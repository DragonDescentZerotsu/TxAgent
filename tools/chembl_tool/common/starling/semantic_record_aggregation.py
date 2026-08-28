"""Resumable LLM aggregation for non-deterministic collapsed record groups."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tools.chembl_tool.common.json_utils import parse_json_content
from tools.chembl_tool.common.openai_reasoning_client import OpenAICompatibleClient
from tools.chembl_tool.common.starling.build_reference_semantics_mapping import (
    BUDGET_EXHAUSTED_EXIT_CODE,
    TokenLedger,
)


SCHEMA_VERSION = "semantic_record_aggregation.v4"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEFAULT_BASE_URL = "http://127.0.0.1:50001/v1"
TEMPLATE_PATH = Path(__file__).parent / "prompt_templates/semantic_record_aggregation_v4.jinja"
_PARENTHETICAL_SMILES_RE = re.compile(
    r"\(\s*(?:global_identifier\s*:\s*)?SMILES\s*:\s*[^\s)]+\s*\)",
    re.IGNORECASE,
)
_COMPOUND_SMILES_RE = re.compile(
    r"\bcompound\s+(?:identified\s+as\s+)?SMILES\s*:\s*\S+", re.IGNORECASE
)
_IDENTIFIED_AS_SMILES_RE = re.compile(
    r"\s+(?:is\s+)?identified\s+as\s+SMILES\s*:\s*\S+", re.IGNORECASE
)
_SMILES_RE = re.compile(r"\bSMILES\s*:\s*\S+", re.IGNORECASE)
_GLOBAL_IDENTIFIER_RE = re.compile(r"\bglobal_identifier\s*:\s*\S+", re.IGNORECASE)
_INTERNAL_IDENTIFIER_RE = re.compile(
    r"\b(?:SMILES|global_identifier)\s*:", re.IGNORECASE
)
_TASK_JUDGMENT_RE = re.compile(r"\b(?:direct[- ]label|informativeness)\b", re.IGNORECASE)


@dataclass(frozen=True)
class SemanticAggregationConfig:
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    provider: str = "openai-compatible"
    reasoning_effort: str = ""
    workers: int = 8
    timeout_s: int = 900
    max_tokens: int = 4096
    max_attempts: int = 3
    max_new_groups: int | None = None
    cache_path: Path | None = None
    token_ledger: TokenLedger | None = None


class SemanticAggregationBudgetExhausted(RuntimeError):
    """The active paid-token epoch cannot reserve another request."""


class SemanticAggregationLimitReached(RuntimeError):
    """A deliberate pilot stopped before the semantic cache was complete."""


def aggregate_semantic_groups(
    groups: Mapping[str, Mapping[str, Any]],
    *,
    config: SemanticAggregationConfig | None,
    prior_paths: Sequence[str | Path] = (),
) -> list[dict[str, Any]]:
    """Return one validated, trace-bearing response per stable group ID."""
    rendered = {
        group_id: render_prompt(payload) for group_id, payload in groups.items()
    }
    cached = _load_cache((*prior_paths, *((config.cache_path,) if config and config.cache_path else ())))
    output: dict[str, dict[str, Any]] = {}
    missing: dict[str, Mapping[str, Any]] = {}
    for group_id, payload in groups.items():
        prompt = rendered[group_id]
        prompt_sha256 = _sha256(prompt)
        prior = cached.get(group_id)
        reusable = match_semantic_cache_row(
            prior,
            payload=payload,
            prompt_sha256=prompt_sha256,
            expected_model=config.model if config else None,
        )
        if reusable:
            output[group_id] = reusable
        else:
            missing[group_id] = payload
    if missing and config is None:
        raise RuntimeError(
            f"{len(missing)} semantic collapse group(s) require LLM aggregation; "
            "provide semantic aggregation endpoint settings"
        )
    if missing:
        assert config is not None
        cache_path = config.cache_path
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
        worker_count = max(1, config.workers)
        ordered = sorted(
            missing.items(),
            key=lambda item: (len(rendered[item[0]].encode("utf-8")), item[0]),
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(worker_count, len(missing))
        ) as executor:
            pending = iter(ordered)
            futures: dict[concurrent.futures.Future[dict[str, Any]], str] = {}
            failures: list[tuple[str, Exception]] = []
            stop_scheduling = False
            while futures or not stop_scheduling:
                while len(futures) < worker_count and not stop_scheduling:
                    try:
                        group_id, payload = next(pending)
                    except StopIteration:
                        stop_scheduling = True
                        break
                    future = executor.submit(
                        _run_one,
                        group_id,
                        payload,
                        rendered[group_id],
                        config,
                    )
                    futures[future] = group_id
                if not futures:
                    break
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    group_id = futures.pop(future)
                    try:
                        row = future.result()
                    except Exception as exc:
                        failures.append((group_id, exc))
                        if isinstance(exc, SemanticAggregationBudgetExhausted):
                            stop_scheduling = True
                        continue
                    output[group_id] = row
                    if cache_path is not None:
                        with cache_path.open("a", encoding="utf-8") as handle:
                            handle.write(
                                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                            )
            if failures:
                if any(
                    isinstance(error, SemanticAggregationBudgetExhausted)
                    for _, error in failures
                ):
                    if config.token_ledger is not None:
                        config.token_ledger.mark_exhausted()
                    raise SemanticAggregationBudgetExhausted(
                        "semantic aggregation token budget exhausted; "
                        f"validated cache rows were preserved at {cache_path}"
                    )
                group_id, error = failures[0]
                raise RuntimeError(
                    f"{len(failures)} semantic aggregation request(s) failed; "
                    f"first={group_id}: {error}"
                ) from error
    return [output[group_id] for group_id in sorted(groups)]


def render_prompt(payload: Mapping[str, Any]) -> str:
    return _prompt_template().render(**payload)


@lru_cache(maxsize=1)
def _prompt_template() -> Any:
    environment = Environment(
        loader=FileSystemLoader(str(TEMPLATE_PATH.parent)),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    return environment.get_template(TEMPLATE_PATH.name)


def validate_semantic_response(response: Any) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        raise ValueError("semantic aggregation response must be a JSON object")
    required = {"summary"}
    if set(response) != required:
        raise ValueError(
            "semantic aggregation response fields differ from the contract: "
            f"{sorted(set(response) ^ required)}"
        )
    summary = response["summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("semantic summary must be nonempty text")
    summary = summary.strip()
    if _INTERNAL_IDENTIFIER_RE.search(summary):
        raise ValueError("semantic summary contains an internal structure identifier")
    if _TASK_JUDGMENT_RE.search(summary):
        raise ValueError("semantic summary contains a task judgment")
    return {"summary": summary}


def sanitize_semantic_prompt_text(
    value: Any, *, omit_structure_only: bool = False
) -> str:
    """Remove structure identifiers while preserving scientific prose."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    text = str(value).strip()
    if text.casefold() in {
        "",
        "nan",
        "none",
        "null",
        "unknown",
        "__unknown__",
        "not applicable",
        "not_applicable",
    }:
        return ""
    text = _PARENTHETICAL_SMILES_RE.sub("", text)
    text = _COMPOUND_SMILES_RE.sub("the tested compound", text)
    text = _IDENTIFIED_AS_SMILES_RE.sub("", text)
    text = _SMILES_RE.sub("the tested compound", text)
    text = _GLOBAL_IDENTIFIER_RE.sub("the tested compound", text)
    text = " ".join(text.split())
    if omit_structure_only and text.casefold() == "the tested compound":
        return ""
    return text


def match_semantic_cache_row(
    cached: Mapping[str, Any] | None,
    *,
    payload: Mapping[str, Any],
    prompt_sha256: str,
    expected_model: str | None,
) -> dict[str, Any] | None:
    """Accept only an exact v4 payload, prompt, and model match."""
    if not cached or (
        expected_model is not None and cached.get("requested_model") != expected_model
    ):
        return None
    if cached.get("schema_version") != SCHEMA_VERSION:
        return None
    try:
        validate_semantic_response(cached.get("response"))
    except ValueError:
        return None
    if cached.get("prompt_sha256") != prompt_sha256 or cached.get("input") != payload:
        return None
    return {**cached, "input": dict(payload)}


def _run_one(
    group_id: str,
    payload: Mapping[str, Any],
    prompt: str,
    config: SemanticAggregationConfig,
) -> dict[str, Any]:
    if config.provider not in {"openai-compatible", "distillation"}:
        raise ValueError(f"unknown semantic aggregation provider: {config.provider}")
    client = (
        OpenAICompatibleClient(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            timeout_s=config.timeout_s,
            max_tokens=config.max_tokens,
            temperature=0.0,
            tool_service_url="http://127.0.0.1:8766",
            enable_group_tools=False,
            max_tool_rounds=0,
            reasoning_effort=config.reasoning_effort,
            enable_thinking=False,
        )
        if config.provider == "openai-compatible"
        else None
    )
    errors: list[str] = []
    trace: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    messages = [{"role": "user", "content": prompt}]
    usage_totals: Counter[str] = Counter()
    for attempt in range(1, config.max_attempts + 1):
        if config.provider == "distillation":
            trace = _distillation_chat_json(
                messages,
                config=config,
                request_id=f"{group_id}:{_sha256(prompt)}:{attempt}",
            )
        else:
            assert client is not None
            trace = client.chat_json(messages)
        usage_totals.update(
            {
                str(key): int(value)
                for key, value in (trace.get("usage") or {}).items()
                if isinstance(value, int)
            }
        )
        try:
            response = validate_semantic_response(trace["content"])
            break
        except ValueError as exc:
            errors.append(str(exc))
            messages = [
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": str(trace.get("raw_content") or "{}"),
                },
                {
                    "role": "user",
                    "content": (
                        "Your previous JSON failed validation: "
                        f"{exc}. Return a corrected JSON object only."
                    ),
                },
            ]
    if response is None or trace is None:
        raise RuntimeError(
            f"semantic aggregation failed validation for {group_id}: {errors}"
        )
    return {
        "group_id": group_id,
        "schema_version": SCHEMA_VERSION,
        "prompt_sha256": _sha256(prompt),
        "template_sha256": _sha256(TEMPLATE_PATH.read_text(encoding="utf-8")),
        "requested_model": config.model,
        "served_model": str(trace.get("model") or ""),
        "endpoint": (
            "therapeutic-tuning/distillation"
            if config.provider == "distillation"
            else config.base_url
        ),
        "attempts": len(errors) + 1,
        "validation_errors": errors,
        "usage": dict(usage_totals),
        "response_id": str(trace.get("id") or ""),
        "raw_content": str(trace.get("raw_content") or ""),
        "reasoning_content": str(trace.get("reasoning_content") or ""),
        "response": response,
        "input": dict(payload),
    }


def _distillation_chat_json(
    messages: Sequence[Mapping[str, Any]],
    *,
    config: SemanticAggregationConfig,
    request_id: str,
) -> dict[str, Any]:
    prompt = "\n\n".join(str(message.get("content") or "") for message in messages)
    reservation = len(prompt.encode("utf-8")) + 1_024 + config.max_tokens
    ledger = config.token_ledger
    if ledger is not None and not ledger.reserve(request_id, reservation):
        raise SemanticAggregationBudgetExhausted(request_id)
    usage: dict[str, int] | None = None
    try:
        result = _distillation_llm()(
            prompt,
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=0.0,
            reasoning_effort=config.reasoning_effort or None,
            max_retries=1,
            verbose=False,
        )
        usage = _usage_dict(result.get("usage"))
    finally:
        if ledger is not None:
            ledger.complete(
                request_id,
                None
                if usage is None
                else {
                    "input_tokens": usage.get(
                        "prompt_tokens", usage.get("input_tokens", 0)
                    ),
                    "output_tokens": usage.get(
                        "completion_tokens", usage.get("output_tokens", 0)
                    ),
                },
            )
    raw_content = str(result.get("content") or "")
    try:
        content = parse_json_content(raw_content)
    except (TypeError, ValueError, json.JSONDecodeError):
        content = {}
    return {
        "content": content,
        "raw_content": raw_content,
        "reasoning_content": str(result.get("reasoning") or ""),
        "usage": usage or {},
        "model": "",
        "id": "",
    }


@lru_cache(maxsize=1)
def _distillation_llm() -> Any:
    from tools.chembl_tool.common.starling.build_measurement_resolution_mapping import (
        distillation_client,
    )

    return distillation_client()[0]


def _usage_dict(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        values = usage.model_dump(mode="json")
    elif isinstance(usage, Mapping):
        values = dict(usage)
    else:
        values = {
            key: getattr(usage, key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if getattr(usage, key, None) is not None
        }
    return {
        str(key): int(value)
        for key, value in values.items()
        if isinstance(value, int)
    }


def _load_cache(paths: Sequence[str | Path | None]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        if raw_path is None:
            continue
        path = Path(raw_path)
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict) and row.get("group_id") and row.get("response"):
                    output[str(row["group_id"])] = row
    return output


def load_semantic_cache(
    paths: Sequence[str | Path | None],
) -> dict[str, dict[str, Any]]:
    """Load the last complete response for each stable semantic group."""
    return _load_cache(paths)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "BUDGET_EXHAUSTED_EXIT_CODE",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "SCHEMA_VERSION",
    "SemanticAggregationBudgetExhausted",
    "SemanticAggregationConfig",
    "SemanticAggregationLimitReached",
    "aggregate_semantic_groups",
    "load_semantic_cache",
    "match_semantic_cache_row",
    "render_prompt",
    "sanitize_semantic_prompt_text",
    "validate_semantic_response",
]
