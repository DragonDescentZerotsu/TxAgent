#!/usr/bin/env python3
"""Build Skin_Reaction auxiliary mappings from embedding-neighbourhood prompts.

Distinct source values are embedded locally, clustered into neighbourhoods of
roughly 100 values, and normalized with one structured-output request per
cluster.  Per-cluster caches and a cumulative token ledger make paid runs
resumable without persisting credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans

from data.processing.evidence_library.versions.v7.prompts import PROMPT_ROOT

from data.processing.evidence_library.versions.v7.tasks.skin_reaction.data_processing.auxiliary_mapping_helpers.reconciliation import (
    DEFAULT_OUTPUT,
    FORBIDDEN_OUTPUT,
    MAPPING_VERSION,
    NULL_LIKE,
    SOURCE_SPECS,
    OutputSpec,
    build_mapping,
    build_output_section,
    _distinct_source_tuples,
    _tuple_key,
    load_reviewed_mapping,
    validate_mapping,
)


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_TOKEN_BUDGET = 500_000
CLUSTER_TARGET_SIZE = 100
CLUSTER_RANDOM_SEED = 20260801
MINIBATCH_SIZE = 4096
MINIBATCH_N_INIT = 3
MINIBATCH_MAX_ITER = 100
GENERAL_PROMPT_VERSION = "starling_skin_embedding_bucket_mapping.v3"
SPECIES_PROMPT_VERSION = "starling_skin_embedding_bucket_mapping.v4"
MIXED_PROMPT_VERSION = "starling_skin_embedding_bucket_mapping.mixed.v3_v4"
LEDGER_VERSION = "starling_skin_auxiliary_token_ledger.v1"
PROMPT_REGISTRY_PATH = PROMPT_ROOT / "auxiliary_canonicalization/skin_reaction.json"
DISTILLATION_API_PATH = Path("/data1/joseph/therapeutic-tuning/distillation/api.py")
DEFAULT_WORK_DIR = Path(__file__).with_name("species_context_v3")
RESPONSE_VALIDATION_VERSION = "skin_species_literal_support_gate.v1"
BUDGET_EXIT_CODE = 75
USAGE_EXIT_CODE = 76
API_EXIT_CODE = 77


@dataclass(frozen=True)
class ExtractionSpec:
    source_id: str
    input_column: str
    output_name: str
    output_column: str
    prompt: str | None
    label_source: str
    reviewed_mapping_path: Path | None
    null_sentinel: str | None
    bucket_pattern: str | None
    clustering: str
    cluster_target_size: int = CLUSTER_TARGET_SIZE
    max_labels_per_call: int | None = None
    input_columns: tuple[str, ...] = ()

    @property
    def resolved_input_columns(self) -> tuple[str, ...]:
        return self.input_columns or (self.input_column,)

    @property
    def is_composite(self) -> bool:
        return len(self.resolved_input_columns) > 1


@dataclass(frozen=True)
class Cluster:
    cluster_id: str
    values: tuple[str, ...]


class BudgetExhausted(RuntimeError):
    pass


class UsageUnavailable(RuntimeError):
    pass


class NonRetryableAPIError(RuntimeError):
    pass


class ClusterQueryFailure(RuntimeError):
    pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_prompt_registry() -> dict[str, Any]:
    payload = json.loads(PROMPT_REGISTRY_PATH.read_text(encoding="utf-8"))
    if payload.get("registry_version") != "starling_skin_auxiliary_prompts.v4":
        raise ValueError("Skin auxiliary prompt registry version mismatch")
    prompts = payload.get("prompts")
    if not isinstance(prompts, dict):
        raise ValueError("Skin auxiliary prompt registry has no prompts")
    expected = {
        (source_id, output.output_name): list(output.extraction_input_columns)
        for source_id, source in SOURCE_SPECS.items()
        for output in source.outputs
        if output.label_source == "llm"
    }
    actual: dict[tuple[str, str], list[str]] = {}
    for source_id, outputs in prompts.items():
        if not isinstance(outputs, dict):
            raise ValueError(f"prompt source {source_id!r} is not an object")
        for output_name, section in outputs.items():
            if not isinstance(section, dict):
                raise ValueError(f"prompt {source_id}/{output_name} is not an object")
            prompt = section.get("prompt")
            input_columns = section.get("input_columns")
            if input_columns is None and isinstance(section.get("input_column"), str):
                input_columns = [section["input_column"]]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"prompt {source_id}/{output_name} is empty")
            actual[(source_id, output_name)] = list(input_columns or [])
    if actual != expected:
        raise ValueError(
            f"prompt inventory differs from SOURCE_SPECS: expected={expected} actual={actual}"
        )
    return payload


_PROMPT_REGISTRY = _load_prompt_registry()


def _extraction(source_id: str, output: OutputSpec) -> ExtractionSpec:
    prompt = None
    if output.label_source == "llm":
        prompt = _PROMPT_REGISTRY["prompts"][source_id][output.output_name]["prompt"]
    elif output.label_source != "reviewed_table":
        raise ValueError(f"unsupported label source {output.label_source!r}")
    return ExtractionSpec(
        source_id=source_id,
        input_column=output.current_input_column,
        output_name=output.output_name,
        output_column=output.current_output_column,
        prompt=prompt,
        label_source=output.label_source,
        reviewed_mapping_path=output.reviewed_mapping_path,
        null_sentinel=output.null_sentinel,
        bucket_pattern=output.bucket_pattern,
        clustering=output.clustering,
        cluster_target_size=output.cluster_target_size,
        max_labels_per_call=output.max_labels_per_call,
        input_columns=output.extraction_input_columns,
    )


def _source_extractions(source_id: str) -> tuple[ExtractionSpec, ...]:
    return tuple(_extraction(source_id, output) for output in SOURCE_SPECS[source_id].outputs)


def _uses_species_v4(extraction: ExtractionSpec) -> bool:
    return (
        extraction.source_id == "sensitization_aop"
        and extraction.output_name == "global_species_context"
    )


def _prompt_version(extraction: ExtractionSpec) -> str:
    return SPECIES_PROMPT_VERSION if _uses_species_v4(extraction) else GENERAL_PROMPT_VERSION


def _client_kind(extraction: ExtractionSpec) -> str:
    return "distillation" if _uses_species_v4(extraction) else "openai"


def _selected_prompt_version(
    selected_outputs: Mapping[str, set[str]],
) -> str:
    versions = {
        _prompt_version(extraction)
        for source_id, output_names in selected_outputs.items()
        for extraction in _source_extractions(source_id)
        if extraction.output_name in output_names
    }
    if versions == {GENERAL_PROMPT_VERSION}:
        return GENERAL_PROMPT_VERSION
    if versions == {SPECIES_PROMPT_VERSION}:
        return SPECIES_PROMPT_VERSION
    return MIXED_PROMPT_VERSION


def _distinct_values(series: pd.Series) -> list[str]:
    values = {
        str(value).strip()
        for value in series.dropna().tolist()
        if str(value).strip().casefold() not in NULL_LIKE
    }
    return sorted(values, key=lambda value: (value.casefold(), value))


def _extraction_values(frame: pd.DataFrame, extraction: ExtractionSpec) -> list[str]:
    if not extraction.is_composite:
        return _distinct_values(frame[extraction.resolved_input_columns[0]])
    return [
        _tuple_key(values)
        for values in _distinct_source_tuples(frame, extraction.resolved_input_columns)
    ]


def _packet_fields(extraction: ExtractionSpec, value: str) -> dict[str, str | None]:
    if not extraction.is_composite:
        return {extraction.resolved_input_columns[0]: value}
    decoded = json.loads(value)
    if not isinstance(decoded, list) or len(decoded) != len(extraction.resolved_input_columns):
        raise ValueError("invalid composite extraction key")
    return dict(zip(extraction.resolved_input_columns, decoded, strict=True))


def _embedding_value(extraction: ExtractionSpec, value: str) -> str:
    if not _uses_species_v4(extraction):
        return value
    return "\n".join(
        f"{field}: {item or '[missing]'}"
        for field, item in _packet_fields(extraction, value).items()
    )


def _mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def _embed_value_groups(
    groups: Mapping[str, list[str]], *, model_name: str, batch_size: int, device: str
) -> dict[str, np.ndarray]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    try:
        output: dict[str, np.ndarray] = {}
        for name, values in groups.items():
            chunks: list[np.ndarray] = []
            started = time.time()
            for start in range(0, len(values), batch_size):
                batch = values[start : start + batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with torch.inference_mode():
                    if device.startswith("cuda"):
                        with torch.autocast("cuda", dtype=torch.float16):
                            hidden = model(**encoded).last_hidden_state
                    else:
                        hidden = model(**encoded).last_hidden_state
                    pooled = _mean_pool(hidden, encoded["attention_mask"])
                    pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
                chunks.append(pooled.cpu().numpy())
                done = min(start + batch_size, len(values))
                if done == len(values) or done == batch_size or done % 4096 < batch_size:
                    print(
                        f"[{name}] embedded={done:,}/{len(values):,} "
                        f"elapsed={time.time() - started:.1f}s",
                        flush=True,
                    )
            output[name] = np.concatenate(chunks, axis=0)
        return output
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def _clusters_from_embeddings(
    values: list[str],
    embeddings: np.ndarray,
    *,
    clustering: str,
    target_size: int = CLUSTER_TARGET_SIZE,
    max_labels_per_call: int | None = None,
) -> list[Cluster]:
    if len(values) != len(embeddings):
        raise ValueError("value and embedding counts differ")
    if not values:
        return []
    if target_size < 1:
        raise ValueError("cluster target size must be positive")
    if max_labels_per_call is not None and max_labels_per_call < 1:
        raise ValueError("maximum labels per call must be positive")
    cluster_count = math.ceil(len(values) / target_size)
    if cluster_count == 1:
        labels = np.zeros(len(values), dtype=np.int64)
    elif clustering == "lloyd":
        labels = KMeans(
            n_clusters=cluster_count,
            random_state=CLUSTER_RANDOM_SEED,
            n_init=10,
            algorithm="lloyd",
        ).fit_predict(embeddings)
    elif clustering == "minibatch":
        labels = MiniBatchKMeans(
            n_clusters=cluster_count,
            random_state=CLUSTER_RANDOM_SEED,
            n_init=MINIBATCH_N_INIT,
            max_iter=MINIBATCH_MAX_ITER,
            batch_size=MINIBATCH_SIZE,
        ).fit_predict(embeddings)
    else:
        raise ValueError(f"unsupported clustering algorithm {clustering!r}")
    clusters: list[Cluster] = []
    for label in sorted(set(int(item) for item in labels)):
        member_indices = np.flatnonzero(labels == label)
        chunk_size = max_labels_per_call or len(member_indices)
        for start in range(0, len(member_indices), chunk_size):
            members = tuple(
                values[index]
                for index in member_indices[start : start + chunk_size]
            )
            digest = hashlib.sha256(
                json.dumps(members, ensure_ascii=False).encode("utf-8")
            ).hexdigest()[:16]
            clusters.append(Cluster(cluster_id=f"cluster_{digest}", values=members))
    return sorted(clusters, key=lambda cluster: cluster.cluster_id)


def _cluster_item_ids(cluster: Cluster) -> dict[str, str]:
    return {f"v{index:04d}": value for index, value in enumerate(cluster.values)}


def _clean_bucket(
    value: Any, *, null_sentinel: str | None, bucket_pattern: str | None
) -> str | None:
    # JSON mode commonly serializes constrained numeric labels as numbers even
    # when the prompt calls them labels.  Numeric output is only permitted for
    # an extraction with an explicit canonical bucket pattern (currently the
    # direct-reaction 0-4 severity grade); free-form reconciliations remain
    # string-only.
    if (
        bucket_pattern
        and isinstance(value, int)
        and not isinstance(value, bool)
    ):
        value = str(value)
    if not isinstance(value, str):
        raise ValueError("bucket is not a string")
    bucket = re.sub(r"\s+", " ", value.strip()).casefold()
    if null_sentinel and bucket == null_sentinel.casefold():
        return None
    if not bucket:
        raise ValueError("bucket is empty")
    if len(bucket) > 120 or "\n" in bucket or "\r" in bucket:
        raise ValueError("bucket is not one concise label")
    if any(token in bucket for token in ("```", "{", "}", "->", "→")):
        raise ValueError("bucket contains non-label syntax")
    if bucket_pattern:
        if re.fullmatch(bucket_pattern, bucket) is None:
            raise ValueError("bucket does not match the required canonical syntax")
    elif FORBIDDEN_OUTPUT.search(bucket):
        raise ValueError("bucket contains an unknown-like value")
    return bucket


_SPECIES_LITERAL_PATTERNS: dict[str, re.Pattern[str]] = {
    "human": re.compile(
        r"\b(?:human|humans|patients?|subjects?|volunteers?|participants?|donors?)\b",
        re.I,
    ),
    "mouse": re.compile(r"\b(?:mouse|mice|murine|mus\s+musculus)\b", re.I),
    "rat": re.compile(r"\b(?:rat|rats|rattus(?:\s+norvegicus)?)\b", re.I),
    "guinea pig": re.compile(r"\bguinea[\s-]*pigs?\b|\bcavia\s+porcellus\b", re.I),
    "rabbit": re.compile(r"\b(?:rabbit|rabbits|oryctolagus\s+cuniculus)\b", re.I),
    "pig": re.compile(r"(?<!guinea[\s-])\b(?:pig|pigs|porcine|swine|minipigs?|micropigs?|sus\s+scrofa)\b", re.I),
    "dog": re.compile(r"\b(?:dog|dogs|canine|canines|beagle|beagles)\b", re.I),
    "cattle": re.compile(r"\b(?:cattle|bovine|cow|cows)\b", re.I),
    "sheep": re.compile(r"\b(?:sheep|ovine)\b", re.I),
    "goat": re.compile(r"\b(?:goat|goats|caprine)\b", re.I),
    "hamster": re.compile(r"\bhamsters?\b", re.I),
    "rhesus monkey": re.compile(r"\b(?:rhesus|macaca\s+mulatta)\b", re.I),
    "cynomolgus monkey": re.compile(r"\b(?:cynomolgus|macaca\s+fascicularis)\b", re.I),
    "monkey": re.compile(r"\b(?:monkey|monkeys|macaque|macaques)\b", re.I),
    "cat": re.compile(r"\b(?:cat|cats|feline)\b", re.I),
    "horse": re.compile(r"\b(?:horse|horses|equine)\b", re.I),
    "chicken": re.compile(r"\bchickens?\b", re.I),
    "frog": re.compile(r"\b(?:frog|frogs|xenopus(?:\s+laevis)?)\b", re.I),
    "snake": re.compile(r"\b(?:snake|snakes|snakeskin)\b", re.I),
}
_SPECIES_LABEL_ALIASES = {
    "bovine": "cattle",
    "cow": "cattle",
    "canine": "dog",
    "murine": "mouse",
    "porcine": "pig",
    "ovine": "sheep",
    "caprine": "goat",
}


def _fail_closed_unsupported_species(
    value: str | None, *, raw_packet: str, extraction: ExtractionSpec
) -> str | None:
    if value is None or not (
        extraction.source_id == "sensitization_aop"
        and extraction.output_name == "global_species_context"
    ):
        return value
    canonical = _SPECIES_LABEL_ALIASES.get(value, value)
    pattern = _SPECIES_LITERAL_PATTERNS.get(canonical)
    if pattern is None or pattern.search(raw_packet) is None:
        return None
    return value


def _validate_response(
    content: str | None, *, item_ids: Mapping[str, str], extraction: ExtractionSpec
) -> dict[str, str | None]:
    try:
        payload = json.loads(content or "")
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"mapping"}:
        raise ValueError("response must contain only the mapping object")
    mapping = payload["mapping"]
    if not isinstance(mapping, dict) or set(mapping) != set(item_ids):
        actual = set(mapping) if isinstance(mapping, dict) else set()
        expected = set(item_ids)
        raise ValueError(
            f"mapping ID mismatch: missing={len(expected - actual)} extra={len(actual - expected)}"
        )
    output: dict[str, str | None] = {}
    for item_id, raw_packet in item_ids.items():
        cleaned = _clean_bucket(
            mapping[item_id],
            null_sentinel=extraction.null_sentinel,
            bucket_pattern=extraction.bucket_pattern,
        )
        output[item_id] = _fail_closed_unsupported_species(
            cleaned, raw_packet=raw_packet, extraction=extraction
        )
    return output


def _empty_usage() -> dict[str, int]:
    return {
        "cached_input_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }


def _response_usage(response: Any) -> tuple[dict[str, int], bool]:
    usage = getattr(response, "usage", None)
    if usage is None or getattr(usage, "total_tokens", None) is None:
        return _empty_usage(), False
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return (
        {
            "cached_input_tokens": int(getattr(prompt_details, "cached_tokens", 0) or 0),
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "reasoning_tokens": int(
                getattr(completion_details, "reasoning_tokens", 0) or 0
            ),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        },
        True,
    )


class TokenLedger:
    def __init__(
        self,
        path: Path,
        *,
        model: str,
        reasoning_effort: str,
        prompt_version: str = GENERAL_PROMPT_VERSION,
    ):
        self.path = path
        self.lock = threading.RLock()
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("ledger_version") != LEDGER_VERSION:
                raise ValueError("token ledger version mismatch")
            if payload.get("prompt_version") != prompt_version:
                raise ValueError("token ledger prompt identity mismatch")
            if payload.get("model") != model or payload.get("reasoning_effort") != reasoning_effort:
                raise ValueError("token ledger model/reasoning identity mismatch")
            self.payload = payload
        else:
            self.payload = {
                "ledger_version": LEDGER_VERSION,
                "prompt_version": prompt_version,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "lifetime_usage": _empty_usage(),
                "attempts": 0,
                "by_extraction": {},
                "status_counts": {},
            }

    def record(self, extraction_key: str, usage: Mapping[str, int], status: str) -> None:
        with self.lock:
            for key in _empty_usage():
                self.payload["lifetime_usage"][key] = int(
                    self.payload["lifetime_usage"].get(key) or 0
                ) + int(usage.get(key) or 0)
            section = self.payload["by_extraction"].setdefault(
                extraction_key, {"usage": _empty_usage(), "attempts": 0}
            )
            for key in _empty_usage():
                section["usage"][key] = int(section["usage"].get(key) or 0) + int(
                    usage.get(key) or 0
                )
            section["attempts"] = int(section.get("attempts") or 0) + 1
            self.payload["attempts"] = int(self.payload.get("attempts") or 0) + 1
            counts = self.payload["status_counts"]
            counts[status] = int(counts.get(status) or 0) + 1
            _atomic_json(self.path, self.payload)


class TokenBudget:
    def __init__(
        self, *, limit: int, ledger: TokenLedger, request_reserve: int = 0
    ):
        self.limit = limit
        self.ledger = ledger
        self.request_reserve = request_reserve
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.accounting_error = threading.Event()
        self.non_retryable_error: str | None = None
        self.run_usage = _empty_usage()
        self.in_flight_attempts = 0

    def before_attempt(self) -> None:
        with self.lock:
            if self.non_retryable_error is not None:
                raise NonRetryableAPIError(self.non_retryable_error)
            if self.accounting_error.is_set():
                raise UsageUnavailable("provider token usage was unavailable")
            projected = self.run_usage["total_tokens"] + (
                self.in_flight_attempts + 1
            ) * self.request_reserve
            if self.stop_event.is_set() or projected > self.limit:
                self.stop_event.set()
                raise BudgetExhausted("per-invocation token budget reached")
            self.in_flight_attempts += 1

    def record(
        self,
        extraction_key: str,
        usage: Mapping[str, int],
        *,
        status: str,
        usage_present: bool,
    ) -> None:
        with self.lock:
            if self.in_flight_attempts < 1:
                raise RuntimeError("token-budget attempt accounting underflow")
            self.in_flight_attempts -= 1
            self.ledger.record(extraction_key, usage, status)
            for key in _empty_usage():
                self.run_usage[key] += int(usage.get(key) or 0)
            if not usage_present:
                self.accounting_error.set()
                self.stop_event.set()
            elif self.run_usage["total_tokens"] >= self.limit:
                self.stop_event.set()

    def stop_for_non_retryable_api_error(self, message: str) -> None:
        with self.lock:
            if self.non_retryable_error is None:
                self.non_retryable_error = message
            self.stop_event.set()

    def summary(self) -> dict[str, Any]:
        return {
            "budget": self.limit,
            "request_reserve": self.request_reserve,
            "run_usage": dict(self.run_usage),
            "lifetime_usage": dict(self.ledger.payload["lifetime_usage"]),
        }


class ClientHolder:
    def __init__(self, *, api_key_env: str | None):
        self.api_key_env = api_key_env
        self.lock = threading.Lock()
        self.clients: dict[str, Any] = {}

    def get(self, extraction: ExtractionSpec) -> Any:
        client_kind = _client_kind(extraction)
        with self.lock:
            if client_kind in self.clients:
                return self.clients[client_kind]
            if client_kind == "openai":
                if not self.api_key_env:
                    raise RuntimeError("--api-key-env is required for v3 paid extraction")
                api_key = os.environ.get(self.api_key_env)
                if not api_key:
                    raise RuntimeError(
                        f"environment variable {self.api_key_env} is unset; its value is never read from a file"
                    )
                from openai import OpenAI

                self.clients[client_kind] = OpenAI(api_key=api_key)
                return self.clients[client_kind]
            if self.api_key_env and not os.environ.get(self.api_key_env):
                raise RuntimeError(
                    f"environment variable {self.api_key_env} is unset; its value is never logged"
                )
            if not DISTILLATION_API_PATH.is_file():
                raise FileNotFoundError(
                    f"distillation API not found: {DISTILLATION_API_PATH}"
                )
            spec = importlib.util.spec_from_file_location(
                "_txagent_skin_species_distillation_api", DISTILLATION_API_PATH
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("could not load the distillation API module")
            module = importlib.util.module_from_spec(spec)
            distillation_root = str(DISTILLATION_API_PATH.parent)
            sys.path.insert(0, distillation_root)
            try:
                spec.loader.exec_module(module)
            finally:
                if sys.path and sys.path[0] == distillation_root:
                    sys.path.pop(0)
            self.clients[client_kind] = _DistillationClient(module.llm)
            return self.clients[client_kind]


class _DistillationCompletions:
    def __init__(self, llm: Any):
        self._llm = llm

    def create(self, **request: Any) -> Any:
        messages = request["messages"]
        result = self._llm(
            {"system": messages[0]["content"], "user": messages[1]["content"]},
            model=request["model"],
            max_tokens=16_384,
            temperature=0.0,
            verbose=False,
            reasoning_effort=request.get("reasoning_effort"),
            max_retries=1,
        )
        content = result.get("content")
        usage = result.get("usage")
        if content is None:
            raise RuntimeError("distillation API returned no completion content")
        return type(
            "DistillationResponse",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": content})()},
                    )()
                ],
                "usage": usage,
            },
        )()


class _DistillationClient:
    def __init__(self, llm: Any):
        self.chat = type(
            "Chat", (), {"completions": _DistillationCompletions(llm)}
        )()


def _cache_identity(
    extraction: ExtractionSpec,
    cluster: Cluster,
    *,
    model: str,
    reasoning_effort: str,
) -> str:
    payload = {
        "prompt_version": _prompt_version(extraction),
        "source_id": extraction.source_id,
        "input_column": extraction.input_column,
        "output_name": extraction.output_name,
        "output_column": extraction.output_column,
        "prompt": extraction.prompt,
        "null_sentinel": extraction.null_sentinel,
        "bucket_pattern": extraction.bucket_pattern,
        "cluster_id": cluster.cluster_id,
        "values": cluster.values,
        "model": model,
        "reasoning_effort": reasoning_effort,
    }
    if _uses_species_v4(extraction):
        payload["input_columns"] = extraction.resolved_input_columns
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return output
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                identity = str(record["identity"])
                output[identity] = record
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid cache line {path}:{line_number}: {exc}") from exc
    return output


def _append_cache(
    path: Path,
    *,
    identity: str,
    cluster: Cluster,
    mapping: Mapping[str, str | None],
    generation: Mapping[str, Any],
    lock: threading.Lock,
) -> None:
    record = json.dumps(
        {
            "generation": generation,
            "identity": identity,
            "mapping": mapping,
            "members": list(cluster.values),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(record + "\n")
            handle.flush()


def _request_items(
    extraction: ExtractionSpec, item_ids: Mapping[str, str]
) -> list[dict[str, Any]]:
    if _uses_species_v4(extraction):
        return [
            {"id": item_id, "fields": _packet_fields(extraction, value)}
            for item_id, value in item_ids.items()
        ]
    return [
        {"id": item_id, "value": value}
        for item_id, value in item_ids.items()
    ]


def _query_cluster(
    client_holder: ClientHolder,
    *,
    extraction: ExtractionSpec,
    cluster: Cluster,
    model: str,
    reasoning_effort: str,
    max_retries: int,
    retry_delay: float,
    budget: TokenBudget,
) -> tuple[dict[str, str | None], dict[str, Any]]:
    item_ids = _cluster_item_ids(cluster)
    attempts: list[dict[str, Any]] = []
    last_error: Exception | None = None
    extraction_key = f"{extraction.source_id}/{extraction.output_name}"
    for attempt in range(1, max_retries + 1):
        budget.before_attempt()
        try:
            prompt = extraction.prompt or ""
            if attempt > 1:
                prompt += (
                    "\n\nThe previous response was invalid. Return every supplied ID exactly once "
                    "inside the single mapping object, with no extra keys. "
                    f"Validator feedback: {last_error}."
                )
            request: dict[str, Any] = {
                "model": model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "response_contract": (
                                    "Return exactly one JSON object containing only the mapping object."
                                ),
                                "items": _request_items(extraction, item_ids),
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
            }
            if reasoning_effort:
                request["reasoning_effort"] = reasoning_effort
            response = client_holder.get(extraction).chat.completions.create(**request)
            content = response.choices[0].message.content or ""
            usage, usage_present = _response_usage(response)
            try:
                mapping = _validate_response(
                    content, item_ids=item_ids, extraction=extraction
                )
            except Exception as exc:
                last_error = exc
                status = "validation_error"
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": status,
                        "usage": usage,
                        "response_sha256": hashlib.sha256(content.encode()).hexdigest(),
                        "validation_error": f"{type(exc).__name__}: {exc}",
                    }
                )
                budget.record(
                    extraction_key,
                    usage,
                    status=status,
                    usage_present=usage_present,
                )
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)
                    continue
                break
            status = "valid"
            attempts.append(
                {
                    "attempt": attempt,
                    "status": status,
                    "usage": usage,
                    "response_sha256": hashlib.sha256(content.encode()).hexdigest(),
                }
            )
            budget.record(
                extraction_key,
                usage,
                status=status,
                usage_present=usage_present,
            )
            generation = {
                "attempt_count": len(attempts),
                "attempts": attempts,
                "usage": {
                    key: sum(int(item["usage"].get(key) or 0) for item in attempts)
                    for key in _empty_usage()
                },
            }
            return mapping, generation
        except (BudgetExhausted, UsageUnavailable, NonRetryableAPIError):
            raise
        except Exception as exc:
            last_error = exc
            status_code = getattr(exc, "status_code", None)
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "api_error",
                    "status_code": status_code,
                    "usage": _empty_usage(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            budget.record(
                extraction_key,
                _empty_usage(),
                status="api_error",
                usage_present=True,
            )
            text = str(exc).casefold()
            non_retryable = status_code in {400, 401, 403, 404} or (
                status_code == 429
                and any(
                    marker in text
                    for marker in (
                        "insufficient_quota",
                        "credit_balance_exhausted",
                        "no credits",
                    )
                )
            )
            if non_retryable:
                message = f"{type(exc).__name__}: {exc}"
                budget.stop_for_non_retryable_api_error(message)
                raise NonRetryableAPIError(message) from exc
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
    raise ClusterQueryFailure(
        f"cluster {cluster.cluster_id} failed after {len(attempts)} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def _map_clusters(
    *,
    client_holder: ClientHolder,
    extraction: ExtractionSpec,
    clusters: list[Cluster],
    model: str,
    reasoning_effort: str,
    workers: int,
    max_retries: int,
    retry_delay: float,
    cache_path: Path,
    budget: TokenBudget,
) -> dict[str, str | None]:
    cached = _load_cache(cache_path)
    cluster_mappings: dict[str, dict[str, str | None]] = {}
    pending: list[tuple[Cluster, str]] = []
    for cluster in clusters:
        identity = _cache_identity(
            extraction, cluster, model=model, reasoning_effort=reasoning_effort
        )
        record = cached.get(identity)
        item_ids = _cluster_item_ids(cluster)
        if record is None:
            pending.append((cluster, identity))
            continue
        if tuple(record.get("members") or ()) != cluster.values:
            raise ValueError(f"cached cluster members differ for {cluster.cluster_id}")
        raw_mapping = record.get("mapping")
        if not isinstance(raw_mapping, dict) or set(raw_mapping) != set(item_ids):
            raise ValueError(f"cached mapping ID mismatch for {cluster.cluster_id}")
        cluster_mappings[cluster.cluster_id] = {
            item_id: _fail_closed_unsupported_species(
                (
                    None
                    if raw_mapping[item_id] is None
                    else _clean_bucket(
                        raw_mapping[item_id],
                        null_sentinel=extraction.null_sentinel,
                        bucket_pattern=extraction.bucket_pattern,
                    )
                ),
                raw_packet=item_ids[item_id],
                extraction=extraction,
            )
            for item_id in item_ids
        }
    print(
        f"[{extraction.source_id}/{extraction.output_name}] clusters={len(clusters):,} "
        f"cached={len(clusters) - len(pending):,} pending={len(pending):,}",
        flush=True,
    )
    failures: list[Exception] = []
    stopped = False
    api_stopped = False
    cache_lock = threading.Lock()
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _query_cluster,
                client_holder,
                extraction=extraction,
                cluster=cluster,
                model=model,
                reasoning_effort=reasoning_effort,
                max_retries=max_retries,
                retry_delay=retry_delay,
                budget=budget,
            ): (cluster, identity)
            for cluster, identity in pending
        }
        for future in as_completed(futures):
            cluster, identity = futures[future]
            try:
                mapping, generation = future.result()
            except BudgetExhausted:
                stopped = True
                continue
            except UsageUnavailable:
                stopped = True
                continue
            except NonRetryableAPIError:
                api_stopped = True
                continue
            except Exception as exc:
                failures.append(exc)
                continue
            cluster_mappings[cluster.cluster_id] = mapping
            _append_cache(
                cache_path,
                identity=identity,
                cluster=cluster,
                mapping=mapping,
                generation=generation,
                lock=cache_lock,
            )
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(pending):
                print(
                    f"[{extraction.source_id}/{extraction.output_name}] "
                    f"completed={completed:,}/{len(pending):,}",
                    flush=True,
                )
    if api_stopped or budget.non_retryable_error is not None:
        raise NonRetryableAPIError(
            budget.non_retryable_error or "provider rejected the request"
        )
    if failures:
        raise RuntimeError(
            f"{len(failures)} cluster(s) failed; first={failures[0]}"
        ) from failures[0]
    if budget.accounting_error.is_set():
        raise UsageUnavailable("provider response omitted total_tokens; cached valid results were retained")
    if stopped or len(cluster_mappings) != len(clusters):
        remaining = len(clusters) - len(cluster_mappings)
        raise BudgetExhausted(
            f"token budget stopped {extraction.source_id}/{extraction.output_name}; "
            f"remaining_clusters={remaining}"
        )
    output: dict[str, str | None] = {}
    for cluster in clusters:
        item_ids = _cluster_item_ids(cluster)
        mapping = cluster_mappings[cluster.cluster_id]
        for item_id, raw_value in item_ids.items():
            output[raw_value] = mapping[item_id]
    return dict(sorted(output.items(), key=lambda item: (item[0].casefold(), item[0])))


def _source_identity(
    source_id: str, *, model: str, reasoning_effort: str, embedding_model: str
) -> str:
    spec = SOURCE_SPECS[source_id]
    extractions = _source_extractions(source_id)
    uses_species_v4 = any(_uses_species_v4(item) for item in extractions)
    extraction_payload = []
    for extraction in extractions:
        reviewed_hash = (
            _file_sha256(extraction.reviewed_mapping_path)
            if extraction.reviewed_mapping_path is not None
            else None
        )
        extraction_identity = dict(extraction.__dict__)
        if not uses_species_v4:
            # Reproduce the frozen v3 dataclass payload exactly so unaffected
            # source snapshots remain valid after the species-only upgrade.
            extraction_identity.pop("input_columns")
        if extraction.cluster_target_size == CLUSTER_TARGET_SIZE:
            extraction_identity.pop("cluster_target_size")
        if extraction.max_labels_per_call is None:
            extraction_identity.pop("max_labels_per_call")
        extraction_payload.append(
            {
                **extraction_identity,
                "reviewed_mapping_path": str(extraction.reviewed_mapping_path or ""),
                "reviewed_mapping_sha256": reviewed_hash,
            }
        )
    payload = {
        "prompt_version": (
            MIXED_PROMPT_VERSION if uses_species_v4 else GENERAL_PROMPT_VERSION
        ),
        "source_id": source_id,
        "source_sha256": _file_sha256(spec.parquet_path),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "embedding_model": embedding_model,
        "cluster_target_size": CLUSTER_TARGET_SIZE,
        "cluster_seed": CLUSTER_RANDOM_SEED,
        "minibatch": {
            "batch_size": MINIBATCH_SIZE,
            "n_init": MINIBATCH_N_INIT,
            "max_iter": MINIBATCH_MAX_ITER,
        },
        "extractions": extraction_payload,
    }
    if uses_species_v4:
        payload["response_validation_version"] = RESPONSE_VALIDATION_VERSION
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()


def _snapshot_path(cache_dir: Path, source_id: str) -> Path:
    return cache_dir / f"{source_id}.complete.json"


def _load_snapshot(
    cache_dir: Path,
    source_id: str,
    *,
    model: str,
    reasoning_effort: str,
    embedding_model: str,
) -> dict[str, Any] | None:
    path = _snapshot_path(cache_dir, source_id)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = _source_identity(
        source_id,
        model=model,
        reasoning_effort=reasoning_effort,
        embedding_model=embedding_model,
    )
    if payload.get("identity") != expected:
        return None
    mapping = payload.get("mapping")
    return mapping if isinstance(mapping, dict) else None


def _load_snapshot_from_models(
    cache_dir: Path,
    source_id: str,
    *,
    models: list[str],
    reasoning_effort: str,
    embedding_model: str,
) -> tuple[dict[str, Any], str] | None:
    for model in dict.fromkeys(models):
        mapping = _load_snapshot(
            cache_dir,
            source_id,
            model=model,
            reasoning_effort=reasoning_effort,
            embedding_model=embedding_model,
        )
        if mapping is not None:
            return mapping, model
    return None


def _build_source(
    args: argparse.Namespace,
    *,
    source_id: str,
    cache_dir: Path,
    client_holder: ClientHolder,
    budget: TokenBudget,
    selected_output_names: set[str] | None = None,
) -> dict[str, Any]:
    spec = SOURCE_SPECS[source_id]
    all_extractions = _source_extractions(source_id)
    extractions = tuple(
        item
        for item in all_extractions
        if selected_output_names is None or item.output_name in selected_output_names
    )
    if not extractions:
        return {}
    complete_source = len(extractions) == len(all_extractions)
    if complete_source:
        cached_snapshot = _load_snapshot(
            cache_dir,
            source_id,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            embedding_model=args.embedding_model,
        )
        if cached_snapshot is not None:
            print(f"[{source_id}] complete source snapshot cached", flush=True)
            return cached_snapshot
    columns = sorted(
        {column for item in extractions for column in item.resolved_input_columns}
    )
    frame = pd.read_parquet(spec.parquet_path, columns=columns)
    values_by_extraction = {
        extraction.output_name: _extraction_values(frame, extraction)
        for extraction in extractions
    }
    groups = {
        f"{source_id}/{extraction.output_name}": [
            _embedding_value(extraction, value)
            for value in values_by_extraction[extraction.output_name]
        ]
        for extraction in extractions
        if extraction.label_source == "llm"
    }
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = (
        _embed_value_groups(
            groups,
            model_name=args.embedding_model,
            batch_size=args.embedding_batch_size,
            device=device,
        )
        if groups
        else {}
    )
    nested: dict[str, dict[str, dict[str, str | None]]] = {}
    for extraction in extractions:
        values = values_by_extraction[extraction.output_name]
        if extraction.label_source == "reviewed_table":
            if extraction.reviewed_mapping_path is None:
                raise ValueError(f"{source_id}/{extraction.output_name} has no reviewed table")
            mapping = load_reviewed_mapping(extraction.reviewed_mapping_path)
            uncovered = set(values) - set(mapping)
            if uncovered:
                raise ValueError(
                    f"reviewed table lacks {len(uncovered)} values; first={sorted(uncovered)[0]!r}"
                )
            resolved = {value: mapping[value] for value in values}
        else:
            clusters = _clusters_from_embeddings(
                values,
                embeddings[f"{source_id}/{extraction.output_name}"],
                clustering=extraction.clustering,
                target_size=extraction.cluster_target_size,
                max_labels_per_call=extraction.max_labels_per_call,
            )
            cache_path = cache_dir / f"{source_id}__{extraction.output_name}.jsonl"
            resolved = _map_clusters(
                client_holder=client_holder,
                extraction=extraction,
                clusters=clusters,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                workers=args.workers,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
                cache_path=cache_path,
                budget=budget,
            )
        nested.setdefault(extraction.input_column, {})[
            extraction.output_column
        ] = resolved
        counts = pd.Series(list(resolved.values()), dtype="string").value_counts(
            dropna=False
        )
        print(
            f"[{source_id}/{extraction.output_name}] values={len(resolved):,} "
            f"labels={counts.index.dropna().nunique():,} null={sum(v is None for v in resolved.values()):,}",
            flush=True,
        )
    if complete_source:
        _atomic_json(
            _snapshot_path(cache_dir, source_id),
            {
                "identity": _source_identity(
                    source_id,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    embedding_model=args.embedding_model,
                ),
                "mapping": nested,
            },
        )
    return nested


def dry_run(args: argparse.Namespace) -> None:
    selected_outputs = _selected_outputs(args)
    total = 0
    for source_id, output_names in selected_outputs.items():
        spec = SOURCE_SPECS[source_id]
        extractions = tuple(
            item for item in _source_extractions(source_id) if item.output_name in output_names
        )
        columns = sorted({column for item in extractions for column in item.resolved_input_columns})
        frame = pd.read_parquet(spec.parquet_path, columns=columns)
        for extraction in extractions:
            values = _extraction_values(frame, extraction)
            calls = 0
            if extraction.label_source == "llm":
                calls = math.ceil(
                    len(values)
                    / min(
                        extraction.cluster_target_size,
                        extraction.max_labels_per_call or extraction.cluster_target_size,
                    )
                )
            total += calls
            print(
                f"[{source_id}/{extraction.output_name}] method={extraction.label_source} "
                f"columns={','.join(extraction.resolved_input_columns)} distinct={len(values):,} "
                f"clustering={extraction.clustering} minimum_calls={calls:,} "
                f"max_labels_per_call={extraction.max_labels_per_call or 'unbounded'}"
            )
    print(f"minimum API calls for selected sources: {total:,}")
    print("dry run complete; no API client or embedding model was loaded")


def _selected_outputs(args: argparse.Namespace) -> dict[str, set[str]]:
    if args.only_output:
        selected: dict[str, set[str]] = {}
        valid = {
            f"{source_id}/{output.output_name}"
            for source_id, source in SOURCE_SPECS.items()
            for output in source.outputs
        }
        for key in args.only_output:
            if key not in valid:
                raise ValueError(f"unknown --only-output {key!r}")
            source_id, output_name = key.split("/", 1)
            selected.setdefault(source_id, set()).add(output_name)
        return selected
    sources = args.sources or list(SOURCE_SPECS)
    return {
        source_id: {output.output_name for output in SOURCE_SPECS[source_id].outputs}
        for source_id in sources
    }


def _publish_focused_mapping(
    *,
    args: argparse.Namespace,
    first_stage: Mapping[str, Any],
    selected_outputs: Mapping[str, set[str]],
    budget: TokenBudget,
    work_dir: Path,
) -> dict[str, Any]:
    uses_species_v4 = any(
        _uses_species_v4(extraction)
        for source_id, output_names in selected_outputs.items()
        for extraction in _source_extractions(source_id)
        if extraction.output_name in output_names
    )
    base_path = Path(args.base_mapping)
    if not base_path.is_file():
        raise FileNotFoundError(f"focused publication requires --base-mapping: {base_path}")
    base_sha256 = _file_sha256(base_path)
    payload = json.loads(base_path.read_text(encoding="utf-8"))
    if set(payload) != {"mapping_version", "sources"}:
        raise ValueError("invalid base auxiliary mapping")
    payload["mapping_version"] = MAPPING_VERSION
    for source_id, output_names in selected_outputs.items():
        for output_name in output_names:
            payload["sources"][source_id][output_name] = build_output_section(
                first_stage[source_id], source=source_id, output_name=output_name
            )
    audit = validate_mapping(payload)
    # A focused paid pass is only a local candidate.  It must not replace the
    # runtime mapping before independent global reconciliation and explicit
    # publication approval.
    destination = work_dir / "candidate_v3_mapping.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    manifest = {
        "artifact_version": "skin_species_context_mapping.v3",
        "mapping_version": MAPPING_VERSION,
        "prompt_version": _selected_prompt_version(selected_outputs),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "selected_outputs": {
            source: sorted(outputs) for source, outputs in sorted(selected_outputs.items())
        },
        "base_mapping": {"path": str(base_path), "sha256": base_sha256},
        "candidate_output": {
            "path": str(destination),
            "sha256": _file_sha256(destination),
            "publication_status": "requires_global_reconciliation",
        },
        "prompt_registry": {
            "path": str(PROMPT_REGISTRY_PATH),
            "sha256": _file_sha256(PROMPT_REGISTRY_PATH),
        },
        "audit": audit,
        "token_usage": budget.summary(),
    }
    if uses_species_v4:
        manifest["response_validation_version"] = RESPONSE_VALIDATION_VERSION
        manifest["distillation_api"] = str(DISTILLATION_API_PATH)
    _atomic_json(work_dir / "mapping_manifest.json", manifest)
    return manifest


def run(args: argparse.Namespace) -> int:
    if args.dry_run:
        dry_run(args)
        return 0
    destination = Path(args.output)
    selected_outputs = _selected_outputs(args)
    focused = bool(args.only_output)
    work_dir = Path(args.work_dir) if focused else destination.parent
    write_target = work_dir / "candidate_v3_mapping.json" if focused else destination
    if write_target.exists() and not args.overwrite:
        raise FileExistsError(
            f"output exists: {write_target}; pass --overwrite to replace it"
        )
    cache_dir = (
        work_dir / "cluster_cache"
        if focused
        else destination.with_suffix(destination.suffix + ".cache")
    )
    ledger_path = Path(args.token_ledger) if args.token_ledger else (
        work_dir / "token_ledger.json"
        if focused
        else destination.with_suffix(destination.suffix + ".token_ledger.json")
    )
    ledger = TokenLedger(
        ledger_path,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        prompt_version=_selected_prompt_version(selected_outputs),
    )
    budget = TokenBudget(
        limit=args.token_budget,
        ledger=ledger,
        request_reserve=args.request_token_reserve,
    )
    client_holder = ClientHolder(api_key_env=args.api_key_env)
    first_stage: dict[str, Any] = {}
    try:
        for source_id, output_names in selected_outputs.items():
            first_stage[source_id] = _build_source(
                args,
                source_id=source_id,
                cache_dir=cache_dir,
                client_holder=client_holder,
                budget=budget,
                selected_output_names=output_names,
            )
    except BudgetExhausted as exc:
        print(f"budget stop: {exc}", flush=True)
        print(json.dumps(budget.summary(), indent=2, sort_keys=True), flush=True)
        return BUDGET_EXIT_CODE
    except UsageUnavailable as exc:
        print(f"token accounting stop: {exc}", flush=True)
        print(json.dumps(budget.summary(), indent=2, sort_keys=True), flush=True)
        return USAGE_EXIT_CODE
    except NonRetryableAPIError as exc:
        print(f"provider stop: {exc}", flush=True)
        print(json.dumps(budget.summary(), indent=2, sort_keys=True), flush=True)
        return API_EXIT_CODE

    if focused:
        manifest = _publish_focused_mapping(
            args=args,
            first_stage=first_stage,
            selected_outputs=selected_outputs,
            budget=budget,
            work_dir=work_dir,
        )
        print(json.dumps(manifest["audit"], indent=2, sort_keys=True), flush=True)
        print(json.dumps(budget.summary(), indent=2, sort_keys=True), flush=True)
        print(f"wrote candidate {manifest['candidate_output']['path']}", flush=True)
        return 0

    first_stage = {}
    missing: list[str] = []
    accepted_snapshot_models = [args.model, *args.compatible_snapshot_model]
    for source_id in SOURCE_SPECS:
        loaded = _load_snapshot_from_models(
            cache_dir,
            source_id,
            models=accepted_snapshot_models,
            reasoning_effort=args.reasoning_effort,
            embedding_model=args.embedding_model,
        )
        if loaded is None:
            missing.append(source_id)
        else:
            snapshot, snapshot_model = loaded
            first_stage[source_id] = snapshot
            print(
                f"[{source_id}] finalization snapshot model={snapshot_model}",
                flush=True,
            )
    if missing:
        print(
            f"selected phase complete; final mapping not published; missing_sources={missing}",
            flush=True,
        )
        print(json.dumps(budget.summary(), indent=2, sort_keys=True), flush=True)
        return 0

    payload = build_mapping(first_stage)
    audit = validate_mapping(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    print(json.dumps(budget.summary(), indent=2, sort_keys=True), flush=True)
    print(f"wrote {destination}", flush=True)
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--compatible-snapshot-model",
        action="append",
        default=[],
        help=(
            "Additional model identity accepted only when loading already-complete "
            "source snapshots for final publication; may be repeated."
        ),
    )
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    parser.add_argument(
        "--request-token-reserve",
        type=int,
        default=12_000,
        help=(
            "Conservative reservation for each in-flight request so concurrency "
            "cannot overshoot the invocation token ceiling."
        ),
    )
    parser.add_argument("--token-ledger")
    parser.add_argument("--api-key-env")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--base-mapping", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--only-output",
        action="append",
        help="Build and replace one source/output section using --base-mapping; may repeat.",
    )
    parser.add_argument(
        "--sources", nargs="+", choices=tuple(SOURCE_SPECS), help="Paid phase sources"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.embedding_batch_size < 1 or args.workers < 1 or args.max_retries < 1:
        parser.error("batch size, workers, and retries must be positive")
    if args.retry_delay < 0:
        parser.error("retry delay cannot be negative")
    if args.token_budget < 1 or args.request_token_reserve < 0:
        parser.error("--token-budget must be positive and --request-token-reserve nonnegative")
    if args.only_output and args.sources:
        parser.error("--only-output and --sources are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
