"""Generate frozen row-level reference semantics with bounded OpenAI usage.

Each task freezes its request size and visible fields. A row is written to the
durable submitted log before the API call and is never submitted again, even
after a malformed response or crash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow.parquet as pq

from data.processing.llm_api import DEFAULT_ENV_FILE, openai_compatible_client
from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256
from data.processing.evidence_library.shared.v2.reference_semantics import (
    ENDPOINT_RATIO_BASES,
    REFERENCE_BASES,
    REFERENCE_SCOPE_ABSOLUTE,
    REFERENCE_SCOPE_ENDPOINT_RATIO,
    REFERENCE_SCOPE_STANDARD_CONTROL,
    REFERENCE_SCOPE_UNKNOWN,
    STANDARD_CONTROL_BASES,
    ReferenceAssignment,
    ReferenceSemanticsConfig,
)
from data.processing.evidence_library.versions.v9.task_registry import import_task_module


MODEL = "gpt-5.4-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
REASONING_EFFORT = "low"
MAX_EPOCH_TOKENS = 9_000_000
LEDGER_VERSION = "reference_semantics_openai_token_ledger.v1"
CACHE_VERSION = "reference_semantics_single_submission_cache.v1"
GENERATION_VERSION = "reference_semantics_generation.v3"
BUDGET_EXHAUSTED_EXIT_CODE = 75
DISTILLATION_ROOT = DEFAULT_ENV_FILE.parent
DEFAULT_LEDGER = Path(
    "outputs/chembl_tool/reference_semantics_generation/openai_token_ledger.json"
)

CORE_FIELDS = (
    "source_id",
    "endpoint_name",
    "canonical_endpoint_name",
    "measurement_text",
    "canonical_measurement_text",
    "unit_text",
    "canonical_unit_text",
    "support_text",
)


@dataclass(frozen=True)
class RequestBatch:
    request_id: str
    source_id: str
    rows: tuple[dict[str, Any], ...]
    prompt: str
    max_completion_tokens: int
    payload_rows: tuple[dict[str, Any], ...] | None = None

    @property
    def row_ids(self) -> tuple[str, ...]:
        return tuple(str(row["id"]) for row in self.rows)

    @property
    def api_rows(self) -> tuple[dict[str, Any], ...]:
        return self.payload_rows if self.payload_rows is not None else self.rows

    @property
    def reservation_tokens(self) -> int:
        # UTF-8 bytes are a conservative upper bound on text tokens.  Add a
        # fixed chat-envelope allowance and the hard completion bound.
        body = json.dumps({"rows": self.api_rows}, ensure_ascii=False, sort_keys=True)
        return (
            len(self.prompt.encode("utf-8"))
            + len(body.encode("utf-8"))
            + 1_024
            + self.max_completion_tokens
        )


class TokenLedger:
    def __init__(
        self,
        path: Path,
        *,
        epoch: str,
        start_new_epoch: bool,
        max_tokens: int = MAX_EPOCH_TOKENS,
    ):
        if max_tokens < 1:
            raise ValueError("token ledger maximum must be positive")
        self.path = path
        self.epoch = epoch
        self.max_tokens = int(max_tokens)
        self._lock = threading.Lock()
        self.payload = self._load()
        epochs = self.payload.setdefault("epochs", {})
        active = str(self.payload.get("active_epoch") or "")
        if epoch not in epochs:
            if active and active != epoch and not start_new_epoch:
                raise ValueError(
                    f"ledger active epoch is {active!r}; pass --start-new-budget-epoch "
                    f"to create {epoch!r}"
                )
            if active and active != epoch:
                previous = epochs[active]
                if previous.get("status") != "budget_exhausted":
                    raise ValueError(
                        f"cannot replace non-exhausted budget epoch {active!r}"
                    )
            epochs[epoch] = {
                "max_tokens": self.max_tokens,
                "input_tokens": 0,
                "output_tokens": 0,
                "conservative_unreported_tokens": 0,
                "requests_completed": 0,
                "reservations": {},
                "status": "active",
            }
            self.payload["active_epoch"] = epoch
        elif active and active != epoch:
            raise ValueError(
                f"cannot resume inactive epoch {epoch!r}; active epoch is {active!r}"
            )
        state = epochs[epoch]
        if int(state.get("max_tokens") or 0) != self.max_tokens:
            raise ValueError(
                "token ledger maximum differs from the requested epoch maximum"
            )
        # A process may have died after submission.  Its rows are terminal, and
        # charging the complete reservation is the only safe accounting choice.
        reservations = dict(state.get("reservations") or {})
        if reservations:
            state["conservative_unreported_tokens"] = int(
                state.get("conservative_unreported_tokens") or 0
            ) + sum(int(value) for value in reservations.values())
            state["reservations"] = {}
        self._write()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"ledger_version": LEDGER_VERSION, "active_epoch": None, "epochs": {}}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("ledger_version") != LEDGER_VERSION:
            raise ValueError("reference token ledger version mismatch")
        return payload

    @property
    def state(self) -> dict[str, Any]:
        return self.payload["epochs"][self.epoch]

    def spent(self) -> int:
        state = self.state
        return (
            int(state.get("input_tokens") or 0)
            + int(state.get("output_tokens") or 0)
            + int(state.get("conservative_unreported_tokens") or 0)
        )

    def reserve(self, request_id: str, maximum: int) -> bool:
        with self._lock:
            state = self.state
            reservations = state.setdefault("reservations", {})
            if request_id in reservations:
                raise ValueError(f"duplicate token reservation {request_id}")
            in_flight = sum(int(value) for value in reservations.values())
            if self.spent() + in_flight + maximum > self.max_tokens:
                return False
            reservations[request_id] = int(maximum)
            self._write()
            return True

    def complete(self, request_id: str, usage: Mapping[str, int] | None) -> None:
        with self._lock:
            state = self.state
            reservation = int(state.setdefault("reservations", {}).pop(request_id))
            if usage is None:
                state["conservative_unreported_tokens"] = int(
                    state.get("conservative_unreported_tokens") or 0
                ) + reservation
            else:
                actual_input = int(usage.get("input_tokens") or 0)
                actual_output = int(usage.get("output_tokens") or 0)
                if actual_input + actual_output > reservation:
                    raise ValueError("actual API usage exceeded conservative reservation")
                state["input_tokens"] = int(state.get("input_tokens") or 0) + actual_input
                state["output_tokens"] = int(state.get("output_tokens") or 0) + actual_output
            state["requests_completed"] = int(state.get("requests_completed") or 0) + 1
            self._write()

    def release(self, request_id: str, *, reason: str) -> None:
        """Release a request rejected before generation while retaining an audit."""
        with self._lock:
            state = self.state
            reservation = int(state.setdefault("reservations", {}).pop(request_id))
            rejected = state.setdefault("released_reservations", {}).setdefault(
                reason,
                {"requests": 0, "tokens": 0},
            )
            rejected["requests"] = int(rejected.get("requests") or 0) + 1
            rejected["tokens"] = int(rejected.get("tokens") or 0) + reservation
            self._write()

    def mark_exhausted(self) -> None:
        with self._lock:
            self.state["status"] = "budget_exhausted"
            self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


class SubmissionCache:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.events = self._load()
        self.attempted: set[str] = set()
        self.assignments: dict[str, dict[str, Any]] = {}
        self.provenance: dict[str, dict[str, str]] = {}
        for event in self.events:
            for record_id in event.get("row_ids") or ():
                record_id = str(record_id)
                self.attempted.add(record_id)
                if event.get("status") == "submitted":
                    # A newer submission supersedes an earlier terminal failure.
                    # If the process stops here, publishing must fail closed instead
                    # of silently restoring the older failed assignment.
                    self.assignments.pop(record_id, None)
                if event.get("model"):
                    self.provenance[record_id] = {
                        "inference_model": str(event["model"]),
                        "inference_base_url": str(event.get("base_url") or ""),
                        "inference_credential_env": str(
                            event.get("credential_env") or ""
                        ),
                    }
            if event.get("status") == "terminal":
                for row in event.get("assignments") or ():
                    self.assignments[str(row["cleaned_record_id"])] = dict(row)

    def allow_retry(self, record_ids: set[str]) -> None:
        """Permit an explicit retry of failed or abandoned submitted rows."""
        missing = record_ids - self.attempted
        if missing:
            raise ValueError(
                "only attempted rows may be retried: "
                f"missing={sorted(missing)[:1]}"
            )
        self.attempted.difference_update(record_ids)

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        output = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(row.get("cache_version") != CACHE_VERSION for row in output):
            raise ValueError("reference semantics submission-cache version mismatch")
        return output

    def submit(
        self,
        batch: RequestBatch,
        *,
        epoch: str,
        credential_env: str = "",
        model: str = "",
        base_url: str = "",
    ) -> None:
        if set(batch.row_ids) & self.attempted:
            raise ValueError("a reference-semantics row would be submitted twice")
        event = {
            "cache_version": CACHE_VERSION,
            "status": "submitted",
            "request_id": batch.request_id,
            "budget_epoch": epoch,
            "credential_env": credential_env,
            "model": model,
            "base_url": base_url,
            "source_id": batch.source_id,
            "row_ids": list(batch.row_ids),
        }
        self._append(event)
        for record_id in batch.row_ids:
            self.attempted.add(record_id)
            self.assignments.pop(record_id, None)
            if model:
                self.provenance[record_id] = {
                    "inference_model": model,
                    "inference_base_url": base_url,
                    "inference_credential_env": credential_env,
                }

    def terminal(
        self,
        batch: RequestBatch,
        *,
        assignments: Sequence[Mapping[str, Any]],
        usage: Mapping[str, int] | None,
        response_status: str,
    ) -> None:
        event = {
            "cache_version": CACHE_VERSION,
            "status": "terminal",
            "request_id": batch.request_id,
            "source_id": batch.source_id,
            "row_ids": list(batch.row_ids),
            "response_status": response_status,
            "usage": dict(usage) if usage is not None else None,
            "assignments": list(assignments),
        }
        self._append(event)
        for row in assignments:
            self.assignments[str(row["cleaned_record_id"])] = dict(row)

    def _append(self, event: Mapping[str, Any]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()


def _load_config(task_id: str) -> ReferenceSemanticsConfig:
    module = import_task_module(task_id, "starling_reference_semantics")
    config = getattr(module, "REFERENCE_SEMANTICS_CONFIG", None)
    if not isinstance(config, ReferenceSemanticsConfig):
        raise TypeError(f"{task_id!r} has no ReferenceSemanticsConfig")
    return config


def _load_llm(
    *,
    api_key_env: str | None,
    env_file: Path | None,
    base_url: str,
    workers: int,
    provider: str | None = None,
):
    return openai_compatible_client(
        base_url=base_url,
        provider=provider,
        env_file=env_file,
        credential_env=api_key_env,
        max_connections=workers,
        timeout_s=900,
    )


def _json_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _candidate_rows(
    records: Sequence[Mapping[str, Any]], config: ReferenceSemanticsConfig
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        if config.deterministic_assignment(record) is not None:
            continue
        source_id = str(record.get("source_id") or "")
        source_spec = config.source_specs.get(source_id)
        if source_spec is None:
            raise ValueError(f"no reference prompt spec for source {source_id!r}")
        record_id = str(record.get("cleaned_record_id") or "")
        if not record_id:
            raise ValueError("reference candidate lacks cleaned_record_id")
        payload = {"id": record_id}
        for field in (*CORE_FIELDS, *source_spec.extra_fields):
            payload[field] = _json_value(record.get(field))
        for field in config.generation_policy_fields:
            payload[field] = _json_value(record.get(field))
        if config.generation_no_call_assignment is not None:
            payload["_generation_no_call_assignment"] = (
                config.generation_no_call_assignment(record)
            )
        output.append(payload)
    return output


def _load_candidate_rows(
    records_path: Path,
    config: ReferenceSemanticsConfig,
    *,
    only_assignment_method: str | None = None,
) -> list[dict[str, Any]]:
    """Stream the wide text parquet and retain only rows requiring an LLM call."""
    available_columns = set(pq.read_schema(records_path).names)
    if (
        only_assignment_method
        and "reference_semantics_assignment_method" not in available_columns
    ):
        raise ValueError("record assignment filtering requires a Stage-02 records file")
    requested_columns = {
        "cleaned_record_id",
        "finite_scalar_value",
        "canonical_measurement_scale_id",
        "categorical_encoder_id",
        "canonicalization_status",
        "normalization_validity_status",
        "reference_semantics_assignment_method",
        "bioavailability_report_type",
        *CORE_FIELDS,
        *config.generation_policy_fields,
        *(
            field
            for source_spec in config.source_specs.values()
            for field in source_spec.extra_fields
        ),
    }
    columns = sorted(requested_columns & available_columns)
    candidates: list[dict[str, Any]] = []
    parquet = pq.ParquetFile(records_path)
    for batch in parquet.iter_batches(batch_size=10_000, columns=columns):
        rows = batch.to_pylist()
        if only_assignment_method is not None:
            rows = [
                row
                for row in rows
                if row.get("reference_semantics_assignment_method")
                == only_assignment_method
            ]
        candidates.extend(_candidate_rows(rows, config))
    candidates.sort(key=lambda row: (str(row["source_id"]), str(row["id"])))
    return candidates


def _load_prompt(config: ReferenceSemanticsConfig) -> str:
    payload = json.loads(config.prompt_registry_path.read_text(encoding="utf-8"))
    if payload.get("prompt_version") != config.prompt_version:
        raise ValueError("reference prompt version mismatch")
    prompt = str(payload.get("system_prompt") or "").strip()
    if not prompt:
        raise ValueError("reference system prompt is empty")
    return prompt


def _batches(
    candidates: Sequence[Mapping[str, Any]],
    *,
    config: ReferenceSemanticsConfig,
    prompt: str,
    attempted: set[str],
    model: str = MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> list[RequestBatch]:
    pending = [
        row
        for row in candidates
        if str(row["id"]) not in attempted
        and row.get("_generation_no_call_assignment") is None
    ]
    output: list[RequestBatch] = []
    if config.batch_across_sources:
        groups = [("mixed", pending)]
    else:
        groups = [
            (
                source_id,
                [row for row in pending if str(row["source_id"]) == source_id],
            )
            for source_id in sorted({str(row["source_id"]) for row in pending})
        ]
    for source_id, rows in groups:
        for offset in range(0, len(rows), config.batch_size):
            chunk = tuple(dict(row) for row in rows[offset : offset + config.batch_size])
            if config.prompt_fields:
                payload_chunk = tuple(
                    {
                        "id": str(local_id),
                        **{
                            field: row.get(field)
                            for field in config.prompt_fields
                        },
                    }
                    for local_id, row in enumerate(chunk)
                )
            else:
                payload_chunk = tuple(
                    {
                        key: value
                        for key, value in row.items()
                        if not key.startswith("_")
                    }
                    for row in chunk
                )
            digest = hashlib.sha256(
                json.dumps(
                    {
                        "task": config.task_id,
                        "source": source_id,
                        "prompt_version": config.prompt_version,
                        "model": model,
                        "base_url": base_url,
                        "rows": [row["id"] for row in chunk],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:24]
            output.append(
                RequestBatch(
                    f"{config.task_id}_{source_id}_{digest}",
                    source_id,
                    chunk,
                    prompt,
                    # Leave room for a complete batch. Truncation is fail-closed,
                    # but wasting a row's only submission would be irrecoverable.
                    max_completion_tokens=8_192,
                    payload_rows=payload_chunk,
                )
            )
    return output


def _usage(result: Mapping[str, Any]) -> dict[str, int] | None:
    usage = result.get("usage")
    if usage is None:
        return None
    if isinstance(usage, Mapping):
        input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        output_tokens = int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )
    else:
        input_tokens = int(
            getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0
        )
        output_tokens = int(
            getattr(usage, "completion_tokens", 0)
            or getattr(usage, "output_tokens", 0)
            or 0
        )
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def _unknown_assignment(
    row: Mapping[str, Any], *, config: ReferenceSemanticsConfig, method: str
) -> dict[str, Any]:
    output = {
        "cleaned_record_id": str(row["id"]),
        "source_id": str(row["source_id"]),
        "reference_scope": REFERENCE_SCOPE_UNKNOWN,
        "assignment_method": method,
        "evidence_field": None,
        "evidence_quote": None,
    }
    if config.output_basis:
        output["reference_basis"] = "unknown"
    return output


def _validate_assignment(
    item: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    config: ReferenceSemanticsConfig,
    model_id: str | None = None,
    model: str = MODEL,
) -> tuple[dict[str, Any] | None, str | None]:
    expected_id = str(row["id"]) if model_id is None else model_id
    if str(item.get("id") or "") != expected_id:
        return None, "missing_or_mismatched_id"
    scope = str(item.get("reference_scope") or "")
    if scope not in {
        "absolute",
        "endpoint_defined_ratio",
        "standardized_control_ratio",
        "comparator_relative",
        "unknown",
    }:
        return None, "unsupported_scope"
    field: str | None = None
    quote: str | None = None
    normalized_field = False
    normalized_quote_field = False
    if not config.labels_only_output:
        field = str(item.get("evidence_field") or "")
        quote = str(item.get("evidence_quote") or "")
        if field not in row:
            matching_fields = [
                name
                for name, value in row.items()
                if name != "id" and value is not None and str(value) == field
            ]
            if len(matching_fields) == 1:
                field = matching_fields[0]
                normalized_field = True
        if field not in row or not quote or len(quote) > 240:
            return None, "invalid_evidence_field_or_quote"
        if quote not in str(row.get(field) or ""):
            quote_fields = [
                name
                for name, value in row.items()
                if name != "id" and value is not None and quote in str(value)
            ]
            if len(quote_fields) != 1:
                return None, "nonverbatim_evidence_quote"
            field = quote_fields[0]
            normalized_quote_field = True
    basis: str | None = None
    if config.output_basis:
        basis = str(item.get("reference_basis") or "")
        if basis not in REFERENCE_BASES:
            return None, "unsupported_basis"
        if scope == REFERENCE_SCOPE_ABSOLUTE and basis != "none":
            return None, "absolute_basis_mismatch"
        if scope == REFERENCE_SCOPE_ENDPOINT_RATIO and basis not in ENDPOINT_RATIO_BASES:
            return None, "endpoint_ratio_basis_mismatch"
        if scope == REFERENCE_SCOPE_STANDARD_CONTROL and basis not in STANDARD_CONTROL_BASES:
            return None, "standard_control_basis_mismatch"
        if scope == REFERENCE_SCOPE_UNKNOWN and basis != "unknown":
            return None, "unknown_basis_mismatch"
    model_method = "".join(
        character if character.isalnum() else "_" for character in model.casefold()
    ).strip("_")
    method_parts = [f"{model_method}_single_pass"]
    if config.labels_only_output:
        method_parts.append("labels_only")
    if normalized_field:
        method_parts.append("normalized_evidence_field")
    if normalized_quote_field:
        method_parts.append("rerouted_exact_evidence_quote")
    output = {
        "cleaned_record_id": str(row["id"]),
        "source_id": str(row["source_id"]),
        "reference_scope": scope,
        "assignment_method": ":".join(method_parts),
        "evidence_field": field,
        "evidence_quote": quote,
    }
    if config.output_basis:
        assert basis is not None
        output["reference_basis"] = basis
    return output, None


def _query_batch(
    batch: RequestBatch,
    *,
    config: ReferenceSemanticsConfig,
    client: Any | None = None,
    model: str = MODEL,
    base_url: str = DEFAULT_BASE_URL,
    credential_env: str = "OPENAI_API_KEY",
    llm: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int] | None, str]:
    user = json.dumps({"rows": batch.api_rows}, ensure_ascii=False, sort_keys=True)
    provenance = {} if llm is not None else {
        "inference_model": model,
        "inference_base_url": base_url,
        "inference_credential_env": credential_env,
    }
    try:
        if llm is not None:
            result = llm(
                {"system": batch.prompt, "user": user},
                model=model,
                max_tokens=batch.max_completion_tokens,
                temperature=1.0,
                verbose=False,
                reasoning_effort=REASONING_EFFORT,
                max_retries=1,
            )
        else:
            if client is None:
                raise ValueError("an OpenAI-compatible client is required")
            parameters: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": batch.prompt},
                    {"role": "user", "content": user},
                ],
                "max_completion_tokens": batch.max_completion_tokens,
                "reasoning_effort": REASONING_EFFORT,
            }
            if "openrouter.ai" in base_url:
                parameters["extra_body"] = {
                    "provider": {"require_parameters": True}
                }
            response = client.chat.completions.create(**parameters)
            result = {
                "content": response.choices[0].message.content,
                "usage": response.usage,
            }
    except Exception:
        return (
            [
                {
                    **_unknown_assignment(row, config=config, method="api_failure"),
                    **provenance,
                }
                for row in batch.rows
            ],
            None,
            "api_failure",
        )
    usage = _usage(result)
    content = result.get("content")
    try:
        parsed = json.loads(str(content or ""))
        returned = parsed.get("rows")
        if not isinstance(returned, list):
            raise ValueError("response rows is not a list")
    except (json.JSONDecodeError, ValueError, TypeError):
        return (
            [
                {
                    **_unknown_assignment(
                        row, config=config, method="invalid_response"
                    ),
                    **provenance,
                }
                for row in batch.rows
            ],
            usage,
            "invalid_response",
        )
    by_id: dict[str, list[Mapping[str, Any]]] = {}
    non_object_rows = 0
    for item in returned:
        if not isinstance(item, Mapping):
            non_object_rows += 1
            continue
        by_id.setdefault(str(item.get("id") or ""), []).append(item)
    expected_ids = {str(row["id"]) for row in batch.api_rows}
    extra_ids = set(by_id) - expected_ids
    assignments: list[dict[str, Any]] = []
    invalid = non_object_rows + len(extra_ids)
    for row, api_row in zip(batch.rows, batch.api_rows, strict=True):
        model_id = str(api_row["id"])
        matches = by_id.get(model_id) or []
        returned_item = matches[0] if len(matches) == 1 else {}
        if len(matches) != 1:
            invalid += 1
        valid, rejection_reason = _validate_assignment(
            returned_item,
            row,
            config=config,
            model_id=model_id,
            model=model,
        )
        if valid is None:
            invalid += 1
            valid = {
                **_unknown_assignment(
                    row,
                    config=config,
                    method=f"invalid_row_response:{rejection_reason}",
                ),
                **provenance,
                "rejected_response_json": json.dumps(
                    returned_item, ensure_ascii=False, sort_keys=True, default=str
                ),
            }
        assignments.append({**valid, **provenance})
    status = "valid" if invalid == 0 else "partial_invalid"
    return assignments, usage, status


def _generation_no_call_assignment(
    candidate: Mapping[str, Any],
) -> ReferenceAssignment | None:
    assignment = candidate.get("_generation_no_call_assignment")
    if assignment is None:
        return None
    if not isinstance(assignment, ReferenceAssignment):
        raise TypeError("generation no-call policy returned a non-assignment")
    return assignment


def _assignment_row(
    candidate: Mapping[str, Any], assignment: ReferenceAssignment
) -> dict[str, Any]:
    return {
        "cleaned_record_id": str(candidate["id"]),
        "source_id": str(candidate["source_id"]),
        "reference_scope": assignment.scope,
        "reference_basis": assignment.basis,
        "assignment_method": assignment.method,
        "evidence_field": None,
        "evidence_quote": None,
    }


def _reconcile_cached_assignment(
    candidate: Mapping[str, Any],
    cached: Mapping[str, Any],
    *,
    config: ReferenceSemanticsConfig,
) -> dict[str, Any]:
    method = str(cached.get("assignment_method") or "")
    if method == "invalid_row_response:comparator_basis_mismatch":
        rejected = json.loads(str(cached["rejected_response_json"]))
        recovered, _ = _validate_assignment(
            rejected,
            candidate,
            config=config,
            model_id=str(rejected.get("id") or ""),
            model=str(cached.get("inference_model") or MODEL),
        )
        if recovered is not None:
            return {
                **recovered,
                **{
                    key: value
                    for key, value in cached.items()
                    if key.startswith("inference_")
                },
            }
    if "normalized_scope_from_basis" in method:
        return {
            **_unknown_assignment(
                candidate, config=config, method="discarded_semantic_override"
            ),
            "prior_assignment_json": json.dumps(
                cached, ensure_ascii=False, sort_keys=True, default=str
            ),
        }
    gate = _generation_no_call_assignment(candidate)
    if gate is None or str(cached.get("reference_scope") or "") == REFERENCE_SCOPE_UNKNOWN:
        return dict(cached)
    cached_scope = str(cached.get("reference_scope") or "")
    cached_basis = str(cached.get("reference_basis") or "")
    basis_conflict = (
        gate.scope
        in {
            REFERENCE_SCOPE_ABSOLUTE,
            REFERENCE_SCOPE_ENDPOINT_RATIO,
            REFERENCE_SCOPE_STANDARD_CONTROL,
        }
        and gate.basis is not None
        and cached_basis != gate.basis
    )
    if cached_scope == gate.scope and not basis_conflict:
        return dict(cached)
    return {
        **_unknown_assignment(
            candidate,
            config=config,
            method=f"safe_gate_conflict:{gate.method}",
        ),
        "safe_gate_scope": gate.scope,
        "safe_gate_basis": gate.basis,
        "prior_assignment_json": json.dumps(
            cached, ensure_ascii=False, sort_keys=True, default=str
        ),
    }


def _write_progress(
    path: Path,
    *,
    config: ReferenceSemanticsConfig,
    candidates: Sequence[Mapping[str, Any]],
    cache: SubmissionCache,
    ledger: TokenLedger,
    completed: bool,
    model: str = MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> None:
    payload = {
        "generation_version": GENERATION_VERSION,
        "task_id": config.task_id,
        "prompt_version": config.prompt_version,
        "model": model,
        "base_url": base_url,
        "reasoning_effort": REASONING_EFFORT,
        "batch_size": config.batch_size,
        "candidate_rows": len(candidates),
        "attempted_rows": len(set(str(row["id"]) for row in candidates) & cache.attempted),
        "terminal_rows": len(cache.assignments),
        "no_call_rows": sum(
            _generation_no_call_assignment(row) is not None for row in candidates
        ),
        "unattempted_api_rows": sum(
            str(row["id"]) not in cache.attempted
            and _generation_no_call_assignment(row) is None
            for row in candidates
        ),
        "completed": completed,
        "budget_epoch": ledger.epoch,
        "budget_spent_tokens": ledger.spent(),
        "budget_max_tokens": ledger.max_tokens,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _materialize_mapping(
    *,
    config: ReferenceSemanticsConfig,
    candidates: Sequence[Mapping[str, Any]],
    cache: SubmissionCache,
    base_mapping_path: Path | None = None,
) -> dict[str, Any]:
    base_rows: list[dict[str, Any]] = []
    if base_mapping_path is not None:
        if not base_mapping_path.is_file():
            raise FileNotFoundError(f"base reference mapping not found: {base_mapping_path}")
        base_rows = pd.read_parquet(base_mapping_path).to_dict("records")
    base_ids = {str(row.get("cleaned_record_id") or "") for row in base_rows}
    candidate_ids = {str(row["id"]) for row in candidates}
    overlap = base_ids & candidate_ids
    if overlap:
        raise ValueError(
            f"base and supplemental mappings overlap on {len(overlap)} record IDs"
        )
    rows = list(base_rows)
    for candidate in candidates:
        record_id = str(candidate["id"])
        cached = cache.assignments.get(record_id)
        if cached is not None:
            assignment = _reconcile_cached_assignment(
                candidate, cached, config=config
            )
        elif record_id in cache.attempted:
            assignment = {
                **_unknown_assignment(
                    candidate, config=config, method="ambiguous_process_termination"
                ),
                **cache.provenance.get(record_id, {}),
            }
        else:
            no_call = _generation_no_call_assignment(candidate)
            if no_call is None:
                raise ValueError(f"unattempted row cannot be published: {record_id}")
            assignment = _assignment_row(candidate, no_call)
        rows.append(assignment)
    rows.sort(key=lambda row: str(row["cleaned_record_id"]))
    config.mapping_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.mapping_path.with_suffix(config.mapping_path.suffix + ".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    os.replace(temporary, config.mapping_path)
    scope_counts = Counter(str(row["reference_scope"]) for row in rows)
    method_counts = Counter(str(row["assignment_method"]) for row in rows)
    model_counts = Counter(
        str(row["inference_model"])
        for row in rows
        if row.get("inference_model")
    )
    endpoint_counts = Counter(
        str(row["inference_base_url"])
        for row in rows
        if row.get("inference_base_url")
    )
    manifest = {
        "generation_version": GENERATION_VERSION,
        "task_id": config.task_id,
        "prompt_version": config.prompt_version,
        "prompt_sha256": file_sha256(config.prompt_registry_path),
        "input_path": str(config.canonical_records_path),
        "input_sha256": file_sha256(config.canonical_records_path),
        "mapping_path": str(config.mapping_path),
        "mapping_sha256": file_sha256(config.mapping_path),
        "mapping_rows": len(rows),
        "base_mapping_path": str(base_mapping_path) if base_mapping_path else None,
        "base_mapping_sha256": (
            file_sha256(base_mapping_path) if base_mapping_path else None
        ),
        "base_mapping_rows": len(base_rows),
        "supplement_mapping_rows": len(candidates),
        "credential_envs": sorted(
            {
                str(event["credential_env"])
                for event in cache.events
                if event.get("credential_env")
            }
        ),
        "inference_model_counts": dict(sorted(model_counts.items())),
        "inference_endpoint_counts": dict(sorted(endpoint_counts.items())),
        "scope_counts": dict(sorted(scope_counts.items())),
        "assignment_method_counts": dict(sorted(method_counts.items())),
        "safe_gate_conflicts": sum(
            str(row["assignment_method"]).startswith("safe_gate_conflict:")
            for row in rows
        ),
        "validations": {
            "one_mapping_per_candidate": len(rows)
            == len(base_rows) + len(candidates),
            "unique_cleaned_record_ids": len(rows)
            == len({row["cleaned_record_id"] for row in rows}),
            "each_row_submitted_at_most_once": True,
        },
    }
    manifest_path = config.mapping_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def run(args: argparse.Namespace) -> int:
    config = _load_config(args.task)
    if args.records:
        config = replace(config, canonical_records_path=Path(args.records))
    if args.mapping:
        config = replace(config, mapping_path=Path(args.mapping))
    if config.batch_size < 1:
        raise ValueError("reference-semantics batch size must be positive")
    candidates = _load_candidate_rows(
        config.canonical_records_path,
        config,
        only_assignment_method=args.only_assignment_method,
    )
    by_source = Counter(str(row["source_id"]) for row in candidates)
    cache_path = Path(args.cache_dir) / config.task_id / "requests.jsonl"
    cache = SubmissionCache(cache_path)
    unattempted_no_call = Counter(
        assignment.method
        for row in candidates
        if str(row["id"]) not in cache.attempted
        if (assignment := _generation_no_call_assignment(row)) is not None
    )
    planned_batches = _batches(
        candidates,
        config=config,
        prompt="",
        attempted=cache.attempted,
        model=args.model,
        base_url=args.base_url,
    )
    print(
        json.dumps(
            {
                "task": config.task_id,
                "only_assignment_method": args.only_assignment_method,
                "candidate_rows": len(candidates),
                "candidate_rows_by_source": dict(sorted(by_source.items())),
                "attempted_rows": len(
                    {str(row["id"]) for row in candidates} & cache.attempted
                ),
                "terminal_rows": len(cache.assignments),
                "unattempted_no_call_rows": sum(unattempted_no_call.values()),
                "unattempted_no_call_method_counts": dict(
                    sorted(unattempted_no_call.items())
                ),
                "unattempted_api_rows": sum(
                    len(batch.rows) for batch in planned_batches
                ),
                "planned_requests": len(planned_batches),
                "batch_size": config.batch_size,
                "prompt_fields": list(config.prompt_fields),
                "labels_only_output": config.labels_only_output,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if args.inventory_only:
        return 0
    if not args.budget_epoch:
        raise ValueError("--budget-epoch is required for paid generation")
    prompt = _load_prompt(config)
    ledger = TokenLedger(
        Path(args.token_ledger),
        epoch=args.budget_epoch,
        start_new_epoch=args.start_new_budget_epoch,
        max_tokens=args.budget_max_tokens,
    )
    batches = [replace(batch, prompt=prompt) for batch in planned_batches]
    limited = args.max_requests is not None and len(batches) > args.max_requests
    if args.max_requests is not None:
        batches = batches[: args.max_requests]
    client, selected_credential_env = _load_llm(
        api_key_env=args.api_key_env,
        env_file=Path(args.env_file) if args.env_file else None,
        base_url=args.base_url,
        workers=args.workers,
        provider=args.provider,
    )
    pending = iter(batches)
    in_flight: dict[Any, RequestBatch] = {}
    exhausted = False
    next_batch: RequestBatch | None = None
    pending_finished = False
    progress_path = config.mapping_path.with_suffix(".generation_progress.json")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while True:
            while len(in_flight) < args.workers and not pending_finished:
                if next_batch is None:
                    try:
                        next_batch = next(pending)
                    except StopIteration:
                        pending_finished = True
                        break
                if not ledger.reserve(
                    next_batch.request_id, next_batch.reservation_tokens
                ):
                    break
                cache.submit(
                    next_batch,
                    epoch=ledger.epoch,
                    credential_env=selected_credential_env,
                    model=args.model,
                    base_url=args.base_url,
                )
                future = executor.submit(
                    _query_batch,
                    next_batch,
                    config=config,
                    client=client,
                    model=args.model,
                    base_url=args.base_url,
                    credential_env=selected_credential_env,
                )
                in_flight[future] = next_batch
                next_batch = None
            if not in_flight:
                exhausted = next_batch is not None
                break
            done, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
            for future in done:
                batch = in_flight.pop(future)
                try:
                    assignments, usage, status = future.result()
                except Exception:
                    assignments = [
                        {
                            **_unknown_assignment(
                                row, config=config, method="worker_failure"
                            ),
                            "inference_model": args.model,
                            "inference_base_url": args.base_url,
                            "inference_credential_env": selected_credential_env,
                        }
                        for row in batch.rows
                    ]
                    usage = None
                    status = "worker_failure"
                ledger.complete(batch.request_id, usage)
                cache.terminal(
                    batch,
                    assignments=assignments,
                    usage=usage,
                    response_status=status,
                )
            _write_progress(
                progress_path,
                config=config,
                candidates=candidates,
                cache=cache,
                ledger=ledger,
                completed=False,
                model=args.model,
                base_url=args.base_url,
            )
    if exhausted:
        ledger.mark_exhausted()
        _write_progress(
            progress_path,
            config=config,
            candidates=candidates,
            cache=cache,
            ledger=ledger,
            completed=False,
            model=args.model,
            base_url=args.base_url,
        )
        return BUDGET_EXHAUSTED_EXIT_CODE
    if limited:
        _write_progress(
            progress_path,
            config=config,
            candidates=candidates,
            cache=SubmissionCache(cache_path),
            ledger=ledger,
            completed=False,
            model=args.model,
            base_url=args.base_url,
        )
        return 0
    # Re-read terminal events written by worker completion before publication.
    cache = SubmissionCache(cache_path)
    manifest = _materialize_mapping(
        config=config,
        candidates=candidates,
        cache=cache,
        base_mapping_path=Path(args.base_mapping) if args.base_mapping else None,
    )
    _write_progress(
        progress_path,
        config=config,
        candidates=candidates,
        cache=cache,
        ledger=ledger,
        completed=True,
        model=args.model,
        base_url=args.base_url,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("bbb_martins", "bioavailability_ma", "skin_reaction"), required=True
    )
    parser.add_argument("--records")
    parser.add_argument("--mapping")
    parser.add_argument("--base-mapping")
    parser.add_argument("--only-assignment-method")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument(
        "--provider", choices=("openai", "openrouter", "local"), default=None
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--budget-epoch")
    parser.add_argument("--start-new-budget-epoch", action="store_true")
    parser.add_argument(
        "--budget-max-tokens",
        type=int,
        default=MAX_EPOCH_TOKENS,
        help="Combined input/output token ceiling for this key epoch.",
    )
    parser.add_argument("--token-ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument(
        "--cache-dir",
        default="outputs/chembl_tool/reference_semantics_generation/cache",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.budget_max_tokens < 1:
        parser.error("--budget-max-tokens must be positive")
    if args.max_requests is not None and args.max_requests < 1:
        parser.error("--max-requests must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
