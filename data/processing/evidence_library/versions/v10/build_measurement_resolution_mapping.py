"""Generate one frozen, endpoint-aware measurement extraction per routed row.

Categorical encoding is independent of this router. Exact scalar pairs and
deterministic scalar rejections are settled before this offline pass; only the
remaining ``extract`` rows reach the model. Responses are
cached before submission, structurally invalid batches are retried once, and no
mapping publishes until every candidate has a terminal result. Paid endpoints may
use the inherited token ledger; uncapped compatible endpoints use
``--no-token-ledger`` while retaining API usage in the response cache.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
import time
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.shared.v2.normalization.measurement_resolution import (
    load_exact_unit_mapping,
)
from data.processing.evidence_library.versions.v10.build_endpoint_unit_profile import (
    PROFILE_VERSION,
    SUPPORTED_TASKS,
    profile_lookup,
    render_profile_block,
)
from data.processing.evidence_library.versions.v10.build_reference_semantics_mapping import (
    BUDGET_EXHAUSTED_EXIT_CODE,
    CACHE_VERSION,
    DISTILLATION_ROOT,
    RequestBatch,
    SubmissionCache,
    TokenLedger,
)
from data.processing.evidence_library.versions.v10.measurement_routing import (
    MEASUREMENT_ROUTING_VERSION,
    ROUTE_BUCKETS,
    route,
)
from data.processing.evidence_library.versions.v10.task_registry import (
    import_task_module,
)
from data.processing.llm_api import (
    DEFAULT_ENV_FILE,
    openai_compatible_client,
    provider_from_base_url,
)

GENERATION_VERSION = "measurement_resolution_generation.v8"
ID_TRANSPORT_VERSION = "batch_local_ids.v1"
RETRY_LIMIT_EXIT_CODE = 4
MODEL = "gpt-5.4-mini"
REASONING_EFFORT = "low"
REVIEWED_SOURCE_INFERENCE = "reviewed_source_decision"
REVIEWED_SOURCE_MODEL = "manual-source-review.v1"
REVIEWED_SOURCE_BASE_URL = "review://source-grounded"

#: The distillation helper lives beside the TxAgent checkout, and TxAgent has more
#: than one checkout -- so resolve it relative to this repository first and treat the
#: node002 path as one candidate rather than the truth.  Hardcoding either location
#: breaks the other host.
DISTILLATION_CANDIDATES = (DISTILLATION_ROOT,)

#: Fixed so a pilot slice is reproducible and never silently re-drawn.
SAMPLE_SEED = 20260820
STATUSES = ("ok", "unsure", "relative", "unavailable")
MAX_COMPLETION_TOKENS = 65_536
OPENROUTER_RETRY_DELAYS_S = (1, 2, 4, 8, 16) + (30,) * 20
CREDENTIAL_UNAVAILABLE_EXIT_CODE = 76
PHASE_BUDGET_EXHAUSTED_EXIT_CODE = 74

#: Row-local columns shown to the model. Endpoint context is supplied separately
#: from deterministically parsed rows and never changes this row payload.
PROMPT_FIELDS = ("endpoint_name", "measurement_text", "unit_text", "support_text")


class TaskConfig:
    """The per-task facts this generator needs, resolved from the task module."""

    def __init__(self, task_id: str, module_name: str | None = None):
        self.task_id = task_id
        self.module = (
            importlib.import_module(module_name)
            if module_name
            else import_task_module(task_id, "starling_measurement_resolution")
        )
        for name in (
            "BATCH_SIZE",
            "DEFAULT_CLEANED_RECORDS",
            "DEFAULT_MAPPING_PATH",
            "DEFAULT_PROFILE_PATH",
            "MAPPING_VERSION",
            "PROMPT_VERSION",
            "SOURCE_IDS",
            "canonical_endpoint_name",
            "prompt_manifest",
            "prompt_row_fields",
            "render_prompt",
            "source_routing_rules",
        ):
            if not hasattr(self.module, name):
                raise TypeError(f"{task_id!r} measurement-resolution config lacks {name}")
        self.rules = self.module.source_routing_rules()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.module, name)


GOLD_FIXTURE = Path(
    "tests/chembl_tool/common/measurement_resolution_quality/gold/"
    "bbb_martins.v8.jsonl"
)


class MissingGoldRows(ValueError):
    """Requested gold rows that the current rules do not send to extraction."""

    def __init__(self, absent: list[str], requested: int):
        self.absent = absent
        self.requested = requested
        super().__init__(
            f"{len(absent)} of {requested} requested id(s) are not extraction "
            f"candidates; first={absent[0]}"
        )


def _candidate_columns(
    records_path: Path, config: TaskConfig
) -> tuple[list[str], list[str]]:
    available = set(pq.read_schema(records_path).names)
    wanted = {
        "cleaned_record_id",
        "source_row_uid",
        "source_id",
        "canonical_endpoint_name",
        "measurement_resolution_route",
        "measurement_resolution_rule_id",
        "measurement_resolution_exact_measurement",
        "measurement_resolution_exact_unit",
        "measurement_resolution_exact_unit_is_canonical",
        *PROMPT_FIELDS,
    }
    for rules in config.rules.values():
        wanted.add(rules.measurement_field)
        if rules.unit_field:
            wanted.add(rules.unit_field)
        wanted.update(rule.field for rule in rules.declarative_non_scalar)
    for source_id in config.SOURCE_IDS:
        wanted.update(config.prompt_row_fields(source_id))
    return sorted(wanted & available), sorted(wanted - available)


def _candidate_route(
    record: dict[str, Any], source_id: str, config: TaskConfig
) -> str:
    rules = config.rules.get(source_id)
    if rules is None:
        raise ValueError(f"no routing rules for source {source_id!r}")
    persisted = record.get("measurement_resolution_route")
    if persisted is not None and str(persisted) not in ROUTE_BUCKETS:
        raise ValueError(f"unsupported persisted measurement route {persisted!r}")
    validator = getattr(config.module, "validate_persisted_measurement_route", None)
    if validator is not None and persisted is not None:
        validator(record, str(persisted))
    if persisted is not None:
        return str(persisted)
    if hasattr(config.module, "route_measurement"):
        return config.module.route_measurement(record).bucket
    return route(record, rules, task=config.task_id).bucket


def _candidate_payload(
    record: dict[str, Any], source_id: str, config: TaskConfig
) -> dict[str, Any]:
    record_id = str(record.get("cleaned_record_id") or "")
    if not record_id:
        raise ValueError("extraction candidate lacks cleaned_record_id")
    if getattr(config, "REQUIRE_SOURCE_ROW_UID", False) and not record.get(
        "source_row_uid"
    ):
        raise ValueError(f"extraction candidate lacks source_row_uid: {record_id}")
    resolver = getattr(config.module, "canonical_endpoint_record", None)
    endpoint = (
        str(record["canonical_endpoint_name"])
        if record.get("canonical_endpoint_name")
        else (
            resolver(record)
            if resolver is not None
            else config.canonical_endpoint_name(source_id, record.get("endpoint_name"))
        )
    )
    validator = getattr(config.module, "validate_candidate_endpoint_identity", None)
    if validator is not None:
        validator(record, endpoint)
    payload = {
        "id": record_id,
        "source_row_uid": record.get("source_row_uid"),
        "source_id": source_id,
        "canonical_endpoint_name": endpoint,
    }
    for field in config.prompt_row_fields(source_id):
        payload[field] = _json_value(record.get(field))
    return payload


def _read_candidate_rows(
    records_path: Path,
    columns: list[str],
    config: TaskConfig,
    source_id: str | None,
    only_ids: set[str] | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    parquet = pq.ParquetFile(records_path)
    for batch in parquet.iter_batches(batch_size=50_000, columns=columns):
        for record in batch.to_pylist():
            row_source = str(record.get("source_id") or "")
            if source_id is not None and row_source != source_id:
                continue
            if row_source not in config.SOURCE_IDS:
                continue
            if _candidate_route(record, row_source, config) != "extract":
                continue
            record_id = str(record.get("cleaned_record_id") or "")
            if only_ids is not None and record_id not in only_ids:
                continue
            candidates.append(_candidate_payload(record, row_source, config))
    return candidates


def gold_extract_ids(path: Path = GOLD_FIXTURE) -> list[str]:
    """The hand-labelled rows that reach the model under current routing.

    ``accept`` and ``reject`` cases are settled by rule and never sent, so replaying
    them would spend tokens on deterministic rows. Gold labels remain valid when a
    later router moves a plain decimal from extraction to the exact source path.
    """
    lines = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest, cases = lines[0], lines[1:]
    config = TaskConfig(str(manifest["task_id"]))
    return [
        case["audit_case_id"].split(":", 1)[1]
        for case in cases
        if (
            str(case["route_bucket"])
            if case.get("route_bucket") is not None
            else (
                config.module.route_measurement(
                    {**case["input"], "source_id": case["source_id"]}
                )
                if hasattr(config.module, "route_measurement")
                else route(
                    case["input"],
                    config.rules[case["source_id"]],
                    task=config.task_id,
                )
            ).bucket
        )
        == "extract"
    ]


def candidate_rows(
    records_path: Path,
    config: TaskConfig,
    *,
    source_id: str | None = None,
    limit: int | None = None,
    only_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the cleaned rows routed to model extraction."""
    columns, missing = _candidate_columns(records_path, config)
    candidates = _read_candidate_rows(
        records_path, columns, config, source_id, only_ids
    )
    candidates.sort(
        key=lambda row: (
            row["source_id"],
            row["canonical_endpoint_name"],
            row["id"],
        )
    )
    if only_ids is not None:
        found = {row["id"] for row in candidates}
        absent = only_ids - found
        if absent:
            raise MissingGoldRows(sorted(absent), len(only_ids))
    if missing:
        print(f"  note: columns requested but absent: {missing}")
    return _stratified_slice(candidates, limit) if limit else candidates


def _stratified_slice(
    candidates: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Take ``limit`` rows spread across sources, not the head of the list.

    A contiguous slice of a list sorted by source would be one source only, which
    for a pilot means never exercising the other prompt variant -- sources without a
    unit column get different instructions, and that branch would go untested.
    Sources are drawn round-robin so a small limit still touches every one.
    """
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_source.setdefault(row["source_id"], []).append(row)
    rng = random.Random(SAMPLE_SEED)
    pools = {}
    for source_id, rows in sorted(by_source.items()):
        pools[source_id] = rng.sample(rows, len(rows))
    output: list[dict[str, Any]] = []
    while len(output) < limit and any(pools.values()):
        for source_id in sorted(pools):
            if not pools[source_id] or len(output) >= limit:
                continue
            output.append(pools[source_id].pop())
    output.sort(
        key=lambda row: (
            row["source_id"],
            row.get("canonical_endpoint_name", ""),
            row["id"],
        )
    )
    return output


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _group_candidate_endpoints(
    candidates: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in candidates:
        endpoint = str(row.get("canonical_endpoint_name") or "")
        if not endpoint:
            raise ValueError(f"candidate {row['id']!r} lacks canonical endpoint")
        grouped.setdefault(row["source_id"], {}).setdefault(endpoint, []).append(row)
    return grouped


def _endpoint_requests(
    endpoint_rows: dict[str, list[dict[str, Any]]], batch_size: int
) -> list[list[tuple[str, list[dict[str, Any]]]]]:
    requests: list[list[tuple[str, list[dict[str, Any]]]]] = []
    small: list[tuple[str, list[dict[str, Any]]]] = []
    small_size = 0
    for endpoint, rows in sorted(endpoint_rows.items()):
        rows.sort(key=lambda row: row["id"])
        if len(rows) >= batch_size:
            requests.extend(
                [(endpoint, rows[offset : offset + batch_size])]
                for offset in range(0, len(rows), batch_size)
            )
            continue
        if small and small_size + len(rows) > batch_size:
            requests.append(small)
            small, small_size = [], 0
        small.append((endpoint, rows))
        small_size += len(rows)
    if small:
        requests.append(small)
    return requests


def _batch_digest(
    endpoint_groups: list[tuple[str, list[dict[str, Any]]]],
    config: TaskConfig,
    source_id: str,
    prompt: str,
    profile_digest: str,
    model: str,
    max_completion_tokens: int,
) -> str:
    composition = [
        {"canonical_endpoint_name": endpoint, "row_ids": [row["id"] for row in rows]}
        for endpoint, rows in endpoint_groups
    ]
    identity = {
        "task": config.task_id,
        "id_transport_version": ID_TRANSPORT_VERSION,
        "source": source_id,
        "prompt_version": config.PROMPT_VERSION,
        "model": model,
        "max_completion_tokens": max_completion_tokens,
        "profile_digest": profile_digest,
        "endpoint_groups": composition,
        "rendered_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }
    if hasattr(config.module, "REASONING_EFFORT"):
        identity["reasoning_effort"] = config.REASONING_EFFORT
    if hasattr(config.module, "TEMPERATURE"):
        identity["temperature"] = float(config.module.TEMPERATURE)
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _profile_blocks(
    endpoint_groups: list[tuple[str, list[dict[str, Any]]]],
    profiles: dict[tuple[str, str], Any],
    source_id: str,
    chunk_ids: set[str],
) -> tuple[str, ...]:
    blocks = []
    for endpoint, _ in endpoint_groups:
        entry = profiles.get((source_id, endpoint))
        blocks.append(
            render_profile_block(entry, exclude_ids=chunk_ids)
            if entry is not None
            else f"[endpoint: {endpoint} - no reliable deterministic summary]"
        )
    return tuple(blocks)


def _wire_payload(chunk: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    omitted = {"source_id", "source_row_uid", "id"}
    return tuple(
        {**{key: row[key] for key in row if key not in omitted}, "id": f"r{index}"}
        for index, row in enumerate(chunk, 1)
    )


def _make_request_batch(
    endpoint_groups: list[tuple[str, list[dict[str, Any]]]],
    config: TaskConfig,
    source_id: str,
    attempted: set[str],
    profiles: dict[tuple[str, str], Any],
    profile_digest: str,
    model: str,
    max_completion_tokens: int,
) -> RequestBatch | None:
    chunk = tuple(dict(row) for _, rows in endpoint_groups for row in rows)
    chunk_ids = {row["id"] for row in chunk}
    overlap = chunk_ids & attempted
    if overlap and overlap != chunk_ids:
        raise ValueError(
            "submission cache contains only part of deterministic request "
            f"{sorted(chunk_ids)}"
        )
    if overlap:
        return None
    prompt = config.render_prompt(
        source_id,
        batch_size=config.BATCH_SIZE,
        endpoint_profiles=_profile_blocks(
            endpoint_groups, profiles, source_id, chunk_ids
        ),
    )
    digest = _batch_digest(
        endpoint_groups,
        config,
        source_id,
        prompt,
        profile_digest,
        model,
        max_completion_tokens,
    )
    return RequestBatch(
        request_id=f"{config.task_id}_{source_id}_{digest}",
        source_id=source_id,
        rows=chunk,
        prompt=prompt,
        max_completion_tokens=max_completion_tokens,
        payload_rows=_wire_payload(chunk),
    )


def _stratify_request_batches(batches: list[RequestBatch]) -> list[RequestBatch]:
    by_source: dict[str, list[RequestBatch]] = {}
    for batch in batches:
        by_source.setdefault(batch.source_id, []).append(batch)
    ranked: list[tuple[float, str, int, RequestBatch]] = []
    for source_id, source_batches in sorted(by_source.items()):
        by_endpoint: dict[tuple[str, ...], list[RequestBatch]] = {}
        for batch in source_batches:
            endpoint_key = tuple(
                dict.fromkeys(str(row["canonical_endpoint_name"]) for row in batch.rows)
            )
            by_endpoint.setdefault(endpoint_key, []).append(batch)
        queues = [by_endpoint[key] for key in sorted(by_endpoint)]
        interleaved: list[RequestBatch] = []
        while queues:
            interleaved.extend(queue.pop(0) for queue in queues)
            queues = [queue for queue in queues if queue]
        total = len(interleaved)
        ranked.extend(
            (index / total, source_id, index, batch)
            for index, batch in enumerate(interleaved)
        )
    return [item[-1] for item in sorted(ranked, key=lambda item: item[:3])]


def plan_batches(
    candidates: list[dict[str, Any]],
    config: TaskConfig,
    *,
    attempted: set[str],
    endpoint_profile: dict[str, Any] | None = None,
    profile_digest: str = "",
    model: str = MODEL,
    max_completion_tokens: int = MAX_COMPLETION_TOKENS,
) -> list[RequestBatch]:
    """Group rows by canonical endpoint without ever crossing a source."""
    if getattr(config.module, "ALLOW_REBATCH_UNATTEMPTED", False):
        candidates = [row for row in candidates if str(row["id"]) not in attempted]
    profiles = profile_lookup(endpoint_profile or {"endpoints": {}})
    by_source = _group_candidate_endpoints(candidates)
    batches: list[RequestBatch] = []
    for source_id in sorted(by_source):
        for groups in _endpoint_requests(by_source[source_id], config.BATCH_SIZE):
            batch = _make_request_batch(
                groups,
                config,
                source_id,
                attempted,
                profiles,
                profile_digest,
                model,
                max_completion_tokens,
            )
            if batch is not None:
                batches.append(batch)
    if not getattr(config.module, "STRATIFY_BATCHES", False):
        return batches
    return _stratify_request_batches(batches)


def _blank(row: Any, *, method: str, reason: str | None = None) -> dict[str, Any]:
    return {
        "cleaned_record_id": str(row["id"]),
        "source_id": str(row["source_id"]),
        "status": "unsure",
        "measurements_json": "[]",
        "quantity_count": 0,
        "assignment_method": method,
        "rejected_response_json": reason,
        "raw_response_json": None,
    }


def is_plain_decimal(measurement: str) -> bool:
    """Accept one finite decimal coefficient, never a bound, range, or exponent."""
    if "e" in measurement.lower() or any(
        character in measurement for character in "<>/×*_"
    ):
        return False
    try:
        value = Decimal(measurement)
    except InvalidOperation:
        return False
    return value.is_finite()


def _refuse(row: Any, returned: Any, *, method: str, reason: str) -> tuple[dict[str, Any], str]:
    """One refusal path, so the raw answer is never lost on any of them."""
    assignment = _blank(row, method=f"invalid_row_response:{method}")
    assignment["raw_response_json"] = json.dumps(
        returned, ensure_ascii=False, default=str
    )
    return assignment, reason


def _response_envelope(
    returned: Any, row: Any, max_measurements: int | None
) -> tuple[str | None, list[Any], tuple[dict[str, Any], str] | None]:
    if not isinstance(returned, dict):
        return None, [], _refuse(
            row, returned, method="not_an_object", reason="not_an_object"
        )
    if str(returned.get("id") or "") != str(row["id"]):
        return None, [], _refuse(
            row, returned, method="id_mismatch", reason="id_mismatch"
        )
    status = str(returned.get("status") or "")
    if status not in STATUSES:
        failure = _refuse(
            row,
            returned,
            method="unsupported_status",
            reason=f"unsupported_status:{status!r}",
        )
        return None, [], failure
    entries = returned.get("measurements")
    entries = [] if entries is None else entries
    if not isinstance(entries, list):
        failure = _refuse(
            row,
            returned,
            method="measurements_not_a_list",
            reason="measurements_not_a_list",
        )
        return None, [], failure
    if status != "ok" and entries:
        failure = _refuse(
            row,
            returned,
            method="non_ok_carries_measurements",
            reason=f"non_ok_carries_measurements:{status}",
        )
        return None, [], failure
    if status == "ok" and not entries:
        failure = _refuse(
            row, returned, method="ok_without_measurements", reason="ok_without_measurements"
        )
        return None, [], failure
    if max_measurements is not None and len(entries) > max_measurements:
        failure = _refuse(
            row,
            returned,
            method="too_many_measurements",
            reason=f"too_many_measurements:{len(entries)}>{max_measurements}",
        )
        return None, [], failure
    return status, entries, None


def _clean_measurements(
    entries: list[Any], row: Any, returned: Any
) -> tuple[list[dict[str, str]] | None, tuple[dict[str, Any], str] | None]:
    cleaned: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            failure = _refuse(
                row, returned, method="entry_not_an_object", reason="entry_not_an_object"
            )
            return None, failure
        measurement = str(entry.get("measurement") or "").strip()
        unit = str(entry.get("unit") or "").strip()
        if not measurement:
            failure = _refuse(
                row, returned, method="empty_measurement", reason="empty_measurement"
            )
            return None, failure
        if not unit:
            failure = _refuse(
                row, returned, method="entry_without_a_unit", reason="entry_without_a_unit"
            )
            return None, failure
        if not is_plain_decimal(measurement):
            failure = _refuse(
                row,
                returned,
                method="measurement_is_not_a_plain_decimal",
                reason=f"measurement_is_not_a_plain_decimal:{measurement!r}",
            )
            return None, failure
        cleaned.append({"measurement": measurement, "unit": unit})
    return cleaned, None


def validate_row(
    returned: Any,
    row: Any,
    *,
    task: str,
    max_measurements: int | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Check one returned row against the schema, failing closed on errors."""
    status, entries, failure = _response_envelope(returned, row, max_measurements)
    if failure is not None:
        return failure
    cleaned, failure = _clean_measurements(entries, row, returned)
    if failure is not None:
        return failure
    assert status is not None and cleaned is not None
    return (
        {
            "cleaned_record_id": str(row["id"]),
            "source_id": str(row["source_id"]),
            "status": status,
            "measurements_json": json.dumps(cleaned, ensure_ascii=False),
            "quantity_count": len(cleaned),
            "assignment_method": "model_single_pass",
            "rejected_response_json": None,
            # Retained on every row, accepted or refused, so a tightened rule can be
            # re-scored against the frozen artifact instead of re-spending tokens.
            "raw_response_json": json.dumps(returned, ensure_ascii=False, default=str),
        },
        None,
    )


def distillation_client() -> tuple[Any, str]:
    """Load the distillation ``llm`` helper from whichever checkout has it."""
    import importlib.util
    import sys

    for root in DISTILLATION_CANDIDATES:
        api_path = root / "api.py"
        if not api_path.exists():
            continue
        sys.path.insert(0, str(root))
        try:
            spec = importlib.util.spec_from_file_location(
                "_txagent_distillation_api", api_path
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"could not load distillation API at {api_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            if sys.path and sys.path[0] == str(root):
                sys.path.pop(0)
        return module.llm, MODEL
    raise FileNotFoundError(
        "no distillation api.py found in "
        + ", ".join(str(root) for root in DISTILLATION_CANDIDATES)
    )


def _openai_request(
    prompt: dict[str, str],
    model: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str,
    request_extra_body: dict[str, Any] | None,
    provider_only: str | None,
    is_openrouter: bool,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": prompt["system"] + "\nReturn the requested output as JSON.",
            },
            {"role": "user", "content": prompt["user"]},
        ],
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort:
        request["reasoning_effort"] = reasoning_effort
    if model.rsplit("/", 1)[-1].startswith("gpt-5"):
        request["max_completion_tokens"] = max_tokens
    else:
        request.update(max_tokens=max_tokens, temperature=temperature)
    extra_body = dict(request_extra_body or {})
    if is_openrouter:
        extra_body["provider"] = {"require_parameters": True}
        if provider_only:
            extra_body["provider"].update(only=[provider_only], allow_fallbacks=False)
    if extra_body:
        request["extra_body"] = extra_body
    return request


def _create_openai_response(
    client: Any, request: dict[str, Any], is_openrouter: bool
) -> Any:
    for attempt in range(len(OPENROUTER_RETRY_DELAYS_S) + 1):
        response = client.chat.completions.create(**request)
        if getattr(response, "choices", None):
            return response
        error = getattr(response, "error", None) or {}
        code = error.get("code") if isinstance(error, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        if is_openrouter and code == 429 and attempt < len(OPENROUTER_RETRY_DELAYS_S):
            time.sleep(OPENROUTER_RETRY_DELAYS_S[attempt])
            continue
        raise RuntimeError(
            f"OpenAI-compatible response has no choices: code={code}, "
            f"message={message or 'not reported'}"
        )
    raise AssertionError("unreachable")


def _api_metadata(response: Any, provider_only: str | None) -> dict[str, Any]:
    return {
        "api_response_id": getattr(response, "id", ""),
        "returned_model": getattr(response, "model", ""),
        "served_provider": getattr(response, "provider", None),
        "requested_provider": provider_only,
    }


def _validate_api_identity(
    metadata: dict[str, Any], model: str, provider_only: str | None
) -> None:
    served_provider = str(metadata["served_provider"]).lower()
    if provider_only == "baidu/fp8" and served_provider not in {
        "baidu",
        "baidu qianfan",
    }:
        raise ValueError(f"Baidu provider not verified: {metadata['served_provider']!r}")
    if provider_only and metadata["returned_model"] != model:
        raise ValueError(
            f"requested model {model!r}, received {metadata['returned_model']!r}"
        )


def openai_compatible_llm(
    client: Any,
    *,
    provider_only: str | None = None,
    request_extra_body: dict[str, Any] | None = None,
    cache: Any = None,
) -> Any:
    """Adapt an OpenAI Chat Completions client to the frozen runner contract."""

    def call(
        prompt: dict[str, str],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str = "",
        request_id: str = "",
        **_: Any,
    ) -> dict[str, Any]:
        is_openrouter = "openrouter.ai" in str(getattr(client, "base_url", ""))
        request = _openai_request(
            prompt,
            model,
            max_tokens,
            temperature,
            reasoning_effort,
            request_extra_body,
            provider_only,
            is_openrouter,
        )
        response = _create_openai_response(client, request, is_openrouter)
        message = response.choices[0].message
        usage = response.usage.model_dump() if response.usage is not None else None
        metadata = _api_metadata(response, provider_only)
        if cache is not None:
            cache._append(
                {
                    "cache_version": CACHE_VERSION,
                    "status": "api_response",
                    "request_id": request_id,
                    **metadata,
                    "response": response.model_dump(),
                }
            )
        _validate_api_identity(metadata, model, provider_only)
        return {
            "content": message.content or "",
            "reasoning": getattr(message, "reasoning_content", None),
            "usage": usage,
            "api_metadata": metadata,
        }

    return call


def _batch_user(wire_rows: list[dict[str, Any]], feedback: Any) -> str:
    payload: dict[str, Any] = {"rows": wire_rows}
    if feedback is not None:
        payload["validation_feedback"] = feedback
        payload["retry_instruction"] = (
            "Correct every listed error, then return all input ids again. "
            "Never return an empty unit: use a non-ok status with [] when the "
            "source does not support a complete unit or named metric."
        )
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _invoke_batch_llm(
    llm: Any,
    batch: RequestBatch,
    user: str,
    model: str,
    reasoning_effort: str,
    temperature: float,
    max_completion_tokens: int,
) -> Any:
    return llm(
        {"system": batch.prompt, "user": user},
        model=model,
        max_tokens=max_completion_tokens,
        temperature=temperature,
        verbose=False,
        reasoning_effort=reasoning_effort,
        max_retries=1,
        request_id=batch.request_id,
        row_ids=list(batch.row_ids),
    )


def _api_failure(
    error: Exception,
    batch: RequestBatch,
    attempt: int,
    usage: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, int] | None, str]:
    method = "api_failure_after_structural_retry" if attempt else "api_failure"
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        detail = body.get("error", body)
        if isinstance(detail, dict) and (
            detail.get("code") in {"credit_balance_exhausted", "insufficient_quota"}
            or detail.get("type") == "insufficient_quota"
        ):
            method = "api_failure_quota_exhausted"
    rows = [_blank(row, method=method) for row in batch.rows]
    failure = json.dumps(
        {"reason": method, "error_type": type(error).__name__, "error": str(error)},
        ensure_ascii=False,
    )
    for row in rows:
        row["rejected_response_json"] = failure
    reported_usage = usage if attempt == 0 and method == "api_failure_quota_exhausted" else None
    return rows, reported_usage, method


def _add_usage(usage: dict[str, int], result: Any) -> bool:
    reported = result.get("usage") if isinstance(result, dict) else None
    if not isinstance(reported, dict):
        return False
    usage["input_tokens"] += int(
        reported.get("prompt_tokens") or reported.get("input_tokens") or 0
    )
    usage["output_tokens"] += int(
        reported.get("completion_tokens") or reported.get("output_tokens") or 0
    )
    return True


def _parse_batch_rows(
    result: Any,
    expected_ids: list[str],
    batch: RequestBatch,
    last_attempt: bool,
    usage: dict[str, int] | None,
    with_feedback: bool,
) -> tuple[list[Any] | None, Any, list[dict[str, str]] | None]:
    try:
        parsed = json.loads(str(result.get("content") or ""))
        returned_rows = parsed.get("rows")
        if not isinstance(returned_rows, list):
            raise ValueError("response rows is not a list")  # noqa: TRY004
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        feedback = [{"id": "batch", "error": "malformed_json_or_rows"}]
        if not last_attempt:
            return None, None, feedback if with_feedback else None
        rows = [
            _blank(row, method="invalid_response_after_structural_retry")
            for row in batch.rows
        ]
        return None, (rows, usage, "invalid_response_after_structural_retry"), None
    returned_ids = [
        str(item["id"])
        for item in returned_rows
        if isinstance(item, dict) and item.get("id") is not None
    ]
    if len(returned_ids) == len(returned_rows) and sorted(returned_ids) == expected_ids:
        return returned_rows, None, None
    feedback = [{"id": "batch", "error": "response_id_mismatch"}]
    if not last_attempt:
        return None, None, feedback if with_feedback else None
    rows = [
        _refuse(
            row,
            returned_rows,
            method="batch_id_mismatch:structural_retry",
            reason="batch_id_mismatch",
        )[0]
        for row in batch.rows
    ]
    return None, (rows, usage, "structural_invalid_after_structural_retry"), None


def _validated_batch_rows(
    batch: RequestBatch,
    returned_rows: list[Any],
    result: Any,
    task: str,
    max_measurements: int | None,
) -> tuple[list[dict[str, Any]], list[str | None]]:
    by_id = {item["id"]: item for item in returned_rows}
    rows, reasons = [], []
    for row, api_row in zip(batch.rows, batch.api_rows, strict=True):
        returned = by_id[str(api_row["id"])]
        assignment, reason = validate_row(
            {**returned, "id": row["id"]},
            row,
            task=task,
            max_measurements=max_measurements,
        )
        assignment.update(result.get("api_metadata") or {})
        assignment["raw_response_json"] = json.dumps(returned, ensure_ascii=False)
        if reason is not None:
            assignment["rejected_response_json"] = json.dumps(
                {"reason": reason, "returned": returned},
                ensure_ascii=False,
                default=str,
            )
        rows.append(assignment)
        reasons.append(reason)
    return rows, reasons


def _retryable_reasons(reasons: list[str | None], retry_semantic_errors: bool) -> bool:
    excluded = ("measurement_is_not_a_plain_decimal", "too_many_measurements")
    return any(
        reason is not None
        and (retry_semantic_errors or not reason.startswith(excluded))
        for reason in reasons
    )


def _finish_batch_rows(
    rows: list[dict[str, Any]], reasons: list[str | None], attempt: int
) -> str:
    if attempt:
        for assignment in rows:
            method = assignment["assignment_method"]
            assignment["assignment_method"] = (
                "model_structural_retry"
                if method == "model_single_pass"
                else f"{method}:structural_retry"
            )
    invalid = sum(reason is not None for reason in reasons)
    suffix = "_after_structural_retry" if attempt else ""
    return f"partial_invalid{suffix}" if invalid else f"valid{suffix}"


def query_batch(
    batch: RequestBatch,
    *,
    llm: Any,
    model: str,
    task: str,
    max_measurements: int | None = None,
    reasoning_effort: str = REASONING_EFFORT,
    temperature: float = 1.0,
    retry_validation_feedback: bool = False,
    retry_semantic_errors: bool = False,
    max_schema_attempts: int = 2,
    initial_validation_feedback: Mapping[str, str] | None = None,
    retry_max_completion_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int] | None, str]:
    """Retry invalid rows with feedback while retaining already valid rows."""
    if max_schema_attempts < 1:
        raise ValueError("max_schema_attempts must be positive")
    retry_max_completion_tokens = (
        batch.max_completion_tokens
        if retry_max_completion_tokens is None
        else int(retry_max_completion_tokens)
    )
    if retry_max_completion_tokens < 1:
        raise ValueError("retry_max_completion_tokens must be positive")
    wire_rows = [
        {**row, "id": f"r{index}"} for index, row in enumerate(batch.api_rows, 1)
    ]
    pending = list(zip(batch.rows, wire_rows, strict=True))
    finished: dict[str, dict[str, Any]] = {}
    usage = {"input_tokens": 0, "output_tokens": 0}
    usage_complete = True
    initial_validation_feedback = initial_validation_feedback or {}
    feedback = [
        {"id": wire["id"], "error": initial_validation_feedback[str(row["id"])]}
        for row, wire in pending
        if str(row["id"]) in initial_validation_feedback
    ] or None
    terminal_status = "valid"
    for attempt in range(max_schema_attempts):
        pending_batch = RequestBatch(
            batch.request_id,
            batch.source_id,
            tuple(row for row, _ in pending),
            batch.prompt,
            batch.max_completion_tokens,
            tuple(wire for _, wire in pending),
        )
        expected_ids = sorted(str(wire["id"]) for _, wire in pending)
        last_attempt = attempt + 1 == max_schema_attempts
        try:
            result = _invoke_batch_llm(
                llm,
                pending_batch,
                _batch_user([wire for _, wire in pending], feedback),
                model,
                reasoning_effort,
                temperature,
                (
                    batch.max_completion_tokens
                    if attempt == 0
                    else retry_max_completion_tokens
                ),
            )
        except Exception as error:  # noqa: BLE001 - API clients raise provider types.
            rows, reported, status = _api_failure(
                error, pending_batch, attempt, usage
            )
            finished.update((row["cleaned_record_id"], row) for row in rows)
            terminal_status = status
            usage_complete = reported is not None
            break
        usage_complete = _add_usage(usage, result) and usage_complete
        returned_rows, terminal, new_feedback = _parse_batch_rows(
            result,
            expected_ids,
            pending_batch,
            last_attempt,
            usage if usage_complete else None,
            retry_validation_feedback,
        )
        if terminal is not None:
            rows, _, terminal_status = terminal
            finished.update((row["cleaned_record_id"], row) for row in rows)
            break
        if returned_rows is None:
            feedback = new_feedback
            continue
        rows, reasons = _validated_batch_rows(
            pending_batch, returned_rows, result, task, max_measurements
        )
        batch_status = _finish_batch_rows(rows, reasons, attempt)
        retry_pairs = []
        retry_feedback = []
        for pair, assignment, reason in zip(pending, rows, reasons, strict=True):
            retryable = reason is not None and _retryable_reasons(
                [reason], retry_semantic_errors
            )
            if retryable and not last_attempt:
                retry_pairs.append(pair)
                if retry_validation_feedback:
                    retry_feedback.append(
                        {"id": str(pair[1]["id"]), "error": str(reason)}
                    )
            else:
                finished[str(assignment["cleaned_record_id"])] = assignment
        if not retry_pairs:
            terminal_status = batch_status
            break
        pending = retry_pairs
        feedback = retry_feedback or None
        terminal_status = f"schema_retry_{attempt + 2}"
    ordered = [finished[str(row["id"])] for row in batch.rows]
    if any(
        str(row.get("assignment_method") or "").startswith(
            ("invalid_response", "invalid_row_response:")
        )
        for row in ordered
    ):
        terminal_status = f"partial_invalid_after_{max_schema_attempts}_attempts"
    return ordered, usage if usage_complete else None, terminal_status


def _checked_prompt_manifest(
    config: TaskConfig,
    expected: Mapping[str, Any] | None,
    error_message: str,
) -> dict[str, Any]:
    current = config.prompt_manifest(batch_size=config.BATCH_SIZE)
    if expected is not None and current != dict(expected):
        raise ValueError(error_message)
    return current


def _unsupported_exact_unit(
    candidate: dict[str, Any],
    assignment: Mapping[str, Any],
    config: TaskConfig,
    exact_mapping: Any,
) -> tuple[str, str, list[dict[str, Any]]] | None:
    if exact_mapping is None or assignment["status"] != "ok":
        return None
    endpoint = str(candidate.get("canonical_endpoint_name") or "")
    entries = json.loads(assignment["measurements_json"])
    for entry in entries:
        unit = str(entry.get("unit") or "").strip()
        if (config.task_id, endpoint, unit) not in exact_mapping and (
            config.task_id,
            "*",
            unit,
        ) not in exact_mapping:
            return endpoint, unit, entries
    return None


def _enforce_exact_units(
    candidate: dict[str, Any],
    assignment: dict[str, Any],
    config: TaskConfig,
    exact_mapping: Any,
) -> dict[str, Any]:
    unsupported = _unsupported_exact_unit(
        candidate, assignment, config, exact_mapping
    )
    if unsupported is None:
        return assignment
    endpoint, unit, entries = unsupported
    refused = _blank(candidate, method="unsupported_exact_unit")
    refused["raw_response_json"] = assignment.get("raw_response_json")
    refused["rejected_response_json"] = json.dumps(
        {
            "reason": "unsupported_exact_unit",
            "canonical_endpoint_name": endpoint,
            "unit": unit,
            "model_assignment": entries,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return refused


def _candidate_assignment(
    candidate: dict[str, Any],
    cache: SubmissionCache,
    config: TaskConfig,
    base_assignments: Mapping[str, Mapping[str, Any]],
    exact_mapping: Any,
) -> tuple[dict[str, Any], bool]:
    record_id = str(candidate["id"])
    from_base = record_id in base_assignments
    prior = base_assignments.get(record_id) or cache.assignments.get(record_id)
    if prior is None:
        if record_id not in cache.attempted:
            raise ValueError(f"unattempted row cannot be published: {record_id}")
        assignment = _blank(candidate, method="ambiguous_process_termination")
    else:
        assignment = dict(prior)
    guard = getattr(config.module, "guard_model_assignment", None)
    if guard is not None and not from_base:
        assignment = guard(candidate, assignment)
    postprocess = getattr(config.module, "postprocess_extracted_assignment", None)
    if postprocess is not None and not from_base:
        assignment = postprocess(candidate, assignment)
    return _enforce_exact_units(candidate, assignment, config, exact_mapping), from_base


def _inference_row(
    candidate: dict[str, Any],
    assignment: dict[str, Any],
    from_base: bool,
    provenance: Mapping[str, Any],
    base_model: str,
    model: str,
    api_base_url: str,
) -> dict[str, Any]:
    return {
        **assignment,
        "source_row_uid": candidate.get("source_row_uid"),
        "inference_source": "base_mapping" if from_base else "delta_inference",
        "inference_model": (
            str(assignment.get("inference_model") or base_model)
            if from_base
            else str(provenance.get("inference_model") or model)
        ),
        "inference_base_url": (
            str(assignment.get("inference_base_url") or "")
            if from_base
            else str(provenance.get("inference_base_url") or api_base_url)
        ),
        "inference_credential_env": (
            str(assignment.get("inference_credential_env") or "")
            if from_base
            else str(provenance.get("inference_credential_env") or "")
        ),
    }


def _materialized_rows(
    candidates: list[dict[str, Any]],
    cache: SubmissionCache,
    config: TaskConfig,
    base_assignments: Mapping[str, Mapping[str, Any]],
    exact_mapping: Any,
    base_model: str,
    model: str,
    api_base_url: str,
) -> list[dict[str, Any]]:
    rows = []
    provenance_by_id = getattr(cache, "provenance", {})
    response_credentials = {
        str(event.get("api_response_id") or ""): str(
            event.get("credential_env") or ""
        )
        for event in getattr(cache, "events", ())
        if event.get("status") == "api_response" and event.get("api_response_id")
    }
    for candidate in candidates:
        assignment, from_base = _candidate_assignment(
            candidate, cache, config, base_assignments, exact_mapping
        )
        provenance = dict(provenance_by_id.get(str(candidate["id"]), {}))
        if not from_base and assignment.get("api_response_id"):
            provenance.update(
                inference_model=str(
                    assignment.get("returned_model")
                    or provenance.get("inference_model")
                    or ""
                ),
                inference_base_url=str(
                    assignment.get("base_url")
                    or provenance.get("inference_base_url")
                    or ""
                ),
                inference_credential_env=response_credentials.get(
                    str(assignment["api_response_id"]),
                    str(provenance.get("inference_credential_env") or ""),
                ),
            )
        rows.append(
            _inference_row(
                candidate,
                assignment,
                from_base,
                provenance,
                base_model,
                model,
                api_base_url,
            )
        )
    rows.sort(key=lambda row: str(row["cleaned_record_id"]))
    return rows


def _usage_by_returned_model(cache: SubmissionCache) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for event in getattr(cache, "events", ()):
        if event.get("status") != "api_response":
            continue
        response = event.get("response") or {}
        usage = response.get("usage") or event.get("usage") or {}
        totals = output.setdefault(
            str(event["returned_model"]),
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "reported_cost": 0.0,
                "responses": 0,
                "responses_without_usage": 0,
            },
        )
        totals["input_tokens"] += int(usage.get("prompt_tokens") or 0)
        totals["output_tokens"] += int(usage.get("completion_tokens") or 0)
        totals["reported_cost"] += float(usage.get("cost") or 0)
        totals["responses"] += 1
        totals["responses_without_usage"] += not bool(usage)
    return output


def _single_or_mixed(values: list[str]) -> str:
    return values[0] if len(values) == 1 else "mixed"


def _model_provenance(
    rows: list[dict[str, Any]], cache: SubmissionCache
) -> dict[str, Any]:
    models = sorted({str(row["inference_model"]) for row in rows})
    base_urls = sorted(
        {str(row["inference_base_url"]) for row in rows if row["inference_base_url"]}
    )
    return {
        "model": _single_or_mixed(models),
        "models": models,
        "api_base_url": _single_or_mixed(base_urls),
        "api_base_urls": base_urls,
        "inference_model_counts": _counter(row["inference_model"] for row in rows),
        "api_usage_by_returned_model": _usage_by_returned_model(cache),
        "served_provider_counts": _counter(row.get("served_provider") for row in rows),
        "credential_counts": _counter(
            row["inference_credential_env"] for row in rows
        ),
        "inference_base_url_counts": _counter(
            row["inference_base_url"] for row in rows if row["inference_base_url"]
        ),
    }


def _input_lineage(
    rows: list[dict[str, Any]],
    profile_path: Path,
    records_path: Path,
    mapping_path: Path,
    base_mapping_path: Path | None,
    base_model: str,
    base_assignments: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    delta_rows = [row for row in rows if row["inference_source"] == "delta_inference"]
    delta_models = sorted({str(row["inference_model"]) for row in delta_rows})
    base_mapping = None
    if base_mapping_path is not None:
        base_mapping = {
            "path": str(base_mapping_path),
            "sha256": file_sha256(base_mapping_path),
            "model": base_model,
            "reused_rows": len(base_assignments),
        }
    return {
        "endpoint_profile": {
            "path": str(profile_path),
            "sha256": file_sha256(profile_path),
            "version": PROFILE_VERSION,
        },
        "cleaned_records_path": str(records_path),
        "cleaned_records_sha256": file_sha256(records_path),
        "mapping_path": str(mapping_path),
        "mapping_sha256": file_sha256(mapping_path),
        "mapping_rows": len(rows),
        "base_mapping": base_mapping,
        "delta_inference": {
            "model": _single_or_mixed(delta_models),
            "models": delta_models,
            "rows": len(delta_rows),
        },
    }


def _assignment_guard_summary(
    rows: list[dict[str, Any]], config: TaskConfig
) -> dict[str, Any] | None:
    if not hasattr(config.module, "ASSIGNMENT_GUARD_VERSION"):
        return None
    reasons = [
        row.get("assignment_guard_reason")
        for row in rows
        if row.get("assignment_guard_reason")
    ]
    return {
        "version": str(config.module.ASSIGNMENT_GUARD_VERSION),
        "guarded_rows": sum(bool(row.get("assignment_guard_reason")) for row in rows),
        "reason_counts": _counter(reasons),
    }


def _exact_unit_summary(
    rows: list[dict[str, Any]], exact_mapping_path: Path | None
) -> dict[str, Any] | None:
    if exact_mapping_path is None:
        return None
    return {
        "path": str(exact_mapping_path),
        "sha256": file_sha256(exact_mapping_path),
        "unsupported_ok_rows_demoted": sum(
            row["assignment_method"] == "unsupported_exact_unit" for row in rows
        ),
    }


def _mapping_validations(
    rows: list[dict[str, Any]], candidates: list[dict[str, Any]], config: TaskConfig
) -> dict[str, bool | None]:
    maximum = None
    if hasattr(config.module, "MAX_MEASUREMENTS_PER_ROW"):
        maximum = max((int(row["quantity_count"]) for row in rows), default=0) <= int(
            config.MAX_MEASUREMENTS_PER_ROW
        )
    return {
        "one_row_per_candidate": len(rows) == len(candidates),
        "unique_cleaned_record_ids": len({row["cleaned_record_id"] for row in rows})
        == len(rows),
        "only_ok_carries_measurements": all(
            (row["quantity_count"] > 0) == (row["status"] == "ok") for row in rows
        ),
        "maximum_measurements_per_row": maximum,
    }


def _result_summary(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    config: TaskConfig,
    exact_mapping_path: Path | None,
) -> dict[str, Any]:
    counts = {status: 0 for status in STATUSES}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    return {
        "status_counts": counts,
        "rejected_rows": sum(1 for row in rows if row["rejected_response_json"]),
        "assignment_method_counts": _counter(
            row["assignment_method"] for row in rows
        ),
        "assignment_guard": _assignment_guard_summary(rows, config),
        "exact_unit_coverage": _exact_unit_summary(rows, exact_mapping_path),
        "validations": _mapping_validations(rows, candidates, config),
    }


def _write_mapping(rows: list[dict[str, Any]], mapping_path: Path) -> None:
    columns = sorted(set().union(*(row.keys() for row in rows)))
    table = pa.Table.from_pylist([{key: row.get(key) for key in columns} for row in rows])
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = mapping_path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary)
    temporary.replace(mapping_path)


def _task_exact_mapping(config: TaskConfig) -> tuple[Path | None, Any]:
    if not getattr(config.module, "ENFORCE_EXACT_UNITS_DURING_EXTRACTION", True):
        return None, None
    path = getattr(config.module, "EXACT_UNIT_MAPPING_PATH", None)
    return path, load_exact_unit_mapping(path) if path is not None else None


def _inference_settings(
    max_completion_tokens: int,
    reasoning_mode: str,
    request_extra_body: dict[str, Any] | None,
    temperature: float,
) -> dict[str, Any]:
    settings = {
        "max_completion_tokens": max_completion_tokens,
        "reasoning_mode": reasoning_mode,
        "temperature": temperature,
    }
    if request_extra_body:
        settings["request_extra_body"] = request_extra_body
    return settings


def _build_materialization_manifest(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    cache: SubmissionCache,
    config: TaskConfig,
    current_prompt_manifest: dict[str, Any],
    mapping_path: Path,
    records_path: Path,
    profile_path: Path,
    base_mapping_path: Path | None,
    base_model: str,
    base_assignments: Mapping[str, Mapping[str, Any]],
    exact_mapping_path: Path | None,
    max_completion_tokens: int,
    reasoning_mode: str,
    request_extra_body: dict[str, Any] | None,
) -> dict[str, Any]:
    manifest = {
        "generation_version": GENERATION_VERSION,
        "routing_version": MEASUREMENT_ROUTING_VERSION,
        "task_id": config.task_id,
        "mapping_version": config.MAPPING_VERSION,
        **_model_provenance(rows, cache),
        "inference": _inference_settings(
            max_completion_tokens,
            reasoning_mode,
            request_extra_body,
            float(getattr(config.module, "TEMPERATURE", 1.0)),
        ),
        "prompt": current_prompt_manifest,
        "source_role_contract": (
            config.source_role_manifest()
            if hasattr(config.module, "source_role_manifest")
            else None
        ),
        "post_extraction_processing": (
            config.module.post_extraction_processing_manifest()
            if hasattr(config.module, "post_extraction_processing_manifest")
            else None
        ),
        **_input_lineage(
            rows,
            profile_path,
            records_path,
            mapping_path,
            base_mapping_path,
            base_model,
            base_assignments,
        ),
        **_result_summary(rows, candidates, config, exact_mapping_path),
    }
    return manifest


def _write_manifest(manifest: dict[str, Any], mapping_path: Path) -> None:
    mapping_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _check_materialization_start(
    config: TaskConfig, expected: Mapping[str, Any] | None
) -> None:
    if expected is None:
        return
    _checked_prompt_manifest(
        config,
        expected,
        "measurement extraction prompt or assignment guard changed after "
        "the run started; refusing to materialize cached assignments",
    )


def materialize(
    candidates: list[dict[str, Any]],
    cache: SubmissionCache,
    config: TaskConfig,
    *,
    mapping_path: Path,
    records_path: Path,
    profile_path: Path,
    model: str,
    api_base_url: str,
    max_completion_tokens: int,
    reasoning_mode: str,
    base_assignments: Mapping[str, Mapping[str, Any]] | None = None,
    base_mapping_path: Path | None = None,
    request_extra_body: dict[str, Any] | None = None,
    expected_prompt_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one mapping row per candidate, refusing to publish an unasked row."""
    _check_materialization_start(config, expected_prompt_manifest)
    base_assignments = base_assignments or {}
    base_model = _base_mapping_model(base_mapping_path) if base_assignments else ""
    exact_unit_mapping_path, exact_unit_mapping = _task_exact_mapping(config)
    rows = _materialized_rows(
        candidates,
        cache,
        config,
        base_assignments,
        exact_unit_mapping,
        base_model,
        model,
        api_base_url,
    )
    current_prompt_manifest = _checked_prompt_manifest(
        config,
        expected_prompt_manifest,
        (
            "measurement extraction prompt or assignment guard changed while "
            "assignments were being materialized"
        ),
    )
    _write_mapping(rows, mapping_path)
    manifest = _build_materialization_manifest(
        rows,
        candidates,
        cache,
        config,
        current_prompt_manifest,
        mapping_path,
        records_path,
        profile_path,
        base_mapping_path,
        base_model,
        base_assignments,
        exact_unit_mapping_path,
        max_completion_tokens,
        reasoning_mode,
        request_extra_body,
    )
    _write_manifest(manifest, mapping_path)
    return manifest


def load_base_mapping(path: Path) -> dict[str, dict[str, Any]]:
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_hash = manifest.get("mapping_sha256") or manifest.get(
        "measurement_resolution_sha256"
    )
    if expected_hash != file_sha256(path):
        raise ValueError(f"base mapping hash mismatch: {path}")
    rows = pq.read_table(path).to_pylist()
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = str(row.get("cleaned_record_id") or "")
        if not record_id or record_id in mapping:
            raise ValueError(f"base mapping has an empty or duplicate ID: {record_id!r}")
        mapping[record_id] = dict(row)
    return mapping


def promote_validated_mapping(
    *,
    pending_path: Path,
    final_path: Path,
    manifest: dict[str, Any],
    expected_record_ids: set[str],
    validator: Any,
) -> None:
    """Validate a pending mapping before replacing any prior canonical artifact."""
    pending_manifest = pending_path.with_suffix(".manifest.json")
    try:
        validator(pending_path, expected_record_ids=expected_record_ids)
    except Exception:
        pending_path.unlink(missing_ok=True)
        pending_manifest.unlink(missing_ok=True)
        raise

    manifest["mapping_path"] = str(final_path)
    final_manifest = final_path.with_suffix(".manifest.json")
    temporary_manifest = final_manifest.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pending_path.replace(final_path)
    temporary_manifest.replace(final_manifest)
    pending_manifest.unlink()
    validator(final_path, expected_record_ids=expected_record_ids)


def validate_full_mapping_provenance(
    mapping_path: str | Path,
    *,
    task: str,
    expected_record_ids: set[str] | None = None,
) -> None:
    """Validate a complete mixed OpenAI/local mapping against its Stage 1 rows."""
    path = Path(mapping_path)
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError(f"measurement mapping or manifest not found: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("generation_version") == "ames_measurement_resolution_mixed.v1":
        from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.consolidate_measurement_resolution import (
            validate_mixed_mapping,
        )

        validate_mixed_mapping(path, expected_record_ids=expected_record_ids)
        return
    config = TaskConfig(task)
    records_path = Path(
        str(manifest.get("cleaned_records_path") or config.DEFAULT_CLEANED_RECORDS)
    )
    base_spec = manifest.get("base_mapping")
    base_assignments: dict[str, dict[str, Any]] = {}
    expected_base = None
    if base_spec is not None:
        if not isinstance(base_spec, Mapping) or not base_spec.get("path"):
            raise ValueError(f"{task} measurement base-mapping lineage is invalid")
        base_path = Path(str(base_spec["path"]))
        base_assignments = load_base_mapping(base_path)
        expected_base = {
            "path": str(base_path),
            "sha256": file_sha256(base_path),
            "model": _base_mapping_model(base_path),
            "reused_rows": 0,
        }
    base_by_uid = {
        str(row.get("source_row_uid") or ""): row
        for row in base_assignments.values()
        if row.get("source_row_uid")
    }
    reviewed_spec = manifest.get("reviewed_source_decisions")
    reviewed_decisions: dict[str, dict[str, Any]] = {}
    reviewed_sha256 = ""
    if reviewed_spec is not None:
        if not isinstance(reviewed_spec, Mapping) or not reviewed_spec.get("path"):
            raise ValueError(f"{task} reviewed-source decision lineage is invalid")
        reviewed_path = Path(str(reviewed_spec["path"]))
        if not reviewed_path.is_file():
            raise ValueError(
                f"{task} reviewed-source decisions are missing: {reviewed_path}"
            )
        reviewed_sha256 = file_sha256(reviewed_path)
        decision_rows = [
            json.loads(line)
            for line in reviewed_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        reviewed_decisions = {
            str(row.get("cleaned_record_id") or ""): row for row in decision_rows
        }
        expected_reviewed = {
            "path": str(reviewed_path),
            "row_count": len(decision_rows),
            "sha256": reviewed_sha256,
        }
        found_reviewed = {
            key: reviewed_spec.get(key) for key in expected_reviewed
        }
        if (
            found_reviewed != expected_reviewed
            or "" in reviewed_decisions
            or len(reviewed_decisions) != len(decision_rows)
        ):
            raise ValueError(
                f"{task} reviewed-source decision receipt mismatch: "
                f"expected={expected_reviewed}, found={found_reviewed}"
            )

    required_columns = {
        "assignment_method",
        "cleaned_record_id",
        "measurements_json",
        "source_id",
        "source_row_uid",
        "status",
        "inference_source",
        "inference_model",
        "inference_base_url",
        "inference_credential_env",
        "rejected_response_json",
        "requested_provider",
        "served_provider",
    }
    if reviewed_spec is not None:
        required_columns.update({"review_decision_sha256", "reviewer_id"})
    available_columns = set(pq.read_schema(path).names)
    missing_columns = required_columns - available_columns
    table = pq.read_table(path, columns=sorted(required_columns & available_columns))
    rows = table.to_pylist()
    observed_ids = [str(row.get("cleaned_record_id") or "") for row in rows]
    selected_ids = set(observed_ids) if expected_record_ids is None else set(
        expected_record_ids
    )
    if expected_base is not None:
        expected_base["reused_rows"] = sum(
            str(row.get("source_row_uid") or "") in base_by_uid
            or str(row.get("cleaned_record_id") or "") in base_assignments
            for row in rows
        )
    mismatches: dict[str, Any] = {}
    expected_manifest = {
        "task_id": task,
        "mapping_version": config.MAPPING_VERSION,
        "mapping_rows": len(rows),
        "mapping_sha256": file_sha256(path),
        "cleaned_records_sha256": file_sha256(records_path),
        "base_mapping": expected_base,
        "prompt": config.prompt_manifest(batch_size=config.BATCH_SIZE),
        "source_role_contract": config.source_role_manifest(),
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            mismatches[key] = {"expected": expected, "found": manifest.get(key)}
    if not rows or len(observed_ids) != len(set(observed_ids)) or "" in observed_ids:
        mismatches["row_identity"] = "mapping must contain unique nonempty IDs"
    if set(observed_ids) != selected_ids:
        mismatches["candidate_coverage"] = {
            "expected_rows": len(selected_ids),
            "observed_rows": len(set(observed_ids)),
            "missing_rows": len(selected_ids - set(observed_ids)),
            "extra_rows": len(set(observed_ids) - selected_ids),
        }
    inference = manifest.get("inference") or {}
    expected_inference = {
        "max_completion_tokens": 8_192,
        "reasoning_mode": "low",
        "temperature": float(getattr(config.module, "TEMPERATURE", 1.0)),
    }
    if inference != expected_inference:
        mismatches["inference"] = {
            "expected": expected_inference,
            "found": inference,
        }
    for key in (
        "one_row_per_candidate",
        "unique_cleaned_record_ids",
        "only_ok_carries_measurements",
        "maximum_measurements_per_row",
    ):
        value = (manifest.get("validations") or {}).get(key)
        if value is not True and not (key == "maximum_measurements_per_row" and value is None):
            mismatches[key] = {"expected": True, "found": value}
    if int(manifest.get("rejected_rows") or 0):
        mismatches["rejected_rows"] = {
            "expected": 0,
            "found": manifest.get("rejected_rows"),
        }

    local_providers = {
        "http://dgx005:50001/v1": (
            "dgx005_50001",
            "deepseek-ai/DeepSeek-V4-Flash-0731",
        ),
        "http://dgx005:50002/v1": (
            "dgx005_50002",
            "deepseek-ai/DeepSeek-V4-Flash-0731",
        ),
        "http://dgx005:8100/v1": ("dgx005_8100", "deepseek-v4-flash"),
        "http://dgx007:50001/v1": (
            "dgx007_50001",
            "deepseek-ai/DeepSeek-V4-Flash-0731",
        ),
        "http://dgx011:50001/v1": (
            "dgx011_50001",
            "deepseek-ai/DeepSeek-V4-Flash-0731",
        ),
        "http://dgx014:50002/v1": (
            "dgx014_50002",
            "deepseek-ai/DeepSeek-V4-Flash-0731",
        ),
        "http://dgx017:50001/v1": (
            "dgx017_50001",
            "deepseek-ai/DeepSeek-V4-Flash-0731",
        ),
        "http://dgx020:50002/v1": (
            "dgx020_50002",
            "deepseek-ai/DeepSeek-V4-Flash-0731",
        ),
    }
    allowed = {
        (
            "gpt-5.4-mini",
            "https://api.openai.com/v1",
            "OPENAI_API_KEY_ONE",
        ),
        (
            "gpt-5.4-mini",
            "https://api.openai.com/v1",
            "OPENAI_API_KEY_TWO",
        ),
        (
            "deepseek/deepseek-v4-flash-0731",
            "https://openrouter.ai/api/v1",
            "OPEN_ROUTER_KEY_TWO",
        ),
    } | {(model, base_url, "") for base_url, (_, model) in local_providers.items()}
    if missing_columns:
        mismatches["mapping_columns"] = {"missing": sorted(missing_columns)}
    else:
        invalid_provenance = []
        source_uids: dict[str, tuple[str, str]] = {}
        columns = ["cleaned_record_id", "source_row_uid", "source_id"]
        parquet = pq.ParquetFile(records_path)
        for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
            for record in batch.to_pylist():
                record_id = str(record.get("cleaned_record_id") or "")
                if record_id in selected_ids:
                    source_uids[record_id] = (
                        str(record.get("source_row_uid") or ""),
                        str(record.get("source_id") or ""),
                    )
        for row in rows:
            provenance = (
                str(row.get("inference_model") or ""),
                str(row.get("inference_base_url") or ""),
                str(row.get("inference_credential_env") or ""),
            )
            record_id = str(row.get("cleaned_record_id") or "")
            source_identity = (
                str(row.get("source_row_uid") or ""),
                str(row.get("source_id") or ""),
            )
            requested_provider = str(row.get("requested_provider") or "")
            served_provider = str(row.get("served_provider") or "")
            expected_local_provider = (
                local_providers.get(provenance[1]) or (None, None)
            )[0]
            if requested_provider or (
                expected_local_provider is not None
                and served_provider != expected_local_provider
            ):
                mismatches["row_provider_provenance"] = {
                    "cleaned_record_id": record_id,
                    "requested_provider": requested_provider,
                    "served_provider": served_provider,
                    "expected_served_provider": expected_local_provider,
                }
                break
            if source_identity != source_uids.get(record_id):
                mismatches["source_identity"] = {
                    "cleaned_record_id": record_id,
                    "expected": source_uids.get(record_id),
                    "found": source_identity,
                }
                break
            inference_source = row.get("inference_source")
            base_row = base_by_uid.get(source_identity[0]) or base_assignments.get(
                record_id
            )
            valid_base = inference_source == "base_mapping" and base_row is not None
            valid_delta = (
                inference_source == "delta_inference" and provenance in allowed
            )
            decision = reviewed_decisions.get(record_id)
            valid_reviewed = (
                inference_source == REVIEWED_SOURCE_INFERENCE
                and provenance
                == (
                    REVIEWED_SOURCE_MODEL,
                    REVIEWED_SOURCE_BASE_URL,
                    "",
                )
                and decision is not None
                and str(row.get("review_decision_sha256") or "")
                == reviewed_sha256
                and str(row.get("reviewer_id") or "")
                == str(decision.get("reviewer_id") or "")
                and str(row.get("source_row_uid") or "")
                == str(decision.get("source_row_uid") or "")
                and str(row.get("status") or "")
                == str(decision.get("status") or "")
                and json.loads(str(row.get("measurements_json") or "[]"))
                == list(decision.get("measurements") or [])
            )
            if (
                not (valid_base or valid_delta or valid_reviewed)
                or row.get("rejected_response_json") not in (None, "")
            ):
                invalid_provenance.append(record_id)
                break
        if invalid_provenance:
            mismatches["row_provenance"] = {
                "first_invalid": invalid_provenance[0]
            }
    model_counts = manifest.get("inference_model_counts") or {}
    credential_counts = manifest.get("credential_counts") or {}
    base_url_counts = manifest.get("inference_base_url_counts") or {}
    expected_counts = {
        "inference_model_counts": _counter(
            row.get("inference_model") for row in rows
        ),
        "credential_counts": _counter(
            row.get("inference_credential_env") for row in rows
        ),
        "inference_base_url_counts": _counter(
            row.get("inference_base_url") for row in rows
        ),
    }
    for key, counts in (
        ("inference_model_counts", model_counts),
        ("credential_counts", credential_counts),
        ("inference_base_url_counts", base_url_counts),
    ):
        if counts != expected_counts[key]:
            mismatches[key] = {"expected": expected_counts[key], "found": counts}
    models = sorted(expected_counts["inference_model_counts"])
    base_urls = sorted(expected_counts["inference_base_url_counts"])
    summary_expected = {
        "model": _single_or_mixed(models),
        "models": models,
        "api_base_url": _single_or_mixed(base_urls),
        "api_base_urls": base_urls,
    }
    for key, expected in summary_expected.items():
        if manifest.get(key) != expected:
            mismatches[key] = {"expected": expected, "found": manifest.get(key)}
    if mismatches:
        raise ValueError(f"{task} measurement mapping provenance mismatch: {mismatches}")


def _base_mapping_model(path: Path | None) -> str:
    if path is None:
        return ""
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        manifest_path = path.parent / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = str(payload.get("model") or "")
    if not model:
        models = {str(row.get("inference_model") or "") for row in load_base_mapping(path).values()}
        if not models or "" in models:
            raise ValueError(f"base mapping lacks per-row model provenance: {manifest_path}")
        return next(iter(models)) if len(models) == 1 else "mixed_per_row"
    return model


def retryable_assignment_ids(
    cache: SubmissionCache,
    *,
    retry_model: str | None = None,
    max_attempts: int | None = None,
) -> set[str]:
    """Rows that reached no publishable model answer and are safe to retry."""
    prefixes = (
        "ambiguous_process_termination",
        "api_failure",
        "invalid_response_after_structural_retry",
        "invalid_row_response:",
        "worker_failure",
    )
    failed = {
        record_id
        for record_id, assignment in cache.assignments.items()
        if str(assignment.get("assignment_method") or "").startswith(prefixes)
    }
    candidates = failed | (cache.attempted - cache.assignments.keys())
    if retry_model is None or max_attempts is None or max_attempts == 0:
        return candidates
    attempts: dict[str, int] = {}
    for event in cache.events:
        if event.get("status") != "submitted" or event.get("model") != retry_model:
            continue
        for record_id in event.get("row_ids") or ():
            key = str(record_id)
            attempts[key] = attempts.get(key, 0) + 1
    return {
        record_id
        for record_id in candidates
        if attempts.get(record_id, 0) < max_attempts
    }


def _gold_sources(path: Path, ids: set[str]) -> dict[str, int]:
    """Which sources the unroutable gold rows belong to."""
    output: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        case = json.loads(line)
        if case["audit_case_id"].split(":", 1)[1] in ids:
            output[case["source_id"]] = output.get(case["source_id"], 0) + 1
    return output


def _counter(values: Any) -> dict[str, int]:
    output: dict[str, int] = {}
    for value in values:
        output[str(value)] = output.get(str(value), 0) + 1
    return dict(sorted(output.items()))


def load_endpoint_profile(path: Path, config: TaskConfig) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("profile_version") != PROFILE_VERSION:
        raise ValueError(f"endpoint profile version mismatch: {path}")
    if payload.get("routing_version") != MEASUREMENT_ROUTING_VERSION:
        raise ValueError(f"endpoint profile routing mismatch: {path}")
    if payload.get("task") != config.task_id:
        raise ValueError(f"endpoint profile task mismatch: {path}")
    if not isinstance(payload.get("endpoints"), dict):
        raise ValueError(f"endpoint profile lacks endpoint entries: {path}")  # noqa: TRY004
    return payload


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True, choices=SUPPORTED_TASKS)
    parser.add_argument(
        "--config-module",
        default=None,
        help="alternate task measurement-resolution config module",
    )
    parser.add_argument("--source", default=None, help="restrict to one source_id")
    parser.add_argument(
        "--limit", type=int, default=None, help="cap the candidate count (pilot runs)"
    )
    parser.add_argument(
        "--gold-replay",
        action="store_true",
        help=(
            "run exactly the hand-labelled gold rows that reach the model, so every "
            "response is scoreable. Use a separate --cache-dir: the submission cache "
            "never re-asks a row, so a replay would otherwise consume these ids for "
            "the full run too."
        ),
    )
    parser.add_argument("--gold-fixture", type=Path, default=None)
    parser.add_argument(
        "--allow-partial-gold-replay",
        action="store_true",
        help=(
            "proceed when some gold rows no longer route to extraction, naming what "
            "is skipped. Any accuracy figure then covers a subset, so say which."
        ),
    )
    parser.add_argument("--cleaned-records", type=Path, default=None)
    parser.add_argument("--endpoint-profile", type=Path, default=None)
    parser.add_argument("--mapping-path", type=Path, default=None)
    parser.add_argument("--base-mapping", type=Path, default=None)


def _add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--validation-feedback",
        type=Path,
        default=None,
        help="JSONL rows with cleaned_record_id and the prior validation reason",
    )
    parser.add_argument("--token-ledger", type=Path, default=None)
    parser.add_argument("--budget-epoch", default=None)
    parser.add_argument("--budget-max-tokens", type=int, default=10_000_000)
    parser.add_argument(
        "--phase-budget-max-tokens",
        type=int,
        default=None,
        help=(
            "stop this invocation after reserving at most this many additional "
            "tokens from the shared ledger; the ledger epoch remains active"
        ),
    )
    parser.add_argument("--start-new-budget-epoch", action="store_true")
    parser.add_argument(
        "--no-token-ledger",
        action="store_true",
        help="disable the token cap; API responses and usage remain cached",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--parallelism",
        type=int,
        default=None,
        help="global request concurrency for --provider-pool-config",
    )
    parser.add_argument("--two-key-baidu-run", action="store_true")
    parser.add_argument(
        "--budget-ledger-dir",
        type=Path,
        default=None,
        help="reuse the original two-key ledgers across a source/prompt migration",
    )
    parser.add_argument("--provider-only", default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--retry-batch-size",
        type=int,
        default=None,
        help="split only released failed batches into smaller retry requests",
    )
    parser.add_argument(
        "--retry-max-attempts",
        type=int,
        default=None,
        help="override per-row attempt cap; 0 allows unlimited technical retries",
    )
    parser.add_argument("--defer-publication", action="store_true")
    parser.add_argument("--require-complete", action="store_true")


def _add_api_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--provider-pool-config",
        type=Path,
        default=None,
        help="probe and schedule across an OpenAI-compatible provider inventory",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="use this OpenAI-compatible Chat Completions endpoint",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="override the provider's standard credential variable name",
    )
    parser.add_argument(
        "--provider", choices=("openai", "openrouter", "parcc", "local"), default=None
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--max-completion-tokens", type=int, default=MAX_COMPLETION_TOKENS
    )
    parser.add_argument(
        "--retry-max-completion-tokens",
        type=int,
        default=None,
        help="completion-token ceiling after the first API call in a batch",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=600,
        help="per-attempt timeout for an OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="print the plan and exit before loading the prompt or the ledger",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_input_arguments(parser)
    _add_execution_arguments(parser)
    _add_api_arguments(parser)
    args = parser.parse_args(argv)
    if args.retry_max_attempts is not None and (
        args.retry_max_attempts < 0 or not args.retry_failed
    ):
        parser.error("--retry-max-attempts requires --retry-failed and a nonnegative value")
    if args.retry_batch_size is not None and (
        args.retry_batch_size < 1 or not args.retry_failed
    ):
        parser.error("--retry-batch-size requires --retry-failed and a positive value")
    if args.phase_budget_max_tokens is not None and args.phase_budget_max_tokens < 1:
        parser.error("--phase-budget-max-tokens must be positive")
    return args


def _validate_task_args(args: argparse.Namespace) -> None:
    allowed_dili_configs = {
        None,
        (
            "data.processing.evidence_library.versions.v10.tasks.dili."
            "starling_measurement_resolution_three_endpoint_256"
        ),
        (
            "data.processing.evidence_library.versions.v10.tasks.dili."
            "starling_measurement_resolution_three_endpoint_512"
        ),
        (
            "data.processing.evidence_library.versions.v10.tasks.dili."
            "starling_measurement_resolution_two_endpoint_512"
        ),
        (
            "data.processing.evidence_library.versions.v10.tasks.dili."
            "starling_measurement_resolution_dgx007_dgx011_512"
        ),
        (
            "data.processing.evidence_library.versions.v10.tasks.dili."
            "starling_measurement_resolution_dgx007_550"
        ),
        (
            "data.processing.evidence_library.versions.v10.tasks.dili."
            "starling_measurement_resolution_dgx005_8100_550"
        ),
        (
            "data.processing.evidence_library.versions.v10.tasks.dili."
            "starling_measurement_resolution_dgx005_550"
        ),
        (
            "data.processing.evidence_library.versions.v10.tasks.dili."
            "starling_measurement_resolution_dgx005_50002_550"
        ),
        (
            "data.processing.evidence_library.versions.v10.tasks.dili."
            "starling_measurement_resolution_openrouter_key_two_256"
        ),
        (
            "data.processing.evidence_library.versions.v10.tasks.dili."
            "starling_measurement_resolution_incremental_three_endpoint_512"
        ),
    }
    if args.task == "dili" and (
        args.config_module not in allowed_dili_configs or args.two_key_baidu_run
    ):
        raise SystemExit(
            "DILI V10 does not permit --config-module or --two-key-baidu-run"
        )


def _run_paths(
    args: argparse.Namespace, config: TaskConfig
) -> tuple[Path, Path, Path | None, Path]:
    records_path = args.cleaned_records or config.DEFAULT_CLEANED_RECORDS
    if not records_path.is_file():
        raise SystemExit(f"cleaned records not found: {records_path}")
    mapping_path = args.mapping_path or config.DEFAULT_MAPPING_PATH
    base_mapping_path = args.base_mapping or getattr(
        config, "DEFAULT_BASE_MAPPING_PATH", None
    )
    if base_mapping_path is not None and not base_mapping_path.is_file():
        raise SystemExit(f"base measurement mapping not found: {base_mapping_path}")
    profile_path = args.endpoint_profile or config.DEFAULT_PROFILE_PATH
    if not profile_path.is_file():
        raise SystemExit(
            f"endpoint profile not found: {profile_path}; build it with "
            "common.starling.build_endpoint_unit_profile"
        )
    return records_path, mapping_path, base_mapping_path, profile_path


def _gold_candidate_ids(args: argparse.Namespace) -> set[str] | None:
    if not args.gold_replay:
        return None
    if args.limit or args.source:
        raise SystemExit("--gold-replay replaces --limit/--source")
    only_ids = set(gold_extract_ids(args.gold_fixture))
    print(f"gold replay: {len(only_ids):,} hand-labelled extraction rows")
    return only_ids


def _report_missing_gold(
    missing: MissingGoldRows, args: argparse.Namespace
) -> set[str]:
    absent = set(missing.absent)
    by_source = _gold_sources(args.gold_fixture, absent)
    print(f"  {len(absent):,} gold row(s) do not route to extraction:")
    for source_id, count in sorted(by_source.items()):
        print(f"     {source_id:24}{count:>5}")
    print(
        "  these rows do not have the persisted Stage-01 `extract` route; "
        "review the gold fixture or rebuild Stage 01 before replaying them."
    )
    if not args.allow_partial_gold_replay:
        raise SystemExit(
            "refusing a partial gold replay; pass --allow-partial-gold-replay "
            "to score the remainder and state which subset it covers"
        )
    return absent


def _run_candidates(
    args: argparse.Namespace, records_path: Path, config: TaskConfig
) -> list[dict[str, Any]]:
    only_ids = _gold_candidate_ids(args)
    try:
        return candidate_rows(
            records_path,
            config,
            source_id=args.source,
            limit=args.limit,
            only_ids=only_ids,
        )
    except MissingGoldRows as missing:
        absent = _report_missing_gold(missing, args)
        assert only_ids is not None
        only_ids -= absent
        print(f"  proceeding over the remaining {len(only_ids):,} gold row(s)")
        return candidate_rows(
            records_path, config, source_id=args.source, only_ids=only_ids
        )


def _mapping_delta(
    candidates: list[dict[str, Any]], base_mapping_path: Path | None
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    base_rows = load_base_mapping(base_mapping_path) if base_mapping_path else {}
    base_by_uid = {
        str(row.get("source_row_uid") or ""): row for row in base_rows.values()
    }
    base_by_uid.pop("", None)
    if len(base_by_uid) != sum(
        bool(row.get("source_row_uid")) for row in base_rows.values()
    ):
        raise ValueError("base mapping has duplicate source_row_uid values")
    base_assignments: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        record_id = str(candidate["id"])
        source_uid = str(candidate.get("source_row_uid") or "")
        prior = base_by_uid.get(source_uid) or base_rows.get(record_id)
        if prior and str(prior.get("source_row_uid") or source_uid) != source_uid:
            raise ValueError(f"base mapping source UID mismatch: {candidate['id']}")
        if prior and str(prior.get("source_id") or "") != str(candidate["source_id"]):
            raise ValueError(f"base mapping source mismatch: {candidate['id']}")
        if prior:
            base_assignments[record_id] = {
                **prior,
                "cleaned_record_id": record_id,
                "source_id": candidate["source_id"],
                "source_row_uid": source_uid,
            }
    delta = [row for row in candidates if str(row["id"]) not in base_assignments]
    return base_assignments, delta


def _print_inventory(
    config: TaskConfig,
    candidates: list[dict[str, Any]],
    base_assignments: Mapping[str, Any],
    delta_candidates: list[dict[str, Any]],
    planned_batches: list[RequestBatch],
) -> None:
    per_source = _counter(row["source_id"] for row in candidates)
    requests_by_source = _counter(batch.source_id for batch in planned_batches)
    print(f"task {config.task_id} | routing {MEASUREMENT_ROUTING_VERSION}")
    print(f"extraction candidates: {len(candidates):,}")
    print(
        f"base rows reused: {len(base_assignments):,} | "
        f"new rows requiring inference: {len(delta_candidates):,}"
    )
    for source_id, count in per_source.items():
        requests = requests_by_source.get(source_id, 0)
        print(f"   {source_id:24}{count:>9,} rows  {requests:>7,} requests")
    print(
        f"planned requests: {len(planned_batches):,} "
        f"at batch_size={config.BATCH_SIZE}"
    )


def _validate_inference_args(args: argparse.Namespace) -> None:
    if args.provider_pool_config and args.base_url:
        raise SystemExit("--provider-pool-config and --base-url are mutually exclusive")
    if args.provider_pool_config and not args.parallelism:
        raise SystemExit("--provider-pool-config requires --parallelism")
    if args.parallelism and not args.provider_pool_config:
        raise SystemExit("--parallelism requires --provider-pool-config")
    if args.no_token_ledger and not (args.base_url or args.provider_pool_config):
        raise SystemExit("--no-token-ledger requires --base-url or --provider-pool-config")
    if not args.no_token_ledger and not args.budget_epoch:
        raise SystemExit("--budget-epoch is required for a paid run")
    if args.phase_budget_max_tokens is not None and args.no_token_ledger:
        raise SystemExit("--phase-budget-max-tokens requires the token ledger")
    if args.validation_feedback is not None and not args.validation_feedback.is_file():
        raise SystemExit(f"validation feedback not found: {args.validation_feedback}")
    if (
        args.retry_max_completion_tokens is not None
        and args.retry_max_completion_tokens < 1
    ):
        raise SystemExit("--retry-max-completion-tokens must be positive")


def _validation_feedback(
    path: Path | None, candidates: list[dict[str, Any]]
) -> dict[str, str]:
    if path is None:
        return {}
    allowed = {str(row["id"]) for row in candidates}
    feedback: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        record_id = str(row.get("cleaned_record_id") or "")
        reason = str(row.get("reason") or "")
        if not record_id or not reason or record_id in feedback:
            raise ValueError("invalid or duplicate validation-feedback row")
        if record_id not in allowed:
            raise ValueError(f"validation feedback names an unselected row: {record_id}")
        feedback[record_id] = reason
    return feedback


def _reviewed_prompt(config: TaskConfig) -> dict[str, Any]:
    prompt = config.prompt_manifest(batch_size=config.BATCH_SIZE)
    print(f"prompt {prompt['prompt_version']} template {prompt['template_sha256'][:16]}")
    return prompt


def _cache_contract(
    config: TaskConfig,
    records_path: Path,
    profile_digest: str,
    prompt_manifest: dict[str, Any],
    request_extra_body: dict[str, Any] | None,
    max_completion_tokens: int,
    validation_feedback_path: Path | None,
) -> dict[str, Any]:
    contract = {
        "task": config.task_id,
        "records_sha256": file_sha256(records_path),
        "profile_sha256": profile_digest,
        "prompt": prompt_manifest,
        "mapping_version": config.MAPPING_VERSION,
    }
    if hasattr(config.module, "REASONING_EFFORT"):
        contract["reasoning_effort"] = config.REASONING_EFFORT
    if hasattr(config.module, "source_role_manifest"):
        contract["source_role_contract"] = config.source_role_manifest()
    if request_extra_body:
        contract["request_extra_body"] = request_extra_body
    if hasattr(config.module, "MAX_SCHEMA_ATTEMPTS"):
        contract["schema_retry"] = {
            "max_attempts": int(config.MAX_SCHEMA_ATTEMPTS),
            "validation_feedback": bool(
                getattr(config, "RETRY_VALIDATION_FEEDBACK", False)
            ),
            "max_completion_tokens": int(max_completion_tokens),
            "initial_feedback": (
                {
                    "path": str(validation_feedback_path),
                    "sha256": file_sha256(validation_feedback_path),
                }
                if validation_feedback_path is not None
                else None
            ),
        }
    return contract


def _run_cache(
    args: argparse.Namespace,
    config: TaskConfig,
    records_path: Path,
    profile_digest: str,
    prompt_manifest: dict[str, Any],
    request_extra_body: dict[str, Any] | None,
) -> tuple[Path, SubmissionCache]:
    cache_dir = args.cache_dir or Path(
        "outputs/chembl_tool/measurement_resolution_generation"
    )
    cache_path = cache_dir / config.task_id / "requests.jsonl"
    contract_path = cache_path.parent / "input_contract.json"
    contract = _cache_contract(
        config,
        records_path,
        profile_digest,
        prompt_manifest,
        request_extra_body,
        args.max_completion_tokens,
        args.validation_feedback,
    )
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("extraction cache input/prompt contract mismatch")
    elif cache_path.exists():
        if args.two_key_baidu_run or args.require_complete:
            raise ValueError("existing extraction cache lacks an input contract")
    else:
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    return cache_path, SubmissionCache(cache_path)


def _release_retryable_rows(
    args: argparse.Namespace, config: TaskConfig, cache: SubmissionCache
) -> None:
    unmetered_retry = args.no_token_ledger and getattr(
        config.module, "RETRY_TERMINAL_FAILURES_ON_UNMETERED", False
    )
    if not args.retry_failed and not unmetered_retry:
        return
    max_attempts = (
        args.retry_max_attempts
        if args.retry_max_attempts is not None
        else getattr(config.module, "MAX_UNMETERED_ATTEMPTS_PER_ROW", None)
    )
    retry_ids = retryable_assignment_ids(
        cache, retry_model=args.model, max_attempts=max_attempts
    )
    cache.allow_retry(retry_ids)
    print(f"failed rows released for fallback retry: {len(retry_ids):,}")


def _run_ledger(args: argparse.Namespace) -> TokenLedger | None:
    if args.no_token_ledger:
        return None
    path = args.token_ledger or Path(
        "outputs/chembl_tool/measurement_resolution_generation/token_ledger.json"
    )
    return TokenLedger(
        path,
        epoch=args.budget_epoch,
        start_new_epoch=args.start_new_budget_epoch,
        max_tokens=args.budget_max_tokens,
    )


def _remaining_batches(
    args: argparse.Namespace,
    config: TaskConfig,
    delta_candidates: list[dict[str, Any]],
    planned_batches: list[RequestBatch],
    cache: SubmissionCache,
    endpoint_profile: dict[str, Any],
    profile_digest: str,
) -> tuple[list[RequestBatch], int | None]:
    batches = (
        plan_batches(
            delta_candidates,
            config,
            attempted=cache.attempted,
            endpoint_profile=endpoint_profile,
            profile_digest=profile_digest,
            model=args.model,
            max_completion_tokens=args.max_completion_tokens,
        )
        if cache.attempted
        else planned_batches
    )
    if args.retry_batch_size is not None:
        if args.retry_batch_size > config.BATCH_SIZE:
            raise ValueError(
                f"retry batch size cannot exceed {config.BATCH_SIZE}"
            )
        size = args.retry_batch_size
        batches = [
            RequestBatch(
                request_id=f"{batch.request_id}_part_{offset // size + 1}",
                source_id=batch.source_id,
                rows=batch.rows[offset : offset + size],
                prompt=batch.prompt,
                max_completion_tokens=batch.max_completion_tokens,
                payload_rows=(
                    batch.api_rows[offset : offset + size]
                    if batch.payload_rows is not None
                    else None
                ),
            )
            for batch in batches
            for offset in range(0, len(batch.rows), size)
        ]
    print(f"unattempted requests this run: {len(batches):,}")
    if not batches and retryable_assignment_ids(cache):
        print("retry attempt cap exhausted; no API requests sent; nothing published")
        return batches, RETRY_LIMIT_EXIT_CODE
    return batches, None


def _run_client(
    args: argparse.Namespace,
    cache: SubmissionCache,
    request_extra_body: dict[str, Any] | None,
    reasoning_effort: str,
) -> tuple[Any, str, str, str]:
    if args.provider_pool_config:
        return _run_provider_pool(args, cache, reasoning_effort)
    if not args.base_url:
        llm, _ = distillation_client()
        return llm, args.model, "distillation_default", ""
    client, credential_env = openai_compatible_client(
        base_url=args.base_url,
        provider=args.provider,
        env_file=args.env_file,
        credential_env=args.api_key_env,
        max_connections=args.workers,
        timeout_s=args.request_timeout_s,
    )
    client = client.with_options(max_retries=0)
    llm = openai_compatible_llm(
        client,
        provider_only=args.provider_only,
        request_extra_body=request_extra_body,
        cache=cache,
    )
    return llm, args.model, args.base_url, credential_env


def _build_measurement_provider_pool(
    args: argparse.Namespace, effort: str
) -> tuple[Any, dict[str, Any]]:
    """Build one bounded pool that can be shared by measurement task queues."""
    from dataclasses import replace

    from predict.api_client.pool import (
        ProviderPoolConfig,
        build_provider_pool,
        load_provider_pool_config,
        select_healthy_providers,
    )

    if args.task not in {"ames", "dili", "skin_reaction", "carcinogens"}:
        raise ValueError(
            "the low-reasoning provider pool is limited to the four approved V10 tasks"
        )
    configured = load_provider_pool_config(args.provider_pool_config)
    models = {provider.model for provider in configured.providers}
    if models != {args.model}:
        raise ValueError(
            f"provider pool model mismatch: requested={args.model!r}, configured={sorted(models)}"
        )
    if effort not in {"low", "high"}:
        raise ValueError(f"unsupported measurement reasoning effort: {effort!r}")
    providers = []
    for provider in configured.providers:
        extra = dict(provider.request_extra_body or {})
        if provider_from_base_url(provider.base_url) == "local":
            template = dict(extra.get("chat_template_kwargs") or {})
            template.update(thinking=True, reasoning_effort=effort)
            extra["chat_template_kwargs"] = template
        providers.append(replace(provider, request_extra_body=extra))
    pool_config = ProviderPoolConfig(
        providers=tuple(providers),
        failure_threshold=configured.failure_threshold,
        cooldown_seconds=configured.cooldown_seconds,
        max_failovers=configured.max_failovers,
        latency_ewma_alpha=configured.latency_ewma_alpha,
    )
    selection = select_healthy_providers(pool_config, args.parallelism)
    args.workers = selection.effective_parallelism
    pool = build_provider_pool(
        selection.config,
        env_file=args.env_file,
        timeout_s=int(args.request_timeout_s),
        max_tokens=args.max_completion_tokens,
        temperature=0.0,
        tool_service_url="",
        enable_group_tools=False,
        max_tool_rounds=0,
        reasoning_effort=effort,
        enable_thinking=False,
        transport_max_retries=0,
        response_format={"type": "json_object"},
    )
    receipt = {
        **selection.public_dict(),
        "effective_reasoning_effort": effort,
        "model": args.model,
    }
    return pool, receipt


def _provider_pool_llm(
    args: argparse.Namespace,
    cache: SubmissionCache,
    effort: str,
    pool: Any,
) -> Any:
    """Bind one task cache to an already validated shared provider pool."""
    credential_by_base_url = {
        provider.base_url: provider.api_key_env for provider in pool.config.providers
    }
    submitted_rows = {
        str(event.get("request_id") or ""): list(event.get("row_ids") or [])
        for event in cache.events
        if event.get("status") == "submitted"
    }
    corrected = {
        str(event.get("request_id") or "")
        for event in cache.events
        if event.get("status") == "provider_provenance"
    }
    for event in cache.events:
        request_id = str(event.get("request_id") or "")
        if (
            event.get("status") != "api_response"
            or request_id in corrected
            or not submitted_rows.get(request_id)
            or not event.get("base_url")
        ):
            continue
        credential_env = str(event.get("credential_env") or "")
        if not credential_env:
            credential_env = credential_by_base_url.get(str(event["base_url"]), "")
        correction = {
            "cache_version": CACHE_VERSION,
            "status": "provider_provenance",
            "request_id": request_id,
            "row_ids": submitted_rows[request_id],
            "model": str(event.get("returned_model") or args.model),
            "base_url": str(event["base_url"]),
            "credential_env": credential_env,
        }
        cache._append(correction)
        for record_id in correction["row_ids"]:
            cache.provenance[str(record_id)] = {
                "inference_model": correction["model"],
                "inference_base_url": correction["base_url"],
                "inference_credential_env": credential_env,
            }

    def call(
        prompt: dict[str, str],
        *,
        model: str,
        reasoning_effort: str,
        request_id: str,
        row_ids: list[str],
        max_tokens: int,
        **_: Any,
    ) -> dict[str, Any]:
        if reasoning_effort != effort:
            raise ValueError(
                f"measurement provider pool requires reasoning_effort={effort!r}"
            )
        response = pool.chat_json(
            [
                {
                    "role": "system",
                    "content": prompt["system"] + "\nReturn the requested output as JSON.",
                },
                {"role": "user", "content": prompt["user"]},
            ],
            max_tokens=max_tokens,
        )
        execution = dict(response.get("execution_provider") or {})
        served_model = str(response.get("model") or execution.get("served_model") or "")
        if served_model != model:
            raise ValueError(f"requested model {model!r}, received {served_model!r}")
        credential_env = credential_by_base_url.get(
            str(execution.get("base_url") or ""), ""
        )
        metadata = {
            "api_response_id": str(response.get("id") or ""),
            "returned_model": served_model,
            "served_provider": execution.get("upstream_provider") or execution.get("provider"),
            "requested_provider": None,
            "base_url": execution.get("base_url"),
        }
        row_ids = row_ids or submitted_rows.get(request_id, [])
        cache._append(
            {
                "cache_version": CACHE_VERSION,
                "status": "api_response",
                "request_id": request_id,
                "row_ids": row_ids,
                "model": served_model,
                "base_url": execution.get("base_url"),
                "credential_env": credential_env,
                **metadata,
                "response": {
                    "id": str(response.get("id") or ""),
                    "model": served_model,
                    "usage": response.get("usage") or None,
                    "raw_content": response.get("raw_content"),
                    "finish_reason": (
                        (response.get("structured_reasoning") or {}).get(
                            "finish_reason"
                        )
                    ),
                    "requested_max_completion_tokens": max_tokens,
                    "request_phase": (
                        "retry"
                        if args.retry_max_completion_tokens is not None
                        and max_tokens == args.retry_max_completion_tokens
                        else "initial"
                    ),
                },
                "provider_attempts": response.get("execution_provider_attempts") or [],
            }
        )
        for record_id in row_ids:
            cache.provenance[str(record_id)] = {
                "inference_model": served_model,
                "inference_base_url": str(execution.get("base_url") or ""),
                "inference_credential_env": credential_env,
            }
        content = response.get("raw_content")
        if not isinstance(content, str):
            content = json.dumps(response.get("content") or {}, ensure_ascii=False)
        return {
            "content": content,
            "reasoning": response.get("reasoning_content"),
            "usage": response.get("usage") or None,
            "api_metadata": metadata,
        }

    return call


def _run_provider_pool(
    args: argparse.Namespace, cache: SubmissionCache, effort: str
) -> tuple[Any, str, str, str]:
    """Build the provider scheduler and bind it to one task cache."""
    pool, receipt = _build_measurement_provider_pool(args, effort)
    receipt_path = Path(cache.path).parent / "provider_pool_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return _provider_pool_llm(args, cache, effort, pool), args.model, "provider_pool", ""


def _submit_request(
    executor: ThreadPoolExecutor,
    batch: RequestBatch,
    cache: SubmissionCache,
    ledger: TokenLedger | None,
    credential_env: str,
    model: str,
    api_base_url: str,
    config: TaskConfig,
    llm: Any,
    initial_validation_feedback: Mapping[str, str] | None = None,
    retry_max_completion_tokens: int | None = None,
) -> Any:
    initial_validation_feedback = initial_validation_feedback or {}
    cache.submit(
        batch,
        epoch=ledger.epoch if ledger is not None else "uncapped_usage_in_cache",
        credential_env=credential_env,
        model=model,
        base_url=api_base_url,
    )
    cache._append(
        {
            "cache_version": CACHE_VERSION,
            "status": "id_transport",
            "request_id": batch.request_id,
            "id_transport_version": ID_TRANSPORT_VERSION,
            "id_mapping": {
                f"r{index}": record_id
                for index, record_id in enumerate(batch.row_ids, 1)
            },
            "initial_max_completion_tokens": batch.max_completion_tokens,
            "schema_retry_max_completion_tokens": (
                retry_max_completion_tokens or batch.max_completion_tokens
            ),
        }
    )
    return executor.submit(
        query_batch,
        batch,
        llm=llm,
        model=model,
        task=config.task_id,
        max_measurements=getattr(config, "MAX_MEASUREMENTS_PER_ROW", None),
        reasoning_effort=getattr(config, "REASONING_EFFORT", REASONING_EFFORT),
        temperature=float(getattr(config.module, "TEMPERATURE", 1.0)),
        retry_validation_feedback=getattr(
            config, "RETRY_VALIDATION_FEEDBACK", False
        ),
        retry_semantic_errors=getattr(config, "RETRY_SEMANTIC_ERRORS", False),
        max_schema_attempts=int(getattr(config, "MAX_SCHEMA_ATTEMPTS", 2)),
        initial_validation_feedback={
            record_id: initial_validation_feedback[record_id]
            for record_id in batch.row_ids
            if record_id in initial_validation_feedback
        },
        retry_max_completion_tokens=retry_max_completion_tokens,
    )


def _complete_request(
    future: Any,
    batch: RequestBatch,
    cache: SubmissionCache,
    ledger: TokenLedger | None,
) -> str:
    try:
        rows, usage, status = future.result()
    except Exception:  # noqa: BLE001 - preserve any worker failure as a cache row.
        rows = [_blank(row, method="worker_failure") for row in batch.rows]
        usage, status = None, "worker_failure"
    if ledger is not None:
        ledger.complete(batch.request_id, usage)
    cache.terminal(batch, assignments=rows, usage=usage, response_status=status)
    return status


def _print_request_progress(
    completed: int,
    total: int,
    cache: SubmissionCache,
    ledger: TokenLedger | None,
) -> None:
    if completed % 25:
        return
    charged = ledger.spent() if ledger is not None else "uncapped; see cache"
    print(
        f"completed {completed:,}/{total:,} batches; "
        f"cached rows={len(cache.assignments):,}; charged tokens={charged}",
        flush=True,
    )


def _execute_batches(
    batches: list[RequestBatch],
    args: argparse.Namespace,
    config: TaskConfig,
    cache: SubmissionCache,
    ledger: TokenLedger | None,
    llm: Any,
    model: str,
    api_base_url: str,
    credential_env: str,
    initial_validation_feedback: Mapping[str, str] | None = None,
) -> tuple[bool, bool, bool]:
    initial_validation_feedback = initial_validation_feedback or {}
    pending = iter(batches)
    in_flight: dict[Any, RequestBatch] = {}
    next_batch: RequestBatch | None = None
    finished = credential_unavailable = phase_exhausted = False
    phase_start_spent = ledger.spent() if ledger is not None else 0
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while True:
            while len(in_flight) < args.workers and not finished:
                if next_batch is None:
                    try:
                        next_batch = next(pending)
                    except StopIteration:
                        finished = True
                        break
                if ledger is not None:
                    reservation = int(
                        getattr(config, "MAX_SCHEMA_ATTEMPTS", 2)
                    ) * next_batch.reservation_tokens
                    in_flight_tokens = sum(
                        int(value)
                        for value in ledger.state.get("reservations", {}).values()
                    )
                    if (
                        args.phase_budget_max_tokens is not None
                        and ledger.spent()
                        - phase_start_spent
                        + in_flight_tokens
                        + reservation
                        > args.phase_budget_max_tokens
                    ):
                        # Reservations are deliberately conservative. Let current
                        # requests settle to their actual usage, then refill until
                        # even one new reservation no longer fits the phase.
                        if not in_flight:
                            phase_exhausted = finished = True
                        break
                    if not ledger.reserve(next_batch.request_id, reservation):
                        break
                future = _submit_request(
                    executor,
                    next_batch,
                    cache,
                    ledger,
                    credential_env,
                    model,
                    api_base_url,
                    config,
                    llm,
                    initial_validation_feedback,
                    args.retry_max_completion_tokens,
                )
                in_flight[future], next_batch = next_batch, None
            if not in_flight:
                return (
                    credential_unavailable,
                    next_batch is not None and not phase_exhausted,
                    phase_exhausted,
                )
            done, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
            for future in done:
                status = _complete_request(
                    future, in_flight.pop(future), cache, ledger
                )
                if status == "api_failure_quota_exhausted":
                    credential_unavailable = finished = True
                    next_batch = None
                completed += 1
                _print_request_progress(completed, len(batches), cache, ledger)


def _execution_exit_code(
    credential_unavailable: bool,
    exhausted: bool,
    phase_exhausted: bool,
    ledger: TokenLedger | None,
) -> int | None:
    if credential_unavailable:
        if ledger is not None:
            ledger.state["status"] = "credential_unavailable"
            ledger._write()
        print("credential has no API credits; checkpointed for the next phase", flush=True)
        return CREDENTIAL_UNAVAILABLE_EXIT_CODE
    if exhausted:
        assert ledger is not None
        ledger.mark_exhausted()
        print(
            "budget epoch exhausted; nothing published. "
            "Resume with a new --budget-epoch; attempted rows are never re-asked."
        )
        return BUDGET_EXHAUSTED_EXIT_CODE
    if phase_exhausted:
        print(
            "phase token allowance reached; shared budget remains active and the "
            "submission cache is ready for the next task or credential phase",
            flush=True,
        )
        return PHASE_BUDGET_EXHAUSTED_EXIT_CODE
    return None


def _materialization_target(
    mapping_path: Path, config: TaskConfig
) -> tuple[bool, Path, Path]:
    auto_validate = bool(
        getattr(config.module, "AUTO_VALIDATE_GENERATED_MAPPING", False)
    )
    target = (
        mapping_path.with_name(f"{mapping_path.stem}.pending.parquet")
        if auto_validate
        else mapping_path
    )
    pending_manifest = target.with_suffix(".manifest.json")
    if auto_validate:
        target.unlink(missing_ok=True)
        pending_manifest.unlink(missing_ok=True)
    return auto_validate, target, pending_manifest


def _verify_publication_prompt(
    config: TaskConfig,
    prompt_manifest: dict[str, Any],
    auto_validate: bool,
    target: Path,
    pending_manifest: Path,
) -> None:
    if config.prompt_manifest(batch_size=config.BATCH_SIZE) == prompt_manifest:
        return
    if auto_validate:
        target.unlink(missing_ok=True)
        pending_manifest.unlink(missing_ok=True)
    raise ValueError(
        "measurement extraction prompt or assignment guard changed before "
        "publication; refusing to promote the mapping"
    )


def _promote_mapping(
    candidates: list[dict[str, Any]],
    config: TaskConfig,
    target: Path,
    mapping_path: Path,
    manifest: dict[str, Any],
) -> None:
    expected_ids = {str(candidate["id"]) for candidate in candidates}
    promote_validated_mapping(
        pending_path=target,
        final_path=mapping_path,
        manifest=manifest,
        expected_record_ids=expected_ids,
        validator=config.validate_mapping_provenance,
    )
    print(f"validated task provenance and candidate coverage: {mapping_path}")


def _publish_run(
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    cache: SubmissionCache,
    config: TaskConfig,
    mapping_path: Path,
    records_path: Path,
    profile_path: Path,
    model: str,
    api_base_url: str,
    base_assignments: Mapping[str, Mapping[str, Any]],
    base_mapping_path: Path | None,
    request_extra_body: dict[str, Any] | None,
    prompt_manifest: dict[str, Any],
) -> int:
    if args.defer_publication:
        return 0
    unresolved = retryable_assignment_ids(cache)
    if args.require_complete and unresolved:
        print(f"unresolved technical failures: {len(unresolved):,}; nothing published")
        return 3
    auto_validate, target, pending_manifest = _materialization_target(
        mapping_path, config
    )
    manifest = materialize(
        candidates,
        cache,
        config,
        mapping_path=target,
        records_path=records_path,
        profile_path=profile_path,
        model=model,
        api_base_url=api_base_url,
        max_completion_tokens=args.max_completion_tokens,
        reasoning_mode=getattr(config, "REASONING_EFFORT", REASONING_EFFORT),
        base_assignments=base_assignments,
        base_mapping_path=base_mapping_path,
        request_extra_body=request_extra_body,
        expected_prompt_manifest=prompt_manifest,
    )
    _verify_publication_prompt(
        config, prompt_manifest, auto_validate, target, pending_manifest
    )
    if auto_validate:
        _promote_mapping(candidates, config, target, mapping_path, manifest)
    print(json.dumps(manifest["status_counts"], indent=2))
    print(f"rejected rows: {manifest['rejected_rows']:,}")
    print(f"wrote {mapping_path}")
    return 0


def _prepare_plan(args: argparse.Namespace) -> tuple[Any, ...]:
    config = TaskConfig(args.task, args.config_module)
    if args.gold_fixture is None:
        args.gold_fixture = getattr(config.module, "DEFAULT_GOLD_FIXTURE", GOLD_FIXTURE)
    validator = getattr(config.module, "validate_generation_args", None)
    if validator is not None:
        validator(args)
    if args.source is not None and args.source not in config.SOURCE_IDS:
        raise SystemExit(f"unknown --source {args.source!r}")
    records_path, mapping_path, base_mapping_path, profile_path = _run_paths(
        args, config
    )
    endpoint_profile = load_endpoint_profile(profile_path, config)
    profile_digest = file_sha256(profile_path)
    candidates = _run_candidates(args, records_path, config)
    base_assignments, delta_candidates = _mapping_delta(
        candidates, base_mapping_path
    )
    planned_batches = plan_batches(
        delta_candidates,
        config,
        attempted=set(),
        endpoint_profile=endpoint_profile,
        profile_digest=profile_digest,
        model=args.model,
        max_completion_tokens=args.max_completion_tokens,
    )
    _print_inventory(
        config, candidates, base_assignments, delta_candidates, planned_batches
    )
    return (
        config,
        records_path,
        mapping_path,
        base_mapping_path,
        profile_path,
        endpoint_profile,
        profile_digest,
        candidates,
        base_assignments,
        delta_candidates,
        planned_batches,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_task_args(args)
    if args.two_key_baidu_run:
        return run_two_key_baidu(list(sys.argv[1:] if argv is None else argv), args)
    (config, records_path, mapping_path, base_mapping_path, profile_path,
     endpoint_profile, profile_digest, candidates, base_assignments,
     delta_candidates, planned_batches) = _prepare_plan(args)
    if args.inventory_only:
        return 0
    _validate_inference_args(args)
    prompt_manifest = _reviewed_prompt(config)
    request_extra_body = getattr(config.module, "REQUEST_EXTRA_BODY", None)
    cache_path, cache = _run_cache(
        args, config, records_path, profile_digest, prompt_manifest, request_extra_body
    )
    _release_retryable_rows(args, config, cache)
    ledger = _run_ledger(args)
    if ledger is not None and ledger.state.get("status") == "credential_unavailable":
        print("credential previously reported no API credits; skipping this phase")
        return CREDENTIAL_UNAVAILABLE_EXIT_CODE
    batches, early_exit = _remaining_batches(
        args,
        config,
        delta_candidates,
        planned_batches,
        cache,
        endpoint_profile,
        profile_digest,
    )
    if early_exit is not None:
        return early_exit
    llm, model, api_base_url, credential_env = _run_client(
        args,
        cache,
        request_extra_body,
        getattr(config, "REASONING_EFFORT", REASONING_EFFORT),
    )
    print(f"model {model}")
    initial_feedback = _validation_feedback(args.validation_feedback, delta_candidates)
    outcome = _execute_batches(
        batches,
        args,
        config,
        cache,
        ledger,
        llm,
        model,
        api_base_url,
        credential_env,
        initial_feedback,
    )
    exit_code = _execution_exit_code(*outcome, ledger)
    if exit_code is not None:
        return exit_code
    return _publish_run(
        args,
        candidates,
        SubmissionCache(cache_path),
        config,
        mapping_path,
        records_path,
        profile_path,
        model,
        api_base_url,
        base_assignments,
        base_mapping_path,
        request_extra_body,
        prompt_manifest,
    )


def run_two_key_baidu(argv: list[str], args: argparse.Namespace) -> int:
    """Resume two capped OpenAI phases, then finish only unresolved rows on Baidu."""
    if args.task != "bioavailability_ma" or args.cache_dir is None or not args.budget_epoch:
        raise ValueError("two-key schedule requires Oral Bio, --cache-dir and --budget-epoch")
    if args.no_token_ledger or args.provider_only or args.retry_failed:
        raise ValueError("the two-key schedule owns budgets and fallback routing")
    common = [value for value in argv if value != "--two-key-baidu-run"]
    common += ["--require-complete", "--workers", "8"]
    for index, credential in enumerate(("OPENAI_API_KEY", "OPENAI_API_KEY_TWO"), 1):
        result = main(common + [
            "--model", "gpt-5.4-mini", "--base-url", "https://api.openai.com/v1",
            "--provider", "openai", "--api-key-env", credential,
            "--token-ledger", str((args.budget_ledger_dir or args.cache_dir) / f"gpt_key_{index}_ledger.json"),
            "--budget-max-tokens", "10000000", "--defer-publication",
        ])
        if result == 0:
            break
        if result not in {BUDGET_EXHAUSTED_EXIT_CODE, CREDENTIAL_UNAVAILABLE_EXIT_CODE}:
            return result
    for _ in range(3):
        result = main(common + [
            "--model", "deepseek/deepseek-v4-flash-0731",
            "--base-url", "https://openrouter.ai/api/v1", "--provider", "openrouter",
            "--api-key-env", "OPEN_ROUTER_KEY", "--provider-only", "baidu/fp8",
            "--no-token-ledger", "--retry-failed",
        ])
        if result != 3:
            return result
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
