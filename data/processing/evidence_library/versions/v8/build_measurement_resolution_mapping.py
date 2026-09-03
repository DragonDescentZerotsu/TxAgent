"""Generate one frozen, endpoint-aware measurement extraction per routed row.

Categorical encoding is independent of this router. Exact scalar pairs and
deterministic scalar rejections are settled before this offline pass; only the
remaining ``extract`` rows reach the model. Responses are
cached before submission, structurally invalid batches are retried once, and no
mapping publishes until every candidate has a terminal result. Paid endpoints may
use the inherited token ledger; unmetered compatible endpoints use
``--no-token-ledger``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.llm_api import DEFAULT_ENV_FILE, openai_compatible_client
from data.processing.evidence_library.versions.v8.build_reference_semantics_mapping import (
    BUDGET_EXHAUSTED_EXIT_CODE,
    DISTILLATION_ROOT,
    RequestBatch,
    SubmissionCache,
    TokenLedger,
)
from data.processing.evidence_library.versions.v8.build_endpoint_unit_profile import (
    PROFILE_VERSION,
    SUPPORTED_TASKS,
    profile_lookup,
    render_profile_block,
)
from data.processing.evidence_library.versions.v8.measurement_routing import (
    MEASUREMENT_ROUTING_VERSION,
    ROUTE_BUCKETS,
    route,
)
from data.processing.evidence_library.shared.v1.normalization.cleaning import file_sha256
from data.processing.evidence_library.versions.v8.task_registry import import_task_module


GENERATION_VERSION = "measurement_resolution_generation.v7"
MODEL = "gpt-5.4-mini"
REASONING_EFFORT = "low"

#: The distillation helper lives beside the TxAgent checkout, and TxAgent has more
#: than one checkout -- so resolve it relative to this repository first and treat the
#: node002 path as one candidate rather than the truth.  Hardcoding either location
#: breaks the other host.
DISTILLATION_CANDIDATES = (DISTILLATION_ROOT,)

#: Fixed so a pilot slice is reproducible and never silently re-drawn.
SAMPLE_SEED = 20260820
STATUSES = ("ok", "unsure", "relative", "unavailable")
MAX_COMPLETION_TOKENS = 65_536

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
            config.module.route_measurement(case["input"])
            if hasattr(config.module, "route_measurement")
            else route(
                case["input"],
                config.rules[case["source_id"]],
                task=config.task_id,
            )
        ).bucket
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
    """Stream the cleaned parquet and keep only rows the router sends to extraction.

    Routing is applied here with the same function the runtime resolver uses, so the
    candidate set is reproducible from the cleaned records alone.
    """
    available = set(pq.read_schema(records_path).names)
    wanted = {
        "cleaned_record_id",
        "source_id",
        "canonical_endpoint_name",
        "measurement_resolution_route",
        *PROMPT_FIELDS,
    }
    for rules in config.rules.values():
        wanted.add(rules.measurement_field)
        if rules.unit_field:
            wanted.add(rules.unit_field)
        wanted.update(rule.field for rule in rules.declarative_non_scalar)
    missing = sorted(wanted - available)
    columns = sorted(wanted & available)

    candidates: list[dict[str, Any]] = []
    parquet = pq.ParquetFile(records_path)
    for batch in parquet.iter_batches(batch_size=50_000, columns=columns):
        for record in batch.to_pylist():
            row_source = str(record.get("source_id") or "")
            if source_id is not None and row_source != source_id:
                continue
            if row_source not in config.SOURCE_IDS:
                continue
            rules = config.rules.get(row_source)
            if rules is None:
                raise ValueError(f"no routing rules for source {row_source!r}")
            persisted_route = record.get("measurement_resolution_route")
            if persisted_route is not None and str(persisted_route) not in ROUTE_BUCKETS:
                raise ValueError(
                    f"unsupported persisted measurement route {persisted_route!r}"
                )
            route_bucket = (
                str(persisted_route)
                if persisted_route is not None
                else (
                    config.module.route_measurement(record).bucket
                    if hasattr(config.module, "route_measurement")
                    else route(record, rules, task=config.task_id).bucket
                )
            )
            if route_bucket != "extract":
                continue
            record_id = str(record.get("cleaned_record_id") or "")
            if not record_id:
                raise ValueError("extraction candidate lacks cleaned_record_id")
            if only_ids is not None and record_id not in only_ids:
                continue
            record_resolver = getattr(config.module, "canonical_endpoint_record", None)
            payload = {
                "id": record_id,
                "source_id": row_source,
                "canonical_endpoint_name": (
                    str(record["canonical_endpoint_name"])
                    if record.get("canonical_endpoint_name")
                    else (
                        record_resolver(record)
                        if record_resolver is not None
                        else config.canonical_endpoint_name(
                            row_source, record.get("endpoint_name")
                        )
                    )
                ),
            }
            for field in config.prompt_row_fields(row_source):
                payload[field] = _json_value(record.get(field))
            candidates.append(payload)
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
            # A requested row that no longer routes to extraction means routing and
            # the gold set disagree.  Usually that is a stale Stage-01: influx's
            # measurement column only exists once ``reported_result`` has been
            # promoted and the clean stage rebuilt, so its rows route to reject
            # until then.  Silently scoring the remainder would report an accuracy
            # figure over an unstated subset, so this is fatal unless waived.
            raise MissingGoldRows(sorted(absent), len(only_ids))
    if missing:
        # Absent prompt fields are legitimate (efflux has no unit column) but a
        # missing measurement column is not, so report rather than silently drop.
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
    batches: list[RequestBatch] = []
    profiles = profile_lookup(endpoint_profile or {"endpoints": {}})
    by_source: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in candidates:
        endpoint = str(row.get("canonical_endpoint_name") or "")
        if not endpoint:
            raise ValueError(f"candidate {row['id']!r} lacks canonical endpoint")
        by_source.setdefault(row["source_id"], {}).setdefault(endpoint, []).append(row)

    for source_id in sorted(by_source):
        requests: list[list[tuple[str, list[dict[str, Any]]]]] = []
        small: list[tuple[str, list[dict[str, Any]]]] = []
        small_size = 0
        for endpoint, endpoint_rows in sorted(by_source[source_id].items()):
            endpoint_rows.sort(key=lambda row: row["id"])
            if len(endpoint_rows) >= config.BATCH_SIZE:
                for offset in range(0, len(endpoint_rows), config.BATCH_SIZE):
                    requests.append(
                        [(endpoint, endpoint_rows[offset : offset + config.BATCH_SIZE])]
                    )
                continue
            if small and small_size + len(endpoint_rows) > config.BATCH_SIZE:
                requests.append(small)
                small, small_size = [], 0
            small.append((endpoint, endpoint_rows))
            small_size += len(endpoint_rows)
        if small:
            requests.append(small)

        for endpoint_groups in requests:
            chunk = tuple(
                dict(row)
                for _, endpoint_rows in endpoint_groups
                for row in endpoint_rows
            )
            chunk_ids = {row["id"] for row in chunk}
            overlap = chunk_ids & attempted
            if overlap:
                if overlap != chunk_ids:
                    raise ValueError(
                        "submission cache contains only part of deterministic request "
                        f"{sorted(chunk_ids)}"
                    )
                continue
            profile_blocks = []
            for endpoint, _ in endpoint_groups:
                entry = profiles.get((source_id, endpoint))
                profile_blocks.append(
                    render_profile_block(entry, exclude_ids=chunk_ids)
                    if entry is not None
                    else f"[endpoint: {endpoint} - no reliable deterministic summary]"
                )
            prompt = config.render_prompt(
                source_id,
                batch_size=config.BATCH_SIZE,
                endpoint_profiles=tuple(profile_blocks),
            )
            composition = [
                {"canonical_endpoint_name": endpoint, "row_ids": [row["id"] for row in rows]}
                for endpoint, rows in endpoint_groups
            ]
            digest = hashlib.sha256(
                json.dumps(
                    {
                        "task": config.task_id,
                        "source": source_id,
                        "prompt_version": config.PROMPT_VERSION,
                        "model": model,
                        "max_completion_tokens": max_completion_tokens,
                        "profile_digest": profile_digest,
                        "endpoint_groups": composition,
                        "rendered_prompt_sha256": hashlib.sha256(
                            prompt.encode("utf-8")
                        ).hexdigest(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:24]
            payload = tuple(
                {key: row[key] for key in row if key != "source_id"} for row in chunk
            )
            batches.append(
                RequestBatch(
                    request_id=f"{config.task_id}_{source_id}_{digest}",
                    source_id=source_id,
                    rows=chunk,
                    prompt=prompt,
                    max_completion_tokens=max_completion_tokens,
                    payload_rows=payload,
                )
            )
    return batches


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


def validate_row(
    returned: Any,
    row: Any,
    *,
    task: str,
    max_measurements: int | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Check one returned row against the schema, or explain why it is refused.

    Every refusal is fail-closed: the row is published as ``unsure`` with the reason
    kept in ``rejected_response_json``, never dropped and never guessed at.
    """
    if not isinstance(returned, dict):
        return _refuse(row, returned, method="not_an_object", reason="not_an_object")
    if str(returned.get("id") or "") != str(row["id"]):
        return _refuse(row, returned, method="id_mismatch", reason="id_mismatch")
    status = str(returned.get("status") or "")
    if status not in STATUSES:
        return _refuse(
            row, returned, method="unsupported_status",
            reason=f"unsupported_status:{status!r}",
        )
    entries = returned.get("measurements")
    if entries is None:
        entries = []
    if not isinstance(entries, list):
        return _refuse(
            row, returned, method="measurements_not_a_list",
            reason="measurements_not_a_list",
        )
    if status != "ok" and entries:
        return _refuse(
            row, returned, method="non_ok_carries_measurements",
            reason=f"non_ok_carries_measurements:{status}",
        )
    if status == "ok" and not entries:
        return _refuse(
            row, returned, method="ok_without_measurements",
            reason="ok_without_measurements",
        )
    if max_measurements is not None and len(entries) > max_measurements:
        return _refuse(
            row,
            returned,
            method="too_many_measurements",
            reason=f"too_many_measurements:{len(entries)}>{max_measurements}",
        )
    cleaned: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            return _refuse(
                row, returned, method="entry_not_an_object",
                reason="entry_not_an_object",
            )
        measurement = str(entry.get("measurement") or "").strip()
        unit = str(entry.get("unit") or "").strip()
        if not measurement:
            return _refuse(
                row, returned, method="empty_measurement", reason="empty_measurement"
            )
        # Every entry must be self-contained so it can become one exploded row.
        if not unit:
            return _refuse(
                row, returned, method="entry_without_a_unit",
                reason="entry_without_a_unit",
            )
        if not is_plain_decimal(measurement):
            return _refuse(
                row,
                returned,
                method="measurement_is_not_a_plain_decimal",
                reason=f"measurement_is_not_a_plain_decimal:{measurement!r}",
            )
        cleaned.append({"measurement": measurement, "unit": unit})
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


def openai_compatible_llm(client: Any) -> Any:
    """Adapt an OpenAI Chat Completions client to the frozen runner contract."""

    def call(
        prompt: dict[str, str],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str = "",
        **_: Any,
    ) -> dict[str, Any]:
        request: dict[str, Any] = dict(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": prompt["system"] + "\nReturn the requested output as JSON.",
                },
                {"role": "user", "content": prompt["user"]},
            ],
            response_format={"type": "json_object"},
        )
        if reasoning_effort:
            request["reasoning_effort"] = reasoning_effort
        if model.startswith("gpt-5"):
            request["max_completion_tokens"] = max_tokens
        else:
            request.update(max_tokens=max_tokens, temperature=temperature)
        response = client.chat.completions.create(**request)
        message = response.choices[0].message
        usage = response.usage.model_dump() if response.usage is not None else None
        return {
            "content": message.content or "",
            "reasoning": getattr(message, "reasoning_content", None),
            "usage": usage,
        }

    return call


def query_batch(
    batch: RequestBatch,
    *,
    llm: Any,
    model: str,
    task: str,
    max_measurements: int | None = None,
    reasoning_effort: str = REASONING_EFFORT,
) -> tuple[list[dict[str, Any]], dict[str, int] | None, str]:
    """Retry one malformed response, but never retry API or semantic failures."""
    user = json.dumps({"rows": batch.api_rows}, ensure_ascii=False, sort_keys=True)
    expected_ids = sorted(batch.row_ids)
    usage = {"input_tokens": 0, "output_tokens": 0}
    usage_complete = True
    for attempt in range(2):
        try:
            result = llm(
                {"system": batch.prompt, "user": user},
                model=model,
                max_tokens=batch.max_completion_tokens,
                temperature=1.0,
                verbose=False,
                reasoning_effort=reasoning_effort,
                max_retries=1,
            )
        except Exception as error:
            method = "api_failure_after_structural_retry" if attempt else "api_failure"
            rows = [_blank(row, method=method) for row in batch.rows]
            failure = json.dumps(
                {"reason": method, "error_type": type(error).__name__, "error": str(error)},
                ensure_ascii=False,
            )
            for row in rows:
                row["rejected_response_json"] = failure
            return rows, None, method
        reported = result.get("usage") if isinstance(result, dict) else None
        if isinstance(reported, dict):
            usage["input_tokens"] += int(
                reported.get("prompt_tokens") or reported.get("input_tokens") or 0
            )
            usage["output_tokens"] += int(
                reported.get("completion_tokens") or reported.get("output_tokens") or 0
            )
        else:
            usage_complete = False
        try:
            parsed = json.loads(str(result.get("content") or ""))
            returned_rows = parsed.get("rows")
            if not isinstance(returned_rows, list):
                raise ValueError("response rows is not a list")
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
            if attempt == 0:
                continue
            return (
                [
                    _blank(row, method="invalid_response_after_structural_retry")
                    for row in batch.rows
                ],
                usage if usage_complete else None,
                "invalid_response_after_structural_retry",
            )

        returned_ids = [
            str(item["id"])
            for item in returned_rows
            if isinstance(item, dict) and item.get("id") is not None
        ]
        ids_match = len(returned_ids) == len(returned_rows) and sorted(
            returned_ids
        ) == expected_ids
        by_id: dict[str, Any] = {}
        for item in returned_rows:
            if isinstance(item, dict) and item.get("id") is not None:
                by_id.setdefault(str(item["id"]), item)
        rows, reasons = [], []
        for row in batch.rows:
            returned = by_id.get(str(row["id"]), {})
            assignment, reason = validate_row(
                returned,
                row,
                task=task,
                max_measurements=max_measurements,
            )
            if reason is not None:
                assignment["rejected_response_json"] = json.dumps(
                    {"reason": reason, "returned": returned},
                    ensure_ascii=False,
                    default=str,
                )
            rows.append(assignment)
            reasons.append(reason)
        retryable = not ids_match or any(
            reason is not None
            and not reason.startswith(
                ("measurement_is_not_a_plain_decimal", "too_many_measurements")
            )
            for reason in reasons
        )
        if attempt == 0 and retryable:
            continue
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
        if not ids_match:
            status = f"structural_invalid{suffix}"
        else:
            status = f"partial_invalid{suffix}" if invalid else f"valid{suffix}"
        return rows, usage if usage_complete else None, status
    raise AssertionError("unreachable")


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
) -> dict[str, Any]:
    """Write one mapping row per candidate, refusing to publish an unasked row."""
    base_assignments = base_assignments or {}
    base_model = _base_mapping_model(base_mapping_path) if base_assignments else ""
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        record_id = str(candidate["id"])
        assignment = base_assignments.get(record_id) or cache.assignments.get(record_id)
        if assignment is None:
            if record_id in cache.attempted:
                assignment = _blank(candidate, method="ambiguous_process_termination")
            else:
                raise ValueError(f"unattempted row cannot be published: {record_id}")
        rows.append(
            {
                **assignment,
                "inference_source": (
                    "base_mapping" if record_id in base_assignments else "delta_inference"
                ),
                "inference_model": (
                    base_model if record_id in base_assignments else model
                ),
            }
        )
    rows.sort(key=lambda row: str(row["cleaned_record_id"]))
    inference_models = sorted({str(row["inference_model"]) for row in rows})

    table = pa.Table.from_pylist(rows)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = mapping_path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary)
    temporary.replace(mapping_path)

    counts: dict[str, int] = {status: 0 for status in STATUSES}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    manifest = {
        "generation_version": GENERATION_VERSION,
        "routing_version": MEASUREMENT_ROUTING_VERSION,
        "task_id": config.task_id,
        "mapping_version": config.MAPPING_VERSION,
        "model": inference_models[0] if len(inference_models) == 1 else "mixed",
        "models": inference_models,
        "api_base_url": api_base_url,
        "inference": {
            "max_completion_tokens": max_completion_tokens,
            "reasoning_mode": reasoning_mode,
            "temperature": 1.0,
        },
        "prompt": config.prompt_manifest(batch_size=config.BATCH_SIZE),
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
        "base_mapping": (
            {
                "path": str(base_mapping_path),
                "sha256": file_sha256(base_mapping_path),
                "model": base_model,
                "reused_rows": len(base_assignments),
            }
            if base_mapping_path is not None
            else None
        ),
        "delta_inference": {
            "model": model,
            "rows": len(rows) - len(base_assignments),
        },
        "status_counts": counts,
        "rejected_rows": sum(1 for row in rows if row["rejected_response_json"]),
        "assignment_method_counts": _counter(row["assignment_method"] for row in rows),
        "validations": {
            "one_row_per_candidate": len(rows) == len(candidates),
            "unique_cleaned_record_ids": len({row["cleaned_record_id"] for row in rows})
            == len(rows),
            "only_ok_carries_measurements": all(
                (row["quantity_count"] > 0) == (row["status"] == "ok") for row in rows
            ),
            "maximum_measurements_per_row": (
                max((int(row["quantity_count"]) for row in rows), default=0)
                <= int(config.MAX_MEASUREMENTS_PER_ROW)
                if hasattr(config.module, "MAX_MEASUREMENTS_PER_ROW")
                else None
            ),
        },
    }
    mapping_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_base_mapping(path: Path) -> dict[str, dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = str(row.get("cleaned_record_id") or "")
        if not record_id or record_id in mapping:
            raise ValueError(f"base mapping has an empty or duplicate ID: {record_id!r}")
        mapping[record_id] = dict(row)
    return mapping


def _base_mapping_model(path: Path | None) -> str:
    if path is None:
        return ""
    manifest_path = path.with_suffix(".manifest.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = str(payload.get("model") or "")
    if not model:
        raise ValueError(f"base mapping manifest lacks a model: {manifest_path}")
    return model


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
        raise ValueError(f"endpoint profile lacks endpoint entries: {path}")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", required=True, choices=SUPPORTED_TASKS
    )
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
    parser.add_argument("--gold-fixture", type=Path, default=GOLD_FIXTURE)
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
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--token-ledger", type=Path, default=None)
    parser.add_argument("--budget-epoch", default=None)
    parser.add_argument("--budget-max-tokens", type=int, default=10_000_000)
    parser.add_argument("--start-new-budget-epoch", action="store_true")
    parser.add_argument(
        "--no-token-ledger",
        action="store_true",
        help="disable token accounting for an unmetered compatible endpoint",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default=MODEL)
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
        "--provider", choices=("openai", "openrouter", "local"), default=None
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--max-completion-tokens", type=int, default=MAX_COMPLETION_TOKENS
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = TaskConfig(args.task, args.config_module)
    if args.source is not None and args.source not in config.SOURCE_IDS:
        raise SystemExit(f"unknown --source {args.source!r}")
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
    endpoint_profile = load_endpoint_profile(profile_path, config)
    profile_digest = file_sha256(profile_path)

    only_ids = None
    if args.gold_replay:
        if args.limit or args.source:
            raise SystemExit("--gold-replay replaces --limit/--source")
        only_ids = set(gold_extract_ids(args.gold_fixture))
        print(f"gold replay: {len(only_ids):,} hand-labelled extraction rows")
    try:
        candidates = candidate_rows(
            records_path,
            config,
            source_id=args.source,
            limit=args.limit,
            only_ids=only_ids,
        )
    except MissingGoldRows as missing:
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
        only_ids = only_ids - absent
        print(f"  proceeding over the remaining {len(only_ids):,} gold row(s)")
        candidates = candidate_rows(
            records_path, config, source_id=args.source, only_ids=only_ids
        )
    per_source = _counter(row["source_id"] for row in candidates)
    candidate_ids = {str(row["id"]) for row in candidates}
    base_assignments = (
        {
            record_id: row
            for record_id, row in load_base_mapping(base_mapping_path).items()
            if record_id in candidate_ids
        }
        if base_mapping_path is not None
        else {}
    )
    delta_candidates = [
        row for row in candidates if str(row["id"]) not in base_assignments
    ]
    planned_batches = plan_batches(
        delta_candidates,
        config,
        attempted=set(),
        endpoint_profile=endpoint_profile,
        profile_digest=profile_digest,
        model=args.model,
        max_completion_tokens=args.max_completion_tokens,
    )
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

    if args.inventory_only:
        return 0
    if args.no_token_ledger and not args.base_url:
        raise SystemExit("--no-token-ledger requires --base-url")
    if not args.no_token_ledger and not args.budget_epoch:
        raise SystemExit("--budget-epoch is required for a paid run")

    # Gate on the reviewed prompt before the ledger opens, so a mismatch is free.
    prompt_manifest = config.prompt_manifest(batch_size=config.BATCH_SIZE)
    print(
        f"prompt {prompt_manifest['prompt_version']} "
        f"template {prompt_manifest['template_sha256'][:16]}"
    )

    cache_dir = args.cache_dir or Path(
        "outputs/chembl_tool/measurement_resolution_generation"
    )
    cache_path = cache_dir / config.task_id / "requests.jsonl"
    cache = SubmissionCache(cache_path)
    ledger = (
        None
        if args.no_token_ledger
        else TokenLedger(
            args.token_ledger
            or Path(
                "outputs/chembl_tool/measurement_resolution_generation/token_ledger.json"
            ),
            epoch=args.budget_epoch,
            start_new_epoch=args.start_new_budget_epoch,
            max_tokens=args.budget_max_tokens,
        )
    )
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
    print(f"unattempted requests this run: {len(batches):,}")

    if args.base_url:
        client, selected_credential_env = openai_compatible_client(
            base_url=args.base_url,
            provider=args.provider,
            env_file=args.env_file,
            credential_env=args.api_key_env,
            max_connections=args.workers,
            timeout_s=args.request_timeout_s,
        )
        llm = openai_compatible_llm(client)
        model = args.model
        api_base_url = args.base_url
    else:
        llm, _ = distillation_client()
        model = args.model
        api_base_url = "distillation_default"
        selected_credential_env = ""
    print(f"model {model}")
    pending = iter(batches)
    in_flight: dict[Any, RequestBatch] = {}
    next_batch: RequestBatch | None = None
    finished = exhausted = False
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while True:
            while len(in_flight) < args.workers and not finished:
                if next_batch is None:
                    try:
                        next_batch = next(pending)
                    except StopIteration:
                        finished = True
                        break
                if ledger is not None and not ledger.reserve(
                    next_batch.request_id, 2 * next_batch.reservation_tokens
                ):
                    break
                cache.submit(
                    next_batch,
                    epoch=ledger.epoch if ledger is not None else "unmetered",
                    credential_env=selected_credential_env,
                    model=model,
                    base_url=api_base_url,
                )
                in_flight[
                    executor.submit(
                        query_batch,
                        next_batch,
                        llm=llm,
                        model=model,
                        task=config.task_id,
                        max_measurements=getattr(
                            config, "MAX_MEASUREMENTS_PER_ROW", None
                        ),
                        reasoning_effort=getattr(
                            config, "REASONING_EFFORT", REASONING_EFFORT
                        ),
                    )
                ] = next_batch
                next_batch = None
            if not in_flight:
                exhausted = next_batch is not None
                break
            done, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
            for future in done:
                batch = in_flight.pop(future)
                try:
                    rows, usage, status = future.result()
                except Exception:
                    rows = [_blank(row, method="worker_failure") for row in batch.rows]
                    usage, status = None, "worker_failure"
                if ledger is not None:
                    ledger.complete(batch.request_id, usage)
                cache.terminal(
                    batch, assignments=rows, usage=usage, response_status=status
                )
    if exhausted:
        assert ledger is not None
        ledger.mark_exhausted()
        print(
            "budget epoch exhausted; nothing published. "
            "Resume with a new --budget-epoch; attempted rows are never re-asked."
        )
        return BUDGET_EXHAUSTED_EXIT_CODE

    cache = SubmissionCache(cache_path)
    manifest = materialize(
        candidates,
        cache,
        config,
        mapping_path=mapping_path,
        records_path=records_path,
        profile_path=profile_path,
        model=model,
        api_base_url=api_base_url,
        max_completion_tokens=args.max_completion_tokens,
        reasoning_mode=getattr(config, "REASONING_EFFORT", REASONING_EFFORT),
        base_assignments=base_assignments,
        base_mapping_path=base_mapping_path,
    )
    print(json.dumps(manifest["status_counts"], indent=2))
    print(f"rejected rows: {manifest['rejected_rows']:,}")
    print(f"wrote {mapping_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
