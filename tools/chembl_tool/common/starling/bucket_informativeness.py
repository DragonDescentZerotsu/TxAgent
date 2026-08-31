"""Score the direct-label informativeness of active normalized-v7 pair buckets."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow.parquet as pq
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tools.chembl_tool.common.openai_reasoning_client import OpenAICompatibleClient
from tools.chembl_tool.common.starling.build_reference_semantics_mapping import (
    BUDGET_EXHAUSTED_EXIT_CODE,
    TokenLedger,
)
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.json_utils import atomic_output_path, write_json_atomic


VERSION = "bucket_informativeness.v1"
CACHE_VERSION = "bucket_informativeness_requests.v1"
PROMPT_VERSION = "bucket_informativeness_prompt.v1"
DEFAULT_MODEL = "openai/gpt-5.6-luna"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_BUDGET = 10_000_000
MAX_COMPLETION_TOKENS = 4_096
MAX_RECORDS = 12
MAX_BATCH_ITEMS = 20
MAX_PROMPT_BYTES = 100_000
TASKS = ("bbb_martins",)
TEMPLATE_PATH = Path(__file__).parent / "prompt_templates/bucket_informativeness_v1.jinja"
DEFAULT_ROOTS = {
    task: Path(f"outputs/chembl_tool/tasks/{task}/evidence_library/starling_normalized_v7")
    for task in TASKS
}
FORBIDDEN_FIELDS = {
    "canonical_record_id",
    "canonical_smiles",
    "cleaned_record_id",
    "direct_vote_label",
    "direct_vote_unit_id",
    "extraction_id",
    "global_identifier",
    "molecule_name",
    "pmid",
    "source_index",
    "source_record_id",
    "source_row_number",
    "source_smiles",
    "smiles",
}
DIRECT_LABEL_DEFINITION = (
    "Whether systemic administration produces meaningful or adequate CNS access "
    "(positive) versus restricted or poor CNS access (negative)."
)
SOURCE_COLUMNS = {
    "direct_bbb": (
        "support_text",
        "bbb_permeability_label",
        "bbb_transport_label",
        "assay_model",
        "species",
        "qualifying_conditions",
        "extra_details",
    ),
    "passive_permeability": (
        "support_text",
        "assay_type",
        "biological_system",
        "metric_uncertainty",
        "passive_bbb_interpretation",
        "qualifying_conditions",
        "extra_details",
        "needs_more_context",
    ),
    "efflux_transport": (
        "support_text",
        "transporter_identifier",
        "evidence_type",
        "interaction_conclusion",
        "assay_system",
        "perturbation",
        "qualifying_conditions",
        "extra_details",
        "needs_more_context",
    ),
    "influx_transport": (
        "support_text",
        "mediator_name",
        "mediator_identifier",
        "transport_mechanism",
        "evidence_basis",
        "assay_model",
        "qualifying_conditions",
        "extra_details",
        "needs_more_context",
    ),
}
OUTPUT_COLUMNS = [
    "task_id",
    "pair_bucket_key",
    "informativeness_score",
    "rationale",
    "requested_model",
    "served_model",
    "reasoning_effort",
    "prompt_version",
    "payload_sha256",
]
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "bucket_informativeness",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["items"],
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", "informativeness_score", "rationale"],
                        "properties": {
                            "id": {"type": "string"},
                            "informativeness_score": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "rationale": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 400,
                            },
                        },
                    },
                }
            },
        },
    },
}


class BudgetExhausted(RuntimeError):
    pass


def _clean(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def visible_record(
    record: Mapping[str, Any], source_columns: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    fields = {
        "source_measurement": _clean(record.get("measurement_text")),
        "source_unit": _clean(record.get("unit_text")),
        "canonical_measurement": _clean(record.get("canonical_measurement_text")),
        "canonical_value": _clean(record.get("finite_scalar_value")),
        "canonical_category": _clean(record.get("canonical_category_id")),
        "canonical_category_rank": _clean(record.get("canonical_category_rank")),
        "reference_scope": _clean(record.get("canonical_reference_scope")),
        "reference_basis": _clean(record.get("canonical_reference_basis")),
    }
    for field in source_columns.get(str(record.get("source_id") or ""), ()):
        if field not in FORBIDDEN_FIELDS:
            fields[field] = _clean(record.get(field))
    molecule_name = str(record.get("molecule_name") or "").strip()
    if len(molecule_name) >= 3:
        fields = {
            key: re.sub(re.escape(molecule_name), "[molecule]", value, flags=re.IGNORECASE)
            if isinstance(value, str)
            else value
            for key, value in fields.items()
        }
    return {key: value for key, value in fields.items() if value is not None}


def _evenly_spaced(records: Sequence[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    if count <= 0 or not records:
        return []
    if len(records) <= count:
        return list(records)
    if count == 1:
        return [records[0]]
    return [records[round(index * (len(records) - 1) / (count - 1))] for index in range(count)]


def select_records(
    records: Sequence[Mapping[str, Any]],
    *,
    measurement_kind: str,
    source_columns: Mapping[str, Sequence[str]],
    limit: int = MAX_RECORDS,
) -> list[Mapping[str, Any]]:
    ordered = sorted(records, key=lambda row: str(row.get("canonical_record_id") or ""))
    if len(ordered) <= limit:
        return ordered

    def measurement_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        if measurement_kind == "continuous":
            value = _clean(row.get("finite_scalar_value"))
            return (value is None, float(value) if value is not None else 0.0, str(row.get("canonical_record_id") or ""))
        rank = _clean(row.get("canonical_category_rank"))
        return (
            rank is None,
            float(rank) if rank is not None else 0.0,
            str(row.get("canonical_category_id") or ""),
            str(row.get("canonical_record_id") or ""),
        )

    chosen = {
        str(row.get("canonical_record_id") or ""): row
        for row in _evenly_spaced(sorted(ordered, key=measurement_key), min(6, limit))
    }
    seen_molecules = {str(row.get("canonical_smiles") or "") for row in chosen.values()}
    seen_contexts = {
        json.dumps(visible_record(row, source_columns), ensure_ascii=False, sort_keys=True)
        for row in chosen.values()
    }
    diverse = sorted(
        ordered,
        key=lambda row: hashlib.sha256(
            json.dumps(
                {
                    "molecule": row.get("canonical_smiles"),
                    "record": visible_record(row, source_columns),
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    )
    for row in diverse:
        if len(chosen) == limit:
            break
        record_id = str(row.get("canonical_record_id") or "")
        molecule = str(row.get("canonical_smiles") or "")
        context = json.dumps(visible_record(row, source_columns), ensure_ascii=False, sort_keys=True)
        if record_id not in chosen and (molecule not in seen_molecules or context not in seen_contexts):
            chosen[record_id] = row
            seen_molecules.add(molecule)
            seen_contexts.add(context)
    for row in ordered:
        if len(chosen) == limit:
            break
        chosen.setdefault(str(row.get("canonical_record_id") or ""), row)
    return sorted(chosen.values(), key=lambda row: str(row.get("canonical_record_id") or ""))


def validate_response(response: Any, size: int) -> list[dict[str, Any]]:
    if not isinstance(response, Mapping) or set(response) != {"items"}:
        raise ValueError("response must contain only items")
    items = response["items"]
    if not isinstance(items, list) or len(items) != size:
        raise ValueError("response has wrong item count")
    expected = {str(index) for index in range(size)}
    output = {}
    for item in items:
        if not isinstance(item, Mapping) or set(item) != {
            "id",
            "informativeness_score",
            "rationale",
        }:
            raise ValueError("response item fields differ from contract")
        item_id = str(item["id"])
        if item_id not in expected or item_id in output:
            raise ValueError("response contains an invalid item ID")
        score = item["informativeness_score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError("informativeness score must be finite numeric")
        if not 0 <= float(score) <= 1:
            raise ValueError("informativeness score is outside [0, 1]")
        rationale = str(item["rationale"]).strip()
        if not rationale or len(rationale.split()) > 40 or len(rationale) > 400:
            raise ValueError("rationale must contain at most 40 words and 400 characters")
        output[item_id] = {
            "id": item_id,
            "informativeness_score": float(
                Decimal(str(score)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            ),
            "rationale": rationale,
        }
    return [output[str(index)] for index in range(size)]


@lru_cache(maxsize=1)
def _template() -> Any:
    environment = Environment(
        loader=FileSystemLoader(str(TEMPLATE_PATH.parent)),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    return environment.get_template(TEMPLATE_PATH.name)


def render_batch(items: Sequence[Mapping[str, Any]], definition: str) -> str:
    visible = [
        {
            "id": str(index),
            "payload": json.dumps(item["payload"], ensure_ascii=False, indent=2, sort_keys=True),
        }
        for index, item in enumerate(items)
    ]
    return _template().render(direct_label_definition=definition, items=visible)


def _payload_sha256(payload: Mapping[str, Any], definition: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "direct_label_definition": definition,
                "payload": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _input_hashes(stage: Path) -> dict[str, str]:
    return {
        name: file_sha256(stage / name)
        for name in (
            "manifest.json",
            "records.parquet",
            "pair_bucket_records.parquet",
            "pair_bucket_distance_calibration.json.gz",
        )
    }


def build_inventory(
    *, task_id: str, normalized_root: Path, output_dir: Path, rebuild: bool = False
) -> dict[str, Any]:
    stage = normalized_root / "03_pair_buckets"
    required = [
        stage / "manifest.json",
        stage / "records.parquet",
        stage / "pair_bucket_records.parquet",
        stage / "pair_bucket_distance_calibration.json.gz",
        TEMPLATE_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing informativeness input(s): {missing}")
    hashes = _input_hashes(stage)
    hashes["prompt_template"] = file_sha256(TEMPLATE_PATH)
    manifest_path = output_dir / "manifest.json"
    payloads_path = output_dir / "payloads.jsonl"
    if not rebuild and manifest_path.is_file() and payloads_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("version") == VERSION and prior.get("input_hashes") == hashes:
            return prior

    definition = DIRECT_LABEL_DEFINITION
    source_columns = SOURCE_COLUMNS
    with gzip.open(stage / "pair_bucket_distance_calibration.json.gz", "rt", encoding="utf-8") as handle:
        calibration = json.load(handle)
    calibration_rows = calibration.get("buckets") or {}
    target_keys = set(calibration_rows)

    sidecar = pq.read_table(
        stage / "pair_bucket_records.parquet",
        columns=["pair_bucket_key", "assay_transfer_eligible"],
    ).to_pylist()
    sidecar_keys = {
        str(row["pair_bucket_key"])
        for row in sidecar
        if row.get("assay_transfer_eligible") is True
    }
    if target_keys != sidecar_keys:
        raise ValueError(
            "calibration and eligible sidecar bucket keys differ: "
            f"calibration_only={len(target_keys - sidecar_keys)}, "
            f"sidecar_only={len(sidecar_keys - target_keys)}"
        )

    available = set(pq.read_schema(stage / "records.parquet").names)
    columns = {
        "assay_transfer_eligible",
        "canonical_category_id",
        "canonical_category_rank",
        "canonical_endpoint_name",
        "canonical_measurement_scale_id",
        "canonical_measurement_text",
        "canonical_pair_fields_json",
        "canonical_record_id",
        "canonical_reference_basis",
        "canonical_reference_scope",
        "canonical_smiles",
        "canonical_unit_text",
        "finite_scalar_value",
        "measurement_kind",
        "measurement_text",
        "molecule_name",
        "pair_bucket_key",
        "source_id",
        "unit_text",
        *(field for fields in source_columns.values() for field in fields),
    }
    records_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parquet = pq.ParquetFile(stage / "records.parquet")
    for batch in parquet.iter_batches(batch_size=10_000, columns=sorted(columns & available)):
        for record in batch.to_pylist():
            key = str(record.get("pair_bucket_key") or "")
            if record.get("assay_transfer_eligible") is True and key in target_keys:
                records_by_bucket[key].append(record)
    if set(records_by_bucket) != target_keys:
        raise ValueError("eligible Stage-03 records do not cover every calibration bucket")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()
    with atomic_output_path(payloads_path) as temporary:
        with temporary.open("w", encoding="utf-8") as handle:
            for key in sorted(target_keys):
                records = records_by_bucket[key]
                calibration_row = calibration_rows[key]
                if len(records) != int(calibration_row["record_count"]):
                    raise ValueError(f"calibration record count mismatch for {key}")
                first = records[0]
                kind = str(calibration_row["measurement_kind"])
                sampled = select_records(
                    records,
                    measurement_kind=kind,
                    source_columns=source_columns,
                )
                scopes = Counter(str(row.get("canonical_reference_scope") or "__unknown__") for row in records)
                bases = Counter(str(row.get("canonical_reference_basis") or "__unknown__") for row in records)
                pair_fields = json.loads(str(first.get("canonical_pair_fields_json") or "{}"))
                payload = {
                    "bucket_identity": json.loads(key),
                    "source": str(first.get("source_id") or ""),
                    "endpoint": str(first.get("canonical_endpoint_name") or ""),
                    "unit": str(first.get("canonical_unit_text") or ""),
                    "measurement_kind": kind,
                    "measurement_scale": _clean(first.get("canonical_measurement_scale_id")),
                    "canonical_context": pair_fields,
                    "record_count": len(records),
                    "distinct_molecule_count": int(calibration_row["distinct_molecule_count"]),
                    "reference_scope_counts": dict(sorted(scopes.items())),
                    "reference_basis_counts": dict(sorted(bases.items())),
                    "all_records_shown": len(records) <= MAX_RECORDS,
                    "sampled_record_count": len(sampled),
                    "sampled_records": [visible_record(row, source_columns) for row in sampled],
                }
                item = {
                    "task_id": task_id,
                    "pair_bucket_key": key,
                    "payload": payload,
                    "payload_sha256": _payload_sha256(payload, definition),
                }
                item["prompt_bytes"] = len(
                    render_batch([item], definition).encode("utf-8")
                )
                if item["prompt_bytes"] > MAX_PROMPT_BYTES:
                    raise ValueError(f"single bucket prompt exceeds {MAX_PROMPT_BYTES} bytes: {key}")
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                source_counts[str(first.get("source_id") or "")] += 1
                kind_counts[kind] += 1
                size_counts[_size_band(len(records))] += 1

    requests = output_dir / "requests.jsonl"
    requests.touch(exist_ok=True)
    manifest = {
        "version": VERSION,
        "prompt_version": PROMPT_VERSION,
        "task_id": task_id,
        "status": "inventory_complete",
        "direct_label_definition": definition,
        "target_buckets": len(target_keys),
        "completed_buckets": 0,
        "pending_buckets": len(target_keys),
        "record_sample_cap": MAX_RECORDS,
        "input_hashes": hashes,
        "payloads_sha256": file_sha256(payloads_path),
        "requests_sha256": file_sha256(requests),
        "bucket_counts_by_source": dict(sorted(source_counts.items())),
        "bucket_counts_by_measurement_kind": dict(sorted(kind_counts.items())),
        "bucket_counts_by_size_band": dict(sorted(size_counts.items())),
        "score_semantics": "semantic_direct_label_informativeness_independent_of_support_and_calibration",
        "score_resolution": 0.01,
    }
    write_json_atomic(manifest_path, manifest)
    _write_example_prompt(output_dir, task_id=task_id, definition=definition)
    return manifest


def _size_band(size: int) -> str:
    if size == 1:
        return "1"
    if size <= 5:
        return "2-5"
    if size <= 12:
        return "6-12"
    if size < 20:
        return "13-19"
    return "20+"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_completed(
    requests_path: Path,
    *,
    model: str,
    base_url: str,
    template_sha256: str,
    payload_hashes: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    completed = {}
    for event in _read_jsonl(requests_path):
        if (
            event.get("cache_version") != CACHE_VERSION
            or event.get("requested_model") != model
            or event.get("base_url") != base_url.rstrip("/")
            or event.get("template_sha256") != template_sha256
            or event.get("status") != "success"
        ):
            continue
        for item in event.get("items") or ():
            key = str(item.get("pair_bucket_key") or "")
            if payload_hashes.get(key) != item.get("payload_sha256"):
                continue
            prior = completed.get(key)
            if prior is not None and {
                field: prior[field]
                for field in ("pair_bucket_key", "payload_sha256", "informativeness_score", "rationale")
            } != {
                field: item[field]
                for field in ("pair_bucket_key", "payload_sha256", "informativeness_score", "rationale")
            }:
                raise ValueError(f"conflicting cached score for {key}")
            completed[key] = {**item, "event": event}
    return completed


def _pilot_items(items: Sequence[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    if size >= len(items):
        return list(items)
    strata: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        payload = item["payload"]
        strata[
            (
                str(payload["source"]),
                str(payload["measurement_kind"]),
                _size_band(int(payload["record_count"])),
                str(payload["endpoint"]),
            )
        ].append(item)
    for rows in strata.values():
        rows.sort(key=lambda row: hashlib.sha256(str(row["pair_bucket_key"]).encode()).hexdigest())
    selected = []
    ordered_strata = sorted(strata, key=lambda value: hashlib.sha256(repr(value).encode()).hexdigest())
    while len(selected) < size:
        changed = False
        for stratum in ordered_strata:
            if strata[stratum]:
                selected.append(strata[stratum].pop(0))
                changed = True
                if len(selected) == size:
                    break
        if not changed:
            break
    return selected


def _pack_batches(
    items: Sequence[dict[str, Any]], definition: str
) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda row: (int(row["prompt_bytes"]), str(row["pair_bucket_key"]))):
        candidate = [*current, item]
        if current and (
            len(candidate) > MAX_BATCH_ITEMS
            or len(render_batch(candidate, definition).encode("utf-8")) > MAX_PROMPT_BYTES
        ):
            batches.append(current)
            current = [item]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def _usage(usage: Mapping[str, Any] | None) -> dict[str, int] | None:
    if not usage:
        return None
    return {
        "input_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
    }


def _query_batch(
    items: Sequence[dict[str, Any]],
    *,
    definition: str,
    client: OpenAICompatibleClient,
    model: str,
    base_url: str,
    reasoning_effort: str,
    ledger: TokenLedger,
    epoch: str,
    max_attempts: int,
) -> dict[str, Any]:
    prompt = render_batch(items, definition)
    template_sha256 = file_sha256(TEMPLATE_PATH)
    batch_id = hashlib.sha256(
        json.dumps(
            {
                "task_id": items[0]["task_id"],
                "model": model,
                "payloads": [item["payload_sha256"] for item in items],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    errors = []
    for attempt in range(1, max_attempts + 1):
        request_id = f"{batch_id}:{attempt}"
        reservation = len(prompt.encode("utf-8")) + 1_024 + MAX_COMPLETION_TOKENS
        if not ledger.reserve(request_id, reservation):
            raise BudgetExhausted(request_id)
        call_usage = None
        result = None
        try:
            result = client.chat_json([{"role": "user", "content": prompt}])
            call_usage = _usage(result.get("usage"))
            validated = validate_response(result.get("content"), len(items))
            served_model = str(result.get("model") or "")
            if "gpt-5.6-luna" not in served_model:
                raise ValueError(f"unexpected served model: {served_model!r}")
        except Exception as exc:
            errors.append(str(exc))
            validated = None
        finally:
            ledger.complete(request_id, call_usage)
        if validated is not None and result is not None:
            return {
                "cache_version": CACHE_VERSION,
                "status": "success",
                "task_id": items[0]["task_id"],
                "batch_id": batch_id,
                "requested_model": model,
                "served_model": str(result.get("model") or ""),
                "base_url": base_url.rstrip("/"),
                "credential_env": "OPENROUTER_API_KEY",
                "reasoning_effort": reasoning_effort,
                "budget_epoch": epoch,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "template_sha256": template_sha256,
                "usage": call_usage or {},
                "attempts": attempt,
                "validation_errors": errors,
                "raw_content": result.get("raw_content"),
                "reasoning_content": result.get("reasoning_content"),
                "items": [
                    {
                        "pair_bucket_key": source["pair_bucket_key"],
                        "payload_sha256": source["payload_sha256"],
                        "informativeness_score": response["informativeness_score"],
                        "rationale": response["rationale"],
                    }
                    for source, response in zip(items, validated, strict=True)
                ],
            }
    return {
        "cache_version": CACHE_VERSION,
        "status": "failed",
        "task_id": items[0]["task_id"],
        "batch_id": batch_id,
        "requested_model": model,
        "base_url": base_url.rstrip("/"),
        "reasoning_effort": reasoning_effort,
        "budget_epoch": epoch,
        "template_sha256": template_sha256,
        "attempts": max_attempts,
        "validation_errors": errors,
        "items": [],
    }


def _write_example_prompt(output_dir: Path, *, task_id: str, definition: str) -> None:
    items = _read_jsonl(output_dir / "payloads.jsonl")
    pilot = _pilot_items(items, 1)
    if pilot:
        (output_dir / "example_prompt.txt").write_text(
            render_batch(pilot, definition), encoding="utf-8"
        )


def _write_progress(
    output_dir: Path,
    *,
    model: str,
    base_url: str,
    reasoning_effort: str,
    run_status: str | None = None,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payloads = _read_jsonl(output_dir / "payloads.jsonl")
    payload_hashes = {str(item["pair_bucket_key"]): str(item["payload_sha256"]) for item in payloads}
    completed = _load_completed(
        output_dir / "requests.jsonl",
        model=model,
        base_url=base_url,
        template_sha256=str(manifest["input_hashes"]["prompt_template"]),
        payload_hashes=payload_hashes,
    )
    rows = []
    for key in sorted(completed):
        item = completed[key]
        event = item["event"]
        rows.append(
            {
                "task_id": manifest["task_id"],
                "pair_bucket_key": key,
                "informativeness_score": item["informativeness_score"],
                "rationale": item["rationale"],
                "requested_model": event["requested_model"],
                "served_model": event.get("served_model"),
                "reasoning_effort": reasoning_effort,
                "prompt_version": PROMPT_VERSION,
                "payload_sha256": item["payload_sha256"],
            }
        )
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    parquet_path = output_dir / "bucket_informativeness.parquet"
    with atomic_output_path(parquet_path) as temporary:
        frame.to_parquet(temporary, index=False)
    token_usage: Counter[str] = Counter()
    successful_requests = 0
    failed_requests = 0
    for event in _read_jsonl(output_dir / "requests.jsonl"):
        if event.get("requested_model") != model or event.get("base_url") != base_url.rstrip("/"):
            continue
        if event.get("status") == "success":
            successful_requests += 1
            token_usage.update(event.get("usage") or {})
        elif event.get("status") == "failed":
            failed_requests += 1
    total = len(payloads)
    manifest.update(
        {
            "status": run_status or ("complete" if len(rows) == total else "partial"),
            "completed_buckets": len(rows),
            "pending_buckets": total - len(rows),
            "requested_model": model,
            "base_url": base_url.rstrip("/"),
            "credential_env": "OPENROUTER_API_KEY",
            "reasoning_effort": reasoning_effort,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "token_usage": dict(token_usage),
            "requests_sha256": file_sha256(output_dir / "requests.jsonl"),
            "output_sha256": file_sha256(parquet_path),
        }
    )
    write_json_atomic(manifest_path, manifest)
    if rows and not (output_dir / "example_response.json").exists():
        write_json_atomic(output_dir / "example_response.json", rows[0])
    return manifest


def run_generation(args: argparse.Namespace) -> int:
    normalized_root = Path(args.normalized_root)
    output_dir = Path(args.output_dir)
    manifest = build_inventory(
        task_id=args.task,
        normalized_root=normalized_root,
        output_dir=output_dir,
        rebuild=args.rebuild_inventory,
    )
    if args.inventory_only:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if _input_hashes(normalized_root / "03_pair_buckets") | {"prompt_template": file_sha256(TEMPLATE_PATH)} != manifest["input_hashes"]:
        raise RuntimeError("Stage-03 or prompt inputs changed after inventory creation")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is unset; refusing paid generation")
    if len(api_key.strip()) < 20:
        raise RuntimeError(
            f"{args.api_key_env} appears to be a placeholder; refusing paid generation"
        )

    payloads = _read_jsonl(output_dir / "payloads.jsonl")
    payload_hashes = {str(item["pair_bucket_key"]): str(item["payload_sha256"]) for item in payloads}
    completed = _load_completed(
        output_dir / "requests.jsonl",
        model=args.model,
        base_url=args.base_url,
        template_sha256=str(manifest["input_hashes"]["prompt_template"]),
        payload_hashes=payload_hashes,
    )
    targets = _pilot_items(payloads, args.pilot_size) if args.pilot_size else payloads
    pending = [item for item in targets if item["pair_bucket_key"] not in completed]
    if not pending:
        result = _write_progress(
            output_dir,
            model=args.model,
            base_url=args.base_url,
            reasoning_effort=args.reasoning_effort,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    ledger = TokenLedger(
        Path(args.token_ledger),
        epoch=args.budget_epoch,
        start_new_epoch=args.start_new_budget_epoch,
        max_tokens=args.budget_max_tokens,
    )
    client = OpenAICompatibleClient(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout_s,
        max_tokens=MAX_COMPLETION_TOKENS,
        temperature=None,
        tool_service_url="http://127.0.0.1:8766",
        enable_group_tools=False,
        max_tool_rounds=0,
        reasoning_effort=args.reasoning_effort,
        enable_thinking=False,
        request_extra_body={"provider": {"require_parameters": True}},
        response_format=RESPONSE_FORMAT,
    )
    batches = _pack_batches(pending, str(manifest["direct_label_definition"]))
    if not completed and batches and len(batches[0]) > 1:
        first = batches[0].pop(0)
        batches.insert(0, [first])
        if not batches[1]:
            batches.pop(1)
    exhausted = False
    failures = 0
    next_batch = 0
    futures: dict[concurrent.futures.Future[Any], list[dict[str, Any]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        while next_batch < len(batches) or futures:
            while not exhausted and next_batch < len(batches) and len(futures) < args.workers:
                batch = batches[next_batch]
                next_batch += 1
                future = executor.submit(
                    _query_batch,
                    batch,
                    definition=str(manifest["direct_label_definition"]),
                    client=client,
                    model=args.model,
                    base_url=args.base_url,
                    reasoning_effort=args.reasoning_effort,
                    ledger=ledger,
                    epoch=args.budget_epoch,
                    max_attempts=args.max_attempts,
                )
                futures[future] = batch
            if not futures:
                break
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                futures.pop(future)
                try:
                    event = future.result()
                except BudgetExhausted:
                    exhausted = True
                    continue
                except Exception as exc:
                    failures += 1
                    event = {
                        "cache_version": CACHE_VERSION,
                        "status": "failed",
                        "requested_model": args.model,
                        "base_url": args.base_url.rstrip("/"),
                        "error": str(exc),
                        "items": [],
                    }
                _append_jsonl(output_dir / "requests.jsonl", event)
                failures += event.get("status") != "success"
    if exhausted:
        ledger.mark_exhausted()
    result = _write_progress(
        output_dir,
        model=args.model,
        base_url=args.base_url,
        reasoning_effort=args.reasoning_effort,
        run_status="budget_exhausted" if exhausted else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if exhausted:
        return BUDGET_EXHAUSTED_EXIT_CODE
    return 1 if failures else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="bbb_martins")
    parser.add_argument("--normalized-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--rebuild-inventory", action="store_true")
    parser.add_argument("--pilot-size", type=int)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--token-ledger", type=Path)
    parser.add_argument("--budget-epoch", default="bbb_bucket_informativeness_openrouter_v1")
    parser.add_argument("--start-new-budget-epoch", action="store_true")
    parser.add_argument("--budget-max-tokens", type=int, default=DEFAULT_BUDGET)
    args = parser.parse_args(argv)
    args.normalized_root = args.normalized_root or DEFAULT_ROOTS[args.task]
    args.output_dir = args.output_dir or args.normalized_root / "bucket_informativeness_v1"
    args.token_ledger = args.token_ledger or args.output_dir / "openrouter_token_ledger.json"
    if args.workers < 1 or args.max_attempts < 1 or args.budget_max_tokens < 1:
        parser.error("workers, attempts, and token budget must be positive")
    if args.pilot_size is not None and args.pilot_size < 1:
        parser.error("pilot size must be positive")
    if args.api_key_env != "OPENROUTER_API_KEY":
        parser.error("bucket informativeness is frozen to OPENROUTER_API_KEY")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return run_generation(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
