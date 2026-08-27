"""Review scientifically suspicious wide-gap pair buckets before collapse."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import itertools
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from tools.chembl_tool.common.openai_reasoning_client import OpenAICompatibleClient
from tools.chembl_tool.common.starling.normalization.audit import write_parquet
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256


VERSION = "final_endpoint_pruning.v2"
SCHEMA_VERSION = "final_endpoint_pruning_review.v1"
ARTIFACT_DIR = "final_endpoint_pruning_v2"
MANIFEST_FILENAME = "manifest.json"
DECISIONS_FILENAME = "decisions.parquet"
REVIEWS_FILENAME = "reviews.jsonl"
CACHE_FILENAME = ".review_cache.jsonl"
DEFAULT_MODEL = "gpt-5.4-mini-2026-03-17"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
GAP_DECADES = 2.0
MINIMUM_RECORDS = 6
MINIMUM_DISTINCT_VALUES = 3
REVIEWERS = ("a", "b")
KEEP_REASONS = {"fits_bucket", "valid_extreme", "insufficient_evidence"}
DROP_REASONS = {
    "wrong_quantity_or_endpoint",
    "wrong_unit_or_scale",
    "corrupt_extraction",
}

_ALWAYS_VISIBLE = {
    "canonical_record_id",
    "endpoint_name",
    "measurement_text",
    "unit_text",
    "measurement_resolution_origin",
    "measurement_resolution_input_measurement",
    "measurement_resolution_input_unit",
    "finite_scalar_value",
    "variation_value",
    "measurement_kind",
    "canonical_endpoint_name",
    "canonical_measurement_text",
    "canonical_unit_text",
    "canonical_measurement_scale_id",
    "canonical_category_id",
    "canonical_category_rank",
    "canonical_reference_scope",
    "canonical_reference_basis",
    "canonical_kinetic_symbol",
    "support_text",
}
_SEMANTIC_TERMS = (
    "endpoint",
    "measurement",
    "metric",
    "unit",
    "assay",
    "species",
    "population",
    "context",
    "condition",
    "dose",
    "exposure",
    "formulation",
    "matrix",
    "comparator",
    "reference",
    "mechanism",
    "transport",
    "study",
    "time",
    "light",
    "support",
    "extra",
    "target",
    "confidence",
    "uncertainty",
    "statistic",
    "site",
    "enzyme",
    "substrate",
    "mediator",
    "biological",
    "reaction",
    "severity",
    "aop",
    "kinetic",
    "category",
    "report",
    "interpretation",
    "count",
    "tested",
)
_PRIVATE_PREFIXES = (
    "source_",
    "direct_",
    "retrieval_",
    "duplicate_",
    "deduplication_",
    "organization_",
    "structure_",
    "auxiliary_",
    "measurement_resolution_",
    "measurement_unit_mapping_",
    "measurement_routing_",
    "measurement_parse_",
    "measurement_unit_",
    "measurement_numeric_",
    "measurement_semantics_",
    "pre_resolution_",
    "resolved_",
    "unit_notation_",
    "assay_transfer_",
    "categorical_encoding_",
    "canonicalization_",
    "canonical_semantics_",
    "canonical_endpoint_policy_",
    "canonical_endpoint_producer_",
    "canonical_endpoint_source_",
    "canonical_endpoint_rule_",
    "canonical_pair_producer_",
    "canonical_unit_rule_",
    "canonical_unit_policy_",
    "canonical_unit_resolution_",
    "reference_semantics_assignment_",
)
_PRIVATE_FIELDS = {
    "pmid",
    "doi",
    "smiles",
    "canonical_smiles",
    "molecule_id",
    "molecule_name",
    "cleaned_record_id",
    "global_identifier",
    "extraction_id",
    "paragraph_idx",
    "record_contract_version",
    "source_column_policy_version",
    "group_id",
    "needs_more_context",
    "condition_group",
    "condition_atoms",
    "condition_scope",
    "condition_key_status",
}

_PROMPT = """You are performing conservative scientific data quality review.

The numeric gap only triggered review. A rare or extreme value is not evidence of error.
Drop a row only when its displayed semantics show that it is a different scientific
quantity or endpoint, has a wrong unit or scale, or is a corrupt/misparsed extraction.
When evidence is ambiguous, keep the row. Do not convert, repair, or reassign rows.

The canonical scalar and unit may already contain a correct deterministic unit conversion.
Different raw and canonical unit strings are not a mismatch when their values are exactly
equivalent. For example, 3.5 cm/h equals 0.0009722 cm/s, 40.4 mg/cm^2 equals
40,400 ug/cm^2, and mg/min/mg simplifies to 1/min. Keep such rows even when the converted
value is extreme. Drop a scale only when the canonical value cannot be justified by the
displayed measurement and unit; never infer an error from unit spelling alone.

For each extracted row, measurement_text/unit_text are the original source fields,
llm_refined_extraction is the literal measurement and unit returned by the earlier
extraction LLM, and canonical_measurement_text/canonical_unit_text/finite_scalar_value are
the final normalized tuple. Use this full lineage to detect a wrong scale or quantity.
The endpoint-aware refined extraction may repair a lost minus sign or table-header scale.
Keep that repair when the raw literal is physically impossible for the endpoint and the
refined value is scientifically plausible. If the raw factor is explicit and plausible
but the refined extraction silently reverses it, drop the row. Treat OCR-like raw forms
such as `101` or an implausible unsigned exponent as ambiguous; they do not override the
refined extraction. A clearly signed factor repeated in source context, such as `10^-2`,
is unambiguous evidence that a conflicting refined scale is wrong. In rate or permeability
reporting, a displayed coefficient followed by `x 10^N` can mean the authors scaled the
reported coefficient by 10^N, so the physical value uses 10^-N. Keep the refined negative
exponent when the raw positive-exponent interpretation would be physically impossible.

Return one decision for every canonical_record_id in this exact JSON form:
{{
  "schema_version": "final_endpoint_pruning_review.v1",
  "rows": [
    {{
      "canonical_record_id": "...",
      "decision": "keep|drop",
      "reason_code": "fits_bucket|valid_extreme|insufficient_evidence|wrong_quantity_or_endpoint|wrong_unit_or_scale|corrupt_extraction",
      "reason": "concise semantic justification"
    }}
  ]
}}

Bucket and complete compact semantic rows:
{payload}
"""
_COMPACT_PROMPT = _PROMPT.replace("canonical_record_id", "row_id")


def supported_gap(
    values: Sequence[Any], *, unit_text: str = ""
) -> dict[str, Any] | None:
    """Return the largest supported 2-dex gap on the unit's native geometry."""
    geometry = _gap_geometry(values, unit_text=unit_text)
    if geometry is None:
        return None
    ordered, coordinates, geometry_id = geometry
    if len(ordered) < MINIMUM_RECORDS or len(set(ordered)) < MINIMUM_DISTINCT_VALUES:
        return None
    minimum_side = max(2, math.ceil(0.05 * len(coordinates)))
    indices = range(minimum_side - 1, len(coordinates) - minimum_side)
    gaps = [(coordinates[index + 1] - coordinates[index], index) for index in indices]
    if not gaps:
        return None
    gap_decades, index = max(gaps)
    if gap_decades + 1e-12 < GAP_DECADES:
        return None
    return {
        "finite_record_count": len(ordered),
        "gap_geometry": geometry_id,
        "minimum_side": minimum_side,
        "gap_decades": gap_decades,
        "gap_lower_value": ordered[index],
        "gap_upper_value": ordered[index + 1],
    }


def _gap_geometry(
    values: Sequence[Any], *, unit_text: str
) -> tuple[list[float], list[float], str] | None:
    unit = unit_text.strip().casefold()
    finite = sorted(float(value) for value in values if _finite(value))
    if unit.startswith(("log10", "-log10", "pec", "pic", "pki", "pka")) or unit == "ph":
        return finite, finite, "canonical_log10"
    if unit.startswith(("ln(", "-ln(")):
        return finite, [value / math.log(10) for value in finite], "canonical_ln"
    if (
        unit.startswith(("log(", "-log(", "log ", "-log ", "log2", "logit"))
        or unit in {"logbb", "logps", "logs"}
        or "log scale" in unit
        or "/log(" in unit
    ):
        return None
    positive = [value for value in finite if value > 0]
    return positive, [math.log10(value) for value in positive], "linear_to_log10"


def semantic_review_columns(schema_names: Sequence[str]) -> tuple[str, ...]:
    """Select scientific row semantics while excluding identity and policy metadata."""
    selected = []
    for field in schema_names:
        lowered = field.casefold()
        if field in _ALWAYS_VISIBLE:
            selected.append(field)
            continue
        if (
            field in _PRIVATE_FIELDS
            or "label" in lowered
            or "vote" in lowered
            or "eligible" in lowered
            or lowered.startswith(_PRIVATE_PREFIXES)
        ):
            continue
        if any(term in lowered for term in _SEMANTIC_TERMS):
            selected.append(field)
    if "canonical_record_id" not in selected:
        raise ValueError("semantic review projection requires canonical_record_id")
    return tuple(selected)


def render_review_prompt(candidate: Mapping[str, Any]) -> str:
    rows = [
        {
            "row_id": f"r{index:04d}",
            **{
                key: value
                for key, value in row.items()
                if key != "canonical_record_id"
            },
        }
        for index, row in enumerate(candidate["rows"], start=1)
    ]
    payload = {
        "task_id": candidate["task_id"],
        "canonical_bucket": candidate["canonical_bucket"],
        "distribution": candidate["distribution"],
        "rows": rows,
    }
    return _COMPACT_PROMPT.format(
        payload=json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def validate_review_response(
    response: Any,
    *,
    record_ids: set[str],
    id_field: str = "canonical_record_id",
) -> dict[str, Any]:
    if not isinstance(response, Mapping) or set(response) != {"schema_version", "rows"}:
        raise ValueError("review response must contain exactly schema_version and rows")
    if response["schema_version"] != SCHEMA_VERSION:
        raise ValueError("review schema version mismatch")
    rows = response["rows"]
    if not isinstance(rows, list):
        raise ValueError("review rows must be a list")
    validated: dict[str, dict[str, str]] = {}
    for row in rows:
        required = {id_field, "decision", "reason_code", "reason"}
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError("review row fields differ from the contract")
        record_id = str(row[id_field])
        if record_id in validated:
            raise ValueError(f"duplicate review record ID: {record_id}")
        decision = row["decision"]
        reason_code = row["reason_code"]
        reason = row["reason"]
        if decision not in {"keep", "drop"}:
            raise ValueError(f"invalid review decision for {record_id}")
        allowed = KEEP_REASONS if decision == "keep" else DROP_REASONS
        if reason_code not in allowed:
            raise ValueError(f"invalid review reason for {record_id}: {reason_code}")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 240:
            raise ValueError(f"review reason must be 1-240 characters for {record_id}")
        validated[record_id] = {
            id_field: record_id,
            "decision": decision,
            "reason_code": reason_code,
            "reason": reason.strip(),
        }
    if set(validated) != record_ids:
        missing = sorted(record_ids - set(validated))
        extra = sorted(set(validated) - record_ids)
        raise ValueError(
            "review IDs differ from input; "
            f"missing_count={len(missing)}, missing_examples={missing[:5]}, "
            f"extra_count={len(extra)}, extra_examples={extra[:5]}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "rows": [validated[record_id] for record_id in sorted(validated)],
    }


def consensus_decisions(
    candidate: Mapping[str, Any],
    review_a: Mapping[str, Any],
    review_b: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_reviewer = []
    for review in (review_a, review_b):
        by_reviewer.append(
            {row["canonical_record_id"]: row for row in review["response"]["rows"]}
        )
    values = {
        row["canonical_record_id"]: row.get("finite_scalar_value")
        for row in candidate["rows"]
    }
    output = []
    for record_id in sorted(values):
        first, second = (review[record_id] for review in by_reviewer)
        agreed_drop = (
            first["decision"] == second["decision"] == "drop"
            and first["reason_code"] == second["reason_code"]
        )
        same_keep = (
            first["decision"] == second["decision"] == "keep"
            and first["reason_code"] == second["reason_code"]
        )
        output.append(
            {
                "task_id": candidate["task_id"],
                "bucket_id": candidate["bucket_id"],
                "pair_bucket_key": candidate["pair_bucket_key"],
                "canonical_record_id": record_id,
                "finite_scalar_value": values[record_id],
                "review_a_decision": first["decision"],
                "review_a_reason_code": first["reason_code"],
                "review_a_reason": first["reason"],
                "review_b_decision": second["decision"],
                "review_b_reason_code": second["reason_code"],
                "review_b_reason": second["reason"],
                "consensus_decision": "drop" if agreed_drop else "keep",
                "consensus_reason_code": (
                    first["reason_code"]
                    if agreed_drop or same_keep
                    else "review_disagreement_keep"
                ),
            }
        )
    return output


def generate_final_endpoint_pruning(
    *,
    task_id: str,
    normalized_root: str | Path,
    output_dir: str | Path | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    workers: int = 16,
    timeout_s: int = 900,
    max_tokens: int = 65_536,
    max_attempts: int = 3,
) -> dict[str, Any]:
    root = Path(normalized_root)
    dedup = root / "05_deduplicated_records"
    records_path = dedup / "records.parquet"
    buckets_path = dedup / "pair_bucket_records.parquet"
    _validate_deduplicated_inputs(root, records_path, buckets_path)
    target = Path(output_dir) if output_dir else root / ARTIFACT_DIR
    target.mkdir(parents=True, exist_ok=True)

    candidates = _candidate_buckets(task_id, records_path, buckets_path)
    _attach_semantic_rows(candidates, records_path)
    prompts = {
        candidate["bucket_id"]: render_review_prompt(candidate)
        for candidate in candidates
    }
    cache_path = target / CACHE_FILENAME
    cached = _load_review_cache(cache_path)
    reviews: dict[tuple[str, str], dict[str, Any]] = {}
    missing: list[tuple[dict[str, Any], str, str]] = []
    for candidate in candidates:
        bucket_id = candidate["bucket_id"]
        prompt = prompts[bucket_id]
        prompt_sha256 = _sha256(prompt)
        record_ids = {row["canonical_record_id"] for row in candidate["rows"]}
        for reviewer in REVIEWERS:
            prior = cached.get((bucket_id, reviewer))
            try:
                if (
                    not prior
                    or prior.get("prompt_sha256") != prompt_sha256
                    or prior.get("requested_model") != model
                ):
                    raise ValueError("cache miss")
                prior = {
                    **prior,
                    "response": validate_review_response(
                        prior.get("response"), record_ids=record_ids
                    ),
                }
                reviews[(bucket_id, reviewer)] = prior
            except ValueError:
                missing.append((candidate, reviewer, prompt))

    calls_this_run = 0
    if missing:
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"{len(missing)} GPT review(s) are missing and {api_key_env} is unset"
            )
        client = OpenAICompatibleClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            temperature=None,
            tool_service_url="http://127.0.0.1:8766",
            enable_group_tools=False,
            max_tool_rounds=0,
            reasoning_effort="medium",
            enable_thinking=False,
        )
        errors = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(max(1, workers), len(missing))
        ) as executor:
            futures = {
                executor.submit(
                    _run_review,
                    client,
                    candidate,
                    reviewer,
                    prompt,
                    model,
                    base_url,
                    max_attempts,
                ): (candidate["bucket_id"], reviewer)
                for candidate, reviewer, prompt in missing
            }
            for future in concurrent.futures.as_completed(futures):
                key = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # cache successful siblings before failing
                    errors.append(f"{key[0]} reviewer {key[1]}: {exc}")
                    continue
                reviews[key] = row
                with cache_path.open("a", encoding="utf-8") as handle:
                    handle.write(_json_dump(row) + "\n")
                calls_this_run += 1
        if errors:
            raise RuntimeError("final endpoint pruning reviews failed: " + "; ".join(errors))

    decisions = []
    for candidate in candidates:
        bucket_id = candidate["bucket_id"]
        decisions.extend(
            consensus_decisions(
                candidate,
                reviews[(bucket_id, "a")],
                reviews[(bucket_id, "b")],
            )
        )
    decisions.sort(key=lambda row: (row["pair_bucket_key"], row["canonical_record_id"]))
    selected_reviews = [
        reviews[(candidate["bucket_id"], reviewer)]
        for candidate in candidates
        for reviewer in REVIEWERS
    ]
    reviews_path = target / REVIEWS_FILENAME
    decisions_path = target / DECISIONS_FILENAME
    _write_jsonl_atomic(reviews_path, selected_reviews)
    temporary_decisions = decisions_path.with_suffix(".parquet.tmp")
    write_parquet(temporary_decisions, decisions)
    os.replace(temporary_decisions, decisions_path)

    dropped = {row["canonical_record_id"] for row in decisions if row["consensus_decision"] == "drop"}
    bucket_summaries = []
    for candidate in candidates:
        bucket_ids = {row["canonical_record_id"] for row in candidate["rows"]}
        bucket_summaries.append(
            {
                "bucket_id": candidate["bucket_id"],
                "pair_bucket_key": candidate["pair_bucket_key"],
                **candidate["canonical_bucket"],
                **candidate["distribution"],
                "dropped_records": len(bucket_ids & dropped),
            }
        )
    manifest = {
        "version": VERSION,
        "task_id": task_id,
        "detector": {
            "unit_aware_geometry": True,
            "linear_values_require_positive_domain": True,
            "ambiguous_log_units_skipped": True,
            "minimum_records": MINIMUM_RECORDS,
            "minimum_distinct_values": MINIMUM_DISTINCT_VALUES,
            "minimum_side": "max(2, ceil(0.05 * finite_record_count))",
            "gap_decades": GAP_DECADES,
            "raw_ratio": 10**GAP_DECADES,
        },
        "review": {
            "requested_model": model,
            "served_models": sorted({str(row.get("served_model") or "") for row in selected_reviews}),
            "endpoint": base_url,
            "reasoning_effort": "medium",
            "max_tokens": max_tokens,
            "max_attempts": max_attempts,
            "reviewers_per_bucket": len(REVIEWERS),
            "consensus": "drop_only_when_both_drop_with_same_reason_code",
            "prompt_sha256": _sha256(_PROMPT),
            "compact_prompt_sha256": _sha256(_COMPACT_PROMPT),
            "schema_version": SCHEMA_VERSION,
        },
        "inputs": {
            "records": {"path": str(records_path), "sha256": file_sha256(records_path)},
            "pair_bucket_records": {"path": str(buckets_path), "sha256": file_sha256(buckets_path)},
        },
        "summary": {
            "candidate_buckets": len(candidates),
            "candidate_rows": len(decisions),
            "trigger_rows": sum(
                candidate["distribution"]["finite_record_count"] for candidate in candidates
            ),
            "review_results": len(selected_reviews),
            "api_calls_this_run": calls_this_run,
            "kept_rows": len(decisions) - len(dropped),
            "dropped_rows": len(dropped),
        },
        "bucket_summaries": bucket_summaries,
        "files": {
            REVIEWS_FILENAME: file_sha256(reviews_path),
            DECISIONS_FILENAME: file_sha256(decisions_path),
        },
        "validations": {
            "complete_bucket_payloads": True,
            "semantic_metadata_only": True,
            "two_independent_reviews_per_bucket": True,
            "exact_row_id_coverage": True,
            "numerical_extremeness_alone_never_drops": True,
        },
    }
    _write_json_atomic(target / MANIFEST_FILENAME, manifest)
    return manifest


def load_final_endpoint_pruning(
    manifest_path: str | Path,
    *,
    task_id: str,
    records_path: str | Path,
    pair_bucket_records_path: str | Path,
) -> tuple[set[str], dict[str, Any]]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION or manifest.get("task_id") != task_id:
        raise ValueError("final endpoint pruning manifest version/task mismatch")
    expected_inputs = {
        "records": Path(records_path),
        "pair_bucket_records": Path(pair_bucket_records_path),
    }
    for key, input_path in expected_inputs.items():
        if manifest.get("inputs", {}).get(key, {}).get("sha256") != file_sha256(input_path):
            raise ValueError(f"final endpoint pruning {key} hash is stale")
    for filename in (REVIEWS_FILENAME, DECISIONS_FILENAME):
        artifact = path.parent / filename
        if manifest.get("files", {}).get(filename) != file_sha256(artifact):
            raise ValueError(f"final endpoint pruning {filename} hash mismatch")
    decisions_path = path.parent / DECISIONS_FILENAME
    rows = pq.read_table(decisions_path).to_pylist()
    required = {"canonical_record_id", "consensus_decision", "consensus_reason_code"}
    if any(not required <= set(row) for row in rows):
        raise ValueError("final endpoint pruning decisions schema is incomplete")
    record_ids = [str(row["canonical_record_id"]) for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("final endpoint pruning decisions contain duplicate record IDs")
    if any(row["consensus_decision"] not in {"keep", "drop"} for row in rows):
        raise ValueError("final endpoint pruning decisions contain an invalid decision")
    summary = manifest.get("summary", {})
    dropped = {
        str(row["canonical_record_id"])
        for row in rows
        if row["consensus_decision"] == "drop"
    }
    if summary.get("candidate_rows") != len(rows) or summary.get("dropped_rows") != len(dropped):
        raise ValueError("final endpoint pruning decision counts differ from manifest")
    return dropped, {
        "version": VERSION,
        "manifest": {"path": str(path), "sha256": file_sha256(path)},
        "decisions": {"path": str(decisions_path), "sha256": file_sha256(decisions_path)},
        "reviewed_rows": len(rows),
        "dropped_rows": len(dropped),
    }


def _candidate_buckets(task_id: str, records_path: Path, buckets_path: Path) -> list[dict[str, Any]]:
    record_columns = [
        "canonical_record_id",
        "retrieval_eligible",
        "canonical_smiles",
        "measurement_kind",
        "finite_scalar_value",
    ]
    bucket_columns = [
        "canonical_record_id",
        "pair_bucket_key",
        "canonical_pair_fields_json",
        "canonical_endpoint_name",
        "canonical_unit_text",
    ]
    groups: dict[str, dict[str, Any]] = {}
    record_batches = pq.ParquetFile(records_path).iter_batches(
        batch_size=5_000, columns=record_columns
    )
    bucket_batches = pq.ParquetFile(buckets_path).iter_batches(
        batch_size=5_000, columns=bucket_columns
    )
    sentinel = object()
    for record_batch, bucket_batch in itertools.zip_longest(
        record_batches, bucket_batches, fillvalue=sentinel
    ):
        if record_batch is sentinel or bucket_batch is sentinel:
            raise ValueError("deduplicated record/pair-bucket coverage differs")
        records = record_batch.to_pylist()
        buckets = bucket_batch.to_pylist()
        if len(records) != len(buckets):
            raise ValueError("deduplicated record/pair-bucket batch lengths differ")
        for record, bucket in zip(records, buckets, strict=True):
            record_id = str(record.get("canonical_record_id") or "")
            if record_id != str(bucket.get("canonical_record_id") or ""):
                raise ValueError("deduplicated record/pair-bucket row order differs")
            key = str(bucket.get("pair_bucket_key") or "")
            if not (
                bool(record.get("retrieval_eligible"))
                and _has_text(record.get("canonical_smiles"))
                and record.get("measurement_kind") == "continuous"
                and key
            ):
                continue
            group = groups.setdefault(
                key,
                {
                    "record_ids": [],
                    "values": [],
                    "canonical_endpoint_name": bucket.get("canonical_endpoint_name"),
                    "canonical_unit_text": bucket.get("canonical_unit_text"),
                    "canonical_pair_fields_json": bucket.get("canonical_pair_fields_json"),
                },
            )
            group["record_ids"].append(record_id)
            group["values"].append(record.get("finite_scalar_value"))
    candidates = []
    for key, group in sorted(groups.items()):
        gap = supported_gap(
            group["values"], unit_text=str(group["canonical_unit_text"] or "")
        )
        if gap is None:
            continue
        try:
            pair_fields = json.loads(str(group["canonical_pair_fields_json"] or "{}"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid canonical pair fields for bucket {key}") from exc
        candidates.append(
            {
                "task_id": task_id,
                "bucket_id": _sha256(task_id + "\0" + key)[:24],
                "pair_bucket_key": key,
                "record_ids": group["record_ids"],
                "canonical_bucket": {
                    "canonical_endpoint_name": group["canonical_endpoint_name"],
                    "canonical_unit_text": group["canonical_unit_text"],
                    "canonical_pair_fields": pair_fields,
                },
                "distribution": {"record_count": len(group["record_ids"]), **gap},
            }
        )
    return candidates


def _attach_semantic_rows(candidates: list[dict[str, Any]], records_path: Path) -> None:
    wanted = {
        record_id
        for candidate in candidates
        for record_id in candidate["record_ids"]
    }
    columns = semantic_review_columns(pq.read_schema(records_path).names)
    rows: dict[str, dict[str, Any]] = {}
    for batch in pq.ParquetFile(records_path).iter_batches(batch_size=5_000, columns=list(columns)):
        for record in batch.to_pylist():
            record_id = str(record.get("canonical_record_id") or "")
            if record_id in wanted:
                rows[record_id] = _compact_semantic_row(record)
    if set(rows) != wanted:
        raise ValueError("semantic review projection did not cover every candidate row")
    for candidate in candidates:
        candidate["rows"] = sorted(
            (rows[record_id] for record_id in candidate.pop("record_ids")),
            key=lambda row: (_numeric_sort(row.get("finite_scalar_value")), row["canonical_record_id"]),
        )


def _run_review(
    client: OpenAICompatibleClient,
    candidate: Mapping[str, Any],
    reviewer: str,
    prompt: str,
    model: str,
    base_url: str,
    max_attempts: int,
) -> dict[str, Any]:
    errors = []
    trace = None
    response = None
    messages = [{"role": "user", "content": prompt}]
    for _ in range(max_attempts):
        try:
            trace = client.chat_json(messages)
            response = _validate_candidate_review(trace["content"], candidate)
            break
        except ValueError as exc:
            errors.append(str(exc))
            messages = [
                {"role": "user", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Your JSON failed validation: {exc}. Return one complete corrected "
                        "JSON object covering every row from the original prompt, not a patch "
                        "or only the corrected rows."
                    ),
                },
            ]
    if response is None or trace is None:
        raise RuntimeError(f"failed validation after {max_attempts} attempts: {errors}")
    return {
        "bucket_id": candidate["bucket_id"],
        "reviewer": reviewer,
        "prompt_sha256": _sha256(prompt),
        "requested_model": model,
        "served_model": str(trace.get("model") or ""),
        "endpoint": base_url,
        "attempts": len(errors) + 1,
        "validation_errors": errors,
        "usage": trace.get("usage") or {},
        "response_id": str(trace.get("id") or ""),
        "raw_content": str(trace.get("raw_content") or ""),
        "reasoning_content": str(trace.get("reasoning_content") or ""),
        "response": response,
    }


def _validate_candidate_review(
    response: Any, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    rows = candidate["rows"]
    alias_to_record_id = {
        f"r{index:04d}": row["canonical_record_id"]
        for index, row in enumerate(rows, start=1)
    }
    validated = validate_review_response(
        response,
        record_ids=set(alias_to_record_id),
        id_field="row_id",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "rows": [
            {
                "canonical_record_id": alias_to_record_id[row["row_id"]],
                **{key: value for key, value in row.items() if key != "row_id"},
            }
            for row in validated["rows"]
        ],
    }


def _validate_deduplicated_inputs(root: Path, records_path: Path, buckets_path: Path) -> None:
    manifest_path = root / "05_deduplicated_records/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_stage03 = root / "03_records/records.parquet"
    checks = (
        (manifest.get("inputs", {}).get("records", {}).get("sha256"), current_stage03),
        (manifest.get("outputs", {}).get("records.parquet"), records_path),
        (manifest.get("outputs", {}).get("pair_bucket_records.parquet"), buckets_path),
    )
    for expected, path in checks:
        if not path.is_file() or expected != file_sha256(path):
            raise ValueError(
                "Stage 05 is missing or stale; rebuild the deterministic prefix before review"
            )


def _load_review_cache(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    output = {}
    if not path.is_file():
        return output
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict) and row.get("bucket_id") and row.get("reviewer"):
                output[(str(row["bucket_id"]), str(row["reviewer"]))] = row
    return output


def _compact_semantic_row(record: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in record.items():
        if key.startswith("measurement_resolution_"):
            continue
        value = _json_value(value)
        if value is not None and value != "":
            output[str(key)] = value
    if record.get("measurement_resolution_origin") == "llm":
        output["llm_refined_extraction"] = {
            "measurement": _json_value(
                record.get("measurement_resolution_input_measurement")
            ),
            "unit": _json_value(record.get("measurement_resolution_input_unit")),
        }
    return output


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json_dump(row) + "\n")
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _has_text(value: Any) -> bool:
    return value is not None and str(value).strip().casefold() not in {"", "nan", "none", "null"}


def _numeric_sort(value: Any) -> tuple[int, float]:
    return (0, float(value)) if _finite(value) else (1, 0.0)


def _rebuild_stage05(task_id: str, normalized_root: Path) -> None:
    modules = {
        "bbb_martins": "tools.chembl_tool.tasks.bbb_martins.build_starling_downstream_artifacts",
        "bioavailability_ma": "tools.chembl_tool.tasks.bioavailability_ma.build_starling_downstream_artifacts",
        "skin_reaction": "tools.chembl_tool.tasks.skin_reaction.build_starling_downstream_artifacts",
    }
    module = importlib.import_module(modules[task_id])
    from tools.chembl_tool.common.starling.split_downstream import (
        build_canonical_deduplicated_records,
    )

    build_canonical_deduplicated_records(module.get_spec(), normalized_root=normalized_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True, choices=("bbb_martins", "bioavailability_ma", "skin_reaction"))
    parser.add_argument("--normalized-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=65_536)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--rebuild-stage05", action="store_true")
    args = parser.parse_args(argv)
    if args.rebuild_stage05:
        _rebuild_stage05(args.task_id, args.normalized_root)
    manifest = generate_final_endpoint_pruning(
        task_id=args.task_id,
        normalized_root=args.normalized_root,
        output_dir=args.output_dir,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        model=args.model,
        workers=args.workers,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
        max_attempts=args.max_attempts,
    )
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_DIR",
    "DECISIONS_FILENAME",
    "MANIFEST_FILENAME",
    "REVIEWS_FILENAME",
    "VERSION",
    "consensus_decisions",
    "generate_final_endpoint_pruning",
    "load_final_endpoint_pruning",
    "render_review_prompt",
    "semantic_review_columns",
    "supported_gap",
    "validate_review_response",
]
