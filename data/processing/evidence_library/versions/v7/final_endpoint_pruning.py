"""Review scientifically suspicious Stage-3 assay-transfer tail records."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import math
import os
import statistics
import warnings
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from scipy.stats import normaltest

from tools.chembl_tool.common.openai_reasoning_client import OpenAICompatibleClient
from data.processing.evidence_library.shared.v1.normalization.audit import write_parquet
from data.processing.evidence_library.shared.v1.normalization.cleaning import file_sha256
from data.processing.evidence_library.versions.v7.build_reference_semantics_mapping import (
    BUDGET_EXHAUSTED_EXIT_CODE,
    TokenLedger,
)
from data.processing.evidence_library.shared.v1.build_runtime import starling_build_session
from data.processing.evidence_library.versions.v7.prompts import PROMPT_ROOT
from data.processing.evidence_library.versions.v7.task_registry import import_task_module


VERSION = "assay_transfer_outlier_review.v1"
SCHEMA_VERSION = "final_endpoint_pruning_review.v1"
ARTIFACT_DIR = "reviews/assay_transfer_outlier_v1"
MANIFEST_FILENAME = "manifest.json"
DECISIONS_FILENAME = "decisions.parquet"
REVIEWS_FILENAME = "reviews.jsonl"
CACHE_FILENAME = ".review_cache.jsonl"
TEMPLATE_PATH = PROMPT_ROOT / "endpoint_pruning/v6.jinja"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
GAP_DECADES = 2.0
GAP_MINIMUM_RECORDS = 6
MINIMUM_RECORDS = 20
MINIMUM_DISTINCT_PARENTS = 16
MINIMUM_DISTINCT_VALUES = 3
TAIL_MINIMUM_RECORDS = 3
TAIL_SIGMA = 5.0
TAIL_DECADES = 2.0
LOG10_GATE_MINIMUM_RECORDS = 20
LOG10_GATE_ALPHA = 0.05
LOG10_GATE_TRANSFORM_ABS_TOLERANCE = 1e-9
MAX_REVIEW_PROMPT_BYTES = 180_000
MAX_TARGET_ROWS = 40
CONTEXT_ROWS = 5
REVIEWER = "single"
KEEP_REASONS = {"fits_bucket", "valid_extreme", "insufficient_evidence"}
DROP_REASONS = {
    "wrong_quantity_or_endpoint",
    "wrong_unit_or_scale",
    "corrupt_extraction",
}
REVIEW_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "final_endpoint_pruning_review",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "schema_version": {"type": "string", "enum": [SCHEMA_VERSION]},
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "row_id": {"type": "string", "pattern": "^r[0-9]{4}$"},
                            "decision": {
                                "type": "string",
                                "enum": ["keep", "drop"],
                            },
                            "reason_code": {
                                "type": "string",
                                "enum": sorted(KEEP_REASONS | DROP_REASONS),
                            },
                            "reason": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 240,
                            },
                        },
                        "required": [
                            "row_id",
                            "decision",
                            "reason_code",
                            "reason",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["schema_version", "rows"],
            "additionalProperties": False,
        },
    },
}


class PruningBudgetExhausted(RuntimeError):
    pass


_REVIEW_COLUMNS = {
    "canonical_record_id",
    "measurement_text",
    "unit_text",
    "measurement_resolution_origin",
    "measurement_resolution_input_measurement",
    "measurement_resolution_input_unit",
    "finite_scalar_value",
    "canonical_measurement_text",
    "canonical_unit_text",
    "support_text",
    "assay_or_test",
    "assay_model",
    "assay_system",
    "assay_type",
    "species",
    "species_or_population",
    "biological_context",
    "experimental_conditions",
    "qualifying_conditions",
    "condition_medium",
    "dose",
    "exposure",
    "exposure_time",
    "formulation_or_solid_form",
    "formulation_vehicle",
    "matrix",
    "comparator",
    "study_design",
    "time",
    "light",
    "light_condition",
    "target",
    "aop_event",
    "extra_details",
}

_PROMPT_SEMANTIC_FIELDS = (
    ("Assay or model", ("assay_or_test", "assay_model", "assay_system", "assay_type")),
    ("Species or population", ("species", "species_or_population")),
    ("Biological context", ("biological_context",)),
    (
        "Experimental conditions",
        ("experimental_conditions", "qualifying_conditions", "condition_medium"),
    ),
    ("Dose", ("dose",)),
    ("Exposure", ("exposure", "exposure_time")),
    (
        "Formulation or vehicle",
        ("formulation_or_solid_form", "formulation_vehicle"),
    ),
    ("Matrix", ("matrix",)),
    ("Comparator", ("comparator",)),
    ("Study design", ("study_design",)),
    ("Time", ("time",)),
    ("Light condition", ("light", "light_condition")),
    ("Target", ("target",)),
    ("AOP event", ("aop_event",)),
    ("Extra details", ("extra_details",)),
)

def supported_gap(
    values: Sequence[Any], *, unit_text: str = ""
) -> dict[str, Any] | None:
    """Return the largest supported 2-dex gap on the unit's native geometry."""
    geometry = _gap_geometry(values, unit_text=unit_text)
    if geometry is None:
        return None
    ordered, coordinates, geometry_id = geometry
    if (
        len(ordered) < GAP_MINIMUM_RECORDS
        or len(set(ordered)) < MINIMUM_DISTINCT_VALUES
    ):
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


def tail_outliers(
    values: Sequence[Any], *, unit_text: str = ""
) -> list[dict[str, Any]]:
    """Return leave-one-out tails on the unit-aware comparison geometry."""
    points, geometry_id, supports_decades = _tail_geometry(
        values, unit_text=unit_text
    )
    if len(points) < TAIL_MINIMUM_RECORDS:
        return []
    coordinates = [point[2] for point in points]
    count = len(coordinates)
    overall_mean = statistics.fmean(coordinates)
    overall_m2 = sum((value - overall_mean) ** 2 for value in coordinates)
    output = []
    for index, value, coordinate in points:
        reference_count = count - 1
        mean = (count * overall_mean - coordinate) / reference_count
        reference_m2 = overall_m2 - (coordinate - overall_mean) * (coordinate - mean)
        sd = math.sqrt(max(0.0, reference_m2) / (reference_count - 1))
        distance = abs(coordinate - mean)
        sigma = distance / sd if sd > 0 else None
        reasons = []
        if (sigma is not None and sigma >= TAIL_SIGMA) or (sd == 0 and distance > 0):
            reasons.append("leave_one_out_5sd")
        if supports_decades and distance >= TAIL_DECADES:
            reasons.append("leave_one_out_2decades")
        if reasons:
            output.append(
                {
                    "record_index": index,
                    "finite_scalar_value": value,
                    "tail_geometry": geometry_id,
                    "tail_reference_count": reference_count,
                    "tail_reference_mean": mean,
                    "tail_reference_sd": sd,
                    "tail_distance_geometry": distance,
                    "tail_distance_decades": distance if supports_decades else None,
                    "tail_sigma_distance": sigma,
                    "tail_zero_reference_sd": sd == 0,
                    "tail_trigger_reasons": reasons,
                }
            )
    return output


def _tail_geometry(
    values: Sequence[Any], *, unit_text: str
) -> tuple[list[tuple[int, float, float]], str, bool]:
    """Compatibility helper describing whether a decade geometry is available."""
    finite = [
        (index, float(value), float(value))
        for index, value in enumerate(values)
        if _finite(value)
    ]
    geometry = _indexed_geometry(values, unit_text=unit_text)
    if geometry is None or len(geometry[0]) != len(finite):
        return finite, "native_scale_sd_only", False
    points, geometry_id = geometry
    return points, geometry_id, True


def log10_approval_gate(
    pretransform_values: Sequence[Any],
    transformed_values: Sequence[Any],
    transform_ids: Sequence[Any],
    *,
    unit_text: str,
) -> dict[str, Any] | None:
    """Approve a verified, justified log10 axis only when its raw values are non-normal."""
    if (
        len(pretransform_values) != len(transformed_values)
        or len(transform_ids) != len(transformed_values)
        or len(transformed_values) < LOG10_GATE_MINIMUM_RECORDS
        or {str(value or "") for value in transform_ids} != {"log10.v1"}
        or not unit_text.strip().casefold().startswith("log10(")
    ):
        return None
    raw: list[float] = []
    logged: list[float] = []
    for raw_value, logged_value in zip(
        pretransform_values, transformed_values, strict=True
    ):
        if not (_finite(raw_value) and _finite(logged_value)):
            return None
        raw_number = float(raw_value)
        logged_number = float(logged_value)
        if raw_number <= 0 or not math.isclose(
            math.log10(raw_number),
            logged_number,
            rel_tol=0.0,
            abs_tol=LOG10_GATE_TRANSFORM_ABS_TOLERANCE,
        ):
            return None
        raw.append(raw_number)
        logged.append(logged_number)
    if (
        len(set(raw)) < MINIMUM_DISTINCT_VALUES
        or len(set(logged)) < MINIMUM_DISTINCT_VALUES
    ):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw_result = normaltest(raw)
    raw_statistic = float(raw_result.statistic)
    raw_p_value = float(raw_result.pvalue)
    if not all(math.isfinite(value) for value in (raw_statistic, raw_p_value)):
        return None
    return {
        "test": "dagostino_pearson_k2",
        "alpha": LOG10_GATE_ALPHA,
        "record_count": len(raw),
        "raw_statistic": raw_statistic,
        "raw_p_value": raw_p_value,
        "raw_normality_rejected": raw_p_value < LOG10_GATE_ALPHA,
        "approved": raw_p_value < LOG10_GATE_ALPHA,
    }


def _gap_geometry(
    values: Sequence[Any], *, unit_text: str
) -> tuple[list[float], list[float], str] | None:
    geometry = _indexed_geometry(values, unit_text=unit_text)
    if geometry is None:
        return None
    points, geometry_id = geometry
    ordered_points = sorted(points, key=lambda point: point[1])
    return (
        [point[1] for point in ordered_points],
        [point[2] for point in ordered_points],
        geometry_id,
    )


def _indexed_geometry(
    values: Sequence[Any], *, unit_text: str
) -> tuple[list[tuple[int, float, float]], str] | None:
    unit = unit_text.strip().casefold()
    finite = [(index, float(value)) for index, value in enumerate(values) if _finite(value)]
    if unit.startswith(("log10", "-log10", "pec", "pic", "pki", "pka")) or unit == "ph":
        return [(index, value, value) for index, value in finite], "canonical_log10"
    if unit.startswith(("ln(", "-ln(")):
        return [
            (index, value, value / math.log(10)) for index, value in finite
        ], "canonical_ln"
    if (
        unit.startswith(("log(", "-log(", "log ", "-log ", "log2", "logit"))
        or unit in {"logbb", "logps", "logs"}
        or "log scale" in unit
        or "/log(" in unit
    ):
        return None
    return [
        (index, value, math.log10(value))
        for index, value in finite
        if value > 0
    ], "linear_to_log10"


def semantic_review_columns(schema_names: Sequence[str]) -> tuple[str, ...]:
    """Load only fields that can appear in the scientific review prompt."""
    selected = [field for field in schema_names if field in _REVIEW_COLUMNS]
    if "canonical_record_id" not in selected:
        raise ValueError("semantic review projection requires canonical_record_id")
    return tuple(selected)


def render_review_prompt(candidate: Mapping[str, Any]) -> str:
    bucket = candidate["canonical_bucket"]
    distribution = candidate["distribution"]
    return _prompt_template().render(
        schema_version=SCHEMA_VERSION,
        task_id=candidate["task_id"],
        bucket_endpoint=_prompt_value(bucket.get("canonical_endpoint_name")),
        bucket_unit=_prompt_value(bucket.get("canonical_unit_text")),
        target_rows=[
            _prompt_row(row, row_id=f"r{index:04d}")
            for index, row in enumerate(candidate["target_rows"], start=1)
        ],
        context_rows=[_prompt_row(row) for row in candidate.get("context_rows", [])],
        complete_bucket_context=(
            not candidate.get("context_rows")
            and distribution["review_scope"] == "complete_supported_gap_bucket"
        ),
    )


@lru_cache(maxsize=1)
def _prompt_template() -> Any:
    environment = Environment(
        loader=FileSystemLoader(str(TEMPLATE_PATH.parent)),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return environment.get_template(TEMPLATE_PATH.name)


def _prompt_row(row: Mapping[str, Any], *, row_id: str = "") -> dict[str, Any]:
    refined = row.get("llm_refined_extraction")
    refined_text = ""
    if isinstance(refined, Mapping):
        measurement = _prompt_value(refined.get("measurement"))
        unit = _prompt_value(refined.get("unit"))
        refined_text = " ".join(part for part in (measurement, unit) if part)
    trigger = []
    sigma = row.get("tail_sigma_distance")
    if _finite(sigma):
        trigger.append(f"{float(sigma):.3g} leave-one-out SD")
    decades = row.get("tail_distance_decades")
    if _finite(decades):
        trigger.append(f"{float(decades):.3g} decades from the reference mean")
    return {
        "row_id": row_id,
        "source_measurement": _prompt_value(row.get("measurement_text")),
        "source_unit": _prompt_value(row.get("unit_text")),
        "canonical_measurement": _prompt_value(
            row.get("canonical_measurement_text", row.get("finite_scalar_value"))
        ),
        "canonical_unit": _prompt_value(row.get("canonical_unit_text")),
        "llm_refined_extraction": refined_text,
        "support_text": _prompt_value(row.get("support_text")),
        "semantic_fields": _prompt_fields(row),
        "trigger": "; ".join(trigger),
    }


def _prompt_fields(values: Mapping[str, Any]) -> list[dict[str, str]]:
    fields = []
    for label, keys in _PROMPT_SEMANTIC_FIELDS:
        texts = []
        for key in keys:
            text = _prompt_value(values.get(key))
            if text and text not in texts:
                texts.append(text)
        if texts:
            fields.append({"label": label, "value": "; ".join(texts)})
    return fields


def _prompt_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    if isinstance(value, Mapping):
        parts = []
        for key, item in value.items():
            text = _prompt_value(item)
            if text:
                parts.append(f"{str(key).replace('_', ' ')}: {text}")
        return "; ".join(parts)
    if isinstance(value, (list, tuple)):
        return "; ".join(text for item in value if (text := _prompt_value(item)))
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "__unknown__"} else text


def _value_spread_rows(
    rows: Sequence[Mapping[str, Any]], *, limit: int = CONTEXT_ROWS
) -> list[Mapping[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            _numeric_sort(row.get("finite_scalar_value")),
            str(row.get("canonical_record_id") or ""),
        ),
    )
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[len(ordered) // 2]]
    return [
        ordered[round(index * (len(ordered) - 1) / (limit - 1))]
        for index in range(limit)
    ]


def split_review_candidate(
    candidate: Mapping[str, Any],
    *,
    max_prompt_bytes: int = MAX_REVIEW_PROMPT_BYTES,
) -> list[dict[str, Any]]:
    """Split only oversized review payloads while preserving bucket context."""
    if max_prompt_bytes < 1:
        raise ValueError("max_prompt_bytes must be positive")
    original = {
        **candidate,
        "target_rows": list(candidate["target_rows"]),
        "context_rows": list(candidate.get("context_rows", [])),
    }
    if (
        len(original["target_rows"]) <= MAX_TARGET_ROWS
        and len(render_review_prompt(original).encode("utf-8")) <= max_prompt_bytes
    ):
        return [original]

    def with_context(target_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        context_rows = original["context_rows"]
        if not context_rows:
            target_ids = {str(row["canonical_record_id"]) for row in target_rows}
            context_rows = _value_spread_rows(
                [
                    row
                    for row in original["target_rows"]
                    if str(row["canonical_record_id"]) not in target_ids
                ]
            )
        return {
            **original,
            "target_rows": list(target_rows),
            "context_rows": list(context_rows),
        }

    row_chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in original["target_rows"]:
        probe = with_context([*current, row])
        if current and (
            len(current) >= MAX_TARGET_ROWS
            or len(render_review_prompt(probe).encode("utf-8")) > max_prompt_bytes
        ):
            row_chunks.append(current)
            current = [row]
        else:
            current.append(row)
        if len(render_review_prompt(with_context(current)).encode("utf-8")) > max_prompt_bytes:
            raise ValueError(
                f"single review row exceeds prompt limit in {candidate['bucket_id']}"
            )
    if current:
        row_chunks.append(current)

    chunk_count = len(row_chunks)
    output = []
    for index, rows in enumerate(row_chunks, start=1):
        output.append(
            {
                **with_context(rows),
                "bucket_id": f"{original['bucket_id']}.chunk{index:04d}",
                "parent_bucket_id": original["bucket_id"],
                "review_chunk_index": index,
                "review_chunk_count": chunk_count,
            }
        )
    return output


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


def review_decisions(
    candidate: Mapping[str, Any],
    review: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reviewed = {
        row["canonical_record_id"]: row for row in review["response"]["rows"]
    }
    values = {
        row["canonical_record_id"]: row.get("finite_scalar_value")
        for row in candidate["target_rows"]
    }
    trigger_fields = {
        row["canonical_record_id"]: {
            key: row.get(key)
            for key in (
                "tail_geometry",
                "tail_reference_count",
                "tail_reference_mean",
                "tail_reference_sd",
                "tail_distance_geometry",
                "tail_distance_decades",
                "tail_sigma_distance",
                "tail_zero_reference_sd",
                "tail_trigger_reasons",
            )
            if key in row
        }
        for row in candidate["target_rows"]
    }
    output = []
    for record_id in sorted(values):
        decision = reviewed[record_id]
        output.append(
            {
                "task_id": candidate["task_id"],
                "bucket_id": candidate["bucket_id"],
                "pair_bucket_key": candidate["pair_bucket_key"],
                "canonical_record_id": record_id,
                "finite_scalar_value": values[record_id],
                **trigger_fields[record_id],
                "review_decision": decision["decision"],
                "review_reason_code": decision["reason_code"],
                "review_reason": decision["reason"],
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
    max_tokens: int = 4_096,
    max_attempts: int = 1,
    token_ledger_path: str | Path | None = None,
    budget_epoch: str = "",
    start_new_budget_epoch: bool = False,
    budget_max_tokens: int = 10_000_000,
    reuse_valid_cache_across_models: bool = False,
) -> dict[str, Any]:
    root = Path(normalized_root)
    stage3 = root / "03_pair_buckets"
    records_path = stage3 / "records.parquet"
    buckets_path = stage3 / "pair_bucket_records.parquet"
    canonical_records_sha256 = _validate_stage3_inputs(
        root, records_path, buckets_path
    )
    target = Path(output_dir) if output_dir else root / ARTIFACT_DIR
    target.mkdir(parents=True, exist_ok=True)

    candidates, detector_summary = _candidate_buckets(
        task_id, records_path, buckets_path
    )
    _attach_semantic_rows(candidates, records_path)
    chunks_by_bucket = {
        candidate["bucket_id"]: split_review_candidate(candidate)
        for candidate in candidates
    }
    review_candidates = [
        chunk
        for candidate in candidates
        for chunk in chunks_by_bucket[candidate["bucket_id"]]
    ]
    prompts = {
        candidate["bucket_id"]: render_review_prompt(candidate)
        for candidate in review_candidates
    }
    cache_path = target / CACHE_FILENAME
    cached = _load_review_cache(cache_path)
    chunk_reviews: dict[str, dict[str, Any]] = {}
    missing: list[tuple[dict[str, Any], str]] = []
    for candidate in review_candidates:
        bucket_id = candidate["bucket_id"]
        prompt = prompts[bucket_id]
        prompt_sha256 = _sha256(prompt)
        record_ids = {
            row["canonical_record_id"] for row in candidate["target_rows"]
        }
        prior = cached.get(bucket_id)
        try:
            if (
                not prior
                or prior.get("prompt_sha256") != prompt_sha256
                or (
                    prior.get("requested_model") != model
                    and not reuse_valid_cache_across_models
                )
            ):
                raise ValueError("cache miss")
            prior = {
                **prior,
                "response": validate_review_response(
                    prior.get("response"), record_ids=record_ids
                ),
            }
            chunk_reviews[bucket_id] = prior
        except ValueError:
            missing.append((candidate, prompt))

    calls_this_run = 0
    ledger = None
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
            request_extra_body=(
                {"provider": {"require_parameters": True}}
                if "openrouter.ai" in base_url
                else None
            ),
            response_format=REVIEW_RESPONSE_FORMAT,
        )
        if not budget_epoch:
            raise RuntimeError("missing pruning reviews require --budget-epoch")
        ledger = TokenLedger(
            Path(token_ledger_path) if token_ledger_path else target / ".token_ledger.json",
            epoch=budget_epoch,
            start_new_epoch=start_new_budget_epoch,
            max_tokens=budget_max_tokens,
        )
        errors = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(max(1, workers), len(missing))
        ) as executor:
            next_missing = 0
            exhausted = False
            while next_missing < len(missing):
                futures = {}
                while len(futures) < max(1, workers) and next_missing < len(missing):
                    candidate, prompt = missing[next_missing]
                    request_id = _sha256(
                        "\0".join((candidate["bucket_id"], model, prompt))
                    )
                    reservation = max_attempts * (
                        len(prompt.encode("utf-8")) + max_tokens + 1_024
                    )
                    if not ledger.reserve(request_id, reservation):
                        break
                    next_missing += 1
                    future = executor.submit(
                        _run_review,
                        client,
                        candidate,
                        prompt,
                        model,
                        base_url,
                        max_attempts,
                    )
                    futures[future] = (candidate["bucket_id"], request_id)
                if not futures:
                    exhausted = True
                    break
                for future in concurrent.futures.as_completed(futures):
                    bucket_id, request_id = futures[future]
                    try:
                        row = future.result()
                    except Exception as exc:  # cache successful siblings before failing
                        if _is_context_limit_rejection(exc):
                            ledger.release(request_id, reason="context_length_exceeded")
                        else:
                            ledger.complete(request_id, None)
                        errors.append(f"{bucket_id}: {exc}")
                        continue
                    ledger.complete(request_id, row.get("usage"))
                    chunk_reviews[bucket_id] = row
                    with cache_path.open("a", encoding="utf-8") as handle:
                        handle.write(_json_dump(row) + "\n")
                    calls_this_run += 1
        if errors:
            raise RuntimeError("final endpoint pruning reviews failed: " + "; ".join(errors))
        if exhausted:
            ledger.mark_exhausted()
            raise PruningBudgetExhausted(
                f"pruning budget {budget_epoch!r} exhausted after "
                f"{calls_this_run} new review(s); successful reviews were cached"
            )

    reviews: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        bucket_id = candidate["bucket_id"]
        record_ids = {
            row["canonical_record_id"] for row in candidate["target_rows"]
        }
        rows = [
            row
            for chunk in chunks_by_bucket[bucket_id]
            for row in chunk_reviews[chunk["bucket_id"]]["response"]["rows"]
        ]
        reviews[bucket_id] = {
            "response": validate_review_response(
                {"schema_version": SCHEMA_VERSION, "rows": rows},
                record_ids=record_ids,
            )
        }

    decisions = []
    for candidate in candidates:
        bucket_id = candidate["bucket_id"]
        decisions.extend(review_decisions(candidate, reviews[bucket_id]))
    decisions.sort(key=lambda row: (row["pair_bucket_key"], row["canonical_record_id"]))
    selected_reviews = [
        chunk_reviews[candidate["bucket_id"]]
        for candidate in review_candidates
    ]
    reviews_path = target / REVIEWS_FILENAME
    decisions_path = target / DECISIONS_FILENAME
    _write_jsonl_atomic(reviews_path, selected_reviews)
    temporary_decisions = decisions_path.with_suffix(".parquet.tmp")
    write_parquet(temporary_decisions, decisions)
    os.replace(temporary_decisions, decisions_path)

    dropped = {
        row["canonical_record_id"]
        for row in decisions
        if row["review_decision"] == "drop"
    }
    bucket_summaries = []
    for candidate in candidates:
        bucket_ids = {
            row["canonical_record_id"] for row in candidate["target_rows"]
        }
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
            "eligibility": "pair_bucket_records.assay_transfer_eligible",
            "unit_aware_geometry": True,
            "decade_trigger_requires_positive_linear_domain": True,
            "ambiguous_log_units_use_native_sd_only": True,
            "nonpositive_linear_buckets_use_native_sd_only": True,
            "minimum_records": MINIMUM_RECORDS,
            "minimum_distinct_parents": MINIMUM_DISTINCT_PARENTS,
            "minimum_distinct_values": MINIMUM_DISTINCT_VALUES,
            "tail_minimum_records": TAIL_MINIMUM_RECORDS,
            "tail_sigma": TAIL_SIGMA,
            "tail_distance_decades": TAIL_DECADES,
            "tail_combination": (
                "leave_one_out_5sd_all_buckets_or_2decades_when_geometry_supports_it"
            ),
            "log10_approval_gate": {
                "scope": "verified_log10.v1_axes_with_no_leave_one_out_5sd_tail",
                "minimum_records": LOG10_GATE_MINIMUM_RECORDS,
                "test": "dagostino_pearson_k2",
                "alpha": LOG10_GATE_ALPHA,
                "approve_if": "raw_p_value < alpha",
                "transform_verification_absolute_tolerance": (
                    LOG10_GATE_TRANSFORM_ABS_TOLERANCE
                ),
            },
            **detector_summary,
        },
        "review": {
            "requested_model": model,
            "selected_review_requested_models": sorted(
                {str(row.get("requested_model") or "") for row in selected_reviews}
            ),
            "served_models": sorted(
                {str(row.get("served_model") or "") for row in selected_reviews}
            ),
            "cache_model_policy": (
                "exact_prompt_schema_valid"
                if reuse_valid_cache_across_models
                else "requested_model_match"
            ),
            "endpoint": base_url,
            "reasoning_effort": "medium",
            "max_tokens": max_tokens,
            "max_attempts": max_attempts,
            "review_passes_per_prompt": 1,
            "decision_policy": "first_schema_valid_review_is_final",
            "template_path": str(TEMPLATE_PATH),
            "template_sha256": file_sha256(TEMPLATE_PATH),
            "schema_version": SCHEMA_VERSION,
            "context_rows": CONTEXT_ROWS,
            "context_selection": "sorted_value_spread_min_to_max",
            "small_bucket_context": "all_available_non_target_rows",
            "max_prompt_bytes": MAX_REVIEW_PROMPT_BYTES,
            "max_target_rows_per_chunk": MAX_TARGET_ROWS,
            "largest_prompt_bytes": max(
                (len(prompt.encode("utf-8")) for prompt in prompts.values()),
                default=0,
            ),
            "review_chunks": len(review_candidates),
            "budget_epoch": ledger.epoch if ledger is not None else None,
            "budget_spent_tokens": ledger.spent() if ledger is not None else 0,
            "budget_max_tokens": ledger.max_tokens if ledger is not None else 0,
        },
        "inputs": {
            "canonical_records": {"sha256": canonical_records_sha256},
            "records": {"path": str(records_path), "sha256": file_sha256(records_path)},
            "pair_bucket_records": {"path": str(buckets_path), "sha256": file_sha256(buckets_path)},
        },
        "summary": {
            "candidate_buckets": len(candidates),
            "candidate_rows": len(decisions),
            "candidate_bucket_finite_rows": sum(
                candidate["distribution"]["finite_record_count"] for candidate in candidates
            ),
            "tail_trigger_rows": detector_summary["tail_candidate_rows"],
            "log10_gate_approved_buckets": detector_summary[
                "log10_gate_approved_buckets"
            ],
            "log10_gate_approved_review_rows": detector_summary[
                "log10_gate_approved_review_rows"
            ],
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
            "all_candidate_rows_assay_transfer_eligible": True,
            "minimum_bucket_size_and_parent_gates": True,
            "tail_only_payloads_contain_only_triggered_rows": True,
            "semantic_metadata_only": True,
            "one_review_vote_per_target_row": True,
            "context_rows_receive_no_decisions": True,
            "exact_row_id_coverage": True,
            "prompts_within_byte_limit": True,
            "numerical_extremeness_alone_never_drops": True,
            "verified_semantic_log10_distributions_bypass_2decade_review": True,
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
    required = {
        "canonical_record_id",
        "review_decision",
        "review_reason_code",
        "review_reason",
    }
    if any(not required <= set(row) for row in rows):
        raise ValueError("final endpoint pruning decisions schema is incomplete")
    record_ids = [str(row["canonical_record_id"]) for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("final endpoint pruning decisions contain duplicate record IDs")
    if any(row["review_decision"] not in {"keep", "drop"} for row in rows):
        raise ValueError("final endpoint pruning decisions contain an invalid decision")
    summary = manifest.get("summary", {})
    dropped = {
        str(row["canonical_record_id"])
        for row in rows
        if row["review_decision"] == "drop"
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


def load_reviewed_record_ineligibility(
    manifest_path: str | Path,
    *,
    task_id: str,
    canonical_records_path: str | Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Load reviewed drops for Stage 3 without deleting canonical records."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION or manifest.get("task_id") != task_id:
        raise ValueError("assay-transfer outlier review version/task mismatch")
    if manifest.get("inputs", {}).get("canonical_records", {}).get(
        "sha256"
    ) != file_sha256(canonical_records_path):
        raise ValueError("assay-transfer outlier review is stale for Stage 2")
    decisions_path = path.parent / DECISIONS_FILENAME
    if manifest.get("files", {}).get(DECISIONS_FILENAME) != file_sha256(
        decisions_path
    ):
        raise ValueError("assay-transfer outlier decisions hash mismatch")
    rows = pq.read_table(
        decisions_path,
        columns=["canonical_record_id", "review_decision", "review_reason_code"],
    ).to_pylist()
    dropped = {
        str(row["canonical_record_id"]): (
            f"llm_review_drop:{row['review_reason_code']}"
        )
        for row in rows
        if row["review_decision"] == "drop"
    }
    if len(dropped) != int(manifest.get("summary", {}).get("dropped_rows") or 0):
        raise ValueError("assay-transfer outlier drop count differs from manifest")
    return dropped, {
        "version": VERSION,
        "manifest_sha256": file_sha256(path),
        "decisions_sha256": file_sha256(decisions_path),
        "reviewed_rows": len(rows),
        "dropped_rows": len(dropped),
    }


def _candidate_buckets(
    task_id: str, records_path: Path, buckets_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    record_columns = [
        "canonical_record_id",
        "canonical_smiles",
        "measurement_kind",
        "finite_scalar_value",
        "assay_transfer_pretransform_scalar_value",
        "assay_transfer_transform_id",
    ]
    bucket_columns = [
        "canonical_record_id",
        "pair_bucket_key",
        "canonical_pair_fields_json",
        "canonical_endpoint_name",
        "canonical_unit_text",
        "assay_transfer_eligible",
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
                bool(bucket.get("assay_transfer_eligible"))
                and _has_text(record.get("canonical_smiles"))
                and record.get("measurement_kind") == "continuous"
                and key
            ):
                continue
            group = groups.setdefault(
                key,
                {
                    "record_ids": [],
                    "canonical_smiles": [],
                    "values": [],
                    "pretransform_values": [],
                    "transform_ids": [],
                    "canonical_endpoint_name": bucket.get("canonical_endpoint_name"),
                    "canonical_unit_text": bucket.get("canonical_unit_text"),
                    "canonical_pair_fields_json": bucket.get("canonical_pair_fields_json"),
                },
            )
            group["record_ids"].append(record_id)
            group["canonical_smiles"].append(str(record["canonical_smiles"]))
            group["values"].append(record.get("finite_scalar_value"))
            group["pretransform_values"].append(
                record.get("assay_transfer_pretransform_scalar_value")
            )
            group["transform_ids"].append(record.get("assay_transfer_transform_id"))
    candidates = []
    summary = {
        "eligible_continuous_buckets": len(groups),
        "minimum_record_eligible_buckets": 0,
        "minimum_parent_eligible_buckets": 0,
        "review_eligible_buckets": 0,
        "unsupported_geometry_buckets": 0,
        "tail_candidate_buckets": 0,
        "tail_candidate_rows": 0,
        "log10_gate_evaluated_buckets": 0,
        "log10_gate_approved_buckets": 0,
        "log10_gate_approved_review_rows": 0,
        "log10_gate_approvals": [],
    }
    for key, group in sorted(groups.items()):
        finite_count = sum(_finite(value) for value in group["values"])
        parent_count = len(set(group["canonical_smiles"]))
        if finite_count >= MINIMUM_RECORDS:
            summary["minimum_record_eligible_buckets"] += 1
        if parent_count >= MINIMUM_DISTINCT_PARENTS:
            summary["minimum_parent_eligible_buckets"] += 1
        if finite_count < MINIMUM_RECORDS or parent_count < MINIMUM_DISTINCT_PARENTS:
            continue
        summary["review_eligible_buckets"] += 1
        tails = tail_outliers(
            group["values"], unit_text=str(group["canonical_unit_text"] or "")
        )
        geometry = _indexed_geometry(
            group["values"], unit_text=str(group["canonical_unit_text"] or "")
        )
        if geometry is None:
            summary["unsupported_geometry_buckets"] += 1
        log10_gate = log10_approval_gate(
            group["pretransform_values"],
            group["values"],
            group["transform_ids"],
            unit_text=str(group["canonical_unit_text"] or ""),
        )
        if log10_gate is not None:
            summary["log10_gate_evaluated_buckets"] += 1
        has_five_sd_tail = any(
            "leave_one_out_5sd" in tail["tail_trigger_reasons"]
            for tail in tails
        )
        if log10_gate is not None and log10_gate["approved"]:
            summary["log10_gate_approved_buckets"] += 1
            summary["log10_gate_approvals"].append(
                {
                    "bucket_id": _sha256(task_id + "\0" + key)[:24],
                    "pair_bucket_key": key,
                    "canonical_endpoint_name": group["canonical_endpoint_name"],
                    "canonical_unit_text": group["canonical_unit_text"],
                    "flagged_rows": len(tails),
                    **log10_gate,
                }
            )
            if tails and not has_five_sd_tail:
                summary["log10_gate_approved_review_rows"] += len(tails)
                continue
        if not tails:
            continue
        summary["tail_candidate_buckets"] += 1
        summary["tail_candidate_rows"] += len(tails)
        trigger_by_record_id = {
            group["record_ids"][tail["record_index"]]: {
                key: value for key, value in tail.items() if key != "record_index"
            }
            for tail in tails
        }
        review_record_ids = sorted(trigger_by_record_id)
        review_record_id_set = set(review_record_ids)
        context_record_ids = [
            str(row["canonical_record_id"])
            for row in _value_spread_rows(
                [
                    {
                        "canonical_record_id": record_id,
                        "finite_scalar_value": value,
                    }
                    for record_id, value in zip(
                        group["record_ids"], group["values"], strict=True
                    )
                    if record_id not in review_record_id_set and _finite(value)
                ]
            )
        ]
        try:
            pair_fields = json.loads(str(group["canonical_pair_fields_json"] or "{}"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid canonical pair fields for bucket {key}") from exc
        candidates.append(
            {
                "task_id": task_id,
                "bucket_id": _sha256(task_id + "\0" + key)[:24],
                "pair_bucket_key": key,
                "target_record_ids": review_record_ids,
                "context_record_ids": context_record_ids,
                "trigger_by_record_id": trigger_by_record_id,
                "canonical_bucket": {
                    "canonical_endpoint_name": group["canonical_endpoint_name"],
                    "canonical_unit_text": group["canonical_unit_text"],
                    "canonical_pair_fields": pair_fields,
                },
                "distribution": {
                    "record_count": len(group["record_ids"]),
                    "finite_record_count": finite_count,
                    "distinct_parent_count": parent_count,
                    "review_record_count": len(review_record_ids),
                    "review_scope": "tail_records_only",
                    "supported_gap": None,
                    "tail_candidate_count": len(tails),
                },
            }
        )
    return candidates, summary


def _attach_semantic_rows(candidates: list[dict[str, Any]], records_path: Path) -> None:
    wanted = {
        record_id
        for candidate in candidates
        for record_id in (
            *candidate["target_record_ids"],
            *candidate["context_record_ids"],
        )
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
        trigger_by_record_id = candidate.pop("trigger_by_record_id")
        candidate["target_rows"] = sorted(
            (
                {**rows[record_id], **trigger_by_record_id.get(record_id, {})}
                for record_id in candidate.pop("target_record_ids")
            ),
            key=lambda row: (
                _numeric_sort(row.get("finite_scalar_value")),
                row["canonical_record_id"],
            ),
        )
        candidate["context_rows"] = sorted(
            (rows[record_id] for record_id in candidate.pop("context_record_ids")),
            key=lambda row: (
                _numeric_sort(row.get("finite_scalar_value")),
                row["canonical_record_id"],
            ),
        )


def _run_review(
    client: OpenAICompatibleClient,
    candidate: Mapping[str, Any],
    prompt: str,
    model: str,
    base_url: str,
    max_attempts: int,
) -> dict[str, Any]:
    errors = []
    trace = None
    response = None
    usage = {"input_tokens": 0, "output_tokens": 0}
    usage_complete = True
    messages = [{"role": "user", "content": prompt}]
    for _ in range(max_attempts):
        try:
            trace = client.chat_json(messages)
            attempt_usage = _normalized_usage(trace.get("usage"))
            if attempt_usage is None:
                usage_complete = False
            else:
                for key in usage:
                    usage[key] += attempt_usage[key]
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
        "parent_bucket_id": candidate.get("parent_bucket_id", candidate["bucket_id"]),
        "review_chunk_index": candidate.get("review_chunk_index", 1),
        "review_chunk_count": candidate.get("review_chunk_count", 1),
        "reviewer": REVIEWER,
        "prompt_sha256": _sha256(prompt),
        "requested_model": model,
        "served_model": str(trace.get("model") or ""),
        "endpoint": base_url,
        "attempts": len(errors) + 1,
        "validation_errors": errors,
        "usage": usage if usage_complete else None,
        "response_id": str(trace.get("id") or ""),
        "raw_content": str(trace.get("raw_content") or ""),
        "reasoning_content": str(trace.get("reasoning_content") or ""),
        "response": response,
    }


def _is_context_limit_rejection(error: Exception) -> bool:
    text = str(error).casefold()
    return (
        "context_length_exceeded" in text
        or "input tokens exceed the configured limit" in text
    )


def _normalized_usage(value: Any) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        "input_tokens": int(value.get("input_tokens") or value.get("prompt_tokens") or 0),
        "output_tokens": int(
            value.get("output_tokens") or value.get("completion_tokens") or 0
        ),
    }


def _validate_candidate_review(
    response: Any, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    rows = candidate["target_rows"]
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


def _validate_stage3_inputs(
    root: Path, records_path: Path, buckets_path: Path
) -> str:
    manifest_path = root / "03_pair_buckets/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "records.parquet": records_path,
        "pair_bucket_records.parquet": buckets_path,
    }
    for filename, path in checks.items():
        expected = manifest.get("outputs", {}).get(filename)
        if not path.is_file() or expected != file_sha256(path):
            raise ValueError(
                "Stage 3 is missing or stale; rebuild it before outlier review"
            )
    canonical_sha256 = str(
        manifest.get("input_hashes", {}).get("canonical_records") or ""
    )
    canonical_path = root / "02_canonicalized/records.parquet"
    if not canonical_sha256 or canonical_sha256 != file_sha256(canonical_path):
        raise ValueError("Stage 3 does not match the active canonical records")
    return canonical_sha256


def _load_review_cache(path: Path) -> dict[str, dict[str, Any]]:
    output = {}
    if not path.is_file():
        return output
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                isinstance(row, dict)
                and row.get("bucket_id")
                and row.get("reviewer") == REVIEWER
            ):
                output[str(row["bucket_id"])] = row
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


def _rebuild_stage3(task_id: str, normalized_root: Path) -> None:
    module = import_task_module(task_id, "build_starling_downstream_artifacts")
    module.build_canonical_artifacts(
        normalized_root=normalized_root, cache_mode="off"
    )


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
    parser.add_argument("--max-tokens", type=int, default=4_096)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--token-ledger", type=Path)
    parser.add_argument("--budget-epoch", default="")
    parser.add_argument("--start-new-budget-epoch", action="store_true")
    parser.add_argument("--budget-max-tokens", type=int, default=10_000_000)
    parser.add_argument("--reuse-valid-cache-across-models", action="store_true")
    parser.add_argument("--rebuild-stage3", action="store_true")
    args = parser.parse_args(argv)
    with starling_build_session(args.normalized_root):
        if args.rebuild_stage3:
            _rebuild_stage3(args.task_id, args.normalized_root)
        try:
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
                token_ledger_path=args.token_ledger,
                budget_epoch=args.budget_epoch,
                start_new_budget_epoch=args.start_new_budget_epoch,
                budget_max_tokens=args.budget_max_tokens,
                reuse_valid_cache_across_models=args.reuse_valid_cache_across_models,
            )
        except PruningBudgetExhausted as error:
            print(str(error))
            return BUDGET_EXHAUSTED_EXIT_CODE
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
    "review_decisions",
    "generate_final_endpoint_pruning",
    "load_final_endpoint_pruning",
    "render_review_prompt",
    "semantic_review_columns",
    "supported_gap",
    "tail_outliers",
    "log10_approval_gate",
    "validate_review_response",
]
