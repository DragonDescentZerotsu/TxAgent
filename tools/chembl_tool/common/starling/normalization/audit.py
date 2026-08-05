"""Audits and deterministic persistence for layered normalization."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .cleaning import file_sha256
from .contracts import MeasurementPair, STAGE_REQUIRED_COLUMNS
from tools.chembl_tool.common.units import canonicalize_unit

from .measurements import (
    EndpointStandardizer,
    SourceMeasurementResolver,
    normalize_measurement_and_unit,
    parse_point_measurement,
)

PARQUET_COMPRESSION_LEVEL = 3


def write_parquet(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import pandas as pd

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(rows)
    frame.to_parquet(
        target,
        index=False,
        engine="pyarrow",
        compression="zstd",
        compression_level=PARQUET_COMPRESSION_LEVEL,
    )


def read_parquet_records(
    path: str | Path, *, columns: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(path, columns=list(columns) if columns is not None else None)
    return [
        {
            key: (
                None
                if value is None
                or (
                    isinstance(value, float)
                    and math.isnan(value)
                )
                else value
            )
            for key, value in row.items()
        }
        for row in frame.to_dict(orient="records")
    ]


def validate_cleaned_normalized_identity(
    cleaned_records: Sequence[Mapping[str, Any]],
    normalized_records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Require a one-to-one cleaned-to-normalized identity before deduplication."""
    errors: list[str] = []
    cleaned_ids = [str(row.get("cleaned_record_id") or "") for row in cleaned_records]
    normalized_counts = Counter(
        str(row.get("cleaned_record_id") or "") for row in normalized_records
    )
    if not all(cleaned_ids):
        errors.append("one or more cleaned record IDs are empty")
    if len(cleaned_ids) != len(set(cleaned_ids)):
        errors.append("cleaned record IDs are not unique")
    missing = [record_id for record_id in cleaned_ids if normalized_counts[record_id] == 0]
    repeated = [
        record_id
        for record_id in cleaned_ids
        if normalized_counts[record_id] > 1
    ]
    unknown = sorted(set(normalized_counts) - set(cleaned_ids))
    if missing:
        errors.append(
            f"{len(missing)} cleaned record(s) produced no normalized record; "
            f"first={missing[0]}"
        )
    if repeated:
        errors.append(
            f"{len(repeated)} cleaned record(s) produced multiple normalized records; "
            f"first={repeated[0]}"
        )
    if unknown:
        errors.append(
            f"{len(unknown)} normalized cleaned record ID(s) are unknown; first={unknown[0]}"
        )
    normalized_ids = [
        str(row.get("normalized_record_id") or "") for row in normalized_records
    ]
    if not all(normalized_ids):
        errors.append("one or more normalized record IDs are empty")
    if len(normalized_ids) != len(set(normalized_ids)):
        errors.append("normalized record IDs are not unique")
    return errors


def validate_stage_schema(
    rows: Sequence[Mapping[str, Any]],
    stage: str,
) -> list[str]:
    required = STAGE_REQUIRED_COLUMNS.get(stage)
    if required is None:
        return [f"unknown normalization stage: {stage}"]
    if not rows:
        return [f"{stage} artifact is empty"]
    actual = set(rows[0])
    missing = sorted(required - actual)
    return [f"{stage} artifact missing required columns: {missing}"] if missing else []


def validate_measurement_pairs(
    records: Sequence[Mapping[str, Any]],
    endpoint_standardizer: EndpointStandardizer | None = None,
    source_measurement_resolver: SourceMeasurementResolver | None = None,
    contextual_standardizer: Callable[
        [Mapping[str, Any], MeasurementPair], MeasurementPair
    ]
    | None = None,
    task: str | None = None,
) -> list[str]:
    """Recompute each authoritative pair and report any atomicity drift.

    ``task`` must match the one the records were built under: the recompute has to use the
    same qualifier vocabulary, or every record carrying a task-scoped token reads as drift.
    """
    errors: list[str] = []
    for record in records:
        baseline = normalize_measurement_and_unit(
            record.get("measurement_text"), record.get("unit_text"), task=task
        )
        if record.get("categorical_encoder_id"):
            # A categorically encoded record's canonical pair comes from a
            # declared encoder, not from the source measurement, so the
            # source-recompute invariant does not apply.  What must hold
            # instead is that the encoder only ever filled a record the source
            # path could not have scored: had the source yielded a value in a
            # recognized unit, that measurement would be authoritative.
            source_value = parse_point_measurement(baseline.canonical_measurement)
            source_unit = canonicalize_unit(baseline.canonical_unit, task=task)
            if source_value.value is not None and (
                bool(source_unit.cleaned) and not source_unit.unknown_tokens
            ):
                errors.append(
                    f"{record.get('normalized_record_id') or '<missing>'}: "
                    "categorical encoding overwrote a scorable source measurement "
                    f"({baseline.canonical_measurement!r}, {baseline.canonical_unit!r})"
                )
            continue
        expected = (
            endpoint_standardizer(str(record.get("canonical_endpoint") or ""), baseline)
            if endpoint_standardizer is not None
            else baseline
        )
        if source_measurement_resolver is not None:
            expected = source_measurement_resolver(
                record,
                str(record.get("canonical_endpoint") or ""),
                expected,
            )
        if contextual_standardizer is not None:
            expected = contextual_standardizer(record, expected)
        if (
            record.get("canonical_measurement") != expected.canonical_measurement
            or record.get("canonical_unit") != expected.canonical_unit
            or record.get("measurement_unit_status") != expected.status
            or record.get("unit_notation_status") != expected.unit_notation_status
            or record.get("unit_notation_factor") != expected.unit_notation_factor
        ):
            errors.append(
                f"{record.get('normalized_record_id') or '<missing>'}: "
                f"actual=({record.get('canonical_measurement')!r}, "
                f"{record.get('canonical_unit')!r}) expected=("
                f"{expected.canonical_measurement!r}, "
                f"{expected.canonical_unit!r}, {expected.status!r}, "
                f"{expected.unit_notation_status!r}, "
                f"{expected.unit_notation_factor!r})"
            )
    return errors


def scalar_distribution_audit(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize clean scalars by canonical endpoint and canonical unit."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    metadata: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for record in records:
        endpoint = str(record.get("canonical_endpoint") or "unassigned")
        unit = str(record.get("canonical_unit") or "")
        key = (endpoint, unit)
        status = str(record.get("measurement_unit_status") or "not_assessed")
        metadata[key][status] += 1
        value = record.get("finite_scalar_value")
        if value is not None:
            grouped[key].append(float(value))

    output: list[dict[str, Any]] = []
    for key in sorted(set(grouped) | set(metadata)):
        values = sorted(grouped.get(key, []))
        positive = [value for value in values if value > 0]
        endpoint, unit = key
        output.append(
            {
                "canonical_endpoint": endpoint,
                "canonical_unit": unit or None,
                "n_records": sum(metadata[key].values()),
                "n_finite_scalars": len(values),
                "n_positive": len(positive),
                "n_nonpositive": len(values) - len(positive),
                "minimum": values[0] if values else None,
                "p01": _quantile(values, 0.01),
                "p05": _quantile(values, 0.05),
                "p25": _quantile(values, 0.25),
                "median": _quantile(values, 0.50),
                "p75": _quantile(values, 0.75),
                "p95": _quantile(values, 0.95),
                "p99": _quantile(values, 0.99),
                "maximum": values[-1] if values else None,
                "measurement_unit_status_counts_json": json.dumps(
                    metadata[key], sort_keys=True
                ),
            }
        )
    return output


def stage_manifest(
    *,
    stage: str,
    version: str,
    inputs: Mapping[str, str | Path],
    output: str | Path,
    row_counts: Mapping[str, int],
    validations: Mapping[str, Any],
) -> dict[str, Any]:
    output_path = Path(output)
    return {
        "stage": stage,
        "version": version,
        "inputs": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for name, path in inputs.items()
            if Path(path).exists()
        },
        "output": {
            "path": str(output_path),
            "sha256": file_sha256(output_path),
        },
        "row_counts": dict(row_counts),
        "validations": dict(validations),
    }


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


__all__ = [
    "read_parquet_records",
    "scalar_distribution_audit",
    "stage_manifest",
    "validate_measurement_pairs",
    "validate_cleaned_normalized_identity",
    "validate_stage_schema",
    "write_parquet",
]
