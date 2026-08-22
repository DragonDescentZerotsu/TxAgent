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
    has_non_atomic_directional_context,
    normalize_measurement_and_unit,
    parse_point_measurement,
)

PARQUET_COMPRESSION_LEVEL = 3
PARQUET_BATCH_ROWS = 10_000


def write_parquet(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pq.write_table(pa.table({}), target, compression="zstd")
        return

    slices = [
        (start, min(start + PARQUET_BATCH_ROWS, len(rows)))
        for start in range(0, len(rows), PARQUET_BATCH_ROWS)
    ]
    schemas = [
        pa.Table.from_pylist(list(rows[start:stop])).schema for start, stop in slices
    ]
    schema = pa.unify_schemas(schemas, promote_options="permissive")
    with pq.ParquetWriter(
        target,
        schema,
        compression="zstd",
        compression_level=PARQUET_COMPRESSION_LEVEL,
    ) as writer:
        for start, stop in slices:
            writer.write_table(
                pa.Table.from_pylist(list(rows[start:stop]), schema=schema)
            )


def read_parquet_records(
    path: str | Path, *, columns: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    records: list[dict[str, Any]] = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=10_000,
        columns=list(columns) if columns is not None else None,
    ):
        records.extend(batch.to_pylist())
    return records


def validate_cleaned_normalized_identity(
    cleaned_records: Sequence[Mapping[str, Any]],
    normalized_records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Require a one-to-one cleaned-to-normalized identity before deduplication."""
    cleaned_ids = [str(row.get("cleaned_record_id") or "") for row in cleaned_records]
    return validate_cleaned_normalized_identity_ids(cleaned_ids, normalized_records)


def validate_cleaned_normalized_identity_ids(
    cleaned_ids: Sequence[str],
    normalized_records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Validate identity when the large cleaned rows were released during projection."""
    errors: list[str] = []
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
        str(
            row.get("normalized_record_id")
            or row.get("canonical_record_id")
            or ""
        )
        for row in normalized_records
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
        resolution_status = str(record.get("measurement_resolution_status") or "")
        if resolution_status in {"ok", "relative", "unsure", "unavailable"}:
            record_id = str(
                record.get("normalized_record_id")
                or record.get("canonical_record_id")
                or "<missing>"
            )
            mapping_status = str(record.get("measurement_unit_mapping_status") or "")
            mapped = mapping_status == "mapped"
            if resolution_status == "ok" and mapping_status not in {
                "mapped",
                "excluded",
                "domain_excluded",
            }:
                errors.append(f"{record_id}: resolved quantity lacks an exact unit decision")
                continue
            if resolution_status != "ok" and mapping_status:
                errors.append(f"{record_id}: unresolved quantity has an exact unit decision")
                continue
            actual_measurement = record.get("canonical_measurement")
            if actual_measurement is None:
                actual_measurement = record.get("canonical_measurement_text")
            actual_unit = record.get("canonical_unit")
            if actual_unit is None:
                actual_unit = record.get("canonical_unit_text")
            actual_scalar = record.get("finite_scalar_value")
            if record.get("assay_transfer_transform_id"):
                actual_measurement = record.get(
                    "assay_transfer_pretransform_measurement_text"
                )
                actual_unit = record.get("assay_transfer_pretransform_unit_text")
                actual_scalar = record.get(
                    "assay_transfer_pretransform_scalar_value"
                )
            if mapped:
                expected_scalar = record.get("resolved_scalar_value")
                scalar_matches = (
                    actual_scalar is not None
                    and expected_scalar is not None
                    and math.isclose(
                        float(actual_scalar),
                        float(expected_scalar),
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                )
                if (
                    actual_measurement != record.get("resolved_measurement_text")
                    or actual_unit != record.get("resolved_unit_text")
                    or not scalar_matches
                    or record.get("measurement_unit_status")
                    != "exact_unit_mapping"
                    or record.get("variation_value") is not None
                ):
                    errors.append(f"{record_id}: exact mapped tuple drifted")
            else:
                expected_status = (
                    f"exact_unit_{mapping_status}"
                    if resolution_status == "ok"
                    else f"measurement_resolution_{resolution_status}"
                )
                if record.get("measurement_unit_status") != expected_status:
                    errors.append(f"{record_id}: unresolved unit status drifted")
                elif any(
                    value is not None
                    for value in (actual_measurement, actual_unit, actual_scalar)
                ):
                    errors.append(f"{record_id}: excluded extraction retained a scalar")
            continue
        resolution_route = str(record.get("measurement_resolution_route") or "")
        if resolution_route and record.get("measurement_resolution_active"):
            record_id = str(
                record.get("normalized_record_id")
                or record.get("canonical_record_id")
                or "<missing>"
            )
            if resolution_route == "categorical":
                if not (
                    record.get("categorical_encoder_id")
                    or record.get("canonical_measurement_scale_id")
                ):
                    errors.append(f"{record_id}: categorical route lacks an encoder")
            elif any(
                record.get(field) is not None
                for field in (
                    "canonical_measurement",
                    "canonical_measurement_text",
                    "canonical_unit",
                    "canonical_unit_text",
                    "finite_scalar_value",
                )
            ):
                errors.append(f"{record_id}: unresolved route retained a scalar")
            continue
        recompute_record = record
        if record.get("assay_transfer_transform_id"):
            recompute_record = {
                **record,
                "canonical_measurement": record.get(
                    "assay_transfer_pretransform_measurement_text"
                ),
                "canonical_unit": record.get(
                    "assay_transfer_pretransform_unit_text"
                ),
                "canonical_measurement_text": record.get(
                    "assay_transfer_pretransform_measurement_text"
                ),
                "canonical_unit_text": record.get(
                    "assay_transfer_pretransform_unit_text"
                ),
            }
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
                recompute_record,
                str(record.get("canonical_endpoint") or ""),
                expected,
            )
        if contextual_standardizer is not None:
            expected = contextual_standardizer(recompute_record, expected)
        if record.get("assay_transfer_scale_factor") is not None:
            # The frozen task policy, rather than the source parser, owns this
            # reviewed base-unit correction. Its arithmetic and final tuple are
            # checked by validate_final_assay_transfer_measurements.
            continue
        actual_measurement = record.get("canonical_measurement")
        actual_unit = record.get("canonical_unit")
        if record.get("assay_transfer_transform_id"):
            actual_measurement = record.get(
                "assay_transfer_pretransform_measurement_text"
            )
            actual_unit = record.get("assay_transfer_pretransform_unit_text")
        expected_status = expected.status
        if (
            parse_point_measurement(actual_measurement).value is not None
            and has_non_atomic_directional_context(actual_measurement, actual_unit)
        ):
            expected_status = "non_atomic_directional_context"
        if (
            actual_measurement != expected.canonical_measurement
            or actual_unit != expected.canonical_unit
            or record.get("measurement_unit_status") != expected_status
            or record.get("unit_notation_status") != expected.unit_notation_status
            or record.get("unit_notation_factor") != expected.unit_notation_factor
        ):
            errors.append(
                f"{record.get('normalized_record_id') or '<missing>'}: "
                f"actual=({actual_measurement!r}, "
                f"{actual_unit!r}, {record.get('measurement_unit_status')!r}, "
                f"{record.get('unit_notation_status')!r}, "
                f"{record.get('unit_notation_factor')!r}) expected=("
                f"{expected.canonical_measurement!r}, "
                f"{expected.canonical_unit!r}, {expected_status!r}, "
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
    "validate_cleaned_normalized_identity_ids",
    "validate_stage_schema",
    "write_parquet",
]
