"""Lean pair-bucket materialization from already-canonicalized records."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from tools.chembl_tool.common.starling.normalization.audit import read_parquet_records
from tools.chembl_tool.common.starling.reference_semantics import (
    ReferenceEligibilitySpec,
    reference_exclusion_reason,
)

PAIR_BUCKET_CONTRACT_VERSION = "source_aware_pair_bucket.v4"
UNKNOWN_TOKEN = "__unknown__"


def read_pair_bucket_input(
    records_path: str | Path,
    *,
    v7_source_fields: Mapping[str, tuple[str, ...]],
    legacy_source_fields: Mapping[str, tuple[str, ...]],
    legacy_endpoint_field_by_source: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], bool, Mapping[str, tuple[str, ...]]]:
    """Read only the Stage-03 columns used to materialize pair buckets."""
    schema = set(pq.read_schema(records_path).names)
    v7 = "canonical_record_id" in schema
    source_fields = v7_source_fields if v7 else legacy_source_fields
    endpoint_field = "canonical_endpoint_name" if v7 else "canonical_endpoint"
    unit_field = "canonical_unit_text" if v7 else "canonical_unit"
    validity_field = (
        "canonicalization_status" if v7 else "normalization_validity_status"
    )
    record_id_field = "canonical_record_id" if v7 else "normalized_record_id"
    columns = {
        record_id_field,
        "source_id",
        endpoint_field,
        unit_field,
        validity_field,
        "molecule_id",
        "canonical_smiles",
        *(field for fields in source_fields.values() for field in fields),
        *((legacy_endpoint_field_by_source or {}).values() if not v7 else ()),
    }
    if v7:
        columns.update(
            {
                "measurement_kind",
                "canonical_measurement_scale_id",
                "canonical_category_id",
                "canonical_category_rank",
            }
        )
    records = read_parquet_records(records_path, columns=sorted(columns & schema))
    return records, v7, source_fields


def materialize_pair_buckets(
    records: Sequence[Mapping[str, Any]],
    *,
    source_required_fields: Mapping[str, tuple[str, ...]],
    contract_version: str = PAIR_BUCKET_CONTRACT_VERSION,
    unknown_token: str = UNKNOWN_TOKEN,
    endpoint_field_by_source: Mapping[str, str] | None = None,
    reference_eligibility_by_source: Mapping[
        str, ReferenceEligibilitySpec
    ] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map persisted canonical fields to one source-aware key per eligible row.

    ``endpoint_field_by_source`` lets a source nominate a different record field
    to occupy the endpoint slot of the bucket key.  A source whose raw endpoint
    is free text can then compare on a reconciled concept while
    ``canonical_endpoint`` stays on the record at full granularity: the
    substitution changes the comparison stratum, never the endpoint's identity.
    """
    output: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    exclusions: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    unknown_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    v7 = any("canonical_record_id" in record for record in records)
    record_id_field = "canonical_record_id" if v7 else "normalized_record_id"
    canonical_endpoint_field = (
        "canonical_endpoint_name" if v7 else "canonical_endpoint"
    )
    canonical_unit_field = "canonical_unit_text" if v7 else "canonical_unit"
    validity_field = "canonicalization_status" if v7 else "normalization_validity_status"

    for record in records:
        record_id = str(record.get(record_id_field) or "")
        if not record_id or record_id in seen_ids:
            raise ValueError(
                f"pair-bucket input {record_id_field} values must be nonempty and unique"
            )
        seen_ids.add(record_id)
        source_id = str(record.get("source_id") or "")
        fields = source_required_fields.get(source_id)
        if fields is None:
            raise ValueError(f"no pair-bucket field mapping for source_id={source_id!r}")
        canonical_endpoint = str(record.get(canonical_endpoint_field) or "")
        endpoint_field = (endpoint_field_by_source or {}).get(
            source_id, canonical_endpoint_field
        )
        endpoint = (
            canonical_endpoint
            if endpoint_field == canonical_endpoint_field
            else str(record.get(endpoint_field) or "")
        )
        unit = str(record.get(canonical_unit_field) or "")
        canonical_fields = {
            field: _persisted_value(record.get(field), unknown_token=unknown_token)
            for field in fields
        }
        for field, value in canonical_fields.items():
            field_counts[field] += 1
            if value == unknown_token:
                unknown_counts[field] += 1

        exclusion = _exclusion_reason(
            record,
            endpoint,
            unit,
            endpoint_field=endpoint_field,
            validity_field=validity_field,
        )
        if exclusion is None and reference_eligibility_by_source is not None:
            reference_spec = reference_eligibility_by_source.get(source_id)
            if reference_spec is None:
                raise ValueError(
                    f"no reference eligibility policy for source_id={source_id!r}"
                )
            exclusion = reference_exclusion_reason(record, reference_spec)
        bucket_values = [
            source_id,
            endpoint,
            unit,
            *[canonical_fields[field] for field in fields],
        ]
        bucket_key = _canonical_json(bucket_values) if exclusion is None else None
        source_counts[source_id]["input_records"] += 1
        if exclusion is None:
            source_counts[source_id]["eligible_records"] += 1
            molecule_key = str(
                record.get("molecule_id") or record.get("canonical_smiles") or ""
            )
            buckets[str(bucket_key)].append((source_id, molecule_key))
        else:
            source_counts[source_id]["excluded_records"] += 1
            exclusions[exclusion] += 1
        sidecar_row = {
                record_id_field: record_id,
                "source_id": source_id,
                canonical_endpoint_field: canonical_endpoint or None,
                canonical_unit_field: unit or None,
                "canonical_pair_fields_json": _canonical_json(canonical_fields),
                "pair_bucket_key": bucket_key,
                "bucket_eligible": exclusion is None,
                "bucket_exclusion_reason": exclusion,
            }
        if v7:
            sidecar_row.update(
                {
                    "measurement_kind": record.get("measurement_kind"),
                    "canonical_measurement_scale_id": record.get(
                        "canonical_measurement_scale_id"
                    ),
                    "canonical_category_id": record.get("canonical_category_id"),
                    "canonical_category_rank": record.get(
                        "canonical_category_rank"
                    ),
                }
            )
        output.append(sidecar_row)

    bucket_sources = {
        bucket: {source for source, _ in members}
        for bucket, members in buckets.items()
    }
    molecule_counts = {
        bucket: len({molecule for _, molecule in members if molecule})
        for bucket, members in buckets.items()
    }
    eligible_count = sum(bool(row["bucket_eligible"]) for row in output)
    has_endpoint_override = any(
        (endpoint_field_by_source or {}).get(source, canonical_endpoint_field)
        != canonical_endpoint_field
        for source in source_required_fields
    )
    audit = {
        "contract_version": contract_version,
        "unknown_token": unknown_token,
        "bucket_tuple_order": [
            "source_id",
            "bucket_endpoint" if has_endpoint_override else canonical_endpoint_field,
            canonical_unit_field,
            "source_specific_canonical_fields_in_mapping_order",
        ],
        "source_required_fields": {
            source: list(fields)
            for source, fields in sorted(source_required_fields.items())
        },
        "bucket_endpoint_field_by_source": {
            source: (endpoint_field_by_source or {}).get(
                source, canonical_endpoint_field
            )
            for source in sorted(source_required_fields)
        },
        "reference_eligibility_by_source": {
            source: {
                "eligible_scopes": list(spec.eligible_scopes),
                "basis_required": spec.basis_required,
            }
            for source, spec in sorted(
                (reference_eligibility_by_source or {}).items()
            )
        },
        "stats": {
            "input_records": len(records),
            "sidecar_records": len(output),
            "eligible_records": eligible_count,
            "excluded_records": len(output) - eligible_count,
            "buckets": len(buckets),
            "pairable_buckets": sum(count >= 2 for count in molecule_counts.values()),
            "singleton_bucket_rate": (
                sum(count == 1 for count in molecule_counts.values()) / len(buckets)
                if buckets
                else 0.0
            ),
        },
        "source_counts": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(source_counts.items())
        },
        "exclusion_reason_counts": dict(sorted(exclusions.items())),
        "unknown_field_rates": {
            field: unknown_counts[field] / count
            for field, count in sorted(field_counts.items())
        },
        "validations": {
            "one_sidecar_row_per_input_record": len(output) == len(records),
            "eligible_records_have_exactly_one_bucket": all(
                bool(row["pair_bucket_key"])
                for row in output
                if row["bucket_eligible"]
            ),
            "ineligible_records_have_no_bucket": all(
                not row["pair_bucket_key"]
                for row in output
                if not row["bucket_eligible"]
            ),
            "no_bucket_spans_sources": all(
                len(sources) == 1 for sources in bucket_sources.values()
            ),
        },
    }
    return output, audit


def _exclusion_reason(
    record: Mapping[str, Any],
    endpoint: str,
    unit: str,
    *,
    endpoint_field: str = "canonical_endpoint",
    validity_field: str = "normalization_validity_status",
) -> str | None:
    status = str(record.get(validity_field) or "")
    if status != "valid":
        return status or f"missing_{validity_field}"
    if not endpoint:
        # Name the field that was actually missing, so a source using a
        # substituted bucket endpoint does not report a misleading reason.
        return f"missing_{endpoint_field}"
    if not unit:
        return "missing_canonical_unit_text" if validity_field == "canonicalization_status" else "missing_canonical_unit"
    return None


def _persisted_value(value: Any, *, unknown_token: str) -> str:
    if value is None:
        return unknown_token
    text = str(value).strip()
    return text if text else unknown_token


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


__all__ = [
    "PAIR_BUCKET_CONTRACT_VERSION",
    "UNKNOWN_TOKEN",
    "materialize_pair_buckets",
    "read_pair_bucket_input",
]
