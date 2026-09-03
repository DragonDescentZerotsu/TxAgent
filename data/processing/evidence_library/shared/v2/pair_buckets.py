"""Source-scoped pair-bucket identity for canonical Starling records."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.normalization.audit import read_parquet_records


PAIR_BUCKET_CONTRACT_VERSION = "source_aware_pair_bucket.v7"
UNKNOWN_TOKEN = "__unknown__"
NUMERIC_TRANSFER_SCOPES = frozenset({"absolute", "endpoint_defined_ratio"})
_REFERENCE_FIELDS = frozenset(
    {"canonical_reference_scope", "canonical_reference_basis"}
)


def raw_pair_bucket_key(
    record: Mapping[str, Any], *, record_contract: Any,
) -> str:
    """Build the source-scoped identity used by Stage 3."""
    source_id = str(record.get("source_id") or "")
    spec = record_contract.pair_buckets[source_id]
    return _canonical_json(
        [
            source_id,
            _persisted_value(
                record.get("canonical_endpoint_name"), unknown_token=UNKNOWN_TOKEN
            ),
            _persisted_value(
                record.get("canonical_unit_text"), unknown_token=UNKNOWN_TOKEN
            ),
            *[
                _persisted_value(record.get(field), unknown_token=UNKNOWN_TOKEN)
                for field in _identity_fields(spec.additional_dimensions)
            ],
        ]
    )


def read_pair_bucket_input(
    records_path: str | Path,
    *,
    v7_source_fields: Mapping[str, tuple[str, ...]],
    legacy_source_fields: Mapping[str, tuple[str, ...]],
    v7_endpoint_field_by_source: Mapping[str, str] | None = None,
    legacy_endpoint_field_by_source: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], bool, Mapping[str, tuple[str, ...]]]:
    """Read only columns needed to materialize source-scoped buckets."""
    schema = set(pq.read_schema(records_path).names)
    v7 = "canonical_record_id" in schema
    source_fields = v7_source_fields if v7 else legacy_source_fields
    endpoint_fields = (
        v7_endpoint_field_by_source if v7 else legacy_endpoint_field_by_source
    ) or {}
    missing_endpoint_fields = set(endpoint_fields.values()) - schema
    if missing_endpoint_fields:
        raise ValueError(
            "pair-bucket input lacks endpoint concept fields: "
            f"{sorted(missing_endpoint_fields)}"
        )
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
        *endpoint_fields.values(),
    }
    if v7:
        columns.update(
            {
                "finite_scalar_value",
                "measurement_kind",
                "canonical_measurement_scale_id",
                "canonical_category_id",
                "canonical_category_rank",
                "canonical_reference_scope",
                "canonical_reference_basis",
                "assay_transfer_ineligibility_reason",
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
    # Ignored compatibility arguments for frozen task bindings.
    reference_eligibility_by_source: Mapping[str, Any] | None = None,
    required_known_fields_by_source: Mapping[str, tuple[str, ...]] | None = None,
    semantic_pair_bucket_sources: Sequence[str] = (),
    assay_transfer_record_ineligibility: Mapping[
        str, str | Mapping[str, Any]
    ]
    | None = None,
    canonical_record_contract: bool | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign every record a key and separately mark calibration candidacy.

    Reference scope, retrieval policy, required-known dimensions, reviewed row
    exclusions, and numeric-domain annotations never change identity. Numeric
    candidates accept only absolute or endpoint-defined-ratio measurements;
    controlled categorical measurements use their declared scale instead.
    """
    del (
        reference_eligibility_by_source,
        required_known_fields_by_source,
        semantic_pair_bucket_sources,
    )
    output: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    reasons: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    scope_counts: dict[str, Counter[str]] = defaultdict(Counter)
    unknown_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    v7 = (
        canonical_record_contract
        if canonical_record_contract is not None
        else any("canonical_record_id" in record for record in records)
    )
    record_id_field = "canonical_record_id" if v7 else "normalized_record_id"
    canonical_endpoint_field = (
        "canonical_endpoint_name" if v7 else "canonical_endpoint"
    )
    canonical_unit_field = "canonical_unit_text" if v7 else "canonical_unit"

    for record in records:
        record_id = str(record.get(record_id_field) or "")
        if not record_id or record_id in seen_ids:
            raise ValueError(
                f"pair-bucket input {record_id_field} values must be nonempty and unique"
            )
        seen_ids.add(record_id)
        source_id = str(record.get("source_id") or "")
        declared_fields = source_required_fields.get(source_id)
        if declared_fields is None:
            raise ValueError(f"no pair-bucket field mapping for source_id={source_id!r}")
        fields = _identity_fields(declared_fields)
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
        bucket_key = _canonical_json(
            [
                source_id,
                _persisted_value(endpoint, unknown_token=unknown_token),
                _persisted_value(unit, unknown_token=unknown_token),
                *[canonical_fields[field] for field in fields],
            ]
        )
        exclusion = str(record.get("assay_transfer_ineligibility_reason") or "") or (
            _transfer_ineligibility_reason(record, endpoint=endpoint, unit=unit)
            if v7
            else _legacy_ineligibility_reason(record, endpoint=endpoint, unit=unit)
        )
        if exclusion in {
            "unreviewed_assay_transfer_axis",
            "outside_reviewed_log10_domain",
        }:
            raise ValueError(
                f"pair-bucket input contains forbidden fallback status: {exclusion}"
            )
        if record_id in (assay_transfer_record_ineligibility or {}):
            reviewed = assay_transfer_record_ineligibility[record_id]
            exclusion = str(
                reviewed.get("reason") if isinstance(reviewed, Mapping) else reviewed
            )
        eligible = exclusion is None
        source_counts[source_id]["input_records"] += 1
        source_counts[source_id][
            "calibration_candidate_records" if eligible else "noncandidate_records"
        ] += 1
        if exclusion:
            reasons[exclusion] += 1
        scope = str(record.get("canonical_reference_scope") or UNKNOWN_TOKEN)
        scope_counts[bucket_key][scope] += 1
        molecule_key = str(
            record.get("molecule_id") or record.get("canonical_smiles") or ""
        )
        buckets[bucket_key].append((source_id, molecule_key))
        row = {
            record_id_field: record_id,
            "source_id": source_id,
            canonical_endpoint_field: canonical_endpoint or None,
            canonical_unit_field: unit or None,
            "canonical_pair_fields_json": _canonical_json(canonical_fields),
            "pair_bucket_key": bucket_key,
            "assay_transfer_eligible": eligible,
            "assay_transfer_ineligibility_reason": exclusion,
            "bucket_eligible": eligible,
            "bucket_exclusion_reason": exclusion,
        }
        if v7:
            row.update(
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
        output.append(row)

    bucket_sources = {
        key: {source for source, _ in members} for key, members in buckets.items()
    }
    molecule_counts = {
        key: len({molecule for _, molecule in members if molecule})
        for key, members in buckets.items()
    }
    eligible_count = sum(bool(row["assay_transfer_eligible"]) for row in output)
    audit = {
        "contract_version": contract_version,
        "unknown_token": unknown_token,
        "bucket_tuple_order": [
            "source_id",
            "bucket_endpoint",
            canonical_unit_field,
            "source_specific_canonical_fields_in_mapping_order",
        ],
        "reference_scope_in_identity": False,
        "numeric_transfer_scopes": sorted(NUMERIC_TRANSFER_SCOPES),
        "controlled_categorical_reference_scope_exempt": True,
        "source_required_fields": {
            source: list(_identity_fields(fields))
            for source, fields in sorted(source_required_fields.items())
        },
        "bucket_endpoint_field_by_source": {
            source: (endpoint_field_by_source or {}).get(
                source, canonical_endpoint_field
            )
            for source in sorted(source_required_fields)
        },
        "stats": {
            "input_records": len(records),
            "sidecar_records": len(output),
            "eligible_records": eligible_count,
            "ineligible_records": len(output) - eligible_count,
            "bucket_records": len(output),
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
        "ineligibility_reason_counts": dict(sorted(reasons.items())),
        "reviewed_record_ineligibility_count": len(
            assay_transfer_record_ineligibility or {}
        ),
        "bucket_reference_scope_counts": {
            key: dict(sorted(counts.items()))
            for key, counts in sorted(scope_counts.items())
        },
        "unknown_field_rates": {
            field: unknown_counts[field] / count
            for field, count in sorted(field_counts.items())
        },
        "validations": {
            "one_sidecar_row_per_input_record": len(output) == len(records),
            "every_record_has_exactly_one_bucket": all(
                bool(row["pair_bucket_key"]) for row in output
            ),
            "no_bucket_spans_sources": all(
                len(sources) == 1 for sources in bucket_sources.values()
            ),
            "reference_scope_not_in_identity": all(
                field not in _REFERENCE_FIELDS
                for fields in source_required_fields.values()
                for field in _identity_fields(fields)
            ),
        },
    }
    return output, audit


def _transfer_ineligibility_reason(
    record: Mapping[str, Any], *, endpoint: str, unit: str
) -> str | None:
    if not _endpoint_is_resolved(endpoint):
        return "unresolved_endpoint"
    if not record.get("canonical_smiles"):
        return "missing_or_invalid_structure"
    kind = str(record.get("measurement_kind") or "")
    if kind == "continuous":
        value = record.get("finite_scalar_value")
        try:
            finite = value is not None and math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            return "missing_or_nonfinite_scalar"
        scope = str(record.get("canonical_reference_scope") or "unknown")
        if scope not in NUMERIC_TRANSFER_SCOPES:
            return f"reference_scope_{scope}"
        if not unit or unit in {"free-text", "unresolved-scalar"}:
            return "unresolved_canonical_unit"
        return None
    if kind in {"binary", "ordinal"}:
        if not record.get("canonical_measurement_scale_id"):
            return "missing_controlled_categorical_scale"
        if record.get("canonical_category_id") is None:
            return "missing_controlled_category"
        if record.get("canonical_category_rank") is None:
            return "missing_controlled_category_rank"
        return None
    return "unsupported_measurement_kind"


def _legacy_ineligibility_reason(
    record: Mapping[str, Any], *, endpoint: str, unit: str
) -> str | None:
    status = str(record.get("normalization_validity_status") or "")
    if status != "valid":
        return status or "missing_normalization_validity_status"
    if not _endpoint_is_resolved(endpoint):
        return "unresolved_endpoint"
    if not unit:
        return "missing_canonical_unit"
    return None


def _identity_fields(fields: Sequence[str]) -> tuple[str, ...]:
    return tuple(field for field in fields if field not in _REFERENCE_FIELDS)


def _persisted_value(value: Any, *, unknown_token: str) -> str:
    if value is None:
        return unknown_token
    text = str(value).strip()
    return text if text else unknown_token


def _endpoint_is_resolved(endpoint: str) -> bool:
    return endpoint.strip().casefold() not in {
        "",
        "missing_endpoint",
        "unknown",
        "__unknown__",
        "__unknown_endpoint__",
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


__all__ = [
    "NUMERIC_TRANSFER_SCOPES",
    "PAIR_BUCKET_CONTRACT_VERSION",
    "UNKNOWN_TOKEN",
    "materialize_pair_buckets",
    "raw_pair_bucket_key",
    "read_pair_bucket_input",
]
