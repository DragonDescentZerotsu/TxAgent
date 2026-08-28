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

PAIR_BUCKET_CONTRACT_VERSION = "source_aware_pair_bucket.v5"
UNKNOWN_TOKEN = "__unknown__"


def raw_pair_bucket_key(
    record: Mapping[str, Any], *, record_contract: Any,
) -> str:
    """Build the pre-transform bucket identity used by scientific review."""
    source_id = str(record.get("source_id") or "")
    spec = record_contract.pair_buckets[source_id]
    values = [
        source_id,
        str(record.get("canonical_endpoint_name") or ""),
        str(record.get("canonical_unit_text") or ""),
        *[
            _persisted_value(record.get(field), unknown_token=UNKNOWN_TOKEN)
            for field in spec.additional_dimensions
        ],
    ]
    return _canonical_json(values)


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
                "retrieval_eligible",
                "organization_status",
                "canonical_measurement_text",
                "finite_scalar_value",
                "measurement_kind",
                "measurement_parse_kind",
                "measurement_unit_status",
                "canonical_measurement_scale_id",
                "canonical_category_id",
                "canonical_category_rank",
            }
        )
        if "collapsed_record_id" in schema:
            columns.update(
                {
                    "collapsed_record_id",
                    "canonical_pair_fields_json",
                    "pair_bucket_key",
                    "assay_transfer_eligible",
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
    reference_eligibility_by_source: Mapping[
        str, ReferenceEligibilitySpec
    ] | None = None,
    required_known_fields_by_source: Mapping[str, tuple[str, ...]] | None = None,
    semantic_pair_bucket_sources: Sequence[str] = (),
    assay_transfer_record_ineligibility: Mapping[
        str, str | Mapping[str, Any]
    ]
    | None = None,
    canonical_record_contract: bool | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Annotate every record with assay-transfer eligibility and an optional key.

    Ineligible rows remain in the Stage-04 sidecar.  Non-semantic sources
    receive no key; retrieval-eligible semantic sources retain a key but stay
    omitted from assay-transfer calibration/model inputs.

    ``endpoint_field_by_source`` lets a source nominate a different record field
    to occupy the endpoint slot of the bucket key.  A source whose raw endpoint
    is free text can then compare on a reconciled concept while
    ``canonical_endpoint`` stays on the record at full granularity: the
    substitution changes the comparison stratum, never the endpoint's identity.

    Sources in ``semantic_pair_bucket_sources`` retain a key whenever they are
    retrieval eligible, even when the row is not eligible for assay transfer.
    Canonical semantic unit tokens are assigned before this stage, so every
    bucket uses the record's single authoritative canonical unit.
    """
    output: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    exclusions: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    unknown_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    semantic_bucket_expectations: list[bool] = []
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
    validity_field = "canonicalization_status" if v7 else "normalization_validity_status"
    semantic_sources = set(semantic_pair_bucket_sources)

    for record in records:
        record_id = str(record.get(record_id_field) or "")
        if not record_id or record_id in seen_ids:
            raise ValueError(
                f"pair-bucket input {record_id_field} values must be nonempty and unique"
            )
        seen_ids.add(record_id)
        source_id = str(record.get("source_id") or "")
        collapsed = bool(record.get("collapsed_record_id"))
        fields = source_required_fields.get(source_id)
        if fields is None and not collapsed:
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
        if collapsed:
            try:
                canonical_fields = json.loads(
                    str(record.get("canonical_pair_fields_json") or "{}")
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid collapsed pair context for record {record_id}"
                ) from exc
            if not isinstance(canonical_fields, Mapping):
                raise ValueError(
                    f"collapsed pair context must be an object for record {record_id}"
                )
        else:
            canonical_fields = {
                field: _persisted_value(record.get(field), unknown_token=unknown_token)
                for field in fields
            }
        for field, value in canonical_fields.items():
            field_counts[field] += 1
            if value == unknown_token:
                unknown_counts[field] += 1

        required_known_fields = () if collapsed else (
            required_known_fields_by_source or {}
        ).get(source_id, ())
        absent_required_fields = set(required_known_fields) - set(fields or ())
        if absent_required_fields:
            raise ValueError(
                f"required-known fields are not in the {source_id!r} pair key: "
                f"{sorted(absent_required_fields)}"
            )

        exclusion = (
            str(record.get("assay_transfer_ineligibility_reason") or "") or None
            if collapsed
            else _exclusion_reason(
                record,
                endpoint,
                unit,
                endpoint_field=endpoint_field,
                validity_field=validity_field,
            )
        )
        if (
            not collapsed
            and exclusion is None
            and reference_eligibility_by_source is not None
        ):
            reference_spec = reference_eligibility_by_source.get(source_id)
            if reference_spec is None:
                raise ValueError(
                    f"no reference eligibility policy for source_id={source_id!r}"
                )
            exclusion = reference_exclusion_reason(record, reference_spec)
        if exclusion is None:
            for field in required_known_fields:
                if canonical_fields[field] == unknown_token:
                    exclusion = f"unknown_{field}"
                    break
        if not collapsed and exclusion is None and record_id in (
            assay_transfer_record_ineligibility or {}
        ):
            reviewed = assay_transfer_record_ineligibility[record_id]
            exclusion = str(
                reviewed.get("reason")
                if isinstance(reviewed, Mapping)
                else reviewed
            )
        retrieval_eligible = bool(record.get("retrieval_eligible"))
        if (
            not collapsed
            and source_id in semantic_sources
            and not retrieval_eligible
            and exclusion is None
        ):
            exclusion = str(
                record.get("organization_status") or "retrieval_ineligible"
            )
        semantic_bucket_eligible = bool(
            source_id in semantic_sources
            and retrieval_eligible
            and _endpoint_is_resolved(endpoint)
        )
        semantic_bucket_expectations.append(semantic_bucket_eligible)
        if collapsed:
            bucket_key = str(record.get("pair_bucket_key") or "") or None
        else:
            bucket_values = [
                source_id,
                endpoint,
                unit,
                *[canonical_fields[field] for field in fields],
            ]
            bucket_key = (
                _canonical_json(bucket_values)
                if exclusion is None or semantic_bucket_eligible
                else None
            )
        source_counts[source_id]["input_records"] += 1
        if exclusion is None:
            source_counts[source_id]["eligible_records"] += 1
        else:
            source_counts[source_id]["excluded_records"] += 1
            exclusions[exclusion] += 1
        if bucket_key is not None:
            source_counts[source_id]["semantic_bucket_records"] += 1
            molecule_key = str(
                record.get("molecule_id") or record.get("canonical_smiles") or ""
            )
            buckets[str(bucket_key)].append((source_id, molecule_key))
        sidecar_row = {
            record_id_field: record_id,
            "source_id": source_id,
            canonical_endpoint_field: canonical_endpoint or None,
            canonical_unit_field: unit or None,
            "canonical_pair_fields_json": _canonical_json(canonical_fields),
            "pair_bucket_key": bucket_key,
            "assay_transfer_eligible": exclusion is None,
            "assay_transfer_ineligibility_reason": exclusion,
            # Historical row-status aliases. Despite their names, these never
            # represented Stage-05 bucket-level eligibility.
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
    semantic_bucket_count = sum(bool(row["pair_bucket_key"]) for row in output)
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
        "semantic_pair_bucket_sources": sorted(semantic_sources),
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
        "required_known_fields_by_source": {
            source: list(fields)
            for source, fields in sorted(
                (required_known_fields_by_source or {}).items()
            )
        },
        "reviewed_record_ineligibility_count": len(
            assay_transfer_record_ineligibility or {}
        ),
        "stats": {
            "input_records": len(records),
            "sidecar_records": len(output),
            "eligible_records": eligible_count,
            "ineligible_records": len(output) - eligible_count,
            "semantic_bucket_records": semantic_bucket_count,
            # Legacy metadata alias retained for frozen readers.
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
        "ineligibility_reason_counts": dict(sorted(exclusions.items())),
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
                for row, record in zip(output, records, strict=True)
                if not row["bucket_eligible"]
                and not record.get("collapsed_record_id")
                and row["source_id"] not in semantic_sources
            ),
            "retrieval_eligible_semantic_records_have_exactly_one_bucket": all(
                bool(row["pair_bucket_key"])
                for row, record, expected in zip(
                    output, records, semantic_bucket_expectations, strict=True
                )
                if not record.get("collapsed_record_id") and expected
            ),
            "retrieval_ineligible_semantic_records_have_no_bucket": all(
                not row["pair_bucket_key"]
                for row, record in zip(output, records, strict=True)
                if row["source_id"] in semantic_sources
                and not record.get("collapsed_record_id")
                and not bool(record.get("retrieval_eligible"))
            ),
            "no_bucket_spans_sources": all(
                len(sources) == 1 for sources in bucket_sources.values()
            ) or any(record.get("collapsed_record_id") for record in records),
            "collapsed_pair_bucket_identity_preserved": all(
                row["pair_bucket_key"] == record.get("pair_bucket_key")
                for row, record in zip(output, records, strict=True)
                if record.get("collapsed_record_id")
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
    "PAIR_BUCKET_CONTRACT_VERSION",
    "UNKNOWN_TOKEN",
    "materialize_pair_buckets",
    "raw_pair_bucket_key",
    "read_pair_bucket_input",
]
