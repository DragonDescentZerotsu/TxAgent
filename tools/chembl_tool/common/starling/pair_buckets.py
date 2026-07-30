"""Lean pair-bucket materialization from already-canonicalized records."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


PAIR_BUCKET_CONTRACT_VERSION = "source_aware_pair_bucket.v4"
UNKNOWN_TOKEN = "__unknown__"


def materialize_pair_buckets(
    records: Sequence[Mapping[str, Any]],
    *,
    source_required_fields: Mapping[str, tuple[str, ...]],
    contract_version: str = PAIR_BUCKET_CONTRACT_VERSION,
    unknown_token: str = UNKNOWN_TOKEN,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map persisted canonical fields to one source-aware key per eligible row."""
    output: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    exclusions: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    unknown_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()

    for record in records:
        record_id = str(record.get("normalized_record_id") or "")
        if not record_id or record_id in seen_ids:
            raise ValueError(
                "pair-bucket input normalized_record_id values must be nonempty and unique"
            )
        seen_ids.add(record_id)
        source_id = str(record.get("source_id") or "")
        fields = source_required_fields.get(source_id)
        if fields is None:
            raise ValueError(f"no pair-bucket field mapping for source_id={source_id!r}")
        endpoint = str(record.get("canonical_endpoint") or "")
        unit = str(record.get("canonical_unit") or "")
        canonical_fields = {
            field: _persisted_value(record.get(field), unknown_token=unknown_token)
            for field in fields
        }
        for field, value in canonical_fields.items():
            field_counts[field] += 1
            if value == unknown_token:
                unknown_counts[field] += 1

        exclusion = _exclusion_reason(record, endpoint, unit)
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
        output.append(
            {
                "normalized_record_id": record_id,
                "source_id": source_id,
                "canonical_endpoint": endpoint or None,
                "canonical_unit": unit or None,
                "canonical_pair_fields_json": _canonical_json(canonical_fields),
                "pair_bucket_key": bucket_key,
                "bucket_eligible": exclusion is None,
                "bucket_exclusion_reason": exclusion,
            }
        )

    bucket_sources = {
        bucket: {source for source, _ in members}
        for bucket, members in buckets.items()
    }
    molecule_counts = {
        bucket: len({molecule for _, molecule in members if molecule})
        for bucket, members in buckets.items()
    }
    eligible_count = sum(bool(row["bucket_eligible"]) for row in output)
    audit = {
        "contract_version": contract_version,
        "unknown_token": unknown_token,
        "bucket_tuple_order": [
            "source_id",
            "canonical_endpoint",
            "canonical_unit",
            "source_specific_canonical_fields_in_mapping_order",
        ],
        "source_required_fields": {
            source: list(fields)
            for source, fields in sorted(source_required_fields.items())
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
) -> str | None:
    status = str(record.get("normalization_validity_status") or "")
    if status != "valid":
        return status or "missing_normalization_validity_status"
    if not endpoint:
        return "missing_canonical_endpoint"
    if not unit:
        return "missing_canonical_unit"
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
]
