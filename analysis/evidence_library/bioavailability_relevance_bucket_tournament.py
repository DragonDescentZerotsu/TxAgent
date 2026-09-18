"""Outcome-blind semantic buckets for oral-bioavailability evidence."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from analysis.evidence_library import relevance_bucket_tournament as shared


RELEVANCE_BUCKET_COLUMNS = {
    "hf_bioavailability": (
        "canonical_endpoint_concept",
        "canonical_bioavailability_report_type", "canonical_bioavailability_evidence_scope",
    ),
    "oral_exposure": (
        "canonical_endpoint_concept", "canonical_biological_matrix",
    ),
    "fa": ("canonical_endpoint_concept",),
    "fg": ("canonical_endpoint_concept",),
    "fh": ("canonical_endpoint_concept",),
}
SAMPLE_CARD_COLUMNS = {
    "hf_bioavailability": (
        "canonical_endpoint_concept", "canonical_unit_text",
        "canonical_measurement_scale_id",
        "canonical_bioavailability_report_type", "canonical_bioavailability_evidence_scope",
    ),
    "oral_exposure": (
        "canonical_endpoint_concept", "canonical_unit_text",
        "canonical_species_context", "canonical_biological_matrix",
        "canonical_oral_dose_key",
    ),
    "fa": (
        "canonical_endpoint_concept", "canonical_unit_text",
        "canonical_assay_context", "canonical_species_context",
    ),
    "fg": (
        "canonical_endpoint_concept", "canonical_unit_text",
        "canonical_measurement_scale_id", "canonical_measurement_target_id",
        "canonical_assay_context", "canonical_species_context",
    ),
    "fh": (
        "canonical_endpoint_concept", "canonical_unit_text",
        "canonical_assay_context", "canonical_species_context",
    ),
}


def relevance_bucket(record: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    source = str(record.get("source_id") or "")
    if source not in RELEVANCE_BUCKET_COLUMNS:
        raise ValueError(f"unsupported oral-bioavailability source_id: {source!r}")
    identity = {"source_id": source}
    identity.update({
        column: str(shared._clean(record.get(column)) or shared.UNKNOWN)
        for column in RELEVANCE_BUCKET_COLUMNS[source]
    })
    return shared._canonical_json(identity), identity


def visible_card(record: Mapping[str, Any]) -> dict[str, Any]:
    source = str(record.get("source_id") or "")
    return {
        column: value
        for column in SAMPLE_CARD_COLUMNS[source]
        if (value := shared._clean(record.get(column))) is not None
    }


def select_diverse_records(
    records: Sequence[Mapping[str, Any]], limit: int = 3,
) -> list[dict[str, Any]]:
    candidates = []
    seen_records = set()
    for record in sorted(
        records,
        key=lambda row: shared._hash(
            "oral-semantic-sample-v2", str(row.get("canonical_record_id") or "")
        ),
    ):
        record_id = str(record.get("canonical_record_id") or "")
        card = visible_card(record)
        if record_id and card and record_id not in seen_records:
            candidates.append((str(record.get("pair_bucket_key") or ""), card))
            seen_records.add(record_id)

    selected = []
    selected_indices = set()
    seen_pair_buckets = set()
    for index, (pair_bucket, card) in enumerate(candidates):
        if pair_bucket in seen_pair_buckets:
            continue
        selected.append(card)
        selected_indices.add(index)
        seen_pair_buckets.add(pair_bucket)
        if len(selected) == limit:
            return selected
    for index, (_, card) in enumerate(candidates):
        if index not in selected_indices:
            selected.append(card)
            if len(selected) == limit:
                break
    return selected


def input_columns() -> list[str]:
    return sorted({
        "source_row_uid", "source_id", "pair_bucket_key", "canonical_record_id",
        *(column for columns in RELEVANCE_BUCKET_COLUMNS.values() for column in columns),
        *(column for columns in SAMPLE_CARD_COLUMNS.values() for column in columns),
    })
