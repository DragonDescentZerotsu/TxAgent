"""Persisted canonical-field mapping for Bioavailability_Ma pair buckets."""

from __future__ import annotations


BIOAVAILABILITY_PAIR_BUCKET_VERSION = "bioavailability_ma_pair_buckets.v12"
BIOAVAILABILITY_V7_PAIR_BUCKET_VERSION = "bioavailability_ma_pair_buckets.v20"

ENDPOINT_FIELD_BY_SOURCE = {
    source_id: "canonical_endpoint_concept"
    for source_id in ("hf_bioavailability", "oral_exposure", "fa", "fg", "fh")
}

SOURCE_PAIR_FIELDS = {
    "hf_bioavailability": (
        "canonical_bioavailability_report_type",
        "canonical_bioavailability_evidence_scope",
    ),
    "oral_exposure": (),
    "fa": ("global_context", "global_species_context"),
    "fg": (
        "canonical_measurement_target_id",
        "global_context",
        "global_species_context",
    ),
    "fh": ("global_context", "global_species_context"),
}


__all__ = [
    "BIOAVAILABILITY_PAIR_BUCKET_VERSION",
    "BIOAVAILABILITY_V7_PAIR_BUCKET_VERSION",
    "ENDPOINT_FIELD_BY_SOURCE",
    "SOURCE_PAIR_FIELDS",
]
