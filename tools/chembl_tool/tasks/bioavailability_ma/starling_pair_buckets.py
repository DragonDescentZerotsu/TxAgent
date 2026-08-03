"""Persisted canonical-field mapping for Bioavailability_Ma pair buckets."""

from __future__ import annotations


BIOAVAILABILITY_PAIR_BUCKET_VERSION = "bioavailability_ma_pair_buckets.v8"

SOURCE_PAIR_FIELDS = {
    "direct_hf": ("canonical_bioavailability_report_type",),
    "oral_exposure": (),
    "fa": ("global_context", "global_species_context"),
    "fg": ("global_context", "global_species_context"),
    "fh": ("global_context", "global_species_context"),
}


__all__ = [
    "BIOAVAILABILITY_PAIR_BUCKET_VERSION",
    "SOURCE_PAIR_FIELDS",
]
