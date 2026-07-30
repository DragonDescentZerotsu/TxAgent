"""Persisted canonical-field mapping for Bioavailability_Ma pair buckets."""

from __future__ import annotations


BIOAVAILABILITY_PAIR_BUCKET_VERSION = "bioavailability_ma_pair_buckets.v5"

SOURCE_PAIR_FIELDS = {
    "direct_hf": ("canonical_bioavailability_report_type",),
    "oral_exposure": (
        "canonical_dose_quantity_kind",
        "canonical_dose_basis",
        "canonical_dose_bin",
        "canonical_dose_regimen",
    ),
    "fa": ("canonical_assay_system",),
    "fg": ("canonical_assay_system",),
    "fh": ("canonical_species", "canonical_assay_system"),
}


__all__ = [
    "BIOAVAILABILITY_PAIR_BUCKET_VERSION",
    "SOURCE_PAIR_FIELDS",
]
