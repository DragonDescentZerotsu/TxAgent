"""Persisted canonical-field mapping for Skin_Reaction pair buckets.

Only ``sensitization_aop`` and ``skin_exposure`` report a measured scalar, so
only they receive reconciled auxiliary context buckets.

The other two sources can reach a bucket through
``starling_categorical_response``. Incidence measurements use raw fractions,
while single-subject binary outcomes and severity grades retain distinct named
scales. ``categorical_encoder_id`` therefore remains a load-bearing part of the
pair key rather than descriptive metadata.

Assay-transfer eligibility is not decided here.  It lives entirely in
``starling_pair_bucket_transfer_policy`` alongside the sample-size, variance and
distinct-level gates.
"""

from __future__ import annotations


SKIN_REACTION_PAIR_BUCKET_VERSION = "skin_reaction_pair_buckets.v4"
SKIN_REACTION_V7_PAIR_BUCKET_VERSION = "skin_reaction_pair_buckets.v10"

SOURCE_PAIR_FIELDS = {
    "direct_skin_reaction": (
        "global_context",
        "global_species_context",
        "categorical_encoder_id",
    ),
    "sensitization_aop": (
        "aop_event",
        "global_context",
        "global_species_context",
    ),
    "phototoxicity_irritation_local_damage": (
        "global_context",
        "global_species_context",
        "categorical_encoder_id",
    ),
    "skin_exposure": ("global_context", "global_species_context"),
}
ENDPOINT_FIELD_BY_SOURCE = {"sensitization_aop": "global_endpoint_context"}
V7_ENDPOINT_FIELD_BY_SOURCE = {
    source_id: "canonical_endpoint_concept" for source_id in SOURCE_PAIR_FIELDS
}


__all__ = [
    "ENDPOINT_FIELD_BY_SOURCE",
    "SKIN_REACTION_PAIR_BUCKET_VERSION",
    "SKIN_REACTION_V7_PAIR_BUCKET_VERSION",
    "SOURCE_PAIR_FIELDS",
    "V7_ENDPOINT_FIELD_BY_SOURCE",
]
