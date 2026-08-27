"""Persisted canonical-field mapping for Skin_Reaction pair buckets.

Only ``sensitization_aop`` and ``skin_exposure`` report a measured scalar, so
only they receive reconciled auxiliary context buckets.

The other two sources are categorical.  Their records reach a bucket only via
``starling_categorical_response``, which places them on a named latent scale,
so they are stratified by ``categorical_encoder_id`` instead.  That field is
load-bearing rather than descriptive: ``count_logit``, ``single_subject_logit``
and ``percent_positive_logit`` all share the ``logit_response`` unit, and
``canonical_unit`` alone would let a 1/1 single-subject report set the
comparison scale for a 45/50 incidence study.

Assay-transfer eligibility is not decided here.  It lives entirely in
``starling_pair_bucket_transfer_policy`` alongside the sample-size, variance and
distinct-level gates.
"""

from __future__ import annotations


SKIN_REACTION_PAIR_BUCKET_VERSION = "skin_reaction_pair_buckets.v4"
SKIN_REACTION_V7_PAIR_BUCKET_VERSION = "skin_reaction_pair_buckets.v7"

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


__all__ = [
    "ENDPOINT_FIELD_BY_SOURCE",
    "SKIN_REACTION_PAIR_BUCKET_VERSION",
    "SKIN_REACTION_V7_PAIR_BUCKET_VERSION",
    "SOURCE_PAIR_FIELDS",
]
