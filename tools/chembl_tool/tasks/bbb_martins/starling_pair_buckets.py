"""Source-aware comparison fields for BBB Martins pair buckets."""

BBB_MARTINS_PAIR_BUCKET_VERSION = "bbb_martins_pair_buckets.v3"
BBB_MARTINS_V7_PAIR_BUCKET_VERSION = "bbb_martins_pair_buckets.v10"

SOURCE_PAIR_FIELDS = {
    "direct_bbb": (
        "categorical_encoder_id",
        "global_context",
        "global_species_context",
    ),
    "passive_permeability": (
        "categorical_encoder_id",
        "canonical_assay_type",
        "global_context",
        "global_species_context",
    ),
    "efflux_transport": (
        "categorical_encoder_id",
        "transporter_identifier",
        "canonical_evidence_type",
        "global_context",
        "global_species_context",
    ),
    "influx_transport": ("canonical_transport_mechanism",),
}

__all__ = [
    "BBB_MARTINS_PAIR_BUCKET_VERSION",
    "BBB_MARTINS_V7_PAIR_BUCKET_VERSION",
    "SOURCE_PAIR_FIELDS",
]
