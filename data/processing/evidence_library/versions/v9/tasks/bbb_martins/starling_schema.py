"""Declarative Starling v7 schema for BBB_Martins.

Cleaned rows keep the source's real field names except for the four universal
record roles.  Canonical fields then state exactly which cleaned inputs they
integrate or extract.  In particular, ``biological_system`` and
``assay_system`` may each support both assay and species canonical dimensions
without creating a misleading cleaned ``species`` alias.
"""

from __future__ import annotations

from data.processing.evidence_library.shared.v2.canonicalization_v7 import (
    CanonicalDimensionSpec,
    CanonicalProducerSpec,
    PairBucketSpec,
    SourceProfile,
    StarlingRecordContract,
)
from data.processing.evidence_library.versions.v9.measurement_routing import (
    MEASUREMENT_ROUTING_VERSION,
)
from data.processing.evidence_library.shared.v2.normalization.measurement_resolution import (
    EXACT_UNIT_MAPPING_VERSION,
)
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.starling_measurement_resolution import (
    MAPPING_VERSION as MEASUREMENT_RESOLUTION_VERSION,
)
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.starling_auxiliary_metadata import (
    MAPPING_VERSION,
)
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.starling_categorical_response import (
    CATEGORICAL_RESPONSE_VERSION,
    MEASUREMENT_SCALES,
)
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.starling_endpoint_normalization import (
    DIRECT_PERMEABILITY_ENDPOINT_PRODUCER_ID,
    EFFLUX_CONCLUSION_ENDPOINT_PRODUCER_ID,
    ENDPOINT_NORMALIZATION_VERSION,
    MISSING_ENDPOINT_MAPPING_VERSION,
    MISSING_ENDPOINT_PRODUCER_ID,
    PASSIVE_INTERPRETATION_ENDPOINT_PRODUCER_ID,
)
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.starling_normalization_policy import (
    SOURCE_MEASUREMENT_RESOLVER_VERSION,
)
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.starling_source_column_contracts import (
    SOURCE_COLUMNS,
)
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.starling_reference_semantics import (
    MAPPING_VERSION as REFERENCE_SEMANTICS_VERSION,
    REFERENCE_SEMANTICS_CONFIG,
)
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.starling_spacing_and_spelling import (
    ENDPOINT_CONCEPT_VERSION,
)


TASK_ID = "bbb_martins"
SMILES_VERSION = "rdkit_standardized_source_smiles.v1"
MEASUREMENT_ATOMIC_GROUP = "canonical_measurement_unit_pair"
ENDPOINT_PRODUCER_FIELD = "canonical_endpoint_producer_id"
PAIR_PRODUCER_FIELD = "canonical_pair_producer_id"
KINETIC_SYMBOL_VERSION = "bbb_kinetic_symbol.v1"

SOURCE_ENDPOINT_PRODUCER_IDS = {
    "direct_bbb": "bbb.direct_endpoint_map.v1",
    "passive_permeability": "bbb.passive_endpoint_rule.v2",
    "efflux_transport": "bbb.efflux_endpoint_rule.v1",
    "influx_transport": "bbb.influx_endpoint_rule.v1",
}
SOURCE_PAIR_PRODUCER_IDS = {
    source_id: f"bbb.{source_id}.source_scalar_pair.v3"
    for source_id in SOURCE_ENDPOINT_PRODUCER_IDS
}
# Rule-routed rows: the source already separated a bare number from a reviewed,
# dimensioned unit, so the pair is taken as written.
SOURCE_RULE_PAIR_PRODUCER_IDS = {
    source_id: f"bbb.{source_id}.routed_rule_pair.v1"
    for source_id in SOURCE_ENDPOINT_PRODUCER_IDS
}
# Rows the rules decline: the pair comes from the frozen per-row extraction.
SOURCE_EXTRACTION_PAIR_PRODUCER_IDS = {
    source_id: f"bbb.{source_id}.frozen_extraction_pair.v1"
    for source_id in SOURCE_ENDPOINT_PRODUCER_IDS
}

MISSING_ENDPOINT_INPUTS = {
    "direct_bbb": (
        "bbb_permeability_label",
        "bbb_transport_label",
        "measurement_text",
        "support_text",
    ),
    "passive_permeability": (
        "assay_type",
        "passive_bbb_interpretation",
    ),
    "efflux_transport": ("evidence_type", "support_text"),
    "influx_transport": ("support_text",),
}


def _base_dimensions(
    *,
    source_id: str,
    endpoint_method: str,
) -> tuple[CanonicalDimensionSpec, ...]:
    source_endpoint_producers = {
        "direct_bbb": (
            CanonicalProducerSpec(
                DIRECT_PERMEABILITY_ENDPOINT_PRODUCER_ID,
                ("bbb_permeability_label",),
                "controlled_encoder",
                ENDPOINT_NORMALIZATION_VERSION,
            ),
        ),
        "passive_permeability": (
            CanonicalProducerSpec(
                PASSIVE_INTERPRETATION_ENDPOINT_PRODUCER_ID,
                ("passive_bbb_interpretation",),
                "controlled_encoder",
                ENDPOINT_NORMALIZATION_VERSION,
            ),
        ),
        "efflux_transport": (
            CanonicalProducerSpec(
                EFFLUX_CONCLUSION_ENDPOINT_PRODUCER_ID,
                ("interaction_conclusion",),
                "controlled_encoder",
                ENDPOINT_NORMALIZATION_VERSION,
            ),
        ),
    }.get(source_id, ())
    endpoint_producer_variants = (
        CanonicalProducerSpec(
            MISSING_ENDPOINT_PRODUCER_ID,
            MISSING_ENDPOINT_INPUTS[source_id],
            "frozen_mapping",
            MISSING_ENDPOINT_MAPPING_VERSION,
        ),
        *source_endpoint_producers,
    )
    categorical_producers = tuple(
        CanonicalProducerSpec(
            scale.scale_id,
            scale.input_fields,
            "controlled_encoder",
            CATEGORICAL_RESPONSE_VERSION,
        )
        for scale in MEASUREMENT_SCALES.values()
        if scale.source_id == source_id
    )
    scalar_inputs = (
        "endpoint_name",
        "measurement_text",
        "unit_text",
    )
    extraction_inputs = (*scalar_inputs, "support_text")
    pair_producers = categorical_producers + (
        CanonicalProducerSpec(
            SOURCE_RULE_PAIR_PRODUCER_IDS[source_id],
            scalar_inputs,
            "deterministic_rule",
            f"{MEASUREMENT_ROUTING_VERSION}+{EXACT_UNIT_MAPPING_VERSION}",
        ),
        CanonicalProducerSpec(
            SOURCE_EXTRACTION_PAIR_PRODUCER_IDS[source_id],
            extraction_inputs,
            "frozen_extraction",
            f"{MEASUREMENT_RESOLUTION_VERSION}+{EXACT_UNIT_MAPPING_VERSION}",
        ),
    )
    dimensions = [
        CanonicalDimensionSpec(
            "canonical_endpoint_name",
            "endpoint_name",
            ("endpoint_name",),
            endpoint_method,
            ENDPOINT_NORMALIZATION_VERSION,
            producer_id=SOURCE_ENDPOINT_PRODUCER_IDS[source_id],
            producer_id_field=ENDPOINT_PRODUCER_FIELD,
            producer_variants=endpoint_producer_variants,
            legacy_value_field="canonical_endpoint",
        ),
        CanonicalDimensionSpec(
            "canonical_endpoint_concept",
            "endpoint_concept",
            ("endpoint_name",),
            "frozen_mapping",
            ENDPOINT_CONCEPT_VERSION,
            depends_on=("canonical_endpoint_name",),
        ),
        CanonicalDimensionSpec(
            "canonical_measurement_text",
            "measurement",
            scalar_inputs,
            "deterministic_rule",
            SOURCE_MEASUREMENT_RESOLVER_VERSION,
            atomic_group=MEASUREMENT_ATOMIC_GROUP,
            depends_on=("canonical_endpoint_name",),
            producer_id=SOURCE_PAIR_PRODUCER_IDS[source_id],
            producer_id_field=PAIR_PRODUCER_FIELD,
            producer_variants=pair_producers,
            legacy_value_field="canonical_measurement",
        ),
        CanonicalDimensionSpec(
            "canonical_unit_text",
            "unit",
            scalar_inputs,
            "deterministic_rule",
            SOURCE_MEASUREMENT_RESOLVER_VERSION,
            atomic_group=MEASUREMENT_ATOMIC_GROUP,
            depends_on=("canonical_endpoint_name",),
            producer_id=SOURCE_PAIR_PRODUCER_IDS[source_id],
            producer_id_field=PAIR_PRODUCER_FIELD,
            producer_variants=pair_producers,
            legacy_value_field="canonical_unit",
        ),
        CanonicalDimensionSpec(
            "canonical_smiles",
            "molecular_structure",
            ("smiles",),
            "deterministic_rule",
            SMILES_VERSION,
            legacy_value_field="canonical_smiles",
        ),
    ]
    if categorical_producers:
        categorical_inputs = tuple(
            dict.fromkeys(
                field
                for producer in categorical_producers
                for field in producer.input_fields
            )
        )
        dimensions.append(
            CanonicalDimensionSpec(
                "canonical_measurement_scale_id",
                "measurement_scale",
                categorical_inputs,
                "controlled_encoder",
                CATEGORICAL_RESPONSE_VERSION,
                legacy_value_field="categorical_encoder_id",
            )
        )
    reference_inputs = tuple(
        dict.fromkeys(
            (
                "endpoint_name",
                "measurement_text",
                "unit_text",
                "support_text",
                *REFERENCE_SEMANTICS_CONFIG.source_specs[source_id].extra_fields,
            )
        )
    )
    dimensions.extend(
        (
            CanonicalDimensionSpec(
                "canonical_reference_scope",
                "measurement_reference_scope",
                reference_inputs,
                "frozen_mapping",
                REFERENCE_SEMANTICS_VERSION,
                missing_policy="explicit_unknown",
                atomic_group="canonical_reference_semantics_pair",
                classification_evidence=True,
                legacy_value_field="canonical_reference_scope",
            ),
            CanonicalDimensionSpec(
                "canonical_reference_basis",
                "measurement_reference_basis",
                reference_inputs,
                "frozen_mapping",
                REFERENCE_SEMANTICS_VERSION,
                missing_policy="explicit_unknown",
                atomic_group="canonical_reference_semantics_pair",
                classification_evidence=True,
                legacy_value_field="canonical_reference_basis",
            ),
        )
    )
    return tuple(dimensions)


def _mapped(
    output_field: str,
    semantic_dimension: str,
    inputs: tuple[str, ...],
    legacy: str,
) -> CanonicalDimensionSpec:
    return CanonicalDimensionSpec(
        output_field,
        semantic_dimension,
        inputs,
        "frozen_mapping",
        MAPPING_VERSION,
        missing_policy="explicit_unknown",
        legacy_value_field=legacy,
    )


def _rule(
    output_field: str,
    semantic_dimension: str,
    inputs: tuple[str, ...],
    legacy: str,
) -> CanonicalDimensionSpec:
    return CanonicalDimensionSpec(
        output_field,
        semantic_dimension,
        inputs,
        "deterministic_rule",
        ENDPOINT_NORMALIZATION_VERSION,
        legacy_value_field=legacy,
    )


SOURCES = {
    "direct_bbb": SourceProfile(
        source_id="direct_bbb",
        source_columns=SOURCE_COLUMNS["direct_bbb"],
        endpoint_field="quant_metric",
        measurement_field="quant_value",
        unit_field="quant_units",
        smiles_field="smiles",
        canonical_dimensions=(
            *_base_dimensions(source_id="direct_bbb", endpoint_method="frozen_mapping"),
            _mapped(
                "canonical_assay_context",
                "assay_context",
                ("assay_model",),
                "global_context",
            ),
            _mapped(
                "canonical_species_context",
                "species_context",
                ("species",),
                "global_species_context",
            ),
        ),
    ),
    "passive_permeability": SourceProfile(
        source_id="passive_permeability",
        source_columns=SOURCE_COLUMNS["passive_permeability"],
        endpoint_field="metric_name",
        measurement_field="metric_value",
        unit_field="metric_units",
        smiles_field="SMILES",
        canonical_dimensions=(
            *_base_dimensions(
                source_id="passive_permeability",
                endpoint_method="deterministic_rule",
            ),
            _rule(
                "canonical_assay_type",
                "assay_type",
                ("assay_type",),
                "canonical_assay_type",
            ),
            _mapped(
                "canonical_assay_context",
                "assay_context",
                ("biological_system",),
                "global_context",
            ),
            _mapped(
                "canonical_species_context",
                "species_context",
                ("biological_system",),
                "global_species_context",
            ),
        ),
    ),
    "efflux_transport": SourceProfile(
        source_id="efflux_transport",
        source_columns=SOURCE_COLUMNS["efflux_transport"],
        endpoint_field="quantitative_metric",
        measurement_field="quantitative_value",
        smiles_field="SMILES",
        canonical_dimensions=(
            *_base_dimensions(
                source_id="efflux_transport",
                endpoint_method="deterministic_rule",
            ),
            _rule(
                "canonical_evidence_type",
                "evidence_type",
                ("evidence_type",),
                "canonical_evidence_type",
            ),
            _rule(
                "canonical_transporter_identifier",
                "transporter_identifier",
                ("transporter_identifier",),
                "transporter_identifier",
            ),
            _mapped(
                "canonical_assay_context",
                "assay_context",
                ("assay_system",),
                "global_context",
            ),
            _mapped(
                "canonical_species_context",
                "species_context",
                ("assay_system",),
                "global_species_context",
            ),
        ),
    ),
    "influx_transport": SourceProfile(
        source_id="influx_transport",
        source_columns=SOURCE_COLUMNS["influx_transport"],
        endpoint_field="transport_endpoint",
        measurement_field="reported_result",
        smiles_field="SMILES",
        canonical_dimensions=(
            *_base_dimensions(
                source_id="influx_transport",
                endpoint_method="deterministic_rule",
            ),
            _rule(
                "canonical_transport_mechanism",
                "transport_mechanism",
                ("transport_mechanism",),
                "canonical_transport_mechanism",
            ),
            CanonicalDimensionSpec(
                "canonical_kinetic_symbol",
                "kinetic_symbol",
                ("measurement_text", "support_text"),
                "deterministic_rule",
                KINETIC_SYMBOL_VERSION,
            ),
        ),
    ),
}


PAIR_BUCKETS = {
    "direct_bbb": PairBucketSpec(
        "direct_bbb",
        (
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_measurement_scale_id",
            "canonical_assay_context",
            "canonical_species_context",
        ),
    ),
    "passive_permeability": PairBucketSpec(
        "passive_permeability",
        (
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_measurement_scale_id",
            "canonical_assay_type",
            "canonical_assay_context",
            "canonical_species_context",
        ),
    ),
    "efflux_transport": PairBucketSpec(
        "efflux_transport",
        (
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_measurement_scale_id",
            "canonical_transporter_identifier",
            "canonical_evidence_type",
            "canonical_assay_context",
            "canonical_species_context",
        ),
    ),
    "influx_transport": PairBucketSpec(
        "influx_transport",
        (
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_transport_mechanism",
            "canonical_kinetic_symbol",
        ),
    ),
}


RECORD_CONTRACT = StarlingRecordContract(
    task_id=TASK_ID,
    sources=SOURCES,
    pair_buckets=PAIR_BUCKETS,
    measurement_scales=MEASUREMENT_SCALES,
)


__all__ = [
    "ENDPOINT_PRODUCER_FIELD",
    "PAIR_BUCKETS",
    "PAIR_PRODUCER_FIELD",
    "RECORD_CONTRACT",
    "SOURCE_ENDPOINT_PRODUCER_IDS",
    "SOURCE_PAIR_PRODUCER_IDS",
    "SOURCES",
    "TASK_ID",
]
