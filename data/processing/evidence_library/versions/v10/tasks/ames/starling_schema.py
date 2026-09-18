"""Source-faithful Stage 2 contract for Ames evidence."""

from __future__ import annotations

from data.processing.evidence_library.shared.v2.canonicalization_v7 import (
    CanonicalDimensionSpec,
    CanonicalProducerSpec,
    PairBucketSpec,
    SourceProfile,
    StarlingRecordContract,
)
from data.processing.evidence_library.shared.v2.normalization.measurement_resolution import (
    EXACT_UNIT_MAPPING_VERSION,
)
from data.processing.evidence_library.versions.v10.measurement_routing import (
    MEASUREMENT_ROUTING_VERSION,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_categorical_response import (
    CATEGORICAL_RESPONSE_VERSION,
    MEASUREMENT_SCALES,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_endpoint_normalization import (
    ENDPOINT_NORMALIZATION_VERSION,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_measurement_resolution import (
    MAPPING_VERSION as MEASUREMENT_RESOLUTION_VERSION,
)

TASK_ID = "ames"
ENDPOINT_MAPPING_VERSION = ENDPOINT_NORMALIZATION_VERSION
MEASUREMENT_ATOMIC_GROUP = "canonical_measurement_unit_pair"
ENDPOINT_PRODUCER_FIELD = "canonical_endpoint_producer_id"
PAIR_PRODUCER_FIELD = "canonical_pair_producer_id"
SOURCE_PAIR_VERSION = "ames_source_scalar_pair.v1"

SOURCE_COLUMNS = {
    "mutagenicity_outcomes": (
        "paragraph_idx",
        "support_text",
        "molecule_name",
        "chemical_entity_type",
        "mutagenicity_result",
        "evidence_basis",
        "assay_family",
        "experimental_context",
        "test_system",
        "metabolic_activation",
        "qualifying_conditions",
        "extra_details",
        "confidence",
        "needs_more_context",
        "pmid",
        "extraction_id",
        "SMILES",
    ),
    "fixed_mutation": (
        "paragraph_idx",
        "support_text",
        "assay_family",
        "endpoint_class",
        "study_context",
        "biological_test_system",
        "genetic_locus_or_chromosome_target",
        "dose_or_concentration",
        "dose_unit",
        "metabolic_activation_status",
        "result_call",
        "response_value",
        "response_unit",
        "extra_details",
        "confidence",
        "needs_more_context",
        "pmid",
        "extraction_id",
        "SMILES",
    ),
    "premutagenic_damage": (
        "paragraph_idx",
        "support_text",
        "molecule_role",
        "assay_family",
        "assay_version",
        "endpoint_class",
        "endpoint_subtype",
        "biological_system",
        "dose_or_concentration",
        "dose_unit",
        "metabolic_activation_presence",
        "result_status",
        "response_value",
        "response_unit",
        "extra_details",
        "confidence",
        "needs_more_context",
        "pmid",
        "extraction_id",
        "SMILES",
    ),
    "mutagenicity_mechanism": (
        "paragraph_idx",
        "support_text",
        "molecule_role",
        "mechanism_category",
        "assay_family",
        "assay_method_and_endpoint",
        "biological_system",
        "metabolic_activation_system",
        "exposure_and_mechanistic_conditions",
        "result_direction",
        "quantitative_readout_value",
        "quantitative_readout_unit",
        "cytotoxicity_status",
        "extra_details",
        "confidence",
        "needs_more_context",
        "pmid",
        "extraction_id",
        "SMILES",
    ),
}

ROLE_FIELDS = {
    "mutagenicity_outcomes": ("assay_family", "mutagenicity_result", ""),
    "fixed_mutation": ("assay_family", "response_value", "response_unit"),
    "premutagenic_damage": ("assay_family", "response_value", "response_unit"),
    "mutagenicity_mechanism": (
        "assay_family",
        "quantitative_readout_value",
        "quantitative_readout_unit",
    ),
}

UNIT_EXCEPTIONS = {
    "mutagenicity_outcomes": {
        "mode": "unitless_categorical",
        "reason": "mutagenicity_result is a qualitative source outcome",
    }
}


SOURCE_ENDPOINT_PRODUCER_IDS = {
    source_id: f"ames.{source_id}.reviewed_endpoint.v1" for source_id in SOURCE_COLUMNS
}
SOURCE_PAIR_PRODUCER_IDS = {
    source_id: f"ames.{source_id}.source_pair.v1" for source_id in SOURCE_COLUMNS
}
SOURCE_RULE_PAIR_PRODUCER_IDS = {
    source_id: f"ames.{source_id}.exact_source_pair.v1" for source_id in SOURCE_COLUMNS
}
SOURCE_EXTRACTION_PAIR_PRODUCER_IDS = {
    source_id: f"ames.{source_id}.frozen_extraction_pair.v1"
    for source_id in SOURCE_COLUMNS
}

EXTRACTION_CONTEXT_FIELDS = {
    "mutagenicity_outcomes": (
        "evidence_basis",
        "test_system",
        "metabolic_activation",
    ),
    "fixed_mutation": (
        "endpoint_class",
        "study_context",
        "biological_test_system",
        "result_call",
    ),
    "premutagenic_damage": (
        "endpoint_class",
        "endpoint_subtype",
        "biological_system",
        "result_status",
    ),
    "mutagenicity_mechanism": (
        "assay_method_and_endpoint",
        "biological_system",
        "mechanism_category",
        "result_direction",
    ),
}


def _categorical_producers(source_id: str) -> tuple[CanonicalProducerSpec, ...]:
    return tuple(
        CanonicalProducerSpec(
            scale.scale_id,
            scale.input_fields,
            "controlled_encoder",
            CATEGORICAL_RESPONSE_VERSION,
        )
        for scale in MEASUREMENT_SCALES.values()
        if scale.source_id == source_id
    )


def _pair_producers(source_id: str) -> tuple[CanonicalProducerSpec, ...]:
    scalar_inputs = ("endpoint_name", "measurement_text", "unit_text")
    return (
        *_categorical_producers(source_id),
        CanonicalProducerSpec(
            SOURCE_RULE_PAIR_PRODUCER_IDS[source_id],
            scalar_inputs,
            "deterministic_rule",
            f"{MEASUREMENT_ROUTING_VERSION}+{EXACT_UNIT_MAPPING_VERSION}",
        ),
        CanonicalProducerSpec(
            SOURCE_EXTRACTION_PAIR_PRODUCER_IDS[source_id],
            (*scalar_inputs, "support_text", *EXTRACTION_CONTEXT_FIELDS[source_id]),
            "frozen_extraction",
            f"{MEASUREMENT_RESOLUTION_VERSION}+{EXACT_UNIT_MAPPING_VERSION}",
        ),
    )


def _pair_dimension(
    source_id: str, output_field: str, semantic_dimension: str, legacy_field: str
) -> CanonicalDimensionSpec:
    scalar_inputs = ("endpoint_name", "measurement_text", "unit_text")
    return CanonicalDimensionSpec(
        output_field,
        semantic_dimension,
        scalar_inputs,
        "deterministic_rule",
        SOURCE_PAIR_VERSION,
        atomic_group=MEASUREMENT_ATOMIC_GROUP,
        depends_on=("canonical_endpoint_name",),
        producer_id=SOURCE_PAIR_PRODUCER_IDS[source_id],
        producer_id_field=PAIR_PRODUCER_FIELD,
        producer_variants=_pair_producers(source_id),
        legacy_value_field=legacy_field,
    )


def _canonical_dimensions(source_id: str) -> tuple[CanonicalDimensionSpec, ...]:
    categorical_producers = _categorical_producers(source_id)
    categorical_inputs = tuple(
        dict.fromkeys(
            field
            for producer in categorical_producers
            for field in producer.input_fields
        )
    )
    return (
        CanonicalDimensionSpec(
            "canonical_endpoint_name",
            "assay_family",
            ("endpoint_name",),
            "frozen_mapping",
            ENDPOINT_MAPPING_VERSION,
            producer_id=SOURCE_ENDPOINT_PRODUCER_IDS[source_id],
            producer_id_field=ENDPOINT_PRODUCER_FIELD,
            legacy_value_field="canonical_endpoint",
        ),
        _pair_dimension(
            source_id,
            "canonical_measurement_text",
            "measurement",
            "canonical_measurement",
        ),
        _pair_dimension(source_id, "canonical_unit_text", "unit", "canonical_unit"),
        CanonicalDimensionSpec(
            "canonical_smiles",
            "molecular_structure",
            ("smiles",),
            "deterministic_rule",
            "rdkit_standardized_source_smiles.v1",
            legacy_value_field="canonical_smiles",
        ),
        CanonicalDimensionSpec(
            "canonical_measurement_scale_id",
            "measurement_scale",
            categorical_inputs,
            "controlled_encoder",
            CATEGORICAL_RESPONSE_VERSION,
            legacy_value_field="categorical_encoder_id",
        ),
    )


SOURCES = {
    source_id: SourceProfile(
        source_id=source_id,
        source_columns=SOURCE_COLUMNS[source_id],
        endpoint_field=roles[0],
        measurement_field=roles[1],
        unit_field=roles[2],
        smiles_field="SMILES",
        structure_mode="direct",
        canonical_dimensions=_canonical_dimensions(source_id),
    )
    for source_id, roles in ROLE_FIELDS.items()
}

PAIR_BUCKETS = {
    source_id: PairBucketSpec(
        source_id,
        (
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_measurement_scale_id",
        ),
    )
    for source_id in SOURCES
}

RECORD_CONTRACT = StarlingRecordContract(
    TASK_ID,
    SOURCES,
    PAIR_BUCKETS,
    measurement_scales=MEASUREMENT_SCALES,
)


__all__ = [
    "ENDPOINT_MAPPING_VERSION",
    "ENDPOINT_PRODUCER_FIELD",
    "EXTRACTION_CONTEXT_FIELDS",
    "MEASUREMENT_ATOMIC_GROUP",
    "PAIR_BUCKETS",
    "PAIR_PRODUCER_FIELD",
    "RECORD_CONTRACT",
    "ROLE_FIELDS",
    "UNIT_EXCEPTIONS",
    "SOURCES",
    "SOURCE_COLUMNS",
    "SOURCE_ENDPOINT_PRODUCER_IDS",
    "SOURCE_EXTRACTION_PAIR_PRODUCER_IDS",
    "SOURCE_PAIR_PRODUCER_IDS",
    "SOURCE_PAIR_VERSION",
    "SOURCE_RULE_PAIR_PRODUCER_IDS",
    "TASK_ID",
]
