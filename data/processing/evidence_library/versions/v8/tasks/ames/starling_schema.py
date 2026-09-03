"""Source-faithful V8 contract for Ames evidence."""

from __future__ import annotations

from data.processing.evidence_library.shared.v1.canonicalization_v7 import (
    CanonicalDimensionSpec,
    PairBucketSpec,
    SourceProfile,
    StarlingRecordContract,
)
from data.processing.evidence_library.versions.v8.tasks.ames.starling_categorical_response import (
    MEASUREMENT_SCALES,
)


TASK_ID = "ames"

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


def _canonical_dimensions(source_id: str) -> tuple[CanonicalDimensionSpec, ...]:
    pair_inputs = ("endpoint_name", "measurement_text", "unit_text")
    categorical_inputs = {
        "mutagenicity_outcomes": ("measurement_text",),
        "fixed_mutation": ("result_call",),
        "premutagenic_damage": ("result_status",),
        "mutagenicity_mechanism": ("result_direction",),
    }[source_id]
    return (
        CanonicalDimensionSpec(
            "canonical_endpoint_name",
            "assay_family",
            ("endpoint_name",),
            "deterministic_rule",
            "ames_assay_family.v1",
            legacy_value_field="canonical_endpoint",
        ),
        CanonicalDimensionSpec(
            "canonical_measurement_text",
            "measurement",
            pair_inputs,
            "deterministic_rule",
            "ames_measurement_pair.v1",
            atomic_group="canonical_measurement_unit_pair",
            legacy_value_field="canonical_measurement",
        ),
        CanonicalDimensionSpec(
            "canonical_unit_text",
            "unit",
            pair_inputs,
            "deterministic_rule",
            "ames_measurement_pair.v1",
            atomic_group="canonical_measurement_unit_pair",
            legacy_value_field="canonical_unit",
        ),
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
            "ames_categorical_response.v1",
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
    "PAIR_BUCKETS",
    "RECORD_CONTRACT",
    "ROLE_FIELDS",
    "SOURCE_COLUMNS",
    "SOURCES",
    "TASK_ID",
]
