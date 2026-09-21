"""Source-faithful Stage 1 contract for six DILI extraction layers."""

from __future__ import annotations

from data.processing.evidence_library.shared.v2.canonicalization_v7 import (
    CanonicalDimensionSpec,
    PairBucketSpec,
    SourceProfile,
    StarlingRecordContract,
)
from data.processing.evidence_library.shared.v2.reference_semantics import (
    REFERENCE_SEMANTICS_VERSION,
)
from data.processing.evidence_library.versions.v10.tasks.dili.starling_reference_semantics import (
    REFERENCE_SEMANTICS_CONFIG,
)


TASK_ID = "dili"
BASE_SOURCE_GROUP = "human_dili_relation"
PAIR_MAPPING_VERSION = "dili_pair_dimensions.v1"

PAIR_DIMENSION_INPUTS = {
    "dili_base": {
        "canonical_assay_context": (
            "human_evidence_basis",
            "clinical_phenotype",
        ),
    },
    "dili_v1": {
        "canonical_assay_context": (
            "assay_and_readout",
            "biological_model_context",
            "culture_format",
        ),
        "canonical_species_context": ("species",),
    },
    "dili_v2": {
        "canonical_assay_context": (
            "assay_detail",
            "biological_system",
            "mechanistic_entities_and_components",
        ),
        "canonical_species_context": ("species",),
    },
    "dili_v3": {
        "canonical_assay_context": (
            "assay_format",
            "target_or_process",
            "probe_or_analyte",
            "biological_system_context",
        ),
    },
    "dili_v4": {
        "canonical_assay_context": (
            "experimental_system",
            "analytical_platform_and_normalization",
            "endpoint_and_comparator",
        ),
    },
    "dili_v5": {
        "canonical_assay_context": (
            "assay_and_platform",
            "biological_model_context",
            "specific_endpoint_name",
            "quantitative_measure_type",
        ),
    },
}

RAW_SOURCE_COLUMNS = {
    "dili_base": (
        "paragraph_idx",
        "support_text",
        "molecule_name",
        "agent_type",
        "liver_injury_identifier",
        "causal_status",
        "human_evidence_basis",
        "exposure_context",
        "reaction_type",
        "biochemical_pattern",
        "clinical_phenotype",
        "maximum_reported_severity",
        "qualifying_conditions",
        "extra_details",
        "confidence",
        "needs_more_context",
        "pmid",
        "extraction_id",
        "SMILES",
    ),
    "dili_v1": (
        "paragraph_idx",
        "support_text",
        "endpoint_category",
        "assay_and_readout",
        "biological_model_context",
        "species",
        "culture_format",
        "concentration_values",
        "concentration_unit",
        "exposure_duration_and_schedule",
        "exposure_regimen",
        "result_value",
        "result_unit",
        "extra_details",
        "confidence",
        "needs_more_context",
        "pmid",
        "extraction_id",
        "SMILES",
    ),
    "dili_v2": (
        "paragraph_idx",
        "support_text",
        "molecule_name",
        "tested_entity_role",
        "assay_category",
        "assay_detail",
        "biological_system",
        "species",
        "mechanistic_entities_and_components",
        "test_concentration",
        "test_concentration_unit",
        "exposure_time",
        "reported_result",
        "extra_details",
        "confidence",
        "needs_more_context",
        "pmid",
        "extraction_id",
        "SMILES",
    ),
    "dili_v3": (
        "paragraph_idx",
        "support_text",
        "tested_entity_form",
        "assay_category",
        "assay_format",
        "target_or_process",
        "probe_or_analyte",
        "biological_system_context",
        "exposure_conditions",
        "endpoint_metric",
        "effect_direction",
        "result_value",
        "result_unit",
        "extra_details",
        "confidence",
        "needs_more_context",
        "pmid",
        "extraction_id",
        "SMILES",
    ),
    "dili_v4": (
        "paragraph_idx",
        "support_text",
        "molecule_dose_value",
        "molecule_dose_unit",
        "immune_stimulus_regimen",
        "exposure_schedule",
        "experimental_system",
        "assay_family",
        "analytical_platform_and_normalization",
        "endpoint_and_comparator",
        "effect_direction",
        "result_value",
        "result_unit",
        "extra_details",
        "confidence",
        "needs_more_context",
        "pmid",
        "extraction_id",
        "SMILES",
    ),
    "dili_v5": (
        "paragraph_idx",
        "support_text",
        "endpoint_class",
        "endpoint_name",
        "assay_and_platform",
        "biological_model_context",
        "exposure_level",
        "exposure_unit",
        "treatment_and_comparator_context",
        "effect_direction",
        "quantitative_measure_type",
        "quantitative_value",
        "quantitative_unit",
        "extra_details",
        "confidence",
        "needs_more_context",
        "pmid",
        "extraction_id",
        "SMILES",
    ),
}

# V5's source field would collide with the common endpoint role, which is supplied
# by endpoint_class. Preserve that distinct source value under an explicit name.
SOURCE_COLUMNS = {
    **{key: columns for key, columns in RAW_SOURCE_COLUMNS.items() if key != "dili_v5"},
    "dili_v5": tuple(
        "specific_endpoint_name" if column == "endpoint_name" else column
        for column in RAW_SOURCE_COLUMNS["dili_v5"]
    ),
}

ROLE_FIELDS = {
    "dili_base": ("", "causal_status", ""),
    "dili_v1": ("endpoint_category", "result_value", "result_unit"),
    "dili_v2": ("assay_category", "reported_result", ""),
    "dili_v3": ("assay_category", "result_value", "result_unit"),
    "dili_v4": ("assay_family", "result_value", "result_unit"),
    "dili_v5": ("endpoint_class", "quantitative_value", "quantitative_unit"),
}

UNIT_EXCEPTIONS = {
    "dili_base": {
        "mode": "unitless_categorical",
        "reason": "causal_status is a qualitative source outcome",
    },
    "dili_v2": {
        "mode": "embedded_in_measurement",
        "reason": "reported_result preserves its source unit inline",
    },
}

def _canonical_dimensions(source_id: str) -> tuple[CanonicalDimensionSpec, ...]:
    universal = (
        CanonicalDimensionSpec(
            "canonical_endpoint_name",
            "endpoint",
            ("endpoint_name",),
            "frozen_mapping",
            PAIR_MAPPING_VERSION,
        ),
        CanonicalDimensionSpec(
            "canonical_endpoint_concept",
            "endpoint_concept",
            ("endpoint_name",),
            "frozen_mapping",
            PAIR_MAPPING_VERSION,
            missing_policy="explicit_unknown",
            legacy_value_field="canonical_endpoint_concept",
        ),
        CanonicalDimensionSpec(
            "canonical_unit_text",
            "unit",
            ("endpoint_name", "measurement_text", "unit_text", "support_text"),
            "frozen_extraction",
            PAIR_MAPPING_VERSION,
        ),
        CanonicalDimensionSpec(
            "canonical_measurement_scale_id",
            "measurement_scale",
            ("measurement_text", "unit_text"),
            "controlled_encoder",
            PAIR_MAPPING_VERSION,
        ),
    )
    contexts = tuple(
        CanonicalDimensionSpec(
            output_field,
            output_field.removeprefix("canonical_"),
            input_fields,
            "frozen_mapping",
            PAIR_MAPPING_VERSION,
            missing_policy="explicit_unknown",
        )
        for output_field, input_fields in PAIR_DIMENSION_INPUTS[source_id].items()
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
    reference = (
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
    return (*universal, *contexts, *reference)


SOURCES = {
    source_id: SourceProfile(
        source_id=source_id,
        source_columns=SOURCE_COLUMNS[source_id],
        endpoint_field=roles[0],
        endpoint_constant=BASE_SOURCE_GROUP if source_id == "dili_base" else "",
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
            *PAIR_DIMENSION_INPUTS[source_id],
        ),
    )
    for source_id in SOURCES
}
RECORD_CONTRACT = StarlingRecordContract(TASK_ID, SOURCES, PAIR_BUCKETS)


__all__ = [
    "BASE_SOURCE_GROUP",
    "PAIR_BUCKETS",
    "PAIR_DIMENSION_INPUTS",
    "PAIR_MAPPING_VERSION",
    "RAW_SOURCE_COLUMNS",
    "RECORD_CONTRACT",
    "ROLE_FIELDS",
    "SOURCES",
    "SOURCE_COLUMNS",
    "TASK_ID",
    "UNIT_EXCEPTIONS",
]
