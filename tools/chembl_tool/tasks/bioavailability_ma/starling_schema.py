"""Declarative Starling v7 schema for Bioavailability_Ma.

This is the single inventory for source-visible cleaning, canonical dimension
lineage, pair-bucket identity, and variance candidates.  Scientific parsing is
implemented by :mod:`starling_canonicalization`; the large Fg scalar rule set
remains a focused helper.
"""

from __future__ import annotations

from tools.chembl_tool.common.starling.canonicalization_v7 import (
    CanonicalDimensionSpec,
    CanonicalProducerSpec,
    PairBucketSpec,
    SourceProfile,
    StarlingRecordContract,
)
from tools.chembl_tool.common.starling.normalization.measurement_resolution import (
    EXACT_UNIT_MAPPING_VERSION,
)
from tools.chembl_tool.tasks.bioavailability_ma.data_processing.auxiliary_mapping_helpers.reconciliation import (
    MAPPING_VERSION,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_categorical_response import (
    CATEGORICAL_RESPONSE_VERSION,
    MEASUREMENT_SCALES,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_source_column_contracts import (
    SOURCE_COLUMNS,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_record_canonicalization import (
    EVIDENCE_SCOPE_VERSION,
    ORAL_DOSE_NORMALIZATION_VERSION,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_reference_semantics import (
    MAPPING_VERSION as REFERENCE_SEMANTICS_VERSION,
    REFERENCE_SEMANTICS_CONFIG,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_spacing_and_spelling import (
    ENDPOINT_CONCEPT_VERSION,
)


TASK_ID = "bioavailability_ma"
ENDPOINT_VERSION = "bioavailability_endpoint_name.v8"
MEASUREMENT_UNIT_VERSION = "bioavailability_measurement_unit.v8"
SMILES_MAPPING_VERSION = (
    "final_smiles_mapping_v2."
    "98ae43b6d9c61b77f0a95e4dce681010694b163a02e7a313667592d985496a0b"
)
DIRECT_SMILES_VERSION = "rdkit_standardized_source_smiles.v1"
REPORT_TYPE_VERSION = "bioavailability_report_type_normalization.v1"
MEASUREMENT_ATOMIC_GROUP = "canonical_measurement_unit_pair"
ENDPOINT_PRODUCER_FIELD = "canonical_endpoint_producer_id"
PAIR_PRODUCER_FIELD = "canonical_pair_producer_id"
FG_SUBSTRATE_ENDPOINT_PRODUCER_ID = (
    "bioavailability.fg.substrate_status_endpoint.v1"
)

SOURCE_ENDPOINT_PRODUCER_IDS = {
    source_id: f"bioavailability.{source_id}.source_endpoint.v1"
    for source_id in SOURCE_COLUMNS
}
SOURCE_PAIR_PRODUCER_IDS = {
    source_id: f"bioavailability.{source_id}.source_scalar_pair.v1"
    for source_id in SOURCE_COLUMNS
}
SOURCE_RULE_PAIR_PRODUCER_IDS = {
    source_id: f"bioavailability.{source_id}.exact_source_pair.v1"
    for source_id in SOURCE_COLUMNS
}
SOURCE_EXTRACTION_PAIR_PRODUCER_IDS = {
    source_id: f"bioavailability.{source_id}.frozen_extraction_pair.v1"
    for source_id in SOURCE_COLUMNS
}


def _base_dimensions(
    *, source_id: str, mapped_structure: bool
) -> tuple[CanonicalDimensionSpec, ...]:
    endpoint_producer_variants = (
        (
            CanonicalProducerSpec(
                FG_SUBSTRATE_ENDPOINT_PRODUCER_ID,
                ("substrate_status", "transporter_or_enzyme"),
                "controlled_encoder",
                CATEGORICAL_RESPONSE_VERSION,
            ),
        )
        if source_id == "fg"
        else ()
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
    scalar_inputs = ("endpoint_name", "measurement_text", "unit_text")
    pair_producers = categorical_producers + (
        CanonicalProducerSpec(
            SOURCE_RULE_PAIR_PRODUCER_IDS[source_id],
            scalar_inputs,
            "deterministic_rule",
            EXACT_UNIT_MAPPING_VERSION,
        ),
        CanonicalProducerSpec(
            SOURCE_EXTRACTION_PAIR_PRODUCER_IDS[source_id],
            scalar_inputs,
            "frozen_extraction",
            EXACT_UNIT_MAPPING_VERSION,
        ),
    )
    dimensions = [
        CanonicalDimensionSpec(
            "canonical_endpoint_name",
            "endpoint_name",
            ("endpoint_name",),
            "deterministic_rule",
            ENDPOINT_VERSION,
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
        ),
        CanonicalDimensionSpec(
            "canonical_measurement_text",
            "measurement",
            scalar_inputs,
            "deterministic_rule",
            MEASUREMENT_UNIT_VERSION,
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
            MEASUREMENT_UNIT_VERSION,
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
            (("global_identifier",) if mapped_structure else ("smiles",)),
            "frozen_mapping" if mapped_structure else "deterministic_rule",
            SMILES_MAPPING_VERSION if mapped_structure else DIRECT_SMILES_VERSION,
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
    dimensions.append(
        CanonicalDimensionSpec(
            "canonical_reference_scope",
            "measurement_reference_scope",
            reference_inputs,
            "frozen_mapping",
            REFERENCE_SEMANTICS_VERSION,
            missing_policy="explicit_unknown",
            classification_evidence=True,
            legacy_value_field="canonical_reference_scope",
        )
    )
    return tuple(dimensions)


def _mapped_context(
    output_field: str,
    semantic_dimension: str,
    input_fields: tuple[str, ...],
    legacy_value_field: str,
) -> CanonicalDimensionSpec:
    return CanonicalDimensionSpec(
        output_field,
        semantic_dimension,
        input_fields,
        "frozen_mapping",
        MAPPING_VERSION,
        missing_policy="explicit_unknown",
        legacy_value_field=legacy_value_field,
    )


SOURCES = {
    "oral_exposure": SourceProfile(
        source_id="oral_exposure",
        source_columns=SOURCE_COLUMNS["oral_exposure"],
        endpoint_field="exposure_measure",
        measurement_field="parameter_value",
        unit_field="parameter_units",
        smiles_field="smiles",
        structure_mode="mapped",
        structure_identity_field="global_identifier",
        canonical_dimensions=(
            *_base_dimensions(source_id="oral_exposure", mapped_structure=True),
            _mapped_context(
                "canonical_species_context",
                "species_context",
                ("study_context",),
                "global_species_context",
            ),
            _mapped_context(
                "canonical_biological_matrix",
                "biological_matrix",
                ("study_context",),
                "global_biological_matrix",
            ),
            CanonicalDimensionSpec(
                "canonical_oral_dose_key",
                "oral_dose",
                ("oral_dose",),
                "deterministic_rule",
                ORAL_DOSE_NORMALIZATION_VERSION,
                missing_policy="explicit_unknown",
                legacy_value_field="canonical_oral_dose_key",
            ),
        ),
    ),
    "fa": SourceProfile(
        source_id="fa",
        source_columns=SOURCE_COLUMNS["fa"],
        endpoint_field="endpoint_category",
        measurement_field="reported_value",
        unit_field="reported_units",
        smiles_field="smiles",
        structure_mode="mapped",
        structure_identity_field="global_identifier",
        canonical_dimensions=(
            *_base_dimensions(source_id="fa", mapped_structure=True),
            _mapped_context(
                "canonical_assay_context",
                "assay_context",
                ("assay_system",),
                "global_context",
            ),
            _mapped_context(
                "canonical_species_context",
                "species_context",
                ("biological_context", "assay_system"),
                "global_species_context",
            ),
        ),
    ),
    "fg": SourceProfile(
        source_id="fg",
        source_columns=SOURCE_COLUMNS["fg"],
        endpoint_field="gut_wall_process",
        measurement_field="measured_value",
        smiles_field="smiles",
        structure_mode="mapped",
        structure_identity_field="global_identifier",
        canonical_dimensions=(
            *_base_dimensions(source_id="fg", mapped_structure=True),
            CanonicalDimensionSpec(
                "canonical_measurement_target_id",
                "measurement_target",
                ("transporter_or_enzyme",),
                "controlled_encoder",
                CATEGORICAL_RESPONSE_VERSION,
            ),
            _mapped_context(
                "canonical_assay_context",
                "assay_context",
                ("assay_system",),
                "global_context",
            ),
            _mapped_context(
                "canonical_species_context",
                "species_context",
                ("assay_system",),
                "global_species_context",
            ),
        ),
    ),
    "fh": SourceProfile(
        source_id="fh",
        source_columns=SOURCE_COLUMNS["fh"],
        endpoint_field="metric_type",
        measurement_field="reported_value",
        unit_field="reported_units",
        smiles_field="smiles",
        structure_mode="mapped",
        structure_identity_field="global_identifier",
        canonical_dimensions=(
            *_base_dimensions(source_id="fh", mapped_structure=True),
            _mapped_context(
                "canonical_assay_context",
                "assay_context",
                ("assay_system",),
                "global_context",
            ),
            _mapped_context(
                "canonical_species_context",
                "species_context",
                ("species", "assay_system"),
                "global_species_context",
            ),
        ),
    ),
    "hf_bioavailability": SourceProfile(
        source_id="hf_bioavailability",
        source_columns=SOURCE_COLUMNS["hf_bioavailability"],
        endpoint_constant="oral_bioavailability",
        measurement_field="oral_bioavailability_value",
        smiles_field="smiles",
        canonical_dimensions=(
            *_base_dimensions(
                source_id="hf_bioavailability", mapped_structure=False
            ),
            CanonicalDimensionSpec(
                "canonical_bioavailability_report_type",
                "bioavailability_report_type",
                ("bioavailability_report_type",),
                "deterministic_rule",
                REPORT_TYPE_VERSION,
                legacy_value_field="canonical_bioavailability_report_type",
            ),
            CanonicalDimensionSpec(
                "canonical_bioavailability_evidence_scope",
                "bioavailability_evidence_scope",
                ("bioavailability_report_type",),
                "deterministic_rule",
                EVIDENCE_SCOPE_VERSION,
                legacy_value_field="canonical_bioavailability_evidence_scope",
            ),
        ),
    ),
}


PAIR_BUCKETS = {
    "hf_bioavailability": PairBucketSpec(
        source_id="hf_bioavailability",
        canonical_dimensions=(
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_measurement_scale_id",
            "canonical_bioavailability_report_type",
            "canonical_bioavailability_evidence_scope",
        ),
    ),
    "oral_exposure": PairBucketSpec(
        source_id="oral_exposure",
        canonical_dimensions=(
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_species_context",
            "canonical_biological_matrix",
            "canonical_oral_dose_key",
        ),
    ),
    "fa": PairBucketSpec(
        source_id="fa",
        canonical_dimensions=(
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_assay_context",
            "canonical_species_context",
        ),
    ),
    "fg": PairBucketSpec(
        source_id="fg",
        canonical_dimensions=(
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_measurement_scale_id",
            "canonical_measurement_target_id",
            "canonical_assay_context",
            "canonical_species_context",
        ),
    ),
    "fh": PairBucketSpec(
        source_id="fh",
        canonical_dimensions=(
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_assay_context",
            "canonical_species_context",
        ),
    ),
}


RECORD_CONTRACT = StarlingRecordContract(
    TASK_ID,
    SOURCES,
    PAIR_BUCKETS,
    measurement_scales=MEASUREMENT_SCALES,
)


__all__ = [
    "ENDPOINT_PRODUCER_FIELD",
    "FG_SUBSTRATE_ENDPOINT_PRODUCER_ID",
    "PAIR_BUCKETS",
    "PAIR_PRODUCER_FIELD",
    "RECORD_CONTRACT",
    "SOURCE_ENDPOINT_PRODUCER_IDS",
    "SOURCE_EXTRACTION_PAIR_PRODUCER_IDS",
    "SOURCE_PAIR_PRODUCER_IDS",
    "SOURCE_RULE_PAIR_PRODUCER_IDS",
    "SOURCES",
    "TASK_ID",
]
