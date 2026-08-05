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
)


TASK_ID = "bioavailability_ma"
ENDPOINT_VERSION = "bioavailability_endpoint_name.v7"
MEASUREMENT_UNIT_VERSION = "bioavailability_measurement_unit.v7"
SMILES_MAPPING_VERSION = (
    "final_smiles_mapping_v2."
    "98ae43b6d9c61b77f0a95e4dce681010694b163a02e7a313667592d985496a0b"
)
DIRECT_SMILES_VERSION = "rdkit_standardized_source_smiles.v1"
REPORT_TYPE_VERSION = "bioavailability_report_type_normalization.v1"
MEASUREMENT_ATOMIC_GROUP = "canonical_measurement_unit_pair"
ENDPOINT_PRODUCER_FIELD = "canonical_endpoint_producer_id"
PAIR_PRODUCER_FIELD = "canonical_pair_producer_id"

SOURCE_ENDPOINT_PRODUCER_IDS = {
    source_id: f"bioavailability.{source_id}.source_endpoint.v1"
    for source_id in SOURCE_COLUMNS
}
SOURCE_PAIR_PRODUCER_IDS = {
    source_id: f"bioavailability.{source_id}.source_scalar_pair.v1"
    for source_id in SOURCE_COLUMNS
}


def _base_dimensions(
    *, source_id: str, mapped_structure: bool
) -> tuple[CanonicalDimensionSpec, ...]:
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
    dimensions = [
        CanonicalDimensionSpec(
            "canonical_endpoint_name",
            "endpoint_name",
            ("endpoint_name",),
            "deterministic_rule",
            ENDPOINT_VERSION,
            producer_id=SOURCE_ENDPOINT_PRODUCER_IDS[source_id],
            producer_id_field=ENDPOINT_PRODUCER_FIELD,
            producer_variants=categorical_producers,
            legacy_value_field="canonical_endpoint",
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
            producer_variants=categorical_producers,
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
            producer_variants=categorical_producers,
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
        canonical_dimensions=_base_dimensions(
            source_id="oral_exposure", mapped_structure=True
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
        "hf_bioavailability",
        (
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_measurement_scale_id",
            "canonical_bioavailability_report_type",
            "canonical_bioavailability_evidence_scope",
        ),
        (
            "species_or_population",
            "dose",
            "oral_exposure_mode",
            "comparator",
            "qualifying_conditions",
        ),
    ),
    "oral_exposure": PairBucketSpec(
        "oral_exposure",
        ("canonical_endpoint_name", "canonical_unit_text"),
        (
            "statistic_type",
            "oral_dose",
            "study_context",
            "comparator_exposure",
            "qualifying_conditions",
        ),
    ),
    "fa": PairBucketSpec(
        "fa",
        (
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_assay_context",
            "canonical_species_context",
        ),
        (
            "condition_medium",
            "formulation_or_solid_form",
            "qualifying_conditions",
        ),
    ),
    "fg": PairBucketSpec(
        "fg",
        (
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_measurement_scale_id",
            "canonical_assay_context",
            "canonical_species_context",
        ),
        (
            "transporter_or_enzyme",
            "substrate_status",
            "intestinal_site",
            "qualifying_conditions",
        ),
    ),
    "fh": PairBucketSpec(
        "fh",
        (
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_assay_context",
            "canonical_species_context",
        ),
        ("molecular_form", "enzyme_or_pathway", "qualifying_conditions"),
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
    "PAIR_BUCKETS",
    "PAIR_PRODUCER_FIELD",
    "RECORD_CONTRACT",
    "SOURCE_ENDPOINT_PRODUCER_IDS",
    "SOURCE_PAIR_PRODUCER_IDS",
    "SOURCES",
    "TASK_ID",
]
