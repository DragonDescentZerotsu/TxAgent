"""Strict source and canonical-dimension contract for ClinTox Starling v7."""

from __future__ import annotations

from tools.chembl_tool.common.starling.canonicalization_v7 import (
    CanonicalDimensionSpec,
    PairBucketSpec,
    SourceProfile,
    StarlingRecordContract,
)
from tools.chembl_tool.tasks.clintox.starling_categorical_response import (
    CATEGORICAL_RESPONSE_VERSION,
    MEASUREMENT_SCALES,
)
from tools.chembl_tool.tasks.clintox.starling_normalization_sources import (
    SOURCE_COLUMNS,
)
from tools.chembl_tool.tasks.clintox.starling_measurement_semantics import (
    MEASUREMENT_SEMANTICS_VERSION,
)
from tools.chembl_tool.tasks.clintox.starling_reference_semantics import (
    REFERENCE_SEMANTICS_VERSION,
)
from tools.chembl_tool.tasks.clintox.starling_source import DIRECT_SOURCE_ID

TASK_ID = "clintox"
ENDPOINT_VERSION = "clintox_endpoint_name.v1"
MEASUREMENT_VERSION = MEASUREMENT_SEMANTICS_VERSION
CONTEXT_VERSION = "clintox_exact_source_context.v1"
REFERENCE_VERSION = REFERENCE_SEMANTICS_VERSION
SMILES_VERSION = "rdkit_standardized_source_smiles.v1"

_CATEGORICAL_SOURCES = {
    scale.source_id for scale in MEASUREMENT_SCALES.values()
}


def _base_dimensions(source_id: str) -> tuple[CanonicalDimensionSpec, ...]:
    categorical = source_id in _CATEGORICAL_SOURCES
    measurement_method = "controlled_encoder" if categorical else "deterministic_rule"
    measurement_version = (
        CATEGORICAL_RESPONSE_VERSION if categorical else MEASUREMENT_VERSION
    )
    dimensions = [
        CanonicalDimensionSpec(
            "canonical_endpoint_name",
            "endpoint_name",
            ("endpoint_name",),
            "deterministic_rule",
            ENDPOINT_VERSION,
            legacy_value_field="canonical_endpoint",
        ),
        CanonicalDimensionSpec(
            "canonical_measurement_text",
            "measurement",
            ("measurement_text",),
            measurement_method,
            measurement_version,
            atomic_group="canonical_measurement_unit_pair",
            depends_on=("canonical_endpoint_name",),
            legacy_value_field="canonical_measurement",
        ),
        CanonicalDimensionSpec(
            "canonical_unit_text",
            "unit",
            ("measurement_text", "unit_text"),
            measurement_method,
            measurement_version,
            atomic_group="canonical_measurement_unit_pair",
            depends_on=("canonical_endpoint_name",),
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
    if categorical:
        dimensions.append(
            CanonicalDimensionSpec(
                "canonical_measurement_scale_id",
                "measurement_scale",
                ("measurement_text",),
                "controlled_encoder",
                CATEGORICAL_RESPONSE_VERSION,
                legacy_value_field="categorical_encoder_id",
            )
        )
    dimensions.extend(
        (
            CanonicalDimensionSpec(
                "canonical_reference_scope",
                "measurement_reference_scope",
                ("endpoint_name", "measurement_text", "unit_text", "support_text"),
                "deterministic_rule",
                REFERENCE_VERSION,
                missing_policy="explicit_unknown",
                atomic_group="canonical_reference_semantics_pair",
                classification_evidence=True,
                legacy_value_field="canonical_reference_scope",
            ),
            CanonicalDimensionSpec(
                "canonical_reference_basis",
                "measurement_reference_basis",
                ("endpoint_name", "measurement_text", "unit_text", "support_text"),
                "deterministic_rule",
                REFERENCE_VERSION,
                missing_policy="explicit_unknown",
                atomic_group="canonical_reference_semantics_pair",
                classification_evidence=True,
                legacy_value_field="canonical_reference_basis",
            ),
        )
    )
    return tuple(dimensions)


def _context(
    output_field: str,
    semantic_dimension: str,
    input_field: str,
) -> CanonicalDimensionSpec:
    return CanonicalDimensionSpec(
        output_field,
        semantic_dimension,
        (input_field,),
        "deterministic_rule",
        CONTEXT_VERSION,
        missing_policy="explicit_unknown",
    )


SOURCES = {
    DIRECT_SOURCE_ID: SourceProfile(
        source_id=DIRECT_SOURCE_ID,
        source_columns=SOURCE_COLUMNS[DIRECT_SOURCE_ID],
        endpoint_constant="human_clinical_toxicity",
        measurement_field="toxicity_category",
        smiles_field="SMILES",
        canonical_dimensions=(
            *_base_dimensions(DIRECT_SOURCE_ID),
            _context(
                "canonical_clinical_context",
                "clinical_context",
                "clinical_context",
            ),
        ),
    ),
    "nonclinical_in_vivo_toxicity": SourceProfile(
        source_id="nonclinical_in_vivo_toxicity",
        source_columns=SOURCE_COLUMNS["nonclinical_in_vivo_toxicity"],
        endpoint_field="evidence_type",
        measurement_field="endpoint_value",
        unit_field="endpoint_unit",
        smiles_field="SMILES",
        canonical_dimensions=(
            *_base_dimensions("nonclinical_in_vivo_toxicity"),
            _context("canonical_animal_context", "animal_context", "animal_context"),
            _context(
                "canonical_exposure_context",
                "exposure_context",
                "exposure_context",
            ),
        ),
    ),
    "organ_specific_toxicity": SourceProfile(
        source_id="organ_specific_toxicity",
        source_columns=SOURCE_COLUMNS["organ_specific_toxicity"],
        endpoint_field="toxicity_endpoint",
        measurement_field="effect_status",
        smiles_field="SMILES",
        canonical_dimensions=(
            *_base_dimensions("organ_specific_toxicity"),
            _context("canonical_organ_system", "organ_system", "organ_system"),
            _context(
                "canonical_evidence_context",
                "evidence_context",
                "evidence_context",
            ),
            _context(
                "canonical_biological_system",
                "biological_system",
                "biological_system",
            ),
        ),
    ),
    "genotoxicity_carcinogenicity": SourceProfile(
        source_id="genotoxicity_carcinogenicity",
        source_columns=SOURCE_COLUMNS["genotoxicity_carcinogenicity"],
        endpoint_field="endpoint",
        measurement_field="result_direction",
        smiles_field="SMILES",
        canonical_dimensions=(
            *_base_dimensions("genotoxicity_carcinogenicity"),
            _context(
                "canonical_evidence_category",
                "evidence_category",
                "evidence_category",
            ),
            _context("canonical_assay_type", "assay_type", "assay_type"),
            _context("canonical_study_context", "study_context", "study_context"),
            _context(
                "canonical_biological_system",
                "biological_system",
                "biological_system",
            ),
        ),
    ),
    "cellular_stress": SourceProfile(
        source_id="cellular_stress",
        source_columns=SOURCE_COLUMNS["cellular_stress"],
        endpoint_field="stress_endpoint",
        measurement_field="effect_direction",
        smiles_field="SMILES",
        canonical_dimensions=(
            *_base_dimensions("cellular_stress"),
            _context("canonical_evidence_basis", "evidence_basis", "evidence_basis"),
            _context(
                "canonical_biological_model",
                "biological_model",
                "biological_model",
            ),
        ),
    ),
    "general_cytotoxicity": SourceProfile(
        source_id="general_cytotoxicity",
        source_columns=SOURCE_COLUMNS["general_cytotoxicity"],
        endpoint_field="endpoint_type",
        measurement_field="result_value",
        unit_field="result_unit",
        smiles_field="SMILES",
        canonical_dimensions=(
            *_base_dimensions("general_cytotoxicity"),
            _context("canonical_cell_model", "cell_model", "cell_model"),
            _context("canonical_assay_method", "assay_method", "assay_method"),
            _context(
                "canonical_exposure_time",
                "exposure_time",
                "exposure_time_h",
            ),
        ),
    ),
    "off_target_ddi_exposure": SourceProfile(
        source_id="off_target_ddi_exposure",
        source_columns=SOURCE_COLUMNS["off_target_ddi_exposure"],
        endpoint_field="target_or_endpoint",
        measurement_field="result_value",
        unit_field="result_unit",
        smiles_field="SMILES",
        canonical_dimensions=(
            *_base_dimensions("off_target_ddi_exposure"),
            _context("canonical_evidence_type", "evidence_type", "evidence_type"),
            _context("canonical_result_metric", "result_metric", "result_metric"),
            _context("canonical_assay_context", "assay_context", "assay_context"),
        ),
    ),
}


_REFERENCE_FIELDS = (
    "canonical_reference_scope",
    "canonical_reference_basis",
)

PAIR_BUCKETS = {
    DIRECT_SOURCE_ID: PairBucketSpec(
        source_id=DIRECT_SOURCE_ID,
        canonical_dimensions=(
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_measurement_scale_id",
            "canonical_clinical_context",
            *_REFERENCE_FIELDS,
        ),
        variance_candidates=("dose_or_exposure", "fda_approval_status"),
        eligible_reference_scopes=("not_applicable",),
        reference_basis_required=True,
        required_known_dimensions=(
            "canonical_measurement_scale_id",
            *_REFERENCE_FIELDS,
        ),
    ),
    "nonclinical_in_vivo_toxicity": PairBucketSpec(
        source_id="nonclinical_in_vivo_toxicity",
        canonical_dimensions=(
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_animal_context",
            "canonical_exposure_context",
            *_REFERENCE_FIELDS,
        ),
        variance_candidates=("administered_dose", "qualifying_conditions"),
        eligible_reference_scopes=(
            "absolute",
            "endpoint_defined_ratio",
            "standardized_control_ratio",
        ),
        reference_basis_required=True,
        required_known_dimensions=_REFERENCE_FIELDS,
    ),
    "organ_specific_toxicity": PairBucketSpec(
        source_id="organ_specific_toxicity",
        canonical_dimensions=(
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_measurement_scale_id",
            "canonical_organ_system",
            "canonical_evidence_context",
            "canonical_biological_system",
            *_REFERENCE_FIELDS,
        ),
        variance_candidates=("exposure_regimen", "qualifying_conditions"),
        eligible_reference_scopes=("not_applicable",),
        reference_basis_required=True,
        required_known_dimensions=(
            "canonical_measurement_scale_id",
            *_REFERENCE_FIELDS,
        ),
    ),
    "genotoxicity_carcinogenicity": PairBucketSpec(
        source_id="genotoxicity_carcinogenicity",
        canonical_dimensions=(
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_measurement_scale_id",
            "canonical_evidence_category",
            "canonical_assay_type",
            "canonical_study_context",
            "canonical_biological_system",
            *_REFERENCE_FIELDS,
        ),
        variance_candidates=("exposure_conditions", "qualifying_conditions"),
        eligible_reference_scopes=("not_applicable",),
        reference_basis_required=True,
        required_known_dimensions=(
            "canonical_measurement_scale_id",
            *_REFERENCE_FIELDS,
        ),
    ),
    "cellular_stress": PairBucketSpec(
        source_id="cellular_stress",
        canonical_dimensions=(
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_measurement_scale_id",
            "canonical_evidence_basis",
            "canonical_biological_model",
            *_REFERENCE_FIELDS,
        ),
        variance_candidates=("dose_and_duration", "qualifying_conditions"),
        eligible_reference_scopes=("not_applicable",),
        reference_basis_required=True,
        required_known_dimensions=(
            "canonical_measurement_scale_id",
            *_REFERENCE_FIELDS,
        ),
    ),
    "general_cytotoxicity": PairBucketSpec(
        source_id="general_cytotoxicity",
        canonical_dimensions=(
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_cell_model",
            "canonical_assay_method",
            "canonical_exposure_time",
            *_REFERENCE_FIELDS,
        ),
        variance_candidates=("test_concentration", "qualifying_conditions"),
        eligible_reference_scopes=(
            "absolute",
            "endpoint_defined_ratio",
            "standardized_control_ratio",
        ),
        reference_basis_required=True,
        required_known_dimensions=_REFERENCE_FIELDS,
    ),
    "off_target_ddi_exposure": PairBucketSpec(
        source_id="off_target_ddi_exposure",
        canonical_dimensions=(
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_evidence_type",
            "canonical_result_metric",
            "canonical_assay_context",
            *_REFERENCE_FIELDS,
        ),
        variance_candidates=("target_identifier", "qualifying_conditions"),
        eligible_reference_scopes=(
            "absolute",
            "endpoint_defined_ratio",
            "standardized_control_ratio",
        ),
        reference_basis_required=True,
        required_known_dimensions=_REFERENCE_FIELDS,
    ),
}

RECORD_CONTRACT = StarlingRecordContract(
    task_id=TASK_ID,
    sources=SOURCES,
    pair_buckets=PAIR_BUCKETS,
    measurement_scales=MEASUREMENT_SCALES,
)


__all__ = ["PAIR_BUCKETS", "RECORD_CONTRACT", "SOURCES", "TASK_ID"]
