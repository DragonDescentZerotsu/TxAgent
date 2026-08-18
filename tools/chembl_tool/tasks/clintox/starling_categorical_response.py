"""Controlled ClinTox categorical measurements for normalized Starling v7.

Only guide-declared values receive a numeric representation. Off-schema and
ambiguous values remain source-visible qualitative evidence and never enter a
pair bucket.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tools.chembl_tool.common.starling.categorical_response import (
    BINARY_OUTCOME_UNIT,
    SIGNED_DIRECTION_UNIT,
    CanonicalCategory,
    CategoricalEncoding,
    CategoricalResponsePolicy,
    ControlledMeasurementSpec,
    render_measurement,
)
from tools.chembl_tool.tasks.clintox.starling_source import DIRECT_SOURCE_ID

CATEGORICAL_RESPONSE_VERSION = "clintox_categorical_response.v2"

TOXICITY_CATEGORIES = frozenset(
    {
        "cardiotoxicity",
        "hepatotoxicity",
        "nephrotoxicity",
        "neurotoxicity",
        "hematologic_toxicity",
        "gastrointestinal_toxicity",
        "dermatologic_toxicity",
        "ocular_toxicity",
        "metabolic_or_electrolyte_toxicity",
        "general_adverse_events",
        "toxicity_absent",
        "other",
    }
)

CONTROLLED_VOCABULARIES: dict[str, dict[str, frozenset[str]]] = {
    DIRECT_SOURCE_ID: {
        "measurement_text": TOXICITY_CATEGORIES,
        "fda_approval_status": frozenset(
            {
                "FDA_approved",
                "FDA_not_approved",
                "FDA_withdrawn_or_discontinued",
                "FDA_black_box_warning",
                "component_of_FDA_approved_product",
                "clinical_development_no_approval_stated",
            }
        ),
    },
    "nonclinical_in_vivo_toxicity": {
        "endpoint_name": frozenset(
            {
                "LD50",
                "LC50",
                "MTD",
                "NOAEL",
                "LOAEL",
                "NOEL",
                "LOEL",
                "dose_limiting_toxicity",
                "mortality_or_survival",
                "body_weight_change",
                "clinical_sign",
                "animal_toxicity_finding",
                "qualitative_toxicity_assessment",
            }
        )
    },
    "organ_specific_toxicity": {
        "organ_system": frozenset(
            {
                "hepatic",
                "renal",
                "cardiac",
                "neurological",
                "hematological",
                "pulmonary",
                "gastrointestinal",
                "reproductive",
                "developmental",
            }
        ),
        "measurement_text": frozenset(
            {"injury_observed", "no_injury_observed"}
        ),
        "evidence_context": frozenset(
            {"human_clinical", "in_vivo_animal", "organ_relevant_in_vitro"}
        ),
    },
    "genotoxicity_carcinogenicity": {
        "evidence_category": frozenset(
            {"genotoxicity", "mutagenicity", "carcinogenicity"}
        ),
        "assay_type": frozenset(
            {
                "Ames_or_bacterial_reverse_mutation",
                "other_mutation_assay",
                "micronucleus_assay",
                "chromosomal_aberration_assay",
                "comet_assay",
                "DNA_damage_or_repair_assay",
                "DNA_adduct_assay",
                "cell_transformation_assay",
                "animal_carcinogenicity_study",
                "human_cancer_epidemiology",
                "overall_weight_of_evidence",
                "other_target_assay",
            }
        ),
        "study_context": frozenset(
            {
                "in_vitro",
                "in_vivo_animal",
                "human_observational",
                "overall_assessment",
                "not_stated",
            }
        ),
        "measurement_text": frozenset(
            {"positive", "negative", "equivocal", "mixed"}
        ),
    },
    "cellular_stress": {
        "endpoint_name": frozenset(
            {
                "mitochondrial_dysfunction",
                "oxidative_stress",
                "ros_generation",
                "er_stress",
                "apoptosis",
                "necrosis",
                "phospholipidosis",
                "lysosomal_disruption",
                "bile_acid_stress",
                "membrane_damage",
                "dna_damage_response",
                "stress_pathway_activation",
            }
        ),
        "measurement_text": frozenset(
            {
                "induces_or_increases",
                "reduces_or_protects",
                "no_effect",
                "mixed_or_context_dependent",
            }
        ),
        "evidence_basis": frozenset(
            {"direct_experimental", "stated_causal_mechanism", "comparative"}
        ),
    },
    "general_cytotoxicity": {
        "endpoint_name": frozenset(
            {
                "IC50",
                "EC50",
                "TC50",
                "GI50",
                "other_icx_gix",
                "viability_percent",
                "proliferation_inhibition_percent",
                "cell_death_percent",
                "membrane_integrity",
                "qualitative_cytotoxicity",
                "other_quantitative",
            }
        )
    },
    "off_target_ddi_exposure": {
        "evidence_type": frozenset(
            {
                "ion_channel_activity",
                "ecg_or_qt",
                "safety_off_target_panel",
                "metabolic_enzyme_inhibition",
                "metabolic_enzyme_induction",
                "metabolic_enzyme_substrate_or_phenotyping",
                "transporter_inhibition",
                "transporter_regulation",
                "transporter_substrate_or_transport",
                "clinical_drug_interaction",
                "phospholipidosis",
                "tissue_accumulation",
                "exposure_or_toxicokinetics",
                "protein_binding_or_partitioning",
            }
        )
    },
}


def controlled_vocabulary_violations(
    record: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return populated controlled fields whose values are off-schema."""
    source_id = str(record.get("source_id") or "")
    violations: list[str] = []
    for field, allowed in CONTROLLED_VOCABULARIES.get(source_id, {}).items():
        value = record.get(field)
        if value is None or str(value).strip() == "":
            continue
        if str(value) not in allowed:
            violations.append(field)
    return tuple(violations)


def _encoding(
    record: Mapping[str, Any],
    *,
    source_id: str,
    scale_id: str,
    values: Mapping[str, float],
    unit: str,
) -> CategoricalEncoding | None:
    if str(record.get("source_id") or "") != source_id:
        return None
    label = str(record.get("measurement_text") or "")
    value = values.get(label)
    if value is None:
        return None
    return CategoricalEncoding(
        encoder_id=scale_id,
        value=value,
        unit=unit,
        measurement_text=render_measurement(value),
        inputs={"measurement_text": label},
    )


def encode_human_toxicity(
    record: Mapping[str, Any],
) -> CategoricalEncoding | None:
    label = str(record.get("measurement_text") or "")
    values = (
        {category: 1.0 for category in TOXICITY_CATEGORIES}
        | {"toxicity_absent": 0.0}
    )
    return _encoding(
        record,
        source_id=DIRECT_SOURCE_ID,
        scale_id="clintox_human_toxicity_binary.v1",
        values=values,
        unit=BINARY_OUTCOME_UNIT,
    ) if label in TOXICITY_CATEGORIES else None


def encode_organ_injury(
    record: Mapping[str, Any],
) -> CategoricalEncoding | None:
    return _encoding(
        record,
        source_id="organ_specific_toxicity",
        scale_id="clintox_organ_injury_binary.v1",
        values={"no_injury_observed": 0.0, "injury_observed": 1.0},
        unit=BINARY_OUTCOME_UNIT,
    )


def encode_genotoxicity(
    record: Mapping[str, Any],
) -> CategoricalEncoding | None:
    return _encoding(
        record,
        source_id="genotoxicity_carcinogenicity",
        scale_id="clintox_genotoxicity_binary.v1",
        values={"negative": 0.0, "positive": 1.0},
        unit=BINARY_OUTCOME_UNIT,
    )


def encode_cellular_stress(
    record: Mapping[str, Any],
) -> CategoricalEncoding | None:
    return _encoding(
        record,
        source_id="cellular_stress",
        scale_id="clintox_cellular_stress_direction.v1",
        values={
            "reduces_or_protects": -1.0,
            "no_effect": 0.0,
            "induces_or_increases": 1.0,
        },
        unit=SIGNED_DIRECTION_UNIT,
    )


CONTROLLED_MEASUREMENTS = (
    ControlledMeasurementSpec(
        scale_id="clintox_human_toxicity_binary.v1",
        source_id=DIRECT_SOURCE_ID,
        input_fields=("measurement_text",),
        encoder=encode_human_toxicity,
        kind="binary",
        parser_id="clintox.controlled_category.v1",
        definition="toxicity_absent versus a declared human toxicity category",
        categories=(
            CanonicalCategory("toxicity_absent", 0, 0.0),
            CanonicalCategory("toxicity_present", 1, 1.0),
        ),
    ),
    ControlledMeasurementSpec(
        scale_id="clintox_organ_injury_binary.v1",
        source_id="organ_specific_toxicity",
        input_fields=("measurement_text",),
        encoder=encode_organ_injury,
        kind="binary",
        parser_id="clintox.controlled_category.v1",
        definition="explicit no-injury versus injury observation",
        categories=(
            CanonicalCategory("no_injury_observed", 0, 0.0),
            CanonicalCategory("injury_observed", 1, 1.0),
        ),
    ),
    ControlledMeasurementSpec(
        scale_id="clintox_genotoxicity_binary.v1",
        source_id="genotoxicity_carcinogenicity",
        input_fields=("measurement_text",),
        encoder=encode_genotoxicity,
        kind="binary",
        parser_id="clintox.controlled_category.v1",
        definition="negative versus positive genotoxicity result",
        categories=(
            CanonicalCategory("negative", 0, 0.0),
            CanonicalCategory("positive", 1, 1.0),
        ),
    ),
    ControlledMeasurementSpec(
        scale_id="clintox_cellular_stress_direction.v1",
        source_id="cellular_stress",
        input_fields=("measurement_text",),
        encoder=encode_cellular_stress,
        kind="ordinal",
        parser_id="clintox.controlled_category.v1",
        definition="protective, no-effect, and harmful cellular-stress direction",
        categories=(
            CanonicalCategory("reduces_or_protects", 0, -1.0),
            CanonicalCategory("no_effect", 1, 0.0),
            CanonicalCategory("induces_or_increases", 2, 1.0),
        ),
    ),
)

POLICY = CategoricalResponsePolicy(
    version=CATEGORICAL_RESPONSE_VERSION,
    controlled_measurements=CONTROLLED_MEASUREMENTS,
)
MEASUREMENT_SCALES = POLICY.measurement_scales


__all__ = [
    "CATEGORICAL_RESPONSE_VERSION",
    "CONTROLLED_MEASUREMENTS",
    "CONTROLLED_VOCABULARIES",
    "MEASUREMENT_SCALES",
    "POLICY",
    "TOXICITY_CATEGORIES",
    "controlled_vocabulary_violations",
]
