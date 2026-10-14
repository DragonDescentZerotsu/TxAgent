"""Versioned source-field adapters for the DILI v5 scoring contract.

These aliases accept TxAgent native, canonical, and prepared records without
requiring its retrieval or model libraries. Reviewed nulls stay authoritative.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any, Mapping

IDENTITY_FIELDS = frozenset(
    {
        "molecule_name",
        "agent_name",
        "compound_name",
        "chemical_name",
        "canonical_smiles",
        "SMILES",
    }
)

GROUP_FIELDS = {
    "endpoint": """canonical_endpoint_name endpoint_name endpoint endpoint_category endpoint_class
 endpoint_subtype endpoint_detail endpoint_measure endpoint_metric endpoint_and_method
 endpoint_and_comparator mechanism_category aop_event target_or_process
 genetic_locus_or_chromosome_target transporter_identifier transporter_or_enzyme
 enzyme_or_pathway mediator_identifier mediator_name clinical_phenotype biochemical_pattern
 cancer_or_tumor phenotype_domain assay_domain assay_category assay_family
 quantitative_measure_type measurement_quantity_kind""".split(),
    "outcome": """canonical_measurement_text canonical_unit_text measurement_text unit_text embedded_unit
 result_label outcome_label effect_direction result_direction result_call result_status
 result_interpretation mutagenicity_result carcinogenicity_conclusion causal_status
 maximum_reported_severity classification_label classification_authority
 bbb_permeability_label bbb_transport_label passive_bbb_interpretation interaction_conclusion
 substrate_status reported_result result_value response_value endpoint_result quantitative_value
 quantitative_readout_value result_unit response_unit endpoint_unit measurement_unit
 quantitative_unit quantitative_readout_unit finite_scalar_value variation_value metric_uncertainty
 positive_count total_tested statistic_type persistence_outcome cytotoxicity_status""".split(),
    "assay": """canonical_assay_context assay_or_test assay_type assay_method assay_and_readout
 assay_and_platform assay_and_detection_method assay_method_and_endpoint assay_detail
 assay_system assay_model assay_version assay_format culture_format analytical_platform_and_normalization
 evidence_basis evidence_type evidence_system human_evidence_basis study_design""".split(),
    "system": """canonical_species_context species species_or_population biological_system
 biological_test_system biological_system_context biological_model biological_model_context
 biological_context test_system experimental_system evidence_population_or_model evidence_scope
 intestinal_site""".split(),
    "exposure": """qualifying_conditions experimental_conditions experimental_context study_context
 exposure_conditions exposure_context exposure_and_mechanistic_conditions exposure_details
 exposure_duration exposure_duration_and_schedule exposure_time exposure_timing exposure_schedule
 exposure_and_readout_timing exposure_regimen exposure_route oral_exposure_mode oral_dose dose
 dose_or_concentration dose_unit concentration_or_dose concentration_values concentration_unit
 molecule_concentration_or_dose molecule_dose_value molecule_dose_unit test_concentration
 test_concentration_unit exposure_level exposure_unit post_exposure_followup condition_medium
 formulation_or_solid_form formulation_vehicle light_conditions metabolic_activation
 metabolic_activation_status metabolic_activation_presence metabolic_activation_system
 bioavailability_report_type assay_interpretation_conditions interpretation_conditions extra_details""".split(),
    "comparison_role": """molecule_role tested_entity_role tested_entity_form molecular_form
 chemical_entity_type agent_type agent_category carcinogenic_role transport_mechanism
 comparator comparator_exposure perturbation immune_stimulus_regimen injury_or_growth_context
 treatment_and_comparator_context probe_or_analyte mechanistic_entities_and_components reaction_type""".split(),
}

SCIENTIFIC_FIELDS = frozenset(f for fields in GROUP_FIELDS.values() for f in fields)

CALL_FIELDS = """result_label outcome_label effect_direction result_direction result_call result_status
 result_interpretation mutagenicity_result carcinogenicity_conclusion causal_status
 bbb_permeability_label bbb_transport_label passive_bbb_interpretation interaction_conclusion
 substrate_status""".split()

MISSING = re.compile(
    r"_*(?:null|none|nan|unknown|unspecified|not[ _]reported|not[ _]applicable|not[ _]available|n/?a|missing)_*",
    re.I,
)


def clean(value: Any) -> str:
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return ""
    text = str(value).strip()
    return "" if not text or MISSING.fullmatch(text) else text


def selection_text(value: Any) -> str:
    """Normalize missing aggregate placeholders for selection, not cached features."""
    text = clean(value)
    without_count = re.sub(r"\s*\(\d+\)$", "", text)
    if not clean(without_count) or without_count.lower() in {
        "not specified",
        "not_specified",
        "not stated",
        "not_stated",
        "-",
        "--",
        "—",
    }:
        return ""
    return text


def normalize_endpoint(value: str) -> str:
    """Conservative spelling aliases, not an inferred biological ontology."""
    name = re.sub(
        r"[\s_\-]+", " ", unicodedata.normalize("NFKC", value).lower()
    ).strip()
    aliases = {
        "ames": "bacterial reverse mutation",
        "ames test": "bacterial reverse mutation",
        "ames mutagenicity": "bacterial reverse mutation",
        "bacterial reverse mutation (ames)": "bacterial reverse mutation",
        "bacterial reverse mutation (ames test)": "bacterial reverse mutation",
        "blood brain barrier permeability": "bbb permeability",
        "bbb permeability outcome": "bbb permeability",
    }
    return aliases.get(name, name)


def native_source(row: Mapping[str, Any], raw: Mapping[str, Any] | None = None) -> dict:
    """Reviewed JSON wins even when it contains explicit null corrections."""
    native = dict(raw or {})
    for field in ("raw_record_json", "reviewed_record_json"):
        value = row.get(field)
        if value:
            native.update(json.loads(value) if isinstance(value, str) else value)
    for key in (
        SCIENTIFIC_FIELDS
        | IDENTITY_FIELDS
        | {
            "support_text",
            "molecule_name",
            "needs_more_context",
        }
    ):
        if key in row and (
            key.startswith("canonical_") or key == "support_text" or key not in native
        ):
            native[key] = row[key]
    return native


def _text(fields, names):
    return " | ".join(
        value for k in names if (value := selection_text(fields.get(k)))
    ).lower()


_DILI_CONDITION_FIELDS = (
    "age",
    "age_group",
    "exposure",
    "population_context",
    "regimen",
    "genotype",
    "co_treatment",
)

_DILI_SYSTEM_FIELDS = (
    "canonical_species_context",
    "species",
    "species_or_population",
    "biological_system",
    "biological_test_system",
    "biological_context",
    "biological_model",
    "biological_model_context",
    "biological_system_context",
    "test_system",
    "experimental_system",
    "evidence_population_or_model",
)
