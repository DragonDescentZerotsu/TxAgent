"""Paper-facing Bioavailability_Ma retrieval views."""

from tools.chembl_tool.common.experiment_retrieval import (
    SourceExperimentConfig,
    evidence_family,
)
from tools.chembl_tool.common.progressive_assay_reasoning import ProgressiveTaskContract
from tools.chembl_tool.tasks.bioavailability_ma.prompt_profiles import (
    DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE,
    get_bioavailability_prompt_profile,
)


CHEMBL = SourceExperimentConfig(
    source_name="chembl",
    direct_groups=(
        evidence_family(
            "direct_oral_bioavailability",
            family_label="Observed",
            source_group_ids=("Tier 1.direct_absolute_bioavailability",),
            legacy_output_group_id="Observed.direct_oral_bioavailability",
        ),
    ),
    mechanism_groups=(
        evidence_family(
            "direct_oral_bioavailability",
            family_label="Observed",
            source_group_ids=("Tier 1.direct_absolute_bioavailability",),
            legacy_output_group_id="Observed.direct_oral_bioavailability",
        ),
        evidence_family(
            "oral_auc_cmax_exposure",
            family_label="Observed",
            source_group_ids=(
                "Tier 2.oral_auc_exposure",
                "Tier 2.oral_cmax_exposure",
                "Tier 6.food_effect_or_fed_fasted",
                "Tier 6.relative_bioavailability_or_formulation",
            ),
            legacy_output_group_id="Observed.oral_auc_cmax_exposure",
        ),
        evidence_family(
            "absorption_solubility_permeability",
            family_label="Fa",
            source_group_ids=(
                "Tier 2.absorption_fraction_or_hia",
                "Tier 2.in_vivo_intestinal_permeability",
                "Tier 3.cell_permeability_papp",
                "Tier 3.pampa_or_artificial_membrane",
                "Tier 4.dissolution",
                "Tier 4.gi_or_chemical_stability",
                "Tier 4.solubility",
            ),
            legacy_output_group_id="Fa.absorption_solubility_permeability",
        ),
        evidence_family(
            "gut_wall_efflux_intestinal_metabolism",
            family_label="Fg",
            source_group_ids=(
                "Tier 3.cell_bidirectional_efflux_ratio",
                "Tier 3.cell_secretory_permeability",
                "Tier 3.transporter_inhibition_or_binding",
                "Tier 3.transporter_substrate_or_efflux",
            ),
            legacy_output_group_id="Fg.gut_wall_efflux_intestinal_metabolism",
        ),
        evidence_family(
            "hepatic_clearance_metabolic_stability",
            family_label="Fh",
            source_group_ids=(
                "Tier 5.first_pass_or_extraction",
                "Tier 5.intrinsic_or_hepatic_clearance",
                "Tier 5.metabolic_stability",
            ),
            legacy_output_group_id="Fh.hepatic_clearance_metabolic_stability",
        ),
    ),
)

STARLING = SourceExperimentConfig(
    source_name="starling",
    direct_groups=(
        evidence_family(
            "direct_oral_bioavailability",
            family_label="Observed",
            source_group_ids=("Observed.direct_oral_bioavailability",),
        ),
    ),
    mechanism_groups=tuple(
        evidence_family(
            family_key,
            family_label=family_label,
            source_group_ids=(source_group_id,),
        )
        for source_group_id, family_key, family_label in (
            (
                "Observed.direct_oral_bioavailability",
                "direct_oral_bioavailability",
                "Observed",
            ),
            (
                "Observed.nondirect_oral_bioavailability",
                "nondirect_oral_bioavailability",
                "Observed",
            ),
            (
                "Observed.oral_auc_cmax_exposure",
                "oral_auc_cmax_exposure",
                "Observed",
            ),
            (
                "Fa.absorption_solubility_permeability",
                "absorption_solubility_permeability",
                "Fa",
            ),
            (
                "Fg.gut_wall_efflux_intestinal_metabolism",
                "gut_wall_efflux_intestinal_metabolism",
                "Fg",
            ),
            (
                "Fh.hepatic_clearance_metabolic_stability",
                "hepatic_clearance_metabolic_stability",
                "Fh",
            ),
        )
    ),
)

SOURCES = {"chembl": CHEMBL, "starling": STARLING}

PROGRESSIVE_ASSAY_LEVEL_DESCRIPTIONS = {
    1: "Direct absolute oral bioavailability outcomes; closest to the F >= 20% benchmark label.",
    2: "Nondirect oral-bioavailability evidence. This is indirect evidence even when its wording resembles the label.",
    3: "Oral AUC or Cmax exposure evidence; indirect because exposure also depends on dose, formulation, clearance, and sampling.",
    4: "Absorption, solubility, dissolution, or permeability evidence contributing to the absorbed fraction.",
    5: "Gut-wall efflux and intestinal metabolism evidence contributing to presystemic loss.",
    6: "Hepatic clearance and metabolic-stability evidence contributing to systemic availability.",
}


def get_source_config(source: str) -> SourceExperimentConfig:
    return SOURCES[source]


def get_progressive_task_contract() -> ProgressiveTaskContract:
    profile = get_bioavailability_prompt_profile(DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE)
    return ProgressiveTaskContract(
        task="bioavailability_ma",
        endpoint_name="absolute oral bioavailability at the F >= 20% threshold",
        label_scope=profile.label_scope,
        prediction_field="bioavailability_prediction",
        positive_prediction="high",
        negative_prediction="low",
        system_role=profile.final_system_role,
        task_instructions=tuple(
            instruction.replace("For every group,", "For the supplied evidence,")
            for instruction in profile.final_instructions
            if not instruction.startswith("You must choose exactly one")
        ),
    )
