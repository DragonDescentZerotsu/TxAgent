"""Paper-facing Bioavailability_Ma retrieval views."""

from tools.chembl_tool.common.experiment_retrieval import EvidenceGroupSpec, SourceExperimentConfig
from tools.chembl_tool.common.progressive_assay_reasoning import ProgressiveTaskContract
from tools.chembl_tool.tasks.bioavailability_ma.prompt_profiles import (
    DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE,
    get_bioavailability_prompt_profile,
)


DIRECT_ORAL_BIOAVAILABILITY_GROUP = "Observed.direct_oral_bioavailability"
NONDIRECT_ORAL_BIOAVAILABILITY_GROUP = "Observed.nondirect_oral_bioavailability"


CHEMBL = SourceExperimentConfig(
    source_name="chembl",
    direct_groups=(
        EvidenceGroupSpec(
            "Observed.direct_oral_bioavailability",
            "Observed",
            "direct_oral_bioavailability",
            source_groups=("Tier 1.direct_absolute_bioavailability",),
        ),
    ),
    mechanism_groups=(
        EvidenceGroupSpec(
            "Observed.direct_oral_bioavailability",
            "Observed",
            "direct_oral_bioavailability",
            source_groups=("Tier 1.direct_absolute_bioavailability",),
        ),
        EvidenceGroupSpec(
            "Observed.oral_auc_cmax_exposure",
            "Observed",
            "oral_auc_cmax_exposure",
            source_groups=(
                "Tier 2.oral_auc_exposure",
                "Tier 2.oral_cmax_exposure",
                "Tier 6.food_effect_or_fed_fasted",
                "Tier 6.relative_bioavailability_or_formulation",
            ),
        ),
        EvidenceGroupSpec(
            "Fa.absorption_solubility_permeability",
            "Fa",
            "absorption_solubility_permeability",
            source_groups=(
                "Tier 2.absorption_fraction_or_hia",
                "Tier 2.in_vivo_intestinal_permeability",
                "Tier 3.cell_permeability_papp",
                "Tier 3.pampa_or_artificial_membrane",
                "Tier 4.dissolution",
                "Tier 4.gi_or_chemical_stability",
                "Tier 4.solubility",
            ),
        ),
        EvidenceGroupSpec(
            "Fg.gut_wall_efflux_intestinal_metabolism",
            "Fg",
            "gut_wall_efflux_intestinal_metabolism",
            source_groups=(
                "Tier 3.cell_bidirectional_efflux_ratio",
                "Tier 3.cell_secretory_permeability",
                "Tier 3.transporter_inhibition_or_binding",
                "Tier 3.transporter_substrate_or_efflux",
            ),
        ),
        EvidenceGroupSpec(
            "Fh.hepatic_clearance_metabolic_stability",
            "Fh",
            "hepatic_clearance_metabolic_stability",
            source_groups=(
                "Tier 5.first_pass_or_extraction",
                "Tier 5.intrinsic_or_hepatic_clearance",
                "Tier 5.metabolic_stability",
            ),
        ),
    ),
)

STARLING = SourceExperimentConfig(
    source_name="starling",
    direct_groups=(
        EvidenceGroupSpec(
            "Observed.direct_oral_bioavailability",
            "Observed",
            "direct_oral_bioavailability",
            source_groups=("Observed.direct_oral_bioavailability",),
        ),
    ),
    mechanism_groups=tuple(
        EvidenceGroupSpec(
            group_id,
            group_id.split(".", 1)[0],
            group_id.split(".", 1)[1],
            source_groups=(group_id,),
        )
        for group_id in (
            "Observed.direct_oral_bioavailability",
            "Observed.nondirect_oral_bioavailability",
            "Observed.oral_auc_cmax_exposure",
            "Fa.absorption_solubility_permeability",
            "Fg.gut_wall_efflux_intestinal_metabolism",
            "Fh.hepatic_clearance_metabolic_stability",
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
        task_instructions=tuple(profile.final_instructions),
    )
