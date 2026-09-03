"""Paper-facing Bioavailability_Ma retrieval views."""

from predict.harnesses.branches.retrieval import BranchDefinition, BranchRetrievalConfig


DIRECT_ORAL_BIOAVAILABILITY_GROUP = "Observed.direct_oral_bioavailability"
NONDIRECT_ORAL_BIOAVAILABILITY_GROUP = "Observed.nondirect_oral_bioavailability"


CHEMBL = BranchRetrievalConfig(
    source_name="chembl",
    direct_groups=(
        BranchDefinition(
            "Observed.direct_oral_bioavailability",
            "Observed",
            "direct_oral_bioavailability",
            source_groups=("Tier 1.direct_absolute_bioavailability",),
        ),
    ),
    mechanism_groups=(
        BranchDefinition(
            "Observed.direct_oral_bioavailability",
            "Observed",
            "direct_oral_bioavailability",
            source_groups=("Tier 1.direct_absolute_bioavailability",),
        ),
        BranchDefinition(
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
        BranchDefinition(
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
        BranchDefinition(
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
        BranchDefinition(
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

STARLING = BranchRetrievalConfig(
    source_name="starling",
    direct_groups=(
        BranchDefinition(
            "Observed.direct_oral_bioavailability",
            "Observed",
            "direct_oral_bioavailability",
            source_groups=("Observed.direct_oral_bioavailability",),
        ),
    ),
    mechanism_groups=tuple(
        BranchDefinition(
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

def get_source_config(source: str) -> BranchRetrievalConfig:
    return SOURCES[source]
