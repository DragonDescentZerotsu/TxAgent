"""Paper-facing Bioavailability_Ma retrieval views."""

from tools.chembl_tool.common.experiment_retrieval import EvidenceGroupSpec, SourceExperimentConfig


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
            "Observed.oral_auc_cmax_exposure",
            "Fa.absorption_solubility_permeability",
            "Fg.gut_wall_efflux_intestinal_metabolism",
            "Fh.hepatic_clearance_metabolic_stability",
        )
    ),
)

# In-distribution retrieval over the starling normalized (hf_cleaned) molecules. The
# neighbor index already keys on the five paper group_ids, so the group mapping is
# identical to STARLING; only the source name (and the index/catalog paths) differ.
STARLING_IN_DISTRIBUTION = SourceExperimentConfig(
    source_name="starling_in_distribution",
    direct_groups=STARLING.direct_groups,
    mechanism_groups=STARLING.mechanism_groups,
)

# Retrieval over the policy-decoupled starling_normalized_v5 evidence library. Its
# neighbor index is re-aggregated to the same five paper group_ids as the legacy
# factor library, so the group mapping is identical to STARLING; only the source
# name (and the index/evidence paths) differ.
STARLING_V5 = SourceExperimentConfig(
    source_name="starling_v5",
    direct_groups=STARLING.direct_groups,
    mechanism_groups=STARLING.mechanism_groups,
)

# Compact normalized-v6 uses a relational Parquet/NPZ index.  Its loader
# hydrates representative evidence from finalized record references once at
# startup, while preserving the same five paper-facing groups.
STARLING_V6 = SourceExperimentConfig(
    source_name="starling_v6",
    direct_groups=STARLING.direct_groups,
    mechanism_groups=STARLING.mechanism_groups,
)

SOURCES = {
    "chembl": CHEMBL,
    "starling": STARLING,
    "starling_in_distribution": STARLING_IN_DISTRIBUTION,
    "starling_v5": STARLING_V5,
    "starling_v6": STARLING_V6,
}

# Retrieval sources whose molecules come from the starling assay-transfer data and are
# therefore eligible for assay-transfer scoring.
STARLING_RETRIEVAL_SOURCES = frozenset({"starling", "starling_in_distribution"})


def get_source_config(source: str) -> SourceExperimentConfig:
    return SOURCES[source]
