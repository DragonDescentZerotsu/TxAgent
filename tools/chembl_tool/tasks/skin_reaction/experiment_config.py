"""Paper-facing Skin_Reaction retrieval views."""

from tools.chembl_tool.common.experiment_retrieval import EvidenceGroupSpec, SourceExperimentConfig


CHEMBL = SourceExperimentConfig(
    source_name="chembl",
    direct_groups=(
        EvidenceGroupSpec(
            "Direct.skin_reaction",
            "Tier 1",
            "direct_skin_reaction",
            source_group_prefixes=("Tier 1.",),
            exclude_source_groups=("Tier 1.context_dependent",),
        ),
    ),
    mechanism_groups=tuple(
        EvidenceGroupSpec(
            f"Mechanism.tier_{tier}",
            f"Tier {tier}",
            {
                1: "direct_skin_reaction",
                2: "sensitisation_aop",
                3: "phototoxicity_irritation_local_damage",
                4: "skin_exposure",
            }[tier],
            source_group_prefixes=(f"Tier {tier}.",),
        )
        for tier in range(1, 5)
    ),
)

STARLING = SourceExperimentConfig(
    source_name="starling",
    direct_groups=(
        EvidenceGroupSpec(
            "Direct.skin_reaction",
            "Tier 1",
            "direct_skin_reaction",
            source_groups=("Direct.skin_reaction",),
        ),
    ),
    mechanism_groups=(
        EvidenceGroupSpec(
            "Mechanism.tier_1",
            "Tier 1",
            "direct_skin_reaction",
            source_groups=("Direct.skin_reaction",),
        ),
        EvidenceGroupSpec(
            "Mechanism.tier_2",
            "Tier 2",
            "sensitisation_aop",
            source_groups=("Mechanism.sensitization_aop",),
        ),
        EvidenceGroupSpec(
            "Mechanism.tier_3",
            "Tier 3",
            "phototoxicity_irritation_local_damage",
            source_groups=("Mechanism.phototoxicity_irritation_local_damage",),
        ),
        EvidenceGroupSpec(
            "Mechanism.tier_4",
            "Tier 4",
            "skin_exposure",
            source_groups=("Mechanism.skin_exposure",),
        ),
    ),
)

SOURCES = {"chembl": CHEMBL, "starling": STARLING}


def get_source_config(source: str) -> SourceExperimentConfig:
    return SOURCES[source]
