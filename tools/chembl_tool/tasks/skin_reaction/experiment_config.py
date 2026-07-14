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

SOURCES = {"chembl": CHEMBL}


def get_source_config(source: str) -> SourceExperimentConfig:
    return SOURCES[source]
