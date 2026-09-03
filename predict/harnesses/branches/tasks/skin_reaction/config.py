"""Paper-facing Skin_Reaction retrieval views."""

from predict.harnesses.branches.retrieval import BranchDefinition, BranchRetrievalConfig


CHEMBL = BranchRetrievalConfig(
    source_name="chembl",
    direct_groups=(
        BranchDefinition(
            "Direct.skin_reaction",
            "Tier 1",
            "direct_skin_reaction",
            source_group_prefixes=("Tier 1.",),
            exclude_source_groups=("Tier 1.context_dependent",),
        ),
    ),
    mechanism_groups=tuple(
        BranchDefinition(
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

STARLING = BranchRetrievalConfig(
    source_name="starling",
    direct_groups=(
        BranchDefinition(
            "Direct.skin_reaction",
            "Tier 1",
            "direct_skin_reaction",
            source_groups=("Direct.skin_reaction",),
        ),
    ),
    mechanism_groups=(
        BranchDefinition(
            "Mechanism.tier_1",
            "Tier 1",
            "direct_skin_reaction",
            source_groups=("Direct.skin_reaction",),
        ),
        BranchDefinition(
            "Mechanism.tier_2",
            "Tier 2",
            "sensitisation_aop",
            source_groups=("Mechanism.sensitization_aop",),
        ),
    ),
)

SOURCES = {"chembl": CHEMBL, "starling": STARLING}

def get_source_config(source: str) -> BranchRetrievalConfig:
    return SOURCES[source]
