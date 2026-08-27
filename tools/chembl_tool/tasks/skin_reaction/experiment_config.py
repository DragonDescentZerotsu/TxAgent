"""Paper-facing Skin_Reaction retrieval views."""

from tools.chembl_tool.common.experiment_retrieval import EvidenceGroupSpec, SourceExperimentConfig
from tools.chembl_tool.common.progressive_assay_reasoning import ProgressiveTaskContract
from tools.chembl_tool.tasks.skin_reaction.prompt_profiles import (
    DEFAULT_SKIN_PROMPT_PROFILE,
    get_skin_prompt_profile,
)


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
    ),
)

SOURCES = {"chembl": CHEMBL, "starling": STARLING}

PROGRESSIVE_ASSAY_LEVEL_DESCRIPTIONS = {
    1: "Direct validated skin-sensitization or contact-allergy outcomes; closest to the benchmark label.",
    2: "Sensitization AOP key-event evidence; indirect mechanism support and not itself a final sensitizer label.",
}


def get_source_config(source: str) -> SourceExperimentConfig:
    return SOURCES[source]


def get_progressive_task_contract() -> ProgressiveTaskContract:
    profile = get_skin_prompt_profile(DEFAULT_SKIN_PROMPT_PROFILE)
    instructions = tuple(
        (
            "Express uncertainty through confidence, conflicts, caveats, and evidence_gaps while "
            "still choosing exactly one prediction."
            if "supplied anonymous query" in instruction
            else instruction
        )
        for instruction in profile.final_instructions
    )
    return ProgressiveTaskContract(
        task="skin_reaction",
        endpoint_name="skin sensitization or contact allergy",
        label_scope=profile.label_scope,
        prediction_field="skin_reaction_prediction",
        positive_prediction="risk",
        negative_prediction="no_risk",
        system_role=profile.final_system_role,
        task_instructions=instructions,
    )
