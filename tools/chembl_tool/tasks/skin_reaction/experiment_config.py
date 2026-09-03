"""Paper-facing Skin_Reaction retrieval views."""

from tools.chembl_tool.common.experiment_retrieval import (
    SourceExperimentConfig,
    evidence_family,
)
from tools.chembl_tool.common.progressive_assay_reasoning import ProgressiveTaskContract
from tools.chembl_tool.tasks.skin_reaction.prompt_profiles import (
    DEFAULT_SKIN_PROMPT_PROFILE,
    get_skin_prompt_profile,
)


CHEMBL = SourceExperimentConfig(
    source_name="chembl",
    direct_groups=(
        evidence_family(
            "direct_skin_reaction",
            family_label="Tier 1",
            source_group_prefixes=("Tier 1.",),
            exclude_source_group_ids=("Tier 1.context_dependent",),
            legacy_output_group_id="Direct.skin_reaction",
        ),
    ),
    mechanism_groups=tuple(
        evidence_family(
            {
                1: "direct_skin_reaction",
                2: "sensitisation_aop",
                3: "phototoxicity_irritation_local_damage",
                4: "skin_exposure",
            }[tier],
            family_label=f"Tier {tier}",
            source_group_prefixes=(f"Tier {tier}.",),
            legacy_output_group_id=f"Mechanism.tier_{tier}",
        )
        for tier in range(1, 5)
    ),
)

STARLING = SourceExperimentConfig(
    source_name="starling",
    direct_groups=(
        evidence_family(
            "direct_skin_reaction",
            family_label="Tier 1",
            source_group_ids=("Direct.skin_reaction",),
        ),
    ),
    mechanism_groups=(
        evidence_family(
            "direct_skin_reaction",
            family_label="Tier 1",
            source_group_ids=("Direct.skin_reaction",),
            legacy_output_group_id="Mechanism.tier_1",
        ),
        evidence_family(
            "nonvoter_sensitization_outcome",
            family_label="Tier 2",
            source_group_ids=("Observed.nonvoter_skin_outcome",),
            legacy_output_group_id="Mechanism.tier_2",
        ),
        evidence_family(
            "sensitisation_aop",
            family_label="Tier 3",
            source_group_ids=("Mechanism.sensitization_aop",),
            legacy_output_group_id="Mechanism.tier_3",
        ),
    ),
)

SOURCES = {"chembl": CHEMBL, "starling": STARLING}

PROGRESSIVE_ASSAY_LEVEL_DESCRIPTIONS = {
    1: "Actual source records that participated in frozen skin-label voting.",
    2: "Observed or direct-like skin outcomes that did not participate in voting.",
    3: "Sensitization AOP key-event evidence; indirect mechanism support and not itself a final sensitizer label.",
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
