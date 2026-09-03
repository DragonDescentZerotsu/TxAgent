"""Skin-specific vocabulary and decision contract for progressive inference."""

from predict.harnesses.progressive.state import ProgressiveTaskContract
from predict.tasks.skin_reaction.prompts import (
    DEFAULT_SKIN_PROMPT_PROFILE,
    get_skin_prompt_profile,
)


PROGRESSIVE_ASSAY_LEVEL_DESCRIPTIONS = {
    1: "Direct validated skin-sensitization or contact-allergy outcomes; closest to the benchmark label.",
    2: "Sensitization AOP key-event evidence; indirect mechanism support and not itself a final sensitizer label.",
}

PROGRESSIVE_ASSAY_ENDPOINT_DESCRIPTIONS: dict[str, str] = {}


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
