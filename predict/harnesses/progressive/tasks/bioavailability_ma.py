"""Bioavailability-specific vocabulary and contract for progressive inference."""

from predict.harnesses.progressive.state import ProgressiveTaskContract
from predict.tasks.bioavailability_ma.prompts import (
    DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE,
    get_bioavailability_prompt_profile,
)


PROGRESSIVE_ASSAY_LEVEL_DESCRIPTIONS = {
    1: "Direct absolute oral bioavailability outcomes; closest to the F >= 20% benchmark label.",
    2: "Nondirect oral-bioavailability evidence. This is indirect evidence even when its wording resembles the label.",
    3: "Oral AUC or Cmax exposure evidence; indirect because exposure also depends on dose, formulation, clearance, and sampling.",
    4: "Absorption, solubility, dissolution, or permeability evidence contributing to the absorbed fraction.",
    5: "Gut-wall efflux and intestinal metabolism evidence contributing to presystemic loss.",
    6: "Hepatic clearance and metabolic-stability evidence contributing to systemic availability.",
}

PROGRESSIVE_ASSAY_ENDPOINT_DESCRIPTIONS: dict[str, str] = {}


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
