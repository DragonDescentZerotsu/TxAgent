"""BBB-specific vocabulary and decision contract for progressive inference."""

from predict.harnesses.progressive.state import ProgressiveTaskContract
from predict.tasks.bbb_martins.prompts import (
    DEFAULT_BBB_PROMPT_PROFILE,
    get_bbb_prompt_profile,
)


PROGRESSIVE_ASSAY_LEVEL_DESCRIPTIONS = {
    1: "Direct measured brain or CNS exposure outcomes; closest to the benchmark label.",
    2: "Passive permeability evidence; indirect and conditional on structural and assay transferability.",
    3: "Efflux-transporter evidence; indirect and directional, with substrate and inhibition claims kept distinct.",
    4: "Influx or uptake-transporter evidence; indirect and dependent on transporter context.",
}

PROGRESSIVE_ASSAY_ENDPOINT_DESCRIPTIONS = {
    "direct_brain_exposure": (
        "Direct measured brain or CNS exposure outcomes eligible for the BBB outcome layer."
    ),
    "central_functional_access_proxy": (
        "Near-direct systemic CNS pharmacodynamic, functional, biomarker, adverse-effect, "
        "or efficacy observations that imply access without directly measuring CNS exposure."
    ),
    "passive_permeability": (
        "PAMPA or epithelial-cell permeability evidence, including predicted passive "
        "permeability readouts retained with their original uncertainty."
    ),
    "efflux_transport": (
        "Efflux-transporter evidence; substrate, non-substrate, inhibition, and directional "
        "permeability readouts remain distinct."
    ),
    "influx_transport": (
        "Influx or uptake-transporter evidence dependent on transporter and assay context."
    ),
}


def get_progressive_task_contract() -> ProgressiveTaskContract:
    profile = get_bbb_prompt_profile(DEFAULT_BBB_PROMPT_PROFILE)
    instructions = tuple(
        instruction
        for instruction in profile.final_instructions
        if "If you recognize the molecule" not in instruction
    )
    return ProgressiveTaskContract(
        task="bbb_martins",
        endpoint_name="meaningful or adequate CNS access after systemic administration",
        label_scope=profile.label_scope,
        prediction_field="bbb_prediction",
        positive_prediction="pass",
        negative_prediction="fail",
        system_role="You are a senior medicinal-chemistry and CNS-exposure reasoning model.",
        task_instructions=instructions,
    )
