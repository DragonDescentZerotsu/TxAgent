"""Paper-facing BBB retrieval views."""

from tools.chembl_tool.common.experiment_retrieval import EvidenceGroupSpec, SourceExperimentConfig
from tools.chembl_tool.common.progressive_assay_reasoning import ProgressiveTaskContract
from tools.chembl_tool.tasks.bbb_martins.prompt_profiles import (
    DEFAULT_BBB_PROMPT_PROFILE,
    get_bbb_prompt_profile,
)


CHEMBL = SourceExperimentConfig(
    source_name="chembl",
    direct_groups=(
        EvidenceGroupSpec(
            "Direct.bbb",
            "Tier 1",
            "direct_bbb",
            source_group_prefixes=("Tier 1.",),
            exclude_source_groups=("Tier 1.context_dependent",),
        ),
    ),
    mechanism_groups=tuple(
        EvidenceGroupSpec(
            f"Mechanism.tier_{tier}",
            f"Tier {tier}",
            {
                1: "direct_brain_exposure",
                2: "passive_permeability",
                3: "efflux_transport",
                4: "influx_transport",
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
            "Direct.bbb",
            "Tier 1",
            "direct_bbb",
            source_groups=("Tier 1.starling_direct_bbb_evidence",),
        ),
    ),
    mechanism_groups=(
        EvidenceGroupSpec(
            "Mechanism.tier_1",
            "Tier 1",
            "direct_brain_exposure",
            source_groups=("Tier 1.starling_direct_bbb_evidence",),
        ),
        EvidenceGroupSpec(
            "Mechanism.tier_2",
            "Tier 2",
            "passive_permeability",
            source_groups=("Mechanism.passive_permeability",),
        ),
        EvidenceGroupSpec(
            "Mechanism.tier_3",
            "Tier 3",
            "efflux_transport",
            source_groups=("Mechanism.efflux_transport",),
        ),
        EvidenceGroupSpec(
            "Mechanism.tier_4",
            "Tier 4",
            "influx_transport",
            source_groups=("Mechanism.influx_transport",),
        ),
    ),
)

# Parallel retrieval-only source-purity view.  Keep ``STARLING`` unchanged so
# historical four-level artifacts remain exactly reproducible.
STARLING_SOURCE_PURITY = SourceExperimentConfig(
    source_name="starling_source_purity",
    direct_groups=STARLING.direct_groups,
    mechanism_groups=(
        EvidenceGroupSpec(
            "Direct.measured_cns_access",
            "Direct measured CNS access",
            "direct_brain_exposure",
            source_groups=("Tier 1.starling_direct_bbb_evidence",),
        ),
        EvidenceGroupSpec(
            "Proxy.central_functional_access",
            "Near-direct functional CNS proxy",
            "central_functional_access_proxy",
            source_groups=("Proxy.central_functional_access",),
        ),
        EvidenceGroupSpec(
            "Mechanism.passive_permeability",
            "Passive permeability",
            "passive_permeability",
            source_groups=("Mechanism.passive_permeability",),
        ),
        EvidenceGroupSpec(
            "Mechanism.efflux_transport",
            "Efflux transport",
            "efflux_transport",
            source_groups=("Mechanism.efflux_transport",),
        ),
        EvidenceGroupSpec(
            "Mechanism.influx_transport",
            "Influx transport",
            "influx_transport",
            source_groups=("Mechanism.influx_transport",),
        ),
    ),
)

SOURCES = {
    "chembl": CHEMBL,
    "starling": STARLING,
    "starling_source_purity": STARLING_SOURCE_PURITY,
}

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


def get_source_config(source: str) -> SourceExperimentConfig:
    return SOURCES[source]


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
