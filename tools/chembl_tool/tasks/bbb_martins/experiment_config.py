"""Paper-facing BBB retrieval views."""

from tools.chembl_tool.common.experiment_retrieval import (
    SourceExperimentConfig,
    evidence_family,
)
from tools.chembl_tool.common.progressive_assay_reasoning import ProgressiveTaskContract
from tools.chembl_tool.tasks.bbb_martins.prompt_profiles import (
    DEFAULT_BBB_PROMPT_PROFILE,
    get_bbb_prompt_profile,
)


CHEMBL = SourceExperimentConfig(
    source_name="chembl",
    direct_groups=(
        evidence_family(
            "direct_bbb",
            family_label="Tier 1",
            source_group_prefixes=("Tier 1.",),
            exclude_source_group_ids=("Tier 1.context_dependent",),
            legacy_output_group_id="Direct.bbb",
        ),
    ),
    mechanism_groups=tuple(
        evidence_family(
            {
                1: "direct_brain_exposure",
                2: "passive_permeability",
                3: "efflux_transport",
                4: "influx_transport",
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
            "direct_bbb",
            family_label="Tier 1",
            source_group_ids=("Tier 1.starling_direct_bbb_evidence",),
            legacy_output_group_id="Direct.bbb",
        ),
    ),
    mechanism_groups=(
        evidence_family(
            "direct_brain_exposure",
            family_label="Tier 1",
            source_group_ids=("Tier 1.starling_direct_bbb_evidence",),
            legacy_output_group_id="Mechanism.tier_1",
        ),
        evidence_family(
            "passive_permeability",
            family_label="Tier 2",
            source_group_ids=("Mechanism.passive_permeability",),
            legacy_output_group_id="Mechanism.tier_2",
        ),
        evidence_family(
            "efflux_transport",
            family_label="Tier 3",
            source_group_ids=("Mechanism.efflux_transport",),
            legacy_output_group_id="Mechanism.tier_3",
        ),
        evidence_family(
            "influx_transport",
            family_label="Tier 4",
            source_group_ids=("Mechanism.influx_transport",),
            legacy_output_group_id="Mechanism.tier_4",
        ),
    ),
)

# Parallel retrieval-only source-purity view.  Keep ``STARLING`` unchanged so
# historical four-level artifacts remain exactly reproducible.
STARLING_SOURCE_PURITY = SourceExperimentConfig(
    source_name="starling_source_purity",
    direct_groups=STARLING.direct_groups,
    mechanism_groups=(
        evidence_family(
            "direct_brain_exposure",
            family_label="Direct measured CNS access",
            source_group_ids=("Tier 1.starling_direct_bbb_evidence",),
            legacy_output_group_id="Direct.measured_cns_access",
        ),
        evidence_family(
            "central_functional_access_proxy",
            family_label="Near-direct functional CNS proxy",
            source_group_ids=("Proxy.central_functional_access",),
        ),
        evidence_family(
            "passive_permeability",
            family_label="Passive permeability",
            source_group_ids=("Mechanism.passive_permeability",),
        ),
        evidence_family(
            "efflux_transport",
            family_label="Efflux transport",
            source_group_ids=("Mechanism.efflux_transport",),
        ),
        evidence_family(
            "influx_transport",
            family_label="Influx transport",
            source_group_ids=("Mechanism.influx_transport",),
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
    2: (
        "Near-direct systemic CNS functional, pharmacodynamic, biomarker, adverse-effect, "
        "or efficacy observations that imply access without directly measuring CNS exposure."
    ),
    3: "Passive permeability evidence; indirect and conditional on structural and assay transferability.",
    4: "Efflux-transporter evidence; indirect and directional, with substrate and inhibition claims kept distinct.",
    5: "Influx or uptake-transporter evidence; indirect and dependent on transporter context.",
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
        # The shared schema/protocol owns formatting, labels and prior handling.
        and not instruction.startswith((
            "Return compact complete JSON.", "Use bbb_prediction=",
            "Use the single-molecule analysis as", "Use group analyses as analog evidence;",
            "You must choose exactly one",
        ))
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
