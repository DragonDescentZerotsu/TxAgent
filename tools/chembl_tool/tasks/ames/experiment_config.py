"""Ames record families and the condition-specific bacterial outcome contract."""

from tools.chembl_tool.common.experiment_retrieval import (
    SourceExperimentConfig,
    evidence_family,
)
from tools.chembl_tool.common.progressive_assay_reasoning import ProgressiveTaskContract
from tools.chembl_tool.tasks.ames.source_contract import (
    DAMAGE,
    DIRECT,
    MECHANISM,
    MUTATION,
    NEAR,
    VERSION,
)


STARLING = SourceExperimentConfig(
    source_name="starling",
    direct_groups=(evidence_family("direct_ames", source_group_ids=(DIRECT,)),),
    mechanism_groups=(
        evidence_family("direct_ames", source_group_ids=(DIRECT,)),
        evidence_family("nonvoter_ames_outcome", source_group_ids=(NEAR,)),
        evidence_family("other_genetic_damage", source_group_ids=(MUTATION,)),
        evidence_family("dna_damage_response", source_group_ids=(DAMAGE,)),
        evidence_family("genotoxicity_mechanisms", source_group_ids=(MECHANISM,)),
    ),
)

SOURCES = {"starling": STARLING}

PROGRESSIVE_ASSAY_LEVEL_DESCRIPTIONS = {
    1: (
        "Actual source records that participated in Ames gold voting under the "
        "exact reported strain panel and metabolic activation condition."
    ),
    2: (
        "Nonvoter bacterial outcomes and predictions, unspecified mutagenicity, "
        "and mixed passages carrying direct-related content."
    ),
    3: "Other gene/chromosome mutation and cytogenetic damage evidence.",
    4: "DNA damage, repair and damage-response evidence, including modifier effects.",
    5: (
        "Genotoxicity mechanisms: bioactivation, detoxification, DNA interaction, "
        "redox/antioxidant defense and chromosome-segregation mechanisms."
    ),
}


def get_source_config(source: str) -> SourceExperimentConfig:
    return SOURCES[source]


def get_progressive_task_contract() -> ProgressiveTaskContract:
    return ProgressiveTaskContract(
        task="ames",
        endpoint_name=(
            "bacterial reverse mutation under the exact reported strain panel "
            "and metabolic activation condition"
        ),
        label_scope=VERSION,
        prediction_field="ames_prediction",
        positive_prediction="positive",
        negative_prediction="negative",
        system_role="Assess condition-specific bacterial reverse-mutation evidence.",
        task_instructions=(
            "Predict the bacterial reverse-mutation result only under the query's "
            "exact reported strain panel and metabolic activation condition; "
            "do not invent missing strains or activation conditions.",
            "A negative result for one strain or a limited panel must not imply "
            "a global negative across other strains or activation conditions.",
            "A panel-positive label means at least one reported positive response "
            "within that exact panel. both_reported is one pooled activation "
            "summary; it does not assert the same result in each activation arm.",
            "Other genetic-damage endpoints, DNA damage and genotoxicity mechanisms "
            "may support, contradict, or overturn a prior Ames prediction when the "
            "evidence is sufficiently strong and transferable to the query's "
            "structure, strain panel and activation condition. New direct Ames "
            "data are not required. Explain the mechanistic link and its limitations; "
            "distinguish the resulting prediction from a measured Ames outcome.",
            "Nonvoter bacterial outcomes retain their source qualifications; "
            "their family membership does not establish a gold-voting result.",
            "Predictions, missing context and mixed or equivocal directions remain "
            "qualified evidence at their endpoint family. Do not treat predictions "
            "as measured outcomes or invent missing experimental details.",
            "Assay application, protocol and analytical records may describe what "
            "was tested without reporting an effect; do not infer an outcome from "
            "assay presence alone. Broad redox or cytogenetic context may have low "
            "transferability to the requested bacterial condition.",
            "Respect the reported molecule role: a modifier/protectant, measured "
            "lesion, metabolite, probe or comparator is not automatically the agent "
            "whose own mutagenicity was tested. Preserve the affected molecule and "
            "co-treatment context when judging relevance to the query.",
            "Choose positive or negative for the requested condition and express "
            "uncertainty through confidence, conflicts, caveats, and evidence_gaps.",
        ),
    )
