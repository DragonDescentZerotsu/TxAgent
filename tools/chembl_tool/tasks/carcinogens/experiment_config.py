"""Carcinogenicity record families and shared progressive reasoning semantics."""

from tools.chembl_tool.common.experiment_retrieval import (
    SourceExperimentConfig,
    evidence_family,
)
from tools.chembl_tool.common.progressive_assay_reasoning import ProgressiveTaskContract
from tools.chembl_tool.tasks.carcinogens.starling_levels import (
    FAMILIES,
    LEVEL_DESCRIPTIONS,
)


_FAMILY_LABELS = {
    1: "Direct carcinogenicity outcomes",
    2: "Additional tumor-outcome evidence",
}
_FAMILY_SPECS = tuple(
    evidence_family(key, family_label=_FAMILY_LABELS.get(level, key),
                    source_group_ids=(f"Group.{key}",))
    for level, key in sorted(FAMILIES.items())
)
STARLING = SourceExperimentConfig(
    source_name="starling",
    direct_groups=_FAMILY_SPECS[:1],
    mechanism_groups=_FAMILY_SPECS,
)
SOURCES = {"starling": STARLING}
PROGRESSIVE_ASSAY_LEVEL_DESCRIPTIONS = {
    **LEVEL_DESCRIPTIONS,
    1: "Observed carcinogenicity outcomes under the reported species, model and exposure.",
    2: "Tumor outcomes, carcinogenicity assertions, predictions and mixed passages; preserve the tested role and limitations.",
}


def get_source_config(source: str) -> SourceExperimentConfig:
    return SOURCES[source]


def get_progressive_task_contract() -> ProgressiveTaskContract:
    return ProgressiveTaskContract(
        task="carcinogens",
        endpoint_name="source-supported carcinogenicity in the reported organism group",
        label_scope="carcinogens_starling_only_five_organism_groups.v1",
        prediction_field="carcinogenicity_prediction",
        positive_prediction="positive",
        negative_prediction="negative",
        system_role="Assess chemical carcinogenic hazard using molecular evidence.",
        evidence_grounding_rules=(
            ("identity_rule", "The indexed neighbor is distinct from the query. Bind each observation to "
             "the actually tested molecule or material and its role. A mixed passage may describe several "
             "agents, including the query: distinguish those observations explicitly. Similarity or a shared "
             "scaffold does not make an analog experiment query-specific. Explain whether the structural "
             "features responsible for the effect are retained in the query, for both positive and negative "
             "transfer. Do not guess the query name from its structure."),
            ("endpoint_decision_rule", "Predict source-supported carcinogenicity in the reported organism "
             "group using the task definition. Distinguish observations, source assertions, predictions and "
             "mechanistic inferences. Transferable mechanisms may support a progressive update without new "
             "direct tumor data; do not describe a hypothesis as an observed tumor outcome."),
            ("comparison_rule", "State what was compared: test agent alone, challenge plus modifier, or a "
             "mixture/material. No reduction of challenge-induced damage means no detected protection, not "
             "absence of damage. Inhibiting one stimulated proliferation pathway does not establish an overall "
             "negative; short-term normal proliferation does not itself establish tumor promotion. Assess "
             "scoped positive and negative findings symmetrically without requiring universal safety."),
        ),
        task_instructions=(
            "Predict source-supported carcinogenicity in the query's reported "
            "organism group. Rodent pools rats, mice and other explicitly "
            "identified rodents; human, dog, monkey and rabbit remain separate.",
            "The benchmark pools sex, strain, route, dose and duration within "
            "each organism group. Source records retain these restrictions: "
            "a pooled label is not a claim about every exposure or every member "
            "of the group. Use the balance of relevant evidence and explain "
            "transfer limitations.",
            "When the query has no reported external condition, assess the "
            "molecule-level carcinogenicity endpoint without inventing a species, "
            "dose, route or duration. Do not silently turn an unspecified "
            "molecule-level label into a human-only cancer conclusion.",
            "Separate observed tumor or epidemiological outcomes from cell "
            "transformation, genotoxicity, DNA damage, molecular mechanisms "
            "and model predictions; these are not interchangeable measurements.",
            "Transformation, DNA damage and repair, reactive metabolism, "
            "oxidative injury, epigenetic changes, disrupted growth restraint "
            "and regenerative or receptor-mediated proliferation may change "
            "the prediction when transferable. Explain the mechanistic link "
            "and limitations; new direct tumor data are not required for an update.",
            "A negative result at one tumor site, for malignant tumors alone, "
            "or for one mechanistic assay does not establish an overall "
            "negative carcinogenicity outcome. Preserve species and exposure "
            "limitations of both positive and negative observations.",
            "Distinguish a carcinogenic test agent from a protectant, treatment, "
            "initiator, promoter, co-carcinogen, metabolite or probe. Anticancer "
            "efficacy and killing tumor cells are not proof of carcinogenicity.",
            "Preserve prediction status, conflicting outcomes and missing "
            "context. Family placement does not create an independent study "
            "or establish that a mechanism caused a measured tumor outcome.",
            "Choose positive or negative and express uncertainty through "
            "confidence, conflicts, caveats and evidence_gaps.",
        ),
    )
