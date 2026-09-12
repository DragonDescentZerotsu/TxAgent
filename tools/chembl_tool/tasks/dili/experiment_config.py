"""DILI record families and the shared progressive reasoning contract."""

from tools.chembl_tool.common.experiment_retrieval import (
    SourceExperimentConfig,
    evidence_family,
)
from tools.chembl_tool.common.progressive_assay_reasoning import ProgressiveTaskContract
from tools.chembl_tool.tasks.dili.constants import (
    DILI_NEGATIVE_PREDICTION,
    DILI_POSITIVE_PREDICTION,
)
from tools.chembl_tool.tasks.dili.starling_levels import FAMILIES, LEVEL_DESCRIPTIONS


_FAMILY_LABELS = {
    1: "Direct human liver injury outcomes",
    2: "Additional liver injury evidence",
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
# The current source release uses the shared L1-only index prefilter and
# query-time disjoint retrieval. Do not reintroduce a union-wide L2 text filter.
PROGRESSIVE_HELDOUT_DIRECT_ALIASES = {}
PROGRESSIVE_ASSAY_LEVEL_DESCRIPTIONS = {
    **LEVEL_DESCRIPTIONS,
    1: "Observed human liver injury outcomes under the reported exposure and population.",
    2: "Liver injury outcomes, clinical signals, predictions and mixed passages; preserve the tested role, system and uncertainty.",
}


def get_source_config(source: str) -> SourceExperimentConfig:
    return SOURCES[source]


def get_progressive_task_contract() -> ProgressiveTaskContract:
    return ProgressiveTaskContract(
        task="dili",
        endpoint_name="clinically meaningful human drug-induced liver injury",
        label_scope="dili_conditioned_or_source_molecule_outcome.v1",
        prediction_field="dili_prediction",
        positive_prediction=DILI_POSITIVE_PREDICTION,
        negative_prediction=DILI_NEGATIVE_PREDICTION,
        system_role="Assess human drug-induced liver injury using molecular evidence.",
        task_instructions=(
            "Predict clinically meaningful human drug-induced liver injury under "
            "the query's reported exposure and population when these are supplied.",
            "When the query has no reported external condition, assess the "
            "molecule-level DILI endpoint without inventing a dose, route, "
            "population or duration. Missing conditions do not imply a universal "
            "claim across all exposure settings.",
            "Separate observed human DILI from animal liver injury, hepatocyte "
            "toxicity, generic cytotoxicity, mechanistic liabilities and model "
            "predictions. Those endpoints are not interchangeable measurements.",
            "Evidence from hepatic cell function, bile-acid transport, "
            "bioactivation, mitochondria, immune interactions and cellular "
            "stress may change the prediction when the mechanistic link and "
            "structural transferability are strong. Explain their limitations; "
            "new direct outcome evidence is not required for a justified update.",
            "Preserve the tested molecule's role and co-exposure context. A "
            "protectant, modifier, metabolite or probe is not automatically "
            "the substance causing the reported injury.",
            "A negative clinical observation applies to its reported population "
            "and exposure; absence of published reports or an isolated negative "
            "mechanism assay does not establish absence of DILI risk.",
            "Keep mixed outcomes, predictions, missing context and uncertainty "
            "explicit. Family placement does not establish a new experiment or "
            "independent evidence support.",
            "Choose dili_risk or no_dili_risk and express uncertainty through "
            "confidence, conflicts, caveats and evidence_gaps.",
        ),
    )
