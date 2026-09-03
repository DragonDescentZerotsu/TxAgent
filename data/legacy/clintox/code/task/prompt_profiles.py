"""Canonical prompt contract for the ClinTox reasoning pipeline."""

from __future__ import annotations

from dataclasses import dataclass


TDC_SOURCE_ALIGNED_V3 = "tdc_source_aligned_v3"
DEFAULT_CLINTOX_PROMPT_PROFILE = TDC_SOURCE_ALIGNED_V3
CLINTOX_PROMPT_PROFILES = (TDC_SOURCE_ALIGNED_V3,)


@dataclass(frozen=True)
class ClinToxPromptProfile:
    name: str
    label_scope: str
    prediction_target: str
    final_system_instruction: str
    final_task_instruction: str
    class_definition_instruction: str
    group_policy_instructions: tuple[str, ...]
    final_policy_instructions: tuple[str, ...]


SOURCE_ALIGNED_PROFILE = ClinToxPromptProfile(
    name=TDC_SOURCE_ALIGNED_V3,
    label_scope="aact_toxicity_failure_association_vs_fda_comparator.v1",
    prediction_target=(
        "the frozen AACT toxicity-failure-associated class versus the FDA-approved "
        "comparator-only class"
    ),
    final_system_instruction=(
        "Predict the source-defined ClinTox CT_TOX class for the query molecular parent. "
        "This is a historical dataset-class prediction, not a clinical causality adjudication."
    ),
    final_task_instruction=(
        "Source-defined AACT toxicity-failure-association versus FDA-comparator-only prediction "
        "from analog evidence."
    ),
    class_definition_instruction=(
        "Interpret clintox_prediction='toxic' as the frozen AACT-derived toxicity-failure-associated "
        "class. Interpret clintox_prediction='non_toxic' as the FDA-approved comparator class with no "
        "recorded AACT toxicity-failure association in the frozen snapshot. These class tokens do not "
        "mean universally toxic or universally safe."
    ),
    group_policy_instructions=(
        "Separate evidence provenance from prediction: only a concrete Direct.clinical_trial_failure row may be described as an observed analog trial/development-failure association.",
        "Treat a direct row as strong only when its text explicitly links toxicity-related trial or development stoppage to the intervention. Downweight ambiguous status-only records, unclear combination attribution, patient-level discontinuation, withdrawal before dosing, postmarketing action, or unspecified safety concern.",
        "Clinical context and mechanistic evidence may support higher or lower class probability when measured, coherent, and structurally transferable, but must never be described as an observed trial-failure event.",
        "For Flat.all_evidence, assess every row using its own minimal_evidence group.id. A direct row does not make neighboring contextual or mechanistic rows direct.",
        "Set direct_evidence_status from the concrete rows in this group, independently of the overall predictive evidence_direction.",
    ),
    final_policy_instructions=(
        "The positive source records are historical AACT-derived associations and were not re-adjudicated here as molecule-intrinsic causal facts. Predict source-defined class membership without claiming that the query caused a trial failure.",
        "A transferable direct analog is strong evidence but is not required for clintox_prediction='toxic'. Absence of a retrieved Direct.clinical_trial_failure analog is an evidence gap, not affirmative evidence for clintox_prediction='non_toxic'.",
        "FDA approval may weakly support the comparator-only class in context, but it is never a veto: approval and an AACT toxicity-failure association can coexist.",
        "Clinical and mechanistic evidence may contribute according to endpoint relevance, measured severity, coherence, structural transferability, exposure context, and uncertainty. A generic hazard signal alone must not determine the class.",
        "Only call an event directly observed in an analog when the corresponding group reports a concrete, intervention-linked Direct.clinical_trial_failure row. Keep that provenance claim separate from the final predicted class.",
    ),
)


def get_clintox_prompt_profile(name: str) -> ClinToxPromptProfile:
    if name != SOURCE_ALIGNED_PROFILE.name:
        raise ValueError(f"Unknown ClinTox prompt profile: {name}")
    return SOURCE_ALIGNED_PROFILE
