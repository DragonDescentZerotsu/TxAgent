"""Versioned prompt contracts for BBB meaningful-CNS-access reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping


MEANINGFUL_CNS_ACCESS_V1 = "meaningful_cns_access_v1"
MEANINGFUL_CNS_ADJUDICATION_V2 = "meaningful_cns_adjudication_v2"
MEANINGFUL_CNS_ADJUDICATION_V3 = "meaningful_cns_adjudication_v3"
HISTORICAL_BBB_PROMPT_PROFILE = MEANINGFUL_CNS_ACCESS_V1
DEFAULT_BBB_PROMPT_PROFILE = MEANINGFUL_CNS_ACCESS_V1
BBB_PROMPT_PROFILES = (
    MEANINGFUL_CNS_ACCESS_V1,
    MEANINGFUL_CNS_ADJUDICATION_V2,
    MEANINGFUL_CNS_ADJUDICATION_V3,
)


SINGLE_SCHEMA: dict[str, Any] = {
    "passive_bbb_plausibility": "high | moderate | low | uncertain",
    "efflux_or_transporter_prior": "high | moderate | low | uncertain",
    "exact_chembl_evidence_assessment": "string",
    "confidence": "high | moderate | low",
    "reasoning_summary": "string",
    "property_drivers": ["string"],
    "caveats": ["string"],
}

GROUP_SCHEMA: dict[str, Any] = {
    "useful_for_bbb_reasoning": "boolean",
    "transferability": "high | moderate | low | not_applicable",
    "evidence_direction": (
        "supports_bbb_crossing | argues_against_bbb_crossing | efflux_risk | "
        "influx_support | neutral_or_unclear"
    ),
    "confidence": "high | moderate | low",
    "reasoning_summary": "string",
    "key_evidence": [
        {
            "molecule_chembl_id": "string",
            "similarity": "number or null",
            "similarity_bucket": "string",
            "assay_signal": "string",
            "activity_values": ["string"],
            "tool_summary": "string",
            "transferability": "high | moderate | low | not_applicable",
            "effect_on_bbb_reasoning": "string",
        }
    ],
    "caveats": ["string"],
}

FINAL_SCHEMA: dict[str, Any] = {
    "bbb_prediction": "pass | fail",
    "confidence": "high | moderate | low",
    "main_reasons": ["string"],
    "single_molecule_assessment": "string",
    "passive_permeability_assessment": "string",
    "direct_brain_exposure_analog_assessment": "string",
    "efflux_risk_assessment": "string",
    "influx_support_assessment": "string",
    "conflicting_evidence": ["string"],
    "evidence_gaps": ["string"],
    "final_summary": "string",
}

INTEGRATED_OUTCOME_STATES = {
    "transferable_positive",
    "transferable_negative",
    "mixed",
    "insufficient",
}
NEGATIVE_EVIDENCE_BASES = {
    "transferable_restricted_cns_outcome",
    "measured_efflux_limitation",
    "none",
}
BALANCED_NEGATIVE_EVIDENCE_BASES = {
    *NEGATIVE_EVIDENCE_BASES,
    "convergent_intrinsic_barriers",
}


@dataclass(frozen=True)
class BbbPromptProfile:
    name: str
    label_scope: str
    single_instructions: tuple[str, ...]
    group_instructions: tuple[str, ...]
    final_instructions: tuple[str, ...]
    group_schema: Mapping[str, Any] = field(default_factory=dict)
    final_schema: Mapping[str, Any] = field(default_factory=dict)
    group_required_fields: tuple[str, ...] = (
        "transferability",
        "confidence",
        "reasoning_summary",
    )
    final_required_fields: tuple[str, ...] = ("bbb_prediction",)
    final_allowed_values: Mapping[str, set[str]] = field(default_factory=dict)


BASE_SINGLE_INSTRUCTIONS = (
    "Assess passive BBB plausibility from molecular weight, logP/logD, TPSA, HBD/HBA, ionization/pKa, charge, rotatable bonds, and functional groups.",
    "Return JSON with passive_bbb_plausibility, efflux_or_transporter_prior, confidence, reasoning_summary, property_drivers, caveats.",
)

BASE_GROUP_INSTRUCTIONS = (
    "Use only this group's evidence.",
    "Each evidence_rows item follows minimal_evidence.v1; read endpoint/measurement, text, annotations, quality, provenance, and examples without assuming a source-specific schema.",
    "Assess structural transferability from neighbors to the query.",
    "Low-similarity analogs are intentionally included. You must explicitly judge whether they are transferable.",
    "Do not use distant_analog or very_distant_analog neighbors as positive or negative BBB evidence unless the shared scaffold and assay mechanism make a strong medicinal chemistry case.",
    "Use mmp_structure_compare to inspect scaffold/MCS/matched-pair differences when similarity bucket alone is not enough.",
    "Use properties_compare when property differences such as pKa, logD, TPSA, charge, HBD/HBA, or logP could affect BBB transferability.",
    "Tool outputs are authoritative only for the pair they compare; cite which neighbor each tool result supports.",
    "Use same_endpoint_activity as direct query-vs-neighbor assay comparison when present.",
    "Use same_assay_different_endpoint_activity only as same-assay context; do not directly compare numeric values across different endpoints.",
    "Distinguish direct BBB exposure, passive permeability, efflux substrate risk, influx support, and weak inhibition/binding evidence.",
    "Do not convert transporter IC50/inhibition directly into substrate/transport unless assay context supports it.",
    "Return key_evidence as structured evidence cards, not a plain list of molecule ids.",
    "For each key_evidence item, derive assay_signal and activity_values from the provided evidence_rows, derive tool_summary from tool outputs, and judge transferability/effect_on_bbb_reasoning yourself.",
    "Return JSON with useful_for_bbb_reasoning, transferability, evidence_direction, confidence, reasoning_summary, key_evidence, caveats.",
)

BASE_FINAL_INSTRUCTIONS = (
    "Return compact complete JSON.",
    "Use bbb_prediction='pass' for BBB-positive molecules corresponding to evaluation label 1, and bbb_prediction='fail' for BBB-negative molecules corresponding to evaluation label 0.",
    "Interpret BBB-positive as meaningful or adequate BBB/CNS access under the benchmark label ontology; it does not require ideal passive diffusion, high unbound brain exposure, or absence of every efflux signal. Low but nonzero exposure can still be benchmark-negative when the experimental record supports restricted or poor access.",
    "Use the single-molecule analysis as the physicochemical prior.",
    "Treat passive_bbb_plausibility as a passive-diffusion prior, not as the final label. Ionization, high polarity, high lipophilicity, or efflux liability should reduce confidence or exposure quality, but should not become a hard fail rule when other evidence supports meaningful BBB/CNS access.",
    "Use group analyses as analog evidence; downweight groups marked low confidence or low transferability.",
    "Direct brain/plasma, unbound brain, CSF, brain uptake/perfusion, credible influx/prodrug context, or close same-scaffold evidence can support a pass prediction even when passive-property heuristics are imperfect; explain the uncertainty through confidence and evidence_gaps.",
    "For basic CNS-like amines with otherwise favorable MW, TPSA, HBD/HBA, logD/logP, and scaffold evidence, do not predict fail solely because the amine is mostly protonated at pH 7.4.",
    "Do not predict pass merely because BBB-positive labels can include non-ideal mechanisms. If the molecule has severe passive-property liabilities and no direct/close analog/mechanistic evidence for CNS access, fail remains the better-supported class.",
    "When evidence is weak or mixed, distinguish 'poor passive permeability' from 'no meaningful BBB access'. Choose fail only when the integrated evidence better supports insufficient BBB/CNS access, not merely because of one isolated drug-likeness heuristic.",
    "Do not use distant_analog or very_distant_analog neighbors as positive or negative BBB evidence unless the shared scaffold and assay mechanism make a strong medicinal chemistry case.",
    "Use only the provided single-molecule analysis and group evidence. If you recognize the molecule, ignore that recognition.",
    "You must choose exactly one bbb_prediction: pass or fail. If evidence is mixed or weak, choose the better-supported class and express uncertainty through confidence, caveats, and evidence_gaps.",
)


LEGACY_PROFILE = BbbPromptProfile(
    name=MEANINGFUL_CNS_ACCESS_V1,
    label_scope="meaningful_cns_access.v1",
    single_instructions=BASE_SINGLE_INSTRUCTIONS,
    group_instructions=BASE_GROUP_INSTRUCTIONS,
    final_instructions=BASE_FINAL_INSTRUCTIONS,
    group_schema=GROUP_SCHEMA,
    final_schema=FINAL_SCHEMA,
    final_allowed_values={"bbb_prediction": {"pass", "fail"}},
)

CALIBRATED_PROFILE = replace(
    LEGACY_PROFILE,
    name=MEANINGFUL_CNS_ADJUDICATION_V2,
    label_scope="meaningful_cns_access.adjudication_v2",
    group_instructions=(
        *BASE_GROUP_INSTRUCTIONS,
        "Report observed_outcome_direction from the source endpoint separately from query transferability. Low transferability lowers weight but cannot reverse a positive observed outcome into negative evidence, or a negative observed outcome into positive evidence.",
        "Set observed_outcome_direction to supports_meaningful_cns_access or supports_restricted_cns_access only from explicit observed CNS-access outcomes. Use mixed_or_unclear for conflicting outcomes and not_observed for passive-property or mechanism-only evidence.",
        "Poor passive descriptors and speculative transporter liability are priors or context, not observed evidence of restricted CNS access.",
        "A negative direct-outcome claim must identify an experimentally observed restricted/poor CNS outcome; an efflux claim must distinguish measured substrate/transport limitation from structural speculation.",
    ),
    final_instructions=(
        *BASE_FINAL_INSTRUCTIONS,
        "First integrate observed outcome direction and transferability into integrated_outcome_state. Do not merge passive-property plausibility into that observed-outcome state.",
        "Low transferability may reduce positive analog evidence to weak or insufficient support, but it cannot make that positive observation support fail. Neutral, missing, or uncertain evidence is not affirmative negative evidence.",
        "Predict fail only when negative_evidence_basis identifies either a transferable observed restricted/poor CNS outcome or measured efflux limitation. Low passive plausibility, ionization, descriptor liabilities, low analog similarity, or speculative efflux alone require negative_evidence_basis='none' and cannot justify fail.",
        "If integrated_outcome_state is transferable_positive, do not overturn it solely because passive diffusion is poor or efflux is merely suspected; retain pass and express those liabilities through confidence and evidence_gaps.",
    ),
    group_schema={
        **GROUP_SCHEMA,
        "observed_outcome_direction": (
            "supports_meaningful_cns_access | supports_restricted_cns_access | "
            "mixed_or_unclear | not_observed"
        ),
    },
    final_schema={
        **FINAL_SCHEMA,
        "integrated_outcome_state": (
            "transferable_positive | transferable_negative | mixed | insufficient"
        ),
        "negative_evidence_basis": (
            "transferable_restricted_cns_outcome | measured_efflux_limitation | none"
        ),
    },
    group_required_fields=(
        "transferability",
        "confidence",
        "reasoning_summary",
        "observed_outcome_direction",
    ),
    final_required_fields=(
        "bbb_prediction",
        "integrated_outcome_state",
        "negative_evidence_basis",
    ),
    final_allowed_values={
        "bbb_prediction": {"pass", "fail"},
        "integrated_outcome_state": INTEGRATED_OUTCOME_STATES,
        "negative_evidence_basis": NEGATIVE_EVIDENCE_BASES,
    },
)

BALANCED_PROFILE = replace(
    CALIBRATED_PROFILE,
    name=MEANINGFUL_CNS_ADJUDICATION_V3,
    label_scope="meaningful_cns_access.adjudication_v3",
    final_instructions=(
        *BASE_FINAL_INSTRUCTIONS,
        "First integrate observed outcome direction and transferability into integrated_outcome_state. Keep that outcome state separate from the physicochemical prior.",
        "Low transferability may reduce an observed analog outcome to mixed or insufficient support, but it cannot reverse the observed direction.",
        "Set negative_evidence_basis='convergent_intrinsic_barriers' only when integrated_outcome_state is mixed or insufficient, no moderate/high-transferability positive outcome remains, passive_bbb_plausibility is low, and at least two independent severe barriers coherently support restricted access (for example very low physiological logD or persistent charge together with high MW, high polarity, or high flexibility). One descriptor or speculative efflux alone is not enough.",
        "Predict fail only from a transferable restricted CNS outcome, measured efflux limitation, or convergent intrinsic barriers. Otherwise use negative_evidence_basis='none'.",
        "If integrated_outcome_state is transferable_positive, retain pass despite poor passive diffusion or suspected efflux; express those liabilities through confidence and evidence_gaps.",
        "For mixed or insufficient analog evidence, do not default to pass: adjudicate the narrowly defined intrinsic-barrier basis explicitly.",
    ),
    final_schema={
        **CALIBRATED_PROFILE.final_schema,
        "negative_evidence_basis": (
            "transferable_restricted_cns_outcome | measured_efflux_limitation | "
            "convergent_intrinsic_barriers | none"
        ),
    },
    final_allowed_values={
        **CALIBRATED_PROFILE.final_allowed_values,
        "negative_evidence_basis": BALANCED_NEGATIVE_EVIDENCE_BASES,
    },
)


_PROFILES = {
    LEGACY_PROFILE.name: LEGACY_PROFILE,
    CALIBRATED_PROFILE.name: CALIBRATED_PROFILE,
    BALANCED_PROFILE.name: BALANCED_PROFILE,
}


def get_bbb_prompt_profile(name: str) -> BbbPromptProfile:
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown BBB prompt profile: {name}") from exc


def final_profile_validation_errors(
    content: Mapping[str, Any],
    *,
    profile: str,
) -> list[str]:
    """Enforce the selected candidate's explicit evidence gate."""
    if profile not in {
        MEANINGFUL_CNS_ADJUDICATION_V2,
        MEANINGFUL_CNS_ADJUDICATION_V3,
    }:
        return []
    prediction = str(content.get("bbb_prediction") or "")
    state = str(content.get("integrated_outcome_state") or "")
    basis = str(content.get("negative_evidence_basis") or "")
    errors: list[str] = []
    if prediction == "fail" and basis == "none":
        errors.append("fail requires affirmative negative_evidence_basis")
    if state == "transferable_positive" and prediction != "pass":
        errors.append("transferable_positive integrated outcome requires pass")
    return errors
