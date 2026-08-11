"""Versioned prompt contracts for the Bioavailability_Ma reasoning pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


LEGACY_BIOAVAILABILITY_V1 = "legacy_bioavailability_v1"
F20_EVIDENCE_CALIBRATED_V2 = "f20_evidence_calibrated_v2"
HISTORICAL_BIOAVAILABILITY_PROMPT_PROFILE = LEGACY_BIOAVAILABILITY_V1
DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE = F20_EVIDENCE_CALIBRATED_V2
BIOAVAILABILITY_PROMPT_PROFILES = (
    LEGACY_BIOAVAILABILITY_V1,
    F20_EVIDENCE_CALIBRATED_V2,
)


SINGLE_SCHEMA: dict[str, Any] = {
    "oral_bioavailability_prior": "high | low | mixed_or_unclear",
    "absorption_prior": "favorable | unfavorable | mixed_or_unclear",
    "solubility_or_dissolution_prior": "favorable | unfavorable | mixed_or_unclear",
    "metabolism_or_clearance_prior": "favorable | unfavorable | mixed_or_unclear",
    "exact_chembl_evidence_assessment": "string",
    "confidence": "high | moderate | low",
    "reasoning_summary": "string",
    "property_drivers": ["string"],
    "caveats": ["string"],
}

GROUP_SCHEMA: dict[str, Any] = {
    "useful_for_bioavailability_reasoning": "boolean",
    "transferability": "high | moderate | low | not_applicable",
    "evidence_direction": (
        "supports_high_bioavailability | argues_against_high_bioavailability | absorption_support | "
        "permeability_support | solubility_support | solubility_risk | metabolic_stability_support | "
        "first_pass_or_clearance_risk | transporter_efflux_risk | neutral_or_unclear"
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
            "effect_on_bioavailability_reasoning": "string",
        }
    ],
    "caveats": ["string"],
}

FINAL_SCHEMA: dict[str, Any] = {
    "bioavailability_prediction": "high | low",
    "confidence": "high | moderate | low",
    "main_reasons": ["string"],
    "single_molecule_assessment": "string",
    "absorption_and_permeability_assessment": "string",
    "solubility_and_dissolution_assessment": "string",
    "metabolism_first_pass_and_clearance_assessment": "string",
    "transporter_efflux_assessment": "string",
    "direct_oral_bioavailability_analog_assessment": "string",
    "conflicting_evidence": ["string"],
    "evidence_gaps": ["string"],
    "final_summary": "string",
}


@dataclass(frozen=True)
class BioavailabilityPromptProfile:
    name: str
    label_scope: str
    single_system_role: str
    single_task: str
    single_instructions: tuple[str, ...]
    group_system_role: str
    group_task: str
    group_instructions: tuple[str, ...]
    final_system_role: str
    final_task: str
    final_instructions: tuple[str, ...]


LEGACY_PROFILE = BioavailabilityPromptProfile(
    name=LEGACY_BIOAVAILABILITY_V1,
    label_scope="absolute_oral_bioavailability_f20_legacy.v1",
    single_system_role=(
        "You are a medicinal chemistry oral bioavailability single-molecule analyst. "
        "Only analyze the query molecule itself, without analog evidence. "
    ),
    single_task="Single-molecule oral bioavailability plausibility analysis.",
    single_instructions=(
        "Assess oral bioavailability prior from molecular weight, logP/logD, TPSA, HBD/HBA, ionization/pKa, charge, rotatable bonds, and functional groups.",
        "Return JSON with oral_bioavailability_prior, absorption_prior, solubility_or_dissolution_prior, metabolism_or_clearance_prior, confidence, reasoning_summary, property_drivers, caveats.",
    ),
    group_system_role=(
        "You are a medicinal chemistry oral bioavailability analog evidence analyst. "
        "Reason about whether analog evidence in one endpoint group is transferable to the query molecule. "
    ),
    group_task="Group-level oral bioavailability analog transferability analysis.",
    group_instructions=(
        "Assess structural transferability from neighbors to the query.",
        "Low-similarity analogs are intentionally included. You must explicitly judge whether they are transferable.",
        "Do not use distant_analog or very_distant_analog neighbors as positive or negative oral bioavailability evidence unless the shared scaffold and assay mechanism make a strong medicinal chemistry case.",
    ),
    final_system_role=(
        "You are a senior oral bioavailability reasoning model. Integrate group-level analog evidence into one final oral bioavailability assessment. "
    ),
    final_task="Final oral bioavailability prediction from analog evidence.",
    final_instructions=(
        "Use the single-molecule analysis as the physicochemical prior.",
        "Use group analyses as analog evidence; downweight groups marked low confidence or low transferability.",
        "Do not use distant_analog or very_distant_analog neighbors as positive or negative oral bioavailability evidence unless the shared scaffold and assay mechanism make a strong medicinal chemistry case.",
        "You must choose exactly one bioavailability_prediction: high or low. If evidence is mixed or weak, choose the better-supported class and express uncertainty through confidence, caveats, and evidence_gaps.",
    ),
)


CALIBRATED_PROFILE = BioavailabilityPromptProfile(
    name=F20_EVIDENCE_CALIBRATED_V2,
    label_scope="absolute_oral_bioavailability_f20.v2",
    single_system_role=(
        "You are an absolute oral bioavailability F-threshold analyst. "
        "Only assess whether the query is plausibly above or below F=20%, without analog evidence. "
    ),
    single_task="Single-molecule prior for absolute oral bioavailability F >= 20% versus F < 20%.",
    single_instructions=(
        "Calibrate every conclusion to the modest binary threshold: high means absolute oral bioavailability F >= 20%, not ideal exposure or general drug-likeness; low means F < 20%.",
        "Use molecular weight, logP/logD, TPSA, HBD/HBA, ionization/pKa, charge, rotatable bonds, and functional groups only as uncertain physicochemical priors.",
        "Lipinski or QED violations, high molecular weight, ionization, polarity, lipophilicity, flexibility, or absence of a favorable property are not individually sufficient to support F < 20%.",
        "Do not infer metabolic instability, first-pass clearance, or transporter efflux from structure alone without direct supporting evidence.",
        "Choose oral_bioavailability_prior='low' only when multiple threshold-relevant liabilities coherently and specifically support F < 20%; otherwise use mixed_or_unclear rather than converting uncertainty into low.",
        "Return JSON with oral_bioavailability_prior, absorption_prior, solubility_or_dissolution_prior, metabolism_or_clearance_prior, confidence, reasoning_summary, property_drivers, caveats.",
    ),
    group_system_role=(
        "You are an absolute oral bioavailability analog evidence analyst. "
        "Keep observed evidence direction separate from query-specific transferability and weight. "
    ),
    group_task="Group-level evidence direction and transferability for absolute oral F >= 20% versus F < 20%.",
    group_instructions=(
        "Assess structural and property transferability explicitly, but report evidence direction from the observed endpoint separately from transferability.",
        "Low or not-applicable transferability lowers evidence weight; it does not reverse the observed direction and is not evidence for the opposite class.",
        "For direct oral-bioavailability outcomes, consistently high analog outcomes remain supports_high_bioavailability even when distant; consistently low outcomes remain argues_against_high_bioavailability. Express distance through transferability, confidence, caveats, and effect_on_bioavailability_reasoning.",
        "Use neutral_or_unclear only when the underlying endpoint direction is genuinely mixed, missing, ambiguous, out of scope, or non-directional—not merely because the analog is distant.",
        "Missing quantitative F, species, formulation, or assay context is uncertainty, not negative evidence. Do not invent F < 20% support from missing context.",
        "Do not flip a direct observed outcome because the query has generic MW, logP, QED, Lipinski, ionization, polarity, or flexibility liabilities.",
    ),
    final_system_role=(
        "You are a senior absolute oral bioavailability F-threshold reasoning model. "
        "Integrate evidence direction and transferability without converting uncertainty into a negative outcome. "
    ),
    final_task="Final absolute oral bioavailability prediction at the F=20% threshold.",
    final_instructions=(
        "Use the single-molecule analysis only as an uncertain physicochemical prior calibrated to F=20%, not as a general drug-likeness verdict.",
        "For every group, keep evidence direction separate from transferability: low transferability lowers weight but cannot turn positive evidence into negative evidence or vice versa.",
        "Insufficient, neutral, low-confidence, low-transferability, or missing evidence is not evidence for F < 20% and must not default to bioavailability_prediction='low'.",
        "Predict low only when affirmative threshold-relevant evidence supports F < 20%, such as transferable direct low outcomes or coherent measured absorption, solubility, first-pass, clearance, or efflux evidence. Generic descriptor liabilities or speculative mechanisms alone are insufficient.",
        "A high prediction means only that F >= 20% is better supported; it does not assert optimal exposure, formulation independence, or high confidence.",
        "When evidence conflicts, compare source directness, endpoint match, direction, transferability, and uncertainty explicitly; express unresolved uncertainty through confidence and evidence_gaps rather than treating it as low.",
        "You must choose exactly one bioavailability_prediction: high or low.",
    ),
)


_PROFILES = {
    LEGACY_PROFILE.name: LEGACY_PROFILE,
    CALIBRATED_PROFILE.name: CALIBRATED_PROFILE,
}


def get_bioavailability_prompt_profile(name: str) -> BioavailabilityPromptProfile:
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown Bioavailability prompt profile: {name}") from exc
