"""Versioned prompt contracts for the Skin_Reaction reasoning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


LEGACY_SKIN_REACTION_V1 = "legacy_skin_reaction_v1"
SENSITIZATION_ALIGNED_V2 = "sensitization_aligned_v2"
SENSITIZATION_NEGATIVE_TRANSFER_V3 = "sensitization_negative_transfer_v3"
HISTORICAL_SKIN_PROMPT_PROFILE = LEGACY_SKIN_REACTION_V1
DEFAULT_SKIN_PROMPT_PROFILE = SENSITIZATION_ALIGNED_V2
SKIN_PROMPT_PROFILES = (
    LEGACY_SKIN_REACTION_V1,
    SENSITIZATION_ALIGNED_V2,
    SENSITIZATION_NEGATIVE_TRANSFER_V3,
)


@dataclass(frozen=True)
class SkinPromptProfile:
    name: str
    label_scope: str
    single_system_role: str
    single_task: str
    single_instructions: tuple[str, ...]
    single_schema: dict[str, Any]
    group_system_role: str
    group_task: str
    group_instructions: tuple[str, ...]
    group_schema: dict[str, Any]
    final_system_role: str
    final_task: str
    final_instructions: tuple[str, ...]
    final_schema: dict[str, Any]
    final_required_fields: tuple[str, ...]
    final_allowed_values: dict[str, set[str]]


LEGACY_PROFILE = SkinPromptProfile(
    name=LEGACY_SKIN_REACTION_V1,
    label_scope="broad_skin_reaction_legacy.v1",
    single_system_role=(
        "You are a medicinal chemistry Skin_Reaction single-molecule analyst. "
        "Only analyze the query molecule itself, without analog evidence. "
    ),
    single_task="Single-molecule Skin_Reaction plausibility analysis.",
    single_instructions=(
        "Assess reactive/haptenation prior from electrophilic groups, Michael acceptors, aldehydes, acylating groups, oxidizable anilines/phenols, thiol/GSH reactivity plausibility, and functional groups.",
        "Assess skin exposure plausibility from logP/logD, ionization, charge, TPSA, HBD/HBA, molecular size, and lipophilicity.",
        "Assess phototoxicity structural prior from aromatic chromophores, extended conjugation, halogenated aromatics, quinones, psoralens-like motifs, or other UV-absorbing alerts when apparent.",
        "Assess irritation/corrosion prior from strong acids/bases, surfactant-like amphiphiles, reactive electrophiles, and local cytotoxicity alerts.",
        "Return JSON with skin_reaction_prior, reactive_or_haptenation_prior, skin_permeation_prior, phototoxicity_structural_prior, irritation_or_corrosion_structural_prior, confidence, reasoning_summary, property_drivers, caveats.",
    ),
    single_schema={
        "skin_reaction_prior": "risk | no_risk | mixed_or_unclear",
        "reactive_or_haptenation_prior": "concerning | not_apparent | mixed_or_unclear",
        "skin_permeation_prior": "high | low | mixed_or_unclear",
        "phototoxicity_structural_prior": "concerning | not_apparent | mixed_or_unclear",
        "irritation_or_corrosion_structural_prior": "concerning | not_apparent | mixed_or_unclear",
        "physicochemical_exposure_prior": "favorable_for_skin_exposure | unfavorable_for_skin_exposure | mixed_or_unclear",
        "exact_chembl_evidence_assessment": "string",
        "confidence": "high | moderate | low",
        "reasoning_summary": "string",
        "property_drivers": ["string"],
        "caveats": ["string"],
    },
    group_system_role=(
        "You are a medicinal chemistry Skin_Reaction analog evidence analyst. "
        "Reason about whether analog evidence in one endpoint group is transferable to the query molecule. "
    ),
    group_task="Group-level Skin_Reaction analog transferability analysis.",
    group_instructions=(
        "Use only this group's evidence.",
        "Each evidence_rows item follows minimal_evidence.v1; read endpoint/measurement, text, annotations, quality, provenance, and examples without assuming a source-specific schema.",
        "Assess structural transferability from neighbors to the query.",
        "Low-similarity analogs are intentionally included. You must explicitly judge whether they are transferable.",
        "Do not use distant_analog or very_distant_analog neighbors as positive or negative Skin_Reaction evidence unless the shared scaffold and assay mechanism make a strong medicinal chemistry case.",
        "Use mmp_structure_compare to inspect scaffold/MCS/matched-pair differences when similarity bucket alone is not enough.",
        "Use properties_compare when property differences such as electrophilic motifs, functional groups, pKa, logD, TPSA, charge, HBD/HBA, logP, molecular size, or polarity could affect Skin_Reaction transferability.",
        "Tool outputs are authoritative only for the pair they compare; cite which neighbor each tool result supports.",
        "Use same_endpoint_activity as direct query-vs-neighbor assay comparison when present.",
        "Use same_assay_different_endpoint_activity only as same-assay context; do not directly compare numeric values across different endpoints.",
        "Distinguish direct human/LLNA/GPMT/dermal toxicity anchors, sensitisation AOP key events, phototoxicity, irritation/corrosion, skin exposure/permeability context, and weak background evidence.",
        "Do not convert skin permeability/retention into skin-reaction hazard; it only modifies exposure plausibility.",
        "Do not convert generic cytotoxicity, dermatology efficacy, target binding, or antimicrobial activity into adverse skin-reaction evidence.",
        "Return key_evidence as structured evidence cards, not a plain list of molecule ids.",
        "For each key_evidence item, derive assay_signal and activity_values from the provided evidence_rows, derive tool_summary from tool outputs, and judge transferability/effect_on_skin_reaction_reasoning yourself.",
        "Return JSON with useful_for_skin_reaction_reasoning, transferability, evidence_direction, confidence, reasoning_summary, key_evidence, caveats.",
    ),
    group_schema={
        "useful_for_skin_reaction_reasoning": "boolean",
        "transferability": "high | moderate | low | not_applicable",
        "evidence_direction": (
            "supports_skin_reaction_risk | argues_against_skin_reaction_risk | sensitization_risk | "
            "irritation_or_corrosion_risk | phototoxicity_risk | local_skin_damage_risk | "
            "skin_exposure_support | reduced_skin_exposure | context_dependent | neutral_or_unclear"
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
                "effect_on_skin_reaction_reasoning": "string",
            }
        ],
        "caveats": ["string"],
    },
    final_system_role=(
        "You are a senior Skin_Reaction reasoning model. Integrate group-level analog evidence into one final skin-reaction risk assessment. "
    ),
    final_task="Final Skin_Reaction prediction from analog evidence.",
    final_instructions=(
        "Return compact complete JSON.",
        "Use skin_reaction_prediction='risk' for Skin_Reaction label 1, and skin_reaction_prediction='no_risk' for label 0.",
        "Use the single-molecule analysis as the structural and physicochemical prior.",
        "Use group analyses as analog evidence; downweight groups marked low confidence or low transferability.",
        "Do not use distant_analog or very_distant_analog neighbors as positive or negative Skin_Reaction evidence unless the shared scaffold and assay mechanism make a strong medicinal chemistry case.",
        "Treat direct human/LLNA/GPMT/dermal toxicity evidence as the strongest anchors.",
        "Treat AOP key-event evidence as sensitisation hazard evidence; one isolated key event is not equal to clinical skin reaction.",
        "Treat phototoxicity, irritation/corrosion, and allergic sensitisation as distinct skin-reaction mechanisms.",
        "Treat skin permeability/retention as exposure context only; it cannot by itself prove skin-reaction risk.",
        "Treat generic cytotoxicity, dermatology efficacy, target binding, and antimicrobial assays as weak context only.",
        "Use only the provided single-molecule analysis and group evidence. If you recognize the molecule, ignore that recognition.",
        "You must choose exactly one skin_reaction_prediction: risk or no_risk. If evidence is mixed or weak, choose the better-supported class and express uncertainty through confidence, caveats, and evidence_gaps.",
    ),
    final_schema={
        "skin_reaction_prediction": "risk | no_risk",
        "confidence": "high | moderate | low",
        "main_evidence_type": "direct_skin_reaction_anchor | sensitization_aop | phototoxicity | irritation_or_corrosion | exposure_context_only | weak_or_no_evidence",
        "main_reasons": ["string"],
        "single_molecule_assessment": "string",
        "direct_skin_reaction_anchor_assessment": "string",
        "sensitization_aop_assessment": "string",
        "phototoxicity_assessment": "string",
        "irritation_or_corrosion_assessment": "string",
        "skin_exposure_context_assessment": "string",
        "weak_context_assessment": "string",
        "conflicting_evidence": ["string"],
        "evidence_gaps": ["string"],
        "final_summary": "string",
    },
    final_required_fields=("skin_reaction_prediction",),
    final_allowed_values={"skin_reaction_prediction": {"risk", "no_risk"}},
)


SENSITIZATION_ALIGNED_PROFILE = SkinPromptProfile(
    name=SENSITIZATION_ALIGNED_V2,
    label_scope="skin_sensitization_contact_allergy.v2",
    single_system_role=(
        "You are a medicinal chemistry skin-sensitization and contact-allergy analyst. "
        "Only analyze the query molecule itself, without analog evidence. "
    ),
    single_task="Single-molecule skin-sensitization/contact-allergy prior.",
    single_instructions=(
        "The binary task is skin sensitization/contact allergy, not generic skin reaction, phototoxicity, irritation, corrosion, dermal toxicity, or exposure.",
        "Assess electrophilic/haptenation potential, including direct haptens, pre-haptens, pro-haptens, oxidizable anilines/phenols, Michael acceptors, aldehydes, acylating groups, and thiol/GSH reactivity plausibility.",
        "Assess whether plausible metabolic or air-oxidation activation could create a reactive sensitizing species; absence of an obvious structural alert is not proof of a non-sensitizer.",
        "Treat lipophilicity, ionization, size, polarity, and skin permeation only as exposure context. They cannot support either sensitizer or non-sensitizer by themselves.",
        "Phototoxicity and irritation/corrosion alerts are outside the binary label scope and must not affect skin_sensitization_prior.",
        "Return only the required JSON fields and keep uncertainty explicit.",
    ),
    single_schema={
        "label_scope": "skin_sensitization_contact_allergy.v2",
        "skin_sensitization_prior": "risk | no_risk | mixed_or_unclear",
        "reactive_or_haptenation_prior": "concerning | not_apparent | mixed_or_unclear",
        "activation_prior": "direct_hapten | pre_hapten | pro_hapten | none_apparent | mixed_or_unclear",
        "skin_exposure_context": "favorable | unfavorable | mixed_or_unclear",
        "confidence": "high | moderate | low",
        "reasoning_summary": "string",
        "property_drivers": ["string"],
        "caveats": ["string"],
    },
    group_system_role=(
        "You are a medicinal chemistry skin-sensitization/contact-allergy analog evidence analyst. "
        "Judge whether one mechanism-family branch is transferable to that exact binary task. "
    ),
    group_task="Group-level skin-sensitization/contact-allergy analog transferability analysis.",
    group_instructions=(
        "Use only this group's evidence and return the requested structured JSON.",
        "The binary label is sensitizer/contact-allergy positive versus non-sensitizer; it is not generic skin hazard.",
        "Direct anchors are explicit human patch/contact-allergy, LLNA, GPMT, Buehler, or other validated sensitization outcomes. Generic dermal toxicity and irritation are not direct anchors.",
        "Validated sensitization AOP key events can support a sensitization mechanism, but one isolated key event is not equivalent to a sensitizer call.",
        "Phototoxicity, irritation/corrosion, generic local injury, and skin-cell cytotoxicity are out of scope. Mark them context_only and do not let them support or oppose the binary label.",
        "Skin permeability, absorption, and retention are exposure context only. They cannot support or oppose the binary label.",
        "Only an adequately described negative sensitization assay can argue against sensitization; missing evidence or absent structural alerts are not negative experimental evidence.",
        "Assess structural transferability explicitly; downweight distant analogs and analogs whose reactive functional group or activation route is absent in the query.",
        "Use mmp_structure_compare and properties_compare only for the query-neighbor pair they describe.",
        "Return key_evidence as source-backed evidence cards, not molecule-id votes.",
    ),
    group_schema={
        "label_scope": "skin_sensitization_contact_allergy.v2",
        "endpoint_scope": "direct_sensitization | sensitization_aop | out_of_scope_other_skin_hazard | exposure_context | weak_context",
        "useful_for_skin_sensitization_reasoning": "boolean",
        "transferability": "high | moderate | low | not_applicable",
        "sensitization_evidence_direction": "supports_sensitizer | argues_against_sensitizer | context_only | neutral_or_unclear",
        "confidence": "high | moderate | low",
        "reasoning_summary": "string",
        "key_evidence": [
            {
                "molecule_chembl_id": "string",
                "similarity": "number or null",
                "assay_signal": "string",
                "tool_summary": "string",
                "transferability": "high | moderate | low | not_applicable",
                "effect_on_skin_sensitization_reasoning": "string",
            }
        ],
        "caveats": ["string"],
    },
    final_system_role=(
        "You are a senior skin-sensitization/contact-allergy reasoning model. "
        "Integrate the supplied branches into that exact binary endpoint. "
    ),
    final_task="Final skin-sensitization/contact-allergy prediction from analog evidence.",
    final_instructions=(
        "Return compact complete JSON.",
        "Set label_scope exactly to skin_sensitization_contact_allergy.v2.",
        "Use skin_reaction_prediction='risk' for sensitizer/contact-allergy positive (label 1) and 'no_risk' for non-sensitizer (label 0).",
        "Only direct sensitization outcomes and transferable, coherent sensitization AOP evidence count as experimental evidence for the binary label.",
        "Treat structural haptenation/activation analysis as a prior, not as a substitute for experimental sensitization evidence.",
        "Phototoxicity, irritation/corrosion, generic local injury, dermal toxicity, and skin-cell cytotoxicity are out of scope and cannot support either prediction.",
        "Skin permeability, absorption, and retention are exposure context only and cannot support either prediction.",
        "Do not interpret missing sensitization evidence or absence of a simple structural alert as proof of no_risk.",
        "Do not count repeated records, cards, and their group summaries as independent evidence.",
        "Use only the supplied anonymous query and evidence. Express uncertainty through confidence, conflicts, caveats, and evidence_gaps while still choosing exactly one prediction.",
    ),
    final_schema={
        "label_scope": "skin_sensitization_contact_allergy.v2",
        "skin_reaction_prediction": "risk | no_risk",
        "confidence": "high | moderate | low",
        "main_evidence_type": "direct_sensitization_anchor | sensitization_aop | structural_haptenation_prior | weak_or_no_sensitization_evidence",
        "main_reasons": ["string"],
        "single_molecule_sensitization_prior": "string",
        "direct_sensitization_anchor_assessment": "string",
        "sensitization_aop_assessment": "string",
        "out_of_scope_context_assessment": "string",
        "conflicting_evidence": ["string"],
        "evidence_gaps": ["string"],
        "final_summary": "string",
    },
    final_required_fields=("label_scope", "skin_reaction_prediction"),
    final_allowed_values={
        "label_scope": {"skin_sensitization_contact_allergy.v2"},
        "skin_reaction_prediction": {"risk", "no_risk"},
        "confidence": {"high", "moderate", "low"},
        "main_evidence_type": {
            "direct_sensitization_anchor",
            "sensitization_aop",
            "structural_haptenation_prior",
            "weak_or_no_sensitization_evidence",
        },
    },
)


SENSITIZATION_NEGATIVE_TRANSFER_PROFILE = replace(
    SENSITIZATION_ALIGNED_PROFILE,
    name=SENSITIZATION_NEGATIVE_TRANSFER_V3,
    group_instructions=(
        *SENSITIZATION_ALIGNED_PROFILE.group_instructions,
        (
            "An analog-only negative sensitization outcome may argue against "
            "sensitization only when transferability is high, the validated assay "
            "used adequate exposure, and the direct records are directionally "
            "consistent. Otherwise mark the direction neutral_or_unclear; low or "
            "moderate transferability is insufficient for a no-risk anchor."
        ),
    ),
    final_instructions=(
        *SENSITIZATION_ALIGNED_PROFILE.final_instructions,
        (
            "Do not use an analog-only negative outcome as a no-risk anchor unless "
            "its group assessment has high transferability and documents an "
            "adequately exposed validated sensitization assay with directionally "
            "consistent records. Treat low- or moderate-transferability negative "
            "analog evidence as insufficient."
        ),
    ),
)


PROFILES = {
    LEGACY_PROFILE.name: LEGACY_PROFILE,
    SENSITIZATION_ALIGNED_PROFILE.name: SENSITIZATION_ALIGNED_PROFILE,
    SENSITIZATION_NEGATIVE_TRANSFER_PROFILE.name: SENSITIZATION_NEGATIVE_TRANSFER_PROFILE,
}


def get_skin_prompt_profile(name: str) -> SkinPromptProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown Skin_Reaction prompt profile: {name}") from exc
