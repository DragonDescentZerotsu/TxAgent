"""Endpoint grouping rules for Skin_Reaction molecule-level evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tools.chembl_tool.common.text import join_text_parts, match_phrases, normalize_text


CONTEXT_DEPENDENT_GROUP = "context_dependent"


@dataclass(frozen=True)
class EndpointAssignment:
    tier: str
    endpoint_group: str
    group_id: str
    evidence_direction: str
    evidence_strength: str
    reason: str


def assign_endpoint_group(row: Mapping[str, Any]) -> EndpointAssignment:
    """Assign a Skin_Reaction evidence row to a Tier.endpoint_group bucket."""
    tier = str(row.get("assay_tier") or row.get("tier") or "").strip() or "unknown"
    standard_type = str(row.get("standard_type") or "").strip()
    endpoint_text = normalize_text(standard_type)
    context_text = _context_text(row)

    if tier == "Tier 1":
        group = _tier1_group(endpoint_text, context_text)
    elif tier == "Tier 2":
        group = _tier2_group(endpoint_text, context_text)
    elif tier == "Tier 3":
        group = _tier3_group(endpoint_text, context_text)
    elif tier == "Tier 4":
        group = _tier4_group(endpoint_text, context_text)
    elif tier == "Tier 5":
        group = _tier5_group(endpoint_text, context_text)
    else:
        group = CONTEXT_DEPENDENT_GROUP

    direction, strength = _direction_and_strength(tier, group, row)
    return EndpointAssignment(
        tier=tier,
        endpoint_group=group,
        group_id=f"{tier}.{group}",
        evidence_direction=direction,
        evidence_strength=strength,
        reason=_reason_for(tier, group, standard_type),
    )


def _tier1_group(endpoint_text: str, context_text: str) -> str:
    full_text = endpoint_text + " " + context_text
    if _has_any(full_text, ["patch test", "hript", "ript", "human maximization", "human maximisation", "contact dermatitis", "contact allergy", "dermatologic adverse", "skin adverse", "skin rash", "rash", "pruritus", "urticaria"]):
        return "human_patch_or_clinical_skin_reaction"
    if _has_any(full_text, ["local lymph node", "llna", "brdu elisa", "brdu fcm", "stimulation index", "ec3", "lymph node proliferation"]):
        return "llna_or_lymph_node_proliferation"
    if _has_any(full_text, ["guinea pig maximization", "guinea pig maximisation", "gpmt", "buehler"]):
        return "guinea_pig_sensitization"
    if _has_any(full_text, ["dermal irritation", "skin irritation", "dermal toxicity", "skin toxicity", "draize", "erythema", "edema", "oedema", "necrosis", "ulceration"]):
        return "in_vivo_dermal_irritation_or_toxicity"
    if _has_any(full_text, ["skin sensitization", "skin sensitisation", "dermal sensitization", "dermal sensitisation"]):
        return "human_patch_or_clinical_skin_reaction"
    return CONTEXT_DEPENDENT_GROUP


def _tier2_group(endpoint_text: str, context_text: str) -> str:
    full_text = endpoint_text + " " + context_text
    if _has_any(full_text, ["dpra", "adra", "kdpra", "direct peptide reactivity", "peptide depletion", "cysteine depletion", "lysine depletion", "hapten", "protein binding", "covalent binding"]):
        return "protein_binding_or_haptenation"
    if _has_any(full_text, ["keratinosens", "lusens", "episensa", "are nrf2", "are-nrf2", "antioxidant response element", "nrf2", "keap1", "keratinocyte activation"]):
        return "keratinocyte_activation_are_nrf2"
    if _has_any(full_text, ["h clat", "h-clat", "u sens", "u-sens", "il 8 luc", "il-8 luc", "gardskin", "dendritic cell activation", "thp 1", "u937", "cd54", "cd86", "il 8 reporter"]):
        return "dendritic_cell_activation"
    if _has_any(full_text, ["glutathione depletion", "gsh reactivity", "thiol reactivity", "cysteine adduct"]):
        return "glutathione_or_thiol_reactivity"
    return CONTEXT_DEPENDENT_GROUP


def _tier3_group(endpoint_text: str, context_text: str) -> str:
    full_text = endpoint_text + " " + context_text
    if _has_any(full_text, ["phototoxicity", "photo toxicity", "photoirritation", "photo irritation", "photoallergy", "photosafety", "3t3 nru", "pif", "mean photo effect", "uva", "uvb", "photocytotoxicity"]):
        return "phototoxicity_3t3_nru_or_rhe"
    if _has_any(full_text, ["skin corrosion", "dermal corrosion", "corrosive", "irreversible tissue damage"]):
        return "skin_corrosion_rhe"
    if _has_any(full_text, ["reconstructed human epidermis", "rhe", "episkin", "epiderm", "skinethic", "labcyte", "skin irritation", "dermal irritation", "mtt skin", "tissue viability"]):
        return "skin_irritation_rhe"
    if _has_any(full_text, ["keratinocyte", "hacat", "epidermal cell", "dermal fibroblast", "skin cell"]) and _has_any(
        full_text,
        ["cytotoxicity", "viability", "cell death", "ldh", "apoptosis", "necrosis", "cc50", "ic50"],
    ):
        return "keratinocyte_or_skin_cell_cytotoxicity"
    if _has_any(full_text, ["il 1 alpha", "il-1 alpha", "il 6", "il-6", "il 8", "il-8", "tnf", "pge2", "cox 2", "barrier disruption", "oxidative stress"]) and _has_any(
        full_text,
        ["skin", "dermal", "epiderm", "keratinocyte", "hacat"],
    ):
        return "skin_inflammation_or_barrier_stress"
    return CONTEXT_DEPENDENT_GROUP


def _tier4_group(endpoint_text: str, context_text: str) -> str:
    full_text = endpoint_text + " " + context_text
    if _has_any(full_text, ["skin pampa", "artificial membrane skin", "silicone", "isopropyl myristate"]) and _has_any(
        full_text,
        ["permeability", "permeation", "retention", "uptake", "pampa"],
    ):
        return "skin_pampa_or_artificial_membrane"
    if _has_any(full_text, ["skin retention", "epidermis retention", "dermis retention", "stratum corneum", "tape stripping", "skin deposition"]):
        return "skin_retention_or_distribution"
    if _has_any(full_text, ["skin absorption", "dermal absorption", "percutaneous absorption", "skin permeation", "skin permeability", "transdermal", "franz diffusion", "diffusion cell", "flux", "logkp", "log kp", " kp ", "jmax"]):
        return "skin_permeability_or_absorption"
    return CONTEXT_DEPENDENT_GROUP


def _tier5_group(endpoint_text: str, context_text: str) -> str:
    full_text = endpoint_text + " " + context_text
    if _has_any(full_text, ["cytotoxicity", "cell viability", "cell survival", "cc50", "gi50", "ldh", "apoptosis", "necrosis"]):
        return "general_cytotoxicity_context"
    if _has_any(full_text, ["cytokine", "inflammation", "nf kb", "nf-kb", "cox", "lox", "immune activation"]):
        return "immune_or_inflammation_context"
    if _has_any(full_text, ["acne", "psoriasis", "atopic dermatitis", "wound healing", "skin whitening", "melanogenesis", "melanoma", "antimicrobial", "antifungal"]):
        return "dermatology_efficacy_context"
    if _has_any(full_text, ["receptor binding", "enzyme inhibition", "kinase", "target binding", "activity"]):
        return "target_binding_context"
    return CONTEXT_DEPENDENT_GROUP


def _direction_and_strength(tier: str, group: str, row: Mapping[str, Any]) -> tuple[str, str]:
    if group == CONTEXT_DEPENDENT_GROUP:
        return "context_dependent", "weak"
    if _is_negative_or_inactive(row):
        if tier in {"Tier 1", "Tier 2"}:
            return "argues_against_skin_reaction_risk", "moderate"
        if tier == "Tier 3":
            return "argues_against_skin_reaction_risk", "weak"
        if tier == "Tier 4":
            return "reduced_skin_exposure", "weak"
        return "context_dependent", "weak"
    if tier == "Tier 1":
        return "supports_skin_reaction_risk", "strong"
    if tier == "Tier 2":
        return "sensitization_risk", "moderate"
    if group.startswith("phototoxicity"):
        return "phototoxicity_risk", "strong"
    if group in {"skin_irritation_rhe", "skin_corrosion_rhe"}:
        return "irritation_or_corrosion_risk", "strong"
    if group in {"keratinocyte_or_skin_cell_cytotoxicity", "skin_inflammation_or_barrier_stress"}:
        return "local_skin_damage_risk", "moderate"
    if tier == "Tier 4":
        return "skin_exposure_support", "moderate"
    return "context_dependent", "weak"


def _context_text(row: Mapping[str, Any]) -> str:
    return join_text_parts(
        row.get("assay_description") or row.get("description"),
        row.get("assay_type"),
        row.get("assay_cell_type"),
        row.get("assay_tissue"),
        row.get("organism"),
        row.get("target_pref_name"),
        row.get("target_genes"),
        row.get("target_synonyms"),
        row.get("assay_standard_types"),
        row.get("matched_keywords"),
        row.get("matched_endpoints"),
        row.get("activity_comment"),
        row.get("standard_type"),
    )


def _is_negative_or_inactive(row: Mapping[str, Any]) -> bool:
    text = join_text_parts(row.get("activity_comment"), row.get("standard_relation"))
    return _has_any(
        text,
        [
            "inactive",
            "not active",
            "negative",
            "non sensitizer",
            "non-sensitizer",
            "not sensitizing",
            "non irritant",
            "non-irritant",
            "not irritant",
            "not phototoxic",
            "no phototoxicity",
            "no effect",
        ],
    )


def _reason_for(tier: str, group: str, standard_type: str) -> str:
    if group == CONTEXT_DEPENDENT_GROUP:
        return f"{tier} evidence could not be assigned to a specific Skin_Reaction endpoint group."
    endpoint = f" endpoint={standard_type}" if standard_type else ""
    return f"Assigned {tier}.{group} from Skin_Reaction assay context and endpoint.{endpoint}"


def _has_any(text: str, phrases: list[str] | tuple[str, ...] | set[str]) -> bool:
    return bool(match_phrases(text, phrases))

