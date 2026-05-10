"""Endpoint grouping rules for Bioavailability_Ma molecule-level evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tools.chembl_tool.common.text import contains_phrase, join_text_parts, normalize_text
from tools.chembl_tool.tasks.bioavailability_ma.constants import BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT


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
    """Assign a bioavailability evidence row to a Tier.endpoint_group bucket."""
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
    elif tier == "Tier 6":
        group = _tier6_group(endpoint_text, context_text)
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
    if _has_any(endpoint_text, ["cl/f", "cl f", "vd/f", "vd f", "vss/f", "vss f", "vc/f", "vc f", "fmax", "fapp"]):
        return CONTEXT_DEPENDENT_GROUP
    if _has_any(endpoint_text, ["bioavailability", "absolute bioavailability", "oral bioavailability", "%f", "f%"]):
        return "direct_absolute_bioavailability"
    if endpoint_text == "f" and _has_any(context_text, ["bioavailability", "oral", "po", "oral iv", "oral/iv"]):
        return "direct_absolute_bioavailability"
    if _has_any(context_text, ["oral/iv", "oral iv", "oral to iv", "po iv"]) and _has_any(
        endpoint_text + " " + context_text,
        ["auc ratio", "exposure ratio", "bioavailability ratio", "ratio"],
    ):
        return "direct_oral_iv_exposure_ratio"
    return CONTEXT_DEPENDENT_GROUP


def _tier2_group(endpoint_text: str, context_text: str) -> str:
    if _has_any(endpoint_text, ["auc", "auc0 t", "auc0 inf", "aucinf", "plasma exposure"]):
        return "oral_auc_exposure" if _has_oral_context(context_text) else CONTEXT_DEPENDENT_GROUP
    if _has_any(endpoint_text, ["cmax", "maximum plasma concentration", "peak plasma concentration"]):
        return "oral_cmax_exposure" if _has_oral_context(context_text) else CONTEXT_DEPENDENT_GROUP
    if _has_any(endpoint_text, ["fraction absorbed", "percent absorbed", "fa", "fabs", "hia"]) or _has_any(
        context_text,
        ["human intestinal absorption", "fraction absorbed", "percent absorbed", "oral absorption"],
    ):
        return "absorption_fraction_or_hia"
    if _has_any(endpoint_text, ["peff", "effective permeability", "intestinal permeability"]) or _has_any(
        context_text,
        ["intestinal perfusion", "jejunal permeability", "in situ intestinal perfusion"],
    ):
        return "in_vivo_intestinal_permeability"
    if _has_any(endpoint_text, ["uptake", "transport", "drug transport"]) and _has_any(
        context_text,
        ["intestinal", "absorptive", "portal"],
    ):
        return "in_vivo_intestinal_uptake_or_transport"
    return CONTEXT_DEPENDENT_GROUP


def _tier3_group(endpoint_text: str, context_text: str) -> str:
    if _has_any(
        context_text,
        [
            "cytotoxicity",
            "cell viability",
            "cell survival",
            "proliferation",
            "mtt assay",
            "mts assay",
            "antitumor",
            "tumor growth",
            "xenograft",
            "reversal of multidrug resistance",
            "binding affinity",
            "pull down",
        ],
    ):
        return CONTEXT_DEPENDENT_GROUP
    if _has_any(context_text, ["pampa", "parallel artificial membrane"]) or _has_any(endpoint_text, ["pampa"]):
        return "pampa_or_artificial_membrane"
    if _has_any(endpoint_text, ["efflux ratio"]) or (
        endpoint_text == "ratio" and _has_any(context_text, ["efflux", "bidirectional", "papp"])
    ):
        return "cell_bidirectional_efflux_ratio"
    if _has_any(endpoint_text, ["papp b to a", "papp basolateral to apical"]) or _has_any(
        context_text,
        ["basolateral to apical", "b to a", "secretory"],
    ):
        return "cell_secretory_permeability"
    if _has_any(endpoint_text, ["papp", "logpapp", "apparent permeability", "permeability coefficient", "permeability"]):
        return "cell_permeability_papp"
    if _has_any(context_text, ["caco 2", "caco2", "mdck"]) and _has_any(
        endpoint_text + " " + context_text,
        ["permeability", "papp", "apical to basolateral", "a to b"],
    ):
        return "cell_permeability_papp"
    if _has_any(context_text, ["p gp substrate", "pgp substrate", "abcb1 substrate", "bcrp substrate", "abcg2 substrate"]):
        return "transporter_substrate_or_efflux"
    if _has_any(endpoint_text, ["substrate", "transport", "drug transport"]) and _has_any(
        context_text,
        ["p gp", "p-gp", "abcb1", "mdr1", "bcrp", "abcg2", "mrp", "abcc", "efflux"],
    ):
        return "transporter_substrate_or_efflux"
    if _has_any(endpoint_text, ["inhibition", "ic50", "ki", "ec50", "kd", "binding"]) and _has_any(
        context_text,
        ["p gp", "p-gp", "abcb1", "mdr1", "bcrp", "abcg2", "mrp", "abcc"],
    ):
        return "transporter_inhibition_or_binding"
    return CONTEXT_DEPENDENT_GROUP


def _tier4_group(endpoint_text: str, context_text: str) -> str:
    if _has_any(
        context_text,
        [
            "cytotoxicity",
            "cell viability",
            "cell survival",
            "proliferation",
            "antitumor",
            "tumor",
            "mic",
            "binding affinity",
            "dissolution dynamic nuclear polarization",
            "thermal solubility",
        ],
    ):
        return CONTEXT_DEPENDENT_GROUP
    if _has_any(endpoint_text, ["solubility", "aqueous solubility", "kinetic solubility", "thermodynamic solubility", "logs", "log s"]):
        return "solubility"
    if _has_any(endpoint_text, ["dissolution", "dissolution rate", "percent dissolved", "dissolved"]) or _has_any(
        context_text,
        ["dissolution"],
    ):
        return "dissolution"
    if _has_any(endpoint_text, ["stability", "half life", "t1/2"]) and _has_any(
        context_text,
        ["gastric", "intestinal", "simulated gastric", "simulated intestinal", "gi fluid"],
    ):
        return "gi_or_chemical_stability"
    if _has_any(endpoint_text, ["ic50", "activity", "inhibition", "kon", "k off", "kd", "ki"]):
        return CONTEXT_DEPENDENT_GROUP
    return CONTEXT_DEPENDENT_GROUP


def _tier5_group(endpoint_text: str, context_text: str) -> str:
    if _has_any(endpoint_text, ["intrinsic clearance", "clint", "hepatic clearance", "clearance", "cl"]):
        return "intrinsic_or_hepatic_clearance"
    if _has_any(context_text, ["first pass", "first-pass", "hepatic extraction", "extraction ratio", "intestinal metabolism"]):
        return "first_pass_or_extraction"
    if _has_any(endpoint_text, ["percent remaining", "remaining", "half life", "t1/2", "metabolic stability", "turnover"]) or _has_any(
        context_text,
        ["microsomal stability", "liver microsome", "hepatocyte stability", "s9 stability", "substrate depletion", "metabolic turnover"],
    ):
        return "metabolic_stability"
    return CONTEXT_DEPENDENT_GROUP


def _tier6_group(endpoint_text: str, context_text: str) -> str:
    if _has_any(context_text, ["toxicity", "adverse event", "antifungal", "antimicrobial", "cfu", "antitumor", "tumor", "calcium entry"]):
        return CONTEXT_DEPENDENT_GROUP
    if _has_any(context_text, ["food effect", "fed fasted", "fed/fasted", "high fat meal", "fasting"]):
        return "food_effect_or_fed_fasted"
    if _has_any(context_text, ["formulation", "tablet", "capsule", "solution", "suspension", "salt form", "solid dispersion", "nanoparticle"]) and _has_any(
        context_text,
        ["bioavailability", "auc", "cmax", "oral", "formulation", "fed", "fasted"],
    ):
        return "relative_bioavailability_or_formulation"
    if _has_any(endpoint_text, ["auc ratio", "cmax ratio", "exposure ratio", "ratio"]):
        return "formulation_auc_cmax_ratio"
    if _has_any(endpoint_text, ["relative bioavailability"]):
        return "relative_bioavailability_or_formulation"
    return CONTEXT_DEPENDENT_GROUP


def _direction_and_strength(tier: str, group: str, row: Mapping[str, Any]) -> tuple[str, str]:
    if group == CONTEXT_DEPENDENT_GROUP:
        return "context_dependent", "weak"
    if tier == "Tier 1":
        direction = _direct_absolute_bioavailability_direction(group, row)
        if direction:
            return direction, "strong"
        return "unknown_direction", "strong"
    if tier == "Tier 2":
        return "absorption_support", "moderate"
    if tier == "Tier 3":
        if group in {"cell_bidirectional_efflux_ratio", "cell_secretory_permeability", "transporter_substrate_or_efflux"}:
            return "transporter_efflux_risk", "moderate"
        if group == "transporter_inhibition_or_binding":
            return "context_dependent", "weak"
        return "permeability_support", "moderate"
    if tier == "Tier 4":
        return "unknown_direction", "moderate"
    if tier == "Tier 5":
        return "unknown_direction", "moderate"
    if tier == "Tier 6":
        return "context_dependent", "weak"
    return "unknown_direction", "weak"


def _direct_absolute_bioavailability_direction(group: str, row: Mapping[str, Any]) -> str:
    if group != "direct_absolute_bioavailability":
        return ""
    value = _parse_float(row.get("standard_value"))
    if value is None:
        return ""
    relation = str(row.get("standard_relation") or "").strip()
    cutoff = BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT
    if relation in {"", "=", "~"}:
        return "supports_high_bioavailability" if value >= cutoff else "argues_against_high_bioavailability"
    if relation in {">", ">="} and value >= cutoff:
        return "supports_high_bioavailability"
    if relation in {"<", "<="} and value <= cutoff:
        return "argues_against_high_bioavailability"
    return ""


def _parse_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _reason_for(tier: str, group: str, standard_type: str) -> str:
    if group == CONTEXT_DEPENDENT_GROUP:
        return f"{tier} endpoint `{standard_type}` lacks enough bioavailability-specific context; keep as weak context."
    return f"{tier} endpoint `{standard_type}` assigned to `{group}`."


def _context_text(row: Mapping[str, Any]) -> str:
    return join_text_parts(
        row.get("assay_description") or row.get("description"),
        row.get("target_pref_name"),
        row.get("target_genes"),
        row.get("target_synonyms"),
        row.get("activity_comment"),
        row.get("standard_type"),
    )


def _has_oral_context(text: str) -> bool:
    return _has_any(text, ["oral", " po ", "p o", "gavage", "intragastric"])


def _has_any(text: str, phrases: list[str]) -> bool:
    return any(contains_phrase(text, phrase) for phrase in phrases)
