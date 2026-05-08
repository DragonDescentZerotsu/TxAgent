"""Endpoint grouping rules for BBB Martins molecule-level evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tools.chembl_tool.common.text import contains_phrase, join_text_parts, normalize_text


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
    """Assign a BBB evidence row to a Tier.endpoint_group bucket."""
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
    else:
        group = CONTEXT_DEPENDENT_GROUP

    direction, strength = _direction_and_strength(tier, group, endpoint_text)
    return EndpointAssignment(
        tier=tier,
        endpoint_group=group,
        group_id=f"{tier}.{group}",
        evidence_direction=direction,
        evidence_strength=strength,
        reason=_reason_for(tier, group, standard_type),
    )


def _tier1_group(endpoint_text: str, context_text: str) -> str:
    if _endpoint_is(
        endpoint_text,
        [
            "bpr",
            "brain plasma",
            "brain/plasma",
            "b p",
            "bbr",
            "ratio auc",
        ],
    ):
        return "direct_brain_plasma"
    if endpoint_text == "ratio" and _has_any(
        context_text,
        ["brain plasma", "brain to plasma", "brain blood", "brain to blood", "brain/plasma"],
    ):
        return "direct_brain_plasma"
    if _has_any(endpoint_text, ["k p uu brain", "k p uu csf", "kp", "fu"]):
        return "direct_unbound_brain"
    if _endpoint_is(
        endpoint_text,
        ["logbb", "log bb", "brain level", "brain concentration", "brain penetration index", "bpi"],
    ):
        return "direct_logbb_or_brain_level"
    if _endpoint_is(endpoint_text, ["drug uptake", "drug uptake free", "uptake", "brain uptake"]):
        return "direct_brain_uptake_or_perfusion"
    return CONTEXT_DEPENDENT_GROUP


def _tier2_group(endpoint_text: str, context_text: str) -> str:
    del context_text
    if _endpoint_is(endpoint_text, ["papp", "logpapp", "logp app", "papp e 6"]) or endpoint_text.startswith("papp "):
        return "passive_papp"
    if _endpoint_is(endpoint_text, ["caco 2 papp", "caco 2 permeability", "pcaco2"]):
        return "passive_caco2"
    if _endpoint_is(
        endpoint_text,
        [
            "permeability",
            "permeability coefficient",
            "peff",
            "logpeff",
            "log pe",
            "pc",
            "pm",
            "pbbb",
        ],
    ):
        return "passive_generic_permeability"
    if _endpoint_is(endpoint_text, ["drug transport", "drug recovery"]):
        return "passive_transport_or_recovery"
    return CONTEXT_DEPENDENT_GROUP


def _tier3_group(endpoint_text: str, context_text: str) -> str:
    if _endpoint_is(endpoint_text, ["efflux ratio", "ratio papp", "ratio_papp", "papp a to b mean", "papp b to a mean"]):
        return "efflux_functional_ratio_or_bidirectional"
    if endpoint_text in {"ratio", "papp"} and _has_any(
        context_text,
        ["efflux", "bidirectional", "basolateral to apical", "apical to basolateral", "mdck mdr1", "mdr1 mdck"],
    ):
        return "efflux_functional_ratio_or_bidirectional"
    if _endpoint_is(endpoint_text, ["drug transport", "drug uptake", "activity", "flu intensity", "rfu", "fluorescence"]):
        if _has_any(context_text, ["efflux", "accumulation", "transport", "rhodamine", "calcein", "hoechst", "mitoxantrone"]):
            return "efflux_transport_or_accumulation"
    if _has_any(endpoint_text, ["atpase", "jc 1 accumulation"]):
        return "efflux_atpase_or_probe"
    if _endpoint_is(
        endpoint_text,
        ["inhibition", "ic50", "ki", "ec50", "kd", "km", "kon", "k off", "ratio ic50", "ratio ec50", "fc"],
    ):
        return "efflux_inhibition_or_binding"
    return CONTEXT_DEPENDENT_GROUP


def _tier4_group(endpoint_text: str, context_text: str) -> str:
    if _endpoint_is(endpoint_text, ["drug uptake", "uptake", "drug transport", "transport"]):
        return "influx_functional_uptake_or_transport"
    if _endpoint_is(endpoint_text, ["activity"]) and _has_any(context_text, ["uptake", "transport", "substrate"]):
        return "influx_functional_uptake_or_transport"
    if _endpoint_is(endpoint_text, ["km", "vmax", "jmax", "kin"]):
        return "influx_kinetic_or_substrate"
    if _endpoint_is(
        endpoint_text,
        ["inhibition", "ic50", "ki", "kd", "kon", "k off", "ec50", "ratio ic50"],
    ):
        return "influx_inhibition_or_binding"
    return CONTEXT_DEPENDENT_GROUP


def _direction_and_strength(tier: str, group: str, endpoint_text: str) -> tuple[str, str]:
    del endpoint_text
    if group == CONTEXT_DEPENDENT_GROUP:
        return "context_dependent", "weak"
    if tier == "Tier 1":
        return "supports_bbb_crossing", "strong"
    if tier == "Tier 2":
        return "permeability_support", "moderate"
    if tier == "Tier 3":
        if group == "efflux_inhibition_or_binding":
            return "context_dependent", "weak"
        return "efflux_risk", "moderate"
    if tier == "Tier 4":
        if group == "influx_inhibition_or_binding":
            return "context_dependent", "weak"
        return "influx_support", "moderate"
    return "unknown_direction", "weak"


def _reason_for(tier: str, group: str, standard_type: str) -> str:
    if group == CONTEXT_DEPENDENT_GROUP:
        return f"{tier} endpoint `{standard_type}` 缺少足够明确的 endpoint family，作为上下文弱证据。"
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


def _endpoint_is(endpoint_text: str, candidates: list[str]) -> bool:
    return any(endpoint_text == normalize_text(candidate) for candidate in candidates)


def _has_any(text: str, phrases: list[str]) -> bool:
    return any(contains_phrase(text, phrase) for phrase in phrases)
