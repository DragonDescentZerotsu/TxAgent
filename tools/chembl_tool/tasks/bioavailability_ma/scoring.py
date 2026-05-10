"""Scoring logic for Bioavailability_Ma assay screening."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tools.chembl_tool.common.text import contains_phrase, join_text_parts, match_phrases

from .rules import (
    BIOAVAILABILITY_RULES,
    CYP_TARGET_GENES,
    EFFLUX_TARGET_GENES,
    EFFLUX_TARGET_PHRASES,
    NEGATIVE_KEYWORDS,
    NON_ORAL_ROUTE_TERMS,
    ORAL_ROUTE_TERMS,
    TRANSPORTER_TARGETS,
    WEAK_TERMS,
)


TIER_ORDER = {
    "direct_absolute_bioavailability": 1,
    "in_vivo_oral_exposure_absorption": 2,
    "in_vitro_permeability_efflux": 3,
    "solubility_dissolution_gi_stability": 4,
    "metabolism_first_pass_clearance": 5,
    "formulation_food_relative_bioavailability": 6,
}


@dataclass
class MatchResult:
    tier_matches: dict[str, bool] = field(default_factory=dict)
    matched_keywords: list[str] = field(default_factory=list)
    matched_endpoints: list[str] = field(default_factory=list)
    matched_targets: list[str] = field(default_factory=list)
    negative_flags: list[str] = field(default_factory=list)
    weak_context_flags: list[str] = field(default_factory=list)


@dataclass
class ScoredAssay:
    tier_key: str
    tier: str
    score: int
    keep: bool
    reason: str
    matches: MatchResult


def match_rules(row: dict[str, Any]) -> MatchResult:
    assay_text = _assay_text(row)
    endpoint_text = _endpoint_text(row)
    genes = {str(gene).upper() for gene in row.get("target_genes", []) if gene}
    target_text = join_text_parts(
        row.get("target_pref_name"),
        " ".join(row.get("target_synonyms", []) or []),
        " ".join(row.get("component_descriptions", []) or []),
    )

    result = MatchResult()
    result.negative_flags = match_phrases(assay_text, NEGATIVE_KEYWORDS)

    efflux_targets = sorted(genes & EFFLUX_TARGET_GENES)
    cyp_targets = sorted(genes & CYP_TARGET_GENES)
    if efflux_targets or _has_any(target_text, EFFLUX_TARGET_PHRASES):
        result.matched_targets.extend(efflux_targets or ["EFFLUX_TRANSPORTER_TARGET"])
    if cyp_targets:
        result.matched_targets.extend(cyp_targets)

    for tier_key, rule in BIOAVAILABILITY_RULES.items():
        keyword_hits = match_phrases(assay_text, rule["keywords"])
        endpoint_hits = match_phrases(endpoint_text, rule["endpoints"])
        target_hit = False
        if tier_key == "in_vitro_permeability_efflux":
            target_hit = bool(efflux_targets) or _has_any(target_text, EFFLUX_TARGET_PHRASES)
        elif tier_key == "metabolism_first_pass_clearance":
            target_hit = bool(cyp_targets)

        valid = _has_valid_context(tier_key, assay_text, endpoint_text, keyword_hits, endpoint_hits, target_hit)
        result.tier_matches[tier_key] = valid
        if valid:
            result.matched_keywords.extend(keyword_hits)
            result.matched_endpoints.extend(endpoint_hits)

    result.matched_keywords = sorted(set(result.matched_keywords))
    result.matched_endpoints = sorted(set(result.matched_endpoints))
    result.matched_targets = sorted(set(result.matched_targets))
    result.weak_context_flags = sorted(_weak_context_flags(assay_text, endpoint_text, result))
    return result


def score_assay(row: dict[str, Any], matches: MatchResult, min_score: int = 40) -> ScoredAssay:
    exclusion_reason = exclusion_reason_for(row, matches)
    if exclusion_reason:
        return ScoredAssay("excluded", "excluded", 0, False, exclusion_reason, matches)

    scores: dict[str, int] = {}
    for tier_key, matched in matches.tier_matches.items():
        if not matched:
            continue
        score = int(BIOAVAILABILITY_RULES[tier_key]["base_score"])
        score += min(20, 3 * len(matches.matched_keywords))
        score += 15 if matches.matched_endpoints else 0
        if tier_key == "direct_absolute_bioavailability" and matches.matched_endpoints:
            score += 15
        if tier_key == "in_vitro_permeability_efflux" and any(target in TRANSPORTER_TARGETS for target in matches.matched_targets):
            score += 15
        if tier_key == "metabolism_first_pass_clearance" and any(str(target).startswith("CYP") for target in matches.matched_targets):
            score += 8
        if _as_int(row.get("confidence_score")) >= 8:
            score += 10
        if row.get("relationship_type") == "D":
            score += 5
        n_unique = _as_int(row.get("n_unique_molecules"))
        if n_unique >= 100:
            score += 10
        elif n_unique >= 20:
            score += 5
        if matches.negative_flags and not _has_strong_adme_positive(matches):
            score -= 60
        scores[tier_key] = score

    if not scores:
        return ScoredAssay(
            tier_key="none",
            tier="none",
            score=-60 if matches.negative_flags else 0,
            keep=False,
            reason="未命中 oral bioavailability 相关 assay 证据。",
            matches=matches,
        )

    tier_key = max(scores, key=lambda key: (scores[key], -TIER_ORDER.get(key, 99)))
    tier = str(BIOAVAILABILITY_RULES[tier_key]["tier"])
    score = scores[tier_key]
    keep = should_keep_assay(row, ScoredAssay(tier_key, tier, score, False, "", matches), min_score)
    return ScoredAssay(tier_key, tier, score, keep, _reason_for(tier_key, matches), matches)


def should_keep_assay(row: dict[str, Any], scored: ScoredAssay, min_score: int = 40) -> bool:
    del row
    return scored.score >= min_score and any(scored.matches.tier_matches.values())


def scored_row(row: dict[str, Any], min_score: int = 40) -> dict[str, Any]:
    matches = match_rules(row)
    scored = score_assay(row, matches, min_score=min_score)
    out = dict(row)
    out.update(
        {
            "tier": scored.tier,
            "score": scored.score,
            "matched_keywords": matches.matched_keywords,
            "matched_endpoints": matches.matched_endpoints,
            "matched_targets": matches.matched_targets,
            "negative_flags": matches.negative_flags,
            "weak_context_flags": matches.weak_context_flags,
            "reason": scored.reason,
            "keep_for_bioavailability_reasoning": scored.keep,
        }
    )
    return out


def exclusion_reason_for(row: dict[str, Any], matches: MatchResult) -> str:
    assay_text = _assay_text(row)
    endpoint_text = _endpoint_text(row)
    full_text = assay_text + " " + endpoint_text

    if _is_nonoral_pk_noise(full_text, matches):
        return "排除非口服 route PK endpoint；没有 oral 或 oral/IV bioavailability context。"

    if matches.tier_matches.get("direct_absolute_bioavailability") and _is_apparent_pk_parameter_with_respect_to_f(full_text, endpoint_text):
        return "排除 CL/F、Vd/F、Vss/F、Fmax 等 apparent PK parameter；不是 direct absolute oral bioavailability。"

    if matches.tier_matches.get("in_vitro_permeability_efflux") and _is_permeability_or_efflux_noise(full_text, endpoint_text):
        return "排除 Caco-2/MDCK/transporter 相关 cytotoxicity、cell viability、binding affinity 或 target activity 噪声。"

    if matches.tier_matches.get("solubility_dissolution_gi_stability") and _is_solubility_protocol_or_enzyme_noise(full_text, endpoint_text):
        return "排除 protocol/stock solution 或 enzyme assay 中的 solubility 误命中；不是 solubility/dissolution/GI stability endpoint。"

    if matches.tier_matches.get("formulation_food_relative_bioavailability") and _is_formulation_context_noise(full_text, endpoint_text):
        return "排除 toxicity/efficacy assay 中的 solution/suspension/fasted/formulation 误命中。"

    if matches.tier_matches.get("metabolism_first_pass_clearance") and _is_cyp_inhibition_without_metabolism_context(full_text):
        return "排除 CYP inhibition/binding 误命中；这不等于 substrate depletion、metabolic stability 或 first-pass metabolism evidence。"

    if matches.tier_matches.get("in_vitro_permeability_efflux") and _is_transporter_inhibition_without_substrate_context(full_text):
        return "排除 transporter inhibition/binding 误命中；这不等于 transporter substrate 或 intestinal efflux evidence。"

    if _is_ppb_without_oral_exposure_context(full_text):
        return "排除 plasma protein binding；PPB 不是 oral bioavailability 或 absorption 的直接 evidence。"

    if matches.negative_flags and not _has_strong_adme_positive(matches):
        return "排除 target potency/cell activity/cytotoxicity 等非 ADME assay。"

    return ""


def _assay_text(row: dict[str, Any]) -> str:
    return join_text_parts(
        row.get("description"),
        row.get("assay_cell_type"),
        row.get("assay_tissue"),
        row.get("assay_organism"),
        row.get("assay_test_type"),
        row.get("assay_category"),
        row.get("target_pref_name"),
        " ".join(row.get("target_genes", []) or []),
        " ".join(row.get("target_synonyms", []) or []),
        " ".join(row.get("component_descriptions", []) or []),
    )


def _endpoint_text(row: dict[str, Any]) -> str:
    return join_text_parts(" ".join(row.get("standard_types", []) or []))


def _has_valid_context(
    tier_key: str,
    assay_text: str,
    endpoint_text: str,
    keyword_hits: list[str],
    endpoint_hits: list[str],
    target_hit: bool,
) -> bool:
    full_text = assay_text + " " + endpoint_text

    if tier_key == "direct_absolute_bioavailability":
        if _has_any(full_text, ["relative bioavailability", "food effect", "fed fasted", "formulation"]):
            return False
        if _is_apparent_pk_parameter_with_respect_to_f(full_text, endpoint_text):
            return False
        direct_context = _has_any(full_text, ["oral bioavailability", "absolute bioavailability", "bioavailability", "oral/iv", "oral iv", "oral to iv"])
        return bool(keyword_hits or endpoint_hits) and direct_context and not _is_nonoral_without_oral_iv(full_text)

    if tier_key == "in_vivo_oral_exposure_absorption":
        oral_exposure = _has_oral_context(full_text) and _has_any(
            full_text,
            ["auc", "cmax", "plasma exposure", "systemic exposure", "plasma concentration"],
        )
        absorption = _has_any(
            full_text,
            [
                "human intestinal absorption",
                "intestinal absorption",
                "fraction absorbed",
                "percent absorbed",
                "oral absorption",
                "peff",
                "effective permeability",
                "intestinal permeability",
                "intestinal perfusion",
            ],
        )
        return bool(keyword_hits or endpoint_hits) and (oral_exposure or absorption)

    if tier_key == "in_vitro_permeability_efflux":
        if _is_permeability_or_efflux_noise(full_text, endpoint_text):
            return False
        permeability_model = _has_any(full_text, ["caco 2", "caco2", "mdck", "pampa", "parallel artificial membrane", "papp", "apparent permeability"])
        efflux_context = _has_any(full_text, ["efflux ratio", "bidirectional", "apical to basolateral", "basolateral to apical"])
        transporter_substrate = target_hit and _has_any(full_text, ["substrate", "transport", "efflux", "papp"])
        return bool(keyword_hits or endpoint_hits or target_hit) and (permeability_model or efflux_context or transporter_substrate)

    if tier_key == "solubility_dissolution_gi_stability":
        if _is_solubility_protocol_or_enzyme_noise(full_text, endpoint_text):
            return False
        solubility_or_dissolution = _has_any(full_text, ["solubility", "logs", "log s", "dissolution", "dissolved"])
        gi_stability = _has_any(full_text, ["gastric", "intestinal", "simulated gastric", "simulated intestinal", "gi fluid"]) and _has_any(
            full_text,
            ["stability", "half life", "t1/2"],
        )
        return bool(keyword_hits or endpoint_hits) and (solubility_or_dissolution or gi_stability)

    if tier_key == "metabolism_first_pass_clearance":
        metabolism_context = _has_any(
            full_text,
            [
                "microsomal stability",
                "liver microsome",
                "hepatocyte stability",
                "s9 stability",
                "metabolic stability",
                "metabolic turnover",
                "substrate depletion",
                "intrinsic clearance",
                "clint",
                "hepatic clearance",
                "first pass",
                "first-pass",
                "hepatic extraction",
            ],
        )
        cyp_metabolism = target_hit and _has_any(full_text, ["substrate", "metabolism", "turnover", "depletion", "metabolite formation"])
        return bool(keyword_hits or endpoint_hits or target_hit) and (metabolism_context or cyp_metabolism)

    if tier_key == "formulation_food_relative_bioavailability":
        if _is_formulation_context_noise(full_text, endpoint_text):
            return False
        specific_context = _has_any(
            full_text,
            ["relative bioavailability", "food effect", "fed fasted", "fed/fasted", "formulation bioavailability", "formulation comparison"],
        )
        dosage_form_context = _has_any(
            full_text,
            ["tablet", "capsule", "solution", "suspension", "salt form", "solid dispersion", "nanoparticle"],
        ) and _has_any(full_text, ["bioavailability", "auc", "cmax", "oral", "formulation", "fed", "fasted"])
        return bool(keyword_hits or endpoint_hits) and (specific_context or dosage_form_context)

    return False


def _weak_context_flags(assay_text: str, endpoint_text: str, matches: MatchResult) -> set[str]:
    full_text = assay_text + " " + endpoint_text
    if any(matches.tier_matches.values()):
        return set()
    return {term for term in WEAK_TERMS if contains_phrase(full_text, term)}


def _has_strong_adme_positive(matches: MatchResult) -> bool:
    evidence = set(matches.matched_keywords) | set(matches.matched_endpoints)
    strong_terms = {
        "bioavailability",
        "absolute bioavailability",
        "oral bioavailability",
        "fraction absorbed",
        "human intestinal absorption",
        "caco 2",
        "caco2",
        "mdck",
        "pampa",
        "papp",
        "solubility",
        "dissolution",
        "microsomal stability",
        "hepatocyte stability",
        "intrinsic clearance",
        "clint",
    }
    return bool(evidence & strong_terms)


def _reason_for(tier_key: str, matches: MatchResult) -> str:
    evidence = matches.matched_endpoints or matches.matched_keywords or matches.matched_targets
    evidence_text = "、".join(evidence[:4]) if evidence else "相关证据"
    if tier_key == "direct_absolute_bioavailability":
        return f"命中 {evidence_text}，属于 absolute oral bioavailability 直接证据。"
    if tier_key == "in_vivo_oral_exposure_absorption":
        return f"命中 {evidence_text}，属于体内口服暴露或肠吸收证据。"
    if tier_key == "in_vitro_permeability_efflux":
        return f"命中 {evidence_text}，属于体外通透或肠道外排相关证据。"
    if tier_key == "solubility_dissolution_gi_stability":
        return f"命中 {evidence_text}，属于溶解度、溶出或 GI stability 证据。"
    if tier_key == "metabolism_first_pass_clearance":
        return f"命中 {evidence_text}，属于代谢稳定性、首过或清除证据。"
    if tier_key == "formulation_food_relative_bioavailability":
        return f"命中 {evidence_text}，属于 formulation、food effect 或 relative bioavailability 上下文证据。"
    return "未命中 oral bioavailability 相关 assay 证据。"


def _is_nonoral_pk_noise(text: str, matches: MatchResult) -> bool:
    pk_like = _has_any(text, ["auc", "cmax", "tmax", "plasma concentration", "clearance", "half life", "t1/2"])
    if not pk_like:
        return False
    if _has_any(text, ["oral/iv", "oral iv", "oral to iv", "po iv", "bioavailability"]):
        return False
    if _has_oral_context(text):
        return False
    return _has_any(text, NON_ORAL_ROUTE_TERMS) or matches.tier_matches.get("in_vivo_oral_exposure_absorption", False)


def _is_nonoral_without_oral_iv(text: str) -> bool:
    if _has_any(text, ["oral/iv", "oral iv", "oral to iv", "po iv"]):
        return False
    return _has_any(text, NON_ORAL_ROUTE_TERMS) and not _has_oral_context(text)


def _is_apparent_pk_parameter_with_respect_to_f(text: str, endpoint_text: str) -> bool:
    apparent_endpoints = [
        "cl/f",
        "cl f",
        "vd/f",
        "vd f",
        "vss/f",
        "vss f",
        "vc/f",
        "vc f",
        "fmax",
        "fapp",
        "max achievable bioavailability",
    ]
    if _has_any(endpoint_text, apparent_endpoints):
        return True
    return _has_any(text, ["clearance with respect to oral bioavailability", "volume of distribution with respect to oral bioavailability"])


def _is_permeability_or_efflux_noise(text: str, endpoint_text: str) -> bool:
    if _has_any(text, ["lucifer yellow", "barrier integrity", "paracellular marker"]) and not _has_any(
        text,
        ["compound permeability", "drug permeability", "papp of compound"],
    ):
        return True
    noisy_biology = _has_any(
        text,
        [
            "cytotoxicity",
            "cell viability",
            "cell survival",
            "proliferation",
            "mtt assay",
            "mts assay",
            "antitumor",
            "anti tumor",
            "tumor growth",
            "xenograft",
            "reversal of multidrug resistance",
            "multidrug resistant",
            "gi50",
            "cc50",
            "mic",
            "antimicrobial",
            "antifungal",
            "cfu",
            "pull down",
        ],
    )
    if noisy_biology:
        return True
    if _has_any(text, ["binding affinity", "receptor binding", "target binding"]):
        return True
    if _has_any(endpoint_text, ["activity", "ic50", "ec50", "ki", "kd", "inhibition", "gi50", "cc50", "fc"]) and not _has_any(
        text,
        [
            "efflux ratio",
            "bidirectional",
            "apical to basolateral",
            "basolateral to apical",
            "papp",
            "apparent permeability",
            "substrate transport",
            "drug transport",
            "uptake",
        ],
    ):
        return True
    return False


def _is_solubility_protocol_or_enzyme_noise(text: str, endpoint_text: str) -> bool:
    if _has_any(
        text,
        [
            "cytotoxicity",
            "cell viability",
            "cell survival",
            "proliferation",
            "antitumor",
            "tumor",
            "mic",
            "binding affinity",
            "receptor binding",
            "target binding",
            "dissolution dynamic nuclear polarization",
            "thermal solubility",
        ],
    ):
        return True
    if _has_any(endpoint_text, ["solubility", "kinetic solubility", "thermodynamic solubility", "logs", "log s", "dissolution", "stability"]):
        return False
    if _has_any(
        text,
        [
            "enzyme inhibition",
            "kinase assay",
            "functional assay",
            "receptor",
            "binding studies",
            "binding affinity",
            "protein solubility",
            "improve solubility",
            "increased solubility",
            "solubility decreased",
            "prepared as",
            "stock",
            "stock solution",
            "dmso",
        ],
    ) and _has_any(endpoint_text, ["ic50", "activity", "inhibition", "kon", "k off", "koff", "kd", "ki"]):
        return True
    return False


def _is_formulation_context_noise(text: str, endpoint_text: str) -> bool:
    if _has_any(text, ["relative bioavailability", "food effect", "fed fasted", "fed/fasted", "formulation bioavailability", "formulation comparison"]):
        return False
    if _has_any(endpoint_text, ["auc ratio", "cmax ratio", "relative bioavailability"]):
        return False
    if _has_any(
        text,
        [
            "toxicity",
            "adverse event",
            "liver function",
            "vital signs",
            "ecg",
            "physical examination",
            "antifungal",
            "antimicrobial",
            "cfu",
            "efficacy",
            "antitumor",
            "tumor",
            "calcium entry",
            "functional activity",
        ],
    ):
        return True
    if _has_any(endpoint_text, ["activity", "ic50", "ec50", "ki", "kd", "kon", "k off", "t1/2", "cl/f", "vc/f"]):
        return True
    return False


def _is_cyp_inhibition_without_metabolism_context(text: str) -> bool:
    cyp_context = _has_any(text, ["cyp", "cytochrome p450", "cyp3a4", "cyp2d6", "cyp2c9", "cyp2c19", "cyp1a2"])
    inhibition = _has_any(text, ["inhibition", "inhibitor", "ic50", "ki", "binding"])
    metabolism = _has_any(text, ["substrate depletion", "metabolic stability", "metabolic turnover", "metabolite formation", "microsomal stability", "hepatocyte stability"])
    return cyp_context and inhibition and not metabolism


def _is_transporter_inhibition_without_substrate_context(text: str) -> bool:
    transporter = _has_any(text, ["p gp", "p-gp", "pgp", "abcb1", "mdr1", "bcrp", "abcg2", "mrp", "abcc"])
    inhibition = _has_any(text, ["inhibition", "inhibitor", "ic50", "ki", "binding"])
    substrate = _has_any(text, ["substrate", "efflux", "bidirectional", "papp", "transport", "drug transport"])
    return transporter and inhibition and not substrate


def _is_ppb_without_oral_exposure_context(text: str) -> bool:
    ppb = _has_any(text, ["plasma protein binding", "protein binding", "albumin binding", "serum binding", "fu plasma"])
    return ppb and not _has_any(text, ["oral bioavailability", "oral exposure", "bioavailability"])


def _has_oral_context(text: str) -> bool:
    return _has_any(text, ORAL_ROUTE_TERMS)


def _has_any(text: str, phrases: list[str]) -> bool:
    return any(contains_phrase(text, phrase) for phrase in phrases)


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
