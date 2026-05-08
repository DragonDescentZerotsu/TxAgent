"""Scoring logic for BBB Martins assay screening."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tools.chembl_tool.common.text import contains_phrase, join_text_parts, match_phrases

from .rules import (
    BBB_RULES,
    EFFLUX_TARGET_GENES,
    EFFLUX_TARGET_PHRASES,
    INFLUX_TARGET_GENES,
    INFLUX_TARGET_PHRASES,
    NEGATIVE_KEYWORDS,
    TRANSPORTER_TARGETS,
    WEAK_TERMS,
)


TIER_ORDER = {
    "direct_bbb": 1,
    "efflux": 3,
    "passive_permeability": 2,
    "influx": 4,
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
    influx_targets = sorted(genes & INFLUX_TARGET_GENES)
    result.matched_targets.extend(efflux_targets + influx_targets)

    for tier_key, rule in BBB_RULES.items():
        keyword_hits = match_phrases(assay_text, rule["keywords"])
        endpoint_hits = match_phrases(endpoint_text, rule["endpoints"])
        target_hit = False
        if tier_key == "efflux":
            target_hit = bool(efflux_targets) or _has_any(target_text, EFFLUX_TARGET_PHRASES)
        elif tier_key == "influx":
            target_hit = bool(influx_targets) or _has_any(target_text, INFLUX_TARGET_PHRASES)

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
        return ScoredAssay(
            tier_key="excluded",
            tier="excluded",
            score=0,
            keep=False,
            reason=exclusion_reason,
            matches=matches,
        )

    scores: dict[str, int] = {}
    for tier_key, matched in matches.tier_matches.items():
        if not matched:
            continue
        score = int(BBB_RULES[tier_key]["base_score"])
        score += min(20, 3 * len(matches.matched_keywords))
        score += 15 if matches.matched_endpoints else 0
        if tier_key == "direct_bbb" and len(matches.matched_endpoints) >= 2:
            score += 10
        if any(target in {"ABCB1", "ABCG2"} for target in matches.matched_targets):
            score += 20
        elif any(target in TRANSPORTER_TARGETS for target in matches.matched_targets):
            score += 10
        if _has_transport_substrate_or_efflux(matches):
            score += 20
        if _as_int(row.get("confidence_score")) >= 8:
            score += 10
        if row.get("relationship_type") == "D":
            score += 5
        n_unique = _as_int(row.get("n_unique_molecules"))
        if n_unique >= 100:
            score += 10
        elif n_unique >= 20:
            score += 5
        if matches.weak_context_flags and not any(matches.tier_matches.values()):
            score -= 50
        if matches.negative_flags and not _has_strong_positive(matches):
            score -= 60
        scores[tier_key] = score

    if not scores:
        return ScoredAssay(
            tier_key="none",
            tier="none",
            score=-60 if matches.negative_flags else 0,
            keep=False,
            reason="未命中 BBB 相关 assay 证据。",
            matches=matches,
        )

    tier_key = max(scores, key=lambda key: (scores[key], -TIER_ORDER.get(key, 99)))
    tier = str(BBB_RULES[tier_key]["tier"])
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
            "keep_for_bbb_reasoning": scored.keep,
        }
    )
    return out


def exclusion_reason_for(row: dict[str, Any], matches: MatchResult) -> str:
    """Return a reason when a row is residual assay-screening noise."""
    assay_text = _assay_text(row)
    genes = {str(gene).upper() for gene in row.get("target_genes", []) if gene}
    matched_targets = {str(target).upper() for target in matches.matched_targets}

    if matches.tier_matches.get("direct_bbb") and _is_brain_plasma_membrane_binding(assay_text, matches):
        return "排除 brain plasma membrane receptor/binding 误命中，不是 brain/plasma 暴露比。"

    assay_endpoint_text = join_text_parts(row.get("description"), " ".join(row.get("standard_types", []) or []))
    if matches.tier_matches.get("influx") and _is_nonfunctional_influx_assay(assay_endpoint_text, matches):
        return "排除非功能性 influx transporter assay，缺少 uptake/transport/substrate 读数。"

    if matches.tier_matches.get("efflux") and _is_nontransporter_mdr_phenotype_noise(
        assay_endpoint_text,
        row,
        genes,
        matched_targets,
    ):
        return "排除非 transporter target 的 resistant-cell-line phenotype 噪声。"

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
    if tier_key == "direct_bbb":
        if endpoint_hits:
            return True
        direct_context = _has_any(
            assay_text,
            [
                "brain penetration",
                "penetrate blood brain barrier",
                "penetrate the blood brain barrier",
                "cross blood brain barrier",
                "cross the blood brain barrier",
                "brain uptake",
                "brain exposure",
                "brain plasma",
                "brain to plasma",
                "brain blood",
                "brain to blood",
                "brain concentration",
                "brain level",
                "brain auc",
                "logbb",
                "kp uu brain",
                "kpuu brain",
                "unbound brain",
                "fraction unbound brain",
                "brain binding",
                "brain perfusion",
                "in situ brain perfusion",
                "brain uptake index",
                "brain penetration index",
                "bui",
                "bpi",
                "csf plasma",
                "csf to plasma",
                "cerebrospinal fluid plasma",
            ],
        )
        csf_context = _has_any(assay_text, ["csf", "cerebrospinal fluid"]) and _has_any(
            assay_text,
            ["plasma", "drug level", "concentration", "ratio", "unbound plasma"],
        )
        direct_context = direct_context or csf_context
        return bool(keyword_hits) and direct_context

    if tier_key == "passive_permeability":
        if _has_any(assay_text, ["blood brain barrier", "bbb permeability"]) and _has_any(
            assay_text + " " + endpoint_text,
            ["papp", "permeability", "apparent permeability", "barrier", "pampa"],
        ):
            return True
        if _has_any(assay_text, ["pampa bbb", "bbb pampa", "brain endothelial", "brain microvessel endothelial"]):
            return bool(keyword_hits or endpoint_hits) and _has_any(
                assay_text + " " + endpoint_text,
                ["papp", "permeability", "transwell", "apical", "basolateral", "barrier", "bbb"],
            )
        model = _has_any(
            assay_text,
            ["caco 2", "caco2", "mdck", "hcmec d3", "bend 3", "bmec", "bbmec", "rbec", "pampa"],
        )
        context = _has_any(
            assay_text + " " + endpoint_text,
            ["papp", "permeability", "transwell", "apical", "basolateral", "efflux ratio", "bbb", "barrier"],
        )
        return bool(keyword_hits or endpoint_hits) and model and context

    if tier_key == "efflux":
        if target_hit and _has_any(assay_text + " " + endpoint_text, ["substrate", "transport", "efflux", "papp", "ic50", "ki", "atpase"]):
            return True
        return _has_any(
            assay_text + " " + endpoint_text,
            ["p gp", "p glycoprotein", "mdr1", "abcb1", "bcrp", "abcg2", "efflux ratio", "bidirectional"],
        )

    if tier_key == "influx":
        if target_hit and _has_any(assay_text + " " + endpoint_text, ["uptake", "transport", "substrate", "influx"]):
            return True
        return _has_any(
            assay_text,
            ["lat1", "slc7a5", "glut1", "slc2a1", "oatp1a2", "slco1a2", "mct1", "slc16a1", "tfrc"],
        )

    return False


def _weak_context_flags(assay_text: str, endpoint_text: str, matches: MatchResult) -> set[str]:
    full_text = assay_text + " " + endpoint_text
    if any(matches.tier_matches.values()):
        return set()
    return {term for term in WEAK_TERMS if contains_phrase(full_text, term)}


def _has_transport_substrate_or_efflux(matches: MatchResult) -> bool:
    evidence = set(matches.matched_keywords) | set(matches.matched_endpoints)
    return bool(evidence & {"efflux ratio", "substrate", "bidirectional", "papp"})


def _has_strong_positive(matches: MatchResult) -> bool:
    return any(matches.tier_matches.values()) and bool(
        set(matches.matched_keywords)
        - {"transport", "uptake", "substrate", "ic50", "ki", "permeability"}
        or matches.matched_endpoints
        or matches.matched_targets
    )


def _reason_for(tier_key: str, matches: MatchResult) -> str:
    evidence = matches.matched_endpoints or matches.matched_keywords or matches.matched_targets
    evidence_text = "、".join(evidence[:4]) if evidence else "相关证据"
    if tier_key == "direct_bbb":
        return f"命中 {evidence_text}，属于直接脑暴露或 BBB 通过证据。"
    if tier_key == "passive_permeability":
        return f"命中 {evidence_text}，属于被动通透或屏障模型证据。"
    if tier_key == "efflux":
        return f"命中 {evidence_text}，属于 BBB 外排转运体相关证据。"
    if tier_key == "influx":
        return f"命中 {evidence_text}，属于 BBB 摄取转运体相关证据。"
    return "未命中 BBB 相关 assay 证据。"


def _has_any(text: str, phrases: list[str]) -> bool:
    return any(contains_phrase(text, phrase) for phrase in phrases)


def _is_brain_plasma_membrane_binding(text: str, matches: MatchResult) -> bool:
    if not _has_any(text, ["brain plasma membrane", "brain plasma membranes"]):
        return False
    direct_endpoints = {
        "logbb",
        "brain plasma",
        "brain plasma ratio",
        "brain concentration",
        "brain level",
        "brain uptake",
        "brain penetration index",
        "brain blood",
    }
    direct_keywords = {
        "brain to plasma",
        "brain concentration",
        "brain level",
        "brain uptake",
        "brain penetration",
        "unbound brain",
        "brain perfusion",
    }
    return not (set(matches.matched_endpoints) & direct_endpoints or set(matches.matched_keywords) & direct_keywords)


def _is_nonfunctional_influx_assay(text: str, matches: MatchResult) -> bool:
    functional_terms = [
        "uptake",
        "transport",
        "substrate",
        "influx",
        "glucose uptake",
        "drug uptake",
        "cellular uptake",
    ]
    if set(matches.matched_endpoints) & set(functional_terms):
        return False
    if set(matches.matched_keywords) & set(functional_terms):
        return False
    if _has_any(text, functional_terms):
        return False
    nonfunctional_terms = [
        "rna stability",
        "phosphorylation",
        "western blot",
        "western blotting",
        "chip seq",
        "kinobead",
        "pull down",
        "binding affinity",
        "gene expression",
        "mrna",
        "transcription",
        "promoter",
    ]
    return _has_any(text, nonfunctional_terms)


def _is_nontransporter_mdr_phenotype_noise(
    text: str,
    row: dict[str, Any],
    genes: set[str],
    matched_targets: set[str],
) -> bool:
    transporter_genes = {
        "ABCB1",
        "ABCG2",
        "ABCC1",
        "ABCC2",
        "ABCC4",
        "ABCC5",
    }
    if genes & transporter_genes or matched_targets & transporter_genes:
        return False
    target_text = join_text_parts(row.get("target_pref_name"), " ".join(row.get("target_synonyms", []) or []))
    if _has_any(
        target_text,
        [
            "atp dependent translocase abcb1",
            "abcg2",
            "bcrp",
            "p glycoprotein",
            "breast cancer resistance protein",
            "multidrug resistance associated protein",
        ],
    ):
        return False
    functional_readout_terms = [
        "rhodamine 123 efflux",
        "rhodamine efflux",
        "rhodamine 123 accumulation",
        "calcein am",
        "hoechst 33342",
        "mitoxantrone accumulation",
        "doxorubicin accumulation",
        "drug accumulation",
        "transepithelial transport",
    ]
    if _has_any(text, functional_readout_terms):
        return False
    phenotype_terms = [
        "abcb1 substrate selected resistant cell line",
        "abcg2 substrate selected resistant cell line",
        "overexpressing abcb1",
        "overexpressing abcg2",
        "abc transporter abcb1",
        "abc transporter abcg2",
        "doxorubicin resistant",
        "paclitaxel resistant",
        "multidrug resistant cell",
    ]
    return _has_any(text, phenotype_terms)


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
