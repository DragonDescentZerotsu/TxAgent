"""Scoring logic for Skin_Reaction assay screening."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tools.chembl_tool.common.text import contains_phrase, join_text_parts, match_phrases

from .rules import (
    DERMATOLOGY_EFFICACY_NOISE,
    SKIN_REACTION_RULES,
    STRONG_SKIN_REACTION_TERMS,
    TARGET_POTENCY_NOISE,
    WEAK_TERMS,
)


TIER_ORDER = {
    "direct_skin_reaction": 1,
    "sensitization_aop_key_event": 2,
    "phototoxicity_or_photosafety": 3,
    "irritation_corrosion_local_damage": 4,
    "skin_exposure_modifier": 5,
    "weak_skin_context": 6,
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
    description_text = _description_text(row)
    endpoint_text = _endpoint_text(row)
    full_text = assay_text + " " + endpoint_text

    result = MatchResult()
    result.negative_flags = sorted(set(match_phrases(full_text, DERMATOLOGY_EFFICACY_NOISE + TARGET_POTENCY_NOISE)))

    for tier_key, rule in SKIN_REACTION_RULES.items():
        keyword_hits = match_phrases(assay_text, rule["keywords"])
        endpoint_hits = match_phrases(endpoint_text, rule["endpoints"])
        valid = _has_valid_context(tier_key, assay_text, description_text, endpoint_text, keyword_hits, endpoint_hits)
        result.tier_matches[tier_key] = valid
        if valid:
            result.matched_keywords.extend(keyword_hits)
            result.matched_endpoints.extend(endpoint_hits)

    result.matched_keywords = sorted(set(result.matched_keywords))
    result.matched_endpoints = sorted(set(result.matched_endpoints))
    result.weak_context_flags = sorted(_weak_context_flags(full_text, result))
    return result


def score_assay(row: dict[str, Any], matches: MatchResult, min_score: int = 40) -> ScoredAssay:
    exclusion_reason = exclusion_reason_for(row, matches)
    if exclusion_reason:
        return ScoredAssay("excluded", "excluded", 0, False, exclusion_reason, matches)

    scores: dict[str, int] = {}
    for tier_key, matched in matches.tier_matches.items():
        if not matched:
            continue
        score = int(SKIN_REACTION_RULES[tier_key]["base_score"])
        score += min(30, 4 * len(matches.matched_keywords))
        score += 15 if matches.matched_endpoints else 0
        if tier_key == "direct_skin_reaction" and _has_any(_assay_text(row), ["human", "patient", "volunteer", "llna", "local lymph node"]):
            score += 15
        if tier_key == "sensitization_aop_key_event" and _has_any(
            _assay_text(row),
            ["dpra", "adra", "keratinosens", "h-clat", "u-sens", "il-8 luc", "gardskin"],
        ):
            score += 12
        if tier_key == "skin_exposure_modifier":
            score -= 10
        if tier_key == "weak_skin_context":
            score -= 20
        if _as_int(row.get("confidence_score")) >= 8:
            score += 8
        if row.get("relationship_type") == "D":
            score += 5
        n_unique = _as_int(row.get("n_unique_molecules"))
        if n_unique >= 100:
            score += 10
        elif n_unique >= 20:
            score += 5
        if matches.negative_flags and not _has_strong_skin_positive(matches):
            score -= 75
        scores[tier_key] = score

    if not scores:
        return ScoredAssay(
            tier_key="none",
            tier="none",
            score=-60 if matches.negative_flags else 0,
            keep=False,
            reason="未命中 Skin_Reaction 相关 skin safety / sensitisation / irritation / exposure assay 证据。",
            matches=matches,
        )

    tier_key = max(scores, key=lambda key: (scores[key], -TIER_ORDER.get(key, 99)))
    tier = str(SKIN_REACTION_RULES[tier_key]["tier"])
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
            "keep_for_skin_reaction_reasoning": scored.keep,
        }
    )
    return out


def exclusion_reason_for(row: dict[str, Any], matches: MatchResult) -> str:
    assay_text = _assay_text(row)
    endpoint_text = _endpoint_text(row)
    full_text = assay_text + " " + endpoint_text
    description_text = _description_text(row) + " " + endpoint_text

    if _is_dermatology_efficacy_noise(full_text) and not _has_strong_skin_assay_context(full_text):
        return "排除 dermatology efficacy / cosmetic / melanoma / antimicrobial context；不是 adverse skin reaction evidence。"

    if matches.tier_matches.get("skin_exposure_modifier") and _is_permeation_enhancer_context(description_text):
        return "排除 permeation/penetration enhancer assay；测试的是促进其他分子透皮，不是 query molecule 自身 skin exposure。"

    if _is_target_potency_noise(description_text) and not _has_strong_skin_assay_context(description_text):
        return "排除普通 target binding/enzyme/kinase biochemical assay；不是 adverse skin-reaction evidence。"

    if _is_target_potency_noise(full_text) and not _has_strong_skin_assay_context(full_text):
        return "排除普通 target binding/enzyme inhibition assay；缺少 skin sensitisation、irritation、phototoxicity 或 dermal toxicity context。"

    if matches.tier_matches.get("skin_exposure_modifier") and _is_non_skin_permeability_noise(description_text):
        return "排除非 skin/dermal/transdermal permeability context；不是皮肤暴露证据。"

    if matches.tier_matches.get("weak_skin_context") and matches.negative_flags and not _has_strong_skin_positive(matches):
        return "排除 weak skin/cytotoxicity context 中的 efficacy/noise assay。"

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


def _description_text(row: dict[str, Any]) -> str:
    return join_text_parts(
        row.get("description"),
        row.get("assay_cell_type"),
        row.get("assay_tissue"),
        row.get("assay_organism"),
        row.get("assay_test_type"),
        row.get("assay_category"),
    )


def _has_valid_context(
    tier_key: str,
    assay_text: str,
    description_text: str,
    endpoint_text: str,
    keyword_hits: list[str],
    endpoint_hits: list[str],
) -> bool:
    full_text = assay_text + " " + endpoint_text
    described_endpoint_text = description_text + " " + endpoint_text

    if tier_key == "direct_skin_reaction":
        return bool(keyword_hits or endpoint_hits) and _has_any(
            full_text,
            [
                "skin reaction",
                "dermatologic adverse",
                "patch test",
                "hript",
                "local lymph node",
                "llna",
                "stimulation index",
                "ec3",
                "guinea pig",
                "gpmt",
                "buehler",
                "skin sensitization",
                "skin sensitisation",
                "contact dermatitis",
                "skin irritation",
                "dermal irritation",
                "dermal toxicity",
                "draize",
            ],
        )

    if tier_key == "sensitization_aop_key_event":
        strong_aop = _has_any(
            full_text,
            [
                "dpra",
                "adra",
                "kdpra",
                "direct peptide reactivity",
                "peptide reactivity",
                "peptide depletion",
                "cysteine depletion",
                "lysine depletion",
                "keratinosens",
                "lusens",
                "episensa",
                "h clat",
                "h-clat",
                "u sens",
                "u-sens",
                "il 8 luc",
                "il-8 luc",
                "gardskin",
                "dendritic cell activation",
                "cd54",
                "cd86",
                "glutathione depletion",
                "gsh reactivity",
                "thiol reactivity",
            ],
        )
        are_nrf2_skin_context = _has_any(full_text, ["are nrf2", "are-nrf2", "antioxidant response element", "are luc", "are-luc"]) and _has_any(
            full_text,
            ["hacat", "keratinocyte", "keratinosens", "lusens", "episensa", "skin sensitization", "skin sensitisation"],
        )
        nrf2_skin_context = _has_any(full_text, ["nrf2", "keap1"]) and _has_any(
            full_text,
            ["hacat", "keratinocyte", "keratinosens", "lusens", "episensa", "skin sensitization", "skin sensitisation"],
        )
        return bool(keyword_hits or endpoint_hits) and (strong_aop or are_nrf2_skin_context or nrf2_skin_context)

    if tier_key == "phototoxicity_or_photosafety":
        return bool(keyword_hits or endpoint_hits) and _has_any(
            full_text,
            ["phototoxicity", "photoirritation", "photo irritation", "photoallergy", "photosafety", "3t3 nru", "pif", "mean photo effect", "uva", "uvb"],
        )

    if tier_key == "irritation_corrosion_local_damage":
        if _is_dermatology_efficacy_noise(full_text) and not _has_any(full_text, ["irritation", "corrosion", "cytotoxicity", "viability"]):
            return False
        if _is_target_potency_noise(description_text) and not _has_any(description_text, ["skin irritation", "skin corrosion", "dermal irritation", "rhe", "reconstructed human epidermis"]):
            return False
        return bool(keyword_hits or endpoint_hits) and _has_any(
            described_endpoint_text,
            [
                "reconstructed human epidermis",
                "episkin",
                "epiderm",
                "skinethic",
                "skin irritation",
                "skin corrosion",
                "dermal irritation",
                "keratinocyte",
                "hacat",
                "epidermal",
                "dermal fibroblast",
                "skin cell",
                "skin inflammation",
            ],
        )

    if tier_key == "skin_exposure_modifier":
        if _is_non_skin_permeability_noise(described_endpoint_text):
            return False
        return bool(keyword_hits or endpoint_hits) and _has_any(
            described_endpoint_text,
            [
                "skin absorption",
                "dermal absorption",
                "percutaneous absorption",
                "skin permeation",
                "skin permeability",
                "transdermal",
                "franz diffusion",
                "diffusion cell",
                "stratum corneum",
                "skin retention",
                "skin pampa",
                "logkp",
                "log kp",
            ],
        )

    if tier_key == "weak_skin_context":
        return (
            bool(keyword_hits or endpoint_hits)
            and _has_weak_skin_context(described_endpoint_text)
            and not _has_strong_skin_assay_context(full_text)
        )

    return False


def _weak_context_flags(full_text: str, matches: MatchResult) -> set[str]:
    if _has_strong_skin_positive(matches):
        return set()
    return {term for term in WEAK_TERMS if contains_phrase(full_text, term)}


def _has_strong_skin_positive(matches: MatchResult) -> bool:
    strong_tiers = {
        "direct_skin_reaction",
        "sensitization_aop_key_event",
        "phototoxicity_or_photosafety",
        "irritation_corrosion_local_damage",
    }
    if any(matches.tier_matches.get(tier) for tier in strong_tiers):
        return True
    evidence = set(matches.matched_keywords) | set(matches.matched_endpoints)
    return bool(evidence & set(STRONG_SKIN_REACTION_TERMS))


def _has_strong_skin_assay_context(text: str) -> bool:
    return _has_any(text, STRONG_SKIN_REACTION_TERMS)


def _has_weak_skin_context(text: str) -> bool:
    return _has_any(
        text,
        [
            "skin",
            "dermal",
            "epiderm",
            "epidermal",
            "keratinocyte",
            "hacat",
            "dermal fibroblast",
            "skin fibroblast",
            "human dermal fibroblast",
            "stratum corneum",
            "cutaneous",
        ],
    )


def _is_dermatology_efficacy_noise(text: str) -> bool:
    return _has_any(text, DERMATOLOGY_EFFICACY_NOISE)


def _is_target_potency_noise(text: str) -> bool:
    return _has_any(text, TARGET_POTENCY_NOISE)


def _is_non_skin_permeability_noise(text: str) -> bool:
    if not _has_any(text, ["permeability", "permeation", "absorption", "flux", "kp", "retention", "pampa"]):
        return False
    if _has_any(text, ["paf induced", "paf-induced", "vascular permeability", "block the effect of exogenously administered paf"]):
        return True
    if _has_any(text, ["skin", "dermal", "percutaneous", "transdermal", "franz", "stratum corneum"]):
        return False
    return _has_any(text, ["caco", "mdck", "bbb", "brain", "intestinal", "pampa"])


def _is_permeation_enhancer_context(text: str) -> bool:
    return _has_any(
        text,
        [
            "permeation enhancer",
            "penetration enhancer",
            "potentiation of",
            "enhancement of transdermal permeation",
            "enhancement of skin permeation",
            "enhancer for",
        ],
    )


def _reason_for(tier_key: str, matches: MatchResult) -> str:
    evidence = matches.matched_endpoints or matches.matched_keywords
    evidence_text = "、".join(evidence[:4]) if evidence else "相关证据"
    if tier_key == "direct_skin_reaction":
        return f"命中 {evidence_text}，属于直接 skin reaction / sensitisation / dermal toxicity anchor。"
    if tier_key == "sensitization_aop_key_event":
        return f"命中 {evidence_text}，属于 validated skin sensitisation AOP key-event evidence。"
    if tier_key == "phototoxicity_or_photosafety":
        return f"命中 {evidence_text}，属于 phototoxicity / photosafety evidence。"
    if tier_key == "irritation_corrosion_local_damage":
        return f"命中 {evidence_text}，属于 irritation、corrosion 或 local skin cell damage evidence。"
    if tier_key == "skin_exposure_modifier":
        return f"命中 {evidence_text}，属于 skin exposure / dermal absorption modifier，不是 hazard 本身。"
    if tier_key == "weak_skin_context":
        return f"命中 {evidence_text}，只作为 weak/context-dependent skin background。"
    return "未命中 Skin_Reaction 相关 assay 证据。"


def _has_any(text: str, phrases: list[str] | tuple[str, ...] | set[str]) -> bool:
    return bool(match_phrases(text, phrases))


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
