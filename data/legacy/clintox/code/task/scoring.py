"""Scoring logic for ClinTox assay screening."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tools.chembl_tool.common.text import join_text_parts, match_phrases

from .rules import (
    CARDIAC_TARGET_GENES,
    CLINTOX_RULES,
    CYP_TARGET_GENES,
    HEPATIC_TARGET_GENES,
    NEGATIVE_KEYWORDS,
    NEURO_TARGET_GENES,
    RENAL_TARGET_GENES,
    SAFETY_TARGET_GENES,
    SAFETY_TARGET_PHRASES,
    WEAK_TERMS,
)


TIER_ORDER = {
    "clinical_human_safety": 1,
    "in_vivo_toxicology": 2,
    "organ_safety_pharmacology": 3,
    "genotoxicity_carcinogenicity": 4,
    "tox21_cell_stress": 5,
    "general_cytotoxicity": 6,
    "offtarget_ddi_exposure": 7,
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
    full_text = assay_text + " " + endpoint_text
    genes = {str(gene).upper() for gene in row.get("target_genes", []) if gene}
    target_text = join_text_parts(
        row.get("target_pref_name"),
        " ".join(row.get("target_synonyms", []) or []),
        " ".join(row.get("component_descriptions", []) or []),
    )

    result = MatchResult()
    result.negative_flags = match_phrases(full_text, NEGATIVE_KEYWORDS)

    safety_targets = sorted(genes & SAFETY_TARGET_GENES)
    if safety_targets or _has_any(target_text, SAFETY_TARGET_PHRASES):
        result.matched_targets.extend(safety_targets or ["SAFETY_RELEVANT_TARGET"])

    for tier_key, rule in CLINTOX_RULES.items():
        keyword_hits = match_phrases(assay_text, rule["keywords"])
        endpoint_hits = match_phrases(endpoint_text, rule["endpoints"])
        target_hit = _target_hit_for_tier(tier_key, genes, target_text)
        valid = _has_valid_context(tier_key, assay_text, endpoint_text, keyword_hits, endpoint_hits, target_hit)
        result.tier_matches[tier_key] = valid
        if valid:
            result.matched_keywords.extend(keyword_hits)
            result.matched_endpoints.extend(endpoint_hits)

    result.matched_keywords = sorted(set(result.matched_keywords))
    result.matched_endpoints = sorted(set(result.matched_endpoints))
    result.matched_targets = sorted(set(result.matched_targets))
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
        score = int(CLINTOX_RULES[tier_key]["base_score"])
        score += min(24, 3 * len(matches.matched_keywords))
        score += 15 if matches.matched_endpoints else 0
        if tier_key == "clinical_human_safety" and matches.matched_endpoints:
            score += 15
        if tier_key == "organ_safety_pharmacology" and matches.matched_targets:
            score += 15
        if tier_key == "offtarget_ddi_exposure" and matches.matched_targets:
            score += 12
        if tier_key == "genotoxicity_carcinogenicity" and _has_any(
            _assay_text(row),
            ["ames", "micronucleus", "chromosomal aberration", "mouse lymphoma", "carcinogenicity"],
        ):
            score += 12
        if _as_int(row.get("confidence_score")) >= 8:
            score += 10
        if row.get("relationship_type") == "D":
            score += 5
        n_unique = _as_int(row.get("n_unique_molecules"))
        if n_unique >= 100:
            score += 10
        elif n_unique >= 20:
            score += 5
        if matches.negative_flags and not _has_strong_toxicity_positive(matches):
            score -= 70
        if matches.weak_context_flags and not matches.matched_keywords:
            score -= 20
        scores[tier_key] = score

    if not scores:
        return ScoredAssay(
            tier_key="none",
            tier="none",
            score=-60 if matches.negative_flags else 0,
            keep=False,
            reason="未命中 ClinTox 相关 safety/toxicity assay 证据。",
            matches=matches,
        )

    tier_key = max(scores, key=lambda key: (scores[key], -TIER_ORDER.get(key, 99)))
    tier = str(CLINTOX_RULES[tier_key]["tier"])
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
            "keep_for_clintox_reasoning": scored.keep,
        }
    )
    return out


def exclusion_reason_for(row: dict[str, Any], matches: MatchResult) -> str:
    assay_text = _assay_text(row)
    endpoint_text = _endpoint_text(row)
    full_text = assay_text + " " + endpoint_text

    if _is_antiinfective_efficacy_noise(full_text):
        return "排除 anti-infective efficacy assay；EC50/replication/cytopathic-effect 不是 host toxicity。"

    if matches.negative_flags and not _has_strong_toxicity_positive(matches):
        return "排除 efficacy/anti-infective/target-potency 语境；缺少明确 toxicity 或 safety-liability context。"

    if matches.tier_matches.get("general_cytotoxicity") and _is_efficacy_cell_growth_noise(full_text):
        return "排除 cancer/anti-infective efficacy cell-growth assay；不是 general safety cytotoxicity。"

    if matches.tier_matches.get("offtarget_ddi_exposure") and _is_generic_target_inhibition_noise(full_text, matches):
        return "排除普通 target inhibition/binding；不是 safety-relevant off-target、DDI 或 exposure liability。"

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
    if tier_key == "clinical_human_safety":
        direct_clinical_context = _has_any(
            full_text,
            [
                "patient",
                "volunteer",
                "clinical trial",
                "adverse event",
                "adverse drug",
                "withdrawal due to toxicity",
                "boxed warning",
                "black box warning",
            ],
        )
        human_subject_context = _has_any(
            full_text,
            [
                "human patient",
                "human patients",
                "healthy volunteer",
                "human volunteer",
                "human subject",
                "human subjects",
                "toxicity in human assessed",
                "toxicity in iv infused human",
                "clinical trial",
            ],
        )
        human_toxicity_context = human_subject_context and _has_any(
            full_text,
            [
                "maximum tolerated dose",
                "mtd",
                "tolerability",
                "dose limiting toxicity",
                "dose-limiting toxicity",
                "toxicity in",
                "creatinine",
                "alt",
                "ast",
                "bilirubin",
            ],
        )
        return bool(keyword_hits or endpoint_hits) and (direct_clinical_context or human_toxicity_context)
    if tier_key == "in_vivo_toxicology":
        if _is_in_vitro_cell_toxicity_context(full_text) and not _has_any(
            full_text,
            ["in vivo", "oral toxicity", "intravenous toxicity", "peroral", "intraperitoneal", "single ip dose"],
        ):
            return False
        return bool(keyword_hits or endpoint_hits) and _has_any(
            full_text,
            ["in vivo", "mouse", "mice", "rat", "dog", "monkey", "animal", "oral", "intravenous", "subchronic", "chronic"],
        )
    if tier_key == "organ_safety_pharmacology":
        return bool(keyword_hits or target_hit) and not _is_plain_efficacy_inhibition(full_text)
    if tier_key == "genotoxicity_carcinogenicity":
        genotox_context = _has_any(
            full_text,
            [
                "ames",
                "bacterial reverse mutation",
                "mutagenicity",
                "revertant",
                "micronucleus",
                "chromosomal aberration",
                "chromosome aberration",
                "mouse lymphoma assay",
                "mouse lymphoma tk",
                "hprt",
                "comet assay",
                "dna strand break",
                "gamma h2ax",
                "atad5",
                "dna damage",
                "carcinogenicity",
                "tumorigenicity",
            ],
        )
        return bool(keyword_hits or endpoint_hits) and genotox_context
    if tier_key == "tox21_cell_stress":
        if _is_enzyme_stress_readout_noise(full_text):
            return False
        if _is_transporter_reversal_stress_noise(full_text):
            return False
        if _is_generic_glutathione_enzyme_noise(full_text):
            return False
        return bool(keyword_hits or endpoint_hits)
    if tier_key == "general_cytotoxicity":
        explicit_cytotoxicity = _has_any(
            full_text,
            [
                "cytotoxicity",
                "cytotoxic",
                "host cell toxicity",
                "not cytotoxic",
                "toxic concentration",
                "cc50",
                "tc50",
                "lc50",
                "ld50",
                "ldh release",
                "neutral red uptake",
                "propidium iodide",
                "apoptosis",
                "necrosis",
                "caspase",
            ],
        )
        return bool(keyword_hits or endpoint_hits) and explicit_cytotoxicity and not _is_plain_cell_viability_counterassay(full_text)
    if tier_key == "offtarget_ddi_exposure":
        return bool(keyword_hits or (target_hit and endpoint_hits)) and not _is_plain_efficacy_inhibition(full_text)
    return False


def _target_hit_for_tier(tier_key: str, genes: set[str], target_text: str) -> bool:
    if tier_key == "organ_safety_pharmacology":
        if genes & (CARDIAC_TARGET_GENES | HEPATIC_TARGET_GENES | RENAL_TARGET_GENES):
            return True
        neuro_functional_context = _has_any(
            target_text,
            ["acetylcholinesterase", "sodium channel", "gaba", "nmda"],
        )
        return bool(genes & NEURO_TARGET_GENES) and neuro_functional_context
    if tier_key == "offtarget_ddi_exposure":
        return bool(genes & (SAFETY_TARGET_GENES | CYP_TARGET_GENES)) or _has_any(target_text, SAFETY_TARGET_PHRASES)
    return False


def _weak_context_flags(full_text: str, matches: MatchResult) -> set[str]:
    if _has_strong_toxicity_positive(matches):
        return set()
    return set(match_phrases(full_text, WEAK_TERMS))


def _has_strong_toxicity_positive(matches: MatchResult) -> bool:
    strong_tiers = {
        "clinical_human_safety",
        "in_vivo_toxicology",
        "organ_safety_pharmacology",
        "genotoxicity_carcinogenicity",
        "tox21_cell_stress",
    }
    return any(matches.tier_matches.get(tier) for tier in strong_tiers) or bool(matches.matched_targets)


def _is_efficacy_cell_growth_noise(full_text: str) -> bool:
    efficacy_context = _has_any(
        full_text,
        [
            "antitumor",
            "anticancer",
            "cancer cell line",
            "carcinoma",
            "tumour",
            "tumor",
            "leukemia",
            "lymphoma",
            "melanoma",
            "glioblastoma",
            "tumor cell",
            "xenograft",
            "antimicrobial",
            "antibacterial",
            "antifungal",
            "antiviral",
            "parasite",
            "plasmodium",
        ],
    )
    toxicity_context = _has_any(
        full_text,
        [
            "normal cell",
            "non tumor",
            "non-tumor",
            "selectivity",
            "safety",
            "host cell toxicity",
            "cytotoxicity counterscreen",
            "cytotoxicity counter screen",
            "counter screen",
            "counterscreen",
            "cc50",
            "tc50",
        ],
    )
    return efficacy_context and not toxicity_context


def _is_generic_target_inhibition_noise(full_text: str, matches: MatchResult) -> bool:
    if matches.matched_targets:
        return False
    if _has_any(full_text, SAFETY_TARGET_PHRASES):
        return False
    return _has_any(full_text, ["enzyme inhibition", "kinase inhibition", "receptor binding"]) and not _has_any(
        full_text,
        ["toxicity", "safety", "ddi", "drug interaction", "reactive metabolite", "covalent", "herg", "bsep", "cyp"],
    )


def _is_plain_efficacy_inhibition(full_text: str) -> bool:
    return _has_any(full_text, ["kinase inhibition", "enzyme inhibition", "receptor binding"]) and not _has_any(
        full_text,
        ["toxicity", "safety", "herg", "bsep", "cyp", "5 ht2b", "htr2b", "reactive metabolite"],
    )


def _is_in_vitro_cell_toxicity_context(full_text: str) -> bool:
    return _has_any(
        full_text,
        [
            "in vitro",
            "cell line",
            "cells",
            "cytotoxicity",
            "cell viability",
            "cell survival",
            "mtt",
            "nih3t3",
            "balb c 3t3",
        ],
    )


def _is_antiinfective_efficacy_noise(full_text: str) -> bool:
    if not _has_any(
        full_text,
        ["antiviral", "viral replication", "hepatitis", "hbv", "hcv", "cytopathic effect", "antibacterial", "antifungal", "antimalarial", "parasite"],
    ):
        return False
    return _has_any(full_text, ["ec50", "effective concentration", "replication", "viral dna", "mic"]) and not _has_any(
        full_text,
        ["cytotoxicity", "host cell toxicity", "cell viability", "cc50", "tc50", "lc50", "selectivity index"],
    )


def _is_enzyme_stress_readout_noise(full_text: str) -> bool:
    enzyme_context = _has_any(full_text, ["lipoxygenase", "hexokinase", "enzyme", "oxidase"])
    if not enzyme_context:
        return False
    if not _has_any(full_text, ["oxygen consumption", "atp"]):
        return False
    return not _has_any(
        full_text,
        ["mitochondrial", "mitochondria", "cellular respiration", "oxygen consumption rate", "seahorse", "tox21", "toxcast"],
    )


def _is_transporter_reversal_stress_noise(full_text: str) -> bool:
    return _has_any(
        full_text,
        ["potentiation of", "reversal of", "multidrug resistance", "doxorubicin induced cytotoxicity", "paclitaxel induced cytotoxicity"],
    ) and _has_any(full_text, ["mrp1", "p gp", "p-gp", "bcrp", "abcb1", "abcg2"])


def _is_generic_glutathione_enzyme_noise(full_text: str) -> bool:
    return _has_any(full_text, ["glutathione"]) and _has_any(
        full_text,
        ["glutathione s transferase", "thioredoxin glutathione reductase", "enzyme inhibition"],
    ) and not _has_any(
        full_text,
        ["reactive metabolite", "gsh adduct", "glutathione adduct", "depletion", "oxidative stress", "tox21", "toxcast"],
    )


def _is_plain_cell_viability_counterassay(full_text: str) -> bool:
    if _has_any(full_text, ["cytotoxicity", "cytotoxic", "host cell toxicity", "cc50", "tc50", "lc50"]):
        return False
    return _has_any(full_text, ["cell viability", "celltiter glo", "mtt assay", "mts assay", "resazurin", "alamar blue"]) and _has_any(
        full_text,
        ["target", "inhibitor", "reporter assay", "binding assay", "activity assay", "counter screen", "counterscreen"],
    )


def _reason_for(tier_key: str, matches: MatchResult) -> str:
    pieces = [f"命中 {CLINTOX_RULES[tier_key]['tier']} ClinTox evidence family: {tier_key}."]
    if matches.matched_keywords:
        pieces.append("keywords=" + ", ".join(matches.matched_keywords[:8]))
    if matches.matched_endpoints:
        pieces.append("endpoints=" + ", ".join(matches.matched_endpoints[:8]))
    if matches.matched_targets:
        pieces.append("targets=" + ", ".join(matches.matched_targets[:8]))
    if matches.negative_flags:
        pieces.append("negative_flags=" + ", ".join(matches.negative_flags[:5]))
    return " ".join(pieces)


def _has_any(text: str, phrases: list[str] | tuple[str, ...] | set[str]) -> bool:
    return bool(match_phrases(text, phrases))


def _as_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
