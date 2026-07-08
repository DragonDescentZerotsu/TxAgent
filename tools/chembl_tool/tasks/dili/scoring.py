"""Scoring logic for DILI assay screening."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tools.chembl_tool.common.text import contains_phrase, join_text_parts, match_phrases

from .rules import (
    DILI_RULES,
    EFFICACY_NOISE,
    GENERIC_CYP_INHIBITION_TERMS,
    GENERIC_TARGET_NOISE,
    HEPATIC_METABOLISM_TARGET_GENES,
    HEPATOBILIARY_TARGET_GENES,
    HUMAN_OR_CLINICAL_TERMS,
    IN_VIVO_TERMS,
    LIVER_CANCER_EFFICACY_NOISE,
    LIVER_CONTEXT_TERMS,
    NON_DILI_SAFETY_NOISE,
    STRONG_DILI_TERMS,
    WEAK_TERMS,
)


TIER_ORDER = {
    "direct_human_dili": 1,
    "in_vivo_liver_injury": 2,
    "cholestasis_transporter": 3,
    "mitochondrial_organelle_stress": 4,
    "reactive_metabolite_immune": 5,
    "hepatic_cell_exposure_context": 6,
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
    genes = _target_genes(row)

    result = MatchResult()
    result.negative_flags = sorted(
        set(match_phrases(full_text, LIVER_CANCER_EFFICACY_NOISE + EFFICACY_NOISE + NON_DILI_SAFETY_NOISE))
    )
    target_hits = sorted(genes & (HEPATOBILIARY_TARGET_GENES | HEPATIC_METABOLISM_TARGET_GENES))
    result.matched_targets.extend(target_hits)

    for tier_key, rule in DILI_RULES.items():
        keyword_hits = match_phrases(assay_text, rule["keywords"])
        endpoint_hits = match_phrases(endpoint_text, rule["endpoints"])
        target_hit = _target_hit_for_tier(tier_key, genes, assay_text)
        valid = _has_valid_context(
            tier_key,
            assay_text,
            description_text,
            endpoint_text,
            keyword_hits,
            endpoint_hits,
            target_hit,
        )
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
        score = int(DILI_RULES[tier_key]["base_score"])
        score += min(28, 4 * len(matches.matched_keywords))
        score += 15 if matches.matched_endpoints else 0
        if tier_key == "direct_human_dili" and _has_any(_assay_text(row), HUMAN_OR_CLINICAL_TERMS):
            score += 15
        if tier_key == "in_vivo_liver_injury" and _has_any(_assay_text(row), IN_VIVO_TERMS):
            score += 10
        if tier_key == "cholestasis_transporter" and matches.matched_targets:
            score += 10
        if tier_key == "hepatic_cell_exposure_context":
            score -= 12
        if _as_int(row.get("confidence_score")) >= 8:
            score += 8
        if row.get("relationship_type") == "D":
            score += 5
        n_unique = _as_int(row.get("n_unique_molecules"))
        if n_unique >= 100:
            score += 10
        elif n_unique >= 20:
            score += 5
        if matches.negative_flags and not _has_strong_dili_positive(matches):
            score -= 80
        if matches.weak_context_flags and tier_key == "hepatic_cell_exposure_context":
            score -= 8
        scores[tier_key] = score

    if not scores:
        return ScoredAssay(
            tier_key="none",
            tier="none",
            score=-60 if matches.negative_flags else 0,
            keep=False,
            reason="未命中 DILI 相关 human/in vivo liver injury 或关键 DILI mechanism assay 证据。",
            matches=matches,
        )

    tier_key = max(scores, key=lambda key: (scores[key], -TIER_ORDER.get(key, 99)))
    tier = str(DILI_RULES[tier_key]["tier"])
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
            "keep_for_dili_reasoning": scored.keep,
        }
    )
    return out


def exclusion_reason_for(row: dict[str, Any], matches: MatchResult) -> str:
    assay_text = _assay_text(row)
    description_text = _description_text(row)
    endpoint_text = _endpoint_text(row)
    full_text = assay_text + " " + endpoint_text
    described_endpoint_text = description_text + " " + endpoint_text

    if _is_liver_cancer_efficacy_noise(described_endpoint_text) and not _has_strong_dili_assay_context(full_text):
        return "排除 liver cancer / hepatocellular carcinoma efficacy 或 antiproliferative assay；不是 compound-induced DILI evidence。"

    if _is_metabolic_efficacy_liver_disease_noise(described_endpoint_text):
        return "排除 hypolipidemic/hyperlipidemic/NAFLD/NASH liver-disease efficacy assay；不是 compound-induced DILI liver injury evidence。"

    if _is_antiinfective_efficacy_noise(described_endpoint_text) and not _has_strong_dili_assay_context(full_text):
        return "排除 anti-infective / antiviral efficacy assay；不是 DILI evidence。"

    if _is_hepatoprotective_efficacy_noise(described_endpoint_text):
        return "排除 hepatoprotective / protection-from-induced-liver-injury efficacy assay；不是 compound-induced DILI evidence。"

    if _is_kidney_or_renal_context_noise(described_endpoint_text):
        return "排除 kidney/renal transporter、microsome 或 necrosis context；不是 hepatic DILI evidence。"

    if _is_non_dili_safety_noise(described_endpoint_text) and not _has_strong_dili_assay_context(full_text):
        return "排除 cardiac/renal/neuro/skin/genotox 等非肝脏 safety context；不是 DILI evidence。"

    if _is_abcb1_or_pgp_noise(full_text):
        return "排除 P-gp/ABCB1/MDR1 assay；不是 DILI cholestasis/hepatobiliary transporter evidence。"

    if _is_asbt_or_ileal_bile_acid_transport_noise(full_text):
        return "排除 ASBT/SLC10A2 或 ileal bile-acid uptake assay；不是 hepatobiliary/cholestatic DILI transporter evidence。"

    if _is_non_hepatobiliary_bile_acid_adjacent_target_noise(full_text):
        return "排除 bile-acid-adjacent enzyme/pathogen/target assay；不是 hepatobiliary transporter 或 DILI bile-acid homeostasis evidence。"

    if _is_nonmetabolic_covalent_binding_noise(full_text):
        return "排除 receptor/target covalent binding assay；缺少 reactive metabolite、bioactivation 或 hepatic metabolism context。"

    if _is_generic_target_biology_noise(described_endpoint_text):
        return "排除 HepG2/target-biology reporter 或 enzyme-target assay；缺少 hepatic injury、DILI mechanism 或 bioactivation context。"

    if _is_generic_cyp_inhibition_noise(described_endpoint_text) and not _has_bioactivation_or_liver_injury_context(full_text):
        return "排除普通 CYP inhibition / TDI assay；缺少 bioactivation、hepatic exposure 或 liver injury context。"

    if _is_generic_target_potency_noise(described_endpoint_text) and not _has_strong_dili_assay_context(full_text):
        return "排除普通 target binding/enzyme potency assay；缺少 DILI 或 liver-injury mechanism context。"

    if matches.tier_matches.get("hepatic_cell_exposure_context") and _is_generic_nonhepatic_cytotoxicity(described_endpoint_text):
        return "排除非肝细胞 generic cytotoxicity / proliferation assay；不是 DILI-specific hepatic cell evidence。"

    if matches.tier_matches.get("hepatic_cell_exposure_context") and _is_antiviral_hepg2_control_cytotoxicity(described_endpoint_text):
        return "排除 HepG2 2.2.15 / antiviral cytotoxicity control；不是专门的 DILI hepatic cell injury model。"

    if matches.tier_matches.get("hepatic_cell_exposure_context") and _is_generic_hepg2_cancer_cytotoxicity_noise(described_endpoint_text):
        return "排除 generic HepG2 cancer-cell cytotoxicity / viability assay；缺少 ADMET、hepatotoxicity 或专门 hepatic injury model context。"

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
        _join_values(row.get("target_genes")),
        _join_values(row.get("target_synonyms")),
        _join_values(row.get("component_descriptions")),
    )


def _description_text(row: dict[str, Any]) -> str:
    return join_text_parts(
        row.get("description"),
        row.get("assay_cell_type"),
        row.get("assay_tissue"),
        row.get("assay_organism"),
        row.get("assay_test_type"),
        row.get("assay_category"),
    )


def _endpoint_text(row: dict[str, Any]) -> str:
    return join_text_parts(_join_values(row.get("standard_types")))


def _has_valid_context(
    tier_key: str,
    assay_text: str,
    description_text: str,
    endpoint_text: str,
    keyword_hits: list[str],
    endpoint_hits: list[str],
    target_hit: bool,
) -> bool:
    full_text = assay_text + " " + endpoint_text
    described_endpoint_text = description_text + " " + endpoint_text
    has_liver_context = _has_liver_context(full_text)
    has_described_liver_context = _has_liver_context(described_endpoint_text)

    if tier_key == "direct_human_dili":
        direct_dili_terms = _has_any(
            described_endpoint_text,
            [
                "drug induced liver injury",
                "drug-induced liver injury",
                "dili",
                "hepatotoxicity",
                "hepatotoxic",
                "liver injury",
                "hepatic injury",
                "drug induced hepatitis",
                "drug-induced hepatitis",
                "toxic hepatitis",
            ],
        )
        severe_liver_outcome = _has_any(
            described_endpoint_text,
            [
                "acute liver failure",
                "fulminant hepatic failure",
                "liver failure",
                "liver transplant",
                "fatal liver injury",
                "liver related death",
                "liver-related death",
                "withdrawal due to hepatotoxicity",
                "withdrawn due to hepatotoxicity",
                "boxed warning",
                "black box warning",
            ],
        )
        direct_dili = severe_liver_outcome or (direct_dili_terms and _has_any(described_endpoint_text, HUMAN_OR_CLINICAL_TERMS))
        clinical_lab = (
            bool(endpoint_hits)
            and _has_any(described_endpoint_text, HUMAN_OR_CLINICAL_TERMS)
            and (has_described_liver_context or _has_liver_lab_endpoint(described_endpoint_text))
        )
        return bool(keyword_hits or endpoint_hits) and (direct_dili or clinical_lab)

    if tier_key == "in_vivo_liver_injury":
        if not bool(keyword_hits or endpoint_hits):
            return False
        in_vivo_liver = _has_any(described_endpoint_text, IN_VIVO_TERMS) and has_described_liver_context
        pathology = _has_any(
            described_endpoint_text,
            [
                "histopathology",
                "pathology",
                "necrosis",
                "degeneration",
            ],
        ) and has_described_liver_context
        liver_weight = _has_any(described_endpoint_text, ["liver weight", "relative liver weight"])
        liver_lab_chemistry = _has_liver_lab_endpoint(described_endpoint_text) and (
            has_described_liver_context
            or _has_any(described_endpoint_text, ["serum", "plasma", "bile fistula", "in vivo"])
        )
        toxic_dose_or_margin = _has_any(described_endpoint_text, ["noael with liver", "loael with liver", "mtd with liver"])
        hepatobiliary_chemistry = _has_any(described_endpoint_text, ["bile fistula", "bile acid", "cholestasis", "cholestatic"])
        return in_vivo_liver or pathology or liver_weight or liver_lab_chemistry or toxic_dose_or_margin or hepatobiliary_chemistry

    if tier_key == "cholestasis_transporter":
        if _is_abcb1_or_pgp_noise(full_text):
            return False
        if target_hit:
            return bool(keyword_hits or endpoint_hits or _has_any(full_text, ["transport", "inhibition", "efflux", "uptake"]))
        if _is_asbt_or_ileal_bile_acid_transport_noise(full_text):
            return False
        if _is_non_hepatobiliary_bile_acid_adjacent_target_noise(full_text):
            return False
        return bool(keyword_hits or endpoint_hits) and _has_any(
            full_text,
            [
                "bsep",
                "bile salt export pump",
                "bile acid",
                "taurocholate",
                "hepatobiliary",
                "cholestasis",
                "cholestatic",
                "ntcp",
                "oatp1b1",
                "oatp1b3",
                "mrp2",
                "mdr3",
            ],
        )

    if tier_key == "mitochondrial_organelle_stress":
        if not bool(keyword_hits or endpoint_hits):
            return False
        return has_liver_context and _has_any(
            full_text,
            [
                "mitochondrial",
                "oxygen consumption",
                "ocr",
                "atp",
                "oxidative stress",
                "ros",
                "glutathione",
                "gsh",
                "er stress",
                "phospholipidosis",
                "steatosis",
            ],
        )

    if tier_key == "reactive_metabolite_immune":
        if not bool(keyword_hits or endpoint_hits):
            return False
        direct_reactive_metabolite = _has_any(
            full_text,
            [
                "reactive metabolite",
                "bioactivation",
                "gsh adduct",
                "glutathione adduct",
                "cysteine trapping",
                "cyanide trapping",
                "acyl glucuronide",
                "metabolic activation",
            ],
        )
        metabolic_covalent_binding = _has_any(full_text, ["covalent binding", "protein adduct", "drug protein adduct", "drug-protein adduct"]) and _has_any(
            full_text,
            ["liver", "hepatic", "hepatocyte", "microsome", "microsomal", "s9", "nadph", "glutathione", "gsh", "metabolism", "metabolic"],
        )
        immune_dili = _has_any(full_text, ["idiosyncratic", "immune mediated", "immune-mediated", "hla", "t cell"]) and has_liver_context
        return direct_reactive_metabolite or metabolic_covalent_binding or immune_dili or (target_hit and _has_bioactivation_or_liver_injury_context(full_text))

    if tier_key == "hepatic_cell_exposure_context":
        if not bool(keyword_hits or endpoint_hits):
            return False
        if _is_antiviral_hepg2_control_cytotoxicity(described_endpoint_text):
            return False
        if _is_generic_hepg2_cancer_cytotoxicity_noise(described_endpoint_text):
            return False
        if _is_hepatoprotective_efficacy_noise(described_endpoint_text):
            return False
        if _is_liver_genotoxicity_noise(described_endpoint_text):
            return False
        if _is_generic_target_biology_noise(described_endpoint_text):
            return False
        hepatic_cell = _has_any(
            described_endpoint_text,
            [
                "primary hepatocyte",
                "hepatocyte",
                "heparg",
                "hepg2",
                "liver spheroid",
                "hepatic spheroid",
                "liver organoid",
                "hepatic organoid",
            ],
        )
        injury_endpoint = _has_any(
            full_text,
            ["viability", "cytotoxicity", "ldh", "apoptosis", "necrosis", "caspase", "high content", "high-content"],
        )
        omics_or_exposure = has_liver_context and _has_any(
            full_text,
            [
                "toxicogenomics",
                "transcriptomic",
                "dili signature",
                "liver stress",
                "high daily dose",
                "high lipophilicity",
                "rule of two",
                "cationic amphiphilicity",
                "liver accumulation",
            ],
        )
        return (hepatic_cell and injury_endpoint) or omics_or_exposure

    return False


def _target_hit_for_tier(tier_key: str, genes: set[str], assay_text: str) -> bool:
    if tier_key == "cholestasis_transporter":
        return bool(genes & HEPATOBILIARY_TARGET_GENES)
    if tier_key == "reactive_metabolite_immune":
        return bool(genes & HEPATIC_METABOLISM_TARGET_GENES) and _has_bioactivation_or_liver_injury_context(assay_text)
    return False


def _weak_context_flags(full_text: str, matches: MatchResult) -> set[str]:
    if _has_strong_dili_positive(matches):
        return set()
    return {term for term in WEAK_TERMS if contains_phrase(full_text, term)}


def _has_strong_dili_positive(matches: MatchResult) -> bool:
    strong_tiers = {
        "direct_human_dili",
        "in_vivo_liver_injury",
        "cholestasis_transporter",
        "mitochondrial_organelle_stress",
        "reactive_metabolite_immune",
    }
    if any(matches.tier_matches.get(tier) for tier in strong_tiers):
        return True
    evidence = set(matches.matched_keywords) | set(matches.matched_endpoints)
    return bool(evidence & set(STRONG_DILI_TERMS))


def _has_strong_dili_assay_context(text: str) -> bool:
    return _has_any(text, STRONG_DILI_TERMS) or (
        _has_liver_context(text)
        and _has_any(
            text,
            [
                "toxicity",
                "injury",
                "necrosis",
                "cholestasis",
                "mitochondrial",
                "reactive metabolite",
                "bioactivation",
                "hepatocyte",
                "heparg",
            ],
        )
    )


def _has_bioactivation_or_liver_injury_context(text: str) -> bool:
    return _has_any(
        text,
        [
            "bioactivation",
            "reactive metabolite",
            "metabolic activation",
            "covalent binding",
            "gsh adduct",
            "glutathione adduct",
            "liver injury",
            "hepatotoxicity",
            "hepatotoxic",
            "hepatocyte",
            "liver microsome",
            "hepatic microsome",
        ],
    )


def _has_liver_context(text: str) -> bool:
    return _has_any(text, LIVER_CONTEXT_TERMS)


def _has_liver_lab_endpoint(text: str) -> bool:
    return _has_any(
        text,
        [
            "alt",
            "ast",
            "alanine aminotransferase",
            "aspartate aminotransferase",
            "alp",
            "alkaline phosphatase",
            "bilirubin",
            "ggt",
            "transaminase",
            "liver enzyme",
        ],
    )


def _is_liver_cancer_efficacy_noise(text: str) -> bool:
    return _has_any(text, LIVER_CANCER_EFFICACY_NOISE)


def _is_metabolic_efficacy_liver_disease_noise(text: str) -> bool:
    return _has_any(
        text,
        [
            "hypolipidemic",
            "hyperlipidemic",
            "hyperlipidemia",
            "normolipidemic",
            "anti hyperlipidemic",
            "anti-hyperlipidemic",
            "antihyperlipidemic",
            "antihypercholesterolemic",
            "cholesterol diet",
            "high cholesterol diet",
            "high fat diet",
            "hfd-fed",
            "hfd fed",
            "hfd-induced",
            "hfd induced",
            "cdahfd",
            "high fat dietary",
            "antidiabetic activity",
            "ob/ob",
            "db/db",
            "hfd/ccl4",
            "anti-hepatic steatosis",
            "anti hepatic steatosis",
            "nonalcoholic hepatic steatosis",
            "non-alcoholic hepatic steatosis",
            "nonalcoholic steatohepatitis",
            "non-alcoholic steatohepatitis",
            "nonalcoholic fatty liver disease",
            "non-alcoholic fatty liver disease",
            "nafld",
            "nash",
            "lipid lowering",
            "lipid-lowering",
            "triglyceride",
        ],
    )


def _is_antiinfective_efficacy_noise(text: str) -> bool:
    return _has_any(text, EFFICACY_NOISE)


def _is_hepatoprotective_efficacy_noise(text: str) -> bool:
    return _has_any(
        text,
        [
            "hepatoprotective",
            "hepato protective",
            "cytoprotective activity",
            "cytoprotection",
            "neuroprotective activity",
            "protective activity",
            "protective effect",
            "protection against",
            "protection in ",
            "inhibition of liver injury",
            "liver protection index",
            "increase in liver protection",
            "rescue of atp",
            "atp rescue",
            "suppression of rotenone-induced",
            "rotenone-induced atp depletion",
            "rotenone induced atp depletion",
            "suppression of h2o2-induced",
            "suppression of h2o2 induced",
            "inhibition of h2o2-induced",
            "inhibition of h2o2 induced",
            "reduction of h2o2-induced",
            "reduction of h2o2 induced",
            "antioxidant activity against h2o2",
            "h2o2-induced ros accumulation",
            "h2o2 induced ros accumulation",
            "prevention of liver",
            "prevention of hepatic",
            "improving ccl4",
            "anti-fibrotic effect in improving",
            "protection from acetaminophen",
            "protection against acetaminophen",
            "protective activity against acetaminophen",
            "protective activity against liver necrosis",
            "protective effect against liver necrosis",
            "protective activity against liver injury",
            "protective effect against liver injury",
            "d galn induced cytotoxicity",
            "d-galn-induced cytotoxicity",
        ],
    )


def _is_non_dili_safety_noise(text: str) -> bool:
    return _has_any(text, NON_DILI_SAFETY_NOISE)


def _is_generic_target_potency_noise(text: str) -> bool:
    if not _has_any(text, GENERIC_TARGET_NOISE):
        return False
    return not _has_any(text, ["bsep", "bile acid", "taurocholate", "bioactivation", "reactive metabolite", "hepatotoxic"])


def _is_generic_cyp_inhibition_noise(text: str) -> bool:
    if not _has_any(text, GENERIC_CYP_INHIBITION_TERMS):
        return False
    if _has_any(text, ["bioactivation", "reactive metabolite", "metabolic activation", "hepatotoxic", "liver injury"]):
        return False
    return _has_any(text, ["cyp", "cyp3a4", "cyp2c9", "cyp2c19", "cyp2d6", "cyp1a2"])


def _is_abcb1_or_pgp_noise(text: str) -> bool:
    return _has_any(text, ["abcb1", "mdr1", "p gp", "p-gp", "pgp", "p glycoprotein", "p-glycoprotein"])


def _is_nonmetabolic_covalent_binding_noise(text: str) -> bool:
    if not _has_any(text, ["covalent binding", "irreversible inhibition"]):
        return False
    if _has_any(text, ["reactive metabolite", "bioactivation", "gsh", "glutathione", "microsome", "microsomal", "nadph", "liver", "hepatic", "hepatocyte", "s9"]):
        return False
    return _has_any(text, ["estrogen receptor", "opioid receptor", "brain membrane", "receptor", "receptor-type", "kinase", "binding to"])


def _is_kidney_or_renal_context_noise(text: str) -> bool:
    if not _has_any(text, ["kidney", "renal"]):
        return False
    if _has_any(
        text,
        [
            "kidney microsome",
            "kidney microsomes",
            "renal microsome",
            "renal microsomes",
            "kidney of",
            "renal proximal",
            "necrosis in kidney",
        ],
    ):
        return True
    if _has_any(text, ["transporter", "quantitative pcr", "western", "branched dna", "rt-pcr"]) and not _has_any(
        text,
        ["liver", "hepatic", "hepatotoxicity", "hepatotoxic"],
    ):
        return True
    return False


def _is_generic_target_biology_noise(text: str) -> bool:
    if _has_any(
        text,
        [
            "pcsk9",
            "ldlr",
            "ldl uptake",
            "ldl receptor",
            "luciferase reporter",
            "reporter plasmid",
            "promoter",
            "abhd10",
            "pme-1",
            "activity-based protein profiling",
            "abpp",
            "ldha",
            "lactate production",
        ],
    ):
        return not _has_any(
            text,
            [
                "hepatotoxicity",
                "hepatotoxic",
                "liver injury",
                "dili",
                "mitochondrial toxicity",
                "cell viability",
                "cytotoxicity",
                "ldh release",
                "apoptosis",
                "caspase",
            ],
        )
    return False


def _is_generic_nonhepatic_cytotoxicity(text: str) -> bool:
    if not _has_any(text, ["cytotoxicity", "viability", "gi50", "cc50", "growth inhibition", "proliferation"]):
        return False
    return not _has_liver_context(text)


def _is_asbt_or_ileal_bile_acid_transport_noise(text: str) -> bool:
    return _has_any(
        text,
        [
            "asbt",
            "slc10a2",
            "ileal sodium",
            "apical sodium dependent bile acid transporter",
            "apical sodium codependent bile acid transporter",
            "ileal ring",
            "ileal taurocholate",
            "ileal brush border",
            "brush border membrane vesicles",
            "intestinal bile acid",
        ],
    )


def _is_non_hepatobiliary_bile_acid_adjacent_target_noise(text: str) -> bool:
    if _has_any(
        text,
        [
            "aldo-keto reductase",
            "20-alpha hsd",
            "20 alpha hsd",
            "akr1c",
            "beta-glucosidase",
            "glucosylceramidase",
            "glucocerebrosidase",
            "gba1",
            "gba2",
            "carboxylic ester hydrolase",
            "cryptosporidium",
            "clostridioides difficile",
        ],
    ):
        return not _has_any(
            text,
            [
                "bsep",
                "abcb11",
                "mrp2",
                "abcc2",
                "mrp3",
                "abcc3",
                "mrp4",
                "abcc4",
                "ntcp",
                "slc10a1",
                "oatp1b1",
                "slco1b1",
                "oatp1b3",
                "slco1b3",
                "fxr",
                "nr1h4",
                "bile acid receptor",
                "cholestasis",
                "cholestatic",
                "hepatotoxicity",
                "liver injury",
                "dili",
            ],
        )
    return False


def _is_antiviral_hepg2_control_cytotoxicity(text: str) -> bool:
    if not _has_any(text, ["2 2 15", "hepg2 2 2 15", "hbv", "hepatitis b", "antiviral"]):
        return False
    return _has_any(text, ["cytotoxicity", "viability", "ic50", "cc50", "selectivity"])


def _is_generic_hepg2_cancer_cytotoxicity_noise(text: str) -> bool:
    if not _has_any(text, ["hepg2"]):
        return False
    if _has_any(
        text,
        [
            "hepg2/c3a",
            "hepg2 c3a",
            "c3a",
            "admet",
            "hepatotoxicity",
            "hepatotoxic",
            "liver injury",
            "dili",
            "primary hepatocyte",
            "heparg",
            "ldh release",
            "apoptosis",
            "caspase",
            "mitochondrial",
            "high content hepatotoxicity",
            "high-content hepatotoxicity",
            "liver spheroid",
            "hepatic spheroid",
            "liver organoid",
            "hepatic organoid",
        ],
    ):
        return False
    return _has_any(
        text,
        [
            "cytotoxicity against",
            "cytotoxicity",
            "cell viability",
            "viability",
            "mtt",
            "alamar blue",
            "acid phosphatase",
            "growth inhibition",
            "proliferation",
            "ic50",
            "cc50",
            "gi50",
        ],
    )


def _is_liver_genotoxicity_noise(text: str) -> bool:
    return _has_any(
        text,
        [
            "dna single strand break",
            "dna single strand breaks",
            "dna single-strand break",
            "dna single-strand breaks",
            "comet assay",
            "genotoxicity",
        ],
    ) and not _has_any(
        text,
        ["hepatotoxicity", "liver injury", "dili"],
    )


def _reason_for(tier_key: str, matches: MatchResult) -> str:
    evidence = matches.matched_endpoints or matches.matched_keywords or matches.matched_targets
    evidence_text = "、".join(evidence[:4]) if evidence else "相关证据"
    if tier_key == "direct_human_dili":
        return f"命中 {evidence_text}，属于 direct human/clinical DILI anchor。"
    if tier_key == "in_vivo_liver_injury":
        return f"命中 {evidence_text}，属于 in vivo liver injury phenotype 或 liver clinical pathology evidence。"
    if tier_key == "cholestasis_transporter":
        return f"命中 {evidence_text}，属于 cholestasis / hepatobiliary transporter DILI mechanism evidence。"
    if tier_key == "mitochondrial_organelle_stress":
        return f"命中 {evidence_text}，属于 hepatic mitochondrial / oxidative / organelle stress evidence。"
    if tier_key == "reactive_metabolite_immune":
        return f"命中 {evidence_text}，属于 reactive metabolite / bioactivation / immune-idiosyncratic DILI evidence。"
    if tier_key == "hepatic_cell_exposure_context":
        return f"命中 {evidence_text}，属于 hepatic cell injury 或 exposure/property modifier evidence。"
    return "未命中 DILI 相关 assay 证据。"


def _target_genes(row: dict[str, Any]) -> set[str]:
    value = row.get("target_genes")
    if isinstance(value, str):
        tokens = value.replace(";", " ").replace(",", " ").split()
    elif isinstance(value, (list, tuple, set)):
        tokens = [str(item) for item in value]
    else:
        tokens = []
    return {token.strip().upper() for token in tokens if token}


def _join_values(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value if item is not None)
    return str(value)


def _has_any(text: str, phrases: list[str] | tuple[str, ...] | set[str]) -> bool:
    return bool(match_phrases(text, phrases))


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
