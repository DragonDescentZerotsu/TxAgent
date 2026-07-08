"""Endpoint grouping rules for DILI molecule-level evidence."""

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
    """Assign a DILI evidence row to a Tier.endpoint_group bucket."""
    tier = str(row.get("assay_tier") or row.get("tier") or "").strip() or "unknown"
    standard_type = str(row.get("standard_type") or "").strip()
    endpoint_text = normalize_text(standard_type)
    context_text = _context_text(row)
    genes = _target_genes(row)

    if tier == "Tier 1":
        group = _tier1_group(endpoint_text, context_text)
    elif tier == "Tier 2":
        group = _tier2_group(endpoint_text, context_text)
    elif tier == "Tier 3":
        group = _tier3_group(endpoint_text, context_text, genes)
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
    full_text = endpoint_text + " " + context_text
    if _has_any(
        full_text,
        [
            "acute liver failure",
            "fulminant hepatic failure",
            "liver failure",
            "liver transplant",
            "fatal liver injury",
            "liver related death",
            "liver-related death",
            "withdrawal",
            "withdrawn",
            "boxed warning",
            "black box warning",
            "discontinuation",
            "contraindication",
        ],
    ):
        return "severe_liver_outcome_or_regulatory_signal"
    if _has_any(
        full_text,
        [
            "alt",
            "ast",
            "alanine aminotransferase",
            "aspartate aminotransferase",
            "bilirubin",
            "alkaline phosphatase",
            "alp",
            "ggt",
            "jaundice",
            "hy's law",
            "hys law",
            "hy law",
            "transaminase",
            "liver enzyme",
        ],
    ):
        return "human_liver_laboratory_signal"
    if _has_any(
        full_text,
        [
            "drug induced liver injury",
            "drug-induced liver injury",
            "dili",
            "hepatotoxicity",
            "hepatotoxic",
            "liver injury",
            "hepatic injury",
            "liver toxicity",
            "hepatic toxicity",
            "drug induced hepatitis",
            "drug-induced hepatitis",
            "toxic hepatitis",
        ],
    ):
        return "human_dili_or_hepatotoxicity"
    return CONTEXT_DEPENDENT_GROUP


def _tier2_group(endpoint_text: str, context_text: str) -> str:
    full_text = endpoint_text + " " + context_text
    if _has_any(
        full_text,
        [
            "histopathology",
            "pathology",
            "necrosis",
            "degeneration",
            "inflammation",
            "bile duct",
            "fibrosis",
            "liver weight",
            "hepatocyte hypertrophy",
            "portal inflammation",
        ],
    ):
        return "in_vivo_liver_histopathology"
    if _has_any(
        full_text,
        [
            "alt",
            "ast",
            "alanine aminotransferase",
            "aspartate aminotransferase",
            "alp",
            "alkaline phosphatase",
            "bilirubin",
            "ggt",
            "bile acid",
            "transaminase",
            "liver enzyme",
        ],
    ):
        return "in_vivo_liver_clinical_chemistry"
    if _has_any(
        full_text,
        [
            "noael",
            "loael",
            "mtd",
            "maximum tolerated dose",
            "toxic dose",
            "repeated dose",
            "repeat dose",
            "subacute",
            "subchronic",
            "chronic",
            "toxicokinetic",
            "exposure margin",
        ],
    ):
        return "in_vivo_hepatotoxic_dose_or_margin"
    return CONTEXT_DEPENDENT_GROUP


def _tier3_group(endpoint_text: str, context_text: str, genes: set[str]) -> str:
    full_text = endpoint_text + " " + context_text
    if _has_any(
        full_text,
        [
            "abcb1",
            "mdr1",
            "p gp",
            "p-gp",
            "pgp",
            "p glycoprotein",
            "p-glycoprotein",
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
    ):
        return CONTEXT_DEPENDENT_GROUP
    if genes & {"ABCB11"} or _has_any(
        full_text,
        [
            "bsep",
            "bile salt export pump",
            "abcb11",
            "bile acid efflux",
            "taurocholate efflux",
            "canalicular bile acid",
        ],
    ):
        return "bsep_or_bile_acid_efflux"
    if genes & {"ABCC2", "ABCC3", "ABCC4", "ABCB4", "SLC10A1", "SLCO1B1", "SLCO1B3"} or _has_any(
        full_text,
        [
            "mrp2",
            "abcc2",
            "mrp3",
            "abcc3",
            "mrp4",
            "abcc4",
            "mdr3",
            "abcb4",
            "ntcp",
            "slc10a1",
            "oatp1b1",
            "slco1b1",
            "oatp1b3",
            "slco1b3",
            "hepatobiliary transporter",
            "bile acid uptake",
        ],
    ):
        return "hepatobiliary_transporter_panel"
    if _has_any(full_text, ["cholestasis", "cholestatic", "bile acid accumulation", "impaired bile flow", "bile canalicular"]):
        return "cholestasis_or_bile_acid_accumulation"
    return CONTEXT_DEPENDENT_GROUP


def _tier4_group(endpoint_text: str, context_text: str) -> str:
    full_text = endpoint_text + " " + context_text
    if _has_any(
        full_text,
        [
            "mitochondrial toxicity",
            "mitochondrial dysfunction",
            "mitochondrial membrane potential",
            "mmp loss",
            "oxygen consumption",
            "ocr",
            "respiratory chain",
            "electron transport chain",
            "complex i",
            "complex ii",
            "complex iii",
            "complex iv",
            "mitochondrial respiration",
            "mitochondrial swelling",
        ],
    ):
        return "mitochondrial_function_or_respiration"
    if _has_any(
        full_text,
        [
            "atp depletion",
            "cellular atp",
            "oxidative stress",
            "reactive oxygen species",
            "ros",
            "glutathione",
            "gsh",
            "glutathione depletion",
            "gsh depletion",
            "nrf2",
            "nfe2l2",
            "antioxidant response",
            "jnk activation",
        ],
    ):
        return "energy_failure_and_oxidative_stress"
    if _has_any(
        full_text,
        [
            "er stress",
            "endoplasmic reticulum",
            "unfolded protein response",
            "upr",
            "phospholipidosis",
            "lysosomal",
            "steatosis",
            "lipid accumulation",
        ],
    ):
        return "er_lysosomal_lipid_stress"
    return CONTEXT_DEPENDENT_GROUP


def _tier5_group(endpoint_text: str, context_text: str) -> str:
    full_text = endpoint_text + " " + context_text
    if _has_any(
        full_text,
        [
            "reactive metabolite",
            "covalent binding",
            "protein adduct",
            "drug protein adduct",
            "drug-protein adduct",
            "gsh adduct",
            "glutathione adduct",
            "cysteine trapping",
            "cyanide trapping",
            "quinone imine",
            "quinoneimine",
            "quinone methide",
            "acyl glucuronide",
            "iminium ion",
        ],
    ):
        return "reactive_metabolite_or_covalent_binding"
    if _has_any(
        full_text,
        [
            "bioactivation",
            "metabolic activation",
            "cyp mediated bioactivation",
            "cyp-mediated bioactivation",
            "metabolite mediated toxicity",
            "metabolite-mediated toxicity",
            "liver microsome",
            "hepatic microsome",
            "s9",
        ],
    ):
        return "hepatic_metabolism_bioactivation"
    if _has_any(
        full_text,
        [
            "idiosyncratic",
            "immune mediated",
            "immune-mediated",
            "hla",
            "t cell",
            "cytokine",
            "inflammasome",
            "danger signal",
            "adaptive immune",
        ],
    ):
        return "immune_or_idiosyncratic_context"
    return CONTEXT_DEPENDENT_GROUP


def _tier6_group(endpoint_text: str, context_text: str) -> str:
    full_text = endpoint_text + " " + context_text
    if _has_any(
        full_text,
        [
            "primary human hepatocyte",
            "primary hepatocyte",
            "hepatocyte",
            "heparg",
            "hepg2",
            "liver spheroid",
            "hepatic spheroid",
            "liver organoid",
            "hepatic organoid",
            "ldh",
            "apoptosis",
            "necrosis",
            "caspase",
            "cytotoxicity",
            "viability",
        ],
    ):
        return "hepatocyte_or_hepatic_cell_injury"
    if _has_any(
        full_text,
        [
            "toxicogenomics",
            "transcriptomic",
            "gene expression",
            "liver stress gene",
            "dili signature",
            "metabolomics",
            "proteomics",
            "high content",
            "high-content",
        ],
    ):
        return "liver_omics_or_stress_signature"
    if _has_any(
        full_text,
        [
            "high daily dose",
            "high lipophilicity",
            "rule of two",
            "logp",
            "logd",
            "cationic amphiphilicity",
            "liver accumulation",
            "hepatic extraction",
            "hepatic metabolism",
            "extensive metabolism",
        ],
    ):
        return "exposure_dose_or_property_context"
    return CONTEXT_DEPENDENT_GROUP


def _direction_and_strength(tier: str, group: str, row: Mapping[str, Any]) -> tuple[str, str]:
    if group == CONTEXT_DEPENDENT_GROUP:
        return "context_dependent", "weak"
    if _is_negative_or_inactive(row):
        if tier in {"Tier 1", "Tier 2"}:
            return "argues_against_dili_risk", "moderate"
        return "argues_against_dili_risk", "weak"
    if tier == "Tier 1":
        return "clinical_dili_signal", "strong"
    if tier == "Tier 2":
        if group == "in_vivo_liver_histopathology" and _is_liver_weight_only_signal(row):
            return "in_vivo_liver_injury_signal", "moderate"
        return "in_vivo_liver_injury_signal", "strong"
    if tier == "Tier 3":
        return "cholestasis_or_bile_acid_transport_risk", "moderate"
    if tier == "Tier 4":
        if group == "energy_failure_and_oxidative_stress" and _is_unclear_gsh_signal(row):
            return "neutral_or_unclear", "weak"
        return "mitochondrial_or_organelle_stress_risk", "moderate"
    if tier == "Tier 5":
        if group == "immune_or_idiosyncratic_context":
            return "immune_or_idiosyncratic_context", "moderate"
        return "reactive_metabolite_or_bioactivation_risk", "moderate"
    if tier == "Tier 6":
        if group == "exposure_dose_or_property_context":
            return "exposure_or_property_context", "weak"
        return "hepatic_cell_injury_risk", "weak"
    return "neutral_or_unclear", "weak"


def _context_text(row: Mapping[str, Any]) -> str:
    return join_text_parts(
        row.get("assay_description") or row.get("description"),
        row.get("assay_type"),
        row.get("assay_cell_type"),
        row.get("assay_tissue"),
        row.get("organism") or row.get("assay_organism"),
        row.get("target_pref_name"),
        row.get("target_genes"),
        row.get("target_synonyms"),
        row.get("component_descriptions"),
        row.get("assay_standard_types"),
        row.get("matched_keywords"),
        row.get("matched_endpoints"),
        row.get("activity_comment"),
        row.get("standard_type"),
    )


def _target_genes(row: Mapping[str, Any]) -> set[str]:
    value = row.get("target_genes")
    if isinstance(value, str):
        tokens = value.replace(";", " ").replace(",", " ").split()
    elif isinstance(value, (list, tuple, set)):
        tokens = [str(item) for item in value]
    else:
        tokens = []
    return {token.strip().upper() for token in tokens if token}


def _is_negative_or_inactive(row: Mapping[str, Any]) -> bool:
    text = join_text_parts(row.get("activity_comment"), row.get("standard_relation"), row.get("standard_text_value"))
    return _has_any(
        text,
        [
            "inactive",
            "not active",
            "negative",
            "non hepatotoxic",
            "non-hepatotoxic",
            "not hepatotoxic",
            "no hepatotoxicity",
            "no liver injury",
            "no alt elevation",
            "no effect",
            "not toxic",
        ],
    )


def _is_liver_weight_only_signal(row: Mapping[str, Any]) -> bool:
    text = _context_text(row)
    if not _has_any(text, ["liver weight", "relative liver weight"]):
        return False
    return not _has_any(
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
            "bile acid",
            "transaminase",
            "liver enzyme",
            "histopathology",
            "pathology",
            "necrosis",
            "degeneration",
            "inflammation",
            "fibrosis",
            "bile duct",
            "hepatocyte hypertrophy",
        ],
    )


def _is_unclear_gsh_signal(row: Mapping[str, Any]) -> bool:
    text = _context_text(row)
    if not _has_any(text, ["gsh", "glutathione"]):
        return False
    if _has_any(
        text,
        [
            "gsh depletion",
            "glutathione depletion",
            "depleted gsh",
            "depleted glutathione",
            "reduced gsh",
            "oxidative stress",
            "reactive oxygen species",
            "ros",
        ],
    ):
        return False
    return True


def _reason_for(tier: str, group: str, standard_type: str) -> str:
    if group == CONTEXT_DEPENDENT_GROUP:
        return f"{tier} evidence could not be assigned to a specific DILI endpoint group."
    endpoint = f" endpoint={standard_type}" if standard_type else ""
    return f"Assigned {tier}.{group} from DILI assay context and endpoint.{endpoint}"


def _has_any(text: str, phrases: list[str] | tuple[str, ...] | set[str]) -> bool:
    return bool(match_phrases(text, phrases))
