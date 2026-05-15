"""Endpoint grouping rules for ClinTox molecule-level evidence."""

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
    """Assign a ClinTox evidence row to a Tier.endpoint_group bucket."""
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
    elif tier == "Tier 7":
        group = _tier7_group(endpoint_text, context_text, genes)
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
    if _has_any(full_text, ["dose limiting toxicity", "dose limiting", "dlt", "adverse event", "serious adverse event"]):
        return "clinical_toxicity_or_trial_failure"
    if _has_any(full_text, ["maximum tolerated dose", "mtd", "noael", "loael", "therapeutic index", "safety margin"]):
        return "human_maximum_tolerated_dose_or_safety_margin"
    if _has_any(full_text, ["alt", "ast", "bilirubin", "creatinine", "bun", "qtc", "qt prolongation", "neutropenia", "thrombocytopenia"]):
        return "human_lab_toxicity_signal"
    if _has_any(full_text, ["clinical toxicity", "tolerability", "withdrawal due to toxicity", "boxed warning", "black box warning"]):
        return "clinical_toxicity_or_trial_failure"
    return CONTEXT_DEPENDENT_GROUP


def _tier2_group(endpoint_text: str, context_text: str) -> str:
    full_text = endpoint_text + " " + context_text
    if _has_any(full_text, ["ld50", "lc50", "td50", "maximum tolerated dose", "mtd", "noael", "loael", "toxic dose"]):
        return "in_vivo_ld50_lc50_mtd_noael_loael"
    if _has_any(full_text, ["histopathology", "pathology", "necrosis", "degeneration", "inflammation", "organ weight"]):
        return "in_vivo_histopathology_or_organ_weight"
    if _has_any(full_text, ["toxicokinetic", "auc at noael", "cmax at noael", "exposure margin", "safety margin"]):
        return "toxicokinetic_exposure_margin"
    if _has_any(full_text, ["acute toxicity", "repeated dose toxicity", "repeat dose toxicity", "systemic toxicity", "mortality", "body weight loss"]):
        return "in_vivo_acute_or_repeat_dose_toxicity"
    return CONTEXT_DEPENDENT_GROUP


def _tier3_group(endpoint_text: str, context_text: str, genes: set[str]) -> str:
    full_text = endpoint_text + " " + context_text
    if genes & {"KCNH2"} or _has_any(full_text, ["herg", "ikr", "potassium current", "thallium influx", "tail current"]):
        return "cardiac_herg_or_ikr_block"
    if _has_any(full_text, ["qt prolongation", "qtc prolongation", "torsade", "repolarization", "action potential duration", "field potential duration"]):
        return "cardiac_qt_or_repolarization"
    if _has_any(full_text, ["cardiomyocyte", "cardiac contractility", "beat rate", "beating", "calcium transient", "troponin"]):
        return "cardiac_contractility_or_cardiomyocyte_toxicity"
    if _has_any(full_text, ["drug induced liver injury", "dili", "hepatotoxicity", "liver injury", "cholestasis", "alt", "ast", "bilirubin"]):
        return "hepatic_dili_or_liver_injury"
    if genes & {"ABCB11", "SLC10A1", "ABCC2", "ABCB4"} or _has_any(
        full_text,
        ["bsep", "bile salt export pump", "bile acid transport", "taurocholate", "ntcp", "mrp2", "mdr3"],
    ):
        return "hepatic_bile_acid_transport_or_bsep"
    if _has_any(full_text, ["mitochondrial toxicity", "mitochondrial membrane potential", "oxygen consumption", "atp depletion", "ros", "glutathione"]):
        return "hepatic_mitochondrial_or_oxidative_stress"
    if _has_any(full_text, ["hepatocyte", "hepg2", "heparg", "steatosis", "lipid accumulation", "phospholipidosis"]):
        return "hepatic_cell_injury_or_steatosis"
    if genes & {"HAVCR1", "LCN2"} or _has_any(
        full_text,
        ["nephrotoxicity", "renal toxicity", "kidney toxicity", "proximal tubule", "kim 1", "kim-1", "ngal", "creatinine", "bun"],
    ):
        return "renal_tubular_injury_or_nephrotoxicity"
    if genes & {"SLC22A6", "SLC22A8", "SLC22A2", "SLC47A1", "SLC47A2"} or _has_any(
        full_text,
        ["oat1", "oat3", "oct2", "mate1", "mate2", "renal uptake", "renal secretion"],
    ):
        return "renal_transporter_accumulation_or_inhibition"
    if _has_any(full_text, ["neurotoxicity", "neuronal toxicity", "neuron viability", "neurite outgrowth", "neural differentiation"]):
        return "neurotoxicity_or_neuronal_viability"
    if genes & {"ACHE", "GABRA1", "GABRA2", "GRIN1", "GRIN2A", "SCN1A", "SCN2A"} or _has_any(
        full_text,
        ["seizure", "convulsion", "microelectrode array", "neuronal firing", "gaba", "nmda", "acetylcholinesterase"],
    ):
        return "neurofunctional_or_seizure_liability"
    if _has_any(full_text, ["developmental neurotoxicity", "dnt", "neural progenitor", "myelination"]):
        return "developmental_neurotoxicity"
    if _has_any(full_text, ["myelosuppression", "bone marrow toxicity", "neutropenia", "thrombocytopenia", "hematopoietic", "cfu gm"]):
        return "hematologic_toxicity_or_myelosuppression"
    if _has_any(full_text, ["immunotoxicity", "cytokine release", "il 6", "tnf", "immune activation", "complement activation"]):
        return "immunotoxicity_or_cytokine_release"
    if _has_any(full_text, ["reproductive toxicity", "developmental toxicity", "embryotoxicity", "teratogenicity", "fetal toxicity", "zebrafish embryo"]):
        return "reproductive_or_developmental_toxicity"
    if genes & {"ESR1", "ESR2", "AR", "CYP19A1", "THRB", "PGR", "NR3C1", "PPARG", "PPARA", "AHR"} or _has_any(
        full_text,
        ["estrogen receptor", "androgen receptor", "aromatase", "thyroid receptor", "ppar", "ahr", "endocrine"],
    ):
        return "endocrine_nuclear_receptor_disruption"
    return CONTEXT_DEPENDENT_GROUP


def _tier4_group(endpoint_text: str, context_text: str) -> str:
    full_text = endpoint_text + " " + context_text
    if _has_any(full_text, ["ames", "bacterial reverse mutation", "salmonella", "revertant", "frameshift", "base pair substitution"]):
        return "ames_or_bacterial_mutagenicity"
    if _has_any(full_text, ["micronucleus", "chromosomal aberration", "chromosome aberration", "mouse lymphoma", "hprt", "comet assay", "dna strand break"]):
        return "in_vitro_mammalian_genotoxicity"
    if _has_any(full_text, ["p53", "gamma h2ax", "h2ax", "atad5", "dna damage", "dna repair", "topoisomerase poison"]):
        return "dna_damage_response"
    if _has_any(full_text, ["carcinogenicity", "tumorigenicity", "carcinogen", "td50", "tumor incidence", "cell transformation"]):
        return "carcinogenicity_or_tumorigenicity"
    return CONTEXT_DEPENDENT_GROUP


def _tier5_group(endpoint_text: str, context_text: str) -> str:
    full_text = endpoint_text + " " + context_text
    if _has_any(full_text, ["mitochondrial membrane potential", "mitochondria", "oxygen consumption", "cellular atp", "uncoupler"]):
        return "mitochondrial_toxicity_or_energy_stress"
    if _has_any(full_text, ["oxidative stress", "reactive oxygen species", "ros", "nrf2", "antioxidant response", "glutathione"]):
        return "oxidative_stress_or_nrf2_are"
    if _has_any(full_text, ["er stress", "endoplasmic reticulum", "unfolded protein", "heat shock", "hsp70", "hsp90"]):
        return "er_stress_heat_shock_unfolded_protein"
    if _has_any(full_text, ["nf kb", "nf-kb", "ap 1", "hif", "creb", "stat", "cytokine", "il 8", "tnf"]):
        return "inflammatory_or_stress_transcription"
    if _has_any(full_text, ["phospholipidosis", "lysotracker", "lysosomal", "cationic amphiphilic"]):
        return "lysosomal_or_phospholipidosis"
    if _has_any(full_text, ["high content", "cellular morphology", "nuclear size", "cell count", "membrane integrity", "ldh", "apoptosis", "caspase"]):
        return "general_cell_stress_or_morphology"
    return CONTEXT_DEPENDENT_GROUP


def _tier6_group(endpoint_text: str, context_text: str) -> str:
    full_text = endpoint_text + " " + context_text
    if _has_any(full_text, ["growth inhibition", "proliferation", "antiproliferative", "cell growth", "gi50", "tgi", "colony formation"]):
        return "antiproliferative_or_growth_inhibition"
    if _has_any(
        full_text,
        [
            "cytotoxicity",
            "cell viability",
            "cell survival",
            "cell death",
            "cc50",
            "tc50",
            "lc50",
            "ldh release",
            "neutral red",
            "propidium iodide",
            "caspase",
        ],
    ):
        return "general_cytotoxicity_or_viability"
    return CONTEXT_DEPENDENT_GROUP


def _tier7_group(endpoint_text: str, context_text: str, genes: set[str]) -> str:
    full_text = endpoint_text + " " + context_text
    if genes & {"HTR2A", "HTR2B", "DRD2", "SLC6A3", "SLC6A4", "SLC6A2", "GABRA1", "GABRA2", "GRIN1", "SCN5A"} or _has_any(
        full_text,
        ["5 ht2b", "htr2b", "5 ht2a", "htr2a", "dopamine transporter", "serotonin transporter", "gaba receptor", "nmda receptor", "sodium channel"],
    ):
        return "cardiac_or_cns_offtarget_binding"
    if any(gene.startswith("CYP") for gene in genes) or _has_any(full_text, ["cyp inhibition", "cyp induction", "time dependent inhibition", "mechanism based inhibition"]):
        return "cytochrome_p450_inhibition_or_induction"
    if genes & {"ABCB1", "ABCG2", "SLCO1B1", "SLCO1B3", "SLC22A6", "SLC22A8", "SLC22A2", "SLC47A1", "ABCB11"} or _has_any(
        full_text,
        ["p gp", "bcrp", "oatp1b1", "oatp1b3", "oat1", "oat3", "oct2", "mate1", "bsep"],
    ):
        return "transporter_ddi_or_exposure_liability"
    if _has_any(full_text, ["reactive metabolite", "glutathione adduct", "gsh adduct", "covalent binding", "protein adduct", "bioactivation"]):
        return "reactive_metabolite_or_covalent_liability"
    return CONTEXT_DEPENDENT_GROUP


def _direction_and_strength(tier: str, group: str, row: Mapping[str, Any]) -> tuple[str, str]:
    if group == CONTEXT_DEPENDENT_GROUP:
        return "context_dependent", "weak"
    if _is_negative_or_inactive(row):
        return "argues_against_clinical_toxicity", "moderate" if tier in {"Tier 1", "Tier 2"} else "weak"
    if tier == "Tier 7":
        return "drug_interaction_or_exposure_risk", "moderate"
    if group.startswith("cardiac_"):
        return "cardiotoxicity_risk", "strong" if tier == "Tier 3" else "moderate"
    if group.startswith("hepatic_"):
        return "hepatotoxicity_risk", "strong" if tier == "Tier 3" else "moderate"
    if group.startswith("renal_"):
        return "nephrotoxicity_risk", "moderate"
    if group.startswith("neuro"):
        return "neurotoxicity_risk", "moderate"
    if group.startswith("ames_") or "genotoxicity" in group or "carcinogenicity" in group or "dna_damage" in group:
        return "genotoxicity_or_carcinogenicity_risk", "strong" if group.startswith("ames_") else "moderate"
    if tier == "Tier 1":
        return "supports_clinical_toxicity", "strong"
    if tier == "Tier 2":
        return "supports_clinical_toxicity", "strong"
    if tier == "Tier 5":
        return "mitochondrial_or_cell_stress_risk", "moderate"
    if tier == "Tier 6":
        return "general_cytotoxicity_risk", "weak"
    return "mechanistic_context", "weak"


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
        row.get("matched_targets"),
        row.get("activity_comment"),
    )


def _target_genes(row: Mapping[str, Any]) -> set[str]:
    value = row.get("target_genes") or ""
    if isinstance(value, str):
        return {part.strip().upper() for part in value.replace("|", ",").replace(";", ",").split(",") if part.strip()}
    return {str(part).strip().upper() for part in value if str(part).strip()}


def _is_negative_or_inactive(row: Mapping[str, Any]) -> bool:
    text = join_text_parts(row.get("activity_comment"), row.get("standard_relation"))
    return _has_any(
        text,
        [
            "inactive",
            "not active",
            "negative",
            "non toxic",
            "non-toxic",
            "not toxic",
            "no toxicity",
            "no adverse effect",
            "not cytotoxic",
            "no effect",
        ],
    )


def _reason_for(tier: str, group: str, standard_type: str) -> str:
    if group == CONTEXT_DEPENDENT_GROUP:
        return f"{tier} evidence could not be assigned to a specific ClinTox endpoint group."
    endpoint = f" endpoint={standard_type}" if standard_type else ""
    return f"Assigned {tier}.{group} from ClinTox assay context and endpoint.{endpoint}"


def _has_any(text: str, phrases: list[str] | tuple[str, ...] | set[str]) -> bool:
    return bool(match_phrases(text, phrases))
