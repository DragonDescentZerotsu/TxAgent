"""Early context canonicalization and policy-independent validity for v5."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Mapping
from difflib import SequenceMatcher
from typing import Any

from tools.chembl_tool.common.starling.normalization.measurements import (
    format_number,
    parse_point_measurement,
)
from tools.chembl_tool.common.units import canonicalize_unit


CONTEXT_CANONICALIZATION_VERSION = "bioavailability_context_canonicalization.v2"
NORMALIZATION_DOMAIN_RULES_VERSION = "bioavailability_normalization_domains.v2"
UNKNOWN_TOKEN = "__unknown__"
UNMAPPED_PREFIX = "unmapped:"

FUZZY_ACCEPT_THRESHOLD = 0.92
FUZZY_WINNER_MARGIN = 0.08

_NULL_LIKE = {"", "nan", "none", "null", "na", "n/a", "-", "unspecified", "unknown"}
_SEPARATORS = re.compile(r"[\s_\-\u2010-\u2015\u2212]+")
_DOSE = re.compile(
    r"^\s*(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
    r"(?P<unit>kg|g|mg|(?:u|µ|μ)g|ng|mol|mmol|(?:u|µ|μ)mol|nmol)"
    r"\s*(?P<basis>/\s*(?:kg|m(?:2|\^2))|"
    r"(?:kg|m(?:2|\^2))\s*(?:\^?\s*-?\s*1|⁻¹))?"
    r"(?P<regimen>.*)$",
    re.IGNORECASE,
)
_MASS_TO_MG = {
    "kg": 1_000_000.0,
    "g": 1_000.0,
    "mg": 1.0,
    "µg": 0.001,
    "ng": 0.000001,
}
_MOLAR_TO_UMOL = {
    "mol": 1_000_000.0,
    "mmol": 1_000.0,
    "µmol": 1.0,
    "nmol": 0.001,
}

_REPORT_TYPE_ALIASES = {
    "absolute": "absolute",
    "absolute_bioavailability": "absolute",
    "relative": "relative_comparison",
    "relative_comparison": "relative_comparison",
    "systemic_availability": "systemic_availability",
    "extent_f": "extent_f",
    "extent_of_bioavailability": "extent_f",
    "apparent": "apparent",
    "apparent_bioavailability": "apparent",
}

_SPECIES_ALIASES = {
    "homo_sapiens": "human",
    "humans": "human",
    "human": "human",
    "man": "human",
    "men": "human",
    "woman": "human",
    "women": "human",
    "patient": "human",
    "patients": "human",
    "subject": "human",
    "subjects": "human",
    "volunteer": "human",
    "volunteers": "human",
    "adult": "human",
    "adults": "human",
    "child": "human",
    "children": "human",
    "rattus_norvegicus": "rat",
    "rat": "rat",
    "rats": "rat",
    "mus_musculus": "mouse",
    "mouse": "mouse",
    "mice": "mouse",
    "dog": "dog",
    "dogs": "dog",
    "canine": "dog",
    "beagle": "dog",
    "beagle_dog": "dog",
    "beagle_dogs": "dog",
    "rabbit": "rabbit",
    "rabbits": "rabbit",
    "pig": "pig",
    "pigs": "pig",
    "porcine": "pig",
    "minipig": "minipig",
    "monkey": "monkey",
    "monkeys": "monkey",
    "cynomolgus_macaque": "cynomolgus_monkey",
    "cynomolgus_monkey": "cynomolgus_monkey",
    "rhesus_macaque": "rhesus_monkey",
    "rhesus_monkey": "rhesus_monkey",
    "marmoset": "marmoset",
    "guinea_pig": "guinea_pig",
    "hamster": "hamster",
    "horse": "horse",
    "sheep": "sheep",
    "cat": "cat",
    "trout": "trout",
    "rainbow_trout": "rainbow_trout",
    "insect": "insect",
    "yeast": "yeast",
    "sf9": "spodoptera_frugiperda",
}

_ASSAY_PLATFORM_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("caco_2", ("caco-2", "caco 2", "caco2")),
    ("mdck", ("mdck",)),
    ("pampa", ("pampa",)),
    ("ussing_chamber", ("ussing",)),
    ("everted_gut_sac", ("everted gut", "gut sac")),
    ("intestinal_perfusion", ("intestinal perfusion", "single pass perfusion")),
    ("intestinal_microsomes", ("intestinal microsome",)),
    ("liver_microsomes", ("liver microsome", "hepatic microsome", "hlm")),
    ("hepatocytes", ("hepatocyte",)),
    ("enterocytes", ("enterocyte",)),
    ("s9_fraction", (" s9", "s9 ", "s9 fraction")),
    ("cytosol", ("cytosol",)),
    ("recombinant_enzyme", ("recombinant", "cyp", "ugt")),
    ("perfused_liver", ("perfused liver",)),
    ("tissue_homogenate", ("homogenate",)),
    ("plasma_or_blood", ("plasma", "whole blood", " blood")),
    ("dissolution_apparatus", ("dissolution",)),
    ("solubility_assay", ("solubility",)),
    ("in_silico", ("in silico", "pbpk")),
    ("in_vivo", ("in vivo", "clinical", "oral administration")),
    ("in_vitro_other", ("in vitro",)),
)

_ASSAY_PROTOCOL_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("bidirectional_transport", ("bidirectional",)),
    ("uptake", ("uptake",)),
    ("efflux", ("efflux", "secretory transport")),
    ("inhibition", ("inhibition", "inhibitor")),
    ("induction", ("induction", "inducer")),
    ("substrate_depletion", ("substrate depletion", "disappearance")),
    ("metabolite_formation", ("metabolite formation",)),
    ("half_life", ("half life", "half-life")),
    ("clearance", ("clearance",)),
    ("permeability", ("permeability", "papp")),
    ("transport", ("transport",)),
    ("absorption", ("absorption",)),
    ("dissolution", ("dissolution",)),
    ("equilibrium_solubility", ("equilibrium solubility",)),
    ("kinetic_solubility", ("kinetic solubility",)),
    ("intrinsic_solubility", ("intrinsic solubility",)),
    ("aqueous_solubility", ("aqueous solubility",)),
    ("solubility", ("solubility",)),
    ("stability", ("stability",)),
    ("metabolism", ("metabolism", "metabolic")),
    ("binding", ("binding",)),
    ("perfusion", ("perfusion",)),
)

_ASSAY_REVIEWED_PHRASES: dict[str, tuple[str, str]] = {
    "caco 2 cell monolayer": ("caco_2", "permeability"),
    "caco 2 monolayer transport assay": ("caco_2", "transport"),
    "caco 2 bidirectional transport": ("caco_2", "bidirectional_transport"),
    "mdck bidirectional transport": ("mdck", "bidirectional_transport"),
    "pampa permeability": ("pampa", "permeability"),
    "in situ intestinal perfusion": ("intestinal_perfusion", "perfusion"),
    "everted gut sac": ("everted_gut_sac", "absorption"),
    "ussing chamber": ("ussing_chamber", "transport"),
    "aqueous solubility measurement": ("solubility_assay", "aqueous_solubility"),
    "equilibrium solubility": ("solubility_assay", "equilibrium_solubility"),
    "in vitro dissolution": ("dissolution_apparatus", "dissolution"),
    "human liver microsomes": ("liver_microsomes", "metabolism"),
    "rat liver microsomes": ("liver_microsomes", "metabolism"),
    "human hepatocytes": ("hepatocytes", "metabolism"),
    "isolated perfused liver": ("perfused_liver", "perfusion"),
}

_SCIENTIFIC_IDENTIFIER = re.compile(
    r"\b(?:CYP\d+[A-Z]?\d*|UGT\d+[A-Z]?\d*|MDR1|P-?GP|BCRP|MRP\d*|"
    r"OATP\d*[A-Z]?\d*|OCT\d+|MATE\d+)\b",
    re.IGNORECASE,
)
_SITE_TERMS = ("duodenum", "jejunum", "ileum", "colon", "intestinal", "hepatic", "liver")
_DIRECTION_TERMS = (
    ("a_to_b", ("a to b", "a-b", "apical to basolateral")),
    ("b_to_a", ("b to a", "b-a", "basolateral to apical")),
    ("bidirectional", ("bidirectional",)),
)

_PERCENT_ENDPOINTS = {
    "absolute_bioavailability",
    "absorption",
    "bioavailability",
    "corrected_bioavailability",
    "dissolution",
    "dissolution_efficiency",
    "fraction_absorbed",
    "fraction_dissolved",
    "gastric_absorption",
    "human_intestinal_absorption",
    "intestinal_absorption",
    "oral_bioavailability",
    "relative_bioavailability",
}
_DURATION_ENDPOINTS = {"metabolic_half_life", "tmax"}


def canonical_text(value: Any) -> str:
    """Return a deterministic matching token without semantic inference."""
    if _is_null_like(value):
        return UNKNOWN_TOKEN
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("μ", "µ")
    return _SEPARATORS.sub("_", text).strip("_") or UNKNOWN_TOKEN


def canonicalize_bioavailability_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Create source-aware canonical context and factual validity fields."""
    output: dict[str, Any] = {}

    report = canonical_text(record.get("bioavailability_report_type"))
    output["canonical_bioavailability_report_type"] = _REPORT_TYPE_ALIASES.get(
        report, report
    )

    dose_value = record.get("oral_dose")
    if _is_null_like(dose_value):
        dose_value = record.get("dose")
    output.update(canonicalize_dose(dose_value))

    species_value = record.get("species")
    if _is_null_like(species_value):
        species_value = record.get("species_or_population")
    output.update(canonicalize_species(species_value))

    output.update(
        canonicalize_assay_system(
            record.get("assay_system"),
            source_id=str(record.get("source_id") or ""),
        )
    )

    enriched = {**dict(record), **output}
    output["normalization_validity_status"] = normalization_validity_status(enriched)
    output["context_canonicalization_version"] = CONTEXT_CANONICALIZATION_VERSION
    return output


def canonicalize_dose(value: Any) -> dict[str, Any]:
    empty = {
        "canonical_dose_value": None,
        "canonical_dose_unit": None,
        "canonical_dose_quantity_kind": UNKNOWN_TOKEN,
        "canonical_dose_basis": UNKNOWN_TOKEN,
        "canonical_dose_bin": UNKNOWN_TOKEN,
        "canonical_dose_regimen": UNKNOWN_TOKEN,
        "canonical_dose_key": UNKNOWN_TOKEN,
        "dose_mapping_status": "missing",
    }
    if _is_null_like(value):
        return empty
    raw = unicodedata.normalize("NFKC", str(value)).replace("μ", "µ")
    raw = raw.translate(str.maketrans({"−": "-", "–": "-", "—": "-", "‑": "-"}))
    match = _DOSE.match(raw.strip())
    regimen = _dose_regimen(raw)
    if match is None:
        fallback = f"{UNMAPPED_PREFIX}{canonical_text(raw)}"
        return {
            **empty,
            "canonical_dose_regimen": regimen,
            "canonical_dose_key": fallback,
            "dose_mapping_status": "unmapped_exact_fallback",
        }
    unit = match.group("unit").casefold().replace("u", "µ").replace("μ", "µ")
    amount = float(match.group("value"))
    if unit in _MASS_TO_MG:
        amount *= _MASS_TO_MG[unit]
        stable_unit = "mg"
        quantity_kind = "mass"
    elif unit in _MOLAR_TO_UMOL:
        amount *= _MOLAR_TO_UMOL[unit]
        stable_unit = "µmol"
        quantity_kind = "molar"
    else:
        fallback = f"{UNMAPPED_PREFIX}{canonical_text(raw)}"
        return {
            **empty,
            "canonical_dose_regimen": regimen,
            "canonical_dose_key": fallback,
            "dose_mapping_status": "unmapped_exact_fallback",
        }
    basis_text = canonical_text(match.group("basis"))
    if basis_text == UNKNOWN_TOKEN:
        basis = "absolute"
    elif "kg" in basis_text:
        basis = "per_kg"
    else:
        basis = "per_m2"
    if not math.isfinite(amount) or amount <= 0:
        fallback = f"{UNMAPPED_PREFIX}{canonical_text(raw)}"
        return {
            **empty,
            "canonical_dose_regimen": regimen,
            "canonical_dose_key": fallback,
            "dose_mapping_status": "nonpositive_exact_fallback",
        }
    magnitude_bin = f"log2:{math.floor(math.log2(amount))}"
    key = "|".join((quantity_kind, basis, stable_unit, magnitude_bin, regimen))
    return {
        "canonical_dose_value": float(amount),
        "canonical_dose_unit": stable_unit,
        "canonical_dose_quantity_kind": quantity_kind,
        "canonical_dose_basis": basis,
        "canonical_dose_bin": magnitude_bin,
        "canonical_dose_regimen": regimen,
        "canonical_dose_key": key,
        "dose_mapping_status": "parsed",
    }


def canonicalize_species(value: Any) -> dict[str, Any]:
    if _is_null_like(value):
        return {
            "canonical_species": UNKNOWN_TOKEN,
            "species_sex": None,
            "species_strain": None,
            "species_age": None,
            "species_model": None,
            "species_mapping_status": "missing",
        }
    raw = unicodedata.normalize("NFKC", str(value)).replace("‐", "-")
    lowered = raw.casefold()
    sex = _first_matching(lowered, ("male", "female", "mixed sex", "both sexes"))
    strain = _first_matching(
        lowered,
        ("sprague-dawley", "sprague dawley", "wistar", "balb/c", "c57bl/6", "beagle"),
    )
    age = _first_matching(lowered, ("adult", "juvenile", "neonatal", "aged", "young"))
    model = _first_matching(
        lowered, ("recombinant", "humanized", "knockout", "transgenic", "wild type")
    )
    base = re.sub(r"\([^)]*\)", " ", lowered)
    base = re.sub(
        r"\b(?:male|female|mixed sex|both sexes|sprague[- ]dawley|wistar|"
        r"balb/c|c57bl/6|adult|juvenile|neonatal|aged|young|recombinant|"
        r"humanized|knockout|transgenic|wild type|healthy|pooled)\b",
        " ",
        base,
    )
    parts = [
        part.strip()
        for part in re.split(r"\s*(?:,|;|\band\b|\+)\s*", base)
        if part.strip()
    ]
    taxa = sorted({_canonical_taxon(part) for part in parts if part})
    taxa = [taxon for taxon in taxa if taxon != UNKNOWN_TOKEN]
    canonical = "+".join(taxa) if taxa else f"{UNMAPPED_PREFIX}{canonical_text(base)}"
    return {
        "canonical_species": canonical,
        "species_sex": sex,
        "species_strain": strain,
        "species_age": age,
        "species_model": model,
        "species_mapping_status": (
            "reviewed_alias" if taxa and not canonical.startswith(UNMAPPED_PREFIX)
            else "unmapped_exact_fallback"
        ),
    }


def canonicalize_assay_system(value: Any, *, source_id: str) -> dict[str, Any]:
    if _is_null_like(value):
        return {
            "canonical_assay_platform": UNKNOWN_TOKEN,
            "canonical_assay_protocol": UNKNOWN_TOKEN,
            "canonical_assay_modifiers": UNKNOWN_TOKEN,
            "canonical_assay_system": UNKNOWN_TOKEN,
            "assay_system_mapping_status": "missing",
            "assay_system_fuzzy_score": None,
            "assay_system_fuzzy_margin": None,
            "assay_system_fuzzy_suggestion": None,
        }
    normalized = _phrase_text(value)
    platform = _first_rule(normalized, _ASSAY_PLATFORM_RULES)
    protocol = _first_rule(normalized, _ASSAY_PROTOCOL_RULES)
    status = "reviewed_rule"
    fuzzy_score: float | None = None
    fuzzy_margin: float | None = None
    suggestion: str | None = None
    if platform is None or protocol is None:
        ranked = sorted(
            (
                (_phrase_similarity(normalized, phrase), phrase, target)
                for phrase, target in _ASSAY_REVIEWED_PHRASES.items()
                if _protected_tokens_compatible(normalized, phrase)
            ),
            reverse=True,
        )
        if ranked:
            fuzzy_score, suggestion, target = ranked[0]
            second = ranked[1][0] if len(ranked) > 1 else 0.0
            fuzzy_margin = fuzzy_score - second
            if (
                fuzzy_score >= FUZZY_ACCEPT_THRESHOLD
                and fuzzy_margin >= FUZZY_WINNER_MARGIN
            ):
                platform = platform or target[0]
                protocol = protocol or target[1]
                status = "reviewed_fuzzy"
    modifiers = _assay_modifiers(normalized, source_id=source_id)
    if platform is None and protocol is None:
        fallback = f"{UNMAPPED_PREFIX}{canonical_text(value)}"
        return {
            "canonical_assay_platform": UNKNOWN_TOKEN,
            "canonical_assay_protocol": UNKNOWN_TOKEN,
            "canonical_assay_modifiers": "|".join(modifiers) if modifiers else UNKNOWN_TOKEN,
            "canonical_assay_system": fallback,
            "assay_system_mapping_status": "unmapped_exact_fallback",
            "assay_system_fuzzy_score": fuzzy_score,
            "assay_system_fuzzy_margin": fuzzy_margin,
            "assay_system_fuzzy_suggestion": suggestion,
        }
    platform = platform or "other"
    protocol = protocol or "other"
    modifier_key = "|".join(modifiers) if modifiers else UNKNOWN_TOKEN
    return {
        "canonical_assay_platform": platform,
        "canonical_assay_protocol": protocol,
        "canonical_assay_modifiers": modifier_key,
        "canonical_assay_system": "|".join((platform, protocol, modifier_key)),
        "assay_system_mapping_status": status,
        "assay_system_fuzzy_score": fuzzy_score,
        "assay_system_fuzzy_margin": fuzzy_margin,
        "assay_system_fuzzy_suggestion": suggestion,
    }


def _domain_kind(record: Mapping[str, Any]) -> str:
    endpoint = str(record.get("canonical_endpoint") or "").casefold()
    unit = str(record.get("canonical_unit") or "")
    source_id = str(record.get("source_id") or "")
    report_type = str(
        record.get("canonical_bioavailability_report_type") or UNKNOWN_TOKEN
    )
    if not endpoint or not unit:
        return "unsupported"
    if canonicalize_unit(unit).transform:
        return "transformed_scalar"
    if source_id == "direct_hf" or endpoint in {
        "absolute_bioavailability",
        "bioavailability",
        "corrected_bioavailability",
        "oral_bioavailability",
        "relative_bioavailability",
    }:
        if unit == "%" and not (
            report_type in {"relative_comparison", "apparent"}
            or endpoint.startswith("relative_")
        ):
            return "bounded_percentage"
        return "positive_scalar"
    if endpoint in _DURATION_ENDPOINTS or endpoint.endswith("_half_life"):
        return "positive_time" if unit == "h" else "positive_scalar"
    if "permeability" in endpoint and unit == "cm/s":
        return "permeability_cm_s"
    if unit == "fraction":
        return "bounded_fraction"
    if unit == "%" and endpoint in _PERCENT_ENDPOINTS:
        return "bounded_percentage"
    if (
        "ratio" in endpoint
        or unit.casefold() in {"ratio", "fold", "dimensionless", "dimensionless_ratio"}
    ):
        return "positive_ratio"
    return "positive_scalar"


def normalization_validity_status(record: Mapping[str, Any]) -> str:
    """Return factual record validity without assigning a labeling policy."""
    if (
        str(record.get("structure_status") or "") != "resolved"
        or not str(record.get("canonical_smiles") or "")
    ):
        return "unresolved_structure"
    if str(record.get("measurement_unit_status") or "") == (
        "ambiguous_scientific_notation"
    ):
        return "ambiguous_scientific_notation"
    if str(record.get("measurement_unit_status") or "") == "incompatible_endpoint_unit":
        return "incompatible_endpoint_unit"
    if not str(record.get("canonical_endpoint") or ""):
        return "missing_canonical_endpoint"
    if not str(record.get("canonical_unit") or ""):
        return "missing_canonical_unit"
    parsed = parse_point_measurement(record.get("canonical_measurement"))
    unit_result = canonicalize_unit(record.get("canonical_unit"))
    if parsed.value is not None and unit_result.unknown_tokens:
        return "incompatible_canonical_unit"
    value = record.get("finite_scalar_value")
    if isinstance(value, bool) or value is None:
        return "non_scalar_measurement"
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return "non_scalar_measurement"
    if not math.isfinite(scalar):
        return "non_scalar_measurement"
    domain_kind = _domain_kind(record)
    if domain_kind == "bounded_percentage" and not 0.0 <= scalar <= 100.0:
        return "outside_bounded_percentage_domain"
    if domain_kind == "bounded_fraction" and not 0.0 <= scalar <= 1.0:
        return "outside_bounded_fraction_domain"
    if domain_kind == "permeability_cm_s" and not 0.0 < scalar <= 1.0:
        return "outside_permeability_domain"
    if scalar <= 0.0 and domain_kind in {
        "positive_scalar",
        "permeability_cm_s",
        "positive_ratio",
        "positive_time",
    }:
        return {
            "positive_scalar": "nonpositive_positive_scalar",
            "positive_ratio": "nonpositive_dimensionless_ratio",
            "positive_time": "nonpositive_time",
        }.get(domain_kind, "outside_comparison_domain")
    variation = record.get("variation_value")
    if variation is not None and float(variation) < 0:
        return "negative_variation"
    return "valid"


def canonicalization_policy_manifest() -> dict[str, Any]:
    return {
        "context_canonicalization_version": CONTEXT_CANONICALIZATION_VERSION,
        "normalization_domain_rules_version": NORMALIZATION_DOMAIN_RULES_VERSION,
        "fuzzy_accept_threshold": FUZZY_ACCEPT_THRESHOLD,
        "fuzzy_winner_margin": FUZZY_WINNER_MARGIN,
        "normalization_domains": {
            "bounded_percentage": {"minimum": 0.0, "maximum": 100.0},
            "bounded_fraction": {"minimum": 0.0, "maximum": 1.0},
            "permeability_cm_s": {
                "minimum_exclusive": 0.0,
                "maximum": 1.0,
            },
            "positive_scalar": {"minimum_exclusive": 0.0},
            "positive_ratio": {"minimum_exclusive": 0.0},
            "positive_time": {"minimum_exclusive": 0.0},
            "transformed_scalar": {"finite": True},
        },
    }


def _dose_regimen(value: str) -> str:
    text = _phrase_text(value)
    if "loading" in text and "maintenance" in text:
        return "loading_and_maintenance"
    if "steady state" in text:
        return "steady_state"
    if re.search(r"\b(range|ranging|escalat|titration|titrated|variable)\w*\b", text):
        return "variable"

    single = bool(
        re.search(
            r"\b(?:single(?:[- ]+oral)?[- ]+dose|single[- ]+administration|"
            r"once[- ]+only|single)\b",
            text,
        )
    )
    repeated = bool(
        re.search(
            r"\b(?:repeated|multiple|multi[- ]+dose|once[- ]+daily|"
            r"twice[- ]+daily|daily|every[- ]+day)\b",
            text,
        )
        or re.search(
            r"(?<![a-z])(?:q\.?\s*d\.?|b\.?\s*i\.?\s*d\.?|"
            r"t\.?\s*i\.?\s*d\.?|q\.?\s*i\.?\s*d\.?)(?![a-z])",
            text,
        )
        or re.search(r"(?:/|\bper\s+)(?:d|day)\b", text)
    )
    if single and repeated:
        return "conflicting_single_and_repeated"
    if single:
        return "single"
    if repeated:
        return "repeated"
    return "unspecified"


def _canonical_taxon(value: str) -> str:
    normalized = canonical_text(value)
    if normalized in _SPECIES_ALIASES:
        return _SPECIES_ALIASES[normalized]
    for phrase in (
        "volunteer",
        "patient",
        "subject",
        "human",
        "rat",
        "mouse",
        "mice",
        "dog",
        "rabbit",
        "monkey",
        "pig",
    ):
        if re.search(rf"\b{re.escape(phrase)}s?\b", value):
            return _SPECIES_ALIASES.get(canonical_text(phrase), canonical_text(phrase))
    return f"{UNMAPPED_PREFIX}{normalized}" if normalized != UNKNOWN_TOKEN else UNKNOWN_TOKEN


def _first_rule(
    text: str,
    rules: tuple[tuple[str, tuple[str, ...]], ...],
) -> str | None:
    for canonical, phrases in rules:
        if any(phrase in text for phrase in phrases):
            return canonical
    return None


def _assay_modifiers(text: str, *, source_id: str) -> list[str]:
    modifiers: set[str] = set()
    for identifier in _SCIENTIFIC_IDENTIFIER.findall(text):
        modifiers.add(identifier.casefold().replace("-", "_"))
    for site in _SITE_TERMS:
        if site in text:
            modifiers.add(site)
    for canonical, phrases in _DIRECTION_TERMS:
        if any(phrase in text for phrase in phrases):
            modifiers.add(canonical)
    for state in ("in vitro", "in vivo", "in situ"):
        if state in text:
            modifiers.add(state.replace(" ", "_"))
    if source_id in {"fa", "fg"}:
        species = canonicalize_species(text)["canonical_species"]
        if (
            species != UNKNOWN_TOKEN
            and not str(species).startswith(UNMAPPED_PREFIX)
        ):
            modifiers.add(f"species:{species}")
    return sorted(modifiers)


def _phrase_similarity(left: str, right: str) -> float:
    left_tokens, right_tokens = set(left.split()), set(right.split())
    union = left_tokens | right_tokens
    token_score = len(left_tokens & right_tokens) / len(union) if union else 0.0
    character_score = SequenceMatcher(None, left, right).ratio()
    return 0.6 * token_score + 0.4 * character_score


def _protected_tokens_compatible(left: str, right: str) -> bool:
    return {
        token.casefold().replace("-", "_")
        for token in _SCIENTIFIC_IDENTIFIER.findall(left)
    } == {
        token.casefold().replace("-", "_")
        for token in _SCIENTIFIC_IDENTIFIER.findall(right)
    }


def _phrase_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("μ", "µ").replace("‐", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"[^a-z0-9µ%+./-]+", " ", text).strip()


def _first_matching(text: str, values: tuple[str, ...]) -> str | None:
    return next((value for value in values if value in text), None)


def _is_null_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().casefold() in _NULL_LIKE


__all__ = [
    "CONTEXT_CANONICALIZATION_VERSION",
    "FUZZY_ACCEPT_THRESHOLD",
    "FUZZY_WINNER_MARGIN",
    "UNKNOWN_TOKEN",
    "canonical_text",
    "canonicalization_policy_manifest",
    "canonicalize_assay_system",
    "canonicalize_bioavailability_record",
    "canonicalize_dose",
    "canonicalize_species",
    "normalization_validity_status",
    "NORMALIZATION_DOMAIN_RULES_VERSION",
]
