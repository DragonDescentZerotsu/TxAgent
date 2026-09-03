"""Direct-report normalization and policy-independent factual validity."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from data.processing.evidence_library.shared.v1.categorical_response import (
    encoded_unit_validity_status,
)
from data.processing.evidence_library.shared.v1.normalization.measurements import (
    SOURCE_SPECIFIC_ATOMIC_SCALAR_STATUS,
    parse_point_measurement,
)
from tools.chembl_tool.common.units import canonicalize_unit
from data.processing.evidence_library.versions.v8.tasks.bioavailability_ma.canonical_source import (
    DIRECT_REPORT_TYPES,
)


REPORT_TYPE_NORMALIZATION_VERSION = "bioavailability_report_type_normalization.v1"
EVIDENCE_SCOPE_VERSION = "bioavailability_evidence_scope.v1"
NORMALIZATION_DOMAIN_RULES_VERSION = "bioavailability_normalization_domains.v4"
ORAL_DOSE_NORMALIZATION_VERSION = "bioavailability_oral_dose.v1"
UNKNOWN_TOKEN = "__unknown__"
DIRECT_EVIDENCE_SCOPE = "direct"
NONDIRECT_EVIDENCE_SCOPE = "nondirect"

_NULL_LIKE = {"", "nan", "none", "null", "na", "n/a", "-", "unspecified", "unknown"}
_SEPARATORS = re.compile(r"[\s_\-\u2010-\u2015\u2212]+")
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
_DOSE = re.compile(
    r"(?<![\w.])(?P<value>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)\s*"
    r"(?P<unit>nmol|[µu]mol|mmol|mol|ng|[µu]g|mcg|mg|g)\b"
    r"(?P<basis>\s*(?:/\s*(?:kg|m(?:2|²))|(?:kg|m(?:2|²))\s*(?:-1|−1|⁻¹)))?",
    re.IGNORECASE,
)
_DOSE_FACTORS = {
    "ng": ("mass", "mg", 1e-6),
    "µg": ("mass", "mg", 1e-3),
    "mcg": ("mass", "mg", 1e-3),
    "mg": ("mass", "mg", 1.0),
    "g": ("mass", "mg", 1e3),
    "nmol": ("molar", "µmol", 1e-3),
    "µmol": ("molar", "µmol", 1.0),
    "mmol": ("molar", "µmol", 1e3),
    "mol": ("molar", "µmol", 1e6),
}

# Qualifier vocabulary for every unit parsed by this task.
_TASK_VOCAB = "bioavailability_ma"



def _is_null_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().casefold() in _NULL_LIKE


def canonical_text(value: Any) -> str:
    """Return a deterministic token without scientific inference."""
    if _is_null_like(value):
        return UNKNOWN_TOKEN
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("μ", "µ")
    return _SEPARATORS.sub("_", text).strip("_") or UNKNOWN_TOKEN


def normalize_bioavailability_report_type(value: Any) -> str:
    if str(value or "").strip().casefold() == "unspecified":
        return "unspecified"
    token = canonical_text(value)
    return _REPORT_TYPE_ALIASES.get(token, token)


def bioavailability_evidence_scope(value: Any) -> str:
    """Classify one HF report without changing its physical source identity."""
    report_type = normalize_bioavailability_report_type(value)
    return (
        DIRECT_EVIDENCE_SCOPE
        if report_type in DIRECT_REPORT_TYPES
        else NONDIRECT_EVIDENCE_SCOPE
    )


def canonical_oral_dose(value: Any) -> dict[str, Any]:
    """Normalize one unambiguous oral dose without interpreting a regimen."""
    empty = {
        "canonical_oral_dose_value": None,
        "canonical_oral_dose_unit": None,
        "canonical_oral_dose_quantity_kind": None,
        "canonical_oral_dose_basis": None,
        "canonical_oral_dose_key": UNKNOWN_TOKEN,
        "canonical_oral_dose_mapping_status": "missing" if _is_null_like(value) else "unparsed",
        "oral_dose_normalization_version": ORAL_DOSE_NORMALIZATION_VERSION,
    }
    if _is_null_like(value):
        return empty
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("μ", "µ")
    matches = list(_DOSE.finditer(text))
    if len(matches) != 1:
        return empty
    match = matches[0]
    # A denominator other than kg or m2 is a concentration, not a dose.
    if not match.group("basis") and text[match.end() :].lstrip().startswith("/"):
        return empty
    number = float(match.group("value").replace(",", ""))
    unit = match.group("unit").casefold().replace("μ", "µ").replace("u", "µ")
    quantity_kind, canonical_unit, factor = _DOSE_FACTORS[unit]
    canonical_value = number * factor
    if not math.isfinite(canonical_value) or canonical_value <= 0:
        return {**empty, "canonical_oral_dose_mapping_status": "nonpositive"}
    basis_text = re.sub(r"\s+", "", match.group("basis") or "").replace("−", "-")
    basis = (
        "per_kg"
        if "kg" in basis_text
        else "per_m2"
        if "m2" in basis_text or "m²" in basis_text
        else "absolute"
    )
    number_text = format(canonical_value, ".12g")
    return {
        "canonical_oral_dose_value": canonical_value,
        "canonical_oral_dose_unit": canonical_unit,
        "canonical_oral_dose_quantity_kind": quantity_kind,
        "canonical_oral_dose_basis": basis,
        "canonical_oral_dose_key": "|".join(
            (quantity_kind, basis, canonical_unit, number_text)
        ),
        "canonical_oral_dose_mapping_status": "resolved",
        "oral_dose_normalization_version": ORAL_DOSE_NORMALIZATION_VERSION,
    }


def enrich_bioavailability_validity(record: Mapping[str, Any]) -> dict[str, Any]:
    report_type = normalize_bioavailability_report_type(
        record.get("bioavailability_report_type")
    )
    evidence_scope = (
        bioavailability_evidence_scope(report_type)
        if str(record.get("source_id") or "") == "hf_bioavailability"
        else None
    )
    enriched = {
        **dict(record),
        "canonical_bioavailability_report_type": report_type,
        "canonical_bioavailability_evidence_scope": evidence_scope,
    }
    return {
        "canonical_bioavailability_report_type": report_type,
        "canonical_bioavailability_evidence_scope": evidence_scope,
        "normalization_validity_status": normalization_validity_status(enriched),
        "report_type_normalization_version": REPORT_TYPE_NORMALIZATION_VERSION,
        "bioavailability_evidence_scope_version": EVIDENCE_SCOPE_VERSION,
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
    if canonicalize_unit(unit, task=_TASK_VOCAB).transform:
        return "transformed_scalar"
    if source_id == "hf_bioavailability" or endpoint in {
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
    if (
        source_id == "fg"
        and unit == "%"
        and str(record.get("measurement_unit_status") or "")
        == SOURCE_SPECIFIC_ATOMIC_SCALAR_STATUS
    ):
        return "bounded_percentage"
    if unit == "%" and endpoint in _PERCENT_ENDPOINTS:
        return "bounded_percentage"
    if (
        "ratio" in endpoint
        or unit.casefold() in {"ratio", "fold", "dimensionless", "dimensionless_ratio"}
    ):
        return "positive_ratio"
    return "positive_scalar"


def normalization_validity_status(record: Mapping[str, Any]) -> str:
    """Return factual record validity without assigning a transfer policy."""
    encoded = encoded_unit_validity_status(record)
    if encoded is not None:
        return encoded
    if (
        str(record.get("structure_status") or "") != "resolved"
        or not str(record.get("canonical_smiles") or "")
    ):
        return "unresolved_structure"
    if str(record.get("measurement_unit_status") or "") == "ambiguous_scientific_notation":
        return "ambiguous_scientific_notation"
    if str(record.get("measurement_unit_status") or "") == "incompatible_endpoint_unit":
        return "incompatible_endpoint_unit"
    if not str(record.get("canonical_endpoint") or ""):
        return "missing_canonical_endpoint"
    if not str(record.get("canonical_unit") or ""):
        return "missing_canonical_unit"
    parsed = parse_point_measurement(record.get("canonical_measurement"))
    unit_result = canonicalize_unit(record.get("canonical_unit"), task=_TASK_VOCAB)
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


def validity_policy_manifest() -> dict[str, Any]:
    return {
        "report_type_normalization_version": REPORT_TYPE_NORMALIZATION_VERSION,
        "report_type_aliases": dict(sorted(_REPORT_TYPE_ALIASES.items())),
        "normalization_domain_rules_version": NORMALIZATION_DOMAIN_RULES_VERSION,
        "normalization_domains": {
            "bounded_percentage": {"minimum": 0.0, "maximum": 100.0},
            "bounded_fraction": {"minimum": 0.0, "maximum": 1.0},
            "permeability_cm_s": {"minimum_exclusive": 0.0, "maximum": 1.0},
            "positive_scalar": {"minimum_exclusive": 0.0},
            "positive_ratio": {"minimum_exclusive": 0.0},
            "positive_time": {"minimum_exclusive": 0.0},
            "transformed_scalar": {"finite": True},
            "controlled_categorical": {
                "validation": "declared_encoder_and_finite_in-domain_anchor"
            },
        },
    }


__all__ = [
    "NORMALIZATION_DOMAIN_RULES_VERSION",
    "ORAL_DOSE_NORMALIZATION_VERSION",
    "REPORT_TYPE_NORMALIZATION_VERSION",
    "UNKNOWN_TOKEN",
    "canonical_text",
    "canonical_oral_dose",
    "enrich_bioavailability_validity",
    "normalization_validity_status",
    "normalize_bioavailability_report_type",
    "validity_policy_manifest",
]
