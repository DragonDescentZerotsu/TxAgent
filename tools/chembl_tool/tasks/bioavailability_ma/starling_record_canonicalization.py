"""Direct-report normalization and policy-independent factual validity."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from tools.chembl_tool.common.starling.normalization.measurements import (
    SOURCE_SPECIFIC_ATOMIC_SCALAR_STATUS,
    parse_point_measurement,
)
from tools.chembl_tool.common.units import canonicalize_unit


REPORT_TYPE_NORMALIZATION_VERSION = "bioavailability_report_type_normalization.v1"
NORMALIZATION_DOMAIN_RULES_VERSION = "bioavailability_normalization_domains.v3"
UNKNOWN_TOKEN = "__unknown__"

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
    token = canonical_text(value)
    return _REPORT_TYPE_ALIASES.get(token, token)


def enrich_bioavailability_validity(record: Mapping[str, Any]) -> dict[str, Any]:
    report_type = normalize_bioavailability_report_type(
        record.get("bioavailability_report_type")
    )
    enriched = {**dict(record), "canonical_bioavailability_report_type": report_type}
    return {
        "canonical_bioavailability_report_type": report_type,
        "normalization_validity_status": normalization_validity_status(enriched),
        "report_type_normalization_version": REPORT_TYPE_NORMALIZATION_VERSION,
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
        },
    }


__all__ = [
    "NORMALIZATION_DOMAIN_RULES_VERSION",
    "REPORT_TYPE_NORMALIZATION_VERSION",
    "UNKNOWN_TOKEN",
    "canonical_text",
    "enrich_bioavailability_validity",
    "normalization_validity_status",
    "normalize_bioavailability_report_type",
    "validity_policy_manifest",
]
