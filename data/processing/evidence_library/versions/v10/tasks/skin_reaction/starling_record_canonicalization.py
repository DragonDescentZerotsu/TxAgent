"""Policy-independent factual validity for normalized Skin_Reaction records.

Skin_Reaction has no direct-report source and therefore no report-type analogue:
the only work this layer does is decide whether a normalized scalar lies inside
the factual domain its endpoint and unit imply.  Records without a scalar are not
invalid — the two qualitative sources legitimately carry a null scalar — they are
simply reported as ``non_scalar_measurement``.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from data.processing.evidence_library.shared.v2.categorical_response import (
    ENCODED_UNITS,
    encoded_unit_validity_status,
)
from data.processing.evidence_library.shared.v2.normalization.measurements import (
    parse_point_measurement,
)
from tools.chembl_tool.common.units import canonicalize_unit


NORMALIZATION_DOMAIN_RULES_VERSION = "skin_reaction_normalization_domains.v1"
UNKNOWN_TOKEN = "__unknown__"

_NULL_LIKE = {"", "nan", "none", "null", "na", "n/a", "-", "unspecified", "unknown"}
_SEPARATORS = re.compile(r"[\s_\-‐-―−]+")

_TIME_ENDPOINTS = {"exposure_time", "lag_time"}
_TIME_DIMENSION = (("time", 1),)
_FOLD_UNITS = {"fold", "ratio", "dimensionless", "dimensionless_ratio", "si", "x"}
# Comparisons against a control legitimately exceed 100 % and are therefore only
# non-negative, never bounded.
_RELATIVE_PREFIXES = ("relative_",)
_RELATIVE_MARKERS = ("_of_control", "_vs_control", "_over_control")

# Qualifier vocabulary for every unit parsed by this task.
_TASK_VOCAB = "skin_reaction"



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


def enrich_skin_reaction_validity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "normalization_validity_status": normalization_validity_status(record),
        "normalization_domain_rules_version": NORMALIZATION_DOMAIN_RULES_VERSION,
    }


def _domain_kind(record: Mapping[str, Any]) -> str:
    """Classify one record's factual domain from its endpoint and canonical unit.

    Skin quantities are amounts, areic amounts, rates, and permeation coefficients,
    for which an exact zero is a genuine observation ("no permeation detected").
    The default domain is therefore non-negative rather than strictly positive;
    only potency (EC3) and fold-change ratios are strictly positive.
    """
    endpoint = str(record.get("canonical_endpoint") or "").casefold()
    unit = str(record.get("canonical_unit") or "")
    if not endpoint or not unit:
        return "unsupported"
    unit_result = canonicalize_unit(unit, task=_TASK_VOCAB)
    if unit_result.transform:
        return "transformed_scalar"
    relative = endpoint.startswith(_RELATIVE_PREFIXES) or any(
        marker in endpoint for marker in _RELATIVE_MARKERS
    )
    if unit == "%":
        if "ec3" in endpoint and not endpoint.startswith("p"):
            return "positive_percentage"
        return "non_negative_scalar" if relative else "bounded_percentage"
    if unit.casefold() in _FOLD_UNITS or "stimulation_index" in endpoint:
        return "positive_fold"
    if endpoint in _TIME_ENDPOINTS or unit_result.dimension == _TIME_DIMENSION:
        return "non_negative_time"
    return "non_negative_scalar"


def normalization_validity_status(record: Mapping[str, Any]) -> str:
    """Return factual record validity without assigning a transfer policy."""
    # A categorically encoded record carries a latent-scale unit rather than a
    # physical one, so it is validated against the encoding contract instead of
    # the physical domains below.
    encoded_status = encoded_unit_validity_status(record)
    if encoded_status is not None:
        return encoded_status
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
    reviewed_domain = str(record.get("measurement_numeric_domain") or "")
    if reviewed_domain:
        if reviewed_domain == "finite_signed":
            pass
        elif reviewed_domain in {"nonnegative", "nonnegative_unbounded"}:
            if scalar < 0.0:
                return "outside_reviewed_numeric_domain"
        elif reviewed_domain == "positive":
            if scalar <= 0.0:
                return "outside_reviewed_numeric_domain"
        elif reviewed_domain == "bounded_0_1":
            if not 0.0 <= scalar <= 1.0:
                return "outside_reviewed_numeric_domain"
        elif reviewed_domain == "bounded_0_100":
            if not 0.0 <= scalar <= 100.0:
                return "outside_reviewed_numeric_domain"
        else:
            return "unsupported_reviewed_numeric_domain"
        variation = record.get("variation_value")
        if variation is not None and float(variation) < 0:
            return "negative_variation"
        return "valid"
    domain_kind = _domain_kind(record)
    if domain_kind in {"bounded_percentage", "positive_percentage"} and not (
        0.0 <= scalar <= 100.0
    ):
        return "outside_bounded_percentage_domain"
    if scalar < 0.0 and domain_kind == "non_negative_time":
        return "nonpositive_time"
    if scalar < 0.0 and domain_kind == "non_negative_scalar":
        return "nonpositive_positive_scalar"
    if scalar <= 0.0 and domain_kind in {"positive_percentage", "positive_fold"}:
        return "nonpositive_positive_scalar"
    variation = record.get("variation_value")
    if variation is not None and float(variation) < 0:
        return "negative_variation"
    return "valid"


def validity_policy_manifest() -> dict[str, Any]:
    return {
        "normalization_domain_rules_version": NORMALIZATION_DOMAIN_RULES_VERSION,
        "normalization_domains": {
            "bounded_percentage": {"minimum": 0.0, "maximum": 100.0},
            "positive_percentage": {"minimum_exclusive": 0.0, "maximum": 100.0},
            "non_negative_scalar": {"minimum": 0.0},
            "non_negative_time": {"minimum": 0.0},
            "positive_fold": {"minimum_exclusive": 0.0},
            "transformed_scalar": {"finite": True},
        },
        "domain_selectors": {
            "positive_percentage": "EC3-style potency reported in %",
            "bounded_percentage": "any other % measurement that is not control-relative",
            "non_negative_scalar": "control-relative % plus every areic amount, flux, "
            "permeability coefficient, mass, and concentration",
            "non_negative_time": sorted(_TIME_ENDPOINTS) + ["any unit of pure time"],
            "positive_fold": sorted(_FOLD_UNITS) + ["stimulation_index endpoints"],
        },
    }


__all__ = [
    "NORMALIZATION_DOMAIN_RULES_VERSION",
    "UNKNOWN_TOKEN",
    "canonical_text",
    "enrich_skin_reaction_validity",
    "normalization_validity_status",
    "validity_policy_manifest",
]
