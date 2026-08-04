"""Policy-independent factual validity for BBB Martins normalized records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from tools.chembl_tool.common.starling.categorical_response import (
    encoded_unit_validity_status,
)
from tools.chembl_tool.common.starling.normalization.measurements import (
    parse_point_measurement,
)
from tools.chembl_tool.common.units import canonicalize_unit


NORMALIZATION_DOMAIN_RULES_VERSION = "bbb_martins_normalization_domains.v2"


def enrich_bbb_validity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "normalization_validity_status": normalization_validity_status(record),
        "normalization_domain_rules_version": NORMALIZATION_DOMAIN_RULES_VERSION,
    }


def normalization_validity_status(record: Mapping[str, Any]) -> str:
    encoded = encoded_unit_validity_status(record)
    if encoded is not None:
        return encoded
    if (
        str(record.get("structure_status") or "") != "resolved"
        or not str(record.get("canonical_smiles") or "")
    ):
        return "unresolved_structure"
    status = str(record.get("measurement_unit_status") or "")
    if status == "ambiguous_scientific_notation":
        return status
    if status == "incompatible_endpoint_unit":
        return status
    endpoint = str(record.get("canonical_endpoint") or "")
    unit = str(record.get("canonical_unit") or "")
    if endpoint in {"", "missing_endpoint"}:
        return "missing_canonical_endpoint"
    if not unit:
        return "missing_canonical_unit"
    parsed = parse_point_measurement(record.get("canonical_measurement"))
    unit_result = canonicalize_unit(unit)
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
    if unit == "%" and not 0.0 <= scalar <= 100.0:
        return "outside_bounded_percentage_domain"
    signed = (
        endpoint in {"logbb", "log_bb"}
        or endpoint.startswith("log_")
        or "change" in endpoint
        or "delta" in endpoint
        or bool(unit_result.transform)
    )
    if scalar < 0.0 and not signed:
        return "nonpositive_positive_scalar"
    variation = record.get("variation_value")
    if variation is not None and float(variation) < 0.0:
        return "negative_variation"
    return "valid"


def validity_policy_manifest() -> dict[str, Any]:
    return {
        "normalization_domain_rules_version": NORMALIZATION_DOMAIN_RULES_VERSION,
        "domains": {
            "percentage": {"minimum": 0.0, "maximum": 100.0},
            "log_or_change": {"finite": True},
            "other_scalar": {"minimum": 0.0},
        },
        "non_scalar_records_retained_as_evidence": True,
    }


__all__ = [
    "NORMALIZATION_DOMAIN_RULES_VERSION",
    "enrich_bbb_validity",
    "normalization_validity_status",
    "validity_policy_manifest",
]
