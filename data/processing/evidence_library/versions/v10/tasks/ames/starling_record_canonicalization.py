"""Policy-independent factual validity for Ames Stage 2 records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from data.processing.evidence_library.shared.v2.categorical_response import (
    encoded_unit_validity_status,
)

NORMALIZATION_DOMAIN_RULES_VERSION = "ames_normalization_domains.v1"
_EXCLUDED_ENDPOINT_STATUSES = frozenset({"exclude", "excluded", "missing"})
_NONSCALAR_RESOLUTION_STATUSES = frozenset({"unavailable", "not_extracted"})


def _text(record: Mapping[str, Any], field: str) -> str:
    return str(record.get(field) or "").strip()


def _endpoint_is_missing(record: Mapping[str, Any]) -> bool:
    endpoint = _text(record, "canonical_endpoint")
    status = _text(record, "canonical_endpoint_mapping_status").casefold()
    return endpoint in {"", "missing_endpoint"} or status in _EXCLUDED_ENDPOINT_STATUSES


def _finite_scalar(record: Mapping[str, Any]) -> float | None:
    value = record.get("finite_scalar_value")
    if isinstance(value, bool) or value is None:
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if math.isfinite(scalar) else None


def _domain_status(record: Mapping[str, Any], scalar: float) -> str | None:
    if _text(record, "measurement_numeric_domain_status") == "outside_declared_domain":
        return "outside_reviewed_numeric_domain"
    domain = _text(record, "measurement_numeric_domain") or "any"
    if domain == "positive" and scalar <= 0:
        return "outside_reviewed_numeric_domain"
    if domain == "nonnegative" and scalar < 0:
        return "outside_reviewed_numeric_domain"
    if domain not in {"any", "positive", "nonnegative"}:
        return "unsupported_reviewed_numeric_domain"
    return None


def normalization_validity_status(record: Mapping[str, Any]) -> str:
    """Return factual validity without assigning transfer or ranking policy."""
    encoded = encoded_unit_validity_status(record)
    if encoded is not None:
        return encoded
    if _text(record, "structure_status") != "resolved" or not _text(
        record, "canonical_smiles"
    ):
        return "unresolved_structure"
    if _endpoint_is_missing(record):
        return "missing_canonical_endpoint"
    if _text(record, "measurement_unit_mapping_status") == "excluded":
        return "exact_measurement_excluded"
    resolution = _text(record, "measurement_resolution_status")
    if resolution == "relative":
        return "relative_measurement"
    if resolution == "unsure":
        return "measurement_resolution_unsure"
    if resolution in _NONSCALAR_RESOLUTION_STATUSES:
        return "non_scalar_measurement"
    if not _text(record, "canonical_unit"):
        return "missing_canonical_unit"
    scalar = _finite_scalar(record)
    if scalar is None:
        return "non_scalar_measurement"
    domain_status = _domain_status(record, scalar)
    if domain_status is not None:
        return domain_status
    variation = record.get("variation_value")
    if variation is not None and float(variation) < 0:
        return "negative_variation"
    return "valid"


def enrich_ames_validity(record: Mapping[str, Any]) -> dict[str, Any]:
    """Attach factual validity and the rule-set identity."""
    return {
        "normalization_validity_status": normalization_validity_status(record),
        "normalization_domain_rules_version": NORMALIZATION_DOMAIN_RULES_VERSION,
    }


def validity_policy_manifest() -> dict[str, Any]:
    """Describe the intentionally narrow validity contract."""
    return {
        "normalization_domain_rules_version": NORMALIZATION_DOMAIN_RULES_VERSION,
        "numeric_domains": ["any", "nonnegative", "positive"],
        "unreviewed_unit_policy": "error",
        "non_scalar_records_retained_as_evidence": True,
        "transfer_semantics": None,
    }


__all__ = [
    "NORMALIZATION_DOMAIN_RULES_VERSION",
    "enrich_ames_validity",
    "normalization_validity_status",
    "validity_policy_manifest",
]
