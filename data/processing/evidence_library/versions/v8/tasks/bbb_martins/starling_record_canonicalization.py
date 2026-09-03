"""Policy-independent factual validity for BBB Martins normalized records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from data.processing.evidence_library.shared.v1.categorical_response import (
    encoded_unit_validity_status,
)
from data.processing.evidence_library.shared.v1.normalization.measurements import (
    parse_point_measurement,
)
from tools.chembl_tool.common.units import canonicalize_unit
from data.processing.evidence_library.versions.v8.tasks.bbb_martins.starling_measurement_semantics import (
    load_measurement_semantics_policy,
    numeric_domain_status,
    resolve_measurement_semantics,
    unit_is_compatible,
)


NORMALIZATION_DOMAIN_RULES_VERSION = "bbb_martins_normalization_domains.v3"

# Qualifier vocabulary for every unit parsed by this task.
_TASK_VOCAB = "bbb_martins"



def enrich_bbb_validity(record: Mapping[str, Any]) -> dict[str, Any]:
    semantics = resolve_measurement_semantics(record)
    return {
        "normalization_validity_status": normalization_validity_status(record),
        "normalization_domain_rules_version": NORMALIZATION_DOMAIN_RULES_VERSION,
        **semantics.fields(),
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
    semantics = resolve_measurement_semantics(record, endpoint)
    if semantics.status != "approved":
        return "unreviewed_endpoint_semantics"
    if not unit:
        return "missing_canonical_unit"
    parsed = parse_point_measurement(record.get("canonical_measurement"))
    unit_result = canonicalize_unit(unit, task=_TASK_VOCAB)
    if parsed.value is not None and unit_result.unknown_tokens:
        return "incompatible_canonical_unit"
    if unit_is_compatible(semantics, unit) is not True:
        return "incompatible_endpoint_unit"
    value = record.get("finite_scalar_value")
    if isinstance(value, bool) or value is None:
        return "non_scalar_measurement"
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return "non_scalar_measurement"
    if not math.isfinite(scalar):
        return "non_scalar_measurement"
    domain_error = numeric_domain_status(semantics, scalar)
    if domain_error is not None:
        return domain_error
    variation = record.get("variation_value")
    if variation is not None and float(variation) < 0.0:
        return "negative_variation"
    return "valid"


def validity_policy_manifest() -> dict[str, Any]:
    semantics = load_measurement_semantics_policy()
    return {
        "normalization_domain_rules_version": NORMALIZATION_DOMAIN_RULES_VERSION,
        "measurement_semantics": semantics.manifest(),
        "domains": {
            "bounded_0_100": {"minimum": 0.0, "maximum": 100.0},
            "bounded_0_1": {"minimum": 0.0, "maximum": 1.0},
            "finite_signed": {"finite": True},
            "nonnegative": {"minimum": 0.0},
        },
        "unreviewed_endpoint_policy": "evidence_only",
        "non_scalar_records_retained_as_evidence": True,
    }


__all__ = [
    "NORMALIZATION_DOMAIN_RULES_VERSION",
    "enrich_bbb_validity",
    "normalization_validity_status",
    "validity_policy_manifest",
]
