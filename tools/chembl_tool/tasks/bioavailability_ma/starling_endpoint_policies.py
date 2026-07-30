"""Standalone endpoint-specific assay-transfer policy assignment."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


ENDPOINT_POLICY_REGISTRY_VERSION = "bioavailability_endpoint_policies.v1"

POLICY_TEMPLATES: dict[str, dict[str, Any]] = {
    "bounded_percentage.absolute_pp.v1": {
        "distance": "absolute",
        "transfer_max": 10.0,
        "not_transfer_min": 30.0,
        "threshold_display": (
            "within 10 percentage points / at least 30 percentage points apart"
        ),
    },
    "bounded_fraction.absolute.v1": {
        "distance": "absolute",
        "transfer_max": 0.10,
        "not_transfer_min": 0.30,
        "threshold_display": "within 0.10 / at least 0.30 apart",
    },
    "positive_scalar.log10_fold.v1": {
        "distance": "absolute_log10_ratio",
        "transfer_max": math.log10(2.0),
        "not_transfer_min": math.log10(5.0),
        "threshold_display": "within 2-fold / at least 5-fold apart",
    },
    "permeability_cm_s.log10_fold.v1": {
        "distance": "absolute_log10_ratio",
        "transfer_max": math.log10(2.0),
        "not_transfer_min": math.log10(5.0),
        "threshold_display": "within 2-fold / at least 5-fold apart",
    },
    "dimensionless_ratio.log10_fold.v1": {
        "distance": "absolute_log10_ratio",
        "transfer_max": math.log10(1.5),
        "not_transfer_min": math.log10(3.0),
        "threshold_display": "within 1.5-fold / at least 3-fold apart",
    },
    "time_hours.absolute.v1": {
        "distance": "absolute",
        "transfer_max": 1.0,
        "not_transfer_min": 3.0,
        "threshold_display": "within 1 hour / at least 3 hours apart",
    },
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
_BIOAVAILABILITY_ENDPOINTS = {
    "absolute_bioavailability",
    "bioavailability",
    "corrected_bioavailability",
    "oral_bioavailability",
    "relative_bioavailability",
}
_DURATION_ENDPOINTS = {"metabolic_half_life", "tmax"}


def inherited_template_key(record: Mapping[str, Any]) -> str | None:
    """Reproduce the reviewed v4 assignment boundary outside normalization."""
    endpoint = str(record.get("canonical_endpoint") or "").casefold()
    unit = str(record.get("canonical_unit") or "")
    source_id = str(record.get("source_id") or "")
    report_type = str(
        record.get("canonical_bioavailability_report_type") or "__unknown__"
    )
    if not endpoint or not unit:
        return None
    if source_id == "direct_hf" or endpoint in _BIOAVAILABILITY_ENDPOINTS:
        if unit != "%":
            return None
        if report_type in {"relative_comparison", "apparent"} or endpoint.startswith(
            "relative_"
        ):
            return "positive_scalar.log10_fold.v1"
        return "bounded_percentage.absolute_pp.v1"
    if endpoint in _DURATION_ENDPOINTS or endpoint.endswith("_half_life"):
        return "time_hours.absolute.v1" if unit == "h" else None
    if "permeability" in endpoint and unit == "cm/s":
        return "permeability_cm_s.log10_fold.v1"
    if unit == "fraction":
        return "bounded_fraction.absolute.v1"
    if unit == "%" and endpoint in _PERCENT_ENDPOINTS:
        return "bounded_percentage.absolute_pp.v1"
    if (
        "ratio" in endpoint
        or unit.casefold() in {"ratio", "fold", "dimensionless", "dimensionless_ratio"}
    ):
        return "dimensionless_ratio.log10_fold.v1"
    return "positive_scalar.log10_fold.v1"


def endpoint_policy_key(endpoint: str, template_key: str) -> str:
    return f"{endpoint}.{template_key}"


def assign_endpoint_policies(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Assign one endpoint-specific policy or one precise non-assignment status."""
    assignments: list[dict[str, Any]] = []
    policies: dict[str, dict[str, Any]] = {}
    statuses: Counter[str] = Counter()
    seen: set[str] = set()

    for record in records:
        record_id = str(record.get("normalized_record_id") or "")
        if not record_id or record_id in seen:
            raise ValueError(
                "endpoint-policy input normalized_record_id values must be nonempty "
                "and unique"
            )
        seen.add(record_id)
        validity = str(record.get("normalization_validity_status") or "")
        endpoint = str(record.get("canonical_endpoint") or "")
        policy_key: str | None = None
        if validity != "valid":
            status = f"normalization_invalid:{validity or 'missing_status'}"
        else:
            template_key = inherited_template_key(record)
            if template_key is None:
                status = "unsupported_assay_transfer_semantics"
            else:
                policy_key = endpoint_policy_key(endpoint, template_key)
                status = "assigned"
                policies.setdefault(
                    policy_key,
                    {
                        "canonical_endpoint": endpoint,
                        "inherited_template_key": template_key,
                        **POLICY_TEMPLATES[template_key],
                    },
                )
        statuses[status] += 1
        assignments.append(
            {
                "normalized_record_id": record_id,
                "endpoint_policy_key": policy_key,
                "policy_assignment_status": status,
            }
        )

    registry = {
        "registry_version": ENDPOINT_POLICY_REGISTRY_VERSION,
        "policy_templates": POLICY_TEMPLATES,
        "endpoint_policies": dict(sorted(policies.items())),
        "scope": {
            "pair_enumeration": False,
            "pair_labels": False,
            "threshold_calibration": False,
            "thresholds_inherited_from_v4": True,
        },
    }
    audit = {
        "registry_version": ENDPOINT_POLICY_REGISTRY_VERSION,
        "stats": {
            "input_records": len(records),
            "assignment_records": len(assignments),
            "assigned_records": statuses["assigned"],
            "unassigned_records": len(assignments) - statuses["assigned"],
            "endpoint_policies": len(policies),
        },
        "assignment_status_counts": dict(sorted(statuses.items())),
        "validations": {
            "one_assignment_per_input_record": len(assignments) == len(records),
            "assigned_rows_have_policy": all(
                bool(row["endpoint_policy_key"])
                for row in assignments
                if row["policy_assignment_status"] == "assigned"
            ),
            "unassigned_rows_have_no_policy": all(
                not row["endpoint_policy_key"]
                for row in assignments
                if row["policy_assignment_status"] != "assigned"
            ),
            "all_assigned_policies_are_registered": all(
                not row["endpoint_policy_key"]
                or row["endpoint_policy_key"] in policies
                for row in assignments
            ),
        },
    }
    return assignments, registry, audit


__all__ = [
    "ENDPOINT_POLICY_REGISTRY_VERSION",
    "POLICY_TEMPLATES",
    "assign_endpoint_policies",
    "endpoint_policy_key",
    "inherited_template_key",
]
