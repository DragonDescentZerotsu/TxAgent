"""Deterministic, fail-closed scalar reference semantics for ClinTox."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tools.chembl_tool.common.starling.reference_semantics import (
    REFERENCE_BASIS_NONE,
    REFERENCE_BASIS_UNKNOWN,
    REFERENCE_SCOPE_NOT_APPLICABLE,
    REFERENCE_SCOPE_UNKNOWN,
    ReferenceAssignment,
)


REFERENCE_SEMANTICS_VERSION = "clintox_reference_semantics.v2"


def deterministic_reference_assignment(
    record: Mapping[str, Any],
) -> ReferenceAssignment:
    """Assign only semantics declared by the frozen measurement registry."""
    if record.get("categorical_encoder_id") or record.get(
        "canonical_measurement_scale_id"
    ):
        return ReferenceAssignment(
            REFERENCE_SCOPE_NOT_APPLICABLE,
            REFERENCE_BASIS_NONE,
            "controlled_measurement_scale",
        )
    if record.get("finite_scalar_value") is None:
        return ReferenceAssignment(
            REFERENCE_SCOPE_UNKNOWN,
            REFERENCE_BASIS_UNKNOWN,
            "non_scalar",
        )
    if str(record.get("canonical_semantics_status") or "") != "approved":
        return ReferenceAssignment(
            REFERENCE_SCOPE_UNKNOWN,
            REFERENCE_BASIS_UNKNOWN,
            "unreviewed_measurement_semantics",
        )
    scope = str(record.get("canonical_reference_scope_hint") or "unknown")
    basis = str(record.get("canonical_reference_basis_hint") or "unknown")
    try:
        return ReferenceAssignment(
            scope,
            basis,
            f"deterministic_measurement_rule:{record.get('canonical_semantics_rule_id')}",
        )
    except ValueError:
        return ReferenceAssignment(
            REFERENCE_SCOPE_UNKNOWN,
            REFERENCE_BASIS_UNKNOWN,
            "invalid_measurement_reference_hint",
        )


def reference_semantics_manifest(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    scope_counts: dict[str, int] = {}
    basis_counts: dict[str, int] = {}
    method_counts: dict[str, int] = {}
    for record in records:
        assignment = deterministic_reference_assignment(record)
        scope_counts[assignment.scope] = scope_counts.get(assignment.scope, 0) + 1
        basis = str(assignment.basis or "")
        basis_counts[basis] = basis_counts.get(basis, 0) + 1
        method_counts[assignment.method] = method_counts.get(assignment.method, 0) + 1
    return {
        "reference_semantics_version": REFERENCE_SEMANTICS_VERSION,
        "assignment_method": "deterministic reviewed measurement rules",
        "scope_counts": dict(sorted(scope_counts.items())),
        "basis_counts": dict(sorted(basis_counts.items())),
        "assignment_method_counts": dict(sorted(method_counts.items())),
        "mapping_path": None,
        "mapping_sha256": None,
        "runtime_llm_calls": False,
        "unresolved_policy": "unknown_and_assay_transfer_ineligible",
    }


__all__ = [
    "REFERENCE_SEMANTICS_VERSION",
    "deterministic_reference_assignment",
    "reference_semantics_manifest",
]
