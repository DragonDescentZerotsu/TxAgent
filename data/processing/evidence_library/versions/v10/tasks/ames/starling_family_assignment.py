"""Non-ranked source-family assignments for Ames Stage 2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from data.processing.evidence_library.shared.v2.normalization.contracts import (
    FamilyAssignment,
)

FAMILY_ASSIGNMENT_VERSION = "ames_source_families.v1"
FAMILY_ASSIGNMENTS = {
    "mutagenicity_outcomes": FamilyAssignment(
        "Observed.mutagenicity_outcomes",
        "Observed",
        "mutagenicity_outcomes",
        "direct_outcome",
        "observed mutagenicity outcome",
    ),
    "fixed_mutation": FamilyAssignment(
        "Observed.fixed_mutation",
        "Observed",
        "fixed_mutation",
        "direct_outcome",
        "observed fixed mutation",
    ),
    "premutagenic_damage": FamilyAssignment(
        "Mechanism.premutagenic_damage",
        "Mechanism",
        "premutagenic_damage",
        "mechanistic_factor",
        "premutagenic damage",
    ),
    "mutagenicity_mechanism": FamilyAssignment(
        "Mechanism.mutagenicity_mechanism",
        "Mechanism",
        "mutagenicity_mechanism",
        "mechanistic_factor",
        "mutagenicity mechanism",
    ),
}


def family_assignment(
    source_id: str,
    endpoint_name: str,
    record: Mapping[str, Any] | None = None,
) -> FamilyAssignment | None:
    """Assign a source layer without implying an order between layers."""
    del endpoint_name, record
    return FAMILY_ASSIGNMENTS.get(source_id)


def family_assignment_manifest() -> dict[str, Any]:
    """Return the frozen family vocabulary used by Stage 2."""
    return {
        "version": FAMILY_ASSIGNMENT_VERSION,
        "semantics": "non_ranked_source_layer_partition",
        "assignments": {
            source_id: asdict(assignment)
            for source_id, assignment in sorted(FAMILY_ASSIGNMENTS.items())
        },
    }


__all__ = [
    "FAMILY_ASSIGNMENTS",
    "FAMILY_ASSIGNMENT_VERSION",
    "family_assignment",
    "family_assignment_manifest",
]
