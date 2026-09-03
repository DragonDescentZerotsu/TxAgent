"""Deterministic Stage 1 measurement routing for the four Ames sources."""

from __future__ import annotations

from collections.abc import Mapping

from data.processing.evidence_library.versions.v8.measurement_routing import (
    SourceRoutingRules,
)


SOURCE_IDS = (
    "mutagenicity_outcomes",
    "fixed_mutation",
    "premutagenic_damage",
    "mutagenicity_mechanism",
)


def source_routing_rules() -> dict[str, SourceRoutingRules]:
    return {
        "mutagenicity_outcomes": SourceRoutingRules(
            "mutagenicity_outcomes", unit_field=""
        ),
        **{
            source_id: SourceRoutingRules(
                source_id,
                unit_field="unit_text",
                require_positive_value=False,
            )
            for source_id in SOURCE_IDS[1:]
        },
    }


def canonical_endpoint_record(record: Mapping[str, object]) -> str:
    return str(record.get("endpoint_name") or "missing_endpoint")


__all__ = ["SOURCE_IDS", "canonical_endpoint_record", "source_routing_rules"]
