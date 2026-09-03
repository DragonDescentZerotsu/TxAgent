"""Conservative endpoint orthography and family assignment for BBB v6."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any, Iterable

from data.processing.evidence_library.shared.v1.endpoint_concepts import (
    load_endpoint_concept_maps,
)
from data.processing.evidence_library.shared.v1.normalization.cleaning import (
    clean_text,
    endpoint_inventory_hash,
)
from data.processing.evidence_library.shared.v1.normalization.contracts import FamilyAssignment
from data.processing.evidence_library.shared.v1.normalization.measurements import EndpointOrthography
from data.processing.evidence_library.versions.v8.tasks.bbb_martins.starling_endpoint_normalization import (
    ENDPOINT_NORMALIZATION_VERSION,
    EndpointNormalizer,
)


SPACING_AND_SPELLING_VERSION = ENDPOINT_NORMALIZATION_VERSION
ENDPOINT_CONCEPT_VERSION = "bbb_martins_endpoint_concepts.v2"

EXPECTED_ENDPOINT_INVENTORIES = {
    "direct_bbb": {
        "count": 22_744,
        "sha256": "2441c887b9b0c2031e0de4ab54663ca23760e97cf1f095365fe3872d375ace8e",
    },
    "passive_permeability": {
        "count": 252,
        "sha256": "0c22aa6427535299918752c4fd5d3357db671fd2fc6eab2b9ee5c87c94b4cc6b",
    },
    "efflux_transport": {
        "count": 23,
        "sha256": "90cd41e13cfb96e32e27822a2b65ecc65147929c1e1c4883d52a522328c657d0",
    },
    "influx_transport": {
        "count": 18,
        "sha256": "8db7a82f95e3c1da2dc1d1841994879f114f637742a2d40d67c740c61875cfeb",
    },
}
ENDPOINT_CONCEPT_PATHS = tuple(
    Path(__file__).resolve().parent
    / "data_processing/canonicalization_v7/endpoint_concepts"
    / f"{source_id}.json"
    for source_id in sorted(EXPECTED_ENDPOINT_INVENTORIES)
)

_DEFAULT_ENDPOINT_NORMALIZER = EndpointNormalizer()


def spacing_and_spelling_decision(
    source_id: str, endpoint_name: str
) -> EndpointOrthography:
    return _DEFAULT_ENDPOINT_NORMALIZER.decision(source_id, endpoint_name)


@cache
def _endpoint_concept_maps() -> dict[str, dict[tuple[str, str], str]]:
    return load_endpoint_concept_maps(
        ENDPOINT_CONCEPT_PATHS,
        version=ENDPOINT_CONCEPT_VERSION,
        expected_inventories=EXPECTED_ENDPOINT_INVENTORIES,
    )


def endpoint_concept(
    source_id: str, endpoint_name: str, canonical_endpoint_name: str
) -> str:
    key = (clean_text(endpoint_name) or "", str(canonical_endpoint_name or ""))
    try:
        return _endpoint_concept_maps()[source_id][key]
    except KeyError as error:
        raise ValueError(
            f"unreviewed endpoint concept: {source_id}/{key[0]}/{key[1]}"
        ) from error


def family_assignment(
    source_id: str,
    endpoint_name: str,
    record: Mapping[str, Any] | None = None,
) -> FamilyAssignment | None:
    del endpoint_name, record
    assignments = {
        "direct_bbb": FamilyAssignment(
            "Tier 1.starling_direct_bbb_evidence",
            "Tier 1",
            "starling_direct_bbb_evidence",
            "direct_outcome",
            "direct brain exposure and BBB permeability",
        ),
        "passive_permeability": FamilyAssignment(
            "Mechanism.passive_permeability",
            "Tier 2",
            "passive_permeability",
            "mechanistic_factor",
            "passive BBB permeability",
        ),
        "efflux_transport": FamilyAssignment(
            "Mechanism.efflux_transport",
            "Tier 3",
            "efflux_transport",
            "mechanistic_factor",
            "BBB efflux transport",
        ),
        "influx_transport": FamilyAssignment(
            "Mechanism.influx_transport",
            "Tier 4",
            "influx_transport",
            "mechanistic_factor",
            "BBB carrier-mediated influx transport",
        ),
    }
    return assignments.get(source_id)


def validate_endpoint_inventory(source_id: str, endpoint_names: Iterable[str]) -> dict:
    values = sorted(set(endpoint_names))
    actual = {"count": len(values), "sha256": endpoint_inventory_hash(values)}
    expected = EXPECTED_ENDPOINT_INVENTORIES.get(source_id)
    if expected is None:
        raise ValueError(f"no frozen endpoint inventory for source {source_id!r}")
    if actual != expected:
        raise ValueError(
            f"endpoint inventory drift for {source_id}: expected {expected}, found {actual}"
        )
    decisions = [spacing_and_spelling_decision(source_id, value) for value in values]
    return {
        "source_id": source_id,
        **actual,
        "n_reviewed_corrections": sum(
            decision.spacing_and_spelling_endpoint != decision.endpoint_name
            for decision in decisions
        ),
        "coverage": 1.0,
        "endpoints": [decision.to_dict() for decision in decisions],
    }


__all__ = [
    "ENDPOINT_CONCEPT_PATHS",
    "ENDPOINT_CONCEPT_VERSION",
    "EXPECTED_ENDPOINT_INVENTORIES",
    "SPACING_AND_SPELLING_VERSION",
    "endpoint_concept",
    "family_assignment",
    "spacing_and_spelling_decision",
    "validate_endpoint_inventory",
]
