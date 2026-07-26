"""Pluggable set selection for structurally eligible molecule neighbors.

Candidate generation, minimum-similarity filtering, molecule-identity policy,
and evidence availability remain the responsibility of the retrieval caller.
This module only orders/selects the already eligible molecule indices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


SIMILARITY_SELECTOR = "similarity"
QUERY_FEATURE_COVERAGE_SELECTOR = "query_feature_coverage"
NEIGHBOR_SELECTORS = (
    SIMILARITY_SELECTOR,
    QUERY_FEATURE_COVERAGE_SELECTOR,
)
SELECTOR_VERSIONS = {
    SIMILARITY_SELECTOR: "similarity.v1",
    QUERY_FEATURE_COVERAGE_SELECTOR: "query_feature_coverage.v1",
}


@dataclass(frozen=True)
class NeighborCandidate:
    """Minimal selector-facing candidate contract."""

    molecule_index: int
    molecule_id: str
    similarity: float


def select_neighbor_candidates(
    candidates: Sequence[NeighborCandidate],
    *,
    query_fingerprint: Any,
    candidate_fingerprints: Sequence[Any],
    top_k: int,
    selector: str = SIMILARITY_SELECTOR,
) -> list[NeighborCandidate]:
    """Select up to ``top_k`` candidates without changing their payload schema."""
    _validate_selector(selector)
    if top_k <= 0 or not candidates:
        return []
    if selector == SIMILARITY_SELECTOR:
        return sorted(candidates, key=_similarity_sort_key)[:top_k]
    return _select_query_feature_coverage(
        candidates,
        query_fingerprint=query_fingerprint,
        candidate_fingerprints=candidate_fingerprints,
        top_k=top_k,
    )


def selector_metadata(selector: str) -> dict[str, Any]:
    """Return compact provenance for non-default selector outputs."""
    _validate_selector(selector)
    metadata: dict[str, Any] = {
        "name": selector,
        "version": SELECTOR_VERSIONS[selector],
    }
    if selector == QUERY_FEATURE_COVERAGE_SELECTOR:
        metadata.update(
            {
                "objective": "greedy_marginal_query_morgan_bit_coverage",
                "similarity_role": "eligibility_threshold_and_tie_break",
                "rank1_forced": False,
            }
        )
    return metadata


def _select_query_feature_coverage(
    candidates: Sequence[NeighborCandidate],
    *,
    query_fingerprint: Any,
    candidate_fingerprints: Sequence[Any],
    top_k: int,
) -> list[NeighborCandidate]:
    query_bits = frozenset(int(bit) for bit in query_fingerprint.GetOnBits())
    if not query_bits:
        return sorted(candidates, key=_similarity_sort_key)[:top_k]

    shared_query_bits: Mapping[int, frozenset[int]] = {
        candidate.molecule_index: frozenset(
            int(bit)
            for bit in candidate_fingerprints[candidate.molecule_index].GetOnBits()
            if int(bit) in query_bits
        )
        for candidate in candidates
    }
    remaining = list(candidates)
    selected: list[NeighborCandidate] = []
    covered: set[int] = set()
    while remaining and len(selected) < top_k:
        choice = min(
            remaining,
            key=lambda candidate: (
                -len(shared_query_bits[candidate.molecule_index] - covered),
                -candidate.similarity,
                candidate.molecule_id,
                candidate.molecule_index,
            ),
        )
        selected.append(choice)
        covered.update(shared_query_bits[choice.molecule_index])
        remaining.remove(choice)
    return selected


def _similarity_sort_key(candidate: NeighborCandidate) -> tuple[float, str, int]:
    return (-candidate.similarity, candidate.molecule_id, candidate.molecule_index)


def _validate_selector(selector: str) -> None:
    if selector not in NEIGHBOR_SELECTORS:
        raise ValueError(
            f"Unknown neighbor selector {selector!r}; expected one of {', '.join(NEIGHBOR_SELECTORS)}"
        )
