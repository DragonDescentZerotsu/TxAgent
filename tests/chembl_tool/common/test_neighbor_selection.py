import pytest
from rdkit import DataStructs

from tools.chembl_tool.common.neighbor_selection import (
    QUERY_FEATURE_COVERAGE_SELECTOR,
    SIMILARITY_SELECTOR,
    NeighborCandidate,
    select_neighbor_candidates,
    selector_metadata,
)


def _fingerprint(*bits: int):
    fingerprint = DataStructs.ExplicitBitVect(16)
    for bit in bits:
        fingerprint.SetBit(bit)
    return fingerprint


def _ids(selected):
    return [candidate.molecule_id for candidate in selected]


def test_similarity_selector_preserves_pointwise_similarity_ranking():
    candidates = [
        NeighborCandidate(0, "A", 0.80),
        NeighborCandidate(1, "B", 0.70),
        NeighborCandidate(2, "C", 0.30),
    ]
    fingerprints = [_fingerprint(0, 1), _fingerprint(0, 1), _fingerprint(2, 3)]

    selected = select_neighbor_candidates(
        candidates,
        query_fingerprint=_fingerprint(0, 1, 2, 3),
        candidate_fingerprints=fingerprints,
        top_k=2,
        selector=SIMILARITY_SELECTOR,
    )

    assert _ids(selected) == ["A", "B"]


def test_query_feature_coverage_selector_prefers_complementary_query_regions():
    candidates = [
        NeighborCandidate(0, "A", 0.80),
        NeighborCandidate(1, "B", 0.70),
        NeighborCandidate(2, "C", 0.30),
    ]
    fingerprints = [_fingerprint(0, 1), _fingerprint(0, 1), _fingerprint(2, 3)]

    selected = select_neighbor_candidates(
        candidates,
        query_fingerprint=_fingerprint(0, 1, 2, 3),
        candidate_fingerprints=fingerprints,
        top_k=2,
        selector=QUERY_FEATURE_COVERAGE_SELECTOR,
    )

    assert _ids(selected) == ["A", "C"]


def test_query_feature_coverage_uses_similarity_only_as_deterministic_tie_break():
    candidates = [
        NeighborCandidate(0, "lower", 0.40),
        NeighborCandidate(1, "higher", 0.60),
    ]
    fingerprints = [_fingerprint(0, 1), _fingerprint(0, 1)]

    selected = select_neighbor_candidates(
        candidates,
        query_fingerprint=_fingerprint(0, 1),
        candidate_fingerprints=fingerprints,
        top_k=1,
        selector=QUERY_FEATURE_COVERAGE_SELECTOR,
    )

    assert _ids(selected) == ["higher"]


def test_query_feature_coverage_metadata_freezes_current_contract():
    assert selector_metadata(QUERY_FEATURE_COVERAGE_SELECTOR) == {
        "name": "query_feature_coverage",
        "version": "query_feature_coverage.v1",
        "objective": "greedy_marginal_query_morgan_bit_coverage",
        "similarity_role": "eligibility_threshold_and_tie_break",
        "rank1_forced": False,
    }


def test_unknown_selector_is_rejected():
    with pytest.raises(ValueError, match="Unknown neighbor selector"):
        select_neighbor_candidates(
            [],
            query_fingerprint=_fingerprint(0),
            candidate_fingerprints=[],
            top_k=1,
            selector="unknown",
        )
