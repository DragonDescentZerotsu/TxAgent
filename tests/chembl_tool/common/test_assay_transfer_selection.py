from rdkit import DataStructs
import pytest

from tools.chembl_tool.common.assay_transfer_selection import (
    ASSAY_TRANSFER_DIVERSITY_ASSAY,
    ASSAY_TRANSFER_DIVERSITY_STRUCTURAL,
    select_assay_transfer_records,
)


def _fingerprint(*bits: int):
    fingerprint = DataStructs.ExplicitBitVect(16)
    for bit in bits:
        fingerprint.SetBit(bit)
    return fingerprint


def _record(molecule_id: str, score: float, endpoint: str, structural_rank: int = 1):
    return {
        "molecule_chembl_id": molecule_id,
        "transfer_selection_score": score,
        "transfer_winning_record_id": f"{molecule_id}-{endpoint}",
        "transfer_winning_record": {"canonical_endpoint_key": endpoint},
        "structural_rank": structural_rank,
    }


def test_zero_slack_preserves_current_score_order_verbatim():
    records = [
        _record("a", 0.90, "q2.endpoint.a"),
        _record("b", 0.80, "q2.endpoint.b"),
    ]
    selected, audit = select_assay_transfer_records(
        records,
        top_k=2,
        mode=ASSAY_TRANSFER_DIVERSITY_STRUCTURAL,
        score_slack=0.0,
        query_fingerprint=_fingerprint(0, 1, 2),
        fingerprints_by_molecule={"a": _fingerprint(0), "b": _fingerprint(1, 2)},
    )

    assert selected == records
    assert [row["selection_reason"] for row in audit["selection_trace"]] == [
        "score_ranking",
        "score_ranking",
    ]


def test_structural_mode_selects_query_feature_coverage_within_score_slack():
    records = [
        _record("high_score", 0.90, "q2.endpoint.a"),
        _record("better_coverage", 0.80, "q2.endpoint.b"),
    ]
    selected, audit = select_assay_transfer_records(
        records,
        top_k=1,
        mode=ASSAY_TRANSFER_DIVERSITY_STRUCTURAL,
        score_slack=0.10,
        query_fingerprint=_fingerprint(0, 1, 2),
        fingerprints_by_molecule={
            "high_score": _fingerprint(0),
            "better_coverage": _fingerprint(0, 1, 2),
        },
    )

    assert [record["molecule_chembl_id"] for record in selected] == ["better_coverage"]
    assert audit["selection_trace"][0]["score_loss"] == pytest.approx(0.1)
    assert audit["query_morgan_bit_coverage"] == 1.0


def test_assay_mode_prefers_an_unseen_canonical_endpoint_within_score_slack():
    records = [
        _record("a", 0.90, "q2.endpoint.a", 1),
        _record("b", 0.86, "q2.endpoint.a", 2),
        _record("c", 0.82, "q2.endpoint.b", 3),
    ]
    selected, audit = select_assay_transfer_records(
        records,
        top_k=2,
        mode=ASSAY_TRANSFER_DIVERSITY_ASSAY,
        score_slack=0.05,
        query_fingerprint=_fingerprint(0, 1),
        fingerprints_by_molecule={
            "a": _fingerprint(0),
            "b": _fingerprint(0),
            "c": _fingerprint(1),
        },
    )

    assert [record["molecule_chembl_id"] for record in selected] == ["a", "c"]
    assert audit["selection_trace"][1]["selection_reason"] == "new_canonical_endpoint_key"
    assert audit["n_distinct_selected_canonical_endpoint_keys"] == 2
