from rdkit import DataStructs
import pytest

from tools.chembl_tool.common.assay_transfer_selection import (
    ASSAY_TRANSFER_DIVERSITY_ASSAY,
    ASSAY_TRANSFER_DIVERSITY_STRUCTURAL,
    ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    collapse_assay_transfer_records_by_molecule,
    select_assay_transfer_records,
    validate_assay_transfer_records_per_molecule,
)


def test_unique_molecule_collapse_keeps_each_molecules_best_record():
    records = [
        _record("a", 0.99, "one", 1),
        _record("a", 0.98, "two", 1),
        _record("a", 0.97, "three", 1),
        _record("b", 0.80, "four", 2),
        _record("c", 0.70, "five", 3),
    ]
    selected, audit = collapse_assay_transfer_records_by_molecule(records)

    assert [row["molecule_chembl_id"] for row in selected] == ["a", "b", "c"]
    assert [row["transfer_winning_record_id"] for row in selected] == [
        "a-one", "b-four", "c-five"
    ]
    assert [row["transfer_scored_record_count"] for row in selected] == [3, 1, 1]
    assert audit["n_valid_records_before_collapse"] == 5
    assert audit["n_unique_molecules_after_collapse"] == 3


def test_unique_molecule_collapse_keeps_score_ordered_endpoint_distinct_records():
    records = [
        _record("a", 0.99, "endpoint_one"),
        _record("a", 0.98, "endpoint_one"),
        _record("a", 0.97, "endpoint_two"),
        _record("a", 0.96, "endpoint_three"),
        _record("b", 0.80, "endpoint_four"),
    ]

    selected, audit = collapse_assay_transfer_records_by_molecule(
        records, records_per_molecule=3
    )

    assert [row["molecule_chembl_id"] for row in selected] == ["a", "b"]
    assert [
        row["canonical_endpoint_key"]
        for row in selected[0]["transfer_selected_records"]
    ] == ["endpoint_one", "endpoint_two", "endpoint_three"]
    assert [
        row["transfer_selection_score"]
        for row in selected[0]["transfer_selected_records"]
    ] == [0.99, 0.97, 0.96]
    assert selected[0]["transfer_winning_record_id"] == "a-endpoint_one"
    assert selected[0]["transfer_duplicate_endpoint_records_skipped"] == 1
    assert selected[1]["transfer_selected_record_count"] == 1
    assert selected[1]["transfer_records_underfilled"] is True
    assert audit["n_duplicate_endpoint_records_skipped"] == 1
    assert audit["n_underfilled_molecules"] == 1


def test_records_per_molecule_never_backfills_duplicate_endpoints():
    selected, _ = collapse_assay_transfer_records_by_molecule(
        [
            _record("a", 0.99, "same"),
            _record("a", 0.98, "same"),
            _record("a", 0.97, "same"),
        ],
        records_per_molecule=3,
    )

    assert len(selected[0]["transfer_selected_records"]) == 1
    assert selected[0]["transfer_records_underfilled"] is True


@pytest.mark.parametrize("value", [0, 11])
def test_records_per_molecule_rejects_values_outside_one_to_ten(value):
    with pytest.raises(ValueError, match="between 1 and 10"):
        validate_assay_transfer_records_per_molecule(
            value, selection_unit="unique_molecule"
        )


def test_multiple_records_require_unique_molecule_selection():
    with pytest.raises(ValueError, match="requires selection unit unique_molecule"):
        validate_assay_transfer_records_per_molecule(
            2, selection_unit=ASSAY_TRANSFER_SELECTION_SCORED_RECORD
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
