import pytest

from optimization.select_records import select_records


def _candidates() -> list[dict]:
    rows = []
    rank = 0
    parents = (
        ("a", "CCO", 0.9, 4),
        ("b", "c1ccccc1", 0.8, 2),
        ("c", "CCN(C)C", 0.7, 1),
    )
    for parent_id, parent_smiles, similarity, count in parents:
        for within_parent_rank in range(1, count + 1):
            rank += 1
            rows.append({
                "item_id": f"u{rank}",
                "parent_id": parent_id,
                "parent_smiles": parent_smiles,
                "morgan_similarity": similarity,
                "morgan_rank": rank,
                "within_parent_rank": within_parent_rank,
                "assay_transfer_score": None,
            })
    return rows


def test_zero_lambda_is_exact_morgan_top_k() -> None:
    selected, summary = select_records(_candidates(), k=3, parent_lambda=0)

    assert [row["item_id"] for row in selected] == ["u1", "u2", "u3"]
    assert summary["distinct_selected_parents"] == 1
    assert summary["similarity_sum"] == pytest.approx(2.7)
    assert summary["objective_score"] == pytest.approx(0.9)


def test_square_root_term_softly_diversifies_parents() -> None:
    selected, summary = select_records(_candidates(), k=3, parent_lambda=0.5)

    assert [row["parent_id"] for row in selected] == ["a", "b", "c"]
    assert summary["distinct_selected_parents"] == 3
    assert summary["max_records_one_parent"] == 1
    assert sum(row["marginal_gain"] for row in selected) == pytest.approx(
        summary["objective_score"]
    )


def test_assay_transfer_score_changes_selection_when_complete() -> None:
    candidates = _candidates()
    assay_by_parent = {"a": 0.1, "b": 0.95, "c": 0.8}
    for row in candidates:
        row["assay_transfer_score"] = assay_by_parent[row["parent_id"]]

    selected, summary = select_records(
        candidates, k=2, parent_lambda=0, assay_lambda=1,
    )

    assert [row["parent_id"] for row in selected] == ["b", "b"]
    assert summary["assay_score_mean"] == pytest.approx(0.95)


def test_fingerprint_coverage_diversifies_without_pairwise_similarity() -> None:
    candidates = _candidates()
    baseline, baseline_summary = select_records(candidates, k=2, parent_lambda=0)
    selected, summary = select_records(
        candidates, k=2, parent_lambda=0, diversity_lambda=2,
    )

    assert [row["parent_id"] for row in baseline] == ["a", "a"]
    assert len({row["parent_id"] for row in selected}) == 2
    assert summary["fingerprint_coverage"] > baseline_summary["fingerprint_coverage"]


def test_positive_assay_lambda_rejects_absent_scores() -> None:
    with pytest.raises(ValueError, match="requires complete"):
        select_records(_candidates(), k=2, parent_lambda=0, assay_lambda=0.1)


@pytest.mark.parametrize("k,parent_lambda", [(0, 0), (8, 0), (1, -0.1)])
def test_invalid_constraints_fail(k: int, parent_lambda: float) -> None:
    with pytest.raises(ValueError):
        select_records(_candidates(), k=k, parent_lambda=parent_lambda)


def test_parent_similarity_must_be_consistent() -> None:
    candidates = _candidates()
    candidates[1]["morgan_similarity"] = 0.89

    with pytest.raises(ValueError, match="varies within parent"):
        select_records(candidates, k=2, parent_lambda=0.05)
