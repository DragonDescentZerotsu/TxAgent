import pytest

from optimization.select_records import (
    PROFILES,
    _feature_coverage,
    _greedy_coverage_ceiling,
    select_records,
)


def _candidates() -> list[dict]:
    return [
        {
            "item_id": "u1", "parent_id": "a", "parent_smiles": "CCO",
            "morgan_similarity": 0.90, "assay_transfer_score": 0.10,
            "semantic_bucket_id": "s1", "semantic_weight": 0.10,
        },
        {
            "item_id": "u2", "parent_id": "a", "parent_smiles": "CCO",
            "morgan_similarity": 0.90, "assay_transfer_score": 0.10,
            "semantic_bucket_id": "s1", "semantic_weight": 0.10,
        },
        {
            "item_id": "u3", "parent_id": "b", "parent_smiles": "c1ccccc1",
            "morgan_similarity": 0.80, "assay_transfer_score": 0.95,
            "semantic_bucket_id": "s2", "semantic_weight": 0.95,
        },
        {
            "item_id": "u4", "parent_id": "c", "parent_smiles": "CCN(C)C",
            "morgan_similarity": 0.70, "assay_transfer_score": 0.80,
            "semantic_bucket_id": "s3", "semantic_weight": 0.80,
        },
    ]


def test_zero_lambdas_are_exact_morgan_top_k() -> None:
    selected, summary = select_records(_candidates(), k=2)

    assert [row["item_id"] for row in selected] == ["u1", "u2"]
    assert summary["morgan"] == pytest.approx(0.9)
    assert summary["objective_score"] == pytest.approx(0.9)


def test_assay_transfer_changes_selection() -> None:
    selected, summary = select_records(_candidates(), k=2, assay_lambda=1)

    assert {row["item_id"] for row in selected} == {"u3", "u4"}
    assert summary["assay"] == pytest.approx(0.875)


def test_gated_assay_requires_both_structure_and_transfer() -> None:
    candidates = _candidates()
    candidates[0]["assay_transfer_score"] = 0.4
    candidates[1]["assay_transfer_score"] = 0.4
    candidates[2]["morgan_similarity"] = 0.2
    candidates[2]["assay_transfer_score"] = 1.0
    candidates[3]["assay_transfer_score"] = 0.1

    additive, _ = select_records(candidates, k=1, assay_lambda=2)
    gated, summary = select_records(candidates, k=1, gated_assay_lambda=2)

    assert additive[0]["item_id"] == "u3"
    assert gated[0]["item_id"] == "u1"
    assert summary["assay_gated"] == pytest.approx(0.36)


def test_normalized_gated_relevance_uses_alpha_and_unit_scale() -> None:
    candidates = _candidates()
    selected, summary = select_records(
        candidates, k=1, gated_assay_lambda=1,
        normalized_gated_objective=True,
    )

    assert selected[0]["item_id"] == "u3"
    assert summary["normalized_relevance"] == pytest.approx((0.8 + 0.76) / 2)
    assert summary["objective_score"] == pytest.approx(summary["normalized_relevance"])


def test_normalized_gated_relevance_falls_back_to_morgan_without_assay() -> None:
    candidates = _candidates()
    for row in candidates:
        row["assay_transfer_score"] = None

    selected, summary = select_records(
        candidates, k=2, gated_assay_lambda=1.25,
        normalized_gated_objective=True,
    )

    assert [row["item_id"] for row in selected] == ["u1", "u2"]
    assert summary["normalized_relevance"] == pytest.approx(summary["morgan"])
    assert summary["gated_assay_lambda_effective"] == 0


def test_feature_coverage_diversifies_without_full_similarity_matrix() -> None:
    baseline, baseline_summary = select_records(_candidates(), k=2)
    selected, summary = select_records(_candidates(), k=2, molecular_lambda=1)

    assert [row["parent_id"] for row in baseline] == ["a", "a"]
    assert len({row["parent_id"] for row in selected}) == 2
    assert summary["molecular_coverage"] > baseline_summary["molecular_coverage"]
    assert summary["molecular_diversity"] > baseline_summary["molecular_diversity"]


def test_feature_coverage_has_diminishing_returns() -> None:
    universe = {1, 2, 3, 4}
    item = {2, 3}
    small = _feature_coverage(({1}, item), universe) - _feature_coverage(({1},), universe)
    large = _feature_coverage(({1}, {2}), universe)
    large = _feature_coverage(({1}, {2}, item), universe) - large

    assert small >= large


def test_greedy_coverage_ceiling_rescales_the_attainable_k_coverage() -> None:
    bits = [frozenset({1, 2}), frozenset({2, 3}), frozenset({4})]

    assert _greedy_coverage_ceiling(bits, frozenset({1, 2, 3, 4}), 2) == 0.75
    _, summary = select_records(
        _candidates(), k=2, molecular_lambda=1,
        normalized_gated_objective=True,
    )
    assert 0 < summary["molecular_coverage_ceiling"] <= 1
    assert 0 <= summary["molecular_coverage_normalized"] <= 1


def test_semantic_relevance_and_diversity_have_separate_effects() -> None:
    relevant, relevance_summary = select_records(
        _candidates(), k=2, semantic_relevance_lambda=1
    )
    diverse, diversity_summary = select_records(
        _candidates(), k=2, semantic_diversity_lambda=1
    )

    assert {row["item_id"] for row in relevant} == {"u3", "u4"}
    assert relevance_summary["semantic_relevance"] == pytest.approx(0.875)
    assert len({row["semantic_bucket_id"] for row in diverse}) == 2
    assert diversity_summary["semantic_coverage"] == pytest.approx(1.0)
    assert diversity_summary["semantic_diversity"] == pytest.approx(1.0)


def test_unavailable_assay_term_is_omitted_for_that_level() -> None:
    candidates = _candidates()
    for row in candidates:
        row["assay_transfer_score"] = None

    selected, summary = select_records(
        candidates, k=2, assay_lambda=1, gated_assay_lambda=1
    )

    assert [row["item_id"] for row in selected] == ["u1", "u2"]
    assert summary["assay_available"] is False
    assert summary["assay_lambda_effective"] == 0
    assert summary["gated_assay_lambda_effective"] == 0


def test_components_are_normalized_for_interpretable_lambdas() -> None:
    _, summary = select_records(
        _candidates(), k=3, assay_lambda=1, molecular_lambda=1,
        semantic_relevance_lambda=1, semantic_diversity_lambda=1,
    )

    for field in (
        "morgan", "assay", "assay_gated", "molecular_coverage", "molecular_diversity",
        "semantic_relevance", "semantic_coverage", "semantic_diversity",
    ):
        assert 0 <= summary[field] <= 1


@pytest.mark.parametrize(
    "kwargs",
    ({"k": 0}, {"k": 5}, {"k": 1, "molecular_lambda": -0.1}),
)
def test_invalid_constraints_fail(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        select_records(_candidates(), **kwargs)


def test_partial_assay_scores_and_inconsistent_parent_similarity_fail() -> None:
    partial = _candidates()
    partial[0]["assay_transfer_score"] = None
    with pytest.raises(ValueError, match="complete or absent"):
        select_records(partial, k=2)

    inconsistent = _candidates()
    inconsistent[1]["morgan_similarity"] = 0.89
    with pytest.raises(ValueError, match="varies within parent"):
        select_records(inconsistent, k=2)


def test_profile_grid_contains_baseline_single_terms_and_balanced_values() -> None:
    assert len(PROFILES) == 16
    assert PROFILES["baseline"].lambdas() == {
        "assay": 0.0, "assay_gated": 0.0, "molecular_coverage": 0.0,
        "semantic_relevance": 0.0, "semantic_coverage": 0.0,
    }
    assert PROFILES["balanced_10"].gated_assay_lambda == 0
    assert set(PROFILES["balanced_10"].lambdas().values()) == {0.0, 1.0}
