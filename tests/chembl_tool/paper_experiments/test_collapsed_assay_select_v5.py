import numpy as np
from scipy import sparse

from tools.chembl_tool.paper_experiments.collapsed_assay_knn.select_v5 import (
    aggregate_categorical,
    aggregate_numeric,
    neighbor_selector,
    numeric_catalog,
    select_configs,
    supported_categorical_catalog,
)
from tools.chembl_tool.paper_experiments.collapsed_assay_knn.select_v6 import (
    select_models,
    surface_matrices,
)
from tools.chembl_tool.paper_experiments.collapsed_assay_knn.select_v7 import (
    aggregate_all_neighbors,
    select_models as select_models_v7,
    surface_matrices as surface_matrices_v7,
)


def test_observed_and_similarity_weighted_aggregation() -> None:
    # Two value columns followed by their two presence columns.
    reference = sparse.csr_matrix(
        [
            [1.0, 0.0, 1.0, 0.0],
            [3.0, 4.0, 1.0, 1.0],
            [0.0, 6.0, 0.0, 1.0],
        ]
    )
    neighbors = [[0, 1, 2]]
    plain = neighbor_selector(neighbors, 3)
    weighted = neighbor_selector(neighbors, 3, [[1.0, 0.5, 0.25]])

    np.testing.assert_allclose(
        aggregate_numeric(reference, plain, 2, observed_only=False).toarray(),
        [[4.0 / 3.0, 10.0 / 3.0]],
    )
    np.testing.assert_allclose(
        aggregate_numeric(reference, plain, 2, observed_only=True).toarray(),
        [[2.0, 5.0]],
    )
    np.testing.assert_allclose(
        aggregate_numeric(reference, weighted, 2, observed_only=True).toarray(),
        [[2.5 / 1.5, 3.5 / 0.75]],
    )
    np.testing.assert_allclose(
        aggregate_numeric(
            reference, plain, 2, observed_only=True, include_presence=True
        ).toarray(),
        [[2.0, 5.0, 2.0 / 3.0, 2.0 / 3.0]],
    )


def test_categorical_distribution_normalizes_within_assay() -> None:
    # Columns 0/1 are categories of assay A; column 2 is assay B.
    reference = sparse.csr_matrix([[1, 0, 1], [0, 1, 0], [1, 0, 0]])
    result = aggregate_categorical(
        reference,
        neighbor_selector([[0, 1, 2]], 3),
        np.asarray([0, 0, 1]),
        2,
    )
    np.testing.assert_allclose(result.toarray(), [[2.0 / 3.0, 1.0 / 3.0, 1.0]])


def test_repeated_mean_selects_the_best_configuration() -> None:
    metrics = []
    for seed, small, large in [(0, 0.60, 0.60), (1, 0.61, 0.64), (2, 0.62, 0.62)]:
        metrics.extend(
            [
                {"seed": seed, "recipe": "r", "k": 3, "c": 0.1, "macro_f1": small},
                {"seed": seed, "recipe": "r", "k": 5, "c": 1.0, "macro_f1": large},
            ]
        )
    selected = select_configs(metrics)[0]
    assert (selected["k"], selected["c"]) == (5, 1.0)
    assert selected["standard_error"] > 0


def test_v6_selects_assay_k_and_lr_c_by_repeated_mean() -> None:
    metrics = []
    for seed, small, large in [(0, 0.60, 0.61), (1, 0.61, 0.64), (2, 0.62, 0.62)]:
        for k, c_value, score in ((3, 0.1, small), (25, 1.0, large)):
            metrics.append(
                {
                    "seed": seed,
                    "surface": "neighbor_indirect_plus_label",
                    "model_family": "logistic_l2",
                    "assay_k": k,
                    "label_k": 5,
                    "c": c_value,
                    "rf_profile": None,
                    "rf_max_depth": None,
                    "rf_min_samples_leaf": None,
                    "macro_f1": score,
                }
            )
    selected = select_models(metrics)[0]
    assert selected["selected_assay_k"] == 25
    assert selected["selected_label_k"] == 5
    assert selected["selected_c"] == 1.0


def test_v6_combined_surface_uses_frozen_label_k() -> None:
    features = (
        sparse.csr_matrix([[1.0, 1.0], [3.0, 1.0], [5.0, 1.0], [7.0, 1.0]]),
        sparse.csr_matrix([[1, 0], [0, 1], [1, 0], [0, 1]]),
        np.asarray([0, 0]),
        1,
        [[0, 1, 2, 3]],
        [[0, 1, 2, 3]],
        None,
        None,
    )
    matrices = surface_matrices(
        features,
        np.asarray([0, 0, 1, 1]),
        assay_k=2,
        label_k=3,
    )
    indirect = matrices["logistic_l2"]["neighbor_indirect"][1].toarray()
    combined = matrices["logistic_l2"]["neighbor_indirect_plus_label"][1].toarray()
    np.testing.assert_allclose(indirect, [[2.0]])
    np.testing.assert_allclose(combined, [[2.0, 1.0 / 3.0]])


def test_v7_categorical_average_uses_all_neighbors() -> None:
    reference = sparse.csr_matrix([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    plain = neighbor_selector([[0, 1, 2]], 3)
    weighted = neighbor_selector([[0, 1, 2]], 3, [[1.0, 0.5, 0.25]])
    np.testing.assert_allclose(
        aggregate_all_neighbors(reference, plain).toarray(), [[1.0 / 3.0, 1.0 / 3.0]]
    )
    np.testing.assert_allclose(
        aggregate_all_neighbors(reference, weighted).toarray(),
        [[1.0 / 1.75, 0.25 / 1.75]],
    )


def test_v7_uses_observed_numeric_and_both_label_k_modes() -> None:
    features = (
        sparse.csr_matrix(
            [[1.0, 0.0, 1.0, 0.0], [3.0, 4.0, 1.0, 1.0], [0.0, 6.0, 0.0, 1.0]]
        ),
        sparse.csr_matrix([[1, 0], [0, 0], [0, 1]]),
        np.asarray([0, 0]),
        1,
        [[0, 1, 2]],
        [[0, 1, 2]],
        [[1.0, 0.5, 0.25]],
        [[1.0, 0.5, 0.25]],
    )
    matrices = surface_matrices_v7(
        features,
        np.asarray([0, 1, 1]),
        assay_k=3,
        standalone_label_k=2,
        weighting="unweighted",
    )["logistic_l2"]
    np.testing.assert_allclose(
        matrices[("neighbor_indirect", None)][1].toarray(), [[2.0, 5.0]]
    )
    np.testing.assert_allclose(
        matrices[("neighbor_indirect_plus_label", "standalone_best")][1].toarray(),
        [[2.0, 5.0, 0.5]],
    )
    np.testing.assert_allclose(
        matrices[("neighbor_indirect_plus_label", "same_assay_k")][1].toarray(),
        [[2.0, 5.0, 2.0 / 3.0]],
    )


def test_v7_exact_tie_prefers_the_simpler_configuration() -> None:
    metrics = []
    for assay_k, weighting, label_mode, label_k, c_value in (
        (25, "similarity", "same_assay_k", 25, 1.0),
        (3, "unweighted", "standalone_best", 3, 0.1),
    ):
        metrics.append(
            {
                "surface": "neighbor_indirect_plus_label",
                "model_family": "logistic_l2",
                "assay_k": assay_k,
                "weighting": weighting,
                "label_mode": label_mode,
                "label_k": label_k,
                "c": c_value,
                "rf_profile": None,
                "rf_max_depth": None,
                "rf_min_samples_leaf": None,
                "macro_f1": 0.6,
            }
        )
    selected = select_models_v7(metrics)[0]
    assert selected["selected_assay_k"] == 3
    assert selected["selected_weighting"] == "unweighted"
    assert selected["selected_label_mode"] == "standalone_best"
    assert selected["selected_c"] == 0.1


def test_feature_catalogs_apply_the_requested_minimum_support() -> None:
    rows = [{"parent_key": f"p{index}"} for index in range(6)]
    numeric = {
        row["parent_key"]: {
            "support6": float(index),
            **({"support5": float(index)} if index < 5 else {}),
        }
        for index, row in enumerate(rows)
    }
    categorical = {
        row["parent_key"]: {
            "support6": "yes",
            **({"support5": "yes"} if index < 5 else {}),
        }
        for index, row in enumerate(rows)
    }
    numeric_features, _ = numeric_catalog(rows, numeric, min_support=6)
    categorical_features, _, _, _ = supported_categorical_catalog(
        rows, categorical, min_support=6
    )
    assert [row["pair_bucket_key"] for row in numeric_features] == ["support6"]
    assert [row["pair_bucket_key"] for row in categorical_features] == ["support6"]
