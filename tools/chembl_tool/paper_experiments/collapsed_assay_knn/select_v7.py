"""Select filtered observed-continuous/all-neighbor-categorical assay models."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from tools.chembl_tool.paper_experiments.collapsed_assay_knn.model_ablation import (
    fit_predict,
)
from tools.chembl_tool.paper_experiments.collapsed_assay_knn.run import (
    REPO_ROOT,
    TASKS,
    _file_entry,
    _write_json,
    binary_metrics,
    encode_assays,
    load_numeric_assays,
    neighbor_label_scores,
    read_split,
)
from tools.chembl_tool.paper_experiments.collapsed_assay_knn.select_k_cv import (
    load_categorical_assays,
    make_folds,
)
from tools.chembl_tool.paper_experiments.collapsed_assay_knn.select_v5 import (
    DEFAULT_INPUT_ROOT,
    DEFAULT_K_VALUES,
    DEFAULT_SEEDS,
    aggregate_numeric,
    fit_lr,
    fold_features,
    neighbor_selector,
    numeric_catalog,
    supported_categorical_catalog,
    write_tsv,
)
from tools.chembl_tool.paper_experiments.collapsed_assay_knn.select_v5_rf import (
    RF_PROFILES,
)
from tools.chembl_tool.paper_experiments.collapsed_assay_knn.select_v6 import (
    select_label_k,
)


VERSION = "collapsed_assay_filtered_observed_weighting.v7"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "outputs/paper/collapsed_assay_filtered_observed_weighting_v7"
)
DEFAULT_C_VALUES = (0.1, 1.0, 10.0)
WEIGHTINGS = ("unweighted", "similarity")
LABEL_MODES = ("standalone_best", "same_assay_k")
NEIGHBOR_SURFACES = (
    "neighbor_indirect",
    "neighbor_indirect_plus_label",
    "neighbor_indirect_plus_categorical_plus_label",
)
WIDE_COLUMNS = (
    ("canonical label KNN", "morgan_label_knn", None),
    ("self LR", "query_self_indirect", "logistic_l2"),
    ("self RF", "query_self_indirect", "random_forest"),
    ("indirect LR", "neighbor_indirect", "logistic_l2"),
    ("indirect RF", "neighbor_indirect", "random_forest"),
    (
        "indirect plus direct labels LR",
        "neighbor_indirect_plus_label",
        "logistic_l2",
    ),
    (
        "indirect plus direct labels RF",
        "neighbor_indirect_plus_label",
        "random_forest",
    ),
    (
        "indirect plus categorical plus direct LR",
        "neighbor_indirect_plus_categorical_plus_label",
        "logistic_l2",
    ),
    (
        "indirect plus categorical plus direct RF",
        "neighbor_indirect_plus_categorical_plus_label",
        "random_forest",
    ),
)


def aggregate_all_neighbors(
    reference_matrix: sparse.csr_matrix, selector: sparse.csr_matrix
) -> sparse.csr_matrix:
    """Average one-hot values over every selected neighbor, including missing zeros."""
    totals = np.asarray(selector.sum(axis=1)).ravel()
    inverse = np.zeros_like(totals)
    inverse[totals > 0] = 1.0 / totals[totals > 0]
    return (sparse.diags(inverse) @ selector @ reference_matrix).tocsr()


def surface_matrices(
    features: tuple[Any, ...],
    reference_labels: np.ndarray,
    assay_k: int,
    standalone_label_k: int,
    weighting: str,
) -> dict[str, dict[tuple[str, str | None], tuple[sparse.csr_matrix, sparse.csr_matrix]]]:
    numeric_reference = features[0]
    n_numeric_assays = numeric_reference.shape[1] // 2
    n_reference = numeric_reference.shape[0]
    similarities = None if weighting == "unweighted" else (features[6], features[7])
    train_selector = neighbor_selector(
        features[4],
        assay_k,
        None if similarities is None else similarities[0],
        n_reference,
    )
    query_selector = neighbor_selector(
        features[5],
        assay_k,
        None if similarities is None else similarities[1],
        n_reference,
    )
    numeric_train_rf = aggregate_numeric(
        numeric_reference,
        train_selector,
        n_numeric_assays,
        observed_only=True,
        include_presence=True,
    )
    numeric_query_rf = aggregate_numeric(
        numeric_reference,
        query_selector,
        n_numeric_assays,
        observed_only=True,
        include_presence=True,
    )
    numeric_train_lr = numeric_train_rf[:, :n_numeric_assays]
    numeric_query_lr = numeric_query_rf[:, :n_numeric_assays]
    categorical_train = aggregate_all_neighbors(features[1], train_selector)
    categorical_query = aggregate_all_neighbors(features[1], query_selector)

    label_columns = {}
    for mode, label_k in (
        ("standalone_best", standalone_label_k),
        ("same_assay_k", assay_k),
    ):
        label_columns[mode] = (
            sparse.csr_matrix(
                neighbor_label_scores(reference_labels, features[4], label_k)[:, None]
            ),
            sparse.csr_matrix(
                neighbor_label_scores(reference_labels, features[5], label_k)[:, None]
            ),
        )

    result = {}
    for model, numeric_pair in (
        ("logistic_l2", (numeric_train_lr, numeric_query_lr)),
        ("random_forest", (numeric_train_rf, numeric_query_rf)),
    ):
        model_matrices = {("neighbor_indirect", None): numeric_pair}
        for mode, (label_train, label_query) in label_columns.items():
            model_matrices[("neighbor_indirect_plus_label", mode)] = (
                sparse.hstack([numeric_pair[0], label_train], format="csr"),
                sparse.hstack([numeric_pair[1], label_query], format="csr"),
            )
            model_matrices[
                ("neighbor_indirect_plus_categorical_plus_label", mode)
            ] = (
                sparse.hstack(
                    [numeric_pair[0], categorical_train, label_train], format="csr"
                ),
                sparse.hstack(
                    [numeric_pair[1], categorical_query, label_query], format="csr"
                ),
            )
        result[model] = model_matrices
    return result


def fit_grid(
    model: str,
    fit_matrix: sparse.csr_matrix,
    labels: np.ndarray,
    query_matrix: sparse.csr_matrix,
    c_values: tuple[float, ...],
) -> list[tuple[float | None, str | None, np.ndarray, np.ndarray]]:
    results = []
    if model == "logistic_l2":
        for c_value in c_values:
            predicted, scores = fit_lr(fit_matrix, labels, query_matrix, c_value)
            results.append((c_value, None, predicted, scores))
        return results
    for profile in RF_PROFILES:
        predicted, scores, _ = fit_predict(
            "random_forest",
            fit_matrix,
            labels,
            query_matrix,
            deterministic_rf_scores=True,
            rf_params=profile,
        )
        results.append((None, profile["profile"], predicted, scores))
    return results


def repeated_model_cv(
    train_rows: list[dict[str, Any]],
    numeric_assays: dict[str, dict[str, float]],
    categorical_assays: dict[str, dict[str, str]],
    standalone_label_k: int,
    k_values: tuple[int, ...],
    c_values: tuple[float, ...],
    seeds: tuple[int, ...],
    min_feature_support: int,
) -> list[dict[str, Any]]:
    labels = np.asarray([row["Y"] for row in train_rows], dtype=np.int64)
    metrics = []
    for seed in seeds:
        print(f"  model CV seed {seed}", flush=True)
        predictions: dict[tuple[Any, ...], np.ndarray] = {}
        scores: dict[tuple[Any, ...], np.ndarray] = {}
        folds, _ = make_folds(train_rows, n_folds=5, seed=seed)
        for reference_indices, query_indices in folds:
            reference_rows = [train_rows[index] for index in reference_indices]
            query_rows = [train_rows[index] for index in query_indices]
            reference_labels = labels[reference_indices]
            features = fold_features(
                reference_rows,
                query_rows,
                numeric_assays,
                categorical_assays,
                max(k_values),
                min_feature_support,
            )
            catalog, feature_index = numeric_catalog(
                reference_rows, numeric_assays, min_feature_support
            )
            self_train, _ = encode_assays(
                reference_rows, numeric_assays, catalog, feature_index
            )
            self_query, _ = encode_assays(
                query_rows, numeric_assays, catalog, feature_index
            )
            n_assays = len(catalog)
            for model, matrix_pair in (
                ("logistic_l2", (self_train[:, :n_assays], self_query[:, :n_assays])),
                ("random_forest", (self_train, self_query)),
            ):
                for c_value, profile, predicted, predicted_scores in fit_grid(
                    model, matrix_pair[0], reference_labels, matrix_pair[1], c_values
                ):
                    key = (
                        "query_self_indirect",
                        model,
                        None,
                        None,
                        None,
                        c_value,
                        profile,
                    )
                    predictions.setdefault(
                        key, np.empty(len(train_rows), dtype=np.int64)
                    )[query_indices] = predicted
                    scores.setdefault(
                        key, np.empty(len(train_rows), dtype=np.float64)
                    )[query_indices] = predicted_scores

            for assay_k in k_values:
                for weighting in WEIGHTINGS:
                    matrices = surface_matrices(
                        features,
                        reference_labels,
                        assay_k,
                        standalone_label_k,
                        weighting,
                    )
                    for model, model_matrices in matrices.items():
                        for (surface, label_mode), matrix_pair in model_matrices.items():
                            if (
                                label_mode == "same_assay_k"
                                and assay_k == standalone_label_k
                            ):
                                continue
                            for c_value, profile, predicted, predicted_scores in fit_grid(
                                model,
                                matrix_pair[0],
                                reference_labels,
                                matrix_pair[1],
                                c_values,
                            ):
                                key = (
                                    surface,
                                    model,
                                    assay_k,
                                    weighting,
                                    label_mode,
                                    c_value,
                                    profile,
                                )
                                predictions.setdefault(
                                    key, np.empty(len(train_rows), dtype=np.int64)
                                )[query_indices] = predicted
                                scores.setdefault(
                                    key, np.empty(len(train_rows), dtype=np.float64)
                                )[query_indices] = predicted_scores

        for key, predicted in predictions.items():
            surface, model, assay_k, weighting, label_mode, c_value, profile_name = key
            result = binary_metrics(labels, predicted, scores[key])
            profile = next(
                (row for row in RF_PROFILES if row["profile"] == profile_name), None
            )
            label_k = None
            if label_mode == "standalone_best":
                label_k = standalone_label_k
            elif label_mode == "same_assay_k":
                label_k = assay_k
            metrics.append(
                {
                    "seed": seed,
                    "surface": surface,
                    "model_family": model,
                    "assay_k": assay_k,
                    "weighting": weighting,
                    "label_mode": label_mode,
                    "label_k": label_k,
                    "c": c_value,
                    "rf_profile": profile_name,
                    "rf_max_depth": None if profile is None else profile["max_depth"],
                    "rf_min_samples_leaf": (
                        None if profile is None else profile["min_samples_leaf"]
                    ),
                    **{
                        name: result[name]
                        for name in ("n", "macro_f1", "accuracy", "auroc")
                    },
                }
            )
    return metrics


def select_models(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    provenance = {}
    for row in metrics:
        key = (
            row["surface"],
            row["model_family"],
            row["assay_k"],
            row["weighting"],
            row["label_mode"],
            row["c"],
            row["rf_profile"],
        )
        grouped[key].append(row["macro_f1"])
        provenance[key] = row

    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for key, values in grouped.items():
        surface, model, assay_k, weighting, label_mode, c_value, profile_name = key
        source = provenance[key]
        repeat_scores = np.asarray(values)
        candidates[(surface, model)].append(
            {
                "surface": surface,
                "model_family": model,
                "selected_assay_k": assay_k,
                "selected_weighting": weighting,
                "selected_label_mode": label_mode,
                "selected_label_k": source["label_k"],
                "selected_c": c_value,
                "selected_rf_profile": profile_name,
                "selected_rf_max_depth": source["rf_max_depth"],
                "selected_rf_min_samples_leaf": source["rf_min_samples_leaf"],
                "mean_oof_macro_f1": float(repeat_scores.mean()),
                "standard_error": (
                    float(repeat_scores.std(ddof=1) / np.sqrt(len(repeat_scores)))
                    if len(repeat_scores) > 1
                    else 0.0
                ),
            }
        )

    selected = []
    for (_, model), rows in sorted(candidates.items()):
        if model == "logistic_l2":
            choice = max(
                rows,
                key=lambda row: (
                    row["mean_oof_macro_f1"],
                    -(row["selected_assay_k"] or 0),
                    row["selected_weighting"] != "similarity",
                    row["selected_label_mode"] != "same_assay_k",
                    -row["selected_c"],
                ),
            )
        else:
            choice = max(
                rows,
                key=lambda row: (
                    row["mean_oof_macro_f1"],
                    -(row["selected_assay_k"] or 0),
                    row["selected_weighting"] != "similarity",
                    row["selected_label_mode"] != "same_assay_k",
                    row["selected_rf_min_samples_leaf"],
                    row["selected_rf_max_depth"] is not None,
                ),
            )
        selected.append(choice)
    return selected


def validate(
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    numeric_assays: dict[str, dict[str, float]],
    categorical_assays: dict[str, dict[str, str]],
    standalone_label_k: int,
    selections: list[dict[str, Any]],
    min_feature_support: int,
) -> list[dict[str, Any]]:
    labels = np.asarray([row["Y"] for row in train_rows], dtype=np.int64)
    valid_labels = np.asarray([row["Y"] for row in valid_rows], dtype=np.int64)
    max_k = max(
        standalone_label_k,
        max(row["selected_assay_k"] or 0 for row in selections),
    )
    features = fold_features(
        train_rows,
        valid_rows,
        numeric_assays,
        categorical_assays,
        max_k,
        min_feature_support,
    )
    catalog, feature_index = numeric_catalog(
        train_rows, numeric_assays, min_feature_support
    )
    self_train, _ = encode_assays(train_rows, numeric_assays, catalog, feature_index)
    self_valid, _ = encode_assays(valid_rows, numeric_assays, catalog, feature_index)
    n_assays = len(catalog)
    matrix_cache = {
        (row["selected_assay_k"], row["selected_weighting"]): surface_matrices(
            features,
            labels,
            row["selected_assay_k"],
            standalone_label_k,
            row["selected_weighting"],
        )
        for row in selections
        if row["selected_assay_k"] is not None
    }

    results = []
    for selection in selections:
        surface = selection["surface"]
        model = selection["model_family"]
        if surface == "query_self_indirect":
            matrix_pair = (
                (self_train[:, :n_assays], self_valid[:, :n_assays])
                if model == "logistic_l2"
                else (self_train, self_valid)
            )
        else:
            matrix_pair = matrix_cache[
                (selection["selected_assay_k"], selection["selected_weighting"])
            ][model][(surface, selection["selected_label_mode"])]
        if model == "logistic_l2":
            predicted, scores = fit_lr(
                matrix_pair[0], labels, matrix_pair[1], selection["selected_c"]
            )
        else:
            predicted, scores, _ = fit_predict(
                "random_forest",
                matrix_pair[0],
                labels,
                matrix_pair[1],
                deterministic_rf_scores=True,
                rf_params={
                    "max_depth": selection["selected_rf_max_depth"],
                    "min_samples_leaf": selection["selected_rf_min_samples_leaf"],
                },
            )
        result = binary_metrics(valid_labels, predicted, scores)
        results.append(
            {
                **selection,
                "n_features": matrix_pair[0].shape[1],
                **{
                    name: result[name]
                    for name in ("n", "macro_f1", "accuracy", "auroc")
                },
            }
        )

    label_scores = neighbor_label_scores(labels, features[5], standalone_label_k)
    label_result = binary_metrics(
        valid_labels, (label_scores >= 0.5).astype(np.int64), label_scores
    )
    results.append(
        {
            "surface": "morgan_label_knn",
            "model_family": None,
            "selected_assay_k": None,
            "selected_weighting": None,
            "selected_label_mode": "standalone_best",
            "selected_label_k": standalone_label_k,
            "selected_c": None,
            "selected_rf_profile": None,
            "selected_rf_max_depth": None,
            "selected_rf_min_samples_leaf": None,
            "mean_oof_macro_f1": None,
            "standard_error": None,
            "n_features": 1,
            **{
                name: label_result[name]
                for name in ("n", "macro_f1", "accuracy", "auroc")
            },
        }
    )
    return results


def write_wide_tables(output_root: Path, valid_metrics: list[dict[str, Any]]) -> None:
    table_names = {
        "validation_macro_f1.tsv": "macro_f1",
        "selected_assay_k.tsv": "selected_assay_k",
        "selected_label_k.tsv": "selected_label_k",
        "selected_weighting.tsv": "selected_weighting",
    }
    columns = ["task"] + [column for column, _, _ in WIDE_COLUMNS]
    for filename, value_key in table_names.items():
        rows = []
        for task in sorted({row["task"] for row in valid_metrics}):
            task_rows = [row for row in valid_metrics if row["task"] == task]
            output_row = {"task": task}
            for column, surface, model in WIDE_COLUMNS:
                row = next(
                    item
                    for item in task_rows
                    if item["surface"] == surface and item["model_family"] == model
                )
                output_row[column] = row[value_key]
            rows.append(output_row)
        write_tsv(output_root / filename, rows, columns)


def report(valid_metrics: list[dict[str, Any]], min_feature_support: int) -> str:
    lookup = {
        (row["task"], row["surface"], row["model_family"]): row
        for row in valid_metrics
    }
    lines = [
        "# Filtered observed-continuous/all-neighbor-categorical validation",
        "",
        "Continuous assays use observed-only means. Categorical one-hot features use all ",
        "selected neighbors as the denominator. Weighting is tied across both blocks. ",
        f"Every retained feature has train support at least {min_feature_support}. ",
        "Direct-label K is selected first; combined models compare it with same-assay K. ",
        "Test and ClinTox were not read.",
        "",
        "| Task | " + " | ".join(column for column, _, _ in WIDE_COLUMNS) + " |",
        "|---|" + "---:|" * len(WIDE_COLUMNS),
    ]
    for task in sorted({row["task"] for row in valid_metrics}):
        values = [
            lookup[(task, surface, model)]["macro_f1"]
            for _, surface, model in WIDE_COLUMNS
        ]
        lines.append(
            f"| {task} | " + " | ".join(f"{value:.4f}" for value in values) + " |"
        )
    return "\n".join(lines) + "\n"


def run(
    tasks: list[str],
    input_root: Path,
    output_root: Path,
    k_values: tuple[int, ...],
    c_values: tuple[float, ...],
    seeds: tuple[int, ...],
    min_feature_support: int,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    all_label_cv = []
    all_label_selections = []
    all_model_cv = []
    all_model_selections = []
    all_valid = []
    feature_counts = []
    for task in tasks:
        print(task, flush=True)
        config = TASKS[task]
        train_rows = read_split(config["data_dir"] / "train.jsonl")
        valid_rows = read_split(config["data_dir"] / "valid.jsonl")
        parent_keys = {row["parent_key"] for row in train_rows + valid_rows}
        numeric_assays, numeric_stats = load_numeric_assays(config["records"], parent_keys)
        categorical_assays, categorical_stats = load_categorical_assays(
            config["records"], parent_keys
        )
        numeric_features, _ = numeric_catalog(
            train_rows, numeric_assays, min_feature_support
        )
        categorical_features, _, _, _ = supported_categorical_catalog(
            train_rows, categorical_assays, min_feature_support
        )
        feature_counts.append(
            {
                "task": task,
                "n_numeric_features": len(numeric_features),
                "n_categorical_features": len(categorical_features),
                "n_train": len(train_rows),
            }
        )

        print("  selecting standalone direct-label K", flush=True)
        label_cv, label_selection = select_label_k(train_rows, k_values, seeds)
        standalone_label_k = label_selection["selected_label_k"]
        print(f"  frozen standalone direct-label K={standalone_label_k}", flush=True)
        model_cv = repeated_model_cv(
            train_rows,
            numeric_assays,
            categorical_assays,
            standalone_label_k,
            k_values,
            c_values,
            seeds,
            min_feature_support,
        )
        model_selections = select_models(model_cv)
        valid_metrics = validate(
            train_rows,
            valid_rows,
            numeric_assays,
            categorical_assays,
            standalone_label_k,
            model_selections,
            min_feature_support,
        )

        task_output = output_root / task
        task_output.mkdir(parents=True, exist_ok=True)
        for name, value in (
            ("label_k_cv_metrics.json", label_cv),
            ("label_k_selection.json", label_selection),
            ("model_cv_metrics.json", model_cv),
            ("model_selections.json", model_selections),
            ("valid_metrics.json", valid_metrics),
        ):
            _write_json(task_output / name, value)
        _write_json(
            task_output / "manifest.json",
            {
                "version": VERSION,
                "task": task,
                "selection_order": [
                    "standalone_direct_label_k",
                    "assay_k_weighting_label_mode_and_model_hyperparameters",
                ],
                "selection_split": "train_repeated_scaffold_oof",
                "evaluation_split": "valid",
                "n_train": len(train_rows),
                "n_valid": len(valid_rows),
                "k_values": list(k_values),
                "c_values": list(c_values),
                "seeds": list(seeds),
                "folds_per_seed": 5,
                "selected_standalone_label_k": standalone_label_k,
                "numeric_aggregation": "observed_only_mean",
                "categorical_aggregation": "all_neighbor_one_hot_mean",
                "weighting_candidates": list(WEIGHTINGS),
                "weighting_tied_between_numeric_and_categorical": True,
                "label_mode_candidates": list(LABEL_MODES),
                "label_vote_weighting": "unweighted",
                "numeric_filter": "train_standard_deviation_gt_0",
                "min_feature_support": min_feature_support,
                "lr_presence_indicators": False,
                "rf_numeric_coverage_indicators": True,
                "n_numeric_features": len(numeric_features),
                "n_categorical_features": len(categorical_features),
                "numeric_assay_loading": numeric_stats,
                "categorical_assay_loading": categorical_stats,
                "source_manifest": _file_entry(input_root / task / "manifest.json"),
                "test_split_read": False,
            },
        )
        all_label_cv.extend({"task": task, **row} for row in label_cv)
        all_label_selections.append({"task": task, **label_selection})
        all_model_cv.extend({"task": task, **row} for row in model_cv)
        all_model_selections.extend({"task": task, **row} for row in model_selections)
        all_valid.extend({"task": task, **row} for row in valid_metrics)

    write_tsv(
        output_root / "feature_counts.tsv",
        feature_counts,
        ["task", "n_numeric_features", "n_categorical_features", "n_train"],
    )
    write_tsv(
        output_root / "label_k_cv_metrics.tsv",
        all_label_cv,
        ["task", "seed", "k", "n", "macro_f1", "accuracy", "auroc"],
    )
    write_tsv(
        output_root / "label_k_selection.tsv",
        all_label_selections,
        ["task", "selected_label_k", "mean_oof_macro_f1", "standard_error"],
    )
    selection_columns = [
        "task",
        "surface",
        "model_family",
        "selected_assay_k",
        "selected_weighting",
        "selected_label_mode",
        "selected_label_k",
        "selected_c",
        "selected_rf_profile",
        "selected_rf_max_depth",
        "selected_rf_min_samples_leaf",
        "mean_oof_macro_f1",
        "standard_error",
    ]
    write_tsv(
        output_root / "model_cv_metrics.tsv",
        all_model_cv,
        [
            "task",
            "seed",
            "surface",
            "model_family",
            "assay_k",
            "weighting",
            "label_mode",
            "label_k",
            "c",
            "rf_profile",
            "rf_max_depth",
            "rf_min_samples_leaf",
            "n",
            "macro_f1",
            "accuracy",
            "auroc",
        ],
    )
    write_tsv(
        output_root / "model_selections.tsv", all_model_selections, selection_columns
    )
    write_tsv(
        output_root / "validation_metrics.tsv",
        all_valid,
        selection_columns + ["n_features", "n", "macro_f1", "accuracy", "auroc"],
    )
    write_wide_tables(output_root, all_valid)
    (output_root / "REPORT.md").write_text(
        report(all_valid, min_feature_support), encoding="utf-8"
    )
    output_names = (
        "feature_counts.tsv",
        "label_k_cv_metrics.tsv",
        "label_k_selection.tsv",
        "model_cv_metrics.tsv",
        "model_selections.tsv",
        "validation_metrics.tsv",
        "validation_macro_f1.tsv",
        "selected_assay_k.tsv",
        "selected_label_k.tsv",
        "selected_weighting.tsv",
        "REPORT.md",
    )
    _write_json(
        output_root / "manifest.json",
        {
            "version": VERSION,
            "tasks": tasks,
            "selection_order": [
                "standalone_direct_label_k",
                "assay_k_weighting_label_mode_and_model_hyperparameters",
            ],
            "k_values": list(k_values),
            "c_values": list(c_values),
            "seeds": list(seeds),
            "min_feature_support": min_feature_support,
            "test_split_read": False,
            "outputs": {
                name: _file_entry(output_root / name) for name in output_names
            },
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=list(TASKS))
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--k-values", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--c-values", nargs="+", type=float, default=list(DEFAULT_C_VALUES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--min-feature-support", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(
        args.tasks,
        args.input_root.resolve(),
        args.output_root.resolve(),
        tuple(args.k_values),
        tuple(args.c_values),
        tuple(args.seeds),
        args.min_feature_support,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
