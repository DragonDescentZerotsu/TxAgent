"""Run one observed-only collapsed-assay strategy with sequential K selection."""

from __future__ import annotations

import argparse
import json
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
    morgan_fingerprints,
    neighbor_label_scores,
    read_split,
    retrieve_neighbors,
)
from tools.chembl_tool.paper_experiments.collapsed_assay_knn.select_k_cv import (
    load_categorical_assays,
    make_folds,
)
from tools.chembl_tool.paper_experiments.collapsed_assay_knn.select_v5 import (
    DEFAULT_INPUT_ROOT,
    DEFAULT_K_VALUES,
    DEFAULT_SEEDS,
    aggregate_categorical,
    aggregate_numeric,
    fit_lr,
    fold_features,
    neighbor_selector,
    numeric_catalog,
    write_tsv,
)
from tools.chembl_tool.paper_experiments.collapsed_assay_knn.select_v5_rf import (
    RF_PROFILES,
)


VERSION = "collapsed_assay_unified_observed_only.v6"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/paper/collapsed_assay_unified_observed_only_v6"
DEFAULT_C_VALUES = (0.1, 1.0, 10.0)
SURFACES = (
    "query_self_indirect",
    "neighbor_indirect",
    "neighbor_indirect_plus_label",
    "neighbor_indirect_plus_categorical",
)
WIDE_COLUMNS = (
    ("query self LR", "query_self_indirect", "logistic_l2"),
    ("query self RF", "query_self_indirect", "random_forest"),
    ("neighbor indirect LR", "neighbor_indirect", "logistic_l2"),
    ("neighbor indirect RF", "neighbor_indirect", "random_forest"),
    ("indirect plus labels LR", "neighbor_indirect_plus_label", "logistic_l2"),
    ("indirect plus labels RF", "neighbor_indirect_plus_label", "random_forest"),
    (
        "indirect plus categorical LR",
        "neighbor_indirect_plus_categorical",
        "logistic_l2",
    ),
    (
        "indirect plus categorical RF",
        "neighbor_indirect_plus_categorical",
        "random_forest",
    ),
    ("direct label KNN", "morgan_label_knn", None),
)


def select_label_k(
    train_rows: list[dict[str, Any]],
    k_values: tuple[int, ...],
    seeds: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = np.asarray([row["Y"] for row in train_rows], dtype=np.int64)
    fingerprints = morgan_fingerprints(train_rows)
    metrics = []
    for seed in seeds:
        predictions = {k: np.empty(len(train_rows), dtype=np.int64) for k in k_values}
        scores = {k: np.empty(len(train_rows), dtype=np.float64) for k in k_values}
        folds, _ = make_folds(train_rows, n_folds=5, seed=seed)
        for reference_indices, query_indices in folds:
            reference_rows = [train_rows[index] for index in reference_indices]
            query_rows = [train_rows[index] for index in query_indices]
            reference_fps = [fingerprints[index] for index in reference_indices]
            query_fps = [fingerprints[index] for index in query_indices]
            neighbors, _ = retrieve_neighbors(
                query_rows,
                query_fps,
                reference_rows,
                reference_fps,
                top_k=max(k_values),
                leave_one_out=False,
            )
            reference_labels = labels[reference_indices]
            for k in k_values:
                fold_scores = neighbor_label_scores(reference_labels, neighbors, k)
                scores[k][query_indices] = fold_scores
                predictions[k][query_indices] = (fold_scores >= 0.5).astype(np.int64)
        for k in k_values:
            result = binary_metrics(labels, predictions[k], scores[k])
            metrics.append(
                {
                    "seed": seed,
                    "k": k,
                    **{
                        name: result[name]
                        for name in ("n", "macro_f1", "accuracy", "auroc")
                    },
                }
            )

    grouped: dict[int, list[float]] = defaultdict(list)
    for row in metrics:
        grouped[row["k"]].append(row["macro_f1"])
    candidates = []
    for k, values in grouped.items():
        scores = np.asarray(values)
        candidates.append(
            {
                "selected_label_k": k,
                "mean_oof_macro_f1": float(scores.mean()),
                "standard_error": (
                    float(scores.std(ddof=1) / np.sqrt(len(scores)))
                    if len(scores) > 1
                    else 0.0
                ),
            }
        )
    return metrics, max(
        candidates, key=lambda row: (row["mean_oof_macro_f1"], -row["selected_label_k"])
    )


def surface_matrices(
    features: tuple[Any, ...],
    reference_labels: np.ndarray,
    assay_k: int,
    label_k: int,
) -> dict[str, dict[str, tuple[sparse.csr_matrix, sparse.csr_matrix]]]:
    numeric_reference = features[0]
    n_assays = numeric_reference.shape[1] // 2
    n_reference = numeric_reference.shape[0]
    train_selector = neighbor_selector(
        features[4], assay_k, n_reference=n_reference
    )
    query_selector = neighbor_selector(
        features[5], assay_k, n_reference=n_reference
    )
    numeric_train_rf = aggregate_numeric(
        numeric_reference,
        train_selector,
        n_assays,
        observed_only=True,
        include_presence=True,
    )
    numeric_query_rf = aggregate_numeric(
        numeric_reference,
        query_selector,
        n_assays,
        observed_only=True,
        include_presence=True,
    )
    numeric_train_lr = numeric_train_rf[:, :n_assays]
    numeric_query_lr = numeric_query_rf[:, :n_assays]
    label_train = sparse.csr_matrix(
        neighbor_label_scores(reference_labels, features[4], label_k)[:, None]
    )
    label_query = sparse.csr_matrix(
        neighbor_label_scores(reference_labels, features[5], label_k)[:, None]
    )
    categorical_train = aggregate_categorical(
        features[1], train_selector, features[2], features[3]
    )
    categorical_query = aggregate_categorical(
        features[1], query_selector, features[2], features[3]
    )
    return {
        "logistic_l2": {
            "neighbor_indirect": (numeric_train_lr, numeric_query_lr),
            "neighbor_indirect_plus_label": (
                sparse.hstack([numeric_train_lr, label_train], format="csr"),
                sparse.hstack([numeric_query_lr, label_query], format="csr"),
            ),
            "neighbor_indirect_plus_categorical": (
                sparse.hstack([numeric_train_lr, categorical_train], format="csr"),
                sparse.hstack([numeric_query_lr, categorical_query], format="csr"),
            ),
        },
        "random_forest": {
            "neighbor_indirect": (numeric_train_rf, numeric_query_rf),
            "neighbor_indirect_plus_label": (
                sparse.hstack([numeric_train_rf, label_train], format="csr"),
                sparse.hstack([numeric_query_rf, label_query], format="csr"),
            ),
            "neighbor_indirect_plus_categorical": (
                sparse.hstack([numeric_train_rf, categorical_train], format="csr"),
                sparse.hstack([numeric_query_rf, categorical_query], format="csr"),
            ),
        },
    }


def repeated_model_cv(
    train_rows: list[dict[str, Any]],
    numeric_assays: dict[str, dict[str, float]],
    categorical_assays: dict[str, dict[str, str]],
    label_k: int,
    k_values: tuple[int, ...],
    c_values: tuple[float, ...],
    seeds: tuple[int, ...],
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
            )
            catalog, feature_index = numeric_catalog(reference_rows, numeric_assays)
            self_train, _ = encode_assays(
                reference_rows, numeric_assays, catalog, feature_index
            )
            self_query, _ = encode_assays(
                query_rows, numeric_assays, catalog, feature_index
            )
            n_assays = len(catalog)

            for c_value in c_values:
                key = ("query_self_indirect", "logistic_l2", None, c_value, None)
                predicted, predicted_scores = fit_lr(
                    self_train[:, :n_assays],
                    reference_labels,
                    self_query[:, :n_assays],
                    c_value,
                )
                predictions.setdefault(key, np.empty(len(train_rows), dtype=np.int64))[
                    query_indices
                ] = predicted
                scores.setdefault(key, np.empty(len(train_rows), dtype=np.float64))[
                    query_indices
                ] = predicted_scores
            for profile in RF_PROFILES:
                key = (
                    "query_self_indirect",
                    "random_forest",
                    None,
                    None,
                    profile["profile"],
                )
                predicted, predicted_scores, _ = fit_predict(
                    "random_forest",
                    self_train,
                    reference_labels,
                    self_query,
                    deterministic_rf_scores=True,
                    rf_params=profile,
                )
                predictions.setdefault(key, np.empty(len(train_rows), dtype=np.int64))[
                    query_indices
                ] = predicted
                scores.setdefault(key, np.empty(len(train_rows), dtype=np.float64))[
                    query_indices
                ] = predicted_scores

            for assay_k in k_values:
                matrices = surface_matrices(
                    features, reference_labels, assay_k, label_k
                )
                for surface in SURFACES[1:]:
                    fit_matrix, query_matrix = matrices["logistic_l2"][surface]
                    for c_value in c_values:
                        key = (surface, "logistic_l2", assay_k, c_value, None)
                        predicted, predicted_scores = fit_lr(
                            fit_matrix, reference_labels, query_matrix, c_value
                        )
                        predictions.setdefault(
                            key, np.empty(len(train_rows), dtype=np.int64)
                        )[query_indices] = predicted
                        scores.setdefault(
                            key, np.empty(len(train_rows), dtype=np.float64)
                        )[query_indices] = predicted_scores
                    fit_matrix, query_matrix = matrices["random_forest"][surface]
                    for profile in RF_PROFILES:
                        key = (
                            surface,
                            "random_forest",
                            assay_k,
                            None,
                            profile["profile"],
                        )
                        predicted, predicted_scores, _ = fit_predict(
                            "random_forest",
                            fit_matrix,
                            reference_labels,
                            query_matrix,
                            deterministic_rf_scores=True,
                            rf_params=profile,
                        )
                        predictions.setdefault(
                            key, np.empty(len(train_rows), dtype=np.int64)
                        )[query_indices] = predicted
                        scores.setdefault(
                            key, np.empty(len(train_rows), dtype=np.float64)
                        )[query_indices] = predicted_scores

        for key, predicted in predictions.items():
            surface, model, assay_k, c_value, profile_name = key
            result = binary_metrics(labels, predicted, scores[key])
            profile = next(
                (row for row in RF_PROFILES if row["profile"] == profile_name), None
            )
            metrics.append(
                {
                    "seed": seed,
                    "surface": surface,
                    "model_family": model,
                    "assay_k": assay_k,
                    "label_k": label_k if surface == "neighbor_indirect_plus_label" else None,
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
    provenance: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in metrics:
        key = (
            row["surface"],
            row["model_family"],
            row["assay_k"],
            row["c"],
            row["rf_profile"],
        )
        grouped[key].append(row["macro_f1"])
        provenance[key] = row
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for key, values in grouped.items():
        surface, model, assay_k, c_value, profile_name = key
        source = provenance[key]
        repeat_scores = np.asarray(values)
        candidates[(surface, model)].append(
            {
                "surface": surface,
                "model_family": model,
                "selected_assay_k": assay_k,
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
                    -row["selected_c"],
                ),
            )
        else:
            choice = max(
                rows,
                key=lambda row: (
                    row["mean_oof_macro_f1"],
                    row["selected_rf_min_samples_leaf"],
                    row["selected_rf_max_depth"] is not None,
                    -(row["selected_assay_k"] or 0),
                ),
            )
        selected.append(choice)
    return selected


def validate(
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    numeric_assays: dict[str, dict[str, float]],
    categorical_assays: dict[str, dict[str, str]],
    label_k: int,
    selections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    labels = np.asarray([row["Y"] for row in train_rows], dtype=np.int64)
    valid_labels = np.asarray([row["Y"] for row in valid_rows], dtype=np.int64)
    max_k = max(
        label_k,
        max(row["selected_assay_k"] or 0 for row in selections),
    )
    features = fold_features(
        train_rows, valid_rows, numeric_assays, categorical_assays, max_k
    )
    catalog, feature_index = numeric_catalog(train_rows, numeric_assays)
    self_train, _ = encode_assays(train_rows, numeric_assays, catalog, feature_index)
    self_valid, _ = encode_assays(valid_rows, numeric_assays, catalog, feature_index)
    n_assays = len(catalog)
    matrix_cache = {
        k: surface_matrices(features, labels, k, label_k)
        for k in {
            row["selected_assay_k"] for row in selections if row["selected_assay_k"]
        }
    }
    results = []
    for selection in selections:
        surface = selection["surface"]
        model = selection["model_family"]
        if surface == "query_self_indirect":
            if model == "logistic_l2":
                fit_matrix, valid_matrix = (
                    self_train[:, :n_assays],
                    self_valid[:, :n_assays],
                )
            else:
                fit_matrix, valid_matrix = self_train, self_valid
        else:
            fit_matrix, valid_matrix = matrix_cache[
                selection["selected_assay_k"]
            ][model][surface]
        if model == "logistic_l2":
            predicted, scores = fit_lr(
                fit_matrix,
                labels,
                valid_matrix,
                selection["selected_c"],
            )
        else:
            predicted, scores, _ = fit_predict(
                "random_forest",
                fit_matrix,
                labels,
                valid_matrix,
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
                "n_features": fit_matrix.shape[1],
                **{
                    name: result[name]
                    for name in ("n", "macro_f1", "accuracy", "auroc")
                },
            }
        )

    label_scores = neighbor_label_scores(labels, features[5], label_k)
    label_predictions = (label_scores >= 0.5).astype(np.int64)
    label_result = binary_metrics(valid_labels, label_predictions, label_scores)
    results.append(
        {
            "surface": "morgan_label_knn",
            "model_family": None,
            "selected_assay_k": None,
            "selected_label_k": label_k,
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
    scores = []
    assay_ks = []
    label_ks = []
    for task in sorted({row["task"] for row in valid_metrics}):
        task_rows = [row for row in valid_metrics if row["task"] == task]
        score_row = {"task": task}
        assay_row = {"task": task}
        label_row = {"task": task}
        for column, surface, model in WIDE_COLUMNS:
            row = next(
                item
                for item in task_rows
                if item["surface"] == surface and item["model_family"] == model
            )
            score_row[column] = row["macro_f1"]
            assay_row[column] = row["selected_assay_k"]
            label_row[column] = row["selected_label_k"]
        scores.append(score_row)
        assay_ks.append(assay_row)
        label_ks.append(label_row)
    columns = ["task"] + [column for column, _, _ in WIDE_COLUMNS]
    write_tsv(output_root / "validation_macro_f1.tsv", scores, columns)
    write_tsv(output_root / "selected_assay_k.tsv", assay_ks, columns)
    write_tsv(output_root / "selected_label_k.tsv", label_ks, columns)


def report(valid_metrics: list[dict[str, Any]]) -> str:
    lookup = {
        (row["task"], row["surface"], row["model_family"]): row
        for row in valid_metrics
    }
    lines = [
        "# Unified observed-only collapsed-assay validation",
        "",
        "Direct-label K is selected first from repeated train OOF and then frozen. All ",
        "neighbor numerical assay surfaces use observed-only means. LR is value-only; RF ",
        "adds numerical assay-coverage fractions. Test and ClinTox were not read.",
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
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    all_label_cv = []
    all_label_selections = []
    all_model_cv = []
    all_model_selections = []
    all_valid = []
    for task in tasks:
        print(task, flush=True)
        config = TASKS[task]
        train_rows = read_split(config["data_dir"] / "train.jsonl")
        valid_rows = read_split(config["data_dir"] / "valid.jsonl")
        parent_keys = {row["parent_key"] for row in train_rows + valid_rows}
        numeric_assays, numeric_stats = load_numeric_assays(
            config["records"], parent_keys
        )
        categorical_assays, categorical_stats = load_categorical_assays(
            config["records"], parent_keys
        )
        print("  selecting direct-label K", flush=True)
        label_cv, label_selection = select_label_k(train_rows, k_values, seeds)
        label_k = label_selection["selected_label_k"]
        print(f"  frozen direct-label K={label_k}", flush=True)
        model_cv = repeated_model_cv(
            train_rows,
            numeric_assays,
            categorical_assays,
            label_k,
            k_values,
            c_values,
            seeds,
        )
        model_selections = select_models(model_cv)
        valid_metrics = validate(
            train_rows,
            valid_rows,
            numeric_assays,
            categorical_assays,
            label_k,
            model_selections,
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
                "selection_order": ["direct_label_k", "assay_k_and_model_hyperparameters"],
                "selection_split": "train_repeated_scaffold_oof",
                "evaluation_split": "valid",
                "n_train": len(train_rows),
                "n_valid": len(valid_rows),
                "k_values": list(k_values),
                "c_values": list(c_values),
                "seeds": list(seeds),
                "folds_per_seed": 5,
                "selected_label_k": label_k,
                "numeric_aggregation": "observed_only_mean",
                "lr_presence_indicators": False,
                "rf_presence_indicators": True,
                "categorical_aggregation": "within_assay_retained_category_distribution",
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
    (output_root / "REPORT.md").write_text(report(all_valid), encoding="utf-8")
    output_names = (
        "label_k_cv_metrics.tsv",
        "label_k_selection.tsv",
        "model_cv_metrics.tsv",
        "model_selections.tsv",
        "validation_metrics.tsv",
        "validation_macro_f1.tsv",
        "selected_assay_k.tsv",
        "selected_label_k.tsv",
        "REPORT.md",
    )
    _write_json(
        output_root / "manifest.json",
        {
            "version": VERSION,
            "tasks": tasks,
            "selection_order": ["direct_label_k", "assay_k_and_model_hyperparameters"],
            "k_values": list(k_values),
            "c_values": list(c_values),
            "seeds": list(seeds),
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
