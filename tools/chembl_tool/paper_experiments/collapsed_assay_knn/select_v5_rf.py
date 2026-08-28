"""Run a minimal repeated-CV random-forest check on the v5 feature recipes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.chembl_tool.paper_experiments.collapsed_assay_knn.model_ablation import (
    fit_predict,
)
from tools.chembl_tool.paper_experiments.collapsed_assay_knn.run import (
    REPO_ROOT,
    TASKS,
    _file_entry,
    _write_json,
    binary_metrics,
    load_numeric_assays,
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
    fold_features,
    matrices_for_k,
    write_tsv,
)


VERSION = "collapsed_assay_feature_selection_rf.v5.1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/paper/collapsed_assay_feature_selection_rf_v5_1"
RECIPES = (
    "numeric_mean_plus_fixed_label",
    "numeric_mean_plus_same_k_label",
    "numeric_observed",
    "numeric_similarity_categorical_plus_fixed_label",
)
RF_PROFILES = (
    {"profile": "flexible", "max_depth": None, "min_samples_leaf": 1},
    {"profile": "regularized", "max_depth": 20, "min_samples_leaf": 5},
)


def repeated_cv(
    train_rows: list[dict[str, Any]],
    numeric_assays: dict[str, dict[str, float]],
    categorical_assays: dict[str, dict[str, str]],
    k_values: tuple[int, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    labels = np.asarray([row["Y"] for row in train_rows], dtype=np.int64)
    metrics = []
    for seed in seeds:
        print(f"  repeated scaffold CV seed {seed}", flush=True)
        predictions: dict[tuple[str, int, str], np.ndarray] = {}
        scores: dict[tuple[str, int, str], np.ndarray] = {}
        folds, _ = make_folds(train_rows, n_folds=5, seed=seed)
        for reference_indices, query_indices in folds:
            reference_rows = [train_rows[index] for index in reference_indices]
            query_rows = [train_rows[index] for index in query_indices]
            features = fold_features(
                reference_rows,
                query_rows,
                numeric_assays,
                categorical_assays,
                max(k_values),
            )
            reference_labels = labels[reference_indices]
            for k in k_values:
                matrices = matrices_for_k(
                    *features,
                    reference_labels,
                    features[0].shape[1] // 2,
                    k,
                    include_presence=True,
                )
                for recipe in RECIPES:
                    fit_matrix, query_matrix = matrices[recipe]
                    for profile in RF_PROFILES:
                        key = (recipe, k, profile["profile"])
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
        for (recipe, k, profile), predicted in predictions.items():
            result = binary_metrics(labels, predicted, scores[(recipe, k, profile)])
            params = next(row for row in RF_PROFILES if row["profile"] == profile)
            metrics.append(
                {
                    "seed": seed,
                    "recipe": recipe,
                    "k": k,
                    "rf_profile": profile,
                    "rf_max_depth": params["max_depth"],
                    "rf_min_samples_leaf": params["min_samples_leaf"],
                    **{
                        name: result[name]
                        for name in ("n", "macro_f1", "accuracy", "auroc")
                    },
                }
            )
    return metrics


def select_configs(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    params: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in metrics:
        key = (row["recipe"], row["k"], row["rf_profile"])
        grouped[key].append(row["macro_f1"])
        params[key] = row
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, values in grouped.items():
        recipe, k, profile = key
        scores = np.asarray(values)
        standard_error = (
            float(scores.std(ddof=1) / np.sqrt(len(scores)))
            if len(scores) > 1
            else 0.0
        )
        candidates[recipe].append(
            {
                "recipe": recipe,
                "k": k,
                "rf_profile": profile,
                "rf_max_depth": params[key]["rf_max_depth"],
                "rf_min_samples_leaf": params[key]["rf_min_samples_leaf"],
                "mean_oof_macro_f1": float(scores.mean()),
                "standard_error": standard_error,
            }
        )
    return [
        max(
            rows,
            key=lambda row: (
                row["mean_oof_macro_f1"],
                row["rf_min_samples_leaf"],
                row["rf_max_depth"] is not None,
                -row["k"],
            ),
        )
        for _, rows in sorted(candidates.items())
    ]


def validate(
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    numeric_assays: dict[str, dict[str, float]],
    categorical_assays: dict[str, dict[str, str]],
    selections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    labels = np.asarray([row["Y"] for row in train_rows], dtype=np.int64)
    valid_labels = np.asarray([row["Y"] for row in valid_rows], dtype=np.int64)
    features = fold_features(
        train_rows,
        valid_rows,
        numeric_assays,
        categorical_assays,
        max(row["k"] for row in selections),
    )
    results = []
    for selection in selections:
        matrices = matrices_for_k(
            *features,
            labels,
            features[0].shape[1] // 2,
            selection["k"],
            include_presence=True,
        )
        fit_matrix, valid_matrix = matrices[selection["recipe"]]
        predicted, scores, _ = fit_predict(
            "random_forest",
            fit_matrix,
            labels,
            valid_matrix,
            deterministic_rf_scores=True,
            rf_params={
                "max_depth": selection["rf_max_depth"],
                "min_samples_leaf": selection["rf_min_samples_leaf"],
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
    return results


def report(metrics: list[dict[str, Any]]) -> str:
    lines = [
        "# Minimal random-forest feature check",
        "",
        "K and one of two RF profiles were selected by mean macro-F1 over three repeated ",
        "five-fold scaffold CV partitions. RF includes numerical assay-coverage fractions. ",
        "Test was not read.",
        "",
        "| Task | Recipe | K | RF profile | OOF macro-F1 | Valid macro-F1 | Accuracy |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['task']} | {row['recipe'].replace('_', ' ')} | {row['k']} | "
            f"{row['rf_profile']} | {row['mean_oof_macro_f1']:.4f} | "
            f"{row['macro_f1']:.4f} | {row['accuracy']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def run(
    tasks: list[str],
    input_root: Path,
    output_root: Path,
    k_values: tuple[int, ...],
    seeds: tuple[int, ...],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    all_cv = []
    all_selections = []
    all_valid = []
    for task in tasks:
        print(task, flush=True)
        config = TASKS[task]
        train_rows = read_split(config["data_dir"] / "train.jsonl")
        valid_rows = read_split(config["data_dir"] / "valid.jsonl")
        parent_keys = {row["parent_key"] for row in train_rows + valid_rows}
        numeric_assays, _ = load_numeric_assays(config["records"], parent_keys)
        categorical_assays, _ = load_categorical_assays(config["records"], parent_keys)
        cv_metrics = repeated_cv(
            train_rows, numeric_assays, categorical_assays, k_values, seeds
        )
        selections = select_configs(cv_metrics)
        valid_metrics = validate(
            train_rows, valid_rows, numeric_assays, categorical_assays, selections
        )
        task_output = output_root / task
        task_output.mkdir(parents=True, exist_ok=True)
        _write_json(task_output / "cv_metrics.json", cv_metrics)
        _write_json(task_output / "selections.json", selections)
        _write_json(task_output / "valid_metrics.json", valid_metrics)
        _write_json(
            task_output / "manifest.json",
            {
                "version": VERSION,
                "task": task,
                "selection_split": "train_repeated_scaffold_oof",
                "evaluation_split": "valid",
                "k_values": list(k_values),
                "seeds": list(seeds),
                "folds_per_seed": 5,
                "recipes": list(RECIPES),
                "rf_profiles": list(RF_PROFILES),
                "numerical_presence_indicators": True,
                "source_manifest": _file_entry(input_root / task / "manifest.json"),
                "test_split_read": False,
            },
        )
        all_cv.extend({"task": task, **row} for row in cv_metrics)
        all_selections.extend({"task": task, **row} for row in selections)
        all_valid.extend({"task": task, **row} for row in valid_metrics)

    cv_columns = [
        "task",
        "seed",
        "recipe",
        "k",
        "rf_profile",
        "rf_max_depth",
        "rf_min_samples_leaf",
        "n",
        "macro_f1",
        "accuracy",
        "auroc",
    ]
    result_columns = [
        "task",
        "recipe",
        "k",
        "rf_profile",
        "rf_max_depth",
        "rf_min_samples_leaf",
        "mean_oof_macro_f1",
        "standard_error",
    ]
    write_tsv(output_root / "cv_metrics.tsv", all_cv, cv_columns)
    write_tsv(output_root / "selections.tsv", all_selections, result_columns)
    write_tsv(
        output_root / "validation_metrics.tsv",
        all_valid,
        result_columns + ["n_features", "n", "macro_f1", "accuracy", "auroc"],
    )
    (output_root / "REPORT.md").write_text(report(all_valid), encoding="utf-8")
    _write_json(
        output_root / "manifest.json",
        {
            "version": VERSION,
            "tasks": tasks,
            "k_values": list(k_values),
            "seeds": list(seeds),
            "test_split_read": False,
            "outputs": {
                name: _file_entry(output_root / name)
                for name in (
                    "cv_metrics.tsv",
                    "selections.tsv",
                    "validation_metrics.tsv",
                    "REPORT.md",
                )
            },
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=list(TASKS))
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--k-values", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(
        args.tasks,
        args.input_root.resolve(),
        args.output_root.resolve(),
        tuple(args.k_values),
        tuple(args.seeds),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
