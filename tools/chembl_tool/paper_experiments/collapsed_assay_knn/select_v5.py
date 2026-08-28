"""Select simple collapsed-assay LR features with repeated scaffold CV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from tools.chembl_tool.paper_experiments.collapsed_assay_knn.run import (
    RANDOM_SEED,
    REPO_ROOT,
    TASKS,
    _file_entry,
    _write_json,
    binary_metrics,
    build_feature_catalog,
    encode_assays,
    load_numeric_assays,
    morgan_fingerprints,
    neighbor_label_scores,
    read_split,
    retrieve_neighbors,
)
from tools.chembl_tool.paper_experiments.collapsed_assay_knn.select_k_cv import (
    build_categorical_catalog,
    encode_categorical_assays,
    load_categorical_assays,
    make_folds,
)


VERSION = "collapsed_assay_feature_selection.v5.1"
DEFAULT_INPUT_ROOT = REPO_ROOT / "outputs/paper/collapsed_assay_knn_stage06_v6_20260827"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/paper/collapsed_assay_feature_selection_v5_1"
DEFAULT_K_VALUES = (3, 5, 10, 25, 40)
DEFAULT_C_VALUES = (0.1, 1.0, 10.0)
DEFAULT_SEEDS = (0, 1, 2)
LABEL_K = 3
MIN_CATEGORY_SUPPORT = 2


def neighbor_selector(
    neighbor_indices: list[list[int]],
    k: int,
    similarities: list[list[float]] | None = None,
    n_reference: int | None = None,
) -> sparse.csr_matrix:
    """Return a sparse query-by-reference selector for the first K neighbors."""
    rows = np.repeat(np.arange(len(neighbor_indices)), k)
    columns = np.asarray(
        [index for neighbors in neighbor_indices for index in neighbors[:k]],
        dtype=np.int64,
    )
    if similarities is None:
        values = np.ones(columns.size, dtype=np.float64)
    else:
        values = np.asarray(
            [value for row in similarities for value in row[:k]], dtype=np.float64
        )
    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(neighbor_indices), n_reference or max(columns) + 1),
    )


def aggregate_numeric(
    reference_matrix: sparse.csr_matrix,
    selector: sparse.csr_matrix,
    n_assays: int,
    *,
    observed_only: bool,
    include_presence: bool = False,
) -> sparse.csr_matrix:
    """Aggregate standardized assay values, optionally over observed neighbors only."""
    values = (selector @ reference_matrix[:, :n_assays]).tocsr()
    counts = (selector @ reference_matrix[:, n_assays:]).tocsr()
    if observed_only:
        inverse = counts.copy()
        inverse.data = 1.0 / inverse.data
        aggregated = values.multiply(inverse).tocsr()
    else:
        aggregated = (values / selector.sum(axis=1)).tocsr()
    if not include_presence:
        return aggregated
    totals = np.asarray(selector.sum(axis=1)).ravel()
    inverse_totals = np.zeros_like(totals)
    inverse_totals[totals > 0] = 1.0 / totals[totals > 0]
    coverage = (sparse.diags(inverse_totals) @ counts).tocsr()
    return sparse.hstack([aggregated, coverage], format="csr")


def aggregate_categorical(
    reference_matrix: sparse.csr_matrix,
    selector: sparse.csr_matrix,
    category_assays: np.ndarray,
    n_assays: int,
) -> sparse.csr_matrix:
    """Encode retained-category fractions within each assay."""
    totals = (selector @ reference_matrix).tocoo()
    mapping = sparse.csr_matrix(
        (
            np.ones(len(category_assays)),
            (np.arange(len(category_assays)), category_assays),
        ),
        shape=(len(category_assays), n_assays),
    )
    assay_totals = (selector @ reference_matrix @ mapping).tocsr()
    denominators = np.asarray(
        assay_totals[totals.row, category_assays[totals.col]]
    ).ravel()
    keep = denominators > 0
    return sparse.csr_matrix(
        (
            totals.data[keep] / denominators[keep],
            (totals.row[keep], totals.col[keep]),
        ),
        shape=totals.shape,
    )


def supported_categorical_catalog(
    rows: list[dict[str, Any]],
    assays: dict[str, dict[str, str]],
    min_support: int = MIN_CATEGORY_SUPPORT,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int], np.ndarray, int]:
    catalog, _ = build_categorical_catalog(rows, assays)
    retained = [row for row in catalog if row["train_support"] >= min_support]
    assay_index = {key: index for index, key in enumerate(sorted({
        row["pair_bucket_key"] for row in retained
    }))}
    for index, row in enumerate(retained):
        row["feature_index"] = index
    feature_index = {
        (row["pair_bucket_key"], row["canonical_category_id"]): row["feature_index"]
        for row in retained
    }
    category_assays = np.asarray(
        [assay_index[row["pair_bucket_key"]] for row in retained], dtype=np.int64
    )
    return retained, feature_index, category_assays, len(assay_index)


def numeric_catalog(
    rows: list[dict[str, Any]],
    assays: dict[str, dict[str, float]],
    min_support: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    catalog, _ = build_feature_catalog(rows, assays)
    retained = [
        row
        for row in catalog
        if row["train_standard_deviation"] > 0
        and row["train_support"] >= min_support
    ]
    for index, row in enumerate(retained):
        row.update(
            feature_index=index,
            value_column=index,
            presence_column=len(retained) + index,
        )
    return retained, {row["pair_bucket_key"]: row["feature_index"] for row in retained}


def fit_lr(
    train_matrix: sparse.csr_matrix,
    labels: np.ndarray,
    query_matrix: sparse.csr_matrix,
    c_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    model = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2_000,
        random_state=RANDOM_SEED,
    )
    model.fit(train_matrix, labels)
    scores = model.predict_proba(query_matrix)[:, 1]
    return (scores >= 0.5).astype(np.int64), scores


def similarities(audit: list[dict[str, Any]]) -> list[list[float]]:
    return [[float(row["similarity"]) for row in item["neighbors"]] for item in audit]


def matrices_for_k(
    numeric_reference: sparse.csr_matrix,
    categorical_reference: sparse.csr_matrix,
    category_assays: np.ndarray,
    n_category_assays: int,
    train_neighbors: list[list[int]],
    query_neighbors: list[list[int]],
    train_similarities: list[list[float]],
    query_similarities: list[list[float]],
    reference_labels: np.ndarray,
    n_numeric_assays: int,
    k: int,
    *,
    include_presence: bool = False,
) -> dict[str, tuple[sparse.csr_matrix, sparse.csr_matrix]]:
    n_reference = numeric_reference.shape[0]
    plain_train = neighbor_selector(train_neighbors, k, n_reference=n_reference)
    plain_query = neighbor_selector(query_neighbors, k, n_reference=n_reference)
    weighted_train = neighbor_selector(
        train_neighbors, k, train_similarities, n_reference
    )
    weighted_query = neighbor_selector(
        query_neighbors, k, query_similarities, n_reference
    )

    numeric_mean_train = aggregate_numeric(
        numeric_reference,
        plain_train,
        n_numeric_assays,
        observed_only=False,
        include_presence=include_presence,
    )
    numeric_mean_query = aggregate_numeric(
        numeric_reference,
        plain_query,
        n_numeric_assays,
        observed_only=False,
        include_presence=include_presence,
    )
    numeric_observed_train = aggregate_numeric(
        numeric_reference,
        plain_train,
        n_numeric_assays,
        observed_only=True,
        include_presence=include_presence,
    )
    numeric_observed_query = aggregate_numeric(
        numeric_reference,
        plain_query,
        n_numeric_assays,
        observed_only=True,
        include_presence=include_presence,
    )
    numeric_weighted_train = aggregate_numeric(
        numeric_reference,
        weighted_train,
        n_numeric_assays,
        observed_only=True,
        include_presence=include_presence,
    )
    numeric_weighted_query = aggregate_numeric(
        numeric_reference,
        weighted_query,
        n_numeric_assays,
        observed_only=True,
        include_presence=include_presence,
    )
    fixed_label_train = sparse.csr_matrix(
        neighbor_label_scores(reference_labels, train_neighbors, LABEL_K)[:, None]
    )
    fixed_label_query = sparse.csr_matrix(
        neighbor_label_scores(reference_labels, query_neighbors, LABEL_K)[:, None]
    )
    same_label_train = sparse.csr_matrix(
        neighbor_label_scores(reference_labels, train_neighbors, k)[:, None]
    )
    same_label_query = sparse.csr_matrix(
        neighbor_label_scores(reference_labels, query_neighbors, k)[:, None]
    )

    result = {
        "numeric_mean": (numeric_mean_train, numeric_mean_query),
        "numeric_mean_plus_fixed_label": (
            sparse.hstack([numeric_mean_train, fixed_label_train], format="csr"),
            sparse.hstack([numeric_mean_query, fixed_label_query], format="csr"),
        ),
        "numeric_mean_plus_same_k_label": (
            sparse.hstack([numeric_mean_train, same_label_train], format="csr"),
            sparse.hstack([numeric_mean_query, same_label_query], format="csr"),
        ),
        "numeric_observed": (numeric_observed_train, numeric_observed_query),
        "numeric_observed_plus_fixed_label": (
            sparse.hstack([numeric_observed_train, fixed_label_train], format="csr"),
            sparse.hstack([numeric_observed_query, fixed_label_query], format="csr"),
        ),
        "numeric_observed_plus_same_k_label": (
            sparse.hstack([numeric_observed_train, same_label_train], format="csr"),
            sparse.hstack([numeric_observed_query, same_label_query], format="csr"),
        ),
        "numeric_similarity_plus_fixed_label": (
            sparse.hstack([numeric_weighted_train, fixed_label_train], format="csr"),
            sparse.hstack([numeric_weighted_query, fixed_label_query], format="csr"),
        ),
    }
    if categorical_reference.shape[1]:
        categorical_train = aggregate_categorical(
            categorical_reference, weighted_train, category_assays, n_category_assays
        )
        categorical_query = aggregate_categorical(
            categorical_reference, weighted_query, category_assays, n_category_assays
        )
        result["numeric_similarity_categorical_plus_fixed_label"] = (
            sparse.hstack(
                [numeric_weighted_train, categorical_train, fixed_label_train],
                format="csr",
            ),
            sparse.hstack(
                [numeric_weighted_query, categorical_query, fixed_label_query],
                format="csr",
            ),
        )
    return result


def fold_features(
    reference_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    numeric_assays: dict[str, dict[str, float]],
    categorical_assays: dict[str, dict[str, str]],
    max_k: int,
    min_feature_support: int | None = None,
) -> tuple[
    sparse.csr_matrix,
    sparse.csr_matrix,
    np.ndarray,
    int,
    list[list[int]],
    list[list[int]],
    list[list[float]],
    list[list[float]],
]:
    numeric_features, numeric_index = numeric_catalog(
        reference_rows,
        numeric_assays,
        min_support=1 if min_feature_support is None else min_feature_support,
    )
    numeric_reference, _ = encode_assays(
        reference_rows, numeric_assays, numeric_features, numeric_index
    )
    categorical_features, categorical_index, category_assays, n_category_assays = (
        supported_categorical_catalog(
            reference_rows,
            categorical_assays,
            min_support=(
                MIN_CATEGORY_SUPPORT
                if min_feature_support is None
                else min_feature_support
            ),
        )
    )
    categorical_reference, _ = encode_categorical_assays(
        reference_rows,
        categorical_assays,
        categorical_features,
        categorical_index,
    )

    reference_fps = morgan_fingerprints(reference_rows)
    query_fps = morgan_fingerprints(query_rows)
    train_neighbors, train_audit = retrieve_neighbors(
        reference_rows,
        reference_fps,
        reference_rows,
        reference_fps,
        top_k=max_k,
        leave_one_out=True,
    )
    query_neighbors, query_audit = retrieve_neighbors(
        query_rows,
        query_fps,
        reference_rows,
        reference_fps,
        top_k=max_k,
        leave_one_out=False,
    )
    return (
        numeric_reference,
        categorical_reference,
        category_assays,
        n_category_assays,
        train_neighbors,
        query_neighbors,
        similarities(train_audit),
        similarities(query_audit),
    )


def repeated_cv(
    train_rows: list[dict[str, Any]],
    numeric_assays: dict[str, dict[str, float]],
    categorical_assays: dict[str, dict[str, str]],
    k_values: tuple[int, ...],
    c_values: tuple[float, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    all_labels = np.asarray([row["Y"] for row in train_rows], dtype=np.int64)
    metrics: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"  repeated scaffold CV seed {seed}", flush=True)
        folds, _ = make_folds(train_rows, n_folds=5, seed=seed)
        predictions: dict[tuple[str, int, float], np.ndarray] = {}
        scores: dict[tuple[str, int, float], np.ndarray] = {}
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
            reference_labels = all_labels[reference_indices]
            for k in k_values:
                matrices = matrices_for_k(
                    *features,
                    reference_labels,
                    features[0].shape[1] // 2,
                    k,
                )
                for recipe, (fit_matrix, query_matrix) in matrices.items():
                    for c_value in c_values:
                        key = (recipe, k, c_value)
                        predicted, predicted_scores = fit_lr(
                            fit_matrix, reference_labels, query_matrix, c_value
                        )
                        predictions.setdefault(key, np.empty(len(train_rows), dtype=np.int64))[
                            query_indices
                        ] = predicted
                        scores.setdefault(key, np.empty(len(train_rows), dtype=np.float64))[
                            query_indices
                        ] = predicted_scores
        for (recipe, k, c_value), predicted in predictions.items():
            result = binary_metrics(all_labels, predicted, scores[(recipe, k, c_value)])
            metrics.append(
                {
                    "seed": seed,
                    "recipe": recipe,
                    "k": k,
                    "c": c_value,
                    **{name: result[name] for name in ("n", "macro_f1", "accuracy", "auroc")},
                }
            )
    return metrics


def select_configs(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    for row in metrics:
        grouped[(row["recipe"], row["k"], row["c"])].append(row["macro_f1"])
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (recipe, k, c_value), values in grouped.items():
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
                "c": c_value,
                "mean_oof_macro_f1": float(scores.mean()),
                "standard_error": standard_error,
            }
        )

    selected = []
    for recipe, rows in sorted(candidates.items()):
        choice = max(
            rows,
            key=lambda row: (row["mean_oof_macro_f1"], -row["k"], -row["c"]),
        )
        selected.append(choice)
    return selected


def validate(
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    numeric_assays: dict[str, dict[str, float]],
    categorical_assays: dict[str, dict[str, str]],
    selections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    max_k = max(row["k"] for row in selections)
    features = fold_features(
        train_rows, valid_rows, numeric_assays, categorical_assays, max_k
    )
    labels = np.asarray([row["Y"] for row in train_rows], dtype=np.int64)
    valid_labels = np.asarray([row["Y"] for row in valid_rows], dtype=np.int64)
    valid_label_scores = neighbor_label_scores(labels, features[5], LABEL_K)
    results = []
    for selection in selections:
        matrices = matrices_for_k(
            *features,
            labels,
            features[0].shape[1] // 2,
            selection["k"],
        )
        fit_matrix, valid_matrix = matrices[selection["recipe"]]
        predicted, scores = fit_lr(
            fit_matrix, labels, valid_matrix, selection["c"]
        )
        result = binary_metrics(valid_labels, predicted, scores)
        results.append(
            {
                "recipe": selection["recipe"],
                "k": selection["k"],
                "c": selection["c"],
                "n_features": fit_matrix.shape[1],
                **{name: result[name] for name in ("n", "macro_f1", "accuracy", "auroc")},
            }
        )
    knn_predictions = (valid_label_scores >= 0.5).astype(np.int64)
    knn = binary_metrics(valid_labels, knn_predictions, valid_label_scores)
    results.append(
        {
            "recipe": "canonical_label_knn",
            "k": LABEL_K,
            "c": None,
            "n_features": 1,
            **{name: knn[name] for name in ("n", "macro_f1", "accuracy", "auroc")},
        }
    )
    return results


def run_task(
    task: str,
    input_root: Path,
    output_root: Path,
    k_values: tuple[int, ...],
    c_values: tuple[float, ...],
    seeds: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    task_config = TASKS[task]
    train_rows = read_split(task_config["data_dir"] / "train.jsonl")
    valid_rows = read_split(task_config["data_dir"] / "valid.jsonl")
    parent_keys = {row["parent_key"] for row in train_rows + valid_rows}
    numeric_assays, numeric_stats = load_numeric_assays(
        task_config["records"], parent_keys
    )
    categorical_assays, categorical_stats = load_categorical_assays(
        task_config["records"], parent_keys
    )
    cv_metrics = repeated_cv(
        train_rows,
        numeric_assays,
        categorical_assays,
        k_values,
        c_values,
        seeds,
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
            "n_train": len(train_rows),
            "n_valid": len(valid_rows),
            "k_values": list(k_values),
            "c_values": list(c_values),
            "seeds": list(seeds),
            "folds_per_seed": 5,
            "label_k": LABEL_K,
            "category_min_train_support": MIN_CATEGORY_SUPPORT,
            "numeric_assay_loading": numeric_stats,
            "categorical_assay_loading": categorical_stats,
            "source_manifest": _file_entry(input_root / task / "manifest.json"),
            "validations": {
                "fold_catalogs_reference_only": True,
                "scaffold_disjoint_folds": True,
                "direct_label_k_fixed": True,
                "test_split_read": False,
            },
        },
    )
    return cv_metrics, selections, valid_metrics


def write_tsv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows({column: row.get(column) for column in columns} for row in rows)


def report(
    valid_metrics: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    n_repeats: int,
) -> str:
    selected = {(row["task"], row["recipe"]): row for row in selections}
    lines = [
        "# Repeated-CV collapsed-assay feature selection",
        "",
        "Canonical direct-label Morgan KNN remains fixed at K=3. Assay K and LR C are ",
        f"selected independently by mean macro-F1 over {n_repeats} repeated ",
        "five-fold scaffold CV partitions. Test was not read.",
        "",
        "| Task | Recipe | Selected K | C | OOF macro-F1 | Valid macro-F1 | Accuracy |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in valid_metrics:
        selection = selected.get((row["task"], row["recipe"]))
        oof_score = "" if selection is None else f"{selection['mean_oof_macro_f1']:.4f}"
        lines.append(
            f"| {row['task']} | {row['recipe'].replace('_', ' ')} | {row['k']} | "
            f"{'' if row['c'] is None else row['c']} | "
            f"{oof_score} | "
            f"{row['macro_f1']:.4f} | {row['accuracy']:.4f} |"
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
    if LABEL_K > min(k_values):
        raise ValueError("K grid must support the fixed direct-label K")
    output_root.mkdir(parents=True, exist_ok=True)
    all_cv: list[dict[str, Any]] = []
    all_selections: list[dict[str, Any]] = []
    all_valid: list[dict[str, Any]] = []
    for task in tasks:
        print(task, flush=True)
        cv_metrics, selections, valid_metrics = run_task(
            task, input_root, output_root, k_values, c_values, seeds
        )
        all_cv.extend({"task": task, **row} for row in cv_metrics)
        all_selections.extend({"task": task, **row} for row in selections)
        all_valid.extend({"task": task, **row} for row in valid_metrics)
    write_tsv(
        output_root / "cv_metrics.tsv",
        all_cv,
        ["task", "seed", "recipe", "k", "c", "n", "macro_f1", "accuracy", "auroc"],
    )
    write_tsv(
        output_root / "selections.tsv",
        all_selections,
        [
            "task",
            "recipe",
            "k",
            "c",
            "mean_oof_macro_f1",
            "standard_error",
        ],
    )
    write_tsv(
        output_root / "validation_metrics.tsv",
        all_valid,
        ["task", "recipe", "k", "c", "n_features", "n", "macro_f1", "accuracy", "auroc"],
    )
    (output_root / "REPORT.md").write_text(
        report(all_valid, all_selections, len(seeds)), encoding="utf-8"
    )
    _write_json(
        output_root / "manifest.json",
        {
            "version": VERSION,
            "tasks": tasks,
            "k_values": list(k_values),
            "c_values": list(c_values),
            "seeds": list(seeds),
            "label_k": LABEL_K,
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
