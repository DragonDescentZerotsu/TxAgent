"""Select K by train-only scaffold CV, then evaluate frozen scaffold validation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold

from tools.chembl_tool.common.molecule_identity import (
    bemis_murcko_scaffold,
    normalize_molecule_identity,
)
from tools.chembl_tool.paper_experiments.collapsed_assay_knn.model_ablation import (
    fit_predict,
    remove_presence,
)
from tools.chembl_tool.paper_experiments.collapsed_assay_knn.run import (
    RANDOM_SEED,
    REPO_ROOT,
    TASKS,
    _file_entry,
    _write_json,
    _write_jsonl,
    binary_metrics,
    build_feature_catalog,
    encode_assays,
    load_numeric_assays,
    mean_neighbor_matrix,
    morgan_fingerprints,
    neighbor_label_scores,
    read_split,
    retrieve_neighbors,
)


CV_VERSION = "collapsed_assay_k_selection_cv.v4"
DEFAULT_INPUT_ROOT = REPO_ROOT / "outputs/paper/collapsed_assay_knn_stage06_v6_20260827"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/paper/collapsed_assay_k_selection_cv_v4"
DEFAULT_PREVIOUS_METRICS = (
    REPO_ROOT / "outputs/paper/collapsed_assay_k_selection_cv_v3/valid_metrics.tsv"
)
DEFAULT_K_VALUES = (3, 5, 10, 15, 25, 40)
MODEL_FAMILIES = ("logistic_l2", "random_forest")
NEIGHBOR_SURFACES = (
    "neighbor_indirect",
    "neighbor_indirect_plus_label",
    "neighbor_indirect_plus_categorical",
)
RF_GRID = (
    {"max_depth": None, "min_samples_leaf": 1},
    {"max_depth": None, "min_samples_leaf": 5},
    {"max_depth": 20, "min_samples_leaf": 1},
    {"max_depth": 20, "min_samples_leaf": 5},
)
VALIDATION_COLUMNS = (
    ("query_self_l2", "query_self_indirect", "logistic_l2"),
    ("query_self_rf", "query_self_indirect", "random_forest"),
    ("neighbor_indirect_l2", "neighbor_indirect", "logistic_l2"),
    ("neighbor_indirect_rf", "neighbor_indirect", "random_forest"),
    ("indirect_plus_labels_l2", "neighbor_indirect_plus_label", "logistic_l2"),
    ("indirect_plus_labels_rf", "neighbor_indirect_plus_label", "random_forest"),
    (
        "indirect_plus_categorical_l2",
        "neighbor_indirect_plus_categorical",
        "logistic_l2",
    ),
    (
        "indirect_plus_categorical_rf",
        "neighbor_indirect_plus_categorical",
        "random_forest",
    ),
    ("direct_label_knn", "morgan_label_knn", None),
)


def scaffold_group(row: dict[str, Any]) -> str:
    scaffold = bemis_murcko_scaffold(row["drug"])
    return f"scaffold:{scaffold}" if scaffold else f"acyclic_parent:{row['parent_key']}"


def make_folds(
    rows: list[dict[str, Any]], *, n_folds: int, seed: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str]]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    labels = np.asarray([row["Y"] for row in rows], dtype=np.int64)
    groups = [scaffold_group(row) for row in rows]
    group_array = np.asarray(groups, dtype=object)
    splitter = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=seed
    )
    folds = list(splitter.split(np.zeros(len(rows)), labels, group_array))
    heldout = np.concatenate([indices for _, indices in folds])
    if sorted(heldout.tolist()) != list(range(len(rows))):
        raise AssertionError("OOF folds must cover every train row exactly once")
    for reference, query in folds:
        if set(group_array[reference]) & set(group_array[query]):
            raise AssertionError("Scaffold leakage detected in OOF folds")
        if len(set(labels[query].tolist())) != 2:
            raise ValueError("Every OOF fold must contain both labels")
    return folds, groups


def load_categorical_assays(
    path: Path, benchmark_parent_keys: set[str]
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    required = {
        "canonical_smiles",
        "pair_bucket_key",
        "canonical_category_id",
        "aggregation_method",
        "aggregation_status",
        "assay_transfer_eligible",
        "retrieval_source_id",
    }
    dataset = ds.dataset(path, format="parquet")
    missing = required - set(dataset.schema.names)
    if missing:
        raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
    predicate = (
        (ds.field("aggregation_method") == "categorical_mode")
        & (ds.field("aggregation_status") == "valid")
        & (ds.field("assay_transfer_eligible") == True)  # noqa: E712
        & ds.field("pair_bucket_key").is_valid()
        & ds.field("canonical_category_id").is_valid()
    )
    table = dataset.to_table(
        columns=["canonical_smiles", "pair_bucket_key", "canonical_category_id"],
        filter=predicate,
    )
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    identity_cache: dict[str, str] = {}
    for row in table.to_pylist():
        smiles = str(row["canonical_smiles"] or "")
        if smiles not in identity_cache:
            identity = normalize_molecule_identity(smiles)
            identity_cache[smiles] = identity.parent_inchi_key or identity.parent_smiles
        parent_key = identity_cache[smiles]
        if parent_key in benchmark_parent_keys:
            grouped[(parent_key, str(row["pair_bucket_key"]))][
                str(row["canonical_category_id"])
            ] += 1

    assays: dict[str, dict[str, str]] = defaultdict(dict)
    tied_cells = 0
    collisions = 0
    for (parent_key, assay_key), counts in grouped.items():
        collisions += sum(counts.values()) - 1
        largest = max(counts.values())
        winners = sorted(category for category, count in counts.items() if count == largest)
        if len(winners) != 1:
            tied_cells += 1
            continue
        assays[parent_key][assay_key] = winners[0]
    retained_cells = sum(len(values) for values in assays.values())
    return dict(assays), {
        "eligible_rows_read": table.num_rows,
        "unique_stage06_structures_normalized": len(identity_cache),
        "matched_parent_assay_cells_before_tie_filter": len(grouped),
        "retained_parent_assay_categories": retained_cells,
        "parent_assay_collisions_resolved_by_mode": collisions,
        "exact_mode_ties_dropped": tied_cells,
        "matched_benchmark_parents": len(assays),
    }


def build_categorical_catalog(
    train_rows: list[dict[str, Any]], assays: dict[str, dict[str, str]]
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int]]:
    support: Counter[tuple[str, str]] = Counter()
    for row in train_rows:
        support.update(assays.get(row["parent_key"], {}).items())
    catalog = [
        {
            "feature_index": index,
            "pair_bucket_key": assay_key,
            "canonical_category_id": category,
            "train_support": support[(assay_key, category)],
        }
        for index, (assay_key, category) in enumerate(sorted(support))
    ]
    return catalog, {
        (row["pair_bucket_key"], row["canonical_category_id"]): row["feature_index"]
        for row in catalog
    }


def encode_categorical_assays(
    rows: list[dict[str, Any]],
    assays: dict[str, dict[str, str]],
    catalog: list[dict[str, Any]],
    feature_index: dict[tuple[str, str], int],
) -> tuple[sparse.csr_matrix, list[dict[str, int]]]:
    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    coverage: list[dict[str, int]] = []
    for row_index, row in enumerate(rows):
        known = assays.get(row["parent_key"], {})
        used = 0
        for assay_key, category in known.items():
            column = feature_index.get((assay_key, category))
            if column is None:
                continue
            matrix_rows.append(row_index)
            matrix_columns.append(column)
            used += 1
        coverage.append(
            {
                "observed_train_catalog_categories": used,
                "ignored_unseen_categories": len(known) - used,
            }
        )
    matrix = sparse.csr_matrix(
        (np.ones(len(matrix_rows)), (matrix_rows, matrix_columns)),
        shape=(len(rows), len(catalog)),
        dtype=np.float64,
    )
    return matrix, coverage


def choose_k(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot select K from no metrics")
    return min(
        rows,
        key=lambda row: (
            -row["macro_f1"],
            -1 if row["k"] is None else row["k"],
            -int(row.get("rf_min_samples_leaf") or 0),
            row.get("rf_max_depth") is None,
        ),
    )


def _prediction_rows(
    task: str,
    split: str,
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    surface: str,
    model_family: str | None,
    k: int | None,
    predictions: np.ndarray,
    scores: np.ndarray,
    *,
    fold: int | None,
    rf_params: dict[str, int | None] | None = None,
) -> list[dict[str, Any]]:
    presence_indicators = model_family == "random_forest"
    return [
        {
            "task": task,
            "split": split,
            "query_index": int(indices[offset]),
            "Y": int(rows[int(indices[offset])]["Y"]),
            "fold": fold,
            "surface": surface,
            "model_family": model_family,
            "k": k,
            "presence_indicators": presence_indicators,
            "rf_max_depth": None if rf_params is None else rf_params["max_depth"],
            "rf_min_samples_leaf": (
                None if rf_params is None else rf_params["min_samples_leaf"]
            ),
            "prediction": int(predictions[offset]),
            "score": float(scores[offset]),
            "correct": int(predictions[offset]) == int(rows[int(indices[offset])]["Y"]),
        }
        for offset in range(len(indices))
    ]


def _pooled_oof_metrics(
    task: str,
    predictions: list[dict[str, Any]],
    n_train: int,
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str | None, int | None, int | None, int | None],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in predictions:
        grouped[
            (
                row["surface"],
                row["model_family"],
                row["k"],
                row["rf_max_depth"],
                row["rf_min_samples_leaf"],
            )
        ].append(row)
    metrics = []
    for (surface, model_family, k, max_depth, min_samples_leaf), condition_rows in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            str(item[0][1]),
            -1 if item[0][2] is None else item[0][2],
            -1 if item[0][3] is None else item[0][3],
            -1 if item[0][4] is None else item[0][4],
        ),
    ):
        condition_rows.sort(key=lambda row: row["query_index"])
        if [row["query_index"] for row in condition_rows] != list(range(n_train)):
            raise AssertionError(
                f"Incomplete OOF coverage for {task} {surface} {model_family} K={k}"
            )
        labels = np.asarray([row["Y"] for row in condition_rows], dtype=np.int64)
        predicted = np.asarray([row["prediction"] for row in condition_rows], dtype=np.int64)
        scores = np.asarray([row["score"] for row in condition_rows], dtype=np.float64)
        result = binary_metrics(labels, predicted, scores)
        result.update(
            {
                "task": task,
                "surface": surface,
                "model_family": model_family,
                "k": k,
                "presence_indicators": model_family == "random_forest",
                "rf_max_depth": max_depth,
                "rf_min_samples_leaf": min_samples_leaf,
                "selection_metric": "pooled_oof_macro_f1",
            }
        )
        metrics.append(result)
    return metrics


def _select_conditions(task: str, metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        grouped[(row["surface"], row["model_family"])].append(row)
    selections = []
    for (surface, model_family), candidates in sorted(
        grouped.items(), key=lambda item: (item[0][0], str(item[0][1]))
    ):
        best = choose_k(candidates)
        selections.append(
            {
                "task": task,
                "surface": surface,
                "model_family": model_family,
                "selected_k": best["k"],
                "selected_rf_max_depth": best["rf_max_depth"],
                "selected_rf_min_samples_leaf": best["rf_min_samples_leaf"],
                "pooled_oof_macro_f1": best["macro_f1"],
                "tie_break": "smallest_k_then_larger_leaf_then_finite_depth",
            }
        )
    return selections


def _outer_validation(
    task: str,
    source_dir: Path,
    train_rows: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    categorical_assays: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    all_rows = [
        json.loads(line)
        for line in (source_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stored_train = sorted(
        (row for row in all_rows if row["split"] == "train"), key=lambda row: row["index"]
    )
    valid_rows = sorted(
        (row for row in all_rows if row["split"] == "valid"), key=lambda row: row["index"]
    )
    if [(row["parent_key"], row["Y"]) for row in stored_train] != [
        (row["parent_key"], row["Y"]) for row in train_rows
    ]:
        raise ValueError(f"{task} stored train rows do not match the frozen train split")
    train_labels = np.asarray([row["Y"] for row in train_rows], dtype=np.int64)
    valid_labels = np.asarray([row["Y"] for row in valid_rows], dtype=np.int64)
    self_train = sparse.load_npz(source_dir / "query_self_train.npz").tocsr()
    self_valid = sparse.load_npz(source_dir / "query_self_valid.npz").tocsr()
    if self_train.shape[1] % 2:
        raise ValueError(f"{task} numerical matrix does not contain value+presence pairs")
    n_assays = self_train.shape[1] // 2
    categorical_catalog, categorical_index = build_categorical_catalog(
        stored_train, categorical_assays
    )
    categorical_train, _ = encode_categorical_assays(
        stored_train, categorical_assays, categorical_catalog, categorical_index
    )
    categorical_valid, _ = encode_categorical_assays(
        valid_rows, categorical_assays, categorical_catalog, categorical_index
    )
    top_k = max(
        int(row["selected_k"])
        for row in selections
        if row["selected_k"] is not None
    )
    train_fingerprints = morgan_fingerprints(stored_train)
    valid_fingerprints = morgan_fingerprints(valid_rows)
    train_neighbors, _ = retrieve_neighbors(
        stored_train,
        train_fingerprints,
        stored_train,
        train_fingerprints,
        top_k=top_k,
        leave_one_out=True,
    )
    valid_neighbors, _ = retrieve_neighbors(
        valid_rows,
        valid_fingerprints,
        stored_train,
        train_fingerprints,
        top_k=top_k,
        leave_one_out=False,
    )

    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    valid_indices = np.arange(len(valid_rows), dtype=np.int64)
    for model_family in MODEL_FAMILIES:
        selection = next(
            (
                row
                for row in selections
                if row["surface"] == "query_self_indirect"
                and row["model_family"] == model_family
            ),
            None,
        )
        rf_params = (
            {
                "max_depth": selection["selected_rf_max_depth"],
                "min_samples_leaf": selection["selected_rf_min_samples_leaf"],
            }
            if model_family == "random_forest" and selection is not None
            else None
        )
        train_matrix = (
            remove_presence(self_train, n_assays)
            if model_family == "logistic_l2"
            else self_train
        )
        valid_matrix = (
            remove_presence(self_valid, n_assays)
            if model_family == "logistic_l2"
            else self_valid
        )
        predicted, scores, details = fit_predict(
            model_family,
            train_matrix,
            train_labels,
            valid_matrix,
            deterministic_rf_scores=True,
            rf_params=rf_params,
        )
        result = binary_metrics(valid_labels, predicted, scores)
        result.update(
            {
                "task": task,
                "surface": "query_self_indirect",
                "model_family": model_family,
                "k": None,
                "presence_indicators": model_family == "random_forest",
                "rf_max_depth": None if rf_params is None else rf_params["max_depth"],
                "rf_min_samples_leaf": (
                    None if rf_params is None else rf_params["min_samples_leaf"]
                ),
                "n_input_features": train_matrix.shape[1],
                "model": details,
                "selection_oof_macro_f1": (
                    None if selection is None else selection["pooled_oof_macro_f1"]
                ),
            }
        )
        metrics.append(result)
        predictions.extend(
            _prediction_rows(
                task,
                "valid",
                valid_rows,
                valid_indices,
                "query_self_indirect",
                model_family,
                None,
                predicted,
                scores,
                fold=None,
                rf_params=rf_params,
            )
        )

    matrix_cache: dict[int, tuple[sparse.csr_matrix, sparse.csr_matrix]] = {}
    categorical_cache: dict[int, tuple[sparse.csr_matrix, sparse.csr_matrix]] = {}
    label_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for selection in selections:
        surface = selection["surface"]
        model_family = selection["model_family"]
        if surface == "query_self_indirect":
            continue
        k = int(selection["selected_k"])
        if k not in matrix_cache:
            matrix_cache[k] = (
                mean_neighbor_matrix(self_train, train_neighbors, k),
                mean_neighbor_matrix(self_train, valid_neighbors, k),
            )
            label_cache[k] = (
                neighbor_label_scores(train_labels, train_neighbors, k),
                neighbor_label_scores(train_labels, valid_neighbors, k),
            )
            categorical_cache[k] = (
                mean_neighbor_matrix(categorical_train, train_neighbors, k),
                mean_neighbor_matrix(categorical_train, valid_neighbors, k),
            )
        numeric_train, numeric_valid = matrix_cache[k]
        train_label_score, valid_label_score = label_cache[k]
        rf_params = (
            {
                "max_depth": selection["selected_rf_max_depth"],
                "min_samples_leaf": selection["selected_rf_min_samples_leaf"],
            }
            if model_family == "random_forest"
            else None
        )
        if surface == "morgan_label_knn":
            scores = valid_label_score
            predicted = (scores >= 0.5).astype(np.int64)
            details = None
            n_input_features = None
        else:
            train_matrix = (
                remove_presence(numeric_train, n_assays)
                if model_family == "logistic_l2"
                else numeric_train
            )
            valid_matrix = (
                remove_presence(numeric_valid, n_assays)
                if model_family == "logistic_l2"
                else numeric_valid
            )
            if surface == "neighbor_indirect_plus_label":
                train_matrix = sparse.hstack(
                    [train_matrix, sparse.csr_matrix(train_label_score[:, None])],
                    format="csr",
                )
                valid_matrix = sparse.hstack(
                    [valid_matrix, sparse.csr_matrix(valid_label_score[:, None])],
                    format="csr",
                )
            elif surface == "neighbor_indirect_plus_categorical":
                categorical_train_mean, categorical_valid_mean = categorical_cache[k]
                train_matrix = sparse.hstack(
                    [train_matrix, categorical_train_mean], format="csr"
                )
                valid_matrix = sparse.hstack(
                    [valid_matrix, categorical_valid_mean], format="csr"
                )
            predicted, scores, details = fit_predict(
                model_family,
                train_matrix,
                train_labels,
                valid_matrix,
                deterministic_rf_scores=True,
                rf_params=rf_params,
            )
            n_input_features = train_matrix.shape[1]
        result = binary_metrics(valid_labels, predicted, scores)
        result.update(
            {
                "task": task,
                "surface": surface,
                "model_family": model_family,
                "k": k,
                "presence_indicators": model_family == "random_forest",
                "rf_max_depth": None if rf_params is None else rf_params["max_depth"],
                "rf_min_samples_leaf": (
                    None if rf_params is None else rf_params["min_samples_leaf"]
                ),
                "n_input_features": n_input_features,
                "model": details,
                "selection_oof_macro_f1": selection["pooled_oof_macro_f1"],
            }
        )
        metrics.append(result)
        predictions.extend(
            _prediction_rows(
                task,
                "valid",
                valid_rows,
                valid_indices,
                surface,
                model_family,
                k,
                predicted,
                scores,
                fold=None,
                rf_params=rf_params,
            )
        )
    return metrics, predictions, len(valid_rows)


def run_task(
    task: str,
    input_root: Path,
    output_dir: Path,
    *,
    k_values: tuple[int, ...],
    n_folds: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_dir = input_root / task
    source_manifest_path = source_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    train_path = Path(source_manifest["inputs"]["train_jsonl"]["path"])
    records_path = Path(source_manifest["inputs"]["stage06_records"]["path"])
    for name, path in (("train_jsonl", train_path), ("stage06_records", records_path)):
        if _file_entry(path)["sha256"] != source_manifest["inputs"][name]["sha256"]:
            raise ValueError(f"{task} source hash changed for {name}")
    train_rows = read_split(train_path)
    train_labels = np.asarray([row["Y"] for row in train_rows], dtype=np.int64)
    assays, _ = load_numeric_assays(records_path, {row["parent_key"] for row in train_rows})
    source_rows = [
        json.loads(line)
        for line in (source_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    categorical_assays, categorical_stats = load_categorical_assays(
        records_path, {row["parent_key"] for row in source_rows}
    )
    fingerprints = morgan_fingerprints(train_rows)
    folds, groups = make_folds(train_rows, n_folds=n_folds, seed=seed)
    output_dir.mkdir(parents=True)

    assignment_rows = []
    cv_predictions: list[dict[str, Any]] = []
    categorical_catalogs: list[dict[str, Any]] = []
    categorical_fold_audit: list[dict[str, Any]] = []
    max_k = max(k_values)
    for fold, (reference_indices, query_indices) in enumerate(folds):
        reference_rows = [train_rows[int(index)] for index in reference_indices]
        query_rows = [train_rows[int(index)] for index in query_indices]
        catalog, feature_index = build_feature_catalog(reference_rows, assays)
        if not catalog:
            raise ValueError(f"{task} fold {fold} has no train-observed assay features")
        reference_self, _ = encode_assays(reference_rows, assays, catalog, feature_index)
        query_self, _ = encode_assays(query_rows, assays, catalog, feature_index)
        categorical_catalog, categorical_index = build_categorical_catalog(
            reference_rows, categorical_assays
        )
        if not categorical_catalog:
            raise ValueError(f"{task} fold {fold} has no categorical assay features")
        categorical_reference, categorical_reference_coverage = encode_categorical_assays(
            reference_rows,
            categorical_assays,
            categorical_catalog,
            categorical_index,
        )
        categorical_query, categorical_query_coverage = encode_categorical_assays(
            query_rows,
            categorical_assays,
            categorical_catalog,
            categorical_index,
        )
        categorical_catalogs.append({"fold": fold, "features": categorical_catalog})
        categorical_fold_audit.append(
            {
                "fold": fold,
                "n_features": len(categorical_catalog),
                "reference_observed_cells": sum(
                    row["observed_train_catalog_categories"]
                    for row in categorical_reference_coverage
                ),
                "query_observed_cells": sum(
                    row["observed_train_catalog_categories"]
                    for row in categorical_query_coverage
                ),
                "query_unseen_cells_ignored": sum(
                    row["ignored_unseen_categories"] for row in categorical_query_coverage
                ),
            }
        )
        reference_fingerprints = [fingerprints[int(index)] for index in reference_indices]
        query_fingerprints = [fingerprints[int(index)] for index in query_indices]
        reference_neighbors, _ = retrieve_neighbors(
            reference_rows,
            reference_fingerprints,
            reference_rows,
            reference_fingerprints,
            top_k=max_k,
            leave_one_out=True,
        )
        query_neighbors, _ = retrieve_neighbors(
            query_rows,
            query_fingerprints,
            reference_rows,
            reference_fingerprints,
            top_k=max_k,
            leave_one_out=False,
        )
        reference_labels = train_labels[reference_indices]
        query_labels = train_labels[query_indices]
        assignment_rows.extend(
            {
                "task": task,
                "train_index": int(index),
                "fold": fold,
                "Y": int(train_rows[int(index)]["Y"]),
                "parent_key": train_rows[int(index)]["parent_key"],
                "fold_group_key": groups[int(index)],
                "fold_catalog_features": len(catalog),
                "fold_categorical_features": len(categorical_catalog),
            }
            for index in query_indices
        )
        for rf_params in RF_GRID:
            predicted, scores, _ = fit_predict(
                "random_forest",
                reference_self,
                reference_labels,
                query_self,
                deterministic_rf_scores=True,
                rf_params=rf_params,
            )
            cv_predictions.extend(
                _prediction_rows(
                    task,
                    "train_oof",
                    train_rows,
                    query_indices,
                    "query_self_indirect",
                    "random_forest",
                    None,
                    predicted,
                    scores,
                    fold=fold,
                    rf_params=rf_params,
                )
            )
        for k in k_values:
            reference_mean = mean_neighbor_matrix(reference_self, reference_neighbors, k)
            query_mean = mean_neighbor_matrix(reference_self, query_neighbors, k)
            categorical_reference_mean = mean_neighbor_matrix(
                categorical_reference, reference_neighbors, k
            )
            categorical_query_mean = mean_neighbor_matrix(
                categorical_reference, query_neighbors, k
            )
            reference_label_score = neighbor_label_scores(
                reference_labels, reference_neighbors, k
            )
            query_label_score = neighbor_label_scores(reference_labels, query_neighbors, k)
            for surface in NEIGHBOR_SURFACES:
                for model_family in MODEL_FAMILIES:
                    fit_matrix = (
                        remove_presence(reference_mean, len(catalog))
                        if model_family == "logistic_l2"
                        else reference_mean
                    )
                    heldout_matrix = (
                        remove_presence(query_mean, len(catalog))
                        if model_family == "logistic_l2"
                        else query_mean
                    )
                    if surface == "neighbor_indirect_plus_label":
                        fit_matrix = sparse.hstack(
                            [fit_matrix, sparse.csr_matrix(reference_label_score[:, None])],
                            format="csr",
                        )
                        heldout_matrix = sparse.hstack(
                            [heldout_matrix, sparse.csr_matrix(query_label_score[:, None])],
                            format="csr",
                        )
                    elif surface == "neighbor_indirect_plus_categorical":
                        fit_matrix = sparse.hstack(
                            [fit_matrix, categorical_reference_mean], format="csr"
                        )
                        heldout_matrix = sparse.hstack(
                            [heldout_matrix, categorical_query_mean], format="csr"
                        )
                    rf_profiles = RF_GRID if model_family == "random_forest" else (None,)
                    for rf_params in rf_profiles:
                        predicted, scores, _ = fit_predict(
                            model_family,
                            fit_matrix,
                            reference_labels,
                            heldout_matrix,
                            deterministic_rf_scores=True,
                            rf_params=rf_params,
                        )
                        cv_predictions.extend(
                            _prediction_rows(
                                task,
                                "train_oof",
                                train_rows,
                                query_indices,
                                surface,
                                model_family,
                                k,
                                predicted,
                                scores,
                                fold=fold,
                                rf_params=rf_params,
                            )
                        )
            direct_prediction = (query_label_score >= 0.5).astype(np.int64)
            cv_predictions.extend(
                _prediction_rows(
                    task,
                    "train_oof",
                    train_rows,
                    query_indices,
                    "morgan_label_knn",
                    None,
                    k,
                    direct_prediction,
                    query_label_score,
                    fold=fold,
                )
            )

    assignment_rows.sort(key=lambda row: row["train_index"])
    cv_metrics = _pooled_oof_metrics(task, cv_predictions, len(train_rows))
    selections = _select_conditions(task, cv_metrics)
    full_categorical_catalog, _ = build_categorical_catalog(train_rows, categorical_assays)
    categorical_catalogs.append(
        {"fold": "full_train", "features": full_categorical_catalog}
    )
    _write_jsonl(output_dir / "fold_assignments.jsonl", assignment_rows)
    _write_jsonl(output_dir / "cv_predictions.jsonl", cv_predictions)
    _write_json(output_dir / "cv_metrics.json", cv_metrics)
    _write_json(output_dir / "selections.json", selections)
    _write_json(output_dir / "categorical_catalogs.json", categorical_catalogs)
    _write_json(
        output_dir / "categorical_audit.json",
        {"source": categorical_stats, "folds": categorical_fold_audit},
    )

    valid_metrics, valid_predictions, n_valid = _outer_validation(
        task, source_dir, train_rows, selections, categorical_assays
    )
    _write_json(output_dir / "valid_metrics.json", valid_metrics)
    _write_jsonl(output_dir / "valid_predictions.jsonl", valid_predictions)
    output_names = (
        "fold_assignments.jsonl",
        "cv_predictions.jsonl",
        "cv_metrics.json",
        "selections.json",
        "categorical_catalogs.json",
        "categorical_audit.json",
        "valid_metrics.json",
        "valid_predictions.jsonl",
    )
    _write_json(
        output_dir / "manifest.json",
        {
            "version": CV_VERSION,
            "task": task,
            "evaluation_split": "valid",
            "selection_split": "train_oof",
            "n_folds": n_folds,
            "fold_policy": "stratified_scaffold_or_acyclic_parent",
            "seed": seed,
            "k_values": list(k_values),
            "selection_metric": "pooled_oof_macro_f1",
            "tie_break": "smallest_k_then_larger_leaf_then_finite_depth",
            "n_train": len(train_rows),
            "n_valid": n_valid,
            "source_manifest": _file_entry(source_manifest_path),
            "outputs": {name: _file_entry(output_dir / name) for name in output_names},
            "validations": {
                "all_train_rows_oof_once": len(assignment_rows) == len(train_rows),
                "zero_fold_group_overlap": True,
                "fold_catalog_fit_on_reference_only": True,
                "categorical_catalog_fit_on_reference_only": True,
                "categorical_exact_mode_ties_dropped": categorical_stats[
                    "exact_mode_ties_dropped"
                ],
                "all_validation_queries_retained_per_condition": n_valid,
                "l2_presence_indicators_enabled": False,
                "random_forest_presence_indicators_enabled": True,
                "test_split_read": False,
            },
        },
    )
    return cv_metrics, valid_metrics


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "task",
        "surface",
        "model_family",
        "k",
        "presence_indicators",
        "rf_max_depth",
        "rf_min_samples_leaf",
        "n_input_features",
        "n",
        "macro_f1",
        "accuracy",
        "auroc",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_wide_tables(output_dir: Path, metrics: list[dict[str, Any]]) -> None:
    lookup = {
        (row["task"], row["surface"], row["model_family"]): row for row in metrics
    }
    tasks = [task for task in TASKS if any(row["task"] == task for row in metrics)]
    score_rows = []
    k_rows = []
    for task in tasks:
        score_row: dict[str, Any] = {"task": task}
        k_row: dict[str, Any] = {"task": task}
        for name, surface, model in VALIDATION_COLUMNS:
            row = lookup[(task, surface, model)]
            score_row[name] = f"{row['macro_f1']:.4f}"
            if row["k"] is not None:
                k_row[name] = row["k"]
        score_rows.append(score_row)
        k_rows.append(k_row)
    for path, rows in (
        (output_dir / "validation_macro_f1.tsv", score_rows),
        (output_dir / "selected_k.tsv", k_rows),
    ):
        rows = [
            {
                key.replace("_", " ").replace("l2", "LR").replace("rf", "RF"): value
                for key, value in row.items()
            }
            for row in rows
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)


def _condition_name(surface: str, model: str | None) -> str:
    names = {
        "logistic_l2": "L2",
        "random_forest": "RF",
        None: "",
    }
    surface_name = {
        "query_self_indirect": "Query self",
        "neighbor_indirect": "Neighbor indirect",
        "neighbor_indirect_plus_label": "Indirect + labels",
        "neighbor_indirect_plus_categorical": "Indirect + indirect categorical",
        "morgan_label_knn": "Direct-label KNN",
    }[surface]
    return f"{surface_name} {names[model]}".strip()


def _format_selection(row: dict[str, Any]) -> str:
    parts = [] if row.get("k") is None else [f"K={row['k']}"]
    if row.get("model_family") == "random_forest":
        depth = row.get("rf_max_depth")
        parts.extend(
            [
                f"depth={depth if depth is not None else 'None'}",
                f"leaf={row['rf_min_samples_leaf']}",
            ]
        )
    return ", ".join(parts) or "fixed"


def _read_metrics_tsv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            row["model_family"] = row["model_family"] or None
            row["k"] = int(row["k"]) if row["k"] else None
            row["macro_f1"] = float(row["macro_f1"])
            row["accuracy"] = float(row["accuracy"])
            rows.append(row)
    return rows


def _report(
    cv_metrics: list[dict[str, Any]],
    valid_metrics: list[dict[str, Any]],
    k_values: tuple[int, ...],
    n_folds: int,
    previous_metrics: list[dict[str, Any]],
) -> str:
    tasks = [task for task in TASKS if any(row["task"] == task for row in cv_metrics)]
    grouped: dict[tuple[str, str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for row in cv_metrics:
        grouped[(row["task"], row["surface"], row["model_family"])].append(row)
    selected = {key: choose_k(rows) for key, rows in grouped.items()}
    lines = [
        f"# {n_folds}-fold collapsed-assay model selection",
        "",
        "Pooled train-only scaffold-OOF macro-F1 selects K and, for RF, max depth "
        "and minimum leaf size. L2 uses numerical values without presence indicators; "
        "RF retains numerical presence indicators. Exact ties choose smaller K, larger "
        "leaves, then finite depth.",
        "",
        "## Train OOF selection",
        "",
        "For RF, each K cell reports the best of the four RF profiles at that fixed K; "
        "the complete grid is in `cv_metrics.tsv`.",
        "",
        "| Task | Condition | " + " | ".join(f"K={k}" for k in k_values) + " | Selected |",
        "|---|---|" + "---:|" * len(k_values) + "---|",
    ]
    condition_keys = [
        ("neighbor_indirect", "logistic_l2"),
        ("neighbor_indirect", "random_forest"),
        ("neighbor_indirect_plus_label", "logistic_l2"),
        ("neighbor_indirect_plus_label", "random_forest"),
        ("neighbor_indirect_plus_categorical", "logistic_l2"),
        ("neighbor_indirect_plus_categorical", "random_forest"),
        ("morgan_label_knn", None),
    ]
    for task in tasks:
        for surface, model in condition_keys:
            candidates = grouped[(task, surface, model)]
            values = [
                choose_k([row for row in candidates if row["k"] == k])["macro_f1"]
                for k in k_values
            ]
            best = selected[(task, surface, model)]
            lines.append(
                f"| {task} | {_condition_name(surface, model)} | "
                + " | ".join(f"{value:.4f}" for value in values)
                + f" | {_format_selection(best)} |"
            )

    lines.extend(
        [
            "",
            "### Query-self RF profile",
            "",
            "| Task | Selected profile | OOF macro-F1 |",
            "|---|---|---:|",
        ]
    )
    for task in tasks:
        row = selected[(task, "query_self_indirect", "random_forest")]
        lines.append(f"| {task} | {_format_selection(row)} | {row['macro_f1']:.4f} |")

    valid_lookup = {
        (row["task"], row["surface"], row["model_family"]): row
        for row in valid_metrics
    }
    columns = [(surface, model) for _, surface, model in VALIDATION_COLUMNS]
    lines.extend(
        [
            "",
            "## Outer scaffold validation",
            "",
            "Each cell is macro-F1 / accuracy. Selected K values are reported separately.",
            "",
            "| Task | Query self L2 | Query self RF | Neighbor indirect L2 | "
            "Neighbor indirect RF | Indirect + labels L2 | Indirect + labels RF | "
            "Indirect + categorical L2 | Indirect + categorical RF | Direct-label KNN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for task in tasks:
        cells = []
        for surface, model in columns:
            row = valid_lookup[(task, surface, model)]
            cells.append(f"{row['macro_f1']:.4f} / {row['accuracy']:.4f}")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "### Selected K",
            "",
            "| Task | Neighbor indirect L2 | Neighbor indirect RF | "
            "Indirect + labels L2 | Indirect + labels RF | Indirect + categorical L2 | "
            "Indirect + categorical RF | Direct-label KNN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for task in tasks:
        values = [
            valid_lookup[(task, surface, model)]["k"]
            for _, surface, model in VALIDATION_COLUMNS
            if surface != "query_self_indirect"
        ]
        lines.append(f"| {task} | " + " | ".join(str(value) for value in values) + " |")

    previous_lookup = {
        (row["task"], row["surface"], row["model_family"]): row
        for row in previous_metrics
    }
    shared = [
        column for column in columns if column[0] != "neighbor_indirect_plus_categorical"
    ]
    if previous_lookup:
        lines.extend(
            [
                "",
                "## Change from v3",
                "",
                "These deltas combine the requested protocol changes: ten to five folds, "
                "no L2 presence columns, and RF profile tuning.",
                "",
                "| Task | Condition | v3 macro-F1 | v4 macro-F1 | Delta |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for task in tasks:
            for surface, model in shared:
                old = previous_lookup[(task, surface, model)]
                new = valid_lookup[(task, surface, model)]
                lines.append(
                    f"| {task} | {_condition_name(surface, model)} | "
                    f"{old['macro_f1']:.4f} | {new['macro_f1']:.4f} | "
                    f"{new['macro_f1'] - old['macro_f1']:+.4f} |"
                )
    lines.extend(
        [
            "",
            "The outer validation split was opened only after train-OOF selections were "
            "written. No test split was read, and these validation results do not "
            "authorize test evaluation.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(
    input_root: Path,
    output_dir: Path,
    tasks: list[str],
    *,
    k_values: tuple[int, ...],
    n_folds: int,
    seed: int,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if not k_values or min(k_values) < 1 or len(k_values) != len(set(k_values)):
        raise ValueError("K values must be unique positive integers")
    all_cv_metrics: list[dict[str, Any]] = []
    all_valid_metrics: list[dict[str, Any]] = []
    for task in tasks:
        print(f"Running {task}", flush=True)
        cv_metrics, valid_metrics = run_task(
            task,
            input_root,
            output_dir / task,
            k_values=k_values,
            n_folds=n_folds,
            seed=seed,
        )
        all_cv_metrics.extend(cv_metrics)
        all_valid_metrics.extend(valid_metrics)
    _write_tsv(output_dir / "cv_metrics.tsv", all_cv_metrics)
    _write_tsv(output_dir / "valid_metrics.tsv", all_valid_metrics)
    _write_wide_tables(output_dir, all_valid_metrics)
    previous_metrics = _read_metrics_tsv(DEFAULT_PREVIOUS_METRICS)
    (output_dir / "REPORT.md").write_text(
        _report(
            all_cv_metrics,
            all_valid_metrics,
            k_values,
            n_folds,
            previous_metrics,
        ),
        encoding="utf-8",
    )
    _write_json(
        output_dir / "manifest.json",
        {
            "version": CV_VERSION,
            "tasks": tasks,
            "k_values": list(k_values),
            "n_folds": n_folds,
            "fold_policy": "stratified_scaffold_or_acyclic_parent",
            "seed": seed,
            "selection_metric": "pooled_oof_macro_f1",
            "tie_break": "smallest_k_then_larger_leaf_then_finite_depth",
            "models": {
                "logistic_l2": {
                    "C": 1.0,
                    "class_weight": "balanced",
                    "solver": "liblinear",
                    "numerical_presence_indicators": False,
                },
                "random_forest": {
                    "n_estimators": 100,
                    "class_weight": "balanced",
                    "max_features": "sqrt",
                    "max_depth_grid": [None, 20],
                    "min_samples_leaf_grid": [1, 5],
                    "numerical_presence_indicators": True,
                },
            },
            "categorical_encoding": "pair_bucket_key_x_canonical_category_id_one_hot",
            "evaluation_split": "valid",
            "test_split_read": False,
            "previous_validation_metrics": (
                _file_entry(DEFAULT_PREVIOUS_METRICS)
                if DEFAULT_PREVIOUS_METRICS.exists()
                else None
            ),
            "summary_outputs": {
                name: _file_entry(output_dir / name)
                for name in (
                    "cv_metrics.tsv",
                    "valid_metrics.tsv",
                    "validation_macro_f1.tsv",
                    "selected_k.tsv",
                    "REPORT.md",
                )
            },
            "task_manifests": {
                task: _file_entry(output_dir / task / "manifest.json") for task in tasks
            },
        },
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    parser.add_argument("--k-values", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run(
        args.input_root.resolve(),
        args.output_dir.resolve(),
        args.tasks,
        k_values=tuple(args.k_values),
        n_folds=args.folds,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
