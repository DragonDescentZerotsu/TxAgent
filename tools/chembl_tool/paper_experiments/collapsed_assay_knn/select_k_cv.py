"""Select K by train-only scaffold CV, then evaluate frozen scaffold validation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold

from tools.chembl_tool.common.molecule_identity import bemis_murcko_scaffold
from tools.chembl_tool.paper_experiments.collapsed_assay_knn.model_ablation import (
    fit_predict,
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


CV_VERSION = "collapsed_assay_k_selection_cv.v1"
DEFAULT_INPUT_ROOT = REPO_ROOT / "outputs/paper/collapsed_assay_knn_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/paper/collapsed_assay_k_selection_cv_v1"
DEFAULT_K_VALUES = (3, 5, 10, 15, 25)
MODEL_FAMILIES = ("logistic_l2", "random_forest")
NEIGHBOR_SURFACES = ("neighbor_indirect", "neighbor_indirect_plus_label")


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


def choose_k(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot select K from no metrics")
    return min(rows, key=lambda row: (-row["macro_f1"], row["k"]))


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
) -> list[dict[str, Any]]:
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
            "presence_indicators": surface != "morgan_label_knn",
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
    grouped: dict[tuple[str, str | None, int], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(row["surface"], row["model_family"], row["k"])].append(row)
    metrics = []
    for (surface, model_family, k), condition_rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], str(item[0][1]), item[0][2])
    ):
        condition_rows.sort(key=lambda row: row["query_index"])
        if [row["query_index"] for row in condition_rows] != list(range(n_train)):
            raise AssertionError(f"Incomplete OOF coverage for {task} {surface} {model_family} K={k}")
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
                "presence_indicators": surface != "morgan_label_knn",
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
                "pooled_oof_macro_f1": best["macro_f1"],
                "tie_break": "smallest_k",
            }
        )
    return selections


def _load_neighbor_indices(path: Path, split: str, n_queries: int) -> list[list[int]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = sorted(
        (row for row in rows if row["split"] == split),
        key=lambda row: row["query_index"],
    )
    if [row["query_index"] for row in selected] != list(range(n_queries)):
        raise ValueError(f"Neighbor rows for {split} are not aligned")
    return [[neighbor["train_index"] for neighbor in row["neighbors"]] for row in selected]


def _outer_validation(
    task: str,
    source_dir: Path,
    train_rows: list[dict[str, Any]],
    selections: list[dict[str, Any]],
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
    neighbor_path = source_dir / "neighbors.jsonl"
    train_neighbors = _load_neighbor_indices(neighbor_path, "train", len(train_rows))
    valid_neighbors = _load_neighbor_indices(neighbor_path, "valid", len(valid_rows))
    max_available = min(min(map(len, train_neighbors)), min(map(len, valid_neighbors)))

    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    valid_indices = np.arange(len(valid_rows), dtype=np.int64)
    for model_family in MODEL_FAMILIES:
        predicted, scores, details = fit_predict(
            model_family,
            self_train,
            train_labels,
            self_valid,
            deterministic_rf_scores=True,
        )
        result = binary_metrics(valid_labels, predicted, scores)
        result.update(
            {
                "task": task,
                "surface": "query_self_indirect",
                "model_family": model_family,
                "k": None,
                "presence_indicators": True,
                "n_input_features": self_train.shape[1],
                "model": details,
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
            )
        )

    matrix_cache: dict[int, tuple[sparse.csr_matrix, sparse.csr_matrix]] = {}
    label_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for selection in selections:
        surface = selection["surface"]
        model_family = selection["model_family"]
        k = int(selection["selected_k"])
        if k > max_available:
            raise ValueError(f"Selected K={k} exceeds stored top-{max_available} neighbors")
        if k not in matrix_cache:
            matrix_cache[k] = (
                mean_neighbor_matrix(self_train, train_neighbors, k),
                mean_neighbor_matrix(self_train, valid_neighbors, k),
            )
            label_cache[k] = (
                neighbor_label_scores(train_labels, train_neighbors, k),
                neighbor_label_scores(train_labels, valid_neighbors, k),
            )
        train_matrix, valid_matrix = matrix_cache[k]
        train_label_score, valid_label_score = label_cache[k]
        if surface == "morgan_label_knn":
            scores = valid_label_score
            predicted = (scores >= 0.5).astype(np.int64)
            details = None
            n_input_features = None
        else:
            if surface == "neighbor_indirect_plus_label":
                train_matrix = sparse.hstack(
                    [train_matrix, sparse.csr_matrix(train_label_score[:, None])],
                    format="csr",
                )
                valid_matrix = sparse.hstack(
                    [valid_matrix, sparse.csr_matrix(valid_label_score[:, None])],
                    format="csr",
                )
            predicted, scores, details = fit_predict(
                model_family,
                train_matrix,
                train_labels,
                valid_matrix,
                deterministic_rf_scores=True,
            )
            n_input_features = train_matrix.shape[1]
        result = binary_metrics(valid_labels, predicted, scores)
        result.update(
            {
                "task": task,
                "surface": surface,
                "model_family": model_family,
                "k": k,
                "presence_indicators": surface != "morgan_label_knn",
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
    fingerprints = morgan_fingerprints(train_rows)
    folds, groups = make_folds(train_rows, n_folds=n_folds, seed=seed)
    output_dir.mkdir(parents=True)

    assignment_rows = []
    cv_predictions: list[dict[str, Any]] = []
    max_k = max(k_values)
    for fold, (reference_indices, query_indices) in enumerate(folds):
        reference_rows = [train_rows[int(index)] for index in reference_indices]
        query_rows = [train_rows[int(index)] for index in query_indices]
        catalog, feature_index = build_feature_catalog(reference_rows, assays)
        if not catalog:
            raise ValueError(f"{task} fold {fold} has no train-observed assay features")
        reference_self, _ = encode_assays(reference_rows, assays, catalog, feature_index)
        query_self, _ = encode_assays(query_rows, assays, catalog, feature_index)
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
            }
            for index in query_indices
        )
        for k in k_values:
            reference_mean = mean_neighbor_matrix(reference_self, reference_neighbors, k)
            query_mean = mean_neighbor_matrix(reference_self, query_neighbors, k)
            reference_label_score = neighbor_label_scores(
                reference_labels, reference_neighbors, k
            )
            query_label_score = neighbor_label_scores(reference_labels, query_neighbors, k)
            for surface in NEIGHBOR_SURFACES:
                fit_matrix, heldout_matrix = reference_mean, query_mean
                if surface == "neighbor_indirect_plus_label":
                    fit_matrix = sparse.hstack(
                        [fit_matrix, sparse.csr_matrix(reference_label_score[:, None])],
                        format="csr",
                    )
                    heldout_matrix = sparse.hstack(
                        [heldout_matrix, sparse.csr_matrix(query_label_score[:, None])],
                        format="csr",
                    )
                for model_family in MODEL_FAMILIES:
                    predicted, scores, _ = fit_predict(
                        model_family,
                        fit_matrix,
                        reference_labels,
                        heldout_matrix,
                        deterministic_rf_scores=True,
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
    _write_jsonl(output_dir / "fold_assignments.jsonl", assignment_rows)
    _write_jsonl(output_dir / "cv_predictions.jsonl", cv_predictions)
    _write_json(output_dir / "cv_metrics.json", cv_metrics)
    _write_json(output_dir / "selections.json", selections)

    valid_metrics, valid_predictions, n_valid = _outer_validation(
        task, source_dir, train_rows, selections
    )
    _write_json(output_dir / "valid_metrics.json", valid_metrics)
    _write_jsonl(output_dir / "valid_predictions.jsonl", valid_predictions)
    output_names = (
        "fold_assignments.jsonl",
        "cv_predictions.jsonl",
        "cv_metrics.json",
        "selections.json",
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
            "tie_break": "smallest_k",
            "n_train": len(train_rows),
            "n_valid": n_valid,
            "source_manifest": _file_entry(source_manifest_path),
            "outputs": {name: _file_entry(output_dir / name) for name in output_names},
            "validations": {
                "all_train_rows_oof_once": len(assignment_rows) == len(train_rows),
                "zero_fold_group_overlap": True,
                "fold_catalog_fit_on_reference_only": True,
                "all_validation_queries_retained_per_condition": n_valid,
                "presence_indicators_enabled": True,
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
        "n",
        "macro_f1",
        "accuracy",
        "auroc",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _condition_name(surface: str, model: str | None) -> str:
    names = {
        "logistic_l2": "L2",
        "random_forest": "RF",
        None: "",
    }
    surface_name = {
        "neighbor_indirect": "Neighbor indirect",
        "neighbor_indirect_plus_label": "Indirect + labels",
        "morgan_label_knn": "Direct-label KNN",
    }[surface]
    return f"{surface_name} {names[model]}".strip()


def _report(
    cv_metrics: list[dict[str, Any]],
    valid_metrics: list[dict[str, Any]],
    k_values: tuple[int, ...],
) -> str:
    tasks = [task for task in TASKS if any(row["task"] == task for row in cv_metrics)]
    cv_lookup = {
        (row["task"], row["surface"], row["model_family"], row["k"]): row["macro_f1"]
        for row in cv_metrics
    }
    selected = {}
    for task in tasks:
        task_rows = [row for row in cv_metrics if row["task"] == task]
        grouped: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
        for row in task_rows:
            grouped[(row["surface"], row["model_family"])].append(row)
        for key, rows in grouped.items():
            selected[(task, *key)] = choose_k(rows)["k"]
    lines = [
        "# Four-fold collapsed-assay K selection",
        "",
        "K is selected independently per condition using pooled train-only scaffold-OOF macro-F1. All learned conditions include assay-presence features. Exact ties choose the smaller K.",
        "",
        "## Train OOF selection",
        "",
        "| Task | Condition | " + " | ".join(f"K={k}" for k in k_values) + " | Selected K |",
        "|---|---|" + "---:|" * len(k_values) + "---:|",
    ]
    condition_keys = [
        ("neighbor_indirect", "logistic_l2"),
        ("neighbor_indirect", "random_forest"),
        ("neighbor_indirect_plus_label", "logistic_l2"),
        ("neighbor_indirect_plus_label", "random_forest"),
        ("morgan_label_knn", None),
    ]
    for task in tasks:
        for surface, model in condition_keys:
            values = [cv_lookup[(task, surface, model, k)] for k in k_values]
            lines.append(
                f"| {task} | {_condition_name(surface, model)} | "
                + " | ".join(f"{value:.4f}" for value in values)
                + f" | {selected[(task, surface, model)]} |"
            )
    valid_lookup = {
        (row["task"], row["surface"], row["model_family"]): row
        for row in valid_metrics
    }
    lines.extend(
        [
            "",
            "## Outer scaffold validation",
            "",
            "| Task | Query self L2 | Query self RF | Neighbor indirect L2 | Neighbor indirect RF | Indirect + labels L2 | Indirect + labels RF | Direct-label KNN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    columns = [
        ("query_self_indirect", "logistic_l2"),
        ("query_self_indirect", "random_forest"),
        ("neighbor_indirect", "logistic_l2"),
        ("neighbor_indirect", "random_forest"),
        ("neighbor_indirect_plus_label", "logistic_l2"),
        ("neighbor_indirect_plus_label", "random_forest"),
        ("morgan_label_knn", None),
    ]
    for task in tasks:
        cells = []
        for surface, model in columns:
            row = valid_lookup[(task, surface, model)]
            suffix = "" if row["k"] is None else f" (K={row['k']})"
            cells.append(f"{row['macro_f1']:.4f}{suffix}")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "The outer validation split was opened only after train-OOF selections were written. No test split was read, and these validation results do not authorize test evaluation.",
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
    (output_dir / "REPORT.md").write_text(
        _report(all_cv_metrics, all_valid_metrics, k_values), encoding="utf-8"
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
            "tie_break": "smallest_k",
            "evaluation_split": "valid",
            "test_split_read": False,
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
    parser.add_argument("--folds", type=int, default=4)
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
