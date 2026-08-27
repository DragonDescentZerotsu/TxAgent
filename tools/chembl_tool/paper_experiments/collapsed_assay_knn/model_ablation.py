"""Compare L2, L1, and random-forest models on frozen collapsed-assay features."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from tools.chembl_tool.paper_experiments.collapsed_assay_knn.run import (
    DEFAULT_K_VALUES,
    RANDOM_SEED,
    REPO_ROOT,
    TASKS,
    _file_entry,
    _write_json,
    _write_jsonl,
    binary_metrics,
)


ABLATION_VERSION = "collapsed_assay_model_ablation.v1"
DEFAULT_INPUT_ROOT = REPO_ROOT / "outputs/paper/collapsed_assay_knn_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/paper/collapsed_assay_model_ablation_v1"
MODEL_FAMILIES = ("logistic_l2", "logistic_l1", "random_forest")
SURFACES = ("query_self_indirect", "neighbor_indirect", "neighbor_indirect_plus_label")


def remove_presence(matrix: sparse.csr_matrix, n_assays: int) -> sparse.csr_matrix:
    if matrix.shape[1] != 2 * n_assays:
        raise ValueError(
            f"Expected {2 * n_assays} value+presence columns, found {matrix.shape[1]}"
        )
    return matrix[:, :n_assays].tocsr()


def load_neighbor_label_scores(
    path: Path, split: str, labels: np.ndarray, k: int
) -> np.ndarray:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = sorted(
        (row for row in rows if row["split"] == split),
        key=lambda row: row["query_index"],
    )
    return np.asarray(
        [
            np.mean([labels[neighbor["train_index"]] for neighbor in row["neighbors"][:k]])
            for row in selected
        ],
        dtype=np.float64,
    )


def build_model(model_family: str) -> Any:
    if model_family == "logistic_l2":
        return LogisticRegression(
            C=1.0,
            l1_ratio=0.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2_000,
            random_state=RANDOM_SEED,
        )
    if model_family == "logistic_l1":
        return LogisticRegression(
            C=1.0,
            l1_ratio=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2_000,
            random_state=RANDOM_SEED,
        )
    if model_family == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            max_features="sqrt",
            random_state=RANDOM_SEED,
            n_jobs=8,
        )
    raise ValueError(f"Unknown model family: {model_family}")


def fit_predict(
    model_family: str,
    train_matrix: sparse.csr_matrix,
    train_labels: np.ndarray,
    valid_matrix: sparse.csr_matrix,
    *,
    deterministic_rf_scores: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    model = build_model(model_family)
    model.fit(train_matrix, train_labels)
    if model_family == "random_forest" and deterministic_rf_scores:
        model.n_jobs = 1
    scores = model.predict_proba(valid_matrix)[:, 1]
    predictions = (scores >= 0.5).astype(np.int64)
    if model_family.startswith("logistic"):
        details = {
            "n_iter": int(model.n_iter_[0]),
            "nonzero_coefficients": int(np.count_nonzero(model.coef_)),
            "total_coefficients": int(model.coef_.size),
        }
    else:
        details = {
            "n_estimators": len(model.estimators_),
            "nonzero_feature_importances": int(np.count_nonzero(model.feature_importances_)),
        }
    return predictions, scores, details


def run_task(
    task: str,
    input_root: Path,
    output_dir: Path,
    k_values: tuple[int, ...],
) -> list[dict[str, Any]]:
    source_dir = input_root / task
    source_manifest_path = source_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if tuple(source_manifest["k_values"]) != k_values:
        raise ValueError(f"{task} source K values do not match the requested ablation")
    n_assays = int(source_manifest["n_assay_features"])
    rows = [json.loads(line) for line in (source_dir / "rows.jsonl").read_text().splitlines()]
    train_rows = sorted((row for row in rows if row["split"] == "train"), key=lambda r: r["index"])
    valid_rows = sorted((row for row in rows if row["split"] == "valid"), key=lambda r: r["index"])
    train_labels = np.asarray([row["Y"] for row in train_rows], dtype=np.int64)
    valid_labels = np.asarray([row["Y"] for row in valid_rows], dtype=np.int64)
    self_train = sparse.load_npz(source_dir / "query_self_train.npz").tocsr()
    self_valid = sparse.load_npz(source_dir / "query_self_valid.npz").tocsr()
    neighbor_path = source_dir / "neighbors.jsonl"

    metrics: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for surface in SURFACES:
        surface_k_values: tuple[int | None, ...] = (None,) if surface == "query_self_indirect" else k_values
        for k in surface_k_values:
            if k is None:
                base_train, base_valid = self_train, self_valid
                train_label_score = valid_label_score = None
            else:
                base_train = sparse.load_npz(
                    source_dir / f"neighbor_indirect_train_k{k}.npz"
                ).tocsr()
                base_valid = sparse.load_npz(
                    source_dir / f"neighbor_indirect_valid_k{k}.npz"
                ).tocsr()
                train_label_score = load_neighbor_label_scores(
                    neighbor_path, "train", train_labels, k
                )
                valid_label_score = load_neighbor_label_scores(
                    neighbor_path, "valid", train_labels, k
                )
            for presence_indicators in (False, True):
                train_matrix = (
                    base_train if presence_indicators else remove_presence(base_train, n_assays)
                )
                valid_matrix = (
                    base_valid if presence_indicators else remove_presence(base_valid, n_assays)
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
                for model_family in MODEL_FAMILIES:
                    predictions, scores, details = fit_predict(
                        model_family, train_matrix, train_labels, valid_matrix
                    )
                    result = binary_metrics(valid_labels, predictions, scores)
                    result.update(
                        {
                            "task": task,
                            "surface": surface,
                            "k": k,
                            "model_family": model_family,
                            "presence_indicators": presence_indicators,
                            "n_input_features": train_matrix.shape[1],
                            "model": details,
                        }
                    )
                    metrics.append(result)
                    prediction_rows.extend(
                        {
                            "task": task,
                            "query_index": row["index"],
                            "Y": row["Y"],
                            "surface": surface,
                            "k": k,
                            "model_family": model_family,
                            "presence_indicators": presence_indicators,
                            "prediction": int(predictions[index]),
                            "score": float(scores[index]),
                            "correct": int(predictions[index]) == row["Y"],
                        }
                        for index, row in enumerate(valid_rows)
                    )

    output_dir.mkdir(parents=True)
    _write_json(output_dir / "metrics.json", metrics)
    _write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    _write_json(
        output_dir / "manifest.json",
        {
            "version": ABLATION_VERSION,
            "task": task,
            "evaluation_split": "valid",
            "reference_split": "train",
            "k_values": list(k_values),
            "n_train": len(train_rows),
            "n_valid": len(valid_rows),
            "n_assay_features": n_assays,
            "source_manifest": _file_entry(source_manifest_path),
            "outputs": {
                name: _file_entry(output_dir / name)
                for name in ("metrics.json", "predictions.jsonl")
            },
            "validations": {
                "l2_presence_matches_v1": _l2_presence_parity(source_dir, metrics),
                "all_validation_queries_retained_per_condition": len(valid_rows),
                "test_split_read": False,
            },
        },
    )
    return metrics


def _l2_presence_parity(source_dir: Path, metrics: list[dict[str, Any]]) -> bool:
    source = json.loads((source_dir / "metrics.json").read_text(encoding="utf-8"))
    expected = {
        (row["method"], row["k"]): row
        for row in source
        if row["method"] in SURFACES
    }
    observed = {
        (row["surface"], row["k"]): row
        for row in metrics
        if row["model_family"] == "logistic_l2" and row["presence_indicators"]
    }
    for key, reference in expected.items():
        candidate = observed[key]
        for metric in ("macro_f1", "accuracy", "auroc"):
            if not np.isclose(candidate[metric], reference[metric], rtol=0, atol=1e-12):
                raise ValueError(f"L2 presence parity failed for {key} {metric}")
    return True


def run(input_root: Path, output_dir: Path, tasks: list[str]) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    source_manifest_path = input_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    k_values = tuple(int(value) for value in source_manifest["k_values"])
    all_metrics: list[dict[str, Any]] = []
    for task in tasks:
        print(f"Running {task}", flush=True)
        all_metrics.extend(run_task(task, input_root, output_dir / task, k_values))
    _write_metrics_tsv(output_dir / "metrics.tsv", all_metrics)
    (output_dir / "REPORT.md").write_text(_report(all_metrics), encoding="utf-8")
    _write_json(
        output_dir / "manifest.json",
        {
            "version": ABLATION_VERSION,
            "source_experiment": _file_entry(source_manifest_path),
            "tasks": tasks,
            "k_values": list(k_values),
            "evaluation_split": "valid",
            "selection_policy": "report_only_no_model_or_k_selection",
            "models": {
                "logistic_l2": {"C": 1.0, "l1_ratio": 0.0, "class_weight": "balanced"},
                "logistic_l1": {"C": 1.0, "l1_ratio": 1.0, "class_weight": "balanced"},
                "random_forest": {
                    "n_estimators": 100,
                    "class_weight": "balanced",
                    "max_features": "sqrt",
                    "random_seed": RANDOM_SEED,
                },
            },
            "presence_variants": [False, True],
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "task_manifests": {
                task: _file_entry(output_dir / task / "manifest.json") for task in tasks
            },
        },
    )


def _write_metrics_tsv(path: Path, metrics: list[dict[str, Any]]) -> None:
    fields = (
        "task",
        "surface",
        "k",
        "model_family",
        "presence_indicators",
        "n_input_features",
        "n",
        "macro_f1",
        "accuracy",
        "auroc",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)


def _report(metrics: list[dict[str, Any]]) -> str:
    lookup = {
        (
            row["task"],
            row["surface"],
            row["k"],
            row["model_family"],
            row["presence_indicators"],
        ): row["macro_f1"]
        for row in metrics
    }
    lines = [
        "# Collapsed-assay model and presence ablation",
        "",
        "All values are scaffold-validation macro-F1. This report does not select a model or K and does not read test data.",
        "",
        "| Task | Surface | K | L2 values | L2 + presence | L1 values | L1 + presence | RF values | RF + presence |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        for surface in SURFACES:
            k_values: tuple[int | None, ...] = (None,) if surface == "query_self_indirect" else DEFAULT_K_VALUES
            for k in k_values:
                values = [
                    lookup[(task, surface, k, model, presence)]
                    for model in MODEL_FAMILIES
                    for presence in (False, True)
                ]
                lines.append(
                    f"| {task} | {surface} | {k if k is not None else 'n/a'} | "
                    + " | ".join(f"{value:.4f}" for value in values)
                    + " |"
                )
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run(args.input_root.resolve(), args.output_dir.resolve(), args.tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
