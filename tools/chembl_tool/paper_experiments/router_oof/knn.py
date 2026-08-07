"""Strict OOF Morgan KNN predictions and uncertainty/density features."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic

from .contract import TaskSpec, fold_root, task_root
from .io import read_jsonl as _read_jsonl


FP_RADIUS = 2
FP_BITS = 2048


def run_task_oof_knn(
    spec: TaskSpec,
    *,
    output_root: str | Path,
    k: int = 3,
) -> dict[str, Any]:
    if k <= 0:
        raise ValueError("k must be positive")
    assignments_path = task_root(output_root, spec.task) / "folds.jsonl"
    rows = _read_jsonl(assignments_path)
    folds = sorted({int(row["fold"]) for row in rows})
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FP_RADIUS,
        fpSize=FP_BITS,
        includeChirality=False,
        useBondTypes=True,
    )
    fingerprints = {
        int(row["train_index"]): _fingerprint(str(row["drug"]), generator)
        for row in rows
    }
    all_predictions: list[dict[str, Any]] = []
    for fold in folds:
        query_rows = [row for row in rows if int(row["fold"]) == fold]
        reference_rows = [row for row in rows if int(row["fold"]) != fold]
        if len(reference_rows) < k:
            raise ValueError(f"Fold {fold} has fewer than k reference rows")
        reference_fps = [fingerprints[int(row["train_index"])] for row in reference_rows]
        fold_predictions: list[dict[str, Any]] = []
        for local_query_index, query in enumerate(query_rows):
            query_fp = fingerprints[int(query["train_index"])]
            similarities = np.asarray(
                DataStructs.BulkTanimotoSimilarity(query_fp, reference_fps),
                dtype=float,
            )
            candidate_indices = np.argpartition(similarities, -k)[-k:]
            ordered = sorted(
                (int(index) for index in candidate_indices),
                key=lambda index: (
                    -float(similarities[index]),
                    int(reference_rows[index]["train_index"]),
                ),
            )
            neighbors = [
                {
                    "train_index": int(reference_rows[index]["train_index"]),
                    "Y": int(reference_rows[index]["Y"]),
                    "similarity": float(similarities[index]),
                }
                for index in ordered
            ]
            summary = summarize_knn_neighbors(
                neighbors,
                reference_size=len(reference_rows),
            )
            prediction = int(summary["knn_p_positive"] >= 0.5)
            result = {
                "task": spec.task,
                "fold": fold,
                "local_query_index": local_query_index,
                "train_index": int(query["train_index"]),
                "Y": int(query["Y"]),
                "prediction": prediction,
                "correct": prediction == int(query["Y"]),
                **summary,
                "neighbors": neighbors,
            }
            fold_predictions.append(result)
            all_predictions.append(result)
        write_jsonl_atomic(fold_root(output_root, spec.task, fold) / "knn_predictions.jsonl", fold_predictions)

    all_predictions.sort(key=lambda row: int(row["train_index"]))
    if [row["train_index"] for row in all_predictions] != list(range(len(rows))):
        raise AssertionError("OOF KNN predictions do not cover train rows exactly once")
    output_path = task_root(output_root, spec.task) / "oof_knn_predictions.jsonl"
    write_jsonl_atomic(output_path, all_predictions)
    metrics = _metrics(all_predictions)
    metrics.update(
        {
            "task": spec.task,
            "method": f"morgan_oof_knn_k{k}",
            "vote": "unweighted_majority",
            "router_features_include_similarity_weighted_vote": True,
            "k": k,
            "n_folds": len(folds),
            "n_train": len(rows),
            "predictions": str(output_path),
        }
    )
    write_json_atomic(task_root(output_root, spec.task) / "oof_knn_metrics.json", metrics)
    return metrics


def binary_entropy(p: float) -> float:
    p = min(max(float(p), 0.0), 1.0)
    if p in (0.0, 1.0):
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def summarize_knn_neighbors(
    neighbors: list[dict[str, Any]],
    *,
    reference_size: int,
) -> dict[str, float | int]:
    """Build the frozen router KNN features from an ordered neighbor list."""
    if not neighbors:
        raise ValueError("KNN feature summary requires at least one neighbor")
    if reference_size < len(neighbors):
        raise ValueError("KNN reference size cannot be smaller than the neighbor list")
    labels = [int(row["Y"]) for row in neighbors]
    similarities = [float(row["similarity"]) for row in neighbors]
    vote_p = sum(labels) / len(labels)
    weights = [max(value, 0.0) for value in similarities]
    weight_sum = sum(weights)
    weighted_p = (
        sum(weight * label for weight, label in zip(weights, labels)) / weight_sum
        if weight_sum
        else vote_p
    )
    return {
        "knn_p_positive": vote_p,
        "knn_margin": abs(2 * vote_p - 1),
        "knn_label_entropy": binary_entropy(vote_p),
        "knn_weighted_p_positive": weighted_p,
        "knn_weighted_margin": abs(2 * weighted_p - 1),
        "knn_weighted_label_entropy": binary_entropy(weighted_p),
        "knn_similarity_max": max(similarities),
        "knn_similarity_mean": sum(similarities) / len(similarities),
        "knn_similarity_kth": min(similarities),
        "knn_top1_top2_gap": (
            similarities[0] - similarities[1] if len(similarities) > 1 else 0.0
        ),
        "knn_reference_size": int(reference_size),
    }


def _fingerprint(smiles: str, generator: Any):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return generator.GetFingerprint(molecule)


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tn = sum(row["Y"] == 0 and row["prediction"] == 0 for row in rows)
    fp = sum(row["Y"] == 0 and row["prediction"] == 1 for row in rows)
    fn = sum(row["Y"] == 1 and row["prediction"] == 0 for row in rows)
    tp = sum(row["Y"] == 1 and row["prediction"] == 1 for row in rows)
    f1_0 = _safe_div(2 * tn, 2 * tn + fp + fn)
    f1_1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    return {
        "accuracy": _safe_div(tn + tp, len(rows)),
        "macro_f1": (f1_0 + f1_1) / 2,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
