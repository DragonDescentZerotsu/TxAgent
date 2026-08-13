"""Evaluate frozen post-selector v3 models on scaffold valid."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic

from .contract import TaskSpec, task_root
from .io import sha256_file as _sha256
from .post_contract import POST_ARTIFACT_DIR, POST_FEATURE_STEM, POST_SELECTOR_SCHEMA_VERSION
from .post_features import build_post_selector_valid_features
from .post_train import predict_post_selector, route_with_direction_thresholds
from .train import evaluation_metrics, paired_bootstrap_deltas


def evaluate_post_selector_valid(
    spec: TaskSpec,
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    root = task_root(output_root, spec.task)
    artifact_dir = root / POST_ARTIFACT_DIR
    model_path = artifact_dir / "model.joblib"
    manifest_path = artifact_dir / "model_manifest.json"
    train_features_path = root / f"{POST_FEATURE_STEM}.jsonl"
    for path in (model_path, manifest_path, train_features_path):
        if not path.exists():
            raise FileNotFoundError(path)
    model = joblib.load(model_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows, feature_summary = build_post_selector_valid_features(
        spec, output_root=output_root
    )
    if tuple(model["feature_columns"]) != tuple(manifest["feature_columns"]):
        raise ValueError("Post-selector model/manifest feature mismatch")
    probability = predict_post_selector(model, rows)
    labels = np.asarray([int(row["Y"]) for row in rows], dtype=int)
    knn = np.asarray([int(row["knn_prediction"]) for row in rows], dtype=int)
    agent = np.asarray([int(row["agent_prediction"]) for row in rows], dtype=int)
    candidate_thresholds = model["candidate_thresholds"]
    thresholds = model["thresholds"]
    candidate, candidate_route = route_with_direction_thresholds(
        knn,
        agent,
        probability,
        float(candidate_thresholds["knn0_agent1"]),
        float(candidate_thresholds["knn1_agent0"]),
    )
    deployed, deployed_route = route_with_direction_thresholds(
        knn,
        agent,
        probability,
        float(thresholds["knn0_agent1"]),
        float(thresholds["knn1_agent0"]),
    )
    candidate_metrics = evaluation_metrics(labels, knn, agent, candidate)
    deployed_metrics = evaluation_metrics(labels, knn, agent, deployed)
    candidate_metrics["router"]["route_count"] = int(candidate_route.sum())
    deployed_metrics["router"]["route_count"] = int(deployed_route.sum())
    candidate_uncertainty = paired_bootstrap_deltas(
        labels, knn, candidate, seed=20260805, repetitions=10000
    )
    deployed_uncertainty = paired_bootstrap_deltas(
        labels, knn, deployed, seed=20260805, repetitions=10000
    )
    output_rows = []
    for index, row in enumerate(rows):
        output_rows.append(
            {
                "schema_version": POST_SELECTOR_SCHEMA_VERSION,
                "task": spec.task,
                "evaluation_subset": "valid",
                "query_index": int(row["query_index"]),
                "Y": int(labels[index]),
                "knn_prediction": int(knn[index]),
                "agent_prediction": int(agent[index]),
                "paired_outcome": row["paired_outcome"],
                "agent_win_probability": float(probability[index]),
                "candidate_route_to_agent": bool(candidate_route[index]),
                "route_to_agent": bool(deployed_route[index]),
                "candidate_selector_prediction": int(candidate[index]),
                "selector_prediction": int(deployed[index]),
            }
        )
    valid_dir = artifact_dir / "valid"
    valid_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(valid_dir / "predictions.jsonl", output_rows)
    metrics = {
        "schema_version": f"{POST_SELECTOR_SCHEMA_VERSION}.valid",
        "task": spec.task,
        "evaluation_subset": "valid",
        "n_rows": len(rows),
        "training_policy": "disagreement-only task-local nested OOF",
        "valid_policy": "development evaluation; no valid fitting or threshold selection",
        "model_family": manifest["model_family"],
        "feature_profile": manifest["feature_profile"],
        "promoted": bool(manifest["promoted"]),
        "candidate_thresholds": candidate_thresholds,
        "thresholds": thresholds,
        "feature_summary": feature_summary,
        "metrics": deployed_metrics,
        "candidate_metrics": candidate_metrics,
        "paired_uncertainty_vs_knn": deployed_uncertainty,
        "candidate_paired_uncertainty_vs_knn": candidate_uncertainty,
        "delta_vs_knn": {
            "accuracy": deployed_metrics["router"]["accuracy"] - deployed_metrics["always_knn"]["accuracy"],
            "macro_f1": deployed_metrics["router"]["macro_f1"] - deployed_metrics["always_knn"]["macro_f1"],
        },
    }
    write_json_atomic(valid_dir / "metrics.json", metrics)
    write_json_atomic(
        valid_dir / "manifest.json",
        {
            "schema_version": f"{POST_SELECTOR_SCHEMA_VERSION}.valid_manifest",
            "task": spec.task,
            "evaluation_subset": "valid",
            "no_valid_fitting": True,
            "source_artifacts": {
                str(path): _sha256(path)
                for path in (model_path, manifest_path, train_features_path)
            },
            "outputs": {
                "features": str(valid_dir / "features.jsonl"),
                "predictions": str(valid_dir / "predictions.jsonl"),
                "metrics": str(valid_dir / "metrics.json"),
            },
        },
    )
    (valid_dir / "report_zh.md").write_text(_render_report(metrics), encoding="utf-8")
    return metrics


def _render_report(metrics: dict[str, Any]) -> str:
    current = metrics["metrics"]
    return (
        f"# {metrics['task']} post-selector v3 scaffold valid\n\n"
        f"- selected: `{metrics['model_family']} / {metrics['feature_profile']}`\n"
        f"- promoted from train-only gate: `{metrics['promoted']}`\n"
        f"- KNN accuracy / macro-F1: {current['always_knn']['accuracy']:.4f} / {current['always_knn']['macro_f1']:.4f}\n"
        f"- selector accuracy / macro-F1: {current['router']['accuracy']:.4f} / {current['router']['macro_f1']:.4f}\n"
        f"- switches / rescues / harms: {current['router']['switch_count']} / {current['router']['rescue_count']} / {current['router']['harm_count']}\n"
    )
