"""Evaluate frozen direction-calibrated post-selector v3.1 on scaffold valid."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic

from .contract import TaskSpec, task_root
from .io import sha256_file as _sha256
from .post_contract import (
    POST_FEATURE_STEM,
    POST_V31_ARTIFACT_DIR,
    POST_V31_SELECTOR_SCHEMA_VERSION,
)
from .post_features import build_post_selector_valid_features
from .post_v31_train import DIRECTIONS, predict_post_selector_v31
from .train import evaluation_metrics, paired_bootstrap_deltas


def evaluate_post_selector_v31_valid(
    spec: TaskSpec,
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    root = task_root(output_root, spec.task)
    artifact_dir = root / POST_V31_ARTIFACT_DIR
    model_path = artifact_dir / "model.joblib"
    manifest_path = artifact_dir / "model_manifest.json"
    train_features_path = root / f"{POST_FEATURE_STEM}.jsonl"
    for path in (model_path, manifest_path, train_features_path):
        if not path.exists():
            raise FileNotFoundError(path)
    model = joblib.load(model_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows, feature_summary = build_post_selector_valid_features(
        spec,
        output_root=output_root,
        output_dir=artifact_dir / "valid",
    )
    labels = np.asarray([int(row["Y"]) for row in rows], dtype=int)
    knn = np.asarray([int(row["knn_prediction"]) for row in rows], dtype=int)
    agent = np.asarray([int(row["agent_prediction"]) for row in rows], dtype=int)
    target = np.asarray(
        [int(row["paired_outcome"] == "agent_only_correct") for row in rows],
        dtype=int,
    )
    probability = predict_post_selector_v31(model, rows, knn, agent)
    candidate_thresholds = model["candidate_thresholds"]
    thresholds = model["thresholds"]
    candidate_route = _route(knn, agent, probability, candidate_thresholds)
    deployed_route = _route(knn, agent, probability, thresholds)
    candidate_prediction = np.where(candidate_route, agent, knn)
    deployed_prediction = np.where(deployed_route, agent, knn)
    candidate_metrics = evaluation_metrics(
        labels, knn, agent, candidate_prediction
    )
    deployed_metrics = evaluation_metrics(labels, knn, agent, deployed_prediction)
    candidate_metrics["router"]["route_count"] = int(candidate_route.sum())
    deployed_metrics["router"]["route_count"] = int(deployed_route.sum())
    candidate_uncertainty = paired_bootstrap_deltas(
        labels,
        knn,
        candidate_prediction,
        seed=20260805,
        repetitions=10000,
    )
    deployed_uncertainty = paired_bootstrap_deltas(
        labels,
        knn,
        deployed_prediction,
        seed=20260805,
        repetitions=10000,
    )
    accuracy_delta = (
        deployed_metrics["router"]["accuracy"]
        - deployed_metrics["always_knn"]["accuracy"]
    )
    macro_f1_delta = (
        deployed_metrics["router"]["macro_f1"]
        - deployed_metrics["always_knn"]["macro_f1"]
    )
    valid_accuracy_evidence_gate = {
        "purpose": "held-out development evidence only; never changes the frozen deployment policy",
        "passed": bool(
            accuracy_delta > 0.0
            and deployed_uncertainty["accuracy_delta_ci95"][0] > 0.0
        ),
        "criteria": {
            "observed_accuracy_delta_gt_0": bool(accuracy_delta > 0.0),
            "accuracy_delta_ci95_lower_gt_0": bool(
                deployed_uncertainty["accuracy_delta_ci95"][0] > 0.0
            ),
        },
    }
    direction_diagnostics = _direction_diagnostics(
        target,
        probability,
        knn,
        agent,
        candidate_route,
    )
    route_01_only = np.where((knn == 0) & (agent == 1), agent, knn)
    baselines = {
        "route_knn0_agent1_only": evaluation_metrics(
            labels, knn, agent, route_01_only
        )["router"]
    }

    output_rows = []
    for index, row in enumerate(rows):
        direction = next(
            (
                name
                for name, pair in DIRECTIONS.items()
                if (int(knn[index]), int(agent[index])) == pair
            ),
            "agreement",
        )
        output_rows.append(
            {
                "schema_version": POST_V31_SELECTOR_SCHEMA_VERSION,
                "task": spec.task,
                "evaluation_subset": "valid",
                "query_index": int(row["query_index"]),
                "Y": int(labels[index]),
                "knn_prediction": int(knn[index]),
                "agent_prediction": int(agent[index]),
                "paired_outcome": row["paired_outcome"],
                "direction": direction,
                "agent_win_probability": float(probability[index]),
                "candidate_route_to_agent": bool(candidate_route[index]),
                "route_to_agent": bool(deployed_route[index]),
                "candidate_selector_prediction": int(candidate_prediction[index]),
                "selector_prediction": int(deployed_prediction[index]),
            }
        )
    valid_dir = artifact_dir / "valid"
    valid_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(valid_dir / "predictions.jsonl", output_rows)
    metrics = {
        "schema_version": f"{POST_V31_SELECTOR_SCHEMA_VERSION}.valid",
        "task": spec.task,
        "evaluation_subset": "valid",
        "n_rows": len(rows),
        "training_policy": "direction-specific task-local nested calibrated ensemble",
        "valid_policy": "development evaluation; no valid fitting, calibration, model selection, or threshold selection",
        # ``promoted`` is retained for compatibility with the first v3.1 receipt.
        # It describes the train-only deployment gate, not held-out validation.
        "promoted": bool(manifest["promoted"]),
        "train_gate_promoted": bool(manifest["promoted"]),
        "valid_accuracy_evidence_gate": valid_accuracy_evidence_gate,
        "directions": manifest["directions"],
        "candidate_thresholds": candidate_thresholds,
        "thresholds": thresholds,
        "feature_summary": feature_summary,
        "metrics": deployed_metrics,
        "candidate_metrics": candidate_metrics,
        "paired_uncertainty_vs_knn": deployed_uncertainty,
        "candidate_paired_uncertainty_vs_knn": candidate_uncertainty,
        "direction_diagnostics": direction_diagnostics,
        "baselines": baselines,
        "delta_vs_knn": {
            "accuracy": accuracy_delta,
            "macro_f1": macro_f1_delta,
        },
    }
    write_json_atomic(valid_dir / "metrics.json", metrics)
    write_json_atomic(
        valid_dir / "manifest.json",
        {
            "schema_version": f"{POST_V31_SELECTOR_SCHEMA_VERSION}.valid_manifest",
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


def _route(
    knn: np.ndarray,
    agent: np.ndarray,
    probability: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    return (
        ((knn == 0) & (agent == 1) & (probability >= float(thresholds["knn0_agent1"])))
        | ((knn == 1) & (agent == 0) & (probability >= float(thresholds["knn1_agent0"])))
    )

def _direction_diagnostics(
    target: np.ndarray,
    probability: np.ndarray,
    knn: np.ndarray,
    agent: np.ndarray,
    route: np.ndarray,
) -> dict[str, Any]:
    output = {}
    for name, pair in DIRECTIONS.items():
        mask = (knn == pair[0]) & (agent == pair[1])
        y = target[mask]
        p = probability[mask]
        selected = route[mask]
        output[name] = {
            "n": int(mask.sum()),
            "agent_win_rate": float(y.mean()) if len(y) else 0.0,
            "mean_probability": float(p.mean()) if len(p) else 0.0,
            "brier": float(np.mean((p - y) ** 2)) if len(y) else 0.0,
            "auc": float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else 0.5,
            "routed_n": int(selected.sum()),
            "routed_precision": float(y[selected].mean()) if selected.any() else 0.0,
        }
    return output


def _render_report(metrics: dict[str, Any]) -> str:
    current = metrics["metrics"]
    return (
        f"# {metrics['task']} post-selector v3.1 scaffold valid\n\n"
        f"- promoted from train-only gate: `{metrics['train_gate_promoted']}`\n"
        f"- held-out valid accuracy evidence gate: `{metrics['valid_accuracy_evidence_gate']['passed']}` "
        "(diagnostic only; does not retune the policy)\n"
        f"- KNN accuracy / macro-F1: {current['always_knn']['accuracy']:.4f} / {current['always_knn']['macro_f1']:.4f}\n"
        f"- selector accuracy / macro-F1: {current['router']['accuracy']:.4f} / {current['router']['macro_f1']:.4f}\n"
        f"- switches / rescues / harms: {current['router']['switch_count']} / {current['router']['rescue_count']} / {current['router']['harm_count']}\n"
    )
