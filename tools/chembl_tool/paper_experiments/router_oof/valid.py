"""Evaluate frozen full-train task-local routers on the formal valid split."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic

from .contract import (
    FEATURE_SCHEMA_VERSION,
    ROUTER_ARTIFACT_DIR,
    ROUTER_FEATURE_STEM,
    ROUTER_SCHEMA_VERSION,
    TaskSpec,
    task_root,
)
from .features import FEATURE_COLUMNS, assemble_router_features, paired_outcome
from .knn import summarize_knn_neighbors
from .io import read_jsonl as _read_jsonl, sha256_file as _sha256
from .train import (
    evaluation_metrics,
    paired_bootstrap_deltas,
    predict_deployment_scores,
    route_predictions,
)


DEFAULT_VALID_DATA_ROOT = Path("data/gold_labels/legacy/processed_starling")
DEFAULT_VALID_KNN_ROOT = Path("outputs/baselines/structure_knn_starling_valid")
DEFAULT_VALID_AGENT_ROOT = Path(
    "outputs/paper/"
    "molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1_valid_gpt_oss_120b/"
    "runs_identity_blind_parent_disjoint"
)


def evaluate_task_valid(
    spec: TaskSpec,
    *,
    output_root: str | Path,
    data_root: str | Path = DEFAULT_VALID_DATA_ROOT,
    knn_root: str | Path = DEFAULT_VALID_KNN_ROOT,
    agent_root: str | Path = DEFAULT_VALID_AGENT_ROOT,
) -> dict[str, Any]:
    """Apply the frozen deployment model once; valid labels are metrics-only."""
    output_root = Path(output_root)
    query_path = Path(data_root) / spec.data_name / "scaffold" / "valid.jsonl"
    knn_dir = Path(knn_root) / spec.data_name / "scaffold"
    knn_path = knn_dir / "valid_predictions.jsonl"
    knn_manifest_path = knn_dir / "manifest.json"
    batch_root = Path(agent_root) / spec.task / spec.direct_condition
    agent_path = batch_root / "predictions.jsonl"
    agent_manifest_path = batch_root / "manifest.json"
    agent_metrics_path = batch_root / "metrics.json"
    router_dir = task_root(output_root, spec.task) / ROUTER_ARTIFACT_DIR
    model_path = router_dir / "model.joblib"
    model_manifest_path = router_dir / "model_manifest.json"
    train_features_path = (
        task_root(output_root, spec.task) / f"{ROUTER_FEATURE_STEM}.jsonl"
    )
    required = (
        query_path,
        knn_path,
        knn_manifest_path,
        agent_path,
        agent_manifest_path,
        agent_metrics_path,
        model_path,
        model_manifest_path,
        train_features_path,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    query_rows = _read_jsonl(query_path)
    knn_rows = _read_jsonl(knn_path)
    agent_rows = _read_jsonl(agent_path)
    knn_by_index = {int(row["query_index"]): row for row in knn_rows}
    agent_by_index = {_agent_query_index(row): row for row in agent_rows}
    expected = set(range(len(query_rows)))
    _require_exact_indices("KNN", knn_by_index, expected)
    _require_exact_indices("agent", agent_by_index, expected)

    knn_manifest = _read_json(knn_manifest_path)
    agent_manifest = _read_json(agent_manifest_path)
    agent_metrics = _read_json(agent_metrics_path)
    model_manifest = _read_json(model_manifest_path)
    _validate_contracts(
        spec,
        query_path=query_path,
        n_rows=len(query_rows),
        knn_manifest=knn_manifest,
        agent_manifest=agent_manifest,
        agent_metrics=agent_metrics,
        model_manifest=model_manifest,
        train_features_path=train_features_path,
    )

    feature_rows: list[dict[str, Any]] = []
    for index, query in enumerate(query_rows):
        label = int(query["Y"])
        smiles = str(query["drug"])
        knn = knn_by_index[index]
        agent = agent_by_index[index]
        if int(knn["Y"]) != label or int(agent["label"]) != label:
            raise ValueError(f"Valid label mismatch for {spec.task} index {index}")
        if str(knn["drug"]) != smiles or str(agent["smiles"]) != smiles:
            raise ValueError(f"Valid molecule mismatch for {spec.task} index {index}")
        if str(knn.get("status") or "") != "ok" or str(agent.get("status") or "") != "ok":
            raise ValueError(f"Non-successful valid prediction for {spec.task} index {index}")

        knn_summary = summarize_knn_neighbors(
            list(knn.get("neighbors") or []),
            reference_size=int(knn["n_eligible_neighbors"]),
        )
        knn_prediction = int(knn["prediction"])
        if knn_prediction != int(float(knn_summary["knn_p_positive"]) >= 0.5):
            raise ValueError(f"KNN vote mismatch for {spec.task} index {index}")
        if not math.isclose(
            float(knn["score"]),
            float(knn_summary["knn_p_positive"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"KNN score mismatch for {spec.task} index {index}")

        retrieval_path = Path(str(agent["run_dir"])) / "retrieval.json"
        if not retrieval_path.exists():
            raise FileNotFoundError(retrieval_path)
        retrieval = _read_json(retrieval_path)
        agent_prediction = int(agent["pred_label"])
        outcome = paired_outcome(label, knn_prediction, agent_prediction)
        feature_rows.append(
            {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "task": spec.task,
                "evaluation_subset": "valid",
                "query_index": index,
                "Y": label,
                "knn_prediction": knn_prediction,
                "agent_prediction": agent_prediction,
                "knn_correct": knn_prediction == label,
                "agent_correct": agent_prediction == label,
                "paired_outcome": outcome,
                "agent_preferred": int(outcome == "agent_only_correct"),
                "features": assemble_router_features(smiles, knn_summary, retrieval),
                "provenance": {
                    "query": str(query_path),
                    "knn_prediction": str(knn_path),
                    "agent_prediction": str(agent_path),
                    "retrieval": str(retrieval_path),
                },
            }
        )

    deployment_model = joblib.load(model_path)
    if not math.isclose(
        float(deployment_model["threshold"]),
        float(model_manifest["threshold"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Router model and manifest threshold mismatch")
    if not math.isclose(
        float(deployment_model["candidate_threshold"]),
        float(model_manifest["candidate_threshold"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Router model and manifest candidate threshold mismatch")
    risks = predict_deployment_scores(deployment_model, feature_rows)
    agent_probabilities = risks["agent_only_probability"]
    knn_probabilities = risks["knn_only_probability"]
    scores = risks["score"]
    threshold = float(deployment_model["threshold"])
    candidate_threshold = float(deployment_model["candidate_threshold"])
    labels = np.asarray([int(row["Y"]) for row in feature_rows], dtype=int)
    knn_predictions = np.asarray(
        [int(row["knn_prediction"]) for row in feature_rows], dtype=int
    )
    agent_predictions = np.asarray(
        [int(row["agent_prediction"]) for row in feature_rows], dtype=int
    )
    agent_only_targets = np.asarray(
        [int(row["paired_outcome"] == "agent_only_correct") for row in feature_rows],
        dtype=int,
    )
    knn_only_targets = np.asarray(
        [int(row["paired_outcome"] == "knn_only_correct") for row in feature_rows],
        dtype=int,
    )
    candidate_routed = route_predictions(
        knn_predictions,
        agent_predictions,
        scores,
        np.full(len(feature_rows), candidate_threshold),
    )
    routed = route_predictions(
        knn_predictions,
        agent_predictions,
        scores,
        np.full(len(feature_rows), threshold),
    )
    evaluation = evaluation_metrics(
        labels,
        knn_predictions,
        agent_predictions,
        routed,
        agent_probabilities=agent_probabilities,
        knn_probabilities=knn_probabilities,
        agent_only_targets=agent_only_targets,
        knn_only_targets=knn_only_targets,
    )
    candidate_evaluation = evaluation_metrics(
        labels,
        knn_predictions,
        agent_predictions,
        candidate_routed,
        agent_probabilities=agent_probabilities,
        knn_probabilities=knn_probabilities,
        agent_only_targets=agent_only_targets,
        knn_only_targets=knn_only_targets,
    )
    deployed_uncertainty = paired_bootstrap_deltas(
        labels,
        knn_predictions,
        routed,
        seed=20260805,
        repetitions=10000,
    )
    candidate_uncertainty = paired_bootstrap_deltas(
        labels,
        knn_predictions,
        candidate_routed,
        seed=20260805,
        repetitions=10000,
    )

    valid_dir = router_dir / "valid"
    valid_dir.mkdir(parents=True, exist_ok=True)
    features_path = valid_dir / "features.jsonl"
    predictions_path = valid_dir / "predictions.jsonl"
    write_jsonl_atomic(features_path, feature_rows)
    prediction_rows = [
        {
            "schema_version": ROUTER_SCHEMA_VERSION,
            "task": spec.task,
            "evaluation_subset": "valid",
            "query_index": index,
            "Y": int(labels[index]),
            "knn_prediction": int(knn_predictions[index]),
            "agent_prediction": int(agent_predictions[index]),
            "paired_outcome": feature_rows[index]["paired_outcome"],
            "agent_only_probability": float(agent_probabilities[index]),
            "knn_only_probability": float(knn_probabilities[index]),
            "router_score": float(scores[index]),
            "candidate_threshold": candidate_threshold,
            "candidate_route_to_agent": bool(scores[index] >= candidate_threshold),
            "candidate_router_prediction": int(candidate_routed[index]),
            "router_threshold": threshold,
            "route_to_agent": bool(scores[index] >= threshold),
            "router_prediction": int(routed[index]),
            "router_correct": bool(routed[index] == labels[index]),
        }
        for index in range(len(feature_rows))
    ]
    write_jsonl_atomic(predictions_path, prediction_rows)

    metrics = {
        "schema_version": "task_local_knn_agent_router.valid.v2",
        "task": spec.task,
        "evaluation_subset": "valid",
        "n_rows": len(feature_rows),
        "training_policy": "all scaffold-train rows with OOF candidate predictions",
        "valid_policy": "frozen v2 evaluation; no valid model, feature, threshold, or promotion tuning",
        "model_family": str(model_manifest["model_family"]),
        "feature_profile": str(model_manifest["feature_profile"]),
        "promoted": bool(model_manifest["promoted"]),
        "candidate_threshold": candidate_threshold,
        "threshold": threshold,
        "training_rows": int(model_manifest["training_rows"]),
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "metrics": evaluation,
        "candidate_metrics": candidate_evaluation,
        "paired_uncertainty_vs_knn": deployed_uncertainty,
        "candidate_paired_uncertainty_vs_knn": candidate_uncertainty,
        "delta_vs_knn": {
            "accuracy": evaluation["router"]["accuracy"]
            - evaluation["always_knn"]["accuracy"],
            "macro_f1": evaluation["router"]["macro_f1"]
            - evaluation["always_knn"]["macro_f1"],
        },
        "paths": {
            "features": str(features_path),
            "predictions": str(predictions_path),
            "model": str(model_path),
        },
    }
    metrics_path = valid_dir / "metrics.json"
    write_json_atomic(metrics_path, metrics)
    write_json_atomic(
        valid_dir / "manifest.json",
        {
            "schema_version": "task_local_knn_agent_router.valid_manifest.v2",
            "task": spec.task,
            "evaluation_subset": "valid",
            "no_valid_tuning": True,
            "source_artifacts": {
                str(path): _sha256(path)
                for path in (
                    query_path,
                    knn_path,
                    knn_manifest_path,
                    agent_path,
                    agent_manifest_path,
                    agent_metrics_path,
                    model_path,
                    model_manifest_path,
                    train_features_path,
                )
            },
            "outputs": {
                "features": str(features_path),
                "predictions": str(predictions_path),
                "metrics": str(metrics_path),
            },
        },
    )
    (valid_dir / "report_zh.md").write_text(
        _render_report(spec, metrics), encoding="utf-8"
    )
    return metrics


def _validate_contracts(
    spec: TaskSpec,
    *,
    query_path: Path,
    n_rows: int,
    knn_manifest: dict[str, Any],
    agent_manifest: dict[str, Any],
    agent_metrics: dict[str, Any],
    model_manifest: dict[str, Any],
    train_features_path: Path,
) -> None:
    if int(knn_manifest.get("k") or 0) != 3:
        raise ValueError("Valid router requires frozen Morgan k=3 KNN")
    if str(knn_manifest.get("evaluation_split") or "") != "valid":
        raise ValueError("KNN artifact is not the valid split")
    if Path(str(knn_manifest.get("test_path") or "")) != query_path:
        raise ValueError("KNN valid input does not match the frozen query file")
    expected_agent = {
        "batch_id": spec.direct_condition,
        "input_jsonl": str(query_path),
        "n_items": n_rows,
        "model": "gpt-oss-120b",
        "visibility_mode": "identity_blind",
        "neighbor_identity_policy": "parent_disjoint",
        "experiment_mode": "direct",
        "retrieval_source": "starling",
        "neighbor_selector": "similarity",
    }
    for key, expected in expected_agent.items():
        if agent_manifest.get(key) != expected:
            raise ValueError(
                f"Agent valid contract mismatch for {key}: "
                f"{agent_manifest.get(key)!r} != {expected!r}"
            )
    if int(agent_metrics.get("n_failed_runs") or 0) != 0:
        raise ValueError("Agent valid artifact contains failed runs")
    model_columns = tuple(model_manifest.get("feature_columns") or ())
    if not model_columns or not set(model_columns).issubset(FEATURE_COLUMNS):
        raise ValueError("Router model manifest feature schema mismatch")
    if tuple(model_manifest.get("available_feature_columns") or ()) != FEATURE_COLUMNS:
        raise ValueError("Router model available-feature contract mismatch")
    if str(model_manifest.get("task") or "") != spec.task:
        raise ValueError("Router model task mismatch")
    with train_features_path.open(encoding="utf-8") as handle:
        training_rows = sum(bool(line.strip()) for line in handle)
    if int(model_manifest.get("training_rows") or 0) != training_rows:
        raise ValueError("Router model was not fit on the complete train feature table")


def _agent_query_index(row: dict[str, Any]) -> int:
    if "query_index" in row:
        return int(row["query_index"])
    run_id = str(row.get("run_id") or "")
    if "_idx" in run_id:
        return int(run_id.rsplit("_idx", 1)[1])
    raise ValueError(f"Agent prediction has no query index: {row}")


def _require_exact_indices(
    name: str,
    rows: dict[int, dict[str, Any]],
    expected: set[int],
) -> None:
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        extra = sorted(set(rows) - expected)
        raise ValueError(f"Incomplete valid {name}: missing={missing[:10]} extra={extra[:10]}")


def _render_report(spec: TaskSpec, result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    candidate = result["candidate_metrics"]
    delta = result["delta_vs_knn"]
    return (
        f"# {spec.data_name} frozen router valid 报告\n\n"
        "模型 family、参数、校准器和 routing threshold 全部由 scaffold-train OOF 冻结；"
        "valid label 只用于本报告评估，没有参与选择或调参。\n\n"
        f"- Valid 样本数：{result['n_rows']}\n"
        f"- Deployment family / profile：{result['model_family']} / "
        f"{result['feature_profile']}\n"
        f"- Promotion gate：{'PASS' if result['promoted'] else 'FAIL -> KNN fallback'}\n"
        f"- Candidate / deployed threshold：{result['candidate_threshold']:.2f} / "
        f"{result['threshold']:.2f}\n"
        f"- KNN accuracy / macro-F1：{metrics['always_knn']['accuracy']:.4f} / "
        f"{metrics['always_knn']['macro_f1']:.4f}\n"
        f"- Agent accuracy / macro-F1：{metrics['always_agent']['accuracy']:.4f} / "
        f"{metrics['always_agent']['macro_f1']:.4f}\n"
        f"- Router accuracy / macro-F1：{metrics['router']['accuracy']:.4f} / "
        f"{metrics['router']['macro_f1']:.4f}\n"
        f"- Ungated candidate accuracy / macro-F1：{candidate['router']['accuracy']:.4f} / "
        f"{candidate['router']['macro_f1']:.4f}\n"
        f"- Router 相对 KNN accuracy / macro-F1：{delta['accuracy']:+.4f} / "
        f"{delta['macro_f1']:+.4f}\n"
        f"- Router 切换率：{metrics['router']['switch_rate']:.4f}\n"
        f"- Oracle accuracy / macro-F1：{metrics['oracle']['accuracy']:.4f} / "
        f"{metrics['oracle']['macro_f1']:.4f}\n"
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value
