"""Nested OOF evaluation and final fitting for task-local dual-risk routers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic

from .contract import (
    ROUTER_ARTIFACT_DIR,
    ROUTER_FEATURE_STEM,
    ROUTER_SCHEMA_VERSION,
    TaskSpec,
    task_root,
)
from .features import FEATURE_COLUMNS, FEATURE_PROFILES
from .io import read_jsonl as _read_jsonl


MODEL_FAMILIES = ("logistic_regression", "hist_gradient_boosting")
NET_SCORE_THRESHOLD_GRID = tuple(
    round(value, 2) for value in np.linspace(-0.5, 0.5, 101)
) + (1.01,)
MAX_ACCURACY_DROP = 0.005
PROMOTION_BOOTSTRAP_REPETITIONS = 2000
RANDOM_SEED = 20260804


def train_task_routers(
    spec: TaskSpec,
    *,
    output_root: str | Path,
    model_families: Iterable[str] = MODEL_FAMILIES,
    feature_profiles: Iterable[str] = tuple(FEATURE_PROFILES),
) -> dict[str, Any]:
    rows = _read_jsonl(
        task_root(output_root, spec.task) / f"{ROUTER_FEATURE_STEM}.jsonl"
    )
    if not rows:
        raise ValueError(f"No router features for {spec.task}")
    families = tuple(model_families)
    profiles = tuple(feature_profiles)
    unknown_families = sorted(set(families) - set(MODEL_FAMILIES))
    unknown_profiles = sorted(set(profiles) - set(FEATURE_PROFILES))
    if unknown_families:
        raise ValueError(f"Unknown router model families: {', '.join(unknown_families)}")
    if unknown_profiles:
        raise ValueError(f"Unknown router feature profiles: {', '.join(unknown_profiles)}")

    folds = np.asarray([int(row["fold"]) for row in rows], dtype=int)
    unique_folds = sorted(set(folds.tolist()))
    if len(unique_folds) < 3:
        raise ValueError("Nested OOF router evaluation requires at least 3 folds")
    labels = np.asarray([int(row["Y"]) for row in rows], dtype=int)
    knn_predictions = np.asarray([int(row["knn_prediction"]) for row in rows], dtype=int)
    agent_predictions = np.asarray([int(row["agent_prediction"]) for row in rows], dtype=int)
    agent_only_target = np.asarray(
        [int(row["paired_outcome"] == "agent_only_correct") for row in rows], dtype=int
    )
    knn_only_target = np.asarray(
        [int(row["paired_outcome"] == "knn_only_correct") for row in rows], dtype=int
    )
    matrices = {
        profile: _feature_matrix(rows, FEATURE_PROFILES[profile]) for profile in profiles
    }
    candidate_names = [
        _candidate_name(family, profile) for family in families for profile in profiles
    ]
    candidate_meta = {
        _candidate_name(family, profile): (family, profile)
        for family in families
        for profile in profiles
    }

    candidate_agent_probabilities = {
        name: np.full(len(rows), np.nan) for name in candidate_names
    }
    candidate_knn_probabilities = {
        name: np.full(len(rows), np.nan) for name in candidate_names
    }
    candidate_thresholds = {name: np.full(len(rows), np.nan) for name in candidate_names}
    selected_agent_probabilities = np.full(len(rows), np.nan)
    selected_knn_probabilities = np.full(len(rows), np.nan)
    selected_thresholds = np.full(len(rows), np.nan)
    selected_candidates = np.full(len(rows), "", dtype=object)
    outer_records: list[dict[str, Any]] = []

    for outer_fold in unique_folds:
        outer_test = folds == outer_fold
        outer_train = ~outer_test
        inner_results: dict[str, dict[str, Any]] = {}
        for name in candidate_names:
            family, profile = candidate_meta[name]
            matrix = matrices[profile]
            inner_raw_agent, inner_raw_knn = _cross_fitted_dual_probabilities(
                family,
                matrix[outer_train],
                agent_only_target[outer_train],
                knn_only_target[outer_train],
                folds[outer_train],
            )
            agent_calibrator = _fit_calibrator(
                inner_raw_agent, agent_only_target[outer_train]
            )
            knn_calibrator = _fit_calibrator(
                inner_raw_knn, knn_only_target[outer_train]
            )
            inner_agent_probability = _apply_calibrator(
                agent_calibrator, inner_raw_agent
            )
            inner_knn_probability = _apply_calibrator(knn_calibrator, inner_raw_knn)
            inner_score = inner_agent_probability - inner_knn_probability
            threshold, inner_metrics = choose_threshold(
                labels[outer_train],
                knn_predictions[outer_train],
                agent_predictions[outer_train],
                inner_score,
            )
            estimators = _fit_dual_estimators(
                family,
                matrix[outer_train],
                agent_only_target[outer_train],
                knn_only_target[outer_train],
            )
            outer_agent_probability = _apply_calibrator(
                agent_calibrator,
                _predict_probability(estimators["agent_only"], matrix[outer_test]),
            )
            outer_knn_probability = _apply_calibrator(
                knn_calibrator,
                _predict_probability(estimators["knn_only"], matrix[outer_test]),
            )
            candidate_agent_probabilities[name][outer_test] = outer_agent_probability
            candidate_knn_probabilities[name][outer_test] = outer_knn_probability
            candidate_thresholds[name][outer_test] = threshold
            inner_results[name] = {
                "threshold": threshold,
                "inner_metrics": inner_metrics,
                "outer_agent_probability": outer_agent_probability,
                "outer_knn_probability": outer_knn_probability,
            }

        selected_name = max(
            candidate_names,
            key=lambda name: _selection_key(
                inner_results[name]["inner_metrics"],
                *candidate_meta[name],
            ),
        )
        selected = inner_results[selected_name]
        selected_agent_probabilities[outer_test] = selected["outer_agent_probability"]
        selected_knn_probabilities[outer_test] = selected["outer_knn_probability"]
        selected_thresholds[outer_test] = selected["threshold"]
        selected_candidates[outer_test] = selected_name
        outer_records.append(
            {
                "outer_fold": outer_fold,
                "selected_candidate": selected_name,
                "selected_family": candidate_meta[selected_name][0],
                "selected_feature_profile": candidate_meta[selected_name][1],
                "selected_threshold": selected["threshold"],
                "inner_candidates": {
                    name: {
                        "threshold": result["threshold"],
                        "router_macro_f1": result["inner_metrics"]["router"]["macro_f1"],
                        "router_accuracy": result["inner_metrics"]["router"]["accuracy"],
                        "switch_rate": result["inner_metrics"]["router"]["switch_rate"],
                    }
                    for name, result in inner_results.items()
                },
            }
        )

    arrays = (
        selected_agent_probabilities,
        selected_knn_probabilities,
        selected_thresholds,
    )
    if any(np.isnan(array).any() for array in arrays):
        raise AssertionError("Nested OOF evaluation left missing predictions")

    candidate_metrics: dict[str, Any] = {}
    candidate_routed: dict[str, np.ndarray] = {}
    for name in candidate_names:
        score = candidate_agent_probabilities[name] - candidate_knn_probabilities[name]
        routed = route_predictions(
            knn_predictions, agent_predictions, score, candidate_thresholds[name]
        )
        candidate_routed[name] = routed
        candidate_metrics[name] = evaluation_metrics(
            labels,
            knn_predictions,
            agent_predictions,
            routed,
            agent_probabilities=candidate_agent_probabilities[name],
            knn_probabilities=candidate_knn_probabilities[name],
            agent_only_targets=agent_only_target,
            knn_only_targets=knn_only_target,
        )

    selected_scores = selected_agent_probabilities - selected_knn_probabilities
    selected_candidate_routed = route_predictions(
        knn_predictions, agent_predictions, selected_scores, selected_thresholds
    )
    nested_candidate_metrics = evaluation_metrics(
        labels,
        knn_predictions,
        agent_predictions,
        selected_candidate_routed,
        agent_probabilities=selected_agent_probabilities,
        knn_probabilities=selected_knn_probabilities,
        agent_only_targets=agent_only_target,
        knn_only_targets=knn_only_target,
    )
    promotion_gate = evaluate_promotion_gate(
        labels,
        knn_predictions,
        selected_candidate_routed,
        seed=RANDOM_SEED,
    )
    selected_deployed_routed = (
        selected_candidate_routed if promotion_gate["passed"] else knn_predictions.copy()
    )
    nested_deployed_metrics = evaluation_metrics(
        labels,
        knn_predictions,
        agent_predictions,
        selected_deployed_routed,
    )

    deployment_candidates: dict[str, Any] = {}
    for name in candidate_names:
        family, profile = candidate_meta[name]
        matrix = matrices[profile]
        raw_agent, raw_knn = _cross_fitted_dual_probabilities(
            family,
            matrix,
            agent_only_target,
            knn_only_target,
            folds,
        )
        agent_calibrator = _fit_calibrator(raw_agent, agent_only_target)
        knn_calibrator = _fit_calibrator(raw_knn, knn_only_target)
        agent_probability = _apply_calibrator(agent_calibrator, raw_agent)
        knn_probability = _apply_calibrator(knn_calibrator, raw_knn)
        score = agent_probability - knn_probability
        threshold, threshold_metrics = choose_threshold(
            labels, knn_predictions, agent_predictions, score
        )
        deployment_candidates[name] = {
            "family": family,
            "feature_profile": profile,
            "threshold": threshold,
            "cross_fitted_metrics": threshold_metrics,
            "agent_calibrator": agent_calibrator,
            "knn_calibrator": knn_calibrator,
        }
    deployment_name = max(
        candidate_names,
        key=lambda name: _selection_key(
            deployment_candidates[name]["cross_fitted_metrics"],
            deployment_candidates[name]["family"],
            deployment_candidates[name]["feature_profile"],
        ),
    )
    deployment = deployment_candidates[deployment_name]
    deployment_family = deployment["family"]
    deployment_profile = deployment["feature_profile"]
    candidate_threshold = float(deployment["threshold"])
    deployed_threshold = candidate_threshold if promotion_gate["passed"] else 1.01
    deployment_estimators = _fit_dual_estimators(
        deployment_family,
        matrices[deployment_profile],
        agent_only_target,
        knn_only_target,
    )
    deployment_model = {
        "agent_only_estimator": deployment_estimators["agent_only"],
        "knn_only_estimator": deployment_estimators["knn_only"],
        "agent_only_calibrator": deployment["agent_calibrator"],
        "knn_only_calibrator": deployment["knn_calibrator"],
        "feature_columns": list(FEATURE_PROFILES[deployment_profile]),
        "feature_profile": deployment_profile,
        "candidate_threshold": candidate_threshold,
        "threshold": deployed_threshold,
        "promoted": bool(promotion_gate["passed"]),
        "fallback": "knn",
    }

    model_dir = task_root(output_root, spec.task) / ROUTER_ARTIFACT_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    joblib.dump(deployment_model, model_path)
    prediction_rows = []
    for index, row in enumerate(rows):
        candidate_route = selected_scores[index] >= selected_thresholds[index]
        deployed_route = bool(candidate_route and promotion_gate["passed"])
        prediction_rows.append(
            {
                "schema_version": ROUTER_SCHEMA_VERSION,
                "task": spec.task,
                "fold": int(row["fold"]),
                "train_index": int(row["train_index"]),
                "Y": int(labels[index]),
                "knn_prediction": int(knn_predictions[index]),
                "agent_prediction": int(agent_predictions[index]),
                "paired_outcome": row["paired_outcome"],
                "selected_candidate": str(selected_candidates[index]),
                "agent_only_probability": float(selected_agent_probabilities[index]),
                "knn_only_probability": float(selected_knn_probabilities[index]),
                "router_score": float(selected_scores[index]),
                "candidate_threshold": float(selected_thresholds[index]),
                "candidate_route_to_agent": bool(candidate_route),
                "route_to_agent": deployed_route,
                "candidate_router_prediction": int(selected_candidate_routed[index]),
                "router_prediction": int(selected_deployed_routed[index]),
                "router_correct": bool(selected_deployed_routed[index] == labels[index]),
                "candidate_predictions": {
                    name: {
                        "agent_only_probability": float(
                            candidate_agent_probabilities[name][index]
                        ),
                        "knn_only_probability": float(
                            candidate_knn_probabilities[name][index]
                        ),
                        "score": float(
                            candidate_agent_probabilities[name][index]
                            - candidate_knn_probabilities[name][index]
                        ),
                        "threshold": float(candidate_thresholds[name][index]),
                        "prediction": int(candidate_routed[name][index]),
                    }
                    for name in candidate_names
                },
            }
        )
    predictions_path = model_dir / "nested_oof_predictions.jsonl"
    write_jsonl_atomic(predictions_path, prediction_rows)

    selection_counts = dict(Counter(str(value) for value in selected_candidates))
    metrics = {
        "schema_version": ROUTER_SCHEMA_VERSION,
        "task": spec.task,
        "n_rows": len(rows),
        "n_folds": len(unique_folds),
        "feature_schema": rows[0]["schema_version"],
        "available_feature_columns": list(FEATURE_COLUMNS),
        "feature_profiles": {
            name: list(columns) for name, columns in FEATURE_PROFILES.items() if name in profiles
        },
        "target_definition": {
            "agent_only": "agent_only_correct",
            "knn_only": "knn_only_correct",
            "router_score": "P(agent_only_correct)-P(knn_only_correct)",
        },
        "default_route": "knn",
        "accuracy_guardrail": {"maximum_drop": MAX_ACCURACY_DROP},
        "nested_candidate_router": nested_candidate_metrics,
        "nested_deployed_router": nested_deployed_metrics,
        "nested_candidate_metrics": candidate_metrics,
        "outer_fold_selection": outer_records,
        "outer_selection_counts": selection_counts,
        "promotion_gate": promotion_gate,
        "deployment_model": {
            "candidate": deployment_name,
            "family": deployment_family,
            "feature_profile": deployment_profile,
            "feature_columns": list(FEATURE_PROFILES[deployment_profile]),
            "candidate_threshold": candidate_threshold,
            "threshold": deployed_threshold,
            "promoted": bool(promotion_gate["passed"]),
            "selection_basis": (
                "all-train dual-risk cross-fitted macro-F1 subject to accuracy guardrail; "
                "nested OOF promotion gate controls KNN fallback"
            ),
            "probability_calibration": (
                "separate Platt sigmoid calibrators for agent-only and KNN-only risks"
            ),
            "cross_fitted_metrics": deployment["cross_fitted_metrics"],
            "path": str(model_path),
        },
        "paths": {"predictions": str(predictions_path), "model": str(model_path)},
    }
    write_json_atomic(model_dir / "metrics.json", metrics)
    write_json_atomic(
        model_dir / "model_manifest.json",
        {
            "schema_version": ROUTER_SCHEMA_VERSION,
            "task": spec.task,
            "model_family": deployment_family,
            "feature_profile": deployment_profile,
            "feature_columns": list(FEATURE_PROFILES[deployment_profile]),
            "available_feature_columns": list(FEATURE_COLUMNS),
            "candidate_threshold": candidate_threshold,
            "threshold": deployed_threshold,
            "promoted": bool(promotion_gate["passed"]),
            "promotion_gate": promotion_gate,
            "target_definition": (
                "P(agent_only_correct)-P(knn_only_correct) dual-risk score"
            ),
            "fallback": "knn",
            "probability_calibration": (
                "separate Platt sigmoid calibrators on cross-fitted raw probabilities"
            ),
            "training_rows": len(rows),
            "training_folds": unique_folds,
            "artifact": str(model_path),
        },
    )
    (model_dir / "report_zh.md").write_text(
        _render_report(spec, metrics), encoding="utf-8"
    )
    return metrics


def choose_threshold(
    labels: np.ndarray,
    knn_predictions: np.ndarray,
    agent_predictions: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    knn_accuracy = float(accuracy_score(labels, knn_predictions))
    candidates: list[tuple[tuple[float, float, float, float], float, dict[str, Any]]] = []
    for threshold in NET_SCORE_THRESHOLD_GRID:
        routed = route_predictions(
            knn_predictions,
            agent_predictions,
            scores,
            np.full(len(scores), threshold),
        )
        metrics = evaluation_metrics(labels, knn_predictions, agent_predictions, routed)
        router = metrics["router"]
        if float(router["accuracy"]) < knn_accuracy - MAX_ACCURACY_DROP:
            continue
        key = (
            float(router["macro_f1"]),
            float(router["accuracy"]),
            -float(router["switch_rate"]),
            float(threshold),
        )
        candidates.append((key, float(threshold), metrics))
    if not candidates:
        raise AssertionError("No threshold satisfied the KNN accuracy guardrail")
    _, threshold, metrics = max(candidates, key=lambda item: item[0])
    return threshold, metrics


def route_predictions(
    knn_predictions: np.ndarray,
    agent_predictions: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    return np.where(scores >= thresholds, agent_predictions, knn_predictions).astype(int)


def predict_deployment_scores(
    deployment_model: dict[str, Any],
    rows: Sequence[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Apply both frozen risk heads using the model's explicit feature profile."""
    columns = tuple(deployment_model.get("feature_columns") or ())
    if not columns or not set(columns).issubset(FEATURE_COLUMNS):
        raise ValueError("Deployment router feature schema does not match the v2 contract")
    matrix = _feature_matrix(rows, columns)
    agent_probability = _apply_calibrator(
        deployment_model["agent_only_calibrator"],
        _predict_probability(deployment_model["agent_only_estimator"], matrix),
    )
    knn_probability = _apply_calibrator(
        deployment_model["knn_only_calibrator"],
        _predict_probability(deployment_model["knn_only_estimator"], matrix),
    )
    return {
        "agent_only_probability": agent_probability,
        "knn_only_probability": knn_probability,
        "score": agent_probability - knn_probability,
    }


def evaluation_metrics(
    labels: np.ndarray,
    knn_predictions: np.ndarray,
    agent_predictions: np.ndarray,
    routed_predictions: np.ndarray,
    *,
    agent_probabilities: np.ndarray | None = None,
    knn_probabilities: np.ndarray | None = None,
    agent_only_targets: np.ndarray | None = None,
    knn_only_targets: np.ndarray | None = None,
) -> dict[str, Any]:
    oracle = np.where(agent_predictions == labels, agent_predictions, knn_predictions)
    switched = (routed_predictions == agent_predictions) & (
        knn_predictions != agent_predictions
    )
    rescues = (routed_predictions == labels) & (knn_predictions != labels)
    harms = (routed_predictions != labels) & (knn_predictions == labels)
    result = {
        "always_knn": _classification_metrics(labels, knn_predictions),
        "always_agent": _classification_metrics(labels, agent_predictions),
        "router": {
            **_classification_metrics(labels, routed_predictions),
            "switch_count": int(switched.sum()),
            "switch_rate": float(switched.mean()),
            "rescue_count": int(rescues.sum()),
            "harm_count": int(harms.sum()),
            "net_rescues": int(rescues.sum() - harms.sum()),
        },
        "oracle": _classification_metrics(labels, oracle),
        "disagreement_count": int((knn_predictions != agent_predictions).sum()),
        "disagreement_rate": float((knn_predictions != agent_predictions).mean()),
    }
    if agent_probabilities is not None and agent_only_targets is not None:
        result["router"].update(
            {
                "agent_only_brier": float(
                    np.mean((agent_probabilities - agent_only_targets) ** 2)
                ),
                "agent_only_ece": expected_calibration_error(
                    agent_only_targets, agent_probabilities
                ),
            }
        )
    if knn_probabilities is not None and knn_only_targets is not None:
        result["router"].update(
            {
                "knn_only_brier": float(
                    np.mean((knn_probabilities - knn_only_targets) ** 2)
                ),
                "knn_only_ece": expected_calibration_error(
                    knn_only_targets, knn_probabilities
                ),
            }
        )
    return result


def evaluate_promotion_gate(
    labels: np.ndarray,
    knn_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    *,
    seed: int,
    repetitions: int = PROMOTION_BOOTSTRAP_REPETITIONS,
) -> dict[str, Any]:
    uncertainty = paired_bootstrap_deltas(
        labels,
        knn_predictions,
        candidate_predictions,
        seed=seed,
        repetitions=repetitions,
    )
    observed_macro_delta = uncertainty["observed_macro_f1_delta"]
    accuracy_ci = uncertainty["accuracy_delta_ci95"]
    macro_ci = uncertainty["macro_f1_delta_ci95"]
    passed = bool(
        observed_macro_delta > 0
        and macro_ci[0] > 0
        and accuracy_ci[0] >= -MAX_ACCURACY_DROP
    )
    reasons = []
    if observed_macro_delta <= 0:
        reasons.append("observed_macro_f1_delta_not_positive")
    if macro_ci[0] <= 0:
        reasons.append("macro_f1_delta_lower_ci_not_positive")
    if accuracy_ci[0] < -MAX_ACCURACY_DROP:
        reasons.append("accuracy_delta_lower_ci_below_guardrail")
    return {
        "passed": passed,
        "fallback_when_failed": "knn",
        "maximum_accuracy_drop": MAX_ACCURACY_DROP,
        **uncertainty,
        "failure_reasons": reasons,
    }


def paired_bootstrap_deltas(
    labels: np.ndarray,
    reference_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    *,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    """Return label-stratified paired uncertainty without selecting a policy."""
    observed_accuracy_delta = float(
        accuracy_score(labels, candidate_predictions)
        - accuracy_score(labels, reference_predictions)
    )
    observed_macro_delta = float(
        f1_score(labels, candidate_predictions, average="macro", zero_division=0)
        - f1_score(labels, reference_predictions, average="macro", zero_division=0)
    )
    rng = np.random.default_rng(seed)
    # Within each true-label stratum, a paired bootstrap is fully described by
    # counts of the four (reference prediction, candidate prediction) states.
    # Sampling those counts from their multinomial distribution is identical
    # to materializing every resampled row, while evaluating all repetitions
    # with vectorized confusion-matrix arithmetic.
    sampled_by_label = []
    for label in (0, 1):
        class_mask = labels == label
        n_class = int(class_mask.sum())
        if n_class == 0:
            sampled_by_label.append(np.zeros((repetitions, 4), dtype=int))
            continue
        state = 2 * reference_predictions[class_mask] + candidate_predictions[class_mask]
        probabilities = np.bincount(state, minlength=4) / n_class
        sampled_by_label.append(rng.multinomial(n_class, probabilities, size=repetitions))
    class_zero, class_one = sampled_by_label

    reference_tn = class_zero[:, 0] + class_zero[:, 1]
    reference_fp = class_zero[:, 2] + class_zero[:, 3]
    reference_fn = class_one[:, 0] + class_one[:, 1]
    reference_tp = class_one[:, 2] + class_one[:, 3]
    candidate_tn = class_zero[:, 0] + class_zero[:, 2]
    candidate_fp = class_zero[:, 1] + class_zero[:, 3]
    candidate_fn = class_one[:, 0] + class_one[:, 2]
    candidate_tp = class_one[:, 1] + class_one[:, 3]

    accuracy_deltas = (
        candidate_tn + candidate_tp - reference_tn - reference_tp
    ) / len(labels)

    def _macro_f1_from_confusion(tn, fp, fn, tp):
        negative_denominator = 2 * tn + fp + fn
        positive_denominator = 2 * tp + fp + fn
        negative_f1 = np.divide(
            2 * tn,
            negative_denominator,
            out=np.zeros(repetitions, dtype=float),
            where=negative_denominator != 0,
        )
        positive_f1 = np.divide(
            2 * tp,
            positive_denominator,
            out=np.zeros(repetitions, dtype=float),
            where=positive_denominator != 0,
        )
        return (negative_f1 + positive_f1) / 2

    macro_deltas = _macro_f1_from_confusion(
        candidate_tn, candidate_fp, candidate_fn, candidate_tp
    ) - _macro_f1_from_confusion(
        reference_tn, reference_fp, reference_fn, reference_tp
    )
    accuracy_ci = [float(value) for value in np.quantile(accuracy_deltas, [0.025, 0.975])]
    macro_ci = [float(value) for value in np.quantile(macro_deltas, [0.025, 0.975])]
    return {
        "bootstrap_repetitions": repetitions,
        "bootstrap_method": "label-stratified paired bootstrap",
        "observed_accuracy_delta": observed_accuracy_delta,
        "observed_macro_f1_delta": observed_macro_delta,
        "accuracy_delta_ci95": accuracy_ci,
        "macro_f1_delta_ci95": macro_ci,
    }


def expected_calibration_error(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    if len(targets) == 0:
        return 0.0
    error = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for index in range(n_bins):
        if index == n_bins - 1:
            mask = (probabilities >= edges[index]) & (
                probabilities <= edges[index + 1]
            )
        else:
            mask = (probabilities >= edges[index]) & (
                probabilities < edges[index + 1]
            )
        if mask.any():
            error += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(targets[mask].mean())
            )
    return error


def _cross_fitted_dual_probabilities(
    family: str,
    matrix: np.ndarray,
    agent_only_target: np.ndarray,
    knn_only_target: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    agent_probabilities = np.full(len(agent_only_target), np.nan)
    knn_probabilities = np.full(len(knn_only_target), np.nan)
    for fold in sorted(set(folds.tolist())):
        heldout = folds == fold
        training = ~heldout
        estimators = _fit_dual_estimators(
            family,
            matrix[training],
            agent_only_target[training],
            knn_only_target[training],
        )
        agent_probabilities[heldout] = _predict_probability(
            estimators["agent_only"], matrix[heldout]
        )
        knn_probabilities[heldout] = _predict_probability(
            estimators["knn_only"], matrix[heldout]
        )
    if np.isnan(agent_probabilities).any() or np.isnan(knn_probabilities).any():
        raise AssertionError("Cross-fitting left missing dual-risk probabilities")
    return agent_probabilities, knn_probabilities


def _fit_dual_estimators(
    family: str,
    matrix: np.ndarray,
    agent_only_target: np.ndarray,
    knn_only_target: np.ndarray,
) -> dict[str, Any]:
    return {
        "agent_only": _fit_estimator(family, matrix, agent_only_target),
        "knn_only": _fit_estimator(family, matrix, knn_only_target),
    }


def _fit_estimator(family: str, matrix: np.ndarray, target: np.ndarray):
    if len(np.unique(target)) < 2:
        estimator = DummyClassifier(strategy="constant", constant=int(target[0]))
        estimator.fit(matrix, target)
        return estimator
    estimator = _make_estimator(family)
    estimator.fit(matrix, target, classifier__sample_weight=_balanced_sample_weights(target))
    return estimator


def _make_estimator(family: str) -> Pipeline:
    if family == "logistic_regression":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
    if family == "hist_gradient_boosting":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=200,
                        max_leaf_nodes=15,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model family: {family}")


def _predict_probability(estimator: Any, matrix: np.ndarray) -> np.ndarray:
    probabilities = estimator.predict_proba(matrix)
    classes = [int(value) for value in estimator.classes_]
    if 1 not in classes:
        return np.zeros(len(matrix), dtype=float)
    return np.asarray(probabilities[:, classes.index(1)], dtype=float)


def _fit_calibrator(probabilities: np.ndarray, target: np.ndarray):
    values = np.asarray(probabilities, dtype=float).reshape(-1, 1)
    if len(np.unique(target)) < 2:
        calibrator = DummyClassifier(strategy="constant", constant=int(target[0]))
    else:
        calibrator = LogisticRegression(
            C=1e6,
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )
    calibrator.fit(values, target)
    return calibrator


def _apply_calibrator(calibrator: Any, probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float).reshape(-1, 1)
    calibrated = calibrator.predict_proba(values)
    classes = [int(value) for value in calibrator.classes_]
    if 1 not in classes:
        return np.zeros(len(values), dtype=float)
    return np.asarray(calibrated[:, classes.index(1)], dtype=float)


def _balanced_sample_weights(target: np.ndarray) -> np.ndarray:
    counts = np.bincount(target, minlength=2)
    weights = np.ones(len(target), dtype=float)
    for label, count in enumerate(counts):
        if count:
            weights[target == label] = len(target) / (2 * count)
    return weights


def _feature_matrix(
    rows: Sequence[dict[str, Any]], columns: Sequence[str] = FEATURE_COLUMNS
) -> np.ndarray:
    return np.asarray(
        [[float(row["features"][column]) for column in columns] for row in rows],
        dtype=float,
    )


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "recall_0": float(tn / (tn + fp)) if tn + fp else 0.0,
        "recall_1": float(tp / (tp + fn)) if tp + fn else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _candidate_name(family: str, profile: str) -> str:
    return f"{family}__{profile}"


def _selection_key(
    metrics: dict[str, Any], family: str, profile: str
) -> tuple[float, float, float, int, int]:
    router = metrics["router"]
    return (
        float(router["macro_f1"]),
        float(router["accuracy"]),
        -float(router["switch_rate"]),
        int(family == "logistic_regression"),
        -len(FEATURE_PROFILES[profile]),
    )


def _render_report(spec: TaskSpec, metrics: dict[str, Any]) -> str:
    candidate = metrics["nested_candidate_router"]
    deployed = metrics["nested_deployed_router"]
    deployment = metrics["deployment_model"]
    gate = metrics["promotion_gate"]
    return (
        f"# {spec.data_name} task-local dual-risk router OOF 报告\n\n"
        "v2 删除 absolute reference-size 和确定性重复 feature，分别估计 agent-only 与 "
        "KNN-only 风险；只有 nested OOF promotion gate 通过才部署，否则回退 KNN。\n\n"
        f"- 样本数：{metrics['n_rows']}\n"
        f"- KNN accuracy / macro-F1：{candidate['always_knn']['accuracy']:.4f} / "
        f"{candidate['always_knn']['macro_f1']:.4f}\n"
        f"- Candidate router accuracy / macro-F1：{candidate['router']['accuracy']:.4f} / "
        f"{candidate['router']['macro_f1']:.4f}\n"
        f"- Promotion gate：{'PASS' if gate['passed'] else 'FAIL'}; "
        f"reasons={gate['failure_reasons']}\n"
        f"- Deployed router accuracy / macro-F1：{deployed['router']['accuracy']:.4f} / "
        f"{deployed['router']['macro_f1']:.4f}\n"
        f"- Deployment candidate：{deployment['candidate']}\n"
        f"- Candidate / deployed threshold：{deployment['candidate_threshold']:.2f} / "
        f"{deployment['threshold']:.2f}\n"
    )
