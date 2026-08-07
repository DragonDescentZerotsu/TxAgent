"""Nested OOF training for the disagreement-only output-aware post-selector."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic

from .contract import TaskSpec, task_root
from .io import read_jsonl as _read_jsonl
from .post_contract import POST_ARTIFACT_DIR, POST_FEATURE_STEM, POST_SELECTOR_SCHEMA_VERSION
from .post_features import POST_FEATURE_COLUMNS, POST_FEATURE_PROFILES
from .train import evaluation_metrics, evaluate_promotion_gate


POST_MODEL_FAMILIES = ("logistic_regression", "hist_gradient_boosting")
THRESHOLD_GRID = tuple(round(value, 3) for value in np.linspace(0.0, 1.0, 41)) + (1.01,)
RANDOM_SEED = 20260805
MAX_ACCURACY_DROP = 0.005


def train_post_selectors(
    spec: TaskSpec,
    *,
    output_root: str | Path,
    model_families: Iterable[str] = POST_MODEL_FAMILIES,
    feature_profiles: Iterable[str] = tuple(POST_FEATURE_PROFILES),
) -> dict[str, Any]:
    rows = _read_jsonl(task_root(output_root, spec.task) / f"{POST_FEATURE_STEM}.jsonl")
    if not rows:
        raise ValueError(f"No post-selector features for {spec.task}")
    families = tuple(model_families)
    profiles = tuple(feature_profiles)
    if set(families) - set(POST_MODEL_FAMILIES):
        raise ValueError(f"Unknown post-selector families: {sorted(set(families) - set(POST_MODEL_FAMILIES))}")
    if set(profiles) - set(POST_FEATURE_PROFILES):
        raise ValueError(f"Unknown post-selector profiles: {sorted(set(profiles) - set(POST_FEATURE_PROFILES))}")

    folds = np.asarray([int(row["fold"]) for row in rows], dtype=int)
    unique_folds = sorted(set(folds.tolist()))
    labels = np.asarray([int(row["Y"]) for row in rows], dtype=int)
    knn = np.asarray([int(row["knn_prediction"]) for row in rows], dtype=int)
    agent = np.asarray([int(row["agent_prediction"]) for row in rows], dtype=int)
    disagreement = knn != agent
    target = np.asarray([int(row["paired_outcome"] == "agent_only_correct") for row in rows], dtype=int)
    if not disagreement.any() or len(set(target[disagreement].tolist())) < 2:
        raise ValueError(f"Post-selector requires both disagreement outcomes for {spec.task}")
    matrices = {profile: _matrix(rows, POST_FEATURE_PROFILES[profile]) for profile in profiles}
    candidates = [(family, profile) for family in families for profile in profiles]
    names = [_name(*candidate) for candidate in candidates]

    candidate_probability = {name: np.full(len(rows), 0.5) for name in names}
    candidate_t01 = {name: np.full(len(rows), 1.01) for name in names}
    candidate_t10 = {name: np.full(len(rows), 1.01) for name in names}
    selected_probability = np.full(len(rows), 0.5)
    selected_t01 = np.full(len(rows), 1.01)
    selected_t10 = np.full(len(rows), 1.01)
    selected_names = np.full(len(rows), "", dtype=object)
    outer_selection = []

    for outer_fold in unique_folds:
        outer_test = folds == outer_fold
        outer_train = ~outer_test
        inner_results = {}
        for family, profile in candidates:
            name = _name(family, profile)
            matrix = matrices[profile]
            inner_probability = _cross_fitted_probability(
                family,
                matrix[outer_train],
                target[outer_train],
                folds[outer_train],
                disagreement[outer_train],
            )
            thresholds, inner_metrics = choose_direction_thresholds(
                labels[outer_train], knn[outer_train], agent[outer_train], inner_probability
            )
            estimator = _fit_estimator(
                family, matrix[outer_train][disagreement[outer_train]], target[outer_train][disagreement[outer_train]]
            )
            outer_probability = _positive_probability(estimator, matrix[outer_test])
            candidate_probability[name][outer_test] = outer_probability
            candidate_t01[name][outer_test] = thresholds["knn0_agent1"]
            candidate_t10[name][outer_test] = thresholds["knn1_agent0"]
            inner_results[name] = {
                "family": family,
                "profile": profile,
                "probability": outer_probability,
                "thresholds": thresholds,
                "metrics": inner_metrics,
            }
        selected_name = max(names, key=lambda name: _selection_key(inner_results[name]["metrics"], inner_results[name]["family"], inner_results[name]["profile"]))
        selected = inner_results[selected_name]
        selected_probability[outer_test] = selected["probability"]
        selected_t01[outer_test] = selected["thresholds"]["knn0_agent1"]
        selected_t10[outer_test] = selected["thresholds"]["knn1_agent0"]
        selected_names[outer_test] = selected_name
        outer_selection.append(
            {
                "outer_fold": outer_fold,
                "selected_candidate": selected_name,
                "thresholds": selected["thresholds"],
                "inner_candidates": {
                    name: {
                        "accuracy": result["metrics"]["router"]["accuracy"],
                        "macro_f1": result["metrics"]["router"]["macro_f1"],
                        "switch_rate": result["metrics"]["router"]["switch_rate"],
                        "thresholds": result["thresholds"],
                    }
                    for name, result in inner_results.items()
                },
            }
        )

    candidate_metrics = {}
    for name in names:
        pred, route = route_with_direction_thresholds(
            knn, agent, candidate_probability[name], candidate_t01[name], candidate_t10[name]
        )
        candidate_metrics[name] = evaluation_metrics(labels, knn, agent, pred)
        candidate_metrics[name]["router"]["route_count"] = int(route.sum())
    selected_pred, selected_route = route_with_direction_thresholds(
        knn, agent, selected_probability, selected_t01, selected_t10
    )
    nested_candidate = evaluation_metrics(labels, knn, agent, selected_pred)
    nested_candidate["router"]["route_count"] = int(selected_route.sum())
    promotion = evaluate_promotion_gate(labels, knn, selected_pred, seed=RANDOM_SEED)
    deployed_pred = selected_pred if promotion["passed"] else knn.copy()
    nested_deployed = evaluation_metrics(labels, knn, agent, deployed_pred)

    deployment_candidates = {}
    for family, profile in candidates:
        name = _name(family, profile)
        probability = _cross_fitted_probability(
            family, matrices[profile], target, folds, disagreement
        )
        thresholds, metrics = choose_direction_thresholds(labels, knn, agent, probability)
        deployment_candidates[name] = {
            "family": family,
            "profile": profile,
            "thresholds": thresholds,
            "metrics": metrics,
        }
    deployment_name = max(
        names,
        key=lambda name: _selection_key(
            deployment_candidates[name]["metrics"],
            deployment_candidates[name]["family"],
            deployment_candidates[name]["profile"],
        ),
    )
    deployment = deployment_candidates[deployment_name]
    estimator = _fit_estimator(
        deployment["family"],
        matrices[deployment["profile"]][disagreement],
        target[disagreement],
    )
    deployed_thresholds = (
        deployment["thresholds"]
        if promotion["passed"]
        else {"knn0_agent1": 1.01, "knn1_agent0": 1.01}
    )
    model = {
        "estimator": estimator,
        "feature_profile": deployment["profile"],
        "feature_columns": list(POST_FEATURE_PROFILES[deployment["profile"]]),
        "candidate_thresholds": deployment["thresholds"],
        "thresholds": deployed_thresholds,
        "promoted": bool(promotion["passed"]),
        "fallback": "knn",
    }
    artifact_dir = task_root(output_root, spec.task) / POST_ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / "model.joblib")
    prediction_rows = []
    for index, row in enumerate(rows):
        prediction_rows.append(
            {
                "schema_version": POST_SELECTOR_SCHEMA_VERSION,
                "task": spec.task,
                "fold": int(folds[index]),
                "train_index": int(row["train_index"]),
                "Y": int(labels[index]),
                "knn_prediction": int(knn[index]),
                "agent_prediction": int(agent[index]),
                "paired_outcome": row["paired_outcome"],
                "selected_candidate": str(selected_names[index]),
                "agent_win_probability": float(selected_probability[index]),
                "candidate_threshold_knn0_agent1": float(selected_t01[index]),
                "candidate_threshold_knn1_agent0": float(selected_t10[index]),
                "candidate_route_to_agent": bool(selected_route[index]),
                "route_to_agent": bool(selected_route[index] and promotion["passed"]),
                "candidate_selector_prediction": int(selected_pred[index]),
                "selector_prediction": int(deployed_pred[index]),
            }
        )
    write_jsonl_atomic(artifact_dir / "nested_oof_predictions.jsonl", prediction_rows)
    metrics = {
        "schema_version": POST_SELECTOR_SCHEMA_VERSION,
        "task": spec.task,
        "n_rows": len(rows),
        "n_disagreements": int(disagreement.sum()),
        "feature_columns": list(POST_FEATURE_COLUMNS),
        "feature_profiles": {key: list(value) for key, value in POST_FEATURE_PROFILES.items()},
        "target": "agent_only_correct vs knn_only_correct on disagreements only",
        "nested_candidate": nested_candidate,
        "nested_deployed": nested_deployed,
        "nested_candidate_metrics": candidate_metrics,
        "outer_selection": outer_selection,
        "outer_selection_counts": dict(Counter(selected_names.tolist())),
        "promotion_gate": promotion,
        "deployment": {
            "candidate": deployment_name,
            "model_family": deployment["family"],
            "feature_profile": deployment["profile"],
            "candidate_thresholds": deployment["thresholds"],
            "thresholds": deployed_thresholds,
            "promoted": bool(promotion["passed"]),
        },
    }
    write_json_atomic(artifact_dir / "metrics.json", metrics)
    write_json_atomic(
        artifact_dir / "model_manifest.json",
        {
            "schema_version": POST_SELECTOR_SCHEMA_VERSION,
            "task": spec.task,
            "model_family": deployment["family"],
            "feature_profile": deployment["profile"],
            "feature_columns": list(POST_FEATURE_PROFILES[deployment["profile"]]),
            "available_feature_columns": list(POST_FEATURE_COLUMNS),
            "candidate_thresholds": deployment["thresholds"],
            "thresholds": deployed_thresholds,
            "promoted": bool(promotion["passed"]),
            "promotion_gate": promotion,
            "training_rows": len(rows),
            "training_disagreements": int(disagreement.sum()),
            "artifact": str(artifact_dir / "model.joblib"),
        },
    )
    (artifact_dir / "report_zh.md").write_text(_render_report(metrics), encoding="utf-8")
    return metrics


def choose_direction_thresholds(
    labels: np.ndarray,
    knn: np.ndarray,
    agent: np.ndarray,
    probability: np.ndarray,
) -> tuple[dict[str, float], dict[str, Any]]:
    best = None
    baseline = evaluation_metrics(labels, knn, agent, knn)
    minimum_accuracy = baseline["always_knn"]["accuracy"] - MAX_ACCURACY_DROP
    # The two disagreement directions modify disjoint subsets of the KNN
    # confusion matrix.  Precompute each direction's delta at every threshold
    # instead of invoking sklearn metrics for every Cartesian-product pair.
    # This is exactly equivalent to materializing all routed predictions, but
    # reduces threshold selection from O(T^2 * n) to O(T * n + T^2).
    base_tn = int(((labels == 0) & (knn == 0)).sum())
    base_fp = int(((labels == 0) & (knn == 1)).sum())
    base_fn = int(((labels == 1) & (knn == 0)).sum())
    base_tp = int(((labels == 1) & (knn == 1)).sum())
    direction_stats: dict[str, list[tuple[int, int, int, int, int]]] = {
        "knn0_agent1": [],
        "knn1_agent0": [],
    }
    direction_01 = (knn == 0) & (agent == 1)
    direction_10 = (knn == 1) & (agent == 0)
    for threshold in THRESHOLD_GRID:
        route_01 = direction_01 & (probability >= threshold)
        route_10 = direction_10 & (probability >= threshold)
        # Switching 0 -> 1 changes TN to FP for y=0 and FN to TP for y=1.
        zero_01 = int((route_01 & (labels == 0)).sum())
        one_01 = int((route_01 & (labels == 1)).sum())
        direction_stats["knn0_agent1"].append(
            (-zero_01, zero_01, -one_01, one_01, int(route_01.sum()))
        )
        # Switching 1 -> 0 changes FP to TN for y=0 and TP to FN for y=1.
        zero_10 = int((route_10 & (labels == 0)).sum())
        one_10 = int((route_10 & (labels == 1)).sum())
        direction_stats["knn1_agent0"].append(
            (zero_10, -zero_10, one_10, -one_10, int(route_10.sum()))
        )
    for index_01, threshold_01 in enumerate(THRESHOLD_GRID):
        delta_01 = direction_stats["knn0_agent1"][index_01]
        for index_10, threshold_10 in enumerate(THRESHOLD_GRID):
            delta_10 = direction_stats["knn1_agent0"][index_10]
            tn = base_tn + delta_01[0] + delta_10[0]
            fp = base_fp + delta_01[1] + delta_10[1]
            fn = base_fn + delta_01[2] + delta_10[2]
            tp = base_tp + delta_01[3] + delta_10[3]
            accuracy = (tn + tp) / len(labels)
            if accuracy + 1e-12 < minimum_accuracy:
                continue
            f1_negative_denominator = 2 * tn + fp + fn
            f1_positive_denominator = 2 * tp + fp + fn
            f1_negative = 0.0 if f1_negative_denominator == 0 else 2 * tn / f1_negative_denominator
            f1_positive = 0.0 if f1_positive_denominator == 0 else 2 * tp / f1_positive_denominator
            macro_f1 = (f1_negative + f1_positive) / 2
            route_count = delta_01[4] + delta_10[4]
            key = (
                macro_f1,
                accuracy,
                -(route_count / len(labels)),
            )
            if best is None or key > best[0]:
                best = (key, threshold_01, threshold_10)
    if best is None:
        raise AssertionError("Direction threshold grid produced no guardrail candidate")
    best_thresholds = {
        "knn0_agent1": float(best[1]),
        "knn1_agent0": float(best[2]),
    }
    best_prediction, best_route = route_with_direction_thresholds(
        knn,
        agent,
        probability,
        best_thresholds["knn0_agent1"],
        best_thresholds["knn1_agent0"],
    )
    best_metrics = evaluation_metrics(labels, knn, agent, best_prediction)
    best_metrics["router"]["route_count"] = int(best_route.sum())
    return best_thresholds, best_metrics


def route_with_direction_thresholds(
    knn: np.ndarray,
    agent: np.ndarray,
    probability: np.ndarray,
    threshold_01: float | np.ndarray,
    threshold_10: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    route = (
        ((knn == 0) & (agent == 1) & (probability >= threshold_01))
        | ((knn == 1) & (agent == 0) & (probability >= threshold_10))
    )
    return np.where(route, agent, knn), route


def predict_post_selector(model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    matrix = _matrix(rows, model["feature_columns"])
    return _positive_probability(model["estimator"], matrix)


def _cross_fitted_probability(
    family: str,
    matrix: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    disagreement: np.ndarray,
) -> np.ndarray:
    probability = np.full(len(target), 0.5)
    for fold in sorted(set(folds.tolist())):
        train = (folds != fold) & disagreement
        test = (folds == fold) & disagreement
        if not test.any():
            continue
        estimator = _fit_estimator(family, matrix[train], target[train])
        probability[test] = _positive_probability(estimator, matrix[test])
    return probability


def _fit_estimator(family: str, matrix: np.ndarray, target: np.ndarray):
    if len(set(target.tolist())) < 2:
        return DummyClassifier(strategy="constant", constant=int(target[0])).fit(matrix, target)
    if family == "logistic_regression":
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED)),
            ]
        )
    elif family == "hist_gradient_boosting":
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, min_samples_leaf=20, learning_rate=0.05, l2_regularization=2.0, random_state=RANDOM_SEED)),
            ]
        )
    else:
        raise ValueError(f"Unknown post-selector family: {family}")
    return estimator.fit(matrix, target)


def _positive_probability(estimator, matrix: np.ndarray) -> np.ndarray:
    probabilities = estimator.predict_proba(matrix)
    classes = list(estimator.classes_)
    if 1 not in classes:
        return np.zeros(len(matrix))
    return probabilities[:, classes.index(1)]


def _matrix(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(row["features"][column]) for column in columns] for row in rows], dtype=float)


def _selection_key(metrics: Mapping[str, Any], family: str, profile: str) -> tuple:
    return (
        float(metrics["router"]["macro_f1"]),
        float(metrics["router"]["accuracy"]),
        -float(metrics["router"]["switch_rate"]),
        family == "logistic_regression",
        -len(POST_FEATURE_PROFILES[profile]),
    )


def _name(family: str, profile: str) -> str:
    return f"{family}__{profile}"


def _render_report(metrics: Mapping[str, Any]) -> str:
    candidate = metrics["nested_candidate"]["router"]
    knn = metrics["nested_candidate"]["always_knn"]
    deployment = metrics["deployment"]
    return (
        f"# {metrics['task']} post-selector v3 train-only nested OOF\n\n"
        f"- rows / disagreements: {metrics['n_rows']} / {metrics['n_disagreements']}\n"
        f"- selected: `{deployment['candidate']}`\n"
        f"- promoted: `{deployment['promoted']}`\n"
        f"- KNN accuracy / macro-F1: {knn['accuracy']:.4f} / {knn['macro_f1']:.4f}\n"
        f"- candidate accuracy / macro-F1: {candidate['accuracy']:.4f} / {candidate['macro_f1']:.4f}\n"
    )
