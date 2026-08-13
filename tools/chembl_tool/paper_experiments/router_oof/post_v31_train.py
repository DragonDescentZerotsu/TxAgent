"""Direction-specific calibrated-ensemble post-selector v3.1.

This module reuses the frozen v3 feature bank but changes the score-generation
contract.  Each disagreement direction has an independent model/profile,
fold-heldout sigmoid calibration, and an accuracy-first routed-risk gate.
"""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic

from .contract import TaskSpec, task_root
from .io import read_jsonl as _read_jsonl
from .post_contract import (
    POST_FEATURE_STEM,
    POST_SELECTOR_DIRECTIONS,
    POST_V31_ARTIFACT_DIR,
    POST_V31_SELECTOR_SCHEMA_VERSION,
)
from .post_features import POST_FEATURE_COLUMNS, POST_FEATURE_PROFILES
from .post_train import (
    POST_MODEL_FAMILIES,
    RANDOM_SEED,
    THRESHOLD_GRID,
    _fit_estimator,
    _matrix,
    _positive_probability,
)
from .train import evaluation_metrics, paired_bootstrap_deltas


# Compatibility alias for existing artifacts and callers; the semantic
# contract lives in post_contract.py so diagnostics do not depend on fitting.
DIRECTIONS = POST_SELECTOR_DIRECTIONS
V31_FEATURE_PROFILES = {
    name: tuple(
        column
        for column in columns
        if column not in {"direction_knn0_agent1", "direction_knn1_agent0"}
    )
    for name, columns in POST_FEATURE_PROFILES.items()
}
MIN_ROUTED_ROWS = 20
WILSON_ONE_SIDED_Z = 1.6448536269514722
MAX_MACRO_F1_DROP = 0.005


class ConstantCalibrator:
    """Serializable calibration fallback for a one-class calibration fold."""

    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        return np.full(len(raw_probability), self.probability, dtype=float)


class SigmoidCalibrator:
    """Platt-style calibration over clipped raw-probability logits."""

    def __init__(self, model: LogisticRegression):
        self.model = model

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        transformed = _probability_logit(raw_probability).reshape(-1, 1)
        return self.model.predict_proba(transformed)[:, 1]


class DirectionCalibratedEnsemble:
    """Average fold-heldout calibrated estimator components."""

    def __init__(self, components: Sequence[tuple[Any, Any]]):
        if not components:
            raise ValueError("Calibrated ensemble requires at least one component")
        self.components = list(components)

    def predict_probability(self, matrix: np.ndarray) -> np.ndarray:
        calibrated = []
        for estimator, calibrator in self.components:
            raw = _positive_probability(estimator, matrix)
            calibrated.append(calibrator.predict(raw))
        return np.mean(np.vstack(calibrated), axis=0)


def train_post_selector_v31(
    spec: TaskSpec,
    *,
    output_root: str | Path,
    model_families: Iterable[str] = POST_MODEL_FAMILIES,
    feature_profiles: Iterable[str] = tuple(V31_FEATURE_PROFILES),
) -> dict[str, Any]:
    root = task_root(output_root, spec.task)
    rows = _read_jsonl(root / f"{POST_FEATURE_STEM}.jsonl")
    if not rows:
        raise ValueError(f"No post-selector features for {spec.task}")
    families = tuple(model_families)
    profiles = tuple(feature_profiles)
    if set(families) - set(POST_MODEL_FAMILIES):
        raise ValueError(f"Unknown model families: {sorted(set(families) - set(POST_MODEL_FAMILIES))}")
    if set(profiles) - set(V31_FEATURE_PROFILES):
        raise ValueError(f"Unknown feature profiles: {sorted(set(profiles) - set(V31_FEATURE_PROFILES))}")

    folds = np.asarray([int(row["fold"]) for row in rows], dtype=int)
    unique_folds = sorted(set(folds.tolist()))
    labels = np.asarray([int(row["Y"]) for row in rows], dtype=int)
    knn = np.asarray([int(row["knn_prediction"]) for row in rows], dtype=int)
    agent = np.asarray([int(row["agent_prediction"]) for row in rows], dtype=int)
    target = np.asarray(
        [int(row["paired_outcome"] == "agent_only_correct") for row in rows],
        dtype=int,
    )
    direction_masks = {
        name: (knn == pair[0]) & (agent == pair[1])
        for name, pair in DIRECTIONS.items()
    }
    for name, mask in direction_masks.items():
        if int(mask.sum()) < MIN_ROUTED_ROWS or len(set(target[mask].tolist())) < 2:
            raise ValueError(f"Insufficient two-class direction rows for {spec.task}/{name}")

    matrices = {
        profile: _matrix(rows, V31_FEATURE_PROFILES[profile])
        for profile in profiles
    }
    candidates = [(family, profile) for family in families for profile in profiles]
    candidate_names = [_candidate_name(family, profile) for family, profile in candidates]

    selected_probability = np.full(len(rows), 0.5, dtype=float)
    selected_threshold = np.full(len(rows), 1.01, dtype=float)
    selected_candidate = np.full(len(rows), "", dtype=object)
    nested_prior_route = np.zeros(len(rows), dtype=bool)
    outer_selection: list[dict[str, Any]] = []
    deployment_scores = {
        direction: {name: np.full(len(rows), 0.5, dtype=float) for name in candidate_names}
        for direction in DIRECTIONS
    }

    for outer_fold in unique_folds:
        outer_test = folds == outer_fold
        outer_train = ~outer_test
        outer_record: dict[str, Any] = {"outer_fold": int(outer_fold), "directions": {}}
        for direction, mask in direction_masks.items():
            train_mask = outer_train & mask
            test_mask = outer_test & mask
            inner_scores: dict[str, np.ndarray] = {}
            external_scores: dict[str, np.ndarray] = {}
            policies: dict[str, dict[str, Any]] = {}
            for family, profile in candidates:
                name = _candidate_name(family, profile)
                matrix = matrices[profile]
                inner_score = _nested_calibrated_scores(
                    family,
                    matrix,
                    target,
                    folds,
                    train_mask,
                )
                inner_scores[name] = inner_score
                policies[name] = choose_safe_direction_policy(
                    target,
                    inner_score,
                    train_mask,
                    family=family,
                    profile=profile,
                )
                ensemble = fit_direction_calibrated_ensemble(
                    family,
                    matrix,
                    target,
                    folds,
                    train_mask,
                )
                predicted = ensemble.predict_probability(matrix[test_mask]) if test_mask.any() else np.asarray([])
                external_scores[name] = predicted
                deployment_scores[direction][name][test_mask] = predicted

            chosen_name = max(
                candidate_names,
                key=lambda name: _policy_selection_key(policies[name]),
            )
            chosen_policy = policies[chosen_name]
            chosen_external = external_scores[chosen_name]
            selected_probability[test_mask] = chosen_external
            selected_threshold[test_mask] = float(chosen_policy["threshold"])
            selected_candidate[test_mask] = chosen_name
            outer_record["directions"][direction] = {
                "selected_candidate": chosen_name,
                "selected_policy": chosen_policy,
                "candidates": policies,
            }

            prior = _prior_safe_policy(target, train_mask)
            if prior["route"]:
                nested_prior_route[test_mask] = True
        outer_selection.append(outer_record)

    nested_route = (
        (direction_masks["knn0_agent1"] | direction_masks["knn1_agent0"])
        & (selected_probability >= selected_threshold)
    )
    nested_prediction = np.where(nested_route, agent, knn)
    nested_metrics = evaluation_metrics(labels, knn, agent, nested_prediction)
    nested_metrics["router"]["route_count"] = int(nested_route.sum())
    promotion = evaluate_accuracy_first_promotion_gate(
        labels,
        knn,
        nested_prediction,
        seed=RANDOM_SEED,
    )
    nested_deployed_prediction = nested_prediction if promotion["passed"] else knn.copy()
    nested_deployed = evaluation_metrics(labels, knn, agent, nested_deployed_prediction)

    deployment: dict[str, Any] = {}
    final_model_directions: dict[str, Any] = {}
    deployment_probability = np.full(len(rows), 0.5, dtype=float)
    candidate_thresholds: dict[str, float] = {}
    for direction, mask in direction_masks.items():
        policies = {}
        for family, profile in candidates:
            name = _candidate_name(family, profile)
            policies[name] = choose_safe_direction_policy(
                target,
                deployment_scores[direction][name],
                mask,
                family=family,
                profile=profile,
            )
        chosen_name = max(candidate_names, key=lambda name: _policy_selection_key(policies[name]))
        chosen = policies[chosen_name]
        family, profile = chosen_name.split("__", 1)
        ensemble = fit_direction_calibrated_ensemble(
            family,
            matrices[profile],
            target,
            folds,
            mask,
        )
        deployment_probability[mask] = deployment_scores[direction][chosen_name][mask]
        candidate_thresholds[direction] = float(chosen["threshold"])
        deployment[direction] = {
            "candidate": chosen_name,
            "model_family": family,
            "feature_profile": profile,
            "policy": chosen,
            "all_candidates": policies,
        }
        final_model_directions[direction] = {
            "ensemble": ensemble,
            "model_family": family,
            "feature_profile": profile,
            "feature_columns": list(V31_FEATURE_PROFILES[profile]),
        }

    deployment_candidate_route = (
        (direction_masks["knn0_agent1"] & (deployment_probability >= candidate_thresholds["knn0_agent1"]))
        | (direction_masks["knn1_agent0"] & (deployment_probability >= candidate_thresholds["knn1_agent0"]))
    )
    deployment_candidate_prediction = np.where(deployment_candidate_route, agent, knn)
    deployment_candidate_metrics = evaluation_metrics(
        labels, knn, agent, deployment_candidate_prediction
    )
    deployment_candidate_metrics["router"]["route_count"] = int(deployment_candidate_route.sum())
    deployed_thresholds = (
        candidate_thresholds
        if promotion["passed"]
        else {direction: 1.01 for direction in DIRECTIONS}
    )

    route_01_only_prediction = np.where(direction_masks["knn0_agent1"], agent, knn)
    prior_prediction = np.where(nested_prior_route, agent, knn)
    baselines = {
        "route_knn0_agent1_only": evaluation_metrics(
            labels, knn, agent, route_01_only_prediction
        )["router"],
        "nested_prior_safe": evaluation_metrics(labels, knn, agent, prior_prediction)["router"],
    }
    pareto_frontier = build_safe_pareto_frontier(
        labels,
        knn,
        agent,
        target,
        deployment_probability,
        direction_masks,
    )

    model = {
        "schema_version": POST_V31_SELECTOR_SCHEMA_VERSION,
        "directions": final_model_directions,
        "candidate_thresholds": candidate_thresholds,
        "thresholds": deployed_thresholds,
        "promoted": bool(promotion["passed"]),
        "fallback": "knn",
        "calibration": "fold-heldout sigmoid ensemble",
    }
    artifact_dir = root / POST_V31_ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / "model.joblib")

    prediction_rows = []
    for index, row in enumerate(rows):
        direction = _row_direction(knn[index], agent[index])
        prediction_rows.append(
            {
                "schema_version": POST_V31_SELECTOR_SCHEMA_VERSION,
                "task": spec.task,
                "fold": int(folds[index]),
                "train_index": int(row["train_index"]),
                "Y": int(labels[index]),
                "knn_prediction": int(knn[index]),
                "agent_prediction": int(agent[index]),
                "paired_outcome": row["paired_outcome"],
                "direction": direction,
                "selected_candidate": str(selected_candidate[index]),
                "agent_win_probability": float(selected_probability[index]),
                "candidate_threshold": float(selected_threshold[index]),
                "candidate_route_to_agent": bool(nested_route[index]),
                "route_to_agent": bool(nested_route[index] and promotion["passed"]),
                "candidate_selector_prediction": int(nested_prediction[index]),
                "selector_prediction": int(nested_deployed_prediction[index]),
            }
        )
    write_jsonl_atomic(artifact_dir / "nested_oof_predictions.jsonl", prediction_rows)

    metrics = {
        "schema_version": POST_V31_SELECTOR_SCHEMA_VERSION,
        "task": spec.task,
        "n_rows": len(rows),
        "n_disagreements": int(sum(mask.sum() for mask in direction_masks.values())),
        "direction_counts": {name: int(mask.sum()) for name, mask in direction_masks.items()},
        "feature_columns": list(POST_FEATURE_COLUMNS),
        "feature_profiles": {key: list(value) for key, value in V31_FEATURE_PROFILES.items()},
        "target": "direction-specific agent_only_correct vs knn_only_correct",
        "risk_gate": {
            "minimum_routed_rows": MIN_ROUTED_ROWS,
            "one_sided_wilson_z": WILSON_ONE_SIDED_Z,
            "minimum_precision_lower_bound": 0.5,
        },
        "nested_candidate": nested_metrics,
        "nested_deployed": nested_deployed,
        "promotion_gate": promotion,
        "outer_selection": outer_selection,
        "outer_selection_counts": {
            direction: dict(
                Counter(
                    record["directions"][direction]["selected_candidate"]
                    for record in outer_selection
                )
            )
            for direction in DIRECTIONS
        },
        "deployment": deployment,
        "deployment_candidate_metrics": deployment_candidate_metrics,
        "candidate_thresholds": candidate_thresholds,
        "thresholds": deployed_thresholds,
        "baselines": baselines,
        "safe_pareto_frontier": pareto_frontier,
    }
    write_json_atomic(artifact_dir / "metrics.json", metrics)
    manifest = {
        "schema_version": POST_V31_SELECTOR_SCHEMA_VERSION,
        "task": spec.task,
        "calibration": "fold-heldout sigmoid ensemble",
        "directions": {
            direction: {
                "model_family": payload["model_family"],
                "feature_profile": payload["feature_profile"],
                "feature_columns": payload["feature_columns"],
                "candidate_threshold": candidate_thresholds[direction],
                "threshold": deployed_thresholds[direction],
                "n_components": len(payload["ensemble"].components),
            }
            for direction, payload in final_model_directions.items()
        },
        "promoted": bool(promotion["passed"]),
        "promotion_gate": promotion,
        "training_rows": len(rows),
        "training_disagreements": int(sum(mask.sum() for mask in direction_masks.values())),
        "artifact": str(artifact_dir / "model.joblib"),
    }
    write_json_atomic(artifact_dir / "model_manifest.json", manifest)
    (artifact_dir / "report_zh.md").write_text(_render_report(metrics), encoding="utf-8")
    return metrics


def fit_direction_calibrated_ensemble(
    family: str,
    matrix: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    eligible: np.ndarray,
) -> DirectionCalibratedEnsemble:
    components = []
    for calibration_fold in sorted(set(folds[eligible].tolist())):
        calibration = eligible & (folds == calibration_fold)
        training = eligible & (folds != calibration_fold)
        if not calibration.any() or not training.any():
            continue
        estimator = _fit_estimator(family, matrix[training], target[training])
        raw = _positive_probability(estimator, matrix[calibration])
        calibrator = _fit_calibrator(raw, target[calibration])
        components.append((estimator, calibrator))
    if not components:
        raise ValueError("No calibrated ensemble components could be fit")
    return DirectionCalibratedEnsemble(components)


def predict_post_selector_v31(
    model: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    knn: np.ndarray,
    agent: np.ndarray,
) -> np.ndarray:
    probability = np.full(len(rows), 0.5, dtype=float)
    for direction, pair in DIRECTIONS.items():
        payload = model["directions"][direction]
        mask = (knn == pair[0]) & (agent == pair[1])
        if not mask.any():
            continue
        matrix = _matrix(rows, payload["feature_columns"])
        probability[mask] = payload["ensemble"].predict_probability(matrix[mask])
    return probability


def choose_safe_direction_policy(
    target: np.ndarray,
    probability: np.ndarray,
    eligible: np.ndarray,
    *,
    family: str,
    profile: str,
) -> dict[str, Any]:
    y = target[eligible]
    p = probability[eligible]
    auc = float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else 0.5
    brier = float(np.mean((p - y) ** 2)) if len(y) else 0.0
    best = {
        "family": family,
        "profile": profile,
        "threshold": 1.01,
        "route_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "net_rescues": 0,
        "precision": 0.0,
        "precision_wilson_lower": 0.5,
        "auc": auc,
        "brier": brier,
        "risk_gate_passed": False,
    }
    for threshold in THRESHOLD_GRID:
        route = p >= threshold
        route_count = int(route.sum())
        if route_count < MIN_ROUTED_ROWS:
            continue
        wins = int(y[route].sum())
        losses = route_count - wins
        lower = wilson_lower_bound(wins, route_count)
        if lower <= 0.5:
            continue
        candidate = {
            "family": family,
            "profile": profile,
            "threshold": float(threshold),
            "route_count": route_count,
            "win_count": wins,
            "loss_count": losses,
            "net_rescues": wins - losses,
            "precision": wins / route_count,
            "precision_wilson_lower": lower,
            "auc": auc,
            "brier": brier,
            "risk_gate_passed": True,
        }
        if _policy_selection_key(candidate) > _policy_selection_key(best):
            best = candidate
    return best


def wilson_lower_bound(successes: int, total: int) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    z2 = WILSON_ONE_SIDED_Z**2
    denominator = 1.0 + z2 / total
    center = proportion + z2 / (2 * total)
    radius = WILSON_ONE_SIDED_Z * math.sqrt(
        proportion * (1.0 - proportion) / total + z2 / (4 * total**2)
    )
    return float((center - radius) / denominator)


def evaluate_accuracy_first_promotion_gate(
    labels: np.ndarray,
    knn: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
    repetitions: int = 2000,
) -> dict[str, Any]:
    uncertainty = paired_bootstrap_deltas(
        labels,
        knn,
        candidate,
        seed=seed,
        repetitions=repetitions,
    )
    accuracy_ci = uncertainty["accuracy_delta_ci95"]
    macro_ci = uncertainty["macro_f1_delta_ci95"]
    passed = bool(
        uncertainty["observed_accuracy_delta"] > 0
        and accuracy_ci[0] > 0
    )
    macro_guardrail_passed = bool(macro_ci[0] >= -MAX_MACRO_F1_DROP)
    reasons = []
    if uncertainty["observed_accuracy_delta"] <= 0:
        reasons.append("observed_accuracy_delta_not_positive")
    if accuracy_ci[0] <= 0:
        reasons.append("accuracy_delta_lower_ci_not_positive")
    return {
        "passed": passed,
        "fallback_when_failed": "knn",
        "primary_metric": "accuracy",
        "maximum_macro_f1_drop": MAX_MACRO_F1_DROP,
        "macro_f1_guardrail_passed": macro_guardrail_passed,
        **uncertainty,
        "failure_reasons": reasons,
    }


def build_safe_pareto_frontier(
    labels: np.ndarray,
    knn: np.ndarray,
    agent: np.ndarray,
    target: np.ndarray,
    probability: np.ndarray,
    direction_masks: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    safe_thresholds: dict[str, list[float]] = {}
    for direction, mask in direction_masks.items():
        thresholds = [1.01]
        y = target[mask]
        p = probability[mask]
        for threshold in THRESHOLD_GRID:
            route = p >= threshold
            n_route = int(route.sum())
            if n_route < MIN_ROUTED_ROWS:
                continue
            if wilson_lower_bound(int(y[route].sum()), n_route) > 0.5:
                thresholds.append(float(threshold))
        safe_thresholds[direction] = sorted(set(thresholds))
    points = []
    for threshold_01 in safe_thresholds["knn0_agent1"]:
        for threshold_10 in safe_thresholds["knn1_agent0"]:
            route = (
                (direction_masks["knn0_agent1"] & (probability >= threshold_01))
                | (direction_masks["knn1_agent0"] & (probability >= threshold_10))
            )
            prediction = np.where(route, agent, knn)
            metrics = evaluation_metrics(labels, knn, agent, prediction)["router"]
            points.append(
                {
                    "threshold_knn0_agent1": threshold_01,
                    "threshold_knn1_agent0": threshold_10,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "switch_count": metrics["switch_count"],
                }
            )
    frontier = []
    for point in points:
        dominated = any(
            other["accuracy"] >= point["accuracy"]
            and other["macro_f1"] >= point["macro_f1"]
            and (
                other["accuracy"] > point["accuracy"]
                or other["macro_f1"] > point["macro_f1"]
            )
            for other in points
        )
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda row: (-row["accuracy"], -row["macro_f1"], row["switch_count"]))


def _nested_calibrated_scores(
    family: str,
    matrix: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    output = np.full(len(target), 0.5, dtype=float)
    for heldout_fold in sorted(set(folds[eligible].tolist())):
        test = eligible & (folds == heldout_fold)
        training_pool = eligible & (folds != heldout_fold)
        if not test.any() or not training_pool.any():
            continue
        ensemble = fit_direction_calibrated_ensemble(
            family,
            matrix,
            target,
            folds,
            training_pool,
        )
        output[test] = ensemble.predict_probability(matrix[test])
    return output


def _fit_calibrator(raw: np.ndarray, target: np.ndarray) -> Any:
    if len(set(target.tolist())) < 2:
        probability = (float(target.sum()) + 1.0) / (len(target) + 2.0)
        return ConstantCalibrator(probability)
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)
    model.fit(_probability_logit(raw).reshape(-1, 1), target)
    return SigmoidCalibrator(model)


def _probability_logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _prior_safe_policy(target: np.ndarray, eligible: np.ndarray) -> dict[str, Any]:
    n = int(eligible.sum())
    wins = int(target[eligible].sum())
    lower = wilson_lower_bound(wins, n)
    return {
        "route": bool(n >= MIN_ROUTED_ROWS and lower > 0.5),
        "n": n,
        "wins": wins,
        "precision": wins / n if n else 0.0,
        "precision_wilson_lower": lower,
    }


def _policy_selection_key(policy: Mapping[str, Any]) -> tuple:
    return (
        int(policy["net_rescues"]),
        float(policy["precision_wilson_lower"]),
        float(policy["auc"]),
        -float(policy["brier"]),
        policy["family"] == "logistic_regression",
        -len(V31_FEATURE_PROFILES[policy["profile"]]),
        -int(policy["route_count"]),
    )


def _candidate_name(family: str, profile: str) -> str:
    return f"{family}__{profile}"


def _row_direction(knn: int, agent: int) -> str:
    for name, pair in DIRECTIONS.items():
        if (knn, agent) == pair:
            return name
    return "agreement"


def _render_report(metrics: Mapping[str, Any]) -> str:
    nested = metrics["nested_candidate"]
    lines = [
        f"# {metrics['task']} post-selector v3.1 train OOF",
        "",
        f"- promotion gate: `{metrics['promotion_gate']['passed']}`",
        f"- KNN accuracy / macro-F1: {nested['always_knn']['accuracy']:.4f} / {nested['always_knn']['macro_f1']:.4f}",
        f"- nested candidate accuracy / macro-F1: {nested['router']['accuracy']:.4f} / {nested['router']['macro_f1']:.4f}",
        f"- switches / rescues / harms: {nested['router']['switch_count']} / {nested['router']['rescue_count']} / {nested['router']['harm_count']}",
        "",
        "## Deployment directions",
        "",
    ]
    for direction, payload in metrics["deployment"].items():
        lines.append(
            f"- `{direction}`: `{payload['candidate']}`, threshold={payload['policy']['threshold']:.3f}, "
            f"OOF routed precision={payload['policy']['precision']:.3f}, "
            f"Wilson lower={payload['policy']['precision_wilson_lower']:.3f}"
        )
    return "\n".join(lines) + "\n"
