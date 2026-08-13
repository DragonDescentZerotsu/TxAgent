"""Train-only cross-task transfer diagnosis for post-selector v3.1.

The shared layer is deliberately narrow: direction-specific, task-balanced
logistic regression over the full generic v3 feature profile.  Calibration,
threshold selection, and risk gates remain target-task specific. Cross-task
rows sharing identity or scaffold/fold-group with the target heldout fold are
excluded before every fit.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic

from .contract import TASK_SPECS, TaskSpec, task_root
from .post_contract import (
    POST_FEATURE_STEM,
    POST_V31_LEARNING_CURVE_ARTIFACT_DIR,
    POST_V31_TRANSFER_ARTIFACT_DIR,
    POST_V31_TRANSFER_SCHEMA_VERSION,
)
from .post_train import _matrix
from .post_v31_common import PostSelectorArrays, direction_only_prediction, read_jsonl
from .post_v31_train import (
    DIRECTIONS,
    V31_FEATURE_PROFILES,
    _fit_calibrator,
    choose_safe_direction_policy,
)
from .train import evaluation_metrics, paired_bootstrap_deltas


TRANSFER_PROFILE = "output_query_knn_evidence_trace"
TRANSFER_FAMILY = "task_balanced_logistic_regression"


@dataclass(frozen=True)
class TransferData:
    task_names: tuple[str, ...]
    rows_by_task: Mapping[str, list[dict[str, Any]]]
    global_task: np.ndarray
    global_fold: np.ndarray
    global_target: np.ndarray
    global_knn: np.ndarray
    global_agent: np.ndarray
    global_identity: np.ndarray
    global_group: np.ndarray
    global_matrix: np.ndarray
    task_slices: Mapping[str, slice]


class SharedLogisticModel:
    """Small serializable scaler/logistic pair with a one-class fallback."""

    def __init__(
        self,
        scaler: StandardScaler | None,
        estimator: LogisticRegression | None,
        constant_probability: float | None = None,
    ):
        self.scaler = scaler
        self.estimator = estimator
        self.constant_probability = constant_probability

    def predict_probability(self, matrix: np.ndarray) -> np.ndarray:
        if self.constant_probability is not None:
            return np.full(len(matrix), self.constant_probability, dtype=float)
        assert self.scaler is not None and self.estimator is not None
        return self.estimator.predict_proba(self.scaler.transform(matrix))[:, 1]


def run_post_v31_transfer_diagnosis(
    *,
    output_root: str | Path,
    workers: int = 1,
    bootstrap_repetitions: int = 2000,
) -> dict[str, Any]:
    """Evaluate one frozen shared representation without reading valid/test."""

    specs = [TASK_SPECS[name] for name in sorted(TASK_SPECS)]
    data = _load_transfer_data(specs, output_root=output_root)
    jobs = [
        (spec.task, outer_fold)
        for spec in specs
        for outer_fold in sorted(
            set(data.global_fold[data.global_task == spec.task].tolist())
        )
    ]
    max_workers = max(1, min(int(workers), len(jobs)))
    with threadpool_limits(limits=1), parallel_config(
        backend="loky", inner_max_num_threads=1
    ):
        completed = Parallel(
            n_jobs=max_workers,
            return_as="generator_unordered",
            max_nbytes="1M",
            mmap_mode="r",
        )(
            delayed(_run_target_outer_fold)(
                data,
                target_task=task,
                outer_fold=outer_fold,
            )
            for task, outer_fold in jobs
        )
        fold_results = list(completed)

    fold_results.sort(key=lambda row: (row["task"], row["outer_fold"]))
    task_results = []
    for spec in specs:
        task = spec.task
        task_rows = data.rows_by_task[task]
        task_slice = data.task_slices[task]
        labels = np.asarray([int(row["Y"]) for row in task_rows], dtype=int)
        knn = data.global_knn[task_slice]
        agent = data.global_agent[task_slice]
        shared = knn.copy()
        probability = np.full(len(labels), 0.5, dtype=float)
        route = np.zeros(len(labels), dtype=bool)
        records = []
        for fold_result in (row for row in fold_results if row["task"] == task):
            indices = np.asarray(fold_result["target_local_indices"], dtype=int)
            shared[indices] = np.asarray(fold_result["prediction"], dtype=int)
            probability[indices] = np.asarray(fold_result["probability"], dtype=float)
            route[indices] = np.asarray(fold_result["route"], dtype=bool)
            records.append(
                {
                    key: value
                    for key, value in fold_result.items()
                    if key
                    not in {
                        "target_local_indices",
                        "prediction",
                        "probability",
                        "route",
                    }
                }
            )
        comparator = direction_only_prediction(knn, agent)
        local = _load_full_local_curve_prediction(
            task_root(output_root, task), expected_rows=len(labels)
        )
        metrics = evaluation_metrics(labels, knn, agent, shared)
        metrics["router"]["route_count"] = int(route.sum())
        direction_metrics = evaluation_metrics(labels, knn, agent, comparator)["router"]
        local_metrics = evaluation_metrics(labels, knn, agent, local)["router"]
        vs_direction = paired_bootstrap_deltas(
            labels,
            comparator,
            shared,
            seed=20260805,
            repetitions=bootstrap_repetitions,
        )
        vs_local = paired_bootstrap_deltas(
            labels,
            local,
            shared,
            seed=20260806,
            repetitions=bootstrap_repetitions,
        )
        artifact_dir = task_root(output_root, task) / POST_V31_TRANSFER_ARTIFACT_DIR
        artifact_dir.mkdir(parents=True, exist_ok=True)
        prediction_rows = [
            {
                "schema_version": POST_V31_TRANSFER_SCHEMA_VERSION,
                "task": task,
                "train_index": int(row["train_index"]),
                "fold": int(row["fold"]),
                "Y": int(labels[index]),
                "knn_prediction": int(knn[index]),
                "agent_prediction": int(agent[index]),
                "direction_only_prediction": int(comparator[index]),
                "local_frozen_prediction": int(local[index]),
                "shared_prediction": int(shared[index]),
                "shared_agent_win_probability": float(probability[index]),
                "shared_route_to_agent": bool(route[index]),
            }
            for index, row in enumerate(task_rows)
        ]
        write_jsonl_atomic(artifact_dir / "predictions.jsonl", prediction_rows)
        task_receipt = {
            "schema_version": POST_V31_TRANSFER_SCHEMA_VERSION,
            "task": task,
            "scope": "train-only scaffold OOF; valid and test not read",
            "n_rows": len(labels),
            "n_disagreements": int((knn != agent).sum()),
            "profile": TRANSFER_PROFILE,
            "family": TRANSFER_FAMILY,
            "metrics": metrics,
            "direction_only": direction_metrics,
            "local_frozen": local_metrics,
            "shared_vs_direction_only": vs_direction,
            "shared_vs_local_frozen": vs_local,
            "outer_folds": records,
        }
        write_json_atomic(artifact_dir / "summary.json", task_receipt)
        task_results.append(task_receipt)

    continuation_gate = _evaluate_transfer_gate(task_results)
    result = {
        "schema_version": f"{POST_V31_TRANSFER_SCHEMA_VERSION}.matrix",
        "scope": "train-only scaffold OOF; valid and test not read",
        "tasks": [spec.task for spec in specs],
        "profile": TRANSFER_PROFILE,
        "family": TRANSFER_FAMILY,
        "task_weighting": "equal total weight per source task within direction",
        "cross_task_exclusion": "target heldout molecule identity and fold-group/scaffold",
        "task_specific_layers": ["sigmoid calibration", "threshold", "Wilson risk gate"],
        "results": task_results,
        "continuation_gate": continuation_gate,
    }
    output_path = Path(output_root) / "post_selector_v31_transfer_result.json"
    write_json_atomic(output_path, result)
    return result


def task_balanced_weights(task: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Give every represented task equal total weight inside one direction."""

    weights = np.zeros(len(task), dtype=float)
    represented = sorted(set(task[eligible].tolist()))
    for name in represented:
        mask = eligible & (task == name)
        weights[mask] = 1.0 / int(mask.sum())
    if eligible.any():
        weights[eligible] *= int(eligible.sum()) / weights[eligible].sum()
    return weights


def cross_task_training_mask(
    data: TransferData,
    *,
    target_task: str,
    direction_mask: np.ndarray,
    target_allowed: np.ndarray,
    target_excluded: np.ndarray,
) -> np.ndarray:
    """Build a leakage-conservative shared training mask."""

    banned_identity = set(data.global_identity[target_excluded].tolist())
    banned_group = set(data.global_group[target_excluded].tolist())
    other = data.global_task != target_task
    cross_task_safe = (
        other
        & ~np.isin(data.global_identity, list(banned_identity))
        & ~np.isin(data.global_group, list(banned_group))
    )
    return direction_mask & (target_allowed | cross_task_safe)


def _run_target_outer_fold(
    data: TransferData,
    *,
    target_task: str,
    outer_fold: int,
) -> dict[str, Any]:
    target = data.global_task == target_task
    outer_test = target & (data.global_fold == outer_fold)
    outer_train = target & ~outer_test
    local_slice = data.task_slices[target_task]
    local_offset = local_slice.start or 0
    outer_global_indices = np.flatnonzero(outer_test)
    output_indices = outer_global_indices - local_offset
    outer_position = np.full(len(data.global_target), -1, dtype=int)
    outer_position[outer_global_indices] = np.arange(len(outer_global_indices))
    output_probability = np.full(len(output_indices), 0.5, dtype=float)
    output_route = np.zeros(len(output_indices), dtype=bool)
    output_prediction = data.global_knn[outer_test].copy()
    direction_records: dict[str, Any] = {}

    for direction, pair in DIRECTIONS.items():
        direction_mask = (
            (data.global_knn == pair[0]) & (data.global_agent == pair[1])
        )
        target_direction_train = outer_train & direction_mask
        inner_raw = np.full(len(data.global_target), 0.5, dtype=float)
        inner_folds = sorted(set(data.global_fold[target_direction_train].tolist()))
        for inner_fold in inner_folds:
            inner_test = target_direction_train & (data.global_fold == inner_fold)
            target_allowed = target_direction_train & ~inner_test
            target_excluded = outer_test | inner_test
            training = cross_task_training_mask(
                data,
                target_task=target_task,
                direction_mask=direction_mask,
                target_allowed=target_allowed,
                target_excluded=target_excluded,
            )
            model = _fit_shared_logistic(data, training)
            inner_raw[inner_test] = model.predict_probability(
                data.global_matrix[inner_test]
            )
        calibrator = _fit_calibrator(
            inner_raw[target_direction_train],
            data.global_target[target_direction_train],
        )
        inner_calibrated = np.full(len(data.global_target), 0.5, dtype=float)
        inner_calibrated[target_direction_train] = calibrator.predict(
            inner_raw[target_direction_train]
        )
        policy = choose_safe_direction_policy(
            data.global_target,
            inner_calibrated,
            target_direction_train,
            family=TRANSFER_FAMILY,
            profile=TRANSFER_PROFILE,
        )
        final_training = cross_task_training_mask(
            data,
            target_task=target_task,
            direction_mask=direction_mask,
            target_allowed=target_direction_train,
            target_excluded=outer_test,
        )
        model = _fit_shared_logistic(data, final_training)
        direction_test = outer_test & direction_mask
        if direction_test.any():
            raw = model.predict_probability(data.global_matrix[direction_test])
            calibrated = calibrator.predict(raw)
            local_positions = outer_position[np.flatnonzero(direction_test)]
            output_probability[local_positions] = calibrated
            selected = calibrated >= float(policy["threshold"])
            output_route[local_positions] = selected
            output_prediction[local_positions[selected]] = data.global_agent[
                np.flatnonzero(direction_test)[selected]
            ]
        direction_records[direction] = {
            "target_outer_train_n": int(target_direction_train.sum()),
            "shared_train_n": int(final_training.sum()),
            "cross_task_train_n": int(
                (final_training & (data.global_task != target_task)).sum()
            ),
            "policy": policy,
        }

    return {
        "schema_version": POST_V31_TRANSFER_SCHEMA_VERSION,
        "task": target_task,
        "outer_fold": int(outer_fold),
        "target_local_indices": output_indices.tolist(),
        "probability": output_probability.tolist(),
        "route": output_route.tolist(),
        "prediction": output_prediction.tolist(),
        "directions": direction_records,
    }


def _fit_shared_logistic(data: TransferData, training: np.ndarray) -> SharedLogisticModel:
    y = data.global_target[training]
    if not len(y):
        raise ValueError("Shared transfer fit received no training rows")
    if len(set(y.tolist())) < 2:
        return SharedLogisticModel(None, None, float(y[0]))
    weights = task_balanced_weights(data.global_task, training)[training]
    scaler = StandardScaler()
    transformed = scaler.fit_transform(
        data.global_matrix[training], sample_weight=weights
    )
    estimator = LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=20260805,
    )
    estimator.fit(transformed, y, sample_weight=weights)
    return SharedLogisticModel(scaler, estimator)


def _load_transfer_data(
    specs: Sequence[TaskSpec],
    *,
    output_root: str | Path,
) -> TransferData:
    task_names = tuple(spec.task for spec in specs)
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    tasks = []
    folds = []
    targets = []
    knn = []
    agent = []
    identity = []
    groups = []
    matrices = []
    task_slices = {}
    start = 0
    for task_index, spec in enumerate(specs):
        root = task_root(output_root, spec.task)
        rows = read_jsonl(root / f"{POST_FEATURE_STEM}.jsonl")
        arrays = PostSelectorArrays.from_rows(rows)
        fold_rows = {
            int(row["train_index"]): row for row in read_jsonl(root / "folds.jsonl")
        }
        if len(fold_rows) != len(rows):
            raise ValueError(f"Fold/feature row mismatch for {spec.task}")
        rows_by_task[spec.task] = rows
        n_rows = len(rows)
        task_slices[spec.task] = slice(start, start + n_rows)
        start += n_rows
        tasks.extend([spec.task] * n_rows)
        folds.extend(arrays.folds.tolist())
        targets.extend(arrays.agent_win_target.tolist())
        knn.extend(arrays.knn.tolist())
        agent.extend(arrays.agent.tolist())
        identity.extend(
            str(fold_rows[int(row["train_index"])]["molecule_identity_key"])
            for row in rows
        )
        groups.extend(
            str(fold_rows[int(row["train_index"])]["fold_group_key"])
            for row in rows
        )
        base = _matrix(rows, V31_FEATURE_PROFILES[TRANSFER_PROFILE])
        one_hot = np.zeros((n_rows, len(task_names)), dtype=float)
        one_hot[:, task_index] = 1.0
        matrices.append(np.hstack([base, one_hot]))
    return TransferData(
        task_names=task_names,
        rows_by_task=rows_by_task,
        global_task=np.asarray(tasks, dtype=object),
        global_fold=np.asarray(folds, dtype=int),
        global_target=np.asarray(targets, dtype=int),
        global_knn=np.asarray(knn, dtype=int),
        global_agent=np.asarray(agent, dtype=int),
        global_identity=np.asarray(identity, dtype=object),
        global_group=np.asarray(groups, dtype=object),
        global_matrix=np.vstack(matrices),
        task_slices=task_slices,
    )


def _load_full_local_curve_prediction(root: Path, *, expected_rows: int) -> np.ndarray:
    summary = json.loads(
        (root / POST_V31_LEARNING_CURVE_ARTIFACT_DIR / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    full = int(summary["n_disagreements"])
    with np.load(
        root / POST_V31_LEARNING_CURVE_ARTIFACT_DIR / "predictions.npz"
    ) as payload:
        indices = np.flatnonzero(payload["budget"] == full)
        if len(indices) != 1:
            raise ValueError(f"Expected one full-budget local curve prediction for {root}")
        prediction = np.asarray(payload["prediction"][indices[0]], dtype=int)
    if len(prediction) != expected_rows:
        raise ValueError(f"Local curve prediction row mismatch for {root}")
    return prediction


def _evaluate_transfer_gate(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    small_tasks = [
        row for row in results if row["task"] in {"bioavailability_ma", "skin_reaction"}
    ]
    passing = []
    for row in small_tasks:
        paired = row["shared_vs_local_frozen"]
        accuracy = (
            paired["accuracy_delta_ci95"][0] > 0
            and paired["observed_macro_f1_delta"] >= 0
        )
        macro = (
            paired["macro_f1_delta_ci95"][0] > 0
            and paired["observed_accuracy_delta"] >= 0
        )
        if accuracy or macro:
            passing.append(row["task"])
    return {
        "passed": bool(passing),
        "decision": "continue_shared_representation" if passing else "stop_router_main_method",
        "passing_small_tasks": passing,
        "criterion": (
            "small-task shared-vs-local paired CI positive for one metric; "
            "other observed delta nonnegative"
        ),
    }
