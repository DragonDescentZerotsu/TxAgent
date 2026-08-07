"""Train-only matched-size diagnosis for the frozen v3.1 post-selector.

The experiment isolates supervision volume from model/profile search.  It uses
the direction-specific family/profile frozen in the v3.1 deployment manifest,
subsamples only outer-train disagreement rows, and evaluates every outer fold
against a direction-only OR comparator.  It never reads valid or test labels.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic

from .contract import TaskSpec, task_root
from .post_contract import (
    POST_FEATURE_STEM,
    POST_V31_ARTIFACT_DIR,
    POST_V31_LEARNING_CURVE_ARTIFACT_DIR,
    POST_V31_LEARNING_CURVE_SCHEMA_VERSION,
)
from .post_train import _matrix
from .post_v31_common import (
    PostSelectorArrays,
    direction_only_prediction,
    read_jsonl,
    sha256_file,
)
from .post_v31_train import (
    DIRECTIONS,
    V31_FEATURE_PROFILES,
    _nested_calibrated_scores,
    choose_safe_direction_policy,
    fit_direction_calibrated_ensemble,
)
from .train import evaluation_metrics, paired_bootstrap_deltas


DEFAULT_TASK_BUDGETS: dict[str, tuple[int, ...]] = {
    "bbb_martins": (800, 1600, 3200, 7130),
    "bioavailability_ma": (772,),
    "skin_reaction": (802,),
}
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def run_post_v31_learning_curve(
    spec: TaskSpec,
    *,
    output_root: str | Path,
    budgets: Iterable[int] | None = None,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    workers: int = 1,
    bootstrap_repetitions: int = 2000,
) -> dict[str, Any]:
    """Run the frozen-spec learning curve and publish an isolated receipt."""

    root = task_root(output_root, spec.task)
    feature_path = root / f"{POST_FEATURE_STEM}.jsonl"
    manifest_path = root / POST_V31_ARTIFACT_DIR / "model_manifest.json"
    for path in (feature_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)

    rows = read_jsonl(feature_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays = PostSelectorArrays.from_rows(rows)
    folds = arrays.folds
    labels = arrays.labels
    knn = arrays.knn
    agent = arrays.agent
    target = arrays.agent_win_target
    direction_masks = arrays.direction_masks()
    n_disagreements = int(sum(mask.sum() for mask in direction_masks.values()))
    if not n_disagreements:
        raise ValueError(f"No KNN-agent disagreements available for {spec.task}")
    selected_budgets = _normalize_budgets(
        budgets if budgets is not None else DEFAULT_TASK_BUDGETS[spec.task],
        n_disagreements,
    )
    selected_seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not selected_seeds:
        raise ValueError("At least one learning-curve seed is required")

    direction_specs = _direction_specs(manifest)
    matrices = {
        profile: _matrix(rows, V31_FEATURE_PROFILES[profile])
        for profile in sorted({payload["feature_profile"] for payload in direction_specs.values()})
    }
    jobs = [
        (budget, seed)
        for budget in selected_budgets
        for seed in (selected_seeds[:1] if budget == n_disagreements else selected_seeds)
    ]

    results: list[dict[str, Any]] = []
    predictions: dict[tuple[int, int], np.ndarray] = {}
    max_workers = max(1, min(int(workers), len(jobs)))
    # Loky starts isolated workers and enforces one native thread per process;
    # this avoids nesting a full OpenMP pool inside every curve job. Large
    # matrices are automatically memmapped rather than copied per worker.
    with threadpool_limits(limits=1), parallel_config(
        backend="loky", inner_max_num_threads=1
    ):
        completed = Parallel(
            n_jobs=max_workers,
            return_as="generator_unordered",
            max_nbytes="1M",
            mmap_mode="r",
        )(
            delayed(_run_curve_job)(
                budget=budget,
                seed=seed,
                full_disagreements=n_disagreements,
                labels=labels,
                knn=knn,
                agent=agent,
                target=target,
                folds=folds,
                direction_masks=direction_masks,
                direction_specs=direction_specs,
                matrices=matrices,
                bootstrap_repetitions=bootstrap_repetitions,
            )
            for budget, seed in jobs
        )
        for result, prediction in completed:
            budget = int(result["budget"])
            seed = int(result["seed"])
            results.append(result)
            predictions[(budget, seed)] = prediction
            print(
                f"[{spec.task}] learning curve budget={budget} seed={seed} "
                "acc_delta_vs_or="
                f"{result['incremental_vs_direction_only']['observed_accuracy_delta']:+.4f} "
                "f1_delta_vs_or="
                f"{result['incremental_vs_direction_only']['observed_macro_f1_delta']:+.4f}",
                file=sys.stderr,
                flush=True,
            )

    results.sort(key=lambda row: (row["budget"], row["seed"]))
    summary_rows = summarize_learning_curve_jobs(results)
    continuation_gate = evaluate_learning_curve_continuation_gate(
        summary_rows,
        full_disagreements=n_disagreements,
    )
    artifact_dir = root / POST_V31_LEARNING_CURVE_ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(artifact_dir / "job_metrics.jsonl", results)
    prediction_path = artifact_dir / "predictions.npz"
    _write_prediction_matrix_atomic(
        prediction_path,
        results,
        predictions,
        labels=labels,
        knn=knn,
        agent=agent,
    )
    receipt = {
        "schema_version": POST_V31_LEARNING_CURVE_SCHEMA_VERSION,
        "task": spec.task,
        "scope": "train-only scaffold OOF; valid and test not read",
        "n_rows": len(rows),
        "n_disagreements": n_disagreements,
        "direction_counts": {
            direction: int(mask.sum()) for direction, mask in direction_masks.items()
        },
        "budgets": list(selected_budgets),
        "seeds": list(selected_seeds),
        "full_budget_runs_one_seed": True,
        "model_selection": "family/profile frozen from v3.1 full-train manifest",
        "threshold_selection": "nested calibrated scores from sampled outer-train only",
        "primary_comparator": "direction-only OR rule",
        "direction_specs": direction_specs,
        "summary": summary_rows,
        "continuation_gate": continuation_gate,
        "artifacts": {
            "job_metrics": str(artifact_dir / "job_metrics.jsonl"),
            "predictions": str(prediction_path),
        },
        "source_sha256": {
            str(feature_path): sha256_file(feature_path),
            str(manifest_path): sha256_file(manifest_path),
        },
    }
    write_json_atomic(artifact_dir / "summary.json", receipt)
    (artifact_dir / "report_zh.md").write_text(
        _render_report(receipt), encoding="utf-8"
    )
    return receipt


def stratified_subsample_mask(
    eligible: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Subsample within fold and target cells while preserving outer isolation."""

    eligible = np.asarray(eligible, dtype=bool)
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if fraction >= 1.0:
        return eligible.copy()
    selected = np.zeros(len(eligible), dtype=bool)
    rng = np.random.default_rng(seed)
    for fold in sorted(set(folds[eligible].tolist())):
        for label in (0, 1):
            indices = np.flatnonzero(eligible & (folds == fold) & (target == label))
            if not len(indices):
                continue
            take = min(len(indices), max(1, int(round(len(indices) * fraction))))
            chosen = rng.choice(indices, size=take, replace=False)
            selected[chosen] = True
    return selected


def summarize_learning_curve_jobs(
    jobs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate seed-level curve jobs without hiding between-seed variation."""

    output = []
    for budget in sorted({int(row["budget"]) for row in jobs}):
        group = [row for row in jobs if int(row["budget"]) == budget]
        accuracy = np.asarray(
            [row["incremental_vs_direction_only"]["observed_accuracy_delta"] for row in group],
            dtype=float,
        )
        macro_f1 = np.asarray(
            [row["incremental_vs_direction_only"]["observed_macro_f1_delta"] for row in group],
            dtype=float,
        )
        output.append(
            {
                "budget": budget,
                "n_seeds": len(group),
                "mean_realized_outer_train_disagreements": float(
                    np.mean([row["mean_realized_outer_train_disagreements"] for row in group])
                ),
                "accuracy_delta_vs_direction_only_mean": float(accuracy.mean()),
                "accuracy_delta_vs_direction_only_min": float(accuracy.min()),
                "accuracy_delta_vs_direction_only_max": float(accuracy.max()),
                "accuracy_positive_seed_fraction": float(np.mean(accuracy > 0)),
                "macro_f1_delta_vs_direction_only_mean": float(macro_f1.mean()),
                "macro_f1_delta_vs_direction_only_min": float(macro_f1.min()),
                "macro_f1_delta_vs_direction_only_max": float(macro_f1.max()),
                "macro_f1_positive_seed_fraction": float(np.mean(macro_f1 > 0)),
                "jobs": group,
            }
        )
    return output


def evaluate_learning_curve_continuation_gate(
    summary: Sequence[Mapping[str, Any]],
    *,
    full_disagreements: int,
) -> dict[str, Any]:
    """Apply the predeclared stop/continue rule to the full-size curve point."""

    full = next(
        (row for row in summary if int(row["budget"]) == int(full_disagreements)),
        None,
    )
    if full is None:
        return {
            "passed": False,
            "decision": "incomplete",
            "failure_reasons": ["missing_full_budget"],
        }
    job = full["jobs"][0]
    paired = job["incremental_vs_direction_only"]
    accuracy_supported = bool(
        paired["accuracy_delta_ci95"][0] > 0
        and paired["observed_macro_f1_delta"] >= 0
    )
    macro_supported = bool(
        paired["macro_f1_delta_ci95"][0] > 0
        and paired["observed_accuracy_delta"] >= 0
    )
    passed = accuracy_supported or macro_supported
    reasons = [] if passed else ["full_size_learned_vs_direction_only_ci_not_positive"]
    return {
        "passed": passed,
        "decision": "continue_shared_representation" if passed else "stop_router_main_method",
        "accuracy_supported": accuracy_supported,
        "macro_f1_supported": macro_supported,
        "full_size_incremental_vs_direction_only": paired,
        "failure_reasons": reasons,
    }


def _run_curve_job(
    *,
    budget: int,
    seed: int,
    full_disagreements: int,
    labels: np.ndarray,
    knn: np.ndarray,
    agent: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    direction_masks: Mapping[str, np.ndarray],
    direction_specs: Mapping[str, Mapping[str, Any]],
    matrices: Mapping[str, np.ndarray],
    bootstrap_repetitions: int,
) -> tuple[dict[str, Any], np.ndarray]:
    fraction = min(1.0, budget / full_disagreements)
    probability = np.full(len(labels), 0.5, dtype=float)
    threshold = np.full(len(labels), 1.01, dtype=float)
    outer_records = []
    realized_counts = []

    for outer_fold in sorted(set(folds.tolist())):
        outer_test = folds == outer_fold
        outer_train = ~outer_test
        outer_record: dict[str, Any] = {
            "outer_fold": int(outer_fold),
            "directions": {},
        }
        for direction_index, (direction, mask) in enumerate(direction_masks.items()):
            payload = direction_specs[direction]
            family = str(payload["model_family"])
            profile = str(payload["feature_profile"])
            eligible = outer_train & mask
            sampled = stratified_subsample_mask(
                eligible,
                target,
                folds,
                fraction=fraction,
                seed=(seed + 1) * 1_000_003 + budget * 101 + outer_fold * 17 + direction_index,
            )
            realized_counts.append(int(sampled.sum()))
            matrix = matrices[profile]
            train_scores = _nested_calibrated_scores(
                family,
                matrix,
                target,
                folds,
                sampled,
            )
            policy = choose_safe_direction_policy(
                target,
                train_scores,
                sampled,
                family=family,
                profile=profile,
            )
            test_mask = outer_test & mask
            if test_mask.any():
                ensemble = fit_direction_calibrated_ensemble(
                    family,
                    matrix,
                    target,
                    folds,
                    sampled,
                )
                probability[test_mask] = ensemble.predict_probability(matrix[test_mask])
                threshold[test_mask] = float(policy["threshold"])
            outer_record["directions"][direction] = {
                "sampled_train_n": int(sampled.sum()),
                "outer_test_n": int(test_mask.sum()),
                "policy": policy,
            }
        outer_records.append(outer_record)

    disagreement = direction_masks["knn0_agent1"] | direction_masks["knn1_agent0"]
    route = disagreement & (probability >= threshold)
    learned = np.where(route, agent, knn)
    direction_only = direction_only_prediction(knn, agent)
    learned_metrics = evaluation_metrics(labels, knn, agent, learned)["router"]
    learned_metrics["route_count"] = int(route.sum())
    comparator_metrics = evaluation_metrics(labels, knn, agent, direction_only)["router"]
    incremental = paired_bootstrap_deltas(
        labels,
        direction_only,
        learned,
        seed=20260805 + seed + budget,
        repetitions=bootstrap_repetitions,
    )
    result = {
        "schema_version": POST_V31_LEARNING_CURVE_SCHEMA_VERSION,
        "budget": int(budget),
        "fraction_of_full_disagreements": float(fraction),
        "seed": int(seed),
        "mean_realized_outer_train_disagreements": float(
            sum(realized_counts) / len(set(folds.tolist()))
        ),
        "learned_selector": learned_metrics,
        "direction_only": comparator_metrics,
        "incremental_vs_direction_only": incremental,
        "outer_folds": outer_records,
    }
    return result, learned.astype(np.int8)


def _direction_specs(manifest: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    output = {}
    for direction in DIRECTIONS:
        payload = manifest["directions"][direction]
        profile = str(payload["feature_profile"])
        if profile not in V31_FEATURE_PROFILES:
            raise ValueError(f"Unknown frozen v3.1 profile: {profile}")
        output[direction] = {
            "model_family": str(payload["model_family"]),
            "feature_profile": profile,
        }
    return output


def _normalize_budgets(budgets: Iterable[int], full: int) -> tuple[int, ...]:
    normalized = sorted({min(full, int(value)) for value in budgets if int(value) > 0})
    if full not in normalized:
        normalized.append(full)
    return tuple(sorted(normalized))


def _write_prediction_matrix_atomic(
    path: Path,
    results: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[int, int], np.ndarray],
    *,
    labels: np.ndarray,
    knn: np.ndarray,
    agent: np.ndarray,
) -> None:
    matrix = np.vstack(
        [predictions[(int(row["budget"]), int(row["seed"]))] for row in results]
    )
    budgets = np.asarray([int(row["budget"]) for row in results], dtype=np.int32)
    seeds = np.asarray([int(row["seed"]) for row in results], dtype=np.int32)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        prediction=matrix,
        budget=budgets,
        seed=seeds,
        Y=labels.astype(np.int8),
        knn=knn.astype(np.int8),
        agent=agent.astype(np.int8),
    )
    os.replace(temporary, path)


def _render_report(receipt: Mapping[str, Any]) -> str:
    lines = [
        f"# {receipt['task']} v3.1 matched-size learning curve",
        "",
        f"- continuation gate: `{receipt['continuation_gate']['passed']}`",
        f"- decision: `{receipt['continuation_gate']['decision']}`",
        f"- comparator: `{receipt['primary_comparator']}`",
        "",
        "| budget | seeds | mean accuracy delta vs OR | mean macro-F1 delta vs OR |",
        "|---:|---:|---:|---:|",
    ]
    for row in receipt["summary"]:
        lines.append(
            f"| {row['budget']} | {row['n_seeds']} | "
            f"{row['accuracy_delta_vs_direction_only_mean']:+.4f} | "
            f"{row['macro_f1_delta_vs_direction_only_mean']:+.4f} |"
        )
    return "\n".join(lines) + "\n"
