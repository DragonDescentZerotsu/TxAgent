"""Plot reusable assay-count performance and retrieval-volume curves.

The plotter reads experiment artifacts rather than embedding metric values.  It
supports the legacy shared-prefix manifest and the task-specific full-catalog
manifest, reuses historical no-retrieval metrics at x=0, and computes retrieval
volume directly from replay payloads.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class TaskPlotSpec:
    label: str
    color: str
    marker: str
    historical_best_condition: str
    historical_best_label: str


TASK_SPECS = {
    "bbb_martins": TaskPlotSpec("BBB", "#0072B2", "o", "starling_full_flat", "full-flat"),
    "bioavailability_ma": TaskPlotSpec(
        "Bioavailability", "#D55E00", "s", "starling_full_flat", "full-flat"
    ),
    "skin_reaction": TaskPlotSpec("Skin", "#CC79A7", "^", "starling_direct", "direct"),
}

CONDITIONED_TASK_SPECS = {
    "bbb_martins": ("BBB", "BBB_Martins"),
    "bioavailability_ma": ("Bioavailability", "Bioavailability_Ma"),
    "skin_reaction": ("Skin", "Skin_Reaction"),
    "clintox": ("ClinTox", "ClinTox"),
}

CONDITIONED_BASELINES = (
    (
        "minimol_head",
        "MiniMol head",
        Path("minimol_train_retest/final/metrics.json"),
    ),
    (
        "minimol_knn_condition",
        "MiniMol KNN condition",
        Path("minimol_knn/same_condition_then_null/metrics.json"),
    ),
    (
        "minimol_knn_all",
        "MiniMol KNN all train",
        Path("minimol_knn/all_train_unique_molecules/metrics.json"),
    ),
    (
        "morgan_knn_condition",
        "Morgan KNN condition",
        Path("morgan_knn/same_condition_then_null/metrics.json"),
    ),
    (
        "morgan_knn_all",
        "Morgan KNN all train",
        Path("morgan_knn/all_train_unique_molecules/metrics.json"),
    ),
)

def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_task_path_overrides(values: list[str]) -> dict[str, Path]:
    """Parse repeatable TASK=PATH CLI overrides without hiding bad task names."""
    overrides: dict[str, Path] = {}
    for value in values:
        task, separator, raw_path = value.partition("=")
        if not separator or task not in CONDITIONED_TASK_SPECS or not raw_path:
            raise ValueError(
                "Task path overrides must use a known conditioned task as "
                f"TASK=PATH; received {value!r}"
            )
        if task in overrides:
            raise ValueError(f"Duplicate task path override: {task}")
        overrides[task] = Path(raw_path)
    return overrides


def _reasoning_tokens_from_usage(usage: dict[str, Any]) -> int | None:
    """Read normalized or OpenAI-compatible nested reasoning-token usage."""
    value = usage.get("reasoning_tokens")
    if value is None:
        details = usage.get("completion_tokens_details") or {}
        value = details.get("reasoning_tokens")
    return None if value is None else int(value)


def _prefixes_by_task(manifest: dict[str, Any]) -> dict[str, list[int]]:
    tasks = list(manifest["tasks"])
    if manifest.get("prefixes_by_task"):
        return {
            task: [int(value) for value in manifest["prefixes_by_task"][task]]
            for task in tasks
        }
    shared = [int(value) for value in manifest.get("prefixes") or []]
    return {task: list(shared) for task in tasks}


def _historical_batch(group_root: Path, task: str, condition: str) -> Path:
    return group_root / task / f"{task}__{condition}"


def _retrieval_counts(retrieval: dict[str, Any]) -> tuple[int, int, int]:
    neighbors = {
        str(neighbor.get("molecule_chembl_id") or neighbor_index)
        for group in retrieval.get("groups") or []
        for neighbor_index, neighbor in enumerate(group.get("neighbors") or [])
    }
    evidence_rows = [
        row
        for group in retrieval.get("groups") or []
        for neighbor in group.get("neighbors") or []
        for row in neighbor.get("evidence_rows") or []
    ]
    source_records = sum(
        max(0, int(row.get("source_record_count") or 0)) for row in evidence_rows
    )
    return len(neighbors), len(evidence_rows), source_records


def _batch_retrieval_means(batch: Path) -> dict[str, float | int]:
    manifest = _load_json(batch / "manifest.json")
    indices = [int(value) for value in manifest.get("indices") or []]
    if not indices:
        indices = list(range(int(manifest["n_items"])))
    values = []
    for query_index in indices:
        run_dir = batch / "runs" / f"{batch.name}_idx{query_index:05d}"
        values.append(_retrieval_counts(_load_json(run_dir / "retrieval.json")))
    if not values:
        raise ValueError(f"Replay batch has no retrieval payloads: {batch}")
    return {
        "n_queries": len(values),
        "mean_unique_molecules": mean(value[0] for value in values),
        "mean_assay_molecule_evidence_rows": mean(value[1] for value in values),
        "mean_source_records_represented": mean(value[2] for value in values),
    }


def collect_curve_data(
    *,
    output_root: Path,
    historical_group_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    experiment = _load_json(output_root / "experiment_manifest.json")
    prefixes_by_task = _prefixes_by_task(experiment)
    performance_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []

    for task in experiment["tasks"]:
        spec = TASK_SPECS.get(
            task,
            TaskPlotSpec(task, "#4C78A8", "o", "starling_full_flat", "historical best"),
        )
        none_batch = _historical_batch(historical_group_root, task, "none")
        none_metrics = _load_json(none_batch / "metrics.json")
        performance_rows.append(
            {
                "task": task,
                "task_label": spec.label,
                "assay_count": 0,
                "condition": "none_reused",
                "macro_f1": float(none_metrics["macro_f1"]),
                "accuracy": float(none_metrics["accuracy"]),
                "n_total": int(none_metrics["n_total"]),
                "n_failed_runs": int(none_metrics["n_failed_runs"]),
                "metrics_path": str(none_batch / "metrics.json"),
            }
        )
        retrieval_rows.append(
            {
                "task": task,
                "task_label": spec.label,
                "assay_count": 0,
                "condition": "none_reused",
                "n_queries": int(none_metrics["n_total"]),
                "mean_unique_molecules": 0.0,
                "mean_assay_molecule_evidence_rows": 0.0,
                "mean_source_records_represented": 0.0,
                "replay_batch": "",
            }
        )

        replay_root = Path(experiment["inputs"][task]["replay_root"])
        for prefix in prefixes_by_task[task]:
            batch = output_root / task / f"assay_flat_top{prefix}"
            metrics_path = batch / "metrics.json"
            replay_batch = replay_root / f"assay_flat_top{prefix}"
            if metrics_path.is_file():
                metrics = _load_json(metrics_path)
                performance_rows.append(
                    {
                        "task": task,
                        "task_label": spec.label,
                        "assay_count": prefix,
                        "condition": "assay_level",
                        "macro_f1": float(metrics["macro_f1"]),
                        "accuracy": float(metrics["accuracy"]),
                        "n_total": int(metrics["n_total"]),
                        "n_failed_runs": int(metrics["n_failed_runs"]),
                        "metrics_path": str(metrics_path),
                    }
                )
            if replay_batch.is_dir():
                retrieval_rows.append(
                    {
                        "task": task,
                        "task_label": spec.label,
                        "assay_count": prefix,
                        "condition": "assay_level",
                        **_batch_retrieval_means(replay_batch),
                        "replay_batch": str(replay_batch),
                    }
                )

    best_rows: list[dict[str, Any]] = []
    for task in experiment["tasks"]:
        spec = TASK_SPECS[task]
        assay_rows = [
            row
            for row in performance_rows
            if row["task"] == task and row["condition"] == "assay_level"
        ]
        if not assay_rows:
            continue
        assay_best = max(assay_rows, key=lambda row: (row["macro_f1"], -row["assay_count"]))
        group_batch = _historical_batch(
            historical_group_root, task, spec.historical_best_condition
        )
        group_metrics = _load_json(group_batch / "metrics.json")
        best_rows.append(
            {
                "task": task,
                "task_label": spec.label,
                "assay_best_count": assay_best["assay_count"],
                "assay_best_macro_f1": assay_best["macro_f1"],
                "group_best_condition": spec.historical_best_label,
                "group_best_macro_f1": float(group_metrics["macro_f1"]),
                "delta_macro_f1": assay_best["macro_f1"] - float(group_metrics["macro_f1"]),
                "delta_percentage_points": 100
                * (assay_best["macro_f1"] - float(group_metrics["macro_f1"])),
                "n_valid": assay_best["n_total"],
                "assay_metrics_path": assay_best["metrics_path"],
                "group_metrics_path": str(group_batch / "metrics.json"),
            }
        )
    return performance_rows, retrieval_rows, {"experiment": experiment, "best": best_rows}


def incomplete_curve_conditions(
    performance_rows: list[dict[str, Any]],
    experiment: dict[str, Any],
) -> list[str]:
    """List absent, failed, or sample-incomplete assay-prefix metrics."""
    prefixes_by_task = _prefixes_by_task(experiment)
    none_n = {
        row["task"]: int(row["n_total"])
        for row in performance_rows
        if row["condition"] == "none_reused"
    }
    observed = {
        (row["task"], int(row["assay_count"])): row
        for row in performance_rows
        if row["condition"] == "assay_level"
    }
    incomplete = []
    for task in experiment["tasks"]:
        for prefix in prefixes_by_task[task]:
            row = observed.get((task, prefix))
            if (
                row is None
                or int(row["n_failed_runs"]) != 0
                or int(row["n_total"]) != none_n[task]
            ):
                incomplete.append(f"{task}__assay_flat_top{prefix}")
    return incomplete


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _metric_n(metrics: dict[str, Any]) -> int:
    """Read evaluation size across agent, KNN, and MiniMol-head receipts."""
    for field in ("n_total", "n_evaluated", "n_evaluation"):
        if metrics.get(field) is not None:
            return int(metrics[field])
    if metrics.get("splits", {}).get("evaluation") is not None:
        return int(metrics["splits"]["evaluation"])
    raise ValueError("Metrics receipt does not expose an evaluation sample count")


def _metric_value(metrics: dict[str, Any], field: str) -> float:
    if metrics.get(field) is not None:
        return float(metrics[field])
    if metrics.get("evaluation_metrics", {}).get(field) is not None:
        return float(metrics["evaluation_metrics"][field])
    raise ValueError(f"Metrics receipt does not expose {field}")


def _agent_metric_issue(metrics: dict[str, Any], expected_n: int) -> str:
    if int(metrics.get("n_failed_runs") or 0):
        return f"n_failed_runs={int(metrics['n_failed_runs'])}"
    actual_n = _metric_n(metrics)
    if actual_n != expected_n:
        return f"n_total={actual_n}, expected={expected_n}"
    return ""


def _complete_agent_metric_issue(metrics: dict[str, Any], expected_n: int) -> str:
    """Require complete successful coverage, not merely a zero failure counter."""
    issue = _agent_metric_issue(metrics, expected_n)
    if issue:
        return issue
    for field in ("n_successful", "n_evaluable"):
        if metrics.get(field) is not None and int(metrics[field]) != expected_n:
            return f"{field}={int(metrics[field])}, expected={expected_n}"
    return ""


def _append_conditioned_baselines(
    rows: list[dict[str, Any]],
    *,
    task: str,
    task_label: str,
    baseline_task: str,
    baseline_root: Path,
    expected_n: int,
    omit_sample_count_mismatch: bool = False,
) -> list[dict[str, Any]]:
    omissions: list[dict[str, Any]] = []
    for method, plot_label, relative_path in CONDITIONED_BASELINES:
        metrics_path = baseline_root / baseline_task / relative_path
        if method == "minimol_head" and not metrics_path.is_file():
            candidates = (
                Path("minimol_train/final/metrics.json"),
                Path("minimol_head/final/metrics.json"),
            )
            metrics_path = next(
                (
                    baseline_root / baseline_task / candidate
                    for candidate in candidates
                    if (baseline_root / baseline_task / candidate).is_file()
                ),
                metrics_path,
            )
        metrics = _load_json(metrics_path)
        actual_n = _metric_n(metrics)
        if actual_n != expected_n:
            if omit_sample_count_mismatch:
                omissions.append(
                    {
                        "task": task,
                        "method": method,
                        "reason": "sample_count_mismatch",
                        "baseline_n": actual_n,
                        "agent_n": expected_n,
                        "metrics_path": str(metrics_path),
                    }
                )
                continue
            raise ValueError(
                f"Baseline sample-count mismatch for {task}/{method}: "
                f"{actual_n} != {expected_n}"
            )
        if metrics.get("n_evaluated") is not None and int(
            metrics["n_evaluated"]
        ) != expected_n:
            raise ValueError(f"Incomplete baseline coverage for {task}/{method}")
        rows.append(
            {
                "task": task,
                "task_label": task_label,
                "result_type": "baseline",
                "method": method,
                "plot_label": plot_label,
                "level": "",
                "assay_count": "",
                "macro_f1": _metric_value(metrics, "macro_f1"),
                "accuracy": _metric_value(metrics, "accuracy"),
                "n": actual_n,
                "metrics_path": str(metrics_path),
            }
        )
    return omissions






def collect_conditioned_progressive_resource_data(
    *,
    progressive_root: Path | None = None,
    progressive_roots_by_task: dict[str, Path] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect performance and prompt-resource statistics by task lineage."""
    if (progressive_root is None) == (progressive_roots_by_task is None):
        raise ValueError(
            "Provide exactly one of progressive_root or progressive_roots_by_task"
        )
    if progressive_root is not None:
        manifest = _load_json(progressive_root / "experiment_manifest.json")
        roots_by_task = {task: progressive_root for task in manifest["tasks"]}
    else:
        roots_by_task = dict(progressive_roots_by_task or {})
    if not roots_by_task:
        raise ValueError("At least one progressive task root is required")

    rows: list[dict[str, Any]] = []
    task_contracts: dict[str, Any] = {}
    split_schemes: set[str] = set()
    manifests: list[dict[str, Any]] = []
    for task, task_root in roots_by_task.items():
        manifest = _load_json(task_root / "experiment_manifest.json")
        manifests.append(manifest)
        split_schemes.add(str(manifest.get("split_scheme") or "scaffold"))
        if manifest.get("experiment") not in {
            "conditioned_assay_progressive_visible.v6",
            "conditioned_assay_progressive_visible.v7",
            "conditioned_assay_progressive_visible.v8",
        }:
            raise ValueError(f"Not a progressive experiment: {task_root}")
        if task not in manifest.get("tasks", []):
            raise ValueError(f"Task {task} is absent from {task_root}")
        if int(manifest.get("n_failed_queries") or 0):
            raise ValueError(
                f"Progressive run has failed queries for {task}: "
                f"{manifest['n_failed_queries']}"
            )
        expected_indices = [
            int(value) for value in manifest["evaluation_indices_by_task"][task]
        ]
        expected_n = len(expected_indices)
        task_label = CONDITIONED_TASK_SPECS[task][0]
        for level_row in manifest["inputs"][task]["levels"]:
            level = int(level_row["level"])
            metrics_path = (
                task_root / task / "levels" / f"level_{level}" / "metrics.json"
            )
            metrics = _load_json(metrics_path)
            issue = _complete_agent_metric_issue(metrics, expected_n)
            if issue:
                raise ValueError(f"Incomplete progressive metric for {task}/L{level}: {issue}")

            active_molecules: list[int] = []
            active_cards: list[int] = []
            cards_per_molecule: list[float] = []
            reasoning_tokens: list[int] = []
            reasoning_chars: list[int] = []
            prompt_tokens: list[int] = []
            completion_tokens: list[int] = []
            missing_reasoning_usage = 0
            statuses: dict[str, int] = {}
            for query_index in expected_indices:
                level_dir = (
                    task_root
                    / task
                    / "queries"
                    / f"query_idx{query_index:05d}"
                    / "levels"
                    / f"level_{level}"
                )
                prepared = _load_json(level_dir / "prepared.json")
                output = _load_json(level_dir / "output.json")
                n_molecules = int(prepared["n_active_molecules"])
                n_cards = int(prepared["n_active_cards"])
                active_molecules.append(n_molecules)
                active_cards.append(n_cards)
                cards_per_molecule.append(
                    n_cards / n_molecules if n_molecules else 0.0
                )
                status = str(output["status"])
                statuses[status] = statuses.get(status, 0) + 1
                if output.get("model_called") is not True:
                    continue
                usage = output.get("llm", {}).get("usage") or {}
                reasoning_text = str(output.get("llm", {}).get("reasoning_content") or "")
                reasoning_token_count = _reasoning_tokens_from_usage(usage)
                if reasoning_token_count is None:
                    missing_reasoning_usage += 1
                    reasoning_token_count = 0
                reasoning_tokens.append(reasoning_token_count)
                reasoning_chars.append(len(reasoning_text))
                prompt_tokens.append(int(usage.get("prompt_tokens") or 0))
                completion_tokens.append(int(usage.get("completion_tokens") or 0))

            n_model_called = len(reasoning_tokens)
            if n_model_called != int(metrics["n_model_called"]):
                raise ValueError(
                    f"Progressive call-count mismatch for {task}/L{level}: "
                    f"{n_model_called} != {metrics['n_model_called']}"
                )
            if missing_reasoning_usage:
                raise ValueError(f"Missing reasoning tokens for {task}/L{level}")

            rows.append(
                {
                    "task": task,
                    "task_label": task_label,
                    "level": level,
                    "family": str(level_row["endpoint_group"]),
                    "cumulative_assays": int(level_row["cumulative_physical_assays"]),
                    "n_queries": expected_n,
                    "macro_f1": _metric_value(metrics, "macro_f1"),
                    "accuracy": _metric_value(metrics, "accuracy"),
                    "mean_active_molecules": mean(active_molecules),
                    "mean_active_record_cards": mean(active_cards),
                    "mean_cards_per_active_molecule": mean(cards_per_molecule),
                    "n_model_called": n_model_called,
                    "model_call_fraction": n_model_called / expected_n,
                    "n_carried_forward": statuses.get("carried_forward", 0),
                    "n_reused_none": statuses.get("reused_none", 0),
                    "mean_reasoning_tokens_per_call": mean(reasoning_tokens),
                    "mean_reasoning_tokens_per_query": sum(reasoning_tokens)
                    / expected_n,
                    "mean_reasoning_chars_per_call": mean(reasoning_chars),
                    "mean_prompt_tokens_per_call": mean(prompt_tokens),
                    "mean_completion_tokens_per_call": mean(completion_tokens),
                    "metrics_path": str(metrics_path),
                }
            )

        task_contracts[task] = {
            "progressive_root": str(task_root),
            "experiment": manifest["experiment"],
            "evaluation_subset": manifest["evaluation_subset"],
            "visibility_mode": manifest["visibility_mode"],
            "reference_pool": manifest["reference_pool"],
            "neighbor_identity_policy": manifest["neighbor_identity_policy"],
            "selection": manifest["selection"],
            "agent_model": manifest["model"],
            "n": expected_n,
            "split_scheme": str(manifest.get("split_scheme") or "scaffold"),
        }

    if len(split_schemes) != 1:
        raise ValueError(f"Mixed progressive split schemes: {sorted(split_schemes)}")
    first_manifest = manifests[0]

    return rows, {
        "comparison_contract": {
            "experiment": first_manifest["experiment"],
            "evaluation_subset": first_manifest["evaluation_subset"],
            "visibility_mode": first_manifest["visibility_mode"],
            "reference_pool": first_manifest["reference_pool"],
            "neighbor_identity_policy": first_manifest["neighbor_identity_policy"],
            "selection": first_manifest["selection"],
            "agent_model": first_manifest["model"],
            "split_scheme": next(iter(split_schemes)),
            "reasoning_mode": {
                "reasoning_effort": first_manifest["reasoning_effort"],
                "thinking": first_manifest["thinking"],
                "length_statistic": "mean reasoning tokens among actual model calls",
            },
            "active_evidence_statistic": (
                "mean cumulative active molecules and per-query record-cards-per-active-"
                "molecule across all queries at each level"
            ),
            "progressive_root": (
                str(progressive_root) if progressive_root is not None else ""
            ),
            "task_contracts": task_contracts,
        },
        "rows": rows,
    }


def collect_conditioned_progressive_agent_baseline_data(
    *,
    progressive_roots_by_task: dict[str, Path],
    none_root: Path,
    baseline_root: Path,
    baseline_roots_by_task: dict[str, Path] | None = None,
    omit_mismatched_baselines: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect mixed-lineage progressive curves with matched current baselines."""
    tasks = tuple(progressive_roots_by_task)
    if not tasks or any(task not in CONDITIONED_TASK_SPECS for task in tasks):
        raise ValueError("Progressive task roots must use known conditioned tasks")

    rows: list[dict[str, Any]] = []
    contracts: dict[str, Any] = {}
    baseline_omissions: list[dict[str, Any]] = []
    effective_baseline_roots: dict[str, str] = {}
    split_schemes: set[str] = set()
    for task, progressive_root in progressive_roots_by_task.items():
        manifest = _load_json(progressive_root / "experiment_manifest.json")
        split_schemes.add(str(manifest.get("split_scheme") or "scaffold"))
        if manifest.get("experiment") not in {
            "conditioned_assay_progressive_visible.v6",
            "conditioned_assay_progressive_visible.v7",
            "conditioned_assay_progressive_visible.v8",
        }:
            raise ValueError(f"Not a supported progressive experiment: {progressive_root}")
        if task not in manifest.get("tasks", []):
            raise ValueError(f"Task {task} is absent from {progressive_root}")
        if int(manifest.get("n_failed_queries") or 0):
            raise ValueError(
                f"Progressive run has failed queries for {task}: "
                f"{manifest['n_failed_queries']}"
            )

        expected_indices = [
            int(value) for value in manifest["evaluation_indices_by_task"][task]
        ]
        expected_n = len(expected_indices)
        task_label, baseline_task = CONDITIONED_TASK_SPECS[task]
        task_baseline_root = (baseline_roots_by_task or {}).get(task, baseline_root)
        effective_baseline_roots[task] = str(task_baseline_root)
        matched_none_path = progressive_root / task / "none" / "metrics.json"
        none_path = (
            matched_none_path
            if matched_none_path.is_file()
            else none_root / task / "none" / "metrics.json"
        )
        none_metrics = _load_json(none_path)
        issue = _complete_agent_metric_issue(none_metrics, expected_n)
        if issue:
            raise ValueError(f"Incomplete no-retrieval metric for {task}: {issue}")
        rows.append(
            {
                "task": task,
                "task_label": task_label,
                "result_type": "agent_level",
                "method": "none",
                "plot_label": "None",
                "level": 0,
                "assay_count": 0,
                "macro_f1": _metric_value(none_metrics, "macro_f1"),
                "accuracy": _metric_value(none_metrics, "accuracy"),
                "n": expected_n,
                "metrics_path": str(none_path),
            }
        )

        levels = list(manifest["inputs"][task]["levels"])
        final_level = int(levels[-1]["level"])
        for level_row in levels:
            level = int(level_row["level"])
            metrics_path = (
                progressive_root / task / "levels" / f"level_{level}" / "metrics.json"
            )
            metrics = _load_json(metrics_path)
            issue = _complete_agent_metric_issue(metrics, expected_n)
            if issue:
                raise ValueError(
                    f"Incomplete progressive metric for {task}/L{level}: {issue}"
                )
            plot_label = f"L{level}"
            if level == final_level:
                plot_label += " (all)"
            rows.append(
                {
                    "task": task,
                    "task_label": task_label,
                    "result_type": "agent_level",
                    "method": f"progressive_level_{level}",
                    "plot_label": plot_label,
                    "level": level,
                    "assay_count": int(level_row["cumulative_physical_assays"]),
                    "macro_f1": _metric_value(metrics, "macro_f1"),
                    "accuracy": _metric_value(metrics, "accuracy"),
                    "n": expected_n,
                    "metrics_path": str(metrics_path),
                }
            )

        baseline_omissions.extend(
            _append_conditioned_baselines(
                rows,
                task=task,
                task_label=task_label,
                baseline_task=baseline_task,
                baseline_root=task_baseline_root,
                expected_n=expected_n,
                omit_sample_count_mismatch=omit_mismatched_baselines,
            )
        )
        contracts[task] = {
            "progressive_root": str(progressive_root),
            "experiment": manifest["experiment"],
            "evaluation_subset": manifest["evaluation_subset"],
            "visibility_mode": manifest["visibility_mode"],
            "reference_pool": manifest["reference_pool"],
            "neighbor_identity_policy": manifest["neighbor_identity_policy"],
            "selection": manifest["selection"],
            "n": expected_n,
            "split_scheme": str(manifest.get("split_scheme") or "scaffold"),
        }

    if len(split_schemes) != 1:
        raise ValueError(f"Mixed progressive split schemes: {sorted(split_schemes)}")
    return rows, {
        "comparison_contract": {
            "tasks": list(tasks),
            "split_scheme": next(iter(split_schemes)),
            "task_contracts": contracts,
            "none_lineage": str(none_root),
            "baseline_lineage": str(baseline_root),
            "baseline_lineage_by_task": effective_baseline_roots,
            "agent_inclusion_gate": (
                "complete n_total, n_successful, n_evaluable, and n_failed_runs=0"
            ),
            "baseline_methods": [method for method, _, _ in CONDITIONED_BASELINES],
            "baseline_omissions": baseline_omissions,
        },
        "rows": rows,
    }


def collect_conditioned_progressive_overview_data(
    *,
    progressive_roots_by_task: dict[str, Path],
    none_root: Path,
    baseline_root: Path,
    baseline_roots_by_task: dict[str, Path] | None = None,
    omit_mismatched_baselines: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect one table for progressive performance, baselines, and resources."""
    rows, performance_summary = collect_conditioned_progressive_agent_baseline_data(
        progressive_roots_by_task=progressive_roots_by_task,
        none_root=none_root,
        baseline_root=baseline_root,
        baseline_roots_by_task=baseline_roots_by_task,
        omit_mismatched_baselines=omit_mismatched_baselines,
    )
    resource_rows, resource_summary = collect_conditioned_progressive_resource_data(
        progressive_roots_by_task=progressive_roots_by_task,
    )
    resource_by_level = {
        (str(row["task"]), int(row["level"])): row for row in resource_rows
    }
    merged: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if row["result_type"] == "agent_level" and int(row["level"]) > 0:
            resource = resource_by_level[(str(row["task"]), int(row["level"]))]
            if abs(float(resource["macro_f1"]) - float(row["macro_f1"])) > 1e-12:
                raise ValueError(
                    f"Progressive metric mismatch for {row['task']}/L{row['level']}"
                )
            for key, value in resource.items():
                if key not in item or key == "metrics_path":
                    item[key] = value
        elif row["result_type"] == "agent_level":
            item.update(
                {
                    "mean_active_molecules": 0.0,
                    "mean_active_record_cards": 0.0,
                    "mean_cards_per_active_molecule": 0.0,
                    "n_model_called": "",
                    "model_call_fraction": "",
                    "mean_prompt_tokens_per_call": "",
                    "mean_reasoning_tokens_per_call": "",
                }
            )
        merged.append(item)

    return merged, {
        "comparison_contract": {
            "figure": "conditioned_progressive_overview.v1",
            "performance": performance_summary["comparison_contract"],
            "resources": resource_summary["comparison_contract"],
            "cards_per_molecule_statistic": (
                "mean over queries of active record cards divided by active molecules; "
                "queries with zero active molecules contribute zero"
            ),
            "prompt_and_reasoning_statistic": (
                "mean tokens among actual DeepSeek model calls; carry-forward and "
                "reused-none checkpoints are excluded"
            ),
        },
        "rows": merged,
    }












def plot_conditioned_progressive_overview(
    *,
    rows: list[dict[str, Any]],
    output_svg: Path,
    output_png: Path,
    tasks: tuple[str, ...] = ("bbb_martins", "bioavailability_ma", "skin_reaction"),
    split_scheme: str = "scaffold",
) -> None:
    """Plot performance, baselines, retrieval volume, and token use together."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(5, len(tasks), figsize=(18.5, 21.5), squeeze=False)
    baseline_styles = {
        "minimol_head": ("D", "#D89C21", "#D89C21"),
        "minimol_knn_condition": ("s", "#666666", "#666666"),
        "minimol_knn_all": ("s", "white", "#666666"),
        "morgan_knn_condition": ("^", "#666666", "#666666"),
        "morgan_knn_all": ("^", "white", "#666666"),
    }
    baseline_ticks = {
        "minimol_head": "MM\nhead",
        "minimol_knn_condition": "MM KNN\ncond.",
        "minimol_knn_all": "MM KNN\nall",
        "morgan_knn_condition": "Morgan\ncond.",
        "morgan_knn_all": "Morgan\nall",
    }
    resource_specs = (
        (
            "mean_active_molecules",
            "B  Mean retrieved molecules",
            "Active molecules/query",
            lambda value: f"{value:.1f}",
            1.18,
        ),
        (
            "mean_cards_per_active_molecule",
            "C  Mean cards per molecule",
            "Record cards/molecule",
            lambda value: f"{value:.2f}",
            1.18,
        ),
        (
            "mean_prompt_tokens_per_call",
            "D  Prompt length",
            "Prompt tokens/call (k)",
            lambda value: f"{value / 1000:.1f}k",
            1.2,
        ),
        (
            "mean_reasoning_tokens_per_call",
            "E  Reasoning length",
            "Reasoning tokens/call (k)",
            lambda value: f"{value / 1000:.1f}k",
            1.2,
        ),
    )

    for column, task in enumerate(tasks):
        task_rows = [row for row in rows if row["task"] == task]
        agent_rows = sorted(
            (row for row in task_rows if row["result_type"] == "agent_level"),
            key=lambda row: int(row["level"]),
        )
        baseline_by_method = {
            str(row["method"]): row
            for row in task_rows
            if row["result_type"] == "baseline"
        }
        baseline_rows = [
            baseline_by_method[method]
            for method, _, _ in CONDITIONED_BASELINES
            if method in baseline_by_method
        ]
        if not agent_rows:
            raise ValueError(f"No progressive agent rows for {task}")
        color = TASK_SPECS[task].color

        ax = axes[0, column]
        agent_x = list(range(len(agent_rows)))
        baseline_start = len(agent_rows) + 0.75
        baseline_x = [
            baseline_start + 0.82 * index for index in range(len(baseline_rows))
        ]
        agent_values = [float(row["macro_f1"]) for row in agent_rows]
        ax.plot(
            agent_x,
            agent_values,
            color=color,
            marker=TASK_SPECS[task].marker,
            linewidth=2.4,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=3,
        )
        for x_value, value in zip(agent_x, agent_values, strict=True):
            ax.annotate(
                f"{value:.3f}",
                (x_value, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7.2,
                color=color,
            )
        for index, (x_value, row) in enumerate(
            zip(baseline_x, baseline_rows, strict=True)
        ):
            marker, face, edge = baseline_styles[str(row["method"])]
            value = float(row["macro_f1"])
            ax.scatter(
                [x_value],
                [value],
                marker=marker,
                s=58,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.25,
                zorder=3,
            )
            ax.annotate(
                f"{value:.3f}",
                (x_value, value),
                xytext=(0, 7 if index % 2 == 0 else -11),
                textcoords="offset points",
                ha="center",
                va="bottom" if index % 2 == 0 else "top",
                fontsize=6.9,
                color="#444444",
            )
        all_values = agent_values + [float(row["macro_f1"]) for row in baseline_rows]
        lower = max(0.0, math.floor((min(all_values) - 0.02) * 50) / 50)
        upper = min(1.0, math.ceil((max(all_values) + 0.02) * 50) / 50)
        ax.set_ylim(lower, upper)
        right_limit = baseline_x[-1] + 0.55 if baseline_x else len(agent_rows) - 0.45
        ax.set_xlim(-0.55, right_limit)
        if baseline_x:
            ax.axvline(
                len(agent_rows) - 0.12,
                color="#C8C8C8",
                linewidth=1,
                linestyle="--",
            )
        else:
            ax.text(
                0.99,
                0.12,
                "Matched baselines pending",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.2,
                color="#777777",
            )
        labels = [
            "None"
            if int(row["level"]) == 0
            else (
                f"L{int(row['level'])}\n"
                + (
                    f"{int(row['assay_count']) / 1000:.1f}k"
                    if int(row["assay_count"]) >= 1000
                    else f"{int(row['assay_count']):,}"
                )
            )
            for row in agent_rows
        ] + [baseline_ticks[str(row["method"])] for row in baseline_rows]
        ax.set_xticks(agent_x + baseline_x, labels=labels)
        ax.tick_params(axis="x", labelrotation=0, labelsize=6.8)
        ax.set_title(
            f"{CONDITIONED_TASK_SPECS[task][0]}  (n={agent_rows[0]['n']})",
            fontsize=13,
            fontweight="bold",
        )
        ax.text(
            0.99,
            0.03,
            "Focused y-axis",
            transform=ax.transAxes,
            ha="right",
            fontsize=7.2,
            color="#777777",
        )
        ax.grid(axis="y", color="#E1E1E1", linewidth=0.8)
        if column == 0:
            ax.set_ylabel("A  Macro-F1")

        level_rows = [row for row in agent_rows if int(row["level"]) > 0]
        x_values = [int(row["level"]) for row in level_rows]
        for row_index, (field, panel_label, ylabel, formatter, padding) in enumerate(
            resource_specs,
            start=1,
        ):
            ax = axes[row_index, column]
            raw_values = [float(row[field]) for row in level_rows]
            plot_values = (
                [value / 1000 for value in raw_values]
                if field in {
                    "mean_prompt_tokens_per_call",
                    "mean_reasoning_tokens_per_call",
                }
                else raw_values
            )
            ax.plot(
                x_values,
                plot_values,
                color=color,
                marker=TASK_SPECS[task].marker,
                linewidth=2.2,
                markersize=6.2,
                markeredgecolor="white",
                markeredgewidth=0.7,
                zorder=3,
            )
            for x_value, raw_value, plot_value in zip(
                x_values, raw_values, plot_values, strict=True
            ):
                ax.annotate(
                    formatter(raw_value),
                    (x_value, plot_value),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.2,
                    color=color,
                )
            maximum = max(plot_values) if plot_values else 0.0
            ax.set_ylim(0, maximum * padding if maximum else 1.0)
            ax.set_xlim(0.65, max(x_values) + 0.35)
            ax.set_xticks(x_values, [f"L{level}" for level in x_values])
            ax.grid(axis="y", color="#E1E1E1", linewidth=0.8)
            if column == 0:
                ax.set_ylabel(f"{panel_label}\n{ylabel}")
            if row_index == len(resource_specs):
                ax.set_xlabel("Progressive assay-family level")

    fig.suptitle(
        "Progressive agent performance, retrieval context, and inference length",
        fontsize=17,
        fontweight="bold",
        x=0.055,
        ha="left",
    )
    fig.text(
        0.055,
        0.958,
        f"{split_scheme.title()} validation · visible append-only DeepSeek-V4-Flash updates · task-specific source-purity contracts · latest complete three-task runs",
        fontsize=10,
        color="#555555",
    )
    legend_handles = [
        Line2D([0], [0], color="#444444", marker="o", linewidth=2.2, label="Progressive agent / None")
    ]
    for method, label, _ in CONDITIONED_BASELINES:
        marker, face, edge = baseline_styles[method]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="none",
                markerfacecolor=face,
                markeredgecolor=edge,
                label=label,
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.026),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.text(
        0.055,
        0.066,
        "None is the matched no-retrieval DeepSeek result; shown baseline points use the latest sample-count-matched condition-aware MiniMol head and k=3 MiniMol/Morgan KNN receipts. "
        "\n"
        "Molecules and cards/molecule are cumulative active evidence averaged over all queries. Prompt and reasoning lengths are means among actual model calls only; carry-forward and reused-none checkpoints are excluded.",
        fontsize=8.2,
        color="#555555",
        linespacing=1.4,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.93,
        bottom=0.12,
        hspace=0.52,
        wspace=0.22,
    )
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, bbox_inches="tight")
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_conditioned_progressive_resources(
    *,
    rows: list[dict[str, Any]],
    output_svg: Path,
    output_png: Path,
    split_scheme: str = "scaffold",
) -> None:
    """Plot progressive performance and cumulative context-resource use."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.8), sharex=True)
    panels = (
        (axes[0, 0], "macro_f1", "A  Predictive performance", "Macro-F1"),
        (
            axes[0, 1],
            "mean_active_molecules",
            "B  Active molecule context",
            "Mean active molecules per query",
        ),
        (
            axes[1, 0],
            "mean_active_record_cards",
            "C  Active evidence-record context",
            "Mean active record cards per query",
        ),
        (
            axes[1, 1],
            "mean_reasoning_tokens_per_call",
            "D  DeepSeek reasoning length",
            "Mean reasoning tokens per model call (k)",
        ),
    )
    annotation_offsets = {
        "bbb_martins": (0, 8, "center"),
        "bioavailability_ma": (0, -15, "center"),
        "skin_reaction": (-8, 8, "right"),
    }

    for ax, field, title, ylabel in panels:
        for task in ("bbb_martins", "bioavailability_ma", "skin_reaction"):
            task_rows = sorted(
                (row for row in rows if row["task"] == task),
                key=lambda row: int(row["level"]),
            )
            spec = TASK_SPECS[task]
            x_values = [int(row["level"]) for row in task_rows]
            raw_values = [float(row[field]) for row in task_rows]
            y_values = (
                [value / 1000 for value in raw_values]
                if field == "mean_reasoning_tokens_per_call"
                else raw_values
            )
            ax.plot(
                x_values,
                y_values,
                color=spec.color,
                marker=spec.marker,
                markersize=6.5,
                linewidth=2.2,
                markeredgecolor="white",
                markeredgewidth=0.7,
                label=spec.label,
                zorder=3,
            )
            dx, dy, ha = annotation_offsets[task]
            for x_value, y_value in zip(x_values, y_values, strict=True):
                if field == "macro_f1":
                    label = f"{y_value:.3f}"
                elif field == "mean_reasoning_tokens_per_call":
                    label = f"{y_value:.1f}k"
                else:
                    label = f"{y_value:.1f}"
                ax.annotate(
                    label,
                    (x_value, y_value),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha=ha,
                    va="bottom" if dy >= 0 else "top",
                    fontsize=7.6,
                    color=spec.color,
                )

        values = [
            float(row[field]) / 1000
            if field == "mean_reasoning_tokens_per_call"
            else float(row[field])
            for row in rows
        ]
        if field == "macro_f1":
            lower = max(0.0, math.floor((min(values) - 0.015) * 50) / 50)
            upper = min(1.0, math.ceil((max(values) + 0.015) * 50) / 50)
            ax.set_ylim(lower, upper)
            ax.text(
                0.99,
                0.03,
                "Focused y-axis",
                transform=ax.transAxes,
                ha="right",
                fontsize=7.5,
                color="#777777",
            )
        else:
            ax.set_ylim(0, max(values) * 1.18)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, 7), [f"L{level}" for level in range(1, 7)])
        ax.set_xlim(0.7, 6.3)
        ax.grid(axis="y", color="#E1E1E1", linewidth=0.8)

    for ax in axes[1, :]:
        ax.set_xlabel("Progressive assay-family level")
    fig.suptitle(
        "Progressive assay-family performance and resource use",
        fontsize=16,
        fontweight="bold",
        x=0.06,
        ha="left",
    )
    fig.text(
        0.06,
        0.925,
        f"{split_scheme.title()} validation · visible append-only updates · L1 direct evidence followed by task-specific indirect families",
        fontsize=9.5,
        color="#555555",
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=TASK_SPECS[task].color,
                marker=TASK_SPECS[task].marker,
                linewidth=2.2,
                label=TASK_SPECS[task].label,
            )
            for task in ("bbb_martins", "bioavailability_ma", "skin_reaction")
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=3,
        frameon=False,
    )
    fig.text(
        0.06,
        0.062,
        "Molecules and record cards are cumulative active evidence averaged over every query at that level. "
        "Reasoning length is averaged only over actual DeepSeek calls; carry-forward and reused-none checkpoints are excluded. "
        "Exact call counts, call fractions, prompt tokens, and amortized reasoning tokens are in the companion TSV.",
        fontsize=8.1,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        top=0.88,
        bottom=0.15,
        hspace=0.34,
        wspace=0.19,
    )
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, bbox_inches="tight")
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)




def collect_relevance_decay_data(
    *,
    ranked_assay_paths: dict[str, Path],
    prefixes_by_task: dict[str, list[int]],
) -> list[dict[str, Any]]:
    """Compute cumulative relevance summaries for frozen assay rankings."""
    rows: list[dict[str, Any]] = []
    for task, prefixes in prefixes_by_task.items():
        path = ranked_assay_paths[task]
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        scores = [float(record["relevance_score"]) for record in records]
        if not scores or any(not 0 <= score <= 100 for score in scores):
            raise ValueError(f"Invalid relevance scores: {path}")
        if any(left < right for left, right in zip(scores, scores[1:])):
            raise ValueError(f"Assay ranking is not score-descending: {path}")
        for prefix in prefixes:
            if prefix > len(scores):
                raise ValueError(
                    f"Prefix {prefix} exceeds {len(scores)} ranked assays: {path}"
                )
            selected = scores[:prefix]
            rows.append(
                {
                    "task": task,
                    "task_label": TASK_SPECS[task].label,
                    "assay_count": prefix,
                    "catalog_size": len(scores),
                    "mean_relevance_score": mean(selected),
                    "boundary_relevance_score": selected[-1],
                    "minimum_selected_score": min(selected),
                    "maximum_selected_score": max(selected),
                    "ranked_assays_path": str(path),
                }
            )
    return rows


def _draw_assay_relevance_decay(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    panel_title: str = "",
) -> None:
    """Draw cumulative assay relevance on an existing axis."""
    tasks = list(dict.fromkeys(row["task"] for row in rows))
    for task in tasks:
        spec = TASK_SPECS[task]
        task_rows = sorted(
            (row for row in rows if row["task"] == task),
            key=lambda row: row["assay_count"],
        )
        ax.plot(
            [row["assay_count"] for row in task_rows],
            [row["mean_relevance_score"] for row in task_rows],
            color=spec.color,
            marker=spec.marker,
            markersize=6.5,
            linewidth=2.2,
            label=spec.label,
        )
        endpoint = task_rows[-1]
        ax.annotate(
            f"{spec.label}: {endpoint['mean_relevance_score']:.1f}",
            (endpoint["assay_count"], endpoint["mean_relevance_score"]),
            xytext=(-5, 8),
            textcoords="offset points",
            ha="right",
            color=spec.color,
            fontsize=8.5,
        )

    ticks = sorted({int(row["assay_count"]) for row in rows})
    tick_labels: list[str] = []
    for index, value in enumerate(ticks):
        label = f"{value:,}"
        if index and value / ticks[index - 1] < 1.6:
            label = "\n" + label
        tick_labels.append(label)
    ax.set_xscale("log", base=4)
    ax.set_xticks(ticks, labels=tick_labels)
    ax.tick_params(axis="x", labelrotation=28)
    ax.set_xlim(min(ticks) / 1.25, max(ticks) * 1.15)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Number of retrieved assays (log scale)")
    ax.set_ylabel("Mean relevance score among retrieved assays (0–100)")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.legend(frameon=False, loc="lower left")
    if panel_title:
        ax.set_title(panel_title, loc="left")


def plot_assay_relevance_decay(
    *,
    rows: list[dict[str, Any]],
    output_svg: Path,
    output_png: Path,
) -> None:
    """Plot cumulative mean relevance as progressively more assays are retrieved."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    _draw_assay_relevance_decay(ax, rows)
    fig.suptitle(
        "Mean relevance of retrieved assays",
        fontsize=15,
        fontweight="bold",
        x=0.08,
        ha="left",
    )
    fig.text(
        0.08,
        0.905,
        "Cumulative mean relevance within each frozen Top-N assay ranking",
        fontsize=9.5,
        color="#555555",
    )
    fig.text(
        0.08,
        0.025,
        "Scores were assigned offline by Codex GPT-5.6 Sol; each curve includes the full assay catalog.",
        fontsize=8.2,
        color="#555555",
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.84, bottom=0.2)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, bbox_inches="tight")
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _current_relevance_rows(tasks: list[str]) -> list[dict[str, Any]]:
    """Load frozen assay rankings using the current task-specific prefix plan."""
    from tools.chembl_tool.paper_experiments.run_assay_retrieval_curve import (
        TASKS,
        prefix_plan,
    )

    return collect_relevance_decay_data(
        ranked_assay_paths={
            task: Path(TASKS[task]["ranked_assays"]) for task in tasks
        },
        prefixes_by_task=prefix_plan(tasks),
    )


def write_analysis(
    *,
    analysis_dir: Path,
    performance_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _write_tsv(analysis_dir / "assay_curve_metrics.tsv", performance_rows)
    _write_tsv(analysis_dir / "assay_retrieval_volume.tsv", retrieval_rows)
    _write_tsv(analysis_dir / "assay_vs_group_best.tsv", summary["best"])
    payload = {
        "comparison_contract": {
            "split": "scaffold-valid",
            "assay_retrieval_contract": summary["experiment"].get(
                "retrieval_contract", ""
            ),
            "assay_reference_pool": summary["experiment"].get("reference_pool", ""),
            "visibility_mode": "identity_blind",
            "neighbor_identity_policy": summary["experiment"].get(
                "neighbor_identity_policy", ""
            ),
            "zero_retrieval": "reused_historical_none",
            "comparison_type": "descriptive_historical_reference_not_endpoint_matched",
        },
        "performance": performance_rows,
        "retrieval_volume": retrieval_rows,
        "best_comparison": summary["best"],
    }
    (analysis_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _set_assay_axis(ax: plt.Axes, ticks: list[int]) -> None:
    max_count = max(ticks)
    displayed = [0]
    value = 5
    while value <= max_count:
        displayed.append(value)
        value *= 4
    ax.set_xscale("symlog", base=4, linthresh=5, linscale=0.8)
    ax.set_xticks(
        displayed,
        labels=["None\n(0)" if value == 0 else f"{value:,}" for value in displayed],
    )
    ax.tick_params(axis="x", labelrotation=25)
    ax.set_xlabel("Number of retrieved assays (symmetric log scale)")


def _best_panel_xlim(best_rows: list[dict[str, Any]]) -> tuple[float, float]:
    """Return rounded, data-driven bounds with room for delta annotations."""
    values = [
        float(row[field])
        for row in best_rows
        for field in ("group_best_macro_f1", "assay_best_macro_f1")
    ]
    if not values:
        return 0.0, 1.0

    value_min = min(values)
    value_max = max(values)
    span = max(value_max - value_min, 0.05)
    lower_padding = max(0.01, span * 0.08)
    upper_padding = max(0.015, span * 0.15)
    tick_step = 0.01
    lower = max(0.0, math.floor((value_min - lower_padding) / tick_step) * tick_step)
    upper = min(1.0, math.ceil((value_max + upper_padding) / tick_step) * tick_step)
    return lower, upper


def plot_assay_retrieval_curves(
    *,
    performance_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    relevance_rows: list[dict[str, Any]],
    output_svg: Path,
    output_png: Path,
) -> None:
    """Render the reusable five-panel assay retrieval figure."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(13.4, 13.0))
    grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 0.95))
    ax_perf = fig.add_subplot(grid[0, 0])
    ax_best = fig.add_subplot(grid[0, 1])
    ax_molecules = fig.add_subplot(grid[1, 0])
    ax_records = fig.add_subplot(grid[1, 1])
    ax_relevance = fig.add_subplot(grid[2, :])
    tasks = list(dict.fromkeys(row["task"] for row in performance_rows))
    performance_ticks = sorted({int(row["assay_count"]) for row in performance_rows})
    assay_ticks = sorted({int(row["assay_count"]) for row in retrieval_rows})

    for task in tasks:
        spec = TASK_SPECS[task]
        perf = sorted(
            (row for row in performance_rows if row["task"] == task),
            key=lambda row: row["assay_count"],
        )
        volume = sorted(
            (row for row in retrieval_rows if row["task"] == task),
            key=lambda row: row["assay_count"],
        )
        style = {
            "color": spec.color,
            "marker": spec.marker,
            "markersize": 6.5,
            "linewidth": 2.0,
            "label": spec.label,
        }
        ax_perf.plot(
            [row["assay_count"] for row in perf],
            [row["macro_f1"] for row in perf],
            **style,
        )
        ax_molecules.plot(
            [row["assay_count"] for row in volume],
            [row["mean_unique_molecules"] for row in volume],
            **style,
        )
        ax_records.plot(
            [row["assay_count"] for row in volume],
            [row["mean_source_records_represented"] for row in volume],
            **style,
        )
        for ax, field in (
            (ax_molecules, "mean_unique_molecules"),
            (ax_records, "mean_source_records_represented"),
        ):
            endpoint = volume[-1]
            ax.annotate(
                f"{spec.label} all\n({endpoint['assay_count']:,})",
                (endpoint["assay_count"], endpoint[field]),
                xytext=(-5, 8),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=7.5,
                color=spec.color,
            )

    for ax, title, ylabel, ticks in (
        (ax_perf, "A. Predictive performance", "Macro-F1", performance_ticks),
        (ax_molecules, "C. Retrieved molecular diversity", "Mean unique molecules per query", assay_ticks),
        (ax_records, "D. Retrieved evidence volume", "Mean source records per query", assay_ticks),
    ):
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
        _set_assay_axis(ax, ticks)
    ax_records.set_yscale("symlog", base=10, linthresh=1, linscale=0.8)
    ax_records.set_ylabel("Mean represented source records per query (symlog)")
    ax_records.set_ylim(bottom=0)
    ax_perf.legend(frameon=False, ncol=3, loc="best")

    best_by_task = {row["task"]: row for row in best_rows}
    best_xlim = _best_panel_xlim(best_rows)
    for y, task in enumerate(tasks):
        if task not in best_by_task:
            continue
        row = best_by_task[task]
        spec = TASK_SPECS[task]
        group_value = row["group_best_macro_f1"]
        assay_value = row["assay_best_macro_f1"]
        ax_best.plot([group_value, assay_value], [y, y], color="#AAAAAA", linewidth=2)
        ax_best.scatter(
            [group_value], [y], s=62, facecolor="white", edgecolor="#555555", linewidth=1.4
        )
        ax_best.scatter(
            [assay_value], [y], s=70, color=spec.color, edgecolor="white", linewidth=0.7
        )
        ax_best.text(
            max(group_value, assay_value) + 0.002,
            y,
            f"{row['delta_percentage_points']:+.1f} pp",
            va="center",
            color=spec.color,
            fontweight="bold",
        )
    ax_best.set_yticks(range(len(tasks)), labels=[TASK_SPECS[task].label for task in tasks])
    ax_best.invert_yaxis()
    ax_best.set_xlabel("Best Macro-F1")
    ax_best.set_title("B. Best assay-level vs historical group-level", loc="left")
    ax_best.grid(axis="x", color="#DDDDDD", linewidth=0.8)
    ax_best.set_xlim(*best_xlim)
    ax_best.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="none", markerfacecolor="white",
                markeredgecolor="#555555", markeredgewidth=1.4, label="Historical group-level best",
            ),
            Line2D(
                [0], [0], marker="o", linestyle="none", markerfacecolor="#555555",
                markeredgecolor="white", label="Current assay-level best",
            ),
        ],
        frameon=False,
        loc="upper left",
    )

    _draw_assay_relevance_decay(
        ax_relevance,
        relevance_rows,
        panel_title="E. Mean relevance of retrieved assays",
    )

    fig.suptitle(
        "Assay-level retrieval scaling on scaffold validation sets",
        fontsize=15,
        fontweight="bold",
        x=0.06,
        ha="left",
    )
    fig.text(
        0.06,
        0.025,
        "No-retrieval and group-level references reuse historical OpenRouter DeepSeek-V4-Flash runs; "
        "assay-level points use PARCC DeepSeek-V4-Flash-0731. This is a descriptive, not endpoint-matched, comparison. "
        "Panel E uses frozen Codex GPT-5.6 Sol assay-relevance scores.",
        fontsize=8.2,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.94,
        bottom=0.09,
        hspace=0.52,
        wspace=0.27,
    )
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, bbox_inches="tight")
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=(
            "outputs/paper/starling_assay_retrieval_curve_v5/"
            "scaffold_valid_direct_only_heldout_filtered_scaffold_disjoint_"
            "epyc_deepseek_v4_flash_0731"
        ),
    )
    parser.add_argument(
        "--historical-group-root",
        default=(
            "outputs/paper/"
            "molecular_evidence_agent_starling_scaffold_current_valid_openrouter_deepseek_v4_flash/"
            "runs_identity_blind_parent_disjoint"
        ),
    )
    parser.add_argument("--analysis-dir", default="")
    parser.add_argument("--output-stem", default="assay_retrieval_scaling")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow a diagnostic partial figure; complete zero-failure curves are required by default.",
    )
    parser.add_argument(
        "--relevance-decay-only",
        action="store_true",
        help="Plot cumulative mean relevance from frozen assay rankings without requiring LLM metrics.",
    )
    parser.add_argument(
        "--conditioned-progressive-performance-resources",
        action="store_true",
        help=(
            "Plot progressive performance, active molecules, active record "
            "cards, and reasoning-token length."
        ),
    )
    parser.add_argument(
        "--conditioned-progressive-overview",
        action="store_true",
        help=(
            "Plot one unified three-task figure containing progressive performance, "
            "None, current baselines, active molecules, cards per molecule, prompt "
            "tokens, and reasoning tokens."
        ),
    )
    parser.add_argument(
        "--omit-mismatched-progressive-baselines",
        action="store_true",
        help=(
            "For progressive overview figures, omit baseline points whose "
            "evaluation sample count does not match the agent cohort. The "
            "omissions remain explicit in the analysis summary."
        ),
    )
    parser.add_argument(
        "--conditioned-baseline-root",
        default="outputs/baselines/starling_conditioned_valid_v1",
    )
    parser.add_argument(
        "--conditioned-progressive-root",
        default=(
            "outputs/paper/"
            "starling_conditioned_assay_progressive_visible_v7_source_purity_v1/"
            "scaffold_valid_deepseek_v4_flash_0731"
        ),
    )
    parser.add_argument(
        "--conditioned-progressive-bbb-root",
        default=(
            "outputs/paper/"
            "starling_conditioned_assay_progressive_visible_v8_global_molecule_"
            "source_purity_v5/scaffold_valid_deepseek_v4_flash_0731"
        ),
    )
    parser.add_argument(
        "--conditioned-progressive-source-purity-root",
        default=(
            "outputs/paper/"
            "starling_conditioned_assay_progressive_visible_v7_source_purity_v1/"
            "scaffold_valid_deepseek_v4_flash_0731"
        ),
    )
    parser.add_argument(
        "--conditioned-progressive-task-root",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help=(
            "Override the progressive artifact root for one task in the unified "
            "overview; repeat when current task lineages live in different roots."
        ),
    )
    parser.add_argument(
        "--conditioned-progressive-baseline-root",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help=(
            "Override the baseline artifact root for one progressive task; "
            "repeat for multiple tasks. Other tasks use --conditioned-baseline-root."
        ),
    )
    parser.add_argument(
        "--conditioned-none-agent-root",
        default=(
            "outputs/paper/starling_conditioned_assay_family_curve_v1/"
            "scaffold_valid_epyc_deepseek_v4_flash_0731"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else output_root / "analysis"
    if args.conditioned_progressive_overview:
        source_purity_root = Path(args.conditioned_progressive_source_purity_root)
        baseline_roots_by_task = _parse_task_path_overrides(
            args.conditioned_progressive_baseline_root
        )
        progressive_roots = {
            "bbb_martins": Path(args.conditioned_progressive_bbb_root),
            "bioavailability_ma": source_purity_root,
            "skin_reaction": source_purity_root,
        }
        progressive_roots.update(
            _parse_task_path_overrides(args.conditioned_progressive_task_root)
        )
        if not args.analysis_dir:
            analysis_dir = (
                progressive_roots["bbb_martins"].parent
                / "analysis"
                / "three_task_progressive_overview"
            )
        rows, summary = collect_conditioned_progressive_overview_data(
            progressive_roots_by_task=progressive_roots,
            none_root=Path(args.conditioned_none_agent_root),
            baseline_root=Path(args.conditioned_baseline_root),
            baseline_roots_by_task=baseline_roots_by_task,
            omit_mismatched_baselines=args.omit_mismatched_progressive_baselines,
        )
        analysis_dir.mkdir(parents=True, exist_ok=True)
        _write_tsv(analysis_dir / "progressive_overview_metrics.tsv", rows)
        (analysis_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        figure_dir = analysis_dir / "figures"
        output_stem = (
            args.output_stem
            if args.output_stem != "assay_retrieval_scaling"
            else "three_task_progressive_overview"
        )
        plot_conditioned_progressive_overview(
            rows=rows,
            output_svg=figure_dir / f"{output_stem}.svg",
            output_png=figure_dir / f"{output_stem}.png",
            tasks=tuple(progressive_roots),
            split_scheme=str(
                summary["comparison_contract"]["performance"]["split_scheme"]
            ),
        )
        print(
            json.dumps(
                {
                    "analysis_dir": str(analysis_dir),
                    "n_metric_rows": len(rows),
                    "tasks": list(progressive_roots),
                    "figure": str(figure_dir / f"{output_stem}.png"),
                },
                indent=2,
            )
        )
        return 0
    if args.conditioned_progressive_performance_resources:
        progressive_root = Path(args.conditioned_progressive_root)
        if not args.analysis_dir:
            analysis_dir = progressive_root / "analysis" / "performance_resources"
        rows, summary = collect_conditioned_progressive_resource_data(
            progressive_root=progressive_root,
        )
        analysis_dir.mkdir(parents=True, exist_ok=True)
        _write_tsv(analysis_dir / "progressive_level_metrics.tsv", rows)
        (analysis_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        figure_dir = analysis_dir / "figures"
        output_stem = (
            args.output_stem
            if args.output_stem != "assay_retrieval_scaling"
            else "progressive_performance_resources"
        )
        plot_conditioned_progressive_resources(
            rows=rows,
            output_svg=figure_dir / f"{output_stem}.svg",
            output_png=figure_dir / f"{output_stem}.png",
            split_scheme=str(summary["comparison_contract"]["split_scheme"]),
        )
        print(
            json.dumps(
                {
                    "analysis_dir": str(analysis_dir),
                    "n_task_levels": len(rows),
                    "n_model_calls": sum(int(row["n_model_called"]) for row in rows),
                },
                indent=2,
            )
        )
        return 0
    if args.relevance_decay_only:
        tasks = list(TASK_SPECS)
        relevance_rows = _current_relevance_rows(tasks)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        _write_tsv(analysis_dir / "assay_relevance_decay.tsv", relevance_rows)
        figure_dir = analysis_dir / "figures"
        plot_assay_relevance_decay(
            rows=relevance_rows,
            output_svg=figure_dir / "assay_relevance_decay.svg",
            output_png=figure_dir / "assay_relevance_decay.png",
        )
        print(
            json.dumps(
                {
                    "analysis_dir": str(analysis_dir),
                    "n_relevance_rows": len(relevance_rows),
                },
                indent=2,
            )
        )
        return 0
    performance, retrieval, summary = collect_curve_data(
        output_root=output_root,
        historical_group_root=Path(args.historical_group_root),
    )
    incomplete = incomplete_curve_conditions(performance, summary["experiment"])
    if incomplete and not args.allow_incomplete:
        raise SystemExit(
            "Refusing to publish an incomplete assay curve:\n" + "\n".join(incomplete)
        )
    relevance_rows = _current_relevance_rows(list(summary["experiment"]["tasks"]))
    write_analysis(
        analysis_dir=analysis_dir,
        performance_rows=performance,
        retrieval_rows=retrieval,
        summary=summary,
    )
    _write_tsv(analysis_dir / "assay_relevance_decay.tsv", relevance_rows)
    figure_dir = analysis_dir / "figures"
    plot_assay_retrieval_curves(
        performance_rows=performance,
        retrieval_rows=retrieval,
        best_rows=summary["best"],
        relevance_rows=relevance_rows,
        output_svg=figure_dir / f"{args.output_stem}.svg",
        output_png=figure_dir / f"{args.output_stem}.png",
    )
    print(
        json.dumps(
            {
                "analysis_dir": str(analysis_dir),
                "n_performance_rows": len(performance),
                "n_retrieval_rows": len(retrieval),
                "n_relevance_rows": len(relevance_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
