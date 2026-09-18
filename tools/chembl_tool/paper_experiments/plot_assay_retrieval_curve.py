"""Plot reusable assay-count performance and retrieval-volume curves.

The plotter reads experiment artifacts rather than embedding metric values.  It
supports the legacy shared-prefix manifest and the task-specific full-catalog
manifest, reuses historical no-retrieval metrics at x=0, and computes retrieval
volume directly from replay payloads.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from multiprocessing import get_context
from pathlib import Path
import re
from statistics import mean, stdev
import textwrap
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tools.chembl_tool.common.json_utils import canonical_json_bytes, sha256_file


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
    "ames": TaskPlotSpec("Ames", "#009E73", "D", "", ""),
    "dili": TaskPlotSpec("DILI", "#0072B2", "o", "", ""),
    "carcinogens": TaskPlotSpec("Carcinogens", "#D55E00", "s", "", ""),
}

CONDITIONED_TASK_SPECS = {
    "bbb_martins": ("BBB", "BBB_Martins"),
    "bioavailability_ma": ("Bioavailability", "Bioavailability_Ma"),
    "skin_reaction": ("Skin", "Skin_Reaction"),
    "clintox": ("ClinTox", "ClinTox"),
    "ames": ("Ames", "Ames"),
    "dili": ("DILI", "DILI"),
    "carcinogens": ("Carcinogens", "Carcinogens"),
}

CONDITIONED_BASELINES = (
    (
        "minimol_head",
        "MiniMol head",
        Path("minimol_train_retest/final/metrics.json"),
    ),
    (
        "minimol_knn_condition",
        "MiniMol KNN condition-first",
        Path("minimol_knn/same_condition_then_null/metrics.json"),
    ),
    (
        "minimol_knn_all",
        "MiniMol KNN all train",
        Path("minimol_knn/all_train_unique_molecules/metrics.json"),
    ),
    (
        "morgan_knn_condition",
        "Morgan KNN condition-first",
        Path("morgan_knn/same_condition_then_null/metrics.json"),
    ),
    (
        "morgan_knn_all",
        "Morgan KNN all train",
        Path("morgan_knn/all_train_unique_molecules/metrics.json"),
    ),
)

CONDITIONED_BASELINE_STYLES = {
    "minimol_head": ("D", "#D89C21", "#D89C21"),
    "minimol_knn_condition": ("s", "#666666", "#666666"),
    "minimol_knn_all": ("s", "white", "#666666"),
    "morgan_knn_condition": ("^", "#666666", "#666666"),
    "morgan_knn_all": ("^", "white", "#666666"),
}

CONDITIONED_BASELINE_TICKS = {
    "minimol_head": "MM\nhead",
    "minimol_knn_condition": "MM KNN\ncond.",
    "minimol_knn_all": "MM KNN\nall",
    "morgan_knn_condition": "Morgan\ncond.",
    "morgan_knn_all": "Morgan\nall",
}


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _selection_without_card_limits(selection: Any) -> Any:
    """Remove only per-molecule card limits from a selection contract."""
    if isinstance(selection, dict):
        normalized = {}
        for key, value in selection.items():
            if "card_limit" in key:
                continue
            if (
                key in {"level_1", "later_new_pool", "later_augmentation_pool"}
                and isinstance(value, str)
            ):
                match = re.search(r"\btop\s+(\d+)\b", value, flags=re.IGNORECASE)
                if match is None:
                    raise ValueError(f"Cannot normalize legacy selection text: {value!r}")
                normalized[key] = {"molecule_limit": int(match.group(1))}
            else:
                normalized[key] = _selection_without_card_limits(value)
        return normalized
    if isinstance(selection, list):
        return [_selection_without_card_limits(value) for value in selection]
    return selection


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


def _parse_configuration_task_path_overrides(
    values: list[str],
) -> dict[str, dict[str, list[Path]]]:
    """Parse CONFIG:TASK=PATH roots, retaining exact-contract replicates."""
    configurations: dict[str, dict[str, list[Path]]] = {}
    for value in values:
        configuration, separator, task_path = value.partition(":")
        task, task_separator, raw_path = task_path.partition("=")
        if (
            not separator
            or not configuration
            or not task_separator
            or task not in CONDITIONED_TASK_SPECS
            or not raw_path
        ):
            raise ValueError(
                "Progressive configuration roots must use "
                f"CONFIG:TASK=PATH; received {value!r}"
            )
        roots = configurations.setdefault(configuration, {})
        task_roots = roots.setdefault(task, [])
        path = Path(raw_path)
        if path in task_roots:
            raise ValueError(
                f"Duplicate progressive configuration root path: "
                f"{configuration}:{task}={path}"
            )
        task_roots.append(path)
    return configurations


def _configuration_root_replicates(value: Path | list[Path]) -> list[Path]:
    """Normalize one historical root or a list of replicate roots."""
    return value if isinstance(value, list) else [value]


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
    evaluation_subset: str | None = None,
    omit_sample_count_mismatch: bool = False,
) -> list[dict[str, Any]]:
    omissions: list[dict[str, Any]] = []
    task_root = baseline_root / baseline_task
    if any((task_root / subset).is_dir() for subset in ("valid", "test")):
        if evaluation_subset not in {"valid", "test"}:
            raise ValueError("Split-scoped baselines require an explicit evaluation subset")
        task_root = task_root / evaluation_subset
        if not task_root.is_dir():
            raise FileNotFoundError(task_root)
    for method, plot_label, relative_path in CONDITIONED_BASELINES:
        metrics_path = task_root / relative_path
        if method == "minimol_head" and not metrics_path.is_file():
            candidates = (
                Path("minimol_train/final/metrics.json"),
                Path("minimol_head/final/metrics.json"),
            )
            metrics_path = next(
                (
                    task_root / candidate
                    for candidate in candidates
                    if (task_root / candidate).is_file()
                ),
                metrics_path,
            )
        if not metrics_path.is_file():
            candidate = task_root / method / "metrics.json"
            if candidate.is_file():
                metrics_path = candidate
        metrics = _load_json(metrics_path)
        actual_n = int(metrics.get("n_evaluated", _metric_n(metrics)))
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
        if manifest.get("experiment") == "conditioned_assay_matched_full_flat.v1":
            source_path = Path(manifest["matched_progressive_root"]) / "experiment_manifest.json"
            if sha256_file(source_path) != manifest["matched_source_manifest_sha256"]:
                raise ValueError(f"Matched progressive source manifest changed: {source_path}")
            source_manifest = _load_json(source_path)
            # Early matched manifests inherit these fields through their pinned source.
            for field in ("min_similarity", "condition_policy"):
                manifest.setdefault(field, source_manifest.get(field))
            if manifest["model"] == source_manifest["model"]:
                manifest.setdefault("model_identity", source_manifest.get("model_identity", manifest["model"]))
        manifests.append(manifest)
        split_schemes.add(str(manifest.get("split_scheme") or "scaffold"))
        if manifest.get("experiment") not in {
            "conditioned_assay_progressive_visible.v6",
            "conditioned_assay_progressive_visible.v7",
            "conditioned_assay_progressive_visible.v8",
            "conditioned_assay_matched_full_flat.v1",
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
        prepared_digests = []
        evidence_digests = []
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
            missing_prompt_usage = 0
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
                prepared_digests.append((query_index, level, _sha256_json(prepared)))
                evidence_digests.append((query_index, level, _sha256_json({
                    key: value for key, value in prepared.items()
                    if key not in {"query_prior", "reused_none_final", "reused_single_source_index"}
                })))
                if output.get("status") not in {"ok", "carried_forward", "reused_none"}:
                    raise ValueError(f"Unsuccessful level output: {level_dir}")
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
                prompt_token_count = usage.get("prompt_tokens")
                if not isinstance(prompt_token_count, int) or prompt_token_count <= 0:
                    missing_prompt_usage += 1
                    prompt_token_count = 0
                prompt_tokens.append(prompt_token_count)
                completion_tokens.append(int(usage.get("completion_tokens") or 0))

            n_model_called = len(reasoning_tokens)
            if n_model_called != int(metrics["n_model_called"]):
                raise ValueError(
                    f"Progressive call-count mismatch for {task}/L{level}: "
                    f"{n_model_called} != {metrics['n_model_called']}"
                )
            if missing_reasoning_usage:
                raise ValueError(f"Missing reasoning tokens for {task}/L{level}")
            if missing_prompt_usage:
                raise ValueError(f"Missing prompt tokens for {task}/L{level}")

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
            "prepared_inputs_sha256": _sha256_json(prepared_digests),
            "prepared_evidence_sha256": _sha256_json(evidence_digests),
            "experiment": manifest["experiment"],
            "evaluation_subset": manifest["evaluation_subset"],
            "visibility_mode": manifest["visibility_mode"],
            "reference_pool": manifest["reference_pool"],
            "neighbor_identity_policy": manifest["neighbor_identity_policy"],
            "selection": manifest["selection"],
            "selection_without_card_limits": _selection_without_card_limits(
                manifest["selection"]
            ),
            "candidate_generation": manifest.get("candidate_generation"),
            "min_similarity": manifest.get("min_similarity"),
            "prompt_profile": manifest.get("prompt_profile"),
            "condition_policy": manifest.get("condition_policy"),
            "max_tokens": manifest.get("max_tokens"),
            "temperature": manifest.get("temperature"),
            "thinking": manifest.get("thinking"),
            "reasoning_effort": manifest.get("reasoning_effort"),
            "tool_prefetch_complete": manifest.get("tool_prefetch_complete"),
            "agent_model": manifest["model"],
            "model_identity": manifest.get("model_identity", manifest["model"]),
            "input_sha256": manifest["inputs"][task].get("input_sha256", ""),
            "evaluation_indices_sha256": _sha256_json(expected_indices),
            "index_sha256": manifest["inputs"][task].get("index_sha256", ""),
            "family_manifest_sha256": manifest["inputs"][task].get(
                "family_manifest_sha256", ""
            ),
            "execution_base_urls": sorted(
                {
                    str(provider.get("base_url") or "")
                    for provider in manifest.get("execution_providers") or []
                }
            ) or [str(manifest.get("base_url") or "")],
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


_PROGRESSIVE_REPLICATE_VALUE_FIELDS = (
    "macro_f1",
    "accuracy",
    "mean_active_molecules",
    "mean_active_record_cards",
    "mean_cards_per_active_molecule",
    "n_model_called",
    "model_call_fraction",
    "n_carried_forward",
    "n_reused_none",
    "mean_reasoning_tokens_per_call",
    "mean_reasoning_tokens_per_query",
    "mean_reasoning_chars_per_call",
    "mean_prompt_tokens_per_call",
    "mean_completion_tokens_per_call",
)


def _aggregate_progressive_replicate_rows(
    rows_by_replicate: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Average exact-contract repeats, retaining sample SD and observed range."""
    if not rows_by_replicate:
        raise ValueError("At least one progressive replicate is required")
    keys = [
        [(str(row["task"]), int(row["level"])) for row in rows]
        for rows in rows_by_replicate
    ]
    if any(item != keys[0] for item in keys[1:]):
        raise ValueError(f"Progressive replicate level mismatch: {keys}")

    aggregated: list[dict[str, Any]] = []
    for row_index in range(len(rows_by_replicate[0])):
        repeats = [rows[row_index] for rows in rows_by_replicate]
        item = dict(repeats[0])
        value_fields = tuple(
            field
            for field in _PROGRESSIVE_REPLICATE_VALUE_FIELDS
            if field in repeats[0]
        )
        for field in value_fields:
            values = [float(row[field]) for row in repeats]
            if len(repeats) > 1:
                item[field] = mean(values)
                item[f"{field}_min"] = min(values)
                item[f"{field}_max"] = max(values)
                item[f"{field}_sd"] = stdev(values)
        stable_fields = (
            set(repeats[0])
            - set(value_fields)
            - {"metrics_path"}
        )
        for field in stable_fields:
            if any(row[field] != repeats[0][field] for row in repeats[1:]):
                raise ValueError(
                    f"Progressive replicate row mismatch for {field}: "
                    f"{[row[field] for row in repeats]}"
                )
        item["n_replicates"] = len(repeats)
        item["metrics_paths"] = [
            str(row["metrics_path"]) for row in repeats if "metrics_path" in row
        ]
        item.pop("metrics_path", None)
        aggregated.append(item)
    return aggregated


def _model_plot_label(model: str) -> str:
    for name in ("flash", "pro"):
        if name in model.lower().split("-"):
            return name.title()
    return model.rsplit("/", 1)[-1]


def _replicate_bounds(row: dict[str, Any], field: str, interval: str) -> tuple[float, float]:
    value = float(row[field])
    if interval == "sd_if_repeated":
        if int(row.get("n_replicates", 1)) == 1:
            return value, value
        interval = "sd"
    if interval == "sd":
        if int(row.get("n_replicates", 1)) < 2:
            raise ValueError("Sample SD requires at least two runs per curve point")
        sd = float(row[f"{field}_sd"])
        return value - sd, value + sd
    if interval == "range":
        return float(row.get(f"{field}_min", value)), float(row.get(f"{field}_max", value))
    raise ValueError(f"Unknown replicate interval: {interval}")


def _matched_organization_comparison(reference: dict, comparison: dict, *, cross_model: bool = False) -> bool:
    """Allow only the frozen full-flat/v8 prompt contrast over identical inputs."""
    profiles = {
        "conditioned_assay_progressive_visible.v8": "progressive_compact_tools_short_aliases.v2",
        "conditioned_assay_matched_full_flat.v1": "independent_cumulative_tools_short_aliases.v1",
    }
    if {reference["experiment"], comparison["experiment"]} != set(profiles):
        return False
    for contract in (reference, comparison):
        if contract["prompt_profile"] != profiles[contract["experiment"]]:
            raise ValueError("Unsupported prompt profile for matched full-flat comparison")
    prepared_field = "prepared_evidence_sha256" if cross_model else "prepared_inputs_sha256"
    for field in ("selection", prepared_field, "index_sha256", "family_manifest_sha256"):
        if reference.get(field) is None or reference[field] != comparison.get(field):
            raise ValueError(f"Matched full-flat comparison differs in {field}")
    return True


def _collect_configuration_resource(key):
    task, root = key
    return collect_conditioned_progressive_resource_data(
        progressive_roots_by_task={task: root}
    )


def collect_conditioned_progressive_configuration_data(
    *,
    configuration_roots: dict[str, dict[str, Path | list[Path]]],
    lineage_receipts_by_task: dict[str, Path] | None = None,
    allow_model_comparison: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect configurations, averaging only exact-contract replicate roots."""
    if len(configuration_roots) < 2:
        raise ValueError("At least two progressive configurations are required")

    invariant_fields = (
        "experiment",
        "evaluation_subset",
        "visibility_mode",
        "reference_pool",
        "neighbor_identity_policy",
        "candidate_generation",
        "min_similarity",
        "selection_without_card_limits",
        "prompt_profile",
        "condition_policy",
        "max_tokens",
        "temperature",
        "thinking",
        "reasoning_effort",
        "tool_prefetch_complete",
        "model_identity",
        "input_sha256",
        "evaluation_indices_sha256",
        "n",
        "split_scheme",
    )
    replicate_invariant_fields = invariant_fields + (
        "prepared_inputs_sha256",
        "selection",
        "agent_model",
        "index_sha256",
        "family_manifest_sha256",
        "execution_base_urls",
    )
    configurations = list(configuration_roots)
    expected_tasks = tuple(
        dict.fromkeys(
            task
            for roots_by_task in configuration_roots.values()
            for task in roots_by_task
        )
    )
    if not expected_tasks:
        raise ValueError("Progressive configurations must provide at least one task")

    collection_keys = list(dict.fromkeys(
        (task, root)
        for roots_by_task in configuration_roots.values()
        for task, root_value in roots_by_task.items()
        for root in _configuration_root_replicates(root_value)
    ))

    # Large comparisons spend substantial CPU parsing and hashing JSON. Processes
    # avoid the GIL; small figures avoid process startup. Both preserve every gate.
    pool = (
        ProcessPoolExecutor(max_workers=min(16, len(collection_keys)), mp_context=get_context("spawn"))
        if len(collection_keys) > 8
        else ThreadPoolExecutor(max_workers=min(8, len(collection_keys)))
    )
    with pool:
        collected_resources = dict(zip(collection_keys, pool.map(_collect_configuration_resource, collection_keys)))

    all_rows: list[dict[str, Any]] = []
    contracts_by_configuration: dict[str, Any] = {}
    for configuration, roots_by_task in configuration_roots.items():
        task_contracts: dict[str, Any] = {}
        split_schemes: set[str] = set()
        for task, root_value in roots_by_task.items():
            replicate_roots = _configuration_root_replicates(root_value)
            replicate_rows: list[list[dict[str, Any]]] = []
            replicate_contracts: list[dict[str, Any]] = []
            for root in replicate_roots:
                resource_rows, resource_summary = collected_resources[(task, root)]
                replicate_rows.append(resource_rows)
                resource_contract = resource_summary["comparison_contract"]
                split_schemes.add(str(resource_contract["split_scheme"]))
                replicate_contracts.append(resource_contract["task_contracts"][task])

            reference_contract = replicate_contracts[0]
            for replicate_index, contract in enumerate(replicate_contracts[1:], start=2):
                mismatches = {
                    field: {
                        "reference": reference_contract[field],
                        "replicate": contract[field],
                    }
                    for field in replicate_invariant_fields
                    if contract[field] != reference_contract[field]
                }
                if mismatches:
                    raise ValueError(
                        f"Incompatible progressive replicate {replicate_index} for "
                        f"{configuration}/{task}: {mismatches}"
                    )
            aggregate_rows = _aggregate_progressive_replicate_rows(replicate_rows)
            all_rows.extend(
                {"configuration": configuration, "model_identity": reference_contract["model_identity"],
                 "organization": "full_flat" if reference_contract["experiment"] == "conditioned_assay_matched_full_flat.v1" else "progressive",
                 **row} for row in aggregate_rows
            )
            task_contracts[task] = {
                **reference_contract,
                "progressive_roots": [str(path) for path in replicate_roots],
                "n_replicates": len(replicate_roots),
            }
        if len(split_schemes) > 1:
            raise ValueError(
                f"Mixed split schemes within {configuration}: {sorted(split_schemes)}"
            )
        contracts_by_configuration[configuration] = {
            "split_scheme": next(iter(split_schemes)) if split_schemes else "",
            "task_contracts": task_contracts,
        }

    audits: dict[str, Any] = {}
    for task in expected_tasks:
        present_configurations = [
            configuration
            for configuration in configurations
            if task in contracts_by_configuration[configuration]["task_contracts"]
        ]
        if not present_configurations:
            continue
        reference_name = present_configurations[0]
        reference_contract = contracts_by_configuration[reference_name][
            "task_contracts"
        ][task]
        reference_levels = [
            (int(row["level"]), str(row["family"]))
            for row in all_rows
            if row["configuration"] == reference_name and row["task"] == task
        ]
        task_audit = {
            "reference_configuration": reference_name,
            "invariants": {},
            "index_sha256_equal": True,
            "family_manifest_sha256_equal": True,
            "execution_base_urls_equal": True,
            "present_configurations": present_configurations,
            "missing_configurations": [
                configuration
                for configuration in configurations
                if configuration not in present_configurations
            ],
        }
        model_references = {reference_contract["model_identity"]: reference_contract}
        for configuration in present_configurations[1:]:
            contract = contracts_by_configuration[configuration]["task_contracts"][task]
            same_model_reference = model_references.setdefault(contract["model_identity"], contract)
            _matched_organization_comparison(same_model_reference, contract)
            cross_model = allow_model_comparison and reference_contract["model_identity"] != contract["model_identity"]
            if cross_model:
                for field in ("prepared_evidence_sha256", "selection", "index_sha256", "family_manifest_sha256"):
                    if not contract.get(field) or contract[field] != reference_contract.get(field):
                        raise ValueError(f"Cross-model comparison differs in {field}: {task}")
                task_audit.setdefault("model_comparisons", []).append(configuration)
            organization_comparison = _matched_organization_comparison(reference_contract, contract, cross_model=cross_model)
            compared_fields = tuple(
                field for field in invariant_fields
                if not (organization_comparison and field in {"experiment", "prompt_profile"})
                and not (cross_model and field == "model_identity")
            )
            mismatches = {
                field: {
                    "reference": reference_contract[field],
                    "comparison": contract[field],
                }
                for field in compared_fields
                if contract[field] != reference_contract[field]
            }
            if mismatches:
                raise ValueError(
                    f"Incompatible progressive configurations for {task}: {mismatches}"
                )
            levels = [
                (int(row["level"]), str(row["family"]))
                for row in all_rows
                if row["configuration"] == configuration and row["task"] == task
            ]
            if levels != reference_levels:
                raise ValueError(
                    f"Progressive level mismatch for {task}: "
                    f"{configuration} has {levels}, expected {reference_levels}"
                )
            task_audit["invariants"][configuration] = "matched"
            if organization_comparison:
                task_audit["organization_comparison"] = {
                    "varying_fields": ["experiment", "prompt_profile"],
                    "prepared_inputs_sha256": contract["prepared_inputs_sha256"],
                    "selection_and_retrieval_equal": True,
                }
            task_audit["index_sha256_equal"] &= (
                contract["index_sha256"] == reference_contract["index_sha256"]
            )
            task_audit["family_manifest_sha256_equal"] &= (
                contract["family_manifest_sha256"]
                == reference_contract["family_manifest_sha256"]
            )
            task_audit["execution_base_urls_equal"] &= (
                contract["execution_base_urls"]
                == reference_contract["execution_base_urls"]
            )
        retrieval_lineage_equal = (
            task_audit["index_sha256_equal"]
            and task_audit["family_manifest_sha256_equal"]
        )
        if not retrieval_lineage_equal:
            receipt_path = (lineage_receipts_by_task or {}).get(task)
            if receipt_path is None:
                raise ValueError(
                    f"Retrieval lineage differs for {task}; provide a verified "
                    "TASK=PATH zero-change receipt"
                )
            receipt = _load_json(receipt_path)
            receipt_artifacts = receipt.get("artifacts") or {}
            compared_contracts = [
                contracts_by_configuration[configuration]["task_contracts"][task]
                for configuration in present_configurations
            ]
            compared_index_hashes = {
                contract["index_sha256"] for contract in compared_contracts
            }
            receipt_index_hashes = {
                receipt_artifacts.get("retrieval_index_sha256_before"),
                receipt_artifacts.get("retrieval_index_sha256_after"),
            }
            compared_family_hashes = {
                contract["family_manifest_sha256"] for contract in compared_contracts
            }
            receipt_family_hashes = {
                receipt_artifacts.get("family_manifest_sha256_before"),
                receipt_artifacts.get("family_manifest_sha256_after"),
            }
            if (
                receipt.get("task") != task
                or receipt.get("split_scheme") != reference_contract["split_scheme"]
                or receipt.get("evaluation_subset")
                != reference_contract["evaluation_subset"]
                or (receipt.get("results") or {}).get(
                    "selected_retrieval_surfaces_equal"
                )
                is not True
                or (
                    not task_audit["index_sha256_equal"]
                    and compared_index_hashes != receipt_index_hashes
                )
                or (
                    not task_audit["family_manifest_sha256_equal"]
                    and compared_family_hashes != receipt_family_hashes
                )
            ):
                raise ValueError(
                    f"Invalid selected-retrieval zero-change receipt for {task}: "
                    f"{receipt_path}"
                )
            task_audit["retrieval_lineage_receipt"] = str(receipt_path)
            task_audit["retrieval_lineage_receipt_sha256"] = sha256_file(
                receipt_path
            )
        else:
            task_audit["retrieval_lineage_receipt"] = "not_required"
        audits[task] = task_audit

    split_schemes = {
        str(contract["split_scheme"])
        for contract in contracts_by_configuration.values()
        if contract["split_scheme"]
    }
    if len(split_schemes) != 1:
        raise ValueError(f"Mixed progressive split schemes: {sorted(split_schemes)}")
    return all_rows, {
        "comparison_contract": {
            "figure": "conditioned_progressive_configuration_comparison.v2",
            "configurations": configurations,
            "tasks": list(expected_tasks),
            "split_scheme": next(iter(split_schemes)),
            "strict_invariants": list(invariant_fields),
            "organization_comparison_exception": (
                "Only frozen v8 versus matched full-flat profiles may differ in "
                "experiment/prompt_profile, with identical prepared inputs, selection and retrieval."
            ),
            "resource_metrics_included": True,
            "allow_model_comparison": allow_model_comparison,
            "model_comparison_exception": (
                "Explicit model comparisons retain identical prepared evidence/tools and selection; "
                "only model identity, endpoint and model-derived query_prior/reused_none_final/"
                "reused_single_source_index may differ. Replicate aggregation remains strict."
                if allow_model_comparison else None
            ),
            "replicate_statistic": "mean with observed min-max range",
            "task_audits": audits,
            "configurations_contract": contracts_by_configuration,
        },
        "rows": all_rows,
    }


def collect_conditioned_progressive_configuration_reference_data(
    *,
    configuration_roots: dict[str, dict[str, Path | list[Path]]],
    none_root: Path,
    baseline_root: Path,
    baseline_roots_by_task: dict[str, Path] | None = None,
    omit_mismatched_baselines: bool = False,
    allow_model_comparison: bool = False,
    omit_baseline_tasks: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect one shared None point and task-matched baseline references."""
    if allow_model_comparison:
        from tools.chembl_tool.paper_experiments.run_conditioned_assay_progressive_curve import _model_identity
        groups = {}
        for name, tasks in configuration_roots.items():
            for task, paths in tasks.items():
                manifest = _load_json(_configuration_root_replicates(paths)[0] / "experiment_manifest.json")
                model = _model_identity(manifest.get("model_identity", manifest["model"]))
                groups.setdefault(model, {}).setdefault(name, {})[task] = paths
        rows, audits, baselines = [], {}, {}
        for model, configs in groups.items():
            model_rows, audit = collect_conditioned_progressive_configuration_reference_data(
                configuration_roots=configs, none_root=none_root, baseline_root=baseline_root,
                baseline_roots_by_task=baseline_roots_by_task,
                omit_mismatched_baselines=omit_mismatched_baselines,
                omit_baseline_tasks=omit_baseline_tasks,
            )
            audits[model] = audit
            for row in model_rows:
                if row["result_type"] == "none":
                    rows.append({**row, "model_identity": model, "plot_label": "None · " + _model_plot_label(model)})
                else:
                    key = (row["task"], row["method"])
                    if key in baselines and baselines[key] != row:
                        raise ValueError(f"Baseline differs between models: {key}")
                    baselines[key] = row
        return rows + list(baselines.values()), {"none_shared_within_model": True, "model_audits": audits}
    if not configuration_roots:
        raise ValueError("At least one progressive configuration is required")
    configurations = list(configuration_roots)
    tasks = tuple(
        dict.fromkeys(
            task
            for roots_by_task in configuration_roots.values()
            for task in roots_by_task
        )
    )
    rows: list[dict[str, Any]] = []
    none_audits: dict[str, Any] = {}
    reference_configuration_by_task: dict[str, str] = {}
    baseline_omissions: list[dict[str, Any]] = []
    effective_baseline_roots: dict[str, str] = {}
    for task in tasks:
        reference_name = next(
            configuration
            for configuration in configurations
            if task in configuration_roots[configuration]
        )
        reference_configuration_by_task[task] = reference_name
        reference_root = _configuration_root_replicates(
            configuration_roots[reference_name][task]
        )[0]
        reference_manifest = _load_json(reference_root / "experiment_manifest.json")
        expected_n = len(reference_manifest["evaluation_indices_by_task"][task])
        task_label, baseline_task = CONDITIONED_TASK_SPECS[task]
        reference_none_path = reference_root / task / "none" / "metrics.json"
        if not reference_none_path.is_file():
            reference_none_path = none_root / task / "none" / "metrics.json"
        reference_none = _load_json(reference_none_path)
        issue = _complete_agent_metric_issue(reference_none, expected_n)
        if issue:
            raise ValueError(f"Incomplete no-retrieval metric for {task}: {issue}")
        none_values = {
            "macro_f1": _metric_value(reference_none, "macro_f1"),
            "accuracy": _metric_value(reference_none, "accuracy"),
        }
        configuration_paths: dict[str, list[str]] = {}
        for configuration, roots_by_task in configuration_roots.items():
            if task not in roots_by_task:
                continue
            configuration_paths[configuration] = []
            for task_root in _configuration_root_replicates(roots_by_task[task]):
                manifest = _load_json(task_root / "experiment_manifest.json")
                configuration_n = len(manifest["evaluation_indices_by_task"][task])
                if configuration_n != expected_n:
                    raise ValueError(
                        f"No-retrieval cohort mismatch for {configuration}/{task}: "
                        f"{configuration_n} != {expected_n}"
                    )
                none_path = task_root / task / "none" / "metrics.json"
                if not none_path.is_file():
                    none_path = none_root / task / "none" / "metrics.json"
                metrics = _load_json(none_path)
                issue = _complete_agent_metric_issue(metrics, expected_n)
                if issue:
                    raise ValueError(
                        f"Incomplete no-retrieval metric for {configuration}/{task}: "
                        f"{issue}"
                    )
                actual = {
                    "macro_f1": _metric_value(metrics, "macro_f1"),
                    "accuracy": _metric_value(metrics, "accuracy"),
                }
                if any(
                    abs(actual[key] - none_values[key]) > 1e-12
                    for key in none_values
                ):
                    raise ValueError(
                        f"No-retrieval metric mismatch across configurations for "
                        f"{task}: {reference_name}={none_values}, "
                        f"{configuration}={actual}"
                    )
                configuration_paths[configuration].append(str(none_path))
        rows.append(
            {
                "task": task,
                "task_label": task_label,
                "result_type": "none",
                "method": "none",
                "plot_label": "None",
                "macro_f1": none_values["macro_f1"],
                "accuracy": none_values["accuracy"],
                "n": expected_n,
                "metrics_path": str(reference_none_path),
            }
        )
        missing_configurations = [
            configuration
            for configuration in configurations
            if task not in configuration_roots[configuration]
        ]
        none_audits[task] = {
            "status": (
                "matched_across_configurations"
                if not missing_configurations
                and all(len(paths) == 1 for paths in configuration_paths.values())
                else "matched_across_available_configuration_replicates"
            ),
            "metrics_by_configuration": configuration_paths,
            "missing_configurations": missing_configurations,
        }
        task_baseline_root = (baseline_roots_by_task or {}).get(task, baseline_root)
        effective_baseline_roots[task] = str(task_baseline_root)
        if task in omit_baseline_tasks:
            baseline_omissions.append({"task": task, "reason": "explicit_task_baseline_omission"})
            continue
        baseline_omissions.extend(
            _append_conditioned_baselines(
                rows,
                task=task,
                task_label=task_label,
                baseline_task=baseline_task,
                baseline_root=task_baseline_root,
                expected_n=expected_n,
                evaluation_subset=reference_manifest.get("evaluation_subset"),
                omit_sample_count_mismatch=omit_mismatched_baselines,
            )
        )
    return rows, {
        "reference_configuration_by_task": reference_configuration_by_task,
        "none_lineage_fallback": str(none_root),
        "none_audits": none_audits,
        "baseline_lineage": str(baseline_root),
        "baseline_lineage_by_task": effective_baseline_roots,
        "baseline_methods": [method for method, _, _ in CONDITIONED_BASELINES],
        "baseline_omissions": baseline_omissions,
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
                evaluation_subset=manifest.get("evaluation_subset"),
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
    evaluation_subset: str = "valid",
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
    fig, axes = plt.subplots(5, len(tasks), figsize=(18.5 / 3 * len(tasks), 21.5), squeeze=False)
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
            marker, face, edge = CONDITIONED_BASELINE_STYLES[str(row["method"])]
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
        ] + [CONDITIONED_BASELINE_TICKS[str(row["method"])] for row in baseline_rows]
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
        f"{split_scheme.title()} {'validation' if evaluation_subset == 'valid' else evaluation_subset} "
        "· visible append-only DeepSeek-V4-Flash "
        "updates · task-specific source-purity contracts · latest complete "
        f"{len(tasks)}-task runs",
        fontsize=10,
        color="#555555",
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            marker="o",
            linewidth=2.2,
            label="Progressive agent / None",
        )
    ]
    for method, label, _ in CONDITIONED_BASELINES:
        marker, face, edge = CONDITIONED_BASELINE_STYLES[method]
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
        "None is the matched no-retrieval DeepSeek result; shown baseline points "
        "use the latest sample-count-matched condition-aware MiniMol head and k=3 "
        "MiniMol/Morgan KNN receipts. "
        "\n"
        "Molecules and cards/molecule are cumulative active evidence averaged over "
        "all queries. Prompt and reasoning lengths are means among actual model "
        "calls only; carry-forward and reused-none checkpoints are excluded.",
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
        f"{split_scheme.title()} validation · visible append-only updates · L1 "
        "direct evidence followed by task-specific indirect families",
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
        "Molecules and record cards are cumulative active evidence averaged over "
        "every query at that level. Reasoning length is averaged only over actual "
        "DeepSeek calls; carry-forward and reused-none checkpoints are excluded. "
        "Exact call counts, call fractions, prompt tokens, and amortized reasoning "
        "tokens are in the companion TSV.",
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


def plot_conditioned_progressive_configuration_comparison(
    *,
    rows: list[dict[str, Any]],
    output_svg: Path,
    output_png: Path,
    tasks: tuple[str, ...],
    configurations: tuple[str, ...],
    reference_rows: list[dict[str, Any]] | None = None,
    split_scheme: str = "scaffold",
    transport_matched: bool = True,
    retrieval_lineage_matched: bool = True,
    evaluation_subset: str = "valid",
    organization_comparison: bool = False,
    performance_only: bool = False,
    replicate_interval: str = "range",
    figure_note: str = "",
    macro_f1_limits: tuple[float, float] | None = None,
) -> None:
    """Plot multiple progressive configurations with full resource panels."""
    if len(configurations) < 2:
        raise ValueError("At least two configurations are required for comparison")
    styles = (("--", "white", 0.72), ("-", "color", 1.0), (":", "white", 1.0), ("-.", "color", 1.0))
    models = list(dict.fromkeys(str(row.get("model_identity", "")) for row in rows))
    cross_model = len(models) > 1
    model_colors = dict(zip(models, ("#2679B5", "#D47A1F"), strict=False))
    if cross_model and len(models) > 2:
        raise ValueError("Use a separate figure for more than two models")
    def series_style(index, configuration):
        if cross_model:
            row = next(row for row in rows if row["configuration"] == configuration)
            model = str(row["model_identity"])
            return styles[0 if row["organization"] == "full_flat" else 1], model_colors[model], ("o" if models.index(model) == 0 else "D")
        return styles[index], None, None
    if len(configurations) > len(styles):
        raise ValueError(f"At most {len(styles)} configurations can be plotted")
    resource_fields = (
        "mean_active_molecules",
        "mean_cards_per_active_molecule",
        "mean_prompt_tokens_per_call",
        "mean_reasoning_tokens_per_call",
    )
    missing_resource_fields = sorted(
        field
        for field in resource_fields
        if any(field not in row for row in rows)
    )
    if missing_resource_fields and not performance_only:
        raise ValueError(
            "Configuration comparison requires full resource metrics: "
            f"{missing_resource_fields}"
        )
    panel_specs = [
        ("macro_f1", "A  Macro-F1", 1.0),
        ("mean_active_molecules", "B  Active molecules/query", 1.0),
        ("mean_cards_per_active_molecule", "C  Record cards/molecule", 1.0),
        ("mean_prompt_tokens_per_call", "D  Prompt tokens/call (k)", 1000.0),
        ("mean_reasoning_tokens_per_call", "E  Reasoning tokens/call (k)", 1000.0),
    ]
    if performance_only:
        panel_specs = panel_specs[:1]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    n_rows = len(panel_specs)
    fig, axes = plt.subplots(
        n_rows,
        len(tasks),
        figsize=(max(8.5, 5.7 * len(tasks)), 6.5 if performance_only else 4.15 * n_rows + 1.1),
        squeeze=False,
        sharex=False,
    )
    reference_rows = reference_rows or []
    has_baselines = any(row.get("result_type") == "baseline" for row in reference_rows)
    baseline_order = {
        method: index for index, (method, _, _) in enumerate(CONDITIONED_BASELINES)
    }
    for column, task in enumerate(tasks):
        task_rows = [row for row in rows if row["task"] == task]
        if not task_rows:
            for ax in axes[:, column]:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "Current results unavailable", ha="center", va="center",
                        transform=ax.transAxes, color="#777777")
            axes[0, column].set_title(CONDITIONED_TASK_SPECS[task][0], fontweight="bold")
            continue
        n_values = {int(row["n_queries"]) for row in task_rows}
        if len(n_values) != 1:
            raise ValueError(f"Mixed evaluation sizes for {task}: {sorted(n_values)}")
        for config_index, configuration in enumerate(configurations):
            config_rows = sorted(
                (row for row in task_rows if row["configuration"] == configuration),
                key=lambda row: int(row["level"]),
            )
            if not config_rows:
                continue
            levels = [int(row["level"]) for row in config_rows]
            (linestyle, fill, alpha), series_color, series_marker = series_style(config_index, configuration)
            color = series_color or TASK_SPECS[task].color
            marker = series_marker or TASK_SPECS[task].marker
            for row_index, (field, _, divisor) in enumerate(panel_specs):
                values = [float(row[field]) / divisor for row in config_rows]
                axes[row_index, column].plot(
                    levels,
                    values,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markerfacecolor=(
                        color if fill == "color" else "white"
                    ),
                    markeredgecolor=color,
                    markeredgewidth=1.1,
                    markersize=6.2,
                    linewidth=2.2,
                    alpha=alpha,
                    zorder=3,
                )
                bounds = [_replicate_bounds(row, field, replicate_interval) for row in config_rows]
                lower = [lo / divisor for lo, _ in bounds]
                upper = [hi / divisor for _, hi in bounds]
                if any(
                    lo < value or hi > value
                    for lo, value, hi in zip(lower, values, upper, strict=True)
                ):
                    axes[row_index, column].errorbar(
                        levels,
                        values,
                        yerr=(
                            [value - lo for value, lo in zip(values, lower, strict=True)],
                            [hi - value for value, hi in zip(values, upper, strict=True)],
                        ),
                        fmt="none",
                        ecolor=color,
                        elinewidth=1.15,
                        capsize=3.2,
                        capthick=1.15,
                        alpha=0.78,
                        zorder=2,
                    )
                if row_index == 0:
                    offset = (7, -13, 20)[config_index % 3]
                    for x, y in (list(zip(levels, values)) if performance_only and not cross_model else [(levels[-1], values[-1])]):
                        if performance_only or len(configurations) == 2:
                            peers = [float(row[field]) / divisor for row in task_rows if int(row["level"]) == x]
                            if config_index and all(abs(value - y) < 1e-12 for value in peers):
                                continue
                            offset = 8 if y >= max(peers) else -14
                        label_y = y
                        if performance_only or len(configurations) == 2:
                            point_index = levels.index(x)
                            label_y = upper[point_index] if offset > 0 else lower[point_index]
                        if cross_model:
                            continue  # Four endpoint labels collide; exact values remain in the TSV.
                        axes[0, column].annotate(
                            f"{y:.3f}", (x, label_y), xytext=(0, offset),
                            textcoords="offset points", ha="center",
                            va="bottom" if offset > 0 else "top",
                            fontsize=7.1, color=TASK_SPECS[task].color,
                        )
        missing_configurations = [
            configuration
            for configuration in configurations
            if not any(
                row["configuration"] == configuration for row in task_rows
            )
        ]
        if missing_configurations:
            axes[0, column].text(
                0.02,
                0.04,
                "Unavailable on current lineage: " + ", ".join(missing_configurations),
                transform=axes[0, column].transAxes,
                ha="left",
                fontsize=7.0,
                color="#777777",
            )
        levels = sorted({int(row["level"]) for row in task_rows})
        task_reference_rows = [row for row in reference_rows if row["task"] == task]
        none_rows = [row for row in task_reference_rows if row["result_type"] == "none"]
        if len(none_rows) > 1 and not cross_model:
            raise ValueError(f"Multiple no-retrieval references for {task}")
        baseline_rows = sorted(
            (row for row in task_reference_rows if row["result_type"] == "baseline"),
            key=lambda row: baseline_order[str(row["method"])],
        )
        performance_ticks = list(levels)
        performance_labels = [f"L{level}" for level in levels]
        for none_index, none_row in enumerate(none_rows):
            none_value = float(none_row["macro_f1"])
            none_x = 0.0 if len(none_rows) == 1 else -0.12 + 0.24 * none_index
            none_color = model_colors.get(none_row.get("model_identity"), "#444444") if cross_model else "#444444"
            axes[0, column].scatter(
                [none_x], [none_value], marker="D" if cross_model and none_index else "o", s=46, facecolor=none_color,
                edgecolor="white", linewidth=0.7, zorder=5,
            )
            axes[0, column].annotate(
                f"{none_value:.3f}", (none_x, none_value),
                xytext=(0, 7 if len(none_rows) == 1 or none_value == max(float(row["macro_f1"]) for row in none_rows) else -14),
                textcoords="offset points", ha="center", fontsize=7.1,
                color=none_color,
            )
        if none_rows:
            performance_ticks.insert(0, 0)
            performance_labels.insert(0, "None")
        baseline_start = max(levels) + 1.8
        baseline_x = [baseline_start + 1.02 * index for index in range(len(baseline_rows))]
        for x_value, row in zip(baseline_x, baseline_rows, strict=True):
            method = str(row["method"])
            marker, face, edge = CONDITIONED_BASELINE_STYLES[method]
            value = float(row["macro_f1"])
            axes[0, column].scatter(
                [x_value], [value], marker=marker, s=47,
                facecolor=face, edgecolor=edge, linewidth=1.1, zorder=5,
            )
            axes[0, column].annotate(
                f"{value:.3f}", (x_value, value), xytext=(0, 7),
                textcoords="offset points", ha="center", fontsize=6.8,
                color="#555555",
            )
        performance_ticks.extend(baseline_x)
        performance_labels.extend(
            CONDITIONED_BASELINE_TICKS[str(row["method"])].replace("MM KNN", "MM\nKNN")
            for row in baseline_rows
        )
        axes[0, column].set_xticks(performance_ticks, performance_labels)
        axes[0, column].tick_params(axis="x", labelsize=6.4)
        right_edge = baseline_x[-1] + 0.5 if baseline_x else max(levels) + 0.45
        axes[0, column].set_xlim(-0.45 if none_rows else 0.55, right_edge)
        molecule_series = [] if performance_only else [
            [
                float(row["mean_active_molecules"])
                for row in sorted(
                    (
                        row
                        for row in task_rows
                        if row["configuration"] == configuration
                    ),
                    key=lambda row: int(row["level"]),
                )
            ]
            for configuration in configurations
            if any(row["configuration"] == configuration for row in task_rows)
        ]
        if not performance_only and len(molecule_series) > 1 and all(
            series == molecule_series[0] for series in molecule_series[1:]
        ):
            axes[1, column].text(
                0.98,
                0.05,
                "All configurations overlap",
                transform=axes[1, column].transAxes,
                ha="right",
                fontsize=7.1,
                color="#777777",
            )
        axes[0, column].set_title(
            f"{CONDITIONED_TASK_SPECS[task][0]}  (n={next(iter(n_values))})",
            fontsize=12.5,
            fontweight="bold",
        )
        for row_index in range(len(panel_specs)):
            axes[row_index, column].grid(axis="y", color="#E1E1E1", linewidth=0.8)
        for row_index in range(1, n_rows):
            axes[row_index, column].set_xlim(min(levels) - 0.2, max(levels) + 0.2)
            axes[row_index, column].set_xticks(
                levels, [f"L{level}" for level in levels]
            )
        axes[n_rows - 1, column].set_xlabel(
            ("Cumulative evidence level / baselines" if has_baselines else "Cumulative evidence level") if performance_only
            else "Cumulative assay-family level"
        )

    macro_values = [float(row["macro_f1"]) for row in rows + reference_rows]
    macro_values.extend(
        bound
        for row in rows
        for bound in _replicate_bounds(row, "macro_f1", replicate_interval)
    )
    macro_limits = (
        max(0.0, math.floor((min(macro_values) - 0.015) * 50) / 50),
        min(1.0, math.ceil((max(macro_values) + 0.015) * 50) / 50),
    )
    if macro_f1_limits is not None:
        lower, upper = macro_f1_limits
        if not (0 <= lower < upper <= 1) or min(macro_values) < lower or max(macro_values) > upper:
            raise ValueError("Macro-F1 limits must contain all plotted values and intervals within [0, 1]")
        macro_limits = (lower, upper)
    for column in range(len(tasks)):
        if not axes[0, column].axison:
            continue
        axes[0, column].set_ylim(*macro_limits)
        axes[0, column].text(
            0.99, 1.02, "Shared focused y-axis", transform=axes[0, column].transAxes,
            ha="right", fontsize=7.0, color="#777777"
        )
    axes[0, 0].set_ylabel(panel_specs[0][1])

    for row_index, (field, ylabel, divisor) in enumerate(panel_specs[1:], start=1):
        maximum = max(
            _replicate_bounds(row, field, replicate_interval)[1] / divisor for row in rows
        )
        for column in range(len(tasks)):
            axes[row_index, column].set_ylim(0, maximum * 1.16 if maximum else 1.0)
        axes[row_index, 0].set_ylabel(ylabel)

    fig.suptitle(
        " vs ".join(_model_plot_label(model) for model in models) + " · full-flat and progressive" if cross_model else
        "Full-flat vs progressive · matched evidence" if organization_comparison
        else "Progressive configuration comparison",
        fontsize=16,
        fontweight="bold",
        x=0.055,
        y=0.995 if performance_only else 0.98,
        ha="left",
    )
    repeat_counts = sorted({int(row.get("n_replicates", 1)) for row in rows})
    interval_note = (
        "mean ± 1 sample SD (ddof=1); "
        + (f"{repeat_counts[0]} runs per point" if len(repeat_counts) == 1 else "run counts vary by point")
        if replicate_interval in {"sd", "sd_if_repeated"} else
        "means across reruns · whiskers show observed min–max (absent for a single run)"
    )
    if all(row.get("n_replicates") == 1 for row in rows):
        interval_note = "single run per setting · no estimated uncertainty interval"
    if cross_model:
        model_intervals = []
        for model in models:
            counts = sorted({int(row.get("n_replicates", 1)) for row in rows
                             if row.get("model_identity") == model})
            if counts == [1]:
                description = "1 run, no SD"
            elif len(counts) == 1:
                description = f"{counts[0]} runs, mean ± SD"
            else:
                description = f"{'/'.join(map(str, counts))} runs by task; SD only for repeated runs"
            model_intervals.append(_model_plot_label(model) + ": " + description)
        interval_note = " · ".join(model_intervals) + " · default 4/2 card budget"
    fig.text(
        0.055,
        0.935 if performance_only else 0.965,
        f"{split_scheme.title()} {'test' if evaluation_subset == 'test' else 'validation'} · "
        + interval_note,
        fontsize=9.5,
        color="#555555",
    )
    handles = []
    for index, configuration in enumerate(configurations):
        (linestyle, fill, alpha), color, marker = series_style(index, configuration)
        color = color or "#333333"
        handles.append(
            Line2D(
                [0], [0], color=color, linestyle=linestyle, marker=marker or "o",
                markerfacecolor=color if fill == "color" else "white",
                markeredgecolor=color, linewidth=2.2, alpha=alpha,
                label=configuration,
            )
        )
    if reference_rows:
        if not cross_model:
            handles.append(
                Line2D(
                    [0], [0], marker="o", linestyle="none", markerfacecolor="#444444",
                    markeredgecolor="white", label="None",
                )
            )
        for method, label, _ in CONDITIONED_BASELINES:
            if not any(row.get("method") == method for row in reference_rows):
                continue
            marker, face, edge = CONDITIONED_BASELINE_STYLES[method]
            handles.append(
                Line2D(
                    [0], [0], marker=marker, linestyle="none",
                    markerfacecolor=face, markeredgecolor=edge, label=label,
                )
            )
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015 if performance_only else 0.025),
        ncol=5 if cross_model else 4, frameon=False, fontsize=8.0,
    )
    transport_note = (
        "Provider transport is matched across configurations; token panels are "
        "directly comparable."
        if transport_matched
        else "Provider transport differs for at least one historical comparison; "
        "token panels show observed usage and are not transport-controlled. Model "
        "identity and evaluation inputs are matched."
    )
    resource_note = (
        "Molecules and cards/molecule are means over all queries; prompt and "
        "reasoning tokens are means over actual model calls. " + interval_note + ". "
    )
    lineage_note = (
        "Retrieval index and family-manifest hashes are matched across configurations."
        if retrieval_lineage_matched
        else "A changed retrieval lineage is admitted only by a registered "
        "selected-surface zero-change receipt."
    )
    reference_note = (
        "None is shared across configurations; baseline markers are task-specific "
        "MiniMol/Morgan references and do not extend into resource panels. "
        "Condition-first KNN fallback follows each task's registered policy.\n"
        if reference_rows else ""
    )
    if cross_model:
        reference_note = "None is model-specific; matched baselines are shared.\n"
        transport_note = "Models and providers differ; token panels show observed usage, not a controlled transport comparison."
    footnote = fig.text(
        0.055,
        0.175 if performance_only else 0.072,
        (f"All curves use the same {evaluation_subset} cohort, cumulative evidence and tools. None and query priors are model-specific.\n"
         "Dashed/open: full-flat; solid/filled: progressive. Right-side baselines are unchanged; KNN references are train-only.\n"
         "Whiskers show ±1 sample SD across repeated runs (not a confidence interval). Single-run curves have no estimated uncertainty interval."
         if cross_model and performance_only else
         "Each level uses identical cumulative evidence and tools; full-flat reasons independently, "
         "progressive carries prior state.\nNone is shared. "
         + (f"Right-side baselines use the same {evaluation_subset} cohort; KNN references are train-only.\n"
            if has_baselines else "Matched baselines have not been included.\n")
         + ("SD describes run-to-run variation with frozen inputs, not a confidence interval. None and baselines are single fixed references."
            if replicate_interval == "sd" else
            ("Single-run curves have no estimated uncertainty interval." if all(row.get("n_replicates") == 1 for row in rows)
             else "Whiskers, when present, are rerun ranges, not confidence intervals."))
         if performance_only and organization_comparison else
         "Same evaluation cohort and model; each curve uses its declared evidence configuration.\n"
         + (f"None and matched baselines use the same {evaluation_subset} cohort; KNN references are train-only.\n"
            if has_baselines else "Matched baselines have not been included.\n")
         + interval_note + "."
         if performance_only else
         reference_note + resource_note + "\n" + lineage_note + " " + transport_note),
        fontsize=8.0, color="#555555", linespacing=1.35,
        va="top" if performance_only else "baseline",
    )
    if figure_note:
        footnote.set_text(footnote.get_text() + "\n" + figure_note)
    if len(tasks) == 1:
        footnote.set_text("\n".join(
            textwrap.fill(line, width=125) for line in footnote.get_text().splitlines()
        ))
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.82 if performance_only else 0.94,
        bottom=0.34 if performance_only else 0.145,
        hspace=0.42,
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


def plot_baseline_curves_csv(points: Path, metadata: Path, analysis_dir: Path, *,
                            output_stem: str, macro_f1_limits=None) -> dict[str, str]:
    """Portable verified-score input; no private prediction directories required."""
    with points.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    seen = set()
    for row in rows:
        key = (row['task'], row['method'])
        if key in seen:
            raise ValueError(f'duplicate task/method: {key}')
        seen.add(key)
        row['n'] = int(row['n'])
        for metric in ('macro_f1', 'accuracy'):
            row[metric] = float(row[metric])
            if not math.isfinite(row[metric]) or not 0 <= row[metric] <= 1:
                raise ValueError(f'invalid {metric}: {key}')
    receipt = json.loads(metadata.read_text())
    receipt['point_source'] = {'path': str(points), 'sha256': sha256_file(points)}
    return plot_baseline_curve_points(rows, receipt, analysis_dir,
        output_stem=output_stem, macro_f1_limits=macro_f1_limits)



def plot_baseline_curve_points(rows, receipt, analysis_dir, *, output_stem, macro_f1_limits=None):
    """Shared renderer for full trace audits and portable CSV handoffs."""
    tasks = list(receipt["conditions"])
    series = list((
        ("Neighbor-fill", ("none", "direct_fill", "both_fill"), "#0072B2", "o", "-"),
        ("Molecule-cap", ("none", "direct_cap", "both_cap"), "#D55E00", "s", "-"),
        ("Group-balanced", ("none", "direct_cap", "both_group"), "#009E73", "^", "--"),
    ))
    if receipt.get('curve_settings'):
        colors = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9')
        series = [(name, methods, colors[i % len(colors)], ('o', 's', '^', 'D')[i % 4], '-' if i != 2 else '--')
                  for i, (name, methods) in enumerate(receipt['curve_settings'].items())]
        if any(len(methods) != 3 for _, methods, *_ in series):
            raise ValueError('each curve must contain None, Direct and Indirect method IDs')
    scores = {(row["task"], row["method"]): row["macro_f1"] for row in rows}
    required = {method for _, methods, *_ in series for method in methods} | {m for m, _, _ in CONDITIONED_BASELINES}
    for task in tasks:
        if required - {r['method'] for r in rows if r['task'] == task}:
            raise ValueError(f'missing curve or ML baseline points for {task}')
        if any(r['n'] != receipt['conditions'][task]['n'] for r in rows if r['task'] == task):
            raise ValueError(f'mixed cohort sizes for {task}')
    values = list(scores.values())
    limits = macro_f1_limits or (
        max(0, math.floor((min(values) - 0.025) * 20) / 20),
        min(1, math.ceil((max(values) + 0.025) * 20) / 20),
    )
    if not (0 <= limits[0] <= min(values) <= max(values) <= limits[1] <= 1):
        raise ValueError("Macro-F1 limits must contain every plotted value within [0, 1]")
    columns = min(3, len(tasks))
    grid_rows = math.ceil(len(tasks) / columns)
    fig, axes = plt.subplots(
        grid_rows, columns, figsize=(6.2 * columns, 3.6 * grid_rows + 1.6),
        sharey=True, squeeze=False,
    )
    baseline_x = [3.15 + 0.62 * i for i in range(len(CONDITIONED_BASELINES))]
    for ax, task in zip(axes.flat, tasks):
        ax.axvspan(2.65, baseline_x[-1] + 0.4, color="#F4F4F4", zorder=0)
        ax.axvline(2.65, color="#BBBBBB", linewidth=0.8)
        for label, methods, color, marker, linestyle in series:
            ax.plot(
                [0, 1, 2], [scores[task, method] for method in methods],
                color=color, marker=marker, linestyle=linestyle, linewidth=2,
                markersize=6, markerfacecolor="white" if linestyle == "--" else color,
                label=label, zorder=3,
            )
        none_methods = {methods[0] for _, methods, *_ in series}
        if len(none_methods) == 1:
            ax.plot(0, scores[task, next(iter(none_methods))], "o", color="#444444", markersize=6, zorder=4)
        for x, (method, _, _) in zip(baseline_x, CONDITIONED_BASELINES):
            marker, face, edge = CONDITIONED_BASELINE_STYLES[method]
            ax.plot(x, scores[task, method], marker=marker, linestyle="none",
                    markerfacecolor=face, markeredgecolor=edge, markersize=7, zorder=3)
        ax.set_title(
            f"{CONDITIONED_TASK_SPECS[task][0]}  ·  n = {receipt['conditions'][task]['n']}",
            loc="left", fontsize=12, fontweight="bold", pad=22,
        )
        ax.text(1, 1.025, "LLM evidence", transform=ax.get_xaxis_transform(),
                ha="center", fontsize=9, color="#666666")
        ax.text(sum(baseline_x) / len(baseline_x), 1.025, "ML baselines",
                transform=ax.get_xaxis_transform(), ha="center", fontsize=9, color="#666666")
        ax.set_xticks([0, 1, 2] + baseline_x)
        ax.set_xticklabels(
            ["None", "Direct", "Direct +\nIndirect"]
            + [CONDITIONED_BASELINE_TICKS[method] for method, _, _ in CONDITIONED_BASELINES],
            fontsize=8,
        )
        ax.set_xlim(-0.3, baseline_x[-1] + 0.4)
        ax.set_ylim(*limits)
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(tasks):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro-F1", fontsize=11)
    handles = [Line2D([0], [0], color=color, marker=marker, linestyle=style,
                      label=label, markerfacecolor="white" if style == "--" else color)
               for label, _, color, marker, style in series]
    for method, label, _ in CONDITIONED_BASELINES:
        marker, face, edge = CONDITIONED_BASELINE_STYLES[method]
        handles.append(Line2D([0], [0], marker=marker, linestyle="none",
                              markerfacecolor=face, markeredgecolor=edge, label=label))
    fig.suptitle("None → Direct → Direct + Indirect", x=0.055, y=0.985,
                 ha="left", fontsize=19, fontweight="bold")
    subset_label = 'validation' if receipt['evaluation_subset'] == 'valid' else receipt['evaluation_subset']
    visibility_label = ('identity visible' if receipt['visibility_mode'] == 'deployment_visible_prefetched'
                        else 'identity blind')
    similarity_label = ('no similarity cutoff' if receipt['min_similarity'] == 0
                        else f"similarity ≥ {receipt['min_similarity']:g}")
    fig.text(0.055, 0.943,
             f"{receipt['split_scheme'].title()} {subset_label} · {similarity_label} · {visibility_label} · "
             f"direct budget {receipt['direct_budget']} + indirect budget {receipt['indirect_budget']} records · "
             "shared Macro-F1 scale", fontsize=10, color="#555555")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.018),
               ncol=4, frameon=False, fontsize=9)
    fig.text(0.055, 0.12,
             "Each point is an independent full-flat setting. Curves use the None/Direct references specified in their metadata.\n"
             f"Right-side ML baselines use the same {subset_label} rows. One run per setting; no uncertainty intervals. "
             f"Recorded tool failures: {receipt['tool_failures']}.",
             fontsize=9, color="#555555", linespacing=1.5)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.86, bottom=0.23,
                        hspace=0.55, wspace=0.14)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for extension in ("png", "pdf", "svg"):
        path = analysis_dir / f"{output_stem}.{extension}"
        fig.savefig(path, dpi=200, facecolor="white")
        paths[extension] = str(path)
    plt.close(fig)
    csv_path = analysis_dir / f"{output_stem}.csv"
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['task', 'method', 'family', 'n', 'macro_f1', 'accuracy'])
        writer.writeheader()
        writer.writerows(rows)
    paths['csv'] = str(csv_path)
    receipt.update({"curve_settings": {label: methods for label, methods, *_ in series},
                    "macro_f1_limits": limits, "scores": rows, "outputs": paths})
    receipt_path = analysis_dir / f"{output_stem}_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return {**paths, "receipt": str(receipt_path)}




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-curves-csv', type=Path, help='Portable task/method score CSV.')
    parser.add_argument('--baseline-curves-metadata', type=Path, help='Cohort/context metadata and optional curve_settings.')
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
    parser.add_argument("--figure-note", default="", help="Additional provenance note on configuration figures.")
    parser.add_argument("--macro-f1-limits", nargs=2, type=float, metavar=("LOWER", "UPPER"),
                        help="Optional common Macro-F1 axis limits for configuration figures; must contain all points and intervals.")
    parser.add_argument(
        "--omit-baseline-tasks", nargs="+", choices=tuple(CONDITIONED_TASK_SPECS), default=[],
        help="Omit unavailable or stale task baselines from configuration figures; retain None and record omissions.",
    )
    parser.add_argument(
        "--plot-tasks", nargs="+", choices=tuple(CONDITIONED_TASK_SPECS),
        help="Ordered task columns for configuration comparisons; unavailable tasks stay explicitly empty.",
    )
    parser.add_argument(
        "--performance-only", action="store_true",
        help="For configuration comparisons, show only level Macro-F1 and baseline references.",
    )
    parser.add_argument(
        "--replicate-interval", choices=("range", "sd", "sd_if_repeated"), default="range",
        help="Configuration-curve whiskers: observed min-max (default) or ±1 sample SD across runs.",
    )
    parser.add_argument("--allow-model-comparison", action="store_true",
                        help="Compare models only with identical prepared evidence/tools; keep model-specific None references.")
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
        "--conditioned-progressive-config-comparison",
        action="store_true",
        help=(
            "Plot multiple progressive configurations together, with Macro-F1 "
            "and active record-card volume by task and level."
        ),
    )
    parser.add_argument(
        "--conditioned-progressive-overview",
        action="store_true",
        help=(
            "Plot one column per supplied task containing progressive performance, "
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
        default="",
        help="Explicit progressive artifact root for the single-lineage resource view.",
    )
    parser.add_argument(
        "--conditioned-progressive-bbb-root",
        default="",
        help="Deprecated compatibility argument; prefer --conditioned-progressive-task-root.",
    )
    parser.add_argument(
        "--conditioned-progressive-source-purity-root",
        default="",
        help="Deprecated compatibility argument; prefer --conditioned-progressive-task-root.",
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
        "--conditioned-progressive-config-task-root",
        action="append",
        default=[],
        metavar="CONFIG:TASK=PATH",
        help=(
            "Register a task artifact root under a named progressive "
            "configuration. Repeat the same CONFIG:TASK with distinct paths "
            "to average exact-contract reruns; unavailable task/configuration "
            "cells may be omitted."
        ),
    )
    parser.add_argument(
        "--conditioned-progressive-config-lineage-receipt",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help=(
            "Required for each task whose compared retrieval index or family "
            "manifest hash differs; the receipt must prove identical selected "
            "model-visible retrieval surfaces."
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
    if args.baseline_curves_csv:
        if not args.baseline_curves_metadata:
            raise ValueError('--baseline-curves-csv requires --baseline-curves-metadata')
        paths = plot_baseline_curves_csv(args.baseline_curves_csv, args.baseline_curves_metadata, analysis_dir,
            output_stem=args.output_stem, macro_f1_limits=tuple(args.macro_f1_limits) if args.macro_f1_limits else None)
        print(json.dumps(paths, indent=2))
        return 0
    if args.conditioned_progressive_config_comparison:
        configuration_roots = _parse_configuration_task_path_overrides(
            args.conditioned_progressive_config_task_root
        )
        lineage_receipts_by_task = _parse_task_path_overrides(
            args.conditioned_progressive_config_lineage_receipt
        )
        rows, summary = collect_conditioned_progressive_configuration_data(
            configuration_roots=configuration_roots,
            lineage_receipts_by_task=lineage_receipts_by_task,
            allow_model_comparison=args.allow_model_comparison,
        )
        baseline_roots_by_task = _parse_task_path_overrides(
            args.conditioned_progressive_baseline_root
        )
        reference_rows, reference_summary = (
            collect_conditioned_progressive_configuration_reference_data(
                configuration_roots=configuration_roots,
                none_root=Path(args.conditioned_none_agent_root),
                baseline_root=Path(args.conditioned_baseline_root),
                baseline_roots_by_task=baseline_roots_by_task,
                omit_mismatched_baselines=(
                    args.omit_mismatched_progressive_baselines
                ),
                allow_model_comparison=args.allow_model_comparison,
                omit_baseline_tasks=tuple(args.omit_baseline_tasks),
            )
        )
        summary["comparison_contract"]["performance_references"] = reference_summary
        summary["comparison_contract"]["replicate_interval"] = args.replicate_interval
        plot_tasks = tuple(args.plot_tasks or summary["comparison_contract"]["tasks"])
        if len(set(plot_tasks)) != len(plot_tasks) or not set(summary["comparison_contract"]["tasks"]) <= set(plot_tasks):
            raise ValueError("--plot-tasks must contain each supplied task exactly once")
        summary["plot_tasks"] = list(plot_tasks)
        if args.replicate_interval in {"sd", "sd_if_repeated"}:
            for row in rows:
                _replicate_bounds(row, "macro_f1", args.replicate_interval)
            summary["comparison_contract"]["replicate_statistic"] = "arithmetic mean ±1 sample SD (ddof=1) for repeated runs; single-run curves have no interval"
        summary["reference_rows"] = reference_rows
        if not args.analysis_dir:
            analysis_dir = Path(
                "outputs/paper/analysis/progressive_configuration_comparison"
            )
        analysis_dir.mkdir(parents=True, exist_ok=True)
        _write_tsv(analysis_dir / "progressive_configuration_metrics.tsv", rows)
        _write_tsv(
            analysis_dir / "progressive_configuration_references.tsv",
            reference_rows,
        )
        (analysis_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        figure_dir = analysis_dir / "figures"
        output_stem = (
            args.output_stem
            if args.output_stem != "assay_retrieval_scaling"
            else "progressive_configuration_comparison"
        )
        comparison_contract = summary["comparison_contract"]
        task_audits = comparison_contract["task_audits"]
        contracts = comparison_contract["configurations_contract"]
        evaluation_subsets = {
            task_contract["evaluation_subset"]
            for configuration in contracts.values()
            for task_contract in configuration["task_contracts"].values()
        }
        if len(evaluation_subsets) != 1:
            raise ValueError("Mixed evaluation subsets in configuration figure")
        plot_conditioned_progressive_configuration_comparison(
            rows=rows,
            output_svg=figure_dir / f"{output_stem}.svg",
            output_png=figure_dir / f"{output_stem}.png",
            tasks=plot_tasks,
            configurations=tuple(comparison_contract["configurations"]),
            reference_rows=reference_rows,
            split_scheme=str(comparison_contract["split_scheme"]),
            evaluation_subset=next(iter(evaluation_subsets)),
            organization_comparison=any("organization_comparison" in audit for audit in task_audits.values()),
            performance_only=args.performance_only,
            replicate_interval=args.replicate_interval,
            figure_note=args.figure_note,
            macro_f1_limits=tuple(args.macro_f1_limits) if args.macro_f1_limits else None,
            transport_matched=all(
                bool(audit["execution_base_urls_equal"])
                for audit in task_audits.values()
            ),
            retrieval_lineage_matched=all(
                bool(audit["index_sha256_equal"])
                and bool(audit["family_manifest_sha256_equal"])
                for audit in task_audits.values()
            ),
        )
        print(
            json.dumps(
                {
                    "analysis_dir": str(analysis_dir),
                    "n_metric_rows": len(rows),
                    "configurations": comparison_contract["configurations"],
                    "tasks": comparison_contract["tasks"],
                    "figure": str(figure_dir / f"{output_stem}.png"),
                },
                indent=2,
            )
        )
        return 0
    if args.conditioned_progressive_overview:
        baseline_roots_by_task = _parse_task_path_overrides(
            args.conditioned_progressive_baseline_root
        )
        progressive_roots = _parse_task_path_overrides(
            args.conditioned_progressive_task_root
        )
        if not progressive_roots or any(task not in TASK_SPECS for task in progressive_roots):
            raise ValueError(
                "--conditioned-progressive-overview requires current "
                "--conditioned-progressive-task-root entries for supported plot tasks"
            )
        if not args.analysis_dir:
            analysis_dir = (
                next(iter(progressive_roots.values())).parent
                / "analysis"
                / "progressive_overview"
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
            else "progressive_overview"
        )
        plot_conditioned_progressive_overview(
            rows=rows,
            output_svg=figure_dir / f"{output_stem}.svg",
            output_png=figure_dir / f"{output_stem}.png",
            tasks=tuple(progressive_roots),
            split_scheme=str(
                summary["comparison_contract"]["performance"]["split_scheme"]
            ),
            evaluation_subset=str(
                summary["comparison_contract"]["resources"]["evaluation_subset"]
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
        if not args.conditioned_progressive_root:
            raise ValueError(
                "--conditioned-progressive-performance-resources requires an "
                "explicit --conditioned-progressive-root"
            )
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
        tasks = [task for task, spec in TASK_SPECS.items() if spec.historical_best_condition]
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
