"""Paired-statistics validation and SVG fragments for the Starling figure.

This module keeps optional paired significance presentation out of the main
model-comparison renderer.  The public figure entry remains
``plot_starling_model_comparison.py``; this file only owns the paired TSV
contract and its two compact panel bands.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from .paper_figure_style import CARD, GRID, INK, MUTED, PURPLE, rect, svg_text
from .plot_starling_benchmark_overview import BASELINE_SOURCES, TASKS


PAIRED_CI_BAND = "#F3F0F9"
PAIRED_CI_EXTRA_HEIGHT = 230
PAIRED_PVALUE_EXTRA_HEIGHT = 165
PAIRED_POINT_FILL = PURPLE
LEGACY_PAIRED_BASELINES = frozenset({"minimol_train_all", "morgan_knn_k3"})
ALL_PAIRED_BASELINES = frozenset(
    {
        *LEGACY_PAIRED_BASELINES,
        "minimol_embedding_cosine_knn_k3",
    }
)


@dataclass(frozen=True)
class PairedComparison:
    baseline_method: str
    baseline_label: str
    agent_model: str
    agent_method: str
    agent_label: str
    n: int
    delta: float
    ci_low: float
    ci_high: float
    bootstrap_replicates: int
    source: Path
    alternative: str = ""
    p_value_one_sided: float | None = None
    permutation_replicates: int | None = None
    permutation_seed: int | None = None
    analysis_status: str = ""


def read_paired_metrics(
    path: Path | None,
    *,
    series_by_label: dict[str, dict[tuple[str, str, str], dict[str, str]]],
    available_splits: tuple[tuple[str, str], ...],
    evaluation_subset: str,
    experiments: dict[tuple[str, str], tuple[Any, ...]],
) -> dict[tuple[str, str], tuple[PairedComparison, ...]]:
    if path is None:
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "benchmark_split",
        "evaluation_subset",
        "task",
        "agent_model",
        "agent_method",
        "baseline",
        "baseline_method",
        "n",
        "agent_macro_f1",
        "baseline_macro_f1",
        "delta_macro_f1_agent_minus_baseline",
        "paired_ci95_low",
        "paired_ci95_high",
        "bootstrap_replicates",
    }
    missing_columns = required - set(rows[0] if rows else ())
    if missing_columns:
        raise ValueError(
            f"Paired-CI metrics {path} are missing columns: {sorted(missing_columns)}"
        )

    supported_splits = {split for split, _ in available_splits}
    supported_tasks = {task.key for task in TASKS}
    grouped: dict[tuple[str, str], list[PairedComparison]] = {}
    seen: set[tuple[str, str, str]] = set()
    allowed_baselines = ALL_PAIRED_BASELINES
    task_by_key = {task.key: task for task in TASKS}

    for row in rows:
        split, task_key = row["benchmark_split"], row["task"]
        if split not in supported_splits or task_key not in supported_tasks:
            raise ValueError(
                "Paired-CI row is outside the plotted task/split matrix: "
                f"{split}/{task_key}"
            )
        if row["evaluation_subset"] != evaluation_subset:
            raise ValueError(
                f"Paired-CI evaluation subset mismatch for {split}/{task_key}: "
                f"{row['evaluation_subset']} != {evaluation_subset}"
            )
        baseline_method = row["baseline_method"]
        unique_key = (split, task_key, baseline_method)
        if unique_key in seen:
            raise ValueError(f"Duplicate paired-CI comparison: {unique_key}")
        seen.add(unique_key)
        if baseline_method not in allowed_baselines:
            raise ValueError(
                f"Unsupported paired-CI baseline for {split}/{task_key}: "
                f"{baseline_method}"
            )

        agent_model = row["agent_model"]
        agent_method = row["agent_method"]
        agent_key = (split, task_key, agent_method)
        agent_results = series_by_label.get(agent_model, {})
        agent_row = agent_results.get(agent_key)
        experiment_agent = next(
            (
                item
                for item in experiments.get((split, task_key), ())
                if item.model_label == agent_model and item.method == agent_method
            ),
            None,
        )
        if agent_row is None and experiment_agent is None:
            raise ValueError(
                "Paired-CI agent method is not plotted in a model series or "
                f"experiment row: {agent_key} in {agent_model!r}"
            )
        if agent_row is not None and agent_row.get("method_family") != "molecular_evidence_agent":
            raise ValueError(f"Paired-CI selected agent is not an agent row: {agent_key}")

        baseline_key = (split, task_key, baseline_method)
        baseline_row = next(
            (
                results[baseline_key]
                for results in series_by_label.values()
                if baseline_key in results
            ),
            None,
        )
        if baseline_row is None:
            raise ValueError(f"Paired-CI baseline is not plotted: {baseline_key}")

        n = int(row["n"])
        agent_n = int(agent_row["n_test"]) if agent_row is not None else experiment_agent.n
        if n != agent_n or n != int(baseline_row["n_test"]):
            raise ValueError(f"Paired-CI sample-count mismatch for {unique_key}")
        agent_f1 = float(row["agent_macro_f1"])
        baseline_f1 = float(row["baseline_macro_f1"])
        plotted_agent_f1 = (
            float(agent_row["macro_f1"])
            if agent_row is not None
            else experiment_agent.macro_f1
        )
        if abs(agent_f1 - plotted_agent_f1) > 5e-6:
            raise ValueError(f"Paired-CI agent macro-F1 mismatch for {unique_key}")
        if abs(baseline_f1 - float(baseline_row["macro_f1"])) > 5e-6:
            raise ValueError(f"Paired-CI baseline macro-F1 mismatch for {unique_key}")

        loaded_agent_values = [
            float(results[(split, task_key, method.key)]["macro_f1"])
            for results in series_by_label.values()
            for method in task_by_key[task_key].methods
            if method.source not in BASELINE_SOURCES
            and (split, task_key, method.key) in results
            and results[(split, task_key, method.key)].get("method_family")
            == "molecular_evidence_agent"
        ]
        loaded_agent_values.extend(
            item.macro_f1 for item in experiments.get((split, task_key), ())
        )
        if agent_f1 + 5e-6 < max(loaded_agent_values):
            raise ValueError(
                "Paired-CI selected agent is not the plotted best agent for "
                f"{split}/{task_key}"
            )

        comparison = _comparison_from_row(
            row,
            path=path,
            baseline_method=baseline_method,
            agent_model=agent_model,
            agent_method=agent_method,
            agent_label=(
                next(
                    method.label
                    for method in task_by_key[task_key].methods
                    if method.key == agent_method
                )
                if agent_row is not None
                else experiment_agent.label
            ),
            n=n,
            agent_f1=agent_f1,
            baseline_f1=baseline_f1,
            unique_key=unique_key,
        )
        grouped.setdefault((split, task_key), []).append(comparison)

    expected_groups = {
        (split, task.key) for split in supported_splits for task in TASKS
    }
    if set(grouped) != expected_groups:
        raise ValueError(
            "Paired-CI metrics must cover every plotted task/split group: "
            f"missing={sorted(expected_groups - set(grouped))}, "
            f"extra={sorted(set(grouped) - expected_groups)}"
        )
    order = {
        "minimol_train_all": 0,
        "morgan_knn_k3": 1,
        "minimol_embedding_cosine_knn_k3": 2,
    }
    output: dict[tuple[str, str], tuple[PairedComparison, ...]] = {}
    comparison_sets = {
        frozenset(item.baseline_method for item in comparisons)
        for comparisons in grouped.values()
    }
    if len(comparison_sets) != 1:
        raise ValueError("Paired-CI baseline sets must match across every task/split")
    expected_baselines = next(iter(comparison_sets))
    if expected_baselines not in {LEGACY_PAIRED_BASELINES, ALL_PAIRED_BASELINES}:
        raise ValueError(
            "Paired-CI metrics require either the historical MiniMol/Morgan pair "
            "or all three current baselines; found "
            f"{sorted(expected_baselines)}"
        )
    for group, comparisons in grouped.items():
        methods = {item.baseline_method for item in comparisons}
        if methods != expected_baselines:
            raise ValueError(
                "Paired-CI metrics require the same complete baseline set for "
                f"{group}; found {sorted(methods)}"
            )
        if len({(item.agent_model, item.agent_method) for item in comparisons}) != 1:
            raise ValueError(f"Paired-CI rows disagree on the best agent for {group}")
        output[group] = tuple(
            sorted(comparisons, key=lambda item: order[item.baseline_method])
        )
    return output


def _comparison_from_row(
    row: dict[str, str],
    *,
    path: Path,
    baseline_method: str,
    agent_model: str,
    agent_method: str,
    agent_label: str,
    n: int,
    agent_f1: float,
    baseline_f1: float,
    unique_key: tuple[str, str, str],
) -> PairedComparison:
    delta = float(row["delta_macro_f1_agent_minus_baseline"])
    ci_low = float(row["paired_ci95_low"])
    ci_high = float(row["paired_ci95_high"])
    if abs(delta - (agent_f1 - baseline_f1)) > 5e-6:
        raise ValueError(f"Paired-CI delta mismatch for {unique_key}")
    if not ci_low <= delta <= ci_high:
        raise ValueError(f"Paired-CI interval does not contain its estimate: {unique_key}")
    bootstrap_replicates = int(row["bootstrap_replicates"])
    if bootstrap_replicates <= 0:
        raise ValueError(f"Paired-CI bootstrap count must be positive: {unique_key}")
    p_value = float(row["p_value_one_sided"]) if row.get("p_value_one_sided") else None
    if p_value is not None and not 0 <= p_value <= 1:
        raise ValueError(f"Paired p-value is outside [0, 1]: {unique_key}")
    permutations = (
        int(row["permutation_replicates"])
        if row.get("permutation_replicates")
        else None
    )
    if permutations is not None and permutations <= 0:
        raise ValueError(f"Paired permutation count must be positive: {unique_key}")
    seed = int(row["permutation_seed"]) if row.get("permutation_seed") else None
    return PairedComparison(
        baseline_method=baseline_method,
        baseline_label=row["baseline"],
        agent_model=agent_model,
        agent_method=agent_method,
        agent_label=agent_label,
        n=n,
        delta=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        bootstrap_replicates=bootstrap_replicates,
        source=path,
        alternative=row.get("alternative", ""),
        p_value_one_sided=p_value,
        permutation_replicates=permutations,
        permutation_seed=seed,
        analysis_status=row.get("analysis_status", ""),
    )


def paired_ci_scale_limit(
    comparisons: dict[tuple[str, str], tuple[PairedComparison, ...]],
) -> float:
    if not comparisons:
        return 0.0
    maximum = max(
        abs(value)
        for rows in comparisons.values()
        for item in rows
        for value in (item.ci_low, item.ci_high)
    )
    return max(0.1, math.ceil(maximum / 0.05) * 0.05)


def render_paired_ci_band(
    parts: list[str],
    *,
    x: float,
    y: float,
    width: float,
    comparisons: tuple[PairedComparison, ...],
    scale_limit: float,
) -> None:
    height = PAIRED_CI_EXTRA_HEIGHT - 20
    parts.append(rect(x, y, width, height, fill=PAIRED_CI_BAND, stroke=GRID, rx=5))
    first = comparisons[0]
    parts.append(svg_text(x + 16, y + 25, "Paired Δ macro-F1: best agent − baseline (95% CI)", size=14, weight=750))
    parts.append(svg_text(x + width - 16, y + 25, f"{first.bootstrap_replicates:,} paired resamples", size=11, weight=650, fill=MUTED, anchor="end"))
    parts.append(svg_text(x + 16, y + 46, f"Best: {_short_model_label(first.agent_model)} · {first.agent_label}", size=11, weight=650, fill=MUTED))

    label_width = min(245.0, width * 0.28)
    value_width = min(185.0, width * 0.22)
    plot_left = x + label_width
    plot_right = x + width - value_width
    zero_x = (plot_left + plot_right) / 2
    axis_bottom = y + 57 + 37 * len(comparisons)
    parts.append(f'<line x1="{zero_x:.1f}" y1="{y + 57:.1f}" x2="{zero_x:.1f}" y2="{axis_bottom:.1f}" stroke="{INK}" stroke-width="1.2" stroke-dasharray="4 4"/>')

    def scale(value: float) -> float:
        clipped = min(scale_limit, max(-scale_limit, value))
        return plot_left + (clipped + scale_limit) / (2 * scale_limit) * (plot_right - plot_left)

    for index, item in enumerate(comparisons):
        row_y = y + 79 + 37 * index
        low_x, high_x, point_x = scale(item.ci_low), scale(item.ci_high), scale(item.delta)
        parts.append(svg_text(x + 16, row_y + 4, item.baseline_label, size=11, weight=650))
        parts.append(f'<line x1="{low_x:.1f}" y1="{row_y:.1f}" x2="{high_x:.1f}" y2="{row_y:.1f}" stroke="{PURPLE}" stroke-width="3"/>')
        for cap_x in (low_x, high_x):
            parts.append(f'<line x1="{cap_x:.1f}" y1="{row_y - 6:.1f}" x2="{cap_x:.1f}" y2="{row_y + 6:.1f}" stroke="{PURPLE}" stroke-width="2"/>')
        parts.append(f'<circle cx="{point_x:.1f}" cy="{row_y:.1f}" r="5" fill="{PAIRED_POINT_FILL}" stroke="{CARD}" stroke-width="1.5"/>')
        interval = f"{item.delta:+.3f} [{item.ci_low:+.3f}, {item.ci_high:+.3f}]"
        parts.append(svg_text(x + width - 16, row_y + 4, interval, size=10, weight=700, anchor="end"))

    for tick in (-scale_limit, -scale_limit / 2, 0.0, scale_limit / 2, scale_limit):
        tick_x = scale(tick)
        parts.append(f'<line x1="{tick_x:.1f}" y1="{axis_bottom + 2:.1f}" x2="{tick_x:.1f}" y2="{axis_bottom + 6:.1f}" stroke="{MUTED}" stroke-width="1"/>')
        parts.append(svg_text(tick_x, axis_bottom + 21, "0" if tick == 0 else f"{tick:+.2f}", size=9, fill=MUTED, anchor="middle"))


def render_paired_pvalue_band(
    parts: list[str],
    *,
    x: float,
    y: float,
    width: float,
    comparisons: tuple[PairedComparison, ...],
) -> None:
    first = comparisons[0]
    if any(
        item.p_value_one_sided is None
        or item.permutation_replicates is None
        or item.alternative != "agent_greater_than_baseline"
        for item in comparisons
    ):
        raise ValueError(
            "One-sided p-value display requires p_value_one_sided, positive "
            "permutation_replicates, and alternative=agent_greater_than_baseline"
        )
    parts.append(rect(x, y, width, PAIRED_PVALUE_EXTRA_HEIGHT - 20, fill=PAIRED_CI_BAND, stroke=GRID, rx=5))
    parts.append(svg_text(x + 16, y + 25, "Exploratory one-sided paired permutation p-value", size=14, weight=750))
    parts.append(svg_text(x + width - 16, y + 25, f"H1: best agent > baseline · {first.permutation_replicates:,} permutations", size=11, weight=650, fill=MUTED, anchor="end"))
    parts.append(svg_text(x + 16, y + 47, f"Best: {_short_model_label(first.agent_model)} · {first.agent_label}", size=11, weight=650, fill=MUTED))
    for index, item in enumerate(comparisons):
        row_y = y + 76 + 25 * index
        parts.append(svg_text(x + 16, row_y, item.baseline_label, size=11, weight=650))
        parts.append(
            svg_text(
                x + width - 16,
                row_y,
                _format_p_value(item.p_value_one_sided),
                size=12,
                weight=750,
                fill=PURPLE,
                anchor="end",
            )
        )


def _format_p_value(value: float) -> str:
    return "p < 0.001" if value < 0.001 else f"p = {value:.3f}"


def _short_model_label(label: str) -> str:
    return label.replace("GLM-5.2 NVFP4", "GLM-5.2").replace("GPT-OSS-", "")
