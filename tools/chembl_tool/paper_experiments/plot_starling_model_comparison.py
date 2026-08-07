"""Render the canonical Starling model-and-experiment comparison figure.

The required reference/candidate summaries provide the frozen GPT-OSS model
comparison. Optional complete model/visibility summaries add grouped bars.
Optional matched experiment TSVs add rows to the same figure; their anchor
rows are validated against an existing loaded model/visibility condition and
are not plotted twice.
An optional paired-bootstrap TSV adds a signed dot-and-whisker interval block
to each task panel for the selected best agent versus train-label baselines.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .paper_figure_style import (
    BG,
    CARD,
    GRID,
    INK,
    MUTED,
    PURPLE,
    append_vertical_grid,
    export_png,
    rect,
    svg_text,
)
from .plot_starling_benchmark_overview import (
    BASELINE_FILLS,
    BASELINE_SOURCES,
    SCALE_MAX,
    SOURCE_COLORS,
    SPLITS,
    TASKS,
    read_metrics,
)
from .starling_paired_figure import (
    PAIRED_CI_EXTRA_HEIGHT,
    PAIRED_POINT_FILL,
    PAIRED_PVALUE_EXTRA_HEIGHT,
    PairedComparison,
    paired_ci_scale_limit,
    read_paired_metrics,
    render_paired_ci_band,
    render_paired_pvalue_band,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE = (
    ROOT
    / "outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b/metrics.tsv"
)
DEFAULT_CANDIDATE = (
    ROOT
    / "outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_120b/metrics.tsv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b_vs_120b"
    / "figures/starling_model_comparison.svg"
)

WIDTH = 1900
HEADER_HEIGHT = 235
PANEL_GAP = 30
FOOTER_HEIGHT = 95
EXPERIMENT_FILL = "#477A74"
EXPERIMENT_BAND = "#EDF5F3"


@dataclass(frozen=True)
class ExperimentMetric:
    method: str
    label: str
    macro_f1: float
    model_label: str
    source: Path


def _lighten(color: str, fraction: float = 0.72) -> str:
    rgb = [int(color[index : index + 2], 16) for index in (1, 3, 5)]
    mixed = [round(value + (255 - value) * fraction) for value in rgb]
    return "#" + "".join(f"{value:02X}" for value in mixed)


def _series_fill(color: str, index: int, count: int) -> str:
    """Encode model/visibility series by shade while retaining source color."""
    if count <= 1:
        return color
    if count == 2:
        fraction = (0.72, 0.0)[index]
    else:
        fraction = 0.82 * (count - index - 1) / (count - 1)
    return _lighten(color, fraction) if fraction else color


def _agent_model_label(results: dict[tuple[str, str, str], dict[str, str]]) -> str:
    for row in results.values():
        if row.get("method_family") == "molecular_evidence_agent":
            return row.get("model_label") or "Agent"
    return next(iter(results.values())).get("model_label") or "Agent"


def _filter_method_family(
    results: dict[tuple[str, str, str], dict[str, str]],
    method_family: str | None,
) -> dict[tuple[str, str, str], dict[str, str]]:
    if method_family is None:
        return results
    filtered = {
        key: row
        for key, row in results.items()
        if row.get("method_family") == method_family
    }
    if not filtered:
        raise ValueError(f"No rows matched method_family={method_family!r}")
    return filtered


def _validate_pair(
    reference: dict[tuple[str, str, str], dict[str, str]],
    candidate: dict[tuple[str, str, str], dict[str, str]],
    *,
    require_shared_baselines: bool = True,
) -> tuple[tuple[str, str], ...]:
    reference_splits = {key[0] for key in reference}
    candidate_splits = {key[0] for key in candidate}
    if reference_splits != candidate_splits:
        raise ValueError(
            "Model-comparison metrics must contain the same benchmark splits: "
            f"reference={sorted(reference_splits)}, candidate={sorted(candidate_splits)}"
        )
    if reference.keys() != candidate.keys():
        raise ValueError("Model-comparison metrics must contain identical task/method keys")
    reference_subsets = {
        row.get("evaluation_subset") or "test" for row in reference.values()
    }
    candidate_subsets = {
        row.get("evaluation_subset") or "test" for row in candidate.values()
    }
    if reference_subsets != candidate_subsets or len(reference_subsets) != 1:
        raise ValueError(
            "Model-comparison metrics must contain the same single evaluation subset: "
            f"reference={sorted(reference_subsets)}, candidate={sorted(candidate_subsets)}"
        )
    for key, reference_row in reference.items():
        candidate_row = candidate[key]
        if int(reference_row["n_test"]) != int(candidate_row["n_test"]):
            raise ValueError(f"Sample-count mismatch between model summaries: {key}")
        method = key[2]
        if method not in {item.key for item in sum((task.methods for task in TASKS), ())}:
            continue
        source = next(
            item.source
            for task in TASKS
            for item in task.methods
            if item.key == method
        )
        if source not in BASELINE_SOURCES or not require_shared_baselines:
            continue
        for field in ("macro_f1", "accuracy", "auroc"):
            reference_value = reference_row.get(field, "")
            candidate_value = candidate_row.get(field, "")
            if reference_value in (None, "") and candidate_value in (None, ""):
                continue
            if reference_value in (None, "") or candidate_value in (None, ""):
                raise ValueError(f"Baseline {field} mismatch between model summaries: {key}")
            if abs(float(reference_value) - float(candidate_value)) > 1e-12:
                raise ValueError(f"Baseline {field} mismatch between model summaries: {key}")
    return tuple((split, label) for split, label in SPLITS if split in reference_splits)


def _read_experiment_metrics(
    paths: tuple[Path, ...],
    *,
    series_by_label: dict[str, dict[tuple[str, str, str], dict[str, str]]],
    available_splits: tuple[tuple[str, str], ...],
    candidate_label: str,
) -> dict[tuple[str, str], tuple[ExperimentMetric, ...]]:
    """Read matched experiment rows and validate their unplotted anchors.

    Each task/split group in an experiment TSV must contain one anchor row.
    ``comparison_role=anchor`` is preferred; the historical
    ``method=morgan_standard`` spelling remains supported.  ``base_method``
    defaults to ``starling_full_mechanism`` and ``base_model_label`` defaults
    to the primary candidate series.
    """
    if not paths:
        return {}
    supported_splits = {split for split, _ in available_splits}
    collected: dict[tuple[str, str], list[ExperimentMetric]] = {}
    seen: set[tuple[str, str, str]] = set()

    for path in paths:
        grouped = _group_experiment_rows(path, supported_splits)
        for group, group_rows in grouped.items():
            anchor, base_row, base_model_label, base_subset = (
                _validate_experiment_anchor(
                    path,
                    group,
                    group_rows,
                    series_by_label=series_by_label,
                    candidate_label=candidate_label,
                )
            )
            _append_experiment_candidates(
                path,
                group,
                group_rows,
                anchor=anchor,
                base_row=base_row,
                base_model_label=base_model_label,
                base_subset=base_subset,
                seen=seen,
                collected=collected,
            )
    return {key: tuple(values) for key, values in collected.items()}


def _group_experiment_rows(
    path: Path,
    supported_splits: set[str],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"benchmark_split", "task", "method", "macro_f1"}
    missing_columns = required - set(rows[0] if rows else ())
    if missing_columns:
        raise ValueError(
            f"Experiment metrics {path} are missing columns: "
            f"{sorted(missing_columns)}"
        )
    supported_tasks = {task.key for task in TASKS}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        split, task = row["benchmark_split"], row["task"]
        if split in supported_splits and task in supported_tasks:
            grouped.setdefault((split, task), []).append(row)
    expected_groups = {
        (split, task.key) for split in supported_splits for task in TASKS
    }
    missing_groups = sorted(expected_groups - set(grouped))
    if missing_groups:
        raise ValueError(
            f"Experiment metrics {path} are missing task/split groups: "
            f"{missing_groups}"
        )
    return grouped


def _validate_experiment_anchor(
    path: Path,
    group: tuple[str, str],
    group_rows: list[dict[str, str]],
    *,
    series_by_label: dict[str, dict[tuple[str, str, str], dict[str, str]]],
    candidate_label: str,
) -> tuple[dict[str, str], dict[str, str], str, str]:
    split, task = group
    anchors = [
        row
        for row in group_rows
        if row.get("comparison_role", "").lower() in {"anchor", "control"}
        or row["method"] == "morgan_standard"
    ]
    if len(anchors) != 1:
        raise ValueError(
            f"Experiment metrics {path} require exactly one anchor row for "
            f"{split}/{task}; found {len(anchors)}"
        )
    anchor = anchors[0]
    base_method = anchor.get("base_method") or "starling_full_mechanism"
    base_model_label = anchor.get("base_model_label") or candidate_label
    if base_model_label not in series_by_label:
        raise ValueError(
            f"Experiment anchor {split}/{task} references missing model/visibility "
            f"series {base_model_label!r}"
        )
    base_results = series_by_label[base_model_label]
    base_key = (split, task, base_method)
    if base_key not in base_results:
        raise ValueError(
            f"Experiment anchor {split}/{task} references missing method "
            f"{base_method!r} in {base_model_label!r}"
        )
    base_row = base_results[base_key]
    n_value = anchor.get("n_test") or anchor.get("n")
    if n_value in (None, "") or int(n_value) != int(base_row["n_test"]):
        raise ValueError(f"Experiment anchor sample-count mismatch for {split}/{task}")
    if abs(float(anchor["macro_f1"]) - float(base_row["macro_f1"])) > 5e-7:
        raise ValueError(
            f"Experiment anchor macro-F1 mismatch for {split}/{task}: "
            f"experiment={anchor['macro_f1']}, candidate={base_row['macro_f1']}"
        )
    base_subset = base_row.get("evaluation_subset") or "test"
    evaluation_subset = anchor.get("evaluation_subset")
    if evaluation_subset and evaluation_subset != base_subset:
        raise ValueError(
            f"Experiment evaluation subset mismatch for {split}/{task}: "
            f"experiment={evaluation_subset}, candidate={base_subset}"
        )
    return anchor, base_row, base_model_label, base_subset


def _append_experiment_candidates(
    path: Path,
    group: tuple[str, str],
    group_rows: list[dict[str, str]],
    *,
    anchor: dict[str, str],
    base_row: dict[str, str],
    base_model_label: str,
    base_subset: str,
    seen: set[tuple[str, str, str]],
    collected: dict[tuple[str, str], list[ExperimentMetric]],
) -> None:
    split, task = group
    for row in group_rows:
        if row is anchor:
            continue
        method = row["method"]
        unique_key = (split, task, method)
        if unique_key in seen:
            raise ValueError(f"Duplicate experiment method row: {unique_key}")
        seen.add(unique_key)
        row_n = row.get("n_test") or row.get("n")
        if row_n in (None, "") or int(row_n) != int(base_row["n_test"]):
            raise ValueError(
                f"Experiment sample-count mismatch for {split}/{task}/{method}"
            )
        row_subset = row.get("evaluation_subset")
        if row_subset and row_subset != base_subset:
            raise ValueError(
                f"Experiment evaluation subset mismatch for {split}/{task}/{method}"
            )
        collected.setdefault(group, []).append(
            ExperimentMetric(
                method=method,
                label=(row.get("plot_label") or row.get("method_label") or method)
                .replace("Coverage retrieve + ", "Coverage · ")
                .replace("coverage-aware context", "coverage-aware"),
                macro_f1=float(row["macro_f1"]),
                model_label=row.get("model_label") or base_model_label,
                source=path,
            )
        )


def _render_panel(
    parts: list[str],
    *,
    x: float,
    y: float,
    task,
    methods,
    split: str,
    split_label: str,
    width: float,
    height: float,
    series: tuple[dict[tuple[str, str, str], dict[str, str]], ...],
    experiments: tuple[ExperimentMetric, ...],
    show_experiment_model_label: bool,
    paired_comparisons: tuple[PairedComparison, ...] = (),
    paired_scale_limit: float = 0.0,
    paired_significance_display: str = "ci",
    baseline_display: str = "shared",
    baseline_series_groups: tuple[str, ...] = (),
) -> None:
    parts.append(rect(x, y, width, height, fill=CARD, stroke=GRID))
    first = series[0][(split, task.key, methods[0].key)]
    parts.append(svg_text(x + 22, y + 34, task.title, size=22, weight=750))
    parts.append(
        svg_text(
            x + width - 22,
            y + 33,
            f"{split_label} · n = {int(first['n_test'])}",
            size=14,
            weight=650,
            fill=MUTED,
            anchor="end",
        )
    )

    if width >= 900:
        plot_left, plot_right = x + 285, x + 805
    else:
        plot_left, plot_right = x + 270, x + 760
    paired_extra = 0
    if paired_comparisons:
        paired_extra = (
            PAIRED_PVALUE_EXTRA_HEIGHT
            if paired_significance_display == "pvalue"
            else PAIRED_CI_EXTRA_HEIGHT
        )
    plot_top, plot_bottom = y + 66, y + height - 65 - paired_extra
    append_vertical_grid(
        parts,
        plot_left=plot_left,
        plot_right=plot_right,
        plot_top=plot_top,
        plot_bottom=plot_bottom,
        ticks=(0.0, 0.2, 0.4, 0.6, 0.8),
        scale_max=SCALE_MAX,
        label_y=y + height - 33 - paired_extra,
    )

    count = len(methods) + len(experiments)
    available = plot_bottom - plot_top - 30
    max_step = 42.0 if len(series) <= 3 else 10.0 * len(series) + 4.0
    step = min(max_step, available / max(1, count - 1))
    row_start = plot_top + 15 + (available - step * (count - 1)) / 2
    if experiments:
        experiment_start = row_start + len(methods) * step
        band_height = max(28.0, (len(experiments) - 1) * step + 30.0)
        parts.append(
            rect(
                x + 12,
                experiment_start - 15,
                width - 24,
                band_height,
                fill=EXPERIMENT_BAND,
                rx=4,
            )
        )
    for index, method in enumerate(methods):
        row_y = row_start + index * step
        series_rows = [result[(split, task.key, method.key)] for result in series]
        color = SOURCE_COLORS[method.source]
        is_baseline = method.source in BASELINE_SOURCES
        is_shared_baseline = is_baseline and baseline_display == "shared"
        if is_shared_baseline:
            band_height = min(34.0, max(24.0, step - 2.0))
            parts.append(
                rect(
                    x + 12,
                    row_y - band_height / 2,
                    width - 24,
                    band_height,
                    fill=BASELINE_FILLS[method.source],
                    rx=4,
                )
            )
        parts.append(
            svg_text(
                x + 22,
                row_y + 5,
                method.label,
                size=13,
                weight=700 if is_baseline else 500,
            )
        )
        if is_shared_baseline:
            value = float(series_rows[0]["macro_f1"])
            bar_width = value / SCALE_MAX * (plot_right - plot_left)
            parts.append(rect(plot_left, row_y - 9, bar_width, 18, fill=color, rx=3))
            parts.append(
                svg_text(
                    plot_left + bar_width + 8,
                    row_y + 5,
                    f"{value:.3f}",
                    size=12,
                    weight=700,
                )
            )
            continue

        display_rows = list(enumerate(series_rows))
        if is_baseline and baseline_display == "series":
            first_index_by_group: dict[str, int] = {}
            for series_index, group in enumerate(baseline_series_groups):
                first_index = first_index_by_group.setdefault(group, series_index)
                if abs(
                    float(series_rows[first_index]["macro_f1"])
                    - float(series_rows[series_index]["macro_f1"])
                ) > 1e-12:
                    raise ValueError(
                        "Baseline series grouped under the same lineage must match: "
                        f"{split}/{task.key}/{method.key}/{group}"
                    )
            display_rows = [
                (series_index, series_rows[series_index])
                for series_index in first_index_by_group.values()
            ]
        bar_gap = 2.0 if len(display_rows) <= 3 else 3.0
        minimum_bar_height = 4.0 if len(display_rows) <= 3 else 5.0
        bar_height = min(
            8.0,
            max(
                minimum_bar_height,
                (step - 8.0) / len(display_rows) - bar_gap,
            ),
        )
        group_height = (
            len(display_rows) * bar_height + (len(display_rows) - 1) * bar_gap
        )
        first_bar_y = row_y - group_height / 2
        for display_index, (series_index, series_row) in enumerate(display_rows):
            value = float(series_row["macro_f1"])
            bar_y = first_bar_y + display_index * (bar_height + bar_gap)
            fill = _series_fill(color, series_index, len(series))
            bar_width = value / SCALE_MAX * (plot_right - plot_left)
            parts.append(
                rect(
                    plot_left,
                    bar_y,
                    bar_width,
                    bar_height,
                    fill=fill,
                    stroke=color,
                    rx=2,
                )
            )
            parts.append(
                svg_text(
                    plot_left + bar_width + 8,
                    bar_y + bar_height,
                    f"{value:.3f}",
                    size=8 if len(series) > 4 else 9 if len(series) > 2 else 10,
                    weight=700,
                    fill=INK,
                )
            )

    for experiment_index, experiment in enumerate(experiments):
        row_y = row_start + (len(methods) + experiment_index) * step
        experiment_label = experiment.label
        if show_experiment_model_label:
            experiment_label = (
                experiment.model_label.replace("GPT-OSS-", "")
                + " · "
                + experiment_label
            )
        parts.append(
            svg_text(x + 22, row_y + 5, experiment_label, size=12, weight=650)
        )
        bar_width = experiment.macro_f1 / SCALE_MAX * (plot_right - plot_left)
        parts.append(
            rect(
                plot_left,
                row_y - 7,
                bar_width,
                14,
                fill=EXPERIMENT_FILL,
                stroke="#315B57",
                rx=3,
            )
        )
        parts.append(
            svg_text(
                plot_left + bar_width + 8,
                row_y + 5,
                f"{experiment.macro_f1:.3f}",
                size=11,
                weight=700,
                fill=INK,
            )
        )
    parts.append(
        svg_text(
            (plot_left + plot_right) / 2,
            y + height - 11 - paired_extra,
            "Macro-F1",
            size=12,
            weight=650,
            fill=MUTED,
            anchor="middle",
        )
    )
    if paired_comparisons:
        if paired_significance_display == "pvalue":
            render_paired_pvalue_band(
                parts,
                x=x + 12,
                y=y + height - paired_extra + 10,
                width=width - 24,
                comparisons=paired_comparisons,
            )
        else:
            render_paired_ci_band(
                parts,
                x=x + 12,
                y=y + height - paired_extra + 10,
                width=width - 24,
                comparisons=paired_comparisons,
                scale_limit=paired_scale_limit,
            )


def render(
    reference_path: Path,
    candidate_path: Path,
    output: Path,
    experiment_paths: tuple[Path, ...] = (),
    comparison_paths: tuple[Path, ...] = (),
    paired_ci_path: Path | None = None,
    paired_significance_display: str = "ci",
    method_family: str | None = None,
    series_label_overrides: tuple[str, ...] = (),
    chart_title: str = "Starling Benchmark Model & Experiment Comparison",
    chart_subtitle: str | None = None,
    context_lines: tuple[str, ...] = (),
    comparison_title_override: str | None = None,
    baseline_display: str = "shared",
    baseline_series_groups: tuple[str, ...] = (),
) -> None:
    if paired_significance_display not in {"ci", "pvalue"}:
        raise ValueError(
            "paired_significance_display must be either 'ci' or 'pvalue'"
        )
    if baseline_display not in {"shared", "series"}:
        raise ValueError("baseline_display must be either 'shared' or 'series'")
    reference = _filter_method_family(read_metrics(reference_path), method_family)
    candidate = _filter_method_family(read_metrics(candidate_path), method_family)
    require_shared_baselines = baseline_display == "shared"
    available_splits = _validate_pair(
        reference,
        candidate,
        require_shared_baselines=require_shared_baselines,
    )
    additional = tuple(
        _filter_method_family(read_metrics(path), method_family)
        for path in comparison_paths
    )
    for result in additional:
        if _validate_pair(
            reference,
            result,
            require_shared_baselines=require_shared_baselines,
        ) != available_splits:
            raise ValueError("Additional comparison metrics changed available splits")
    series = (reference, candidate, *additional)
    series_labels = (
        series_label_overrides
        if series_label_overrides
        else tuple(_agent_model_label(result) for result in series)
    )
    if len(series_labels) != len(series):
        raise ValueError(
            "series_label_overrides must contain one label for each metrics series"
        )
    if series_label_overrides and len(set(series_labels)) != len(series_labels):
        raise ValueError("Comparison series labels must be unique")
    effective_baseline_groups = baseline_series_groups or series_labels
    if len(effective_baseline_groups) != len(series):
        raise ValueError(
            "baseline_series_groups must contain one group for each metrics series"
        )
    series_by_label = dict(zip(series_labels, series, strict=True))
    reference_label = series_labels[0]
    candidate_label = (
        series_labels[1] if series_label_overrides else _agent_model_label(candidate)
    )
    # Preserve the historical default-anchor behavior even when a synthetic
    # test or legacy summary reuses the same display label for both series.
    series_by_label[candidate_label] = candidate
    experiments = _read_experiment_metrics(
        experiment_paths,
        series_by_label=series_by_label,
        available_splits=available_splits,
        candidate_label=candidate_label,
    )
    experiment_model_labels = {
        item.model_label for values in experiments.values() for item in values
    }
    show_experiment_model_label = len(experiment_model_labels) > 1
    first = next(iter(reference.values()))
    evaluation_subset = first.get("evaluation_subset") or "test"
    paired_comparisons = read_paired_metrics(
        paired_ci_path,
        series_by_label=series_by_label,
        available_splits=available_splits,
        evaluation_subset=evaluation_subset,
        experiments=experiments,
    )
    paired_scale_limit = paired_ci_scale_limit(paired_comparisons)
    split_title = " and ".join(label.replace(" split", "") for _, label in available_splits)
    methods_by_task = {
        task.key: tuple(
            method
            for method in task.methods
            if all(
                (split, task.key, method.key) in reference
                for split, _ in available_splits
            )
        )
        for task in TASKS
    }
    missing_tasks = [task for task, methods in methods_by_task.items() if not methods]
    if missing_tasks:
        raise ValueError(f"No plotted methods remain for tasks: {missing_tasks}")
    has_failure = any(
        int(row.get("n_failed") or 0) > 0
        for result in series
        for row in result.values()
    )
    generated = date.today().isoformat()
    max_rows = max(
        len(methods_by_task[task.key])
        + len(experiments.get((split, task.key), ()))
        for split, _ in available_splits
        for task in TASKS
    )
    dense_series_extra = max(0, len(series) - 3) * 110
    panel_height = max(
        510,
        510
        + (max_rows - max(len(methods_by_task[task.key]) for task in TASKS)) * 38
        + dense_series_extra,
    )
    if paired_comparisons:
        panel_height += (
            PAIRED_PVALUE_EXTRA_HEIGHT
            if paired_significance_display == "pvalue"
            else PAIRED_CI_EXTRA_HEIGHT
        )
    effective_context_lines = context_lines or (
        "Same scaffold-valid data and parent-disjoint retrieval · Visibility/tool-execution contract varies by series",
    )
    context_extra = max(0, len(effective_context_lines) - 1) * 22
    header_height = HEADER_HEIGHT + context_extra
    height = header_height + len(TASKS) * panel_height + (len(TASKS) - 1) * PANEL_GAP + FOOTER_HEIGHT
    source_paths = (
        reference_path,
        candidate_path,
        *comparison_paths,
        *experiment_paths,
        *((paired_ci_path,) if paired_ci_path is not None else ()),
    )
    comparison_title = comparison_title_override or (
        f"{reference_label} vs {candidate_label}"
        if len(series_labels) == 2
        else f"{len(series_labels)} model/visibility settings"
    )
    subtitle = chart_subtitle or f"{split_title} {evaluation_subset} · Macro-F1"
    baseline_description = (
        "lineage-specific train-label baseline series"
        if baseline_display == "series"
        else "shared train-label baselines"
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="chart-title chart-desc">',
        '<title id="chart-title">Starling benchmark model and experiment comparison</title>',
        f'<desc id="chart-desc">Grouped horizontal bars compare {", ".join(series_labels)} '
        f'macro-F1 for Starling scaffold valid tasks, with {baseline_description} and matched experiment additions'
        + (
            ", plus exploratory one-sided paired permutation p-values.</desc>"
            if paired_comparisons and paired_significance_display == "pvalue"
            else ", plus paired bootstrap intervals.</desc>"
            if paired_comparisons
            else ".</desc>"
        ),
        f'<metadata>Sources: {"; ".join(str(path) for path in source_paths)}; generated {generated}.</metadata>',
        rect(0, 0, WIDTH, height, fill=BG, rx=0),
        svg_text(70, 58, chart_title, size=36, weight=750),
        svg_text(70, 94, subtitle, size=20, fill=MUTED),
        svg_text(1830, 58, comparison_title, size=18, weight=700, fill=PURPLE, anchor="end"),
    ]
    for index, line in enumerate(effective_context_lines):
        parts.append(svg_text(70, 132 + index * 22, line, size=14, fill=MUTED))
    primary_legend_y = 157 + context_extra
    auxiliary_legend_y = 183 + context_extra
    legend_x = 70.0
    for series_index, label in enumerate(series_labels):
        parts.append(
            rect(
                legend_x,
                primary_legend_y,
                30,
                10,
                fill=_series_fill(PURPLE, series_index, len(series_labels)),
                stroke=PURPLE,
                rx=2,
            )
        )
        parts.append(
            svg_text(legend_x + 42, primary_legend_y + 11, label, size=13, weight=600)
        )
        legend_x += max(225.0, 82.0 + len(label) * 7.2)
    # Keep shared baselines and opt-in experiments on a second legend row.  A
    # sixth model/visibility series fills most of the first row, so appending
    # these entries horizontally would clip the final label at the SVG edge.
    auxiliary_legend_x = 70.0
    has_shared_baseline = baseline_display == "shared" and any(
        method.source in BASELINE_SOURCES
        for methods in methods_by_task.values()
        for method in methods
    )
    if has_shared_baseline:
        parts.extend(
            [
                rect(auxiliary_legend_x, auxiliary_legend_y, 30, 18, fill="#7A8338", rx=3),
                svg_text(
                    auxiliary_legend_x + 42,
                    auxiliary_legend_y + 15,
                    "Shared train-label baseline",
                    size=13,
                    weight=600,
                ),
            ]
        )
    if experiments:
        experiment_legend_x = auxiliary_legend_x + (300 if has_shared_baseline else 0)
        experiment_legend = (
            "Matched opt-in experiments"
            if show_experiment_model_label
            else f"Matched {next(iter(experiment_model_labels))} experiment"
        )
        parts.extend(
            [
                rect(experiment_legend_x, auxiliary_legend_y, 30, 18, fill=EXPERIMENT_FILL, stroke="#315B57", rx=3),
                svg_text(
                    experiment_legend_x + 42,
                    auxiliary_legend_y + 15,
                    experiment_legend,
                    size=14,
                    weight=600,
                ),
            ]
        )
    if paired_comparisons:
        paired_legend_x = 1030.0
        paired_legend_label = (
            "Exploratory one-sided paired p-value"
            if paired_significance_display == "pvalue"
            else "Best agent paired Δ · 95% CI"
        )
        parts.extend(
            [
                f'<line x1="{paired_legend_x:.1f}" y1="{auxiliary_legend_y + 9:.1f}" '
                f'x2="{paired_legend_x + 30:.1f}" y2="{auxiliary_legend_y + 9:.1f}" '
                f'stroke="{PURPLE}" stroke-width="3"/>',
                f'<circle cx="{paired_legend_x + 15:.1f}" cy="{auxiliary_legend_y + 9:.1f}" r="4.5" '
                f'fill="{PAIRED_POINT_FILL}" stroke="{CARD}" stroke-width="1"/>',
                svg_text(
                    paired_legend_x + 42,
                    auxiliary_legend_y + 15,
                    paired_legend_label,
                    size=13,
                    weight=600,
                ),
            ]
        )

    for task_index, task in enumerate(TASKS):
        y = header_height + task_index * (panel_height + PANEL_GAP)
        panel_width = 980 if len(available_splits) == 1 else 860
        for column, (split, split_label) in enumerate(available_splits):
            x = 460 if len(available_splits) == 1 else 70 + column * 900
            _render_panel(
                parts,
                x=x,
                y=y,
                task=task,
                methods=methods_by_task[task.key],
                split=split,
                split_label=split_label,
                width=panel_width,
                height=panel_height,
                series=series,
                experiments=experiments.get((split, task.key), ()),
                show_experiment_model_label=show_experiment_model_label,
                paired_comparisons=paired_comparisons.get((split, task.key), ()),
                paired_scale_limit=paired_scale_limit,
                paired_significance_display=paired_significance_display,
                baseline_display=baseline_display,
                baseline_series_groups=effective_baseline_groups,
            )

    footer_y = height - FOOTER_HEIGHT + 25
    parts.extend(
        [
            svg_text(
                70,
                footer_y,
                "Bar shade follows the legend"
                + (" · Shared baselines are shown once" if has_shared_baseline else "")
                + (" · Teal rows are matched opt-in experiments" if experiments else "")
                + "."
                + (
                    " · P-values are exploratory one-sided paired permutation tests."
                    if paired_comparisons and paired_significance_display == "pvalue"
                    else " · Paired whiskers show best-agent minus baseline 95% CIs."
                    if paired_comparisons
                    else ""
                ),
                size=13,
                fill=MUTED,
            ),
            svg_text(70, footer_y + 28, "Source: frozen Starling benchmark results"
                     + (" · Failed samples are counted as incorrect." if has_failure else ""), size=13, fill=MUTED),
            svg_text(1830, footer_y + 28, f"Generated {generated}", size=13, fill=MUTED, anchor="end"),
            "</svg>",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-metrics", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--candidate-metrics", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--experiment-metrics",
        type=Path,
        action="append",
        default=[],
        help=(
            "Matched experiment TSV to append to the canonical figure. Repeat for "
            "future experiments; each task/split group must include one anchor row."
        ),
    )
    parser.add_argument(
        "--comparison-metrics",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional complete model/visibility metrics TSV. Repeat to add grouped "
            "bars; task/method keys, subsets, sample counts, and baselines must match."
        ),
    )
    parser.add_argument(
        "--paired-ci-metrics",
        type=Path,
        help=(
            "Paired statistics TSV for the plotted best agent versus MiniMol "
            "train-all and Morgan KNN. The displayed statistic is selected with "
            "--paired-significance-display."
        ),
    )
    parser.add_argument(
        "--paired-significance-display",
        choices=("ci", "pvalue"),
        default="ci",
        help=(
            "Render either paired-bootstrap delta CIs or exploratory one-sided "
            "paired permutation p-values from --paired-ci-metrics."
        ),
    )
    parser.add_argument(
        "--method-family",
        help=(
            "Plot only one method_family from every summary. This is useful for "
            "dataset-version comparisons where train-derived baselines differ."
        ),
    )
    parser.add_argument(
        "--baseline-display",
        choices=("shared", "series"),
        default="shared",
        help=(
            "Use 'shared' for same-dataset model comparisons and 'series' when "
            "each dataset lineage has its own train-derived baselines."
        ),
    )
    parser.add_argument(
        "--baseline-series-group",
        action="append",
        default=[],
        help=(
            "Dataset-lineage group in metrics-series order. Repeated groups collapse "
            "visibility-duplicate baseline bars after equality validation."
        ),
    )
    parser.add_argument(
        "--series-label",
        action="append",
        default=[],
        help=(
            "Legend label in reference, candidate, then --comparison-metrics order. "
            "Repeat exactly once per metrics series."
        ),
    )
    parser.add_argument("--chart-title")
    parser.add_argument("--chart-subtitle")
    parser.add_argument(
        "--context-line",
        action="append",
        default=[],
        help="Header context line; repeat to add auditable dataset notes.",
    )
    parser.add_argument("--comparison-title")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--png-output", type=Path)
    args = parser.parse_args()
    render(
        args.reference_metrics,
        args.candidate_metrics,
        args.output,
        tuple(args.experiment_metrics),
        tuple(args.comparison_metrics),
        args.paired_ci_metrics,
        args.paired_significance_display,
        args.method_family,
        tuple(args.series_label),
        args.chart_title or "Starling Benchmark Model & Experiment Comparison",
        args.chart_subtitle,
        tuple(args.context_line),
        args.comparison_title,
        args.baseline_display,
        tuple(args.baseline_series_group),
    )
    print(args.output)
    if args.png_output is not None:
        export_png(args.output, args.png_output)
        print(args.png_output)


if __name__ == "__main__":
    main()
