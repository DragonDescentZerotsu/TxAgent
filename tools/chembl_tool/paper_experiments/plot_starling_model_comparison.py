"""Render the canonical Starling model-and-experiment comparison figure.

The required reference/candidate summaries provide the frozen GPT-OSS model
comparison. Optional complete model/visibility summaries add grouped bars.
Optional matched experiment TSVs add rows to the same figure; their anchor
rows are validated against an existing loaded model/visibility condition and
are not plotted twice.
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


def _validate_pair(
    reference: dict[tuple[str, str, str], dict[str, str]],
    candidate: dict[tuple[str, str, str], dict[str, str]],
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
        if source not in BASELINE_SOURCES:
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
    split: str,
    split_label: str,
    width: float,
    height: float,
    series: tuple[dict[tuple[str, str, str], dict[str, str]], ...],
    experiments: tuple[ExperimentMetric, ...],
    show_experiment_model_label: bool,
) -> None:
    parts.append(rect(x, y, width, height, fill=CARD, stroke=GRID))
    first = series[0][(split, task.key, task.methods[0].key)]
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
    plot_top, plot_bottom = y + 66, y + height - 65
    append_vertical_grid(
        parts,
        plot_left=plot_left,
        plot_right=plot_right,
        plot_top=plot_top,
        plot_bottom=plot_bottom,
        ticks=(0.0, 0.2, 0.4, 0.6, 0.8),
        scale_max=SCALE_MAX,
        label_y=y + height - 33,
    )

    count = len(task.methods) + len(experiments)
    available = plot_bottom - plot_top - 30
    max_step = 42.0 if len(series) <= 3 else 10.0 * len(series) + 4.0
    step = min(max_step, available / max(1, count - 1))
    row_start = plot_top + 15 + (available - step * (count - 1)) / 2
    if experiments:
        experiment_start = row_start + len(task.methods) * step
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
    for index, method in enumerate(task.methods):
        row_y = row_start + index * step
        series_rows = [result[(split, task.key, method.key)] for result in series]
        color = SOURCE_COLORS[method.source]
        is_baseline = method.source in BASELINE_SOURCES
        if is_baseline:
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
        if is_baseline:
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

        bar_gap = 2.0 if len(series) <= 3 else 3.0
        minimum_bar_height = 4.0 if len(series) <= 3 else 5.0
        bar_height = min(
            8.0,
            max(
                minimum_bar_height,
                (step - 8.0) / len(series) - bar_gap,
            ),
        )
        group_height = len(series) * bar_height + (len(series) - 1) * bar_gap
        first_bar_y = row_y - group_height / 2
        for series_index, series_row in enumerate(series_rows):
            value = float(series_row["macro_f1"])
            bar_y = first_bar_y + series_index * (bar_height + bar_gap)
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
        row_y = row_start + (len(task.methods) + experiment_index) * step
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
            y + height - 11,
            "Macro-F1",
            size=12,
            weight=650,
            fill=MUTED,
            anchor="middle",
        )
    )


def render(
    reference_path: Path,
    candidate_path: Path,
    output: Path,
    experiment_paths: tuple[Path, ...] = (),
    comparison_paths: tuple[Path, ...] = (),
) -> None:
    reference = read_metrics(reference_path)
    candidate = read_metrics(candidate_path)
    available_splits = _validate_pair(reference, candidate)
    additional = tuple(read_metrics(path) for path in comparison_paths)
    for result in additional:
        if _validate_pair(reference, result) != available_splits:
            raise ValueError("Additional comparison metrics changed available splits")
    series = (reference, candidate, *additional)
    series_labels = tuple(_agent_model_label(result) for result in series)
    series_by_label = dict(zip(series_labels, series, strict=True))
    reference_label = series_labels[0]
    candidate_label = _agent_model_label(candidate)
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
    split_title = " and ".join(label.replace(" split", "") for _, label in available_splits)
    has_failure = any(
        int(row.get("n_failed") or 0) > 0
        for result in series
        for row in result.values()
    )
    generated = date.today().isoformat()
    max_rows = max(
        len(task.methods) + len(experiments.get((split, task.key), ()))
        for split, _ in available_splits
        for task in TASKS
    )
    dense_series_extra = max(0, len(series) - 3) * 110
    panel_height = max(
        510,
        510
        + (max_rows - max(len(task.methods) for task in TASKS)) * 38
        + dense_series_extra,
    )
    height = HEADER_HEIGHT + len(TASKS) * panel_height + (len(TASKS) - 1) * PANEL_GAP + FOOTER_HEIGHT
    source_paths = (
        reference_path,
        candidate_path,
        *comparison_paths,
        *experiment_paths,
    )
    comparison_title = (
        f"{reference_label} vs {candidate_label}"
        if len(series_labels) == 2
        else f"{len(series_labels)} model/visibility settings"
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="chart-title chart-desc">',
        '<title id="chart-title">Starling benchmark model and experiment comparison</title>',
        f'<desc id="chart-desc">Grouped horizontal bars compare {", ".join(series_labels)} '
        f'macro-F1 for Starling scaffold valid tasks, with shared train-label baselines and matched experiment additions.</desc>',
        f'<metadata>Sources: {"; ".join(str(path) for path in source_paths)}; generated {generated}.</metadata>',
        rect(0, 0, WIDTH, height, fill=BG, rx=0),
        svg_text(70, 58, "Starling Benchmark Model & Experiment Comparison", size=36, weight=750),
        svg_text(70, 94, f"{split_title} {evaluation_subset} · Macro-F1", size=20, fill=MUTED),
        svg_text(1830, 58, comparison_title, size=18, weight=700, fill=PURPLE, anchor="end"),
        svg_text(
            70,
            132,
            "Same scaffold-valid data and parent-disjoint retrieval · Visibility/tool-execution contract varies by series",
            size=14,
            fill=MUTED,
        ),
    ]
    legend_x = 70.0
    for series_index, label in enumerate(series_labels):
        parts.append(
            rect(
                legend_x,
                157,
                30,
                10,
                fill=_series_fill(PURPLE, series_index, len(series_labels)),
                stroke=PURPLE,
                rx=2,
            )
        )
        parts.append(svg_text(legend_x + 42, 168, label, size=13, weight=600))
        legend_x += max(225.0, 82.0 + len(label) * 7.2)
    # Keep shared baselines and opt-in experiments on a second legend row.  A
    # sixth model/visibility series fills most of the first row, so appending
    # these entries horizontally would clip the final label at the SVG edge.
    auxiliary_legend_x = 70.0
    parts.extend(
        [
            rect(auxiliary_legend_x, 183, 30, 18, fill="#7A8338", rx=3),
            svg_text(
                auxiliary_legend_x + 42,
                198,
                "Shared train-label baseline",
                size=13,
                weight=600,
            ),
        ]
    )
    if experiments:
        experiment_legend_x = auxiliary_legend_x + 300
        experiment_legend = (
            "Matched opt-in experiments"
            if show_experiment_model_label
            else f"Matched {next(iter(experiment_model_labels))} experiment"
        )
        parts.extend(
            [
                rect(experiment_legend_x, 183, 30, 18, fill=EXPERIMENT_FILL, stroke="#315B57", rx=3),
                svg_text(
                    experiment_legend_x + 42,
                    198,
                    experiment_legend,
                    size=14,
                    weight=600,
                ),
            ]
        )

    for task_index, task in enumerate(TASKS):
        y = HEADER_HEIGHT + task_index * (panel_height + PANEL_GAP)
        panel_width = 980 if len(available_splits) == 1 else 860
        for column, (split, split_label) in enumerate(available_splits):
            x = 460 if len(available_splits) == 1 else 70 + column * 900
            _render_panel(
                parts,
                x=x,
                y=y,
                task=task,
                split=split,
                split_label=split_label,
                width=panel_width,
                height=panel_height,
                series=series,
                experiments=experiments.get((split, task.key), ()),
                show_experiment_model_label=show_experiment_model_label,
            )

    footer_y = height - FOOTER_HEIGHT + 25
    parts.extend(
        [
            svg_text(70, footer_y, "Bar shade follows the legend · Shared baselines are shown once · Teal rows are matched opt-in experiments.", size=13, fill=MUTED),
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--png-output", type=Path)
    args = parser.parse_args()
    render(
        args.reference_metrics,
        args.candidate_metrics,
        args.output,
        tuple(args.experiment_metrics),
        tuple(args.comparison_metrics),
    )
    print(args.output)
    if args.png_output is not None:
        export_png(args.output, args.png_output)
        print(args.png_output)


if __name__ == "__main__":
    main()
