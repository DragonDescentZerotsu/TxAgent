"""Plot matched Morgan-versus-coverage LLM metrics across Starling conditions."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from .paper_figure_style import (
    BG,
    BLIND,
    CARD,
    GRID,
    MUTED,
    VISIBLE,
    append_vertical_grid,
    rect,
    run_metric_plot_cli,
    svg_text,
)
from .summarize_coverage_selector_llm_matrix import CONDITIONS as ANALYSIS_CONDITIONS


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANALYSIS_DIR = ROOT / "outputs/paper/coverage_selector_llm/analysis"
DEFAULT_METRICS = DEFAULT_ANALYSIS_DIR / "metrics.tsv"
DEFAULT_OUTPUT = DEFAULT_ANALYSIS_DIR / "figures/coverage_selector_llm_matrix.svg"

WIDTH = 1700
HEIGHT = 1130
METHODS = (
    ("morgan_similarity", "Morgan similarity retrieve", BLIND),
    ("query_feature_coverage", "Coverage retrieve", VISIBLE),
)
TASK_LABELS = {
    "bbb_martins": "BBB",
    "bioavailability_ma": "Bioavailability",
    "skin_reaction": "Skin reaction",
}
CONDITIONS = tuple(
    (task, split, f"{TASK_LABELS[task]} · {split.title()}")
    for split, task, _, _ in ANALYSIS_CONDITIONS
)


def read_metrics(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = {
            (row["task"], row["benchmark_split"], row["method"]): row
            for row in csv.DictReader(handle, delimiter="\t")
        }
    expected = {
        (task, split, method)
        for task, split, _ in CONDITIONS
        for method, _, _ in METHODS
    }
    missing = sorted(expected - set(rows))
    if missing:
        raise ValueError(f"Missing matched matrix rows: {missing}")
    return rows


def _load_plot_contract(
    path: Path,
) -> tuple[
    dict[tuple[str, str, str], dict[str, str]],
    tuple[tuple[str, str, str], ...],
    tuple[tuple[str, str, str], ...],
]:
    with path.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle, delimiter="\t"))
    rows = {
        (row["task"], row["benchmark_split"], row["method"]): row
        for row in raw_rows
    }
    method_order = list(dict.fromkeys(row["method"] for row in raw_rows))
    if method_order == [method for method, _, _ in METHODS]:
        return rows, CONDITIONS, METHODS
    if not 2 <= len(method_order) <= 3:
        raise ValueError(f"Expected two or three methods, found {method_order}")
    task_order = {task: index for index, task in enumerate(TASK_LABELS)}
    split_order = {"random": 0, "scaffold": 1}
    pairs = sorted(
        {(row["task"], row["benchmark_split"]) for row in raw_rows},
        key=lambda pair: (
            task_order.get(pair[0], len(task_order)),
            split_order.get(pair[1], len(split_order)),
        ),
    )
    conditions = tuple(
        (
            task,
            split,
            f"{TASK_LABELS.get(task, task)} · {split.title()}",
        )
        for task, split in pairs
    )
    palette = (BLIND, "#6C7A89", VISIBLE) if len(method_order) == 3 else (BLIND, VISIBLE)
    methods = tuple(
        (
            method,
            next(
                (
                    row.get("method_label") or method
                    for row in raw_rows
                    if row["method"] == method
                ),
                method,
            ),
            color,
        )
        for method, color in zip(method_order, palette, strict=True)
    )
    expected = {
        (task, split, method)
        for task, split, _ in conditions
        for method, _, _ in methods
    }
    missing = sorted(expected - set(rows))
    if missing:
        raise ValueError(f"Missing matched matrix rows: {missing}")
    return rows, conditions, methods


def render_panel(
    parts: list[str],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    metric: str,
    metric_label: str,
    rows: dict[tuple[str, str, str], dict[str, str]],
    conditions: tuple[tuple[str, str, str], ...],
    methods: tuple[tuple[str, str, str], ...],
) -> None:
    parts.append(rect(x, y, width, height, fill=CARD, stroke=GRID))
    parts.append(svg_text(x + 24, y + 38, metric_label, size=24, weight=750))
    plot_left, plot_right = x + 270, x + width - 72
    plot_top, plot_bottom = y + 74, y + height - 54
    append_vertical_grid(
        parts,
        plot_left=plot_left,
        plot_right=plot_right,
        plot_top=plot_top,
        plot_bottom=plot_bottom,
        ticks=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        scale_max=1.0,
        label_y=plot_bottom + 28,
    )

    row_step = (plot_bottom - plot_top) / len(conditions)
    bar_height = 16
    for index, (task, split, condition_label) in enumerate(conditions):
        center_y = plot_top + row_step * (index + 0.5)
        n_value = int(rows[(task, split, methods[0][0])]["n"])
        parts.append(
            svg_text(
                x + 22,
                center_y + 4,
                condition_label,
                size=14,
                weight=600,
            )
        )
        parts.append(
            svg_text(
                plot_left - 14,
                center_y + 4,
                f"n={n_value}",
                size=12,
                fill=MUTED,
                anchor="end",
            )
        )
        total_bar_height = (len(methods) - 1) * 22 + bar_height
        for method_index, (method, _, color) in enumerate(methods):
            value = float(rows[(task, split, method)][metric])
            bar_y = center_y - total_bar_height / 2 + method_index * 22
            bar_width = value * (plot_right - plot_left)
            parts.append(
                rect(plot_left, bar_y, bar_width, bar_height, fill=color, rx=3)
            )
            parts.append(
                svg_text(
                    plot_left + bar_width + 8,
                    bar_y + 13,
                    f"{value:.3f}",
                    size=12,
                    weight=700,
                )
            )


def render(metrics_path: Path, output: Path) -> None:
    rows, conditions, methods = _load_plot_contract(metrics_path)
    legacy = methods == METHODS and conditions == CONDITIONS
    title = (
        "Retrieval Selector Comparison Across Starling"
        if legacy
        else "Coverage-Aware Reasoning Comparison"
    )
    subtitle = (
        "Three tasks · random and scaffold splits · strict full-sample matched comparisons"
        if legacy
        else f"{len(conditions)} task/split conditions · strict full-sample matched comparisons"
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="chart-title chart-desc">',
        f'<title id="chart-title">{title}</title>',
        '<desc id="chart-desc">Two horizontal grouped-bar panels compare accuracy and macro-F1 for strictly matched Starling agent profiles.</desc>',
        f"<metadata>Source: {metrics_path}; generated {date.today().isoformat()}.</metadata>",
        rect(0, 0, WIDTH, HEIGHT, fill=BG, rx=0),
        svg_text(70, 62, title, size=36, weight=750),
        svg_text(
            70,
            100,
            subtitle,
            size=19,
            fill=MUTED,
        ),
        svg_text(
            70,
            130,
            (
                "Full-mechanism GLM · parent-disjoint · top-k = 3 · minimum similarity = 0.30"
                if legacy
                else "Full-mechanism agent · parent-disjoint · top-k = 3 · minimum similarity = 0.30"
            ),
            size=14,
            fill=MUTED,
        ),
    ]
    if legacy:
        legend_positions = ((910, 68), (1240, 68))
    elif len(methods) == 2:
        legend_positions = ((650, 68), (1150, 68))
    else:
        legend_positions = ((780, 58), (780, 94), (1190, 94))
    for (_, label, color), (legend_x, legend_y) in zip(
        methods, legend_positions, strict=True
    ):
        parts.append(rect(legend_x, legend_y, 20, 20, fill=color, rx=3))
        parts.append(svg_text(legend_x + 30, legend_y + 16, label, size=15, weight=650))

    render_panel(
        parts,
        x=70,
        y=170,
        width=1560,
        height=410,
        metric="accuracy",
        metric_label="Accuracy",
        rows=rows,
        conditions=conditions,
        methods=methods,
    )
    render_panel(
        parts,
        x=70,
        y=610,
        width=1560,
        height=410,
        metric="macro_f1",
        metric_label="Macro-F1",
        rows=rows,
        conditions=conditions,
        methods=methods,
    )
    parts.extend(
        [
            svg_text(
                70,
                1080,
                (
                    "Only the retrieval selector differs within each task/split pair; all other inference and evidence contracts are matched."
                    if legacy
                    else (
                        "The two profiles use the same coverage-selected neighbors; only the LLM-visible coverage context differs."
                        if len(methods) == 2
                        else "Morgan is the original baseline; the two coverage profiles share neighbors and differ only in LLM-visible coverage context."
                    )
                ),
                size=13,
                fill=MUTED,
            ),
            svg_text(
                WIDTH - 70,
                1080,
                f"Generated {date.today().isoformat()}",
                size=13,
                fill=MUTED,
                anchor="end",
            ),
            "</svg>",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    run_metric_plot_cli(
        description=__doc__,
        render=render,
        default_metrics=DEFAULT_METRICS,
        default_output=DEFAULT_OUTPUT,
    )


if __name__ == "__main__":
    main()
