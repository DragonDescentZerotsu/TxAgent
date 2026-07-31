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
    INK,
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

    row_step = (plot_bottom - plot_top) / len(CONDITIONS)
    bar_height = 16
    for index, (task, split, condition_label) in enumerate(CONDITIONS):
        center_y = plot_top + row_step * (index + 0.5)
        n_value = int(rows[(task, split, METHODS[0][0])]["n"])
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
        for method_index, (method, _, color) in enumerate(METHODS):
            value = float(rows[(task, split, method)][metric])
            bar_y = center_y - 20 + method_index * 22
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
    rows = read_metrics(metrics_path)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="chart-title chart-desc">',
        '<title id="chart-title">Morgan similarity and coverage retrieval across Starling tasks</title>',
        '<desc id="chart-desc">Two horizontal grouped-bar panels compare accuracy and macro-F1 for Morgan similarity and coverage retrieval on random and scaffold splits of three Starling tasks.</desc>',
        f"<metadata>Source: {metrics_path}; generated {date.today().isoformat()}.</metadata>",
        rect(0, 0, WIDTH, HEIGHT, fill=BG, rx=0),
        svg_text(70, 62, "Retrieval Selector Comparison Across Starling", size=36, weight=750),
        svg_text(
            70,
            100,
            "Three tasks · random and scaffold splits · strict full-sample matched comparisons",
            size=19,
            fill=MUTED,
        ),
        svg_text(
            70,
            130,
            "Full-mechanism GLM · parent-disjoint · top-k = 3 · minimum similarity = 0.30",
            size=14,
            fill=MUTED,
        ),
    ]
    legend_x = 910
    for _, label, color in METHODS:
        parts.append(rect(legend_x, 68, 20, 20, fill=color, rx=3))
        parts.append(svg_text(legend_x + 30, 84, label, size=15, weight=650))
        legend_x += 330

    render_panel(
        parts,
        x=70,
        y=170,
        width=1560,
        height=410,
        metric="accuracy",
        metric_label="Accuracy",
        rows=rows,
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
    )
    parts.extend(
        [
            svg_text(
                70,
                1080,
                "Only the retrieval selector differs within each task/split pair; all other inference and evidence contracts are matched.",
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
