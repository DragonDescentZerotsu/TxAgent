"""Plot retrieval coverage against paired macro-F1 change.

The renderer consumes ``coverage_performance.tsv`` produced by
``summarize_results.py``.  It does not recompute model metrics, which keeps the
statistical summary and visual presentation independently testable.
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import date
from pathlib import Path

from .molecular_evidence_agent import DEPLOYMENT_VISIBLE
from .paper_figure_style import (
    BG,
    BLIND,
    CARD,
    GOLD,
    GRID,
    INK,
    MUTED,
    PURPLE,
    VISIBLE,
    rect,
    svg_text,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANALYSIS_DIR = ROOT / "outputs/paper/molecular_evidence_agent/analysis"
DEFAULT_OUTPUT = DEFAULT_ANALYSIS_DIR / "figures/coverage_performance_relationship.svg"
WIDTH = 1800
HEIGHT = 1880

TASKS = (
    ("bbb_martins", "BBB penetration", BLIND),
    ("skin_reaction", "Skin reaction", VISIBLE),
    ("clintox", "Clinical toxicity", PURPLE),
    ("bioavailability_ma", "Oral bioavailability", GOLD),
)
TASK_ORDER = {task: index for index, (task, _, _) in enumerate(TASKS)}
TASK_TITLES = {task: title for task, title, _ in TASKS}
TASK_COLORS = {task: color for task, _, color in TASKS}


def read_rows(analysis_dir: Path, visibility_mode: str) -> list[dict[str, str]]:
    path = analysis_dir / "coverage_performance.tsv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["visibility_mode"] == visibility_mode
        ]
    if not rows:
        raise ValueError(f"No coverage-performance rows for {visibility_mode} in {path}")
    unknown = sorted({row["task"] for row in rows} - TASK_ORDER.keys())
    if unknown:
        raise ValueError(f"Unknown tasks in coverage-performance input: {unknown}")
    return sorted(rows, key=lambda row: (TASK_ORDER[row["task"]], row["condition_label"]))


def _marker(
    parts: list[str],
    x: float,
    y: float,
    *,
    view: str,
    source: str,
    color: str,
    size: float = 9,
) -> None:
    if source not in {"chembl", "starling"}:
        raise ValueError(f"Unknown retrieval source: {source}")
    fill = color if source == "chembl" else "white"
    stroke = "white" if source == "chembl" else color
    stroke_width = 2 if source == "chembl" else 3
    metadata = f'data-source="{source}" data-view="{view}"'
    if view == "direct":
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size:g}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}" {metadata}/>'
        )
    elif view == "full_flat":
        parts.append(
            f'<rect x="{x-size:.1f}" y="{y-size:.1f}" width="{2*size:g}" height="{2*size:g}" '
            f'rx="2" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" {metadata}/>'
        )
    else:
        points = f"{x:.1f},{y-size-1:.1f} {x+size+1:.1f},{y:.1f} {x:.1f},{y+size+1:.1f} {x-size-1:.1f},{y:.1f}"
        parts.append(
            f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width}" {metadata}/>'
        )


def _tick_range(values: list[float]) -> tuple[float, float, list[float]]:
    step = 0.05
    lower = min(-step, math.floor((min(values) - 0.015) / step) * step)
    upper = max(step, math.ceil((max(values) + 0.015) / step) * step)
    ticks = []
    value = lower
    while value <= upper + 1e-9:
        ticks.append(round(value, 10))
        value += step
    return lower, upper, ticks


def render(
    analysis_dir: Path,
    output: Path,
    *,
    data_split: str = "test",
    visibility_mode: str = DEPLOYMENT_VISIBLE,
) -> None:
    if data_split not in {"test", "valid"}:
        raise ValueError(f"Unsupported data split: {data_split}")
    rows = read_rows(analysis_dir, visibility_mode)
    generated = date.today().isoformat()
    point_values = [float(row["delta_macro_f1"]) for row in rows]
    values = [
        value
        for row in rows
        for value in (
            float(row["delta_macro_f1"]),
            float(row["delta_ci_low"]),
            float(row["delta_ci_high"]),
        )
    ]
    y_min, y_max, y_ticks = _tick_range(values)
    positive_count = sum(value > 0 for value in point_values)
    negative_count = sum(value < 0 for value in point_values)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="chart-title chart-desc">',
        '<title id="chart-title">Retrieval coverage and macro-F1 change</title>',
        '<desc id="chart-desc">Overall retrieval coverage is plotted against paired macro-F1 change. Color identifies task, shape identifies retrieval view, and filled versus open markers distinguish ChEMBL from Starling.</desc>',
        f'<metadata>Source: coverage_performance.tsv; split {data_split}; visibility {visibility_mode}; generated {generated}.</metadata>',
        rect(0, 0, WIDTH, HEIGHT, fill=BG, rx=0),
        svg_text(70, 60, "Retrieval Coverage and Macro-F1 Change", size=36, weight=750),
        svg_text(70, 99, f"{visibility_mode} · {data_split} split · Δ versus the same-task no-retrieval condition", size=19, fill=MUTED),
        svg_text(1730, 60, "DESCRIPTIVE ASSOCIATION", size=14, weight=750, fill=PURPLE, anchor="end", spacing=1.0),
    ]

    # Legends retain independent task, retrieval-view, and source encodings.
    parts.append(svg_text(70, 128, "TASK", size=10, weight=750, fill=MUTED, spacing=0.8))
    parts.append(svg_text(998, 128, "VIEW", size=10, weight=750, fill=MUTED, spacing=0.8))
    parts.append(svg_text(1473, 128, "SOURCE", size=10, weight=750, fill=MUTED, spacing=0.8))
    legend_x = 82
    for task, title, color in TASKS:
        parts.append(f'<circle cx="{legend_x}" cy="149" r="7" fill="{color}"/>')
        parts.append(svg_text(legend_x + 14, 154, title, size=13, weight=600))
        legend_x += 225
    shape_x = 1010
    for view, label in (("direct", "Direct"), ("full_flat", "Full / Flat"), ("full_mechanism", "Full / Mechanism")):
        _marker(parts, shape_x, 149, view=view, source="chembl", color=MUTED, size=7)
        parts.append(svg_text(shape_x + 14, 154, label, size=13, fill=MUTED))
        shape_x += 150
    _marker(parts, 1485, 149, view="direct", source="chembl", color=MUTED, size=7)
    parts.append(svg_text(1499, 154, "ChEMBL", size=13, fill=MUTED))
    _marker(parts, 1600, 149, view="direct", source="starling", color=MUTED, size=7)
    parts.append(svg_text(1614, 154, "Starling", size=13, fill=MUTED))

    # Overall coverage versus paired macro-F1 change.
    card_x, card_y, card_w, card_h = 70, 185, 1660, 610
    parts.append(rect(card_x, card_y, card_w, card_h, fill=CARD, stroke=GRID))
    parts.append(svg_text(94, 222, "A · CONDITION-LEVEL RELATIONSHIP", size=14, weight=750, fill=PURPLE, spacing=0.8))
    parts.append(svg_text(180, 249, "Δ macro-F1 (percentage points)", size=12, weight=650, fill=MUTED))
    parts.append(svg_text(1688, 222, f"{positive_count} positive · {negative_count} negative", size=13, weight=650, fill=MUTED, anchor="end"))
    parts.append(svg_text(1688, 246, "Vertical bars: paired-bootstrap 95% CI", size=12, fill=MUTED, anchor="end"))
    left, right, top, bottom = 180, 1680, 265, 725
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + tick * (right - left)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(svg_text(x, bottom + 25, f"{tick * 100:.0f}%", size=12, fill=MUTED, anchor="middle"))
    for tick in y_ticks:
        y = bottom - (tick - y_min) / (y_max - y_min) * (bottom - top)
        width = 2 if abs(tick) < 1e-9 else 1
        color = INK if abs(tick) < 1e-9 else GRID
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{color}" stroke-width="{width}"/>')
        parts.append(svg_text(left - 14, y + 5, f"{tick * 100:+.0f}", size=12, fill=MUTED, anchor="end"))
    for index, row in enumerate(rows, start=1):
        x = left + float(row["coverage"]) * (right - left)
        delta = float(row["delta_macro_f1"])
        y = bottom - (delta - y_min) / (y_max - y_min) * (bottom - top)
        color = TASK_COLORS[row["task"]]
        ci_low = max(y_min, float(row["delta_ci_low"]))
        ci_high = min(y_max, float(row["delta_ci_high"]))
        ci_y_low = bottom - (ci_low - y_min) / (y_max - y_min) * (bottom - top)
        ci_y_high = bottom - (ci_high - y_min) / (y_max - y_min) * (bottom - top)
        parts.append(f'<line x1="{x:.1f}" y1="{ci_y_high:.1f}" x2="{x:.1f}" y2="{ci_y_low:.1f}" stroke="{color}" stroke-width="2" opacity="0.55"/>')
        parts.append(f'<line x1="{x-5:.1f}" y1="{ci_y_high:.1f}" x2="{x+5:.1f}" y2="{ci_y_high:.1f}" stroke="{color}" stroke-width="2" opacity="0.55"/>')
        parts.append(f'<line x1="{x-5:.1f}" y1="{ci_y_low:.1f}" x2="{x+5:.1f}" y2="{ci_y_low:.1f}" stroke="{color}" stroke-width="2" opacity="0.55"/>')
        _marker(
            parts,
            x,
            y,
            view=row["retrieval_view"],
            source=row["source"],
            color=color,
        )
        parts.append(svg_text(x + 12, y - 10, str(index), size=11, weight=750, fill=color))
    parts.append(svg_text((left + right) / 2, 776, "Overall retrieval coverage", size=14, weight=650, fill=MUTED, anchor="middle"))

    # Class-conditional coverage dumbbells. Each row maps back to its scatter ID.
    lower_y = 825
    row_h = 42
    lower_h = 100 + len(rows) * row_h
    parts.append(rect(70, lower_y, 1660, lower_h, fill=CARD, stroke=GRID))
    parts.append(svg_text(94, lower_y + 38, "B · CLASS-CONDITIONAL COVERAGE", size=14, weight=750, fill=PURPLE, spacing=0.8))
    parts.append(svg_text(94, lower_y + 66, "Open endpoint = negative class · Filled endpoint = positive class · Solid segment = ChEMBL · Dashed segment = Starling", size=13, fill=MUTED))
    axis_left, axis_right = 960, 1680
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = axis_left + tick * (axis_right - axis_left)
        parts.append(f'<line x1="{x:.1f}" y1="{lower_y+84}" x2="{x:.1f}" y2="{lower_y+lower_h-20}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(svg_text(x, lower_y + 91, f"{tick * 100:.0f}%", size=11, fill=MUTED, anchor="middle"))

    previous_task = None
    for index, row in enumerate(rows, start=1):
        y = lower_y + 118 + (index - 1) * row_h
        task = row["task"]
        color = TASK_COLORS[task]
        if task != previous_task:
            if previous_task is not None:
                parts.append(f'<line x1="92" y1="{y-row_h/2:.1f}" x2="1705" y2="{y-row_h/2:.1f}" stroke="{GRID}" stroke-width="1"/>')
            previous_task = task
        parts.append(svg_text(94, y + 5, str(index), size=12, weight=750, fill=color))
        parts.append(svg_text(126, y + 5, TASK_TITLES[task], size=12, weight=650, fill=color))
        parts.append(svg_text(310, y + 5, row["condition_label"], size=12, weight=500))
        parts.append(svg_text(690, y + 5, f"n0={row['n_negative']} · n1={row['n_positive']}", size=11, fill=MUTED))
        negative_x = axis_left + float(row["negative_coverage"]) * (axis_right - axis_left)
        positive_x = axis_left + float(row["positive_coverage"]) * (axis_right - axis_left)
        dash = ' stroke-dasharray="7 5"' if row["source"] == "starling" else ""
        parts.append(
            f'<line x1="{negative_x:.1f}" y1="{y}" x2="{positive_x:.1f}" y2="{y}" '
            f'stroke="{color}" stroke-width="3" opacity="0.65"{dash} '
            f'data-source="{row["source"]}"/>'
        )
        parts.append(f'<circle cx="{negative_x:.1f}" cy="{y}" r="7" fill="white" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<circle cx="{positive_x:.1f}" cy="{y}" r="7" fill="{color}" stroke="white" stroke-width="1.5"/>')

    footer_y = lower_y + lower_h + 42
    parts.append(svg_text(70, footer_y, "Interpretation: coverage is retrieval availability, not evidence quality. Task, source, and retrieval view change jointly; the association is not causal.", size=13, fill=MUTED))
    parts.append(svg_text(70, footer_y + 25, "Macro-F1 is primary because labels are imbalanced; class-conditional coverage diagnoses whether retrieval availability is itself label-skewed.", size=13, fill=MUTED))
    parts.append(svg_text(1730, footer_y + 25, f"Generated {generated}", size=12, fill=MUTED, anchor="end"))
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-split", choices=("test", "valid"), default="test")
    parser.add_argument("--visibility-mode", default=DEPLOYMENT_VISIBLE)
    args = parser.parse_args()
    render(
        args.analysis_dir,
        args.output,
        data_split=args.data_split,
        visibility_mode=args.visibility_mode,
    )
    print(args.output)


if __name__ == "__main__":
    main()
