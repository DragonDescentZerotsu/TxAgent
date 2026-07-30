"""Compare MiniMol operational agent retrieval with all existing Starling results."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path
from typing import Any

from .minimol_retrieval_contract import paper_root_for_minimol_retrieval
from .paper_figure_style import (
    BG,
    BLIND,
    CARD,
    GOLD,
    GRID,
    INK,
    MUTED,
    NEUTRAL,
    PARENT,
    PURPLE,
    VISIBLE,
    rect,
    svg_text,
)
from .plot_starling_benchmark_overview import (
    BASELINE_SOURCES,
    DEFAULT_METRICS,
    SCALE_MAX,
    SPLITS,
    TASKS,
    Task,
    export_png,
    read_metrics,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = ROOT / "outputs/paper/minimol_retrieval_agent_results"
DEFAULT_OUTPUT = (
    DEFAULT_RESULTS_DIR
    / "figures/starling_benchmark_with_minimol_agent_operational.svg"
)
DEFAULT_DATA_OUTPUT = DEFAULT_RESULTS_DIR / "operational_all_results_comparison.tsv"

WIDTH = 1900
HEIGHT = 1840
AGENT_SOURCES = {"chembl", "starling"}
SINGLE_COLORS = {
    "none": NEUTRAL,
    "minimol": BLIND,
    "knn": VISIBLE,
    "minimol_knn": PARENT,
}
SINGLE_FILLS = {
    "minimol": "#EDF3FA",
    "knn": "#FBEDE9",
    "minimol_knn": "#F2F4E8",
}


def read_minimol_operational() -> dict[tuple[str, str, str], dict[str, Any]]:
    results: dict[tuple[str, str, str], dict[str, Any]] = {}
    missing: list[str] = []
    failed: list[str] = []
    for split, _ in SPLITS:
        root = paper_root_for_minimol_retrieval(split) / "runs_deployment_visible"
        for task in TASKS:
            for method in task.methods:
                if method.source not in AGENT_SOURCES:
                    continue
                batch = root / task.key / f"{task.key}__{method.key}"
                metrics_path = batch / "metrics.json"
                key = (split, task.key, method.key)
                if not metrics_path.is_file():
                    missing.append(":".join(key))
                    continue
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                if int(metrics.get("n_failed_runs") or 0):
                    failed.append(":".join(key))
                results[key] = {
                    **metrics,
                    "metrics_path": str(metrics_path),
                }
    if missing:
        raise ValueError(f"Missing MiniMol operational agent results: {missing}")
    if failed:
        raise ValueError(f"Cannot plot MiniMol operational rows with failures: {failed}")
    return results


def write_comparison_data(
    path: Path,
    *,
    existing: dict[tuple[str, str, str], dict[str, str]],
    minimol: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    for split, _ in SPLITS:
        for task in TASKS:
            for method in task.methods:
                key = (split, task.key, method.key)
                row = existing[key]
                rows.append(
                    {
                        "benchmark_split": split,
                        "task": task.key,
                        "method": method.key,
                        "display_label": method.label,
                        "result_series": _existing_series(method.source),
                        "neighbor_identity_policy": row.get(
                            "neighbor_identity_policy"
                        )
                        or "",
                        "n_test": row["n_test"],
                        "n_failed": row["n_failed"],
                        "macro_f1": row["macro_f1"],
                        "metrics_path": row.get("metrics_path") or "",
                    }
                )
                if method.source in AGENT_SOURCES:
                    feature_row = minimol[key]
                    rows.append(
                        {
                            "benchmark_split": split,
                            "task": task.key,
                            "method": method.key,
                            "display_label": method.label,
                            "result_series": "MiniMol agent retrieval",
                            "neighbor_identity_policy": "operational",
                            "n_test": feature_row["n_evaluable"],
                            "n_failed": feature_row["n_failed_runs"],
                            "macro_f1": feature_row["macro_f1"],
                            "metrics_path": feature_row["metrics_path"],
                        }
                    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _existing_series(source: str) -> str:
    return {
        "none": "No retrieval",
        "chembl": "Morgan agent retrieval",
        "starling": "Morgan agent retrieval",
        "minimol": "MiniMol train-all head",
        "knn": "Morgan KNN",
        "minimol_knn": "MiniMol KNN",
    }[source]


def render_panel(
    parts: list[str],
    *,
    x: float,
    y: float,
    task: Task,
    split: str,
    split_label: str,
    existing: dict[tuple[str, str, str], dict[str, str]],
    minimol: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    width, height = 860, 490
    parts.append(rect(x, y, width, height, fill=CARD, stroke=GRID))
    first = existing[(split, task.key, task.methods[0].key)]
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
    plot_left, plot_right = x + 270, x + 748
    plot_top, plot_bottom = y + 66, y + 427
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8):
        tick_x = plot_left + tick / SCALE_MAX * (plot_right - plot_left)
        parts.append(
            f'<line x1="{tick_x:.1f}" y1="{plot_top}" x2="{tick_x:.1f}" '
            f'y2="{plot_bottom}" stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            svg_text(
                tick_x,
                y + 459,
                f"{tick:.1f}",
                size=12,
                fill=MUTED,
                anchor="middle",
            )
        )

    count = len(task.methods)
    step = min(44.0, 340.0 / max(1, count - 1))
    row_start = y + 87 + (340.0 - step * (count - 1)) / 2
    for index, method in enumerate(task.methods):
        row_y = row_start + index * step
        key = (split, task.key, method.key)
        if method.source in BASELINE_SOURCES:
            band_height = min(34.0, max(24.0, step - 2.0))
            parts.append(
                rect(
                    x + 12,
                    row_y - band_height / 2,
                    width - 24,
                    band_height,
                    fill=SINGLE_FILLS[method.source],
                    rx=4,
                )
            )
        parts.append(
            svg_text(
                x + 22,
                row_y + 5,
                method.label,
                size=13,
                weight=700 if method.source in BASELINE_SOURCES else 500,
            )
        )
        if method.source in AGENT_SOURCES:
            values = (
                (float(existing[key]["macro_f1"]), -10, GOLD),
                (float(minimol[key]["macro_f1"]), 3, PURPLE),
            )
            for value, offset, color in values:
                bar_width = value / SCALE_MAX * (plot_right - plot_left)
                parts.append(
                    rect(plot_left, row_y + offset, bar_width, 10, fill=color, rx=2)
                )
                parts.append(
                    svg_text(
                        plot_left + bar_width + 6,
                        row_y + offset + 9,
                        f"{value:.3f}",
                        size=10,
                        weight=650,
                        fill=color,
                    )
                )
            continue

        value = float(existing[key]["macro_f1"])
        color = SINGLE_COLORS[method.source]
        bar_width = value / SCALE_MAX * (plot_right - plot_left)
        parts.append(rect(plot_left, row_y - 9, bar_width, 18, fill=color, rx=3))
        parts.append(
            svg_text(
                plot_left + bar_width + 8,
                row_y + 5,
                f"{value:.3f}",
                size=12,
                weight=700,
                fill=INK,
            )
        )
    parts.append(
        svg_text(
            (plot_left + plot_right) / 2,
            y + 480,
            "Macro-F1",
            size=12,
            weight=650,
            fill=MUTED,
            anchor="middle",
        )
    )


def render(metrics_path: Path, output: Path, *, data_output: Path) -> None:
    existing = read_metrics(metrics_path)
    minimol = read_minimol_operational()
    write_comparison_data(data_output, existing=existing, minimol=minimol)
    generated = date.today().isoformat()
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="chart-title chart-desc">',
        '<title id="chart-title">Starling benchmark results with MiniMol agent retrieval</title>',
        '<desc id="chart-desc">Six horizontal bar panels compare existing formal Starling benchmark results with MiniMol embedding cosine operational agent retrieval across tasks and random or scaffold splits.</desc>',
        f"<metadata>Sources: {metrics_path}; MiniMol operational roots; generated {generated}.</metadata>",
        rect(0, 0, WIDTH, HEIGHT, fill=BG, rx=0),
        svg_text(
            70,
            58,
            "Starling Benchmark with MiniMol Agent Retrieval",
            size=34,
            weight=750,
        ),
        svg_text(
            70,
            94,
            "Existing formal results plus MiniMol/cosine operational agent · Macro-F1",
            size=19,
            fill=MUTED,
        ),
        svg_text(
            1830,
            58,
            "GLM-5.2 + train-label baselines",
            size=18,
            weight=700,
            fill=PURPLE,
            anchor="end",
        ),
    ]
    legend = (
        ("Morgan agent · parent-disjoint", GOLD, 270),
        ("MiniMol agent · operational", PURPLE, 250),
        ("No retrieval", NEUTRAL, 160),
        ("MiniMol head", BLIND, 170),
        ("Morgan KNN", VISIBLE, 170),
        ("MiniMol KNN", PARENT, 170),
    )
    legend_x = 70
    for label, color, width in legend:
        parts.append(rect(legend_x, 125, 18, 18, fill=color, rx=3))
        parts.append(svg_text(legend_x + 27, 139, label, size=13, weight=600))
        legend_x += width

    panel_y = (185, 705, 1225)
    for task, y in zip(TASKS, panel_y, strict=True):
        for column, (split, split_label) in enumerate(SPLITS):
            render_panel(
                parts,
                x=70 + column * 900,
                y=y,
                task=task,
                split=split,
                split_label=split_label,
                existing=existing,
                minimol=minimol,
            )
    parts.extend(
        [
            svg_text(
                70,
                1762,
                "Agent comparison changes retrieval feature and identity policy: existing Morgan bars are formal parent-disjoint; MiniMol bars are operational.",
                size=13,
                fill=MUTED,
            ),
            svg_text(
                70,
                1792,
                "No-retrieval and train-label baselines are shared context. Exact values and provenance are in operational_all_results_comparison.tsv.",
                size=13,
                fill=MUTED,
            ),
            svg_text(
                1830,
                1792,
                f"Generated {generated}",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-output", type=Path, default=DEFAULT_DATA_OUTPUT)
    parser.add_argument("--png-output", type=Path)
    args = parser.parse_args()
    render(args.metrics, args.output, data_output=args.data_output)
    print(args.output)
    print(args.data_output)
    if args.png_output is not None:
        export_png(args.output, args.png_output)
        print(args.png_output)


if __name__ == "__main__":
    main()
