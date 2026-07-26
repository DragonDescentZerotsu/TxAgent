"""Render the Starling random/scaffold benchmark grouped-bar overview.

The figure compares the parent-disjoint molecular-evidence-agent conditions
with the train-all MiniMol baseline. It reads the consolidated benchmark TSV,
writes a canonical SVG, and optionally exports a high-resolution PNG.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .paper_figure_style import (
    BG,
    BLIND,
    CARD,
    GRID,
    GOLD,
    INK,
    MUTED,
    NEUTRAL,
    PURPLE,
    VISIBLE,
    rect,
    svg_text,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = ROOT / "outputs/paper/starling_benchmark_results"
DEFAULT_METRICS = DEFAULT_RESULTS_DIR / "metrics.tsv"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "figures/starling_benchmark_overview.svg"

WIDTH = 1900
HEIGHT = 1840
SCALE_MAX = 0.85


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    source: str


@dataclass(frozen=True)
class Task:
    key: str
    title: str
    methods: tuple[Method, ...]


COMMON_METHODS = (
    Method("none", "No retrieval", "none"),
    Method("chembl_direct", "ChEMBL · Direct", "chembl"),
    Method("chembl_full_flat", "ChEMBL · Full / Flat", "chembl"),
    Method("chembl_full_mechanism", "ChEMBL · Full / Mechanism", "chembl"),
)

TASKS = (
    Task(
        "bbb_martins",
        "BBB penetration",
        COMMON_METHODS
        + (
            Method("starling_direct", "Starling · Direct", "starling"),
            Method("starling_full_flat", "Starling · Full / Flat", "starling"),
            Method("starling_full_mechanism", "Starling · Full / Mechanism", "starling"),
            Method("minimol_train_all", "MiniMol · Train all", "minimol"),
            Method("morgan_knn_k3", "Morgan KNN · k=3", "knn"),
        ),
    ),
    Task(
        "skin_reaction",
        "Skin reaction",
        COMMON_METHODS
        + (
            Method("starling_direct", "Starling · Direct", "starling"),
            Method("starling_full_flat", "Starling · Full / Flat", "starling"),
            Method("starling_full_mechanism", "Starling · Full / Mechanism", "starling"),
            Method("minimol_train_all", "MiniMol · Train all", "minimol"),
            Method("morgan_knn_k3", "Morgan KNN · k=3", "knn"),
        ),
    ),
    Task(
        "bioavailability_ma",
        "Oral bioavailability",
        COMMON_METHODS
        + (
            Method("starling_direct_numeric", "Starling · Direct (numeric)", "starling"),
            Method("starling_direct_full", "Starling · Direct (full)", "starling"),
            Method("starling_full_flat", "Starling · Full / Flat", "starling"),
            Method("starling_full_mechanism", "Starling · Full / Mechanism", "starling"),
            Method("minimol_train_all", "MiniMol · Train all", "minimol"),
            Method("morgan_knn_k3", "Morgan KNN · k=3", "knn"),
        ),
    ),
)

SPLITS = (("random", "Random split"), ("scaffold", "Scaffold split"))
SOURCE_COLORS = {
    "none": NEUTRAL,
    "chembl": BLIND,
    "starling": VISIBLE,
    "minimol": PURPLE,
    "knn": GOLD,
}


def read_metrics(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    results = {
        (row["benchmark_split"], row["task"], row["method"]): row
        for row in rows
    }
    expected = {
        (split, task.key, method.key)
        for split, _ in SPLITS
        for task in TASKS
        for method in task.methods
    }
    missing = sorted(expected - results.keys())
    if missing:
        raise ValueError(f"Missing Starling benchmark results: {missing}")
    failed = [
        key for key in expected if int(results[key].get("n_failed") or 0) != 0
    ]
    if failed:
        raise ValueError(f"Cannot plot benchmark rows with failures: {sorted(failed)}")
    return results


def render_panel(
    parts: list[str],
    *,
    x: float,
    y: float,
    task: Task,
    split: str,
    split_label: str,
    results: dict[tuple[str, str, str], dict[str, str]],
) -> None:
    width, height = 860, 490
    parts.append(rect(x, y, width, height, fill=CARD, stroke=GRID))
    first = results[(split, task.key, task.methods[0].key)]
    n_test = int(first["n_test"])
    parts.append(svg_text(x + 22, y + 34, task.title, size=22, weight=750))
    parts.append(svg_text(x + width - 22, y + 33, f"{split_label} · n = {n_test}", size=14, weight=650, fill=MUTED, anchor="end"))

    plot_left, plot_right = x + 270, x + 760
    plot_top, plot_bottom = y + 66, y + 427
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8):
        tick_x = plot_left + tick / SCALE_MAX * (plot_right - plot_left)
        parts.append(
            f'<line x1="{tick_x:.1f}" y1="{plot_top}" x2="{tick_x:.1f}" '
            f'y2="{plot_bottom}" stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(svg_text(tick_x, y + 459, f"{tick:.1f}", size=12, fill=MUTED, anchor="middle"))

    count = len(task.methods)
    step = min(44.0, 340.0 / max(1, count - 1))
    row_start = y + 87 + (340.0 - step * (count - 1)) / 2
    for index, method in enumerate(task.methods):
        row_y = row_start + index * step
        row = results[(split, task.key, method.key)]
        value = float(row["macro_f1"])
        color = SOURCE_COLORS[method.source]
        if method.source in {"minimol", "knn"}:
            baseline_fill = "#F2EFF8" if method.source == "minimol" else "#FBF3E6"
            parts.append(rect(x + 12, row_y - 18, width - 24, 36, fill=baseline_fill, rx=4))
        parts.append(
            svg_text(
                x + 22,
                row_y + 5,
                method.label,
                size=13,
                weight=700 if method.source in {"minimol", "knn"} else 500,
            )
        )
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
    parts.append(svg_text((plot_left + plot_right) / 2, y + 480, "Macro-F1", size=12, weight=650, fill=MUTED, anchor="middle"))


def render(metrics_path: Path, output: Path) -> None:
    results = read_metrics(metrics_path)
    generated = date.today().isoformat()
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="chart-title chart-desc">',
        '<title id="chart-title">Starling benchmark performance by task and split</title>',
        '<desc id="chart-desc">Horizontal bar charts compare no retrieval, ChEMBL retrieval, Starling retrieval, and MiniMol macro-F1 for three tasks on random and scaffold splits.</desc>',
        f'<metadata>Source: {metrics_path}; generated {generated}.</metadata>',
        rect(0, 0, WIDTH, HEIGHT, fill=BG, rx=0),
        svg_text(70, 58, "Starling Benchmark Performance", size=36, weight=750),
        svg_text(70, 94, "Random and scaffold held-out tests · Macro-F1", size=20, fill=MUTED),
        svg_text(1830, 58, "GLM-5.2 + supervised baselines", size=18, weight=700, fill=PURPLE, anchor="end"),
        svg_text(70, 132, "Retrieval: parent-disjoint · MiniMol: train-all · Morgan KNN: train labels only, k=3", size=14, fill=MUTED),
    ]

    legend = (
        ("No retrieval", NEUTRAL),
        ("ChEMBL retrieval", BLIND),
        ("Starling retrieval", VISIBLE),
        ("MiniMol baseline", PURPLE),
        ("Morgan KNN · k=3", GOLD),
    )
    legend_x = 70
    for label, color in legend:
        parts.append(rect(legend_x, 157, 18, 18, fill=color, rx=3))
        parts.append(svg_text(legend_x + 27, 171, label, size=14, weight=600))
        legend_x += 220 if label != "Starling retrieval" else 230

    panel_y = (205, 725, 1245)
    for task, y in zip(TASKS, panel_y, strict=True):
        for column, (split, split_label) in enumerate(SPLITS):
            render_panel(
                parts,
                x=70 + column * 900,
                y=y,
                task=task,
                split=split,
                split_label=split_label,
                results=results,
            )

    parts.extend(
        [
            svg_text(70, 1782, "Pipeline bars use the formal parent-disjoint result for every retrieval condition; no-retrieval is operational because neighbor eligibility is not applicable.", size=13, fill=MUTED),
            svg_text(70, 1812, "Source: frozen Starling benchmark results · Accuracy and positive-class metrics remain available in metrics.tsv.", size=13, fill=MUTED),
            svg_text(1830, 1812, f"Generated {generated}", size=13, fill=MUTED, anchor="end"),
            "</svg>",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def export_png(svg_path: Path, png_path: Path) -> None:
    converter = shutil.which("convert")
    if converter is None:
        raise RuntimeError("ImageMagick 'convert' is required for --png-output")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [converter, "-background", "white", str(svg_path), str(png_path)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--png-output", type=Path)
    args = parser.parse_args()
    render(args.metrics, args.output)
    print(args.output)
    if args.png_output is not None:
        export_png(args.output, args.png_output)
        print(args.png_output)


if __name__ == "__main__":
    main()
