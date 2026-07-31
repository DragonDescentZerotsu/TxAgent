"""Plot paired Morgan- versus MiniMol-retrieval GLM agent results."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .paper_figure_style import (
    BG,
    CARD,
    GOLD,
    GRID,
    INK,
    MUTED,
    PURPLE,
    append_vertical_grid,
    rect,
    run_metric_plot_cli,
    svg_text,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = ROOT / "outputs/paper/minimol_retrieval_agent_results"
DEFAULT_METRICS = DEFAULT_RESULTS_DIR / "condition_results.tsv"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "figures/minimol_vs_morgan_agent_retrieval.svg"

WIDTH = 1900
HEIGHT = 1630
SCALE_MAX = 0.85
SPLITS = (("random", "Random split"), ("scaffold", "Scaffold split"))


@dataclass(frozen=True)
class Task:
    key: str
    title: str
    conditions: tuple[tuple[str, str], ...]


COMMON = (
    ("chembl_direct", "ChEMBL · Direct"),
    ("chembl_full_flat", "ChEMBL · Full / Flat"),
    ("chembl_full_mechanism", "ChEMBL · Full / Mechanism"),
)
TASKS = (
    Task(
        "bbb_martins",
        "BBB penetration",
        COMMON
        + (
            ("starling_direct", "Starling · Direct"),
            ("starling_full_flat", "Starling · Full / Flat"),
            ("starling_full_mechanism", "Starling · Full / Mechanism"),
        ),
    ),
    Task(
        "skin_reaction",
        "Skin reaction",
        COMMON
        + (
            ("starling_direct", "Starling · Direct"),
            ("starling_full_flat", "Starling · Full / Flat"),
            ("starling_full_mechanism", "Starling · Full / Mechanism"),
        ),
    ),
    Task(
        "bioavailability_ma",
        "Oral bioavailability",
        COMMON
        + (
            ("starling_direct_numeric", "Starling · Direct (numeric)"),
            ("starling_direct_full", "Starling · Direct (full)"),
            ("starling_full_flat", "Starling · Full / Flat"),
            ("starling_full_mechanism", "Starling · Full / Mechanism"),
        ),
    ),
)


def read_results(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    results = {
        (row["split"], row["task"], row["condition"]): row
        for row in rows
    }
    expected = {
        (split, task.key, condition)
        for split, _ in SPLITS
        for task in TASKS
        for condition, _ in task.conditions
    }
    missing = sorted(expected - results.keys())
    if missing:
        raise ValueError(f"Missing MiniMol retrieval agent results: {missing}")
    failed = [
        key
        for key in expected
        if int(results[key].get("morgan_n_failed") or 0)
        or int(results[key].get("minimol_n_failed") or 0)
    ]
    if failed:
        raise ValueError(f"Cannot plot conditions with failed runs: {sorted(failed)}")
    return results


def render_panel(
    parts: list[str],
    *,
    x: float,
    y: float,
    split: str,
    split_label: str,
    task: Task,
    results: dict[tuple[str, str, str], dict[str, str]],
) -> None:
    width, height = 860, 400
    parts.append(rect(x, y, width, height, fill=CARD, stroke=GRID))
    first = results[(split, task.key, task.conditions[0][0])]
    parts.append(svg_text(x + 22, y + 34, task.title, size=22, weight=750))
    parts.append(
        svg_text(
            x + width - 22,
            y + 33,
            f"{split_label} · n = {int(first['n_total'])}",
            size=14,
            weight=650,
            fill=MUTED,
            anchor="end",
        )
    )

    plot_left, plot_right = x + 280, x + 735
    plot_top, plot_bottom = y + 62, y + 345
    append_vertical_grid(
        parts,
        plot_left=plot_left,
        plot_right=plot_right,
        plot_top=plot_top,
        plot_bottom=plot_bottom,
        ticks=(0.0, 0.2, 0.4, 0.6, 0.8),
        scale_max=SCALE_MAX,
        label_y=y + 374,
    )

    step = 270 / max(1, len(task.conditions) - 1)
    row_start = y + 82
    for index, (condition, label) in enumerate(task.conditions):
        row_y = row_start + index * step
        row = results[(split, task.key, condition)]
        morgan = float(row["morgan_macro_f1"])
        minimol = float(row["minimol_macro_f1"])
        parts.append(svg_text(x + 22, row_y + 5, label, size=13, weight=550))
        for value, offset, color in (
            (morgan, -8, GOLD),
            (minimol, 5, PURPLE),
        ):
            bar_width = value / SCALE_MAX * (plot_right - plot_left)
            parts.append(rect(plot_left, row_y + offset, bar_width, 10, fill=color, rx=2))
        delta = minimol - morgan
        parts.append(
            svg_text(
                x + width - 18,
                row_y + 5,
                f"{delta:+.3f}",
                size=12,
                weight=700,
                fill=PURPLE if delta >= 0 else GOLD,
                anchor="end",
            )
        )
    parts.append(
        svg_text(
            (plot_left + plot_right) / 2,
            y + 392,
            "Macro-F1",
            size=12,
            weight=650,
            fill=MUTED,
            anchor="middle",
        )
    )


def render(metrics_path: Path, output: Path) -> None:
    results = read_results(metrics_path)
    generated = date.today().isoformat()
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="chart-title chart-desc">',
        '<title id="chart-title">Morgan versus MiniMol retrieval in the Starling GLM agent</title>',
        '<desc id="chart-desc">Paired horizontal bars compare parent-disjoint Morgan Tanimoto and MiniMol embedding cosine neighbor retrieval across 38 task, split, and evidence conditions.</desc>',
        f"<metadata>Source: {metrics_path}; generated {generated}.</metadata>",
        rect(0, 0, WIDTH, HEIGHT, fill=BG, rx=0),
        svg_text(70, 58, "Retrieval Feature Ablation in the Starling GLM Agent", size=34, weight=750),
        svg_text(70, 94, "Morgan/Tanimoto vs MiniMol/cosine · Parent-disjoint neighbors · Macro-F1", size=19, fill=MUTED),
        rect(70, 123, 20, 12, fill=GOLD, rx=2),
        svg_text(100, 135, "Morgan fingerprint", size=14, weight=650),
        rect(270, 123, 20, 12, fill=PURPLE, rx=2),
        svg_text(300, 135, "MiniMol embedding", size=14, weight=650),
        svg_text(1830, 135, "Right label: MiniMol − Morgan", size=13, fill=MUTED, anchor="end"),
    ]
    for task_index, task in enumerate(TASKS):
        y = 170 + task_index * 435
        for column, (split, split_label) in enumerate(SPLITS):
            render_panel(
                parts,
                x=70 + column * 900,
                y=y,
                split=split,
                split_label=split_label,
                task=task,
                results=results,
            )
    parts.extend(
        [
            svg_text(70, 1515, "All paired bars use identical Starling queries, evidence sources, top-k, similarity threshold, GLM-5.2 settings, and parent-disjoint policy.", size=13, fill=MUTED),
            svg_text(70, 1543, "Only neighbor ranking changes: Morgan/Tanimoto is replaced by L2-normalized MiniMol embedding cosine.", size=13, fill=MUTED),
            svg_text(1830, 1577, f"Generated {generated}", size=13, fill=MUTED, anchor="end"),
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
