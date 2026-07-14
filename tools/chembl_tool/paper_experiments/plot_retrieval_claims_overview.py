"""Render the grouped-bar overview of paper retrieval claims.

The chart compares Identity-Blind, Deployment-Visible, and the paired
Parent-disjoint retrieval-policy ablation.  It reads the frozen
machine-readable summaries and requires only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANALYSIS_DIR = ROOT / "outputs/paper/molecular_evidence_agent/analysis"
DEFAULT_OUTPUT = DEFAULT_ANALYSIS_DIR / "figures/retrieval_claims_overview.svg"

WIDTH = 1800
HEIGHT = 2040
FONT = "Inter, DejaVu Sans, Arial, sans-serif"

BG = "#F7F8FA"
CARD = "#FFFFFF"
INK = "#172033"
MUTED = "#5E6878"
GRID = "#D9DEE7"
BLIND = "#2F6FB0"
VISIBLE = "#D95F3D"
PARENT = "#7A8338"
NEUTRAL = "#8993A2"
PURPLE = "#6557A4"
GOLD = "#A66B12"
POSITIVE = "#177A58"
NEGATIVE = "#B54848"


@dataclass(frozen=True)
class Condition:
    suffix: str
    label: str


@dataclass(frozen=True)
class Task:
    key: str
    title: str
    n: int
    conditions: tuple[Condition, ...]


TASKS = (
    Task(
        "bbb_martins",
        "BBB penetration",
        392,
        (
            Condition("none", "No retrieval"),
            Condition("chembl_direct", "ChEMBL · Direct"),
            Condition("chembl_full_flat", "ChEMBL · Full / Flat"),
            Condition("chembl_full_mechanism", "ChEMBL · Full / Mechanism"),
            Condition("starling_direct", "Starling · Direct"),
        ),
    ),
    Task(
        "skin_reaction",
        "Skin reaction",
        82,
        (
            Condition("none", "No retrieval"),
            Condition("chembl_direct", "ChEMBL · Direct"),
            Condition("chembl_full_flat", "ChEMBL · Full / Flat"),
            Condition("chembl_full_mechanism", "ChEMBL · Full / Mechanism"),
        ),
    ),
    Task(
        "clintox",
        "Clinical toxicity",
        286,
        (
            Condition("none", "No retrieval"),
            Condition("chembl_direct", "ChEMBL · Direct"),
            Condition("chembl_full_flat", "ChEMBL · Full / Flat"),
            Condition("chembl_full_mechanism", "ChEMBL · Full / Mechanism"),
        ),
    ),
    Task(
        "bioavailability_ma",
        "Oral bioavailability",
        128,
        (
            Condition("none", "No retrieval"),
            Condition("chembl_direct", "ChEMBL · Direct"),
            Condition("starling_direct_numeric", "Starling · Direct (numeric)"),
            Condition("starling_direct_full", "Starling · Direct (full)"),
            Condition("chembl_full_flat", "ChEMBL · Full / Flat"),
            Condition("chembl_full_mechanism", "ChEMBL · Full / Mechanism"),
            Condition("starling_full_flat", "Starling · Full / Flat"),
            Condition("starling_full_mechanism", "Starling · Full / Mechanism"),
        ),
    ),
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def svg_text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 16,
    weight: int = 400,
    fill: str = INK,
    anchor: str = "start",
    spacing: float | None = None,
) -> str:
    attrs = [
        f'x="{x:g}"',
        f'y="{y:g}"',
        f'font-family="{FONT}"',
        f'font-size="{size}"',
        f'font-weight="{weight}"',
        f'fill="{fill}"',
        f'text-anchor="{anchor}"',
    ]
    if spacing is not None:
        attrs.append(f'letter-spacing="{spacing:g}"')
    return f"<text {' '.join(attrs)}>{escape(value)}</text>"


def multiline(
    x: float,
    y: float,
    lines: Iterable[str],
    *,
    size: int = 14,
    weight: int = 400,
    fill: str = MUTED,
    line_height: int = 20,
) -> str:
    spans = []
    for index, line in enumerate(lines):
        spans.append(f'<tspan x="{x:g}" dy="{0 if index == 0 else line_height:g}">{escape(line)}</tspan>')
    return (
        f'<text x="{x:g}" y="{y:g}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="start">'
        + "".join(spans)
        + "</text>"
    )


def rect(x: float, y: float, width: float, height: float, *, fill: str, stroke: str = "none", rx: int = 8) -> str:
    return (
        f'<rect x="{x:g}" y="{y:g}" width="{width:g}" height="{height:g}" '
        f'rx="{rx}" fill="{fill}" stroke="{stroke}"/>'
    )


def load_results(analysis_dir: Path) -> dict[str, dict[str, float]]:
    experiments = read_tsv(analysis_dir / "experiment_summary.tsv")
    parent_rows = read_tsv(analysis_dir / "parent_disjoint_ablation/condition_results.tsv")
    results: dict[str, dict[str, float]] = {
        "identity_blind": {},
        "deployment_visible": {},
        "parent_disjoint": {},
    }
    for row in experiments:
        experiment = row["experiment"]
        if row["visibility_mode"] == "identity_blind":
            results["identity_blind"][experiment] = float(row["macro_f1"])
        elif row["visibility_mode"] == "deployment_visible":
            prefix = "deployment_visible__"
            if experiment.startswith(prefix):
                experiment = experiment[len(prefix) :]
            results["deployment_visible"][experiment] = float(row["macro_f1"])
    for row in parent_rows:
        results["parent_disjoint"][row["experiment"]] = float(row["parent_disjoint_macro_f1"])

    # Parent-disjoint changes neighbor eligibility only.  The query-only
    # condition therefore reuses the identical Deployment-Visible baseline.
    for task in TASKS:
        experiment = f"{task.key}__none"
        results["parent_disjoint"][experiment] = results["deployment_visible"][experiment]

    expected = {f"{task.key}__{condition.suffix}" for task in TASKS for condition in task.conditions}
    for regime, values in results.items():
        missing = sorted(expected - values.keys())
        if missing:
            raise ValueError(f"Missing {regime} results: {missing}")
    return results


def strategy_card(
    parts: list[str],
    x: float,
    *,
    color: str,
    eyebrow: str,
    heading: str,
    bullets: tuple[str, str],
) -> None:
    width, y, height = 401.5, 122, 150
    parts.append(rect(x, y, width, height, fill=CARD, stroke=GRID))
    parts.append(rect(x, y, width, 6, fill=color, rx=3))
    parts.append(svg_text(x + 20, y + 34, eyebrow, size=14, weight=750, fill=color, spacing=1.0))
    parts.append(svg_text(x + 20, y + 64, heading, size=21, weight=700))
    for index, bullet in enumerate(bullets):
        bullet_y = y + 91 + index * 25
        parts.append(f'<circle cx="{x + 24:g}" cy="{bullet_y:g}" r="3" fill="{color}"/>')
        parts.append(svg_text(x + 36, bullet_y + 6, bullet, size=14, fill=MUTED))


def setting_card(
    parts: list[str],
    x: float,
    *,
    color: str,
    eyebrow: str,
    heading: str,
    lines: tuple[str, str],
) -> None:
    y, width, height = 365, 530, 116
    parts.append(rect(x, y, width, height, fill=CARD, stroke=GRID))
    parts.append(rect(x, y, 7, height, fill=color, rx=3))
    parts.append(svg_text(x + 20, y + 27, eyebrow, size=12, weight=750, fill=color, spacing=0.9))
    parts.append(svg_text(x + 20, y + 54, heading, size=18, weight=700))
    parts.append(multiline(x + 20, y + 78, lines, size=13, line_height=18))


def signed_gain(value: float) -> tuple[str, str]:
    return f"{value:+.3f}", POSITIVE if value >= 0 else NEGATIVE


def render_panel(
    parts: list[str],
    task: Task,
    x: float,
    y: float,
    results: dict[str, dict[str, float]],
) -> None:
    width, height = 815, 590
    parts.append(rect(x, y, width, height, fill=CARD, stroke=GRID))
    parts.append(svg_text(x + 22, y + 34, task.title, size=22, weight=750))
    parts.append(svg_text(x + width - 22, y + 33, f"n = {task.n}", size=14, weight=600, fill=MUTED, anchor="end"))

    baseline_key = f"{task.key}__none"
    gains = {}
    for regime in results:
        baseline = results[regime][baseline_key]
        best = max(results[regime][f"{task.key}__{condition.suffix}"] for condition in task.conditions[1:])
        gains[regime] = best - baseline
    parts.append(svg_text(x + 22, y + 62, "Best retrieval gain vs no retrieval:", size=13, weight=600, fill=MUTED))
    gain_positions = (("identity_blind", "Blind", x + 314), ("deployment_visible", "Visible", x + 445), ("parent_disjoint", "Parent", x + 590))
    for regime, label, gain_x in gain_positions:
        gain_text, gain_color = signed_gain(gains[regime])
        parts.append(svg_text(gain_x, y + 62, f"{label} {gain_text}", size=13, weight=750, fill=gain_color))

    plot_left, plot_right = x + 235, x + 724
    plot_top, plot_bottom = y + 80, y + 548
    scale_max = 0.85
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8):
        tick_x = plot_left + tick / scale_max * (plot_right - plot_left)
        parts.append(f'<line x1="{tick_x:.1f}" y1="{plot_top}" x2="{tick_x:.1f}" y2="{plot_bottom}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(svg_text(tick_x, y + 570, f"{tick:.1f}", size=12, fill=MUTED, anchor="middle"))

    count = len(task.conditions)
    step = min(58.0, 420.0 / max(1, count - 1)) if count > 1 else 0.0
    center_span = step * (count - 1)
    row_start = y + 112 + (420.0 - center_span) / 2
    regimes = (
        ("identity_blind", BLIND, -13),
        ("deployment_visible", VISIBLE, 0),
        ("parent_disjoint", PARENT, 13),
    )
    for index, condition in enumerate(task.conditions):
        row_y = row_start + index * step
        experiment = f"{task.key}__{condition.suffix}"
        if condition.suffix == "none":
            parts.append(rect(x + 12, row_y - 29, width - 24, 58, fill="#F3F5F8", rx=4))
        parts.append(svg_text(x + 22, row_y + 5, condition.label, size=13, weight=600 if condition.suffix == "none" else 500))
        for regime, color, offset in regimes:
            value = results[regime][experiment]
            bar_width = value / scale_max * (plot_right - plot_left)
            bar_y = row_y + offset - 4.5
            if regime == "parent_disjoint" and condition.suffix == "none":
                # The olive outline communicates that this is a shared Visible
                # baseline rather than an independently rerun condition.
                parts.append(rect(plot_left, bar_y, bar_width, 9, fill="none", stroke=PARENT, rx=2))
                label = f"{value:.3f} shared"
            else:
                parts.append(rect(plot_left, bar_y, bar_width, 9, fill=color, rx=2))
                label = f"{value:.3f}"
            parts.append(svg_text(plot_left + bar_width + 7, row_y + offset + 4, label, size=10, weight=650, fill=color))
    parts.append(svg_text((plot_left + plot_right) / 2, y + 586, "Macro-F1", size=12, weight=600, fill=MUTED, anchor="middle"))


def claim_card(
    parts: list[str],
    x: float,
    *,
    color: str,
    eyebrow: str,
    heading: str,
    lines: tuple[str, str],
) -> None:
    y, width, height = 1820, 540, 145
    parts.append(rect(x, y, width, height, fill=CARD, stroke=GRID))
    parts.append(rect(x, y, 7, height, fill=color, rx=3))
    parts.append(svg_text(x + 22, y + 30, eyebrow, size=13, weight=750, fill=color, spacing=0.7))
    parts.append(svg_text(x + 22, y + 62, heading, size=20, weight=750))
    parts.append(multiline(x + 22, y + 91, lines, size=14, line_height=22))


def render(analysis_dir: Path, output: Path) -> None:
    results = load_results(analysis_dir)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="chart-title chart-desc">',
        '<title id="chart-title">Does molecular evidence retrieval improve property classification?</title>',
        '<desc id="chart-desc">Grouped bar charts compare Identity-Blind, Deployment-Visible, and Parent-disjoint retrieval across four molecular property tasks.</desc>',
        '<metadata>Sources: experiment_summary.tsv and parent_disjoint_ablation/condition_results.tsv; generated 2026-07-13.</metadata>',
        rect(0, 0, WIDTH, HEIGHT, fill=BG, rx=0),
        svg_text(70, 58, "Does Molecular Evidence Retrieval Improve Property Classification?", size=36, weight=750),
        svg_text(70, 94, "Two evaluation regimes plus a paired parent-disjoint retrieval-policy ablation", size=20, fill=MUTED),
        svg_text(1730, 58, "GLM-5.2", size=18, weight=700, fill=PURPLE, anchor="end"),
    ]

    strategy_card(parts, 70, color=NEUTRAL, eyebrow="NO RETRIEVAL", heading="Query-only prior", bullets=("Molecular properties only", "No neighbor evidence"))
    strategy_card(parts, 489.5, color=BLIND, eyebrow="DIRECT", heading="Outcome evidence only", bullets=("Retrieve direct task measurements", "One evidence reasoning branch"))
    strategy_card(parts, 909, color=GOLD, eyebrow="FULL · FLAT", heading="All evidence pooled", bullets=("Retrieve all task-relevant evidence", "One pooled reasoning branch"))
    strategy_card(parts, 1328.5, color=PURPLE, eyebrow="FULL · MECHANISM", heading="All evidence structured", bullets=("Same full evidence rows as Flat", "Mechanism families → parallel reasoning"))

    parts.extend(
        [
            rect(70, 287, 1660, 59, fill="#EEF2F6", stroke=GRID),
            svg_text(92, 313, "Flat vs Mechanism:", size=16, weight=750, fill=PURPLE),
            svg_text(285, 313, "the retrieved evidence rows are held constant; only the reasoning orchestration changes.", size=16, weight=500),
            svg_text(92, 337, "Flat = all evidence → one branch.    Mechanism = evidence grouped by task mechanism → parallel branches → final synthesis.", size=15, fill=MUTED),
        ]
    )

    setting_card(parts, 70, color=BLIND, eyebrow="IDENTITY-BLIND · SUPPLEMENTARY CONTROL", heading="Evidence-only evaluation", lines=("Structures, names, and source IDs are hidden from the LLM.", "The harness supplies fixed property/comparison summaries."))
    setting_card(parts, 635, color=VISIBLE, eyebrow="DEPLOYMENT-VISIBLE · MAIN EXPERIMENT", heading="Structure-visible deployment setting", lines=("Structures are visible; the query name is hidden.", "The LLM chooses comparison tools; same-parent records may remain."))
    setting_card(parts, 1200, color=PARENT, eyebrow="PARENT-DISJOINT · ANALOG-ONLY ABLATION", heading="Same Visible setting, stricter retrieval", lines=("Exclude exact, same-connectivity, and same-parent records.", "Backfill top-k only at similarity ≥ 0.30; no-retrieval is shared."))

    render_panel(parts, TASKS[0], 70, 510, results)
    render_panel(parts, TASKS[1], 915, 510, results)
    render_panel(parts, TASKS[2], 70, 1125, results)
    render_panel(parts, TASKS[3], 915, 1125, results)

    parts.append(svg_text(70, 1802, "WHAT DO THE CURRENT RESULTS SUPPORT?", size=14, weight=750, fill=PURPLE, spacing=1.2))
    claim_card(parts, 70, color=POSITIVE, eyebrow="CLAIM 1 · RETRIEVAL HELPS", heading="SUPPORTED, TASK-DEPENDENT", lines=("Parent-disjoint retains +0.238 Bioavailability and +0.029 BBB gains;", "Skin remains negative, and ClinTox is metric-sensitive."))
    claim_card(parts, 630, color=GOLD, eyebrow="CLAIM 2 · STARLING > CHEMBL", heading="PROMISING, PARENT-SENSITIVE", lines=("Bioavailability Starling remains strongest after exclusion, but declines;", "BBB Starling and ChEMBL Direct become nearly tied; coverage is incomplete."))
    claim_card(parts, 1190, color=NEGATIVE, eyebrow="CLAIM 3 · MECHANISM > FLAT", heading="NOT CONSISTENTLY PROVEN", lines=("Parent-disjoint Bioavailability Starling favors Mechanism by +0.023;", "ClinTox moves the other way, and differences remain task-dependent."))

    parts.extend(
        [
            svg_text(70, 2012, "Source: frozen GLM-5.2 full-run artifacts · Values are test macro-F1 · Parent-disjoint covers 17 retrieval conditions", size=13, fill=MUTED),
            svg_text(1730, 2012, "Generated 2026-07-13", size=13, fill=MUTED, anchor="end"),
            "</svg>",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.analysis_dir, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
