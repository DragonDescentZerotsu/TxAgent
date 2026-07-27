"""Render the grouped-bar overview of paper retrieval claims.

The chart compares Identity-Blind, Deployment-Visible, and the paired
Parent-disjoint retrieval-policy ablation.  It reads the frozen
machine-readable summaries. SVG generation requires only the Python standard
library; optional PNG export uses ImageMagick.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from .paper_figure_style import (
    BG,
    BLIND,
    CARD,
    GOLD,
    GRID,
    MUTED,
    NEGATIVE,
    NEUTRAL,
    PARENT,
    POSITIVE,
    PURPLE,
    VISIBLE,
    multiline,
    rect,
    svg_text,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANALYSIS_DIR = ROOT / "outputs/paper/molecular_evidence_agent/analysis"
DEFAULT_OUTPUT = DEFAULT_ANALYSIS_DIR / "figures/retrieval_claims_overview.svg"

WIDTH = 1800
HEIGHT = 2040


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
            Condition("starling_full_flat", "Starling · Full / Flat"),
            Condition("starling_full_mechanism", "Starling · Full / Mechanism"),
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
            Condition("starling_direct", "Starling · Direct"),
            Condition("starling_full_flat", "Starling · Full / Flat"),
            Condition("starling_full_mechanism", "Starling · Full / Mechanism"),
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


# Every paper condition now has an identity-blind visibility control. Keep this
# named set as an explicit schema hook so a future intentionally omitted control
# must be declared here rather than silently disappearing from the chart.
IDENTITY_BLIND_NOT_RUN_CONDITIONS: frozenset[str] = frozenset()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


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
    required_by_regime = {
        "identity_blind": expected - IDENTITY_BLIND_NOT_RUN_CONDITIONS,
        "deployment_visible": expected,
        "parent_disjoint": expected,
    }
    for regime, values in results.items():
        missing = sorted(required_by_regime[regime] - values.keys())
        if missing:
            raise ValueError(f"Missing {regime} results: {missing}")
    return results


def load_task_sizes(analysis_dir: Path) -> dict[str, int]:
    rows = read_tsv(analysis_dir / "experiment_summary.tsv")
    sizes = {
        row["task"]: int(row["n_total"])
        for row in rows
        if row["visibility_mode"] == "identity_blind" and row["experiment"].endswith("__none")
    }
    missing = sorted({task.key for task in TASKS} - sizes.keys())
    if missing:
        raise ValueError(f"Missing task sizes: {missing}")
    return sizes


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
        retrieval_values = [
            results[regime][experiment]
            for condition in task.conditions[1:]
            if (experiment := f"{task.key}__{condition.suffix}") in results[regime]
        ]
        best = max(retrieval_values)
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
            if experiment not in results[regime]:
                parts.append(rect(plot_left, row_y + offset - 4.5, 24, 9, fill="none", stroke=GRID, rx=2))
                parts.append(svg_text(plot_left + 31, row_y + offset + 4, "not run", size=10, weight=600, fill=MUTED))
                continue
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


def render(analysis_dir: Path, output: Path, *, data_split: str = "test") -> None:
    if data_split not in {"test", "valid"}:
        raise ValueError(f"Unsupported data split: {data_split}")
    results = load_results(analysis_dir)
    task_sizes = load_task_sizes(analysis_dir)
    tasks = tuple(replace(task, n=task_sizes[task.key]) for task in TASKS)
    generated = date.today().isoformat()
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="chart-title chart-desc">',
        '<title id="chart-title">Does molecular evidence retrieval improve property classification?</title>',
        '<desc id="chart-desc">Grouped bar charts compare Identity-Blind, Deployment-Visible, and Parent-disjoint retrieval across four molecular property tasks.</desc>',
        f'<metadata>Sources: experiment_summary.tsv and parent_disjoint_ablation/condition_results.tsv; split {data_split}; generated {generated}.</metadata>',
        rect(0, 0, WIDTH, HEIGHT, fill=BG, rx=0),
        svg_text(70, 58, "Does Molecular Evidence Retrieval Improve Property Classification?", size=36, weight=750),
        svg_text(70, 94, "Primary parent-disjoint analog results with operational and identity-blind controls", size=20, fill=MUTED),
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
    setting_card(parts, 635, color=VISIBLE, eyebrow="DEPLOYMENT-VISIBLE · OPERATIONAL REFERENCE", heading="Structure-visible deployment setting", lines=("Structures are visible; the query name is hidden.", "The LLM chooses comparison tools; same-parent records may remain."))
    setting_card(parts, 1200, color=PARENT, eyebrow="PARENT-DISJOINT · PRIMARY ANALOG SETTING", heading="Same Visible setting, analog-only retrieval", lines=("Exclude exact, same-connectivity, and same-parent records.", "Backfill top-k only at similarity ≥ 0.30; no-retrieval is shared."))

    render_panel(parts, tasks[0], 70, 510, results)
    render_panel(parts, tasks[1], 915, 510, results)
    render_panel(parts, tasks[2], 70, 1125, results)
    render_panel(parts, tasks[3], 915, 1125, results)

    parts.append(svg_text(70, 1748, "All displayed retrieval conditions have measured Identity-Blind, Operational, and Parent-disjoint results.", size=13, weight=600, fill=MUTED))
    parts.append(svg_text(70, 1772, "No-retrieval is shared between Operational and Parent-disjoint because neighbor eligibility is not applicable.", size=13, fill=MUTED))
    parts.append(svg_text(70, 1802, "WHAT DO THE CURRENT RESULTS SUPPORT?", size=14, weight=750, fill=PURPLE, spacing=1.2))
    if data_split == "valid":
        parent = results["parent_disjoint"]

        def parent_gain(task: Task) -> float:
            baseline = parent[f"{task.key}__none"]
            return max(parent[f"{task.key}__{condition.suffix}"] for condition in task.conditions[1:]) - baseline

        bbb_gain, skin_gain, clin_gain, bio_gain = (parent_gain(task) for task in tasks)
        bbb_starling_lead = parent["bbb_martins__starling_direct"] - max(
            parent["bbb_martins__chembl_direct"],
            parent["bbb_martins__chembl_full_flat"],
            parent["bbb_martins__chembl_full_mechanism"],
        )
        bio_starling_lead = max(
            parent["bioavailability_ma__starling_direct_numeric"],
            parent["bioavailability_ma__starling_direct_full"],
            parent["bioavailability_ma__starling_full_flat"],
            parent["bioavailability_ma__starling_full_mechanism"],
        ) - max(
            parent["bioavailability_ma__chembl_direct"],
            parent["bioavailability_ma__chembl_full_flat"],
            parent["bioavailability_ma__chembl_full_mechanism"],
        )
        bio_mechanism_delta = (
            parent["bioavailability_ma__starling_full_mechanism"]
            - parent["bioavailability_ma__starling_full_flat"]
        )
        skin_mechanism_delta = parent["skin_reaction__chembl_full_mechanism"] - parent["skin_reaction__chembl_full_flat"]
        bbb_mechanism_delta = parent["bbb_martins__chembl_full_mechanism"] - parent["bbb_martins__chembl_full_flat"]
        clin_mechanism_delta = parent["clintox__chembl_full_mechanism"] - parent["clintox__chembl_full_flat"]
        claim_card(parts, 70, color=POSITIVE, eyebrow="CLAIM 1 · RETRIEVAL HELPS", heading="POSITIVE POINT ESTIMATES", lines=(f"Parent-disjoint gains: {bio_gain:+.3f} Bioavailability, {clin_gain:+.3f} ClinTox;", f"{bbb_gain:+.3f} BBB and {skin_gain:+.3f} Skin reaction."))
        claim_card(parts, 630, color=GOLD, eyebrow="CLAIM 2 · STARLING > CHEMBL", heading="PROMISING, TASK-LIMITED", lines=(f"Parent-disjoint Starling leads best ChEMBL by {bio_starling_lead:+.3f} Bioavailability", f"and {bbb_starling_lead:+.3f} BBB; Starling coverage remains task-limited."))
        claim_card(parts, 1190, color=NEGATIVE, eyebrow="CLAIM 3 · MECHANISM > FLAT", heading="NOT CONSISTENTLY PROVEN", lines=(f"Parent-disjoint Mechanism − Flat: Bio Starling {bio_mechanism_delta:+.3f}, Skin {skin_mechanism_delta:+.3f};", f"BBB {bbb_mechanism_delta:+.3f}, ClinTox {clin_mechanism_delta:+.3f} — strongly task-dependent."))
    else:
        claim_card(parts, 70, color=POSITIVE, eyebrow="CLAIM 1 · RETRIEVAL HELPS", heading="SUPPORTED, TASK-DEPENDENT", lines=("Parent-disjoint retains +0.238 Bioavailability and +0.029 BBB gains;", "Skin remains negative, and ClinTox is metric-sensitive."))
        claim_card(parts, 630, color=GOLD, eyebrow="CLAIM 2 · STARLING > CHEMBL", heading="PROMISING, PARENT-SENSITIVE", lines=("Bioavailability Starling remains strongest after exclusion, but declines;", "BBB Starling and ChEMBL Direct become nearly tied; coverage is incomplete."))
        claim_card(parts, 1190, color=NEGATIVE, eyebrow="CLAIM 3 · MECHANISM > FLAT", heading="NOT CONSISTENTLY PROVEN", lines=("Parent-disjoint Bioavailability Starling favors Mechanism by +0.023;", "ClinTox moves the other way, and differences remain task-dependent."))

    n_parent_conditions = sum(
        f"{task.key}__{condition.suffix}" in results["parent_disjoint"]
        for task in tasks
        for condition in task.conditions
        if condition.suffix != "none"
    )
    parts.extend(
        [
            svg_text(70, 2012, f"Source: frozen GLM-5.2 full-run artifacts · Values are {data_split} macro-F1 · Parent-disjoint covers {n_parent_conditions} retrieval conditions", size=13, fill=MUTED),
            svg_text(1730, 2012, f"Generated {generated}", size=13, fill=MUTED, anchor="end"),
            "</svg>",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def export_png(svg_path: Path, png_path: Path) -> None:
    """Export the canonical SVG through ImageMagick when a PNG is requested."""
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
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--png-output", type=Path)
    parser.add_argument("--data-split", choices=("test", "valid"), default="test")
    args = parser.parse_args()
    render(args.analysis_dir, args.output, data_split=args.data_split)
    print(args.output)
    if args.png_output is not None:
        export_png(args.output, args.png_output)
        print(args.png_output)


if __name__ == "__main__":
    main()
