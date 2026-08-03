"""Analyze record-majority label thresholds for frozen Starling parents."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from tools.chembl_tool.paper_experiments.analyze_starling_parent_provenance import (
    BENCHMARK_ROOT,
    PMID_BINS,
    RECORD_BINS,
    TASKS,
)
from tools.chembl_tool.paper_experiments.paper_figure_style import (
    BG,
    BLIND,
    GRID,
    INK,
    MUTED,
    VISIBLE,
)


DEFAULT_PARENT_STATS = Path("outputs/paper/starling_parent_provenance/parent_stats.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/paper/starling_majority_thresholds")
THRESHOLDS = (50, 60, 70, 80, 90)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    parent_rows = _load_parent_rows(args.parent_stats)
    decisions = _make_decisions(parent_rows)
    summary_rows = _summarize(decisions)
    histogram_rows = _histogram_rows(decisions)

    _write_table(args.output_dir / "parent_threshold_decisions.csv", decisions)
    _write_table(args.output_dir / "summary.tsv", summary_rows, delimiter="\t")
    _write_table(args.output_dir / "histogram_bins.tsv", histogram_rows, delimiter="\t")

    figure_paths = {
        "yield_svg": figures_dir / "parent_yield_by_threshold.svg",
        "yield_png": figures_dir / "parent_yield_by_threshold_highres.png",
        "record_svg": figures_dir / "record_distribution_by_threshold.svg",
        "record_png": figures_dir / "record_distribution_by_threshold_highres.png",
        "pmid_svg": figures_dir / "unique_pmid_distribution_by_threshold.svg",
        "pmid_png": figures_dir / "unique_pmid_distribution_by_threshold_highres.png",
    }
    _plot_yield(summary_rows, figure_paths["yield_svg"], figure_paths["yield_png"])
    _plot_distribution_grid(
        decisions,
        field="source_record_count",
        bins=RECORD_BINS,
        title="Accepted Starling records per parent across majority thresholds",
        subtitle=(
            "Each panel compares kept and rejected parents within one task/threshold; "
            "bars are percentages within decision state"
        ),
        x_label="Accepted source records per parent",
        svg_output=figure_paths["record_svg"],
        png_output=figure_paths["record_png"],
    )
    _plot_distribution_grid(
        decisions,
        field="n_unique_pmids",
        bins=PMID_BINS,
        title="Unique publication sources per parent across majority thresholds",
        subtitle=(
            "Publication sources are unique non-empty PMIDs; bars are percentages "
            "within decision state"
        ),
        x_label="Unique PMIDs per parent",
        svg_output=figure_paths["pmid_svg"],
        png_output=figure_paths["pmid_png"],
    )

    payload = {
        "analysis_version": "starling_record_majority_thresholds.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds_percent": list(THRESHOLDS),
        "decision_rule": {
            "agreement": "max(n_label_0, n_label_1) / (n_label_0 + n_label_1)",
            "keep": "agreement >= threshold and n_label_0 != n_label_1",
            "assigned_label": "argmax(n_label_0, n_label_1)",
            "tie_policy": "exact 50/50 ties are rejected at every threshold",
            "vote_unit": "accepted source record; multiple records from one PMID vote separately",
        },
        "validation": {
            "n_parent_rows": len(parent_rows),
            "n_decision_rows": len(decisions),
            "expected_decision_rows": len(parent_rows) * len(THRESHOLDS),
            "record_count_identity": "passed",
            "parent_key_coverage": "passed",
        },
        "summary": summary_rows,
        "paths": {
            "parent_threshold_decisions": str(
                args.output_dir / "parent_threshold_decisions.csv"
            ),
            "summary_tsv": str(args.output_dir / "summary.tsv"),
            "histogram_bins": str(args.output_dir / "histogram_bins.tsv"),
            **{key: str(value) for key, value in figure_paths.items()},
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        _render_report(summary_rows), encoding="utf-8"
    )
    print(json.dumps(payload["paths"], ensure_ascii=False, indent=2), flush=True)
    return 0


def _load_parent_rows(parent_stats: Path) -> list[dict[str, Any]]:
    if not parent_stats.exists():
        raise FileNotFoundError(parent_stats)
    with parent_stats.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_task_key = {
        (row["task"], row["molecule_identity_key"]): row for row in rows
    }
    if len(by_task_key) != len(rows):
        raise AssertionError("parent_stats contains duplicate task/parent keys")

    for task in TASKS:
        task_keys: set[str] = set()
        for filename in ("molecule_labels.jsonl", "conflicting_molecules.jsonl"):
            path = BENCHMARK_ROOT / task.output_name / filename
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    artifact = json.loads(line)
                    key = artifact["molecule_identity_key"]
                    task_keys.add(key)
                    parent = by_task_key[(task.output_name, key)]
                    n0 = int(artifact["label_counts"].get("0", 0))
                    n1 = int(artifact["label_counts"].get("1", 0))
                    if n0 + n1 != int(parent["source_record_count"]):
                        raise AssertionError(
                            f"{task.output_name}/{key}: label counts do not match record count"
                        )
                    parent["n_label_0"] = n0
                    parent["n_label_1"] = n1
        expected = {
            row["molecule_identity_key"]
            for row in rows
            if row["task"] == task.output_name
        }
        if task_keys != expected:
            raise AssertionError(f"{task.output_name}: parent key coverage mismatch")

    integer_fields = (
        "source_record_count",
        "records_with_pmid",
        "records_without_pmid",
        "n_unique_pmids",
        "n_unique_source_ids",
        "n_label_0",
        "n_label_1",
    )
    for row in rows:
        for field in integer_fields:
            row[field] = int(row[field])
    return rows


def _make_decisions(parent_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for parent in parent_rows:
        n0 = parent["n_label_0"]
        n1 = parent["n_label_1"]
        total = n0 + n1
        majority_count = max(n0, n1)
        minority_count = min(n0, n1)
        agreement = majority_count / total
        tie = n0 == n1
        majority_label = "" if tie else int(n1 > n0)
        for threshold in THRESHOLDS:
            kept = not tie and agreement >= threshold / 100.0
            if kept:
                reason = "kept_record_majority"
            elif tie:
                reason = "rejected_exact_tie"
            else:
                reason = "rejected_below_threshold"
            output.append(
                {
                    "task": parent["task"],
                    "threshold_percent": threshold,
                    "decision": "kept" if kept else "rejected",
                    "decision_reason": reason,
                    "assigned_label": majority_label if kept else "",
                    "molecule_identity_key": parent["molecule_identity_key"],
                    "strict_status": parent["status"],
                    "n_label_0": n0,
                    "n_label_1": n1,
                    "source_record_count": total,
                    "majority_record_count": majority_count,
                    "minority_record_count": minority_count,
                    "agreement_fraction": agreement,
                    "is_exact_tie": tie,
                    "n_unique_pmids": parent["n_unique_pmids"],
                    "pmids": parent["pmids"],
                }
            )
    return output


def _summarize(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for task in TASKS:
        task_rows = [row for row in decisions if row["task"] == task.output_name]
        strict_kept = sum(
            row["threshold_percent"] == THRESHOLDS[0]
            and row["strict_status"] == "retained"
            for row in task_rows
        )
        strict_conflicts = sum(
            row["threshold_percent"] == THRESHOLDS[0]
            and row["strict_status"] == "conflict"
            for row in task_rows
        )
        for threshold in THRESHOLDS:
            selected = [row for row in task_rows if row["threshold_percent"] == threshold]
            kept = [row for row in selected if row["decision"] == "kept"]
            rejected = [row for row in selected if row["decision"] == "rejected"]
            kept_records = sum(row["source_record_count"] for row in kept)
            kept_minority = sum(row["minority_record_count"] for row in kept)
            recovered = sum(row["strict_status"] == "conflict" for row in kept)
            kept_pmids = {
                pmid
                for row in kept
                for pmid in str(row["pmids"]).split("|")
                if pmid
            }
            rejected_pmids = {
                pmid
                for row in rejected
                for pmid in str(row["pmids"]).split("|")
                if pmid
            }
            output.append(
                {
                    "task": task.output_name,
                    "threshold_percent": threshold,
                    "n_parent_candidates": len(selected),
                    "n_kept": len(kept),
                    "n_rejected": len(rejected),
                    "keep_rate_pct": _pct(len(kept), len(selected)),
                    "n_current_strict_kept": strict_kept,
                    "n_current_conflict_parents": strict_conflicts,
                    "n_recovered_vs_strict": recovered,
                    "recovered_current_conflicts_pct": _pct(recovered, strict_conflicts),
                    "n_kept_label_0": sum(row["assigned_label"] == 0 for row in kept),
                    "n_kept_label_1": sum(row["assigned_label"] == 1 for row in kept),
                    "n_rejected_exact_ties": sum(row["is_exact_tie"] for row in rejected),
                    "n_kept_source_records": kept_records,
                    "n_rejected_source_records": sum(
                        row["source_record_count"] for row in rejected
                    ),
                    "n_minority_records_inside_kept": kept_minority,
                    "minority_record_share_inside_kept_pct": _pct(
                        kept_minority, kept_records
                    ),
                    "kept_records_per_parent_median": _median(
                        [row["source_record_count"] for row in kept]
                    ),
                    "rejected_records_per_parent_median": _median(
                        [row["source_record_count"] for row in rejected]
                    ),
                    "kept_unique_pmids_per_parent_median": _median(
                        [row["n_unique_pmids"] for row in kept]
                    ),
                    "rejected_unique_pmids_per_parent_median": _median(
                        [row["n_unique_pmids"] for row in rejected]
                    ),
                    "n_distinct_pmids_kept": len(kept_pmids),
                    "n_distinct_pmids_rejected": len(rejected_pmids),
                }
            )
    return output


def _histogram_rows(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for task in TASKS:
        for threshold in THRESHOLDS:
            for decision in ("kept", "rejected"):
                selected = [
                    row
                    for row in decisions
                    if row["task"] == task.output_name
                    and row["threshold_percent"] == threshold
                    and row["decision"] == decision
                ]
                for metric, field, bins in (
                    ("source_record_count", "source_record_count", RECORD_BINS),
                    ("unique_pmid_count", "n_unique_pmids", PMID_BINS),
                ):
                    counts = _bin_counts([row[field] for row in selected], bins)
                    for (label, lower, upper), count in zip(bins, counts, strict=True):
                        output.append(
                            {
                                "task": task.output_name,
                                "threshold_percent": threshold,
                                "decision": decision,
                                "metric": metric,
                                "bin": label,
                                "bin_lower": lower,
                                "bin_upper": "" if upper is None else upper,
                                "n_parents": count,
                                "parent_share_pct": _pct(count, len(selected)),
                                "decision_group_n": len(selected),
                            }
                        )
    return output


def _plot_yield(
    summary_rows: list[dict[str, Any]], svg_output: Path, png_output: Path
) -> None:
    _set_plot_style()
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 10.2), sharex=True)
    fig.patch.set_facecolor(BG)
    x = np.arange(len(THRESHOLDS))
    width = 0.36
    for axis, task in zip(axes, TASKS, strict=True):
        rows = [row for row in summary_rows if row["task"] == task.output_name]
        kept = np.asarray([row["n_kept"] for row in rows])
        rejected = np.asarray([row["n_rejected"] for row in rows])
        axis.set_facecolor("white")
        kept_bars = axis.bar(
            x - width / 2,
            kept,
            width,
            color=BLIND,
            edgecolor=BLIND,
            label="Kept",
            zorder=3,
        )
        rejected_bars = axis.bar(
            x + width / 2,
            rejected,
            width,
            color="white",
            edgecolor=VISIBLE,
            linewidth=1.5,
            label="Rejected",
            zorder=3,
        )
        for bars in (kept_bars, rejected_bars):
            for bar in bars:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(kept.max(), rejected.max()) * 0.018,
                    f"{int(bar.get_height()):,}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=INK,
                )
        axis.set_title(task.display_name, loc="left", fontsize=12, fontweight="bold")
        axis.set_ylabel("Parent count")
        axis.set_ylim(0, max(kept.max(), rejected.max()) * 1.15)
        axis.set_xticks(x, [f"{value}%" for value in THRESHOLDS])
        axis.tick_params(axis="x", labelbottom=True)
        axis.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Minimum record agreement threshold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.96, 0.95), frameon=False, ncol=2)
    fig.suptitle(
        "Parent yield under record-majority thresholds",
        x=0.08,
        y=0.985,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.955,
        "Agreement = majority-label records / all accepted binary records; exact ties are rejected",
        color=MUTED,
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.08, hspace=0.34)
    _save_figure(fig, svg_output, png_output)


def _plot_distribution_grid(
    decisions: list[dict[str, Any]],
    *,
    field: str,
    bins: tuple[tuple[str, int, int | None], ...],
    title: str,
    subtitle: str,
    x_label: str,
    svg_output: Path,
    png_output: Path,
) -> None:
    _set_plot_style()
    fig, axes = plt.subplots(
        len(TASKS),
        len(THRESHOLDS),
        figsize=(22, 12.5),
        sharex=True,
        sharey=True,
    )
    fig.patch.set_facecolor(BG)
    x = np.arange(len(bins))
    width = 0.38
    for row_index, task in enumerate(TASKS):
        for column_index, threshold in enumerate(THRESHOLDS):
            axis = axes[row_index, column_index]
            axis.set_facecolor("white")
            group_sizes: dict[str, int] = {}
            for state_index, decision in enumerate(("kept", "rejected")):
                selected = [
                    row[field]
                    for row in decisions
                    if row["task"] == task.output_name
                    and row["threshold_percent"] == threshold
                    and row["decision"] == decision
                ]
                group_sizes[decision] = len(selected)
                percentages = (
                    np.asarray(_bin_counts(selected, bins), dtype=float) / len(selected) * 100
                )
                positions = x + (state_index - 0.5) * width
                bars = axis.bar(
                    positions,
                    percentages,
                    width,
                    color=BLIND if decision == "kept" else "white",
                    edgecolor=BLIND if decision == "kept" else VISIBLE,
                    linewidth=0.7 if decision == "kept" else 1.2,
                    zorder=3,
                    clip_on=False,
                    label="Kept" if decision == "kept" else "Rejected",
                )
                for bar, value in zip(bars, percentages, strict=True):
                    if value >= 20:
                        axis.text(
                            bar.get_x() + bar.get_width() / 2,
                            value + 1.3,
                            f"{value:.0f}%",
                            ha="center",
                            va="bottom",
                            fontsize=6.5,
                            color=INK,
                        )
            axis.set_ylim(0, 105)
            axis.set_xticks(x, [item[0] for item in bins], rotation=45, ha="right")
            axis.tick_params(axis="x", labelbottom=True, labelsize=7)
            axis.tick_params(axis="y", labelsize=8)
            axis.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
            axis.spines[["top", "right"]].set_visible(False)
            if row_index == 0:
                axis.set_title(f"Threshold {threshold}%", fontsize=11, fontweight="bold")
            if column_index == 0:
                axis.set_ylabel(f"{task.display_name}\nParents within state (%)", fontsize=9)
            axis.text(
                0.98,
                0.94,
                f"keep {group_sizes['kept']:,}\nreject {group_sizes['rejected']:,}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                color=MUTED,
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.965, 0.965), frameon=False, ncol=2)
    fig.suptitle(title, x=0.055, y=0.992, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.055, 0.965, subtitle, color=MUTED, fontsize=10.5)
    fig.text(0.5, 0.015, x_label, ha="center", fontsize=11)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.92, bottom=0.075, wspace=0.13, hspace=0.40)
    _save_figure(fig, svg_output, png_output)


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "svg.fonttype": "path",
        }
    )


def _save_figure(fig: Any, svg_output: Path, png_output: Path) -> None:
    svg_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_output, format="svg", facecolor=fig.get_facecolor())
    fig.savefig(png_output, format="png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def _bin_counts(
    values: list[int], bins: tuple[tuple[str, int, int | None], ...]
) -> list[int]:
    return [
        sum(value >= lower and (upper is None or value <= upper) for value in values)
        for _, lower, upper in bins
    ]


def _median(values: list[int]) -> float:
    return float(np.median(np.asarray(values, dtype=int))) if values else 0.0


def _pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def _write_table(
    path: Path, rows: list[dict[str, Any]], *, delimiter: str = ","
) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def _render_report(summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Starling record-majority 阈值敏感性分析",
        "",
        "本分析只改变 parent-level label aggregation：每个 parent 先统计 accepted binary records 中",
        "Y=0 与 Y=1 的数量，再令 `agreement=max(n0,n1)/(n0+n1)`。agreement 大于等于阈值且",
        "不是 50/50 平票时保留，标签取多数类；平票在所有阈值下拒绝。",
        "",
        "> 注意：这是 record-weighted majority。同一 PMID 下的多条 extraction 会分别计票，",
        "> 因而它不是独立论文等权投票。",
        "",
        "## Parent yield",
        "",
        "| task | threshold | keep | reject | keep rate | 比当前 strict 新增 | 恢复原 conflict | kept Y=0 / Y=1 | kept 内 minority record 占比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {task} | {threshold_percent}% | {n_kept:,} | {n_rejected:,} | "
            "{keep_rate_pct:.1f}% | +{n_recovered_vs_strict:,} | "
            "{recovered_current_conflicts_pct:.1f}% | {n_kept_label_0:,} / "
            "{n_kept_label_1:,} | {minority_record_share_inside_kept_pct:.2f}% |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## 产物",
            "",
            "```text",
            "parent_threshold_decisions.csv",
            "summary.tsv",
            "summary.json",
            "histogram_bins.tsv",
            "figures/parent_yield_by_threshold.svg",
            "figures/parent_yield_by_threshold_highres.png",
            "figures/record_distribution_by_threshold.svg",
            "figures/record_distribution_by_threshold_highres.png",
            "figures/unique_pmid_distribution_by_threshold.svg",
            "figures/unique_pmid_distribution_by_threshold_highres.png",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-stats", type=Path, default=DEFAULT_PARENT_STATS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
