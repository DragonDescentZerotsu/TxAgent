"""Plot reusable assay-count performance and retrieval-volume curves.

The plotter reads experiment artifacts rather than embedding metric values.  It
supports the legacy shared-prefix manifest and the task-specific full-catalog
manifest, reuses historical no-retrieval metrics at x=0, and computes retrieval
volume directly from replay payloads.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class TaskPlotSpec:
    label: str
    color: str
    marker: str
    historical_best_condition: str
    historical_best_label: str


TASK_SPECS = {
    "bbb_martins": TaskPlotSpec("BBB", "#0072B2", "o", "starling_full_flat", "full-flat"),
    "bioavailability_ma": TaskPlotSpec(
        "Bioavailability", "#D55E00", "s", "starling_full_flat", "full-flat"
    ),
    "skin_reaction": TaskPlotSpec("Skin", "#CC79A7", "^", "starling_direct", "direct"),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prefixes_by_task(manifest: dict[str, Any]) -> dict[str, list[int]]:
    tasks = list(manifest["tasks"])
    if manifest.get("prefixes_by_task"):
        return {
            task: [int(value) for value in manifest["prefixes_by_task"][task]]
            for task in tasks
        }
    shared = [int(value) for value in manifest.get("prefixes") or []]
    return {task: list(shared) for task in tasks}


def _historical_batch(group_root: Path, task: str, condition: str) -> Path:
    return group_root / task / f"{task}__{condition}"


def _retrieval_counts(retrieval: dict[str, Any]) -> tuple[int, int, int]:
    neighbors = {
        str(neighbor.get("molecule_chembl_id") or neighbor_index)
        for group in retrieval.get("groups") or []
        for neighbor_index, neighbor in enumerate(group.get("neighbors") or [])
    }
    evidence_rows = [
        row
        for group in retrieval.get("groups") or []
        for neighbor in group.get("neighbors") or []
        for row in neighbor.get("evidence_rows") or []
    ]
    source_records = sum(
        max(0, int(row.get("source_record_count") or 0)) for row in evidence_rows
    )
    return len(neighbors), len(evidence_rows), source_records


def _batch_retrieval_means(batch: Path) -> dict[str, float | int]:
    manifest = _load_json(batch / "manifest.json")
    indices = [int(value) for value in manifest.get("indices") or []]
    if not indices:
        indices = list(range(int(manifest["n_items"])))
    values = []
    for query_index in indices:
        run_dir = batch / "runs" / f"{batch.name}_idx{query_index:05d}"
        values.append(_retrieval_counts(_load_json(run_dir / "retrieval.json")))
    if not values:
        raise ValueError(f"Replay batch has no retrieval payloads: {batch}")
    return {
        "n_queries": len(values),
        "mean_unique_molecules": mean(value[0] for value in values),
        "mean_assay_molecule_evidence_rows": mean(value[1] for value in values),
        "mean_source_records_represented": mean(value[2] for value in values),
    }


def collect_curve_data(
    *,
    output_root: Path,
    historical_group_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    experiment = _load_json(output_root / "experiment_manifest.json")
    prefixes_by_task = _prefixes_by_task(experiment)
    performance_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []

    for task in experiment["tasks"]:
        spec = TASK_SPECS.get(
            task,
            TaskPlotSpec(task, "#4C78A8", "o", "starling_full_flat", "historical best"),
        )
        none_batch = _historical_batch(historical_group_root, task, "none")
        none_metrics = _load_json(none_batch / "metrics.json")
        performance_rows.append(
            {
                "task": task,
                "task_label": spec.label,
                "assay_count": 0,
                "condition": "none_reused",
                "macro_f1": float(none_metrics["macro_f1"]),
                "accuracy": float(none_metrics["accuracy"]),
                "n_total": int(none_metrics["n_total"]),
                "n_failed_runs": int(none_metrics["n_failed_runs"]),
                "metrics_path": str(none_batch / "metrics.json"),
            }
        )
        retrieval_rows.append(
            {
                "task": task,
                "task_label": spec.label,
                "assay_count": 0,
                "condition": "none_reused",
                "n_queries": int(none_metrics["n_total"]),
                "mean_unique_molecules": 0.0,
                "mean_assay_molecule_evidence_rows": 0.0,
                "mean_source_records_represented": 0.0,
                "replay_batch": "",
            }
        )

        replay_root = Path(experiment["inputs"][task]["replay_root"])
        for prefix in prefixes_by_task[task]:
            batch = output_root / task / f"assay_flat_top{prefix}"
            metrics_path = batch / "metrics.json"
            replay_batch = replay_root / f"assay_flat_top{prefix}"
            if metrics_path.is_file():
                metrics = _load_json(metrics_path)
                performance_rows.append(
                    {
                        "task": task,
                        "task_label": spec.label,
                        "assay_count": prefix,
                        "condition": "assay_level",
                        "macro_f1": float(metrics["macro_f1"]),
                        "accuracy": float(metrics["accuracy"]),
                        "n_total": int(metrics["n_total"]),
                        "n_failed_runs": int(metrics["n_failed_runs"]),
                        "metrics_path": str(metrics_path),
                    }
                )
            if replay_batch.is_dir():
                retrieval_rows.append(
                    {
                        "task": task,
                        "task_label": spec.label,
                        "assay_count": prefix,
                        "condition": "assay_level",
                        **_batch_retrieval_means(replay_batch),
                        "replay_batch": str(replay_batch),
                    }
                )

    best_rows: list[dict[str, Any]] = []
    for task in experiment["tasks"]:
        spec = TASK_SPECS[task]
        assay_rows = [
            row
            for row in performance_rows
            if row["task"] == task and row["condition"] == "assay_level"
        ]
        if not assay_rows:
            continue
        assay_best = max(assay_rows, key=lambda row: (row["macro_f1"], -row["assay_count"]))
        group_batch = _historical_batch(
            historical_group_root, task, spec.historical_best_condition
        )
        group_metrics = _load_json(group_batch / "metrics.json")
        best_rows.append(
            {
                "task": task,
                "task_label": spec.label,
                "assay_best_count": assay_best["assay_count"],
                "assay_best_macro_f1": assay_best["macro_f1"],
                "group_best_condition": spec.historical_best_label,
                "group_best_macro_f1": float(group_metrics["macro_f1"]),
                "delta_macro_f1": assay_best["macro_f1"] - float(group_metrics["macro_f1"]),
                "delta_percentage_points": 100
                * (assay_best["macro_f1"] - float(group_metrics["macro_f1"])),
                "n_valid": assay_best["n_total"],
                "assay_metrics_path": assay_best["metrics_path"],
                "group_metrics_path": str(group_batch / "metrics.json"),
            }
        )
    return performance_rows, retrieval_rows, {"experiment": experiment, "best": best_rows}


def incomplete_curve_conditions(
    performance_rows: list[dict[str, Any]],
    experiment: dict[str, Any],
) -> list[str]:
    """List absent, failed, or sample-incomplete assay-prefix metrics."""
    prefixes_by_task = _prefixes_by_task(experiment)
    none_n = {
        row["task"]: int(row["n_total"])
        for row in performance_rows
        if row["condition"] == "none_reused"
    }
    observed = {
        (row["task"], int(row["assay_count"])): row
        for row in performance_rows
        if row["condition"] == "assay_level"
    }
    incomplete = []
    for task in experiment["tasks"]:
        for prefix in prefixes_by_task[task]:
            row = observed.get((task, prefix))
            if (
                row is None
                or int(row["n_failed_runs"]) != 0
                or int(row["n_total"]) != none_n[task]
            ):
                incomplete.append(f"{task}__assay_flat_top{prefix}")
    return incomplete


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def collect_relevance_decay_data(
    *,
    ranked_assay_paths: dict[str, Path],
    prefixes_by_task: dict[str, list[int]],
) -> list[dict[str, Any]]:
    """Compute cumulative relevance summaries for frozen assay rankings."""
    rows: list[dict[str, Any]] = []
    for task, prefixes in prefixes_by_task.items():
        path = ranked_assay_paths[task]
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        scores = [float(record["relevance_score"]) for record in records]
        if not scores or any(not 0 <= score <= 100 for score in scores):
            raise ValueError(f"Invalid relevance scores: {path}")
        if any(left < right for left, right in zip(scores, scores[1:])):
            raise ValueError(f"Assay ranking is not score-descending: {path}")
        for prefix in prefixes:
            if prefix > len(scores):
                raise ValueError(
                    f"Prefix {prefix} exceeds {len(scores)} ranked assays: {path}"
                )
            selected = scores[:prefix]
            rows.append(
                {
                    "task": task,
                    "task_label": TASK_SPECS[task].label,
                    "assay_count": prefix,
                    "catalog_size": len(scores),
                    "mean_relevance_score": mean(selected),
                    "boundary_relevance_score": selected[-1],
                    "minimum_selected_score": min(selected),
                    "maximum_selected_score": max(selected),
                    "ranked_assays_path": str(path),
                }
            )
    return rows


def _draw_assay_relevance_decay(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    panel_title: str = "",
) -> None:
    """Draw cumulative assay relevance on an existing axis."""
    tasks = list(dict.fromkeys(row["task"] for row in rows))
    for task in tasks:
        spec = TASK_SPECS[task]
        task_rows = sorted(
            (row for row in rows if row["task"] == task),
            key=lambda row: row["assay_count"],
        )
        ax.plot(
            [row["assay_count"] for row in task_rows],
            [row["mean_relevance_score"] for row in task_rows],
            color=spec.color,
            marker=spec.marker,
            markersize=6.5,
            linewidth=2.2,
            label=spec.label,
        )
        endpoint = task_rows[-1]
        ax.annotate(
            f"{spec.label}: {endpoint['mean_relevance_score']:.1f}",
            (endpoint["assay_count"], endpoint["mean_relevance_score"]),
            xytext=(-5, 8),
            textcoords="offset points",
            ha="right",
            color=spec.color,
            fontsize=8.5,
        )

    ticks = sorted({int(row["assay_count"]) for row in rows})
    tick_labels: list[str] = []
    for index, value in enumerate(ticks):
        label = f"{value:,}"
        if index and value / ticks[index - 1] < 1.6:
            label = "\n" + label
        tick_labels.append(label)
    ax.set_xscale("log", base=4)
    ax.set_xticks(ticks, labels=tick_labels)
    ax.tick_params(axis="x", labelrotation=28)
    ax.set_xlim(min(ticks) / 1.25, max(ticks) * 1.15)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Number of retrieved assays (log scale)")
    ax.set_ylabel("Mean relevance score among retrieved assays (0–100)")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.legend(frameon=False, loc="lower left")
    if panel_title:
        ax.set_title(panel_title, loc="left")


def plot_assay_relevance_decay(
    *,
    rows: list[dict[str, Any]],
    output_svg: Path,
    output_png: Path,
) -> None:
    """Plot cumulative mean relevance as progressively more assays are retrieved."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    _draw_assay_relevance_decay(ax, rows)
    fig.suptitle(
        "Mean relevance of retrieved assays",
        fontsize=15,
        fontweight="bold",
        x=0.08,
        ha="left",
    )
    fig.text(
        0.08,
        0.905,
        "Cumulative mean relevance within each frozen Top-N assay ranking",
        fontsize=9.5,
        color="#555555",
    )
    fig.text(
        0.08,
        0.025,
        "Scores were assigned offline by Codex GPT-5.6 Sol; each curve includes the full assay catalog.",
        fontsize=8.2,
        color="#555555",
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.84, bottom=0.2)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, bbox_inches="tight")
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _current_relevance_rows(tasks: list[str]) -> list[dict[str, Any]]:
    """Load frozen assay rankings using the current task-specific prefix plan."""
    from tools.chembl_tool.paper_experiments.run_assay_retrieval_curve import (
        TASKS,
        prefix_plan,
    )

    return collect_relevance_decay_data(
        ranked_assay_paths={
            task: Path(TASKS[task]["ranked_assays"]) for task in tasks
        },
        prefixes_by_task=prefix_plan(tasks),
    )


def write_analysis(
    *,
    analysis_dir: Path,
    performance_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _write_tsv(analysis_dir / "assay_curve_metrics.tsv", performance_rows)
    _write_tsv(analysis_dir / "assay_retrieval_volume.tsv", retrieval_rows)
    _write_tsv(analysis_dir / "assay_vs_group_best.tsv", summary["best"])
    payload = {
        "comparison_contract": {
            "split": "scaffold-valid",
            "assay_retrieval_contract": summary["experiment"].get(
                "retrieval_contract", ""
            ),
            "assay_reference_pool": summary["experiment"].get("reference_pool", ""),
            "visibility_mode": "identity_blind",
            "neighbor_identity_policy": summary["experiment"].get(
                "neighbor_identity_policy", ""
            ),
            "zero_retrieval": "reused_historical_none",
            "comparison_type": "descriptive_historical_reference_not_endpoint_matched",
        },
        "performance": performance_rows,
        "retrieval_volume": retrieval_rows,
        "best_comparison": summary["best"],
    }
    (analysis_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _set_assay_axis(ax: plt.Axes, ticks: list[int]) -> None:
    max_count = max(ticks)
    displayed = [0]
    value = 5
    while value <= max_count:
        displayed.append(value)
        value *= 4
    ax.set_xscale("symlog", base=4, linthresh=5, linscale=0.8)
    ax.set_xticks(
        displayed,
        labels=["None\n(0)" if value == 0 else f"{value:,}" for value in displayed],
    )
    ax.tick_params(axis="x", labelrotation=25)
    ax.set_xlabel("Number of retrieved assays (symmetric log scale)")


def _best_panel_xlim(best_rows: list[dict[str, Any]]) -> tuple[float, float]:
    """Return rounded, data-driven bounds with room for delta annotations."""
    values = [
        float(row[field])
        for row in best_rows
        for field in ("group_best_macro_f1", "assay_best_macro_f1")
    ]
    if not values:
        return 0.0, 1.0

    value_min = min(values)
    value_max = max(values)
    span = max(value_max - value_min, 0.05)
    lower_padding = max(0.01, span * 0.08)
    upper_padding = max(0.015, span * 0.15)
    tick_step = 0.01
    lower = max(0.0, math.floor((value_min - lower_padding) / tick_step) * tick_step)
    upper = min(1.0, math.ceil((value_max + upper_padding) / tick_step) * tick_step)
    return lower, upper


def plot_assay_retrieval_curves(
    *,
    performance_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    relevance_rows: list[dict[str, Any]],
    output_svg: Path,
    output_png: Path,
) -> None:
    """Render the reusable five-panel assay retrieval figure."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(13.4, 13.0))
    grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 0.95))
    ax_perf = fig.add_subplot(grid[0, 0])
    ax_best = fig.add_subplot(grid[0, 1])
    ax_molecules = fig.add_subplot(grid[1, 0])
    ax_records = fig.add_subplot(grid[1, 1])
    ax_relevance = fig.add_subplot(grid[2, :])
    tasks = list(dict.fromkeys(row["task"] for row in performance_rows))
    performance_ticks = sorted({int(row["assay_count"]) for row in performance_rows})
    assay_ticks = sorted({int(row["assay_count"]) for row in retrieval_rows})

    for task in tasks:
        spec = TASK_SPECS[task]
        perf = sorted(
            (row for row in performance_rows if row["task"] == task),
            key=lambda row: row["assay_count"],
        )
        volume = sorted(
            (row for row in retrieval_rows if row["task"] == task),
            key=lambda row: row["assay_count"],
        )
        style = {
            "color": spec.color,
            "marker": spec.marker,
            "markersize": 6.5,
            "linewidth": 2.0,
            "label": spec.label,
        }
        ax_perf.plot(
            [row["assay_count"] for row in perf],
            [row["macro_f1"] for row in perf],
            **style,
        )
        ax_molecules.plot(
            [row["assay_count"] for row in volume],
            [row["mean_unique_molecules"] for row in volume],
            **style,
        )
        ax_records.plot(
            [row["assay_count"] for row in volume],
            [row["mean_source_records_represented"] for row in volume],
            **style,
        )
        for ax, field in (
            (ax_molecules, "mean_unique_molecules"),
            (ax_records, "mean_source_records_represented"),
        ):
            endpoint = volume[-1]
            ax.annotate(
                f"{spec.label} all\n({endpoint['assay_count']:,})",
                (endpoint["assay_count"], endpoint[field]),
                xytext=(-5, 8),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=7.5,
                color=spec.color,
            )

    for ax, title, ylabel, ticks in (
        (ax_perf, "A. Predictive performance", "Macro-F1", performance_ticks),
        (ax_molecules, "C. Retrieved molecular diversity", "Mean unique molecules per query", assay_ticks),
        (ax_records, "D. Retrieved evidence volume", "Mean source records per query", assay_ticks),
    ):
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
        _set_assay_axis(ax, ticks)
    ax_records.set_yscale("symlog", base=10, linthresh=1, linscale=0.8)
    ax_records.set_ylabel("Mean represented source records per query (symlog)")
    ax_records.set_ylim(bottom=0)
    ax_perf.legend(frameon=False, ncol=3, loc="best")

    best_by_task = {row["task"]: row for row in best_rows}
    best_xlim = _best_panel_xlim(best_rows)
    for y, task in enumerate(tasks):
        if task not in best_by_task:
            continue
        row = best_by_task[task]
        spec = TASK_SPECS[task]
        group_value = row["group_best_macro_f1"]
        assay_value = row["assay_best_macro_f1"]
        ax_best.plot([group_value, assay_value], [y, y], color="#AAAAAA", linewidth=2)
        ax_best.scatter(
            [group_value], [y], s=62, facecolor="white", edgecolor="#555555", linewidth=1.4
        )
        ax_best.scatter(
            [assay_value], [y], s=70, color=spec.color, edgecolor="white", linewidth=0.7
        )
        ax_best.text(
            max(group_value, assay_value) + 0.002,
            y,
            f"{row['delta_percentage_points']:+.1f} pp",
            va="center",
            color=spec.color,
            fontweight="bold",
        )
    ax_best.set_yticks(range(len(tasks)), labels=[TASK_SPECS[task].label for task in tasks])
    ax_best.invert_yaxis()
    ax_best.set_xlabel("Best Macro-F1")
    ax_best.set_title("B. Best assay-level vs historical group-level", loc="left")
    ax_best.grid(axis="x", color="#DDDDDD", linewidth=0.8)
    ax_best.set_xlim(*best_xlim)
    ax_best.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="none", markerfacecolor="white",
                markeredgecolor="#555555", markeredgewidth=1.4, label="Historical group-level best",
            ),
            Line2D(
                [0], [0], marker="o", linestyle="none", markerfacecolor="#555555",
                markeredgecolor="white", label="Current assay-level best",
            ),
        ],
        frameon=False,
        loc="upper left",
    )

    _draw_assay_relevance_decay(
        ax_relevance,
        relevance_rows,
        panel_title="E. Mean relevance of retrieved assays",
    )

    fig.suptitle(
        "Assay-level retrieval scaling on scaffold validation sets",
        fontsize=15,
        fontweight="bold",
        x=0.06,
        ha="left",
    )
    fig.text(
        0.06,
        0.025,
        "No-retrieval and group-level references reuse historical OpenRouter DeepSeek-V4-Flash runs; "
        "assay-level points use PARCC DeepSeek-V4-Flash-0731. This is a descriptive, not endpoint-matched, comparison. "
        "Panel E uses frozen Codex GPT-5.6 Sol assay-relevance scores.",
        fontsize=8.2,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.94,
        bottom=0.09,
        hspace=0.52,
        wspace=0.27,
    )
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, bbox_inches="tight")
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=(
            "outputs/paper/starling_assay_retrieval_curve_v5/"
            "scaffold_valid_direct_only_heldout_filtered_scaffold_disjoint_"
            "epyc_deepseek_v4_flash_0731"
        ),
    )
    parser.add_argument(
        "--historical-group-root",
        default=(
            "outputs/paper/"
            "molecular_evidence_agent_starling_scaffold_current_valid_openrouter_deepseek_v4_flash/"
            "runs_identity_blind_parent_disjoint"
        ),
    )
    parser.add_argument("--analysis-dir", default="")
    parser.add_argument("--output-stem", default="assay_retrieval_scaling")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow a diagnostic partial figure; complete zero-failure curves are required by default.",
    )
    parser.add_argument(
        "--relevance-decay-only",
        action="store_true",
        help="Plot cumulative mean relevance from frozen assay rankings without requiring LLM metrics.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else output_root / "analysis"
    if args.relevance_decay_only:
        tasks = list(TASK_SPECS)
        relevance_rows = _current_relevance_rows(tasks)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        _write_tsv(analysis_dir / "assay_relevance_decay.tsv", relevance_rows)
        figure_dir = analysis_dir / "figures"
        plot_assay_relevance_decay(
            rows=relevance_rows,
            output_svg=figure_dir / "assay_relevance_decay.svg",
            output_png=figure_dir / "assay_relevance_decay.png",
        )
        print(
            json.dumps(
                {
                    "analysis_dir": str(analysis_dir),
                    "n_relevance_rows": len(relevance_rows),
                },
                indent=2,
            )
        )
        return 0
    performance, retrieval, summary = collect_curve_data(
        output_root=output_root,
        historical_group_root=Path(args.historical_group_root),
    )
    incomplete = incomplete_curve_conditions(performance, summary["experiment"])
    if incomplete and not args.allow_incomplete:
        raise SystemExit(
            "Refusing to publish an incomplete assay curve:\n" + "\n".join(incomplete)
        )
    relevance_rows = _current_relevance_rows(list(summary["experiment"]["tasks"]))
    write_analysis(
        analysis_dir=analysis_dir,
        performance_rows=performance,
        retrieval_rows=retrieval,
        summary=summary,
    )
    _write_tsv(analysis_dir / "assay_relevance_decay.tsv", relevance_rows)
    figure_dir = analysis_dir / "figures"
    plot_assay_retrieval_curves(
        performance_rows=performance,
        retrieval_rows=retrieval,
        best_rows=summary["best"],
        relevance_rows=relevance_rows,
        output_svg=figure_dir / f"{args.output_stem}.svg",
        output_png=figure_dir / f"{args.output_stem}.png",
    )
    print(
        json.dumps(
            {
                "analysis_dir": str(analysis_dir),
                "n_performance_rows": len(performance),
                "n_retrieval_rows": len(retrieval),
                "n_relevance_rows": len(relevance_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
