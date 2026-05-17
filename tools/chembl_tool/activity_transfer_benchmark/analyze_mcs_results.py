"""Analyze MCS coverage as an activity-transfer threshold baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_MCS_RESULTS = (
    "outputs/chembl_tool/activity_transfer_benchmark/mcs_runtime/"
    "dynamic_v1_mcs_full_t2_w128_stream/mcs_sample_results.tsv.tmp"
)
DEFAULT_OUT_ROOT = "outputs/chembl_tool/activity_transfer_benchmark/mcs_analysis"
MCS_BUCKETS = [
    ("very_high", 0.95, 1.01),
    ("high", 0.90, 0.95),
    ("moderately_high", 0.80, 0.90),
    ("moderate", 0.65, 0.80),
    ("low", 0.40, 0.65),
    ("very_low", 0.00, 0.40),
]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = args.run_id or datetime.now().strftime("mcs_analysis_%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / run_id
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(Path(args.mcs_results))
    if not rows:
        raise RuntimeError("No usable MCS rows found.")

    metrics = {
        "mean_mcs_coverage": scan_thresholds(rows, "mean_mcs_coverage"),
        "tanimoto": scan_thresholds(rows, "tanimoto"),
    }
    best = {name: max(values, key=lambda row: row["macro_f1"]) for name, values in metrics.items()}
    bucket_summary = summarize_mcs_buckets(rows)
    heatmap = summarize_heatmap(rows, grid_size=args.heatmap_bins)

    write_tsv(out_dir / "mcs_threshold_metrics.tsv", metrics["mean_mcs_coverage"])
    write_tsv(out_dir / "tanimoto_threshold_metrics_on_mcs_subset.tsv", metrics["tanimoto"])
    write_tsv(out_dir / "mcs_bucket_summary.tsv", bucket_summary)
    write_tsv(out_dir / "mcs_tanimoto_heatmap.tsv", heatmap)

    plot_threshold_metrics(
        figures_dir / "mcs_vs_tanimoto_threshold_metrics.svg",
        metrics,
    )
    plot_bucket_rates(figures_dir / "mcs_label_rates_by_bucket.svg", bucket_summary)
    plot_heatmap(figures_dir / "mcs_tanimoto_similar_rate_heatmap.svg", heatmap, args.heatmap_bins)

    summary = {
        "input": str(args.mcs_results),
        "n_non_ambiguous": len(rows),
        "best": best,
        "files": {
            "mcs_threshold_metrics": str(out_dir / "mcs_threshold_metrics.tsv"),
            "tanimoto_threshold_metrics_on_mcs_subset": str(out_dir / "tanimoto_threshold_metrics_on_mcs_subset.tsv"),
            "mcs_bucket_summary": str(out_dir / "mcs_bucket_summary.tsv"),
            "mcs_tanimoto_heatmap": str(out_dir / "mcs_tanimoto_heatmap.tsv"),
            "report": str(out_dir / "report_zh.md"),
            "figures": str(figures_dir),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_report(out_dir / "report_zh.md", summary, bucket_summary)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcs-results", default=DEFAULT_MCS_RESULTS)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default="dynamic_v1_mcs_t2_analysis")
    parser.add_argument("--heatmap-bins", type=int, default=10)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            label = row["label"]
            if label not in {"similar", "different"}:
                continue
            if row["status"] != "ok":
                continue
            rows.append(
                {
                    "label": 1 if label == "similar" else 0,
                    "label_name": label,
                    "tanimoto": float(row["tanimoto"]),
                    "mean_mcs_coverage": float(row["mean_mcs_coverage"]),
                    "similarity_bucket": row["similarity_bucket"],
                    "timed_out": parse_bool(row["timed_out"]),
                }
            )
    return rows


def scan_thresholds(rows: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    out = []
    for idx in range(101):
        threshold = idx / 100.0
        tp = fp = tn = fn = 0
        for row in rows:
            pred_similar = row[score_key] >= threshold
            actual_similar = row["label"] == 1
            if pred_similar and actual_similar:
                tp += 1
            elif pred_similar and not actual_similar:
                fp += 1
            elif not pred_similar and actual_similar:
                fn += 1
            else:
                tn += 1
        out.append(metrics_row(threshold, tp, fp, tn, fn))
    return out


def metrics_row(threshold: float, tp: int, fp: int, tn: int, fn: int) -> dict[str, Any]:
    precision_similar = safe_div(tp, tp + fp)
    recall_similar = safe_div(tp, tp + fn)
    precision_different = safe_div(tn, tn + fn)
    recall_different = safe_div(tn, tn + fp)
    f1_similar = safe_div(2 * precision_similar * recall_similar, precision_similar + recall_similar)
    f1_different = safe_div(2 * precision_different * recall_different, precision_different + recall_different)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": safe_div(tp + tn, tp + fp + tn + fn),
        "balanced_accuracy": (recall_similar + recall_different) / 2.0,
        "macro_f1": (f1_similar + f1_different) / 2.0,
        "precision_similar": precision_similar,
        "recall_similar": recall_similar,
        "precision_different": precision_different,
        "recall_different": recall_different,
    }


def summarize_mcs_buckets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {
        name: {
            "bucket": name,
            "min_mcs": low,
            "max_mcs": high,
            "n": 0,
            "similar": 0,
            "different": 0,
            "timed_out": 0,
            "sum_tanimoto": 0.0,
        }
        for name, low, high in MCS_BUCKETS
    }
    for row in rows:
        bucket = mcs_bucket(float(row["mean_mcs_coverage"]))
        item = grouped[bucket]
        item["n"] += 1
        item["similar"] += int(row["label"] == 1)
        item["different"] += int(row["label"] == 0)
        item["timed_out"] += int(row["timed_out"])
        item["sum_tanimoto"] += float(row["tanimoto"])
    out = []
    for name, _low, _high in MCS_BUCKETS:
        item = grouped[name]
        n = item["n"]
        item["similar_rate"] = safe_div(item["similar"], n)
        item["different_rate"] = safe_div(item["different"], n)
        item["timeout_rate"] = safe_div(item["timed_out"], n)
        item["mean_tanimoto"] = safe_div(item["sum_tanimoto"], n)
        out.append(item)
    return out


def summarize_heatmap(rows: list[dict[str, Any]], grid_size: int) -> list[dict[str, Any]]:
    cells: dict[tuple[int, int], dict[str, Any]] = defaultdict(lambda: {"n": 0, "similar": 0, "different": 0})
    for row in rows:
        tanimoto_bin = min(grid_size - 1, max(0, int(float(row["tanimoto"]) * grid_size)))
        mcs_bin = min(grid_size - 1, max(0, int(float(row["mean_mcs_coverage"]) * grid_size)))
        cell = cells[(tanimoto_bin, mcs_bin)]
        cell["n"] += 1
        cell["similar"] += int(row["label"] == 1)
        cell["different"] += int(row["label"] == 0)
    out = []
    for tanimoto_bin in range(grid_size):
        for mcs_bin in range(grid_size):
            cell = cells[(tanimoto_bin, mcs_bin)]
            n = cell["n"]
            out.append(
                {
                    "tanimoto_bin": tanimoto_bin,
                    "mcs_bin": mcs_bin,
                    "tanimoto_min": tanimoto_bin / grid_size,
                    "tanimoto_max": (tanimoto_bin + 1) / grid_size,
                    "mcs_min": mcs_bin / grid_size,
                    "mcs_max": (mcs_bin + 1) / grid_size,
                    "n": n,
                    "similar": cell["similar"],
                    "different": cell["different"],
                    "similar_rate": safe_div(cell["similar"], n),
                }
            )
    return out


def mcs_bucket(value: float) -> str:
    for name, low, high in MCS_BUCKETS:
        if low <= value < high:
            return name
    return MCS_BUCKETS[-1][0]


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def plot_threshold_metrics(path: Path, metrics: dict[str, list[dict[str, Any]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    styles = {
        "mean_mcs_coverage": ("#1f77b4", "-"),
        "tanimoto": ("#d62728", "--"),
    }
    for name, rows in metrics.items():
        color, linestyle = styles[name]
        xs = [row["threshold"] for row in rows]
        ax.plot(xs, [row["macro_f1"] for row in rows], color=color, linestyle=linestyle, label=f"{name} macro-F1")
        ax.plot(xs, [row["balanced_accuracy"] for row in rows], color=color, linestyle=":", label=f"{name} balanced acc")
    ax.set_xlabel("threshold")
    ax.set_ylabel("metric")
    ax.set_ylim(0.45, 0.65)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_bucket_rates(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["bucket"] for row in rows]
    rates = [row["similar_rate"] for row in rows]
    counts = [row["n"] for row in rows]
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.bar(labels, rates, color="#4c78a8")
    ax1.set_ylabel("similar rate")
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis="x", rotation=30)
    ax2 = ax1.twinx()
    ax2.plot(labels, counts, color="#f58518", marker="o")
    ax2.set_ylabel("pair count")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_heatmap(path: Path, rows: list[dict[str, Any]], grid_size: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = [[math.nan for _ in range(grid_size)] for _ in range(grid_size)]
    for row in rows:
        if row["n"] >= 50:
            matrix[int(row["mcs_bin"])][int(row["tanimoto_bin"])] = row["similar_rate"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, origin="lower", vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xlabel("Tanimoto bin")
    ax.set_ylabel("Mean MCS coverage bin")
    ticks = list(range(grid_size))
    labels = [f"{idx / grid_size:.1f}" for idx in ticks]
    ax.set_xticks(ticks, labels=labels)
    ax.set_yticks(ticks, labels=labels)
    fig.colorbar(im, ax=ax, label="similar rate")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(path: Path, summary: dict[str, Any], bucket_summary: list[dict[str, Any]]) -> None:
    best_mcs = summary["best"]["mean_mcs_coverage"]
    best_tanimoto = summary["best"]["tanimoto"]
    lines = [
        "# MCS activity-transfer threshold 分析",
        "",
        f"- non-ambiguous usable pairs: {summary['n_non_ambiguous']:,}",
        f"- best mean MCS coverage threshold: {best_mcs['threshold']:.2f}",
        f"- best MCS macro-F1: {best_mcs['macro_f1']:.4f}",
        f"- best MCS balanced accuracy: {best_mcs['balanced_accuracy']:.4f}",
        f"- best Tanimoto threshold on same subset: {best_tanimoto['threshold']:.2f}",
        f"- best Tanimoto macro-F1 on same subset: {best_tanimoto['macro_f1']:.4f}",
        f"- best Tanimoto balanced accuracy on same subset: {best_tanimoto['balanced_accuracy']:.4f}",
        "",
        "## MCS Coverage Buckets",
        "",
        "| bucket | n | similar rate | different rate | timeout rate | mean Tanimoto |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in bucket_summary:
        lines.append(
            "| {bucket} | {n:,} | {similar_rate:.4f} | {different_rate:.4f} | {timeout_rate:.4f} | {mean_tanimoto:.4f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
