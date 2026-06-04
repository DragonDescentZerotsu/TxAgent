"""Plot and summarize multiple activity-transfer LLM benchmark runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.chembl_tool.activity_transfer_benchmark.run_llm_benchmark import (
    compute_metrics,
    prepare_input_records,
    read_jsonl,
)


DEFAULT_OUT_ROOT = "outputs/chembl_tool/activity_transfer_benchmark/comparisons"
BASELINE_METHODS = ["Tanimoto >= 0.5", "Bucket majority"]
PALETTE = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#E45756",
    "#72B7B2",
    "#9D755D",
    "#BAB0AC",
]


def main() -> int:
    args = parse_args()
    runs = parse_runs(args.run)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics = {label: load_run_metrics(Path(path)) for label, path in runs}
    baseline_metrics = full_validation_baselines(Path(args.input_jsonl))
    methods = list(metrics) + BASELINE_METHODS
    colors = [PALETTE[index % len(PALETTE)] for index in range(len(methods))]
    groups_by_scope = {
        "similarity_bucket": sorted(baseline_metrics["per_similarity_bucket"], key=lambda item: int(item)),
        "assay_type": sorted(baseline_metrics["per_assay_type"]),
    }
    if baseline_metrics.get("per_eval_subset"):
        groups_by_scope["eval_subset"] = sorted(baseline_metrics["per_eval_subset"])

    dashboard_svg = fig_dir / "comparison_dashboard.svg"
    dashboard_png = fig_dir / "comparison_dashboard.png"
    overall_svg = fig_dir / "overall_metrics.svg"
    overall_png = fig_dir / "overall_metrics.png"
    label_recall_paths = {
        scope: (
            fig_dir / f"label_recall_by_{scope}.svg",
            fig_dir / f"label_recall_by_{scope}.png",
        )
        for scope in groups_by_scope
    }
    table_path = out_dir / "comparison_metrics.tsv"
    report_path = out_dir / "comparison_report.md"

    plot_dashboard(dashboard_svg, dashboard_png, args.title, metrics, baseline_metrics, methods, colors, groups_by_scope)
    plot_overall(overall_svg, overall_png, args.title, metrics, baseline_metrics, methods, colors)
    for scope, groups in groups_by_scope.items():
        svg_path, png_path = label_recall_paths[scope]
        plot_label_recalls(svg_path, png_path, scope, groups, metrics, baseline_metrics, methods, colors)
    write_tables(table_path, metrics, baseline_metrics, methods, groups_by_scope)
    write_report(
        report_path,
        args.title,
        args.input_jsonl,
        runs,
        table_path,
        dashboard_svg,
        dashboard_png,
        overall_svg,
        label_recall_paths,
        metrics,
        baseline_metrics,
        methods,
        groups_by_scope,
    )

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "dashboard_png": str(dashboard_png),
                "overall_png": str(overall_png),
                "label_recall_png": {scope: str(paths[1]) for scope, paths in label_recall_paths.items()},
                "report": str(report_path),
                "tsv": str(table_path),
                "overall_macro_f1": {
                    method: metric_for_method(metrics, baseline_metrics, method, "macro_f1") for method in methods
                },
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Benchmark run directory with a display label. Repeat for multiple runs.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--title", default="Activity-transfer LLM comparison")
    return parser.parse_args()


def parse_runs(values: list[str]) -> list[tuple[str, str]]:
    runs = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--run must be LABEL=PATH, got: {value}")
        label, path = value.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise SystemExit(f"--run must be LABEL=PATH, got: {value}")
        if label in seen:
            raise SystemExit(f"Duplicate run label: {label}")
        if not (Path(path) / "metrics.json").exists():
            raise SystemExit(f"Missing metrics.json for run: {path}")
        seen.add(label)
        runs.append((label, path))
    return runs


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_metrics(run_dir: Path) -> dict[str, Any]:
    metrics = read_json(run_dir / "metrics.json")
    if metrics.get("per_eval_subset") or not (run_dir / "predictions.jsonl").exists():
        return metrics
    return compute_metrics(read_jsonl(run_dir / "predictions.jsonl"))


def full_validation_baselines(input_jsonl: Path) -> dict[str, Any]:
    records = prepare_input_records(read_jsonl(input_jsonl))
    rows = []
    for record in records:
        rows.append(
            {
                "status": "ok",
                "prediction": record["baseline_tanimoto_0_50_prediction"],
                "label": record["label"],
                "input_record": {"tanimoto": record["tanimoto"], "hf_metadata": record["metadata"]},
                "baseline_tanimoto_0_50_prediction": record["baseline_tanimoto_0_50_prediction"],
                "baseline_tanimoto_0_48_prediction": record["baseline_tanimoto_0_48_prediction"],
                "baseline_similarity_bucket_majority_prediction": record[
                    "baseline_similarity_bucket_majority_prediction"
                ],
                "baseline_assay_type_tanimoto_0_50_prediction": record[
                    "baseline_assay_type_tanimoto_0_50_prediction"
                ],
                "usage": {},
                "tool_count": 0,
            }
        )
    return compute_metrics(rows)


def metric_for_method(metrics: dict[str, Any], baseline_metrics: dict[str, Any], method: str, metric: str) -> float:
    if method in metrics:
        return float(metrics[method]["llm"][metric])
    if method == "Tanimoto >= 0.5":
        return float(baseline_metrics["baselines"]["tanimoto_0_50"][metric])
    if method == "Bucket majority":
        return float(baseline_metrics["baselines"]["similarity_bucket_majority"][metric])
    raise KeyError(method)


def metric_for_group(
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    scope: str,
    group: str,
    method: str,
) -> dict[str, Any]:
    if scope == "overall":
        if method in metrics:
            return metrics[method]["llm"]
        if method == "Tanimoto >= 0.5":
            return baseline_metrics["baselines"]["tanimoto_0_50"]
        if method == "Bucket majority":
            return baseline_metrics["baselines"]["similarity_bucket_majority"]
    if method in metrics:
        return metrics[method][f"per_{scope}"][group]["llm"]
    if method == "Tanimoto >= 0.5":
        return baseline_metrics[f"per_{scope}"][group]["baselines"]["tanimoto_0_50"]
    if method == "Bucket majority":
        return baseline_metrics[f"per_{scope}"][group]["baselines"]["similarity_bucket_majority"]
    raise KeyError((scope, group, method))


def true_label_n(metric: dict[str, Any], label: str) -> int:
    if label == "similar":
        return int(metric.get("tp", 0)) + int(metric.get("fn", 0))
    if label == "different":
        return int(metric.get("tn", 0)) + int(metric.get("fp", 0))
    raise KeyError(label)


def group_count_label(baseline_metrics: dict[str, Any], scope: str, group: str) -> str:
    metric = baseline_metrics[f"per_{scope}"][group]["baselines"]["tanimoto_0_50"]
    return f"{group}\nS={true_label_n(metric, 'similar')}\nD={true_label_n(metric, 'different')}"


def bar_offsets(n_methods: int) -> tuple[float, list[float]]:
    width = min(0.8 / max(1, n_methods), 0.16)
    center = (n_methods - 1) / 2
    return width, [(index - center) * width for index in range(n_methods)]


def plot_dashboard(
    svg_path: Path,
    png_path: Path,
    title: str,
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    methods: list[str],
    colors: list[str],
    groups_by_scope: dict[str, list[str]],
) -> None:
    n_group_panels = len(groups_by_scope)
    fig = plt.figure(figsize=(16, 5 + 3.8 * n_group_panels))
    grid = fig.add_gridspec(1 + n_group_panels, 1, height_ratios=[1.0] + [1.25] * n_group_panels, hspace=0.45)

    ax = fig.add_subplot(grid[0, 0])
    values = [metric_for_method(metrics, baseline_metrics, method, "macro_f1") for method in methods]
    bars = ax.bar(methods, values, color=colors)
    annotate_bars(ax, bars, values)
    ax.set_ylim(0, max(0.75, max(values) + 0.08))
    ax.set_ylabel("Macro-F1")
    ax.set_title("Overall performance")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)

    for row_index, (scope, groups) in enumerate(groups_by_scope.items(), start=1):
        plot_group_macro_f1_panel(
            fig.add_subplot(grid[row_index, 0]),
            scope,
            groups,
            metrics,
            baseline_metrics,
            methods,
            colors,
        )
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def plot_group_macro_f1_panel(
    ax: Any,
    scope: str,
    groups: list[str],
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    methods: list[str],
    colors: list[str],
) -> None:
    x_values = np.arange(len(groups))
    width, offsets = bar_offsets(len(methods))
    for index, method in enumerate(methods):
        values = [
            float(metric_for_group(metrics, baseline_metrics, scope, group, method).get("macro_f1", 0.0))
            for group in groups
        ]
        ax.bar(x_values + offsets[index], values, width, label=method, color=colors[index])
    ax.set_xticks(x_values)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel(scope)
    ax.set_title(f"Performance by {scope}")
    ax.legend(ncol=min(4, len(methods)), fontsize=8)
    ax.grid(axis="y", alpha=0.25)


def plot_overall(
    svg_path: Path,
    png_path: Path,
    title: str,
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    methods: list[str],
    colors: list[str],
) -> None:
    metric_names = ["accuracy", "balanced_accuracy", "macro_f1"]
    fig, ax = plt.subplots(figsize=(11, 5))
    x_values = np.arange(len(metric_names))
    width, offsets = bar_offsets(len(methods))
    for index, method in enumerate(methods):
        values = [metric_for_method(metrics, baseline_metrics, method, metric) for metric in metric_names]
        ax.bar(x_values + offsets[index], values, width, label=method, color=colors[index])
    ax.set_xticks(x_values)
    ax.set_xticklabels(["Accuracy", "Balanced accuracy", "Macro-F1"])
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=min(3, len(methods)))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def plot_label_recalls(
    svg_path: Path,
    png_path: Path,
    scope: str,
    groups: list[str],
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    methods: list[str],
    colors: list[str],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(16, 8.5), sharex=True)
    x_values = np.arange(len(groups))
    width, offsets = bar_offsets(len(methods))
    label_specs = [
        ("similar", "Recall on true similar / positive transfer"),
        ("different", "Recall on true different / negative transfer"),
    ]
    for ax, (label_key, title) in zip(axes, label_specs):
        metric_key = f"recall_{label_key}"
        for index, method in enumerate(methods):
            values = [
                float(metric_for_group(metrics, baseline_metrics, scope, group, method).get(metric_key, 0.0))
                for group in groups
            ]
            ax.bar(x_values + offsets[index], values, width, label=method, color=colors[index])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Recall")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=min(4, len(methods)), fontsize=8)
    axes[-1].set_xticks(x_values)
    axes[-1].set_xticklabels([group_count_label(baseline_metrics, scope, group) for group in groups])
    axes[-1].set_xlabel(scope)
    fig.suptitle(f"Label-specific performance by {scope}", y=0.995)
    fig.tight_layout()
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def annotate_bars(ax: Any, bars: Any, values: list[float]) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def metric_row(scope: str, group: str, method: str, metric: dict[str, Any]) -> list[str]:
    return [
        scope,
        group,
        method,
        str(metric.get("n", "")),
        str(true_label_n(metric, "similar")),
        str(true_label_n(metric, "different")),
        f"{float(metric.get('accuracy', 0.0)):.6f}",
        f"{float(metric.get('balanced_accuracy', 0.0)):.6f}",
        f"{float(metric.get('macro_f1', 0.0)):.6f}",
        f"{float(metric.get('recall_similar', 0.0)):.6f}",
        f"{float(metric.get('recall_different', 0.0)):.6f}",
        f"{float(metric.get('precision_similar', 0.0)):.6f}",
        f"{float(metric.get('precision_different', 0.0)):.6f}",
    ]


def iter_metric_rows(
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    methods: list[str],
    groups_by_scope: dict[str, list[str]],
) -> list[list[str]]:
    rows = []
    for method in methods:
        rows.append(metric_row("overall", "all", method, metric_for_group(metrics, baseline_metrics, "overall", "all", method)))
    for scope, groups in groups_by_scope.items():
        for group in groups:
            for method in methods:
                rows.append(metric_row(scope, group, method, metric_for_group(metrics, baseline_metrics, scope, group, method)))
    return rows


def write_tables(
    path: Path,
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    methods: list[str],
    groups_by_scope: dict[str, list[str]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "scope\tgroup\tmethod\tn\tn_true_similar\tn_true_different\taccuracy\tbalanced_accuracy\tmacro_f1\t"
            "recall_similar\trecall_different\tprecision_similar\tprecision_different\n"
        )
        for row in iter_metric_rows(metrics, baseline_metrics, methods, groups_by_scope):
            handle.write("\t".join(row) + "\n")


def write_report(
    path: Path,
    title: str,
    input_jsonl: str,
    runs: list[tuple[str, str]],
    table_path: Path,
    dashboard_svg: Path,
    dashboard_png: Path,
    overall_svg: Path,
    label_recall_paths: dict[str, tuple[Path, Path]],
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    methods: list[str],
    groups_by_scope: dict[str, list[str]],
) -> None:
    rows = iter_metric_rows(metrics, baseline_metrics, methods, groups_by_scope)
    by_key = {(row[0], row[1], row[2]): row for row in rows}
    lines = [
        f"# {title}",
        "",
        f"- input: `{input_jsonl}`",
    ]
    for label, run_path in runs:
        metric = metrics[label]
        lines.append(
            f"- {label}: `{run_path}`, ok {int(metric['n_ok']):,}/{int(metric['n']):,}, "
            f"failed {int(metric['n_failed']):,}, wall_s {float(metric.get('wall_s', 0.0)):.1f}"
        )
    lines.extend(["", "## Overall", "", overall_table(methods, by_key)])
    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"- Dashboard SVG: `{dashboard_svg}`",
            f"- Dashboard PNG: `{dashboard_png}`",
            f"- Overall metrics SVG: `{overall_svg}`",
            f"- Metrics table: `{table_path}`",
        ]
    )
    for scope, (svg_path, png_path) in label_recall_paths.items():
        lines.append(f"- {scope} label recall SVG: `{svg_path}`")
        lines.append(f"- {scope} label recall PNG: `{png_path}`")
    for scope, groups in groups_by_scope.items():
        lines.extend(["", f"## {scope} Macro-F1", "", macro_f1_table(scope, groups, methods, by_key)])
        append_label_recall_table(lines, by_key, scope, groups, methods)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def overall_table(methods: list[str], by_key: dict[tuple[str, str, str], list[str]]) -> str:
    lines = ["| method | n | accuracy | balanced accuracy | macro-F1 |", "| --- | ---: | ---: | ---: | ---: |"]
    for method in methods:
        row = by_key[("overall", "all", method)]
        lines.append(f"| {method} | {int(row[3]):,} | {float(row[6]):.4f} | {float(row[7]):.4f} | {float(row[8]):.4f} |")
    return "\n".join(lines)


def macro_f1_table(scope: str, groups: list[str], methods: list[str], by_key: dict[tuple[str, str, str], list[str]]) -> str:
    lines = [
        f"| {scope} | " + " | ".join(methods) + " |",
        "| --- | " + " | ".join(["---:"] * len(methods)) + " |",
    ]
    for group in groups:
        values = [float(by_key[(scope, group, method)][8]) for method in methods]
        lines.append(f"| {group} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    return "\n".join(lines)


def append_label_recall_table(
    lines: list[str],
    by_key: dict[tuple[str, str, str], list[str]],
    scope: str,
    groups: list[str],
    methods: list[str],
) -> None:
    method_headers = " | ".join(f"{method} recall" for method in methods)
    lines.extend(
        [
            "",
            f"## {scope} Label-Specific Recall",
            "",
            "Metric: recall within each true label subset. true similar is positive transfer; true different is negative transfer.",
            "",
            f"| {scope} | true label | n_true | {method_headers} |",
            f"| --- | --- | ---: | {' | '.join(['---:'] * len(methods))} |",
        ]
    )
    for group in groups:
        for label, n_index, recall_index in (("similar", 4, 9), ("different", 5, 10)):
            full_set = by_key[(scope, group, "Tanimoto >= 0.5")]
            values = [float(by_key[(scope, group, method)][recall_index]) for method in methods]
            lines.append(
                f"| {group} | {label} | {int(full_set[n_index]):,} | "
                + " | ".join(f"{value:.4f}" for value in values)
                + " |"
            )


if __name__ == "__main__":
    raise SystemExit(main())
