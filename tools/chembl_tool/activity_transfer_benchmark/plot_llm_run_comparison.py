"""Plot and summarize two activity-transfer LLM benchmark runs."""

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


DEFAULT_INPUT = (
    "outputs/chembl_tool/activity_transfer_benchmark/hf_jiosephlee_valid10k/"
    "chembl-mol12-stdsep-assay-mol-disjoint-no-props-no-tanimoto/validation.jsonl"
)
DEFAULT_OUT_DIR = (
    "outputs/chembl_tool/activity_transfer_benchmark/comparisons/hf_jiosephlee_valid10k/"
    "chembl-mol12-stdsep-assay-mol-disjoint-no-props-no-tanimoto"
)
DEFAULT_GPT_RUN = (
    "outputs/chembl_tool/activity_transfer_benchmark/llm_runs/"
    "gpt_oss_120b_hf_assay_mol_disjoint_no_tanimoto_valid10k_tools"
)
DEFAULT_DEEPSEEK_RUN = (
    "outputs/chembl_tool/activity_transfer_benchmark/llm_runs/"
    "deepseek_v4_pro_hf_assay_mol_disjoint_no_tanimoto_valid10k_tools"
)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "gpt-oss-120b": read_json(Path(args.gpt_run) / "metrics.json"),
        "DeepSeek-v4-pro": read_json(Path(args.deepseek_run) / "metrics.json"),
    }
    baseline_metrics = full_validation_baselines(Path(args.input_jsonl))
    methods = ["gpt-oss-120b", "DeepSeek-v4-pro", "Tanimoto >= 0.5", "Bucket majority"]
    colors = ["#4C78A8", "#B279A2", "#F58518", "#54A24B"]

    dashboard_svg = fig_dir / "comparison_dashboard.svg"
    dashboard_png = fig_dir / "comparison_dashboard.png"
    overall_svg = fig_dir / "overall_metrics.svg"
    overall_png = fig_dir / "overall_metrics.png"
    bucket_label_svg = fig_dir / "label_recall_by_similarity_bucket.svg"
    bucket_label_png = fig_dir / "label_recall_by_similarity_bucket.png"
    assay_label_svg = fig_dir / "label_recall_by_assay_type.svg"
    assay_label_png = fig_dir / "label_recall_by_assay_type.png"
    table_path = out_dir / "comparison_metrics.tsv"
    report_path = out_dir / "comparison_report.md"

    plot_dashboard(dashboard_svg, dashboard_png, metrics, baseline_metrics, methods, colors, args.input_jsonl)
    plot_overall(overall_svg, overall_png, metrics, baseline_metrics, methods, colors)
    plot_label_recalls(
        bucket_label_svg,
        bucket_label_png,
        "similarity_bucket",
        sorted(baseline_metrics["per_similarity_bucket"], key=lambda item: int(item)),
        metrics,
        baseline_metrics,
        methods,
        colors,
    )
    plot_label_recalls(
        assay_label_svg,
        assay_label_png,
        "assay_type",
        sorted(baseline_metrics["per_assay_type"]),
        metrics,
        baseline_metrics,
        methods,
        colors,
    )
    write_tables(table_path, metrics, baseline_metrics)
    write_report(
        report_path,
        table_path,
        dashboard_svg,
        dashboard_png,
        overall_svg,
        bucket_label_svg,
        assay_label_svg,
        metrics,
        baseline_metrics,
        args.input_jsonl,
    )

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "dashboard_svg": str(dashboard_svg),
                "dashboard_png": str(dashboard_png),
                "bucket_label_recall_svg": str(bucket_label_svg),
                "assay_type_label_recall_svg": str(assay_label_svg),
                "report": str(report_path),
                "tsv": str(table_path),
                "overall_macro_f1": {
                    "gpt-oss-120b": metrics["gpt-oss-120b"]["llm"]["macro_f1"],
                    "DeepSeek-v4-pro": metrics["DeepSeek-v4-pro"]["llm"]["macro_f1"],
                    "Tanimoto >= 0.5": baseline_metrics["baselines"]["tanimoto_0_50"]["macro_f1"],
                    "Bucket majority": baseline_metrics["baselines"]["similarity_bucket_majority"]["macro_f1"],
                },
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT)
    parser.add_argument("--gpt-run", default=DEFAULT_GPT_RUN)
    parser.add_argument("--deepseek-run", default=DEFAULT_DEEPSEEK_RUN)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def plot_dashboard(
    svg_path: Path,
    png_path: Path,
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    methods: list[str],
    colors: list[str],
    input_jsonl: str,
) -> None:
    fig = plt.figure(figsize=(14, 13))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.25, 1.35], hspace=0.42)

    ax = fig.add_subplot(grid[0, 0])
    values = [metric_for_method(metrics, baseline_metrics, method, "macro_f1") for method in methods]
    bars = ax.bar(methods, values, color=colors)
    annotate_bars(ax, bars, values)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Overall performance on assay-mol-disjoint no-tanimoto valid10k")
    ax.tick_params(axis="x", rotation=10)
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(grid[1, 0])
    buckets = sorted(baseline_metrics["per_similarity_bucket"], key=lambda item: int(item))
    x_values = np.arange(len(buckets))
    width = 0.18
    series = [
        ("gpt-oss-120b", [metrics["gpt-oss-120b"]["per_similarity_bucket"][b]["llm"]["macro_f1"] for b in buckets]),
        (
            "DeepSeek-v4-pro",
            [metrics["DeepSeek-v4-pro"]["per_similarity_bucket"][b]["llm"]["macro_f1"] for b in buckets],
        ),
        (
            "Tanimoto >= 0.5",
            [baseline_metrics["per_similarity_bucket"][b]["baselines"]["tanimoto_0_50"]["macro_f1"] for b in buckets],
        ),
        (
            "Bucket majority",
            [
                baseline_metrics["per_similarity_bucket"][b]["baselines"]["similarity_bucket_majority"]["macro_f1"]
                for b in buckets
            ],
        ),
    ]
    for index, (label, values) in enumerate(series):
        ax.bar(x_values + (index - 1.5) * width, values, width, label=label, color=colors[index])
    ax.set_xticks(x_values)
    ax.set_xticklabels(buckets)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("similarity_bucket")
    ax.set_title("Performance by similarity_bucket")
    ax.legend(ncol=4, fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(grid[2, 0])
    assay_types = sorted(baseline_metrics["per_assay_type"])
    x_values = np.arange(len(assay_types))
    series = [
        ("gpt-oss-120b", [metrics["gpt-oss-120b"]["per_assay_type"][a]["llm"]["macro_f1"] for a in assay_types]),
        ("DeepSeek-v4-pro", [metrics["DeepSeek-v4-pro"]["per_assay_type"][a]["llm"]["macro_f1"] for a in assay_types]),
        (
            "Tanimoto >= 0.5",
            [baseline_metrics["per_assay_type"][a]["baselines"]["tanimoto_0_50"]["macro_f1"] for a in assay_types],
        ),
        (
            "Bucket majority",
            [
                baseline_metrics["per_assay_type"][a]["baselines"]["similarity_bucket_majority"]["macro_f1"]
                for a in assay_types
            ],
        ),
    ]
    for index, (label, values) in enumerate(series):
        ax.bar(x_values + (index - 1.5) * width, values, width, label=label, color=colors[index])
    ax.set_xticks(x_values)
    ax.set_xticklabels(assay_types)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("assay_type")
    ax.set_title("Performance by assay_type")
    ax.legend(ncol=4, fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(str(input_jsonl), fontsize=9, y=0.995)
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
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True)
    x_values = np.arange(len(groups))
    width = 0.18
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
            ax.bar(x_values + (index - 1.5) * width, values, width, label=method, color=colors[index])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Recall")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=4, fontsize=9)
    axes[-1].set_xticks(x_values)
    axes[-1].set_xticklabels([group_count_label(baseline_metrics, scope, group) for group in groups])
    axes[-1].set_xlabel(scope)
    fig.suptitle(f"Label-specific performance by {scope}", y=0.995)
    fig.tight_layout()
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def plot_overall(
    svg_path: Path,
    png_path: Path,
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    methods: list[str],
    colors: list[str],
) -> None:
    metric_names = ["accuracy", "balanced_accuracy", "macro_f1"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x_values = np.arange(len(metric_names))
    width = 0.18
    for index, method in enumerate(methods):
        values = [metric_for_method(metrics, baseline_metrics, method, metric) for metric in metric_names]
        ax.bar(x_values + (index - 1.5) * width, values, width, label=method, color=colors[index])
    ax.set_xticks(x_values)
    ax.set_xticklabels(["Accuracy", "Balanced accuracy", "Macro-F1"])
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Score")
    ax.set_title("Overall metrics")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


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


def iter_metric_rows(metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> list[list[str]]:
    rows = []
    rows.append(metric_row("overall", "all", "gpt-oss-120b", metrics["gpt-oss-120b"]["llm"]))
    rows.append(metric_row("overall", "all", "DeepSeek-v4-pro", metrics["DeepSeek-v4-pro"]["llm"]))
    rows.append(metric_row("overall", "all", "Tanimoto >= 0.5", baseline_metrics["baselines"]["tanimoto_0_50"]))
    rows.append(
        metric_row("overall", "all", "Bucket majority", baseline_metrics["baselines"]["similarity_bucket_majority"])
    )
    for bucket in sorted(baseline_metrics["per_similarity_bucket"], key=lambda item: int(item)):
        rows.append(metric_row("similarity_bucket", bucket, "gpt-oss-120b", metrics["gpt-oss-120b"]["per_similarity_bucket"][bucket]["llm"]))
        rows.append(metric_row("similarity_bucket", bucket, "DeepSeek-v4-pro", metrics["DeepSeek-v4-pro"]["per_similarity_bucket"][bucket]["llm"]))
        rows.append(
            metric_row(
                "similarity_bucket",
                bucket,
                "Tanimoto >= 0.5",
                baseline_metrics["per_similarity_bucket"][bucket]["baselines"]["tanimoto_0_50"],
            )
        )
        rows.append(
            metric_row(
                "similarity_bucket",
                bucket,
                "Bucket majority",
                baseline_metrics["per_similarity_bucket"][bucket]["baselines"]["similarity_bucket_majority"],
            )
        )
    for assay_type in sorted(baseline_metrics["per_assay_type"]):
        rows.append(metric_row("assay_type", assay_type, "gpt-oss-120b", metrics["gpt-oss-120b"]["per_assay_type"][assay_type]["llm"]))
        rows.append(metric_row("assay_type", assay_type, "DeepSeek-v4-pro", metrics["DeepSeek-v4-pro"]["per_assay_type"][assay_type]["llm"]))
        rows.append(
            metric_row(
                "assay_type",
                assay_type,
                "Tanimoto >= 0.5",
                baseline_metrics["per_assay_type"][assay_type]["baselines"]["tanimoto_0_50"],
            )
        )
        rows.append(
            metric_row(
                "assay_type",
                assay_type,
                "Bucket majority",
                baseline_metrics["per_assay_type"][assay_type]["baselines"]["similarity_bucket_majority"],
            )
        )
    return rows


def write_tables(path: Path, metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "scope\tgroup\tmethod\tn\tn_true_similar\tn_true_different\taccuracy\tbalanced_accuracy\tmacro_f1\t"
            "recall_similar\trecall_different\tprecision_similar\tprecision_different\n"
        )
        for row in iter_metric_rows(metrics, baseline_metrics):
            handle.write("\t".join(row) + "\n")


def write_report(
    path: Path,
    table_path: Path,
    dashboard_svg: Path,
    dashboard_png: Path,
    overall_svg: Path,
    bucket_label_svg: Path,
    assay_label_svg: Path,
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    input_jsonl: str,
) -> None:
    rows = iter_metric_rows(metrics, baseline_metrics)
    by_key = {(row[0], row[1], row[2]): row for row in rows}
    lines = [
        "# HF Assay-Mol-Disjoint No-Tanimoto Valid10k Comparison",
        "",
        f"- input: `{input_jsonl}`",
    ]
    for name, metric in metrics.items():
        lines.append(
            f"- {name}: ok {int(metric['n_ok']):,}/{int(metric['n']):,}, "
            f"failed {int(metric['n_failed']):,}, tool_calls {int(metric['tool_calls']):,}, "
            f"wall_s {float(metric['wall_s']):.1f}"
        )
    lines.extend(["", "## Overall", "", "| method | n | accuracy | balanced accuracy | macro-F1 |", "| --- | ---: | ---: | ---: | ---: |"])
    for method in ["gpt-oss-120b", "DeepSeek-v4-pro", "Tanimoto >= 0.5", "Bucket majority"]:
        row = by_key[("overall", "all", method)]
        label, n, accuracy, balanced, macro_f1 = row[2], row[3], row[6], row[7], row[8]
        lines.append(f"| {label} | {int(n):,} | {float(accuracy):.4f} | {float(balanced):.4f} | {float(macro_f1):.4f} |")
    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"- Dashboard SVG: `{dashboard_svg}`",
            f"- Dashboard PNG: `{dashboard_png}`",
            f"- Overall metrics SVG: `{overall_svg}`",
            f"- Similarity-bucket label recall SVG: `{bucket_label_svg}`",
            f"- Assay-type label recall SVG: `{assay_label_svg}`",
            f"- Metrics table: `{table_path}`",
            "",
            "## Similarity Bucket Macro-F1",
            "",
            "| bucket | gpt-oss | DeepSeek | Tanimoto >= 0.5 | Bucket majority |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for bucket in sorted(baseline_metrics["per_similarity_bucket"], key=lambda item: int(item)):
        lines.append(
            f"| {bucket} | "
            f"{float(by_key[('similarity_bucket', bucket, 'gpt-oss-120b')][8]):.4f} | "
            f"{float(by_key[('similarity_bucket', bucket, 'DeepSeek-v4-pro')][8]):.4f} | "
            f"{float(by_key[('similarity_bucket', bucket, 'Tanimoto >= 0.5')][8]):.4f} | "
            f"{float(by_key[('similarity_bucket', bucket, 'Bucket majority')][8]):.4f} |"
        )
    append_label_recall_table(lines, by_key, "similarity_bucket", sorted(baseline_metrics["per_similarity_bucket"], key=lambda item: int(item)))
    lines.extend(
        [
            "",
            "## Assay Type Macro-F1",
            "",
            "| assay_type | gpt-oss | DeepSeek | Tanimoto >= 0.5 | Bucket majority |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for assay_type in sorted(baseline_metrics["per_assay_type"]):
        lines.append(
            f"| {assay_type} | "
            f"{float(by_key[('assay_type', assay_type, 'gpt-oss-120b')][8]):.4f} | "
            f"{float(by_key[('assay_type', assay_type, 'DeepSeek-v4-pro')][8]):.4f} | "
            f"{float(by_key[('assay_type', assay_type, 'Tanimoto >= 0.5')][8]):.4f} | "
            f"{float(by_key[('assay_type', assay_type, 'Bucket majority')][8]):.4f} |"
        )
    append_label_recall_table(lines, by_key, "assay_type", sorted(baseline_metrics["per_assay_type"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_label_recall_table(lines: list[str], by_key: dict[tuple[str, str, str], list[str]], scope: str, groups: list[str]) -> None:
    title = "Similarity Bucket" if scope == "similarity_bucket" else "Assay Type"
    methods = ["gpt-oss-120b", "DeepSeek-v4-pro", "Tanimoto >= 0.5", "Bucket majority"]
    method_headers = " | ".join(f"{method} recall" for method in methods)
    lines.extend(
        [
            "",
            f"## {title} Label-Specific Recall",
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
