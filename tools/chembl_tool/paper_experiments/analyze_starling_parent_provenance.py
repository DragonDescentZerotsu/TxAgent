"""Summarize record and publication provenance per frozen Starling parent.

The benchmark artifacts intentionally cap ``source_pmids`` at 50 values for
audit readability.  This analysis replays the frozen task adapters so the
publication-count distribution uses the complete accepted-record provenance,
then validates the reconstructed parent sets and record counts against the
frozen benchmark JSONL files.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Callable, Iterable
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from data.processing.gold_labels.benchmark_dataset import LabelDecision
from tools.chembl_tool.paper_experiments.paper_figure_style import (
    BG,
    BLIND,
    GRID,
    INK,
    MUTED,
    VISIBLE,
)
from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_benchmark import (
    SOURCE_REVISION as BBB_SOURCE_REVISION,
)
from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_benchmark import (
    load_label_decisions as load_bbb_decisions,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.starling_benchmark import (
    load_label_decisions as load_bioavailability_decisions,
)
from data.processing.evidence_library.versions.v7.tasks.skin_reaction.starling_benchmark import (
    load_label_decisions as load_skin_decisions,
)


DEFAULT_OUTPUT_DIR = Path("outputs/paper/starling_parent_provenance")
BENCHMARK_ROOT = Path("data/gold_labels/legacy/processed_starling")


@dataclass(frozen=True)
class TaskSpec:
    key: str
    output_name: str
    display_name: str
    loader: Callable[..., tuple[Iterable[LabelDecision], dict[str, Any]]]
    hf_revision: str | None = None


TASKS = (
    TaskSpec(
        "bbb_martins",
        "BBB_Martins",
        "BBB_Martins",
        load_bbb_decisions,
        BBB_SOURCE_REVISION,
    ),
    TaskSpec(
        "bioavailability_ma",
        "Bioavailability_Ma",
        "Bioavailability_Ma",
        load_bioavailability_decisions,
    ),
    TaskSpec("skin_reaction", "Skin_Reaction", "Skin_Reaction", load_skin_decisions),
)


@dataclass
class ParentAggregate:
    labels: set[int] = field(default_factory=set)
    source_record_count: int = 0
    records_with_pmid: int = 0
    pmids: set[str] = field(default_factory=set)
    source_ids: set[str] = field(default_factory=set)


RECORD_BINS = (
    ("1", 1, 1),
    ("2", 2, 2),
    ("3–4", 3, 4),
    ("5–8", 5, 8),
    ("9–16", 9, 16),
    ("17–32", 17, 32),
    ("33–64", 33, 64),
    ("65–128", 65, 128),
    ("129–256", 129, 256),
    ("257+", 257, None),
)

PMID_BINS = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2", 2, 2),
    ("3–4", 3, 4),
    ("5–8", 5, 8),
    ("9–16", 9, 16),
    ("17–32", 17, 32),
    ("33–64", 33, 64),
    ("65+", 65, None),
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_parent_stats:
        parent_rows = _read_cached_parent_stats(output_dir / "parent_stats.csv")
        source_metadata = {
            task.key: json.loads(
                (BENCHMARK_ROOT / task.output_name / "summary.json").read_text(encoding="utf-8")
            )["source_metadata"]
            for task in TASKS
        }
        validations = {
            task.key: _validate_cached_task(task, parent_rows) for task in TASKS
        }
    else:
        parent_rows = []
        source_metadata = {}
        validations = {}
        for task in TASKS:
            print(f"[starling-parent-provenance] reconstructing {task.output_name}", flush=True)
            task_rows, metadata, validation = _reconstruct_task(task)
            parent_rows.extend(task_rows)
            source_metadata[task.key] = metadata
            validations[task.key] = validation

    summary_rows = _summarize(parent_rows)
    histogram_rows = _histogram_rows(parent_rows)

    _write_csv(output_dir / "parent_stats.csv", parent_rows)
    _write_tsv(output_dir / "summary.tsv", summary_rows)
    _write_tsv(output_dir / "histogram_bins.tsv", histogram_rows)

    record_svg = figures_dir / "record_count_distribution.svg"
    record_png = figures_dir / "record_count_distribution_highres.png"
    pmid_svg = figures_dir / "unique_pmid_distribution.svg"
    pmid_png = figures_dir / "unique_pmid_distribution_highres.png"
    _plot_distribution(
        parent_rows,
        field="source_record_count",
        bins=RECORD_BINS,
        title="Accepted Starling records per parent",
        subtitle="Faceted by task; bars show within-status parent shares; count bins grow by powers of two",
        x_label="Accepted source records per parent",
        output=record_svg,
        png_output=record_png,
    )
    _plot_distribution(
        parent_rows,
        field="n_unique_pmids",
        bins=PMID_BINS,
        title="Unique publication sources per parent",
        subtitle="Publication sources are deduplicated by non-empty PMID; 0 denotes no PMID on accepted records",
        x_label="Unique PMIDs per parent",
        output=pmid_svg,
        png_output=pmid_png,
    )

    payload = {
        "analysis_version": "starling_parent_provenance.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "record_count": "accepted binary-labeled source records grouped by rdkit_fragment_parent.v1",
            "publication_count": "unique non-empty PMID values among those accepted source records",
            "strict_status": (
                "retained means unanimous records; conflict means both labels occur. "
                "This analysis stratum is distinct from the current 70% keep/reject decision."
            ),
        },
        "artifact_pmid_limit_note": (
            "Frozen molecule_labels/conflicting_molecules source_pmids arrays are capped at 50; "
            "this analysis reconstructs complete PMID sets from frozen source adapters."
        ),
        "validation": validations,
        "source_metadata": source_metadata,
        "summary": summary_rows,
        "paths": {
            "parent_stats": str(output_dir / "parent_stats.csv"),
            "summary_tsv": str(output_dir / "summary.tsv"),
            "histogram_bins": str(output_dir / "histogram_bins.tsv"),
            "record_figure_svg": str(record_svg),
            "record_figure_png": str(record_png),
            "pmid_figure_svg": str(pmid_svg),
            "pmid_figure_png": str(pmid_png),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        _render_report(summary_rows, validations),
        encoding="utf-8",
    )
    print(json.dumps(payload["paths"], ensure_ascii=False, indent=2), flush=True)
    return 0


def _reconstruct_task(
    task: TaskSpec,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    decisions, metadata = _load_task_decisions(task)
    aggregates: dict[str, ParentAggregate] = defaultdict(ParentAggregate)
    for decision in decisions:
        record = decision.record
        if record is None:
            continue
        identity = normalize_molecule_identity(record.smiles)
        if identity.status != "ok" or not identity.parent_smiles:
            continue
        identity_key = identity.parent_inchi_key or identity.parent_smiles
        parent = aggregates[identity_key]
        parent.labels.add(record.label)
        parent.source_record_count += 1
        parent.source_ids.add(record.source_id)
        pmid = _normalize_pmid(record.pmid)
        if pmid:
            parent.records_with_pmid += 1
            parent.pmids.add(pmid)

    artifact_rows = _read_artifact_rows(task)
    reconstructed_keys = set(aggregates)
    artifact_keys = set(artifact_rows)
    if reconstructed_keys != artifact_keys:
        raise AssertionError(
            f"{task.output_name}: reconstructed/artifact parent mismatch: "
            f"missing={len(artifact_keys - reconstructed_keys)} "
            f"extra={len(reconstructed_keys - artifact_keys)}"
        )

    rows: list[dict[str, Any]] = []
    record_count_mismatches = 0
    status_mismatches = 0
    artifact_pmid_prefix_mismatches = 0
    n_actual_pmid_sets_over_artifact_limit = 0
    for identity_key, parent in sorted(aggregates.items()):
        artifact = artifact_rows[identity_key]
        status = "retained" if len(parent.labels) == 1 else "conflict"
        if artifact["status"] != status:
            status_mismatches += 1
        if artifact["source_record_count"] != parent.source_record_count:
            record_count_mismatches += 1
        artifact_pmids = {_normalize_pmid(value) for value in artifact["source_pmids"]}
        artifact_pmids.discard("")
        if not artifact_pmids.issubset(parent.pmids):
            artifact_pmid_prefix_mismatches += 1
        if len(parent.pmids) > 50:
            n_actual_pmid_sets_over_artifact_limit += 1
        rows.append(
            {
                "task": task.output_name,
                "status": status,
                "molecule_identity_key": identity_key,
                "label_set": ",".join(str(value) for value in sorted(parent.labels)),
                "source_record_count": parent.source_record_count,
                "records_with_pmid": parent.records_with_pmid,
                "records_without_pmid": parent.source_record_count - parent.records_with_pmid,
                "n_unique_pmids": len(parent.pmids),
                "pmids": "|".join(sorted(parent.pmids)),
                "n_unique_source_ids": len(parent.source_ids),
                "source_ids": "|".join(sorted(parent.source_ids)),
            }
        )

    validation = {
        "status": "passed",
        "n_reconstructed_parents": len(reconstructed_keys),
        "n_artifact_parents": len(artifact_keys),
        "record_count_mismatches": record_count_mismatches,
        "status_mismatches": status_mismatches,
        "artifact_pmid_subset_mismatches": artifact_pmid_prefix_mismatches,
        "n_parents_with_more_than_50_unique_pmids": n_actual_pmid_sets_over_artifact_limit,
    }
    if any(
        validation[key]
        for key in (
            "record_count_mismatches",
            "status_mismatches",
            "artifact_pmid_subset_mismatches",
        )
    ):
        raise AssertionError(f"{task.output_name}: validation failed: {validation}")
    return rows, metadata, validation


def _load_task_decisions(
    task: TaskSpec,
) -> tuple[Iterable[LabelDecision], dict[str, Any]]:
    """Load frozen sources from local cache without requiring an HF API lookup."""
    if task.hf_revision is None:
        return task.loader()
    from huggingface_hub import HfApi

    offline_env = {"HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"}
    with (
        patch.dict(os.environ, offline_env),
        patch.object(
            HfApi,
            "dataset_info",
            return_value=SimpleNamespace(sha=task.hf_revision),
        ),
    ):
        return task.loader()


def _read_artifact_rows(task: TaskSpec) -> dict[str, dict[str, Any]]:
    task_dir = BENCHMARK_ROOT / task.output_name
    result: dict[str, dict[str, Any]] = {}
    for status, filename in (
        ("retained", "molecule_labels.jsonl"),
        ("conflict", "conflicting_molecules.jsonl"),
    ):
        with (task_dir / filename).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                result[row["molecule_identity_key"]] = {
                    "status": status,
                    "source_record_count": int(row["source_record_count"]),
                    "source_pmids": row.get("source_pmids") or [],
                }
    return result


def _read_cached_parent_stats(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"--reuse-parent-stats requires {path}")
    integer_fields = {
        "source_record_count",
        "records_with_pmid",
        "records_without_pmid",
        "n_unique_pmids",
        "n_unique_source_ids",
    }
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field_name in integer_fields:
            row[field_name] = int(row[field_name])
    return rows


def _validate_cached_task(
    task: TaskSpec, parent_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    artifact_rows = _read_artifact_rows(task)
    cached = {
        row["molecule_identity_key"]: row
        for row in parent_rows
        if row["task"] == task.output_name
    }
    record_count_mismatches = sum(
        int(cached[key]["source_record_count"]) != artifact["source_record_count"]
        for key, artifact in artifact_rows.items()
        if key in cached
    )
    status_mismatches = sum(
        cached[key]["status"] != artifact["status"]
        for key, artifact in artifact_rows.items()
        if key in cached
    )
    if set(cached) != set(artifact_rows) or record_count_mismatches or status_mismatches:
        raise AssertionError(f"{task.output_name}: cached parent_stats validation failed")
    return {
        "status": "passed_cached_full_provenance",
        "n_reconstructed_parents": len(cached),
        "n_artifact_parents": len(artifact_rows),
        "record_count_mismatches": record_count_mismatches,
        "status_mismatches": status_mismatches,
        "artifact_pmid_subset_mismatches": 0,
        "n_parents_with_more_than_50_unique_pmids": sum(
            int(row["n_unique_pmids"]) > 50 for row in cached.values()
        ),
    }


def _normalize_pmid(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def _summarize(parent_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for status in ("retained", "conflict"):
            selected = [
                row
                for row in parent_rows
                if row["task"] == task.output_name and row["status"] == status
            ]
            record_values = np.asarray(
                [row["source_record_count"] for row in selected], dtype=int
            )
            pmid_values = np.asarray([row["n_unique_pmids"] for row in selected], dtype=int)
            total_records = int(record_values.sum())
            records_with_pmid = sum(int(row["records_with_pmid"]) for row in selected)
            distinct_pmids = {
                pmid
                for row in selected
                for pmid in str(row["pmids"]).split("|")
                if pmid
            }
            rows.append(
                {
                    "task": task.output_name,
                    "status": status,
                    "n_parents": len(selected),
                    "total_source_records": total_records,
                    "records_per_parent_mean": _mean(record_values),
                    "records_per_parent_median": _quantile(record_values, 0.50),
                    "records_per_parent_p90": _quantile(record_values, 0.90),
                    "records_per_parent_p95": _quantile(record_values, 0.95),
                    "records_per_parent_p99": _quantile(record_values, 0.99),
                    "records_per_parent_max": int(record_values.max()),
                    "parents_with_one_record": int((record_values == 1).sum()),
                    "parents_with_multiple_records": int((record_values > 1).sum()),
                    "records_with_pmid": records_with_pmid,
                    "record_pmid_coverage_pct": _pct(records_with_pmid, total_records),
                    "distinct_pmids_across_group": len(distinct_pmids),
                    "unique_pmids_per_parent_mean": _mean(pmid_values),
                    "unique_pmids_per_parent_median": _quantile(pmid_values, 0.50),
                    "unique_pmids_per_parent_p90": _quantile(pmid_values, 0.90),
                    "unique_pmids_per_parent_p95": _quantile(pmid_values, 0.95),
                    "unique_pmids_per_parent_p99": _quantile(pmid_values, 0.99),
                    "unique_pmids_per_parent_max": int(pmid_values.max()),
                    "parents_with_zero_pmids": int((pmid_values == 0).sum()),
                    "parents_with_one_pmid": int((pmid_values == 1).sum()),
                    "parents_with_multiple_pmids": int((pmid_values > 1).sum()),
                }
            )
    return rows


def _histogram_rows(parent_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for task in TASKS:
        for status in ("retained", "conflict"):
            selected = [
                row
                for row in parent_rows
                if row["task"] == task.output_name and row["status"] == status
            ]
            for metric, field, bins in (
                ("source_record_count", "source_record_count", RECORD_BINS),
                ("unique_pmid_count", "n_unique_pmids", PMID_BINS),
            ):
                counts = _bin_counts([int(row[field]) for row in selected], bins)
                for (label, lower, upper), count in zip(bins, counts, strict=True):
                    output.append(
                        {
                            "task": task.output_name,
                            "status": status,
                            "metric": metric,
                            "bin": label,
                            "bin_lower": lower,
                            "bin_upper": "" if upper is None else upper,
                            "n_parents": count,
                            "parent_share_pct": _pct(count, len(selected)),
                            "status_group_n": len(selected),
                        }
                    )
    return output


def _plot_distribution(
    parent_rows: list[dict[str, Any]],
    *,
    field: str,
    bins: tuple[tuple[str, int, int | None], ...],
    title: str,
    subtitle: str,
    x_label: str,
    output: Path,
    png_output: Path,
) -> None:
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
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 10.8), sharex=True, sharey=True)
    fig.patch.set_facecolor(BG)
    x = np.arange(len(bins))
    width = 0.37
    styles = {
        "retained": {
            "label": "Retained binary parents",
            "facecolor": BLIND,
            "edgecolor": BLIND,
            "linewidth": 0.8,
        },
        "conflict": {
            "label": "Discarded conflict parents",
            "facecolor": "white",
            "edgecolor": VISIBLE,
            "linewidth": 1.5,
        },
    }

    for axis, task in zip(axes, TASKS, strict=True):
        axis.set_facecolor("white")
        for index, status in enumerate(("retained", "conflict")):
            selected = [
                int(row[field])
                for row in parent_rows
                if row["task"] == task.output_name and row["status"] == status
            ]
            counts = np.asarray(_bin_counts(selected, bins), dtype=float)
            percentages = counts / len(selected) * 100.0
            positions = x + (index - 0.5) * width
            bars = axis.bar(
                positions,
                percentages,
                width=width,
                label=styles[status]["label"],
                color=styles[status]["facecolor"],
                edgecolor=styles[status]["edgecolor"],
                linewidth=styles[status]["linewidth"],
                alpha=0.95,
                zorder=3,
                clip_on=False,
            )
            for bar, value in zip(bars, percentages, strict=True):
                if value >= 4.0:
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + 1.2,
                        f"{value:.0f}%",
                        ha="center",
                        va="bottom",
                        fontsize=8.5,
                        color=INK,
                    )
        retained_n = sum(
            row["task"] == task.output_name and row["status"] == "retained"
            for row in parent_rows
        )
        conflict_n = sum(
            row["task"] == task.output_name and row["status"] == "conflict"
            for row in parent_rows
        )
        axis.set_title(
            f"{task.display_name}  ·  retained n={retained_n:,}  ·  conflict n={conflict_n:,}",
            loc="left",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        axis.set_ylim(0, 105)
        axis.set_ylabel("Parents within group (%)", fontsize=10)
        axis.set_xticks(x, [item[0] for item in bins])
        axis.tick_params(axis="x", labelbottom=True)
        axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel(x_label, fontsize=11, labelpad=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.965, 0.955),
        frameon=False,
        ncol=2,
        fontsize=10,
    )
    fig.suptitle(title, x=0.075, y=0.985, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.075, 0.955, subtitle, ha="left", va="top", fontsize=10.5, color=MUTED)
    fig.text(
        0.075,
        0.012,
        "Source: current frozen Starling benchmark adapters and parent artifacts; normalization = rdkit_fragment_parent.v1.",
        ha="left",
        fontsize=9,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.075, right=0.97, top=0.90, bottom=0.09, hspace=0.33)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", facecolor=fig.get_facecolor())
    fig.savefig(
        png_output,
        format="png",
        dpi=200,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def _bin_counts(
    values: list[int], bins: tuple[tuple[str, int, int | None], ...]
) -> list[int]:
    return [
        sum(value >= lower and (upper is None or value <= upper) for value in values)
        for _, lower, upper in bins
    ]


def _quantile(values: np.ndarray, quantile: float) -> int:
    return int(np.quantile(values, quantile, method="higher"))


def _mean(values: np.ndarray) -> float:
    return float(values.mean())


def _pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _render_report(
    summary_rows: list[dict[str, Any]], validations: dict[str, Any]
) -> str:
    lines = [
        "# Starling parent record 与论文来源分布",
        "",
        "本分析按当前 frozen task adapter 重放 accepted binary records，并用",
        "`rdkit_fragment_parent.v1` 聚合。`retained` 表示 parent 的有效标签一致；",
        "`conflict` 表示同一 parent 同时存在 Y=0 与 Y=1，因而从 benchmark 丢弃。",
        "论文来源按每个 parent 的 unique non-empty PMID 计数。",
        "",
        "## 汇总",
        "",
        "| task | parent 状态 | parent 数 | record 总数 | record 中位数 / P90 / P99 / 最大值 | unique PMID 中位数 / P90 / P99 / 最大值 | PMID record 覆盖率 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    status_label = {"retained": "保留", "conflict": "冲突丢弃"}
    for row in summary_rows:
        lines.append(
            "| {task} | {status} | {n_parents:,} | {total_source_records:,} | "
            "{records_per_parent_median} / {records_per_parent_p90} / "
            "{records_per_parent_p99} / {records_per_parent_max} | "
            "{unique_pmids_per_parent_median} / {unique_pmids_per_parent_p90} / "
            "{unique_pmids_per_parent_p99} / {unique_pmids_per_parent_max} | "
            "{record_pmid_coverage_pct:.1f}% |".format(
                **{**row, "status": status_label[row["status"]]}
            )
        )
    lines.extend(
        [
            "",
            "## Provenance 与限制",
            "",
            "- benchmark JSONL 中的 `source_pmids` 最多保存 50 个；本分析从冻结 source adapter 重建完整 PMID 集合。",
            "- PMID=0 表示对应有效 records 没有提供可用 PMID，不代表一定没有论文来源。",
            "- `source_record_count` 是通过 task scope、单位、population、context 与 ambiguity gate 后的有效 records，不是原始 parquet/HF 行数。",
            "- 两张图展示状态组内占比，因此 retained/conflict 样本量不同不会直接决定柱高。",
            "",
            "## 校验",
            "",
        ]
    )
    for task, validation in validations.items():
        lines.append(
            f"- `{task}`：{validation['n_reconstructed_parents']:,} parents；"
            f"record-count mismatch={validation['record_count_mismatches']}；"
            f"status mismatch={validation['status_mismatches']}；"
            f"artifact PMID subset mismatch={validation['artifact_pmid_subset_mismatches']}。"
        )
    lines.extend(
        [
            "",
            "## 产物",
            "",
            "```text",
            "parent_stats.csv",
            "summary.tsv",
            "summary.json",
            "histogram_bins.tsv",
            "figures/record_count_distribution.svg",
            "figures/record_count_distribution_highres.png",
            "figures/unique_pmid_distribution.svg",
            "figures/unique_pmid_distribution_highres.png",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reuse-parent-stats",
        action="store_true",
        help="Regenerate summaries/figures from an already validated parent_stats.csv.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
