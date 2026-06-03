"""Summarize molecule-count distributions for task-selected ChEMBL endpoints."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = "outputs/chembl_tool/task_endpoint_molecule_counts"


@dataclass(frozen=True)
class TaskInput:
    task: str
    evidence_jsonl: Path
    assay_candidates_csv: Path


DEFAULT_TASKS = (
    TaskInput(
        "bbb_martins",
        Path("outputs/chembl_tool/tasks/bbb_martins/evidence_library/bbb_molecule_evidence.jsonl"),
        Path("outputs/chembl_tool/tasks/bbb_martins/assay_screening/v6/bbb_assay_candidates.csv"),
    ),
    TaskInput(
        "bioavailability_ma",
        Path("outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/bioavailability_molecule_evidence.jsonl"),
        Path("outputs/chembl_tool/tasks/bioavailability_ma/assay_screening/v4/bioavailability_assay_candidates.csv"),
    ),
    TaskInput(
        "clintox",
        Path("outputs/chembl_tool/tasks/clintox/evidence_library/clintox_molecule_evidence.jsonl"),
        Path("outputs/chembl_tool/tasks/clintox/assay_screening/v6/clintox_assay_candidates.csv"),
    ),
    TaskInput(
        "skin_reaction",
        Path("outputs/chembl_tool/tasks/skin_reaction/evidence_library/skin_reaction_molecule_evidence.jsonl"),
        Path("outputs/chembl_tool/tasks/skin_reaction/assay_screening/v1/skin_reaction_assay_candidates.csv"),
    ),
)


COUNT_BINS = (
    ("1", 1, 1),
    ("2-4", 2, 4),
    ("5-9", 5, 9),
    ("10-19", 10, 19),
    ("20-49", 20, 49),
    ("50-99", 50, 99),
    ("100-199", 100, 199),
    ("200-499", 200, 499),
    ("500-999", 500, 999),
    ("1000+", 1000, None),
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = list(DEFAULT_TASKS)
    group_rows: list[dict[str, Any]] = []
    assay_rows: list[dict[str, Any]] = []

    for task_input in tasks:
        _validate_input(task_input)
        group_rows.extend(_summarize_endpoint_groups(task_input))
        assay_rows.extend(_summarize_assay_endpoints(task_input))

    group_tsv = out_dir / "endpoint_group_molecule_counts.tsv"
    assay_tsv = out_dir / "assay_endpoint_molecule_counts.tsv"
    summary_json = out_dir / "summary.json"
    group_svg = out_dir / "endpoint_group_molecule_count_histogram.svg"
    assay_svg = out_dir / "assay_endpoint_molecule_count_histogram.svg"
    report_md = out_dir / "report.md"

    _write_tsv(group_tsv, group_rows)
    _write_tsv(assay_tsv, assay_rows)
    summary = _build_summary(group_rows, assay_rows)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_histogram_svg(
        group_svg,
        "Endpoint-group molecule-count distribution",
        _counts_by_task(group_rows, "n_unique_molecules"),
        unit_label="endpoint groups",
    )
    _write_histogram_svg(
        assay_svg,
        "Selected-assay molecule-count distribution",
        _counts_by_task(assay_rows, "n_unique_molecules_candidate"),
        unit_label="assay endpoints",
    )
    _write_report(report_md, summary, group_tsv, assay_tsv, group_svg, assay_svg)

    print(f"wrote {group_tsv}")
    print(f"wrote {assay_tsv}")
    print(f"wrote {summary_json}")
    print(f"wrote {group_svg}")
    print(f"wrote {assay_svg}")
    print(f"wrote {report_md}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def _validate_input(task_input: TaskInput) -> None:
    missing = [
        str(path)
        for path in (task_input.evidence_jsonl, task_input.assay_candidates_csv)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing input for {task_input.task}: {', '.join(missing)}")


def _summarize_endpoint_groups(task_input: TaskInput) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    with task_input.evidence_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            group_id = str(row.get("group_id") or "").strip()
            molecule_id = str(row.get("molecule_chembl_id") or "").strip()
            if not group_id or not molecule_id:
                continue
            item = groups.setdefault(
                group_id,
                {
                    "task": task_input.task,
                    "group_id": group_id,
                    "assay_tier": row.get("assay_tier", ""),
                    "endpoint_group": row.get("endpoint_group", ""),
                    "molecules": set(),
                    "assays": set(),
                    "activity_rows": 0,
                },
            )
            item["molecules"].add(molecule_id)
            if row.get("assay_chembl_id"):
                item["assays"].add(str(row["assay_chembl_id"]))
            item["activity_rows"] += 1

    rows: list[dict[str, Any]] = []
    for item in groups.values():
        rows.append(
            {
                "task": item["task"],
                "group_id": item["group_id"],
                "assay_tier": item["assay_tier"],
                "endpoint_group": item["endpoint_group"],
                "n_unique_molecules": len(item["molecules"]),
                "n_assays": len(item["assays"]),
                "n_activity_rows": item["activity_rows"],
            }
        )
    return sorted(rows, key=lambda row: (row["task"], -int(row["n_unique_molecules"]), row["group_id"]))


def _summarize_assay_endpoints(task_input: TaskInput) -> list[dict[str, Any]]:
    evidence_molecules: dict[str, set[str]] = defaultdict(set)
    evidence_rows: dict[str, int] = defaultdict(int)
    with task_input.evidence_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            assay_id = str(row.get("assay_chembl_id") or "").strip()
            molecule_id = str(row.get("molecule_chembl_id") or "").strip()
            if assay_id and molecule_id:
                evidence_molecules[assay_id].add(molecule_id)
                evidence_rows[assay_id] += 1

    rows: list[dict[str, Any]] = []
    with task_input.assay_candidates_csv.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            assay_id = str(row.get("assay_chembl_id") or "").strip()
            if not assay_id:
                continue
            rows.append(
                {
                    "task": task_input.task,
                    "assay_chembl_id": assay_id,
                    "tier": row.get("tier", ""),
                    "score": row.get("score", ""),
                    "n_unique_molecules_candidate": _as_int(row.get("n_unique_molecules")),
                    "n_activities_candidate": _as_int(row.get("n_activities")),
                    "n_unique_molecules_in_evidence": len(evidence_molecules.get(assay_id, set())),
                    "n_activity_rows_in_evidence": evidence_rows.get(assay_id, 0),
                    "standard_types": row.get("standard_types", ""),
                    "assay_type": row.get("assay_type", ""),
                    "target_pref_name": row.get("target_pref_name", ""),
                    "description": row.get("description", ""),
                }
            )
    return sorted(
        rows,
        key=lambda row: (row["task"], -int(row["n_unique_molecules_candidate"]), row["assay_chembl_id"]),
    )


def _as_int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(group_rows: list[dict[str, Any]], assay_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for task in sorted({row["task"] for row in group_rows} | {row["task"] for row in assay_rows}):
        group_counts = [
            int(row["n_unique_molecules"])
            for row in group_rows
            if row["task"] == task and int(row["n_unique_molecules"]) > 0
        ]
        assay_counts = [
            int(row["n_unique_molecules_candidate"])
            for row in assay_rows
            if row["task"] == task and int(row["n_unique_molecules_candidate"]) > 0
        ]
        task_groups = [row for row in group_rows if row["task"] == task]
        task_assays = [row for row in assay_rows if row["task"] == task]
        summary[task] = {
            "endpoint_groups": _distribution_summary(group_counts),
            "assay_endpoints": _distribution_summary(assay_counts),
            "total_unique_molecules_in_evidence_library": _total_unique_molecules_for_task(task_groups),
            "n_selected_assays": len(task_assays),
            "largest_endpoint_groups": task_groups[:10],
            "largest_assay_endpoints": task_assays[:10],
        }
    return summary


def _distribution_summary(counts: list[int]) -> dict[str, Any]:
    if not counts:
        return {"n": 0}
    sorted_counts = sorted(counts)
    return {
        "n": len(sorted_counts),
        "min": sorted_counts[0],
        "p25": _percentile(sorted_counts, 0.25),
        "median": _percentile(sorted_counts, 0.50),
        "p75": _percentile(sorted_counts, 0.75),
        "max": sorted_counts[-1],
        "mean": round(sum(sorted_counts) / len(sorted_counts), 2),
        "histogram_bins": _histogram_bins(sorted_counts),
    }


def _percentile(sorted_counts: list[int], q: float) -> float:
    if len(sorted_counts) == 1:
        return float(sorted_counts[0])
    pos = (len(sorted_counts) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_counts[lo])
    return round(sorted_counts[lo] + (sorted_counts[hi] - sorted_counts[lo]) * (pos - lo), 2)


def _histogram_bins(counts: list[int]) -> dict[str, int]:
    bins = {label: 0 for label, _low, _high in COUNT_BINS}
    for count in counts:
        for label, low, high in COUNT_BINS:
            if count >= low and (high is None or count <= high):
                bins[label] += 1
                break
    return bins


def _total_unique_molecules_for_task(group_rows: list[dict[str, Any]]) -> str:
    # Exact task-wide unique molecule totals are available from the evidence-library index;
    # this table stores per-group counts, where molecules can appear in multiple groups.
    return "see evidence library metadata"


def _counts_by_task(rows: list[dict[str, Any]], count_key: str) -> dict[str, list[int]]:
    counts_by_task: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        count = int(row.get(count_key) or 0)
        if count > 0:
            counts_by_task[str(row["task"])].append(count)
    return dict(sorted(counts_by_task.items()))


def _write_histogram_svg(
    path: Path,
    title: str,
    counts_by_task: dict[str, list[int]],
    *,
    unit_label: str,
) -> None:
    panel_w = 520
    panel_h = 250
    margin_l = 58
    margin_r = 18
    margin_t = 42
    margin_b = 56
    gap = 28
    width = panel_w * max(1, len(counts_by_task))
    height = panel_h + 72
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;font-size:12px;fill:#222}.title{font-size:18px;font-weight:700}.panel-title{font-size:13px;font-weight:700}.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.bar-label{font-size:10px;fill:#333}</style>",
        f'<text class="title" x="{width / 2:.1f}" y="24" text-anchor="middle">{html.escape(title)}</text>',
    ]
    plot_w = panel_w - margin_l - margin_r
    plot_h = panel_h - margin_t - margin_b
    labels = [label for label, _low, _high in COUNT_BINS]
    for panel_idx, (task, counts) in enumerate(counts_by_task.items()):
        x0 = panel_idx * panel_w
        bins = _histogram_bins(counts)
        max_y = max(1, max(bins.values()))
        y_ticks = _nice_ticks(max_y)
        parts.append(f'<g transform="translate({x0},52)">')
        parts.append(
            f'<text class="panel-title" x="{panel_w / 2:.1f}" y="-12" text-anchor="middle">'
            f'{html.escape(task)} ({len(counts)} {html.escape(unit_label)})</text>'
        )
        for tick in y_ticks:
            y = margin_t + plot_h - (tick / max_y) * plot_h
            parts.append(f'<line class="grid" x1="{margin_l}" y1="{y:.1f}" x2="{margin_l + plot_w}" y2="{y:.1f}"/>')
            parts.append(f'<text x="{margin_l - 8}" y="{y + 4:.1f}" text-anchor="end">{tick}</text>')
        parts.append(f'<line class="axis" x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}"/>')
        parts.append(
            f'<line class="axis" x1="{margin_l}" y1="{margin_t + plot_h}" '
            f'x2="{margin_l + plot_w}" y2="{margin_t + plot_h}"/>'
        )
        bar_gap = 5
        bar_w = (plot_w - bar_gap * (len(labels) - 1)) / len(labels)
        color = colors[panel_idx % len(colors)]
        for i, label in enumerate(labels):
            value = bins[label]
            bar_h = 0 if value == 0 else max(1, (value / max_y) * plot_h)
            x = margin_l + i * (bar_w + bar_gap)
            y = margin_t + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
            if value:
                parts.append(
                    f'<text class="bar-label" x="{x + bar_w / 2:.1f}" y="{y - 3:.1f}" '
                    f'text-anchor="middle">{value}</text>'
                )
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{margin_t + plot_h + 17}" '
                f'text-anchor="middle" transform="rotate(45 {x + bar_w / 2:.1f} {margin_t + plot_h + 17})">'
                f'{html.escape(label)}</text>'
            )
        parts.append("</g>")
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _nice_ticks(max_y: int) -> list[int]:
    if max_y <= 5:
        return list(range(0, max_y + 1))
    step = max(1, math.ceil(max_y / 4))
    return list(range(0, max_y + step, step))


def _write_report(
    path: Path,
    summary: dict[str, Any],
    group_tsv: Path,
    assay_tsv: Path,
    group_svg: Path,
    assay_svg: Path,
) -> None:
    lines = [
        "# Task Endpoint Molecule Counts",
        "",
        "This report summarizes molecule-count distributions for the selected ChEMBL evidence used by the task pipelines.",
        "",
        "Definitions:",
        "",
        "- `endpoint_group`: the Tier.endpoint_group bucket used by reasoning retrieval.",
        "- `assay_endpoint`: one selected ChEMBL assay candidate; molecule counts come from the candidate CSV `n_unique_molecules` field.",
        "",
        f"- Endpoint-group table: `{group_tsv.name}`",
        f"- Assay-endpoint table: `{assay_tsv.name}`",
        f"- Endpoint-group histogram: `{group_svg.name}`",
        f"- Assay-endpoint histogram: `{assay_svg.name}`",
        "",
        "## Summary",
        "",
        "| task | endpoint groups | group median | group max | selected assays | assay median | assay max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, task_summary in summary.items():
        group = task_summary["endpoint_groups"]
        assay = task_summary["assay_endpoints"]
        lines.append(
            "| "
            + " | ".join(
                [
                    task,
                    str(group.get("n", 0)),
                    str(group.get("median", "")),
                    str(group.get("max", "")),
                    str(task_summary.get("n_selected_assays", 0)),
                    str(assay.get("median", "")),
                    str(assay.get("max", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
