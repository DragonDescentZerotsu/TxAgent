"""Summarize molecule-count distributions for all ChEMBL assay endpoints."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CHEMBL_SQLITE = "tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db"
DEFAULT_OUT_DIR = "outputs/chembl_tool/chembl_endpoint_molecule_distribution"

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
    ("1000-4999", 1000, 4999),
    ("5000-9999", 5000, 9999),
    ("10000+", 10000, None),
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    db_path = Path(args.chembl_sqlite)
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assay_tsv = out_dir / "all_chembl_assay_molecule_counts.tsv"
    endpoint_tsv = out_dir / "all_chembl_assay_standard_type_endpoint_molecule_counts.tsv"
    summary_json = out_dir / "summary.json"
    histogram_svg = out_dir / "molecule_count_histograms.svg"
    report_md = out_dir / "report.md"

    conn = sqlite3.connect(str(db_path))
    try:
        _prepare_connection(conn)
        print("querying assay-level molecule counts")
        assay_rows = _query_assay_counts(conn)
        print(f"assay rows={len(assay_rows):,}")
        print("querying assay_id + standard_type endpoint molecule counts")
        endpoint_rows = _query_endpoint_counts(conn)
        print(f"endpoint rows={len(endpoint_rows):,}")
    finally:
        conn.close()

    _write_tsv(assay_tsv, assay_rows)
    _write_tsv(endpoint_tsv, endpoint_rows)

    summary = {
        "chembl_sqlite": str(db_path),
        "filters": {
            "molregno": "not null",
            "assay_id": "not null",
            "task_specific_screening": "not applied",
        },
        "definitions": {
            "assay": "one ChEMBL assay_id",
            "assay_standard_type_endpoint": "one ChEMBL assay_id plus normalized lower-case standard_type",
        },
        "assay": _summarize_rows(assay_rows, "n_unique_molecules"),
        "assay_standard_type_endpoint": _summarize_rows(endpoint_rows, "n_unique_molecules"),
        "top_assays": assay_rows[:20],
        "top_assay_standard_type_endpoints": endpoint_rows[:20],
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_histogram_svg(
        histogram_svg,
        {
            "ChEMBL assays": [int(row["n_unique_molecules"]) for row in assay_rows],
            "ChEMBL assay + standard_type endpoints": [
                int(row["n_unique_molecules"]) for row in endpoint_rows
            ],
        },
    )
    _write_report(report_md, summary, assay_tsv, endpoint_tsv, histogram_svg)

    print(f"wrote {assay_tsv}")
    print(f"wrote {endpoint_tsv}")
    print(f"wrote {summary_json}")
    print(f"wrote {histogram_svg}")
    print(f"wrote {report_md}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-sqlite", default=DEFAULT_CHEMBL_SQLITE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def _prepare_connection(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA temp_store = FILE")
    conn.execute("PRAGMA cache_size = -200000")


def _query_assay_counts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    query = """
        SELECT
          a.assay_id,
          ass.chembl_id AS assay_chembl_id,
          COUNT(*) AS n_activities,
          COUNT(DISTINCT a.molregno) AS n_unique_molecules,
          COUNT(DISTINCT COALESCE(NULLIF(LOWER(TRIM(a.standard_type)), ''), '[missing]')) AS n_standard_types,
          ass.assay_type,
          ass.description
        FROM activities a
        LEFT JOIN assays ass ON a.assay_id = ass.assay_id
        WHERE a.assay_id IS NOT NULL
          AND a.molregno IS NOT NULL
        GROUP BY a.assay_id
        ORDER BY n_unique_molecules DESC, n_activities DESC, a.assay_id
    """
    return [dict(row) for row in conn.execute(query)]


def _query_endpoint_counts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    query = """
        SELECT
          a.assay_id,
          ass.chembl_id AS assay_chembl_id,
          COALESCE(NULLIF(LOWER(TRIM(a.standard_type)), ''), '[missing]') AS standard_type,
          COUNT(*) AS n_activities,
          COUNT(DISTINCT a.molregno) AS n_unique_molecules,
          COUNT(DISTINCT COALESCE(NULLIF(LOWER(TRIM(a.standard_units)), ''), '[missing]')) AS n_standard_units,
          ass.assay_type,
          ass.description
        FROM activities a
        LEFT JOIN assays ass ON a.assay_id = ass.assay_id
        WHERE a.assay_id IS NOT NULL
          AND a.molregno IS NOT NULL
        GROUP BY a.assay_id, COALESCE(NULLIF(LOWER(TRIM(a.standard_type)), ''), '[missing]')
        ORDER BY n_unique_molecules DESC, n_activities DESC, a.assay_id, standard_type
    """
    return [dict(row) for row in conn.execute(query)]


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summarize_rows(rows: list[dict[str, Any]], count_key: str) -> dict[str, Any]:
    counts = [int(row[count_key]) for row in rows if int(row[count_key]) > 0]
    return {
        **_distribution_summary(counts),
        "histogram_bins": _histogram_bins(counts),
    }


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
        "p90": _percentile(sorted_counts, 0.90),
        "p95": _percentile(sorted_counts, 0.95),
        "p99": _percentile(sorted_counts, 0.99),
        "max": sorted_counts[-1],
        "mean": round(sum(sorted_counts) / len(sorted_counts), 2),
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
    bins = Counter({label: 0 for label, _low, _high in COUNT_BINS})
    for count in counts:
        for label, low, high in COUNT_BINS:
            if count >= low and (high is None or count <= high):
                bins[label] += 1
                break
    return dict(bins)


def _write_histogram_svg(path: Path, title_to_counts: dict[str, list[int]]) -> None:
    panel_w = 660
    panel_h = 300
    width = panel_w * max(1, len(title_to_counts))
    height = panel_h + 80
    margin_l = 64
    margin_r = 18
    margin_t = 46
    margin_b = 76
    plot_w = panel_w - margin_l - margin_r
    plot_h = panel_h - margin_t - margin_b
    colors = ["#4C78A8", "#F58518"]
    labels = [label for label, _low, _high in COUNT_BINS]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;font-size:12px;fill:#222}.title{font-size:18px;font-weight:700}.panel-title{font-size:13px;font-weight:700}.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.bar-label{font-size:10px;fill:#333}</style>",
        f'<text class="title" x="{width / 2:.1f}" y="24" text-anchor="middle">ChEMBL molecule-count distributions</text>',
    ]
    for panel_idx, (title, counts) in enumerate(title_to_counts.items()):
        x0 = panel_idx * panel_w
        bins = _histogram_bins(counts)
        max_y = max(1, max(bins.values()))
        ticks = _nice_ticks(max_y)
        color = colors[panel_idx % len(colors)]
        parts.append(f'<g transform="translate({x0},56)">')
        parts.append(
            f'<text class="panel-title" x="{panel_w / 2:.1f}" y="-12" text-anchor="middle">'
            f'{html.escape(title)} (n={len(counts):,})</text>'
        )
        for tick in ticks:
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
        for i, label in enumerate(labels):
            value = bins[label]
            bar_h = 0 if value == 0 else max(1, (value / max_y) * plot_h)
            x = margin_l + i * (bar_w + bar_gap)
            y = margin_t + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
            if value:
                parts.append(
                    f'<text class="bar-label" x="{x + bar_w / 2:.1f}" y="{y - 3:.1f}" '
                    f'text-anchor="middle">{value:,}</text>'
                )
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{margin_t + plot_h + 19}" '
                f'text-anchor="middle" transform="rotate(45 {x + bar_w / 2:.1f} {margin_t + plot_h + 19})">'
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
    assay_tsv: Path,
    endpoint_tsv: Path,
    histogram_svg: Path,
) -> None:
    assay = summary["assay"]
    endpoint = summary["assay_standard_type_endpoint"]
    lines = [
        "# ChEMBL Endpoint Molecule Distribution",
        "",
        "This report uses all ChEMBL activities with non-null `assay_id` and `molregno`; no task-specific assay screening is applied.",
        "",
        "Definitions:",
        "",
        "- `assay`: one ChEMBL `assay_id`.",
        "- `assay_standard_type_endpoint`: one ChEMBL `assay_id` plus normalized lower-case `standard_type`.",
        "",
        f"- Assay table: `{assay_tsv.name}`",
        f"- Endpoint table: `{endpoint_tsv.name}`",
        f"- Histogram: `{histogram_svg.name}`",
        "",
        "## Summary",
        "",
        "| level | n | median | p75 | p90 | p95 | p99 | max | mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _summary_row("assay", assay),
        _summary_row("assay + standard_type endpoint", endpoint),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _summary_row(label: str, summary: dict[str, Any]) -> str:
    return (
        f"| {label} | {summary['n']} | {summary['median']} | {summary['p75']} | "
        f"{summary['p90']} | {summary['p95']} | {summary['p99']} | "
        f"{summary['max']} | {summary['mean']} |"
    )


if __name__ == "__main__":
    raise SystemExit(main())
