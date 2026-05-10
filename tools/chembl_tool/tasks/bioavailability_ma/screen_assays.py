"""CLI for screening ChEMBL assays relevant to oral bioavailability reasoning."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.assay_loader import (
    iter_assay_metadata,
    load_activity_summary,
    load_target_annotations,
    merge_assay_record,
)
from tools.chembl_tool.common.export import ensure_dir, write_csv, write_jsonl
from tools.chembl_tool.common.sqlite import connect_sqlite
from tools.chembl_tool.tasks.bioavailability_ma.report import write_report
from tools.chembl_tool.tasks.bioavailability_ma.scoring import scored_row


OUTPUT_FIELDS = [
    "assay_chembl_id",
    "assay_id",
    "tier",
    "score",
    "assay_type",
    "description",
    "target_chembl_id",
    "target_pref_name",
    "target_genes",
    "target_synonyms",
    "organism",
    "confidence_score",
    "relationship_type",
    "assay_cell_type",
    "assay_tissue",
    "n_activities",
    "n_unique_molecules",
    "standard_types",
    "matched_keywords",
    "matched_endpoints",
    "matched_targets",
    "negative_flags",
    "weak_context_flags",
    "reason",
    "keep_for_bioavailability_reasoning",
]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = ensure_dir(args.out_dir)
    run_start = time.monotonic()
    _log_stage("Opening ChEMBL SQLite database")
    conn = connect_sqlite(args.chembl_sqlite)
    try:
        total_available = _count_assays(conn)
        total_to_scan = min(total_available, args.limit) if args.limit else total_available
        _log_stage(f"Preparing scan scope: total_assays={total_available:,}, scan_assays={total_to_scan:,}")
        limit_assay_ids = _limit_assay_ids(conn, args.limit) if args.limit else None

        activity_start = time.monotonic()
        _log_stage("Loading activity summaries")
        activity_summaries = load_activity_summary(conn, assay_ids=limit_assay_ids)
        _log_done(
            "Loaded activity summaries",
            activity_start,
            f"assays_with_activities={len(activity_summaries):,}",
        )

        target_start = time.monotonic()
        _log_stage("Loading target annotations")
        target_annotations = load_target_annotations(conn)
        _log_done(
            "Loaded target annotations",
            target_start,
            f"targets_with_annotations={len(target_annotations):,}",
        )

        scan_start = time.monotonic()
        _log_stage("Scanning assays and applying oral bioavailability rules")
        total_assays = 0
        candidates: list[dict[str, Any]] = []
        candidate_assay_ids: list[int] = []
        for metadata in iter_assay_metadata(conn):
            total_assays += 1
            assay_id = int(metadata["assay_id"])
            row = merge_assay_record(
                metadata,
                activity_summaries.get(assay_id),
                target_annotations.get(int(metadata["tid"])) if metadata.get("tid") is not None else None,
            )
            row["organism"] = row.get("target_organism") or row.get("assay_organism")
            scored = scored_row(row, min_score=args.min_score)
            if scored["keep_for_bioavailability_reasoning"]:
                candidates.append(_candidate_output_row(scored))
                candidate_assay_ids.append(assay_id)
            if args.progress_every and total_assays % args.progress_every == 0:
                _log_progress(total_assays, total_to_scan, len(candidates), scan_start)
            if args.limit and total_assays >= args.limit:
                break

        _log_progress(total_assays, total_to_scan, len(candidates), scan_start, final=True)

        export_start = time.monotonic()
        _log_stage("Writing output files")
        candidates.sort(key=lambda row: (str(row["tier"]), -int(row["score"]), str(row["assay_chembl_id"])))
        csv_path = out_dir / "bioavailability_assay_candidates.csv"
        jsonl_path = out_dir / "bioavailability_assay_candidates.jsonl"
        report_path = out_dir / "bioavailability_assay_report.md"
        write_csv(csv_path, candidates, OUTPUT_FIELDS)
        write_jsonl(jsonl_path, candidates)
        write_report(
            report_path,
            total_assays=total_assays,
            candidates=candidates,
            min_score=args.min_score,
            chembl_sqlite=args.chembl_sqlite,
        )
        if args.export_activities:
            _log_stage("Exporting candidate activity evidence")
            export_activity_evidence(conn, candidate_assay_ids, out_dir / "bioavailability_activity_evidence.csv")
        _log_done("Wrote output files", export_start, f"out_dir={out_dir}")

        _log_done(
            "Finished oral bioavailability assay screening",
            run_start,
            f"scanned={total_assays:,} retained={len(candidates):,}",
        )
        print(f"[output] {csv_path}", file=sys.stderr)
        print(f"[output] {jsonl_path}", file=sys.stderr)
        print(f"[output] {report_path}", file=sys.stderr)
    finally:
        conn.close()
    return 0


def export_activity_evidence(conn: sqlite3.Connection, assay_ids: list[int], path: Path) -> int:
    if not assay_ids:
        path.write_text("", encoding="utf-8")
        return 0
    fields = [
        "assay_chembl_id",
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "data_validity_comment",
        "activity_comment",
    ]
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for chunk in _chunks(sorted(set(assay_ids)), 500):
            placeholders = ",".join("?" for _ in chunk)
            query = f"""
                SELECT
                  a.chembl_id AS assay_chembl_id,
                  md.chembl_id AS molecule_chembl_id,
                  cs.canonical_smiles,
                  act.standard_type,
                  act.standard_relation,
                  act.standard_value,
                  act.standard_units,
                  act.pchembl_value,
                  act.data_validity_comment,
                  act.activity_comment
                FROM activities act
                LEFT JOIN assays a ON act.assay_id = a.assay_id
                LEFT JOIN molecule_dictionary md ON act.molregno = md.molregno
                LEFT JOIN compound_structures cs ON act.molregno = cs.molregno
                WHERE act.assay_id IN ({placeholders})
            """
            for row in conn.execute(query, chunk):
                writer.writerow({field: row[field] for field in fields})
                count += 1
    return count


def _candidate_output_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in OUTPUT_FIELDS}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-sqlite", required=True, help="Path to ChEMBL SQLite database.")
    parser.add_argument("--out-dir", default="outputs/chembl_tool/tasks/bioavailability_ma/assay_screening/raw", help="Output directory.")
    parser.add_argument("--min-score", type=int, default=40, help="Minimum score to retain an assay.")
    parser.add_argument("--export-activities", action="store_true", help="Export candidate activity evidence as CSV.")
    parser.add_argument("--limit", type=int, default=0, help="Optional assay scan limit for smoke tests.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Print scan progress every N assays. Set 0 to disable.",
    )
    return parser.parse_args(argv)


def _count_assays(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM assays").fetchone()
    return int(row["n"] or 0)


def _limit_assay_ids(conn: sqlite3.Connection, limit: int) -> list[int]:
    return [
        int(row["assay_id"])
        for row in conn.execute("SELECT assay_id FROM assays ORDER BY assay_id LIMIT ?", (limit,))
    ]


def _chunks(values: list[int], size: int) -> list[list[int]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _log_stage(message: str) -> None:
    print(f"[stage] {message}", file=sys.stderr, flush=True)


def _log_done(message: str, start: float, extra: str = "") -> None:
    suffix = f" {extra}" if extra else ""
    print(f"[done] {message} elapsed={_format_elapsed(time.monotonic() - start)}{suffix}", file=sys.stderr, flush=True)


def _log_progress(
    scanned: int,
    total: int,
    retained: int,
    start: float,
    *,
    final: bool = False,
) -> None:
    elapsed = max(time.monotonic() - start, 1e-9)
    rate = scanned / elapsed
    if total:
        pct = 100.0 * scanned / total
        prefix = "[progress-final]" if final else "[progress]"
        print(
            f"{prefix} scanned={scanned:,}/{total:,} ({pct:.2f}%) retained={retained:,} "
            f"rate={rate:,.1f}/s elapsed={_format_elapsed(elapsed)}",
            file=sys.stderr,
            flush=True,
        )
    else:
        prefix = "[progress-final]" if final else "[progress]"
        print(
            f"{prefix} scanned={scanned:,} retained={retained:,} "
            f"rate={rate:,.1f}/s elapsed={_format_elapsed(elapsed)}",
            file=sys.stderr,
            flush=True,
        )


def _format_elapsed(seconds: float) -> str:
    seconds_int = int(seconds)
    hours, rem = divmod(seconds_int, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{seconds:.1f}s"


if __name__ == "__main__":
    raise SystemExit(main())
