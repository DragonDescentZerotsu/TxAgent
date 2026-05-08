"""Re-score existing BBB Martins output files after rule changes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.export import write_csv, write_jsonl
from tools.chembl_tool.tasks.bbb_martins.report import write_report
from tools.chembl_tool.tasks.bbb_martins.scoring import scored_row
from tools.chembl_tool.tasks.bbb_martins.screen_assays import OUTPUT_FIELDS
from tools.chembl_tool.tasks.bbb_martins.summarize_outputs import summarize_rows


LIST_FIELDS = {
    "standard_types",
    "target_genes",
    "target_synonyms",
    "matched_keywords",
    "matched_endpoints",
    "matched_targets",
    "negative_flags",
    "weak_context_flags",
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_candidates = in_dir / "bbb_assay_candidates.csv"
    rows = _read_candidate_rows(input_candidates)
    rescored: list[dict[str, Any]] = []
    for row in rows:
        scored = scored_row(_row_for_scoring(row), min_score=args.min_score)
        if scored["keep_for_bbb_reasoning"]:
            output_row = dict(row)
            for field in OUTPUT_FIELDS:
                output_row.setdefault(field, "")
            output_row.update({field: scored.get(field) for field in OUTPUT_FIELDS if field in scored})
            rescored.append(output_row)

    rescored.sort(key=lambda row: (str(row["tier"]), -int(row["score"]), str(row["assay_chembl_id"])))
    write_csv(out_dir / "bbb_assay_candidates.csv", rescored, OUTPUT_FIELDS)
    write_jsonl(out_dir / "bbb_assay_candidates.jsonl", rescored)
    write_report(
        out_dir / "bbb_assay_report.md",
        total_assays=args.total_assays,
        candidates=rescored,
        min_score=args.min_score,
        chembl_sqlite=args.chembl_sqlite,
    )
    (out_dir / "bbb_health_check.md").write_text(
        "\n".join(summarize_rows(_rows_for_summary(rescored), candidates_path=out_dir / "bbb_assay_candidates.csv")) + "\n",
        encoding="utf-8",
    )

    activity_in = in_dir / "bbb_activity_evidence.csv"
    if args.filter_activities and activity_in.exists():
        _filter_activity_evidence(
            activity_in,
            out_dir / "bbb_activity_evidence.csv",
            {str(row["assay_chembl_id"]) for row in rescored},
        )

    print(f"input_candidates={len(rows)}")
    print(f"rescored_candidates={len(rescored)}")
    print(f"out_dir={out_dir}")
    return 0


def _read_candidate_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_for_scoring(row: dict[str, str]) -> dict[str, Any]:
    converted: dict[str, Any] = dict(row)
    for field in LIST_FIELDS:
        converted[field] = _split_multi(row.get(field, ""))
    converted["confidence_score"] = _as_int(row.get("confidence_score"))
    converted["n_unique_molecules"] = _as_int(row.get("n_unique_molecules"))
    converted["n_activities"] = _as_int(row.get("n_activities"))
    converted.setdefault("component_descriptions", [])
    return converted


def _rows_for_summary(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if isinstance(value, list):
                item[key] = "|".join(str(v) for v in value)
            else:
                item[key] = "" if value is None else str(value)
        out.append(item)
    return out


def _filter_activity_evidence(input_path: Path, output_path: Path, assay_ids: set[str]) -> int:
    count = 0
    with input_path.open(newline="", encoding="utf-8") as src, output_path.open("w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            return 0
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row.get("assay_chembl_id") in assay_ids:
                writer.writerow(row)
                count += 1
    return count


def _split_multi(value: object) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def _as_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", default="outputs/chembl_bbb")
    parser.add_argument("--out-dir", default="outputs/chembl_bbb_cleaned")
    parser.add_argument("--min-score", type=int, default=40)
    parser.add_argument("--total-assays", type=int, default=1890749)
    parser.add_argument("--chembl-sqlite", default="tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db")
    parser.add_argument("--filter-activities", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
