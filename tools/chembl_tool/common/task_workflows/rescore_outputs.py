"""Shared re-scoring CLI for assay-screening outputs."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.export import write_csv, write_jsonl


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


@dataclass(frozen=True)
class RescoreConfig:
    description: str
    default_in_dir: str
    default_out_dir: str
    candidates_filename: str
    candidates_jsonl_filename: str
    report_filename: str
    health_check_filename: str
    activity_evidence_filename: str
    keep_field: str
    output_fields: tuple[str, ...]
    scored_row: Callable[[dict[str, Any]], dict[str, Any]]
    write_report: Callable[..., None]
    summarize_rows: Callable[..., list[str]]


def main(config: RescoreConfig, argv: list[str] | None = None) -> int:
    args = _parse_args(config, argv)
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_candidates = in_dir / config.candidates_filename
    rows = _read_candidate_rows(input_candidates)
    rescored: list[dict[str, Any]] = []
    for row in rows:
        scored = config.scored_row(_row_for_scoring(row), min_score=args.min_score)
        if scored[config.keep_field]:
            output_row = dict(row)
            for field in config.output_fields:
                output_row.setdefault(field, "")
            output_row.update({field: scored.get(field) for field in config.output_fields if field in scored})
            rescored.append(output_row)

    rescored.sort(key=lambda row: (str(row["tier"]), -int(row["score"]), str(row["assay_chembl_id"])))
    write_csv(out_dir / config.candidates_filename, rescored, list(config.output_fields))
    write_jsonl(out_dir / config.candidates_jsonl_filename, rescored)
    config.write_report(
        out_dir / config.report_filename,
        total_assays=args.total_assays,
        candidates=rescored,
        min_score=args.min_score,
        chembl_sqlite=args.chembl_sqlite,
    )
    (out_dir / config.health_check_filename).write_text(
        "\n".join(
            config.summarize_rows(
                _rows_for_summary(rescored),
                candidates_path=out_dir / config.candidates_filename,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    activity_in = in_dir / config.activity_evidence_filename
    if args.filter_activities and activity_in.exists():
        _filter_activity_evidence(
            activity_in,
            out_dir / config.activity_evidence_filename,
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


def _parse_args(config: RescoreConfig, argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=config.description)
    parser.add_argument("--in-dir", default=config.default_in_dir)
    parser.add_argument("--out-dir", default=config.default_out_dir)
    parser.add_argument("--min-score", type=int, default=40)
    parser.add_argument("--total-assays", type=int, default=1890749)
    parser.add_argument("--chembl-sqlite", default="tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db")
    parser.add_argument("--filter-activities", action="store_true")
    return parser.parse_args(argv)
