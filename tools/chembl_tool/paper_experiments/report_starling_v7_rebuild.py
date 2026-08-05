"""Snapshot and compare canonical v7 pair-bucket/calibration artifacts."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256


TASK_ROOTS = {
    "bbb_martins": Path(
        "outputs/chembl_tool/tasks/bbb_martins/evidence_library/starling_normalized_v7"
    ),
    "bioavailability_ma": Path(
        "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_normalized_v7"
    ),
    "skin_reaction": Path(
        "outputs/chembl_tool/tasks/skin_reaction/evidence_library/starling_normalized_v7"
    ),
}
REPORT_VERSION = "starling_v7_rebuild_delta.v1"


def capture_snapshot(label: str, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"version": REPORT_VERSION, "label": label, "tasks": {}}
    for task, root in TASK_ROOTS.items():
        pair_meta_path = root / "04_pair_buckets/pair_bucket_metadata.json"
        calibration_path = (
            root
            / "05_distance_calibration/pair_bucket_distance_calibration.json.gz"
        )
        assignments_path = root / "04_pair_buckets/pair_bucket_records.parquet"
        pair_meta = json.loads(pair_meta_path.read_text(encoding="utf-8"))
        with gzip.open(calibration_path, "rt", encoding="utf-8") as handle:
            calibration = json.load(handle)
        valid_buckets = {
            str(key)
            for key, entry in calibration["buckets"].items()
            if entry.get("calibration_valid")
        }
        assignments = pd.read_parquet(assignments_path)
        assignments["calibration_valid"] = assignments["pair_bucket_key"].isin(
            valid_buckets
        )
        snapshot_assignments = output_dir / f"{label}_{task}_assignments.parquet"
        assignments.to_parquet(snapshot_assignments, index=False)
        task_result: dict[str, Any] = {
            "root": str(root),
            "pair_bucket_stats": pair_meta["stats"],
            "calibration_summary": calibration["summary"],
            "calibration_valid_bucket_keys": sorted(valid_buckets),
            "files": {
                "pair_bucket_metadata_sha256": file_sha256(pair_meta_path),
                "pair_bucket_records_sha256": file_sha256(assignments_path),
                "distance_calibration_sha256": file_sha256(calibration_path),
                "snapshot_assignments": str(snapshot_assignments),
                "snapshot_assignments_sha256": file_sha256(snapshot_assignments),
            },
        }
        if task == "bioavailability_ma":
            task_result["hf_semantic_breakdown"] = _bio_hf_breakdown(root)
        result["tasks"][task] = task_result
    snapshot_path = output_dir / f"{label}.json"
    snapshot_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def compare_snapshots(before_path: Path, after_path: Path, output_dir: Path) -> dict[str, Any]:
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    comparison: dict[str, Any] = {
        "version": REPORT_VERSION,
        "before": str(before_path),
        "after": str(after_path),
        "tasks": {},
    }
    for task in TASK_ROOTS:
        old = before["tasks"][task]
        new = after["tasks"][task]
        old_frame = pd.read_parquet(old["files"]["snapshot_assignments"])
        new_frame = pd.read_parquet(new["files"]["snapshot_assignments"])
        record_id = "canonical_record_id"
        old_by_id = old_frame.set_index(record_id, drop=False)
        new_by_id = new_frame.set_index(record_id, drop=False)
        old_eligible = set(old_by_id.index[old_by_id["bucket_eligible"]])
        new_eligible = set(new_by_id.index[new_by_id["bucket_eligible"]])
        old_calibration = set(old_by_id.index[old_by_id["calibration_valid"]])
        new_calibration = set(new_by_id.index[new_by_id["calibration_valid"]])
        common_eligible = old_eligible & new_eligible
        common_index = sorted(common_eligible)
        changed_bucket = int(
            old_by_id.loc[common_index, "pair_bucket_key"]
            .astype(str)
            .ne(new_by_id.loc[common_index, "pair_bucket_key"].astype(str))
            .sum()
        )
        old_bucket_keys = set(old_frame.loc[old_frame["bucket_eligible"], "pair_bucket_key"])
        new_bucket_keys = set(new_frame.loc[new_frame["bucket_eligible"], "pair_bucket_key"])
        old_valid_buckets = set(old["calibration_valid_bucket_keys"])
        new_valid_buckets = set(new["calibration_valid_bucket_keys"])
        comparison["tasks"][task] = {
            "pair_bucket_stats": _numeric_delta(
                old["pair_bucket_stats"], new["pair_bucket_stats"]
            ),
            "calibration_summary": _numeric_delta(
                old["calibration_summary"], new["calibration_summary"]
            ),
            "pair_bucket_keys": _set_delta(old_bucket_keys, new_bucket_keys),
            "eligible_record_ids": _set_delta(old_eligible, new_eligible),
            "common_eligible_records_reassigned_bucket": changed_bucket,
            "calibration_valid_bucket_keys": _set_delta(
                old_valid_buckets, new_valid_buckets
            ),
            "calibration_valid_record_ids": _set_delta(
                old_calibration, new_calibration
            ),
        }
        if task == "bioavailability_ma":
            comparison["tasks"][task]["hf_semantic_breakdown"] = {
                "before": old.get("hf_semantic_breakdown", []),
                "after": new.get("hf_semantic_breakdown", []),
            }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    summary_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_markdown_report(comparison), encoding="utf-8")
    return comparison


def _bio_hf_breakdown(root: Path) -> list[dict[str, Any]]:
    records_path = root / "03_records/records.parquet"
    wanted = [
        "source_id",
        "bioavailability_report_type",
        "canonical_bioavailability_report_type",
        "canonical_bioavailability_evidence_scope",
        "group_id",
        "measurement_kind",
        "canonical_unit_text",
        "finite_scalar_value",
    ]
    available = set(pq.read_schema(records_path).names)
    frame = pd.read_parquet(records_path, columns=[field for field in wanted if field in available])
    frame = frame[frame["source_id"] == "hf_bioavailability"].copy()
    frame["has_finite_scalar"] = frame["finite_scalar_value"].notna()
    dimensions = [
        field
        for field in wanted
        if field not in {"source_id", "finite_scalar_value"} and field in frame
    ] + ["has_finite_scalar"]
    grouped = frame.groupby(dimensions, dropna=False).size().reset_index(name="records")
    grouped = grouped.sort_values(["records", *dimensions], ascending=[False, *([True] * len(dimensions))])
    return json.loads(grouped.to_json(orient="records"))


def _numeric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            output[key] = {"before": old, "after": new, "delta": new - old}
    return output


def _set_delta(before: set[str], after: set[str]) -> dict[str, Any]:
    added = sorted(after - before)
    removed = sorted(before - after)
    return {
        "before": len(before),
        "after": len(after),
        "delta": len(after) - len(before),
        "common": len(before & after),
        "added": len(added),
        "removed": len(removed),
        "added_examples": added[:20],
        "removed_examples": removed[:20],
    }


def _markdown_report(comparison: dict[str, Any]) -> str:
    lines = ["# Starling v7 rebuild delta", ""]
    for task, values in comparison["tasks"].items():
        pair = values["pair_bucket_stats"]
        calibration = values["calibration_summary"]
        lines.extend(
            [
                f"## {task}",
                "",
                f"- Pair buckets: {_format_delta(pair['buckets'])}",
                f"- Stage 04 eligible records: {_format_delta(pair['eligible_records'])}",
                "- Stage 05 calibration-valid buckets: "
                f"{_format_delta(calibration['calibration_valid_buckets'])}",
                "- Stage 05 calibration-valid records: "
                f"{_format_delta(calibration['calibration_valid_records'])}",
                "- Common eligible records assigned to a different bucket: "
                f"{values['common_eligible_records_reassigned_bucket']:,}",
                "- Pair-bucket keys added / removed: "
                f"{values['pair_bucket_keys']['added']:,} / "
                f"{values['pair_bucket_keys']['removed']:,}",
                "- Eligible record IDs added / removed: "
                f"{values['eligible_record_ids']['added']:,} / "
                f"{values['eligible_record_ids']['removed']:,}",
                "- Calibration-valid record IDs added / removed: "
                f"{values['calibration_valid_record_ids']['added']:,} / "
                f"{values['calibration_valid_record_ids']['removed']:,}",
                "",
            ]
        )
        if task == "bioavailability_ma":
            semantic = values["hf_semantic_breakdown"]
            old_scope = _scope_totals(semantic["before"])
            new_scope = _scope_totals(semantic["after"])
            lines.extend(
                [
                    "HF semantic correction:",
                    "",
                    "- Literal `unspecified` report rows: "
                    f"{old_scope['unspecified']:,} → {new_scope['unspecified']:,}",
                    "- Canonical direct-scope rows: "
                    f"{old_scope['direct']:,} → {new_scope['direct']:,}",
                    "- Canonical nondirect-scope rows: "
                    f"{old_scope['nondirect']:,} → {new_scope['nondirect']:,}",
                    "- HF rows with finite scalar values: "
                    f"{old_scope['finite']:,} → {new_scope['finite']:,}",
                    "",
                ]
            )
    return "\n".join(lines) + "\n"


def _scope_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "unspecified": sum(
            int(row["records"])
            for row in rows
            if str(row.get("bioavailability_report_type")) == "unspecified"
        ),
        "direct": sum(
            int(row["records"])
            for row in rows
            if str(row.get("canonical_bioavailability_evidence_scope")) == "direct"
        ),
        "nondirect": sum(
            int(row["records"])
            for row in rows
            if str(row.get("canonical_bioavailability_evidence_scope")) == "nondirect"
        ),
        "finite": sum(
            int(row["records"])
            for row in rows
            if bool(row.get("has_finite_scalar"))
        ),
    }


def _format_delta(value: dict[str, Any]) -> str:
    return f"{value['before']:,} → {value['after']:,} ({value['delta']:+,})"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--label", required=True)
    snapshot.add_argument("--output-dir", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--before", type=Path, required=True)
    compare.add_argument("--after", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "snapshot":
        payload = capture_snapshot(args.label, args.output_dir)
    else:
        payload = compare_snapshots(args.before, args.after, args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
