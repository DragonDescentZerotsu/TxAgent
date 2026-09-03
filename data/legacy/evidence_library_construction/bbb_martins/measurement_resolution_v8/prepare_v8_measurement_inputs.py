"""Derive fail-closed BBB v8 measurement inputs from the frozen v7 artifacts."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.paths import evidence_library_root
from data.processing.evidence_library.versions.v8.tasks.bbb_martins.starling_measurement_resolution import (
    SOURCE_MEASUREMENT_FIELDS,
)


TASK_ASSET_ROOT = Path(__file__).resolve().parent / "data_processing"
V7_EXACT_UNIT_MAPPING = (
    TASK_ASSET_ROOT / "canonicalization_v7/exact_measurement_unit_map.v2.json"
)
V8_EXACT_UNIT_MAPPING = (
    TASK_ASSET_ROOT / "canonicalization_v8/exact_measurement_unit_map.v8.json"
)
V7_MEASUREMENT_MAPPING = (
    TASK_ASSET_ROOT / "measurement_resolution_v3/measurement_resolution.parquet"
)
V8_MEASUREMENT_MAPPING = (
    TASK_ASSET_ROOT / "measurement_resolution_v8/measurement_resolution.parquet"
)
V7_CLEANED_RECORDS = (
    evidence_library_root("bbb_martins", "v7") / "01_cleaned/records.parquet"
)

_NUMBER = re.compile(r"(?<![A-Za-z0-9.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?![A-Za-z0-9.])")


def _decimal(value: Any) -> Decimal | None:
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _coefficient_occurs_in_declared_field(measurement: Any, field_value: Any) -> bool:
    target = _decimal(measurement)
    if target is None or field_value is None:
        return False
    if not isinstance(field_value, str):
        return _decimal(field_value) == target
    return any(_decimal(match.group()) == target for match in _NUMBER.finditer(field_value))


def build_exact_unit_mapping(
    source_path: Path = V7_EXACT_UNIT_MAPPING,
    output_path: Path = V8_EXACT_UNIT_MAPPING,
) -> dict[str, int]:
    """Move every conversion scale into unit identity and leave values untouched."""
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    changed = 0
    for entry in payload["entries"]:
        if entry.get("action") != "map":
            continue
        scale = str(entry.get("scale") or "1")
        if Decimal(scale) == 1:
            continue
        entry["canonical_unit"] = f"scale[{scale}]({entry['canonical_unit']})"
        entry["scale"] = "1"
        changed += 1
    payload["v8_contract"] = {
        "value_transform": "identity",
        "scale_representation": "distinct_unit_identity",
        "source": str(source_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"entries": len(payload["entries"]), "scales_moved_to_unit": changed}


def build_measurement_mapping(
    cleaned_path: Path = V7_CLEANED_RECORDS,
    source_path: Path = V7_MEASUREMENT_MAPPING,
    output_path: Path = V8_MEASUREMENT_MAPPING,
) -> dict[str, int]:
    """Keep one extraction only when its coefficient occurs in the declared field."""
    needed = ["cleaned_record_id", "source_id", "measurement_text"]
    cleaned = pq.read_table(cleaned_path, columns=needed).to_pylist()
    declared: dict[str, Any] = {}
    for row in cleaned:
        declared[str(row["cleaned_record_id"])] = row.get("measurement_text")

    table = pq.read_table(source_path)
    rows = table.to_pylist()
    counts = {
        "rows": len(rows),
        "retained_single_measurement": 0,
        "rejected_multiple_measurements": 0,
        "rejected_outside_declared_field": 0,
        "unchanged_non_ok": 0,
        "out_of_scope_structure_rejections": 0,
    }
    for row in rows:
        if str(row.get("status") or "") != "ok":
            counts["unchanged_non_ok"] += 1
            continue
        entries = json.loads(row.get("measurements_json") or "[]")
        record_id = str(row["cleaned_record_id"])
        if record_id not in declared:
            counts["out_of_scope_structure_rejections"] += 1
            continue
        if len(entries) != 1:
            row["status"] = "unsure"
            row["measurements_json"] = "[]"
            row["quantity_count"] = 0
            row["assignment_method"] = "v8_fail_closed_multiple_measurements"
            row["rejected_response_json"] = json.dumps(
                {"reason": "v8_requires_exactly_one_measurement", "count": len(entries)}
            )
            counts["rejected_multiple_measurements"] += 1
            continue
        if not _coefficient_occurs_in_declared_field(
            entries[0].get("measurement"), declared[record_id]
        ):
            row["status"] = "unavailable"
            row["measurements_json"] = "[]"
            row["quantity_count"] = 0
            row["assignment_method"] = "v8_fail_closed_outside_declared_field"
            row["rejected_response_json"] = json.dumps(
                {"reason": "measurement_not_in_declared_source_column"}
            )
            counts["rejected_outside_declared_field"] += 1
            continue
        counts["retained_single_measurement"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = pa.Table.from_pylist(rows, schema=table.schema)
    pq.write_table(output, output_path, compression="zstd")
    manifest = {
        "version": "bbb_martins_measurement_resolution.v8",
        "source_mapping": str(source_path),
        "cleaned_records": str(cleaned_path),
        "contract": {
            "maximum_measurements_per_row": 1,
            "coefficient_source": "declared_source_measurement_field_only",
            "support_only_coefficients": "unavailable",
            "multiple_measurements": "unsure",
        },
        "counts": counts,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return counts


def main() -> int:
    print(json.dumps({
        "exact_unit_mapping": build_exact_unit_mapping(),
        "measurement_mapping": build_measurement_mapping(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
