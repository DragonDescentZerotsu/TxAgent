"""Build a deliberately minimal v8 unit map from a 200-row review sample."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.paths import evidence_library_root
from data.processing.evidence_library.shared.v1.normalization.cleaning import file_sha256
from data.processing.evidence_library.shared.v1.normalization.measurement_resolution import (
    load_exact_unit_mapping,
)
from tools.chembl_tool.common.units import canonicalize_unit
from data.processing.evidence_library.versions.v8.tasks.bbb_martins.prepare_v8_measurement_inputs import (
    V8_EXACT_UNIT_MAPPING,
    V8_MEASUREMENT_MAPPING,
    build_exact_unit_mapping,
)
from data.processing.evidence_library.versions.v8.tasks.bbb_martins.starling_measurement_semantics import (
    resolve_measurement_semantics,
    unit_is_compatible,
)


V8_CLEANED_RECORDS = (
    evidence_library_root("bbb_martins", "v8") / "01_cleaned/records.parquet"
)
REVIEW_PATH = V8_EXACT_UNIT_MAPPING.parent / "minimal_unit_review_200.jsonl"
RULE_ID = "v8_minimal_unit_review_200.v1"
SAMPLE_SIZE = 200
SAMPLE_SEED = 20260831


def _scaled_unit(canonical: str, factor: float | None) -> str:
    if factor is None or Decimal(str(factor)) == 1:
        return canonical
    return f"scale[{format(Decimal(str(factor)), 'f')}]({canonical})"


def _original_entry(row: dict[str, Any]) -> dict[str, str] | None:
    if row.get("status") == "ok":
        entries = json.loads(row.get("measurements_json") or "[]")
        return entries[0] if len(entries) == 1 else None
    if row.get("assignment_method") != "v8_fail_closed_unit_preflight":
        return None
    raw = json.loads(row.get("raw_response_json") or "{}")
    entries = raw.get("measurements") or []
    return entries[0] if raw.get("status") == "ok" and len(entries) == 1 else None


def _decision(endpoint: str, unit_text: str) -> tuple[str | None, str]:
    parsed = canonicalize_unit(unit_text, task="bbb_martins")
    if not parsed.canonical or parsed.unknown_tokens:
        return None, "unresolved_unit_spelling"
    semantics = resolve_measurement_semantics(
        {"canonical_endpoint": endpoint}, endpoint
    )
    if semantics.status != "approved":
        return None, "unreviewed_endpoint_semantics"
    if unit_is_compatible(semantics, parsed.canonical) is not True:
        return None, "incompatible_endpoint_unit"
    return _scaled_unit(parsed.canonical, parsed.notation_factor), "map"


def finalize() -> dict[str, Any]:
    # Reset broader generated additions: v8 starts from v7-reviewed vocabulary,
    # with conversion scales represented as distinct unit identities.
    build_exact_unit_mapping()
    exact_payload = json.loads(V8_EXACT_UNIT_MAPPING.read_text(encoding="utf-8"))
    existing = load_exact_unit_mapping(V8_EXACT_UNIT_MAPPING)
    source_rows = {
        str(row["cleaned_record_id"]): row
        for row in pq.read_table(
            V8_CLEANED_RECORDS,
            columns=[
                "cleaned_record_id",
                "source_id",
                "canonical_endpoint_name",
                "measurement_text",
                "support_text",
            ],
        ).to_pylist()
    }
    table = pq.read_table(V8_MEASUREMENT_MAPPING)
    rows = table.to_pylist()
    missing_rows: list[tuple[dict[str, Any], dict[str, str], dict[str, Any]]] = []
    for row in rows:
        entry = _original_entry(row)
        if entry is None:
            continue
        source = source_rows[str(row["cleaned_record_id"])]
        key = (
            "bbb_martins",
            str(source["canonical_endpoint_name"]),
            str(entry.get("unit") or "").strip(),
        )
        if key not in existing:
            missing_rows.append((row, entry, source))

    sample = random.Random(SAMPLE_SEED).sample(
        missing_rows, min(SAMPLE_SIZE, len(missing_rows))
    )
    reviewed_keys: dict[tuple[str, str], str] = {}
    review_records: list[dict[str, Any]] = []
    for row, entry, source in sample:
        endpoint = str(source["canonical_endpoint_name"])
        input_unit = str(entry.get("unit") or "").strip()
        output_unit, decision = _decision(endpoint, input_unit)
        if output_unit is not None:
            reviewed_keys[(endpoint, input_unit)] = output_unit
        else:
            output_unit = input_unit
            decision = "preserve_exact_source_unit"
        review_records.append(
            {
                "cleaned_record_id": str(row["cleaned_record_id"]),
                "source_id": source["source_id"],
                "canonical_endpoint": endpoint,
                "measurement_text": source["measurement_text"],
                "support_text": source["support_text"],
                "extracted_measurement": entry.get("measurement"),
                "extracted_unit": input_unit,
                "decision": decision,
                "canonical_unit": output_unit,
            }
        )

    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for _, entry, source in missing_rows:
        endpoint = str(source["canonical_endpoint_name"])
        input_unit = str(entry.get("unit") or "").strip()
        output_unit = reviewed_keys.get((endpoint, input_unit), input_unit)
        grouped[(input_unit, output_unit)].add(endpoint)
    for (input_unit, output_unit), endpoints in sorted(grouped.items()):
        exact_payload["entries"].append(
            {
                "task": "bbb_martins",
                "canonical_endpoints": sorted(endpoints),
                "input_unit": input_unit,
                "action": "map",
                "canonical_unit": output_unit,
                "scale": "1",
                "domain": "any",
                "v8_rule_id": RULE_ID,
            }
        )

    status_before = Counter(row["status"] for row in rows)
    unified_rows = 0
    preserved_rows = 0
    for row, entry, source in missing_rows:
        endpoint = str(source["canonical_endpoint_name"])
        input_unit = str(entry.get("unit") or "").strip()
        key = (endpoint, input_unit)
        row.update(
            {
                "status": "ok",
                "measurements_json": json.dumps([entry], ensure_ascii=False),
                "quantity_count": 1,
            }
        )
        if row.get("assignment_method") == "v8_fail_closed_unit_preflight":
            row["assignment_method"] = "model_recovered_by_minimal_unit_review"
            row["rejected_response_json"] = None
        if key in reviewed_keys:
            unified_rows += 1
        else:
            preserved_rows += 1

    exact_payload.setdefault("v8_contract", {})["minimal_unit_review"] = {
        "rule_id": RULE_ID,
        "sample_size": len(sample),
        "sample_seed": SAMPLE_SEED,
        "review_path": str(REVIEW_PATH),
        "approved_endpoint_unit_keys": len(reviewed_keys),
    }
    V8_EXACT_UNIT_MAPPING.write_text(
        json.dumps(exact_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    REVIEW_PATH.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in review_records
        ),
        encoding="utf-8",
    )
    pq.write_table(
        pa.Table.from_pylist(rows, schema=table.schema),
        V8_MEASUREMENT_MAPPING,
        compression="zstd",
    )

    status_after = Counter(row["status"] for row in rows)
    summary = {
        "rule_id": RULE_ID,
        "sample_size": len(sample),
        "sample_seed": SAMPLE_SEED,
        "missing_rows_before_review": len(missing_rows),
        "sampled_approved_keys": len(reviewed_keys),
        "rows_unified_by_sampled_keys": unified_rows,
        "rows_preserved_as_exact_unit_identity": preserved_rows,
        "status_counts_before": dict(sorted(status_before.items())),
        "status_counts_after": dict(sorted(status_after.items())),
        "review_path": str(REVIEW_PATH),
        "review_sha256": file_sha256(REVIEW_PATH),
        "exact_unit_mapping_sha256": file_sha256(V8_EXACT_UNIT_MAPPING),
    }
    manifest_path = V8_MEASUREMENT_MAPPING.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["mapping_sha256"] = file_sha256(V8_MEASUREMENT_MAPPING)
    manifest["status_counts"] = summary["status_counts_after"]
    manifest["rejected_rows"] = sum(
        bool(row.get("rejected_response_json")) for row in rows
    )
    manifest["assignment_method_counts"] = dict(
        sorted(Counter(row["assignment_method"] for row in rows).items())
    )
    manifest["unit_preflight"] = summary
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    print(json.dumps(finalize(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
