"""Build and validate the frozen AMES V10 measurement-candidate inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_candidate_resolution import (
    CANDIDATE_CONTRACT_VERSION,
    candidate_set_sha256,
    load_candidates,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_measurement_candidates import (
    CANDIDATE_GENERATOR_VERSION,
    EVIDENCE_FIELDS,
    MAX_CANDIDATE_FIELD_BYTES,
    MAX_CANDIDATE_JSON_BYTES,
    MAX_CANDIDATES,
    candidate_generation_disposition,
    candidates_for_record,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_measurement_resolution import (
    DEFAULT_CLEANED_RECORDS,
    DEFAULT_PROFILE_PATH,
    SOURCE_IDS,
    prompt_row_fields,
)

TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[8]
DEFAULT_OUTPUT = (
    TASK_ROOT
    / "data_processing/measurement_resolution_v2/measurement_candidates.parquet"
)
DEFAULT_GOLD = (
    REPO_ROOT
    / "tests/chembl_tool/common/measurement_resolution_quality/gold/ames.v10.jsonl"
)
GOLD_SHA256 = "14ced688f4e4bfb31df915eff79543acfc78c047df552efe280d9f7541d6be1d"
GOLD_CASES = 500
GOLD_SOURCE_COUNTS = {
    "fixed_mutation": 150,
    "mutagenicity_mechanism": 150,
    "premutagenic_damage": 200,
}
GOLD_STATUS_COUNTS = {
    "ok": 153,
    "relative": 72,
    "unavailable": 59,
    "unsure": 216,
}
MIN_GOLD_PAIR_COVERAGE = 146
MIN_SOURCE_GOLD_COVERAGE = 0.90


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def _canonical_hash(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _source_columns() -> list[str]:
    fields = {
        "cleaned_record_id",
        "source_row_uid",
        "source_id",
        "canonical_endpoint_name",
        "measurement_resolution_route",
    }
    for source_id in SOURCE_IDS:
        fields.update(prompt_row_fields(source_id))
    fields.update(EVIDENCE_FIELDS)
    return sorted(fields)


def _stage1_contract(records_path: Path, profile_path: Path) -> dict[str, Any]:
    manifest_path = records_path.parent / "manifest.json"
    manifest = _read_object(manifest_path)
    output = manifest.get("output") or {}
    profile = (manifest.get("sidecars") or {}).get("endpoint_unit_profile") or {}
    expected_rows = int(
        ((manifest.get("validations") or {}).get("measurement_route_counts") or {}).get(
            "extract", -1
        )
    )
    if output.get("sha256") != file_sha256(records_path):
        raise ValueError("Stage 1 cleaned-record hash mismatch")
    if profile.get("sha256") != file_sha256(profile_path):
        raise ValueError("Stage 1 endpoint-profile hash mismatch")
    if expected_rows < 1:
        raise ValueError("Stage 1 manifest lacks extract-route count")
    return {
        "records": _reference(records_path),
        "manifest": _reference(manifest_path),
        "endpoint_profile": _reference(profile_path),
        "expected_extract_rows": expected_rows,
    }


def _reference(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _candidate_row(record: Mapping[str, Any]) -> dict[str, Any]:
    candidates = candidates_for_record(dict(record))
    payload = json.dumps(
        candidates, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    load_candidates(payload)
    if len(candidates) > MAX_CANDIDATES:
        raise ValueError("candidate generator exceeded its declared maximum")
    if len(payload.encode("utf-8")) > MAX_CANDIDATE_JSON_BYTES:
        raise ValueError("candidate generator exceeded its byte bound")
    return {
        **dict(record),
        "measurement_candidates_json": payload,
        "candidate_set_sha256": candidate_set_sha256(payload),
        "candidate_contract_version": CANDIDATE_CONTRACT_VERSION,
        "candidate_generator_version": CANDIDATE_GENERATOR_VERSION,
        "candidate_generation_disposition": candidate_generation_disposition(
            dict(record)
        ),
        "candidate_count": len(candidates),
    }


def build_candidate_rows(records_path: Path) -> list[dict[str, Any]]:
    """Read every Stage 1 extract row into one frozen candidate inventory."""
    available = set(pq.read_schema(records_path).names)
    columns = _source_columns()
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError(f"Stage 1 lacks candidate source columns: {missing}")
    rows: list[dict[str, Any]] = []
    for batch in pq.ParquetFile(records_path).iter_batches(
        batch_size=50_000, columns=columns
    ):
        for record in batch.to_pylist():
            if record.get("measurement_resolution_route") == "extract":
                rows.append(_candidate_row(record))
    rows.sort(
        key=lambda row: (
            row["source_id"],
            row["canonical_endpoint_name"],
            row["cleaned_record_id"],
        )
    )
    return rows


def _identity_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    identities = [
        (
            row["cleaned_record_id"],
            row["source_row_uid"],
            row["source_id"],
            row["canonical_endpoint_name"],
            row["candidate_set_sha256"],
        )
        for row in rows
    ]
    return _canonical_hash(sorted(identities))


def _row_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["cleaned_record_id"])):
        payload = json.dumps(
            dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _gold_cases(path: Path) -> list[dict[str, Any]]:
    if path.resolve() != DEFAULT_GOLD.resolve() or file_sha256(path) != GOLD_SHA256:
        raise ValueError("AMES candidate development fixture is not the frozen gold")
    lines = [json.loads(line) for line in path.read_text().splitlines() if line]
    manifest, cases = lines[0], lines[1:]
    ids = [str(case.get("audit_case_id") or "") for case in cases]
    uids = [str(case.get("source_row_uid") or "") for case in cases]
    source_counts = Counter(str(case.get("source_id") or "") for case in cases)
    status_counts = Counter(
        str((case.get("expected") or {}).get("status") or "") for case in cases
    )
    valid = (
        manifest.get("task_id") == "ames"
        and manifest.get("cases") == len(cases) == GOLD_CASES
        and len(set(ids)) == len(ids)
        and len(set(uids)) == len(uids)
        and all(ids)
        and all(uids)
        and dict(source_counts) == GOLD_SOURCE_COUNTS
        and dict(status_counts) == GOLD_STATUS_COUNTS
    )
    if not valid:
        raise ValueError("AMES candidate gold fixture contract mismatch")
    return cases


def _gold_record(
    case: Mapping[str, Any], by_id: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any]:
    record_id = str(case["audit_case_id"]).split(":", 1)[1]
    row = by_id.get(record_id)
    if row is None:
        raise ValueError(f"gold row is absent from candidate inventory: {record_id}")
    expected = {
        "source_row_uid": case["source_row_uid"],
        "source_id": case["source_id"],
        "canonical_endpoint_name": case["canonical_endpoint_name"],
    }
    source_fields = prompt_row_fields(str(case["source_id"]))
    expected.update(
        {field: (case.get("input") or {}).get(field) for field in source_fields}
    )
    mismatches = [key for key, value in expected.items() if row.get(key) != value]
    if mismatches:
        raise ValueError(f"gold row drifted from Stage 1: {record_id}: {mismatches}")
    return row


def gold_coverage(rows: Sequence[Mapping[str, Any]], gold_path: Path) -> dict[str, Any]:
    """Measure development-fixture recall without treating labels as inputs."""
    by_id = {str(row["cleaned_record_id"]): row for row in rows}
    cases = _gold_cases(gold_path)
    if len(by_id) != len(rows):
        raise ValueError("candidate inventory has duplicate cleaned IDs")
    ok_cases = [case for case in cases if case["expected"]["status"] == "ok"]
    source_totals = Counter(case["source_id"] for case in ok_cases)
    source_hits: Counter[str] = Counter()
    misses: list[str] = []
    for case in cases:
        _gold_record(case, by_id)
    for case in ok_cases:
        record_id = case["audit_case_id"].split(":", 1)[1]
        row = _gold_record(case, by_id)
        expected = case["expected"]["measurements"][0]
        candidates = load_candidates(row["measurement_candidates_json"])
        numeric_pairs = {
            (str(item["measurement"]), str(item["unit"])) for item in candidates
        }
        hit = any(
            _same_number(value, expected["measurement"]) and unit == expected["unit"]
            for value, unit in numeric_pairs
        )
        if hit:
            source_hits[case["source_id"]] += 1
        else:
            misses.append(record_id)
    by_source = {
        source: {
            "hits": source_hits[source],
            "total": total,
            "coverage": source_hits[source] / total,
        }
        for source, total in sorted(source_totals.items())
    }
    return {
        "cases": len(cases),
        "ok_cases": len(ok_cases),
        "pair_hits": sum(source_hits.values()),
        "pair_misses": misses,
        "by_source": by_source,
    }


def _same_number(left: object, right: object) -> bool:
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except (InvalidOperation, ValueError):
        return False


def _validate_gold_gate(coverage: Mapping[str, Any]) -> None:
    if int(coverage["pair_hits"]) < MIN_GOLD_PAIR_COVERAGE:
        raise ValueError("candidate inventory misses the overall gold recall gate")
    failed = [
        source
        for source, result in coverage["by_source"].items()
        if float(result["coverage"]) < MIN_SOURCE_GOLD_COVERAGE
    ]
    if failed:
        raise ValueError(f"candidate inventory misses source gold gates: {failed}")


def _counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field) or "") for row in rows).items()))


def _maximum_field_bytes(rows: Sequence[Mapping[str, Any]]) -> int:
    maximum = 0
    for row in rows:
        for candidate in load_candidates(row["measurement_candidates_json"]):
            for field in ("measurement", "unit"):
                maximum = max(maximum, len(str(candidate[field]).encode("utf-8")))
    return maximum


def _maximum_payload_bytes(rows: Sequence[Mapping[str, Any]]) -> int:
    return max(
        len(str(row["measurement_candidates_json"]).encode("utf-8")) for row in rows
    )


def _manifest_validations(
    rows: Sequence[Mapping[str, Any]], stage1: Mapping[str, Any]
) -> dict[str, bool]:
    counts = [int(row["candidate_count"]) for row in rows]
    return {
        "exact_stage1_extract_coverage": len(rows)
        == int(stage1["expected_extract_rows"]),
        "unique_cleaned_record_ids": len(rows)
        == len({row["cleaned_record_id"] for row in rows}),
        "unique_source_row_uids": len(rows)
        == len({row["source_row_uid"] for row in rows}),
        "nonempty_identity_fields": all(
            str(row.get(field) or "").strip()
            for row in rows
            for field in ("cleaned_record_id", "source_row_uid")
        ),
        "known_sources": all(row.get("source_id") in SOURCE_IDS for row in rows),
        "complete_canonical_endpoints": all(
            str(row.get("canonical_endpoint_name") or "").strip() for row in rows
        ),
        "candidate_sets_hash_valid": True,
        "candidate_sets_not_truncated": True,
        "candidate_cap_respected": max(counts) <= MAX_CANDIDATES,
        "candidate_field_byte_bound_respected": _maximum_field_bytes(rows)
        <= MAX_CANDIDATE_FIELD_BYTES,
        "candidate_json_byte_bound_respected": _maximum_payload_bytes(rows)
        <= MAX_CANDIDATE_JSON_BYTES,
    }


def _manifest(
    output_path: Path,
    temporary_path: Path,
    rows: Sequence[Mapping[str, Any]],
    stage1: Mapping[str, Any],
    gold_path: Path,
) -> dict[str, Any]:
    coverage = gold_coverage(rows, gold_path)
    _validate_gold_gate(coverage)
    counts = [int(row["candidate_count"]) for row in rows]
    return {
        "inventory_version": CANDIDATE_CONTRACT_VERSION,
        "candidate_generator_version": CANDIDATE_GENERATOR_VERSION,
        "candidate_path": str(output_path.resolve()),
        "candidate_sha256": file_sha256(temporary_path),
        "candidate_rows": len(rows),
        "candidate_identity_sha256": _identity_digest(rows),
        "candidate_rows_sha256": _row_digest(rows),
        "candidate_schema": pq.read_schema(temporary_path).to_string(),
        "source_counts": _counts(rows, "source_id"),
        "disposition_counts": _counts(rows, "candidate_generation_disposition"),
        "candidate_count_distribution": dict(sorted(Counter(counts).items())),
        "zero_candidate_rows": counts.count(0),
        "maximum_candidates": MAX_CANDIDATES,
        "maximum_observed_candidates": max(counts),
        "maximum_candidate_field_bytes": MAX_CANDIDATE_FIELD_BYTES,
        "maximum_observed_candidate_field_bytes": _maximum_field_bytes(rows),
        "maximum_candidate_json_bytes": MAX_CANDIDATE_JSON_BYTES,
        "maximum_observed_candidate_json_bytes": _maximum_payload_bytes(rows),
        "stage1": dict(stage1),
        "implementation": {
            "writer": _reference(Path(__file__)),
            "generator": _reference(TASK_ROOT / "starling_measurement_candidates.py"),
            "compiler_contract": _reference(
                TASK_ROOT / "starling_candidate_resolution.py"
            ),
            "source_fields": _reference(
                TASK_ROOT / "starling_measurement_resolution.py"
            ),
        },
        "development_gold_validation": {
            **coverage,
            **_reference(gold_path),
            "construction_input": True,
            "independent_generalization_evidence": False,
        },
        "validations": _manifest_validations(rows, stage1),
    }


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = sorted(set().union(*(row.keys() for row in rows)))
    table = pa.Table.from_pylist(
        [{column: row.get(column) for column in columns} for row in rows]
    )
    pq.write_table(table, path)


def write_candidate_inventory(
    records_path: Path,
    profile_path: Path,
    output_path: Path,
    gold_path: Path,
) -> dict[str, Any]:
    """Write only a complete, recursively validated candidate artifact."""
    manifest_path = _manifest_path(output_path)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    paths = (output_path, manifest_path, temporary, temporary_manifest)
    if any(path.exists() for path in paths):
        raise FileExistsError(f"refusing to replace candidate artifact: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage1 = _stage1_contract(records_path, profile_path)
    rows = build_candidate_rows(records_path)
    try:
        _write_parquet(temporary, rows)
        manifest = _manifest(output_path, temporary, rows, stage1, gold_path)
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output_path)
        os.replace(temporary_manifest, manifest_path)
        validate_candidate_inventory(output_path)
    except BaseException:
        output_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
    return manifest


def _validate_candidate_rows(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> None:
    for row in rows:
        identity = {
            field: str(row.get(field) or "").strip()
            for field in ("cleaned_record_id", "source_row_uid")
        }
        if not all(identity.values()):
            raise ValueError("candidate inventory contains an empty identity")
        if row.get("source_id") not in SOURCE_IDS:
            raise ValueError("candidate inventory contains an unknown source")
        if not str(row.get("canonical_endpoint_name") or "").strip():
            raise ValueError("candidate inventory contains an empty canonical endpoint")
        payload = row.get("measurement_candidates_json")
        candidates = load_candidates(payload)
        if len(str(payload).encode("utf-8")) > MAX_CANDIDATE_JSON_BYTES:
            raise ValueError("candidate inventory row exceeds the JSON byte bound")
        if any(
            len(str(candidate[field]).encode("utf-8")) > MAX_CANDIDATE_FIELD_BYTES
            for candidate in candidates
            for field in ("measurement", "unit")
        ):
            raise ValueError("candidate inventory row exceeds the field byte bound")
        if candidate_set_sha256(payload) != row.get("candidate_set_sha256"):
            raise ValueError("candidate inventory row hash mismatch")
        if row.get("candidate_contract_version") != CANDIDATE_CONTRACT_VERSION:
            raise ValueError("candidate inventory row contract mismatch")
        if row.get("candidate_generator_version") != CANDIDATE_GENERATOR_VERSION:
            raise ValueError("candidate inventory row generator mismatch")
        if int(row.get("candidate_count") or 0) != len(candidates):
            raise ValueError("candidate inventory row count disagrees")
        if len(candidates) > MAX_CANDIDATES:
            raise ValueError("candidate inventory row exceeds the candidate cap")
    if _identity_digest(rows) != manifest.get("candidate_identity_sha256"):
        raise ValueError("candidate inventory identity digest mismatch")
    if _row_digest(rows) != manifest.get("candidate_rows_sha256"):
        raise ValueError("candidate inventory row digest mismatch")


def _validate_implementation(manifest: Mapping[str, Any]) -> None:
    implementation = manifest.get("implementation") or {}
    expected = {
        "writer": Path(__file__).resolve(),
        "generator": TASK_ROOT / "starling_measurement_candidates.py",
        "compiler_contract": TASK_ROOT / "starling_candidate_resolution.py",
        "source_fields": TASK_ROOT / "starling_measurement_resolution.py",
    }
    if set(implementation) != set(expected):
        raise ValueError("candidate inventory implementation inventory mismatch")
    for role, expected_path in expected.items():
        reference = implementation[role]
        source = Path(str(reference.get("path") or ""))
        if source.resolve() != expected_path.resolve():
            raise ValueError("candidate inventory implementation role mismatch")
        if not source.is_file() or file_sha256(source) != reference.get("sha256"):
            raise ValueError("candidate inventory implementation drift")


def _validate_stage1_rebuild(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> None:
    stage1 = manifest.get("stage1") or {}
    records_path = Path(str((stage1.get("records") or {}).get("path") or ""))
    profile_path = Path(str((stage1.get("endpoint_profile") or {}).get("path") or ""))
    if _stage1_contract(records_path, profile_path) != stage1:
        raise ValueError("candidate inventory Stage 1 contract drift")
    expected = build_candidate_rows(records_path)
    if _row_digest(expected) != _row_digest(rows):
        raise ValueError("candidate inventory differs from the frozen generator")


def _validate_gold(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> None:
    recorded = manifest.get("development_gold_validation") or {}
    path = Path(str(recorded.get("path") or ""))
    if not path.is_file() or file_sha256(path) != recorded.get("sha256"):
        raise ValueError("candidate inventory development-gold drift")
    actual = gold_coverage(rows, path)
    if any(recorded.get(key) != value for key, value in actual.items()):
        raise ValueError("candidate inventory development-gold result mismatch")
    _validate_gold_gate(actual)


def _validate_manifest_stats(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> None:
    counts = [int(row["candidate_count"]) for row in rows]
    distribution = {str(key): value for key, value in Counter(counts).items()}
    expected = {
        "source_counts": _counts(rows, "source_id"),
        "disposition_counts": _counts(rows, "candidate_generation_disposition"),
        "candidate_count_distribution": dict(sorted(distribution.items())),
        "zero_candidate_rows": counts.count(0),
        "maximum_observed_candidates": max(counts),
        "maximum_observed_candidate_field_bytes": _maximum_field_bytes(rows),
        "maximum_observed_candidate_json_bytes": _maximum_payload_bytes(rows),
    }
    mismatches = {key for key, value in expected.items() if manifest.get(key) != value}
    if mismatches:
        raise ValueError(f"candidate inventory summary mismatch: {sorted(mismatches)}")


def _validate_manifest_claims(path: Path, manifest: Mapping[str, Any]) -> None:
    gold = manifest.get("development_gold_validation") or {}
    expected = {
        "candidate_path": str(path.resolve()),
        "candidate_schema": pq.read_schema(path).to_string(),
        "maximum_candidates": MAX_CANDIDATES,
        "maximum_candidate_field_bytes": MAX_CANDIDATE_FIELD_BYTES,
        "maximum_candidate_json_bytes": MAX_CANDIDATE_JSON_BYTES,
    }
    mismatches = {key for key, value in expected.items() if manifest.get(key) != value}
    if mismatches:
        raise ValueError(f"candidate inventory claim mismatch: {sorted(mismatches)}")
    if gold.get("construction_input") is not True:
        raise ValueError("candidate inventory misstates development-gold use")
    if gold.get("independent_generalization_evidence") is not False:
        raise ValueError("candidate inventory misstates generalization evidence")


def validate_candidate_inventory(path: Path) -> dict[str, Any]:
    """Rebuild the inventory from pinned Stage 1 inputs before accepting it."""
    manifest = _read_object(_manifest_path(path))
    expected = {
        "inventory_version": CANDIDATE_CONTRACT_VERSION,
        "candidate_generator_version": CANDIDATE_GENERATOR_VERSION,
        "candidate_sha256": file_sha256(path),
        "candidate_rows": pq.read_metadata(path).num_rows,
    }
    mismatches = {key for key, value in expected.items() if manifest.get(key) != value}
    if mismatches:
        raise ValueError(f"candidate inventory manifest mismatch: {sorted(mismatches)}")
    rows = pq.read_table(path).to_pylist()
    _validate_candidate_rows(rows, manifest)
    _validate_implementation(manifest)
    _validate_stage1_rebuild(rows, manifest)
    _validate_gold(rows, manifest)
    _validate_manifest_stats(rows, manifest)
    _validate_manifest_claims(path, manifest)
    validations = manifest.get("validations") or {}
    expected_validations = {
        "exact_stage1_extract_coverage",
        "unique_cleaned_record_ids",
        "unique_source_row_uids",
        "nonempty_identity_fields",
        "known_sources",
        "complete_canonical_endpoints",
        "candidate_sets_hash_valid",
        "candidate_sets_not_truncated",
        "candidate_cap_respected",
        "candidate_field_byte_bound_respected",
        "candidate_json_byte_bound_respected",
    }
    if set(validations) != expected_validations or not all(validations.values()):
        raise ValueError("candidate inventory contains a failed validation")
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_CLEANED_RECORDS)
    parser.add_argument("--endpoint-profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = write_candidate_inventory(
        args.records, args.endpoint_profile, args.output, args.gold
    )
    print(
        json.dumps(
            {"rows": manifest["candidate_rows"], "sha256": manifest["candidate_sha256"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "build_candidate_rows",
    "gold_coverage",
    "validate_candidate_inventory",
    "write_candidate_inventory",
]
