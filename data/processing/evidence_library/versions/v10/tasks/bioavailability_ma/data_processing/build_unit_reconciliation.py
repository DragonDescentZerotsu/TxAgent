"""Build reviewed, endpoint-independent Oral Bio V10 unit spelling aliases."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from data.processing.paths import REPO_ROOT
from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256
from data.processing.evidence_library.shared.v2.normalization.measurement_resolution import load_exact_unit_mapping

TASK = "bioavailability_ma"
ASSET_ROOT = Path(__file__).parent / "canonicalization_v8"
REVIEW = ASSET_ROOT / "bioavailability_unit_review.v1.jsonl"
REVIEW_MANIFEST = ASSET_ROOT / "bioavailability_unit_review.v1.manifest.json"
OUTPUT = ASSET_ROOT / "bioavailability_unit_reconciliation.v1.json"


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_inputs(manifest_path: Path, review_path: Path) -> tuple[dict, dict]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("task") != TASK:
        raise ValueError("unit-review task mismatch")
    for name, expected in manifest["inputs"].items():
        if file_sha256(_path(name)) != expected["sha256"]:
            raise ValueError(f"unit-review input hash mismatch: {name}")
    if file_sha256(review_path) != manifest.get("review_sha256"):
        raise ValueError("unit-review decision hash mismatch")
    selected = manifest["selected_inputs"]
    names = ("cleaned_records", "measurement_resolution", "unit_vocabulary")
    if any(selected[name] not in manifest["inputs"] for name in names):
        raise ValueError("selected input is not frozen by manifest")
    return manifest, {name: _path(selected[name]) for name in names}


def _observed_units(paths: dict) -> tuple[set[str], int]:
    columns = ["cleaned_record_id", "source_row_uid", "unit_text",
               "measurement_resolution_route", "measurement_resolution_exact_unit"]
    source = pq.read_table(paths["cleaned_records"], columns=columns).to_pylist()
    by_id = {r["cleaned_record_id"]: r for r in source}
    if len(by_id) != len(source) or any(not r["source_row_uid"] for r in source):
        raise ValueError("invalid cleaned source identity")
    units = {str(r["unit_text"]).strip() for r in source if r["unit_text"]}
    for row in source:
        if row["measurement_resolution_route"] == "accept":
            unit = row["measurement_resolution_exact_unit"] or row["unit_text"]
            if not unit or not str(unit).strip():
                raise ValueError("accepted source scalar has no unit")
            units.add(str(unit).strip())
    extraction = pq.read_table(paths["measurement_resolution"], columns=[
        "cleaned_record_id", "source_row_uid", "status", "measurements_json"
    ]).to_pylist()
    if len({r["cleaned_record_id"] for r in extraction}) != len(extraction):
        raise ValueError("duplicate extraction identity")
    ok_rows = 0
    for row in extraction:
        parent = by_id.get(row["cleaned_record_id"])
        if parent is None or parent["source_row_uid"] != row["source_row_uid"]:
            raise ValueError("extraction/source identity mismatch")
        if row["status"] != "ok":
            continue
        measurements = json.loads(row["measurements_json"] or "[]")
        if len(measurements) != 1 or not str(measurements[0].get("unit") or "").strip():
            raise ValueError("ok extraction requires one unit-bearing outcome")
        units.add(str(measurements[0]["unit"]).strip())
        ok_rows += 1
    vocabulary = json.loads(paths["unit_vocabulary"].read_text())
    units.update(entry["unit"] for entry in vocabulary["units"] if any(
        name.startswith(f"{TASK}:") for name in entry.get("sources", [])
    ))
    if "" in units:
        raise ValueError("observed unit is empty")
    return units, ok_rows


def _review_entries(reviews: list[dict], units: set[str]) -> list[dict]:
    by_unit = {row["input_unit"]: row for row in reviews}
    if len(by_unit) != len(reviews) or set(by_unit) != units:
        raise ValueError("review must cover every observed unit exactly once")
    entries = []
    for row in reviews:
        unit, target = row["input_unit"], row["canonical_unit"]
        decision = "keep_distinct" if unit == target else "merge_alias"
        if row.get("decision") != decision or not row.get("rationale"):
            raise ValueError(f"invalid unit-only decision: {unit}")
        if target not in by_unit or by_unit[target]["canonical_unit"] != target:
            raise ValueError(f"alias target must be an observed stable key: {unit}")
        if any(row.get(name) for name in (
            "endpoint_rules", "excluded_endpoints", "canonical_endpoint", "canonical_endpoints"
        )):
            raise ValueError("endpoint decisions are outside unit unification")
        if row.get("scale", "1") != "1" or row.get("domain", "any") != "any":
            raise ValueError("unit unification cannot transform or restrict values")
        if unit in {"ratio", "fraction", "%"} and target != unit:
            raise ValueError("ratio, fraction, and percent must remain distinct")
        if not isinstance(row.get("review_round"), int) or row["review_round"] < 1:
            raise ValueError("invalid review round")
        entries.append(dict(task=TASK, canonical_endpoints=["*"], input_unit=unit,
                            action="map", canonical_unit=target, scale="1", domain="any",
                            review_basis=f"manual_unit_review_500_round_{row['review_round']:02d}"))
    return sorted(entries, key=lambda entry: entry["input_unit"])


def build_mapping(manifest_path: Path = REVIEW_MANIFEST, review_path: Path = REVIEW) -> dict:
    manifest, paths = _load_inputs(manifest_path, review_path)
    units, ok_rows = _observed_units(paths)
    reviews = [json.loads(line) for line in review_path.read_text().splitlines() if line]
    entries = _review_entries(reviews, units)
    round_counts = Counter(row["review_round"] for row in reviews)
    if any(count > 500 for count in round_counts.values()):
        raise ValueError("review round exceeds 500 units")
    return dict(version="starling_exact_measurement_units.v2", entries=entries,
                bioavailability_v10_contract=dict(
                    scope="oral_bioavailability_endpoint_independent",
                    value_transform="identity", scale_representation="distinct_unit_identity",
                    inputs=manifest["inputs"], selected_inputs=manifest["selected_inputs"],
                    review_path=str(review_path), review_sha256=file_sha256(review_path),
                    review_manifest_path=str(manifest_path),
                    review_manifest_sha256=file_sha256(manifest_path),
                    counts=dict(combined_units=len(units), llm_ok_rows=ok_rows),
                    per_round_counts={str(k): v for k, v in sorted(round_counts.items())},
                    decision_counts=dict(Counter(row["decision"] for row in reviews))))


def main() -> int:
    payload = build_mapping()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    load_exact_unit_mapping(OUTPUT)
    print(f"wrote {len(payload['entries']):,} Oral Bio unit rules to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
