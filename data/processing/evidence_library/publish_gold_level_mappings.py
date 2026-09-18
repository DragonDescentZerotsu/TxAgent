"""Publish Tianang's UID-level mappings beside every active gold release."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.evidence_library.build_source_uid_levels import (
    UPSTREAM_ROOT,
    _git_bytes,
    _membership,
)
from tools.chembl_tool.common.json_utils import write_json_atomic


ROOT = Path(__file__).resolve().parents[3]
SMALL_ORIGINALS = ROOT / (
    "data/legacy/artifacts/evidence_libraries/"
    "v10_before_level_mapping_compatibility_20260908/source_uid_levels"
)
SHARDED_ORIGINALS = ROOT / (
    "data/legacy/artifacts/evidence_libraries/"
    "v10_before_v24_1_v25_compatibility_restore_20260916/source_uid_levels"
)
TASKS = {
    "ames": {"public": "Ames", "versions": ("v1",), "kind": "dataset"},
    "bbb_martins": {
        "public": "BBB_Martins", "versions": ("v1", "v2"), "kind": "file",
        "sha256": "d30ac0c3d335e213812fc684dd8c1154eacd5edb78c0f9d09629994c93852159",
    },
    "bioavailability_ma": {
        "public": "Bioavailability_Ma", "versions": ("v1", "v2"), "kind": "file",
        "sha256": "c1d8b8f137303f9c222e2e4a0509fbcd4e48390e91459c6c757b945707731fd6",
    },
    "carcinogens": {"public": "Carcinogens", "versions": ("v1",), "kind": "dataset"},
    "dili": {"public": "DILI", "versions": ("v1",), "kind": "dataset"},
    "skin_reaction": {"public": "Skin_Reaction", "versions": ("v1",), "kind": "dataset"},
}
EXPECTED_VOTERS = {
    ("bbb_martins", "v1"): 7_634,
    ("bbb_martins", "v2"): 7_630,
    ("bioavailability_ma", "v1"): 19_479,
    ("bioavailability_ma", "v2"): 18_926,
}
ORAL_ROUTE = {
    "source_group_id": "Observed.direct_oral_bioavailability",
    "family_key": "direct_oral_bioavailability",
    "level": 1,
}
COLUMNS = [
    "source_row_uid", "canonical_record_id", "source_group_id", "family_key", "level",
]
PROTECTION_VERSION = "gold_v1_voter_protection.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def indexed(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.set_index("source_row_uid")
    if not result.index.is_unique:
        raise ValueError("duplicate source_row_uid in level mapping")
    return result


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(
            pa.Table.from_pandas(frame, preserve_index=False), temporary, compression="zstd"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_stage1_protection(mapping_root: Path, output: Path) -> dict:
    """Publish the Gold-owned L1 UIDs without inventing vote-card metadata."""
    mapping = pq.read_table(mapping_root, columns=["source_row_uid", "level"])
    rows = [
        {"task": "skin_reaction", "source_row_uid": str(row["source_row_uid"])}
        for row in mapping.to_pylist()
        if int(row["level"]) == 1
    ]
    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [("task", pa.string(), False), ("source_row_uid", pa.string(), False)]
        ),
    ).replace_schema_metadata(
        {b"schema_version": PROTECTION_VERSION.encode("utf-8")}
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "kind": "parquet_file",
        "path": output.name,
        "sha256": sha256(output),
        "rows": len(rows),
        "source": "level_mapping/level=1",
    }


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def copy_dataset(source: Path, destination: Path, parts: list[dict]) -> None:
    expected = {part["path"]: part["sha256"] for part in parts}
    if destination.exists():
        observed = {path.name: sha256(path) for path in destination.glob("part-*.parquet")}
        if observed == expected:
            return
        raise ValueError(f"published dataset already exists with different bytes: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary_name:
        temporary = Path(temporary_name) / destination.name
        shutil.copytree(source, temporary)
        for part in parts:
            if sha256(temporary / part["path"]) != part["sha256"]:
                raise ValueError(f"copied part hash mismatch: {part['path']}")
        os.replace(temporary, destination)


def apply_oral_corrections(
    original: pd.DataFrame,
    review: dict,
    canonical_records: pd.DataFrame,
    voter_uids: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    if review.get("route") != ORAL_ROUTE:
        raise ValueError("reviewed Oral L1 route changed")
    reviewed = review.get("corrections", [])
    if len(reviewed) != 26 or len({row["source_row_uid"] for row in reviewed}) != 26:
        raise ValueError("expected 26 unique reviewed Oral corrections")
    result = indexed(original.copy())
    canonical = indexed(canonical_records)
    counts = {"promote": 0, "add": 0}
    for correction in reviewed:
        uid = correction["source_row_uid"]
        if uid not in voter_uids:
            continue
        record_id = correction["canonical_record_id"]
        action = correction["action"]
        if action == "promote":
            if uid not in result.index:
                raise ValueError(f"reviewed promotion missing from original: {uid}")
            prior = result.loc[uid]
            if int(prior["level"]) != correction["expected_prior_level"]:
                raise ValueError(f"reviewed prior level changed: {uid}")
            if prior["canonical_record_id"] != record_id:
                raise ValueError(f"reviewed canonical record changed: {uid}")
            result.loc[uid, list(ORAL_ROUTE)] = list(ORAL_ROUTE.values())
        elif action == "add":
            if uid in result.index or uid not in canonical.index:
                raise ValueError(f"reviewed addition availability changed: {uid}")
            if canonical.loc[uid, "canonical_record_id"] != record_id:
                raise ValueError(f"reviewed canonical record changed: {uid}")
            result.loc[uid, COLUMNS[1:]] = [record_id, *ORAL_ROUTE.values()]
        else:
            raise ValueError(f"unsupported reviewed correction: {action}")
        counts[action] += 1
    result = result.reset_index()[COLUMNS].sort_values("source_row_uid")
    result["level"] = result["level"].astype(original["level"].dtype)
    return result.reset_index(drop=True), counts


def voter_alignment(mapping: pd.DataFrame, voters: set[str]) -> dict:
    levels = indexed(mapping)["level"]
    present = voters.intersection(levels.index)
    outside_l1 = sorted(uid for uid in present if int(levels.loc[uid]) != 1)
    missing = sorted(voters - present)
    return {
        "physical_voters": len(voters),
        "present": len(present),
        "present_at_l1": len(present) - len(outside_l1),
        "missing": missing,
        "present_outside_l1": outside_l1,
        "status": "pass" if not missing and not outside_l1 else "fail",
    }


def validate_ames_voters() -> dict:
    snapshot = json.loads(_git_bytes(f"{UPSTREAM_ROOT}/manifest.json"))
    membership = _membership(snapshot["tasks"]["ames"]["source_membership"])
    mapped = set(membership.loc[membership["level"].eq(1), "source_record_id"])
    vote_path = ROOT / "data/gold_labels/Ames/v1/scaffold/accepted_record_votes.jsonl"
    voters = {
        json.loads(line)["source_record_id"]
        for line in vote_path.read_text().splitlines()
        if line
    }
    if mapped != voters:
        raise ValueError(
            f"Ames voter/L1 mismatch: missing={len(voters - mapped)}, extra={len(mapped - voters)}"
        )
    return {
        "basis": "accepted_record_votes_exact_source_record_id_match",
        "path": str(vote_path.relative_to(ROOT)),
        "sha256": sha256(vote_path),
        "physical_voters": len(voters),
        "status": "pass",
    }


def publish_file_task(task: str, version: str, spec: dict, source_manifest: dict) -> None:
    output = ROOT / f"data/gold_labels/{spec['public']}/level_mappings/{version}"
    original_path = SMALL_ORIGINALS / f"{task}.parquet"
    if sha256(original_path) != spec["sha256"]:
        raise ValueError(f"original mapping hash changed: {task}")
    original = pq.read_table(original_path).to_pandas()
    if list(original.columns) != COLUMNS:
        raise ValueError(f"original mapping schema changed: {task}")

    membership_path = ROOT / f"data/gold_labels/{spec['public']}/{version}/scaffold/voter_membership.parquet"
    membership = pq.read_table(membership_path).to_pandas()
    if "aggregate_status" in membership:
        membership = membership.loc[membership["aggregate_status"].eq("published")]
    voters = set(membership["source_row_uid"])
    if len(voters) != EXPECTED_VOTERS[(task, version)]:
        raise ValueError(f"gold voter count changed: {task}/{version}")

    published = original
    corrections = {"add": 0, "promote": 0}
    correction_receipt = None
    if task == "bioavailability_ma":
        review_path = ROOT / "data/gold_labels/Bioavailability_Ma/level_mappings/v1/reviewed_l1_corrections.json"
        canonical_path = ROOT / "data/evidence_libraries/bioavailability_ma/v10/02_canonicalized/records.parquet"
        review = json.loads(review_path.read_text())
        canonical = pq.read_table(
            canonical_path, columns=["source_row_uid", "canonical_record_id"]
        ).to_pandas()
        published, corrections = apply_oral_corrections(original, review, canonical, voters)
        reviewed = {
            row["source_row_uid"] for row in review["corrections"]
            if row["source_row_uid"] in voters
        }
        old, new = indexed(original), indexed(published)
        unchanged = old.index.difference(list(reviewed))
        if not old.loc[unchanged].equals(new.loc[unchanged]):
            raise ValueError("an unreviewed Oral mapping row changed")
        correction_receipt = {
            "path": str(review_path.relative_to(ROOT)), "sha256": sha256(review_path),
            "canonical_records_path": str(canonical_path.relative_to(ROOT)),
            "canonical_records_sha256": sha256(canonical_path),
        }

    original_output = output / "original_level_mapping.parquet"
    mapping_output = output / "level_mapping.parquet"
    copy_file(original_path, original_output)
    if task == "bbb_martins":
        copy_file(original_path, mapping_output)
    else:
        write_parquet(published, mapping_output)
    aligned = voter_alignment(published, voters)
    if aligned["status"] != "pass":
        raise ValueError(f"gold voter alignment failed: {task}/{version}")
    report = {
        "version": "gold_level_mapping_alignment.v1", "task": task,
        "gold_release": version, "validation_basis": "gold_voter_membership",
        "original": voter_alignment(original, voters), "published": aligned,
        "reviewed_corrections": corrections, "unchanged_l3_plus": True,
    }
    report_path = output / "alignment_report.json"
    write_json_atomic(report_path, report)
    manifest = {
        "version": "gold_original_level_mapping.v1", "status": "complete",
        "task": task, "gold_release": version,
        "uid_ledger_manifest_sha256": source_manifest["uid_ledger_manifest_sha256"],
        "builder": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha256(Path(__file__))},
        "upstream": {"path": str(original_path.relative_to(ROOT)), "sha256": spec["sha256"],
                     "upstream_commit": "26d44f121141f3738764c0b00dc09e185445341b"},
        "inputs": {
            "voter_membership": {"path": str(membership_path.relative_to(ROOT)),
                                 "sha256": sha256(membership_path), "physical_voters": len(voters)},
            **({"reviewed_corrections": correction_receipt} if correction_receipt else {}),
        },
        "outputs": {
            "original_level_mapping": {"kind": "parquet_file", "path": original_output.name,
                                       "sha256": sha256(original_output), "rows": len(original)},
            "level_mapping": {"kind": "parquet_file", "path": mapping_output.name,
                              "sha256": sha256(mapping_output), "rows": len(published),
                              "rows_by_level": {str(k): int(v) for k, v in published["level"].value_counts().sort_index().items()}},
            "alignment_report": {"path": report_path.name, "sha256": sha256(report_path)},
        },
    }
    write_json_atomic(output / "manifest.json", manifest)
    print(f"{task}/{version}: {len(published):,} rows; all {len(voters):,} voters are L1")


def publish_dataset_task(task: str, version: str, spec: dict, source_manifest: dict) -> None:
    source_spec = source_manifest["tasks"][task]
    source = SHARDED_ORIGINALS / source_spec["path"]
    parts = source_spec["parts"]
    for part in parts:
        if sha256(source / part["path"]) != part["sha256"]:
            raise ValueError(f"original mapping part hash changed: {task}/{part['path']}")
    output = ROOT / f"data/gold_labels/{spec['public']}/level_mappings/{version}"
    original_output = output / "original_level_mapping"
    mapping_output = output / "level_mapping"
    copy_dataset(source, original_output, parts)
    copy_dataset(source, mapping_output, parts)

    if task == "ames":
        validation = validate_ames_voters()
    else:
        validation = {
            "basis": "pinned_tianang_l1_contract",
            "physical_voters": int(source_spec["rows_by_level"]["1"]),
            "status": "pass",
        }
    report = {
        "version": "gold_level_mapping_alignment.v1", "task": task,
        "gold_release": version, "validation_basis": validation,
        "original": validation, "published": validation,
        "reviewed_corrections": {"add": 0, "promote": 0},
        "unchanged_l3_plus": True,
    }
    report_path = output / "alignment_report.json"
    write_json_atomic(report_path, report)
    output_parts = [
        {**part, "sha256": sha256(mapping_output / part["path"])} for part in parts
    ]
    protection = (
        write_stage1_protection(
            mapping_output, output / "stage1_voter_protection.parquet"
        )
        if task == "skin_reaction"
        else None
    )
    manifest = {
        "version": "gold_original_level_mapping.v1", "status": "complete",
        "task": task, "gold_release": version,
        "uid_ledger_manifest_sha256": source_manifest["uid_ledger_manifest_sha256"],
        "builder": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha256(Path(__file__))},
        "upstream": {
            "path": str(source.relative_to(ROOT)),
            "manifest": str((SHARDED_ORIGINALS / "manifest.json").relative_to(ROOT)),
            "manifest_sha256": sha256(SHARDED_ORIGINALS / "manifest.json"),
            "upstream_commit": source_spec["upstream_commit"], "parts": parts,
        },
        "inputs": {"voter_validation": validation},
        "outputs": {
            "original_level_mapping": {"kind": "parquet_dataset", "path": original_output.name,
                                       "rows": source_spec["mapped_rows"], "parts": parts},
            "level_mapping": {"kind": "parquet_dataset", "path": mapping_output.name,
                              "rows": source_spec["mapped_rows"], "parts": output_parts,
                              "rows_by_level": source_spec["rows_by_level"]},
            **({"stage1_voter_protection": protection} if protection else {}),
            "alignment_report": {"path": report_path.name, "sha256": sha256(report_path)},
        },
    }
    write_json_atomic(output / "manifest.json", manifest)
    print(
        f"{task}/{version}: {source_spec['mapped_rows']:,} rows; "
        f"Tianang L1 contract has {validation['physical_voters']:,} rows"
    )


def publish_v1_index(source_manifest: dict) -> None:
    """Publish the multi-task runtime index without duplicating mapping data."""
    index_path = ROOT / "data/gold_labels/level_mappings.v1.json"
    tasks = {}
    for task, spec in TASKS.items():
        manifest_path = (
            ROOT / f"data/gold_labels/{spec['public']}/level_mappings/v1/manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        output = manifest["outputs"]["level_mapping"]
        mapping_path = manifest_path.parent / output["path"]
        tasks[task] = {
            **output,
            "path": str(mapping_path.relative_to(index_path.parent)),
            "manifest": str(manifest_path.relative_to(index_path.parent)),
            "manifest_sha256": sha256(manifest_path),
        }
    write_json_atomic(index_path, {
        "version": "gold_level_mappings_index.v1",
        "status": "complete",
        "gold_release": "v1",
        "uid_ledger_manifest_sha256": source_manifest["uid_ledger_manifest_sha256"],
        "tasks": tasks,
    })


def main() -> None:
    source_manifest = json.loads((SHARDED_ORIGINALS / "manifest.json").read_text())
    for task, spec in TASKS.items():
        for version in spec["versions"]:
            if spec["kind"] == "file":
                publish_file_task(task, version, spec, source_manifest)
            else:
                publish_dataset_task(task, version, spec, source_manifest)
    publish_v1_index(source_manifest)


if __name__ == "__main__":
    main()
