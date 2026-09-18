"""Import the pinned main-tree Starling universe into active raw data.

This is a one-time acquisition step. Evidence-library builds consume only the
published ``data/raw`` output and never depend on ``artifacts/`` at runtime.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_COMMIT = "26d44f121141f3738764c0b00dc09e185445341b"
UPSTREAM_MANIFEST = "artifacts/chembl_tool/starling/current_records/manifest.json"
DEFAULT_OUTPUT_ROOT = ROOT / "data/raw/starling/main_universe_v1"
UID_LEDGER = ROOT / "data/raw/starling/source_row_uid_ledger"
PUBLIC_NAMES = {
    "bbb_martins": "BBB_Martins",
    "bioavailability_ma": "Bioavailability_Ma",
    "skin_reaction": "Skin_Reaction",
    "ames": "Ames",
    "dili": "DILI",
    "carcinogens": "Carcinogens",
}


def _git_bytes(path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{UPSTREAM_COMMIT}:{path}"], cwd=ROOT
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _restore_records(task: str, spec: dict[str, Any], temporary: Path) -> Path:
    compressed = temporary / "stage.tar.zst"
    archive_digest = hashlib.sha256()
    with compressed.open("wb") as output:
        for part in spec["parts"]:
            process = subprocess.Popen(
                ["git", "show", f"{UPSTREAM_COMMIT}:{part['path']}"],
                cwd=ROOT,
                stdout=subprocess.PIPE,
            )
            assert process.stdout is not None
            part_digest = hashlib.sha256()
            size = 0
            while chunk := process.stdout.read(1024 * 1024):
                output.write(chunk)
                part_digest.update(chunk)
                archive_digest.update(chunk)
                size += len(chunk)
            if process.wait() or size != int(part["size"]):
                raise ValueError(f"failed to restore upstream archive part: {part['path']}")
            if part_digest.hexdigest() != part["sha256"]:
                raise ValueError(f"upstream archive part hash mismatch: {part['path']}")
    if archive_digest.hexdigest() != spec["archive_sha256"]:
        raise ValueError(f"upstream archive hash mismatch: {task}")
    archive = temporary / "stage.tar"
    subprocess.run(
        ["zstd", "-q", "-d", "-f", str(compressed), "-o", str(archive)],
        check=True,
    )
    records = temporary / "records.parquet"
    with tarfile.open(archive) as handle:
        member = next(
            (item for item in handle.getmembers() if item.name == "records.parquet"),
            None,
        )
        if member is None or not member.isfile():
            raise ValueError(f"upstream archive lacks records.parquet: {task}")
        source = handle.extractfile(member)
        if source is None:
            raise ValueError(f"cannot read upstream records: {task}")
        with source, records.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        duplicate_member = next(
            (item for item in handle.getmembers() if item.name == "duplicates.parquet"),
            None,
        )
        if duplicate_member is not None and duplicate_member.isfile():
            duplicate_source = handle.extractfile(duplicate_member)
            if duplicate_source is None:
                raise ValueError(f"cannot read upstream duplicates: {task}")
            with duplicate_source, (temporary / "duplicates.parquet").open("wb") as output:
                shutil.copyfileobj(duplicate_source, output, length=1024 * 1024)
    if _sha256(records) != spec["records_sha256"]:
        raise ValueError(f"upstream records hash mismatch: {task}")
    return records


def _uid_ledger_hash() -> str:
    manifest = UID_LEDGER / "manifest.json"
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    for name, expected in spec["partition_sha256"].items():
        if _sha256(UID_LEDGER / name) != expected:
            raise ValueError(f"UID ledger hash mismatch: {name}")
    return _sha256(manifest)


def _gold_voter_uids(task: str) -> set[str]:
    if task not in {"bbb_martins", "bioavailability_ma"}:
        return set()
    path = ROOT / "data/gold_labels" / PUBLIC_NAMES[task] / "v1/scaffold/voter_membership.parquet"
    return {
        str(value)
        for value in pq.read_table(path, columns=["source_row_uid"])["source_row_uid"].to_pylist()
    }


def _gold_canonical_ids(task: str, voter_uids: set[str]) -> dict[str, str]:
    if not voter_uids:
        return {}
    root = ROOT / "data/gold_labels" / PUBLIC_NAMES[task] / "level_mappings/v1"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    path = root / manifest["outputs"]["original_level_mapping"]["path"]
    table = pq.read_table(path, columns=["source_row_uid", "canonical_record_id"])
    result = {
        str(uid): str(canonical_id)
        for uid, canonical_id in zip(
            table["source_row_uid"].to_pylist(),
            table["canonical_record_id"].to_pylist(),
            strict=True,
        )
        if str(uid) in voter_uids
    }
    corrections = root / "reviewed_l1_corrections.json"
    if corrections.is_file():
        for row in json.loads(corrections.read_text(encoding="utf-8"))["corrections"]:
            uid = str(row["source_row_uid"])
            if uid in voter_uids:
                result[uid] = str(row["canonical_record_id"])
    return result


def _uid_mapping(task: str) -> tuple[dict[tuple[Any, ...], str], dict[str, dict[str, Any]]]:
    table = ds.dataset(UID_LEDGER, format="parquet", exclude_invalid_files=True).to_table(
        filter=ds.field("task_id") == task,
        columns=[
            "source_id", "source_record_id", "acquisition_source_row_number",
            "source_row_uid", "legacy_cleaned_record_id",
        ],
    )
    aliases = {
        "ames_base": "mutagenicity_outcomes",
        "ames_v1": "fixed_mutation",
        "ames_v2": "premutagenic_damage",
        "ames_v3": "mutagenicity_mechanism",
    }
    mapping: dict[tuple[Any, ...], str] = {}
    details: dict[str, dict[str, Any]] = {}
    for row in table.to_pylist():
        source_id = str(row["source_id"])
        row_number = int(row["acquisition_source_row_number"])
        key = (
            source_id,
            row_number,
        ) if task == "ames" else (
            source_id,
            row_number,
            str(row["source_record_id"]),
        )
        if key in mapping:
            raise ValueError(f"UID ledger has duplicate physical identity: {key}")
        uid = str(row["source_row_uid"])
        mapping[key] = uid
        details[uid] = row
    if task == "ames":
        return (
            {(upstream, row): mapping[(ledger, row)] for upstream, ledger in aliases.items()
             for row in (key[1] for key in mapping if key[0] == ledger)},
            details,
        )
    return mapping, details


def _attach_uids(
    task: str, source: Path, duplicates_path: Path, destination: Path
) -> tuple[int, int, int]:
    source_file = pq.ParquetFile(source)
    has_uid = "source_row_uid" in source_file.schema_arrow.names
    mapping, uid_details = ({}, {}) if has_uid else _uid_mapping(task)
    voter_uids = _gold_voter_uids(task)
    gold_canonical_ids = _gold_canonical_ids(task, voter_uids)
    used_voter_uids: set[str] = set()
    emitted_uids: set[str] = set()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    writer = None
    rows = 0
    duplicate_rows = (
        pq.read_table(duplicates_path).to_pylist() if duplicates_path.is_file() else []
    )
    duplicate_by_removed = {
        str(row["removed_normalized_record_id"]): row for row in duplicate_rows
    }
    protected_duplicates = {
        uid: duplicate_by_removed[canonical_id]
        for uid, canonical_id in gold_canonical_ids.items()
        if canonical_id in duplicate_by_removed
    }
    kept_ids = {
        str(row["kept_normalized_record_id"]) for row in protected_duplicates.values()
    }
    kept_rows: dict[str, dict[str, Any]] = {}
    try:
        for batch in source_file.iter_batches(batch_size=100_000):
            table = pa.Table.from_batches([batch])
            source_ids = table["source_id"].to_pylist()
            source_rows = table["source_row_number"].to_pylist()
            record_ids = table["source_record_id"].to_pylist()
            if kept_ids:
                canonical_ids = table["canonical_record_id"].to_pylist()
                for index, canonical_id in enumerate(canonical_ids):
                    if canonical_id in kept_ids:
                        kept_rows[str(canonical_id)] = table.slice(index, 1).to_pylist()[0]
            if has_uid:
                batch_uids = [str(value) for value in table["source_row_uid"].to_pylist()]
            else:
                keys = [
                    (str(source_id), int(source_row) + 1)
                    if task == "ames"
                    else (str(source_id), int(source_row), str(record_id))
                    for source_id, source_row, record_id in zip(
                        source_ids, source_rows, record_ids, strict=True
                    )
                ]
                missing = [key for key in keys if key not in mapping]
                if missing:
                    raise ValueError(f"main rows lack permanent acquisition UIDs: {missing[:5]}")
                batch_uids = [mapping[key] for key in keys]
                table = table.append_column(
                    "source_row_uid", pa.array(batch_uids, type=pa.string())
                )
            for uid in batch_uids:
                if uid in voter_uids:
                    used_voter_uids.add(uid)
            duplicates = emitted_uids & set(batch_uids)
            within_batch = {
                uid for uid, count in Counter(batch_uids).items() if count > 1
            }
            if duplicates or within_batch:
                raise ValueError(
                    f"UID bridge produced duplicate identity: {min(duplicates or within_batch)}"
                )
            emitted_uids.update(batch_uids)
            table = table.replace_schema_metadata()
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(table)
        missing_uids = voter_uids - used_voter_uids
        rehydrated = []
        for uid in sorted(missing_uids):
            duplicate = protected_duplicates.get(uid)
            if duplicate is None:
                raise ValueError(f"Gold voter is absent and not an upstream duplicate: {uid}")
            kept_id = str(duplicate["kept_normalized_record_id"])
            if kept_id not in kept_rows:
                raise ValueError(f"upstream duplicate target is absent: {kept_id}")
            row = dict(kept_rows[kept_id])
            detail = uid_details[uid]
            row.update(
                source_id=str(detail["source_id"]),
                source_record_id=str(detail["source_record_id"]),
                source_row_number=int(detail["acquisition_source_row_number"]),
                source_row_uid=uid,
                canonical_record_id=str(duplicate["removed_normalized_record_id"]),
            )
            if "cleaned_record_id" in row and detail["legacy_cleaned_record_id"]:
                row["cleaned_record_id"] = str(detail["legacy_cleaned_record_id"])
            if "source_index" in row and detail["source_id"] in {
                "direct_bbb",
                "hf_bioavailability",
            }:
                row["source_index"] = int(detail["source_record_id"])
            rehydrated.append(row)
            used_voter_uids.add(uid)
        if rehydrated:
            table = pa.Table.from_pylist(rehydrated, schema=source_file.schema_arrow.append(
                pa.field("source_row_uid", pa.string())
            )).replace_schema_metadata()
            assert writer is not None
            writer.write_table(table)
            emitted_uids.update(row["source_row_uid"] for row in rehydrated)
            rows += len(rehydrated)
    finally:
        if writer is not None:
            writer.close()
    if used_voter_uids != voter_uids:
        missing = voter_uids - used_voter_uids
        raise ValueError(f"Gold voter physical rows absent from main: {sorted(missing)[:10]}")
    os.replace(temporary, destination)
    return rows, len(used_voter_uids), len(rehydrated)


def import_task(task: str, root: Path, upstream: dict[str, Any]) -> dict[str, Any]:
    spec = upstream["tasks"][task]
    with tempfile.TemporaryDirectory(prefix=f"{task}-main-universe-", dir="/local") as name:
        records = _restore_records(task, spec, Path(name))
        destination = root / task / "records.parquet"
        expected_rows = pq.ParquetFile(records).metadata.num_rows
        rows, voter_uid_rows, rehydrated_rows = _attach_uids(
            task, records, Path(name) / "duplicates.parquet", destination
        )
        expected_rows += rehydrated_rows
    if rows != expected_rows:
        raise ValueError(f"{task} imported {rows:,} rows; expected {expected_rows:,}")
    return {
        "path": str(destination.relative_to(root)),
        "rows": rows,
        "sha256": _sha256(destination),
        "upstream_records_sha256": spec["records_sha256"],
        "uid_join": "permanent acquisition identity by source, physical row, and source record ID",
        "gold_voter_physical_uid_rows": voter_uid_rows,
        "gold_voter_duplicates_rehydrated": rehydrated_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tasks", nargs="+", choices=tuple(PUBLIC_NAMES), default=list(PUBLIC_NAMES))
    args = parser.parse_args(argv)
    upstream_bytes = _git_bytes(UPSTREAM_MANIFEST)
    upstream = json.loads(upstream_bytes)
    uid_ledger_manifest_sha256 = _uid_ledger_hash()
    args.output_root.mkdir(parents=True, exist_ok=True)
    existing = args.output_root / "manifest.json"
    tasks = json.loads(existing.read_text()).get("tasks", {}) if existing.is_file() else {}
    manifest = {
        "version": "main_source_universe.v1",
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_manifest": UPSTREAM_MANIFEST,
        "upstream_manifest_sha256": hashlib.sha256(upstream_bytes).hexdigest(),
        "source_structure_policy": "upstream canonical_smiles is the V10 source seed",
        "uid_ledger_manifest": str((UID_LEDGER / "manifest.json").relative_to(ROOT)),
        "uid_ledger_manifest_sha256": uid_ledger_manifest_sha256,
        "tasks": tasks,
    }
    for task in args.tasks:
        manifest["tasks"][task] = import_task(task, args.output_root, upstream)
        temporary = existing.with_name(".manifest.json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, existing)
        print(f"{task}: imported {manifest['tasks'][task]['rows']:,} records", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
