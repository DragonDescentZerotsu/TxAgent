"""Assign and verify permanent identities for authoritative Starling rows.

Raw Parquet files own ``source_row_uid``.  The partitioned ledger is the
global uniqueness and lifecycle index; compressed files are publication
mirrors and never create identities.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import uuid
from math import ceil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v1.normalization.cleaning import (
    clean_text,
    stable_id,
)
from data.processing.paths import REPO_ROOT


UID_FIELD = "source_row_uid"
UID_RE = re.compile(r"^sr_[0-9a-f]{32}$")
LEDGER_VERSION = "starling_source_row_uid_ledger.v1"
RAW_ROOT = REPO_ROOT / "data/raw/starling"
LEDGER_ROOT = RAW_ROOT / "source_row_uid_ledger"
LOCK_PATH = RAW_ROOT / ".source-row-uid.lock"
INCOMPLETE_PATH = RAW_ROOT / ".source-row-uid-incomplete.json"
PARTITIONS = "0123456789abcdef"
MAX_COMPRESSED_BYTES = 100_000_000
TARGET_COMPRESSED_BYTES = 95_000_000


@dataclass(frozen=True)
class SourceSpec:
    task_id: str
    source_id: str
    path: Path
    record_id_field: str = "extraction_id"
    preserve_legacy_cleaned_id: bool = False

    @property
    def relative_path(self) -> str:
        return self.path.relative_to(RAW_ROOT).as_posix()


_SOURCE_IDS = {
    "bbb_martins/Direct_BBB": ("direct_bbb", "source_index"),
    "bbb_martins/passive_permeability": ("passive_permeability", "extraction_id"),
    "bbb_martins/efflux_transport": ("efflux_transport", "extraction_id"),
    "bbb_martins/influx_transport": ("influx_transport", "extraction_id"),
    "bioavailability_ma/Direct_HF": ("hf_bioavailability", "source_index"),
    "bioavailability_ma/Oral_AUC-Cmax_Exposure": ("oral_exposure", "extraction_id"),
    "bioavailability_ma/Fa": ("fa", "extraction_id"),
    "bioavailability_ma/Fg": ("fg", "extraction_id"),
    "bioavailability_ma/Fh": ("fh", "extraction_id"),
    "ames/ames_base": ("mutagenicity_outcomes", "extraction_id"),
    "ames/ames_v1": ("fixed_mutation", "extraction_id"),
    "ames/ames_v2": ("premutagenic_damage", "extraction_id"),
    "ames/ames_v3": ("mutagenicity_mechanism", "extraction_id"),
}
_LEGACY_TASKS = frozenset({"bbb_martins", "bioavailability_ma", "skin_reaction", "ames"})


LEDGER_SCHEMA = pa.schema(
    [
        (UID_FIELD, pa.string()),
        ("task_id", pa.string()),
        ("source_id", pa.string()),
        ("acquisition_source_path", pa.string()),
        ("acquisition_source_sha256", pa.string()),
        ("acquisition_source_row_number", pa.int64()),
        ("active_source_path", pa.string()),
        ("active_source_row_number", pa.int64()),
        ("source_record_id", pa.string()),
        ("legacy_cleaned_record_id", pa.string()),
        ("lifecycle_status", pa.string()),
    ]
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authoritative_sources(raw_root: Path = RAW_ROOT) -> list[SourceSpec]:
    sources: list[SourceSpec] = []
    for path in sorted(raw_root.glob("*/*/*.parquet")):
        if path.parent.name in {"compressed", "source_row_uid_ledger"}:
            continue
        relative_parent = path.parent.relative_to(raw_root).as_posix()
        task_id, directory = relative_parent.split("/", 1)
        source_id, record_id_field = _SOURCE_IDS.get(
            relative_parent, (directory, "extraction_id")
        )
        sources.append(
            SourceSpec(
                task_id=task_id,
                source_id=source_id,
                path=path,
                record_id_field=record_id_field,
                preserve_legacy_cleaned_id=task_id in _LEGACY_TASKS,
            )
        )
    return sources


def _connect_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.executescript(
        """
        CREATE TABLE ledger (
            source_row_uid TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            acquisition_source_path TEXT NOT NULL,
            acquisition_source_sha256 TEXT NOT NULL,
            acquisition_source_row_number INTEGER NOT NULL,
            active_source_path TEXT,
            active_source_row_number INTEGER,
            source_record_id TEXT NOT NULL,
            legacy_cleaned_record_id TEXT,
            lifecycle_status TEXT NOT NULL
        );
        CREATE TABLE seen (
            source_row_uid TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            source_row_number INTEGER NOT NULL,
            source_record_id TEXT NOT NULL,
            new_cleaned_record_id TEXT
        );
        """
    )
    return connection


def _load_ledger(connection: sqlite3.Connection, ledger_root: Path) -> None:
    for path in sorted(ledger_root.glob("part-*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=50_000):
            connection.executemany(
                "INSERT INTO ledger VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                zip(*(batch.column(name).to_pylist() for name in LEDGER_SCHEMA.names)),
            )
    connection.commit()


def _source_record_ids(batch: pa.RecordBatch, field: str, start: int) -> list[str]:
    values = (
        batch.column(batch.schema.get_field_index(field)).to_pylist()
        if field in batch.schema.names
        else [None] * batch.num_rows
    )
    return [clean_text(value) or str(start + index + 1) for index, value in enumerate(values)]


def _new_uid() -> str:
    return f"sr_{uuid.uuid4().hex}"


def _stage_source(
    spec: SourceSpec,
    stage_root: Path,
    connection: sqlite3.Connection,
    *,
    initial_bootstrap: bool,
) -> tuple[Path | None, dict[str, Any]]:
    source_hash = file_sha256(spec.path)
    parquet = pq.ParquetFile(spec.path)
    has_uid = UID_FIELD in parquet.schema_arrow.names
    if initial_bootstrap and has_uid:
        raise ValueError(f"bootstrap source already contains {UID_FIELD}: {spec.path}")

    output_path = stage_root / spec.relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_schema = parquet.schema_arrow if has_uid else parquet.schema_arrow.append(pa.field(UID_FIELD, pa.string()))
    writer: pq.ParquetWriter | None = None
    changed = not has_uid or pq.read_table(spec.path, columns=[UID_FIELD])[UID_FIELD].null_count > 0
    row_number = 0
    try:
        for batch in parquet.iter_batches(batch_size=25_000):
            record_ids = _source_record_ids(batch, spec.record_id_field, row_number)
            if has_uid:
                uid_index = batch.schema.get_field_index(UID_FIELD)
                raw_uids = batch.column(uid_index).to_pylist()
                uids = [str(value) if value else _new_uid() for value in raw_uids]
                if any(not value for value in raw_uids):
                    changed = True
                    batch = batch.set_column(uid_index, UID_FIELD, pa.array(uids, pa.string()))
            else:
                uids = [_new_uid() for _ in range(batch.num_rows)]
                batch = batch.append_column(UID_FIELD, pa.array(uids, pa.string()))
            for uid in uids:
                if not UID_RE.fullmatch(uid):
                    raise ValueError(f"invalid {UID_FIELD} in {spec.relative_path}: {uid!r}")
            cleaned_ids = [
                (
                    stable_id("cleaned", spec.source_id, source_hash, row_number + index + 1, record_ids[index])
                    if initial_bootstrap and spec.preserve_legacy_cleaned_id
                    else stable_id("cleaned", uid)
                )
                for index, uid in enumerate(uids)
            ]
            try:
                connection.executemany(
                    "INSERT INTO seen VALUES (?,?,?,?,?,?,?,?)",
                    (
                        (
                            uid,
                            spec.task_id,
                            spec.source_id,
                            spec.relative_path,
                            source_hash,
                            row_number + index + 1,
                            record_ids[index],
                            cleaned_ids[index],
                        )
                        for index, uid in enumerate(uids)
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError(f"duplicate {UID_FIELD} while reading {spec.relative_path}") from error
            if changed:
                if writer is None:
                    writer = pq.ParquetWriter(output_path, output_schema, compression="zstd", compression_level=3)
                writer.write_batch(batch)
            row_number += batch.num_rows
    finally:
        if writer is not None:
            writer.close()
    connection.commit()
    return (
        output_path if changed else None,
        {
            "task_id": spec.task_id,
            "source_id": spec.source_id,
            "path": spec.relative_path,
            "rows": row_number,
            "input_sha256": source_hash,
            "output_sha256": file_sha256(output_path) if changed else source_hash,
        },
    )


def _validate_published_legacy_ids(connection: sqlite3.Connection, repo_root: Path) -> None:
    releases = {
        "bbb_martins": "v8",
        "bioavailability_ma": "v7",
        "skin_reaction": "v7",
    }
    connection.execute(
        """
        CREATE TABLE expected_legacy (
            task_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_row_number INTEGER NOT NULL,
            cleaned_record_id TEXT NOT NULL,
            PRIMARY KEY(task_id, source_id, source_row_number)
        )
        """
    )
    for task_id, release in releases.items():
        stage = repo_root / "data/evidence_libraries" / task_id / release / "01_cleaned"
        paths = [stage / "records.parquet", stage / "structure_rejections.parquet"]
        for path in paths:
            table = pq.read_table(
                path,
                columns=["source_id", "source_row_number", "cleaned_record_id"],
            )
            connection.executemany(
                "INSERT INTO expected_legacy VALUES (?,?,?,?)",
                (
                    (task_id, source_id, row_number, cleaned_id)
                    for source_id, row_number, cleaned_id in zip(
                        table["source_id"].to_pylist(),
                        table["source_row_number"].to_pylist(),
                        table["cleaned_record_id"].to_pylist(),
                    )
                ),
            )
        audit = pq.read_table(stage / "source_value_cleaning_audit.parquet")
        for row in audit.to_pylist():
            if row.get("field") == "record" and row.get("after") == "dropped":
                connection.execute(
                    "INSERT INTO expected_legacy VALUES (?,?,?,?)",
                    (
                        task_id,
                        row["source_id"],
                        row["source_row_number"],
                        row["cleaned_record_id"],
                    ),
                )
    connection.commit()
    mismatch = connection.execute(
        """
        SELECT seen.task_id, seen.source_id, seen.source_row_number
        FROM seen LEFT JOIN expected_legacy
          ON expected_legacy.task_id=seen.task_id
         AND expected_legacy.source_id=seen.source_id
         AND expected_legacy.source_row_number=seen.source_row_number
         AND expected_legacy.cleaned_record_id=seen.new_cleaned_record_id
        WHERE seen.task_id IN ('bbb_martins','bioavailability_ma','skin_reaction')
          AND expected_legacy.cleaned_record_id IS NULL LIMIT 1
        """
    ).fetchone()
    orphan = connection.execute(
        """
        SELECT expected_legacy.task_id, expected_legacy.source_id,
               expected_legacy.source_row_number
        FROM expected_legacy LEFT JOIN seen
          ON expected_legacy.task_id=seen.task_id
         AND expected_legacy.source_id=seen.source_id
         AND expected_legacy.source_row_number=seen.source_row_number
        WHERE seen.source_row_uid IS NULL LIMIT 1
        """
    ).fetchone()
    if mismatch or orphan:
        raise ValueError(f"published Stage-1 identity does not match raw input: {mismatch or orphan}")


def _reconcile_ledger(connection: sqlite3.Connection) -> None:
    mismatch = connection.execute(
        """
        SELECT seen.source_row_uid FROM seen JOIN ledger USING(source_row_uid)
        WHERE seen.task_id != ledger.task_id OR seen.source_id != ledger.source_id
        LIMIT 1
        """
    ).fetchone()
    if mismatch:
        raise ValueError(f"an existing UID changed task/source identity: {mismatch[0]}")
    connection.execute(
        "UPDATE ledger SET lifecycle_status='retired', active_source_path=NULL, active_source_row_number=NULL"
    )
    connection.execute(
        """
        UPDATE ledger SET lifecycle_status='active',
          active_source_path=(SELECT source_path FROM seen WHERE seen.source_row_uid=ledger.source_row_uid),
          active_source_row_number=(SELECT source_row_number FROM seen WHERE seen.source_row_uid=ledger.source_row_uid)
        WHERE source_row_uid IN (SELECT source_row_uid FROM seen)
        """
    )
    connection.execute(
        """
        INSERT INTO ledger
        SELECT source_row_uid, task_id, source_id, source_path, source_sha256,
               source_row_number, source_path, source_row_number, source_record_id,
               new_cleaned_record_id, 'active'
        FROM seen WHERE source_row_uid NOT IN (SELECT source_row_uid FROM ledger)
        """
    )
    connection.commit()


def _export_ledger(connection: sqlite3.Connection, target: Path) -> dict[str, Any]:
    target.mkdir(parents=True, exist_ok=True)
    partition_rows: dict[str, int] = {}
    partition_hashes: dict[str, str] = {}
    columns = ",".join(LEDGER_SCHEMA.names)
    for partition in PARTITIONS:
        path = target / f"part-{partition}.parquet"
        writer = pq.ParquetWriter(path, LEDGER_SCHEMA, compression="zstd", compression_level=3)
        count = 0
        cursor = connection.execute(
            f"SELECT {columns} FROM ledger WHERE substr(source_row_uid,4,1)=? ORDER BY source_row_uid",
            (partition,),
        )
        while rows := cursor.fetchmany(25_000):
            writer.write_table(pa.Table.from_pylist([dict(zip(LEDGER_SCHEMA.names, row)) for row in rows], schema=LEDGER_SCHEMA))
            count += len(rows)
        writer.close()
        partition_rows[path.name] = count
        partition_hashes[path.name] = file_sha256(path)
    return {"partition_rows": partition_rows, "partition_sha256": partition_hashes}


def _verify_database(connection: sqlite3.Connection) -> dict[str, int]:
    unknown = connection.execute(
        "SELECT source_row_uid FROM seen WHERE source_row_uid NOT IN (SELECT source_row_uid FROM ledger) LIMIT 1"
    ).fetchone()
    orphan = connection.execute(
        "SELECT source_row_uid FROM ledger WHERE lifecycle_status='active' AND source_row_uid NOT IN (SELECT source_row_uid FROM seen) LIMIT 1"
    ).fetchone()
    mismatch = connection.execute(
        """
        SELECT seen.source_row_uid FROM seen JOIN ledger USING(source_row_uid)
        WHERE ledger.lifecycle_status!='active' OR seen.task_id!=ledger.task_id
           OR seen.source_id!=ledger.source_id OR seen.source_path!=ledger.active_source_path
           OR seen.source_row_number!=ledger.active_source_row_number LIMIT 1
        """
    ).fetchone()
    if unknown or orphan or mismatch:
        raise ValueError(f"raw/ledger identity mismatch: {unknown or orphan or mismatch}")
    active = connection.execute("SELECT COUNT(*) FROM ledger WHERE lifecycle_status='active'").fetchone()[0]
    retired = connection.execute("SELECT COUNT(*) FROM ledger WHERE lifecycle_status='retired'").fetchone()[0]
    seen = connection.execute("SELECT COUNT(*) FROM seen").fetchone()[0]
    if active != seen:
        raise ValueError(f"active ledger/raw counts differ: {active} != {seen}")
    return {"active_rows": active, "retired_rows": retired, "raw_rows": seen}


def sync(*, raw_root: Path = RAW_ROOT, ledger_root: Path = LEDGER_ROOT, bootstrap: bool = False) -> dict[str, Any]:
    raw_root = raw_root.resolve()
    ledger_root = ledger_root.resolve()
    if bootstrap != (not ledger_root.exists()):
        expected = "bootstrap" if not ledger_root.exists() else "ingest"
        raise ValueError(f"use {expected} for the current ledger state")
    receipt = {"version": LEDGER_VERSION, "started_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()}
    raw_root.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        INCOMPLETE_PATH.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        stage_root = Path(tempfile.mkdtemp(prefix="starling-uids-", dir=raw_root))
        descriptor, database_name = tempfile.mkstemp(prefix="starling-uids-", suffix=".sqlite3")
        os.close(descriptor)
        database_path = Path(database_name)
        connection = _connect_database(database_path)
        try:
            if ledger_root.exists():
                _load_ledger(connection, ledger_root)
            changed: list[tuple[Path, Path]] = []
            sources = []
            for spec in authoritative_sources(raw_root):
                staged, source = _stage_source(spec, stage_root, connection, initial_bootstrap=bootstrap)
                sources.append(source)
                if staged is not None:
                    changed.append((staged, spec.path))
            if bootstrap and raw_root == (REPO_ROOT / "data/raw/starling").resolve():
                _validate_published_legacy_ids(connection, REPO_ROOT)
            _reconcile_ledger(connection)
            counts = _verify_database(connection)
            staged_ledger = stage_root / "source_row_uid_ledger"
            partitions = _export_ledger(connection, staged_ledger)
            manifest = {
                **receipt,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "uid_format": "sr_<uuid4 lowercase hex>",
                "sources": sources,
                **counts,
                **partitions,
            }
            (staged_ledger / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            for staged, destination in changed:
                os.replace(staged, destination)
            ledger_root.mkdir(parents=True, exist_ok=True)
            for path in staged_ledger.iterdir():
                os.replace(path, ledger_root / path.name)
            for stale in ledger_root.glob("part-*.parquet"):
                if stale.name not in partitions["partition_rows"]:
                    stale.unlink()
            INCOMPLETE_PATH.unlink()
            return manifest
        finally:
            connection.close()
            database_path.unlink(missing_ok=True)
            shutil.rmtree(stage_root, ignore_errors=True)


def verify(*, raw_root: Path = RAW_ROOT, ledger_root: Path = LEDGER_ROOT) -> dict[str, Any]:
    if INCOMPLETE_PATH.exists():
        raise RuntimeError(f"incomplete source-row UID transaction: {INCOMPLETE_PATH}")
    manifest_path = ledger_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    descriptor, database_name = tempfile.mkstemp(prefix="starling-uids-verify-", suffix=".sqlite3")
    os.close(descriptor)
    database_path = Path(database_name)
    connection = _connect_database(database_path)
    try:
        _load_ledger(connection, ledger_root)
        sources = authoritative_sources(raw_root)
        for spec in sources:
            source_hash = file_sha256(spec.path)
            parquet = pq.ParquetFile(spec.path)
            if UID_FIELD not in parquet.schema_arrow.names:
                raise ValueError(f"authoritative source lacks {UID_FIELD}: {spec.path}")
            row_number = 0
            for batch in parquet.iter_batches(batch_size=50_000, columns=[UID_FIELD, spec.record_id_field] if spec.record_id_field in parquet.schema_arrow.names else [UID_FIELD]):
                uids = batch.column(batch.schema.get_field_index(UID_FIELD)).to_pylist()
                record_ids = _source_record_ids(batch, spec.record_id_field, row_number)
                if any(not isinstance(uid, str) or not UID_RE.fullmatch(uid) for uid in uids):
                    raise ValueError(f"invalid or missing UID in {spec.relative_path}")
                try:
                    connection.executemany(
                        "INSERT INTO seen VALUES (?,?,?,?,?,?,?,NULL)",
                        ((uid, spec.task_id, spec.source_id, spec.relative_path, source_hash, row_number + index + 1, record_ids[index]) for index, uid in enumerate(uids)),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError(f"duplicate UID in authoritative raws: {spec.relative_path}") from error
                row_number += batch.num_rows
        connection.commit()
        counts = _verify_database(connection)
        expected = {item["path"]: item for item in manifest["sources"]}
        for spec in sources:
            item = expected.get(spec.relative_path)
            if item is None or item["output_sha256"] != file_sha256(spec.path):
                raise ValueError(f"source hash is absent or stale in ledger manifest: {spec.relative_path}")
        for name, expected_hash in manifest["partition_sha256"].items():
            if file_sha256(ledger_root / name) != expected_hash:
                raise ValueError(f"ledger partition hash mismatch: {name}")
        return {"version": LEDGER_VERSION, "sources": len(sources), **counts}
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)


def validate_task_sources(
    task_id: str,
    *,
    raw_root: Path = RAW_ROOT,
    ledger_root: Path = LEDGER_ROOT,
) -> dict[str, str]:
    """Validate one task against the globally hashed ledger and return old IDs."""
    if (raw_root / ".source-row-uid-incomplete.json").exists():
        raise RuntimeError("source-row UID ledger has an incomplete transaction")
    manifest = json.loads((ledger_root / "manifest.json").read_text(encoding="utf-8"))
    for name, expected_hash in manifest["partition_sha256"].items():
        if file_sha256(ledger_root / name) != expected_hash:
            raise ValueError(f"ledger partition hash mismatch: {name}")
    ledger: dict[str, tuple[str, str, int, str | None]] = {}
    for path in sorted(ledger_root.glob("part-*.parquet")):
        table = pq.read_table(
            path,
            columns=[
                UID_FIELD,
                "task_id",
                "source_id",
                "active_source_path",
                "active_source_row_number",
                "legacy_cleaned_record_id",
                "lifecycle_status",
            ],
            filters=[("task_id", "=", task_id), ("lifecycle_status", "=", "active")],
        )
        for row in table.to_pylist():
            uid = str(row[UID_FIELD])
            if uid in ledger:
                raise ValueError(f"duplicate UID in task ledger: {uid}")
            ledger[uid] = (
                str(row["source_id"]),
                str(row["active_source_path"]),
                int(row["active_source_row_number"]),
                str(row["legacy_cleaned_record_id"]) if row["legacy_cleaned_record_id"] else None,
            )
    seen: set[str] = set()
    expected_sources = {item["path"]: item for item in manifest["sources"]}
    for spec in authoritative_sources(raw_root):
        if spec.task_id != task_id:
            continue
        expected = expected_sources.get(spec.relative_path)
        if expected is None or expected["output_sha256"] != file_sha256(spec.path):
            raise ValueError(f"task source hash is absent or stale: {spec.relative_path}")
        row_number = 0
        for batch in pq.ParquetFile(spec.path).iter_batches(batch_size=50_000, columns=[UID_FIELD]):
            for uid in batch.column(0).to_pylist():
                row_number += 1
                if not isinstance(uid, str) or uid in seen:
                    raise ValueError(f"missing or duplicate task UID: {uid!r}")
                expected_row = ledger.get(uid)
                if expected_row is None or expected_row[:3] != (
                    spec.source_id,
                    spec.relative_path,
                    row_number,
                ):
                    raise ValueError(f"task raw/ledger mismatch: {spec.relative_path}:{row_number}")
                seen.add(uid)
    if seen != set(ledger):
        raise ValueError(f"task raw/ledger cardinality mismatch: raw={len(seen)} ledger={len(ledger)}")
    return {uid: values[3] for uid, values in ledger.items() if values[3]}


def _compressed_part_paths(compressed_root: Path, source_name: str) -> list[Path]:
    single = compressed_root / f"{source_name}.parquet"
    parts = sorted(compressed_root.glob(f"{source_name}.part-*.parquet"))
    if single.exists() and parts:
        raise ValueError(f"mixed single/partitioned mirror for {source_name}")
    return [single] if single.exists() else parts


def _write_compressed_source(spec: SourceSpec, target: Path) -> dict[str, Any]:
    table = pq.read_table(spec.path)
    source_name = spec.path.parent.name
    part_count = max(1, ceil(spec.path.stat().st_size / TARGET_COMPRESSED_BYTES))
    while True:
        for stale in target.glob(f"{source_name}*.parquet"):
            stale.unlink()
        rows_per_part = ceil(table.num_rows / part_count)
        part_paths: list[Path] = []
        for index, start in enumerate(range(0, table.num_rows, rows_per_part), start=1):
            name = (
                f"{source_name}.parquet"
                if part_count == 1
                else f"{source_name}.part-{index:04d}.parquet"
            )
            path = target / name
            pq.write_table(
                table.slice(start, min(rows_per_part, table.num_rows - start)),
                path,
                compression="zstd",
                compression_level=3,
            )
            part_paths.append(path)
        if max(path.stat().st_size for path in part_paths) < MAX_COMPRESSED_BYTES:
            break
        part_count += 1
    mirrored = pa.concat_tables([pq.read_table(path) for path in part_paths])
    if not mirrored.equals(table):
        raise ValueError(f"compressed mirror differs from {spec.relative_path}")
    return {
        "task_id": spec.task_id,
        "source_id": spec.source_id,
        "authoritative_path": spec.relative_path,
        "authoritative_sha256": file_sha256(spec.path),
        "rows": table.num_rows,
        "parts": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in part_paths
        ],
    }


def publish_compressed(*, raw_root: Path = RAW_ROOT) -> dict[str, Any]:
    verify(raw_root=raw_root, ledger_root=raw_root / "source_row_uid_ledger")
    marker = raw_root / ".compressed-publication-incomplete.json"
    if marker.exists():
        raise RuntimeError(f"incomplete compressed publication: {marker}")
    stage_root = Path(tempfile.mkdtemp(prefix="starling-compressed-", dir=raw_root))
    receipt = {
        "version": "starling_compressed_mirrors.v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    marker.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        entries: list[dict[str, Any]] = []
        by_task: dict[str, list[SourceSpec]] = {}
        for spec in authoritative_sources(raw_root):
            by_task.setdefault(spec.task_id, []).append(spec)
        for task_id, specs in sorted(by_task.items()):
            target = stage_root / task_id / "compressed"
            target.mkdir(parents=True)
            task_entries = [_write_compressed_source(spec, target) for spec in specs]
            entries.extend(task_entries)
            (target / "manifest.json").write_text(
                json.dumps(
                    {"version": receipt["version"], "task_id": task_id, "sources": task_entries},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        for task_id in sorted(by_task):
            destination = raw_root / task_id / "compressed"
            staged = stage_root / task_id / "compressed"
            backup = raw_root / task_id / ".compressed-backup"
            if backup.exists():
                raise RuntimeError(f"stale compressed backup: {backup}")
            if destination.exists():
                os.replace(destination, backup)
            try:
                os.replace(staged, destination)
            except BaseException:
                if backup.exists():
                    os.replace(backup, destination)
                raise
            shutil.rmtree(backup, ignore_errors=True)
        for path in (raw_root / "ames").glob("*/*.parquet.gz"):
            path.unlink()
        marker.unlink()
        return {**receipt, "completed_at": datetime.now(timezone.utc).isoformat(), "sources": entries}
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def verify_compressed(*, raw_root: Path = RAW_ROOT) -> dict[str, int]:
    marker = raw_root / ".compressed-publication-incomplete.json"
    if marker.exists():
        raise RuntimeError(f"incomplete compressed publication: {marker}")
    sources = authoritative_sources(raw_root)
    parts_count = 0
    for spec in sources:
        compressed = raw_root / spec.task_id / "compressed"
        manifest = json.loads((compressed / "manifest.json").read_text(encoding="utf-8"))
        entry = next(
            (item for item in manifest["sources"] if item["source_id"] == spec.source_id),
            None,
        )
        if entry is None or entry["authoritative_sha256"] != file_sha256(spec.path):
            raise ValueError(f"stale compressed manifest for {spec.relative_path}")
        paths = _compressed_part_paths(compressed, spec.path.parent.name)
        if not paths or any(path.stat().st_size >= MAX_COMPRESSED_BYTES for path in paths):
            raise ValueError(f"missing or oversized compressed mirror for {spec.relative_path}")
        expected_parts = entry["parts"]
        if [path.name for path in paths] != [item["name"] for item in expected_parts]:
            raise ValueError(f"compressed part inventory mismatch for {spec.relative_path}")
        for path, expected in zip(paths, expected_parts):
            if file_sha256(path) != expected["sha256"]:
                raise ValueError(f"compressed part hash mismatch: {path}")
        if not pa.concat_tables([pq.read_table(path) for path in paths]).equals(pq.read_table(spec.path)):
            raise ValueError(f"compressed rows differ from {spec.relative_path}")
        parts_count += len(paths)
    misplaced = sorted((raw_root / "carcinogens/compressed").glob("dili*.parquet"))
    redundant = sorted((raw_root / "ames").glob("*/*.parquet.gz"))
    if misplaced or redundant:
        raise ValueError(f"superseded compressed mirrors remain: {misplaced + redundant}")
    return {"sources": len(sources), "parts": parts_count}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("bootstrap", "ingest", "verify", "publish-compressed", "verify-compressed"),
    )
    args = parser.parse_args(argv)
    if args.command == "verify":
        result = verify()
    elif args.command == "publish-compressed":
        result = publish_compressed()
    elif args.command == "verify-compressed":
        result = verify_compressed()
    else:
        result = sync(bootstrap=args.command == "bootstrap")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
