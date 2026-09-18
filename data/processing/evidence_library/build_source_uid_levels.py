"""Join an imported level snapshot to permanent acquisition UIDs via frozen lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_COMMIT = "26d44f121141f3738764c0b00dc09e185445341b"
UPSTREAM_ROOT = "artifacts/chembl_tool/starling/current_level_records"
TARGET_TASKS = ("ames", "dili", "carcinogens", "skin_reaction")
LEDGER = ROOT / "data/raw/starling/source_row_uid_ledger"


def _git_bytes(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{UPSTREAM_COMMIT}:{path}"], cwd=ROOT)


def _membership(spec: dict) -> pd.DataFrame:
    parts = spec.get("parts") or [spec]
    tables = []
    columns = [
        "canonical_record_id", "source_id", "source_record_id",
        "source_group_id", "family_key", "level",
    ]
    for part in parts:
        data = _git_bytes(f"{UPSTREAM_ROOT}/{part['path']}")
        if hashlib.sha256(data).hexdigest() != part["sha256"]:
            raise ValueError(f"Snapshot hash mismatch: {part['path']}")
        tables.append(pq.read_table(pa.BufferReader(data), columns=columns))
    return pa.concat_tables(tables).to_pandas()


def _canonical_records(spec: dict, task: str) -> pd.DataFrame:
    """Stream a pinned archive through local disk instead of retaining it in RAM."""
    with tempfile.TemporaryDirectory(prefix=f"{task}-records-", dir="/local") as name:
        temporary = Path(name)
        compressed_path = temporary / "stage.tar.zst"
        archive_digest = hashlib.sha256()
        with compressed_path.open("wb") as output:
            for part in spec["parts"]:
                data = _git_bytes(part["path"])
                if hashlib.sha256(data).hexdigest() != part["sha256"]:
                    raise ValueError(f"Archive part hash mismatch: {part['path']}")
                archive_digest.update(data)
                output.write(data)
        if archive_digest.hexdigest() != spec["archive_sha256"]:
            raise ValueError(f"Archive hash mismatch: {task}")
        tar_path = temporary / "stage.tar"
        subprocess.run(["zstd", "-d", "-f", str(compressed_path), "-o", str(tar_path)], check=True)
        parquet_path = temporary / "records.parquet"
        with tarfile.open(tar_path) as archive:
            member = next(
                (member for member in archive.getmembers() if member.name == "records.parquet"),
                None,
            )
            if member is None:
                raise ValueError(f"Missing records.parquet: {task}")
            with archive.extractfile(member) as source, parquet_path.open("wb") as output:
                shutil.copyfileobj(source, output)
        with parquet_path.open("rb") as handle:
            records_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        if records_sha256 != spec["records_sha256"]:
            raise ValueError(f"Canonical records hash mismatch: {task}")
        return pq.read_table(parquet_path, columns=[
            "canonical_record_id", "source_id", "source_record_id", "source_row_number",
        ]).to_pandas()


def _write_parts(result: pd.DataFrame, destination: Path) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix=f"{destination.name}-", dir=destination.parent) as name:
        temporary = Path(name) / destination.name
        temporary.mkdir()
        parts = []
        for number, start in enumerate(range(0, len(result), 250_000)):
            path = temporary / f"part-{number:05d}.parquet"
            pq.write_table(
                pa.Table.from_pandas(result.iloc[start:start + 250_000], preserve_index=False),
                path,
                compression="zstd",
            )
            parts.append({
                "path": str(path.relative_to(temporary)),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        old_file = destination.with_suffix(".parquet")
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
        if old_file.exists():
            old_file.unlink()
        return parts


def map_uids(
    membership: pd.DataFrame,
    canonical: pd.DataFrame,
    ledger: pd.DataFrame,
    source_aliases: dict[str, str],
) -> pd.DataFrame:
    """Require exact record lineage and a single family/level per acquisition UID."""
    record_keys = ["canonical_record_id", "source_id", "source_record_id"]
    result = membership.merge(canonical, on=record_keys, how="left", validate="one_to_one")
    if result["source_row_number"].isna().any():
        raise ValueError("Membership records are missing from the frozen canonical lineage")
    result["ledger_source_id"] = result["source_id"].map(source_aliases)
    if result["ledger_source_id"].isna().any():
        raise ValueError("Canonical lineage contains an unknown source directory")
    generated_ids = result.apply(
        lambda row: row["source_record_id"]
        == f"{row['source_id']}:{int(row['source_row_number'])}",
        axis=1,
    )
    uid_ids = result["source_record_id"].str.fullmatch(r"sr_[0-9a-f]{32}")
    if generated_ids.all():
        result["acquisition_row_number"] = result["source_row_number"] + 1
        left_keys = ["ledger_source_id", "acquisition_row_number"]
        right_keys = ["source_id", "acquisition_source_row_number"]
    elif uid_ids.all():
        result["acquisition_row_number"] = result["source_row_number"] + 1
        left_keys = ["ledger_source_id", "acquisition_row_number", "source_record_id"]
        right_keys = ["source_id", "acquisition_source_row_number", "source_row_uid"]
    else:
        result["acquisition_row_number"] = result["source_row_number"]
        left_keys = ["ledger_source_id", "acquisition_row_number", "source_record_id"]
        right_keys = ["source_id", "acquisition_source_row_number", "source_record_id"]
    result = result.merge(
        ledger,
        left_on=left_keys,
        right_on=right_keys,
        how="left", validate="many_to_one",
        suffixes=("", "_ledger"),
    )
    if result["source_row_uid"].isna().any():
        missing = result.loc[
            result["source_row_uid"].isna(),
            ["source_id", "ledger_source_id", "source_record_id", "source_row_number"],
        ]
        raise ValueError(
            f"Canonical lineage has {len(missing)} rows without an exact acquisition UID; "
            f"examples={missing.head(5).to_dict('records')}"
        )
    if not result["source_row_uid"].str.fullmatch(r"sr_[0-9a-f]{32}").all():
        raise ValueError("Invalid acquisition UID")
    if not (result["acquisition_row_number"] == result["acquisition_source_row_number"]).all():
        raise ValueError("Canonical and acquisition row numbers disagree")
    if (result.groupby("source_row_uid")[["level", "family_key"]].nunique() > 1).any().any():
        raise ValueError("An acquisition UID has conflicting family/level assignments")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TARGET_TASKS, default=TARGET_TASKS)
    args = parser.parse_args()
    snapshot_bytes = _git_bytes(f"{UPSTREAM_ROOT}/manifest.json")
    snapshot = json.loads(snapshot_bytes)
    ledger_manifest = json.loads((LEDGER / "manifest.json").read_text())
    for name, expected in ledger_manifest["partition_sha256"].items():
        with (LEDGER / name).open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise ValueError(f"UID ledger hash mismatch: {name}")
    archives = json.loads(_git_bytes("artifacts/chembl_tool/starling/current_records/manifest.json"))
    existing = args.output_dir / "manifest.json"
    existing_tasks = json.loads(existing.read_text()).get("tasks", {}) if existing.exists() else {}
    receipt = {
        "version": "source_uid_levels.v2",
        "snapshot_manifest_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "uid_ledger_manifest_sha256": hashlib.sha256((LEDGER / "manifest.json").read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "join": "canonical_record_id + upstream source directory + source_record_id -> source directory alias + acquisition row number -> permanent source_row_uid",
        "scope": "Imported retrieval-eligible source membership only; absent UIDs are unmapped, not inferred exclusions.",
        "tasks": existing_tasks,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        spec = snapshot["tasks"][task]
        source = spec["source_membership"]
        membership = _membership(source)
        if len(membership) != source["n_records"]:
            raise ValueError(f"Snapshot row-count mismatch: {task}")
        archive_spec = archives["tasks"][task]
        canonical = _canonical_records(archive_spec, task)
        ledger = pq.read_table(
            LEDGER, filters=[("task_id", "=", task)], ignore_prefixes=[".", "_", "manifest"],
            columns=["source_row_uid", "source_id", "source_record_id", "acquisition_source_row_number"],
        ).to_pandas()
        task_manifest_path = ROOT / f"data/processing/evidence_library/versions/v10/tasks/{task}/source_manifest.json"
        if task_manifest_path.is_file():
            task_manifest = json.loads(task_manifest_path.read_text())
            source_aliases = {
                spec["directory"]: source_id
                for source_id, spec in task_manifest["sources"].items()
            }
        else:
            source_aliases = {source_id: source_id for source_id in ledger["source_id"].unique()}
        joined = map_uids(membership, canonical, ledger, source_aliases)
        result = joined[["source_row_uid", "canonical_record_id", "source_group_id", "family_key", "level"]].sort_values("source_row_uid")
        destination = args.output_dir / task
        output_parts = _write_parts(result, destination)
        receipt["tasks"][task] = {
            "upstream_commit": UPSTREAM_COMMIT,
            "path": destination.name,
            "parts": output_parts,
            "source_membership": source,
            "canonical_records_sha256": archive_spec["records_sha256"],
            "source_rows": len(membership), "mapped_rows": len(result),
            "unique_uids": int(result["source_row_uid"].nunique()),
            "unmatched_rows": 0, "conflicting_uid_levels": 0,
            "rows_by_level": {str(k): int(v) for k, v in result["level"].value_counts().sort_index().items()},
        }
        print(f"{task}: mapped {len(result):,} rows into {len(output_parts)} part(s)", flush=True)
        (args.output_dir / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
