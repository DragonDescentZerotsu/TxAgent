"""Join an imported level snapshot to permanent acquisition UIDs via frozen lineage."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_COMMIT = "00800a5b890440ac52e4e7689c4a157d0ff2a365"
SNAPSHOT = ROOT / "artifacts/chembl_tool/starling/current_level_records"
LEDGER = ROOT / "data/raw/starling/source_row_uid_ledger"
OUTPUT = ROOT / "data/artifacts/starling/source_uid_levels"


def map_uids(membership: pd.DataFrame, canonical: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """Require exact record lineage and a single family/level per acquisition UID."""
    record_keys = ["canonical_record_id", "source_id", "source_record_id"]
    result = membership.merge(canonical, on=record_keys, how="left", validate="one_to_one")
    if result["cleaned_record_id"].isna().any():
        raise ValueError("Membership records are missing from the frozen canonical lineage")
    result = result.merge(
        ledger,
        left_on=["cleaned_record_id", "source_id", "source_record_id"],
        right_on=["legacy_cleaned_record_id", "source_id", "source_record_id"],
        how="left", validate="many_to_one",
    )
    if result["source_row_uid"].isna().any():
        raise ValueError("Canonical lineage has no exact acquisition UID match")
    if not result["source_row_uid"].str.fullmatch(r"sr_[0-9a-f]{32}").all():
        raise ValueError("Invalid acquisition UID")
    if not (result["source_row_number"] == result["acquisition_source_row_number"]).all():
        raise ValueError("Canonical and acquisition row numbers disagree")
    if (result.groupby("source_row_uid")[["level", "family_key"]].nunique() > 1).any().any():
        raise ValueError("An acquisition UID has conflicting family/level assignments")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    snapshot = json.loads((SNAPSHOT / "manifest.json").read_text())
    ledger_manifest = json.loads((LEDGER / "manifest.json").read_text())
    for name, expected in ledger_manifest["partition_sha256"].items():
        with (LEDGER / name).open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise ValueError(f"UID ledger hash mismatch: {name}")
    archives = json.loads(subprocess.check_output([
        "git", "show", f"{UPSTREAM_COMMIT}:artifacts/chembl_tool/starling/current_records/manifest.json",
    ], cwd=ROOT))
    receipt = {
        "upstream_commit": UPSTREAM_COMMIT,
        "snapshot_manifest_sha256": hashlib.sha256((SNAPSHOT / "manifest.json").read_bytes()).hexdigest(),
        "uid_ledger_manifest_sha256": hashlib.sha256((LEDGER / "manifest.json").read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "join": "canonical_record_id + source_id + source_record_id -> cleaned_record_id -> legacy_cleaned_record_id -> source_row_uid",
        "scope": "Imported retrieval-eligible source membership only; absent UIDs are unmapped, not inferred exclusions.",
        "tasks": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task, spec in snapshot["tasks"].items():
        for surface in [spec["source_membership"], *spec["indices"].values()]:
            data = (SNAPSHOT / surface["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != surface["sha256"]:
                raise ValueError(f"Snapshot hash mismatch: {surface['path']}")
        source = spec["source_membership"]
        membership = pq.read_table(SNAPSHOT / source["path"], columns=[
            "canonical_record_id", "source_id", "source_record_id", "source_group_id", "family_key", "level",
        ]).to_pandas()
        archive_spec = archives["tasks"][task]
        parts = []
        for part in archive_spec["parts"]:
            data = subprocess.check_output(["git", "show", f"{UPSTREAM_COMMIT}:{part['path']}"], cwd=ROOT)
            if hashlib.sha256(data).hexdigest() != part["sha256"]:
                raise ValueError(f"Archive part hash mismatch: {part['path']}")
            parts.append(data)
        compressed = b"".join(parts)
        if hashlib.sha256(compressed).hexdigest() != archive_spec["archive_sha256"]:
            raise ValueError(f"Archive hash mismatch: {task}")
        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(compressed)) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                for member in archive:
                    if member.name == "records.parquet":
                        data = archive.extractfile(member).read()
                        break
                else:
                    raise ValueError(f"Missing records.parquet: {task}")
        if hashlib.sha256(data).hexdigest() != archive_spec["records_sha256"]:
            raise ValueError(f"Canonical records hash mismatch: {task}")
        canonical = pq.read_table(pa.BufferReader(data), columns=[
            "canonical_record_id", "cleaned_record_id", "source_id", "source_record_id", "source_row_number",
        ]).to_pandas()
        ledger = pq.read_table(
            LEDGER, filters=[("task_id", "=", task)], ignore_prefixes=[".", "_", "manifest"],
            columns=["source_row_uid", "legacy_cleaned_record_id", "source_id", "source_record_id", "acquisition_source_row_number"],
        ).to_pandas()
        joined = map_uids(membership, canonical, ledger)
        result = joined[["source_row_uid", "canonical_record_id", "source_group_id", "family_key", "level"]].sort_values("source_row_uid")
        destination = args.output_dir / f"{task}.parquet"
        temporary = destination.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pandas(result, preserve_index=False), temporary, compression="zstd")
        temporary.replace(destination)
        receipt["tasks"][task] = {
            "path": destination.name,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "source_membership_sha256": source["sha256"],
            "canonical_records_sha256": archive_spec["records_sha256"],
            "source_rows": len(membership), "mapped_rows": len(result),
            "unique_uids": int(result["source_row_uid"].nunique()),
            "unmatched_rows": 0, "conflicting_uid_levels": 0,
            "rows_by_level": {str(k): int(v) for k, v in result["level"].value_counts().sort_index().items()},
        }
        print(task, receipt["tasks"][task], flush=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
