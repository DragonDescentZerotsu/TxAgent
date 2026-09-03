"""Publish UID-only V9 copies of the active completed evidence libraries."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.paths import EVIDENCE_LIBRARIES_ROOT, REPO_ROOT
from data.processing.starling_source_row_identity import (
    LEDGER_ROOT,
    UID_FIELD,
    authoritative_sources,
    file_sha256,
    validate_task_sources,
)


BASELINES = {
    "bbb_martins": "v8",
    "bioavailability_ma": "v7",
    "skin_reaction": "v7",
}
CORE_ENTRIES = (
    "00_source",
    "01_cleaned",
    "02_canonicalized",
    "03_pair_buckets",
    "manifest.json",
)
LINEAGE_SCHEMA = pa.schema(
    [("stage", pa.string()), ("record_id", pa.string()), (UID_FIELD, pa.string())]
)


def _coordinate_uids(task_id: str) -> dict[tuple[str, int], str]:
    output: dict[tuple[str, int], str] = {}
    for spec in authoritative_sources():
        if spec.task_id != task_id:
            continue
        row_number = 0
        for batch in pq.ParquetFile(spec.path).iter_batches(
            batch_size=50_000, columns=[UID_FIELD]
        ):
            for uid in batch.column(0).to_pylist():
                row_number += 1
                key = (spec.source_id, row_number)
                if key in output:
                    raise ValueError(f"duplicate raw coordinate: {key}")
                output[key] = str(uid)
    return output


def _row_uids(
    table: pa.Table,
    coordinates: dict[tuple[str, int], str],
    legacy_ids: dict[str, str],
) -> list[str] | None:
    columns = set(table.column_names)
    if UID_FIELD in columns:
        values = [str(value or "") for value in table[UID_FIELD].to_pylist()]
        if not all(values):
            raise ValueError("partially populated source_row_uid column")
        return values
    values: list[str] = []
    if {"source_id", "source_row_number"} <= columns:
        for source_id, row_number in zip(
            table["source_id"].to_pylist(), table["source_row_number"].to_pylist()
        ):
            uid = coordinates.get((str(source_id or ""), int(row_number or 0)))
            if not uid:
                raise ValueError(f"unknown evidence row coordinate: {source_id}/{row_number}")
            values.append(uid)
        return values
    if "cleaned_record_id" in columns:
        for cleaned_id in table["cleaned_record_id"].to_pylist():
            uid = legacy_ids.get(str(cleaned_id or ""))
            if not uid:
                raise ValueError(f"evidence row cannot be traced to raw: {cleaned_id}")
            values.append(uid)
        return values
    return None


def _write_uid_parquet(
    path: Path,
    coordinates: dict[tuple[str, int], str],
    legacy_ids: dict[str, str],
) -> tuple[str, str] | None:
    before_hash = file_sha256(path)
    table = pq.read_table(path)
    uids = _row_uids(table, coordinates, legacy_ids)
    if uids is None:
        return None
    before = table
    table = table.append_column(UID_FIELD, pa.array(uids, pa.string()))
    temporary = path.with_name(f".{path.name}.uid.tmp")
    pq.write_table(table, temporary, compression="zstd", compression_level=3)
    if not pq.read_table(temporary).drop([UID_FIELD]).equals(before):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"UID migration changed existing values: {path}")
    os.replace(temporary, path)
    return before_hash, file_sha256(path)


def _append_columns(path: Path, columns: dict[str, list[str]]) -> tuple[str, str]:
    before_hash = file_sha256(path)
    table = pq.read_table(path)
    original = table
    for name, values in columns.items():
        if name in table.column_names:
            continue
        table = table.append_column(name, pa.array(values, pa.string()))
    temporary = path.with_name(f".{path.name}.uid.tmp")
    pq.write_table(table, temporary, compression="zstd", compression_level=3)
    if not pq.read_table(temporary).select(original.column_names).equals(original):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"UID sidecar migration changed existing values: {path}")
    os.replace(temporary, path)
    return before_hash, file_sha256(path)


def _augment_stage3_sidecars(root: Path) -> dict[str, str]:
    records = pq.read_table(
        root / "02_canonicalized/records.parquet",
        columns=["canonical_record_id", UID_FIELD],
    )
    uid_by_id = dict(
        zip(
            records["canonical_record_id"].to_pylist(),
            records[UID_FIELD].to_pylist(),
            strict=True,
        )
    )
    replacements: dict[str, str] = {}
    specs = {
        "pair_bucket_records.parquet": (("canonical_record_id", UID_FIELD),),
        "direct_record_mapping.parquet": (("canonical_record_id", UID_FIELD),),
        "duplicates.parquet": (
            ("discarded_canonical_record_id", UID_FIELD),
            ("retained_canonical_record_id", "retained_source_row_uid"),
        ),
    }
    for filename, mappings in specs.items():
        path = root / "03_pair_buckets" / filename
        table = pq.read_table(path)
        columns = {
            output_field: [str(uid_by_id[str(value)]) for value in table[id_field].to_pylist()]
            for id_field, output_field in mappings
            if output_field not in table.column_names
        }
        if columns:
            old_hash, new_hash = _append_columns(path, columns)
            replacements[old_hash] = new_hash
    return replacements


def _replace_hashes(
    value: Any,
    replacements: dict[str, str],
    path_replacements: dict[str, str],
) -> tuple[Any, bool]:
    if isinstance(value, str):
        replaced = replacements.get(value, value)
        for before, after in path_replacements.items():
            replaced = replaced.replace(before, after)
        return replaced, replaced != value
    if isinstance(value, list):
        output, changed = [], False
        for item in value:
            replacement, item_changed = _replace_hashes(
                item, replacements, path_replacements
            )
            output.append(replacement)
            changed |= item_changed
        return output, changed
    if isinstance(value, dict):
        output, changed = {}, False
        for key, item in value.items():
            replacement_key = key
            if isinstance(key, str):
                for before, after in path_replacements.items():
                    replacement_key = replacement_key.replace(before, after)
            replacement, item_changed = _replace_hashes(
                item, replacements, path_replacements
            )
            if replacement_key in output:
                raise ValueError(f"metadata path collision: {replacement_key}")
            output[replacement_key] = replacement
            changed |= item_changed or replacement_key != key
        return output, changed
    return value, False


def _refresh_file_hashes(value: Any, json_path: Path, root: Path) -> bool:
    if not isinstance(value, (dict, list)):
        return False
    if isinstance(value, list):
        changed = False
        for item in value:
            changed |= _refresh_file_hashes(item, json_path, root)
        return changed
    changed = False
    for item in value.values():
        changed |= _refresh_file_hashes(item, json_path, root)
    for key, expected in list(value.items()):
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            continue
        candidates = [Path(key)] if Path(key).is_absolute() else [
            json_path.parent / key,
            root / key,
            REPO_ROOT / key,
        ]
        target = next((path for path in candidates if path.is_file()), None)
        if target is not None:
            actual = file_sha256(target)
            if actual != expected:
                value[key] = actual
                changed = True
    artifact_path = value.get("path")
    expected = value.get("sha256")
    if isinstance(artifact_path, str) and isinstance(expected, str):
        candidates = [Path(artifact_path)] if Path(artifact_path).is_absolute() else [
            json_path.parent / artifact_path,
            root / artifact_path,
            REPO_ROOT / artifact_path,
        ]
        target = next((path for path in candidates if path.is_file()), None)
        if target is not None:
            actual = file_sha256(target)
            if actual != expected:
                value["sha256"] = actual
                changed = True
    return changed


def _repair_json_hashes(
    root: Path,
    replacements: dict[str, str],
    *,
    task_id: str,
    baseline_release: str,
) -> None:
    json_paths = sorted(root.rglob("*.json"), key=lambda path: len(path.parts), reverse=True)
    known_hashes = dict(replacements)
    path_replacements = {
        str(EVIDENCE_LIBRARIES_ROOT / task_id / baseline_release): str(root),
        f"outputs/chembl_tool/tasks/{task_id}/evidence_library/starling_normalized_{baseline_release}": str(root),
        f"data/starling_data/{task_id}": str(REPO_ROOT / "data/raw/starling" / task_id),
        str(REPO_ROOT / "tools/chembl_tool/tasks" / task_id / "data_processing"): str(
            REPO_ROOT
            / "data/processing/evidence_library/versions/v9/tasks"
            / task_id
            / "data_processing"
        ),
        str(REPO_ROOT / "data/processing/evidence_library/versions" / baseline_release): str(
            REPO_ROOT / "data/processing/evidence_library/versions/v9"
        ),
        f"data/processing/evidence_library/versions/{baseline_release}": (
            "data/processing/evidence_library/versions/v9"
        ),
    }
    for _ in range(8):
        changed_any = False
        for path in json_paths:
            before_hash = file_sha256(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload, changed = _replace_hashes(
                payload, known_hashes, path_replacements
            )
            changed |= _refresh_file_hashes(payload, path, root)
            if changed:
                path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                after_hash = file_sha256(path)
                known_hashes[before_hash] = after_hash
                changed_any = True
        if not changed_any:
            return
    raise RuntimeError("manifest hash repair did not converge")


def _write_lineage(root: Path) -> dict[str, Any]:
    output = root / "lineage/source_row_lineage.parquet"
    output.parent.mkdir(parents=True)
    writer = pq.ParquetWriter(output, LINEAGE_SCHEMA, compression="zstd", compression_level=3)
    rows = 0
    inputs = (
        "01_cleaned/records.parquet",
        "01_cleaned/structure_rejections.parquet",
        "02_canonicalized/records.parquet",
        "03_pair_buckets/records.parquet",
    )
    try:
        for relative in inputs:
            path = root / relative
            if not path.exists():
                continue
            parquet = pq.ParquetFile(path)
            id_field = next(
                field
                for field in ("canonical_record_id", "normalized_record_id", "cleaned_record_id")
                if field in parquet.schema_arrow.names
            )
            for batch in parquet.iter_batches(
                batch_size=50_000, columns=[id_field, UID_FIELD]
            ):
                writer.write_table(
                    pa.table(
                        {
                            "stage": pa.array([relative] * batch.num_rows, pa.string()),
                            "record_id": batch.column(0).cast(pa.string()),
                            UID_FIELD: batch.column(1).cast(pa.string()),
                        },
                        schema=LINEAGE_SCHEMA,
                    )
                )
                rows += batch.num_rows
    finally:
        writer.close()
    return {"path": str(output.relative_to(root)), "rows": rows, "sha256": file_sha256(output)}


def publish_task(task_id: str) -> dict[str, Any]:
    baseline_release = BASELINES[task_id]
    source_root = EVIDENCE_LIBRARIES_ROOT / task_id / baseline_release
    destination = EVIDENCE_LIBRARIES_ROOT / task_id / "v9"
    if destination.exists():
        raise FileExistsError(f"V9 destination already exists: {destination}")
    legacy_by_uid = validate_task_sources(task_id)
    legacy_by_id = {cleaned_id: uid for uid, cleaned_id in legacy_by_uid.items()}
    if len(legacy_by_id) != len(legacy_by_uid):
        raise ValueError(f"legacy cleaned IDs are not unique for {task_id}")
    coordinates = _coordinate_uids(task_id)
    if len(coordinates) != len(legacy_by_uid):
        raise ValueError(f"raw/legacy UID counts differ for {task_id}")

    stage = Path(tempfile.mkdtemp(prefix=f"{task_id}-v9-", dir=destination.parent))
    root = stage / "v9"
    root.mkdir()
    try:
        for name in CORE_ENTRIES:
            source = source_root / name
            target = root / name
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
        replacements: dict[str, str] = {}
        changed_files = 0
        for path in sorted(root.rglob("*.parquet")):
            changed = _write_uid_parquet(path, coordinates, legacy_by_id)
            if changed:
                old_hash, new_hash = changed
                replacements[old_hash] = new_hash
                changed_files += 1
        sidecar_replacements = _augment_stage3_sidecars(root)
        replacements.update(sidecar_replacements)
        changed_files += len(sidecar_replacements)
        raw_manifest = json.loads((LEDGER_ROOT / "manifest.json").read_text(encoding="utf-8"))
        for source in raw_manifest["sources"]:
            if source["task_id"] == task_id:
                replacements[source["input_sha256"]] = source["output_sha256"]
        _repair_json_hashes(
            root,
            replacements,
            task_id=task_id,
            baseline_release=baseline_release,
        )
        lineage = _write_lineage(root)
        receipt = {
            "version": "evidence_library_source_row_uid_migration.v1",
            "task_id": task_id,
            "release": "v9",
            "baseline_release": baseline_release,
            "uid_field": UID_FIELD,
            "ledger_manifest_sha256": file_sha256(LEDGER_ROOT / "manifest.json"),
            "changed_parquet_files": changed_files,
            "source_rows": len(coordinates),
            "lineage": lineage,
        }
        (root / "source_row_uid_migration.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(root, destination)
        return receipt
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(BASELINES))
    args = parser.parse_args(argv)
    print(json.dumps(publish_task(args.task), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
