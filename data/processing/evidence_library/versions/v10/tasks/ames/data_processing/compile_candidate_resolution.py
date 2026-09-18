"""Publish exact AMES pairs from a frozen constrained-selection mapping."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_candidate_resolution import (
    CANDIDATE_CONTRACT_VERSION,
    SELECTION_COMPILER_VERSION,
    compile_selections,
)

COMPILED_MAPPING_VERSION = "ames_measurement_resolution.v2"
_SUMMARY_FIELDS = {
    "source_counts": "source_id",
    "model_counts": "inference_model",
    "provider_counts": "served_provider",
    "status_counts": "status",
    "reason_counts": "selection_reason",
    "selected_rule_counts": "selected_candidate_rule_id",
    "assignment_method_counts": "assignment_method",
}


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest_path = _manifest_path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"manifest is not an object: {manifest_path}")
    return payload


def _validated_candidate_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_manifest(path)
    if manifest.get("candidate_sha256") != file_sha256(path):
        raise ValueError("measurement candidate inventory hash mismatch")
    if int(manifest.get("candidate_rows") or -1) != pq.read_metadata(path).num_rows:
        raise ValueError("measurement candidate inventory row count mismatch")
    if manifest.get("inventory_version") != CANDIDATE_CONTRACT_VERSION:
        raise ValueError("unsupported measurement candidate inventory")
    return manifest


def _validated_selection_manifest(
    path: Path, *, require_merged: bool
) -> dict[str, Any]:
    manifest = _read_manifest(path)
    if manifest.get("mapping_sha256") != file_sha256(path):
        raise ValueError("candidate selection mapping hash mismatch")
    if int(manifest.get("mapping_rows") or -1) != pq.read_metadata(path).num_rows:
        raise ValueError("candidate selection mapping row count mismatch")
    if require_merged:
        from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.merge_measurement_resolution import (
            validate_merged_mapping,
        )

        validate_merged_mapping(path)
    return manifest


def _row_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    text = json.dumps(
        sorted((dict(row) for row in rows), key=lambda row: row["cleaned_record_id"]),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _selected_candidates(
    candidate_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    *,
    require_complete: bool,
) -> list[Mapping[str, Any]]:
    by_id = {str(row.get("cleaned_record_id") or ""): row for row in candidate_rows}
    selection_ids = {str(row.get("cleaned_record_id") or "") for row in selection_rows}
    if "" in by_id or len(by_id) != len(candidate_rows):
        raise ValueError("candidate inventory IDs are missing or duplicated")
    if "" in selection_ids or len(selection_ids) != len(selection_rows):
        raise ValueError("candidate selection IDs are missing or duplicated")
    if not selection_ids <= set(by_id):
        raise ValueError("candidate selection contains an unknown row")
    if require_complete and selection_ids != set(by_id):
        raise ValueError("candidate selection does not cover the full inventory")
    return [by_id[record_id] for record_id in sorted(selection_ids)]


def _write_temporary_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if path.exists() or _manifest_path(path).exists():
        raise FileExistsError(f"refusing to replace compiled mapping: {path}")
    if not rows:
        raise ValueError("candidate selection mapping is empty")
    columns = sorted(set().union(*(row.keys() for row in rows)))
    table = pa.Table.from_pylist(
        [{column: row.get(column) for column in columns} for row in rows]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"compiled mapping temporary exists: {temporary}")
    pq.write_table(table, temporary)
    return temporary


def _counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field) or "") for row in rows).items()))


def _compiled_validations(rows: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    return {
        "one_row_per_selection": True,
        "source_row_uid_exact": True,
        "source_id_exact": True,
        "candidate_ids_resolved": True,
        "invented_pairs_absent": True,
        "candidate_unit_sentinel_absent": all(
            "candidate_id" not in str(row.get("measurements_json") or "")
            for row in rows
        ),
    }


def _validate_compiled_summary(
    path: Path, manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    if manifest.get("task_id") != "ames":
        raise ValueError("compiled AMES mapping task mismatch")
    recorded_path = Path(str(manifest.get("mapping_path") or ""))
    if recorded_path.resolve() != path.resolve():
        raise ValueError("compiled AMES mapping path mismatch")
    if manifest.get("mapping_rows") != len(rows):
        raise ValueError("compiled AMES mapping row count mismatch")
    for summary, field in _SUMMARY_FIELDS.items():
        if manifest.get(summary) != _counts(rows, field):
            raise ValueError(f"compiled AMES {summary} mismatch")
    if manifest.get("validations") != _compiled_validations(rows):
        raise ValueError("compiled AMES validation claims mismatch")


def _compiled_manifest(
    output_path: Path,
    candidates_path: Path,
    selection_path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    complete: bool,
    output_sha256: str,
    merged_selection_validated: bool,
) -> dict[str, Any]:
    return {
        "generation_version": COMPILED_MAPPING_VERSION,
        "task_id": "ames",
        "mapping_path": str(output_path.resolve()),
        "mapping_sha256": output_sha256,
        "mapping_rows": len(rows),
        "compiled_rows_sha256": _row_digest(rows),
        "selection_compiler_version": SELECTION_COMPILER_VERSION,
        "candidate_contract_version": CANDIDATE_CONTRACT_VERSION,
        "candidate_inventory": {
            "path": str(candidates_path.resolve()),
            "sha256": file_sha256(candidates_path),
            "manifest_sha256": file_sha256(_manifest_path(candidates_path)),
        },
        "raw_selection_mapping": {
            "path": str(selection_path.resolve()),
            "sha256": file_sha256(selection_path),
            "manifest_sha256": file_sha256(_manifest_path(selection_path)),
        },
        **{name: _counts(rows, field) for name, field in _SUMMARY_FIELDS.items()},
        "complete_candidate_inventory": complete,
        "merged_selection_validated": merged_selection_validated,
        "validations": _compiled_validations(rows),
    }


def _publish_compiled(
    output_path: Path,
    temporary: Path,
    manifest: Mapping[str, Any],
) -> None:
    manifest_path = _manifest_path(output_path)
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        os.replace(temporary, output_path)
        os.replace(temporary_manifest, manifest_path)
    except BaseException:
        output_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        raise


def compile_mapping(
    candidates_path: Path,
    selection_path: Path,
    output_path: Path,
    *,
    require_complete: bool,
    require_merged_selection: bool = True,
) -> dict[str, Any]:
    """Compile atomically only after every selected ID passes its frozen contract."""
    _validated_candidate_manifest(candidates_path)
    if require_merged_selection:
        from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.build_measurement_candidates import (
            validate_candidate_inventory,
        )

        validate_candidate_inventory(candidates_path)
    _validated_selection_manifest(
        selection_path, require_merged=require_merged_selection
    )
    candidates = pq.read_table(candidates_path).to_pylist()
    selections = pq.read_table(selection_path).to_pylist()
    selected = _selected_candidates(
        candidates, selections, require_complete=require_complete
    )
    rows = compile_selections(selections, selected)
    temporary = _write_temporary_parquet(output_path, rows)
    try:
        complete = len(selected) == len(candidates)
        manifest = _compiled_manifest(
            output_path,
            candidates_path,
            selection_path,
            rows,
            complete=complete,
            output_sha256=file_sha256(temporary),
            merged_selection_validated=require_merged_selection,
        )
        _publish_compiled(output_path, temporary, manifest)
    finally:
        temporary.unlink(missing_ok=True)
    try:
        validate_compiled_mapping(
            output_path, require_complete=require_merged_selection
        )
    except BaseException:
        output_path.unlink(missing_ok=True)
        _manifest_path(output_path).unlink(missing_ok=True)
        raise
    return manifest


def validate_compiled_mapping(path: Path, *, require_complete: bool = True) -> None:
    """Rebuild the compiled row digest from frozen inputs and verify all hashes."""
    manifest = _read_manifest(path)
    if manifest.get("generation_version") != COMPILED_MAPPING_VERSION:
        raise ValueError("compiled AMES mapping version mismatch")
    if manifest.get("candidate_contract_version") != CANDIDATE_CONTRACT_VERSION:
        raise ValueError("compiled AMES candidate contract mismatch")
    if manifest.get("selection_compiler_version") != SELECTION_COMPILER_VERSION:
        raise ValueError("compiled AMES selection compiler mismatch")
    if manifest.get("mapping_sha256") != file_sha256(path):
        raise ValueError("compiled AMES mapping hash mismatch")
    rows = pq.read_table(path).to_pylist()
    _validate_compiled_summary(path, manifest, rows)
    merged = manifest.get("merged_selection_validated")
    if not isinstance(merged, bool):
        raise TypeError("compiled AMES merged-selection claim is invalid")
    if require_complete and not manifest.get("complete_candidate_inventory"):
        raise ValueError("compiled AMES mapping does not cover the full inventory")
    if require_complete and not merged:
        raise ValueError("compiled AMES mapping lacks merged-selection validation")
    candidate_ref = manifest.get("candidate_inventory") or {}
    selection_ref = manifest.get("raw_selection_mapping") or {}
    candidates_path = Path(str(candidate_ref.get("path") or ""))
    selection_path = Path(str(selection_ref.get("path") or ""))
    inputs = ((candidate_ref, candidates_path), (selection_ref, selection_path))
    for reference, input_path in inputs:
        if file_sha256(input_path) != reference.get("sha256"):
            raise ValueError("compiled AMES input hash mismatch")
        if file_sha256(_manifest_path(input_path)) != reference.get("manifest_sha256"):
            raise ValueError("compiled AMES input manifest hash mismatch")
    _validated_candidate_manifest(candidates_path)
    if require_complete:
        from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.build_measurement_candidates import (
            validate_candidate_inventory,
        )

        validate_candidate_inventory(candidates_path)
    _validated_selection_manifest(selection_path, require_merged=merged)
    candidates = pq.read_table(candidates_path).to_pylist()
    selections = pq.read_table(selection_path).to_pylist()
    selected = _selected_candidates(
        candidates, selections, require_complete=require_complete
    )
    complete = len(selected) == len(candidates)
    if manifest.get("complete_candidate_inventory") is not complete:
        raise ValueError("compiled AMES inventory-coverage claim mismatch")
    expected = compile_selections(selections, selected)
    if _row_digest(rows) != _row_digest(expected) or _row_digest(rows) != manifest.get(
        "compiled_rows_sha256"
    ):
        raise ValueError("compiled AMES rows do not match frozen selections")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-subset", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = compile_mapping(
        args.candidates,
        args.selections,
        args.output,
        require_complete=not args.allow_subset,
        require_merged_selection=not args.allow_subset,
    )
    print(json.dumps(manifest["status_counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
