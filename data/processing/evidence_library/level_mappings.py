"""Resolve the UID-level mapping owned by an evidence-library release."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.paths import (
    EVIDENCE_LIBRARIES_ROOT,
    KNOWN_RELEASES,
    REPO_ROOT,
    evidence_library_root,
    task_id,
)
from tools.chembl_tool.common.json_utils import write_json_atomic


INDEX_PATH = EVIDENCE_LIBRARIES_ROOT / "level_mappings.v1.json"
UID_LEDGER_MANIFEST = (
    REPO_ROOT / "data/raw/starling/source_row_uid_ledger/manifest.json"
)


def evidence_level_mapping_release(
    task: str, version: str | None = None
) -> tuple[Path, Path, dict[str, Any]]:
    """Return the manifest, mapping path, and output receipt for one release."""
    task = task_id(task)
    root = evidence_library_root(task, version)
    manifest_path = root / "level_mapping/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") != task or not _is_complete_release_mapping(manifest):
        raise ValueError(f"invalid evidence-library level mapping: {manifest_path}")
    if (
        manifest.get("version") == "evidence_library_level_mapping.v1"
        and manifest.get("evidence_library_version") != root.name
    ):
        raise ValueError(f"level mapping targets another release: {manifest_path}")
    receipt = manifest["output"]
    mapping_path = Path(receipt["path"])
    if not mapping_path.is_absolute():
        mapping_path = REPO_ROOT / mapping_path
    if not mapping_path.is_file():
        raise FileNotFoundError(mapping_path)
    return manifest_path, mapping_path, receipt


def _is_complete_release_mapping(manifest: dict[str, Any]) -> bool:
    if manifest.get("version") == "evidence_library_level_mapping.v1":
        return (
            manifest.get("status") == "complete"
            and (manifest.get("validation") or {}).get("stage3_uid_coverage")
            == "exact"
        )
    validations = manifest.get("validations") or {}
    return (
        manifest.get("version") == "gold_owned_level_mapping.complete_stage3_v4"
        and manifest.get("unmapped_stage3_rows") == 0
        and validations.get("exact_stage3_uid_coverage") is True
        and validations.get("l1_equals_current_physical_voters") is True
    )


def refresh_active_level_mapping_index(
    output_path: Path = INDEX_PATH,
) -> dict[str, Any]:
    """Atomically index every active release with a complete level projection."""
    tasks: dict[str, Any] = {}
    for task in KNOWN_RELEASES:
        current = EVIDENCE_LIBRARIES_ROOT / task / "CURRENT"
        if not current.is_file():
            continue
        release = current.read_text(encoding="utf-8").strip()
        root = evidence_library_root(task, release)
        manifest_path = root / "level_mapping/manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("task") != task or not _is_complete_release_mapping(manifest):
            continue
        records_path = root / "level_mapping/records.parquet"
        eligibility_path = (
            root / "level_mapping/assay_transfer_record_eligibility.parquet"
        )
        output = manifest.get("output") or {}
        eligibility = manifest.get("assay_transfer_record_eligibility") or {}
        if (
            file_sha256(records_path) != output.get("sha256")
            or file_sha256(eligibility_path) != eligibility.get("sha256")
        ):
            raise ValueError(f"active {task} level mapping differs from its manifest")
        tasks[task] = {
            "release": release,
            "kind": "parquet_file",
            "path": str(records_path.relative_to(EVIDENCE_LIBRARIES_ROOT)),
            "sha256": output["sha256"],
            "rows": output["rows"],
            "rows_by_level": output["rows_by_level"],
            "manifest": str(manifest_path.relative_to(EVIDENCE_LIBRARIES_ROOT)),
            "manifest_sha256": file_sha256(manifest_path),
            "assay_transfer_record_eligibility": str(
                eligibility_path.relative_to(EVIDENCE_LIBRARIES_ROOT)
            ),
            "assay_transfer_record_eligibility_sha256": eligibility["sha256"],
        }
    document = {
        "version": "evidence_library_level_mappings_index.v1",
        "status": "complete",
        "gold_release": "v1",
        "uid_ledger_manifest_sha256": file_sha256(UID_LEDGER_MANIFEST),
        "tasks": tasks,
    }
    write_json_atomic(output_path, document)
    return document
