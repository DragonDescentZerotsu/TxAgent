"""Resolve the UID-level mapping owned by an evidence-library release."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.processing.paths import REPO_ROOT, evidence_library_root, task_id


def evidence_level_mapping_release(
    task: str, version: str | None = None
) -> tuple[Path, Path, dict[str, Any]]:
    """Return the manifest, mapping path, and output receipt for one release."""
    task = task_id(task)
    root = evidence_library_root(task, version)
    manifest_path = root / "level_mapping/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("version") != "evidence_library_level_mapping.v1"
        or manifest.get("status") != "complete"
        or manifest.get("task") != task
        or manifest.get("evidence_library_version") != root.name
    ):
        raise ValueError(f"invalid evidence-library level mapping: {manifest_path}")
    receipt = manifest["output"]
    mapping_path = Path(receipt["path"])
    if not mapping_path.is_absolute():
        mapping_path = REPO_ROOT / mapping_path
    if not mapping_path.is_file():
        raise FileNotFoundError(mapping_path)
    return manifest_path, mapping_path, receipt
