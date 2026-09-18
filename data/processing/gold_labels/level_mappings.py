"""Resolve the UID-level mapping published with a gold-label release."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.processing.paths import GOLD_LABELS_ROOT, task_id


PUBLIC_TASK_NAMES = {
    "ames": "Ames",
    "bbb_martins": "BBB_Martins",
    "bioavailability_ma": "Bioavailability_Ma",
    "carcinogens": "Carcinogens",
    "dili": "DILI",
    "skin_reaction": "Skin_Reaction",
}


def level_mapping_release(
    task: str, version: str | None = None
) -> tuple[Path, Path, dict[str, Any]]:
    """Return and validate the manifest, mapping path, and output receipt."""
    task = task_id(task)
    task_root = GOLD_LABELS_ROOT / PUBLIC_TASK_NAMES[task]
    if version is None:
        version = (task_root / "CURRENT").read_text(encoding="utf-8").strip()
    root = task_root / "level_mappings" / version
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("task") != task
        or manifest.get("gold_release") != version
    ):
        raise ValueError(f"invalid gold level-mapping manifest: {manifest_path}")
    receipt = manifest["outputs"]["level_mapping"]
    mapping_path = root / receipt["path"]
    if receipt.get("kind") not in {"parquet_file", "parquet_dataset"}:
        raise ValueError(f"unsupported gold level-mapping output: {manifest_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(mapping_path)
    return manifest_path, mapping_path, receipt


def level_mapping_path(task: str, version: str | None = None) -> Path:
    """Return the published mapping path for one active gold release."""
    return level_mapping_release(task, version)[1]
