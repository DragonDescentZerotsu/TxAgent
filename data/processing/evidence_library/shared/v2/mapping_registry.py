"""Shared validation for task-owned, hash-pinned canonical mappings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256
from data.processing.evidence_library.shared.v2.reviewed_mapping import (
    validate_mapping_review,
)
from data.processing.paths import REPO_ROOT


def load_mapping_registry(path: Path, task_id: str) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("task_id") != task_id:
        raise ValueError(f"{task_id} mapping registry has the wrong task_id")
    sources = set(registry.get("sources") or ())
    mappings = registry.get("mappings") or {}
    if not sources or not mappings:
        raise ValueError(f"{task_id} mapping registry is empty")
    for mapping_id, entry in mappings.items():
        applicable = set(entry.get("sources") or ())
        if not applicable or not applicable <= sources:
            raise ValueError(f"{mapping_id} has invalid source coverage")
        if entry.get("stage") not in {"01_cleaned", "02_canonicalized"}:
            raise ValueError(f"{mapping_id} has an invalid application stage")
        if ("outputs" in entry) == ("outputs_by_source" in entry):
            raise ValueError(f"{mapping_id} must define outputs xor outputs_by_source")
        if "outputs_by_source" in entry and set(entry["outputs_by_source"]) != applicable:
            raise ValueError(f"{mapping_id} has incomplete per-source outputs")
        if ("path" in entry) == ("paths_by_source" in entry):
            raise ValueError(f"{mapping_id} must define path xor paths_by_source")
        if "paths_by_source" in entry and set(entry["paths_by_source"]) != applicable:
            raise ValueError(f"{mapping_id} has incomplete per-source paths")
        unknown = set(entry.get("depends_on_mappings") or ()) - set(mappings)
        if unknown:
            raise ValueError(f"{mapping_id} depends on unknown mappings: {sorted(unknown)}")
        for asset in mapping_assets(entry):
            resolved = resolve_mapping_path(asset["path"])
            if not resolved.is_file():
                raise FileNotFoundError(f"{mapping_id} mapping is absent: {resolved}")
    return registry


def mapping_assets(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry] if "path" in entry else list(entry["paths_by_source"].values())


def mapping_path(
    registry: dict[str, Any], mapping_id: str, source_id: str | None = None
) -> Path:
    entry = registry["mappings"][mapping_id]
    if "path" in entry:
        if source_id is not None and source_id not in entry["sources"]:
            raise ValueError(f"{mapping_id} does not apply to {source_id}")
        return resolve_mapping_path(entry["path"])
    if source_id is None:
        raise ValueError(f"{mapping_id} requires source_id")
    return resolve_mapping_path(entry["paths_by_source"][source_id]["path"])


def mapping_paths(registry: dict[str, Any], mapping_id: str) -> tuple[Path, ...]:
    return tuple(
        resolve_mapping_path(asset["path"])
        for asset in mapping_assets(registry["mappings"][mapping_id])
    )


def validate_mapping_registry(registry: dict[str, Any]) -> None:
    task_id = str(registry["task_id"])
    for mapping_id, entry in registry["mappings"].items():
        for asset in mapping_assets(entry):
            path = resolve_mapping_path(asset["path"])
            observed = file_sha256(path)
            if observed != asset["sha256"]:
                raise ValueError(
                    f"{mapping_id} hash mismatch: expected {asset['sha256']}, "
                    f"found {observed}"
                )
        validate_mapping_review(task_id, mapping_id, entry)


def resolve_mapping_path(path: str) -> Path:
    return REPO_ROOT / path


__all__ = [
    "load_mapping_registry",
    "mapping_path",
    "mapping_paths",
    "validate_mapping_registry",
]
