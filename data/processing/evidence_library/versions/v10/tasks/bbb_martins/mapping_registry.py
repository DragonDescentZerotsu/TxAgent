"""Single registry for BBB's frozen file-backed mappings."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256
from data.processing.paths import REPO_ROOT


REGISTRY_PATH = Path(__file__).resolve().parent / "data_processing/mapping_registry.v1.json"


@cache
def mapping_registry() -> dict[str, Any]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if registry.get("task_id") != "bbb_martins":
        raise ValueError("BBB mapping registry has the wrong task_id")
    sources = set(registry.get("sources") or ())
    mappings = registry.get("mappings") or {}
    if not sources or not mappings:
        raise ValueError("BBB mapping registry is empty")
    for mapping_id, entry in mappings.items():
        applicable = set(entry.get("sources") or ())
        if not applicable or not applicable <= sources:
            raise ValueError(f"{mapping_id} has invalid source coverage")
        if entry.get("stage") not in {"01_cleaned", "02_canonicalized"}:
            raise ValueError(f"{mapping_id} has an invalid application stage")
        if ("outputs" in entry) == ("outputs_by_source" in entry):
            raise ValueError(f"{mapping_id} must define outputs xor outputs_by_source")
        if ("path" in entry) == ("paths_by_source" in entry):
            raise ValueError(f"{mapping_id} must define path xor paths_by_source")
        if "paths_by_source" in entry and set(entry["paths_by_source"]) != applicable:
            raise ValueError(f"{mapping_id} has incomplete per-source paths")
        unknown = set(entry.get("depends_on_mappings") or ()) - set(mappings)
        if unknown:
            raise ValueError(f"{mapping_id} depends on unknown mappings: {sorted(unknown)}")
        for path in _entry_paths(entry):
            if not path.is_file():
                raise FileNotFoundError(f"{mapping_id} mapping is absent: {path}")
    return registry


def mapping_path(mapping_id: str, source_id: str | None = None) -> Path:
    entry = mapping_registry()["mappings"][mapping_id]
    if "path" in entry:
        if source_id is not None and source_id not in entry["sources"]:
            raise ValueError(f"{mapping_id} does not apply to {source_id}")
        return _resolved(entry["path"])
    if source_id is None:
        raise ValueError(f"{mapping_id} requires source_id")
    return _resolved(entry["paths_by_source"][source_id]["path"])


@cache
def validate_mapping_hashes() -> None:
    for mapping_id, entry in mapping_registry()["mappings"].items():
        assets = [entry] if "path" in entry else list(entry["paths_by_source"].values())
        for asset in assets:
            path = _resolved(asset["path"])
            observed = file_sha256(path)
            if observed != asset["sha256"]:
                raise ValueError(
                    f"{mapping_id} hash mismatch: expected {asset['sha256']}, found {observed}"
                )


def _entry_paths(entry: dict[str, Any]):
    if "path" in entry:
        yield _resolved(entry["path"])
    else:
        for source_id in entry["sources"]:
            yield _resolved(entry["paths_by_source"][source_id]["path"])


def _resolved(path: str) -> Path:
    return REPO_ROOT / path


__all__ = ["REGISTRY_PATH", "mapping_path", "mapping_registry", "validate_mapping_hashes"]
