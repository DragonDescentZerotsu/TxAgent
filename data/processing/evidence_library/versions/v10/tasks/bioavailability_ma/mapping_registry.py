"""Single registry for Bioavailability's frozen file-backed mappings."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from data.processing.evidence_library.shared.v2.mapping_registry import (
    load_mapping_registry,
    mapping_path as shared_mapping_path,
    mapping_paths as shared_mapping_paths,
    validate_mapping_registry,
)


REGISTRY_PATH = (
    Path(__file__).resolve().parent / "data_processing/mapping_registry.v1.json"
)


@cache
def mapping_registry() -> dict[str, Any]:
    return load_mapping_registry(REGISTRY_PATH, "bioavailability_ma")


def mapping_path(mapping_id: str, source_id: str | None = None) -> Path:
    return shared_mapping_path(mapping_registry(), mapping_id, source_id)


def mapping_paths(mapping_id: str) -> tuple[Path, ...]:
    return shared_mapping_paths(mapping_registry(), mapping_id)


@cache
def validate_mapping_hashes() -> None:
    validate_mapping_registry(mapping_registry())


__all__ = [
    "REGISTRY_PATH",
    "mapping_path",
    "mapping_paths",
    "mapping_registry",
    "validate_mapping_hashes",
]
