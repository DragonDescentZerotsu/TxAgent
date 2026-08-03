"""Bioavailability_Ma binding for the shared compact-artifact implementation.

The algorithms live in ``common/starling/compact_artifacts.py``.  This module
only pins this task's artifact/index versions, evidence-source label, and
source-column contract, and re-exports the wrappers under their historical
names so existing callers and tests keep working.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.compact_artifacts import (
    BANNED_PERSISTED_FIELDS,
    CompactArtifactProfile,
    assert_compact_schema,
    build_relational_evidence_catalog,
    compact_persisted_records,
)
from tools.chembl_tool.common.starling.compact_artifacts import (
    load_compact_neighbor_index as _load_compact_neighbor_index,
)
from tools.chembl_tool.common.starling.compact_artifacts import (
    write_compact_neighbor_index as _write_compact_neighbor_index,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_source_column_contracts import (
    SOURCE_COLUMNS,
    llm_source_projection,
)


COMPACT_ARTIFACT_VERSION = "bioavailability_ma.compact_v6.v1"
COMPACT_INDEX_VERSION = "bioavailability_ma.compact_neighbor_index.v1"
EVIDENCE_SOURCE_LABEL = "Starling normalized oral bioavailability"

COMPACT_PROFILE = CompactArtifactProfile(
    task_id="bioavailability_ma",
    artifact_version=COMPACT_ARTIFACT_VERSION,
    index_version=COMPACT_INDEX_VERSION,
    evidence_source_label=EVIDENCE_SOURCE_LABEL,
    source_columns=SOURCE_COLUMNS,
    llm_source_projection=llm_source_projection,
)


def write_compact_neighbor_index(
    *,
    families: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    workers: int = 1,
    progress_every: int = 0,
) -> dict[str, Any]:
    return _write_compact_neighbor_index(
        profile=COMPACT_PROFILE,
        families=families,
        output_dir=output_dir,
        workers=workers,
        progress_every=progress_every,
    )


def load_compact_neighbor_index(
    index_dir: str | Path,
    *,
    evidence_dir: str | Path | None = None,
    records_path: str | Path | None = None,
) -> dict[str, Any]:
    return _load_compact_neighbor_index(
        index_dir,
        profile=COMPACT_PROFILE,
        evidence_dir=evidence_dir,
        records_path=records_path,
    )


__all__ = [
    "BANNED_PERSISTED_FIELDS",
    "COMPACT_ARTIFACT_VERSION",
    "COMPACT_INDEX_VERSION",
    "COMPACT_PROFILE",
    "EVIDENCE_SOURCE_LABEL",
    "assert_compact_schema",
    "build_relational_evidence_catalog",
    "compact_persisted_records",
    "load_compact_neighbor_index",
    "write_compact_neighbor_index",
]
