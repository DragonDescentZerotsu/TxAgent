"""BBB Martins profile for shared compact normalized-v6 artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.compact_artifacts import (
    CompactArtifactProfile,
)
from tools.chembl_tool.common.starling.compact_artifacts import (
    load_compact_neighbor_index as _load,
)
from tools.chembl_tool.common.starling.compact_artifacts import (
    write_compact_neighbor_index as _write,
)
from tools.chembl_tool.tasks.bbb_martins.starling_source_column_contracts import (
    SOURCE_COLUMNS,
    llm_source_projection,
)


TASK_ID = "bbb_martins"
COMPACT_ARTIFACT_VERSION = "bbb_martins.compact_v6.v1"
COMPACT_INDEX_VERSION = "bbb_martins.compact_neighbor_index.v1"

COMPACT_PROFILE = CompactArtifactProfile(
    task_id=TASK_ID,
    artifact_version=COMPACT_ARTIFACT_VERSION,
    index_version=COMPACT_INDEX_VERSION,
    evidence_source_label="Starling normalized BBB evidence",
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
    return _write(
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
    return _load(
        index_dir,
        profile=COMPACT_PROFILE,
        evidence_dir=evidence_dir,
        records_path=records_path,
    )


__all__ = [
    "COMPACT_ARTIFACT_VERSION",
    "COMPACT_INDEX_VERSION",
    "COMPACT_PROFILE",
    "load_compact_neighbor_index",
    "write_compact_neighbor_index",
]

