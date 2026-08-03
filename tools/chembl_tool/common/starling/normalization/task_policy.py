"""Per-task plug-in contract for the layered normalized-Starling builder.

The staged builder in ``common/starling/build_normalized_evidence_library.py``
is task-agnostic.  Everything that differs between tasks -- source schemas,
endpoint orthography, unit standardization, factual-validity domains, auxiliary
context attachment, and the version strings that land in ``manifest.json`` --
is supplied by a :class:`StarlingTaskPolicy`.

Each task publishes exactly one policy as ``POLICY`` in
``tools/chembl_tool/tasks/<task>/starling_policy.py``.  The builder and the
directory-index loader resolve it by that convention; there is no separate
registry to keep in sync.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.compact_artifacts import CompactArtifactProfile
from tools.chembl_tool.common.starling.normalization.contracts import (
    MeasurementPair,
    NormalizedSourceProfile,
)


@dataclass(frozen=True)
class NormalizationHooks:
    """The callables ``normalize_cleaned_records`` injects for one build.

    ``record_enricher`` is built per run because it may close over run-scoped
    state (an auxiliary-mapping attacher, for example), which is why the policy
    exposes a factory rather than a bare function.  ``run_state`` carries that
    same state back to ``stage_documents``, which needs it to write the
    stage's coverage audit.
    """

    endpoint_normalizer: Callable[[str, str], Any]
    endpoint_standardizer: Callable[[str, MeasurementPair], MeasurementPair]
    family_resolver: Callable[[str, str], Any]
    record_enricher: Callable[[dict[str, Any]], dict[str, Any]]
    source_measurement_resolver: Callable[..., Any] | None = None
    contextual_standardizer: Callable[..., Any] | None = None
    run_state: Any = None


@dataclass(frozen=True)
class ExtraSourceBatch:
    """Rows from a source that is loaded outside the profile-driven sweep.

    Bioavailability's pinned Direct-HF snapshot is the motivating case: it has
    its own column contract and row-coverage assertion rather than a plain
    ``extractions.parquet`` read.
    """

    source_id: str
    profile: NormalizedSourceProfile
    rows: list[dict[str, Any]]
    source_path: Path
    source_sha256: str
    endpoint_names: list[str]
    inventory_key: str = "direct_records"
    inventory_entry: dict[str, Any] | None = None


@dataclass(frozen=True)
class StageDocuments:
    """Stage-scoped JSON side artifacts and manifest fields for ``normalize``."""

    validity_policy: dict[str, Any]
    auxiliary_mapping_manifest: dict[str, Any]
    source_column_contract: dict[str, Any]
    validations: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StarlingTaskPolicy:
    """Everything the staged builder needs that is specific to one task."""

    task_id: str
    dataset_name: str
    default_data_dir: str
    default_out_dir: str
    compact: CompactArtifactProfile
    expected_source_rows: Mapping[str, int]

    # Source ingestion.
    source_profiles: Callable[[Path], Sequence[NormalizedSourceProfile]]
    endpoint_inventory: Callable[..., dict[str, Any]]

    # (source_id, endpoint_name) -> FamilyAssignment | None.  Used by the
    # normalization hooks and, independently, by the evidence catalog to
    # re-derive family labels that compaction stripped from persisted records.
    family_resolver: Callable[[str, str], Any]

    # Normalization.
    build_hooks: Callable[[argparse.Namespace], NormalizationHooks]
    attach_source_columns: Callable[
        [list[dict[str, Any]]], list[dict[str, Any]]
    ]
    stage_documents: Callable[..., StageDocuments]

    # Manifest.  Called as ``manifest_versions()`` for a complete build and
    # ``manifest_versions(complete=False)`` for a partial one, so a task can
    # withhold keys that only mean something once the index exists.
    manifest_versions: Callable[..., dict[str, Any]]

    # Optional extensions.
    add_cli_arguments: Callable[[argparse.ArgumentParser], None] | None = None
    validate_arguments: Callable[[argparse.ArgumentParser, argparse.Namespace], None] | None = None
    load_extra_source: Callable[[argparse.Namespace], ExtraSourceBatch | None] | None = None
    # Raises if a source file no longer matches the digest the task pinned.
    verify_source_digest: Callable[[str, Path], None] | None = None
    census_extras: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None
    smiles_mapping: Callable[[argparse.Namespace], "SmilesMappingSpec | None"] | None = None


@dataclass(frozen=True)
class SmilesMappingSpec:
    """An identifier-to-SMILES mapping and the hash the task pins it to."""

    path: Path
    expected_sha256: str = ""
    allow_unpinned: bool = False


__all__ = [
    "ExtraSourceBatch",
    "NormalizationHooks",
    "SmilesMappingSpec",
    "StageDocuments",
    "StarlingTaskPolicy",
]
