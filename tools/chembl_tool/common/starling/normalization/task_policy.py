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
from typing import TYPE_CHECKING, Any

from tools.chembl_tool.common.starling.compact_artifacts import CompactArtifactProfile
from tools.chembl_tool.common.starling.normalization.contracts import (
    MeasurementPair,
    NormalizedSourceProfile,
)

if TYPE_CHECKING:
    from tools.chembl_tool.common.starling.canonicalization_v7 import (
        StarlingRecordContract,
    )
    from tools.chembl_tool.common.starling.normalization.source_value_cleaning import (
        SourceValueCleaningResult,
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
    family_resolver: Callable[[str, str, Mapping[str, Any] | None], Any]
    record_enricher: Callable[[dict[str, Any]], dict[str, Any]]
    assay_transfer_revalidator: (
        Callable[[dict[str, Any]], dict[str, Any]] | None
    ) = None
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
    endpoint_registry: dict[str, Any] | None = None
    reference_semantics_manifest: dict[str, Any] | None = None


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

    # (source_id, endpoint_name, record) -> FamilyAssignment | None.  Used by the
    # normalization hooks and, independently, by the evidence catalog to
    # re-derive family labels that compaction stripped from persisted records.
    family_resolver: Callable[[str, str, Mapping[str, Any] | None], Any]

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
    # v7 persists strict source-visible and canonical projections while the
    # established scientific parsers continue to use private in-memory v6
    # aliases.  ``None`` preserves frozen v6 behavior byte for byte.
    record_contract: "StarlingRecordContract | None" = None
    # Optional Stage-01 source-visible value cleaning.  The hook runs after
    # common ingestion and before the v7 source projection, and returns the
    # field-level audit published beside the cleaned records.
    source_value_cleaner: (
        Callable[
            [list[dict[str, Any]], argparse.Namespace],
            "SourceValueCleaningResult",
        ]
        | None
    ) = None
    add_cli_arguments: Callable[[argparse.ArgumentParser], None] | None = None
    validate_arguments: Callable[[argparse.ArgumentParser, argparse.Namespace], None] | None = None
    load_extra_source: (
        Callable[
            [argparse.Namespace],
            ExtraSourceBatch | Sequence[ExtraSourceBatch] | None,
        ]
        | None
    ) = None
    # Raises if a source file no longer matches the digest the task pinned.
    verify_source_digest: Callable[[str, Path], None] | None = None
    census_extras: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None
    smiles_mapping: Callable[[argparse.Namespace], "SmilesMappingSpec | None"] | None = None
    # Frozen, task-owned scientific registries that affect normalization but
    # are not necessarily exposed as CLI arguments.  The shared build cache
    # fingerprints these alongside implementation code.
    scientific_assets: tuple[Path, ...] = ()
    # Frozen assay-transfer measurement policy. Canonical measurement/unit
    # fields are numerical training geometry, never retrieval presentation.
    assay_transfer_measurement_policy: Path | None = None
    # Stage 01 publishes the canonical endpoint and pre-LLM measurement route.
    measurement_resolution_enabled: bool = False
    # When enabled, Stage 02 publishes the frozen row-level reference
    # classification mapping and coverage as a first-class side artifact.
    reference_semantics_enabled: bool = False
    # Non-direct sources must resolve an endpoint identity before retrieval.
    # Direct-label sources keep their historical retrieval policy.
    endpoint_identity_required_sources: tuple[str, ...] = ()
    # Additional compact columns required when the task's family resolver
    # re-derives labels after a persisted-stage reload.
    family_resolver_input_fields: tuple[str, ...] = ()

    def compact_profile_for_contract(
        self, record_contract_version: str = ""
    ) -> CompactArtifactProfile:
        """Return the exact projection profile declared by an index manifest."""
        if not record_contract_version:
            return self.compact
        contract = self.record_contract
        if contract is None or record_contract_version != contract.version:
            raise ValueError(
                f"unsupported record contract for {self.task_id!r}: "
                f"{record_contract_version!r}"
            )
        from tools.chembl_tool.common.starling.canonicalization_v7 import (
            SOURCE_CONTRACT_VERSION,
        )

        return CompactArtifactProfile(
            task_id=self.compact.task_id,
            artifact_version=f"{self.compact.artifact_version}.v7",
            index_version=f"{self.compact.index_version}.v7",
            evidence_source_label=self.compact.evidence_source_label,
            source_columns={
                source_id: profile.source_visible_fields
                for source_id, profile in contract.sources.items()
            },
            llm_source_projection=contract.source_projection,
            record_contract_version=contract.version,
            source_contract_version=SOURCE_CONTRACT_VERSION,
        )


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
