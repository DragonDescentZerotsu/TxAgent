"""Skin_Reaction binding for the shared split-aware downstream builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling import split_downstream as _shared
from tools.chembl_tool.common.starling.split_downstream import (
    AUDIT_STAGE,
    DISTANCE_CALIBRATION_STAGE,
    DOWNSTREAM_STAGES,
    FILTERED_RECORDS_FILENAME,
    HELDOUT_STAGE,
    MOLECULE_EVIDENCE_STAGE,
    NEIGHBOR_INDEX_STAGE,
    PAIR_BUCKET_STAGE,
    TRANSFER_POLICY_STAGE,
    SplitDownstreamSpec,
)
from tools.chembl_tool.tasks.skin_reaction.build_starling_pair_bucket_sidecar import (
    PAIR_BUCKET_METADATA_FILENAME,
    PAIR_BUCKET_RECORDS_FILENAME,
    build_sidecar,
)
from tools.chembl_tool.tasks.skin_reaction.build_starling_pair_bucket_transfer_policy import (
    build_pair_bucket_transfer_policy,
)
from tools.chembl_tool.tasks.skin_reaction.starling_pair_buckets import (
    SKIN_REACTION_V7_PAIR_BUCKET_VERSION,
)
from tools.chembl_tool.tasks.skin_reaction.starling_policy import POLICY


PIPELINE_LAYOUT_VERSION = "skin_reaction.normalized_v7_layout.v1"
HELDOUT_OVERLAP_VERSION = "skin_reaction.remove_heldout_overlap.v1"
BENCHMARK_SPLITS = ("random", "scaffold")
DEFAULT_SPLIT_ROOT = "data/processed_starling/Skin_Reaction"
EXCLUSIONS_FILENAME = "excluded_direct_skin_reaction_records.parquet"
LEGACY_DOWNSTREAM_STAGES = (
    "04_evidence_catalog",
    "05_neighbor_index",
    "06_pair_buckets",
    "07_assay_transfer_policy",
    "08_audits",
)


def _spec() -> SplitDownstreamSpec:
    return SplitDownstreamSpec(
        task_id="skin_reaction",
        policy=POLICY,
        pipeline_layout_version=PIPELINE_LAYOUT_VERSION,
        heldout_overlap_version=HELDOUT_OVERLAP_VERSION,
        pair_bucket_version=SKIN_REACTION_V7_PAIR_BUCKET_VERSION,
        filter_source_id="direct_skin_reaction",
        benchmark_splits=BENCHMARK_SPLITS,
        build_sidecar=build_sidecar,
        build_transfer_policy=build_pair_bucket_transfer_policy,
        exclusions_filename=EXCLUSIONS_FILENAME,
        legacy_downstream_stages=LEGACY_DOWNSTREAM_STAGES,
    )


def get_spec() -> SplitDownstreamSpec:
    return _spec()


def build_canonical_artifacts(**kwargs: Any) -> dict[str, Any]:
    return _shared.build_canonical_artifacts(_spec(), **kwargs)


def build_downstream_artifacts(**kwargs: Any) -> dict[str, Any]:
    return _shared.build_downstream_artifacts(_spec(), **kwargs)


def materialize_filtered_record_views(**kwargs: Any) -> dict[str, Any]:
    return _shared.materialize_filtered_record_views(_spec(), **kwargs)


def build_filtered_molecule_evidence(**kwargs: Any) -> dict[str, dict[str, Any]]:
    return _shared.build_filtered_molecule_evidence(_spec(), **kwargs)


def build_filtered_neighbor_indices(**kwargs: Any) -> dict[str, dict[str, Any]]:
    return _shared.build_filtered_neighbor_indices(_spec(), **kwargs)


def _publish_downstream_candidate(root: Path, candidate_root: Path) -> None:
    _shared._publish_downstream_candidate(_spec(), root, candidate_root)


__all__ = [
    "AUDIT_STAGE",
    "BENCHMARK_SPLITS",
    "DEFAULT_SPLIT_ROOT",
    "DOWNSTREAM_STAGES",
    "DISTANCE_CALIBRATION_STAGE",
    "EXCLUSIONS_FILENAME",
    "FILTERED_RECORDS_FILENAME",
    "HELDOUT_STAGE",
    "LEGACY_DOWNSTREAM_STAGES",
    "MOLECULE_EVIDENCE_STAGE",
    "NEIGHBOR_INDEX_STAGE",
    "PAIR_BUCKET_METADATA_FILENAME",
    "PAIR_BUCKET_RECORDS_FILENAME",
    "PAIR_BUCKET_STAGE",
    "PIPELINE_LAYOUT_VERSION",
    "TRANSFER_POLICY_STAGE",
    "build_downstream_artifacts",
    "build_canonical_artifacts",
    "build_filtered_molecule_evidence",
    "build_filtered_neighbor_indices",
    "get_spec",
    "materialize_filtered_record_views",
]
