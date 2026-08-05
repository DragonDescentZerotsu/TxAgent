"""BBB Martins binding for the shared split-aware normalized-v7 stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling import split_downstream as _shared
from tools.chembl_tool.common.starling.split_downstream import (
    AUDIT_STAGE,
    DEFAULT_LEGACY_DOWNSTREAM_STAGES,
    DISTANCE_CALIBRATION_STAGE,
    DOWNSTREAM_STAGES,
    FILTERED_RECORDS_FILENAME,
    HELDOUT_STAGE,
    MOLECULE_EVIDENCE_STAGE,
    NEIGHBOR_INDEX_STAGE,
    PAIR_BUCKET_METADATA_FILENAME,
    PAIR_BUCKET_RECORDS_FILENAME,
    PAIR_BUCKET_STAGE,
    TRANSFER_POLICY_STAGE,
    SplitDownstreamSpec,
)
from tools.chembl_tool.tasks.bbb_martins.build_starling_pair_bucket_sidecar import (
    build_sidecar,
)
from tools.chembl_tool.tasks.bbb_martins.build_starling_pair_bucket_transfer_policy import (
    build_pair_bucket_transfer_policy,
)
from tools.chembl_tool.tasks.bbb_martins.starling_pair_buckets import (
    BBB_MARTINS_V7_PAIR_BUCKET_VERSION,
)
from tools.chembl_tool.tasks.bbb_martins.starling_policy import POLICY


PIPELINE_LAYOUT_VERSION = "bbb_martins.normalized_v7_layout.v1"
HELDOUT_OVERLAP_VERSION = "bbb_martins.remove_heldout_overlap.v1"
BENCHMARK_SPLITS = ("random", "scaffold")
DEFAULT_SPLIT_ROOT = "data/processed_starling/BBB_Martins"
EXCLUSIONS_FILENAME = "excluded_direct_bbb_records.parquet"
LEGACY_DOWNSTREAM_STAGES = DEFAULT_LEGACY_DOWNSTREAM_STAGES


def _spec() -> SplitDownstreamSpec:
    return SplitDownstreamSpec(
        task_id="bbb_martins",
        policy=POLICY,
        pipeline_layout_version=PIPELINE_LAYOUT_VERSION,
        heldout_overlap_version=HELDOUT_OVERLAP_VERSION,
        pair_bucket_version=BBB_MARTINS_V7_PAIR_BUCKET_VERSION,
        filter_source_id="direct_bbb",
        exclusions_filename=EXCLUSIONS_FILENAME,
        build_sidecar=build_sidecar,
        build_transfer_policy=build_pair_bucket_transfer_policy,
        benchmark_splits=BENCHMARK_SPLITS,
        legacy_downstream_stages=LEGACY_DOWNSTREAM_STAGES,
    )


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
    "DISTANCE_CALIBRATION_STAGE",
    "DOWNSTREAM_STAGES",
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
    "build_filtered_molecule_evidence",
    "build_filtered_neighbor_indices",
    "materialize_filtered_record_views",
]
