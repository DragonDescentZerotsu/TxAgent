"""Bioavailability_Ma binding for the shared split-aware downstream builder."""

from __future__ import annotations

import os
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
from tools.chembl_tool.tasks.bioavailability_ma.build_starling_pair_bucket_sidecar import (
    PAIR_BUCKET_METADATA_FILENAME,
    PAIR_BUCKET_RECORDS_FILENAME,
    build_sidecar,
)
from tools.chembl_tool.tasks.bioavailability_ma.build_starling_pair_bucket_transfer_policy import (
    build_pair_bucket_transfer_policy,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_pair_buckets import (
    BIOAVAILABILITY_PAIR_BUCKET_VERSION,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_policy import POLICY


PIPELINE_LAYOUT_VERSION = "bioavailability_ma.normalized_v7_layout.v2"
HELDOUT_OVERLAP_VERSION = "bioavailability_ma.remove_heldout_overlap.v2"
BENCHMARK_SPLITS = ("random", "scaffold")
DEFAULT_SPLIT_ROOT = "data/processed_starling/Bioavailability_Ma"
EXCLUSIONS_FILENAME = "excluded_direct_bioavailability_records.parquet"
LEGACY_DOWNSTREAM_STAGES = _shared.LEGACY_DOWNSTREAM_STAGES


def _spec() -> SplitDownstreamSpec:
    return SplitDownstreamSpec(
        task_id="bioavailability_ma",
        policy=POLICY,
        pipeline_layout_version=PIPELINE_LAYOUT_VERSION,
        heldout_overlap_version=HELDOUT_OVERLAP_VERSION,
        pair_bucket_version=BIOAVAILABILITY_PAIR_BUCKET_VERSION,
        filter_source_id="hf_bioavailability",
        filter_scope_field="canonical_bioavailability_evidence_scope",
        filter_scope_value="direct",
        benchmark_splits=BENCHMARK_SPLITS,
        build_sidecar=build_sidecar,
        build_transfer_policy=build_pair_bucket_transfer_policy,
        exclusions_filename=EXCLUSIONS_FILENAME,
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
    "build_filtered_molecule_evidence",
    "build_filtered_neighbor_indices",
    "materialize_filtered_record_views",
]
