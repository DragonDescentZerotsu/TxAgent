"""ClinTox binding for canonical Stage 04-05 and split-aware Stage 06-09."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling import split_downstream as _shared
from tools.chembl_tool.common.starling.build_pair_bucket_distance_calibration import (
    build_pair_bucket_distance_calibration,
)
from tools.chembl_tool.common.starling.build_pair_bucket_transfer_policy import (
    TransferPolicyBuildSpec,
)
from tools.chembl_tool.common.starling.assay_transfer_measurements import (
    load_measurement_policy,
)
from tools.chembl_tool.common.starling.normalization.audit import write_parquet
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.starling.pair_bucket_transfer_policy import (
    TransferPolicyProfile,
)
from tools.chembl_tool.common.starling.pair_buckets import (
    materialize_pair_buckets,
    read_pair_bucket_input,
)
from tools.chembl_tool.common.starling.reference_semantics import (
    ReferenceEligibilitySpec,
)
from tools.chembl_tool.common.starling.split_downstream import (
    AUDIT_STAGE,
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
from tools.chembl_tool.tasks.clintox.starling_policy import (
    DEFAULT_BENCHMARK_SPLIT_ROOT,
    EXACT_CONTEXT_ATTACHMENT_VERSION,
    EXACT_CONTEXT_MAPPING_VERSION,
    POLICY,
)
from tools.chembl_tool.tasks.clintox.starling_schema import RECORD_CONTRACT

PAIR_BUCKET_VERSION = "clintox_pair_buckets.v8"
DISTANCE_PROFILE_VERSION = "clintox_distance_calibration_profile.v2"
PIPELINE_LAYOUT_VERSION = "clintox.normalized_v7_layout.v2"
HELDOUT_OVERLAP_VERSION = "clintox.remove_heldout_direct_overlap.v2"
BENCHMARK_SPLITS = ("scaffold",)
EXCLUSIONS_FILENAME = "excluded_clinical_trial_failure_records.parquet"
GOLD_FILTER_SOURCE_ID = "clinical_trial_failure"
LEGACY_DOWNSTREAM_STAGES = (
    "04_evidence_catalog",
    "05_neighbor_index",
    "06_pair_buckets",
    "07_assay_transfer_policy",
    "08_audits",
)

TRANSFER_PROFILE = TransferPolicyProfile(
    version=DISTANCE_PROFILE_VERSION,
    source_candidate_fields={
        source: spec.variance_candidates
        for source, spec in RECORD_CONTRACT.pair_buckets.items()
    },
    minimum_distinct_levels=3,
    minimum_distinct_levels_by_scale={
        "clintox_organ_injury_binary.v1": 2,
        "clintox_genotoxicity_binary.v1": 2,
    },
)

TRANSFER_BUILD_SPEC = TransferPolicyBuildSpec(
    profile=TRANSFER_PROFILE,
    pair_bucket_version=PAIR_BUCKET_VERSION,
    source_pair_fields={
        source: spec.additional_dimensions
        for source, spec in RECORD_CONTRACT.pair_buckets.items()
    },
    auxiliary_mapping_version=EXACT_CONTEXT_MAPPING_VERSION,
    auxiliary_attachment_version=EXACT_CONTEXT_ATTACHMENT_VERSION,
    required_auxiliary_output_fields=(),
    endpoint_field_by_source={
        source: "canonical_endpoint_name"
        for source in RECORD_CONTRACT.pair_buckets
    },
)


def build_sidecar(
    *, records_path: str | Path, out_dir: str | Path
) -> dict[str, Any]:
    records_path = Path(records_path)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    records, v7, pair_fields = read_pair_bucket_input(
        records_path,
        v7_source_fields={
            source: spec.additional_dimensions
            for source, spec in RECORD_CONTRACT.pair_buckets.items()
        },
        legacy_source_fields={source: () for source in RECORD_CONTRACT.sources},
    )
    if not v7:
        raise ValueError("ClinTox normalized downstream artifacts require v7 records")
    rows, metadata = materialize_pair_buckets(
        records,
        source_required_fields=pair_fields,
        contract_version=PAIR_BUCKET_VERSION,
        reference_eligibility_by_source={
            source: ReferenceEligibilitySpec(
                spec.eligible_reference_scopes,
                spec.reference_basis_required,
            )
            for source, spec in RECORD_CONTRACT.pair_buckets.items()
        },
        required_known_fields_by_source={
            source: spec.required_known_dimensions
            for source, spec in RECORD_CONTRACT.pair_buckets.items()
            if spec.required_known_dimensions
        },
        assay_transfer_record_ineligibility=load_measurement_policy(
            POLICY.assay_transfer_measurement_policy
        )["record_ineligibility"],
    )
    if not all(metadata["validations"].values()):
        raise ValueError(f"ClinTox pair-bucket audit failed: {metadata['validations']}")
    records_output = target / PAIR_BUCKET_RECORDS_FILENAME
    write_parquet(records_output, rows)
    metadata.update(
        {
            "input": {
                "path": str(records_path),
                "sha256": file_sha256(records_path),
                "records": len(records),
            },
            "output": {
                "path": str(records_output),
                "sha256": file_sha256(records_output),
            },
            "scope": {
                "pair_enumeration": False,
                "pair_labels": False,
                "benchmark_split": None,
            },
        }
    )
    (target / PAIR_BUCKET_METADATA_FILENAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def build_distance_calibration(
    *,
    records_path: str | Path,
    pair_bucket_records_path: str | Path,
    pair_bucket_metadata_path: str | Path,
    auxiliary_manifest_path: str | Path,
    out_dir: str | Path,
    workers: int = 1,
) -> dict[str, Any]:
    return build_pair_bucket_distance_calibration(
        spec=TRANSFER_BUILD_SPEC,
        record_contract=RECORD_CONTRACT,
        records_path=records_path,
        pair_bucket_records_path=pair_bucket_records_path,
        pair_bucket_metadata_path=pair_bucket_metadata_path,
        auxiliary_manifest_path=auxiliary_manifest_path,
        out_dir=out_dir,
        workers=workers,
    )


def _spec() -> SplitDownstreamSpec:
    return SplitDownstreamSpec(
        task_id="clintox",
        policy=POLICY,
        pipeline_layout_version=PIPELINE_LAYOUT_VERSION,
        heldout_overlap_version=HELDOUT_OVERLAP_VERSION,
        pair_bucket_version=PAIR_BUCKET_VERSION,
        filter_source_id=GOLD_FILTER_SOURCE_ID,
        benchmark_splits=BENCHMARK_SPLITS,
        build_sidecar=build_sidecar,
        build_transfer_policy=build_distance_calibration,
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


__all__ = [
    "AUDIT_STAGE",
    "BENCHMARK_SPLITS",
    "DEFAULT_BENCHMARK_SPLIT_ROOT",
    "DISTANCE_CALIBRATION_STAGE",
    "DOWNSTREAM_STAGES",
    "EXCLUSIONS_FILENAME",
    "FILTERED_RECORDS_FILENAME",
    "HELDOUT_STAGE",
    "MOLECULE_EVIDENCE_STAGE",
    "NEIGHBOR_INDEX_STAGE",
    "PAIR_BUCKET_METADATA_FILENAME",
    "PAIR_BUCKET_RECORDS_FILENAME",
    "PAIR_BUCKET_STAGE",
    "PIPELINE_LAYOUT_VERSION",
    "TRANSFER_POLICY_STAGE",
    "build_canonical_artifacts",
    "build_distance_calibration",
    "build_downstream_artifacts",
    "build_filtered_molecule_evidence",
    "build_filtered_neighbor_indices",
    "build_sidecar",
    "get_spec",
    "materialize_filtered_record_views",
]
