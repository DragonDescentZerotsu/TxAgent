"""Build the seven-source ClinTox normalized Starling v7 evidence library."""

from __future__ import annotations

from copy import copy

from tools.chembl_tool.common.starling.build_normalized_evidence_library import (
    CLEANED_FILENAME,
    DISTRIBUTION_AUDIT_FILENAME,
    DUPLICATES_FILENAME,
    ENDPOINT_INVENTORY_FILENAME,
    MANIFEST_FILENAME,
    RECORDS_FILENAME,
    REJECTIONS_FILENAME,
    SOURCE_INVENTORY_FILENAME,
    STAGES,
    parse_args,
    run_with_args,
)
from tools.chembl_tool.tasks.clintox.build_starling_downstream_artifacts import (
    AUDIT_STAGE,
    DISTANCE_CALIBRATION_STAGE,
    HELDOUT_STAGE,
    MOLECULE_EVIDENCE_STAGE,
    NEIGHBOR_INDEX_STAGE,
    PAIR_BUCKET_STAGE,
    build_canonical_artifacts,
    build_downstream_artifacts,
)
from tools.chembl_tool.tasks.clintox.starling_policy import POLICY

NORMALIZED_RECORDS_FILENAME = "02_canonicalized/records.parquet"
AUXILIARY_MAPPING_MANIFEST_FILENAME = (
    "02_canonicalized/auxiliary_mapping_manifest.json"
)
ENDPOINT_REGISTRY_FILENAME = "02_canonicalized/endpoint_registry.json"
SOURCE_COLUMN_CONTRACT_FILENAME = "02_canonicalized/source_contract.json"
VALIDITY_POLICY_FILENAME = "02_canonicalized/record_validity_policy.json"

CANONICAL_ARTIFACT_STAGES = (
    "01_cleaned",
    "02_canonicalized",
    "03_records",
    PAIR_BUCKET_STAGE,
    DISTANCE_CALIBRATION_STAGE,
)
LEGACY_ARTIFACT_STAGES = (
    *CANONICAL_ARTIFACT_STAGES,
    HELDOUT_STAGE,
    MOLECULE_EVIDENCE_STAGE,
    NEIGHBOR_INDEX_STAGE,
    AUDIT_STAGE,
)
ARTIFACT_STAGES = CANONICAL_ARTIFACT_STAGES


def main(argv: list[str] | None = None) -> int:
    args = parse_args(POLICY, argv)
    if args.through_stage != "index":
        return run_with_args(POLICY, args)

    if args.from_stage != "index":
        record_args = copy(args)
        record_args.through_stage = "organize"
        result = run_with_args(POLICY, record_args)
        if result:
            return result

    common = {
        "normalized_root": args.out_dir,
        "workers": args.workers,
        "rebuild_request": {
            "from_stage": args.from_stage,
            "through_stage": args.through_stage,
        },
        "validation_level": args.validation_level,
        "cache_mode": args.cache_mode,
    }
    if args.legacy_task_local_downstream:
        build_downstream_artifacts(
            **common,
            benchmark_split_root=args.benchmark_split_root,
            progress_every=args.progress_every,
            max_record_examples=args.max_record_examples,
        )
    else:
        build_canonical_artifacts(**common)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_STAGES",
    "AUXILIARY_MAPPING_MANIFEST_FILENAME",
    "CANONICAL_ARTIFACT_STAGES",
    "CLEANED_FILENAME",
    "DISTRIBUTION_AUDIT_FILENAME",
    "DUPLICATES_FILENAME",
    "ENDPOINT_INVENTORY_FILENAME",
    "ENDPOINT_REGISTRY_FILENAME",
    "LEGACY_ARTIFACT_STAGES",
    "MANIFEST_FILENAME",
    "NORMALIZED_RECORDS_FILENAME",
    "RECORDS_FILENAME",
    "REJECTIONS_FILENAME",
    "SOURCE_COLUMN_CONTRACT_FILENAME",
    "SOURCE_INVENTORY_FILENAME",
    "STAGES",
    "VALIDITY_POLICY_FILENAME",
    "main",
]
