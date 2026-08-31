"""Build the three-stage canonical Starling v7 BBB Martins library."""

from __future__ import annotations

from copy import copy

from tools.chembl_tool.common.starling.build_runtime import starling_build_session
from tools.chembl_tool.common.starling.build_normalized_evidence_library import (
    build,
    parse_args,
    run_with_args,
)
from tools.chembl_tool.tasks.bbb_martins.build_starling_downstream_artifacts import (
    AUDIT_STAGE,
    CORE_PAIR_BUCKET_STAGE,
    HELDOUT_STAGE,
    MOLECULE_EVIDENCE_STAGE,
    NEIGHBOR_INDEX_STAGE,
    PAIR_BUCKET_STAGE,
    build_canonical_artifacts,
)
from tools.chembl_tool.tasks.bbb_martins.starling_policy import (
    DEFAULT_OUT_DIR,
    POLICY,
)


NORMALIZED_RECORDS_FILENAME = "02_canonicalized/records.parquet"
RECORDS_FILENAME = "03_pair_buckets/records.parquet"
DUPLICATES_FILENAME = "03_pair_buckets/duplicates.parquet"


CANONICAL_ARTIFACT_STAGES = (
    "00_source",
    "01_cleaned",
    "02_canonicalized",
    CORE_PAIR_BUCKET_STAGE,
)
LEGACY_ARTIFACT_STAGES = (
    "00_source",
    "01_cleaned",
    "02_canonicalized",
    "03_records",
    PAIR_BUCKET_STAGE,
    "05_distance_calibration",
    HELDOUT_STAGE,
    MOLECULE_EVIDENCE_STAGE,
    NEIGHBOR_INDEX_STAGE,
    AUDIT_STAGE,
)
ARTIFACT_STAGES = CANONICAL_ARTIFACT_STAGES
EVIDENCE_FAMILIES_TEMPLATE = (
    f"{MOLECULE_EVIDENCE_STAGE}/{{benchmark_split}}/molecule_families.parquet"
)
EVIDENCE_BRIDGE_TEMPLATE = (
    f"{MOLECULE_EVIDENCE_STAGE}/{{benchmark_split}}/molecule_family_records.parquet"
)
INDEX_MOLECULES_TEMPLATE = (
    f"{NEIGHBOR_INDEX_STAGE}/{{benchmark_split}}/molecules.parquet"
)
INDEX_FINGERPRINTS_TEMPLATE = (
    f"{NEIGHBOR_INDEX_STAGE}/{{benchmark_split}}/fingerprints.npz"
)
INDEX_MEMBERSHIP_TEMPLATE = (
    f"{NEIGHBOR_INDEX_STAGE}/{{benchmark_split}}/group_membership.parquet"
)
INDEX_META_TEMPLATE = f"{NEIGHBOR_INDEX_STAGE}/{{benchmark_split}}/manifest.json"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(POLICY, argv, "pair-buckets", True)
    with starling_build_session(args.out_dir):
        if args.through_stage != "pair-buckets":
            return run_with_args(POLICY, args)

        if args.from_stage != "pair-buckets":
            record_args = copy(args)
            record_args.through_stage = "normalize"
            result = run_with_args(POLICY, record_args)
            if result:
                return result

        build_canonical_artifacts(
            normalized_root=args.out_dir,
            workers=args.workers,
            rebuild_request={
                "from_stage": args.from_stage,
                "through_stage": "03_pair_buckets",
            },
            validation_level=args.validation_level,
            cache_mode=args.cache_mode,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
