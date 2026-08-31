"""Build the layered v7 Skin_Reaction canonical-Starling evidence library.

The staged builder is shared (``common/starling/build_normalized_evidence_library``);
this entry point only binds the Skin_Reaction policy.
"""

from __future__ import annotations

from copy import copy

from tools.chembl_tool.common.starling.build_runtime import starling_build_session
from tools.chembl_tool.common.starling.build_normalized_evidence_library import (
    AUXILIARY_MAPPING_MANIFEST_FILENAME,
    CLEANED_FILENAME,
    DISTRIBUTION_AUDIT_FILENAME,
    DUPLICATES_FILENAME,
    ENDPOINT_INVENTORY_FILENAME,
    ENDPOINT_REGISTRY_FILENAME,
    EVIDENCE_BRIDGE_FILENAME,
    EVIDENCE_FAMILIES_FILENAME,
    MANIFEST_FILENAME,
    NORMALIZED_RECORDS_FILENAME,
    RECORDS_FILENAME,
    REJECTIONS_FILENAME,
    SOURCE_COLUMN_CONTRACT_FILENAME,
    SOURCE_INVENTORY_FILENAME,
    STAGES,
    VALIDITY_POLICY_FILENAME,
    build,
    parse_args,
    run_with_args,
)
NORMALIZED_RECORDS_FILENAME = "02_canonicalized/records.parquet"
RECORDS_FILENAME = "03_pair_buckets/records.parquet"
DUPLICATES_FILENAME = "03_pair_buckets/duplicates.parquet"
AUXILIARY_MAPPING_MANIFEST_FILENAME = (
    "02_canonicalized/auxiliary_mapping_manifest.json"
)
ENDPOINT_REGISTRY_FILENAME = "02_canonicalized/endpoint_registry.json"
SOURCE_COLUMN_CONTRACT_FILENAME = "02_canonicalized/source_contract.json"
VALIDITY_POLICY_FILENAME = "02_canonicalized/record_validity_policy.json"
from tools.chembl_tool.tasks.skin_reaction.build_starling_downstream_artifacts import (
    AUDIT_STAGE,
    CORE_PAIR_BUCKET_STAGE,
    HELDOUT_STAGE,
    MOLECULE_EVIDENCE_STAGE,
    NEIGHBOR_INDEX_STAGE,
    PAIR_BUCKET_STAGE,
    build_canonical_artifacts,
)
from tools.chembl_tool.tasks.skin_reaction.starling_policy import (
    DEFAULT_OUT_DIR,
    DEFAULT_STARLING_DATA_DIR,
    POLICY,
)


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
        if (
            args.allow_missing_auxiliary_mapping
            and args.through_stage not in {"source", "clean"}
        ):
            # Pair buckets and canonicalization require the reconciled mapping.
            args.through_stage = "clean"
            return run_with_args(POLICY, args)
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
