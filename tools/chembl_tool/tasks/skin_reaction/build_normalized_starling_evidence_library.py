"""Build the layered v6 Skin_Reaction normalized-Starling evidence library.

The staged builder is shared (``common/starling/build_normalized_evidence_library``);
this entry point only binds the Skin_Reaction policy.
"""

from __future__ import annotations

from copy import copy

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
from tools.chembl_tool.tasks.skin_reaction.build_starling_downstream_artifacts import (
    AUDIT_STAGE,
    HELDOUT_STAGE,
    MOLECULE_EVIDENCE_STAGE,
    NEIGHBOR_INDEX_STAGE,
    PAIR_BUCKET_STAGE,
    TRANSFER_POLICY_STAGE,
    build_downstream_artifacts,
)
from tools.chembl_tool.tasks.skin_reaction.starling_policy import (
    DEFAULT_OUT_DIR,
    DEFAULT_STARLING_DATA_DIR,
    POLICY,
)


ARTIFACT_STAGES = (
    "01_cleaned",
    "02_normalized",
    "03_records",
    PAIR_BUCKET_STAGE,
    TRANSFER_POLICY_STAGE,
    HELDOUT_STAGE,
    MOLECULE_EVIDENCE_STAGE,
    NEIGHBOR_INDEX_STAGE,
    AUDIT_STAGE,
)
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
    args = parse_args(POLICY, argv)
    if args.allow_missing_auxiliary_mapping:
        # Pair buckets and every later stage require the reconciled v2 mapping.
        # The explicit pending mode is only a way to inspect source cleaning,
        # normalization, and categorical encoding before that paid pass exists.
        args.through_stage = "organize"
        return run_with_args(POLICY, args)
    if args.through_stage != "index":
        return run_with_args(POLICY, args)

    if args.from_stage != "index":
        record_args = copy(args)
        record_args.through_stage = "organize"
        result = run_with_args(POLICY, record_args)
        if result:
            return result

    build_downstream_artifacts(
        normalized_root=args.out_dir,
        benchmark_split_root=args.benchmark_split_root,
        workers=args.workers,
        progress_every=args.progress_every,
        max_record_examples=args.max_record_examples,
        rebuild_request={
            "from_stage": args.from_stage,
            "through_stage": args.through_stage,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
