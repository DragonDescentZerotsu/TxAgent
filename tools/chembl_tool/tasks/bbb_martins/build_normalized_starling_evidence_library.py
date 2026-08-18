"""Build the split-aware canonical Starling v7 BBB Martins library.

Stages 01-03 use the shared normalization engine. Stages 04-09 are produced by
the shared split-aware downstream engine, with BBB-specific policies supplied
by this task package.
"""

from __future__ import annotations

from copy import copy

from tools.chembl_tool.common.starling.build_normalized_evidence_library import (
    build,
    parse_args,
    run_with_args,
)
from tools.chembl_tool.tasks.bbb_martins.build_starling_downstream_artifacts import (
    AUDIT_STAGE,
    DISTANCE_CALIBRATION_STAGE,
    HELDOUT_STAGE,
    MOLECULE_EVIDENCE_STAGE,
    NEIGHBOR_INDEX_STAGE,
    PAIR_BUCKET_STAGE,
    build_canonical_artifacts,
    build_downstream_artifacts,
)
from tools.chembl_tool.tasks.bbb_martins.starling_policy import (
    DEFAULT_OUT_DIR,
    POLICY,
)


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
