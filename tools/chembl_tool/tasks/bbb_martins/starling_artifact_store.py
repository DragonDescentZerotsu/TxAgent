"""Package, restore, and verify BBB Martins normalized-v6 stage bundles."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.processing.evidence_library.stage_artifact_store import (
    DEFAULT_PART_SIZE,
    StageArtifactStoreProfile,
    package_stages,
    restore_stages,
    verify_local,
    verify_tracked,
)
# This store targets the historical v6 layout.  Keep its inventory local and
# frozen so the v7 builder's stage names cannot change restore semantics.
ARTIFACT_STAGES = (
    "01_cleaned",
    "02_normalized",
    "03_records",
    "04_pair_buckets",
    "05_assay_transfer_policy",
    "06_remove_heldout_overlap",
    "07_molecule_evidence",
    "08_neighbor_index",
    "09_audits",
)
PROFILE = StageArtifactStoreProfile(
    store_version="bbb_martins.normalized_v6_store.v1",
    task_id="bbb_martins",
    stages=ARTIFACT_STAGES,
    local_root=Path(
        "outputs/chembl_tool/tasks/bbb_martins/evidence_library/"
        "starling_normalized_v6"
    ),
    tracked_root=Path("artifacts/chembl_tool/tasks/bbb_martins/starling_normalized_v6"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("package", "restore", "verify-tracked", "verify-local"))
    parser.add_argument("--local-root", default=str(PROFILE.local_root))
    parser.add_argument("--tracked-root", default=str(PROFILE.tracked_root))
    parser.add_argument("--stages", nargs="+", choices=PROFILE.stages, default=list(PROFILE.stages))
    parser.add_argument("--part-size", type=int, default=DEFAULT_PART_SIZE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    options = {
        "local_root": args.local_root,
        "tracked_root": args.tracked_root,
        "stages": args.stages,
    }
    if args.action == "package":
        package_stages(PROFILE, part_size=args.part_size, **options)
    elif args.action == "restore":
        restore_stages(PROFILE, force=args.force, **options)
    elif args.action == "verify-tracked":
        verify_tracked(PROFILE, tracked_root=args.tracked_root, stages=args.stages)
    else:
        verify_local(PROFILE, **options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
