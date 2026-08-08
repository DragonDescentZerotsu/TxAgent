"""Package, restore, and verify Skin_Reaction normalized-v7 stage bundles."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.chembl_tool.common.starling.stage_artifact_store import (
    DEFAULT_PART_SIZE,
    StageArtifactStoreProfile,
    package_stages,
    restore_stages,
    verify_local,
    verify_tracked,
)


ARTIFACT_STAGES = (
    "01_cleaned",
    "02_canonicalized",
    "03_records",
    "04_pair_buckets",
    "05_distance_calibration",
    "06_remove_heldout_overlap",
    "07_molecule_evidence",
    "08_neighbor_index",
    "09_audits",
)
PROFILE = StageArtifactStoreProfile(
    store_version="skin_reaction.normalized_v7_store.v1",
    task_id="skin_reaction",
    stages=ARTIFACT_STAGES,
    local_root=Path(
        "outputs/chembl_tool/tasks/skin_reaction/evidence_library/"
        "starling_normalized_v7"
    ),
    tracked_root=Path(
        "artifacts/chembl_tool/tasks/skin_reaction/starling_normalized_v7"
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("package", "restore", "verify-tracked", "verify-local")
    )
    parser.add_argument("--local-root", default=str(PROFILE.local_root))
    parser.add_argument("--tracked-root", default=str(PROFILE.tracked_root))
    parser.add_argument(
        "--stages", nargs="+", choices=PROFILE.stages, default=list(PROFILE.stages)
    )
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
