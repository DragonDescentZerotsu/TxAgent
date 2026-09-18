"""Package or verify portable v10 core-stage artifacts."""

from __future__ import annotations

import argparse

from data.processing.paths import ARTIFACTS_ROOT, evidence_library_root
from data.processing.evidence_library.stage_artifact_store import (
    DEFAULT_PART_SIZE,
    StageArtifactStoreProfile,
    package_stages,
    restore_stages,
    verify_local,
    verify_tracked,
)


TASKS = ("ames", "dili", "carcinogens", "skin_reaction")
STAGE1_STAGES = ("00_source", "01_cleaned")
ALL_STAGES = STAGE1_STAGES


def _profile(task: str) -> StageArtifactStoreProfile:
    return StageArtifactStoreProfile(
        store_version=f"{task}.normalized_v10_stage1_store.v1",
        task_id=task,
        stages=STAGE1_STAGES,
        local_root=evidence_library_root(task, "v10"),
        tracked_root=ARTIFACTS_ROOT / f"evidence_library_compressed/{task}/v10",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument(
        "action", choices=("package", "restore", "verify-tracked", "verify-local")
    )
    parser.add_argument("--local-root")
    parser.add_argument("--tracked-root")
    parser.add_argument("--stages", nargs="+", choices=ALL_STAGES)
    parser.add_argument("--part-size", type=int, default=DEFAULT_PART_SIZE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    profile = _profile(args.task)
    stages = args.stages or profile.stages
    options = {
        key: value
        for key, value in {
            "local_root": args.local_root,
            "tracked_root": args.tracked_root,
        }.items()
        if value is not None
    }
    if args.action == "package":
        package_stages(profile, stages=stages, part_size=args.part_size, **options)
    elif args.action == "restore":
        restore_stages(profile, stages=stages, force=args.force, **options)
    elif args.action == "verify-tracked":
        verify_tracked(profile, tracked_root=args.tracked_root, stages=stages)
    else:
        verify_local(profile, stages=stages, **options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
