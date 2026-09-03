"""Build the BBB v8 evidence library through canonical pair buckets."""

from __future__ import annotations

from copy import copy

from data.processing.evidence_library.shared.v2.build_runtime import (
    assert_unpublished_build_root,
    starling_build_session,
)
from data.processing.evidence_library.versions.v9.build_normalized_evidence_library import (
    parse_args,
    run_with_args,
)
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.build_starling_downstream_artifacts import (
    CORE_PAIR_BUCKET_STAGE,
    build_canonical_artifacts,
)
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.starling_policy import (
    DEFAULT_OUT_DIR,
    POLICY,
)

ARTIFACT_STAGES = (
    "00_source",
    "01_cleaned",
    "02_canonicalized",
    CORE_PAIR_BUCKET_STAGE,
)
__all__ = ["ARTIFACT_STAGES", "DEFAULT_OUT_DIR", "POLICY", "main"]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(POLICY, argv, "pair-buckets", True)
    assert_unpublished_build_root(args.out_dir)
    with starling_build_session(
        args.out_dir, complete=args.through_stage == "pair-buckets"
    ):
        if args.through_stage != "pair-buckets":
            return run_with_args(POLICY, args)
        if args.from_stage != "pair-buckets":
            record_args = copy(args)
            record_args.through_stage = "normalize"
            if result := run_with_args(POLICY, record_args):
                return result
        build_canonical_artifacts(
            normalized_root=args.out_dir,
            workers=args.workers,
            rebuild_request={
                "from_stage": args.from_stage,
                "through_stage": CORE_PAIR_BUCKET_STAGE,
            },
            validation_level=args.validation_level,
            cache_mode=args.cache_mode,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
