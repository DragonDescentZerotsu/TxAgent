"""Build L2+ ranked-UID caches for TDC queries over the active evidence library.

This is a thin CLI around ``build_ranked_uid_retrieval``. It changes only the
query cohort and output profile; candidate construction, prompt rendering,
model pins, score reuse, finalization, and validation remain owned there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import build_ranked_uid_retrieval as builder
from .runtime import cache_profile_root


PROFILE = "ranked_level_retrieval_tdc_v1_indirect_v2"
TASKS = ("bbb_martins", "bioavailability_ma")
LEVELS = {
    "bbb_martins": ("L2", "L3", "L4", "L5"),
    "bioavailability_ma": ("L2", "L3", "L4", "L5", "L6"),
}


def _configure(
    profile: str = PROFILE,
    evidence_release: str | None = None,
    score_reuse_roots: tuple[Path, ...] = (),
) -> None:
    builder.PROFILE = profile
    builder.QUERY_BENCHMARK = "tdc"
    builder.LABEL_RELEASE = {"benchmark": "tdc", "version": "v1"}
    builder.TASK_LEVELS = LEVELS
    builder.EVIDENCE_RELEASE = evidence_release
    builder.SCORE_REUSE_ROOTS = score_reuse_roots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare-level", "score", "finalize", "write-index", "validate")
    )
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--subset", choices=("valid", "test"))
    parser.add_argument("--level")
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--evidence-release")
    parser.add_argument("--score-reuse-root", action="append", type=Path, default=[])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    score_reuse_roots = tuple(path.resolve() for path in args.score_reuse_root)
    _configure(args.profile, args.evidence_release, score_reuse_roots)
    args.output_root = args.output_root or cache_profile_root(args.profile)
    if args.command in {"prepare-level", "score", "finalize"}:
        if not args.subset or args.level not in LEVELS[args.task]:
            parser.error("--subset and a task-supported --level are required")
    if args.command == "prepare-level":
        result = builder.prepare_level(
            args.task, args.subset, args.level, args.output_root, args.evidence_manifest
        )
    elif args.command == "score":
        result = builder.score_level(
            args.task, args.subset, args.level, args.output_root, args.device,
            args.num_shards, args.shard_index, args.batch_size,
        )
    elif args.command == "finalize":
        result = builder.finalize_level(args.task, args.subset, args.level, args.output_root)
    elif args.command == "write-index":
        result = builder.write_index(args.task, args.output_root, args.evidence_manifest)
    else:
        result = builder.validate_release(args.task, args.output_root, args.evidence_manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
