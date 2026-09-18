"""Build and run degree-25 rankings for the reviewed BBB V10 semantic map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_buckets import bbb_semantic_readout_v1 as bbb


def configure_degree(review_manifest: Path | None = None):
    bbb.configure_core()
    from semantic_buckets import bioavailability_semantic_degree25 as degree

    degree.VERSION = "bbb_semantic_degree25.v1"
    degree.TASK_ID = "bbb_martins"
    degree.RECORD_COUNT = sum(bbb.EXPECTED_LEVEL_COUNTS[level] for level in bbb.LEVELS)
    degree.SEMANTIC_ROOT = bbb.DEFAULT_OUTPUT
    degree.FINAL_MAP = bbb.DEFAULT_OUTPUT / "semantic_bucket_map.parquet"
    degree.FINAL_MAP_MANIFEST = bbb.DEFAULT_OUTPUT / "semantic_bucket_map_manifest.json"
    degree.DEFAULT_OUTPUT = bbb.DEFAULT_OUTPUT / "degree25_rankings"
    degree.TEMPLATE = bbb.RANKING_TEMPLATE
    if review_manifest is not None:
        degree.RANKING_PROMPT_APPROVAL_MANIFEST = review_manifest
        degree.RANKING_PROMPT_APPROVAL_FIELD = "ranking_prompt_sha256"
    return degree


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "run", "build-and-run"))
    parser.add_argument("--output", type=Path, default=bbb.DEFAULT_OUTPUT / "degree25_rankings")
    parser.add_argument("--final-map", type=Path, default=bbb.DEFAULT_OUTPUT / "semantic_bucket_map.parquet")
    parser.add_argument("--review-manifest", type=Path)
    parser.add_argument("--parallelism", type=int, default=128)
    args = parser.parse_args()
    if args.command in {"build", "build-and-run"} and args.review_manifest is None:
        parser.error("--review-manifest is required when building")
    degree = configure_degree(args.review_manifest)
    if args.command in {"build", "build-and-run"}:
        result = degree.build(args.output, final_map=args.final_map)
    if args.command in {"run", "build-and-run"}:
        result = degree.run(args.output, parallelism=args.parallelism)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
