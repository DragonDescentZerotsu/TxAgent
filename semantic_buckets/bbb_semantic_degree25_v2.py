"""Run degree-25 rankings over the reviewed BBB V10 semantic V4 map."""

import argparse
import json
from pathlib import Path

from semantic_buckets import bbb_semantic_degree25 as degree_v1
from semantic_buckets import bbb_semantic_readout_v4 as semantic_v4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "run", "build-and-run"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--final-map", type=Path)
    parser.add_argument("--review-manifest", type=Path)
    parser.add_argument("--parallelism", type=int, default=64)
    args = parser.parse_args()
    if args.command in {"build", "build-and-run"} and args.review_manifest is None:
        parser.error("--review-manifest is required when building")
    semantic_v4.configure_workflow()
    degree = degree_v1.configure_degree(args.review_manifest)
    degree.VERSION = "bbb_semantic_degree25.v2"
    output = args.output or degree.DEFAULT_OUTPUT
    final_map = args.final_map or degree.FINAL_MAP
    if args.command in {"build", "build-and-run"}:
        result = degree.build(output, final_map=final_map)
    if args.command in {"run", "build-and-run"}:
        result = degree.run(output, parallelism=args.parallelism)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
