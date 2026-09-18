"""Run degree-25 rankings over the reviewed Oral V10 semantic V4 map."""

import argparse
import json
from pathlib import Path

from semantic_buckets import bioavailability_semantic_degree25 as degree
from semantic_buckets import bioavailability_semantic_readout_v4 as semantic_v4


def configure_degree() -> None:
    semantic_v4.configure_core()
    degree.VERSION = "bioavailability_semantic_degree25.v2"
    degree.SEMANTIC_ROOT = semantic_v4.DEFAULT_OUTPUT
    degree.FINAL_MAP = semantic_v4.DEFAULT_OUTPUT / "semantic_bucket_map.parquet"
    degree.FINAL_MAP_MANIFEST = semantic_v4.DEFAULT_OUTPUT / "semantic_bucket_map_manifest.json"
    degree.DEFAULT_OUTPUT = semantic_v4.DEFAULT_OUTPUT / "degree25_rankings"


def main() -> None:
    configure_degree()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "run", "build-and-run"))
    parser.add_argument("--output", type=Path, default=degree.DEFAULT_OUTPUT)
    parser.add_argument("--final-map", type=Path, default=degree.FINAL_MAP)
    parser.add_argument("--parallelism", type=int, default=64)
    args = parser.parse_args()
    if args.command in {"build", "build-and-run"}:
        result = degree.build(args.output, final_map=args.final_map)
    if args.command in {"run", "build-and-run"}:
        result = degree.run(args.output, parallelism=args.parallelism)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
