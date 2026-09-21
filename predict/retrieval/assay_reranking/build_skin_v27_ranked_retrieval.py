"""Build Skin V27 L2/L3 ranked-UID caches for Gold-v1 or TDC-v1 queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from predict.utils.json import sha256_file

from . import build_ranked_uid_retrieval as builder
from . import v27_skin
from .runtime import cache_profile_root


TASK = "skin_reaction"
LEVELS = {TASK: ("L2", "L3")}
PROFILES = {
    "gold": "ranked_level_retrieval_skin_v27_gold_v1",
    "tdc": "ranked_level_retrieval_skin_v27_tdc_v1",
}
EVIDENCE_MANIFEST = builder.REPO_ROOT / (
    "data/evidence_libraries/skin_reaction/v10_main_universe_v5/"
    "retrieval_projection/ranked_evidence_v1/VERSION.json"
)
EVIDENCE_MANIFEST_SHA256 = "c4391ed6d429c59d870e6061ac9b84d6325d897f2685d6c49c9aada5438f739a"


def _configure(benchmark: str) -> str:
    evidence = json.loads(EVIDENCE_MANIFEST.read_text(encoding="utf-8"))
    if sha256_file(EVIDENCE_MANIFEST) != EVIDENCE_MANIFEST_SHA256 or any(
        evidence["inputs"].get(name) != digest
        for name, digest in v27_skin.MODEL["source_snapshot"].items()
    ):
        raise ValueError("Skin V27 cache inputs no longer match the trained V10 snapshot")
    profile = PROFILES[benchmark]
    builder.PROFILE = profile
    builder.QUERY_BENCHMARK = benchmark
    builder.LABEL_RELEASE = (
        "v1" if benchmark == "gold" else {"benchmark": "tdc", "version": "v1"}
    )
    builder.TASK_LEVELS = LEVELS
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare-level", "score", "finalize", "write-index", "validate")
    )
    parser.add_argument("--benchmark", choices=tuple(PROFILES), required=True)
    parser.add_argument("--subset", choices=("valid", "test"))
    parser.add_argument("--level", choices=LEVELS[TASK])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    profile = _configure(args.benchmark)
    output = args.output_root or cache_profile_root(profile)
    if args.command in {"prepare-level", "score", "finalize"} and (
        not args.subset or not args.level
    ):
        parser.error("--subset and --level are required")
    if args.command == "prepare-level":
        result = builder.prepare_level(TASK, args.subset, args.level, output, EVIDENCE_MANIFEST)
    elif args.command == "score":
        result = builder.score_level(
            TASK, args.subset, args.level, output, args.device,
            args.num_shards, args.shard_index, args.batch_size,
        )
    elif args.command == "finalize":
        result = builder.finalize_level(TASK, args.subset, args.level, output)
    elif args.command == "write-index":
        result = builder.write_index(TASK, output, EVIDENCE_MANIFEST)
    else:
        result = builder.validate_release(TASK, output, EVIDENCE_MANIFEST)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
