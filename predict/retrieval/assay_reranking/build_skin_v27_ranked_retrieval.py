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
TRAINING_PROFILES = {
    benchmark: f"flat_v5/{benchmark}_v1/skin_reaction/l2plus/assay_transfer/v27/training_candidate_copy_shared_parent100_v2"
    for benchmark in PROFILES
}
EVIDENCE_MANIFEST = builder.REPO_ROOT / (
    "data/evidence_libraries/skin_reaction/v10_main_universe_v5/"
    "retrieval_projection/ranked_evidence_v1/VERSION.json"
)
EVIDENCE_MANIFEST_SHA256 = "c4391ed6d429c59d870e6061ac9b84d6325d897f2685d6c49c9aada5438f739a"
TRAINING_EVIDENCE_MANIFEST = EVIDENCE_MANIFEST.parents[1] / (
    "ranked_evidence_v27_training_v1/VERSION.json"
)


def _configure(benchmark: str, *, training_template: bool = False) -> str:
    evidence = json.loads(EVIDENCE_MANIFEST.read_text(encoding="utf-8"))
    if sha256_file(EVIDENCE_MANIFEST) != EVIDENCE_MANIFEST_SHA256 or any(
        evidence["inputs"].get(name) != digest
        for name, digest in v27_skin.MODEL["source_snapshot"].items()
    ):
        raise ValueError("Skin V27 cache inputs no longer match the trained V10 snapshot")
    profile = (TRAINING_PROFILES if training_template else PROFILES)[benchmark]
    builder.PROFILE = profile
    builder.SCORING_FAMILY = "skin_training" if training_template else "default"
    builder.BASE_RANKING_ROOT = cache_profile_root(PROFILES[benchmark])
    builder.SHARED_LATER_PARENT_UNIVERSE = training_template
    builder.QUERY_BENCHMARK = benchmark
    builder.LABEL_RELEASE = (
        "v1" if benchmark == "gold" else {"benchmark": "tdc", "version": "v1"}
    )
    builder.TASK_LEVELS = LEVELS
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("build-projection", "prepare-universe", "prepare-level", "score", "finalize", "write-index", "validate")
    )
    parser.add_argument("--benchmark", choices=tuple(PROFILES), required=True)
    parser.add_argument("--subset", choices=("valid", "test"))
    parser.add_argument("--level", choices=LEVELS[TASK])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--training-template", action="store_true")
    parser.add_argument("--projection-output", type=Path)
    parser.add_argument("--score-reuse-root", action="append", type=Path, default=[])
    args = parser.parse_args()
    profile = _configure(args.benchmark, training_template=args.training_template)
    builder.SCORE_REUSE_ROOTS = tuple(args.score_reuse_root)
    output = args.output_root or cache_profile_root(profile)
    evidence_manifest = TRAINING_EVIDENCE_MANIFEST if args.training_template else EVIDENCE_MANIFEST
    if args.command == "build-projection":
        if not args.training_template or args.projection_output is None:
            parser.error("build-projection requires --training-template and --projection-output")
        from .build_safety_v27_ranked_retrieval import build_projection

        result = build_projection(
            TASK, args.projection_output, base_path=EVIDENCE_MANIFEST,
            source=EVIDENCE_MANIFEST.parents[2] / "03_pair_buckets/records.parquet",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.training_template:
        evidence = json.loads(evidence_manifest.read_text(encoding="utf-8"))
        if evidence["inputs"].get("base_projection_manifest_sha256") != EVIDENCE_MANIFEST_SHA256:
            raise ValueError("Skin training projection is not based on the pinned evidence release")
    if args.command in {"prepare-level", "score", "finalize"} and (
        not args.subset or not args.level
    ):
        parser.error("--subset and --level are required")
    if args.command == "prepare-universe":
        if not args.subset:
            parser.error("--subset is required")
        result = builder.prepare_shared_parent_universe(TASK, args.subset, output, evidence_manifest)
    elif args.command == "prepare-level":
        result = builder.prepare_level(TASK, args.subset, args.level, output, evidence_manifest)
    elif args.command == "score":
        result = builder.score_level(
            TASK, args.subset, args.level, output, args.device,
            args.num_shards, args.shard_index, args.batch_size,
        )
    elif args.command == "finalize":
        result = builder.finalize_level(TASK, args.subset, args.level, output)
    elif args.command == "write-index":
        result = builder.write_index(TASK, output, evidence_manifest)
    else:
        result = builder.validate_release(TASK, output, evidence_manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
