"""Build V27 safety scores over frozen Gold or new TDC Morgan-100 UID universes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from predict.utils.json import sha256_file

from . import build_ranked_uid_retrieval as builder
from .runtime import cache_profile_root
from .v27_safety import TASKS


LEVELS = {
    "ames": ("L2", "L3", "L4", "L5"),
    "dili": ("L2", "L3", "L4", "L5", "L6", "L7"),
    "carcinogens": ("L2", "L3", "L4", "L5", "L6", "L7"),
}
GOLD_BASE = cache_profile_root("ranked_level_retrieval_gold_v1_addon_v2")
RELEASE = "v10_main_universe_v3"
PROJECTION = "ranked_evidence_safety_v27_v1"
NARROW_PARENT_TASKS = frozenset({("gold", "carcinogens"), ("tdc", "ames")})


def profile(
    task: str, benchmark: str, parent_capacity: int = 100, *,
    supplement: bool = False, snapshot: bool = False, hybrid: bool = False,
) -> str:
    if supplement:
        variant = "shared_parent41to50_score_only_v1"
    elif hybrid:
        return (
            f"flat_v5/{benchmark}_v1/{task}/l2plus/hybrid/v27/"
            "assay_complete_or_hidden_morgan_tail_shared_parent40_v1"
        )
    elif snapshot:
        variant = "shared_parent40_partial_snapshot_v1"
    else:
        variant = f"shared_parent{parent_capacity}_v1" if parent_capacity < 100 else "shared_parent100_v2"
    return f"flat_v5/{benchmark}_v1/{task}/l2plus/assay_transfer/v27/general_candidate_copy_{variant}"


def _configure(
    task: str, benchmark: str, parent_capacity: int = 100, *,
    supplement: bool = False, snapshot: bool = False, hybrid: bool = False,
) -> None:
    if hybrid and (supplement or snapshot or parent_capacity != 40
                   or (benchmark, task) not in NARROW_PARENT_TASKS):
        raise ValueError("Hybrid releases require Gold Carcinogens or TDC AMES parent-40 inputs")
    if snapshot and (supplement or parent_capacity != 40 or (benchmark, task) not in NARROW_PARENT_TASKS):
        raise ValueError("Partial snapshots require Gold Carcinogens or TDC AMES parent-40 inputs")
    if supplement and (parent_capacity != 100 or (benchmark, task) not in NARROW_PARENT_TASKS):
        raise ValueError("Ranks 41-50 supplement requires a narrow-parent task and default capacity")
    if not supplement and (
        parent_capacity not in {40, 50, 100}
        or (parent_capacity < 100 and (benchmark, task) not in NARROW_PARENT_TASKS)
    ):
        raise ValueError("Reduced-parent successors are limited to Gold Carcinogens and TDC AMES")
    builder.PROFILE = profile(
        task, benchmark, parent_capacity,
        supplement=supplement, snapshot=snapshot, hybrid=hybrid,
    )
    builder.CAPACITY = 10 if supplement else parent_capacity
    builder.SOURCE_PARENT_OFFSET = 40 if supplement else 0
    builder.EVIDENCE_RELEASE = RELEASE
    builder.QUERY_BENCHMARK = benchmark
    builder.LABEL_RELEASE = "v1" if benchmark == "gold" else {"benchmark": "tdc", "version": "v1"}
    builder.TASK_LEVELS = {task: LEVELS[task]}
    builder.SCORING_FAMILY = "safety_v27"
    builder.BASE_RANKING_ROOT = GOLD_BASE if benchmark == "gold" else None
    builder.SHARED_LATER_PARENT_UNIVERSE = True
    builder.SCORE_REUSE_ROOTS = ()
    builder.TRUST_PREDECESSOR_ROWS = False


def snapshot_partial_release(
    task: str, source_root: Path, output_root: Path, evidence_manifest: Path,
) -> dict:
    """Close a separate, hash-pinned release from scored rows without touching journals."""
    if output_root.exists() or source_root.resolve() == output_root.resolve():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_root.name}.", dir=output_root.parent) as temporary:
        stage = Path(temporary)
        source_stages = {}
        for subset in ("valid", "test"):
            source_split = source_root / task / "scaffold" / subset
            destination_split = stage / task / "scaffold" / subset
            destination_split.mkdir(parents=True)
            shutil.copy2(source_split / "PARENT_UNIVERSE.json", destination_split)
            for level in LEVELS[task]:
                source = source_split / level
                destination = destination_split / level
                manifest_path = source / "VERSION.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    manifest.get("task_id") != task or manifest.get("subset") != subset
                    or manifest.get("level") != level
                    or manifest.get("status") not in {"complete", "prepared", "ready_to_finalize"}
                    or manifest.get("model") != builder._model_spec(task, level)
                ):
                    raise ValueError(f"Incompatible score source: {manifest_path}")
                destination.mkdir()
                database = source / str(manifest["database"])
                source_hashes = {
                    "manifest_sha256": sha256_file(manifest_path),
                    "database_sha256": sha256_file(database),
                    "score_journals_sha256": {
                        journal.name: sha256_file(journal)
                        for journal in sorted((source / ".scores").glob("*.jsonl"))
                    },
                }
                source_stages[f"{subset}/{level}"] = source_hashes
                shutil.copy2(database, destination / database.name)
                if sha256_file(destination / database.name) != source_hashes["database_sha256"]:
                    raise ValueError(f"Score source changed while copying: {database}")
                if manifest["status"] == "complete":
                    if source_hashes["database_sha256"] != manifest["database_sha256"]:
                        raise ValueError(f"Completed source database changed: {database}")
                    shutil.copy2(manifest_path, destination / "VERSION.json")
                    continue
                if (source / ".scores").exists():
                    shutil.copytree(source / ".scores", destination / ".scores")
                    if any(
                        sha256_file(destination / ".scores" / name) != digest
                        for name, digest in source_hashes["score_journals_sha256"].items()
                    ):
                        raise ValueError(f"Score journals changed while copying: {source}")
                manifest["snapshot_source"] = {
                    "root": str(source_root.resolve()), **source_hashes,
                }
                (destination / "VERSION.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                builder.finalize_level(task, subset, level, stage, partial_snapshot=True)
        index = builder.write_index(task, stage, evidence_manifest, partial_snapshot=True)
        index["source_stages"] = source_stages
        (stage / task / "RELEASE_INDEX.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        builder.validate_release(task, stage, evidence_manifest, allow_partial_snapshot=True)
        stage.rename(output_root)
        output_root.chmod(0o2775)
    return index


HYBRID_POLICY = {
    "schema_version": "assay_complete_or_hidden_morgan_tail.v1",
    "scope": "query_level",
    "complete_order": "assay_transfer",
    "incomplete_order": "scored_assay_then_unscored_morgan",
    "incomplete_prompt_scores": "hidden",
}


def snapshot_hybrid_release(
    task: str, source_root: Path, output_root: Path, evidence_manifest: Path,
) -> dict:
    """Publish a complete-selection hybrid view of one validated partial snapshot."""
    if output_root.exists() or source_root.resolve() == output_root.resolve():
        raise FileExistsError(output_root)
    source_index_path = source_root / task / "RELEASE_INDEX.json"
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    if (
        source_index.get("status") != "complete"
        or source_index.get("task_id") != task
        or source_index.get("score_coverage") != "partial_snapshot"
    ):
        raise ValueError("Hybrid source must be a complete validated partial snapshot")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_root.name}.", dir=output_root.parent) as temporary:
        stage = Path(temporary)
        shutil.copytree(source_root / task, stage / task)
        index_path = stage / task / "RELEASE_INDEX.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index.update(
            profile=builder.PROFILE,
            selection_coverage="complete",
            hybrid_policy=HYBRID_POLICY,
            hybrid_source={
                "release_index": str(source_index_path.resolve()),
                "release_index_sha256": sha256_file(source_index_path),
            },
        )
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        builder.validate_release(
            task, stage, evidence_manifest,
            allow_partial_snapshot=True, allow_hybrid=True,
        )
        stage.rename(output_root)
        output_root.chmod(0o2775)
    return index


def _base_projection(task: str) -> Path:
    index = json.loads((GOLD_BASE / task / "RELEASE_INDEX.json").read_text(encoding="utf-8"))
    path = GOLD_BASE / task / index["evidence"]["manifest"]
    if sha256_file(path) != index["evidence"]["manifest_sha256"]:
        raise ValueError(f"Frozen Gold evidence manifest changed: {task}")
    return path.resolve()


def build_projection(
    task: str, output: Path, *, base_path: Path | None = None,
    source: Path | None = None,
) -> dict:
    """Add canonical endpoints without changing published UID or level membership."""
    if output.exists():
        raise FileExistsError(output)
    base_path = base_path or _base_projection(task)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    source = source or builder.REPO_ROOT / "data/evidence_libraries" / task / RELEASE / "03_pair_buckets/records.parquet"
    source_sha256 = sha256_file(source)
    if source_sha256 != base["inputs"]["records"]:
        raise ValueError(f"Safety source differs from frozen Gold projection: {task}")
    endpoints = {}
    for batch in pq.ParquetFile(source).iter_batches(
        batch_size=50_000, columns=["source_row_uid", "canonical_endpoint_name"]
    ):
        for row in batch.to_pylist():
            uid = str(row["source_row_uid"])
            if uid in endpoints:
                raise ValueError(f"Duplicate safety source UID: {uid}")
            endpoints[uid] = str(row["canonical_endpoint_name"] or "")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".safety-v27.", dir=output.parent) as temporary:
        root = Path(temporary)
        records_path = root / "records.parquet"
        old_records = base_path.with_name(str(base["records"]))
        if sha256_file(old_records) != base["records_sha256"]:
            raise ValueError(f"Frozen Gold projection bytes changed: {task}")
        source_file = pq.ParquetFile(old_records)
        with pq.ParquetWriter(records_path, source_file.schema_arrow, compression="zstd") as writer:
            count = 0
            for batch in source_file.iter_batches(batch_size=10_000):
                updated = []
                for row in batch.to_pylist():
                    uid = str(row["source_row_uid"])
                    payload = json.loads(row["payload"])
                    endpoint = endpoints.pop(uid, "")
                    if row["level"] != "L1" and not endpoint:
                        raise ValueError(f"Missing canonical endpoint for safety UID: {uid}")
                    payload["canonical_endpoint_name"] = endpoint
                    row["payload"] = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                    updated.append(row)
                writer.write_table(pa.Table.from_pylist(updated, schema=source_file.schema_arrow))
                count += len(updated)
        if count != base["record_count"]:
            raise ValueError(f"Safety projection lost records: {task}")
        identity = {key: value for key, value in base.items() if key not in {
            "content_id", "records", "records_sha256",
        }}
        identity["inputs"] = {
            **base["inputs"], "base_projection_manifest_sha256": sha256_file(base_path),
            "canonical_endpoint_source_sha256": source_sha256,
        }
        identity["content_id"] = builder._digest(identity)
        manifest = {
            **identity, "records": records_path.name,
            "records_sha256": sha256_file(records_path),
        }
        (root / "VERSION.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        root.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build-projection", "prepare-universe", "prepare-level", "tokenize", "score", "finalize", "write-index", "validate", "snapshot-partial", "validate-partial", "snapshot-hybrid", "validate-hybrid"))
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--benchmark", choices=("gold", "tdc"), required=True)
    parser.add_argument("--subset", choices=("valid", "test"))
    parser.add_argument("--level")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--tokenized-root", type=Path)
    parser.add_argument("--projection-output", type=Path)
    parser.add_argument("--evidence-manifest", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--tokenize-workers", type=int, default=1)
    parser.add_argument("--score-reuse-root", action="append", type=Path, default=[])
    parser.add_argument("--source-parent-universe", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--parent-capacity", type=int, choices=(40, 50, 100), default=100)
    parser.add_argument("--supplement-41-50", action="store_true")
    args = parser.parse_args()
    if args.command == "snapshot-partial" and (args.source_root is None or args.parent_capacity != 40):
        parser.error("snapshot-partial requires --source-root and --parent-capacity 40")
    if args.command == "validate-partial" and args.parent_capacity != 40:
        parser.error("validate-partial requires --parent-capacity 40")
    if args.command == "snapshot-hybrid" and (args.source_root is None or args.parent_capacity != 40):
        parser.error("snapshot-hybrid requires --source-root and --parent-capacity 40")
    if args.command == "validate-hybrid" and args.parent_capacity != 40:
        parser.error("validate-hybrid requires --parent-capacity 40")
    if args.supplement_41_50 and (args.output_root is None or args.command in {"build-projection", "write-index", "validate"}):
        parser.error("The score-only supplement requires --output-root and cannot be published as a cache")
    if args.source_parent_universe is not None and args.command != "prepare-universe":
        parser.error("--source-parent-universe is only valid for prepare-universe")
    if (args.parent_capacity < 100 or args.supplement_41_50) and args.command == "prepare-universe" and args.source_parent_universe is None:
        parser.error("Reduced-parent preparation requires a frozen --source-parent-universe")
    is_snapshot = args.command in {"snapshot-partial", "validate-partial"}
    is_hybrid = args.command in {"snapshot-hybrid", "validate-hybrid"}
    _configure(
        args.task, args.benchmark, args.parent_capacity,
        supplement=args.supplement_41_50, snapshot=is_snapshot, hybrid=is_hybrid,
    )
    builder.SCORE_REUSE_ROOTS = tuple(args.score_reuse_root)
    output = args.output_root or cache_profile_root(
        profile(args.task, args.benchmark, args.parent_capacity,
                supplement=args.supplement_41_50, snapshot=is_snapshot, hybrid=is_hybrid)
    )
    if args.command == "build-projection":
        if args.projection_output is None:
            parser.error("--projection-output is required")
        result = build_projection(args.task, args.projection_output)
    else:
        if args.evidence_manifest is None:
            parser.error("--evidence-manifest is required")
        if args.command in {"prepare-level", "tokenize", "score", "finalize"} and (
            args.subset is None or args.level not in LEVELS[args.task]
        ):
            parser.error("--subset and a task-supported --level are required")
        if args.command == "tokenize" and args.tokenized_root is None:
            parser.error("--tokenized-root is required for tokenize")
        if args.command == "prepare-universe" and args.subset is None:
            parser.error("--subset is required")
        function = {
            "prepare-universe": lambda: builder.prepare_shared_parent_universe(
                args.task, args.subset, output, args.evidence_manifest, args.source_parent_universe
            ),
            "prepare-level": lambda: builder.prepare_level(args.task, args.subset, args.level, output, args.evidence_manifest),
            "tokenize": lambda: builder.pretokenize_level(args.task, args.subset, args.level, output, args.tokenized_root, args.num_shards, args.batch_size, args.tokenize_workers),
            "score": lambda: builder.score_level(args.task, args.subset, args.level, output, args.device, args.num_shards, args.shard_index, args.batch_size, args.tokenized_root),
            "finalize": lambda: builder.finalize_level(args.task, args.subset, args.level, output),
            "write-index": lambda: builder.write_index(args.task, output, args.evidence_manifest),
            "validate": lambda: builder.validate_release(args.task, output, args.evidence_manifest),
            "validate-partial": lambda: builder.validate_release(
                args.task, output, args.evidence_manifest, allow_partial_snapshot=True
            ),
            "validate-hybrid": lambda: builder.validate_release(
                args.task, output, args.evidence_manifest,
                allow_partial_snapshot=True, allow_hybrid=True,
            ),
            "snapshot-partial": lambda: snapshot_partial_release(
                args.task, args.source_root, output, args.evidence_manifest
            ),
            "snapshot-hybrid": lambda: snapshot_hybrid_release(
                args.task, args.source_root, output, args.evidence_manifest
            ),
        }[args.command]
        result = function()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
