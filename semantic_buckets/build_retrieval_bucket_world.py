"""Publish the semantic buckets reachable from valid/test Morgan-top-100 parents."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any, Mapping

import pandas as pd

from semantic_buckets.publication import sha256_file
from tools.chembl_tool.common.json_utils import write_json_atomic


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent
CACHE_ROOT = REPOSITORY_ROOT / "data/caches/assay_reranking/active/ranked_level_retrieval_v4"
RELEASE_ROOT = ROOT / "releases"
OUTPUT_ROOT = ROOT / "provenance/retrieval_worlds"
WORLD_ID = "morgan_top100_valid_test_l2plus_semantic_world_v2"
SKIN_WORLD_ID = "skin_morgan_top100_valid_test_l2_l3_v1"
SKIN_SEMANTIC_ROOT = (
    ROOT / "provenance/source_local_semantic_v4/"
    "skin_main_universe_v5_semantic_parent_v1_20260919/semantic_run"
)
FROZEN_SEMANTIC_ROOTS = {
    "ames": (
        ROOT / "provenance/source_local_semantic_v5/"
        "ames_dili_carcinogens_main_universe_v3_frozen_queued_boundary_20260922/"
        "ames/semantic_run"
    ),
}
SELECTED_BUCKET_ONLY_TASKS = {"dili", "carcinogens"}
TASK_LEVELS = {
    "bbb_martins": ("L2", "L3", "L4", "L5"),
    "bioavailability_ma": ("L2", "L3", "L4", "L5", "L6"),
    "skin_reaction": ("L2", "L3"),
    "ames": ("L2", "L3", "L4", "L5"),
    "dili": ("L2", "L3", "L4", "L5", "L6", "L7"),
    "carcinogens": ("L2", "L3", "L4", "L5", "L6", "L7"),
}
DEFAULT_TASK_LEVELS = {
    task: TASK_LEVELS[task] for task in ("bbb_martins", "bioavailability_ma")
}
EXPECTED = {
    "bbb_martins": {"records": 389_760, "buckets": 11_253},
    "bioavailability_ma": {"records": 374_645, "buckets": 6_454},
    "skin_reaction": {"records": 22_569, "buckets": 1_244},
    "ames": {"records": 556_755, "buckets": 11_391},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mapping(release_root: Path, task: str) -> tuple[
    dict[str, tuple[str, str]], set[tuple[str, str]], dict[str, Any]
]:
    if task == "skin_reaction":
        return _reviewed_skin_mapping()
    if task in FROZEN_SEMANTIC_ROOTS:
        return _frozen_semantic_mapping(task)
    if task in SELECTED_BUCKET_ONLY_TASKS:
        return _selected_bucket_only_mapping(release_root, task)
    release = (
        REPOSITORY_ROOT / "data/evidence_libraries" / task / "CURRENT"
    ).read_text(encoding="utf-8").strip()
    root = release_root / task / release
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    relative = manifest["selected"]["record_readout_bucket_map"]
    path = root / relative
    frame = pd.read_parquet(path, columns=["source_row_uid", "level", "semantic_bucket_id"])
    frame = frame.drop_duplicates()
    conflicts = frame.groupby("source_row_uid")[["level", "semantic_bucket_id"]].nunique()
    if conflicts.gt(1).any().any():
        raise ValueError(f"record has conflicting semantic assignments: {task}")
    mapping = {
        str(row.source_row_uid): (str(row.level), str(row.semantic_bucket_id))
        for row in frame.drop_duplicates("source_row_uid").itertuples(index=False)
    }
    buckets = set(zip(frame.level.astype(str), frame.semantic_bucket_id.astype(str)))
    return mapping, buckets, {
        "release_manifest": str(manifest_path),
        "release_manifest_sha256": sha256_file(manifest_path),
        "record_map": str(path),
        "record_map_sha256": sha256_file(path),
    }


def _selected_bucket_only_mapping(release_root: Path, task: str) -> tuple[
    dict[str, tuple[str, str]], set[tuple[str, str]], dict[str, Any]
]:
    release = (REPOSITORY_ROOT / "data/evidence_libraries" / task / "CURRENT").read_text().strip()
    root = release_root / task / release
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete_approved_unweighted" or manifest.get("task") != task:
        raise ValueError(f"{task} has no selected bucket-only release")
    record_path = root / manifest["selected"]["record_semantic_bucket_map"]
    semantic_path = root / manifest["selected"]["semantic_map"]
    for name, path in (("record_semantic_bucket_map", record_path), ("semantic_map", semantic_path)):
        if sha256_file(path) != manifest["files"][name]["sha256"]:
            raise ValueError(f"selected {task} {name} changed")
    frame = pd.read_parquet(record_path, columns=["source_row_uid", "level", "semantic_bucket_id"])
    frame = frame.drop_duplicates()
    conflicts = frame.groupby("source_row_uid")[["level", "semantic_bucket_id"]].nunique()
    if conflicts.gt(1).any().any():
        raise ValueError(f"record has conflicting semantic assignments: {task}")
    mapping = {str(row.source_row_uid): (str(row.level), str(row.semantic_bucket_id))
               for row in frame.drop_duplicates("source_row_uid").itertuples(index=False)}
    semantic = pd.read_parquet(semantic_path, columns=["level", "semantic_bucket_id"])
    buckets = set(zip(semantic.level.astype(str), semantic.semantic_bucket_id.astype(str)))
    return mapping, buckets, {
        "release_manifest": str(manifest_path), "release_manifest_sha256": sha256_file(manifest_path),
        "record_map": str(record_path), "record_map_sha256": sha256_file(record_path),
        "semantic_map": str(semantic_path), "semantic_map_sha256": sha256_file(semantic_path),
    }


def _frozen_semantic_mapping(task: str) -> tuple[
    dict[str, tuple[str, str]], set[tuple[str, str]], dict[str, Any]
]:
    root = FROZEN_SEMANTIC_ROOTS[task]
    manifest_path = root / "semantic_bucket_map_manifest.json"
    manifest = _read_json(manifest_path)
    required = {"status": "frozen_queued_boundary", "task": task}
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError(f"{task} semantic generation is not the pinned frozen candidate")
    record_path = root / "record_semantic_bucket_map.parquet"
    semantic_path = root / "semantic_bucket_map.parquet"
    if (
        sha256_file(record_path) != manifest["record_semantic_bucket_map_sha256"]
        or sha256_file(semantic_path) != manifest["semantic_bucket_map_sha256"]
    ):
        raise ValueError(f"{task} frozen semantic generation hash changed")
    frame = pd.read_parquet(
        record_path, columns=["source_row_uid", "level", "semantic_bucket_id"]
    ).drop_duplicates()
    conflicts = frame.groupby("source_row_uid")[["level", "semantic_bucket_id"]].nunique()
    if conflicts.gt(1).any().any():
        raise ValueError(f"record has conflicting semantic assignments: {task}")
    mapping = {
        str(row.source_row_uid): (str(row.level), str(row.semantic_bucket_id))
        for row in frame.drop_duplicates("source_row_uid").itertuples(index=False)
    }
    semantic = pd.read_parquet(semantic_path, columns=["level", "semantic_bucket_id"])
    buckets = set(zip(semantic.level.astype(str), semantic.semantic_bucket_id.astype(str)))
    return mapping, buckets, {
        "semantic_review_status": "unreviewed_candidate",
        "semantic_manifest": str(manifest_path),
        "semantic_manifest_sha256": sha256_file(manifest_path),
        "record_map": str(record_path), "record_map_sha256": sha256_file(record_path),
        "semantic_map": str(semantic_path), "semantic_map_sha256": sha256_file(semantic_path),
    }


def _reviewed_skin_mapping() -> tuple[
    dict[str, tuple[str, str]], set[tuple[str, str]], dict[str, Any]
]:
    manifest_path = SKIN_SEMANTIC_ROOT / "semantic_bucket_map_manifest.json"
    manifest = _read_json(manifest_path)
    required = {
        "status": "complete_reviewed", "task": "skin_reaction",
        "evidence_release": "v10_main_universe_v5",
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("Skin semantic generation is not the reviewed V10 generation")
    record_path = SKIN_SEMANTIC_ROOT / "record_semantic_bucket_map.parquet"
    semantic_path = SKIN_SEMANTIC_ROOT / "semantic_bucket_map.parquet"
    if (
        sha256_file(record_path) != manifest["record_semantic_bucket_map_sha256"]
        or sha256_file(semantic_path) != manifest["semantic_bucket_map_sha256"]
    ):
        raise ValueError("reviewed Skin semantic generation hash changed")
    frame = pd.read_parquet(
        record_path, columns=["source_row_uid", "level", "semantic_bucket_id"]
    ).drop_duplicates()
    conflicts = frame.groupby("source_row_uid")[["level", "semantic_bucket_id"]].nunique()
    if conflicts.gt(1).any().any():
        raise ValueError("record has conflicting semantic assignments: skin_reaction")
    mapping = {
        str(row.source_row_uid): (str(row.level), str(row.semantic_bucket_id))
        for row in frame.drop_duplicates("source_row_uid").itertuples(index=False)
    }
    semantic = pd.read_parquet(semantic_path, columns=["level", "semantic_bucket_id"])
    buckets = set(zip(semantic.level.astype(str), semantic.semantic_bucket_id.astype(str)))
    return mapping, buckets, {
        "semantic_manifest": str(manifest_path),
        "semantic_manifest_sha256": sha256_file(manifest_path),
        "record_map": str(record_path), "record_map_sha256": sha256_file(record_path),
        "semantic_map": str(semantic_path), "semantic_map_sha256": sha256_file(semantic_path),
    }


def _split_uids(index_path: Path, index: Mapping[str, Any], split: str,
                level: str) -> tuple[set[str], dict[str, Any]]:
    entry = index["splits"][split]["levels"][level]
    manifest_path = (index_path.parent / entry["manifest"]).resolve()
    manifest = _read_json(manifest_path)
    if sha256_file(manifest_path) != entry["manifest_sha256"] or (
        manifest.get("content_id") != entry["content_id"]
    ):
        raise ValueError(f"ranking manifest drift: {index['task_id']}/{split}/{level}")
    database = manifest_path.with_name(manifest["database"])
    connection = sqlite3.connect(f"file:{database}?mode=ro&immutable=1", uri=True)
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        query_count = connection.execute("SELECT count(*) FROM queries").fetchone()[0]
        uids = {str(row[0]) for row in connection.execute("SELECT item_id FROM rankings")}
    finally:
        connection.close()
    if metadata.get("content_id") != manifest["content_id"] or query_count != manifest["query_count"]:
        raise ValueError(f"ranking database drift: {index['task_id']}/{split}/{level}")
    return uids, {
        "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
        "content_id": manifest["content_id"], "database": str(database),
        "database_sha256": manifest["database_sha256"], "query_count": query_count,
    }


def _bucket_rows(task: str, level: str, valid: set[str], test: set[str],
                 mapping: Mapping[str, tuple[str, str]]) -> list[dict[str, Any]]:
    missing = sorted((valid | test) - mapping.keys())
    if missing:
        raise ValueError(f"unmapped retrieval UIDs for {task}/{level}: {missing[:10]}")
    wrong = sorted(uid for uid in valid | test if mapping[uid][0] != level)
    if wrong:
        raise ValueError(f"retrieval UID level mismatch for {task}/{level}: {wrong[:10]}")
    valid_counts = Counter(mapping[uid][1] for uid in valid)
    test_counts = Counter(mapping[uid][1] for uid in test)
    union_counts = Counter(mapping[uid][1] for uid in valid | test)
    return [{
        "task": task, "level": level, "semantic_bucket_id": bucket,
        "record_count": union_counts[bucket], "valid_record_count": valid_counts[bucket],
        "test_record_count": test_counts[bucket],
    } for bucket in sorted(union_counts)]


def build(output: Path | None = None, *, cache_root: Path = CACHE_ROOT,
          release_root: Path = RELEASE_ROOT, expected: Mapping[str, Any] | None = EXPECTED,
          task_levels: Mapping[str, tuple[str, ...]] = DEFAULT_TASK_LEVELS) -> dict[str, Any]:
    if task_levels and cache_root == CACHE_ROOT and set(task_levels) <= SELECTED_BUCKET_ONLY_TASKS:
        cache_root = (REPOSITORY_ROOT / "data/caches/assay_reranking/active/"
                      "ranked_level_retrieval_gold_v1_addon_v2")
    destination = output or OUTPUT_ROOT / WORLD_ID
    if destination.exists():
        raise FileExistsError(destination)
    rows, inputs, all_buckets = [], {}, {}
    for task, levels in task_levels.items():
        index_path = cache_root / task / "RELEASE_INDEX.json"
        index = _read_json(index_path)
        required = (index.get("schema_version") == "ranked_uid_task_release_index.v1"
                    and index.get("status") == "complete" and index.get("task_id") == task
                    and index.get("later_candidate_universe") == "all_uids_under_morgan_top_100_parents")
        if not required:
            raise ValueError(f"incompatible ranked UID release: {index_path}")
        mapping, buckets, mapping_inputs = _mapping(release_root, task)
        all_buckets[task] = {row for row in buckets if row[0] in levels}
        task_inputs = {"release_index": str(index_path),
                       "release_index_sha256": sha256_file(index_path), **mapping_inputs, "levels": {}}
        for level in levels:
            valid, valid_input = _split_uids(index_path, index, "valid", level)
            test, test_input = _split_uids(index_path, index, "test", level)
            rows.extend(_bucket_rows(task, level, valid, test, mapping))
            task_inputs["levels"][level] = {"valid": valid_input, "test": test_input}
        inputs[task] = task_inputs
    scoped_expected = None if expected is None or any(task not in expected for task in task_levels) else {
        task: expected[task] for task in task_levels
    }
    return _write(destination, pd.DataFrame(rows), inputs, all_buckets, scoped_expected)


def _write(destination: Path, frame: pd.DataFrame, inputs: Mapping[str, Any],
           all_buckets: Mapping[str, set[tuple[str, str]]],
           expected: Mapping[str, Any] | None) -> dict[str, Any]:
    frame = frame.sort_values(["task", "level", "semantic_bucket_id"]).reset_index(drop=True)
    summary = frame.groupby(["task", "level"], as_index=False).agg(
        record_count=("record_count", "sum"), bucket_count=("semantic_bucket_id", "size")
    )
    totals = {task: {"records": int(group.record_count.sum()), "buckets": len(group)}
              for task, group in frame.groupby("task")}
    if expected is not None and totals != dict(expected):
        raise ValueError(f"retrieval world changed: {totals}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        frame.to_parquet(staging / "semantic_bucket_world.parquet", index=False)
        summary.to_csv(staging / "summary.tsv", sep="\t", index=False)
        covered = {
            task: set(zip(group.level.astype(str), group.semantic_bucket_id.astype(str)))
            for task, group in frame.groupby("task")
        }
        uncovered = pd.DataFrame([
            {"task": task, "level": level, "semantic_bucket_id": bucket}
            for task, buckets in all_buckets.items()
            for level, bucket in sorted(buckets - covered.get(task, set()))
        ], columns=["task", "level", "semantic_bucket_id"])
        uncovered.to_parquet(staging / "uncovered_semantic_buckets.parquet", index=False)
        uncovered.to_csv(staging / "uncovered_semantic_buckets.tsv", sep="\t", index=False)
        manifest = {
            "version": "semantic_bucket_retrieval_world.v1", "status": "complete",
            "world_id": destination.name, "created_at": _now(), "scope": "L2+ valid/test",
            "candidate_universe": "all_uids_under_morgan_top_100_parents",
            "counts": {"tasks": totals, "records": sum(x["records"] for x in totals.values()),
            "buckets": len(frame), "unmapped_uids": 0,
            "uncovered_buckets": len(uncovered)}, "inputs": inputs,
            "files": {name: sha256_file(staging / name) for name in
                      ("semantic_bucket_world.parquet", "summary.tsv",
                       "uncovered_semantic_buckets.parquet", "uncovered_semantic_buckets.tsv")},
        }
        write_json_atomic(staging / "manifest.json", manifest)
        staging.replace(destination)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASK_LEVELS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    args = parser.parse_args()
    levels = DEFAULT_TASK_LEVELS if args.task is None else {args.task: TASK_LEVELS[args.task]}
    default = WORLD_ID if args.task is None else (
        SKIN_WORLD_ID if args.task == "skin_reaction" else f"{args.task}_{WORLD_ID}"
    )
    if args.task in SELECTED_BUCKET_ONLY_TASKS:
        default = f"{args.task}_morgan_top100_valid_test_l2_l7_selected_v1_20260923"
    result = build(
        args.output or OUTPUT_ROOT / default,
        cache_root=args.cache_root,
        task_levels=levels,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
