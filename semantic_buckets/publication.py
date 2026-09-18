"""Build and verify release manifests for semantic-bucket sidecars."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import socket
from typing import Any, Iterable

import pandas as pd

from tools.chembl_tool.common.json_utils import write_json_atomic

from .artifacts import REPOSITORY_ROOT, normalize_task


SCHEMA_VERSION = "semantic_buckets.release.v1"
SELECTIONS = {
    "bbb_martins": {
        "generation": "gold_v1_protected_incremental_v1",
        "eligibility_generation": "gold_v1_protected_incremental_v1_weighted_v2",
    },
    "bioavailability_ma": {
        "generation": "gold_v1_protected_incremental_v1",
        "eligibility_generation": "gold_v1_protected_incremental_v1_weighted_v2",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def tree_inventory(root: Path) -> dict[str, Any]:
    files = []
    tree_digest = hashlib.sha256()
    total_bytes = 0
    for path in _files(root):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        files.append({"path": relative, "bytes": size, "sha256": digest})
        total_bytes += size
        tree_digest.update(relative.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(str(size).encode("ascii"))
        tree_digest.update(b"\0")
        tree_digest.update(digest.encode("ascii"))
        tree_digest.update(b"\n")
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "tree_sha256": tree_digest.hexdigest(),
        "files": files,
    }


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def selected_paths(task: str, root: Path) -> dict[str, str]:
    choice = SELECTIONS[task]
    generation = root / "generations" / choice["generation"]
    readout = generation / "readout_buckets_v5_depth1"
    ranking = generation / "degree25_rankings"
    eligibility = (
        root
        / "eligibility"
        / choice["eligibility_generation"]
        / "record_eligibility.parquet"
    )
    return {
        "semantic_map": _relative(root, generation / "semantic_bucket_map.parquet"),
        "semantic_map_manifest": _relative(
            root, generation / "semantic_bucket_map_manifest.json"
        ),
        "readout_bucket_map": _relative(root, readout / "readout_bucket_map.parquet"),
        "record_readout_bucket_map": _relative(
            root, readout / "record_readout_bucket_map.parquet"
        ),
        "semantic_bucket_rankings": _relative(
            root, ranking / "semantic_bucket_rankings.parquet"
        ),
        "record_relevance_rankings": _relative(
            root, ranking / "record_relevance_rankings.parquet"
        ),
        "retrieval_eligibility": _relative(root, eligibility),
    }


def _coverage(root: Path, selected: dict[str, str]) -> dict[str, Any]:
    semantic = pd.read_parquet(root / selected["semantic_map"])
    readout = pd.read_parquet(root / selected["readout_bucket_map"])
    records = pd.read_parquet(root / selected["record_readout_bucket_map"])
    eligibility = pd.read_parquet(root / selected["retrieval_eligibility"])

    if semantic.duplicated(["level", "atom_id"]).any():
        raise ValueError("selected semantic map repeats a level/atom assignment")
    if readout.duplicated(["level", "atom_id"]).any():
        raise ValueError("selected readout map repeats a level/atom assignment")
    semantic_keys = semantic[["level", "atom_id", "semantic_bucket_id"]]
    readout_keys = readout[["level", "atom_id", "semantic_bucket_id"]]
    if set(map(tuple, semantic_keys.itertuples(index=False, name=None))) != set(
        map(tuple, readout_keys.itertuples(index=False, name=None))
    ):
        raise ValueError("selected readout map does not preserve semantic assignments")
    parents = readout.groupby("readout_bucket_id", sort=False).agg(
        levels=("level", "nunique"), semantic_parents=("semantic_bucket_id", "nunique")
    )
    if parents.ne(1).any().any():
        raise ValueError("a selected readout bucket crosses level or semantic parent")
    record_keys = ["canonical_record_id", "source_row_uid", "level", "semantic_bucket_id"]
    if records.duplicated(record_keys[:3]).any():
        raise ValueError("selected record/readout map repeats a record-level assignment")
    if eligibility.duplicated(record_keys).any():
        raise ValueError("selected eligibility map repeats a semantic record assignment")

    def level_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
        values = frame.groupby("level", sort=True)[column].nunique()
        return {str(level): int(count) for level, count in values.items()}

    return {
        "semantic_atoms": len(semantic),
        "semantic_buckets_by_level": level_counts(semantic, "semantic_bucket_id"),
        "readout_atoms": len(readout),
        "readout_buckets_by_level": level_counts(readout, "readout_bucket_id"),
        "record_assignments": len(records),
        "record_assignments_by_level": dict(
            sorted(Counter(map(str, records["level"])).items())
        ),
        "eligibility_records": len(eligibility),
        "eligible_records": int(eligibility["retrieval_eligible"].sum()),
    }


def build_manifest(task: str, root: Path, source_root: Path) -> dict[str, Any]:
    task = normalize_task(task)
    root = root.resolve()
    source_root = source_root.resolve()
    selected = selected_paths(task, root)
    for relative in selected.values():
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)
    generations = {
        path.name: tree_inventory(path)
        for path in sorted((root / "generations").iterdir())
        if path.is_dir()
    }
    eligibility = {
        path.name: tree_inventory(path)
        for path in sorted((root / "eligibility").iterdir())
        if path.is_dir()
    }
    pair_records = (
        REPOSITORY_ROOT
        / "data/evidence_libraries"
        / task
        / "v10/03_pair_buckets/records.parquet"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_reviewed",
        "task": task,
        "evidence_library_version": "v10",
        "selected": selected,
        "coverage": _coverage(root, selected),
        "upstream": {
            "pair_bucket_records": str(pair_records.resolve()),
            "pair_bucket_records_sha256": sha256_file(pair_records),
        },
        "generations": generations,
        "eligibility_generations": eligibility,
        "migration": {
            "hostname": socket.gethostname(),
            "source_root": str(source_root),
            "staging_root": str(root),
            "source_directories": sorted(generations),
            "eligibility_source_directories": sorted(eligibility),
        },
    }


def verify_manifest(root: Path) -> None:
    root = root.resolve()
    document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected semantic-bucket manifest schema")
    selected = document["selected"]
    _coverage(root, selected)
    for namespace, key in (
        ("generations", "generations"),
        ("eligibility", "eligibility_generations"),
    ):
        expected = document[key]
        observed = {
            path.name: tree_inventory(path)
            for path in sorted((root / namespace).iterdir())
            if path.is_dir()
        }
        if observed != expected:
            raise ValueError(f"published {namespace} inventory differs from manifest")


def refresh_upstream(task: str, root: Path) -> dict[str, Any]:
    """Refresh only the Stage-3 binding of an unchanged reviewed release."""
    task = normalize_task(task)
    root = root.resolve()
    verify_manifest(root)
    manifest_path = root / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("task") != task:
        raise ValueError(f"semantic release task mismatch: {manifest_path}")
    records = (
        REPOSITORY_ROOT
        / "data/evidence_libraries"
        / task
        / "v10/03_pair_buckets/records.parquet"
    )
    document["upstream"] = {
        "pair_bucket_records": str(records.resolve()),
        "pair_bucket_records_sha256": sha256_file(records),
    }
    write_json_atomic(manifest_path, document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--task", required=True, choices=tuple(SELECTIONS))
    build.add_argument("--root", required=True, type=Path)
    build.add_argument("--source-root", required=True, type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", required=True, type=Path)
    refresh = subparsers.add_parser("refresh-upstream")
    refresh.add_argument("--task", required=True, choices=tuple(SELECTIONS))
    refresh.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "build":
        manifest = build_manifest(args.task, args.root, args.source_root)
        write_json_atomic(args.root / "manifest.json", manifest)
    elif args.command == "verify":
        verify_manifest(args.root)
    else:
        refresh_upstream(args.task, args.root)


if __name__ == "__main__":
    main()
