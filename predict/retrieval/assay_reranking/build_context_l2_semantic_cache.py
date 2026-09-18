"""Publish Morgan and semantic-lap L2 selections beside the frozen context L1 cache."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

import pandas as pd

from semantic_buckets.artifacts import (
    resolve_semantic_bucket_artifacts,
)
from predict.retrieval.policies import normalize_molecule_identity, seeded_rank_tie_key
from predict.utils.json import sha256_file

from . import three_pools
from .build_cache_matched_v2 import _similarity
from .runtime import cache_profile_root
from .semantic_bucket_selection import select_semantic_lap


SCHEMA_VERSION = "l1_context_semantic_l2.v1"
OUTPUT_ROOT = cache_profile_root("l1_context_semantic_l2_v1")
L1_ROOT = cache_profile_root("l1_context_morgan100_v1")
SOURCE_ROOT = three_pools.ROOT
L2_LIMIT = 12
PER_BUCKET = 3
SEMANTIC_TASKS = ("bbb_martins", "bioavailability_ma")


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _semantic_index(task: str) -> tuple[dict[str, tuple[str, int]], dict[str, object]]:
    artifacts = resolve_semantic_bucket_artifacts(task)
    record_path = artifacts.record_relevance_rankings
    map_manifest = artifacts.semantic_map_manifest
    release_manifest = artifacts.manifest
    manifest = json.loads(release_manifest.read_text())
    if manifest.get("status") != "complete_reviewed":
        raise ValueError(f"reviewed semantic rankings required: {task}")
    mapping = pd.read_parquet(
        record_path,
        columns=["source_row_uid", "level", "semantic_bucket_id", "level_rank"],
    )
    mapping = mapping[mapping.level.eq("L2")]
    if mapping.source_row_uid.duplicated().any():
        raise ValueError(f"duplicate semantic identity: {task}/L2")
    index = {
        str(row.source_row_uid): (str(row.semantic_bucket_id), int(row.level_rank))
        for row in mapping.itertuples(index=False)
    }
    inputs = {
        str(path.resolve()): sha256_file(path)
        for path in (record_path, map_manifest, release_manifest)
    }
    return index, {"inputs": inputs, "mapped_l2_records": len(index)}


def _l1_visible_record_ids(connection: sqlite3.Connection, query_id: int) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            """SELECT r.external_record_id
               FROM candidates AS c
               JOIN gold_contexts AS g USING(context_key)
               JOIN context_records AS cr USING(context_key)
               JOIN records AS r USING(record_key)
               WHERE c.query_id=? AND c.morgan_parent_rank<=10
               ORDER BY c.morgan_parent_rank,g.external_context_id,
                        cr.within_context_rank,r.external_record_id""",
            (query_id,),
        )
    }


def build(task: str, subset: str = "valid", output_root: Path | None = None) -> dict[str, object]:
    if task not in SEMANTIC_TASKS or subset != "valid":
        raise ValueError("the context-L2 experiment supports BBB/oral validation only")
    destination = (output_root or OUTPUT_ROOT / task / "scaffold" / subset).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing cache: {destination}")
    semantic, semantic_audit = _semantic_index(task)
    l1_root = L1_ROOT / task / "scaffold" / subset
    source_root = SOURCE_ROOT / task / "scaffold" / subset
    l1_manifest_path, source_manifest_path = l1_root / "VERSION.json", source_root / "VERSION.json"
    l1_manifest, source_manifest = (
        json.loads(l1_manifest_path.read_text()), json.loads(source_manifest_path.read_text())
    )
    if l1_manifest.get("status") != "complete" or source_manifest.get("status") != "complete":
        raise ValueError("complete L1 and Morgan-100 source caches are required")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    database = temporary / "retrieval.sqlite3"
    shutil.copy2(l1_root / str(l1_manifest["database"]), database)
    database.chmod(0o664)
    output = sqlite3.connect(database)
    output.executescript(
        """
        CREATE TABLE l2_records(
          record_key INTEGER PRIMARY KEY,external_record_id TEXT UNIQUE NOT NULL,
          parent_id TEXT NOT NULL,parent_smiles TEXT NOT NULL,payload TEXT NOT NULL
        );
        CREATE TABLE l2_assignments(
          query_id INTEGER NOT NULL,mode TEXT NOT NULL,selection_rank INTEGER NOT NULL,
          record_key INTEGER NOT NULL,semantic_bucket_id TEXT,semantic_rank INTEGER,
          semantic_lap INTEGER,morgan_similarity REAL NOT NULL,
          PRIMARY KEY(query_id,mode,selection_rank),UNIQUE(query_id,mode,record_key)
        ) WITHOUT ROWID;
        CREATE TABLE l2_selection_counts(
          query_id INTEGER PRIMARY KEY,candidate_record_count INTEGER NOT NULL,
          candidate_parent_count INTEGER NOT NULL,candidate_bucket_count INTEGER NOT NULL,
          unmapped_record_count INTEGER NOT NULL
        );
        """
    )
    record_keys: dict[str, int] = {}
    summary: dict[str, int] = defaultdict(int)
    source_db = source_root / "scores.sqlite3"
    if sha256_file(source_db) != source_manifest["cache_sha256"]:
        raise ValueError(f"source cache hash changed: {source_db}")
    source = sqlite3.connect(f"file:{source_db.resolve()}?mode=ro&immutable=1", uri=True)
    source.row_factory = sqlite3.Row
    output.row_factory = sqlite3.Row
    source_queries = {
        str(row[0]): int(row[1])
        for row in source.execute("SELECT benchmark_row_id,query_id FROM benchmark_queries")
    }
    for query in output.execute(
        """SELECT b.benchmark_row_id,b.query_id,q.query_parent_smiles
           FROM benchmark_queries AS b JOIN queries AS q USING(query_id)
           ORDER BY b.query_id"""
    ).fetchall():
        query_id = int(query["query_id"])
        seen = _l1_visible_record_ids(output, query_id)
        candidates, unmapped = [], 0
        similarity_by_parent: dict[str, float] = {}
        smiles_by_parent: dict[str, str] = {}
        rows = source.execute(
            """SELECT r.external_record_id,r.parent_id,r.payload
               FROM assignments AS a JOIN records AS r USING(record_key)
               WHERE a.query_id=? AND a.pool='all' AND a.level='L2'""",
            (source_queries[str(query["benchmark_row_id"])],),
        )
        for row in rows:
            payload = json.loads(row["payload"])
            binding = semantic.get(str(payload.get("source_row_uid") or ""))
            if binding is None:
                unmapped += 1
                continue
            record_id = str(row["external_record_id"])
            if record_id in seen:
                continue
            parent_id = str(row["parent_id"])
            parent_smiles = smiles_by_parent.get(parent_id)
            if parent_smiles is None:
                identity = normalize_molecule_identity(str(payload["canonical_smiles"]))
                parent_smiles = str(identity.parent_smiles or "")
                if (identity.status != "ok" or not parent_smiles
                        or (identity.parent_inchi_key or parent_smiles) != parent_id):
                    raise ValueError(f"unresolved or changed L2 parent: {record_id}")
                smiles_by_parent[parent_id] = parent_smiles
                similarity_by_parent[parent_id] = _similarity(
                    str(query["query_parent_smiles"]), parent_smiles
                )
            bucket, semantic_rank = binding
            candidates.append(
                {
                    "record_id": record_id,
                    "parent_id": parent_id,
                    "parent_smiles": parent_smiles,
                    "payload": payload,
                    "semantic_bucket_id": bucket,
                    "semantic_rank": semantic_rank,
                    "morgan_similarity": similarity_by_parent[parent_id],
                    "tie_key": seeded_rank_tie_key(
                        0, task, str(query["benchmark_row_id"]), "L2", parent_id, record_id
                    ),
                }
            )
        global_morgan = sorted(
            candidates, key=lambda row: (-row["morgan_similarity"], row["tie_key"])
        )[:L2_LIMIT]
        semantic_lap = select_semantic_lap(candidates, limit=L2_LIMIT, per_bucket=PER_BUCKET)
        if len(global_morgan) != L2_LIMIT or len(semantic_lap) != L2_LIMIT:
            raise ValueError(f"{task}/{query['benchmark_row_id']} cannot supply 12 L2 records")
        for mode, selected in (("morgan", global_morgan), ("semantic_lap", semantic_lap)):
            for selection_rank, row in enumerate(selected, 1):
                record_id = row["record_id"]
                if record_id not in record_keys:
                    record_keys[record_id] = len(record_keys) + 1
                    output.execute(
                        "INSERT INTO l2_records VALUES (?,?,?,?,?)",
                        (record_keys[record_id], record_id, row["parent_id"], row["parent_smiles"],
                         json.dumps(row["payload"], sort_keys=True, separators=(",", ":"))),
                    )
                output.execute(
                    "INSERT INTO l2_assignments VALUES (?,?,?,?,?,?,?,?)",
                    (query_id, mode, selection_rank, record_keys[record_id],
                     row["semantic_bucket_id"] if mode == "semantic_lap" else None,
                     row["semantic_rank"] if mode == "semantic_lap" else None,
                     row.get("semantic_lap") if mode == "semantic_lap" else None,
                     row["morgan_similarity"]),
                )
                summary[f"{mode}_assignments"] += 1
        output.execute(
            "INSERT INTO l2_selection_counts VALUES (?,?,?,?,?)",
            (query_id, len(candidates), len({row["parent_id"] for row in candidates}),
             len({row["semantic_bucket_id"] for row in candidates}), unmapped),
        )
        summary["unmapped_assignments"] += unmapped
    source.close()
    summary["l2_unique_records"] = len(record_keys)

    identity = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task,
        "subset": subset,
        "l1_content_id": l1_manifest["content_id"],
        "source_cache_sha256": source_manifest["cache_sha256"],
        "semantic_inputs": semantic_audit["inputs"],
        "assignment_counts": dict(sorted(summary.items())),
    }
    content_id = _digest(identity)
    output.execute("UPDATE metadata SET value=? WHERE key='schema_version'", (SCHEMA_VERSION,))
    output.execute("UPDATE metadata SET value=? WHERE key='content_id'", (content_id,))
    output.commit()
    output.execute("VACUUM")
    output.close()
    manifest = {
        **l1_manifest,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "database": database.name,
        "content_id": content_id,
        "capacities": {**l1_manifest["capacities"], "l2_records": L2_LIMIT,
                       "semantic_records_per_bucket_per_lap": PER_BUCKET},
        "assignment_counts": {**l1_manifest["assignment_counts"], **dict(sorted(summary.items()))},
        "l1_record_count": int(l1_manifest["record_count"]),
        "l2_record_count": len(record_keys),
        "record_count": int(l1_manifest["record_count"]) + len(record_keys),
        "selection": {
            "candidate_pool": "all_mapped_current_v10_records_within_frozen_morgan100_parents",
            "morgan": "global_similarity_then_seeded_record_tie",
            "semantic_lap": "semantic_level_rank_then_three_records_per_bucket_per_lap",
            "semantic_metadata_visible": False,
        },
        "semantic_binding": semantic_audit,
        "inputs": {
            **l1_manifest.get("inputs", {}),
            "l1_manifest": {"path": str(l1_manifest_path.resolve()), "sha256": sha256_file(l1_manifest_path)},
            "source_manifest": {"path": str(source_manifest_path.resolve()), "sha256": sha256_file(source_manifest_path)},
            "l2_builder": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
            "runtime_selector": {
                "path": str(Path(__file__).with_name("l1_context_cache.py").resolve()),
                "sha256": sha256_file(Path(__file__).with_name("l1_context_cache.py")),
            },
            **semantic_audit["inputs"],
        },
        "database_sha256": sha256_file(database),
    }
    (temporary / "VERSION.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.rename(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=SEMANTIC_TASKS)
    parser.add_argument("--subset", default="valid", choices=("valid",))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.task, args.subset, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
