"""Publish matched molecule-first Morgan/control and semantic-ordered L2 plans."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any

import pandas as pd

from semantic_buckets.artifacts import resolve_semantic_bucket_artifacts
from predict.retrieval.policies import normalize_molecule_identity, seeded_rank_tie_key
from predict.utils.json import sha256_file

from . import three_pools
from .build_cache_matched_v2 import _similarity
from .build_context_l2_semantic_cache import _l1_visible_record_ids
from .runtime import cache_profile_root
from .semantic_bucket_selection import select_molecule_records


SCHEMA_VERSION = "l1_context_morgan_semantic_l2.v1"
PROFILE = "l1_context_morgan_semantic_l2_v1"
OUTPUT_ROOT = cache_profile_root(PROFILE)
L1_ROOT = cache_profile_root("l1_context_morgan100_v1")
SOURCE_ROOT = three_pools.ROOT
PARENT_LIMIT = 10
RECORD_LIMIT = 10
MODES = ("morgan_parent_control", "morgan_parent_semantic")


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _eligibility(task: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    artifacts = resolve_semantic_bucket_artifacts(task)
    manifest_path = artifacts.manifest
    path = artifacts.retrieval_eligibility
    if path is None:
        raise ValueError(f"complete L2 eligibility artifact required: {task}")
    frame = pd.read_parquet(path, columns=["canonical_record_id", "source_row_uid", "level",
        "semantic_bucket_id", "level_rank", "retrieval_eligible"])
    frame = frame[frame.level.eq("L2")].copy()
    if set(frame.level) != {"L2"} or frame.source_row_uid.duplicated().any():
        raise ValueError(f"invalid L2 eligibility mapping: {task}")
    binding = {
        str(row.source_row_uid): {"record_id": str(row.canonical_record_id),
            "semantic_bucket_id": str(row.semantic_bucket_id), "semantic_rank": int(row.level_rank),
            "retrieval_eligible": bool(row.retrieval_eligible)}
        for row in frame.itertuples(index=False)
    }
    return binding, {"manifest": str(manifest_path.resolve()),
                     "manifest_sha256": sha256_file(manifest_path),
                     "record_map": str(path.resolve()), "record_map_sha256": sha256_file(path),
                     "eligible_records": int(frame.retrieval_eligible.sum()),
                     "excluded_records": int((~frame.retrieval_eligible).sum())}


def _source_contract(task: str, subset: str) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    l1_root = L1_ROOT / task / "scaffold" / subset
    source_root = SOURCE_ROOT / task / "scaffold" / subset
    l1_manifest = json.loads((l1_root / "VERSION.json").read_text())
    source_manifest = json.loads((source_root / "VERSION.json").read_text())
    if l1_manifest.get("status") != "complete" or source_manifest.get("status") != "complete":
        raise ValueError("complete frozen L1 and Morgan-100 caches are required")
    l1_database = l1_root / l1_manifest["database"]
    l1_sha256 = sha256_file(l1_database)
    if l1_manifest.get("database_sha256") not in {None, l1_sha256}:
        raise ValueError("frozen L1 database changed")
    with sqlite3.connect(
        f"file:{l1_database.resolve()}?mode=ro&immutable=1", uri=True
    ) as connection:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    if metadata.get("content_id") != l1_manifest.get("content_id"):
        raise ValueError("frozen L1 database identity changed")
    l1_manifest = {**l1_manifest, "database_sha256": l1_sha256}
    if sha256_file(source_root / "scores.sqlite3") != source_manifest["cache_sha256"]:
        raise ValueError("Morgan-100 source database changed")
    return l1_root, l1_manifest, source_root, source_manifest


def _create_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE l2_records(
          record_key INTEGER PRIMARY KEY,external_record_id TEXT UNIQUE NOT NULL,
          parent_id TEXT NOT NULL,parent_smiles TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE l2_parent_assignments(
          query_id INTEGER NOT NULL,mode TEXT NOT NULL,parent_rank INTEGER NOT NULL,
          parent_id TEXT NOT NULL,morgan_similarity REAL NOT NULL,
          available_eligible_records INTEGER NOT NULL,selected_records INTEGER NOT NULL,
          PRIMARY KEY(query_id,mode,parent_rank),UNIQUE(query_id,mode,parent_id));
        CREATE TABLE l2_assignments(
          query_id INTEGER NOT NULL,mode TEXT NOT NULL,parent_rank INTEGER NOT NULL,
          within_parent_rank INTEGER NOT NULL,record_key INTEGER NOT NULL,
          semantic_bucket_id TEXT NOT NULL,semantic_rank INTEGER NOT NULL,
          morgan_similarity REAL NOT NULL,
          PRIMARY KEY(query_id,mode,parent_rank,within_parent_rank),
          UNIQUE(query_id,mode,record_key));
        CREATE TABLE l2_selection_counts(
          query_id INTEGER PRIMARY KEY,candidate_records INTEGER NOT NULL,
          eligible_records INTEGER NOT NULL,excluded_records INTEGER NOT NULL,
          unmapped_records INTEGER NOT NULL,
          candidate_parents INTEGER NOT NULL,eligible_parents INTEGER NOT NULL,
          selected_parents INTEGER NOT NULL);
        """
    )


def _query_candidates(source: sqlite3.Connection, source_query_id: int,
                      binding: dict[str, dict[str, Any]], seen: set[str],
                      task: str, benchmark_id: str, query_smiles: str) -> tuple[list[dict[str, Any]], Counter]:
    rows, counts = [], Counter()
    query = source.execute(
        """SELECT r.external_record_id,r.parent_id,r.payload FROM assignments AS a
           JOIN records AS r USING(record_key)
           WHERE a.query_id=? AND a.pool='all' AND a.level='L2'""", (source_query_id,)
    )
    for stored in query:
        payload = json.loads(stored["payload"])
        info = binding.get(str(payload.get("source_row_uid") or ""))
        counts["candidate_records"] += 1
        if info is None:
            counts["unmapped_records"] += 1
            counts["excluded_records"] += 1
            continue
        if info["record_id"] != str(stored["external_record_id"]):
            raise ValueError(f"missing or changed eligibility binding: {stored['external_record_id']}")
        if str(stored["external_record_id"]) in seen or not info["retrieval_eligible"]:
            counts["excluded_records"] += 1
            continue
        identity = normalize_molecule_identity(str(payload["canonical_smiles"]))
        parent_smiles = str(identity.parent_smiles or "")
        parent_id = str(stored["parent_id"])
        if identity.status != "ok" or (identity.parent_inchi_key or parent_smiles) != parent_id:
            raise ValueError(f"unresolved or changed L2 parent: {stored['external_record_id']}")
        rows.append({**info, "parent_id": parent_id, "parent_smiles": parent_smiles,
            "payload": payload, "morgan_similarity": _similarity(query_smiles, parent_smiles),
            "tie_key": seeded_rank_tie_key(0, task, benchmark_id, "L2", parent_id,
                                             str(stored["external_record_id"]))})
    counts["eligible_records"] = len(rows)
    return rows, counts


def _select(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["parent_id"]].append(row)
    parents = sorted(grouped, key=lambda parent: (-grouped[parent][0]["morgan_similarity"], parent))
    selected = parents[:PARENT_LIMIT]
    if len(selected) != PARENT_LIMIT:
        raise ValueError(f"only {len(selected)} eligible L2 parents; require {PARENT_LIMIT}")
    return selected, grouped


def _insert_selection(connection: sqlite3.Connection, query_id: int, mode: str,
                      parents: list[str], grouped: dict[str, list[dict[str, Any]]],
                      record_keys: dict[str, int]) -> int:
    method = "semantic" if mode.endswith("semantic") else "control"
    total = 0
    for parent_rank, parent in enumerate(parents, 1):
        rows = grouped[parent]
        chosen = select_molecule_records(rows, method=method, limit=RECORD_LIMIT)
        connection.execute("INSERT INTO l2_parent_assignments VALUES (?,?,?,?,?,?,?)",
            (query_id, mode, parent_rank, parent, rows[0]["morgan_similarity"], len(rows), len(chosen)))
        for within_rank, row in enumerate(chosen, 1):
            record_id = row["record_id"]
            if record_id not in record_keys:
                record_keys[record_id] = len(record_keys) + 1
                connection.execute("INSERT INTO l2_records VALUES (?,?,?,?,?)", (record_keys[record_id],
                    record_id, parent, row["parent_smiles"], json.dumps(row["payload"], sort_keys=True,
                    separators=(",", ":"))))
            connection.execute("INSERT INTO l2_assignments VALUES (?,?,?,?,?,?,?,?)",
                (query_id, mode, parent_rank, within_rank, record_keys[record_id],
                 row["semantic_bucket_id"], row["semantic_rank"], row["morgan_similarity"]))
        total += len(chosen)
    return total


def _populate(output: sqlite3.Connection, source: sqlite3.Connection,
              binding: dict[str, dict[str, Any]], task: str) -> Counter:
    source_queries = {str(row[0]): int(row[1]) for row in source.execute(
        "SELECT benchmark_row_id,query_id FROM benchmark_queries")}
    record_keys, summary = {}, Counter()
    queries = output.execute("""SELECT b.benchmark_row_id,b.query_id,q.query_parent_smiles
        FROM benchmark_queries AS b JOIN queries AS q USING(query_id) ORDER BY b.query_id""").fetchall()
    for query in queries:
        query_id, benchmark_id = int(query["query_id"]), str(query["benchmark_row_id"])
        seen = _l1_visible_record_ids(output, query_id)
        rows, counts = _query_candidates(source, source_queries[benchmark_id], binding, seen,
                                          task, benchmark_id, str(query["query_parent_smiles"]))
        parents, grouped = _select(rows)
        for mode in MODES:
            summary[f"{mode}_records"] += _insert_selection(
                output, query_id, mode, parents, grouped, record_keys)
        counts["candidate_parents"] = len({row["parent_id"] for row in rows})
        counts["eligible_parents"] = len(grouped)
        summary.update(counts)
        output.execute("INSERT INTO l2_selection_counts VALUES (?,?,?,?,?,?,?,?)", (query_id,
            counts["candidate_records"], counts["eligible_records"], counts["excluded_records"],
            counts["unmapped_records"], counts["candidate_parents"],
            counts["eligible_parents"], len(parents)))
    summary["l2_unique_records"] = len(record_keys)
    summary["queries"] = len(queries)
    return summary


def _manifest(task: str, subset: str, l1_manifest: dict[str, Any],
              source_manifest_path: Path, source_manifest: dict[str, Any],
              eligibility: dict[str, Any], database: Path, summary: Counter,
              content_id: str) -> dict[str, Any]:
    identity = {"schema_version": SCHEMA_VERSION, "task_id": task, "subset": subset,
        "l1_content_id": l1_manifest["content_id"], "source_cache_sha256": source_manifest["cache_sha256"],
        "eligibility_record_map_sha256": eligibility["record_map_sha256"],
        "builder_sha256": sha256_file(Path(__file__)),
        "parent_limit": PARENT_LIMIT, "record_limit_per_parent": RECORD_LIMIT,
        "assignment_counts": dict(sorted(summary.items()))}
    return {**l1_manifest, **identity, "status": "complete", "database": database.name,
        "content_id": content_id, "capacities": {**l1_manifest["capacities"],
            "l2_molecules": PARENT_LIMIT, "l2_records_per_molecule": RECORD_LIMIT},
        "assignment_counts": {**l1_manifest["assignment_counts"], **dict(sorted(summary.items()))},
        "l1_record_count": int(l1_manifest["record_count"]),
        "l2_record_count": int(summary["l2_unique_records"]),
        "record_count": int(l1_manifest["record_count"]) + int(summary["l2_unique_records"]),
        "selection": {"candidate_pool": "all_mapped_v10_L2_within_frozen_morgan100_parents",
            "parent_order": "morgan_tanimoto_then_parent_id", "parent_limit": PARENT_LIMIT,
            "record_limit_per_parent": RECORD_LIMIT, "control_order": "seeded_record_tie",
            "semantic_order": "drain_ascending_semantic_rank_then_seeded_record_tie",
            "semantic_metadata_visible": False, "zero_record_parents_backfilled": True,
            "unmapped_semantic_records": "exclude"},
        "eligibility": eligibility,
        "inputs": {**l1_manifest.get("inputs", {}), "source_manifest": {
            "path": str(source_manifest_path.resolve()), "sha256": sha256_file(source_manifest_path)},
            "l2_builder": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
            "runtime_selector": {"path": str(Path(__file__).with_name("l1_context_cache.py").resolve()),
                                 "sha256": sha256_file(Path(__file__).with_name("l1_context_cache.py"))}},
        "database_sha256": sha256_file(database)}


def build(task: str, subset: str = "valid", output_root: Path | None = None) -> dict[str, Any]:
    if task not in three_pools.MODULES or subset != "valid":
        raise ValueError("molecule-first L2 supports BBB/oral validation only")
    destination = (output_root or OUTPUT_ROOT / task / "scaffold" / subset).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing cache: {destination}")
    binding, eligibility = _eligibility(task)
    l1_root, l1_manifest, source_root, source_manifest = _source_contract(task, subset)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    database = temporary / "retrieval.sqlite3"
    shutil.copy2(l1_root / l1_manifest["database"], database)
    database.chmod(0o600)
    output = sqlite3.connect(database)
    output.row_factory = sqlite3.Row
    source = sqlite3.connect(f"file:{(source_root / 'scores.sqlite3').resolve()}?mode=ro&immutable=1", uri=True)
    source.row_factory = sqlite3.Row
    try:
        _create_tables(output)
        summary = _populate(output, source, binding, task)
        identity = _digest({"schema_version": SCHEMA_VERSION, "task_id": task, "subset": subset,
            "l1_content_id": l1_manifest["content_id"],
            "source_cache_sha256": source_manifest["cache_sha256"],
            "eligibility_record_map_sha256": eligibility["record_map_sha256"],
            "builder_sha256": sha256_file(Path(__file__)),
            "parent_limit": PARENT_LIMIT, "record_limit_per_parent": RECORD_LIMIT,
            "assignment_counts": dict(sorted(summary.items()))})
        output.execute("UPDATE metadata SET value=? WHERE key='schema_version'", (SCHEMA_VERSION,))
        output.execute("UPDATE metadata SET value=? WHERE key='content_id'", (identity,))
        output.commit()
        output.execute("VACUUM")
        if output.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("molecule-first L2 cache failed SQLite quick_check")
    finally:
        source.close()
        output.close()
    source_manifest_path = source_root / "VERSION.json"
    manifest = _manifest(task, subset, l1_manifest, source_manifest_path, source_manifest,
                         eligibility, database, summary, identity)
    (temporary / "VERSION.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.rename(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(three_pools.MODULES))
    parser.add_argument("--subset", default="valid", choices=("valid",))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.task, args.subset, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
