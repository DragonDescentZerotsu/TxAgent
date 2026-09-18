"""Publish Morgan-parent L2 control and semantic panels with a global record cap."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any

from predict.retrieval.policies import normalize_molecule_identity, seeded_rank_tie_key
from predict.utils.json import sha256_file

from . import three_pools
from .build_cache_matched_v2 import _similarity
from .build_context_l2_morgan_semantic_cache import _eligibility
from .runtime import cache_profile_root
from .semantic_bucket_selection import select_global_molecule_records


SCHEMA_VERSION = "l1_context_morgan_semantic_l2.v2"
PROFILE = "l1_context_morgan_semantic_l2_v2"
OUTPUT_ROOT = cache_profile_root(PROFILE)
L1_ROOT = cache_profile_root("l1_context_morgan10_v2")
SOURCE_ROOT = three_pools.ROOT
PARENT_LIMIT = 10
TOTAL_RECORD_LIMIT = 25
PER_PARENT_LIMIT = 10
MODES = ("morgan_parent_control", "morgan_parent_semantic")


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_contract(task: str, subset: str) -> tuple[Path, dict, Path, dict]:
    l1_root = L1_ROOT / task / "scaffold" / subset
    source_root = SOURCE_ROOT / task / "scaffold" / subset
    l1 = json.loads((l1_root / "VERSION.json").read_text())
    source = json.loads((source_root / "VERSION.json").read_text())
    expected = {
        "schema_version": "l1_context_retrieval.v2",
        "status": "complete",
        "morgan_primary_parent_width": 10,
    }
    if any(l1.get(key) != value for key, value in expected.items()):
        raise ValueError("complete Morgan-10 V2 L1 cache is required")
    if source.get("status") != "complete":
        raise ValueError("complete Morgan-100 all-record source cache is required")
    database = l1_root / l1["database"]
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro&immutable=1", uri=True) as db:
        metadata = dict(db.execute("SELECT key,value FROM metadata"))
    if metadata.get("content_id") != l1.get("content_id"):
        raise ValueError("L1 database and manifest identities differ")
    scores = source_root / "scores.sqlite3"
    if sha256_file(scores) != source["cache_sha256"]:
        raise ValueError("Morgan-100 all-record source database changed")
    return l1_root, l1, source_root, source


def _create_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE l2_records(
          record_key INTEGER PRIMARY KEY,external_record_id TEXT UNIQUE NOT NULL,
          parent_id TEXT NOT NULL,parent_smiles TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE l2_parent_assignments(
          query_id INTEGER NOT NULL,mode TEXT NOT NULL,parent_rank INTEGER NOT NULL,
          parent_id TEXT NOT NULL,morgan_similarity REAL NOT NULL,
          available_records INTEGER NOT NULL,available_semantic_records INTEGER NOT NULL,
          selected_records INTEGER NOT NULL,
          PRIMARY KEY(query_id,mode,parent_rank),UNIQUE(query_id,mode,parent_id));
        CREATE TABLE l2_assignments(
          query_id INTEGER NOT NULL,mode TEXT NOT NULL,selection_rank INTEGER NOT NULL,
          parent_rank INTEGER NOT NULL,within_parent_rank INTEGER NOT NULL,
          record_key INTEGER NOT NULL,semantic_bucket_id TEXT,semantic_rank INTEGER,
          retrieval_eligible INTEGER,morgan_similarity REAL NOT NULL,
          PRIMARY KEY(query_id,mode,selection_rank),UNIQUE(query_id,mode,record_key));
        CREATE TABLE l2_selection_counts(
          query_id INTEGER NOT NULL,mode TEXT NOT NULL,candidate_records INTEGER NOT NULL,
          mapped_records INTEGER NOT NULL,eligible_records INTEGER NOT NULL,
          excluded_bad_records INTEGER NOT NULL,unmapped_records INTEGER NOT NULL,
          candidate_parents INTEGER NOT NULL,selected_parents INTEGER NOT NULL,
          selected_records INTEGER NOT NULL,shortfall INTEGER NOT NULL,
          PRIMARY KEY(query_id,mode));
        """
    )


def _l1_visible_record_ids(connection: sqlite3.Connection, query_id: int) -> set[str]:
    contexts = [row[0] for row in connection.execute(
        """SELECT c.context_key FROM candidates AS c JOIN gold_contexts AS g USING(context_key)
           WHERE c.query_id=? AND c.morgan_parent_rank<=10
           ORDER BY c.morgan_parent_rank,g.external_context_id LIMIT 10""",
        (query_id,),
    )]
    if len(contexts) != 10:
        raise ValueError(f"L1 query {query_id} does not select ten contexts")
    placeholders = ",".join("?" for _ in contexts)
    return {str(row[0]) for row in connection.execute(
        f"""SELECT r.external_record_id FROM context_records AS cr
             JOIN records AS r USING(record_key)
             WHERE cr.context_key IN ({placeholders}) AND cr.within_context_rank<=10""",
        contexts,
    )}


def _query_rows(
    source: sqlite3.Connection, source_query_id: int, binding: dict[str, dict[str, Any]],
    seen: set[str], task: str, benchmark_id: str, query_smiles: str,
) -> tuple[list[dict[str, Any]], Counter]:
    rows, counts = [], Counter()
    stored_rows = source.execute(
        """SELECT r.external_record_id,r.parent_id,r.payload FROM assignments AS a
           JOIN records AS r USING(record_key)
           WHERE a.query_id=? AND a.pool='all' AND a.level='L2'""",
        (source_query_id,),
    )
    for stored in stored_rows:
        record_id = str(stored["external_record_id"])
        if record_id in seen:
            counts["l1_overlap"] += 1
            continue
        payload = json.loads(stored["payload"])
        info = binding.get(str(payload.get("source_row_uid") or ""))
        if info is not None and info["record_id"] != record_id:
            raise ValueError(f"changed semantic binding: {record_id}")
        identity = normalize_molecule_identity(str(payload["canonical_smiles"]))
        parent_smiles = str(identity.parent_smiles or "")
        parent_id = str(stored["parent_id"])
        if identity.status != "ok" or (identity.parent_inchi_key or parent_smiles) != parent_id:
            raise ValueError(f"unresolved or changed L2 parent: {record_id}")
        rows.append({
            "record_id": record_id, "parent_id": parent_id,
            "parent_smiles": parent_smiles, "payload": payload,
            "morgan_similarity": _similarity(query_smiles, parent_smiles),
            "semantic_bucket_id": info["semantic_bucket_id"] if info else None,
            "semantic_rank": info["semantic_rank"] if info else None,
            "retrieval_eligible": info["retrieval_eligible"] if info else None,
            "tie_key": seeded_rank_tie_key(
                0, task, benchmark_id, "L2", parent_id, record_id,
            ),
        })
    counts["candidate_records"] = len(rows)
    counts["mapped_records"] = sum(row["semantic_rank"] is not None for row in rows)
    counts["eligible_records"] = sum(row["retrieval_eligible"] is True for row in rows)
    return rows, counts


def _top_parents(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, list[dict]]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["parent_id"]].append(row)
    parents = sorted(
        grouped, key=lambda parent: (-grouped[parent][0]["morgan_similarity"], parent)
    )[:PARENT_LIMIT]
    if len(parents) != PARENT_LIMIT:
        raise ValueError(f"only {len(parents)} L2 parents; require {PARENT_LIMIT}")
    for rank, parent in enumerate(parents, 1):
        for row in grouped[parent]:
            row["parent_rank"] = rank
    return parents, grouped


def _store_mode(
    connection: sqlite3.Connection, query_id: int, mode: str, parents: list[str],
    grouped: dict[str, list[dict]], record_keys: dict[str, int], summary: Counter,
) -> None:
    candidates = [row for parent in parents for row in grouped[parent]]
    method = "semantic" if mode.endswith("semantic") else "control"
    selected = select_global_molecule_records(
        candidates, method=method, limit=TOTAL_RECORD_LIMIT,
        per_parent_limit=PER_PARENT_LIMIT,
    )
    selected_counts = Counter(str(row["parent_id"]) for row in selected)
    within_counts: Counter[str] = Counter()
    for selection_rank, row in enumerate(selected, 1):
        parent = str(row["parent_id"])
        within_counts[parent] += 1
        record_id = str(row["record_id"])
        if record_id not in record_keys:
            record_keys[record_id] = len(record_keys) + 1
            connection.execute("INSERT INTO l2_records VALUES (?,?,?,?,?)", (
                record_keys[record_id], record_id, parent, row["parent_smiles"],
                json.dumps(row["payload"], sort_keys=True, separators=(",", ":")),
            ))
        connection.execute("INSERT INTO l2_assignments VALUES (?,?,?,?,?,?,?,?,?,?)", (
            query_id, mode, selection_rank, row["parent_rank"], within_counts[parent],
            record_keys[record_id], row["semantic_bucket_id"], row["semantic_rank"],
            row["retrieval_eligible"], row["morgan_similarity"],
        ))
    for parent_rank, parent in enumerate(parents, 1):
        rows = grouped[parent]
        connection.execute("INSERT INTO l2_parent_assignments VALUES (?,?,?,?,?,?,?,?)", (
            query_id, mode, parent_rank, parent, rows[0]["morgan_similarity"], len(rows),
            sum(row["retrieval_eligible"] is True for row in rows), selected_counts[parent],
        ))
    mapped = sum(row["semantic_rank"] is not None for row in candidates)
    eligible = sum(row["retrieval_eligible"] is True for row in candidates)
    values = (query_id, mode, len(candidates), mapped, eligible, mapped - eligible,
              len(candidates) - mapped, len(parents), len(selected_counts), len(selected),
              TOTAL_RECORD_LIMIT - len(selected))
    connection.execute("INSERT INTO l2_selection_counts VALUES (?,?,?,?,?,?,?,?,?,?,?)", values)
    summary[f"{mode}_records"] += len(selected)
    summary[f"{mode}_shortfall"] += TOTAL_RECORD_LIMIT - len(selected)


def _populate(
    output: sqlite3.Connection, source: sqlite3.Connection,
    binding: dict[str, dict[str, Any]], task: str,
) -> Counter:
    source_queries = {str(row[0]): int(row[1]) for row in source.execute(
        "SELECT benchmark_row_id,query_id FROM benchmark_queries"
    )}
    record_keys, summary = {}, Counter()
    queries = output.execute(
        """SELECT b.benchmark_row_id,b.query_id,q.query_parent_smiles
           FROM benchmark_queries AS b JOIN queries AS q USING(query_id)
           ORDER BY b.query_id"""
    ).fetchall()
    for query in queries:
        query_id, benchmark_id = int(query["query_id"]), str(query["benchmark_row_id"])
        rows, counts = _query_rows(
            source, source_queries[benchmark_id], binding,
            _l1_visible_record_ids(output, query_id), task, benchmark_id,
            str(query["query_parent_smiles"]),
        )
        parents, grouped = _top_parents(rows)
        for mode in MODES:
            _store_mode(output, query_id, mode, parents, grouped, record_keys, summary)
        summary.update(counts)
        summary["selected_parent_slots"] += len(parents)
    summary["l2_unique_records"] = len(record_keys)
    summary["queries"] = len(queries)
    return summary


def _manifest(
    task: str, subset: str, l1: dict, source_path: Path, source: dict,
    eligibility: dict, database: Path, summary: Counter, content_id: str,
) -> dict[str, Any]:
    selection = {
        "candidate_pool": "all_v10_L2_records_within_frozen_morgan100_parents",
        "parent_order": "morgan_tanimoto_then_parent_id", "parent_limit": PARENT_LIMIT,
        "total_record_limit": TOTAL_RECORD_LIMIT, "record_limit_per_parent": PER_PARENT_LIMIT,
        "control_order": "global_seeded_record_tie_without_semantic_filter",
        "semantic_order": "global_semantic_rank_then_morgan_parent_rank_then_stable_tie",
        "semantic_filter": "retrieval_eligible_true_and_ranked",
        "semantic_metadata_visible": False, "parent_backfill_after_filter": False,
    }
    return {
        **l1, "schema_version": SCHEMA_VERSION, "status": "complete",
        "task_id": task, "subset": subset, "database": database.name,
        "content_id": content_id, "l1_content_id": l1["content_id"],
        "capacities": {**l1["capacities"], "l2_molecules": PARENT_LIMIT,
                       "l2_total_records": TOTAL_RECORD_LIMIT,
                       "l2_records_per_molecule": PER_PARENT_LIMIT},
        "assignment_counts": {**l1["assignment_counts"], **dict(sorted(summary.items()))},
        "l1_record_count": int(l1["record_count"]),
        "l2_record_count": int(summary["l2_unique_records"]),
        "record_count": int(l1["record_count"]) + int(summary["l2_unique_records"]),
        "selection": selection, "eligibility": eligibility,
        "inputs": {**l1.get("inputs", {}), "source_manifest": {
            "path": str(source_path.resolve()), "sha256": sha256_file(source_path)},
            "l2_builder": {"path": str(Path(__file__).resolve()),
                           "sha256": sha256_file(Path(__file__))},
            "runtime_selector": {
                "path": str(Path(__file__).with_name("l1_context_cache.py").resolve()),
                "sha256": sha256_file(Path(__file__).with_name("l1_context_cache.py")),
            }},
        "database_sha256": sha256_file(database),
    }


def build(task: str, subset: str = "valid", output_root: Path | None = None) -> dict:
    if task not in three_pools.MODULES or subset != "valid":
        raise ValueError("Morgan L2 V2 supports BBB/oral validation only")
    destination = (output_root or OUTPUT_ROOT / task / "scaffold" / subset).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing cache: {destination}")
    binding, eligibility = _eligibility(task)
    l1_root, l1, source_root, source = _source_contract(task, subset)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    database = temporary / "retrieval.sqlite3"
    try:
        shutil.copy2(l1_root / l1["database"], database)
        database.chmod(0o600)
        with sqlite3.connect(database) as output, sqlite3.connect(
            f"file:{(source_root / 'scores.sqlite3').resolve()}?mode=ro&immutable=1", uri=True,
        ) as source_db:
            output.row_factory = source_db.row_factory = sqlite3.Row
            _create_tables(output)
            summary = _populate(output, source_db, binding, task)
            identity = _digest({"schema_version": SCHEMA_VERSION, "task": task,
                "l1_content_id": l1["content_id"], "source": source["cache_sha256"],
                "eligibility": eligibility["record_map_sha256"],
                "builder": sha256_file(Path(__file__)), "summary": dict(sorted(summary.items()))})
            output.execute("UPDATE metadata SET value=? WHERE key='schema_version'", (SCHEMA_VERSION,))
            output.execute("UPDATE metadata SET value=? WHERE key='content_id'", (identity,))
            output.commit()
            output.execute("VACUUM")
            if output.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Morgan L2 V2 cache failed SQLite quick_check")
        source_manifest = source_root / "VERSION.json"
        manifest = _manifest(task, subset, l1, source_manifest, source, eligibility,
                             database, summary, identity)
        (temporary / "VERSION.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.chmod(database, 0o444)
        os.chmod(temporary / "VERSION.json", 0o444)
        temporary.rename(destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(three_pools.MODULES))
    parser.add_argument("--subset", default="valid", choices=("valid",))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.task, args.subset, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
