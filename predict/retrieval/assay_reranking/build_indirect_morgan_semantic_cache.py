"""Publish direct-free Morgan-parent control and semantic panels for L2-L4."""

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

import pandas as pd

from semantic_buckets.artifacts import resolve_semantic_bucket_artifacts
from predict.retrieval.policies import normalize_molecule_identity, seeded_rank_tie_key
from predict.utils.json import sha256_file

from . import three_pools
from .build_cache_matched_v2 import _similarity
from .runtime import cache_profile_root
from .semantic_bucket_selection import select_global_molecule_records


SCHEMA_VERSION = "indirect_morgan_semantic_l2_l4.v1"
PROFILE = "indirect_morgan_semantic_l2_l4_v1"
OUTPUT_ROOT = cache_profile_root(PROFILE)
V4_ROOT = cache_profile_root("l1_context_morgan_semantic_l2_v4")
LEVELS = ("L2", "L3", "L4")
MODES = ("morgan_parent_control", "morgan_parent_semantic")
PARENT_LIMIT = 100
TOTAL_RECORD_LIMIT = 25
PER_PARENT_LIMIT = 10
PER_PARENT_BUCKET_LIMIT = 5


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _eligibility(task: str) -> tuple[dict[tuple[str, str], dict[str, Any]], dict]:
    artifacts = resolve_semantic_bucket_artifacts(task)
    manifest_path = artifacts.manifest
    path = artifacts.retrieval_eligibility
    if path is None:
        raise ValueError(f"complete L2-L4 eligibility artifact required: {task}")
    frame = pd.read_parquet(path, columns=[
        "canonical_record_id", "source_row_uid", "level", "semantic_bucket_id",
        "level_rank", "retrieval_eligible",
    ])
    if set(frame.level) != set(LEVELS) or frame.source_row_uid.duplicated().any():
        raise ValueError(f"invalid L2-L4 eligibility mapping: {task}")
    binding = {
        (str(row.level), str(row.source_row_uid)): {
            "record_id": str(row.canonical_record_id),
            "semantic_bucket_id": str(row.semantic_bucket_id),
            "semantic_rank": int(row.level_rank),
            "retrieval_eligible": bool(row.retrieval_eligible),
        }
        for row in frame.itertuples(index=False)
    }
    return binding, {
        "manifest": str(manifest_path.resolve()), "manifest_sha256": sha256_file(manifest_path),
        "record_map": str(path.resolve()), "record_map_sha256": sha256_file(path),
    }


def _source(task: str, subset: str) -> tuple[Path, dict[str, Any]]:
    root = three_pools.ROOT / task / "scaffold" / subset
    manifest_path = root / "VERSION.json"
    manifest = json.loads(manifest_path.read_text())
    database = root / "scores.sqlite3"
    if (manifest.get("status") != "complete" or manifest.get("morgan_pool_size") != 100
            or sha256_file(database) != manifest.get("cache_sha256")):
        raise ValueError(f"complete Morgan-100 three-pool source required: {task}/{subset}")
    return database, manifest


def _create_tables(connection: sqlite3.Connection) -> None:
    connection.executescript("""
      CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
      CREATE TABLE queries(query_id INTEGER PRIMARY KEY,query_parent_smiles TEXT NOT NULL);
      CREATE TABLE benchmark_queries(benchmark_row_id TEXT PRIMARY KEY,drug TEXT NOT NULL,
                                     query_id INTEGER NOT NULL);
      CREATE TABLE records(record_key INTEGER PRIMARY KEY,external_record_id TEXT UNIQUE NOT NULL,
                           parent_id TEXT NOT NULL,parent_smiles TEXT NOT NULL,level TEXT NOT NULL,
                           payload TEXT NOT NULL);
      CREATE TABLE assignments(query_id INTEGER NOT NULL,level TEXT NOT NULL,mode TEXT NOT NULL,
        selection_rank INTEGER NOT NULL,parent_rank INTEGER NOT NULL,within_parent_rank INTEGER NOT NULL,
        record_key INTEGER NOT NULL,semantic_bucket_id TEXT,semantic_rank INTEGER,
        retrieval_eligible INTEGER,morgan_similarity REAL NOT NULL,
        PRIMARY KEY(query_id,level,mode,selection_rank),UNIQUE(query_id,level,mode,record_key));
      CREATE TABLE parent_assignments(query_id INTEGER NOT NULL,level TEXT NOT NULL,mode TEXT NOT NULL,
        parent_rank INTEGER NOT NULL,parent_id TEXT NOT NULL,morgan_similarity REAL NOT NULL,
        available_records INTEGER NOT NULL,available_semantic_records INTEGER NOT NULL,
        selected_records INTEGER NOT NULL,PRIMARY KEY(query_id,level,mode,parent_rank));
      CREATE TABLE selection_counts(query_id INTEGER NOT NULL,level TEXT NOT NULL,mode TEXT NOT NULL,
        candidate_records INTEGER NOT NULL,mapped_records INTEGER NOT NULL,eligible_records INTEGER NOT NULL,
        excluded_bad_records INTEGER NOT NULL,unmapped_records INTEGER NOT NULL,candidate_parents INTEGER NOT NULL,
        selected_parents INTEGER NOT NULL,selected_records INTEGER NOT NULL,shortfall INTEGER NOT NULL,
        PRIMARY KEY(query_id,level,mode));
    """)


def _query_rows(source: sqlite3.Connection, query_id: int, level: str,
                binding: dict, task: str, benchmark: str, query_smiles: str) -> list[dict]:
    rows = []
    stored = source.execute(
        """SELECT r.external_record_id,r.parent_id,r.payload FROM assignments a
           JOIN records r USING(record_key) WHERE a.query_id=? AND a.pool='all' AND a.level=?""",
        (query_id, level),
    )
    for item in stored:
        record_id, parent_id, payload = str(item[0]), str(item[1]), json.loads(item[2])
        info = binding.get((level, str(payload.get("source_row_uid") or "")))
        if info is not None and info["record_id"] != record_id:
            raise ValueError(f"changed semantic binding: {record_id}")
        identity = normalize_molecule_identity(str(payload["canonical_smiles"]))
        parent_smiles = str(identity.parent_smiles or "")
        if identity.status != "ok" or (identity.parent_inchi_key or parent_smiles) != parent_id:
            raise ValueError(f"unresolved parent for {record_id}")
        rows.append({"record_id": record_id, "parent_id": parent_id, "parent_smiles": parent_smiles,
            "payload": payload, "morgan_similarity": _similarity(query_smiles, parent_smiles),
            "semantic_bucket_id": info["semantic_bucket_id"] if info else None,
            "semantic_rank": info["semantic_rank"] if info else None,
            "retrieval_eligible": info["retrieval_eligible"] if info else None,
            "tie_key": seeded_rank_tie_key(0, task, benchmark, level, parent_id, record_id)})
    return rows


def _parent_prefix(rows: list[dict]) -> tuple[list[str], dict[str, list[dict]]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["parent_id"]].append(row)
    ordered = sorted(grouped, key=lambda parent: (-grouped[parent][0]["morgan_similarity"], parent))
    candidates: list[dict] = []
    for rank, parent in enumerate(ordered[:PARENT_LIMIT], 1):
        for row in grouped[parent]:
            row["parent_rank"] = rank
        candidates.extend(grouped[parent])
        control = _select(candidates, "control")
        semantic = _select(candidates, "semantic")
        if len(control) == len(semantic) == TOTAL_RECORD_LIMIT:
            return ordered[:rank], grouped
    raise ValueError("Morgan-100 parents cannot fill both 25-record panels")


def _select(rows: list[dict], method: str) -> list[dict]:
    return select_global_molecule_records(
        rows, method=method, limit=TOTAL_RECORD_LIMIT, per_parent_limit=PER_PARENT_LIMIT,
        per_parent_bucket_limit=PER_PARENT_BUCKET_LIMIT if method == "semantic" else None,
    )


def _store_mode(connection: sqlite3.Connection, query_id: int, level: str, mode: str,
                parents: list[str], grouped: dict, record_keys: dict, summary: Counter) -> None:
    candidates = [row for parent in parents for row in grouped[parent]]
    selected = _select(candidates, "semantic" if mode.endswith("semantic") else "control")
    parent_counts, within = Counter(row["parent_id"] for row in selected), Counter()
    for rank, row in enumerate(selected, 1):
        parent, record_id = row["parent_id"], row["record_id"]
        within[parent] += 1
        if record_id not in record_keys:
            record_keys[record_id] = len(record_keys) + 1
            connection.execute("INSERT INTO records VALUES (?,?,?,?,?,?)", (
                record_keys[record_id], record_id, parent, row["parent_smiles"], level,
                json.dumps(row["payload"], sort_keys=True, separators=(",", ":"))))
        connection.execute("INSERT INTO assignments VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
            query_id, level, mode, rank, row["parent_rank"], within[parent], record_keys[record_id],
            row["semantic_bucket_id"], row["semantic_rank"], row["retrieval_eligible"],
            row["morgan_similarity"]))
    _store_counts(connection, query_id, level, mode, parents, grouped, selected)
    summary[f"{level}/{mode}/records"] += len(selected)
    summary[f"{level}/{mode}/parent_slots"] += len(parents)


def _store_counts(connection: sqlite3.Connection, query_id: int, level: str, mode: str,
                  parents: list[str], grouped: dict, selected: list[dict]) -> None:
    candidates = [row for parent in parents for row in grouped[parent]]
    selected_counts = Counter(row["parent_id"] for row in selected)
    for rank, parent in enumerate(parents, 1):
        rows = grouped[parent]
        connection.execute("INSERT INTO parent_assignments VALUES (?,?,?,?,?,?,?,?,?)", (
            query_id, level, mode, rank, parent, rows[0]["morgan_similarity"], len(rows),
            sum(row["retrieval_eligible"] is True for row in rows), selected_counts[parent]))
    mapped = sum(row["semantic_rank"] is not None for row in candidates)
    eligible = sum(row["retrieval_eligible"] is True for row in candidates)
    values = (query_id, level, mode, len(candidates), mapped, eligible, mapped - eligible,
              len(candidates) - mapped, len(parents), len(selected_counts), len(selected),
              TOTAL_RECORD_LIMIT - len(selected))
    connection.execute("INSERT INTO selection_counts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", values)


def _populate(output: sqlite3.Connection, source: sqlite3.Connection,
              binding: dict, task: str) -> Counter:
    source_queries = dict(source.execute(
        "SELECT benchmark_row_id,query_id FROM benchmark_queries"
    ))
    benchmarks = list(source.execute(
        """SELECT b.benchmark_row_id,b.drug,q.query_smiles FROM benchmark_queries b
           JOIN queries q USING(query_id) ORDER BY b.benchmark_row_id"""
    ))
    output.executemany("INSERT INTO queries VALUES (?,?)", (
        (query_id, row[2]) for query_id, row in enumerate(benchmarks)
    ))
    output.executemany("INSERT INTO benchmark_queries VALUES (?,?,?)", (
        (row[0], row[1], query_id) for query_id, row in enumerate(benchmarks)
    ))
    record_keys, summary = {}, Counter()
    for query_id, (benchmark, _, query_smiles) in enumerate(benchmarks):
        for level in LEVELS:
            rows = _query_rows(
                source, source_queries[benchmark], level, binding, task,
                str(benchmark), str(query_smiles),
            )
            parents, grouped = _parent_prefix(rows)
            for mode in MODES:
                _store_mode(output, int(query_id), level, mode, parents, grouped, record_keys, summary)
            summary[f"{level}/candidate_records"] += len(rows)
            summary[f"{level}/maximum_parent_prefix"] = max(
                summary[f"{level}/maximum_parent_prefix"], len(parents))
    summary["queries"], summary["unique_records"] = len(benchmarks), len(record_keys)
    return summary


def _verify_l2(task: str, subset: str, database: Path) -> dict[str, Any]:
    old_root = V4_ROOT / task / "scaffold" / subset
    old_manifest = old_root / "VERSION.json"
    old_database = old_root / json.loads(old_manifest.read_text())["database"]
    with sqlite3.connect(database) as new, sqlite3.connect(old_database) as old:
        new.row_factory = old.row_factory = sqlite3.Row
        for benchmark, new_id in new.execute("SELECT benchmark_row_id,query_id FROM benchmark_queries"):
            old_id = old.execute(
                "SELECT query_id FROM benchmark_queries WHERE benchmark_row_id=?", (benchmark,)
            ).fetchone()[0]
            for mode in MODES:
                current = [row[0] for row in new.execute(
                    """SELECT r.external_record_id FROM assignments a JOIN records r USING(record_key)
                       WHERE a.query_id=? AND a.level='L2' AND a.mode=? ORDER BY a.selection_rank""",
                    (new_id, mode))]
                frozen = [row[0] for row in old.execute(
                    """SELECT r.external_record_id FROM l2_assignments a JOIN l2_records r USING(record_key)
                       WHERE a.query_id=? AND a.mode=? ORDER BY a.selection_rank""", (old_id, mode))]
                if current != frozen:
                    raise ValueError(f"L2 selection differs from V4: {benchmark}/{mode}")
    return {"manifest": str(old_manifest.resolve()), "manifest_sha256": sha256_file(old_manifest),
            "database_sha256": sha256_file(old_database), "record_order_identical": True}


def build(task: str, subset: str = "valid", output_root: Path | None = None) -> dict:
    if task not in three_pools.MODULES or subset != "valid":
        raise ValueError("indirect cache supports BBB/oral validation only")
    destination = (output_root or OUTPUT_ROOT / task / "scaffold" / subset).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing cache: {destination}")
    binding, eligibility = _eligibility(task)
    source_path, source_manifest = _source(task, subset)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    database = temporary / "retrieval.sqlite3"
    try:
        with sqlite3.connect(database) as output, sqlite3.connect(
            f"file:{source_path.resolve()}?mode=ro&immutable=1", uri=True
        ) as source:
            _create_tables(output)
            summary = _populate(output, source, binding, task)
            content_id = _digest({"schema": SCHEMA_VERSION, "task": task,
                "source": source_manifest["cache_sha256"], "eligibility": eligibility["record_map_sha256"],
                "builder": sha256_file(Path(__file__)), "summary": dict(sorted(summary.items()))})
            output.executemany("INSERT INTO metadata VALUES (?,?)", (
                ("schema_version", SCHEMA_VERSION), ("content_id", content_id)))
            output.commit()
            output.execute("VACUUM")
            if output.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("indirect cache failed SQLite quick_check")
        l2_equivalence = _verify_l2(task, subset, database)
        manifest = _manifest(task, subset, database, source_path, source_manifest,
                             eligibility, l2_equivalence, summary, content_id)
        (temporary / "VERSION.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(database, 0o444)
        os.chmod(temporary / "VERSION.json", 0o444)
        temporary.rename(destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _manifest(task: str, subset: str, database: Path, source_path: Path,
              source: dict, eligibility: dict, l2_equivalence: dict,
              summary: Counter, content_id: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "status": "complete", "task_id": task,
        "subset": subset, "database": database.name, "content_id": content_id,
        "database_sha256": sha256_file(database), "levels_in_cache": list(LEVELS),
        "morgan_primary_parent_width": 100, "morgan_fallback_parent_width": 100,
        "morgan_candidate_parent_widths": [100], "tie_seed": 0,
        "capacities": {"parent_candidates": PARENT_LIMIT, "records_per_level": TOTAL_RECORD_LIMIT,
            "records_per_parent": PER_PARENT_LIMIT,
            "records_per_parent_semantic_bucket": PER_PARENT_BUCKET_LIMIT},
        "selection": {"candidate_pool": "all", "parent_order": "morgan_tanimoto_then_parent_id",
            "parent_selection": "minimal_prefix_filling_both_25_record_arms",
            "control_filter": "none", "semantic_filter": "retrieval_eligible_true_and_ranked",
            "semantic_order": "semantic_rank_then_parent_rank_then_bucket_then_stable_tie",
            "semantic_metadata_visible": False, "full_panel_required": True},
        "assignment_counts": dict(sorted(summary.items())), "eligibility": eligibility,
        "l2_v4_equivalence": l2_equivalence, "inputs": {
            "source_database": {"path": str(source_path.resolve()), "sha256": source["cache_sha256"]},
            "builder": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
            "selector": {"path": str(Path(select_global_molecule_records.__code__.co_filename).resolve()),
                         "sha256": sha256_file(Path(select_global_molecule_records.__code__.co_filename))}},
        "contains_direct_records": False, "neighbor_identity_policy": source["neighbor_identity_policy"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(three_pools.MODULES))
    parser.add_argument("--subset", default="valid", choices=("valid",))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.task, args.subset, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
