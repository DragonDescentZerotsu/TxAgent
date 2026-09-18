"""Read pooled-L5 cache-matched retrieval from one immutable SQLite database.

V3 retains the V2 tables and L1 selection, but every later level, including
Morgan-only L5, uses the caller's selected record pool. Runtime performs indexed
SQLite reads only; cache publication owns chemistry and source validation.
"""
from __future__ import annotations

from pathlib import Path
import json
import math
from typing import Any, Mapping

from .cache_matched_v2 import (
    DATABASE_NAME,
    FIXED_POOL,
    POOLS,
    _open_immutable,
    _record,
    _require_current_gold,
    _rows,
    _select_l1,
)


SCHEMA_VERSION = "cache_matched_retrieval.v3"


def _manifest(
    policy: Mapping[str, Any], task: str, subset: str,
) -> tuple[Path, dict[str, Any]]:
    path = Path(policy["cache_manifest"]).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "task_id": task,
        "subset": subset,
        "tie_seed": 0,
        "joint_materialized": False,
        "fixed_pool_levels": ["L1"],
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError(f"Incompatible cache-matched v3 manifest: {path}")
    _require_current_gold(manifest, task)
    if manifest.get("pools") != list(POOLS):
        raise ValueError(f"Incompatible cache pools: {path}")
    database = path.with_name(manifest.get("database", DATABASE_NAME))
    if not database.is_file():
        raise ValueError(f"Missing cache database: {database}")
    return database, manifest


def load_candidates(
    queries: Mapping[str, str], *, task: str, subset: str, policy: Mapping[str, Any],
    molecule_limit: int = 10, l1_limit: int = 10, later_limit: int = 50,
    tie_seed: int = 0, cache_pool: str = "tool-accepted",
    joint_panel_sizes: tuple[int, int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Select fixed L1 and pool-specific later records with indexed reads only."""
    if not queries or min(molecule_limit, l1_limit, later_limit) < 1:
        raise ValueError("Queries and positive shape limits are required")
    if cache_pool not in POOLS:
        raise ValueError(f"Unknown cache pool: {cache_pool}")
    database, manifest = _manifest(policy, task, subset)
    capacities = manifest.get("capacities") or {}
    if tie_seed != manifest["tie_seed"]:
        raise ValueError("The cache fixes ranking tie seed 0")
    if molecule_limit > capacities.get("l1_molecules", 0):
        raise ValueError("Requested L1 molecule count exceeds cache capacity")
    if l1_limit > capacities.get("l1_records_per_molecule", 0):
        raise ValueError("Requested L1 record count exceeds cache capacity")
    if later_limit > capacities.get("later_records_per_level", 0):
        raise ValueError("Requested later-level record count exceeds cache capacity")
    if policy["stages"]["L1"] == "joint" and molecule_limit != 10:
        raise ValueError("Joint requires ten panel slots")
    if policy["stages"]["L1"] != "joint" and joint_panel_sizes is not None:
        raise ValueError("Joint panel sizes require joint L1 selection")

    molecules_by_query: dict[str, Any] = {}
    later_by_query: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    query_identities: dict[str, Any] = {}
    with _open_immutable(database) as connection:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if (metadata.get("schema_version") != SCHEMA_VERSION
                or metadata.get("content_id") != manifest.get("content_id")):
            raise ValueError("Cache database and manifest identity differ")
        for benchmark_row_id, drug in queries.items():
            registered = connection.execute(
                """SELECT b.query_id,b.drug,q.query_parent_id,q.query_parent_smiles
                   FROM benchmark_queries AS b JOIN queries AS q USING(query_id)
                   WHERE b.benchmark_row_id=?""",
                (str(benchmark_row_id),),
            ).fetchone()
            if registered is None or registered[1] != drug:
                raise ValueError(f"Query differs from frozen cache ledger: {benchmark_row_id}")
            query_id = int(registered[0])
            query_identities[str(benchmark_row_id)] = {
                "parent_id": str(registered[2]),
                "parent_smiles": str(registered[3]),
            }
            molecules, l1_audit = _select_l1(
                connection, query_id, policy["stages"]["L1"],
                molecule_limit, l1_limit, joint_panel_sizes=joint_panel_sizes,
            )
            seen = {
                row["record_id"]
                for molecule in molecules
                for row in molecule["l1_records"]
            }
            level_audits = {"L1": l1_audit}
            later = {}
            for level, method in policy["stages"].items():
                if level == "L1":
                    continue
                counts = connection.execute(
                    """SELECT candidate_record_count,candidate_parent_count
                       FROM selection_counts WHERE query_id=? AND pool=? AND level=?""",
                    (query_id, cache_pool, level),
                ).fetchone()
                if counts is None:
                    raise ValueError(
                        f"Cache lacks selection counts: {benchmark_row_id}/{cache_pool}/{level}"
                    )
                candidate_records, candidate_molecules = counts
                chosen = _rows(
                    connection, query_id, cache_pool, level,
                    method=method, limit=later_limit,
                )
                overlap = {str(row["external_record_id"]) for row in chosen} & seen
                if overlap:
                    raise ValueError(f"A cached record occurs in multiple levels: {sorted(overlap)}")
                records = [_record(row, method) for row in chosen]
                seen.update(row["record_id"] for row in records)
                level_audits[level] = {
                    "candidate_records": candidate_records,
                    "candidate_molecules": candidate_molecules,
                    "selected_records": len(records),
                    "selected_molecules": len({
                        row["reference_molecule_id"] for row in records
                    }),
                    "shortfall": later_limit - len(records),
                }
                later[level] = {
                    "records": records,
                    "available_record_count": candidate_records,
                    "allow_shortfall": True,
                    "ranking_method": method,
                    **level_audits[level],
                }
            molecules_by_query[str(benchmark_row_id)] = molecules
            later_by_query[str(benchmark_row_id)] = later
            audits[str(benchmark_row_id)] = level_audits

    contract = {
        "policy": dict(policy),
        "molecule_limit": molecule_limit,
        "l1_limit": l1_limit,
        "later_limit": later_limit,
        "tie_seed": tie_seed,
        "cache_pool": cache_pool,
        "joint_panel_sizes": list(joint_panel_sizes) if joint_panel_sizes else None,
        "cache_content_id": manifest["content_id"],
    }
    return molecules_by_query, later_by_query, {
        "selection_policy": SCHEMA_VERSION,
        "inputs": dict(policy.get("inputs") or {}),
        "contract": contract,
        "cache_pool": cache_pool,
        "cache_version": str(Path(policy["cache_manifest"]).resolve()),
        "cache_content_id": manifest["content_id"],
        "cache_capacities": dict(manifest["capacities"]),
        "cache_assignment_counts": dict(manifest["assignment_counts"]),
        "cache_record_count": int(manifest["record_count"]),
        "query_identities": query_identities,
        "query_audits": audits,
        "neighbor_identity_policy": "scaffold_disjoint",
        "similarity_floor": None,
        "scaffold_overlap": 0,
        "parent_overlap": 0,
    }
