"""Read the immutable semantic/pair-bucket-balanced retrieval cache."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .cache_matched_v2 import DATABASE_NAME, FIXED_POOL, _open_immutable, _record, _rows, _select_l1


SCHEMA_VERSION = "semantic_bucket_reranking.v1"


def _manifest(policy: Mapping[str, Any], task: str, subset: str) -> tuple[Path, dict[str, Any]]:
    path = Path(policy["cache_manifest"]).resolve()
    manifest = json.loads(path.read_text())
    required = {
        "schema_version": SCHEMA_VERSION, "status": "complete", "task_id": task,
        "subset": subset, "tie_seed": 0, "joint_materialized": False,
        "pools": ["all"], "fixed_pool_levels": ["L1"],
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError(f"Incompatible semantic-bucket cache manifest: {path}")
    database = path.with_name(manifest.get("database", DATABASE_NAME))
    if not database.is_file():
        raise ValueError(f"Missing semantic-bucket cache database: {database}")
    return database, manifest


def load_candidates(
    queries: Mapping[str, str], *, task: str, subset: str, policy: Mapping[str, Any],
    molecule_limit: int = 10, l1_limit: int = 10, later_limit: int = 50,
    tie_seed: int = 0, cache_pool: str = "all",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not queries or min(molecule_limit, l1_limit, later_limit) < 1:
        raise ValueError("Queries and positive shape limits are required")
    if cache_pool != "all":
        raise ValueError("Semantic-bucket retrieval has one fixed filtered all pool")
    if policy.get("reranking") not in {"morgan", "assay-transfer"}:
        raise ValueError("Semantic-bucket retrieval supports morgan or assay-transfer only")
    database, manifest = _manifest(policy, task, subset)
    capacities = manifest["capacities"]
    if tie_seed != 0 or molecule_limit > capacities["l1_molecules"] \
            or l1_limit > capacities["l1_records_per_molecule"] \
            or later_limit > capacities["later_records_per_level"]:
        raise ValueError("Requested shape exceeds the frozen cache contract")

    molecules_by_query, later_by_query, audits, identities = {}, {}, {}, {}
    with _open_immutable(database) as connection:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if metadata.get("schema_version") != SCHEMA_VERSION \
                or metadata.get("content_id") != manifest["content_id"]:
            raise ValueError("Cache database and manifest identity differ")
        for benchmark_row_id, drug in queries.items():
            registered = connection.execute(
                """SELECT b.query_id,b.drug,q.query_parent_id,q.query_parent_smiles
                   FROM benchmark_queries AS b JOIN queries AS q USING(query_id)
                   WHERE b.benchmark_row_id=?""", (str(benchmark_row_id),),
            ).fetchone()
            if registered is None or registered[1] != drug:
                raise ValueError(f"Query differs from frozen cache ledger: {benchmark_row_id}")
            query_id = int(registered[0])
            identities[str(benchmark_row_id)] = {
                "parent_id": str(registered[2]), "parent_smiles": str(registered[3]),
            }
            l1_method = policy["stages"]["L1"]
            molecules, l1_audit = _select_l1(
                connection, query_id, l1_method, molecule_limit, l1_limit,
            )
            seen = {row["record_id"] for molecule in molecules for row in molecule["l1_records"]}
            level_audits, later = {"L1": l1_audit}, {}
            for level, method in policy["stages"].items():
                if level == "L1":
                    continue
                counts = connection.execute(
                    """SELECT candidate_record_count,candidate_parent_count
                       FROM selection_counts WHERE query_id=? AND pool='all' AND level=?""",
                    (query_id, level),
                ).fetchone()
                if counts is None:
                    raise ValueError(f"Cache lacks {benchmark_row_id}/{level}")
                chosen = _rows(connection, query_id, "all", level, method=method, limit=later_limit)
                overlap = {str(row["external_record_id"]) for row in chosen} & seen
                if overlap:
                    raise ValueError(f"A cached record occurs in multiple levels: {sorted(overlap)}")
                records = [_record(row, method) for row in chosen]
                seen.update(row["record_id"] for row in records)
                level_audits[level] = {
                    "candidate_records": int(counts[0]), "candidate_molecules": int(counts[1]),
                    "selected_records": len(records),
                    "selected_molecules": len({row["reference_molecule_id"] for row in records}),
                    "shortfall": later_limit - len(records),
                }
                later[level] = {
                    "records": records, "available_record_count": int(counts[0]),
                    "allow_shortfall": True, "ranking_method": method, **level_audits[level],
                }
            molecules_by_query[str(benchmark_row_id)] = molecules
            later_by_query[str(benchmark_row_id)] = later
            audits[str(benchmark_row_id)] = level_audits
    contract = {
        "policy": dict(policy), "molecule_limit": molecule_limit, "l1_limit": l1_limit,
        "later_limit": later_limit, "tie_seed": tie_seed, "cache_pool": cache_pool,
        "cache_content_id": manifest["content_id"],
    }
    return molecules_by_query, later_by_query, {
        "selection_policy": SCHEMA_VERSION, "inputs": dict(policy.get("inputs") or {}),
        "contract": contract, "cache_pool": cache_pool,
        "cache_version": str(Path(policy["cache_manifest"]).resolve()),
        "cache_content_id": manifest["content_id"], "cache_capacities": dict(capacities),
        "cache_assignment_counts": dict(manifest["assignment_counts"]),
        "cache_record_count": int(manifest["record_count"]), "query_identities": identities,
        "query_audits": audits, "neighbor_identity_policy": "scaffold_disjoint",
        "similarity_floor": None, "scaffold_overlap": 0, "parent_overlap": 0,
    }
