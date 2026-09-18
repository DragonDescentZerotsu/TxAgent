"""Read immutable direct-free L2-L4 Morgan/semantic selections for prompts."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping

from predict.utils.json import sha256_file

from .cache_matched_v2 import _open_immutable, _payload


SCHEMA_V1 = "indirect_morgan_semantic_l2_l4.v1"
SCHEMA_V2 = "indirect_morgan_semantic_l2_l4.v2"
SCHEMA_V3 = "indirect_morgan_semantic_l2_l4.v3"


def _expected_contract(schema: str) -> dict[str, Any]:
    if schema == SCHEMA_V1:
        return {"parent_candidates": 100, "records_per_level": 25,
                "records_per_parent": 10,
                "records_per_parent_semantic_bucket": 5}
    if schema == SCHEMA_V2:
        return {"parent_candidates": 100, "control_records_per_level": 25,
                "semantic_candidates_per_level": 100,
                "semantic_records_per_parent": 15,
                "semantic_records_per_parent_semantic_bucket": 8,
                "final_records_min": 10, "final_records_max": 25}
    if schema == SCHEMA_V3:
        return {"control_records_per_level": 25,
                "semantic_records_per_level": 25,
                "llm_semantic_candidates_per_level": 100,
                "records_per_parent": 10}
    raise ValueError(f"unsupported indirect cache schema: {schema}")


def _manifest(policy: Mapping[str, Any], task: str, subset: str) -> tuple[Path, dict]:
    path = Path(policy["cache_manifest"]).resolve()
    manifest = json.loads(path.read_text())
    schema = str(policy.get("selection_contract") or "")
    required = {"schema_version": schema, "status": "complete",
                "task_id": task, "subset": subset, "tie_seed": 0,
                "contains_direct_records": False}
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError(f"incompatible indirect-only cache manifest: {path}")
    capacities = manifest.get("capacities") or {}
    expected_capacities = _expected_contract(schema)
    selection = manifest.get("selection") or {}
    if (manifest.get("levels_in_cache") != ["L2", "L3", "L4"]
            or capacities != expected_capacities
            or selection.get("candidate_pool") != "all"
            or selection.get("full_panel_required") is not True
            or selection.get("semantic_metadata_visible") is not False):
        raise ValueError(f"invalid indirect-only cache contract: {path}")
    database = path.with_name(manifest["database"])
    if (not database.is_file()
            or sha256_file(database) != manifest.get("database_sha256")):
        raise ValueError(f"missing or changed indirect-only cache database: {database}")
    return database, manifest


def _selected_records(connection: sqlite3.Connection, query_id: int,
                      level: str, mode: str) -> list[dict]:
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """SELECT a.*,r.external_record_id,r.parent_id,r.parent_smiles,r.payload
           FROM assignments a JOIN records r USING(record_key)
           WHERE a.query_id=? AND a.level=? AND a.mode=? ORDER BY a.selection_rank""",
        (query_id, level, mode),
    ).fetchall()
    return [{"record_id": str(row["external_record_id"]),
             "reference_molecule_id": str(row["parent_id"]),
             "reference_parent_smiles": str(row["parent_smiles"]),
             "morgan_similarity": float(row["morgan_similarity"]),
             "morgan_rank": int(row["parent_rank"]), "assay_rank": None,
            "ranking_method": mode,
            "_semantic_bucket_id": row["semantic_bucket_id"],
            "_semantic_rank": row["semantic_rank"],
             "payload": _payload(str(row["payload"]), str(row["parent_smiles"]))}
            for row in rows]


def _level_audit(connection: sqlite3.Connection, query_id: int,
                 level: str, mode: str, records: list[dict]) -> dict:
    counts = connection.execute(
        "SELECT * FROM selection_counts WHERE query_id=? AND level=? AND mode=?",
        (query_id, level, mode),
    ).fetchone()
    available = int(counts["candidate_records" if mode.endswith("control") else "eligible_records"])
    return {"available_record_count": available, "selected_records": len(records),
            "candidate_records": int(counts["candidate_records"]),
            "mapped_records": int(counts["mapped_records"]),
            "eligible_records": int(counts["eligible_records"]),
            "excluded_records": int(counts["excluded_bad_records"]),
            "unmapped_records": int(counts["unmapped_records"]),
            "candidate_molecules": int(counts["candidate_parents"]),
            "selected_molecules": len({row["reference_molecule_id"] for row in records}),
            "selected_parent_panel": int(counts["candidate_parents"]),
            "shortfall": int(counts["shortfall"])}


def _v1_semantic_ids(manifest: Mapping[str, Any], benchmark: str, level: str) -> list[str]:
    manifest_path = Path(manifest["v1_control_equivalence"]["manifest"])
    old_manifest = json.loads(manifest_path.read_text())
    database = manifest_path.with_name(old_manifest["database"])
    with _open_immutable(database) as connection:
        row = connection.execute(
            "SELECT query_id FROM benchmark_queries WHERE benchmark_row_id=?", (benchmark,)
        ).fetchone()
        if row is None:
            raise ValueError(f"query absent from V1 reference cache: {benchmark}")
        return [item["record_id"] for item in _selected_records(
            connection, int(row[0]), level, "morgan_parent_semantic"
        )]


def load_candidates(queries: Mapping[str, str], *, task: str, subset: str,
                    policy: Mapping[str, Any], later_limit: int = 25,
                    tie_seed: int = 0, cache_pool: str = "all", **_: Any):
    """Return no direct contexts and one independently selected indirect level."""
    mode = next(iter(policy.get("stages", {}).values()), "")
    schema = str(policy.get("selection_contract") or "")
    llm_semantic = schema == SCHEMA_V3 and mode == "morgan_parent_llm_semantic"
    semantic_panel = schema in {SCHEMA_V2, SCHEMA_V3} and mode.endswith("semantic")
    expected = 100 if (llm_semantic or schema == SCHEMA_V2 and semantic_panel) else 25
    if not queries or later_limit != expected or tie_seed != 0 or cache_pool != "all":
        raise ValueError(
            f"indirect-only retrieval requires queries, all pool, seed 0, and {expected} records"
        )
    if len(policy["stages"]) != 1:
        raise ValueError("indirect-only retrieval requires exactly one selected level")
    level, mode = next(iter(policy["stages"].items()))
    database, manifest = _manifest(policy, task, subset)
    direct, later, audits, identities = {}, {}, {}, {}
    with _open_immutable(database) as connection:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if metadata != {"schema_version": schema, "content_id": manifest["content_id"]}:
            raise ValueError("indirect-only database and manifest identity differ")
        for benchmark, drug in queries.items():
            row = connection.execute(
                """SELECT b.query_id,q.query_parent_smiles FROM benchmark_queries b
                   JOIN queries q USING(query_id) WHERE b.benchmark_row_id=? AND b.drug=?""",
                (benchmark, drug),
            ).fetchone()
            if row is None:
                raise ValueError(f"query differs from frozen indirect cache: {benchmark}")
            records = _selected_records(connection, int(row[0]), level, mode)
            if len(records) != later_limit:
                raise ValueError(f"{benchmark}/{level}/{mode} lacks {later_limit} records")
            audit = _level_audit(connection, int(row[0]), level, mode, records)
            direct[str(benchmark)] = []
            control_ids = (
                [item["record_id"] for item in _selected_records(
                    connection, int(row[0]), level, "morgan_parent_control"
                )]
                if semantic_panel else []
            )
            later[str(benchmark)] = {level: {"records": records,
                "available_record_count": audit["available_record_count"],
                "matched_control_record_ids": control_ids,
                "original_semantic_record_ids": (
                    _v1_semantic_ids(manifest, str(benchmark), level)
                    if schema == SCHEMA_V2 and mode.endswith("semantic") else []
                ),
                "allow_shortfall": False, "ranking_method": mode, **audit}}
            audits[str(benchmark)] = {level: audit}
            identities[str(benchmark)] = {"parent_smiles": str(row[1])}
    manifest_path = Path(policy["cache_manifest"]).resolve()
    inputs = dict(policy.get("inputs") or {})
    inputs[str(Path(__file__).resolve())] = sha256_file(Path(__file__))
    inputs[str(manifest_path)] = sha256_file(manifest_path)
    inputs[str(database)] = manifest["database_sha256"]
    return direct, later, {"selection_policy": schema, "inputs": inputs,
        "contract": {"policy": dict(policy), "later_limit": later_limit, "cache_pool": cache_pool},
        "cache_version": str(Path(policy["cache_manifest"]).resolve()),
        "cache_content_id": manifest["content_id"], "cache_capacities": manifest["capacities"],
        "cache_assignment_counts": manifest["assignment_counts"], "query_identities": identities,
        "query_audits": audits, "neighbor_identity_policy": manifest["neighbor_identity_policy"],
        "similarity_floor": None, "scaffold_overlap": 0, "parent_overlap": 0}
