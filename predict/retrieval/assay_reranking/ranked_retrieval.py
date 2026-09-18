"""Fast runtime reader for split rankings joined to one task evidence index.

Publication performs chemistry, policy checks, ranking, and display adaptation.
Runtime reads only the split ``VERSION.json`` and indexed SQLite rows.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Mapping


SCHEMA_VERSION = "ranked_evidence_retrieval.v1"
POOLS = ("tool-accepted", "tool-compatible", "all")
FIXED_POOL = "fixed"


def _open_immutable(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)


def _paths(policy: Mapping[str, Any], task: str, subset: str) -> tuple[Path, Path, dict[str, Any]]:
    manifest_path = Path(policy["cache_manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "task_id": task,
        "subset": subset,
        "gold_release": "v1",
        "neighbor_identity_policy": "l1_voter_membership_later_parent_disjoint",
        "joint_materialized": False,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError(f"Incompatible ranked retrieval manifest: {manifest_path}")
    if manifest.get("pools") != list(POOLS):
        raise ValueError(f"Incompatible ranked retrieval pools: {manifest_path}")
    rankings = manifest_path.with_name(manifest["rankings_database"])
    evidence = (manifest_path.parent / manifest["evidence_database"]).resolve()
    if not rankings.is_file() or not evidence.is_file():
        raise ValueError(f"Missing ranked retrieval database: {manifest_path}")
    return rankings, evidence, manifest


def _record(row: Mapping[str, Any], method: str) -> dict[str, Any]:
    similarity = float(row["morgan_similarity"])
    if not math.isfinite(similarity) or not 0 <= similarity <= 1:
        raise ValueError("Invalid cached Morgan similarity")
    payload = json.loads(str(row["payload"]))
    parent_smiles = str(row["parent_smiles"])
    payload.setdefault("source_canonical_smiles", payload.get("canonical_smiles"))
    payload.setdefault("evidence_parent_smiles", payload.get("canonical_smiles"))
    payload["canonical_smiles"] = parent_smiles
    payload["reference_parent_smiles"] = parent_smiles
    result = {
        "record_id": str(row["external_record_id"]),
        "reference_molecule_id": str(row["parent_id"]),
        "reference_parent_smiles": parent_smiles,
        "morgan_similarity": similarity,
        "morgan_rank": int(row["morgan_rank"]),
        "assay_rank": int(row["assay_rank"]) if row["assay_rank"] is not None else None,
        "ranking_method": method,
        "payload": payload,
    }
    if row["context_id"] is not None:
        result["gold_context_id"] = str(row["context_id"])
    if method == "assay_transfer":
        score = row["assay_transfer_score"]
        if score is None or not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
            raise ValueError("Invalid cached assay-transfer score")
        result["transfer_likelihood"] = float(score)
    return result


def _rows(
    connection: sqlite3.Connection, query_id: int, pool: str, level: str,
    method: str, *, limit: int | None = None, parents: list[str] | None = None,
) -> list[dict[str, Any]]:
    where = "r.query_id=? AND r.pool=? AND r.level=?"
    values: list[Any] = [query_id, pool, level]
    if parents is not None:
        if not parents:
            return []
        where += f" AND r.parent_id IN ({','.join('?' for _ in parents)})"
        values.extend(parents)
    rank = "morgan_rank" if method == "morgan" else "assay_rank"
    suffix = ""
    if limit is not None:
        suffix = " LIMIT ?"
        values.append(limit)
    connection.row_factory = sqlite3.Row
    return [dict(row) for row in connection.execute(
        f"""SELECT r.parent_id,r.parent_smiles,r.context_id,r.within_parent_rank,
                   r.morgan_similarity,r.morgan_rank,r.assay_transfer_score,r.assay_rank,
                   e.external_record_id,e.payload
            FROM rankings AS r
            JOIN evidence.records AS e USING(source_row_uid)
            WHERE {where} ORDER BY r.{rank}{suffix}""",
        values,
    )]


def _ranked_parents(
    connection: sqlite3.Connection, query_id: int, method: str, limit: int,
) -> list[str]:
    rank = "morgan_rank" if method == "morgan" else "assay_rank"
    return [str(row[0]) for row in connection.execute(
        f"""SELECT parent_id FROM rankings
            WHERE query_id=? AND pool=? AND level='L1'
            GROUP BY parent_id ORDER BY MIN({rank}),parent_id LIMIT ?""",
        (query_id, FIXED_POOL, limit),
    )]


def _select_l1(
    connection: sqlite3.Connection, query_id: int, method: str,
    molecule_limit: int, record_limit: int,
    joint_panel_sizes: tuple[int, int] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    panel_ranks: dict[tuple[str, str], int] = {}
    if method == "joint":
        assay_size, morgan_size = joint_panel_sizes or (5, 5)
        if min(assay_size, morgan_size) < 1 or assay_size + morgan_size != molecule_limit:
            raise ValueError("Joint panel sizes must be positive and sum to the molecule limit")
        specs = (("assay_transfer", assay_size), ("morgan", morgan_size))
        parents = []
        for panel, size in specs:
            ranked = _ranked_parents(connection, query_id, panel, size)
            for rank, parent in enumerate(ranked, 1):
                panel_ranks[panel, parent] = rank
                if parent not in parents:
                    parents.append(parent)
    else:
        specs = ()
        parents = _ranked_parents(connection, query_id, method, molecule_limit)
    if len(parents) < (1 if method == "joint" else molecule_limit):
        raise ValueError("Insufficient cached L1 molecules")

    rows = _rows(connection, query_id, FIXED_POOL, "L1", "morgan", parents=parents)
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_parent[str(row["parent_id"])].append(row)
    count = connection.execute(
        "SELECT candidate_record_count,candidate_parent_count FROM selection_counts "
        "WHERE query_id=? AND pool=? AND level='L1'", (query_id, FIXED_POOL),
    ).fetchone()
    if count is None:
        raise ValueError("Cache lacks L1 selection counts")

    molecules = []
    for selection_rank, parent in enumerate(parents):
        available = sorted(
            by_parent[parent],
            key=lambda row: (int(row["within_parent_rank"]), str(row["external_record_id"])),
        )
        exemplar = available[0]
        ranking_method = "joint" if method == "joint" else method
        context_choices = connection.execute(
            "SELECT morgan_context_id,assay_context_id FROM l1_parent_contexts "
            "WHERE query_id=? AND parent_id=?", (query_id, parent),
        ).fetchone()
        if context_choices is None:
            raise ValueError(f"Cache lacks L1 context choice: {query_id}/{parent}")
        context_method = (
            "assay_transfer" if method == "assay_transfer"
            or (method == "joint" and ("assay_transfer", parent) in panel_ranks)
            else "morgan"
        )
        selected_context = str(context_choices[1 if context_method == "assay_transfer" else 0])
        context = connection.execute(
            "SELECT parent_id,condition_group,gold_label FROM evidence.contexts WHERE context_id=?",
            (selected_context,),
        ).fetchone()
        if context is None or str(context[0]) != parent:
            raise ValueError(f"Cache has incompatible L1 context choice: {selected_context}")
        molecule = {
            "reference_molecule_id": parent,
            "canonical_smiles": str(exemplar["parent_smiles"]),
            "morgan_similarity": float(exemplar["morgan_similarity"]),
            "morgan_rank": int(exemplar["morgan_rank"]),
            "assay_rank": int(exemplar["assay_rank"]),
            "selection_rank": selection_rank,
            "available_l1": len(available),
            "available_l2": 0,
            "ranking_method": ranking_method,
            "l2_records": [],
            "context_card_id": selected_context,
            "selected_condition": (
                "" if str(context[1]) == "no_reported_external_condition" else str(context[1])
            ),
            "_diagnostic_label": int(context[2]),
            "_diagnostic_context_id": selected_context,
            "_diagnostic_parent_id": parent,
            "gold_context_ids": {
                "morgan": str(context_choices[0]),
                "assay_transfer": str(context_choices[1]),
            },
            "l1_records": [_record(row, ranking_method) for row in available[:record_limit]],
        }
        if method in {"assay_transfer", "joint"}:
            molecule["transfer_likelihood"] = float(exemplar["assay_transfer_score"])
        if method == "joint":
            for panel, _ in specs:
                molecule[f"{panel}_panel_rank"] = panel_ranks.get((panel, parent), "not_selected_in_panel")
        molecules.append(molecule)
    return molecules, {
        "candidate_records": int(count[0]),
        "candidate_molecules": int(count[1]),
        "selected_records": sum(len(item["l1_records"]) for item in molecules),
        "selected_molecules": len(molecules),
    }


def load_candidates(
    queries: Mapping[str, str], *, task: str, subset: str, policy: Mapping[str, Any],
    molecule_limit: int = 10, l1_limit: int = 10, later_limit: int = 50,
    tie_seed: int = 0, cache_pool: str = "tool-accepted",
    joint_panel_sizes: tuple[int, int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return prompt-ready selections from the two compact immutable databases."""
    if not queries or min(molecule_limit, l1_limit, later_limit) < 1:
        raise ValueError("Queries and positive shape limits are required")
    if cache_pool not in POOLS or tie_seed != 0:
        raise ValueError("Unknown pool or non-frozen tie seed")
    rankings, evidence, manifest = _paths(policy, task, subset)
    capacities = manifest["capacities"]
    if molecule_limit > capacities["l1_molecules"] or l1_limit > capacities["l1_records_per_molecule"]:
        raise ValueError("Requested L1 shape exceeds cache capacity")
    if later_limit > capacities["later_records_per_level"]:
        raise ValueError("Requested later-level shape exceeds cache capacity")
    if policy["stages"]["L1"] == "joint" and molecule_limit != 10:
        raise ValueError("Joint requires ten panel slots")

    molecules_by_query: dict[str, Any] = {}
    later_by_query: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    with _open_immutable(rankings) as connection:
        connection.execute("ATTACH DATABASE ? AS evidence", (str(evidence),))
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        evidence_metadata = dict(connection.execute("SELECT key,value FROM evidence.metadata"))
        if (metadata.get("schema_version") != SCHEMA_VERSION
                or metadata.get("content_id") != manifest["content_id"]
                or evidence_metadata.get("content_id") != manifest["evidence_content_id"]):
            raise ValueError("Rankings, evidence, and manifest identities differ")
        for benchmark_row_id, drug in queries.items():
            registered = connection.execute(
                "SELECT query_id,drug,query_parent_id,query_parent_smiles FROM benchmark_queries "
                "JOIN queries USING(query_id) WHERE benchmark_row_id=?", (str(benchmark_row_id),),
            ).fetchone()
            if registered is None or str(registered[1]) != drug:
                raise ValueError(f"Query differs from frozen cache ledger: {benchmark_row_id}")
            query_id = int(registered[0])
            identities[str(benchmark_row_id)] = {
                "parent_id": str(registered[2]), "parent_smiles": str(registered[3]),
            }
            molecules, l1_audit = _select_l1(
                connection, query_id, policy["stages"]["L1"], molecule_limit,
                l1_limit, joint_panel_sizes,
            )
            seen = {row["record_id"] for molecule in molecules for row in molecule["l1_records"]}
            later: dict[str, Any] = {}
            level_audits = {"L1": l1_audit}
            for level, method in policy["stages"].items():
                if level == "L1":
                    continue
                pool = cache_pool
                count = connection.execute(
                    "SELECT candidate_record_count,candidate_parent_count FROM selection_counts "
                    "WHERE query_id=? AND pool=? AND level=?", (query_id, pool, level),
                ).fetchone()
                if count is None:
                    raise ValueError(f"Cache lacks selection counts: {benchmark_row_id}/{pool}/{level}")
                selected = _rows(connection, query_id, pool, level, method, limit=later_limit)
                overlap = {str(row["external_record_id"]) for row in selected} & seen
                if overlap:
                    raise ValueError(f"A cached record occurs in multiple levels: {sorted(overlap)[:5]}")
                records = [_record(row, method) for row in selected]
                seen.update(row["record_id"] for row in records)
                audit = {
                    "candidate_records": int(count[0]),
                    "candidate_molecules": int(count[1]),
                    "selected_records": len(records),
                    "selected_molecules": len({row["reference_molecule_id"] for row in records}),
                    "shortfall": later_limit - len(records),
                }
                level_audits[level] = audit
                later[level] = {
                    "records": records, "available_record_count": int(count[0]),
                    "allow_shortfall": True, "ranking_method": method, **audit,
                }
            molecules_by_query[str(benchmark_row_id)] = molecules
            later_by_query[str(benchmark_row_id)] = later
            audits[str(benchmark_row_id)] = level_audits
        connection.execute("DETACH DATABASE evidence")

    return molecules_by_query, later_by_query, {
        "selection_policy": SCHEMA_VERSION,
        "inputs": dict(policy.get("inputs") or {}),
        "contract": {
            "policy": dict(policy), "molecule_limit": molecule_limit,
            "l1_limit": l1_limit, "later_limit": later_limit,
            "tie_seed": tie_seed, "cache_pool": cache_pool,
            "joint_panel_sizes": list(joint_panel_sizes) if joint_panel_sizes else None,
            "cache_content_id": manifest["content_id"],
        },
        "cache_pool": cache_pool,
        "cache_version": str(Path(policy["cache_manifest"]).resolve()),
        "cache_content_id": manifest["content_id"],
        "cache_capacities": dict(capacities),
        "cache_assignment_counts": dict(manifest["assignment_counts"]),
        "cache_record_count": int(manifest["evidence_record_count"]),
        "query_identities": identities,
        "query_audits": audits,
        "neighbor_identity_policy": "l1_voter_membership_later_parent_disjoint",
        "similarity_floor": None,
        "scaffold_overlap": None,
        "parent_overlap": 0,
    }


__all__ = ["SCHEMA_VERSION", "load_candidates"]
