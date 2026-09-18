"""Fast, read-only access to self-contained cache-matched retrieval caches.

The cache stores source payloads and both native rankings. Runtime selection is
therefore SQLite-only: no Parquet reads, molecule normalization, fingerprints,
scaffold checks, database hashing, or integrity scan.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Mapping


SCHEMA_VERSION = "cache_matched_retrieval.v2"
POOLS = ("tool-accepted", "tool-compatible", "all")
FIXED_POOL = "fixed"
DATABASE_NAME = "retrieval.sqlite3"
ACTIVE_GOLD_RELEASES = {
    "bbb_martins": ("BBB_Martins", "v2"),
    "bioavailability_ma": ("Bioavailability_Ma", "v2"),
}


def _require_current_gold(manifest: Mapping[str, Any], task: str) -> None:
    """Reject caches pinned to a superseded benchmark without reading its files."""
    expected = ACTIVE_GOLD_RELEASES.get(task)
    if expected is None:
        return
    task_directory, release = expected
    declared = manifest.get("gold_release")
    query_path = str(
        ((manifest.get("inputs") or {}).get("query_ledger") or {}).get("path") or ""
    ).replace("\\", "/")
    path_declares_release = f"/{task_directory}/{release}/scaffold/" in query_path
    if declared != release and not path_declares_release:
        raise ValueError(
            f"Cache is stale for active {task_directory} {release}; rebuild it before retrieval"
        )


def _open_immutable(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)


def _manifest(policy: Mapping[str, Any], task: str, subset: str) -> tuple[Path, dict[str, Any]]:
    path = Path(policy["cache_manifest"]).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "task_id": task,
        "subset": subset,
        "tie_seed": 0,
        "joint_materialized": False,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError(f"Incompatible cache-matched v2 manifest: {path}")
    _require_current_gold(manifest, task)
    if manifest.get("pools") != list(POOLS):
        raise ValueError(f"Incompatible cache pools: {path}")
    database = path.with_name(manifest.get("database", DATABASE_NAME))
    if not database.is_file():
        raise ValueError(f"Missing cache database: {database}")
    return database, manifest


def _payload(raw: str, parent_smiles: str) -> dict[str, Any]:
    payload = json.loads(raw)
    cached_source_smiles = payload.get("canonical_smiles")
    payload.setdefault("source_canonical_smiles", cached_source_smiles)
    payload["canonical_smiles"] = parent_smiles
    payload["reference_parent_smiles"] = parent_smiles
    return payload


def _record(row: Mapping[str, Any], method: str) -> dict[str, Any]:
    similarity = float(row["morgan_similarity"])
    if not math.isfinite(similarity) or not 0 <= similarity <= 1:
        raise ValueError("Invalid cached Morgan similarity")
    result = {
        "record_id": str(row["external_record_id"]),
        "reference_molecule_id": str(row["parent_id"]),
        "reference_parent_smiles": str(row["parent_smiles"]),
        "morgan_similarity": similarity,
        "morgan_rank": int(row["morgan_rank"]),
        "assay_rank": int(row["assay_rank"]) if row["assay_rank"] is not None else None,
        "ranking_method": method,
        "payload": _payload(str(row["payload"]), str(row["parent_smiles"])),
    }
    if method == "assay_transfer":
        score = row["assay_transfer_score"]
        if score is None or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid cached assay-transfer score")
        result["transfer_likelihood"] = float(score)
    return result


def _rows(
    connection: sqlite3.Connection, query_id: int, pool: str, level: str, *,
    method: str | None = None, limit: int | None = None,
    parents: list[str] | None = None,
) -> list[dict[str, Any]]:
    connection.row_factory = sqlite3.Row
    where = "a.query_id=? AND a.pool=? AND a.level=?"
    values: list[Any] = [query_id, pool, level]
    if parents is not None:
        if not parents:
            return []
        where += f" AND a.parent_id IN ({','.join('?' for _ in parents)})"
        values.extend(parents)
    order = ""
    if method is not None:
        column = "morgan_rank" if method == "morgan" else "assay_rank"
        order = f" ORDER BY a.{column},r.external_record_id"
    suffix = ""
    if limit is not None:
        suffix = " LIMIT ?"
        values.append(limit)
    return [dict(row) for row in connection.execute(
        f"""SELECT a.parent_id,a.parent_smiles,a.morgan_similarity,a.morgan_rank,
                   a.assay_rank,a.assay_transfer_score,a.within_parent_rank,
                   r.external_record_id,r.payload
            FROM assignments AS a JOIN records AS r USING(record_key)
            WHERE {where}{order}{suffix}""",
        values,
    )]


def _select_l1(
    connection: sqlite3.Connection, query_id: int, method: str,
    molecule_limit: int, record_limit: int,
    *, joint_panel_sizes: tuple[int, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def ranked_parents(panel: str, limit: int) -> list[str]:
        column = "morgan_rank" if panel == "morgan" else "assay_rank"
        return [str(row[0]) for row in connection.execute(
            f"""SELECT parent_id FROM assignments
                WHERE query_id=? AND pool=? AND level='L1'
                GROUP BY parent_id ORDER BY MIN({column}),parent_id LIMIT ?""",
            (query_id, FIXED_POOL, limit),
        )]

    if method == "joint":
        if joint_panel_sizes is None:
            panel_specs = (("morgan", 5), ("assay_transfer", 5))
            rank_suffix = "top5_rank"
            missing_rank = "not_selected_in_top5"
        else:
            assay_size, morgan_size = joint_panel_sizes
            if min(assay_size, morgan_size) < 1 or assay_size + morgan_size != molecule_limit:
                raise ValueError("Joint panel sizes must be positive and sum to the molecule limit")
            panel_specs = (("assay_transfer", assay_size), ("morgan", morgan_size))
            rank_suffix = "panel_rank"
            missing_rank = "not_selected_in_panel"
        panels: dict[str, int] = {}
        ordered_parents: list[str] = []
        for panel, panel_size in panel_specs:
            ranked = ranked_parents(panel, panel_size)
            for rank, parent in enumerate(ranked, 1):
                panels[f"{panel}:{parent}"] = rank
                if parent not in ordered_parents:
                    ordered_parents.append(parent)
    else:
        ordered_parents = ranked_parents(method, molecule_limit)
        panels = {}
    if len(ordered_parents) < (1 if method == "joint" else molecule_limit):
        raise ValueError("Insufficient cached L1 molecules")

    rows = _rows(
        connection, query_id, FIXED_POOL, "L1", method="morgan",
        parents=ordered_parents,
    )
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_parent[str(row["parent_id"])].append(row)
    counts = connection.execute(
        """SELECT candidate_record_count,candidate_parent_count FROM selection_counts
           WHERE query_id=? AND pool=? AND level='L1'""",
        (query_id, FIXED_POOL),
    ).fetchone()
    if counts is None:
        raise ValueError("Cache lacks L1 selection counts")
    candidate_records, candidate_molecules = counts

    molecules = []
    selected_records = 0
    for selection_rank, parent in enumerate(ordered_parents):
        exemplar = by_parent[parent][0]
        cards = sorted(
            by_parent[parent],
            key=lambda row: (int(row["within_parent_rank"]), str(row["external_record_id"])),
        )[:record_limit]
        ranking_method = "joint" if method == "joint" else method
        molecule = {
            "reference_molecule_id": parent,
            "canonical_smiles": str(exemplar["parent_smiles"]),
            "morgan_similarity": float(exemplar["morgan_similarity"]),
            "morgan_rank": int(exemplar["morgan_rank"]),
            "assay_rank": int(exemplar["assay_rank"]),
            "selection_rank": selection_rank,
            "available_l1": len(by_parent[parent]),
            "available_l2": 0,
            "ranking_method": ranking_method,
            "l2_records": [],
            "gold_context_ids": {},
            "l1_records": [_record(row, ranking_method) for row in cards],
        }
        score = exemplar["assay_transfer_score"]
        if method in {"assay_transfer", "joint"}:
            if score is None or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid cached L1 assay-transfer score")
            molecule["transfer_likelihood"] = float(score)
        if method == "joint":
            for panel, _ in panel_specs:
                molecule[f"{panel}_{rank_suffix}"] = panels.get(
                    f"{panel}:{parent}", missing_rank
                )
        selected_records += len(cards)
        molecules.append(molecule)
    audit = {
        "candidate_molecules": candidate_molecules,
        "candidate_records": candidate_records,
        "selected_molecules": len(molecules),
        "selected_records": selected_records,
    }
    if method == "joint":
        audit["joint_panel_order"] = [panel for panel, _ in panel_specs]
        audit["joint_panel_sizes"] = {
            panel: size for panel, size in panel_specs
        }
        audit["joint_overlap_molecules"] = sum(
            1 for parent in ordered_parents
            if all(f"{panel}:{parent}" in panels for panel, _ in panel_specs)
        )
        audit["joint_overlap_refill"] = False
    return molecules, audit


def load_candidates(
    queries: Mapping[str, str], *, task: str, subset: str, policy: Mapping[str, Any],
    molecule_limit: int = 10, l1_limit: int = 10, later_limit: int = 50,
    tie_seed: int = 0, cache_pool: str = "tool-accepted",
    joint_panel_sizes: tuple[int, int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Select cached records using only indexed reads from one immutable database."""
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
                pool = FIXED_POOL if level == "L5" else cache_pool
                counts = connection.execute(
                    """SELECT candidate_record_count,candidate_parent_count
                       FROM selection_counts WHERE query_id=? AND pool=? AND level=?""",
                    (query_id, pool, level),
                ).fetchone()
                if counts is None:
                    raise ValueError(f"Cache lacks selection counts: {benchmark_row_id}/{pool}/{level}")
                candidate_records, candidate_molecules = counts
                chosen = _rows(
                    connection, query_id, pool, level,
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
                    "selected_molecules": len({row["reference_molecule_id"] for row in records}),
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
