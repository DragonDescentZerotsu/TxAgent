"""SQLite-only selection of assay-ranked gold condition contexts at L1."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Mapping

from predict.utils.json import sha256_file

from .cache_matched_v2 import _open_immutable, _payload


SCHEMA_VERSION = "l1_context_retrieval.v1"
ACTIVE_CONTEXT_SCHEMA_VERSION = "l1_context_retrieval.v2"
SEMANTIC_L2_SCHEMA_VERSION = "l1_context_semantic_l2.v1"
SEMANTIC_WEIGHTED_L2_SCHEMA_VERSION = "l1_context_semantic_weighted_l2.v1"
MORGAN_SEMANTIC_L2_SCHEMA_VERSION = "l1_context_morgan_semantic_l2.v1"
MORGAN_SEMANTIC_L2_V2_SCHEMA_VERSION = "l1_context_morgan_semantic_l2.v2"
MORGAN_SEMANTIC_L2_V3_SCHEMA_VERSION = "l1_context_morgan_semantic_l2.v3"
MORGAN_SEMANTIC_L2_V4_SCHEMA_VERSION = "l1_context_morgan_semantic_l2.v4"
DATABASE_NAME = "retrieval.sqlite3"
METHODS = {
    "morgan", "morgan_contrastive", "assay_transfer",
    "assay_transfer_within_morgan",
    "assay_transfer_contrastive",
}


def _manifest(
    policy: Mapping[str, Any], task: str, subset: str
) -> tuple[Path, dict[str, Any]]:
    path = Path(policy["cache_manifest"]).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    schema_version = str(policy.get("selection_contract") or SCHEMA_VERSION)
    required = {
        "schema_version": schema_version,
        "status": "complete",
        "task_id": task,
        "subset": subset,
        "tie_seed": 0,
        "morgan_fallback_parent_width": 100,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError(f"Incompatible L1 context cache manifest: {path}")
    primary_width = manifest.get("morgan_primary_parent_width")
    fallback_width = manifest["morgan_fallback_parent_width"]
    candidate_widths = manifest.get(
        "morgan_candidate_parent_widths", sorted({primary_width, fallback_width})
    )
    supported_widths = (
        {10, 15, 25, 50, 100}
        if schema_version in {
            ACTIVE_CONTEXT_SCHEMA_VERSION, MORGAN_SEMANTIC_L2_V2_SCHEMA_VERSION,
            MORGAN_SEMANTIC_L2_V3_SCHEMA_VERSION, MORGAN_SEMANTIC_L2_V4_SCHEMA_VERSION,
        }
        else {15, 25, 100}
    )
    if (primary_width not in supported_widths
            or not isinstance(candidate_widths, list)
            or not candidate_widths
            or not all(isinstance(width, int) for width in candidate_widths)
            or candidate_widths != sorted(set(candidate_widths))
            or candidate_widths[0] != primary_width
            or candidate_widths[-1] != fallback_width):
        raise ValueError(f"Unsupported L1 context Morgan width: {path}")
    if (schema_version == MORGAN_SEMANTIC_L2_V4_SCHEMA_VERSION
            and manifest.get("selection", {}).get(
                "semantic_records_per_parent_bucket"
            ) != 5):
        raise ValueError(f"Morgan-semantic L2 V4 requires its five-record cell cap: {path}")
    database = path.with_name(manifest.get("database", DATABASE_NAME))
    if not database.is_file():
        raise ValueError(f"Missing cache database: {database}")
    return database, manifest


def _l2_records(
    connection: sqlite3.Connection, query_id: int, *, mode: str, limit: int,
) -> list[dict[str, Any]]:
    if mode not in {"morgan", "semantic_lap", "semantic_weighted"}:
        raise ValueError(f"unsupported context-L2 mode: {mode}")
    rows = connection.execute(
        """SELECT a.*,r.external_record_id,r.parent_id,r.parent_smiles,r.payload
           FROM l2_assignments AS a JOIN l2_records AS r USING(record_key)
           WHERE a.query_id=? AND a.mode=? ORDER BY a.selection_rank""",
        (query_id, mode),
    ).fetchall()
    if len(rows) < limit:
        raise ValueError(f"context-L2 cache has only {len(rows)} of {limit} required records")
    output = []
    for row in rows[:limit]:
        item = {
            "record_id": str(row["external_record_id"]),
            "reference_molecule_id": str(row["parent_id"]),
            "reference_parent_smiles": str(row["parent_smiles"]),
            "morgan_similarity": float(row["morgan_similarity"]),
            "morgan_rank": int(row["selection_rank"]),
            "assay_rank": None,
            "ranking_method": mode,
            "payload": _payload(str(row["payload"]), str(row["parent_smiles"])),
        }
        if mode == "semantic_lap":
            item.update(
                semantic_bucket_id=str(row["semantic_bucket_id"]),
                semantic_rank=int(row["semantic_rank"]),
                semantic_lap=int(row["semantic_lap"]),
            )
        elif mode == "semantic_weighted":
            item.update(
                semantic_bucket_id=str(row["semantic_bucket_id"]),
                semantic_rank=int(row["semantic_rank"]),
                expert_weight=float(row["expert_weight"]),
                semantic_utility=float(row["semantic_utility"]),
            )
        output.append(item)
    return output


def _molecule_l2_records(
    connection: sqlite3.Connection, query_id: int, *, mode: str,
    parent_limit: int, record_limit: int,
) -> list[dict[str, Any]]:
    if mode not in {"morgan_parent_control", "morgan_parent_semantic"}:
        raise ValueError(f"unsupported molecule-first L2 mode: {mode}")
    rows = connection.execute(
        """SELECT a.*,r.external_record_id,r.parent_id,r.parent_smiles,r.payload
           FROM l2_assignments AS a JOIN l2_records AS r USING(record_key)
           WHERE a.query_id=? AND a.mode=? AND a.parent_rank<=?
             AND a.within_parent_rank<=?
           ORDER BY a.parent_rank,a.within_parent_rank""",
        (query_id, mode, parent_limit, record_limit),
    ).fetchall()
    parents = {str(row["parent_id"]) for row in rows}
    if len(parents) != parent_limit:
        raise ValueError(f"molecule-first L2 has {len(parents)} of {parent_limit} parents")
    return [{"record_id": str(row["external_record_id"]),
        "reference_molecule_id": str(row["parent_id"]),
        "reference_parent_smiles": str(row["parent_smiles"]),
        "morgan_similarity": float(row["morgan_similarity"]),
        "morgan_rank": int(row["parent_rank"]), "assay_rank": None,
        "ranking_method": mode, "payload": _payload(str(row["payload"]), str(row["parent_smiles"]))}
        for row in rows]


def _molecule_l2_records_v2(
    connection: sqlite3.Connection, query_id: int, *, mode: str, limit: int,
) -> list[dict[str, Any]]:
    if mode not in {"morgan_parent_control", "morgan_parent_semantic"}:
        raise ValueError(f"unsupported molecule-first L2 mode: {mode}")
    rows = connection.execute(
        """SELECT a.*,r.external_record_id,r.parent_id,r.parent_smiles,r.payload
           FROM l2_assignments AS a JOIN l2_records AS r USING(record_key)
           WHERE a.query_id=? AND a.mode=? AND a.selection_rank<=?
           ORDER BY a.selection_rank""",
        (query_id, mode, limit),
    ).fetchall()
    return [{"record_id": str(row["external_record_id"]),
        "reference_molecule_id": str(row["parent_id"]),
        "reference_parent_smiles": str(row["parent_smiles"]),
        "morgan_similarity": float(row["morgan_similarity"]),
        "morgan_rank": int(row["parent_rank"]), "assay_rank": None,
        "ranking_method": mode, "payload": _payload(str(row["payload"]), str(row["parent_smiles"]))}
        for row in rows]


def _ranked_candidates(connection: sqlite3.Connection, query_id: int) -> list[sqlite3.Row]:
    connection.row_factory = sqlite3.Row
    return list(connection.execute(
        """SELECT c.*,g.external_context_id,g.parent_id,g.parent_smiles,
                  g.condition_group,g.gold_label,g.available_voter_count
           FROM candidates AS c JOIN gold_contexts AS g USING(context_key)
           WHERE c.query_id=?
           ORDER BY c.transfer_score DESC,c.transfer_rank_top100,g.external_context_id""",
        (query_id,),
    ))


def _select(
    rows: list[sqlite3.Row], *, method: str, context_limit: int, min_contrast: int,
    primary_width: int, fallback_width: int,
    candidate_widths: list[int] | None = None,
) -> tuple[list[sqlite3.Row], dict[str, Any]]:
    if method not in METHODS:
        raise ValueError(f"Unsupported L1 context method: {method}")
    if min_contrast < 0:
        raise ValueError("Minimum contrast must be non-negative")
    candidate_widths = candidate_widths or [primary_width, fallback_width]
    rows = [row for row in rows if int(row["morgan_parent_rank"]) <= fallback_width]
    if method in {
        "morgan", "morgan_contrastive", "assay_transfer_within_morgan"
    }:
        rows.sort(key=lambda row: (
            int(row["morgan_parent_rank"]), str(row["external_context_id"])
        ))
    primary = [row for row in rows if int(row["morgan_parent_rank"]) <= primary_width]
    if len(primary) < context_limit:
        raise ValueError(
            f"Insufficient assay-ranked contexts within the Morgan top-{primary_width} parents"
        )
    selected = primary[:context_limit]
    if method == "assay_transfer_within_morgan":
        selected.sort(key=lambda row: (
            -float(row["transfer_score"]),
            int(row["transfer_rank_top100"]),
            str(row["external_context_id"]),
        ))
    replacements: list[dict[str, Any]] = []
    if method in {
        "morgan_contrastive", "assay_transfer_contrastive"
    } and min_contrast:
        if context_limit < 2 * min_contrast:
            raise ValueError("Contrastive L1 requires K >= 2*M")
        labels = {int(row["gold_label"]) for row in rows}
        if labels != {0, 1}:
            raise ValueError("Contrastive L1 cache must contain both binary labels")
        available = Counter(int(row["gold_label"]) for row in rows)
        if min(available[0], available[1]) < min_contrast:
            raise ValueError("Insufficient top-100 contexts for the requested label balance")
        counts = Counter(int(row["gold_label"]) for row in selected)
        selected_ids = {int(row["context_key"]) for row in selected}
        for missing_label in (0, 1):
            while counts[missing_label] < min_contrast:
                added_width = None
                for width in candidate_widths:
                    candidates = [
                        row for row in rows
                        if int(row["gold_label"]) == missing_label
                        and int(row["context_key"]) not in selected_ids
                        and int(row["morgan_parent_rank"]) <= width
                    ]
                    if candidates:
                        added_width = width
                        break
                if not candidates:
                    raise ValueError(
                        f"No unused contrastive context remains in the Morgan top-{fallback_width}"
                    )
                added = candidates[0]
                removable = [
                    row for row in selected
                    if int(row["gold_label"]) != missing_label
                    and counts[int(row["gold_label"])] > min_contrast
                ]
                if not removable:
                    raise ValueError("Cannot satisfy the requested binary label balance")
                if method == "morgan_contrastive":
                    removed = max(removable, key=lambda row: (
                        int(row["morgan_parent_rank"]), str(row["external_context_id"])
                    ))
                else:
                    removed = min(removable, key=lambda row: (
                        float(row["transfer_score"]),
                        -int(row["transfer_rank_top100"]),
                        str(row["external_context_id"]),
                    ))
                selected.remove(removed)
                selected.append(added)
                selected_ids.remove(int(removed["context_key"]))
                selected_ids.add(int(added["context_key"]))
                counts[int(removed["gold_label"])] -= 1
                counts[missing_label] += 1
                replacements.append({
                    "added_context_id": str(added["external_context_id"]),
                    "added_label": missing_label,
                    "added_parent_rank": int(added["morgan_parent_rank"]),
                    "added_parent_width": added_width,
                    "removed_context_id": str(removed["external_context_id"]),
                    "removed_label": int(removed["gold_label"]),
                })
        if method == "morgan_contrastive":
            selected.sort(key=lambda row: (
                int(row["morgan_parent_rank"]), str(row["external_context_id"])
            ))
        else:
            selected.sort(key=lambda row: (
                -float(row["transfer_score"]),
                int(row["transfer_rank_top100"]),
                str(row["external_context_id"]),
            ))
    counts = Counter(int(row["gold_label"]) for row in selected)
    unique_parents = len({str(row["parent_id"]) for row in selected})
    return selected, {
        "morgan_primary_parent_width": primary_width,
        "morgan_fallback_parent_width": fallback_width,
        "morgan_candidate_parent_widths": candidate_widths,
        "candidate_contexts_primary": len(primary),
        "candidate_contexts_fallback": len(rows),
        "selected_contexts": len(selected),
        "selected_unique_parents": unique_parents,
        "selected_repeated_parent_cards": len(selected) - unique_parents,
        "selected_label_counts": {str(label): counts[label] for label in (0, 1)},
        "replacement_count": len(replacements),
        "fallback_beyond_primary": any(
            item["added_parent_rank"] > primary_width for item in replacements
        ),
        "replacements": replacements,
    }


def _records(
    connection: sqlite3.Connection, row: sqlite3.Row, *, limit: int, method: str
) -> list[dict[str, Any]]:
    connection.row_factory = sqlite3.Row
    records = list(connection.execute(
        """SELECT r.external_record_id,r.payload,cr.within_context_rank
           FROM context_records AS cr JOIN records AS r USING(record_key)
           WHERE cr.context_key=?
           ORDER BY cr.within_context_rank,r.external_record_id LIMIT ?""",
        (int(row["context_key"]), limit),
    ))
    if not records:
        raise ValueError(f"Cached context has no voter records: {row['external_context_id']}")
    output = []
    for record in records:
        output.append({
            "record_id": str(record["external_record_id"]),
            "reference_molecule_id": str(row["parent_id"]),
            "context_card_id": str(row["external_context_id"]),
            "reference_parent_smiles": str(row["parent_smiles"]),
            "morgan_similarity": float(row["morgan_similarity"]),
            "morgan_rank": int(row["morgan_parent_rank"]),
            "assay_rank": int(row["transfer_rank_top100"]),
            "ranking_method": method,
            "transfer_likelihood": float(row["transfer_score"]),
            "payload": _payload(str(record["payload"]), str(row["parent_smiles"])),
        })
    return output


def load_candidates(
    queries: Mapping[str, str], *, task: str, subset: str, policy: Mapping[str, Any],
    molecule_limit: int = 10, l1_limit: int = 10, later_limit: int = 50,
    tie_seed: int = 0, min_contrast: int = 3, cache_pool: str = "all", **_: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return one card per gold parent-condition context using indexed reads only."""
    if not queries or min(molecule_limit, l1_limit, later_limit) < 1:
        raise ValueError("Queries and positive shape limits are required")
    database, manifest = _manifest(policy, task, subset)
    if tie_seed != 0:
        raise ValueError("The cache fixes ranking tie seed 0")
    if molecule_limit > int(manifest["capacities"]["l1_contexts"]):
        raise ValueError("Requested L1 context count exceeds cache capacity")
    if l1_limit > int(manifest["capacities"]["voter_records_per_context"]):
        raise ValueError("Requested L1 voter-record count exceeds cache capacity")
    semantic_l2 = manifest["schema_version"] in {
        SEMANTIC_L2_SCHEMA_VERSION, SEMANTIC_WEIGHTED_L2_SCHEMA_VERSION,
    }
    molecule_l2 = manifest["schema_version"] in {
        MORGAN_SEMANTIC_L2_SCHEMA_VERSION, MORGAN_SEMANTIC_L2_V2_SCHEMA_VERSION,
        MORGAN_SEMANTIC_L2_V3_SCHEMA_VERSION, MORGAN_SEMANTIC_L2_V4_SCHEMA_VERSION,
    }
    global_molecule_l2 = manifest["schema_version"] in {
        MORGAN_SEMANTIC_L2_V2_SCHEMA_VERSION, MORGAN_SEMANTIC_L2_V3_SCHEMA_VERSION,
        MORGAN_SEMANTIC_L2_V4_SCHEMA_VERSION,
    }
    adaptive_molecule_l2 = manifest["schema_version"] in {
        MORGAN_SEMANTIC_L2_V3_SCHEMA_VERSION, MORGAN_SEMANTIC_L2_V4_SCHEMA_VERSION,
    }
    if semantic_l2 and (cache_pool != "all" or later_limit > int(manifest["capacities"]["l2_records"])):
        raise ValueError("context-L2 requires the all pool and at most 12 records")
    if molecule_l2 and not global_molecule_l2 and (cache_pool != "all"
            or molecule_limit > int(manifest["capacities"]["l2_molecules"])
            or later_limit > int(manifest["capacities"]["l2_records_per_molecule"])):
        raise ValueError("molecule-first L2 exceeds its parent or per-parent capacity")
    if global_molecule_l2 and (cache_pool != "all"
            or later_limit > int(manifest["capacities"]["l2_total_records"])):
        raise ValueError("global-cap molecule-first L2 exceeds its cache capacity")
    method = str(policy["stages"]["L1"])
    molecules_by_query: dict[str, Any] = {}
    later_by_query: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    with _open_immutable(database) as connection:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if (metadata.get("schema_version") != manifest["schema_version"]
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
            identities[str(benchmark_row_id)] = {
                "parent_id": str(registered[2]), "parent_smiles": str(registered[3])
            }
            selected, audit = _select(
                _ranked_candidates(connection, query_id), method=method,
                context_limit=molecule_limit, min_contrast=min_contrast,
                primary_width=int(manifest["morgan_primary_parent_width"]),
                fallback_width=int(manifest["morgan_fallback_parent_width"]),
                candidate_widths=manifest.get("morgan_candidate_parent_widths"),
            )
            molecules = []
            for selection_rank, row in enumerate(selected):
                score = float(row["transfer_score"])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Invalid cached assay-transfer score")
                cards = _records(connection, row, limit=l1_limit, method=method)
                molecules.append({
                    "reference_molecule_id": str(row["parent_id"]),
                    "context_card_id": str(row["external_context_id"]),
                    "_diagnostic_label": int(row["gold_label"]),
                    "_diagnostic_context_id": str(row["external_context_id"]),
                    "_diagnostic_parent_id": str(row["parent_id"]),
                    "canonical_smiles": str(row["parent_smiles"]),
                    "selected_condition": (
                        "" if row["condition_group"] == "no_reported_external_condition"
                        else str(row["condition_group"])
                    ),
                    "morgan_similarity": float(row["morgan_similarity"]),
                    "morgan_rank": int(row["morgan_parent_rank"]),
                    "assay_rank": int(row["transfer_rank_top100"]),
                    "selection_rank": selection_rank,
                    "available_l1": int(row["available_voter_count"]),
                    "available_l2": 0,
                    "ranking_method": method,
                    "transfer_likelihood": score,
                    "l1_records": cards,
                    "l2_records": [],
                    "gold_context_ids": {},
                })
            molecules_by_query[str(benchmark_row_id)] = molecules
            level_audit = {"L1": audit}
            if semantic_l2:
                mode = str(policy["stages"]["L2"])
                records = _l2_records(connection, query_id, mode=mode, limit=later_limit)
                counts = connection.execute(
                    "SELECT * FROM l2_selection_counts WHERE query_id=?", (query_id,)
                ).fetchone()
                weighted = mode == "semantic_weighted"
                candidate_records = int(counts[
                    "weighted_candidate_record_count" if weighted else "candidate_record_count"
                ])
                candidate_molecules = int(counts[
                    "weighted_candidate_parent_count" if weighted else "candidate_parent_count"
                ])
                candidate_buckets = int(counts[
                    "weighted_candidate_bucket_count" if weighted else "candidate_bucket_count"
                ])
                later = {
                    "L2": {
                        "records": records,
                        "available_record_count": candidate_records,
                        "allow_shortfall": False,
                        "ranking_method": mode,
                        "candidate_records": candidate_records,
                        "candidate_molecules": candidate_molecules,
                        "candidate_semantic_buckets": candidate_buckets,
                        "unmapped_records": 0 if weighted else int(counts["unmapped_record_count"]),
                        "selected_records": len(records),
                        "selected_molecules": len({row["reference_molecule_id"] for row in records}),
                        "shortfall": later_limit - len(records),
                    }
                }
                if weighted:
                    later["L2"].update(
                        capped_capacity=int(counts["weighted_capped_capacity"]),
                        selected_semantic_buckets=len({
                            row["semantic_bucket_id"] for row in records
                        }),
                        selected_utility=sum(row["semantic_utility"] for row in records),
                    )
                later_by_query[str(benchmark_row_id)] = later
                level_audit["L2"] = {key: value for key, value in later["L2"].items()
                                     if key not in {"records", "ranking_method", "allow_shortfall"}}
            elif molecule_l2:
                mode = str(policy["stages"]["L2"])
                records = (
                    _molecule_l2_records_v2(
                        connection, query_id, mode=mode, limit=later_limit,
                    )
                    if global_molecule_l2 else
                    _molecule_l2_records(connection, query_id, mode=mode,
                        parent_limit=molecule_limit, record_limit=later_limit)
                )
                if adaptive_molecule_l2 and len(records) != later_limit:
                    raise ValueError("adaptive molecule-first L2 did not fill its record cap")
                counts = connection.execute(
                    "SELECT * FROM l2_selection_counts WHERE query_id=?"
                    + (" AND mode=?" if global_molecule_l2 else ""),
                    (query_id, mode) if global_molecule_l2 else (query_id,),
                ).fetchone()
                available = int(counts[
                    "candidate_records" if mode == "morgan_parent_control" else "eligible_records"
                ])
                selected_molecules = len({row["reference_molecule_id"] for row in records})
                excluded = int(counts[
                    "excluded_bad_records" if global_molecule_l2 else "excluded_records"
                ])
                later = {"L2": {"records": records, "available_record_count": available,
                    "allow_shortfall": not adaptive_molecule_l2, "ranking_method": mode,
                    "candidate_records": int(counts["candidate_records"]),
                    "eligible_records": int(counts["eligible_records"]),
                    "excluded_records": excluded,
                    "candidate_molecules": int(counts["candidate_parents"]),
                    "selected_molecules": selected_molecules, "selected_records": len(records),
                    "shortfall": later_limit - len(records)}}
                if global_molecule_l2:
                    later["L2"].update(
                        mapped_records=int(counts["mapped_records"]),
                        unmapped_records=int(counts["unmapped_records"]),
                        selected_parent_panel=int(counts["candidate_parents"]),
                    )
                later_by_query[str(benchmark_row_id)] = later
                level_audit["L2"] = {key: value for key, value in later["L2"].items()
                                     if key not in {"records", "ranking_method", "allow_shortfall"}}
            audits[str(benchmark_row_id)] = level_audit
    inputs = dict(policy.get("inputs") or {})
    inputs[str(Path(__file__).resolve())] = sha256_file(Path(__file__))
    return molecules_by_query, (
        later_by_query if semantic_l2 or molecule_l2 else {str(query_id): {} for query_id in queries}
    ), {
        "selection_policy": manifest["schema_version"],
        "inputs": inputs,
        "contract": {
            "policy": dict(policy), "context_limit": molecule_limit,
            "l1_limit": l1_limit, "min_contrast": min_contrast,
            "cache_content_id": manifest["content_id"],
            "morgan_primary_parent_width": manifest["morgan_primary_parent_width"],
            "morgan_fallback_parent_width": manifest["morgan_fallback_parent_width"],
            **({"later_limit": later_limit, "cache_pool": cache_pool}
               if semantic_l2 or molecule_l2 else {}),
        },
        "cache_version": str(Path(policy["cache_manifest"]).resolve()),
        "cache_content_id": manifest["content_id"],
        "cache_capacities": dict(manifest["capacities"]),
        "cache_assignment_counts": dict(manifest["assignment_counts"]),
        "cache_record_count": int(manifest["record_count"]),
        "query_identities": identities,
        "query_audits": audits,
        "neighbor_identity_policy": "scaffold_disjoint",
        "similarity_floor": None,
        "scaffold_overlap": 0,
        "parent_overlap": 0,
    }
