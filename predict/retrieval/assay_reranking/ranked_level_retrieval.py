"""Read independent top-100 level rankings and join selected evidence by UID."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Mapping


SCHEMA_VERSION = "ranked_level_retrieval.v2"
CAPACITY = 100


def _open(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _chunks(values: list[str], size: int = 900):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _rows_by_key(
    connection: sqlite3.Connection, table: str, key: str, values: list[str]
) -> dict[str, sqlite3.Row]:
    output: dict[str, sqlite3.Row] = {}
    for chunk in _chunks(values):
        placeholders = ",".join("?" for _ in chunk)
        for row in connection.execute(
            f"SELECT * FROM {table} WHERE {key} IN ({placeholders})", chunk
        ):
            output[str(row[key])] = row
    return output


def _limits(
    stages: Mapping[str, str], later_limit: int | Mapping[str, int]
) -> dict[str, int]:
    later_levels = [level for level in stages if level != "L1"]
    if isinstance(later_limit, Mapping):
        unknown = set(later_limit) - set(later_levels)
        if unknown:
            raise ValueError(f"Record limits contain unsupported levels: {sorted(unknown)}")
        missing = set(later_levels) - set(later_limit)
        if missing:
            raise ValueError(f"Record limits omit levels: {sorted(missing)}")
        result = {level: int(later_limit[level]) for level in later_levels}
    else:
        result = {level: int(later_limit) for level in later_levels}
    invalid = {level: value for level, value in result.items() if not 1 <= value <= CAPACITY}
    if invalid:
        raise ValueError(f"Requested level counts exceed cache capacity {CAPACITY}: {invalid}")
    return result


def _manifest(path: Path, *, task: str, subset: str, level: str, method: str) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "task_id": task,
        "subset": subset,
        "level": level,
        "ranking_method": method,
        "pool": "all" if level != "L1" else "fixed",
        "capacity": CAPACITY,
    }
    if any(document.get(key) != value for key, value in required.items()):
        raise ValueError(f"Incompatible independent level cache: {path}")
    return document


def _ranked_rows(
    manifest_path: Path,
    *,
    task: str,
    subset: str,
    level: str,
    method: str,
    queries: Mapping[str, str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, tuple[str, str]], dict[str, Any]]:
    manifest = _manifest(
        manifest_path, task=task, subset=subset, level=level, method=method
    )
    database = manifest_path.with_name(str(manifest["database"]))
    if not database.is_file():
        raise ValueError(f"Missing independent level database: {database}")
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: dict[str, tuple[str, str]] = {}
    with _open(database) as connection:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if metadata.get("content_id") != manifest["content_id"]:
            raise ValueError(f"Level database and manifest differ: {manifest_path}")
        query_ids = [str(query_id) for query_id in queries]
        registered_queries = _rows_by_key(
            connection, "queries", "benchmark_row_id", query_ids
        )
        for query_id, drug in queries.items():
            registered = registered_queries.get(str(query_id))
            if registered is None or str(registered["drug"]) != str(drug):
                raise ValueError(f"Query differs from frozen cache ledger: {query_id}")
            identities[str(query_id)] = (
                str(registered["query_parent_id"]), str(registered["query_parent_smiles"])
            )
        for chunk in _chunks(query_ids):
            placeholders = ",".join("?" for _ in chunk)
            for row in connection.execute(
                "SELECT * FROM rankings "
                f"WHERE benchmark_row_id IN ({placeholders}) "
                "ORDER BY benchmark_row_id,rank",
                chunk,
            ):
                output[str(row["benchmark_row_id"])].append(dict(row))
        for query_id in query_ids:
            output.setdefault(query_id, [])
    return dict(output), identities, manifest


def _record(
    ranked: Mapping[str, Any], evidence: Mapping[str, Any], method: str
) -> dict[str, Any]:
    similarity = float(ranked["morgan_similarity"])
    if not math.isfinite(similarity) or not 0 <= similarity <= 1:
        raise ValueError("Invalid cached Morgan similarity")
    payload = json.loads(str(evidence["payload"]))
    source_fields = payload.get("source_fields")
    if isinstance(source_fields, Mapping) and "source_contract" not in payload:
        payload["source_contract"] = {
            "contract_version": "ranked_evidence_projection.v1",
            "source_or_simply_cleaned": {name: True for name in source_fields},
        }
    parent_smiles = str(ranked["parent_smiles"])
    payload.setdefault("source_canonical_smiles", payload.get("canonical_smiles"))
    payload.setdefault("evidence_parent_smiles", payload.get("canonical_smiles"))
    payload["canonical_smiles"] = parent_smiles
    payload["reference_parent_smiles"] = parent_smiles
    rank = int(ranked["rank"])
    result = {
        "record_id": str(evidence["external_record_id"]),
        "reference_molecule_id": str(ranked["parent_id"]),
        "reference_parent_smiles": parent_smiles,
        "morgan_similarity": similarity,
        "morgan_rank": rank if method == "morgan" else None,
        "assay_rank": rank if method == "assay_transfer" else None,
        "ranking_method": method,
        "payload": payload,
    }
    if method == "assay_transfer":
        score = ranked["transfer_likelihood"]
        if score is None or not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
            raise ValueError("Invalid cached assay-transfer score")
        result["transfer_likelihood"] = float(score)
    return result


def load_candidates(
    queries: Mapping[str, str], *, task: str, subset: str, policy: Mapping[str, Any],
    molecule_limit: int = 10, l1_limit: int = 10,
    later_limit: int | Mapping[str, int] = 50, tie_seed: int = 0,
    cache_pool: str = "all", **_: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return prompt-ready selections from independent level caches."""
    if not queries or not 1 <= molecule_limit <= CAPACITY or l1_limit < 1:
        raise ValueError(f"Queries and positive limits are required; L1 capacity is {CAPACITY}")
    if cache_pool != "all" or tie_seed != 0:
        raise ValueError("ranked_level_retrieval.v2 supports only pool=all and tie_seed=0")
    stages = dict(policy["stages"])
    if stages.get("L1") not in {"morgan", "assay_transfer"}:
        raise ValueError("ranked_level_retrieval.v2 does not support joint L1")
    limits = _limits(stages, later_limit)
    manifests = {str(level): Path(path).resolve() for level, path in policy["cache_manifests"].items()}
    if set(manifests) != set(stages):
        raise ValueError("Independent cache policy does not cover the requested levels exactly")

    ranked: dict[str, dict[str, list[dict[str, Any]]]] = {}
    content_ids: dict[str, str] = {}
    query_identities: dict[str, tuple[str, str]] | None = None
    manifest_documents: dict[str, dict[str, Any]] = {}
    for level, method in stages.items():
        rows, identities, manifest = _ranked_rows(
            manifests[level], task=task, subset=subset, level=level,
            method=method, queries=queries,
        )
        if query_identities is not None and identities != query_identities:
            raise ValueError("Independent level caches disagree on query identity")
        query_identities = identities
        ranked[level] = rows
        content_ids[level] = str(manifest["content_id"])
        manifest_documents[level] = manifest

    index_path = Path(policy["cache_index"]).resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    evidence_manifest_path = (index_path.parent / index["evidence"]["manifest"]).resolve()
    evidence_manifest = json.loads(evidence_manifest_path.read_text(encoding="utf-8"))
    if (
        evidence_manifest.get("status") != "complete"
        or evidence_manifest.get("content_id") != index["evidence"]["content_id"]
    ):
        raise ValueError("Evidence store differs from the task release index")
    evidence_path = evidence_manifest_path.with_name(str(evidence_manifest["database"]))

    selected_l1 = {
        query_id: rows[:molecule_limit] for query_id, rows in ranked["L1"].items()
    }
    context_ids = sorted({
        str(row["item_id"]) for rows in selected_l1.values() for row in rows
    })
    with _open(evidence_path) as evidence:
        evidence_metadata = dict(evidence.execute("SELECT key,value FROM metadata"))
        if evidence_metadata.get("content_id") != evidence_manifest["content_id"]:
            raise ValueError("Evidence database and manifest differ")
        context_metadata = _rows_by_key(evidence, "contexts", "context_id", context_ids)
        context_uids: dict[str, list[str]] = defaultdict(list)
        for chunk in _chunks(context_ids):
            placeholders = ",".join("?" for _ in chunk)
            for row in evidence.execute(
                "SELECT context_id,source_row_uid FROM context_records "
                f"WHERE context_id IN ({placeholders}) ORDER BY context_id,within_context_rank",
                chunk,
            ):
                if len(context_uids[str(row["context_id"])]) < l1_limit:
                    context_uids[str(row["context_id"])].append(str(row["source_row_uid"]))

        later_candidates: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
        all_uids = {uid for values in context_uids.values() for uid in values}
        for query_id in queries:
            seen_uids = {
                uid for row in selected_l1[str(query_id)]
                for uid in context_uids.get(str(row["item_id"]), [])
            }
            for level in stages:
                if level == "L1":
                    continue
                requested = limits[level]
                chosen = []
                for row in ranked[level][str(query_id)]:
                    uid = str(row["item_id"])
                    if uid in seen_uids:
                        continue
                    chosen.append(row)
                    seen_uids.add(uid)
                    if len(chosen) == requested:
                        break
                later_candidates[str(query_id)][level] = chosen
                all_uids.update(str(row["item_id"]) for row in chosen)

        evidence_rows = _rows_by_key(evidence, "records", "source_row_uid", sorted(all_uids))

    missing = sorted(all_uids - set(evidence_rows))
    if missing:
        raise ValueError(f"Selected evidence UIDs are absent: {missing[:5]}")

    molecules_by_query: dict[str, Any] = {}
    later_by_query: dict[str, Any] = {}
    query_audits: dict[str, Any] = {}
    for query_id in queries:
        molecules = []
        seen_record_ids: set[str] = set()
        l1_method = stages["L1"]
        for selection_rank, row in enumerate(selected_l1[str(query_id)]):
            context_id = str(row["item_id"])
            context = context_metadata.get(context_id)
            if context is None or str(context["parent_id"]) != str(row["parent_id"]):
                raise ValueError(f"L1 context identity mismatch: {context_id}")
            records = [
                _record(row, evidence_rows[uid], l1_method)
                for uid in context_uids.get(context_id, [])
            ]
            seen_record_ids.update(record["record_id"] for record in records)
            molecule = {
                "reference_molecule_id": str(row["parent_id"]),
                "canonical_smiles": str(row["parent_smiles"]),
                "morgan_similarity": float(row["morgan_similarity"]),
                "morgan_rank": int(row["rank"]) if l1_method == "morgan" else None,
                "assay_rank": int(row["rank"]) if l1_method == "assay_transfer" else None,
                "selection_rank": selection_rank,
                "available_l1": int(row["member_count"]),
                "available_l2": 0,
                "ranking_method": l1_method,
                "l2_records": [],
                "context_card_id": context_id,
                "selected_condition": (
                    "" if str(context["condition_group"]) == "no_reported_external_condition"
                    else str(context["condition_group"])
                ),
                "_diagnostic_label": int(context["gold_label"]),
                "_diagnostic_context_id": context_id,
                "_diagnostic_parent_id": str(row["parent_id"]),
                "gold_context_ids": {l1_method: context_id},
                "l1_records": records,
            }
            if l1_method == "assay_transfer":
                molecule["transfer_likelihood"] = float(row["transfer_likelihood"])
            molecules.append(molecule)

        later: dict[str, Any] = {}
        audits = {
            "L1": {
                "candidate_records": sum(int(row["member_count"]) for row in ranked["L1"][str(query_id)]),
                "candidate_molecules": len(ranked["L1"][str(query_id)]),
                "selected_records": sum(len(item["l1_records"]) for item in molecules),
                "selected_molecules": len(molecules),
                "shortfall": max(0, molecule_limit - len(molecules)),
            }
        }
        for level in stages:
            if level == "L1":
                continue
            method = stages[level]
            records = [_record(row, evidence_rows[str(row["item_id"])], method)
                       for row in later_candidates[str(query_id)][level]]
            overlap = {row["record_id"] for row in records} & seen_record_ids
            if overlap:
                raise ValueError(f"A selected record occurs in multiple levels: {sorted(overlap)[:5]}")
            seen_record_ids.update(row["record_id"] for row in records)
            requested = limits[level]
            manifest = manifest_documents[level]
            counts = manifest["query_counts"][str(query_id)]
            audit = {
                "candidate_records": int(counts["candidate_records"]),
                "candidate_molecules": int(counts["candidate_parents"]),
                "selected_records": len(records),
                "selected_molecules": len({row["reference_molecule_id"] for row in records}),
                "shortfall": requested - len(records),
            }
            audits[level] = audit
            later[level] = {
                "records": records,
                "available_record_count": int(counts["candidate_records"]),
                "allow_shortfall": True,
                "ranking_method": method,
                **audit,
            }
        molecules_by_query[str(query_id)] = molecules
        later_by_query[str(query_id)] = later
        query_audits[str(query_id)] = audits

    return molecules_by_query, later_by_query, {
        "selection_policy": SCHEMA_VERSION,
        "inputs": dict(policy.get("inputs") or {}),
        "contract": {
            "policy": dict(policy), "molecule_limit": molecule_limit,
            "l1_limit": l1_limit, "later_limits": limits,
            "tie_seed": tie_seed, "cache_pool": cache_pool,
            "cache_content_ids": content_ids,
        },
        "cache_pool": "all",
        "cache_index": str(index_path),
        "cache_content_ids": content_ids,
        "cache_capacities": {level: CAPACITY for level in stages},
        "query_identities": {
            query_id: {"parent_id": value[0], "parent_smiles": value[1]}
            for query_id, value in (query_identities or {}).items()
        },
        "query_audits": query_audits,
        "neighbor_identity_policy": "level_specific_disjoint",
        "neighbor_identity_policy_by_level": {
            level: "scaffold_disjoint" if level == "L1" else "parent_disjoint"
            for level in stages
        },
        "similarity_floor": None,
        "scaffold_overlap": 0,
        "parent_overlap": 0,
    }


__all__ = ["CAPACITY", "SCHEMA_VERSION", "load_candidates"]
