"""Select ranked UIDs, hydrate them once, and return prompt-ready evidence."""
from __future__ import annotations

from data.processing.paths import resolve_legacy_path
from predict.utils.json import read_json

from collections import defaultdict
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq

from predict.utils.json import sha256_file
from predict.retrieval.policies import (
    MOLECULAR_REPRESENTATION_POLICY, source_smiles,
)
from predict.retrieval.selection import assert_disjoint


SCHEMA_VERSION = "ranked_uid_retrieval.v1"
CAPACITY = 100
HYBRID_METHOD = "assay_transfer_complete_or_hidden"
HIDDEN_PARTIAL_METHOD = "assay_transfer_partial_hidden"
QUERY_PURE_METHOD = "assay_transfer_complete_else_morgan"


def _open(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{resolve_legacy_path(path).resolve()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _limits(stages: Mapping[str, str], later_limit: int | Mapping[str, int]) -> dict[str, int]:
    levels = [level for level in stages if level != "L1"]
    if isinstance(later_limit, Mapping):
        if set(later_limit) != set(levels):
            raise ValueError("Record limits must cover the requested later levels exactly")
        result = {level: int(later_limit[level]) for level in levels}
    else:
        result = {level: int(later_limit) for level in levels}
    invalid = {level: value for level, value in result.items() if value < 1}
    if invalid:
        raise ValueError(f"Requested level counts must be positive: {invalid}")
    return result


def _manifest(path: Path, task: str, subset: str, level: str) -> dict[str, Any]:
    document = read_json(path)
    required = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "task_id": task,
        "subset": subset,
        "level": level,
        "pool": "fixed" if level == "L1" else "all",
    }
    capacities = {CAPACITY} if level == "L1" else {40, 50, CAPACITY}
    if (any(document.get(key) != value for key, value in required.items())
            or document.get("parent_capacity") not in capacities
            or document.get("neighbor_identity_policy", "parent_disjoint") not in {
                "parent_disjoint", "scaffold_disjoint",
            }):
        raise ValueError(f"Incompatible ranked UID cache: {path}")
    if level == "L1" and document.get("neighbor_identity_policy") != "scaffold_disjoint":
        raise ValueError("Direct L1 retrieval requires scaffold-disjoint candidates")
    return document


def _ranked_rows(
    manifest_path: Path, *, task: str, subset: str, level: str, method: str,
    queries: Mapping[str, str], limit: int | None,
    parent_morgan_width: int | None = None,
    selected_uids: Mapping[str, Sequence[str]] | None = None,
    allow_unscored_preselected: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, tuple[str, str]], dict[str, Any]]:
    manifest = _manifest(manifest_path, task, subset, level)
    database = manifest_path.with_name(str(manifest["database"]))
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: dict[str, tuple[str, str]] = {}
    if method not in {"morgan", "assay_transfer", HYBRID_METHOD, QUERY_PURE_METHOD}:
        raise ValueError(f"Unsupported ranked UID method: {method}")
    rank_column = "morgan_rank" if method == "morgan" else "assay_rank"
    with _open(database) as connection:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if metadata.get("content_id") != manifest["content_id"]:
            raise ValueError(f"Ranking database and manifest differ: {manifest_path}")
        for query_id, drug in queries.items():
            registered = connection.execute(
                "SELECT drug,query_parent_id,query_parent_smiles FROM queries "
                "WHERE benchmark_row_id=?", (str(query_id),),
            ).fetchone()
            if registered is None or str(registered["drug"]) != str(drug):
                raise ValueError(f"Query differs from frozen cache ledger: {query_id}")
            identities[str(query_id)] = (
                str(registered["query_parent_id"]), str(registered["query_parent_smiles"])
            )
            width_clause = (
                " AND parent_morgan_rank<=?" if parent_morgan_width is not None else ""
            )
            sql = (
                f"SELECT * FROM rankings WHERE benchmark_row_id=? AND {rank_column} IS NOT NULL"
                f"{width_clause} ORDER BY {rank_column},item_id"
            )
            parameters: tuple[Any, ...] = (
                (str(query_id), parent_morgan_width)
                if parent_morgan_width is not None else (str(query_id),)
            )
            if method in {HYBRID_METHOD, QUERY_PURE_METHOD}:
                all_rows = connection.execute(
                    "SELECT * FROM rankings WHERE benchmark_row_id=?",
                    (str(query_id),),
                ).fetchall()
                scored = [
                    row for row in all_rows if row["assay_transfer_score"] is not None
                ]
                if method == QUERY_PURE_METHOD:
                    if len(scored) not in {0, len(all_rows)} or any(
                        (row["assay_rank"] is not None) != (row["assay_transfer_score"] is not None)
                        for row in all_rows
                    ):
                        raise ValueError(f"Query-pure cache contains mixed ranks: {query_id}/{level}")
                    selection_method = "assay_transfer" if scored else "morgan"
                else:
                    selection_method = (
                        "assay_transfer" if len(scored) == len(all_rows)
                        else HIDDEN_PARTIAL_METHOD
                    )
                if selected_uids is not None:
                    ordered_uids = [str(uid) for uid in selected_uids[str(query_id)]]
                    by_uid = {str(row["item_id"]): row for row in all_rows}
                    missing = [uid for uid in ordered_uids if uid not in by_uid]
                    if len(set(ordered_uids)) != len(ordered_uids) or missing:
                        raise ValueError(
                            f"Invalid preselected UIDs for {query_id}/{level}: {missing[:5]}"
                        )
                    rows = [by_uid[uid] for uid in ordered_uids]
                elif selection_method == "assay_transfer":
                    rows = sorted(all_rows, key=lambda row: int(row["assay_rank"]))
                elif selection_method == "morgan":
                    rows = sorted(all_rows, key=lambda row: int(row["morgan_rank"]))
                else:
                    rows = sorted(
                        scored,
                        key=lambda row: (
                            -float(row["assay_transfer_score"]), str(row["item_id"])
                        ),
                    ) + sorted(
                        (row for row in all_rows if row["assay_transfer_score"] is None),
                        key=lambda row: (int(row["morgan_rank"]), str(row["item_id"])),
                    )
                if limit is not None:
                    rows = rows[:limit]
                rows = [
                    {**dict(row), "_selection_method": selection_method}
                    for row in rows
                ]
            elif selected_uids is not None:
                ordered_uids = [str(uid) for uid in selected_uids[str(query_id)]]
                if len(set(ordered_uids)) != len(ordered_uids):
                    raise ValueError(f"Preselected UIDs must be unique: {query_id}/{level}")
                rows = []
                if ordered_uids:
                    placeholders = ",".join("?" for _ in ordered_uids)
                    selected_rows = connection.execute(
                        f"SELECT * FROM rankings WHERE benchmark_row_id=? "
                        f"AND item_id IN ({placeholders})"
                        + ("" if allow_unscored_preselected else f" AND {rank_column} IS NOT NULL"),
                        (str(query_id), *ordered_uids),
                    ).fetchall()
                    by_uid = {str(row["item_id"]): row for row in selected_rows}
                    missing = [uid for uid in ordered_uids if uid not in by_uid]
                    if missing:
                        raise ValueError(
                            f"Preselected UIDs are absent from {query_id}/{level}/{method}: "
                            f"{missing[:5]}"
                        )
                    rows = [
                        {**dict(by_uid[uid]), "_selection_method": HIDDEN_PARTIAL_METHOD}
                        if by_uid[uid]["assay_transfer_score"] is None else by_uid[uid]
                        for uid in ordered_uids
                    ]
            else:
                rows = connection.execute(
                    f"{sql} LIMIT ?", (*parameters, limit),
                ).fetchall() if limit is not None else connection.execute(
                    sql, parameters,
                ).fetchall()
            if (limit is not None and len(rows) != limit
                    and not (method == "assay_transfer"
                             and manifest.get("score_coverage") == "partial_snapshot")):
                if method in {HYBRID_METHOD, QUERY_PURE_METHOD}:
                    raise ValueError(f"Requested {limit} rows but {query_id}/{level} has {len(rows)}")
                available = connection.execute(
                    f"SELECT COUNT(*) FROM rankings WHERE benchmark_row_id=? "
                    f"AND {rank_column} IS NOT NULL{width_clause}", parameters,
                ).fetchone()[0]
                raise ValueError(
                    f"Requested {limit} rows but {query_id}/{level}/{method} has {available}"
                )
            output[str(query_id)] = [dict(row) for row in rows]
    return dict(output), identities, manifest


def load_ranked_universe(
    release_index: Path, *, task: str, subset: str, levels: list[str] | tuple[str, ...],
    queries: Mapping[str, str],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], Path, dict[str, Any]]:
    """Load every record in selected later-level Morgan universes.

    The release index remains the authority for level manifests and the shared
    V10 evidence projection. This reader deliberately does not hydrate the
    records so callers can optimize over compact ranking rows first.
    """
    index_path = Path(release_index).resolve()
    index = read_json(index_path)
    if (
        index.get("schema_version") != "ranked_uid_task_release_index.v1"
        or index.get("status") != "complete"
        or index.get("task_id") != task
        or index.get("pool") != "all"
        or index.get("later_candidate_universe") not in {
            "all_uids_under_morgan_top_100_parents",
            "all_uids_under_shared_l2plus_morgan_top_100_parents",
            "all_uids_under_shared_l2plus_morgan_top_50_parents",
            "all_uids_under_shared_l2plus_morgan_top_40_parents",
        }
    ):
        raise ValueError(f"Incompatible ranked UID task release index: {index_path}")
    if not queries or not levels or len(set(levels)) != len(levels) or "L1" in levels:
        raise ValueError("Queries and unique later levels are required")

    try:
        indexed_levels = index["splits"][subset]["levels"]
    except KeyError as error:
        raise ValueError(f"Subset is absent from ranked UID release: {subset}") from error

    ranked: dict[str, dict[str, list[dict[str, Any]]]] = {}
    identities: dict[str, tuple[str, str]] | None = None
    manifests: dict[str, dict[str, Any]] = {}
    manifest_paths: dict[str, Path] = {}
    for level in levels:
        if level not in indexed_levels:
            raise ValueError(f"Level is absent from ranked UID release: {level}")
        entry = indexed_levels[level]
        manifest_path = (index_path.parent / str(entry["manifest"])).resolve()
        document = _manifest(manifest_path, task, subset, level)
        if (
            sha256_file(manifest_path) != entry["manifest_sha256"]
            or document["content_id"] != entry["content_id"]
            or document["parent_capacity"] != index["parent_capacity"]
            or index["neighbor_identity_policy_by_level"].get(level) not in {
                "parent_disjoint", "scaffold_disjoint",
            }
            or document.get("neighbor_identity_policy", "parent_disjoint")
            != index["neighbor_identity_policy_by_level"].get(level)
        ):
            raise ValueError(f"{level} cache differs from the task release index")
        rows, current_identities, _ = _ranked_rows(
            manifest_path, task=task, subset=subset, level=level, method="morgan",
            queries=queries, limit=None,
        )
        for query_id, query_rows in rows.items():
            counts = document["query_counts"][query_id]
            if (
                len(query_rows) != int(counts["candidate_records"])
                or len({str(row["parent_id"]) for row in query_rows})
                != int(counts["candidate_parents"])
            ):
                raise ValueError(f"{query_id}/{level} candidate universe is incomplete")
        if identities is not None and current_identities != identities:
            raise ValueError("Independent level caches disagree on query identity")
        ranked[level] = rows
        identities = current_identities
        manifests[level] = document
        manifest_paths[level] = manifest_path

    evidence_entry = index["evidence"]
    evidence_manifest_path = (index_path.parent / str(evidence_entry["manifest"])).resolve()
    evidence_manifest = read_json(evidence_manifest_path)
    if (
        evidence_manifest.get("schema_version") != "ranked_evidence_projection.v1"
        or evidence_manifest.get("status") != "complete"
        or evidence_manifest.get("task_id") != task
        or evidence_manifest.get("content_id") != evidence_entry["content_id"]
        or sha256_file(evidence_manifest_path) != evidence_entry["manifest_sha256"]
    ):
        raise ValueError("Evidence projection differs from the task release index")

    return ranked, evidence_manifest_path, {
        "release_index": str(index_path),
        "release_index_sha256": sha256_file(index_path),
        "profile": index.get("profile"),
        "gold_release": index.get("gold_release"),
        "label_release": index.get("label_release"),
        "neighbor_identity_policy_by_level": {
            level: index["neighbor_identity_policy_by_level"][level] for level in levels
        },
        "parent_capacity": int(index["parent_capacity"]),
        "level_manifests": {
            level: {
                "path": str(manifest_paths[level]),
                "sha256": sha256_file(manifest_paths[level]),
                "content_id": manifests[level]["content_id"],
            }
            for level in levels
        },
        "evidence_manifest": str(evidence_manifest_path),
        "evidence_manifest_sha256": sha256_file(evidence_manifest_path),
        "evidence_content_id": evidence_manifest["content_id"],
        "query_identities": {
            query_id: {"parent_id": value[0], "parent_smiles": value[1]}
            for query_id, value in (identities or {}).items()
        },
    }


def load_ranked_panels(
    manifest_path: Path, *, task: str, subset: str, level: str,
    queries: Mapping[str, str], morgan_limit: int, assay_limit: int,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, list[str]]]:
    """Return two independent orders and their deduplicated item union."""
    morgan, identities, _ = _ranked_rows(
        manifest_path, task=task, subset=subset, level=level,
        method="morgan", queries=queries, limit=morgan_limit,
    )
    assay, assay_identities, _ = _ranked_rows(
        manifest_path, task=task, subset=subset, level=level,
        method="assay_transfer", queries=queries, limit=assay_limit,
    )
    if identities != assay_identities:
        raise ValueError("Morgan and assay panels disagree on query identity")
    union = {
        query_id: list(dict.fromkeys(
            str(row["item_id"])
            for panel in (morgan[query_id], assay[query_id])
            for row in panel
        ))
        for query_id in queries
    }
    return {"morgan": morgan, "assay_transfer": assay}, union


def _payload(row: Mapping[str, Any], ranked: Mapping[str, Any], method: str) -> dict[str, Any]:
    similarity = float(ranked["morgan_similarity"])
    if not math.isfinite(similarity) or not 0 <= similarity <= 1:
        raise ValueError("Invalid cached Morgan similarity")
    payload = json.loads(str(row["payload"]))
    canonical_card = payload.get("label_source") == "canonical_context_cards.v1"
    if canonical_card or payload.get("label_source") == "tdc" or str(payload.get("label_source") or "").startswith("tdc_v"):
        source_fields = dict(payload.get("source_fields") or {})
        if canonical_card and (
            set(source_fields) != {"label"}
            or source_fields["label"] not in {0, 1}
            or source_fields["label"] != payload.get("Y")
        ):
            raise ValueError("Canonical context card has an invalid label contract")
        payload.setdefault("family_key", "tdc_training_labels")
        payload.setdefault("source_id", str(payload["label_source"]))
        payload.setdefault("record_id", str(row["external_record_id"]))
        payload.setdefault("source_row_uid", str(row["source_row_uid"]))
        payload.setdefault("measurement_kind", "categorical_label")
        payload.setdefault(
            "source_contract",
            {
                "source_or_simply_cleaned": {
                    name: True for name in source_fields
                }
            },
        )
    parent_smiles = str(ranked["parent_smiles"])
    payload.setdefault("source_canonical_smiles", payload.get("canonical_smiles"))
    payload.setdefault("evidence_parent_smiles", payload.get("canonical_smiles"))
    payload["canonical_smiles"] = str(ranked.get("canonical_smiles") or parent_smiles)
    payload["reference_parent_smiles"] = parent_smiles
    effective_method = str(ranked.get("_selection_method") or method)
    hidden_partial = effective_method == HIDDEN_PARTIAL_METHOD
    result = {
        "record_id": str(row["external_record_id"]),
        "reference_molecule_id": str(ranked.get("canonical_smiles") or ranked["parent_id"]),
        "reference_parent_smiles": parent_smiles,
        "morgan_similarity": similarity,
        "morgan_rank": int(ranked["morgan_rank"]),
        "assay_rank": (
            int(ranked["assay_rank"])
            if ranked["assay_rank"] is not None and not hidden_partial else None
        ),
        "ranking_method": effective_method,
        "payload": payload,
    }
    if effective_method == "assay_transfer":
        score = ranked["assay_transfer_score"]
        if score is None or not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
            raise ValueError("Invalid cached assay-transfer score")
        result["transfer_likelihood"] = float(score)
    return result




def _hydrate(
    path: Path, uids: set[str], *, require_complete: bool = True
) -> dict[str, dict[str, Any]]:
    if not uids:
        return {}
    if any(uid.startswith("skin_gold_v1_member_") for uid in uids):
        raise ValueError(
            "Skin Gold cache contains synthetic member IDs, not canonical evidence UIDs. "
            "Rebuild from verified voter membership before inference; "
            "see docs/look/gold_voter_provenance.md."
        )
    table = pq.read_table(
        path, columns=["source_row_uid", "external_record_id", "payload"],
        filters=[("source_row_uid", "in", sorted(uids))],
    )
    rows = {str(row["source_row_uid"]): row for row in table.to_pylist()}
    missing = sorted(uids - set(rows))
    if require_complete and missing:
        raise ValueError(f"Selected evidence UIDs are absent: {missing[:5]}")
    return rows


def hydrate_uids(evidence_manifest: Path, uids: set[str]) -> dict[str, dict[str, Any]]:
    """Resolve a deduplicated UID set from one evidence-owned projection."""
    manifest = read_json(evidence_manifest)
    if (
        manifest.get("schema_version") != "ranked_evidence_projection.v1"
        or manifest.get("status") != "complete"
    ):
        raise ValueError(f"Incompatible evidence projection: {evidence_manifest}")
    return _hydrate(evidence_manifest.with_name(str(manifest["records"])), uids)


def load_candidates(
    queries: Mapping[str, str], *, task: str, subset: str, policy: Mapping[str, Any],
    molecule_limit: int = 10, l1_limit: int = 10,
    later_limit: int | Mapping[str, int] = 50, tie_seed: int = 0,
    cache_pool: str = "all",
    morgan_primary_parent_width: int = CAPACITY,
    preselected_uids: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    preselected_contexts: Mapping[str, Sequence[str]] | None = None,
    preselected_uid_budget: int | None = None,
    physical_selection: Sequence[Mapping[str, Any]] | None = None,
    **_: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Select ranks first, hydrate their UIDs once, and return prompt-ready rows."""
    if not queries or not 1 <= molecule_limit <= CAPACITY or l1_limit < 1:
        raise ValueError("Queries and positive L1 limits are required")
    if cache_pool != "all" or tie_seed != 0:
        raise ValueError("ranked_uid_retrieval.v1 supports only pool=all and tie_seed=0")
    stages = dict(policy["stages"])
    l1_selection = stages.get("L1")
    if l1_selection not in {"morgan", "assay_transfer", QUERY_PURE_METHOD}:
        raise ValueError("ranked_uid_retrieval.v1 does not support this L1 selection")
    limits = _limits(stages, later_limit)
    physical_context_uids: dict[tuple[str, str], list[str]] = defaultdict(list)
    if physical_selection is not None:
        from predict.retrieval.selection import resolve_selection
        resolved = resolve_selection(
            physical_selection, task=task, subset=subset, queries=queries,
            cache_manifests=policy["cache_manifests"], molecule_limit=molecule_limit,
            l1_limit=l1_limit, limits=limits,
        )
        preselected_contexts = {str(query_id): [] for query_id in queries}
        preselected_uids = {str(query_id): {level: [] for level in stages if level != "L1"}
                            for query_id in queries}
        for row in resolved:
            query_id, level = row["benchmark_row_id"], row["level"]
            if level == "L1":
                context = row["context_id"]
                if context not in preselected_contexts[query_id]:
                    preselected_contexts[query_id].append(context)
                physical_context_uids[(query_id, context)].append(row["source_row_uid"])
            else:
                preselected_uids[query_id][level].append(row["source_row_uid"])
    all_later_records = policy.get("all_later_records") is True and preselected_uids is None
    later_levels = {level for level in stages if level != "L1"}
    if preselected_uid_budget is not None and (preselected_uid_budget < 1 or preselected_uids is None):
        raise ValueError("Joint UID budget requires a positive budget and preselected UIDs")
    if preselected_uids is not None:
        if policy.get("allow_unscored_preselected") and (
            task != "carcinogens" or subset != "valid"
        ):
            raise ValueError("Unscored preselected records require Gold Carcinogens validation")
        if set(preselected_uids) != {str(query_id) for query_id in queries}:
            raise ValueError("Preselected UID queries must match requested queries exactly")
        for query_id, levels in preselected_uids.items():
            if set(levels) != later_levels:
                raise ValueError(f"Preselected UID levels differ for {query_id}")
            uids = [str(uid) for rows in levels.values() for uid in rows]
            if len(uids) != len(set(uids)) or any(not uid for uid in uids):
                raise ValueError(f"Preselected UIDs must be nonempty and globally unique: {query_id}")
            if preselected_uid_budget is not None:
                if len(uids) != preselected_uid_budget:
                    raise ValueError(f"Preselected UID budget differs for {query_id}")
            elif physical_selection is None and any(len(rows) != limits[level] for level, rows in levels.items()):
                raise ValueError(f"Preselected UID level limits differ for {query_id}")
    if preselected_contexts is not None:
        if set(preselected_contexts) != {str(query_id) for query_id in queries}:
            raise ValueError("Preselected context queries must match requested queries exactly")
        for query_id, context_ids in preselected_contexts.items():
            if ((physical_selection is None and len(context_ids) != molecule_limit)
                    or len(set(map(str, context_ids))) != len(context_ids)):
                raise ValueError(
                    f"Preselected contexts must contain exactly {molecule_limit} unique "
                    f"cards for {query_id}/L1"
                )
    manifests = {level: Path(path).resolve() for level, path in policy["cache_manifests"].items()}
    if set(manifests) != set(stages):
        raise ValueError("Independent caches do not cover the requested levels exactly")

    ranked: dict[str, dict[str, list[dict[str, Any]]]] = {}
    identities = None
    documents = {}
    for level, method in stages.items():
        rows, current, document = _ranked_rows(
            manifests[level], task=task, subset=subset, level=level, method=method,
            queries=queries,
            limit=(
                None if physical_selection is not None and level == "L1" else
                molecule_limit if level == "L1"
                else None if task == "bbb_martins" or all_later_records or preselected_uids is not None else limits[level]
            ),
            parent_morgan_width=(
                morgan_primary_parent_width
                if level != "L1" and method == "assay_transfer"
                and preselected_uids is None and not all_later_records else None
            ),
            selected_uids=(
                {
                    str(query_id): preselected_uids[str(query_id)][level]
                    for query_id in queries
                }
                if level != "L1" and preselected_uids is not None else None
            ),
            allow_unscored_preselected=(
                level != "L1" and (physical_selection is not None or bool(policy.get("allow_unscored_preselected")))
            ),
        )
        if identities is not None and (
            current.keys() != identities.keys()
            or any(
                current[query_id][1] != parent_smiles
                or (current[query_id][0] and parent_id
                    and current[query_id][0] != parent_id)
                for query_id, (parent_id, parent_smiles) in identities.items()
            )
        ):
            raise ValueError("Independent level caches disagree on query identity")
        if (all_later_records and level != "L1" and document.get("parent_universe_source")
                != f"shared_l2plus_morgan_{document['parent_capacity']}"):
            raise ValueError(f"All-record L2+ selection requires a shared-parent cache: {level}")
        ranked[level], identities, documents[level] = rows, current, document

    if preselected_contexts is not None:
        l1_database = manifests["L1"].with_name(str(documents["L1"]["database"]))
        with _open(l1_database) as connection:
            for query_id, ordered_contexts in preselected_contexts.items():
                placeholders = ",".join("?" for _ in ordered_contexts)
                rows = connection.execute(
                    "SELECT * FROM rankings WHERE benchmark_row_id=? AND "
                    f"(morgan_context_id IN ({placeholders}) OR assay_context_id IN ({placeholders}))",
                    (str(query_id), *map(str, ordered_contexts), *map(str, ordered_contexts)),
                ).fetchall()
                by_context: dict[str, tuple[sqlite3.Row, str]] = {}
                for row in rows:
                    for field, method in (("morgan_context_id", "morgan"),
                                          ("assay_context_id", "assay_transfer")):
                        context_id = row[field]
                        if context_id is not None and str(context_id) in ordered_contexts:
                            by_context[str(context_id)] = (row, method)
                missing = [context_id for context_id in ordered_contexts if context_id not in by_context]
                if missing:
                    raise ValueError(
                        f"Preselected contexts are absent from {query_id}/L1: {missing[:5]}"
                    )
                ranked["L1"][str(query_id)] = [
                    {**dict(by_context[str(context_id)][0]),
                     "_selection_method": by_context[str(context_id)][1],
                     "_selection_context_id": str(context_id)}
                    for context_id in ordered_contexts
                ]

    index_paths = {
        level: Path(path).resolve()
        for level, path in (
            policy.get("cache_indexes")
            or {level: policy["cache_index"] for level in stages}
        ).items()
    }
    if set(index_paths) != set(stages):
        raise ValueError("Cache indexes do not cover the requested levels exactly")
    indexes = {
        path: read_json(path)
        for path in set(index_paths.values())
    }
    for path, index in indexes.items():
        if (
            index.get("schema_version") != "ranked_uid_task_release_index.v1"
            or index.get("status") != "complete"
            or index.get("task_id") != task
        ):
            raise ValueError(f"Incompatible ranked UID task release index: {path}")
    for level, document in documents.items():
        entry = indexes[index_paths[level]]["splits"][subset]["levels"][level]
        if (
            sha256_file(manifests[level]) != entry["manifest_sha256"]
            or document["content_id"] != entry["content_id"]
            or document["parent_capacity"] != indexes[index_paths[level]]["parent_capacity"]
            or document.get("neighbor_identity_policy", "parent_disjoint")
            != indexes[index_paths[level]]["neighbor_identity_policy_by_level"][level]
        ):
            raise ValueError(f"{level} cache differs from the task release index")
    evidence_entries: dict[Path, Mapping[str, Any]] = {}
    evidence_paths_by_index: dict[Path, list[Path]] = {}
    for index_path, index in indexes.items():
        entries = index.get("evidence_sources") or [index["evidence"]]
        evidence_paths_by_index[index_path] = []
        for entry in entries:
            manifest_path = (index_path.parent / entry["manifest"]).resolve()
            evidence_entries[manifest_path] = entry
            evidence_paths_by_index[index_path].append(manifest_path)
    evidence_manifests = {}
    for manifest_path, entry in evidence_entries.items():
        manifest = read_json(manifest_path)
        if (
            manifest.get("schema_version") != "ranked_evidence_projection.v1"
            or manifest.get("status") != "complete"
            or manifest.get("task_id") != task
            or manifest.get("content_id") != entry["content_id"]
            or sha256_file(manifest_path) != entry["manifest_sha256"]
        ):
            raise ValueError("Evidence projection differs from its task release index")
        evidence_manifests[manifest_path] = manifest
    l1_manifest_path = manifests["L1"]
    l1_document = documents["L1"]
    l1_database = l1_manifest_path.with_name(str(l1_document["database"]))
    def selected_context_id(row: Mapping[str, Any]) -> str:
        if row.get("_selection_context_id") is not None:
            return str(row["_selection_context_id"])
        method = row.get("_selection_method", l1_selection)
        return str(row["morgan_context_id" if method == "morgan" else "assay_context_id"])

    candidate_contexts = {
        selected_context_id(row) for rows in ranked["L1"].values() for row in rows
    }
    with _open(l1_database) as connection:
        placeholders = ",".join("?" for _ in candidate_contexts)
        contexts = {
            str(row["context_id"]): dict(row)
            for row in connection.execute(
                f"SELECT * FROM contexts WHERE context_id IN ({placeholders})",
                sorted(candidate_contexts),
            )
        }
        if candidate_contexts != set(contexts):
            raise ValueError("Ranked L1 contexts are absent from the context ledger")
        chosen_contexts = {
            selected_context_id(row) for rows in ranked["L1"].values() for row in rows
        }
        contexts = {context_id: contexts[context_id] for context_id in chosen_contexts}
        placeholders = ",".join("?" for _ in chosen_contexts)
        context_uids: dict[str, list[str]] = defaultdict(list)
        for row in connection.execute(
            "SELECT context_id,source_row_uid FROM context_records "
            f"WHERE context_id IN ({placeholders}) "
            "ORDER BY context_id,within_context_rank",
            sorted(chosen_contexts),
        ):
            if len(context_uids[str(row["context_id"])]) < l1_limit:
                context_uids[str(row["context_id"])].append(str(row["source_row_uid"]))
    if chosen_contexts != set(contexts) or chosen_contexts != set(context_uids):
        raise ValueError("Selected L1 contexts do not resolve to frozen physical members")

    if physical_selection is not None:
        context_uids = defaultdict(list)
        for (_, context), uids in physical_context_uids.items():
            context_uids[context] = list(dict.fromkeys([*context_uids[context], *uids]))
    uids_by_index: dict[Path, set[str]] = defaultdict(set)
    uids_by_index[index_paths["L1"]].update(
        uid for rows in context_uids.values() for uid in rows
    )
    for level in stages:
        if level != "L1":
            uids_by_index[index_paths[level]].update(
                str(row["item_id"]) for rows in ranked[level].values() for row in rows
            )
    evidence_by_index: dict[Path, dict[str, dict[str, Any]]] = {}
    for index_path, selected_uids in uids_by_index.items():
        evidence: dict[str, dict[str, Any]] = {}
        for manifest_path in evidence_paths_by_index[index_path]:
            manifest = evidence_manifests[manifest_path]
            rows = _hydrate(
                manifest_path.with_name(str(manifest["records"])),
                selected_uids,
                require_complete=False,
            )
            duplicates = set(evidence) & set(rows)
            if duplicates:
                raise ValueError(
                    f"Selected evidence UIDs resolve from multiple sources in {index_path}: "
                    f"{sorted(duplicates)[:5]}"
                )
            evidence.update(rows)
        missing_uids = selected_uids - set(evidence)
        if missing_uids:
            raise ValueError(f"Selected evidence UIDs are absent from {index_path}: {sorted(missing_uids)[:5]}")
        evidence_by_index[index_path] = evidence

    eligibility_audits = {}
    bbb_policy = {}
    if task == "bbb_martins":
        from predict.tasks.bbb_martins import EVIDENCE_POLICY, prediction_only_uids
        bbb_policy = {"bbb_evidence_policy": EVIDENCE_POLICY}
        excluded = set()
        for index_path, evidence in evidence_by_index.items():
            later_uids = {str(row["item_id"]) for level in stages if level != "L1"
                          and index_paths[level] == index_path
                          for rows in ranked[level].values() for row in rows}
            excluded.update(prediction_only_uids({uid: evidence[uid] for uid in later_uids}))
        for level in stages:
            if level == "L1":
                continue
            for query_id, rows in ranked[level].items():
                eligible = [row for row in rows if str(row["item_id"]) not in excluded]
                removed = len(rows) - len(eligible)
                if removed and preselected_uids is not None:
                    invalid = [str(row["item_id"]) for row in rows if str(row["item_id"]) in excluded]
                    raise ValueError(f"Prediction-only BBB UIDs are ineligible: {query_id}/{level}: {invalid[:5]}")
                eligibility_audits[(level, query_id)] = {
                    "prediction_only_excluded": removed, "eligible_records": len(eligible),
                }
                if not all_later_records and preselected_uids is None:
                    if (len(eligible) < limits[level] and not (
                            documents[level].get("score_coverage") == "partial_snapshot"
                            and stages[level] == "assay_transfer")):
                        raise ValueError(f"Requested {limits[level]} rows but {query_id}/{level} has {len(eligible)} eligible records after BBB prediction-only exclusion")
                    eligible = eligible[:limits[level]]
                ranked[level][query_id] = eligible

    molecules_by_query: dict[str, list[dict[str, Any]]] = {}
    later_by_query: dict[str, dict[str, Any]] = {}
    query_audits: dict[str, Any] = {}
    for query_id in queries:
        method = l1_selection
        molecules = []
        for selection_rank, row in enumerate(ranked["L1"][str(query_id)]):
            row_method = row.get("_selection_method", method)
            context_id = selected_context_id(row)
            member_count = int(row["morgan_member_count"] if row_method == "morgan" else row["assay_member_count"])
            context = contexts.get(context_id)
            if context is None or str(context["parent_id"]) != str(row["parent_id"]):
                raise ValueError(f"Frozen L1 context identity mismatch: {context_id}")
            records = [
                _payload(evidence_by_index[index_paths["L1"]][uid], row, row_method)
                for uid in (physical_context_uids[(str(query_id), context_id)]
                            if physical_selection is not None else context_uids[context_id])
            ]
            if documents["L1"].get("molecular_representation_policy") == MOLECULAR_REPRESENTATION_POLICY:
                assert_disjoint(queries[query_id], row["canonical_smiles"], "scaffold_disjoint")
                for record in records:
                    assert_disjoint(queries[query_id], source_smiles(record["payload"]), "scaffold_disjoint")
            molecule = {
                "reference_molecule_id": str(row.get("canonical_smiles") or row["parent_id"]),
                "canonical_smiles": str(row.get("canonical_smiles") or row["parent_smiles"]),
                "morgan_similarity": float(row["morgan_similarity"]),
                "morgan_rank": int(row["morgan_rank"]),
                "assay_rank": (
                    int(row["assay_rank"])
                    if row["assay_rank"] is not None else None
                ),
                "selection_rank": selection_rank,
                "available_l1": member_count,
                "available_l2": 0,
                "ranking_method": row_method,
                "l1_records": records,
                "l2_records": [],
                "context_card_id": context_id,
                "selected_condition": "" if context["condition_group"] == "no_reported_external_condition" else str(context["condition_group"]),
                "_diagnostic_context_id": context_id,
                "_diagnostic_parent_id": str(row["parent_id"]),
                "gold_context_ids": {
                    "morgan": str(row["morgan_context_id"]),
                    "assay_transfer": (
                        str(row["assay_context_id"])
                        if row["assay_context_id"] is not None else None
                    ),
                },
            }
            if int(context["gold_label"]) in {0, 1}:
                molecule["_diagnostic_label"] = int(context["gold_label"])
            elif any(
                int(record.get("payload", {}).get("Y", -1)) in {0, 1}
                for record in records
            ):
                molecule["visible_training_labels"] = sorted({
                    int(record["payload"]["Y"])
                    for record in records
                    if int(record.get("payload", {}).get("Y", -1)) in {0, 1}
                })
            if row_method == "assay_transfer":
                molecule["transfer_likelihood"] = float(row["assay_transfer_score"])
            molecules.append(molecule)
        later = {}
        level_audits = {"L1": {
            "candidate_records": sum(
                int(row["morgan_member_count"] if row.get("_selection_method", method) == "morgan" else row["assay_member_count"])
                for row in ranked["L1"][str(query_id)]
            ),
            "candidate_molecules": CAPACITY,
            "selected_records": sum(len(molecule["l1_records"]) for molecule in molecules),
            "selected_molecules": len(molecules),
            **({"selection_mode": molecules[0]["ranking_method"]}
               if l1_selection == QUERY_PURE_METHOD else {}),
        }}
        for level, level_method in stages.items():
            if level == "L1":
                continue
            records = []
            selected_method = (
                ranked[level][str(query_id)][0].get("_selection_method", level_method)
                if ranked[level][str(query_id)] else level_method
            )
            for row in ranked[level][str(query_id)]:
                record = _payload(
                    evidence_by_index[index_paths[level]][str(row["item_id"])],
                    row, row.get("_selection_method", selected_method),
                )
                if documents[level].get("molecular_representation_policy") == MOLECULAR_REPRESENTATION_POLICY:
                    assert_disjoint(queries[query_id], source_smiles(record["payload"]),
                                    documents[level]["neighbor_identity_policy"])
                records.append(record)
            counts = documents[level]["query_counts"][str(query_id)]
            partial_level = documents[level].get("score_coverage") == "partial_snapshot"
            hybrid_level = level_method == HYBRID_METHOD
            hybrid_scored = sum(
                row["assay_transfer_score"] is not None
                for row in ranked[level][str(query_id)]
            )
            score_complete = hybrid_scored == int(counts["candidate_records"])
            later[level] = {
                "records": records,
                **eligibility_audits.get((level, str(query_id)), {}),
                "ranking_method": selected_method,
                "available_record_count": int(
                    counts["scored_records"] if partial_level and level_method == "assay_transfer"
                    and not policy.get("allow_unscored_preselected")
                    else counts["candidate_records"]
                ),
                "allow_shortfall": partial_level and level_method == "assay_transfer",
            }
            if (level, str(query_id)) in eligibility_audits and preselected_uids is None:
                later[level]["available_record_count"] = eligibility_audits[(level, str(query_id))]["eligible_records"]
            level_audits[level] = {
                "candidate_records": int(counts["candidate_records"]),
                **eligibility_audits.get((level, str(query_id)), {}),
                **({"scored_records": int(counts["scored_records"]),
                    "missing_scores": int(counts["missing_scores"])} if partial_level else {}),
                "candidate_molecules": int(counts["candidate_parents"]),
                "selected_records": len(records),
                "selected_molecules": len({row["parent_id"] for row in ranked[level][str(query_id)]}),
                **({
                    "scored_records": hybrid_scored,
                    "missing_scores": int(counts["candidate_records"]) - hybrid_scored,
                    "selection_mode": (
                        "assay_transfer"
                        if score_complete else "scored_assay_then_unscored_morgan"
                    ),
                    "prompt_assay_scores": "visible" if score_complete else "hidden",
                } if hybrid_level else {}),
                **({"selection_mode": selected_method} if level_method == QUERY_PURE_METHOD else {}),
            }
        molecules_by_query[str(query_id)] = molecules
        later_by_query[str(query_id)] = later
        query_audits[str(query_id)] = level_audits

    content_ids = {
        level: document["content_id"] for level, document in documents.items()
    }
    capacities = {
        level: (
            CAPACITY
            if level == "L1"
            else min(
                int(
                    counts["scored_records"]
                    if document.get("score_coverage") == "partial_snapshot"
                    and stages[level] == "assay_transfer"
                    else counts["candidate_records"]
                )
                for counts in document["query_counts"].values()
            )
        )
        for level, document in documents.items()
    }
    return molecules_by_query, later_by_query, {
        "selection_policy": SCHEMA_VERSION,
        "selection_contract": SCHEMA_VERSION,
        "inputs": dict(policy.get("inputs") or {}),
        "contract": {
            "policy": dict(policy),
            **bbb_policy,
            "molecule_limit": molecule_limit,
            "l1_limit": l1_limit,
            "later_limits": limits,
            "preselected_uid_budget": preselected_uid_budget,
            "all_later_records": all_later_records,
            "min_contrast": 0,
            "morgan_primary_parent_width": (
                morgan_primary_parent_width
                if any(
                    level != "L1" and method == "assay_transfer"
                    for level, method in stages.items()
                ) else None
            ),
            "tie_seed": tie_seed,
            "cache_pool": cache_pool,
            "cache_content_ids": content_ids,
            "later_selection": (
                "preselected_uid_order" if preselected_uids is not None else "cached_rank"
            ),
            "l1_selection": (
                "preselected_context_order"
                if preselected_contexts is not None else "cached_rank"
            ),
        },
        "pool": "all",
        "cache_pool": "all",
        "parent_capacity": CAPACITY,
        "cache_index": str(index_paths["L1"]),
        "cache_indexes": {level: str(path) for level, path in index_paths.items()},
        "cache_capacities": capacities,
        "cache_content_ids": content_ids,
        "evidence_manifest": str(evidence_paths_by_index[index_paths["L1"]][0]),
        "evidence_manifests": sorted(str(path) for path in evidence_manifests),
        "evidence_manifests_by_level": {
            level: [str(path) for path in evidence_paths_by_index[index_paths[level]]]
            for level in stages
        },
        "neighbor_identity_policy_by_level": {
            level: indexes[index_paths[level]]["neighbor_identity_policy_by_level"][level]
            for level in stages
        },
        "neighbor_identity_policy": "level_specific_disjoint",
        "similarity_floor": None,
        "scaffold_overlap": 0,
        "parent_overlap": 0,
        "query_identities": {
            query_id: {"parent_id": value[0], "parent_smiles": value[1]}
            for query_id, value in (identities or {}).items()
        },
        "query_audits": query_audits,
    }


__all__ = [
    "CAPACITY", "SCHEMA_VERSION", "hydrate_uids", "load_candidates",
    "load_ranked_panels", "load_ranked_universe",
]
