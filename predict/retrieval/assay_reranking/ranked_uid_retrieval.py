"""Select ranked UIDs, hydrate them once, and return prompt-ready evidence."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq

from predict.utils.json import sha256_file


SCHEMA_VERSION = "ranked_uid_retrieval.v1"
CAPACITY = 100
CONTRAST_WIDTHS = (15, 25, 50, 100)


def _open(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)
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
    document = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "task_id": task,
        "subset": subset,
        "level": level,
        "parent_capacity": CAPACITY,
        "pool": "fixed" if level == "L1" else "all",
    }
    if any(document.get(key) != value for key, value in required.items()):
        raise ValueError(f"Incompatible ranked UID cache: {path}")
    return document


def _ranked_rows(
    manifest_path: Path, *, task: str, subset: str, level: str, method: str,
    queries: Mapping[str, str], limit: int | None,
    parent_morgan_width: int | None = None,
    selected_uids: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, tuple[str, str]], dict[str, Any]]:
    manifest = _manifest(manifest_path, task, subset, level)
    database = manifest_path.with_name(str(manifest["database"]))
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: dict[str, tuple[str, str]] = {}
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
            if selected_uids is not None:
                ordered_uids = [str(uid) for uid in selected_uids[str(query_id)]]
                if not ordered_uids or len(set(ordered_uids)) != len(ordered_uids):
                    raise ValueError(f"Preselected UIDs must be nonempty and unique: {query_id}/{level}")
                placeholders = ",".join("?" for _ in ordered_uids)
                rows = connection.execute(
                    f"SELECT * FROM rankings WHERE benchmark_row_id=? "
                    f"AND {rank_column} IS NOT NULL AND item_id IN ({placeholders})",
                    (str(query_id), *ordered_uids),
                ).fetchall()
                by_uid = {str(row["item_id"]): row for row in rows}
                missing = [uid for uid in ordered_uids if uid not in by_uid]
                if missing:
                    raise ValueError(
                        f"Preselected UIDs are absent from {query_id}/{level}/{method}: "
                        f"{missing[:5]}"
                    )
                rows = [by_uid[uid] for uid in ordered_uids]
            else:
                rows = connection.execute(
                    f"{sql} LIMIT ?", (*parameters, limit),
                ).fetchall() if limit is not None else connection.execute(
                    sql, parameters,
                ).fetchall()
            if limit is not None and len(rows) != limit:
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
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if (
        index.get("schema_version") != "ranked_uid_task_release_index.v1"
        or index.get("status") != "complete"
        or index.get("task_id") != task
        or index.get("pool") != "all"
        or index.get("later_candidate_universe")
        != "all_uids_under_morgan_top_100_parents"
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
            or index["neighbor_identity_policy_by_level"].get(level) != "parent_disjoint"
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
    evidence_manifest = json.loads(evidence_manifest_path.read_text(encoding="utf-8"))
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
    if payload.get("label_source") == "tdc_v1":
        source_fields = dict(payload.get("source_fields") or {})
        payload.setdefault("family_key", "tdc_training_labels")
        payload.setdefault("source_id", "tdc_v1")
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
    payload["canonical_smiles"] = parent_smiles
    payload["reference_parent_smiles"] = parent_smiles
    result = {
        "record_id": str(row["external_record_id"]),
        "reference_molecule_id": str(ranked["parent_id"]),
        "reference_parent_smiles": parent_smiles,
        "morgan_similarity": similarity,
        "morgan_rank": int(ranked["morgan_rank"]),
        "assay_rank": int(ranked["assay_rank"]) if ranked["assay_rank"] is not None else None,
        "ranking_method": method,
        "payload": payload,
    }
    if method == "assay_transfer":
        score = ranked["assay_transfer_score"]
        if score is None or not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
            raise ValueError("Invalid cached assay-transfer score")
        result["transfer_likelihood"] = float(score)
    return result


def _select_assay_contrastive(
    rows: list[dict[str, Any]], contexts: Mapping[str, Mapping[str, Any]], *,
    molecule_limit: int, primary_width: int, min_contrast: int,
    context_scores: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank assay scores inside a Morgan pool, widening only for label balance."""
    if primary_width not in CONTRAST_WIDTHS or min_contrast < 0:
        raise ValueError("Unsupported Morgan width or negative contrast")
    if molecule_limit < 2 * min_contrast:
        raise ValueError("Contrastive L1 requires K >= 2*M")

    def label(row: Mapping[str, Any]) -> int:
        context = contexts.get(str(row["assay_context_id"]))
        if context is None:
            raise ValueError("Assay-ranked L1 context is missing")
        return int(context["gold_label"])

    primary = [row for row in rows if int(row["morgan_rank"]) <= primary_width]
    if len(primary) < molecule_limit:
        raise ValueError(f"Morgan top-{primary_width} has fewer than K candidates")
    assay_key = lambda row: (int(row["assay_rank"]), str(row["item_id"]))
    selected = sorted(primary, key=assay_key)[:molecule_limit]
    counts = Counter(label(row) for row in selected)
    base_by_parent = {
        str(row.get("parent_id", row["item_id"])): row for row in rows
    }
    alternatives = []
    for score in context_scores:
        base = base_by_parent.get(str(score["parent_id"]))
        if base is None:
            continue
        alternative = dict(base)
        alternative.update({
            "assay_context_id": str(score["context_id"]),
            "assay_transfer_score": float(score["assay_transfer_score"]),
            "assay_rank": int(score["assay_rank"]),
            "assay_member_count": int(score["member_count"]),
            "score_key": str(score["score_key"]),
        })
        alternatives.append(alternative)
    candidates_by_score = alternatives or rows
    score_key = lambda row: (
        -float(row["assay_transfer_score"]), str(row["assay_context_id"])
    )
    candidate_key = score_key if alternatives else assay_key
    replacements = []
    fallback_widths = [width for width in CONTRAST_WIDTHS if width >= primary_width]
    for missing_label in (0, 1):
        while counts[missing_label] < min_contrast:
            added = None
            effective_width = None
            for width in fallback_widths:
                candidates = [
                    row for row in candidates_by_score
                    if int(row["morgan_rank"]) <= width
                    and label(row) == missing_label
                    and not any(
                        str(current["item_id"]) == str(row["item_id"])
                        and str(current["assay_context_id"]) == str(row["assay_context_id"])
                        for current in selected
                    )
                ]
                if candidates:
                    added = min(candidates, key=candidate_key)
                    effective_width = width
                    break
            if added is None:
                raise ValueError("Morgan top-100 cannot satisfy the requested label contrast")
            removed = next(
                (row for row in selected if str(row["item_id"]) == str(added["item_id"])),
                None,
            )
            if removed is None:
                removable = [
                    row for row in selected
                    if label(row) != missing_label and counts[label(row)] > min_contrast
                ]
                if not removable:
                    raise ValueError("Cannot satisfy the requested binary label balance")
                removed = max(removable, key=candidate_key)
            elif counts[label(removed)] <= min_contrast:
                raise ValueError("Cannot satisfy the requested binary label balance")
            selected[selected.index(removed)] = added
            counts[label(removed)] -= 1
            counts[missing_label] += 1
            replacements.append({
                "added_context_id": str(added["assay_context_id"]),
                "added_label": missing_label,
                "added_parent_rank": int(added["morgan_rank"]),
                "effective_parent_width": effective_width,
                "removed_context_id": str(removed["assay_context_id"]),
                "removed_label": label(removed),
            })
    selected.sort(key=candidate_key)
    return selected, {
        "morgan_primary_parent_width": primary_width,
        "morgan_fallback_parent_width": CAPACITY,
        "morgan_candidate_parent_widths": fallback_widths,
        "candidate_contexts_primary": len(primary),
        "selected_label_counts": {str(value): counts[value] for value in (0, 1)},
        "replacement_count": len(replacements),
        "fallback_beyond_primary": any(
            replacement["added_parent_rank"] > primary_width
            for replacement in replacements
        ),
        "replacements": replacements,
    }


def _hydrate(
    path: Path, uids: set[str], *, require_complete: bool = True
) -> dict[str, dict[str, Any]]:
    if not uids:
        return {}
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
    manifest = json.loads(evidence_manifest.read_text(encoding="utf-8"))
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
    cache_pool: str = "all", min_contrast: int = 3,
    morgan_primary_parent_width: int = CAPACITY,
    preselected_uids: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    **_: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Select ranks first, hydrate their UIDs once, and return prompt-ready rows."""
    if not queries or not 1 <= molecule_limit <= CAPACITY or l1_limit < 1:
        raise ValueError("Queries and positive L1 limits are required")
    if cache_pool != "all" or tie_seed != 0:
        raise ValueError("ranked_uid_retrieval.v1 supports only pool=all and tie_seed=0")
    stages = dict(policy["stages"])
    l1_selection = stages.get("L1")
    if l1_selection not in {"morgan", "assay_transfer", "assay_transfer_contrastive"}:
        raise ValueError("ranked_uid_retrieval.v1 does not support this L1 selection")
    limits = _limits(stages, later_limit)
    later_levels = {level for level in stages if level != "L1"}
    if preselected_uids is not None:
        if set(preselected_uids) != {str(query_id) for query_id in queries}:
            raise ValueError("Preselected UID queries must match requested queries exactly")
        for query_id, levels in preselected_uids.items():
            if set(levels) != later_levels:
                raise ValueError(f"Preselected UID levels differ for {query_id}")
            for level, uids in levels.items():
                if len(uids) != limits[level] or len(set(map(str, uids))) != len(uids):
                    raise ValueError(
                        f"Preselected UIDs must contain exactly {limits[level]} unique "
                        f"records for {query_id}/{level}"
                    )
    manifests = {level: Path(path).resolve() for level, path in policy["cache_manifests"].items()}
    if set(manifests) != set(stages):
        raise ValueError("Independent caches do not cover the requested levels exactly")

    ranked: dict[str, dict[str, list[dict[str, Any]]]] = {}
    identities = None
    documents = {}
    for level, method in stages.items():
        read_method = (
            "morgan" if level == "L1" and method == "assay_transfer_contrastive"
            else method
        )
        rows, current, document = _ranked_rows(
            manifests[level], task=task, subset=subset, level=level, method=read_method,
            queries=queries,
            limit=(
                CAPACITY
                if level == "L1" and method == "assay_transfer_contrastive"
                else molecule_limit if level == "L1"
                else None if preselected_uids is not None else limits[level]
            ),
            parent_morgan_width=(
                morgan_primary_parent_width
                if level != "L1" and method == "assay_transfer"
                and preselected_uids is None else None
            ),
            selected_uids=(
                {
                    str(query_id): preselected_uids[str(query_id)][level]
                    for query_id in queries
                }
                if level != "L1" and preselected_uids is not None else None
            ),
        )
        if identities is not None and current != identities:
            raise ValueError("Independent level caches disagree on query identity")
        ranked[level], identities, documents[level] = rows, current, document

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
        path: json.loads(path.read_text(encoding="utf-8"))
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
        ):
            raise ValueError(f"{level} cache differs from the task release index")
    evidence_entries: dict[Path, Mapping[str, Any]] = {}
    for index_path, index in indexes.items():
        entries = index.get("evidence_sources") or [index["evidence"]]
        for entry in entries:
            manifest_path = (index_path.parent / entry["manifest"]).resolve()
            evidence_entries[manifest_path] = entry
    evidence_manifests = {}
    for manifest_path, entry in evidence_entries.items():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    context_field = "morgan_context_id" if l1_selection == "morgan" else "assay_context_id"
    candidate_contexts = {
        str(row[context_field]) for rows in ranked["L1"].values() for row in rows
    }
    contrast_audits: dict[str, Any] = {}
    with _open(l1_database) as connection:
        context_scores: dict[str, list[dict[str, Any]]] = {}
        if (
            l1_selection == "assay_transfer_contrastive"
            and l1_document.get("l1_context_score_contract") == "l1_context_scores.v1"
        ):
            for query_id, rows in ranked["L1"].items():
                parents = {str(row["parent_id"]) for row in rows}
                placeholders = ",".join("?" for _ in parents)
                context_scores[query_id] = [
                    dict(row) for row in connection.execute(
                        "SELECT * FROM context_scores WHERE benchmark_row_id=? "
                        f"AND parent_id IN ({placeholders}) ORDER BY assay_rank,context_id",
                        (query_id, *sorted(parents)),
                    )
                ]
                candidate_contexts.update(
                    str(row["context_id"]) for row in context_scores[query_id]
                )
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
        if l1_selection == "assay_transfer_contrastive":
            for query_id, rows in ranked["L1"].items():
                ranked["L1"][query_id], contrast_audits[query_id] = (
                    _select_assay_contrastive(
                        rows,
                        contexts,
                        molecule_limit=molecule_limit,
                        primary_width=morgan_primary_parent_width,
                        min_contrast=min_contrast,
                        context_scores=context_scores.get(query_id, ()),
                    )
                )
        chosen_contexts = {
            str(row[context_field]) for rows in ranked["L1"].values() for row in rows
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

    all_uids = {uid for rows in context_uids.values() for uid in rows}
    for level in stages:
        if level != "L1":
            all_uids.update(str(row["item_id"]) for rows in ranked[level].values() for row in rows)
    evidence: dict[str, dict[str, Any]] = {}
    for manifest_path, manifest in evidence_manifests.items():
        rows = _hydrate(
            manifest_path.with_name(str(manifest["records"])),
            all_uids,
            require_complete=False,
        )
        duplicates = set(evidence) & set(rows)
        if duplicates:
            raise ValueError(f"Selected evidence UIDs resolve from multiple sources: {sorted(duplicates)[:5]}")
        evidence.update(rows)
    missing_uids = all_uids - set(evidence)
    if missing_uids:
        raise ValueError(f"Selected evidence UIDs are absent: {sorted(missing_uids)[:5]}")

    molecules_by_query: dict[str, list[dict[str, Any]]] = {}
    later_by_query: dict[str, dict[str, Any]] = {}
    query_audits: dict[str, Any] = {}
    for query_id in queries:
        method = "assay_transfer" if l1_selection == "assay_transfer_contrastive" else l1_selection
        molecules = []
        for selection_rank, row in enumerate(ranked["L1"][str(query_id)]):
            context_id = str(row["morgan_context_id"] if method == "morgan" else row["assay_context_id"])
            member_count = int(row["morgan_member_count"] if method == "morgan" else row["assay_member_count"])
            context = contexts.get(context_id)
            if context is None or str(context["parent_id"]) != str(row["parent_id"]):
                raise ValueError(f"Frozen L1 context identity mismatch: {context_id}")
            records = [_payload(evidence[uid], row, method) for uid in context_uids[context_id]]
            molecule = {
                "reference_molecule_id": str(row["parent_id"]),
                "canonical_smiles": str(row["parent_smiles"]),
                "morgan_similarity": float(row["morgan_similarity"]),
                "morgan_rank": int(row["morgan_rank"]),
                "assay_rank": (
                    int(row["assay_rank"])
                    if row["assay_rank"] is not None else None
                ),
                "selection_rank": selection_rank,
                "available_l1": member_count,
                "available_l2": 0,
                "ranking_method": method,
                "l1_records": records,
                "l2_records": [],
                "context_card_id": context_id,
                "selected_condition": "" if context["condition_group"] == "no_reported_external_condition" else str(context["condition_group"]),
                "_diagnostic_label": int(context["gold_label"]),
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
            if method == "assay_transfer":
                molecule["transfer_likelihood"] = float(row["assay_transfer_score"])
            molecules.append(molecule)
        later = {}
        level_audits = {"L1": {
            "candidate_records": sum(
                int(row["morgan_member_count"] if method == "morgan" else row["assay_member_count"])
                for row in ranked["L1"][str(query_id)]
            ),
            "candidate_molecules": CAPACITY,
            "selected_records": sum(len(molecule["l1_records"]) for molecule in molecules),
            "selected_molecules": len(molecules),
        }}
        if query_id in contrast_audits:
            level_audits["L1"].update(contrast_audits[query_id])
        for level, level_method in stages.items():
            if level == "L1":
                continue
            records = []
            for row in ranked[level][str(query_id)]:
                record = _payload(evidence[str(row["item_id"])], row, level_method)
                records.append(record)
            counts = documents[level]["query_counts"][str(query_id)]
            later[level] = {
                "records": records,
                "ranking_method": level_method,
                "available_record_count": int(counts["candidate_records"]),
                "allow_shortfall": False,
            }
            level_audits[level] = {
                "candidate_records": int(counts["candidate_records"]),
                "candidate_molecules": int(counts["candidate_parents"]),
                "selected_records": len(records),
                "selected_molecules": len({row["parent_id"] for row in ranked[level][str(query_id)]}),
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
                int(counts["candidate_records"])
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
            "molecule_limit": molecule_limit,
            "l1_limit": l1_limit,
            "later_limits": limits,
            "min_contrast": min_contrast if l1_selection == "assay_transfer_contrastive" else 0,
            "morgan_primary_parent_width": (
                morgan_primary_parent_width
                if l1_selection == "assay_transfer_contrastive"
                or any(
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
        },
        "pool": "all",
        "cache_pool": "all",
        "parent_capacity": CAPACITY,
        "cache_index": str(index_paths["L1"]),
        "cache_indexes": {level: str(path) for level, path in index_paths.items()},
        "cache_capacities": capacities,
        "cache_content_ids": content_ids,
        "evidence_manifest": str(next(iter(evidence_manifests))),
        "evidence_manifests": [str(path) for path in evidence_manifests],
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
