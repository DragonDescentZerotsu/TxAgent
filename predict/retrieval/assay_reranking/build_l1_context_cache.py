"""Publish validation-only assay-ranked condition-context caches."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any

import pyarrow.parquet as pq

from data.processing.gold_labels.conditioned_benchmark import split_path
from predict.harnesses.progressive._records import (
    BIOAVAILABILITY_CLAIMS,
    BIOAVAILABILITY_CLAIMS_MANIFEST,
    _bioavailability_source_id,
)
from predict.retrieval.policies import normalize_molecule_identity
from predict.retrieval.assay_reranking.runtime import cache_profile_root
from predict.utils.json import read_jsonl, sha256_file

from . import v9 as direct_gold
from .l1_context_cache import ACTIVE_CONTEXT_SCHEMA_VERSION as SCHEMA_VERSION, load_candidates


TASKS = ("bbb_martins", "bioavailability_ma")
LINEAGES = ("v9", "v10_4")
V9_ROOT = cache_profile_root("v9_direct_gold_morgan100")
V10_4_ROOT = cache_profile_root("v10_4_direct_gold_morgan100_v1")
GOLD_TASK_NAMES = {
    "bbb_martins": "BBB_Martins",
    "bioavailability_ma": "Bioavailability_Ma",
}
PUBLISH_V10_RECEIPT = (
    Path(__file__).resolve().parents[4]
    / "starling_assay_transfer/assay_transfer/context_conditioned/results/publish_v10_receipt.json"
)
PUBLISHED_RELEASES = {
    "bbb_martins": ("BBB_Martins", "d32ed5ed7423f7c182e65e42f8d15c50458ccdc5"),
    "bioavailability_ma": (
        "Bioavailability_Ma", "5b9d51e22a2dc6876a00705d7e185ba2a175f4a4"
    ),
}


def _profile_root(primary_width: int, lineage: str) -> Path:
    profile = (
        f"l1_context_morgan{primary_width}_v2"
        if lineage == "v9"
        else f"l1_context_morgan{primary_width}_v10_4_gold_v2_v1"
    )
    return cache_profile_root(profile)


def _analysis_root(primary_width: int, lineage: str) -> Path:
    suffix = "contrastive_v2" if lineage == "v9" else "v10_4_gold_v2_v1"
    return Path(
        f"outputs/analysis/prediction/l1_context_morgan{primary_width}_{suffix}"
    )


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=DELETE;
        CREATE TABLE metadata(key TEXT PRIMARY KEY NOT NULL,value TEXT NOT NULL);
        CREATE TABLE queries(
          query_id INTEGER PRIMARY KEY,query_parent_id TEXT NOT NULL,
          query_parent_smiles TEXT NOT NULL
        );
        CREATE TABLE benchmark_queries(
          benchmark_row_id TEXT PRIMARY KEY NOT NULL,drug TEXT NOT NULL,
          query_id INTEGER NOT NULL REFERENCES queries(query_id)
        );
        CREATE TABLE gold_contexts(
          context_key INTEGER PRIMARY KEY,external_context_id TEXT UNIQUE NOT NULL,
          parent_id TEXT NOT NULL,parent_smiles TEXT NOT NULL,
          condition_group TEXT NOT NULL,condition_scope TEXT NOT NULL,
          condition_atoms TEXT NOT NULL,
          gold_label INTEGER NOT NULL CHECK(gold_label IN (0,1)),
          available_voter_count INTEGER NOT NULL CHECK(available_voter_count > 0)
        );
        CREATE TABLE records(
          record_key INTEGER PRIMARY KEY,external_record_id TEXT UNIQUE NOT NULL,
          payload TEXT NOT NULL
        );
        CREATE TABLE context_records(
          context_key INTEGER NOT NULL REFERENCES gold_contexts(context_key),
          record_key INTEGER NOT NULL REFERENCES records(record_key),
          within_context_rank INTEGER NOT NULL,
          PRIMARY KEY(context_key,record_key)
        );
        CREATE TABLE candidates(
          query_id INTEGER NOT NULL REFERENCES queries(query_id),
          context_key INTEGER NOT NULL REFERENCES gold_contexts(context_key),
          morgan_parent_rank INTEGER NOT NULL,
          morgan_similarity REAL NOT NULL,
          transfer_score REAL NOT NULL,
          transfer_rank_top25 INTEGER,
          transfer_rank_top100 INTEGER NOT NULL,
          PRIMARY KEY(query_id,context_key)
        );
        CREATE INDEX candidates_query_score
          ON candidates(query_id,transfer_score DESC,transfer_rank_top100);
        CREATE INDEX context_records_context_rank
          ON context_records(context_key,within_context_rank);
        """
    )


def _inputs(task: str, lineage: str) -> dict[str, Path]:
    ranking = (V9_ROOT if lineage == "v9" else V10_4_ROOT) / task / "scaffold/valid/VERSION.json"
    if lineage == "v9":
        gold_root = split_path(task, "train").parent
    else:
        gold_root = Path("data/gold_labels") / GOLD_TASK_NAMES[task] / "v2/scaffold"
    return {
        "ranking_version": ranking,
        "rankings": ranking.with_name("rankings.parquet"),
        "train_gold": gold_root / "train_molecule_condition_labels.jsonl",
        "valid_gold": gold_root / "valid_molecule_condition_labels.jsonl",
        **({
            "voter_membership": gold_root / "voter_membership.parquet",
            "voter_membership_manifest": gold_root / "voter_membership.manifest.json",
        } if lineage == "v10_4" else {}),
    }


def _context_by_uid(
    task: str, gold: dict[str, dict[str, Any]], paths: dict[str, Path], lineage: str,
) -> tuple[dict[str, str], set[str], dict[str, Path]]:
    from predict.retrieval.assay_reranking import three_pools

    stage3 = Path(three_pools.MODULES[task].STAGE3)
    uid_to_context: dict[str, str] = {}
    provenance: dict[str, Path] = {"v10_stage3": stage3}
    if lineage == "v10_4":
        membership_path = paths["voter_membership"]
        manifest_path = paths["voter_membership_manifest"]
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("schema_version") != "gold_voter_membership.v1"
                or manifest.get("path") != membership_path.name
                or manifest.get("sha256") != sha256_file(membership_path)):
            raise ValueError(f"Incompatible gold-v2 voter membership: {task}")
        rows = pq.read_table(
            membership_path,
            columns=[
                "task", "source_row_uid", "molecule_identity_key",
                "condition_group", "condition_atoms", "aggregate_label",
                "benchmark_row_id",
            ],
            filters=[("split", "=", "train"), ("aggregate_status", "=", "published")],
        ).to_pylist()
        expected_task = GOLD_TASK_NAMES[task]
        for row in rows:
            context_id = str(row["benchmark_row_id"])
            context = gold.get(context_id)
            if (row["task"] != expected_task or context is None
                    or str(row["molecule_identity_key"]) != str(context["molecule_identity_key"])
                    or str(row["condition_group"]) != str(context["condition_group"])
                    or list(row["condition_atoms"] or []) != list(context.get("condition_atoms") or [])
                    or int(row["aggregate_label"]) != int(context["Y"])):
                raise ValueError(f"Gold-v2 voter identity differs from its context: {context_id}")
            uid = str(row["source_row_uid"])
            prior = uid_to_context.setdefault(uid, context_id)
            if prior != context_id:
                raise ValueError(f"Gold-v2 voter UID maps to multiple contexts: {uid}")
        if set(uid_to_context.values()) != set(gold):
            raise ValueError(f"Gold-v2 training contexts lack published voters: {task}")
        return uid_to_context, set(), provenance
    if task == "bbb_martins":
        by_index = {}
        destinations = {}
        for context_id, row in gold.items():
            key = (str(row["molecule_identity_key"]), str(row["condition_group"]))
            if key in destinations:
                raise ValueError(f"Duplicate BBB destination context identity: {key}")
            destinations[key] = context_id
            for source_id in row.get("source_record_ids") or []:
                tail = str(source_id).rsplit(":", 1)[-1]
                if not tail.isdigit() or int(tail) in by_index:
                    raise ValueError(f"Invalid or duplicate BBB source index: {source_id!r}")
                by_index[int(tail)] = context_id
        rows = pq.read_table(
            stage3,
            columns=["source_row_uid", "source_index", "canonical_smiles"],
        ).to_pylist()
        source_references = 0
        for row in rows:
            index = row.get("source_index")
            if index is None or int(index) not in by_index:
                continue
            origin = gold[by_index[int(index)]]
            identity = normalize_molecule_identity(str(row["canonical_smiles"]))
            parent_id = identity.parent_inchi_key or identity.parent_smiles
            context_id = destinations.get((str(parent_id), str(origin["condition_group"])))
            if context_id is None:
                raise ValueError("BBB voter has no active parent-condition destination")
            uid = str(row["source_row_uid"])
            prior = uid_to_context.setdefault(uid, context_id)
            if prior != context_id:
                raise ValueError(f"BBB voter UID maps to multiple contexts: {uid}")
            source_references += 1
        retired = set(gold) - set(uid_to_context.values())
        v3_audit = (
            cache_profile_root("cache_matched_retrieval_v3")
            / task / "scaffold/valid/VERSION.json"
        )
        receipt = json.loads(v3_audit.read_text())["legacy_selection_audit"][
            "gold_context_mapping"
        ]
        if (source_references != int(receipt["source_references"])
                or len(set(uid_to_context.values())) != int(receipt["destination_contexts"])
                or retired != set(map(str, receipt["retired_contexts"]))):
            raise ValueError("Reconstructed BBB active-context mapping differs from its frozen audit")
        provenance["bbb_context_mapping_audit"] = v3_audit
    else:
        columns = [
            "source_row_uid", "source_id", "source_index",
            "source_row_number", "source_record_id",
        ]
        available = set(pq.read_schema(stage3).names)
        rows = pq.read_table(
            stage3, columns=[column for column in columns if column in available]
        ).to_pylist()
        claims = pq.read_table(
            BIOAVAILABILITY_CLAIMS,
            columns=["canonical_claim_id", "source_record_ids"],
        ).to_pylist()
        claim_to_context = {}
        for context_id, row in gold.items():
            for claim_id in row.get("source_record_ids") or []:
                prior = claim_to_context.setdefault(str(claim_id), context_id)
                if prior != context_id:
                    raise ValueError(f"Bioavailability claim maps to multiple contexts: {claim_id}")
        source_to_claim = {}
        for row in claims:
            claim_id = str(row["canonical_claim_id"])
            if claim_id not in claim_to_context:
                continue
            for source_id in row.get("source_record_ids") or []:
                prior = source_to_claim.setdefault(str(source_id), claim_id)
                if prior != claim_id:
                    raise ValueError(f"Bioavailability source maps to multiple claims: {source_id}")
        for row in rows:
            claim_id = source_to_claim.get(_bioavailability_source_id(row))
            if claim_id in claim_to_context:
                uid_to_context[str(row["source_row_uid"])] = claim_to_context[claim_id]
        retirement = stage3.parents[1] / "reviews/l1_training_context_retirement/manifest.json"
        document = json.loads(retirement.read_text())
        if (document.get("version") != "l1_training_context_retirement.v1"
                or document.get("task_id") != task
                or document.get("gold_train_sha256") != sha256_file(
                    split_path(task, "train").with_name("train_molecule_condition_labels.jsonl")
                )
                or document.get("stage3_records_sha256") != sha256_file(stage3)):
            raise ValueError("Incompatible oral active-context retirement receipt")
        retired = {str(row["context_id"]) for row in document["contexts"]}
        provenance.update(
            bioavailability_claims=BIOAVAILABILITY_CLAIMS,
            bioavailability_claims_manifest=BIOAVAILABILITY_CLAIMS_MANIFEST,
            bioavailability_context_retirement=retirement,
        )
    if not uid_to_context:
        raise ValueError(f"No exact voter provenance mapped for {task}")
    if retired & set(uid_to_context.values()):
        raise ValueError(f"Retired contexts retain active voter mappings: {task}")
    return uid_to_context, retired, provenance


def _source_catalog(
    task: str, paths: dict[str, Path], lineage: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    from predict.retrieval.assay_reranking import three_pools

    gold_rows = read_jsonl(paths["train_gold"])
    gold = {str(row["benchmark_row_id"]): row for row in gold_rows}
    if len(gold) != len(gold_rows):
        raise ValueError(f"Duplicate training context IDs: {task}")
    published_task, published_commit = PUBLISHED_RELEASES[task]
    releases = json.loads(PUBLISH_V10_RECEIPT.read_text())["releases"]
    matches = [row for row in releases if row.get("task_id") == published_task]
    if (len(matches) != 1 or matches[0].get("version") != "v10"
            or matches[0].get("commit") != published_commit
            or matches[0].get("upload_commit") != published_commit):
        raise ValueError(f"Published V10 release receipt differs for {task}")
    uid_to_context, retired, provenance = _context_by_uid(task, gold, paths, lineage)
    module = three_pools.MODULES[task]
    stage3 = Path(module.STAGE3)
    mapping = Path(module.LEVEL_MAPPING)
    source_contract = Path(module.SOURCE_CONTRACT)
    level_rows = pq.read_table(
        mapping, columns=["source_row_uid", "level", "family_key"],
        filters=[("level", "=", 1)],
    ).to_pylist()
    families = {str(row["source_row_uid"]): str(row["family_key"]) for row in level_rows}
    if len(families) != len(level_rows):
        raise ValueError(f"V10 L1 membership contains duplicate voter UIDs: {task}")
    non_l1_source_rows = len(set(uid_to_context) - set(families))
    uid_to_context = {
        uid: context_id for uid, context_id in uid_to_context.items() if uid in families
    }
    contract = json.loads(source_contract.read_text())
    if contract.get("contract_version") not in {
        "source_column_contract.v1", "source_column_contract.v2"
    }:
        raise ValueError(f"Unsupported V10 source contract: {task}")
    fields = {
        source_id: [
            name for name, spec in source["normalized_artifact_columns"].items()
            if spec.get("source_or_simply_cleaned") is True
        ]
        for source_id, source in contract["sources"].items()
    }
    table = pq.read_table(
        stage3, filters=[("source_row_uid", "in", sorted(uid_to_context))]
    )
    if table.num_rows != len(uid_to_context):
        raise ValueError(f"V10 Stage 3 does not contain every mapped voter UID: {task}")
    if task == "bbb_martins":
        from data.processing.evidence_library.versions.v10.tasks.bbb_martins import (
            semantic_display,
        )
    records_by_context: dict[str, list[tuple[str, str]]] = defaultdict(list)
    seen_record_ids: set[str] = set()
    for row in table.to_pylist():
        uid = str(row["source_row_uid"])
        context_id = uid_to_context[uid]
        source_id = str(row["source_id"])
        values = {name: row.get(name) for name in fields[source_id]}
        if task == "bbb_martins":
            values = semantic_display.semantic_prompt_payload(row, values)
        record_id = str(row["canonical_record_id"])
        if record_id in seen_record_ids:
            raise ValueError(f"Duplicate mapped voter record: {record_id}")
        seen_record_ids.add(record_id)
        payload = {
            "record_id": record_id,
            "source_row_uid": uid,
            "task_id": task,
            "source_id": source_id,
            "progressive_level": "L1",
            "family_key": families[uid],
            "canonical_smiles": str(gold[context_id]["drug"]),
            "source_canonical_smiles": str(row["canonical_smiles"]),
            "source_fields": values,
            "source_contract": {
                "contract_version": contract["contract_version"],
                "source_or_simply_cleaned": {name: True for name in values},
            },
            "source_projection": "library_source_contract.v2",
            "measurement_kind": row["measurement_kind"],
        }
        records_by_context[context_id].append((record_id, _json(payload)))
    for rows in records_by_context.values():
        rows.sort(key=lambda item: item[0])
    active = set(gold) - retired
    missing = active - set(records_by_context)
    if missing:
        raise ValueError(f"Active training contexts lack exact voters: {sorted(missing)[:10]}")
    valid_rows = read_jsonl(paths["valid_gold"])
    queries = {}
    for query_id, row in enumerate(valid_rows, 1):
        identity = normalize_molecule_identity(str(row["drug"]))
        if identity.status != "ok" or not identity.parent_smiles:
            raise ValueError(f"Invalid validation query parent: {row['benchmark_row_id']}")
        benchmark_row_id = str(row["benchmark_row_id"])
        queries[benchmark_row_id] = {
            "query_id": query_id,
            "drug": str(row["drug"]),
            "query_parent_id": identity.parent_inchi_key or identity.parent_smiles,
            "query_parent_smiles": identity.parent_smiles,
        }
    provenance.update(
        v10_level_mapping=mapping,
        v10_source_contract=source_contract,
        publish_v10_receipt=PUBLISH_V10_RECEIPT,
    )
    return {
        "gold": gold,
        "records_by_context": dict(records_by_context),
        "retired_contexts": retired,
        "queries": queries,
        "source_audit": {
            "training_contexts": len(gold),
            "active_contexts": len(active),
            "retired_contexts": len(retired),
            "exact_voter_records": len(seen_record_ids),
            "gold_source_rows_outside_l1": non_l1_source_rows,
            "published_dataset": {
                "repo_id": matches[0]["repo_id"],
                "commit": published_commit,
                "rows": matches[0]["rows"],
            },
        },
    }, provenance


def _candidate_rows(
    task: str, paths: dict[str, Path], catalog: dict[str, Any], *,
    primary_width: int, context_capacity: int, lineage: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ranking = json.loads(paths["ranking_version"].read_text())
    expected_policy = "morgan_top100_distinct_gold_train_parents_then_all_context_rows"
    if (ranking.get("status") != "complete"
            or ranking.get("candidate_policy") != expected_policy
            or ranking.get("rankings_sha256") != sha256_file(paths["rankings"])):
        raise ValueError(f"Incompatible {lineage} Morgan-100 rankings: {task}")
    if lineage == "v10_4" and (
            ranking.get("schema_version") != direct_gold.RANKING_SCHEMA_VERSION
            or ranking.get("model_lineage") != lineage
            or ranking.get("model") != direct_gold.model_profile(task, lineage)
            or ranking.get("reference_provenance", {})
            != direct_gold.reference_provenance(task, lineage)
            or ranking.get("morgan_pool_size") != 100):
        raise ValueError(f"Incompatible {lineage} ranking provenance: {task}")
    expected_release = "v2" if lineage == "v10_4" else "v1"
    if ranking.get("inputs", {}).get("gold_release", expected_release) != expected_release:
        raise ValueError(f"Incompatible {lineage} gold release: {task}")
    columns = [
        "query_record_id", "retrieval_record_id", "retrieval_molecule_identity_key",
        "retrieval_condition_group", "retrieval_gold_Y", "morgan_tanimoto_similarity",
        "retrieval_parent_rank", "model_rank", "prob_transfer",
    ]
    ranking_rows = pq.read_table(paths["rankings"], columns=columns).to_pylist()
    query_rows = catalog["queries"]
    if {str(row["query_record_id"]) for row in ranking_rows} != set(query_rows):
        raise ValueError(f"{lineage} queries differ from the selected valid ledger: {task}")
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    discarded = Counter()
    for row in ranking_rows:
        query_id = str(row["query_record_id"])
        context_id = str(row["retrieval_record_id"])
        gold = catalog["gold"].get(context_id)
        if gold is None:
            raise ValueError(f"Unknown {lineage} training context: {context_id}")
        if not catalog["records_by_context"].get(context_id):
            if context_id not in catalog["retired_contexts"]:
                raise ValueError(f"Active {lineage} context has no exact voters: {context_id}")
            discarded["retired_context"] += 1
            continue
        parent_id = str(row["retrieval_molecule_identity_key"])
        if (parent_id != str(gold["molecule_identity_key"])
                or str(row["retrieval_condition_group"]) != str(gold["condition_group"])
                or int(row["retrieval_gold_Y"]) != int(gold["Y"])):
            raise ValueError(f"{lineage} context identity or label mismatch: {context_id}")
        similarity = float(row["morgan_tanimoto_similarity"])
        parent_rank = int(row["retrieval_parent_rank"]) + 1
        score = float(row["prob_transfer"])
        if not 0 <= score <= 1:
            raise ValueError(f"Invalid {lineage} transfer score: {query_id}/{context_id}")
        by_query[query_id].append({
            "query_id": query_rows[query_id]["query_id"], "context_id": context_id,
            "parent_id": parent_id, "parent_rank": parent_rank,
            "raw_parent_rank": parent_rank,
            "similarity": similarity, "score": score, "model_rank": int(row["model_rank"]),
        })
    output = []
    active_parent_counts = []
    for benchmark_row_id in sorted(query_rows):
        rows = by_query[benchmark_row_id]
        if len({row["context_id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate query-context candidates: {benchmark_row_id}")
        parent_identity = {}
        for row in rows:
            identity = (
                row["raw_parent_rank"]
                if lineage == "v10_4"
                else (row["raw_parent_rank"], row["similarity"])
            )
            prior = parent_identity.setdefault(row["parent_id"], identity)
            if prior != identity:
                raise ValueError(
                    f"Inconsistent {lineage} parent rank: {benchmark_row_id}/{row['parent_id']}"
                )
        active_parents = sorted(
            parent_identity,
            key=lambda parent: (
                parent_identity[parent]
                if lineage == "v10_4"
                else parent_identity[parent][0],
                parent,
            ),
        )
        active_parent_counts.append(len(active_parents))
        if primary_width < 100 and len(active_parents) < primary_width:
            raise ValueError(
                f"Fewer than {primary_width} active parents in frozen Morgan-100: "
                f"{benchmark_row_id}"
            )
        active_ranks = {parent: rank for rank, parent in enumerate(active_parents, 1)}
        for row in rows:
            row["parent_rank"] = active_ranks[row["parent_id"]]
        ranked100 = sorted(rows, key=lambda row: (-row["score"], row["model_rank"], row["context_id"]))
        ranks100 = {row["context_id"]: rank for rank, row in enumerate(ranked100, 1)}
        ranked25 = [row for row in ranked100 if row["parent_rank"] <= 25]
        ranks25 = {row["context_id"]: rank for rank, row in enumerate(ranked25, 1)}
        primary = [row for row in ranked100 if row["parent_rank"] <= primary_width]
        if len(primary) < context_capacity:
            raise ValueError(
                f"Fewer than {context_capacity} active contexts in Morgan top-{primary_width}: "
                f"{benchmark_row_id}"
            )
        label_counts = Counter(int(catalog["gold"][row["context_id"]]["Y"]) for row in ranked100)
        if min(label_counts[0], label_counts[1]) < 3:
            raise ValueError(f"Morgan top-100 cannot supply three contexts per label: {benchmark_row_id}")
        for row in ranked100:
            output.append({
                **row,
                "transfer_rank_top25": ranks25.get(row["context_id"]),
                "transfer_rank_top100": ranks100[row["context_id"]],
            })
    return output, {
        "discarded_candidate_occurrences": dict(discarded),
        "parent_rank_policy": "compact_surviving_parents_from_frozen_morgan100",
        "source_catalog": catalog["source_audit"],
        "candidate_occurrences": len(output),
        "queries": len(query_rows),
        "active_parent_counts": {
            "minimum": min(active_parent_counts),
            "maximum": max(active_parent_counts),
            "queries_below_primary_width": sum(
                count < primary_width for count in active_parent_counts
            ),
        },
    }


def build(
    task: str, *, primary_width: int = 25, context_capacity: int = 10,
    root: Path | None = None, lineage: str = "v9",
) -> Path:
    """Build one immutable task/valid cache and return its VERSION path."""
    if task not in TASKS:
        raise ValueError(f"Unsupported task: {task}")
    if lineage not in LINEAGES:
        raise ValueError(f"Unsupported direct-gold lineage: {lineage}")
    if primary_width not in {10, 15, 25, 50, 100}:
        raise ValueError("Morgan primary width must be 10, 15, 25, 50, or 100")
    if not 1 <= context_capacity <= primary_width:
        raise ValueError("Context capacity must be positive and fit the primary width")
    root = root or _profile_root(primary_width, lineage)
    destination = root / task / "scaffold/valid"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite immutable cache: {destination}")
    paths = _inputs(task, lineage)
    catalog, extra_inputs = _source_catalog(task, paths, lineage)
    candidates, build_audit = _candidate_rows(
        task, paths, catalog, primary_width=primary_width,
        context_capacity=context_capacity, lineage=lineage,
    )
    used_context_ids = {row["context_id"] for row in candidates}
    contexts = {context_id: catalog["gold"][context_id] for context_id in used_context_ids}
    input_paths = {
        **paths,
        **extra_inputs,
        "builder": Path(__file__),
        "runtime_selector": Path(__file__).with_name("l1_context_cache.py"),
    }
    input_receipts = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in sorted(input_paths.items())
    }
    candidate_widths = {
        10: [10, 15, 25, 100],
        15: [15, 25, 100],
        25: [25, 100],
        50: [50, 100],
        100: [100],
    }[primary_width]
    content_id = hashlib.sha256(_json({
        "schema": SCHEMA_VERSION, "task": task,
        "inputs": {name: item["sha256"] for name, item in input_receipts.items()},
        "policy": {
            "primary": primary_width,
            "candidate_widths": candidate_widths,
            "context_capacity": context_capacity,
        },
    }).encode()).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    database = temporary / "retrieval.sqlite3"
    try:
        with sqlite3.connect(database) as connection:
            _schema(connection)
            connection.executemany("INSERT INTO metadata VALUES (?,?)", [
                ("schema_version", SCHEMA_VERSION), ("content_id", content_id),
            ])
            query_rows = {
                benchmark_row_id: (
                    row["query_id"], row["drug"], row["query_parent_id"],
                    row["query_parent_smiles"],
                )
                for benchmark_row_id, row in catalog["queries"].items()
            }
            connection.executemany("INSERT INTO queries VALUES (?,?,?)", [
                (query_id, parent_id, parent_smiles)
                for query_id, _, parent_id, parent_smiles in query_rows.values()
            ])
            connection.executemany("INSERT INTO benchmark_queries VALUES (?,?,?)", [
                (benchmark_row_id, drug, query_id)
                for benchmark_row_id, (query_id, drug, _, _) in query_rows.items()
            ])
            context_keys = {context_id: key for key, context_id in enumerate(sorted(contexts), 1)}
            connection.executemany("INSERT INTO gold_contexts VALUES (?,?,?,?,?,?,?,?,?)", [
                (
                    context_keys[context_id], context_id, str(row["molecule_identity_key"]),
                    str(row["drug"]),
                    str(row["condition_group"]), str(row["condition_scope"]),
                    _json(row.get("condition_atoms") or []), int(row["Y"]),
                    len(catalog["records_by_context"][context_id]),
                )
                for context_id, row in sorted(contexts.items())
            ])
            all_records = {
                external_id: payload
                for context_id in contexts
                for external_id, payload in catalog["records_by_context"][context_id]
            }
            record_keys = {record_id: key for key, record_id in enumerate(sorted(all_records), 1)}
            connection.executemany("INSERT INTO records VALUES (?,?,?)", [
                (record_keys[record_id], record_id, all_records[record_id])
                for record_id in sorted(all_records)
            ])
            connection.executemany("INSERT INTO context_records VALUES (?,?,?)", [
                (context_keys[context_id], record_keys[record_id], rank)
                for context_id in sorted(contexts)
                for rank, (record_id, _) in enumerate(
                    catalog["records_by_context"][context_id], 1
                )
            ])
            connection.executemany("INSERT INTO candidates VALUES (?,?,?,?,?,?,?)", [
                (
                    row["query_id"], context_keys[row["context_id"]], row["parent_rank"],
                    row["similarity"], row["score"], row["transfer_rank_top25"],
                    row["transfer_rank_top100"],
                )
                for row in candidates
            ])
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Published SQLite integrity check failed")
        os.chmod(database, 0o444)
        manifest = {
            "schema_version": SCHEMA_VERSION, "status": "complete", "task_id": task,
            "subset": "valid", "database": database.name, "content_id": content_id,
            "model_lineage": lineage,
            "gold_release": "v2" if lineage == "v10_4" else "v1",
            "tie_seed": 0, "morgan_primary_parent_width": primary_width,
            "morgan_fallback_parent_width": 100,
            "morgan_candidate_parent_widths": candidate_widths,
            "morgan_parent_rank_policy": (
                "compact_surviving_parents_from_frozen_morgan100"
            ),
            "selection_unit": "parent_condition_context",
            "condition_identity_fields": [
                "condition_group", "condition_scope", "condition_atoms"
            ],
            "label_source": "exact_condition_row_gold_Y_hidden_at_runtime",
            "record_membership": "full_v10_active_context_exact_voter_provenance",
            "query_target_stored": False,
            "capacities": {
                "l1_contexts": context_capacity, "voter_records_per_context": 10
            },
            "assignment_counts": {
                "queries": len(query_rows), "contexts": len(contexts),
                "candidate_occurrences": len(candidates),
                "context_record_memberships": sum(
                    len(catalog["records_by_context"][context_id]) for context_id in contexts
                ),
            },
            "record_count": len(all_records), "inputs": input_receipts,
            "build_audit": build_audit,
        }
        (temporary / "VERSION.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.chmod(temporary / "VERSION.json", 0o444)
        temporary.replace(destination)
    except Exception:
        for path in temporary.iterdir():
            path.unlink()
        temporary.rmdir()
        raise
    return destination / "VERSION.json"


def write_reports(
    manifests: dict[str, Path], *, primary_width: int = 25, output: Path | None = None,
    lineage: str = "v9",
) -> None:
    output = output or _analysis_root(primary_width, lineage)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite analysis artifact: {output}")
    output.mkdir(parents=True)
    details = []
    summaries = []
    modes = {"assay-transfer-contrastive": "assay_transfer_contrastive"}
    minimum_contrasts = (0, 1, 2) if lineage == "v9" else (1, 2, 3)
    for task, manifest_path in manifests.items():
        manifest = json.loads(manifest_path.read_text())
        database = manifest_path.with_name(manifest["database"])
        with sqlite3.connect(database) as connection:
            queries = {
                str(benchmark_row_id): str(drug)
                for benchmark_row_id, drug in connection.execute(
                    "SELECT benchmark_row_id,drug FROM benchmark_queries"
                )
            }
        for mode, method in modes.items():
            for minimum in minimum_contrasts:
                policy = {
                    "selection_contract": SCHEMA_VERSION, "reranking": mode,
                    "stages": {"L1": method}, "cache_manifest": str(manifest_path),
                    "inputs": {},
                }
                molecules, _, audit = load_candidates(
                    queries, task=task, subset="valid", policy=policy,
                    molecule_limit=int(manifest["capacities"]["l1_contexts"]),
                    l1_limit=10, min_contrast=minimum,
                )
                for query_id, selected in molecules.items():
                    item = audit["query_audits"][query_id]["L1"]
                    labels = item["selected_label_counts"]
                    details.append({
                        "task": task, "query_record_id": query_id, "mode": mode,
                        "minimum_contrast": minimum,
                        "label_0_count": labels["0"], "label_1_count": labels["1"],
                        "minority_fraction": min(labels.values()) / len(selected),
                        "unique_parent_count": item["selected_unique_parents"],
                        "repeated_parent_cards": item["selected_repeated_parent_cards"],
                        "replacement_count": item["replacement_count"],
                        "fallback_beyond_primary": item["fallback_beyond_primary"],
                    })
                group = [row for row in details if row["task"] == task
                         and row["mode"] == mode
                         and row["minimum_contrast"] == minimum]
                summaries.append({
                    "task": task, "mode": mode, "minimum_contrast": minimum,
                    "n_queries": len(group),
                    "mean_label_0_fraction": sum(
                        row["label_0_count"] / manifest["capacities"]["l1_contexts"]
                        for row in group
                    ) / len(group),
                    "mean_label_1_fraction": sum(
                        row["label_1_count"] / manifest["capacities"]["l1_contexts"]
                        for row in group
                    ) / len(group),
                    "mean_minority_fraction": sum(
                        row["minority_fraction"] for row in group
                    ) / len(group),
                    "queries_rebalanced": sum(row["replacement_count"] > 0 for row in group),
                    "queries_using_fallback": sum(row["fallback_beyond_primary"] for row in group),
                    "mean_unique_parent_count": sum(
                        row["unique_parent_count"] for row in group
                    ) / len(group),
                })
    for filename, rows in (("summary.tsv", summaries), ("per_query.tsv", details)):
        with (output / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    receipt = {
        "schema_version": f"l1_context_morgan{primary_width}_contrastive_analysis.v2",
        "status": "complete", "query_target_read": False,
        "cache_manifests": {
            task: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for task, path in manifests.items()
        },
        "outputs": {
            name: sha256_file(output / name) for name in ("summary.tsv", "per_query.tsv")
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument(
        "--morgan-primary-parent-width", type=int,
        choices=(10, 15, 25, 50, 100), default=25
    )
    parser.add_argument("--context-capacity", type=int, default=10)
    parser.add_argument("--lineage", choices=LINEAGES, default="v9")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--analysis-output", type=Path)
    args = parser.parse_args(argv)
    manifests = {
        task: build(
            task, primary_width=args.morgan_primary_parent_width,
            context_capacity=args.context_capacity, root=args.root,
            lineage=args.lineage,
        )
        for task in dict.fromkeys(args.tasks)
    }
    write_reports(
        manifests,
        primary_width=args.morgan_primary_parent_width,
        output=args.analysis_output, lineage=args.lineage,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
