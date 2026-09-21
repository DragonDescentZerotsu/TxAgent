"""Publish parent-disjoint V1 rankings backed by the current V10 evidence library."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Iterable

import pyarrow.parquet as pq
from rdkit import DataStructs

from predict.retrieval.policies import (
    normalize_molecule_identity,
    seeded_rank_tie_key,
    standardize_smiles_and_fp,
)
from predict.utils.json import read_jsonl, sha256_file

from . import three_pools
from .build_cache_matched_v3 import L5_RELEASES
from .ranked_retrieval import FIXED_POOL, POOLS, SCHEMA_VERSION
from .runtime import cache_profile_root


REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILE = "ranked_evidence_retrieval_parent_v1"
DEFAULT_OUTPUT_ROOT = cache_profile_root(PROFILE)
L1_ROOT = cache_profile_root("v10_3_best_parent_morgan100_v1")
GOLD_NAMES = {"bbb_martins": "BBB_Martins", "bioavailability_ma": "Bioavailability_Ma"}
L1_PARENT_CAPACITY = 100
L1_RECORD_CAPACITY = 10
LATER_RECORD_CAPACITY = 50


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_release_index(task: str, output_root: Path) -> dict[str, Any]:
    """Refresh the small task-level navigation index atomically."""
    task_root = output_root.resolve() / task
    evidence_manifest = task_root / "evidence/VERSION.json"
    splits = {}
    for subset in ("valid", "test"):
        path = task_root / "scaffold" / subset / "VERSION.json"
        if path.is_file():
            manifest = json.loads(path.read_text())
            splits[subset] = {
                "manifest": str(path.relative_to(task_root)),
                "manifest_sha256": sha256_file(path),
                "content_id": manifest["content_id"],
            }
    evidence = json.loads(evidence_manifest.read_text())
    index = {
        "schema_version": "ranked_evidence_task_release_index.v1",
        "profile": PROFILE,
        "task_id": task,
        "status": "complete" if set(splits) == {"valid", "test"} else "partial",
        "gold_release": "v1",
        "evidence_release": "v10_current",
        "neighbor_identity_policy": "l1_voter_membership_later_parent_disjoint",
        "evidence": {
            "manifest": str(evidence_manifest.relative_to(task_root)),
            "manifest_sha256": sha256_file(evidence_manifest),
            "content_id": evidence["content_id"],
        },
        "splits": splits,
    }
    temporary = task_root / ".RELEASE_INDEX.json.tmp"
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, task_root / "RELEASE_INDEX.json")
    return index


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _present(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _display_fields(task: str, record: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Apply semantic display, then canonical-pair/source-pair atomic fallback."""
    scale = str(record.get("canonical_measurement_scale_id") or "")
    category = record.get("canonical_category_id")
    if task == "bbb_martins" and scale:
        from data.processing.evidence_library.versions.v10.tasks.bbb_martins.semantic_display import (
            SEMANTIC_DISPLAY_VERSION,
            semantic_display_fields,
        )

        semantic = semantic_display_fields(record)
        return {
            "measurement_text": semantic["measurement_text"],
            "unit_text": semantic["unit_text"],
            "origin": "semantic_category",
            "adapter": SEMANTIC_DISPLAY_VERSION,
        }
    oral = {
        ("direct_oral_bioavailability_ordinal.v1", "low"): ("low", "oral bioavailability category"),
        ("direct_oral_bioavailability_ordinal.v1", "middle"): ("middle", "oral bioavailability category"),
        ("direct_oral_bioavailability_ordinal.v1", "high"): ("high", "oral bioavailability category"),
        ("fg_substrate_status_binary.v1", "not_substrate"): ("not substrate", "transporter substrate status"),
        ("fg_substrate_status_binary.v1", "substrate"): ("substrate", "transporter substrate status"),
    }
    if task == "bioavailability_ma" and scale:
        try:
            measurement, unit = oral[scale, str(category)]
        except KeyError as exc:
            raise ValueError(f"Unsupported oral semantic display: {scale}/{category}") from exc
        return {
            "measurement_text": measurement, "unit_text": unit,
            "origin": "semantic_category", "adapter": "oral_semantic_display.v1",
        }

    canonical = (
        record.get("canonical_measurement_text"), record.get("canonical_unit_text")
    )
    source_pair = (source.get("measurement_text"), source.get("unit_text"))
    if all(_present(value) for value in canonical):
        measurement, unit = canonical
        origin = "canonical_pair"
    elif all(_present(value) for value in source_pair):
        measurement, unit = source_pair
        origin = "source_pair"
    elif str(record.get("measurement_kind") or "") == "non_scalar":
        result_fields = (
            "measurement_text", "reported_result", "bbb_permeability_label",
            "bbb_transport_label", "interaction_conclusion",
            "passive_bbb_interpretation", "substrate_status", "support_text",
            "extra_details", "effect_direction", "result_interpretation",
            "classification_label", "persistence_outcome", "result_call",
            "result_direction", "result_status", "maximum_reported_severity",
        )
        measurement = next((source.get(name) for name in result_fields if _present(source.get(name))), None)
        unit = None
        origin = "source_non_scalar_text" if measurement is not None else "unavailable"
    else:
        measurement = unit = None
        origin = "unavailable"
    return {
        "measurement_text": None if measurement is None else str(measurement),
        "unit_text": None if unit is None else str(unit),
        "origin": origin,
        "adapter": "canonical_first_atomic_pair.v1",
    }


def _evidence_schema(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        PRAGMA journal_mode=DELETE;
        PRAGMA user_version=1;
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE records(
            source_row_uid TEXT PRIMARY KEY,
            external_record_id TEXT NOT NULL UNIQUE,
            parent_id TEXT NOT NULL,
            parent_smiles TEXT NOT NULL,
            level TEXT NOT NULL,
            tool_accepted INTEGER NOT NULL,
            tool_compatible INTEGER NOT NULL,
            payload TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE context_records(
            context_id TEXT NOT NULL,
            source_row_uid TEXT NOT NULL,
            within_context_rank INTEGER NOT NULL,
            PRIMARY KEY(context_id,source_row_uid)
        ) WITHOUT ROWID;
        CREATE TABLE contexts(
            context_id TEXT PRIMARY KEY,
            parent_id TEXT NOT NULL,
            condition_group TEXT NOT NULL,
            gold_label INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE INDEX records_level_parent ON records(level,parent_id);
        CREATE INDEX context_records_context ON context_records(context_id,within_context_rank);
    """)


def _ranking_schema(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        PRAGMA journal_mode=DELETE;
        PRAGMA user_version=1;
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE queries(
            query_id INTEGER PRIMARY KEY,
            query_parent_id TEXT NOT NULL,
            query_parent_smiles TEXT NOT NULL
        );
        CREATE TABLE benchmark_queries(
            benchmark_row_id TEXT PRIMARY KEY,
            drug TEXT NOT NULL,
            query_id INTEGER NOT NULL UNIQUE
        ) WITHOUT ROWID;
        CREATE TABLE rankings(
            query_id INTEGER NOT NULL,
            pool TEXT NOT NULL,
            level TEXT NOT NULL,
            source_row_uid TEXT NOT NULL,
            parent_id TEXT NOT NULL,
            parent_smiles TEXT NOT NULL,
            context_id TEXT,
            within_parent_rank INTEGER NOT NULL,
            morgan_similarity REAL NOT NULL,
            morgan_rank INTEGER NOT NULL,
            assay_transfer_score REAL,
            assay_rank INTEGER,
            PRIMARY KEY(query_id,pool,level,source_row_uid)
        ) WITHOUT ROWID;
        CREATE TABLE selection_counts(
            query_id INTEGER NOT NULL,
            pool TEXT NOT NULL,
            level TEXT NOT NULL,
            candidate_record_count INTEGER NOT NULL,
            candidate_parent_count INTEGER NOT NULL,
            PRIMARY KEY(query_id,pool,level)
        ) WITHOUT ROWID;
        CREATE TABLE l1_parent_contexts(
            query_id INTEGER NOT NULL,
            parent_id TEXT NOT NULL,
            morgan_context_id TEXT NOT NULL,
            assay_context_id TEXT NOT NULL,
            PRIMARY KEY(query_id,parent_id)
        ) WITHOUT ROWID;
        CREATE INDEX rankings_morgan ON rankings(query_id,pool,level,morgan_rank);
        CREATE INDEX rankings_assay ON rankings(query_id,pool,level,assay_rank);
        CREATE INDEX rankings_parent ON rankings(query_id,pool,level,parent_id);
    """)


def _source_fields(task: str) -> tuple[dict[str, list[str]], set[str]]:
    contract = json.loads(three_pools.MODULES[task].SOURCE_CONTRACT.read_text())
    fields: dict[str, list[str]] = {}
    union: set[str] = set()
    for source, spec in contract["sources"].items():
        names = [
            name for name, value in spec["normalized_artifact_columns"].items()
            if value.get("source_or_simply_cleaned") is True
        ]
        fields[source] = names
        union.update(names)
    return fields, union


def _accepted_buckets(task: str) -> dict[str, set[str]]:
    accepted: dict[str, set[str]] = defaultdict(set)
    for level, profile in three_pools.MODULES[task].MODELS.items():
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(
            profile["dataset"], repo_type="dataset", revision=profile["dataset_revision"],
            local_files_only=True,
        ))
        buckets = json.loads((root / "calibration.json").read_text())["accepted_buckets"]
        accepted[level].update(key for key, value in buckets.items() if value.get("level") == level)
    l5_root = L5_RELEASES[task]
    buckets = json.loads((l5_root / "calibration.json").read_text())["accepted_buckets"]
    accepted["L5"].update(key for key, value in buckets.items() if value.get("level") == "L5")
    return dict(accepted)


def build_evidence(task: str, output_root: Path) -> dict[str, Any]:
    """Publish one current-V10 evidence index shared by valid and test."""
    module = three_pools.MODULES[task]
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to replace evidence index: {output_root}")
    fields, source_union = _source_fields(task)
    mapping_rows = pq.read_table(
        module.LEVEL_MAPPING, columns=["source_row_uid", "canonical_record_id", "level", "family_key"]
    ).to_pylist()
    mapping = {
        str(row["source_row_uid"]): (f"L{int(row['level'])}", str(row["family_key"]), str(row["canonical_record_id"]))
        for row in mapping_rows
    }
    if len(mapping) != len(mapping_rows):
        raise ValueError("Duplicate current V10 level-mapping UID")
    accepted = _accepted_buckets(task)
    core = {
        "source_row_uid", "canonical_record_id", "canonical_smiles", "source_id",
        "pair_bucket_key", "measurement_kind", "finite_scalar_value",
        "canonical_pair_fields_json", "canonical_measurement_scale_id",
        "canonical_category_id", "categorical_encoder_id",
        "canonical_measurement_text", "canonical_unit_text",
        "canonical_transporter_identifier",
    }
    schema = set(pq.read_schema(module.STAGE3).names)
    columns = sorted((core | source_union) & schema)
    gold_root = REPO_ROOT / "data/gold_labels" / GOLD_NAMES[task] / "v1/scaffold"
    membership_path = gold_root / "voter_membership.parquet"
    input_hashes = {
        "records": sha256_file(module.STAGE3),
        "level_mapping": sha256_file(module.LEVEL_MAPPING),
        "level_manifest": sha256_file(module.LEVEL_MANIFEST),
        "source_contract": sha256_file(module.SOURCE_CONTRACT),
        "voter_membership": sha256_file(membership_path),
    }
    membership = pq.read_table(
        membership_path, columns=["source_row_uid", "benchmark_row_id", "split", "physical_member_index"]
    ).to_pylist()
    context_rows = [row for row in membership if row["split"] == "train"]
    train_contexts = read_jsonl(gold_root / "train_molecule_condition_labels.jsonl")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_root.name}.", dir=output_root.parent) as tmp:
        tmp_root = Path(tmp)
        database = tmp_root / "evidence.sqlite3"
        connection = sqlite3.connect(database)
        _evidence_schema(connection)
        display_counts: Counter[str] = Counter()
        seen: set[str] = set()
        try:
            for batch in pq.ParquetFile(module.STAGE3).iter_batches(batch_size=10_000, columns=columns):
                rows = []
                for raw in batch.to_pylist():
                    uid = str(raw["source_row_uid"])
                    mapped = mapping.get(uid)
                    if mapped is None:
                        raise ValueError(f"Current V10 record lacks finalized level: {uid}")
                    level, family, mapped_record_id = mapped
                    record_id = str(raw["canonical_record_id"])
                    if record_id != mapped_record_id or uid in seen:
                        raise ValueError(f"Current V10 evidence identity mismatch: {uid}")
                    identity = normalize_molecule_identity(str(raw["canonical_smiles"] or ""))
                    parent_smiles = str(identity.parent_smiles or "")
                    if identity.status != "ok" or not parent_smiles:
                        raise ValueError(f"Current V10 record lacks valid parent: {uid}")
                    parent_id = str(identity.parent_inchi_key or parent_smiles)
                    source = {name: raw.get(name) for name in fields[str(raw["source_id"])]}
                    display = _display_fields(task, raw, source)
                    display_counts[display["origin"]] += 1
                    projected = dict(source)
                    projected["measurement_text"] = display["measurement_text"]
                    projected["unit_text"] = display["unit_text"]
                    bucket = None
                    if raw.get("pair_bucket_key"):
                        bucket = json.dumps(
                            [*json.loads(raw["pair_bucket_key"]), level], separators=(",", ":")
                        )
                    scalar = raw.get("finite_scalar_value")
                    compatible = scalar is not None and math.isfinite(float(scalar))
                    payload = {
                        **{key: raw.get(key) for key in core},
                        "record_id": record_id,
                        "task_id": task,
                        "progressive_level": level,
                        "family_key": family,
                        "canonical_smiles": parent_smiles,
                        "source_canonical_smiles": str(raw["canonical_smiles"] or ""),
                        "source_fields": projected,
                        "source_fields_raw": source,
                        "display_measurement_text": display["measurement_text"],
                        "display_unit_text": display["unit_text"],
                        "display_origin": display["origin"],
                        "display_adapter": display["adapter"],
                    }
                    rows.append((
                        uid, record_id, parent_id, parent_smiles, level,
                        int(bucket in accepted.get(level, set())), int(compatible),
                        json.dumps(_clean(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":")),
                    ))
                    seen.add(uid)
                connection.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?,?)", rows)
                connection.commit()
            if set(mapping) != seen:
                raise ValueError(f"Finalized map and current V10 differ by {len(set(mapping) ^ seen)} UIDs")
            context_insert = sorted(
                (str(row["benchmark_row_id"]), str(row["source_row_uid"]), int(row["physical_member_index"]) + 1)
                for row in context_rows
            )
            context_metadata = [(
                str(row["benchmark_row_id"]), str(row["molecule_identity_key"]),
                str(row["condition_group"]), int(row["Y"]),
            ) for row in train_contexts]
            if len({row[0] for row in context_metadata}) != len(context_metadata):
                raise ValueError("Duplicate V1 train context ID")
            connection.executemany("INSERT INTO contexts VALUES (?,?,?,?)", context_metadata)
            connection.executemany("INSERT INTO context_records VALUES (?,?,?)", context_insert)
            missing = connection.execute(
                "SELECT COUNT(*) FROM context_records c LEFT JOIN records r USING(source_row_uid) "
                "WHERE r.source_row_uid IS NULL"
            ).fetchone()[0]
            if missing:
                raise ValueError(f"V1 gold context edges missing from current V10 evidence: {missing}")
            missing_contexts = connection.execute(
                "SELECT COUNT(*) FROM contexts c LEFT JOIN context_records r USING(context_id) "
                "WHERE r.context_id IS NULL"
            ).fetchone()[0]
            if missing_contexts:
                raise ValueError(f"V1 train contexts lack physical evidence: {missing_contexts}")
            identity = {
                "schema_version": SCHEMA_VERSION,
                "task_id": task,
                "evidence_release": "v10_current",
                "stage3_sha256": input_hashes["records"],
                "level_mapping_sha256": input_hashes["level_mapping"],
                "source_contract_sha256": input_hashes["source_contract"],
                "voter_membership_sha256": input_hashes["voter_membership"],
                "record_count": len(seen),
                "context_edge_count": len(context_insert),
                "context_count": len(context_metadata),
                "display_counts": dict(sorted(display_counts.items())),
            }
            content_id = _digest(identity)
            connection.executemany("INSERT INTO metadata VALUES (?,?)", [
                ("schema_version", SCHEMA_VERSION), ("content_id", content_id),
                ("task_id", task), ("evidence_release", "v10_current"),
            ])
            connection.commit()
            connection.execute("VACUUM")
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise ValueError("Evidence index failed integrity check")
        finally:
            connection.close()
        manifest = {
            **identity, "content_id": content_id, "status": "complete",
            "database": "evidence.sqlite3", "database_sha256": sha256_file(database),
            "inputs": {
                "records": {"path": str(module.STAGE3.resolve()), "sha256": input_hashes["records"]},
                "level_mapping": {"path": str(module.LEVEL_MAPPING.resolve()), "sha256": input_hashes["level_mapping"]},
                "level_manifest": {"path": str(module.LEVEL_MANIFEST.resolve()), "sha256": input_hashes["level_manifest"]},
                "source_contract": {"path": str(module.SOURCE_CONTRACT.resolve()), "sha256": input_hashes["source_contract"]},
                "voter_membership": {"path": str(membership_path.resolve()), "sha256": input_hashes["voter_membership"]},
            },
        }
        (tmp_root / "VERSION.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(tmp_root, output_root)
    return manifest


@lru_cache(maxsize=None)
def _fingerprint(smiles: str):
    _, _, fingerprint = standardize_smiles_and_fp(smiles)
    if fingerprint is None:
        raise ValueError(f"Invalid parent SMILES: {smiles}")
    return fingerprint


def _similarity(left: str, right: str) -> float:
    return float(DataStructs.TanimotoSimilarity(_fingerprint(left), _fingerprint(right)))


def _insert_rankings(connection: sqlite3.Connection, rows: Iterable[tuple[Any, ...]]) -> int:
    rows = list(rows)
    connection.executemany("INSERT INTO rankings VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def _rank_scored_rows(
    rows: list[dict[str, Any]], *, task: str, benchmark_row_id: str, level: str,
    query_parent_smiles: str,
) -> list[dict[str, Any]]:
    similarities = {
        row["parent_id"]: _similarity(query_parent_smiles, row["parent_smiles"])
        for row in rows
    }
    for row in rows:
        row["similarity"] = similarities[row["parent_id"]]
        row["tie"] = seeded_rank_tie_key(
            0, task, benchmark_row_id, level, row["parent_id"], row["source_row_uid"]
        )
    by_morgan = sorted(rows, key=lambda row: (-row["similarity"], row["tie"]))
    by_assay = sorted(rows, key=lambda row: (-row["score"], row["tie"]))
    morgan = {row["source_row_uid"]: rank for rank, row in enumerate(by_morgan, 1)}
    assay = {row["source_row_uid"]: rank for rank, row in enumerate(by_assay, 1)}
    within: Counter[str] = Counter()
    for row in by_morgan:
        within[row["parent_id"]] += 1
        row["within_parent_rank"] = within[row["parent_id"]]
        row["morgan_rank"] = morgan[row["source_row_uid"]]
        row["assay_rank"] = assay[row["source_row_uid"]]
    return by_morgan


def _l1_rows(
    *, task: str, subset: str, benchmark_row_id: str, query_id: int,
    query_parent_smiles: str, rankings_by_query: dict[str, list[dict[str, Any]]],
    evidence: sqlite3.Connection,
) -> tuple[
    list[tuple[Any, ...]], tuple[int, int], list[tuple[int, str, str, str]]
]:
    candidates = rankings_by_query.get(benchmark_row_id) or []
    if not candidates:
        raise ValueError(f"L1 cache lacks query: {benchmark_row_id}")
    context_ids = sorted({str(row["retrieval_record_id"]) for row in candidates})
    placeholders = ",".join("?" for _ in context_ids)
    context_records: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for context_id, uid, within_rank in evidence.execute(
        f"""SELECT c.context_id,c.source_row_uid,c.within_context_rank
            FROM context_records c
            WHERE c.context_id IN ({placeholders})""", context_ids,
    ):
        context_records[str(context_id)].append((str(uid), int(within_rank)))
    if set(context_records) != set(context_ids):
        raise ValueError(f"L1 contexts do not resolve in current V10 evidence: {benchmark_row_id}")

    parents: dict[str, dict[str, Any]] = {}
    expanded: dict[str, dict[str, Any]] = {}
    for row in sorted(candidates, key=lambda value: int(value["model_rank"])):
        context_id = str(row["retrieval_record_id"])
        expected_parent = str(row["retrieval_molecule_identity_key"])
        edges = context_records[context_id]
        context_parent = evidence.execute(
            "SELECT parent_id FROM contexts WHERE context_id=?", (context_id,)
        ).fetchone()
        parent_smiles = str(row["retrieval_smiles"])
        if context_parent is None or str(context_parent[0]) != expected_parent:
            raise ValueError(f"L1 frozen context identity mismatch: {context_id}")
        parent = parents.setdefault(expected_parent, {
            "parent_smiles": parent_smiles,
            "similarity": _similarity(query_parent_smiles, parent_smiles),
            "morgan_source_rank": int(row["retrieval_parent_rank"]),
            "assay_source_rank": int(row["model_rank"]),
            "score": float(row["prob_transfer"]),
            "assay_context_key": (int(row["model_rank"]), context_id),
            "morgan_context_key": (int(row["retrieval_parent_context_index"]), context_id),
        })
        assay_key = (int(row["model_rank"]), context_id)
        if assay_key < parent["assay_context_key"]:
            parent["assay_context_key"] = assay_key
            parent["assay_source_rank"] = int(row["model_rank"])
            parent["score"] = float(row["prob_transfer"])
        parent["morgan_context_key"] = min(
            parent["morgan_context_key"],
            (int(row["retrieval_parent_context_index"]), context_id),
        )
        for uid, within_context in edges:
            value = {
                "source_row_uid": uid, "parent_id": expected_parent,
                "parent_smiles": parent_smiles, "context_id": context_id,
                "context_rank": int(row["model_rank"]),
                "within_context_rank": within_context,
            }
            prior = expanded.get(uid)
            if prior is None or (value["context_rank"], value["within_context_rank"], context_id) < (
                prior["context_rank"], prior["within_context_rank"], prior["context_id"]
            ):
                expanded[uid] = value
    if len(parents) != L1_PARENT_CAPACITY:
        raise ValueError(f"L1 does not contain {L1_PARENT_CAPACITY} parents: {benchmark_row_id}")
    morgan_order = sorted(
        parents, key=lambda parent: (-parents[parent]["similarity"], parents[parent]["morgan_source_rank"], parent)
    )
    assay_order = sorted(
        parents, key=lambda parent: (parents[parent]["assay_source_rank"], parent)
    )
    morgan_rank = {parent: rank for rank, parent in enumerate(morgan_order, 1)}
    assay_rank = {parent: rank for rank, parent in enumerate(assay_order, 1)}
    within: Counter[str] = Counter()
    output = []
    for row in sorted(
        expanded.values(),
        key=lambda value: (
            morgan_rank[value["parent_id"]], value["context_rank"],
            value["within_context_rank"], value["source_row_uid"],
        ),
    ):
        parent_id = row["parent_id"]
        within[parent_id] += 1
        parent = parents[parent_id]
        output.append((
            query_id, FIXED_POOL, "L1", row["source_row_uid"], parent_id,
            row["parent_smiles"], row["context_id"], within[parent_id],
            parent["similarity"], morgan_rank[parent_id], parent["score"], assay_rank[parent_id],
        ))
    represented = {row["parent_id"] for row in expanded.values()}
    contexts = [(
        query_id, parent_id, parent["morgan_context_key"][1], parent["assay_context_key"][1]
    ) for parent_id, parent in sorted(parents.items()) if parent_id in represented]
    if len(represented) < 10:
        raise ValueError(f"L1 voter-membership records cannot fill ten cards: {benchmark_row_id}")
    return output, (len(output), len(represented)), contexts


def _later_rows(
    *, task: str, benchmark_row_id: str, query_id: int, query_parent_smiles: str,
    old: sqlite3.Connection, old_query_id: int,
    evidence_lookup: dict[str, tuple[str, str, str, str]],
    pool: str, level: str,
) -> tuple[list[tuple[Any, ...]], tuple[int, int]]:
    raw = old.execute(
        """SELECT r.external_record_id,r.parent_id,r.payload,s.transfer_probability
           FROM assignments a JOIN records r USING(record_key) JOIN scores s USING(score_key)
           WHERE a.query_id=? AND a.pool=? AND a.level=?""",
        (old_query_id, pool, level),
    ).fetchall()
    rows = []
    for record_id, cached_parent, payload_json, score in raw:
        payload = json.loads(payload_json)
        uid = str(payload["source_row_uid"])
        current = evidence_lookup.get(uid)
        if (current is None or str(current[0]) != str(record_id) or str(current[1]) != str(cached_parent)
                or str(current[3]) != level):
            raise ValueError(f"Scored cache record differs from current V10 evidence: {uid}")
        rows.append({
            "source_row_uid": uid, "parent_id": str(current[1]),
            "parent_smiles": str(current[2]), "score": float(score),
        })
    ranked = _rank_scored_rows(
        rows, task=task, benchmark_row_id=benchmark_row_id, level=level,
        query_parent_smiles=query_parent_smiles,
    )
    output = [(
        query_id, pool, level, row["source_row_uid"], row["parent_id"],
        row["parent_smiles"], None, row["within_parent_rank"], row["similarity"],
        row["morgan_rank"], row["score"], row["assay_rank"],
    ) for row in ranked]
    return output, (len(rows), len({row["parent_id"] for row in rows}))


def _l5_catalog(evidence: sqlite3.Connection) -> dict[str, dict[str, list[tuple[str, str]]]]:
    grouped = {pool: defaultdict(list) for pool in POOLS}
    for uid, parent_id, parent_smiles, accepted, compatible in evidence.execute(
        "SELECT source_row_uid,parent_id,parent_smiles,tool_accepted,tool_compatible "
        "FROM records WHERE level='L5'"
    ):
        pools = ["all"]
        if accepted:
            pools.append("tool-accepted")
        if compatible:
            pools.append("tool-compatible")
        for pool in pools:
            grouped[pool][str(parent_id)].append((str(uid), str(parent_smiles)))
    if any(not grouped[pool] for pool in POOLS):
        raise ValueError("Current V10 L5 pool is empty")
    return {pool: dict(parents) for pool, parents in grouped.items()}


def _l5_rows(
    *, task: str, benchmark_row_id: str, query_id: int, query_parent_id: str,
    query_parent_smiles: str, pool: str,
    catalog: dict[str, dict[str, list[tuple[str, str]]]],
) -> tuple[list[tuple[Any, ...]], tuple[int, int]]:
    groups = {
        parent: rows for parent, rows in catalog[pool].items() if parent != query_parent_id
    }
    similarities = {
        parent: _similarity(query_parent_smiles, rows[0][1])
        for parent, rows in groups.items()
    }
    selected_parents = sorted(groups, key=lambda parent: (-similarities[parent], parent))[:L1_PARENT_CAPACITY]
    if len(selected_parents) != L1_PARENT_CAPACITY:
        raise ValueError(f"Insufficient current V10 L5 parents: {benchmark_row_id}/{pool}")
    staged = []
    for parent in selected_parents:
        for uid, parent_smiles in groups[parent]:
            staged.append({
                "source_row_uid": uid, "parent_id": parent, "parent_smiles": parent_smiles,
                "similarity": similarities[parent],
                "tie": seeded_rank_tie_key(0, task, benchmark_row_id, "L5", parent, uid),
            })
    staged.sort(key=lambda row: (-row["similarity"], row["tie"]))
    within: Counter[str] = Counter()
    output = []
    for rank, row in enumerate(staged, 1):
        within[row["parent_id"]] += 1
        output.append((
            query_id, pool, "L5", row["source_row_uid"], row["parent_id"],
            row["parent_smiles"], None, within[row["parent_id"]], row["similarity"],
            rank, None, None,
        ))
    return output, (len(output), len(selected_parents))


def build_rankings(
    task: str, subset: str, evidence_root: Path, score_root: Path, output_root: Path,
) -> dict[str, Any]:
    """Publish one split ranking DB over the shared evidence index."""
    if subset not in {"valid", "test"}:
        raise ValueError(f"Unsupported split: {subset}")
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to replace rankings: {output_root}")
    evidence_manifest_path = evidence_root.resolve() / "VERSION.json"
    evidence_manifest = json.loads(evidence_manifest_path.read_text())
    evidence_database = evidence_root.resolve() / evidence_manifest["database"]
    evidence_database_sha256 = sha256_file(evidence_database)
    if (evidence_manifest.get("schema_version") != SCHEMA_VERSION
            or evidence_manifest.get("status") != "complete"
            or evidence_manifest.get("task_id") != task
            or evidence_manifest.get("database_sha256") != evidence_database_sha256):
        raise ValueError("Evidence index is incomplete or changed")
    score_manifest_path = score_root.resolve() / task / "scaffold" / subset / "VERSION.json"
    score_manifest = json.loads(score_manifest_path.read_text())
    score_database = score_manifest_path.with_name("scores.sqlite3")
    score_database_sha256 = sha256_file(score_database)
    if (score_manifest.get("schema_version") != three_pools.SCHEMA
            or score_manifest.get("status") != "complete"
            or score_manifest.get("task_id") != task
            or score_manifest.get("subset") != subset
            or score_manifest.get("gold_release") != "v1"
            or score_manifest.get("evidence_release") != "current_v10"
            or score_manifest.get("neighbor_identity_policy") != "all_parent_forms_parent_disjoint"
            or score_manifest.get("cache_sha256") != score_database_sha256):
        raise ValueError("Later-level score cache is incomplete or incompatible")
    l1_manifest_path = L1_ROOT / task / "scaffold" / subset / "VERSION.json"
    l1_manifest = json.loads(l1_manifest_path.read_text())
    l1_rankings = l1_manifest_path.with_name("rankings.parquet")
    l1_rankings_sha256 = sha256_file(l1_rankings)
    l1_inputs = l1_manifest.get("inputs") or {}
    if (l1_manifest.get("status") != "complete" or l1_inputs.get("gold_release") != "v1"
            or l1_manifest.get("model_lineage") != "v10_3_best_parent"
            or l1_inputs.get("neighbor_identity_policy") != "parent_disjoint"
            or l1_manifest.get("rankings_sha256") != l1_rankings_sha256):
        raise ValueError("L1 ranking cache is incomplete or incompatible")
    l1_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pq.read_table(l1_rankings).to_pylist():
        l1_by_query[str(row["query_record_id"])].append(row)

    gold_root = REPO_ROOT / "data/gold_labels" / GOLD_NAMES[task] / "v1/scaffold"
    query_path = gold_root / f"{subset}_molecule_condition_labels.jsonl"
    query_sha256 = sha256_file(query_path)
    queries = read_jsonl(query_path)
    if len({str(row["benchmark_row_id"]) for row in queries}) != len(queries):
        raise ValueError("Duplicate frozen benchmark row ID")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_root.name}.", dir=output_root.parent) as tmp:
        tmp_root = Path(tmp)
        database = tmp_root / "rankings.sqlite3"
        connection = sqlite3.connect(database)
        _ranking_schema(connection)
        evidence = sqlite3.connect(f"file:{evidence_database}?mode=ro", uri=True)
        old = sqlite3.connect(f"file:{score_database}?mode=ro", uri=True)
        assignment_counts: Counter[str] = Counter()
        l5_catalog = _l5_catalog(evidence)
        evidence_lookup = {
            str(uid): (str(record_id), str(parent_id), str(parent_smiles), str(level))
            for uid, record_id, parent_id, parent_smiles, level in evidence.execute(
                "SELECT source_row_uid,external_record_id,parent_id,parent_smiles,level FROM records"
            )
        }
        try:
            for query_id, query in enumerate(queries, 1):
                benchmark_row_id = str(query["benchmark_row_id"])
                drug = str(query["drug"])
                identity = normalize_molecule_identity(drug)
                query_parent_smiles = str(identity.parent_smiles or "")
                query_parent_id = str(identity.parent_inchi_key or query_parent_smiles)
                if identity.status != "ok" or not query_parent_smiles:
                    raise ValueError(f"Invalid frozen query parent: {benchmark_row_id}")
                connection.execute(
                    "INSERT INTO queries VALUES (?,?,?)",
                    (query_id, query_parent_id, query_parent_smiles),
                )
                connection.execute(
                    "INSERT INTO benchmark_queries VALUES (?,?,?)",
                    (benchmark_row_id, drug, query_id),
                )

                l1_rows, counts, l1_contexts = _l1_rows(
                    task=task, subset=subset, benchmark_row_id=benchmark_row_id,
                    query_id=query_id, query_parent_smiles=query_parent_smiles,
                    rankings_by_query=l1_by_query, evidence=evidence,
                )
                assignment_counts["fixed/L1"] += _insert_rankings(connection, l1_rows)
                connection.executemany("INSERT INTO l1_parent_contexts VALUES (?,?,?,?)", l1_contexts)
                connection.execute(
                    "INSERT INTO selection_counts VALUES (?,?,?,?,?)",
                    (query_id, FIXED_POOL, "L1", *counts),
                )

                old_query = old.execute(
                    "SELECT query_id,drug FROM benchmark_queries WHERE benchmark_row_id=?",
                    (benchmark_row_id,),
                ).fetchone()
                if old_query is None or str(old_query[1]) != drug:
                    raise ValueError(f"Later-level cache query mismatch: {benchmark_row_id}")
                for pool in POOLS:
                    for level in sorted(score_manifest["models"], key=lambda value: int(value[1:])):
                        rows, counts = _later_rows(
                            task=task, benchmark_row_id=benchmark_row_id, query_id=query_id,
                            query_parent_smiles=query_parent_smiles, old=old,
                            old_query_id=int(old_query[0]), evidence_lookup=evidence_lookup,
                            pool=pool, level=level,
                        )
                        assignment_counts[f"{pool}/{level}"] += _insert_rankings(connection, rows)
                        connection.execute(
                            "INSERT INTO selection_counts VALUES (?,?,?,?,?)",
                            (query_id, pool, level, *counts),
                        )
                    rows, counts = _l5_rows(
                        task=task, benchmark_row_id=benchmark_row_id, query_id=query_id,
                        query_parent_id=query_parent_id, query_parent_smiles=query_parent_smiles,
                        pool=pool, catalog=l5_catalog,
                    )
                    assignment_counts[f"{pool}/L5"] += _insert_rankings(connection, rows)
                    connection.execute(
                        "INSERT INTO selection_counts VALUES (?,?,?,?,?)",
                        (query_id, pool, "L5", *counts),
                    )
                connection.commit()
                print(f"{task}/{subset}: indexed {query_id}/{len(queries)} queries", flush=True)

            identity_receipt = {
                "schema_version": SCHEMA_VERSION,
                "task_id": task,
                "subset": subset,
                "gold_release": "v1",
                "evidence_content_id": evidence_manifest["content_id"],
                "query_sha256": query_sha256,
                "l1_rankings_sha256": l1_rankings_sha256,
                "later_scores_sha256": score_database_sha256,
                "assignment_counts": dict(sorted(assignment_counts.items())),
                "neighbor_identity_policy": "l1_voter_membership_later_parent_disjoint",
            }
            content_id = _digest(identity_receipt)
            connection.executemany("INSERT INTO metadata VALUES (?,?)", [
                ("schema_version", SCHEMA_VERSION), ("content_id", content_id),
                ("task_id", task), ("subset", subset),
            ])
            connection.commit()
            connection.execute("VACUUM")
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise ValueError("Rankings index failed integrity check")
        finally:
            old.close()
            evidence.close()
            connection.close()

        manifest = {
            **identity_receipt,
            "content_id": content_id,
            "status": "complete",
            "rankings_database": "rankings.sqlite3",
            "rankings_database_sha256": sha256_file(database),
            "evidence_database": "../../evidence/evidence.sqlite3",
            "evidence_database_sha256": evidence_manifest["database_sha256"],
            "evidence_record_count": evidence_manifest["record_count"],
            "pools": list(POOLS),
            "fixed_pool_levels": ["L1"],
            "native_rankings": ["morgan", "assay_transfer"],
            "joint_materialized": False,
            "tie_seed": 0,
            "morgan_fingerprint": {"radius": 2, "bits": 2048, "similarity": "Tanimoto"},
            "morgan_similarity_scope": "query_parent_to_reference_parent",
            "capacities": {
                "l1_molecules": 10,
                "l1_records_per_molecule": L1_RECORD_CAPACITY,
                "later_records_per_level": LATER_RECORD_CAPACITY,
                "cached_parent_pool": L1_PARENT_CAPACITY,
            },
            "inputs": {
                "query_ledger": {"path": str(query_path.resolve()), "sha256": query_sha256},
                "evidence_manifest": {"path": str(evidence_manifest_path), "sha256": sha256_file(evidence_manifest_path)},
                "l1_manifest": {"path": str(l1_manifest_path), "sha256": sha256_file(l1_manifest_path)},
                "l1_rankings": {"path": str(l1_rankings), "sha256": l1_rankings_sha256},
                "later_manifest": {"path": str(score_manifest_path), "sha256": sha256_file(score_manifest_path)},
                "later_scores": {"path": str(score_database), "sha256": score_database_sha256},
            },
        }
        (tmp_root / "VERSION.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(tmp_root, output_root)
    _write_release_index(task, output_root.parents[2])
    return manifest


def validate(root: Path, *, deep: bool = True) -> dict[str, Any]:
    """Validate locally in full, or verify hashes after copying the tree."""
    root = root.resolve()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "validation_mode": "deep" if deep else "copied_tree_hashes",
        "tasks": {},
    }
    for task in GOLD_NAMES:
        evidence_root = root / task / "evidence"
        evidence_manifest = json.loads((evidence_root / "VERSION.json").read_text())
        evidence_database = evidence_root / evidence_manifest["database"]
        if evidence_manifest["database_sha256"] != sha256_file(evidence_database):
            raise ValueError(f"Evidence hash mismatch: {task}")
        evidence_records = int(evidence_manifest["record_count"])
        evidence_parents: dict[str, str] = {}
        if deep:
            with sqlite3.connect(evidence_database) as connection:
                if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                    raise ValueError(f"Evidence integrity failure: {task}")
                if connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] != evidence_records:
                    raise ValueError(f"Evidence count mismatch: {task}")
                evidence_parents = dict(
                    connection.execute("SELECT source_row_uid,parent_id FROM records")
                )
        unavailable = int((evidence_manifest.get("display_counts") or {}).get("unavailable", 0))
        splits = {}
        for subset in ("valid", "test"):
            split_root = root / task / "scaffold" / subset
            manifest_path = split_root / "VERSION.json"
            manifest = json.loads(manifest_path.read_text())
            database = split_root / manifest["rankings_database"]
            if manifest["rankings_database_sha256"] != sha256_file(database):
                raise ValueError(f"Rankings hash mismatch: {task}/{subset}")
            if not deep:
                splits[subset] = {
                    "assignments": sum(manifest["assignment_counts"].values()),
                    "database_sha256_verified": True,
                }
                continue
            with sqlite3.connect(database) as connection:
                if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                    raise ValueError(f"Rankings integrity failure: {task}/{subset}")
                overlap = connection.execute(
                    """SELECT COUNT(*) FROM rankings r JOIN queries q USING(query_id)
                       WHERE r.parent_id=q.query_parent_id"""
                ).fetchone()[0]
                missing = later_identity_mismatches = 0
                for uid, parent_id in connection.execute(
                    "SELECT source_row_uid,parent_id FROM rankings WHERE level!='L1'"
                ):
                    evidence_parent = evidence_parents.get(uid)
                    missing += evidence_parent is None
                    later_identity_mismatches += (
                        evidence_parent is not None and evidence_parent != parent_id
                    )
                l1_identity_overrides = 0
                for uid, parent_id in connection.execute(
                    "SELECT source_row_uid,parent_id FROM rankings WHERE level='L1'"
                ):
                    evidence_parent = evidence_parents.get(uid)
                    missing += evidence_parent is None
                    l1_identity_overrides += (
                        evidence_parent is not None and evidence_parent != parent_id
                    )
                assignments = connection.execute("SELECT COUNT(*) FROM rankings").fetchone()[0]
            if missing or overlap or later_identity_mismatches or unavailable:
                raise ValueError(
                    f"Validation failure {task}/{subset}: missing={missing} "
                    f"parent_overlap={overlap} later_identity_mismatches={later_identity_mismatches} "
                    f"unavailable_display={unavailable}"
                )
            splits[subset] = {
                "assignments": assignments, "parent_overlap": 0,
                "later_identity_mismatches": 0,
                "l1_voter_membership_identity_overrides": l1_identity_overrides,
                "unavailable_display": 0,
            }
        report["tasks"][task] = {"evidence_records": evidence_records, "splits": splits}
        index_path = root / task / "RELEASE_INDEX.json"
        index = json.loads(index_path.read_text())
        if index.get("status") != "complete" or set(index.get("splits") or {}) != {"valid", "test"}:
            raise ValueError(f"Incomplete task release index: {task}")
        for subset, item in index["splits"].items():
            path = root / task / item["manifest"]
            if item["manifest_sha256"] != sha256_file(path):
                raise ValueError(f"Task release index hash mismatch: {task}/{subset}")
    report["status"] = "complete"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build-evidence", "build-rankings", "validate"))
    parser.add_argument("--task", choices=tuple(GOLD_NAMES))
    parser.add_argument("--subset", choices=("valid", "test"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--score-root", type=Path, default=Path("/local/jojolee-starling-retrieval-v1/later"))
    parser.add_argument("--hashes-only", action="store_true")
    args = parser.parse_args()
    if args.command == "validate":
        result = validate(args.output_root, deep=not args.hashes_only)
    elif args.command == "build-evidence":
        if not args.task:
            parser.error("--task is required")
        result = build_evidence(args.task, args.output_root / args.task / "evidence")
    else:
        if not args.task or not args.subset:
            parser.error("--task and --subset are required")
        result = build_rankings(
            args.task, args.subset, args.output_root / args.task / "evidence",
            args.score_root, args.output_root / args.task / "scaffold" / args.subset,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
