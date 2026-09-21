"""Build compact shared-universe rankings and an evidence-owned UID projection."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import DataStructs

from data.processing.gold_labels.conditioned_benchmark import tdc_task_root
from data.processing.paths import evidence_library_root
from predict.retrieval.policies import normalize_molecule_identity, standardize_smiles_and_fp
from predict.utils.json import read_jsonl, sha256_file

from . import runtime, three_pools, v27_skin
from .build_ranked_retrieval import (
    GOLD_NAMES as BASE_GOLD_NAMES,
    _clean,
    _digest,
    _display_fields,
)
from .ranked_uid_retrieval import CAPACITY, SCHEMA_VERSION


PROFILE = "ranked_level_retrieval_v4"
EVIDENCE_RELEASE: str | None = None
SCORE_REUSE_ROOTS: tuple[Path, ...] = ()
TRUST_PREDECESSOR_ROWS = False
TASK_LEVELS = {
    "bbb_martins": ("L1", "L2", "L3", "L4", "L5"),
    "bioavailability_ma": ("L1", "L2", "L3", "L4", "L5", "L6"),
    "skin_reaction": ("L2", "L3"),
    "ames": ("L1", "L2", "L3", "L4", "L5"),
    "dili": ("L1", "L2", "L3", "L4", "L5", "L6", "L7"),
    "carcinogens": ("L1", "L2", "L3", "L4", "L5", "L6", "L7"),
}
REPO_ROOT = Path(__file__).resolve().parents[3]
GOLD_NAMES = {
    **BASE_GOLD_NAMES,
    "skin_reaction": "Skin_Reaction",
    "ames": "Ames",
    "dili": "DILI",
    "carcinogens": "Carcinogens",
}
DEFAULT_OUTPUT = runtime.cache_profile_root(PROFILE)
L1_ROOT = runtime.cache_profile_root("v10_3_best_scaffold_morgan100_v1")
ADDON_L1_ROOT = runtime.cache_profile_root("ranked_level_retrieval_gold_v1_addon_v1")
ADDON_TASKS = frozenset({"ames", "dili", "carcinogens"})
REUSE_ROOT = runtime.cache_profile_root("recent_models_three_pools_morgan100_v2_gold_v1")
QUERY_BENCHMARK = "gold"
LABEL_RELEASE: str | dict[str, str] = "v1"


def _evidence_paths(task: str) -> tuple[Path, Path, Path, Path]:
    if EVIDENCE_RELEASE is not None:
        release = evidence_library_root(task, EVIDENCE_RELEASE)
        return (
            release / "03_pair_buckets/records.parquet",
            release / "level_mapping/records.parquet",
            release / "level_mapping/manifest.json",
            release / "02_canonicalized/source_contract.json",
        )
    module = three_pools.MODULES.get(task)
    if module is not None:
        return module.STAGE3, module.LEVEL_MAPPING, module.LEVEL_MANIFEST, module.SOURCE_CONTRACT
    release = REPO_ROOT / "data/evidence_libraries/skin_reaction/v10_main_universe_v5"
    return (
        release / "03_pair_buckets/records.parquet",
        release / "level_mapping/records.parquet",
        release / "level_mapping/manifest.json",
        release / "02_canonicalized/source_contract.json",
    )


def _source_fields(task: str) -> tuple[dict[str, list[str]], set[str]]:
    contract = json.loads(_evidence_paths(task)[3].read_text(encoding="utf-8"))
    fields = {}
    for source, spec in contract["sources"].items():
        if "source_visible_fields" in spec:
            fields[source] = list(spec["source_visible_fields"])
        else:
            fields[source] = [
                name for name, value in spec["normalized_artifact_columns"].items()
                if value.get("source_or_simply_cleaned") is True
            ]
    return fields, set().union(*map(set, fields.values()))


def _model_spec(task: str, level: str) -> dict[str, Any] | None:
    if task == v27_skin.TASK_ID:
        return v27_skin.MODEL if level in v27_skin.LEVELS else None
    module = three_pools.MODULES.get(task)
    return None if module is None else module.MODELS.get(level)


def build_evidence(task: str, output: Path) -> dict[str, Any]:
    """Write the display-ready projection once under the evidence release."""
    if output.exists():
        raise FileExistsError(f"Refusing to replace evidence projection: {output}")
    stage3, mapping_path, level_manifest, source_contract = _evidence_paths(task)
    fields, source_union = _source_fields(task)
    mapping_columns = ["source_row_uid", "canonical_record_id", "level"]
    if "family_key" in pq.read_schema(mapping_path).names:
        mapping_columns.append("family_key")
    mapping_rows = pq.read_table(mapping_path, columns=mapping_columns).to_pylist()
    scoped_rows = [
        row for row in mapping_rows if f"L{int(row['level'])}" in TASK_LEVELS[task]
    ]
    mapping = {
        str(row["source_row_uid"]): (
            f"L{int(row['level'])}", str(row.get("family_key") or ""),
            str(row["canonical_record_id"]),
        )
        for row in scoped_rows
    }
    if len(mapping) != len(scoped_rows):
        raise ValueError("Duplicate current V10 level-mapping UID")
    core = {
        "source_row_uid", "canonical_record_id", "canonical_smiles", "source_id",
        "pair_bucket_key", "measurement_kind", "finite_scalar_value",
        "canonical_pair_fields_json", "canonical_measurement_scale_id",
        "canonical_category_id", "canonical_measurement_text", "canonical_unit_text",
        "canonical_transporter_identifier",
    }
    columns = sorted((core | source_union) & set(pq.read_schema(stage3).names))
    projection_adapter = Path(__import__(_display_fields.__module__, fromlist=["x"]).__file__)
    task_adapter = projection_adapter
    if task == "bbb_martins":
        from data.processing.evidence_library.versions.v10.tasks.bbb_martins import (
            semantic_display,
        )

        task_adapter = Path(semantic_display.__file__)
    input_hashes = {
        "records": sha256_file(stage3),
        "level_mapping": sha256_file(mapping_path),
        "level_manifest": sha256_file(level_manifest),
        "source_contract": sha256_file(source_contract),
        "display_adapter": {
            "projection": sha256_file(projection_adapter),
            "task": sha256_file(task_adapter),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        root = Path(temporary)
        records_path = root / "records.parquet"
        projected = []
        seen: set[str] = set()
        display_counts: Counter[str] = Counter()
        for batch in pq.ParquetFile(stage3).iter_batches(batch_size=10_000, columns=columns):
            for raw in batch.to_pylist():
                uid = str(raw["source_row_uid"])
                mapped = mapping.get(uid)
                if mapped is None:
                    continue
                if uid in seen:
                    raise ValueError(f"Current V10 evidence identity mismatch: {uid}")
                level, family, record_id = mapped
                if str(raw["canonical_record_id"]) != record_id:
                    raise ValueError(f"Mapped record identity changed: {uid}")
                identity = normalize_molecule_identity(str(raw["canonical_smiles"] or ""))
                parent_smiles = str(identity.parent_smiles or "")
                if identity.status != "ok" or not parent_smiles:
                    raise ValueError(f"Current V10 record lacks valid parent: {uid}")
                parent_id = str(identity.parent_inchi_key or parent_smiles)
                source = {name: raw.get(name) for name in fields[str(raw["source_id"])]}
                display = _display_fields(task, raw, source)
                if display["origin"] == "unavailable":
                    raise ValueError(f"Current V10 record lacks a valid display: {uid}")
                display_counts[display["origin"]] += 1
                visible = dict(source)
                visible["measurement_text"] = display["measurement_text"]
                visible["unit_text"] = display["unit_text"]
                payload = {
                    **{key: raw.get(key) for key in core},
                    "record_id": record_id,
                    "task_id": task,
                    "progressive_level": level,
                    "family_key": family,
                    "canonical_smiles": parent_smiles,
                    "source_canonical_smiles": str(raw["canonical_smiles"] or ""),
                    "source_fields": visible,
                    "source_fields_raw": source,
                    "source_contract": {
                        "contract_version": "ranked_evidence_projection.v1",
                        "source_or_simply_cleaned": {
                            name: True for name in visible
                        },
                    },
                    "display_measurement_text": display["measurement_text"],
                    "display_unit_text": display["unit_text"],
                    "display_origin": display["origin"],
                    "display_adapter": display["adapter"],
                }
                projected.append({
                    "source_row_uid": uid,
                    "external_record_id": record_id,
                    "parent_id": parent_id,
                    "parent_smiles": parent_smiles,
                    "level": level,
                    "payload": json.dumps(
                        _clean(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":")
                    ),
                })
                seen.add(uid)
        if set(mapping) != seen:
            raise ValueError(f"Finalized map and current V10 differ by {len(set(mapping) ^ seen)} UIDs")
        schema = pa.schema([
            ("source_row_uid", pa.string()),
            ("external_record_id", pa.string()),
            ("parent_id", pa.string()),
            ("parent_smiles", pa.large_string()),
            ("level", pa.string()),
            ("payload", pa.large_string()),
        ])
        table = pa.Table.from_pylist(projected, schema=schema).sort_by("source_row_uid")
        pq.write_table(table, records_path, compression="zstd", row_group_size=10_000)
        identity = {
            "schema_version": "ranked_evidence_projection.v1",
            "status": "complete",
            "task_id": task,
            "evidence_library_release": EVIDENCE_RELEASE,
            "record_count": len(seen),
            "display_counts": dict(sorted(display_counts.items())),
            "inputs": input_hashes,
        }
        identity["content_id"] = _digest(identity)
        manifest = {
            **identity,
            "records": records_path.name,
            "records_sha256": sha256_file(records_path),
            "source_paths": {
                "records": str(stage3.resolve()),
                "level_mapping": str(mapping_path.resolve()),
                "level_manifest": str(level_manifest.resolve()),
                "source_contract": str(source_contract.resolve()),
            },
        }
        (root / "VERSION.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(root, output)
    return manifest


def _schema(connection: sqlite3.Connection, level: str) -> None:
    connection.executescript("""
        PRAGMA journal_mode=DELETE;
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE queries(
            benchmark_row_id TEXT PRIMARY KEY,drug TEXT NOT NULL,
            query_parent_id TEXT NOT NULL,query_parent_smiles TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE rankings(
            benchmark_row_id TEXT NOT NULL,item_id TEXT NOT NULL,
            parent_id TEXT NOT NULL,parent_smiles TEXT NOT NULL,
            morgan_similarity REAL NOT NULL,parent_morgan_rank INTEGER NOT NULL,
            within_parent_rank INTEGER NOT NULL,morgan_rank INTEGER NOT NULL,
            assay_transfer_score REAL,assay_rank INTEGER,
            morgan_context_id TEXT,assay_context_id TEXT,
            morgan_member_count INTEGER,assay_member_count INTEGER,
            score_key TEXT,
            PRIMARY KEY(benchmark_row_id,item_id)
        ) WITHOUT ROWID;
        CREATE INDEX rankings_morgan ON rankings(benchmark_row_id,morgan_rank);
        CREATE INDEX rankings_assay ON rankings(benchmark_row_id,assay_rank);
        CREATE TABLE contexts(
            context_id TEXT PRIMARY KEY,parent_id TEXT NOT NULL,
            condition_group TEXT NOT NULL,gold_label INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE context_records(
            context_id TEXT NOT NULL,source_row_uid TEXT NOT NULL,
            within_context_rank INTEGER NOT NULL,
            PRIMARY KEY(context_id,within_context_rank),
            UNIQUE(context_id,source_row_uid)
        ) WITHOUT ROWID;
        CREATE INDEX context_records_uid ON context_records(source_row_uid);
    """)


def _queries(task: str, subset: str) -> tuple[Path, list[dict[str, Any]], list[tuple[str, str, str, str]]]:
    path = (
        tdc_task_root(task) / f"{subset}_molecule_condition_labels.jsonl"
        if QUERY_BENCHMARK == "tdc"
        else REPO_ROOT / "data/gold_labels" / GOLD_NAMES[task] / "v1/scaffold" / f"{subset}_molecule_condition_labels.jsonl"
    )
    rows = read_jsonl(path)
    output = []
    for row in rows:
        identity = row.get("molecule_identity") or {}
        if not identity.get("parent_inchi_key") or not identity.get("parent_smiles"):
            normalized = normalize_molecule_identity(str(row["drug"]))
            if normalized.status != "ok":
                raise ValueError(f"Frozen Gold query cannot be normalized: {row['benchmark_row_id']}")
            identity = {
                "parent_inchi_key": normalized.parent_inchi_key,
                "parent_smiles": normalized.parent_smiles,
            }
        if str(identity["parent_inchi_key"]) != str(row["molecule_identity_key"]):
            raise ValueError(f"Frozen Gold query identity changed: {row['benchmark_row_id']}")
        output.append((
            str(row["benchmark_row_id"]), str(row["drug"]),
            str(identity["parent_inchi_key"]), str(identity["parent_smiles"]),
        ))
    return path, rows, output


def _row_digest(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    for row in connection.execute(
        "SELECT * FROM rankings ORDER BY benchmark_row_id,morgan_rank,item_id"
    ):
        digest.update(json.dumps(row, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _write_complete(
    connection: sqlite3.Connection, database: Path, *, task: str, subset: str,
    level: str, identity: dict[str, Any], target: Path,
) -> dict[str, Any]:
    ranking_sha256 = _row_digest(connection)
    content_id = _digest({**identity, "ranking_rows_sha256": ranking_sha256})
    connection.executemany("INSERT OR REPLACE INTO metadata VALUES (?,?)", [
        ("schema_version", SCHEMA_VERSION), ("content_id", content_id),
        ("task_id", task), ("subset", subset), ("level", level),
    ])
    connection.commit()
    if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
        raise ValueError("Ranking database failed integrity check")
    manifest = {
        **identity,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "content_id": content_id,
        "ranking_rows_sha256": ranking_sha256,
        "database": database.name,
        "database_sha256": sha256_file(database),
    }
    (target / "VERSION.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def rebind_evidence(
    task: str, output_root: Path, previous_manifest: Path, current_manifest: Path,
) -> dict[str, Any]:
    """Rebind provenance only when the projected evidence bytes are identical."""
    previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
    current = json.loads(current_manifest.read_text(encoding="utf-8"))
    stable = ("schema_version", "task_id", "record_count", "records_sha256")
    if any(previous.get(key) != current.get(key) for key in stable):
        raise ValueError("Evidence projection changed; rankings must be rebuilt")
    old_hash = sha256_file(previous_manifest)
    new_hash = sha256_file(current_manifest)
    rebound = 0
    for subset in ("valid", "test"):
        for level in TASK_LEVELS[task]:
            target = output_root / task / "scaffold" / subset / level
            version_path = target / "VERSION.json"
            manifest = json.loads(version_path.read_text(encoding="utf-8"))
            inputs = manifest.get("inputs") or {}
            if inputs.get("evidence_manifest_sha256") != old_hash:
                raise ValueError(f"Cache is not bound to the expected projection: {target}")
            inputs["evidence_manifest_sha256"] = new_hash
            manifest["inputs"] = inputs
            if manifest.get("status") == "complete":
                database = target / str(manifest["database"])
                connection = sqlite3.connect(database)
                try:
                    manifest = _write_complete(
                        connection,
                        database,
                        task=task,
                        subset=subset,
                        level=level,
                        identity={
                            key: value for key, value in manifest.items()
                            if key not in {
                                "schema_version", "status", "content_id", "database",
                                "database_sha256", "ranking_rows_sha256",
                            }
                        },
                        target=target,
                    )
                finally:
                    connection.close()
            else:
                temporary = version_path.with_name(".VERSION.json.tmp")
                temporary.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, version_path)
            rebound += 1
    return {
        "status": "complete",
        "task_id": task,
        "rebound_levels": rebound,
        "previous_manifest_sha256": old_hash,
        "current_manifest_sha256": new_hash,
        "records_sha256": current["records_sha256"],
    }


def build_l1(task: str, subset: str, output_root: Path, evidence_manifest: Path) -> dict[str, Any]:
    if task in ADDON_TASKS:
        return _copy_addon_l1(task, subset, output_root, evidence_manifest)
    target = output_root / task / "scaffold" / subset / "L1"
    if target.exists():
        raise FileExistsError(f"Refusing to replace L1 cache: {target}")
    query_path, _, queries = _queries(task, subset)
    source_manifest = L1_ROOT / task / "scaffold" / subset / "VERSION.json"
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    rankings_path = source_manifest.with_name(str(source["rankings"]))
    rows = pq.read_table(rankings_path).to_pylist()
    membership_path = REPO_ROOT / "data/gold_labels" / GOLD_NAMES[task] / "v1/scaffold/voter_membership.parquet"
    membership_rows = pq.read_table(
        membership_path,
        columns=["benchmark_row_id", "source_row_uid", "vote_id", "physical_member_index"],
    ).to_pylist()
    members: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for row in membership_rows:
        members[str(row["benchmark_row_id"])].append(
            (
                str(row["vote_id"]), int(row["physical_member_index"]),
                str(row["source_row_uid"]),
            )
        )
    gold_train = REPO_ROOT / "data/gold_labels" / GOLD_NAMES[task] / "v1/scaffold/train_molecule_condition_labels.jsonl"
    contexts = {
        str(row["benchmark_row_id"]): row for row in read_jsonl(gold_train)
    }
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["query_record_id"])][str(row["retrieval_molecule_identity_key"])].append(row)

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".L1.", dir=target.parent) as temporary:
        root = Path(temporary)
        database = root / "rankings.sqlite3"
        connection = sqlite3.connect(database)
        _schema(connection, "L1")
        connection.executemany("INSERT INTO queries VALUES (?,?,?,?)", queries)
        stored = []
        for benchmark_row_id, _, _, _ in queries:
            parents = grouped[benchmark_row_id]
            if len(parents) != CAPACITY:
                raise ValueError(f"L1 cache lacks {CAPACITY} parents: {benchmark_row_id}")
            staged = []
            for parent_id, context_rows in parents.items():
                morgan = min(context_rows, key=lambda row: (
                    int(row["retrieval_parent_context_index"]), str(row["retrieval_record_id"])
                ))
                assay = min(context_rows, key=lambda row: (
                    int(row["model_rank"]), str(row["retrieval_record_id"])
                ))
                staged.append({
                    "parent_id": parent_id,
                    "parent_smiles": str(morgan["retrieval_smiles"]),
                    "similarity": float(morgan["morgan_tanimoto_similarity"]),
                    "morgan_source_rank": int(morgan["retrieval_parent_rank"]),
                    "assay_source_rank": int(assay["model_rank"]),
                    "score": float(assay["prob_transfer"]),
                    "morgan_context": str(morgan["retrieval_record_id"]),
                    "assay_context": str(assay["retrieval_record_id"]),
                })
            morgan_order = sorted(staged, key=lambda row: (row["morgan_source_rank"], row["parent_id"]))
            assay_order = sorted(staged, key=lambda row: (row["assay_source_rank"], row["parent_id"]))
            morgan_ranks = {row["parent_id"]: rank for rank, row in enumerate(morgan_order, 1)}
            assay_ranks = {row["parent_id"]: rank for rank, row in enumerate(assay_order, 1)}
            for row in staged:
                if not members[row["morgan_context"]] or not members[row["assay_context"]]:
                    raise ValueError("L1 selected context has no physical Gold members")
                stored.append((
                    benchmark_row_id, row["parent_id"], row["parent_id"], row["parent_smiles"],
                    row["similarity"], morgan_ranks[row["parent_id"]], 1,
                    morgan_ranks[row["parent_id"]], row["score"],
                    assay_ranks[row["parent_id"]], row["morgan_context"], row["assay_context"],
                    len(members[row["morgan_context"]]), len(members[row["assay_context"]]), None,
                ))
        connection.executemany("INSERT INTO rankings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", stored)
        selected_contexts = sorted({
            context_id for row in stored for context_id in (row[10], row[11])
        })
        connection.executemany(
            "INSERT INTO contexts VALUES (?,?,?,?)",
            [
                (
                    context_id,
                    str(contexts[context_id]["molecule_identity_key"]),
                    str(contexts[context_id]["condition_group"]),
                    int(contexts[context_id]["Y"]),
                )
                for context_id in selected_contexts
            ],
        )
        connection.executemany(
            "INSERT INTO context_records VALUES (?,?,?)",
            [
                (context_id, uid, rank)
                for context_id in selected_contexts
                for rank, (_, _, uid) in enumerate(sorted(members[context_id]), 1)
            ],
        )
        identity = {
            "task_id": task, "subset": subset, "level": "L1", "pool": "fixed",
            "capacity": CAPACITY, "parent_capacity": CAPACITY, "gold_release": "v1",
            "neighbor_identity_policy": "scaffold_disjoint",
            "shared_candidate_universe": True,
            "query_count": len(queries), "stored_rows": len(stored),
            "query_counts": {
                query_id: {
                    "candidate_parents": CAPACITY,
                    "morgan_candidate_records": sum(
                        row[12] for row in stored if row[0] == query_id
                    ),
                    "assay_candidate_records": sum(
                        row[13] for row in stored if row[0] == query_id
                    ),
                }
                for query_id, *_ in queries
            },
            "inputs": {
                "query_sha256": sha256_file(query_path),
                "source_manifest_sha256": sha256_file(source_manifest),
                "source_rankings_sha256": sha256_file(rankings_path),
                "voter_membership_sha256": sha256_file(membership_path),
                "gold_train_sha256": sha256_file(gold_train),
                "evidence_manifest_sha256": sha256_file(evidence_manifest),
            },
        }
        manifest = _write_complete(
            connection, database, task=task, subset=subset, level="L1",
            identity=identity, target=root,
        )
        connection.close()
        os.replace(root, target)
    return manifest


def _copy_addon_l1(
    task: str, subset: str, output_root: Path, evidence_manifest: Path,
) -> dict[str, Any]:
    """Rebind the reviewed Morgan-only L1 cache to the complete projection."""
    target = output_root / task / "scaffold" / subset / "L1"
    if target.exists():
        raise FileExistsError(f"Refusing to replace L1 cache: {target}")
    source_root = ADDON_L1_ROOT / task / "scaffold" / subset / "L1"
    source_manifest = source_root / "VERSION.json"
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    source_database = source_root / str(source["database"])
    evidence = json.loads(evidence_manifest.read_text(encoding="utf-8"))
    evidence_uids = set(
        pq.read_table(
            evidence_manifest.with_name(str(evidence["records"])),
            columns=["source_row_uid"],
        )["source_row_uid"].to_pylist()
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".L1.", dir=target.parent) as temporary:
        root = Path(temporary)
        database = root / "rankings.sqlite3"
        shutil.copy2(source_database, database)
        with sqlite3.connect(database) as connection:
            selected = {
                str(row[0]) for row in connection.execute(
                    "SELECT DISTINCT source_row_uid FROM context_records"
                )
            }
            if not selected <= evidence_uids:
                raise ValueError(f"L1 predecessor has {len(selected - evidence_uids)} missing UIDs")
            _, _, expected_queries = _queries(task, subset)
            if list(connection.execute("SELECT * FROM queries ORDER BY benchmark_row_id")) != sorted(
                expected_queries
            ):
                raise ValueError(f"L1 predecessor query identity changed: {task}/{subset}")
            identity = {
                key: value for key, value in source.items()
                if key not in {
                    "schema_version", "status", "content_id", "database",
                    "database_sha256", "ranking_rows_sha256",
                }
            }
            identity["inputs"] = {
                **identity["inputs"],
                "evidence_manifest_sha256": sha256_file(evidence_manifest),
                "l1_predecessor_manifest_sha256": sha256_file(source_manifest),
                "l1_predecessor_database_sha256": sha256_file(source_database),
            }
            manifest = _write_complete(
                connection, database, task=task, subset=subset, level="L1",
                identity=identity, target=root,
            )
        os.replace(root, target)
    return manifest


def _records(
    evidence_manifest: Path, level: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, list[str]]]]:
    manifest = json.loads(evidence_manifest.read_text(encoding="utf-8"))
    path = evidence_manifest.with_name(str(manifest["records"]))
    records = {}
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for batch in pq.ParquetFile(path).iter_batches(batch_size=10_000):
        for row in batch.to_pylist():
            if level is not None and str(row["level"]) != level:
                continue
            payload = json.loads(row["payload"])
            record = {
                **payload,
                "source_row_uid": str(row["source_row_uid"]),
                "record_id": str(row["external_record_id"]),
                "parent_id": str(row["parent_id"]),
                "canonical_smiles": str(row["parent_smiles"]),
                "progressive_level": str(row["level"]),
                "has_scalar": payload.get("finite_scalar_value") is not None,
            }
            uid = record["source_row_uid"]
            records[uid] = record
            grouped[record["progressive_level"]][record["parent_id"]].append(uid)
    return records, {level: dict(parents) for level, parents in grouped.items()}


def _reuse_scores(
    task: str, subset: str, level: str, keys: set[str]
) -> dict[str, float]:
    output: dict[str, float] = {}
    values = sorted(keys)
    for reuse_subset in ("valid", "test"):
        path = REUSE_ROOT / task / "scaffold" / reuse_subset / "scores.sqlite3"
        if not path.is_file():
            continue
        with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
            for start in range(0, len(values), 900):
                chunk = values[start:start + 900]
                placeholders = ",".join("?" for _ in chunk)
                for key, value in connection.execute(
                    f"SELECT cache_key,transfer_probability FROM scores WHERE cache_key IN ({placeholders}) "
                    "AND transfer_probability IS NOT NULL", chunk,
                ):
                    output[str(key)] = float(value)
    for reuse_root in SCORE_REUSE_ROOTS:
        manifest_path = reuse_root / task / "scaffold" / subset / level / "VERSION.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        database = manifest_path.with_name(str(manifest["database"]))
        with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
            for start in range(0, len(values), 900):
                chunk = values[start:start + 900]
                placeholders = ",".join("?" for _ in chunk)
                for key, value in connection.execute(
                    f"SELECT DISTINCT score_key,assay_transfer_score FROM rankings "
                    f"WHERE score_key IN ({placeholders}) "
                    "AND assay_transfer_score IS NOT NULL",
                    chunk,
                ):
                    key, value = str(key), float(value)
                    if key in output and output[key] != value:
                        raise ValueError(f"Conflicting reusable score: {key}")
                    output[key] = value
    return output


def _predecessor_rows(
    task: str, subset: str, level: str,
) -> dict[tuple[str, str], tuple[str, float]]:
    """Load reviewed predecessor score identities without rerendering old rows."""
    if not TRUST_PREDECESSOR_ROWS:
        return {}
    output: dict[tuple[str, str], tuple[str, float]] = {}
    for reuse_root in SCORE_REUSE_ROOTS:
        manifest_path = reuse_root / task / "scaffold" / subset / level / "VERSION.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        database = manifest_path.with_name(str(manifest["database"]))
        with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
            for query_id, item_id, score_key, score in connection.execute(
                "SELECT benchmark_row_id,item_id,score_key,assay_transfer_score "
                "FROM rankings WHERE score_key IS NOT NULL "
                "AND assay_transfer_score IS NOT NULL"
            ):
                key = (str(query_id), str(item_id))
                value = (str(score_key), float(score))
                if key in output and output[key] != value:
                    raise ValueError(f"Conflicting predecessor row: {key}")
                output[key] = value
    return output


def prepare_level(
    task: str, subset: str, level: str, output_root: Path, evidence_manifest: Path,
) -> dict[str, Any]:
    if level == "L1":
        return build_l1(task, subset, output_root, evidence_manifest)
    if level not in TASK_LEVELS[task]:
        raise ValueError(f"Unsupported level: {task}/{level}")
    target = output_root / task / "scaffold" / subset / level
    if target.exists():
        raise FileExistsError(f"Refusing to replace level cache: {target}")
    query_path, query_rows, queries = _queries(task, subset)
    records, grouped = _records(evidence_manifest, level)
    level_parents = grouped.get(level, {})
    parents = sorted(level_parents)
    parent_smiles = {
        parent: records[level_parents[parent][0]]["canonical_smiles"] for parent in parents
    }
    fps = [standardize_smiles_and_fp(parent_smiles[parent])[2] for parent in parents]
    if any(fp is None for fp in fps):
        raise ValueError("Evidence projection contains an invalid parent fingerprint")
    packed = np.stack([
        np.frombuffer(DataStructs.BitVectToBinaryText(fp), dtype=np.uint8) for fp in fps
    ])
    popcount = three_pools.bbb.POPCOUNT[packed].sum(axis=1, dtype=np.uint16)
    model_spec = _model_spec(task, level)
    if task == v27_skin.TASK_ID:
        renderer = v27_skin.SkinV27PromptRenderer()
    else:
        renderer = three_pools.Renderer(task) if model_spec is not None else None
    predecessor_rows = _predecessor_rows(task, subset, level)
    trusted_score_keys: set[str] = set()
    trusted_row_count = 0

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{level}.", dir=target.parent) as temporary:
        root = Path(temporary)
        database = root / "rankings.sqlite3"
        connection = sqlite3.connect(database)
        _schema(connection, level)
        connection.executemany("INSERT INTO queries VALUES (?,?,?,?)", queries)
        staged_rows = []
        prompt_rows: dict[str, tuple[str, str]] = {}
        query_identities = {row[0]: (row[2], row[3]) for row in queries}
        for query in query_rows:
            benchmark_row_id = str(query["benchmark_row_id"])
            query_parent, query_smiles = query_identities[benchmark_row_id]
            qfp = standardize_smiles_and_fp(query_smiles)[2]
            similarities = three_pools.bbb._similarities(qfp, packed, popcount)
            ordered = sorted(range(len(parents)), key=lambda index: (-float(similarities[index]), parents[index]))
            selected_parents = []
            for index in ordered:
                parent = parents[index]
                similarity = float(similarities[index])
                if parent == query_parent:
                    continue
                selected_parents.append((parent, similarity))
                if len(selected_parents) == CAPACITY:
                    break
            if len(selected_parents) != CAPACITY:
                raise ValueError(f"Insufficient Morgan parents: {benchmark_row_id}/{level}")
            morgan_rank = 0
            for parent_rank, (parent, similarity) in enumerate(selected_parents, 1):
                for within_parent_rank, uid in enumerate(sorted(level_parents[parent]), 1):
                    morgan_rank += 1
                    record = records[uid]
                    score_key = None
                    score = None
                    if renderer is not None:
                        predecessor = predecessor_rows.get((benchmark_row_id, uid))
                        if predecessor is not None:
                            score_key, score = predecessor
                            trusted_score_keys.add(score_key)
                            trusted_row_count += 1
                        else:
                            prompt = renderer.prompt_task(record, query_smiles)
                            score_key = prompt.cache_key
                            prompt_rows.setdefault(score_key, (prompt.prompt, prompt.projection_hash))
                    staged_rows.append([
                        benchmark_row_id, uid, parent, record["canonical_smiles"], similarity,
                        parent_rank, within_parent_rank, morgan_rank, score, None,
                        None, None, None, None, score_key,
                    ])

        reused = _reuse_scores(task, subset, level, set(prompt_rows))
        missing = set(prompt_rows) - set(reused)
        for row in staged_rows:
            if row[-1] in reused:
                row[8] = reused[row[-1]]
        connection.executemany("INSERT INTO rankings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", staged_rows)
        prompts_path = root / "prompts.parquet"
        if missing:
            pq.write_table(
                pa.Table.from_pylist([
                    {
                        "score_key": key,
                        "prompt": prompt_rows[key][0],
                        "projection_hash": prompt_rows[key][1],
                    }
                    for key in sorted(missing)
                ]),
                prompts_path,
                compression="zstd",
            )
        record_counts = Counter(row[0] for row in staged_rows)
        identity = {
            "task_id": task, "subset": subset, "level": level, "pool": "all",
            "capacity": CAPACITY, "parent_capacity": CAPACITY,
            **({"gold_release": LABEL_RELEASE} if isinstance(LABEL_RELEASE, str)
               else {"label_release": LABEL_RELEASE}),
            "neighbor_identity_policy": "parent_disjoint",
            "shared_candidate_universe": True,
            "query_count": len(queries), "stored_rows": len(staged_rows),
            "query_counts": {
                query_id: {
                    "candidate_parents": CAPACITY,
                    "candidate_records": record_counts[query_id],
                }
                for query_id, *_ in queries
            },
            "model": model_spec,
            "inputs": {
                "query_sha256": sha256_file(query_path),
                "evidence_manifest_sha256": sha256_file(evidence_manifest),
            },
            "score_counts": {
                "exact_reuse": len(reused) + len(trusted_score_keys),
                "pending": len(missing),
                "trusted_predecessor_rows": trusted_row_count,
            },
            "trusted_predecessor_row_reuse": TRUST_PREDECESSOR_ROWS,
            "score_reuse_sources": [
                {
                    "path": str(root.resolve()),
                    "release_index_sha256": sha256_file(root / task / "RELEASE_INDEX.json"),
                }
                for root in SCORE_REUSE_ROOTS
                if (root / task / "RELEASE_INDEX.json").is_file()
            ],
        }
        if renderer is None:
            for benchmark_row_id, in connection.execute("SELECT benchmark_row_id FROM queries"):
                connection.execute(
                    "UPDATE rankings SET assay_rank=NULL WHERE benchmark_row_id=?", (benchmark_row_id,)
                )
            manifest = _write_complete(
                connection, database, task=task, subset=subset, level=level,
                identity=identity, target=root,
            )
        else:
            for benchmark_row_id, in connection.execute("SELECT benchmark_row_id FROM queries"):
                scored = connection.execute(
                    "SELECT item_id FROM rankings WHERE benchmark_row_id=? AND assay_transfer_score IS NOT NULL "
                    "ORDER BY assay_transfer_score DESC,item_id", (benchmark_row_id,),
                ).fetchall()
                expected = connection.execute(
                    "SELECT COUNT(*) FROM rankings WHERE benchmark_row_id=?", (benchmark_row_id,)
                ).fetchone()[0]
                if len(scored) == expected:
                    connection.executemany(
                        "UPDATE rankings SET assay_rank=? WHERE benchmark_row_id=? AND item_id=?",
                        [(rank, benchmark_row_id, row[0]) for rank, row in enumerate(scored, 1)],
                    )
            connection.commit()
            manifest = {
                **identity,
                "schema_version": SCHEMA_VERSION,
                "status": "prepared" if missing else "ready_to_finalize",
                "database": database.name,
                "prompts": prompts_path.name if missing else None,
            }
            (root / "VERSION.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        connection.close()
        os.replace(root, target)
    return manifest


def score_level(
    task: str, subset: str, level: str, output_root: Path, device: int,
    num_shards: int, shard_index: int, batch_size: int,
) -> dict[str, Any]:
    target = output_root / task / "scaffold" / subset / level
    manifest = json.loads((target / "VERSION.json").read_text(encoding="utf-8"))
    if manifest.get("status") not in {"prepared", "ready_to_finalize"}:
        raise ValueError("Level cache is not prepared")
    database = target / manifest["database"]
    prompt_path = target / str(manifest.get("prompts") or "")
    rows = sorted(
        (
            (str(row["score_key"]), str(row["prompt"]), str(row["projection_hash"]))
            for row in pq.read_table(prompt_path).to_pylist()
        ),
        key=lambda row: (len(row[1]), row[0]),
    )[shard_index::num_shards]
    journal_dir = target / ".scores"
    journal_dir.mkdir(exist_ok=True)
    journal = journal_dir / f"{shard_index:02d}-of-{num_shards:02d}.jsonl"
    done = read_jsonl(journal) if journal.exists() else []
    if [row["score_key"] for row in done] != [row[0] for row in rows[:len(done)]]:
        raise ValueError("Score journal is not the exact completed shard prefix")
    spec = _model_spec(task, level)
    if spec is None:
        raise ValueError(f"No assay-transfer model is configured for {task}/{level}")
    renderer = (
        v27_skin.SkinV27PromptRenderer()
        if task == v27_skin.TASK_ID
        else three_pools.Renderer(task)
    )
    snapshot = runtime.resolve_model_snapshot(spec["model"], spec["revision"], local_files_only=True)
    model, tokenizer = runtime.load_model(snapshot, device=device)
    with journal.open("a", encoding="utf-8") as handle:
        for offset in range(len(done), len(rows), batch_size):
            batch = [runtime.PromptTask(
                cache_key=key, prompt=prompt,
                prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
                task_id=task, query_smiles="", group_id=level, molecule_id="", record_id="",
                model=spec["model"], model_revision=spec["revision"],
                scoring_contract_version=runtime.SCORING_CONTRACT_VERSION,
                template_hash=renderer.template_hash, projection_hash=projection,
            ) for key, prompt, projection in rows[offset:offset + batch_size]]
            for result in runtime.score_prompt_batch(model, tokenizer, batch, device=device):
                handle.write(json.dumps({
                    "score_key": result.cache_key,
                    "assay_transfer_score": result.transfer_probability,
                }) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(f"{task}/{subset}/{level}/{shard_index}: {offset + len(batch)}/{len(rows)}", flush=True)
    return {"status": "complete", "scores": len(rows), "journal": str(journal)}


def finalize_level(task: str, subset: str, level: str, output_root: Path) -> dict[str, Any]:
    target = output_root / task / "scaffold" / subset / level
    version_path = target / "VERSION.json"
    manifest = json.loads(version_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"prepared", "ready_to_finalize"}:
        raise ValueError("Level cache is not prepared")
    database = target / manifest["database"]
    connection = sqlite3.connect(database)
    try:
        expected = {
            str(row[0]) for row in connection.execute(
                "SELECT DISTINCT score_key FROM rankings "
                "WHERE score_key IS NOT NULL AND assay_transfer_score IS NULL"
            )
        }
        fresh = {}
        for journal in sorted((target / ".scores").glob("*.jsonl")) if (target / ".scores").exists() else []:
            for row in read_jsonl(journal):
                key, value = str(row["score_key"]), float(row["assay_transfer_score"])
                if key in fresh or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid or duplicate fresh score: {key}")
                fresh[key] = value
        if set(fresh) != expected:
            raise ValueError(f"Incomplete fresh scores: missing={len(expected-set(fresh))} extra={len(set(fresh)-expected)}")
        connection.execute(
            "CREATE TEMP TABLE fresh_scores("
            "score_key TEXT PRIMARY KEY,assay_transfer_score REAL NOT NULL) WITHOUT ROWID"
        )
        connection.executemany(
            "INSERT INTO fresh_scores VALUES (?,?)", fresh.items()
        )
        connection.execute(
            "UPDATE rankings SET assay_transfer_score=("
            "SELECT assay_transfer_score FROM fresh_scores "
            "WHERE fresh_scores.score_key=rankings.score_key) "
            "WHERE score_key IN (SELECT score_key FROM fresh_scores)"
        )
        for benchmark_row_id, in connection.execute("SELECT benchmark_row_id FROM queries"):
            rows = connection.execute(
                "SELECT item_id FROM rankings WHERE benchmark_row_id=? "
                "ORDER BY assay_transfer_score DESC,item_id", (benchmark_row_id,),
            ).fetchall()
            expected = connection.execute(
                "SELECT COUNT(*) FROM rankings WHERE benchmark_row_id=?", (benchmark_row_id,)
            ).fetchone()[0]
            if len(rows) != expected:
                raise ValueError(f"Incomplete assay ranking: {benchmark_row_id}/{level}")
            connection.executemany(
                "UPDATE rankings SET assay_rank=? WHERE benchmark_row_id=? AND item_id=?",
                [(rank, benchmark_row_id, row[0]) for rank, row in enumerate(rows, 1)],
            )
        manifest["score_counts"] = {
            "exact_reuse": manifest["score_counts"]["exact_reuse"],
            "fresh": len(fresh),
        }
        manifest = _write_complete(
            connection, database, task=task, subset=subset, level=level,
            identity={key: value for key, value in manifest.items() if key not in {
                "schema_version", "status", "database", "content_id", "database_sha256",
                "ranking_rows_sha256", "prompts",
            }},
            target=target,
        )
    finally:
        connection.close()
    prompt_path = target / str(manifest.get("prompts") or "prompts.parquet")
    if prompt_path.is_file():
        prompt_path.unlink()
    if (target / ".scores").is_dir():
        shutil.rmtree(target / ".scores")
    return manifest


def write_index(task: str, output_root: Path, evidence_manifest: Path) -> dict[str, Any]:
    task_root = output_root / task
    levels = {}
    complete = True
    for subset in ("valid", "test"):
        split = {}
        for level in TASK_LEVELS[task]:
            path = task_root / "scaffold" / subset / level / "VERSION.json"
            if not path.is_file():
                complete = False
                continue
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("status") != "complete":
                complete = False
            split[level] = {
                "manifest": str(path.relative_to(task_root)),
                "manifest_sha256": sha256_file(path),
                "content_id": manifest.get("content_id"),
            }
        levels[subset] = {"levels": split}
    evidence = json.loads(evidence_manifest.read_text(encoding="utf-8"))
    index = {
        "schema_version": "ranked_uid_task_release_index.v1",
        "selection_contract": SCHEMA_VERSION,
        "profile": PROFILE,
        "task_id": task,
        "status": "complete" if complete else "partial",
        **({"gold_release": LABEL_RELEASE} if isinstance(LABEL_RELEASE, str)
           else {"label_release": LABEL_RELEASE}),
        "pool": "all",
        "parent_capacity": CAPACITY,
        "levels_independent": True,
        "ranking_modes": (
            ["morgan"]
            if all(_model_spec(task, level) is None for level in TASK_LEVELS[task])
            else ["morgan", "assay-transfer"]
        ),
        "later_candidate_universe": "all_uids_under_morgan_top_100_parents",
        "neighbor_identity_policy_by_level": {
            level: "scaffold_disjoint" if level == "L1" else "parent_disjoint"
            for level in TASK_LEVELS[task]
        },
        "evidence": {
            "manifest": os.path.relpath(evidence_manifest, task_root),
            "manifest_sha256": sha256_file(evidence_manifest),
            "content_id": evidence["content_id"],
            "record_count": evidence["record_count"],
        },
        "splits": levels,
    }
    task_root.mkdir(parents=True, exist_ok=True)
    path = task_root / "RELEASE_INDEX.json"
    temporary = path.with_name(".RELEASE_INDEX.json.tmp")
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return index


def validate_release(
    task: str, output_root: Path, evidence_manifest: Path,
) -> dict[str, Any]:
    """Run expensive integrity checks once, before atomic publication."""
    task_root = output_root / task
    index_path = task_root / "RELEASE_INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if (
        index.get("schema_version") != "ranked_uid_task_release_index.v1"
        or index.get("status") != "complete"
        or index.get("task_id") != task
    ):
        raise ValueError("Task release index is incomplete or incompatible")

    evidence = json.loads(evidence_manifest.read_text(encoding="utf-8"))
    indexed_evidence = (index_path.parent / index["evidence"]["manifest"]).resolve()
    if (
        indexed_evidence != evidence_manifest.resolve()
        or index["evidence"]["manifest_sha256"] != sha256_file(evidence_manifest)
        or index["evidence"]["content_id"] != evidence["content_id"]
    ):
        raise ValueError("Release index points to different evidence")
    stage3, mapping, level_manifest, source_contract = _evidence_paths(task)
    expected_inputs = {
        "records": sha256_file(stage3),
        "level_mapping": sha256_file(mapping),
        "level_manifest": sha256_file(level_manifest),
        "source_contract": sha256_file(source_contract),
    }
    if any(evidence["inputs"].get(key) != value for key, value in expected_inputs.items()):
        raise ValueError("Evidence projection no longer matches the current V10 release")
    projection_adapter = Path(__import__(_display_fields.__module__, fromlist=["x"]).__file__)
    task_adapter = projection_adapter
    if task == "bbb_martins":
        from data.processing.evidence_library.versions.v10.tasks.bbb_martins import (
            semantic_display,
        )

        task_adapter = Path(semantic_display.__file__)
    if evidence["inputs"].get("display_adapter") != {
        "projection": sha256_file(projection_adapter),
        "task": sha256_file(task_adapter),
    }:
        raise ValueError("Evidence display adapter code changed")
    records_path = evidence_manifest.with_name(str(evidence["records"]))
    if sha256_file(records_path) != evidence["records_sha256"]:
        raise ValueError("Evidence projection bytes changed")
    evidence_uids = set(
        pq.read_table(records_path, columns=["source_row_uid"])["source_row_uid"].to_pylist()
    )
    if len(evidence_uids) != evidence["record_count"]:
        raise ValueError("Evidence projection UID count changed")
    gold_root = REPO_ROOT / "data/gold_labels" / GOLD_NAMES[task] / "v1/scaffold"
    gold_member_rows = (
        pq.read_table(
            gold_root / "voter_membership.parquet",
            columns=["benchmark_row_id", "source_row_uid", "vote_id", "physical_member_index"],
        ).to_pylist()
        if "L1" in TASK_LEVELS[task]
        else []
    )
    gold_members: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for row in gold_member_rows:
        gold_members[str(row["benchmark_row_id"])].append((
            str(row["vote_id"]), int(row["physical_member_index"]),
            str(row["source_row_uid"]),
        ))
    member_order = (
        (lambda row: (row[1], row[2]))
        if task in ADDON_TASKS else (lambda row: row)
    )
    ordered_gold_uids = {
        context_id: [row[2] for row in sorted(rows, key=member_order)]
        for context_id, rows in gold_members.items()
    }
    gold_train_labels = gold_root / "train_molecule_condition_labels.jsonl"
    if not gold_train_labels.is_file():
        gold_train_labels = gold_root / "train.jsonl"
    gold_contexts = (
        {
            str(row["benchmark_row_id"]): (
                str(row["molecule_identity_key"]), str(row["condition_group"]), int(row["Y"])
            )
            for row in read_jsonl(gold_train_labels)
        }
        if "L1" in TASK_LEVELS[task]
        else {}
    )

    report: dict[str, Any] = {}
    evidence_hash = sha256_file(evidence_manifest)
    for subset in ("valid", "test"):
        _, _, expected_queries = _queries(task, subset)
        query_ledger = {row[0]: row for row in expected_queries}
        split_report = {}
        for level in TASK_LEVELS[task]:
            entry = index["splits"][subset]["levels"][level]
            manifest_path = task_root / str(entry["manifest"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            database = manifest_path.with_name(str(manifest["database"]))
            required = {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "task_id": task,
                "subset": subset,
                "level": level,
                "parent_capacity": CAPACITY,
                "neighbor_identity_policy": (
                    "scaffold_disjoint" if level == "L1" else "parent_disjoint"
                ),
            }
            if any(manifest.get(key) != value for key, value in required.items()):
                raise ValueError(f"Incompatible level manifest: {task}/{subset}/{level}")
            expected_model = _model_spec(task, level)
            if manifest.get("model") != expected_model:
                raise ValueError(f"Level model provenance changed: {task}/{subset}/{level}")
            if (
                sha256_file(manifest_path) != entry["manifest_sha256"]
                or manifest["content_id"] != entry["content_id"]
                or sha256_file(database) != manifest["database_sha256"]
                or manifest["inputs"]["evidence_manifest_sha256"] != evidence_hash
            ):
                raise ValueError(f"Published hashes differ: {task}/{subset}/{level}")
            if level == "L1":
                source_manifest = (
                    ADDON_L1_ROOT / task / "scaffold" / subset / "L1/VERSION.json"
                    if task in ADDON_TASKS
                    else L1_ROOT / task / "scaffold" / subset / "VERSION.json"
                )
                source = json.loads(source_manifest.read_text(encoding="utf-8"))
                source_payload = source_manifest.with_name(
                    str(source.get("rankings") or source["database"])
                )
                l1_inputs = manifest["inputs"]
                if task in ADDON_TASKS:
                    valid_l1 = (
                        l1_inputs.get("l1_predecessor_manifest_sha256")
                        == sha256_file(source_manifest)
                        and l1_inputs.get("l1_predecessor_database_sha256")
                        == sha256_file(source_payload)
                    )
                else:
                    role = "direct_v10_3" if task == "bbb_martins" else "direct_v10_3_0_2"
                    expected_model = runtime.model_profile(task, role)
                    valid_l1 = (
                        source.get("model") == expected_model
                        and l1_inputs["source_manifest_sha256"] == sha256_file(source_manifest)
                        and l1_inputs["source_rankings_sha256"] == sha256_file(source_payload)
                        and l1_inputs["voter_membership_sha256"]
                        == sha256_file(gold_root / "voter_membership.parquet")
                        and l1_inputs["gold_train_sha256"]
                        == sha256_file(gold_train_labels)
                    )
                if not valid_l1:
                    raise ValueError(f"L1 source provenance changed: {task}/{subset}")

            with sqlite3.connect(database) as connection:
                connection.row_factory = sqlite3.Row
                if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError(f"SQLite integrity failed: {task}/{subset}/{level}")
                metadata = dict(connection.execute("SELECT key,value FROM metadata"))
                if metadata.get("content_id") != manifest["content_id"]:
                    raise ValueError(f"SQLite identity changed: {task}/{subset}/{level}")
                frozen_queries = {
                    str(row["benchmark_row_id"]): tuple(row)
                    for row in connection.execute("SELECT * FROM queries")
                }
                if set(frozen_queries) != set(query_ledger):
                    raise ValueError(f"Query coverage changed: {task}/{subset}/{level}")
                context_parents: dict[str, str] = {}
                context_members: dict[str, list[str]] = defaultdict(list)
                if level == "L1":
                    frozen_contexts = {
                        str(row["context_id"]): (
                            str(row["parent_id"]), str(row["condition_group"]),
                            int(row["gold_label"]),
                        )
                        for row in connection.execute("SELECT * FROM contexts")
                    }
                    if any(
                        gold_contexts.get(context_id) != values
                        for context_id, values in frozen_contexts.items()
                    ):
                        raise ValueError(f"L1 Gold context changed: {task}/{subset}")
                    context_parents = {
                        context_id: values[0]
                        for context_id, values in frozen_contexts.items()
                    }
                    for row in connection.execute(
                        "SELECT context_id,source_row_uid FROM context_records "
                        "ORDER BY context_id,within_context_rank"
                    ):
                        context_members[str(row["context_id"])].append(
                            str(row["source_row_uid"])
                        )
                    if any(
                        uid not in evidence_uids
                        for members in context_members.values()
                        for uid in members
                    ):
                        raise ValueError(f"L1 context UID missing from evidence: {task}/{subset}")

                stored_rows = 0
                for query_id, (_, drug, query_parent, query_smiles) in query_ledger.items():
                    frozen = frozen_queries[query_id]
                    if frozen != (query_id, drug, query_parent, query_smiles):
                        raise ValueError(f"Query identity changed: {query_id}/{level}")
                    rows = connection.execute(
                        "SELECT * FROM rankings WHERE benchmark_row_id=? ORDER BY morgan_rank",
                        (query_id,),
                    ).fetchall()
                    stored_rows += len(rows)
                    count = CAPACITY if level == "L1" else int(
                        manifest["query_counts"][query_id]["candidate_records"]
                    )
                    if len(rows) != count:
                        raise ValueError(f"Candidate count changed: {query_id}/{level}")
                    parents = {str(row["parent_id"]) for row in rows}
                    if len(parents) != CAPACITY or query_parent in parents:
                        raise ValueError(f"Parent capacity/disjointness failed: {query_id}/{level}")
                    if [int(row["morgan_rank"]) for row in rows] != list(range(1, count + 1)):
                        raise ValueError(f"Morgan ranks are not dense: {query_id}/{level}")
                    if {int(row["parent_morgan_rank"]) for row in rows} != set(
                        range(1, CAPACITY + 1)
                    ):
                        raise ValueError(f"Parent ranks are not dense: {query_id}/{level}")
                    by_parent: dict[str, list[int]] = defaultdict(list)
                    for row in rows:
                        by_parent[str(row["parent_id"])].append(int(row["within_parent_rank"]))
                    if any(sorted(values) != list(range(1, len(values) + 1)) for values in by_parent.values()):
                        raise ValueError(f"Within-parent ranks are not dense: {query_id}/{level}")

                    if level == "L1":
                        if task in ADDON_TASKS:
                            if any(
                                row["assay_rank"] is not None
                                or row["assay_transfer_score"] is not None
                                or row["assay_context_id"] is not None
                                for row in rows
                            ):
                                raise ValueError(f"Morgan-only L1 has assay values: {query_id}")
                            prefixes = ("morgan",)
                        else:
                            if (
                                sorted(int(row["assay_rank"]) for row in rows)
                                != list(range(1, count + 1))
                                or any(
                                    row["assay_transfer_score"] is None
                                    or not 0 <= float(row["assay_transfer_score"]) <= 1
                                    for row in rows
                                )
                            ):
                                raise ValueError(f"L1 assay ranks/scores are incomplete: {query_id}")
                            prefixes = ("morgan", "assay")
                        for row in rows:
                            for prefix in prefixes:
                                context_id = str(row[f"{prefix}_context_id"])
                                if (
                                    context_parents.get(context_id) != str(row["parent_id"])
                                    or len(context_members.get(context_id, ()))
                                    != int(row[f"{prefix}_member_count"])
                                    or context_members.get(context_id)
                                    != ordered_gold_uids.get(context_id)
                                ):
                                    raise ValueError(f"L1 context membership changed: {context_id}")
                    else:
                        if any(str(row["item_id"]) not in evidence_uids for row in rows):
                            raise ValueError(f"Ranked UID missing from evidence: {query_id}/{level}")
                        assay_ranks = [row["assay_rank"] for row in rows]
                        scores = [row["assay_transfer_score"] for row in rows]
                        if expected_model is None:
                            if any(value is not None for value in (*assay_ranks, *scores)):
                                raise ValueError(f"Morgan-only level has assay values: {query_id}/{level}")
                        elif (
                            sorted(int(value) for value in assay_ranks if value is not None)
                            != list(range(1, count + 1))
                            or any(value is None or not 0 <= float(value) <= 1 for value in scores)
                        ):
                            raise ValueError(f"Assay ranks/scores are incomplete: {query_id}/{level}")
                if stored_rows != manifest["stored_rows"]:
                    raise ValueError(f"Stored-row count changed: {task}/{subset}/{level}")
            split_report[level] = {
                "content_id": manifest["content_id"],
                "database_sha256": manifest["database_sha256"],
                "stored_rows": manifest["stored_rows"],
            }
        report[subset] = split_report

    receipt = {
        "schema_version": "ranked_uid_release_validation.v1",
        "status": "complete",
        "task_id": task,
        "evidence_content_id": evidence["content_id"],
        "evidence_records_sha256": evidence["records_sha256"],
        "splits": report,
    }
    receipt["content_id"] = _digest(receipt)
    (task_root / "VALIDATION.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    global EVIDENCE_RELEASE, PROFILE, SCORE_REUSE_ROOTS, TRUST_PREDECESSOR_ROWS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "build-evidence", "prepare-level", "score", "finalize",
            "rebind-evidence", "index", "validate",
        ),
    )
    parser.add_argument("--task", choices=tuple(TASK_LEVELS), required=True)
    parser.add_argument("--subset", choices=("valid", "test"))
    parser.add_argument("--level")
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--evidence-release")
    parser.add_argument("--score-reuse-root", action="append", type=Path, default=[])
    parser.add_argument("--trust-existing-score-keys", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--previous-evidence-manifest", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    PROFILE = args.profile
    EVIDENCE_RELEASE = args.evidence_release
    SCORE_REUSE_ROOTS = tuple(path.resolve() for path in args.score_reuse_root)
    TRUST_PREDECESSOR_ROWS = args.trust_existing_score_keys
    args.output_root = args.output_root or runtime.cache_profile_root(PROFILE)
    if args.command == "build-evidence":
        result = build_evidence(args.task, args.evidence_manifest.parent)
    elif args.command == "rebind-evidence":
        if args.previous_evidence_manifest is None:
            parser.error("--previous-evidence-manifest is required")
        result = rebind_evidence(
            args.task, args.output_root,
            args.previous_evidence_manifest, args.evidence_manifest,
        )
    elif args.command == "index":
        result = write_index(args.task, args.output_root, args.evidence_manifest)
    elif args.command == "validate":
        result = validate_release(args.task, args.output_root, args.evidence_manifest)
    else:
        if not args.subset or not args.level:
            parser.error("--subset and --level are required")
        if args.command == "prepare-level":
            result = prepare_level(args.task, args.subset, args.level, args.output_root, args.evidence_manifest)
        elif args.command == "score":
            result = score_level(
                args.task, args.subset, args.level, args.output_root, args.device,
                args.num_shards, args.shard_index, args.batch_size,
            )
        else:
            result = finalize_level(args.task, args.subset, args.level, args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
