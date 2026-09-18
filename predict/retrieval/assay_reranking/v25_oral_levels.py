"""Build split-specific Oral V25 L2, L3, and L4/L6 record-transfer score caches.

The builder joins Oral V10 records to Tianang's acquisition-UID level mapping,
constructs independent scaffold-disjoint Morgan-75 pools, renders the frozen
V25 prompt with copied non-result retrieval context, and publishes one compact SQLite
cache per benchmark split. Scoring is resumable and level-specific because each
level has its own immutable checkpoint.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
import os
from pathlib import Path
import socket
import sqlite3
import sys
from typing import Any, Iterable, Mapping, Sequence

from jinja2 import Environment, FileSystemLoader, StrictUndefined
import numpy as np
import pyarrow.parquet as pq

from data.processing.evidence_library.level_mappings import evidence_level_mapping_release
from rdkit import DataStructs

from predict.retrieval.assay_reranking.runtime import (
    BACKBONE_DTYPE,
    COMPACT_CACHE_SCHEMA_VERSION,
    LOGIT_EXTRACTION_DTYPE,
    PromptTask,
    SCORING_CONTRACT_VERSION,
    file_sha256,
    load_model,
    resolve_model_snapshot,
    score_prompt_batch,
    cache_profile_root,
)
from predict.retrieval.policies import (
    decide_candidate,
    normalize_molecule_identity,
    standardize_smiles_and_fp,
)


TASK_ID = "bioavailability_ma"
PROFILE = "v25_oral_uid_levels_morgan75"
L2_PROFILE = PROFILE + "_l2"
LEVELS = ("L4", "L6")
# Independently audited caches retain their original construction provenance.
AUDITED_BUILDERS = {
    PROFILE: "c0a5e43d755754ed8571cfe6c922385cbb56d6a2bdee4e8ac6c5f1686209c1ce",
    L2_PROFILE: "cc32ee115bf10a8bd7dd33e0a577de5a6ccbcd5545f6f434b24bd9c7920f2e4a",
}
POOL_SIZE = 75
RECORD_SCOPE = (
    "v10_stage3_record_eligible_and_v25_assay_transfer_bucket_eligible_uid_mapped.v1"
)
PROMPT_ROOT = Path(__file__).with_name("prompts") / "v25_oral"
TRAINING_TEMPLATE_SHA256 = "435d70f963d9b55ac9b13da6f8c32af5414ee402d2af3af08fc5be01c4c7203d"
TRAINING_ORAL_PROJECTION_SHA256 = "2cc407ee11a82f1d27965c5234c583ed32717d5986f3aeb259aa309ea4678eaa"
STAGE3 = Path("data/evidence_libraries/bioavailability_ma/v10/03_pair_buckets/records.parquet")
STAGE3_MANIFEST = STAGE3.with_name("manifest.json")
SOURCE_CONTRACT = Path("data/evidence_libraries/bioavailability_ma/v10/02_canonicalized/source_contract.json")
LEVEL_MANIFEST, LEVEL_MAPPING, LEVEL_MAPPING_RECEIPT = evidence_level_mapping_release(
    "bioavailability_ma", "v10"
)
GOLD_ROOT = Path("data/gold_labels/Bioavailability_Ma/v1/scaffold")
MODELS = {
    "L2": {
        "model": "jiosephlee/intern-s1-mini-assay-transfer-record-level-v25-bioavailability-ma-l2-best",
        "revision": "c8bcca2e541bc5fdedb7a74be9881ea91eec7539",
        "dataset": "jiosephlee/assay-transfer-record-level-v25-bioavailability-ma-l2-intern",
        "dataset_revision": "b07ad9337a7f97a75db5770b521fd5a8779f8620",
    },
    "L3": {
        "model": "jiosephlee/intern-s1-mini-assay-transfer-record-level-v25-bioavailability-ma-l3-best",
        "revision": "089199baaac086dfb97f5944f520717c69044a94",
        "dataset": "jiosephlee/assay-transfer-record-level-v25-bioavailability-ma-l3-intern",
        "dataset_revision": "b6901b3e7eafca06835a2b9d5b60135ca2432521",
    },
    "L4": {
        "model": "jiosephlee/intern-s1-mini-assay-transfer-record-level-v25-bioavailability-ma-l4-best",
        "revision": "0c3c134a33ef4b67cba78fad228b485ccbd0d05b",
        "dataset": "jiosephlee/assay-transfer-record-level-v25-bioavailability-ma-l4-intern",
        "dataset_revision": "2c14fa0784e03125a89d4d52636023b7e3cd07e2",
    },
    "L6": {
        "model": "jiosephlee/intern-s1-mini-assay-transfer-record-level-v25-bioavailability-ma-l6-best",
        "revision": "6c5f76d6afafd0517c7630e8fc374915d4c2b98a",
        "dataset": "jiosephlee/assay-transfer-record-level-v25-bioavailability-ma-l6-intern",
        "dataset_revision": "7df75e575fb60c297f02b4b21c75b470ae1ec261",
    },
}
MEASUREMENT_KINDS = {"L2": {"continuous", "ordinal"}, "L3": {"continuous"}, "L4": {"continuous"}, "L6": {"continuous"}}
ALWAYS_HIDDEN = {"needs_more_context", "smiles", "source_smiles"}
QUERY_RESULT_FIELDS = {
    "measurement_text", "support_text", "extra_details", "substrate_status",
    "comparator_exposure",
}
QUERY_IDENTITY_FIELDS = {"molecule_name"}
HIDDEN = {
    "confidence", "extraction_id", "global_identifier", "paragraph_idx", "pmid",
    "source_record_id", "source_index",
}
LABELS = {
    "endpoint_name": "Endpoint",
    "measurement_text": "Known source measurement",
    "unit_text": "Source unit",
    "support_text": "Supporting evidence",
    "source_record_id": "Source record ID",
    "source_index": "Source index",
    "source_smiles": "Source SMILES",
    "pmid": "PMID",
}
POPCOUNT = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)


def _profile_levels(single_level: str | None = None, *, l2_only: bool = False) -> tuple[str, tuple[str, ...]]:
    if l2_only:
        if single_level is not None:
            raise ValueError("Specify either l2_only or single_level")
        single_level = "L2"
    if single_level is None:
        return PROFILE, LEVELS
    if single_level not in {"L2", "L3"}:
        raise ValueError(f"Unsupported separate Oral V25 cache level: {single_level}")
    return f"{PROFILE}_{single_level.lower()}", (single_level,)


def cache_paths(subset: str, *, l2_only: bool = False, single_level: str | None = None) -> dict[str, Path]:
    if subset not in {"valid", "test"}:
        raise ValueError(f"unsupported benchmark subset: {subset}")
    profile, _ = _profile_levels(single_level, l2_only=l2_only)
    root = cache_profile_root(profile) / TASK_ID / "scaffold" / subset
    return {
        "root": root,
        "cache": root / "scores.sqlite3",
        "version": root / "VERSION.json",
        "journals": root / ".scores",
        "queries": GOLD_ROOT / f"{subset}_molecule_condition_labels.jsonl",
    }


def _clean(value: Any) -> str | None:
    if value is None or isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(value).strip()
    return text or None


def _fields(payload: Mapping[str, Any], *, known: bool) -> list[tuple[str, str]]:
    hidden = ALWAYS_HIDDEN if known else ALWAYS_HIDDEN | QUERY_RESULT_FIELDS | QUERY_IDENTITY_FIELDS
    return [
        (LABELS.get(name, name.replace("_", " ").capitalize()), value)
        for name, raw in payload.items()
        if name not in hidden and (value := _clean(raw)) is not None
    ]


class V25OralPromptRenderer:
    """Render the released Oral V25 prompt with copied non-result query context."""

    task_id = TASK_ID

    def __init__(self) -> None:
        self.template_hash = file_sha256(PROMPT_ROOT / "prompt.jinja")
        if self.template_hash != TRAINING_TEMPLATE_SHA256:
            raise ValueError("Oral V25 prompt differs from the frozen training template")
        projection = {
            "always_hidden": sorted(ALWAYS_HIDDEN),
            "query_result_fields": sorted(QUERY_RESULT_FIELDS),
            "query_identity_fields": sorted(QUERY_IDENTITY_FIELDS),
            "release_hidden": sorted(HIDDEN),
            "source_field_policy": "source_column_contract.v2:source_or_simply_cleaned",
        }
        self.projection_hash = hashlib.sha256(
            json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.environment = Environment(
            loader=FileSystemLoader(str(PROMPT_ROOT)),
            undefined=StrictUndefined,
            autoescape=False,
        )

    def render(self, record: Mapping[str, Any], query_smiles: str) -> str:
        payload = prompt_payload(record, record["source_fields"])
        known_smiles = str(
            record["source_fields"].get("source_smiles")
            or record["source_fields"].get("smiles")
            or record["canonical_smiles"]
        )
        return self.environment.get_template("prompt.jinja").render(
            known_smiles=known_smiles,
            query_smiles=query_smiles,
            known_fields=_fields(payload, known=True),
            query_fields=_fields(payload, known=False),
        ).strip()


def prompt_payload(record: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen Oral V25/V24 source projection without BBB transforms."""
    return {key: value for key, value in payload.items() if key not in HIDDEN}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _dataset_inputs(levels: Sequence[str] = LEVELS) -> tuple[dict[str, set[str]], dict[str, Path]]:
    from huggingface_hub import snapshot_download

    accepted, files = {}, {}
    for level in levels:
        profile = MODELS[level]
        root = Path(snapshot_download(
            repo_id=profile["dataset"], repo_type="dataset",
            revision=profile["dataset_revision"], local_files_only=True,
        ))
        calibration = root / "calibration.json"
        manifest = root / "manifest.json"
        document = json.loads(calibration.read_text())
        release = json.loads(manifest.read_text())
        expected = {
            "version": "v25", "release_id": "oral_v25",
            "task_id": TASK_ID, "level": level,
        }
        if any(release.get(key) != value for key, value in expected.items()):
            raise ValueError(f"unexpected Oral V25 dataset manifest for {level}")
        accepted[level] = {
            key for key, value in document["accepted_buckets"].items()
            if value.get("level") == level
        }
        kinds = {
            value.get("measurement_kind")
            for key, value in document["accepted_buckets"].items()
            if key in accepted[level]
        }
        if not accepted[level] or kinds != MEASUREMENT_KINDS[level]:
            raise ValueError(f"unexpected Oral V25 {level} measurement kinds: {kinds}")
        for name in ("records", "source_contract", "uid_levels", "gold_validation", "gold_test"):
            item = release["source_snapshot"][name]
            if file_sha256(item["path"]) != item["sha256"]:
                raise ValueError(f"Oral V25 training input changed: {name}")
        files[f"{level}_dataset_calibration"] = calibration
        files[f"{level}_dataset_manifest"] = manifest
    return accepted, files


def _source_fields() -> tuple[dict[str, list[str]], set[str]]:
    contract = json.loads(SOURCE_CONTRACT.read_text())
    if contract.get("contract_version") != "source_column_contract.v2":
        raise ValueError("unexpected Oral V10 source contract")
    by_source, union = {}, set()
    for source, spec in contract["sources"].items():
        fields = [
            name for name, value in spec["normalized_artifact_columns"].items()
            if value.get("source_or_simply_cleaned") is True
        ]
        by_source[source] = fields
        union.update(fields)
    return by_source, union


def _load_records(levels: Sequence[str] = LEVELS) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Path]]:
    accepted, dataset_files = _dataset_inputs(levels)
    source_fields, source_union = _source_fields()
    if file_sha256(LEVEL_MAPPING) != LEVEL_MAPPING_RECEIPT["sha256"]:
        raise ValueError("Tianang-derived Oral source UID mapping is inconsistent")
    core = {
        "source_row_uid", "canonical_record_id", "canonical_smiles", "source_id",
        "pair_bucket_key", "assay_transfer_eligible", "measurement_kind",
        "canonical_pair_fields_json", "canonical_measurement_scale_id",
        "canonical_category_id", "finite_scalar_value", "canonical_measurement_text",
        "canonical_unit_text",
    }
    available = set(pq.read_schema(STAGE3).names)
    columns = sorted((core | source_union) & available)
    missing = core - set(columns)
    if missing:
        raise ValueError(f"Oral V10 Stage 3 lacks required fields: {sorted(missing)}")
    mapping = pq.read_table(LEVEL_MAPPING, columns=["source_row_uid", "level", "family_key"]).to_pylist()
    level_by_uid = {}
    for row in mapping:
        uid = str(row["source_row_uid"])
        value = (f"L{int(row['level'])}", str(row["family_key"]))
        if uid in level_by_uid and level_by_uid[uid] != value:
            raise ValueError(f"conflicting Tianang level mapping for {uid}")
        level_by_uid[uid] = value
    output, mapped_counts = [], {level: 0 for level in levels}
    for raw in pq.read_table(STAGE3, columns=columns).to_pylist():
        membership = level_by_uid.get(str(raw["source_row_uid"]))
        if membership is None or membership[0] not in levels:
            continue
        level, family = membership
        bucket = json.dumps(
            [*json.loads(str(raw["pair_bucket_key"])), level], separators=(",", ":")
        )
        if raw["assay_transfer_eligible"] is not True or bucket not in accepted[level]:
            continue
        if raw["measurement_kind"] not in MEASUREMENT_KINDS[level]:
            raise ValueError(f"unexpected Oral V25 measurement kind: {raw['canonical_record_id']}")
        identity = normalize_molecule_identity(str(raw["canonical_smiles"]))
        if identity.status != "ok" or not identity.parent_smiles:
            raise ValueError(f"cannot normalize V10 record {raw['canonical_record_id']}")
        source_id = str(raw["source_id"])
        payload = {field: raw.get(field) for field in source_fields[source_id]}
        output.append({
            **raw,
            "record_id": str(raw["canonical_record_id"]),
            "level": level,
            "level_family": family,
            "augmented_pair_bucket_key": bucket,
            "parent_smiles": identity.parent_smiles,
            "parent_id": identity.parent_inchi_key or identity.parent_smiles,
            "source_fields": payload,
        })
        mapped_counts[level] += 1
    if any(not mapped_counts[level] for level in levels):
        raise ValueError(f"empty Oral V25 level after UID mapping: {mapped_counts}")
    return output, {"eligible_record_counts": mapped_counts}, dataset_files


def _open_build_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=OFF")
    connection.executescript("""
        CREATE TABLE cache_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE queries(query_id INTEGER PRIMARY KEY, query_smiles TEXT UNIQUE NOT NULL);
        CREATE TABLE groups_dim(group_key INTEGER PRIMARY KEY, group_id TEXT UNIQUE NOT NULL);
        CREATE TABLE molecules(molecule_key INTEGER PRIMARY KEY, molecule_chembl_id TEXT UNIQUE NOT NULL);
        CREATE TABLE records(record_key INTEGER PRIMARY KEY, external_record_id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE refs(record_id TEXT NOT NULL, query_smiles TEXT NOT NULL, group_id TEXT NOT NULL,
                          molecule_id TEXT NOT NULL, PRIMARY KEY(record_id, query_smiles, group_id, molecule_id)) WITHOUT ROWID;
        CREATE TABLE prompt_tasks(prompt_key INTEGER PRIMARY KEY, cache_key TEXT UNIQUE NOT NULL, prompt TEXT NOT NULL);
        CREATE TABLE prompt_assignments(query_id INTEGER NOT NULL, group_key INTEGER NOT NULL,
             molecule_key INTEGER NOT NULL, record_key INTEGER NOT NULL, prompt_key INTEGER NOT NULL,
             PRIMARY KEY(query_id, group_key, molecule_key, record_key)) WITHOUT ROWID;
    """)
    return connection


def _dimension(connection: sqlite3.Connection, table: str, key: str, field: str, value: str) -> int:
    connection.execute(f"INSERT OR IGNORE INTO {table}({field}) VALUES (?)", (value,))
    return int(connection.execute(f"SELECT {key} FROM {table} WHERE {field}=?", (value,)).fetchone()[0])


def _similarities(query_fp: Any, packed: np.ndarray, bit_counts: np.ndarray) -> np.ndarray:
    query = np.frombuffer(DataStructs.BitVectToBinaryText(query_fp), dtype=np.uint8)
    intersection = POPCOUNT[np.bitwise_and(packed, query)].sum(axis=1, dtype=np.uint16)
    union = bit_counts.astype(np.int32) + int(POPCOUNT[query].sum()) - intersection
    return np.divide(intersection, union, out=np.zeros(len(packed)), where=union != 0)


def prepare(subset: str, pool_size: int = POOL_SIZE, *, l2_only: bool = False,
            single_level: str | None = None) -> dict[str, Any]:
    if pool_size != POOL_SIZE:
        raise ValueError(f"Oral V25 cache contract requires Morgan pool size {POOL_SIZE}")
    profile, levels = _profile_levels(single_level, l2_only=l2_only)
    paths = cache_paths(subset, l2_only=l2_only, single_level=single_level)
    paths["root"].mkdir(parents=True, exist_ok=True)
    if paths["cache"].exists() or paths["version"].exists():
        raise FileExistsError(f"Oral V25 cache already exists: {paths['root']}")
    records, record_audit, dataset_files = _load_records(levels)
    queries = _read_jsonl(paths["queries"])
    by_parent: dict[str, dict[str, Any]] = {}
    records_by_level = {level: {} for level in levels}
    for record in records:
        parent = str(record["parent_id"])
        if parent not in by_parent:
            _, _, fp = standardize_smiles_and_fp(str(record["parent_smiles"]))
            if fp is None:
                raise ValueError(f"cannot fingerprint V10 parent {parent}")
            by_parent[parent] = {"canonical_smiles": record["parent_smiles"], "fingerprint": fp,
                                 "parent_smiles_variants": set()}
        by_parent[parent]["parent_smiles_variants"].add(record["parent_smiles"])
        records_by_level[record["level"]].setdefault(parent, []).append(record)
    parents = sorted(by_parent)
    packed = np.stack([
        np.frombuffer(DataStructs.BitVectToBinaryText(by_parent[parent]["fingerprint"]), dtype=np.uint8)
        for parent in parents
    ])
    bit_counts = POPCOUNT[packed].sum(axis=1, dtype=np.uint16)
    connection = _open_build_db(paths["cache"])
    candidate_counts = {level: 0 for level in levels}
    try:
        for index, query in enumerate(queries):
            frozen_identity = query.get("molecule_identity") or {}
            query_smiles = str(frozen_identity.get("parent_smiles") or "")
            query_identity = normalize_molecule_identity(str(query["drug"]))
            frozen_parent = str(frozen_identity.get("parent_inchi_key") or query_smiles)
            current_parent = query_identity.parent_inchi_key or query_identity.parent_smiles
            if not query_smiles or current_parent != frozen_parent:
                raise ValueError(f"frozen query parent identity drift at row {index}")
            _, _, query_fp = standardize_smiles_and_fp(query_smiles)
            if query_fp is None:
                raise ValueError(f"cannot fingerprint query row {index}")
            sims = _similarities(query_fp, packed, bit_counts)
            ranked = sorted(range(len(parents)), key=lambda i: (-float(sims[i]), parents[i]))
            selected = {level: 0 for level in levels}
            for parent_index in ranked:
                parent = parents[parent_index]
                relevant = [level for level in levels if selected[level] < pool_size and parent in records_by_level[level]]
                if relevant and any(
                    decide_candidate(query_identity, {"canonical_smiles": smiles}, "scaffold_disjoint").excluded
                    for smiles in by_parent[parent]["parent_smiles_variants"]
                ):
                    continue
                for level in relevant:
                    rows = records_by_level[level][parent]
                    connection.executemany(
                        "INSERT OR IGNORE INTO refs VALUES (?, ?, ?, ?)",
                        ((row["record_id"], query_smiles, level, parent) for row in rows),
                    )
                    selected[level] += 1
                    candidate_counts[level] += len(rows)
                if all(selected[level] == pool_size for level in levels):
                    break
            if any(selected[level] != pool_size for level in levels):
                raise ValueError(f"query {index} lacks Morgan-{pool_size} coverage: {selected}")
            if (index + 1) % 25 == 0:
                connection.commit()
                print(f"prepared {index + 1}/{len(queries)} queries", flush=True)
        connection.commit()

        wanted = {row[0] for row in connection.execute("SELECT DISTINCT record_id FROM refs")}
        renderer = V25OralPromptRenderer()
        for record in records:
            if record["record_id"] not in wanted:
                continue
            stored = {
                "record_id": record["record_id"], "task_id": TASK_ID,
                "source_id": record["source_id"], "progressive_level": record["level"],
                "level_family": record["level_family"], "canonical_smiles": record["parent_smiles"],
                "measurement_kind": record["measurement_kind"], "source_fields": record["source_fields"],
            }
            connection.execute(
                "INSERT INTO records(external_record_id, payload) VALUES (?, ?)",
                (record["record_id"], json.dumps(stored, ensure_ascii=False, sort_keys=True, default=str)),
            )
        connection.commit()
        if int(connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]) != len(wanted):
            raise ValueError("Oral V25 record hydration is incomplete")

        sql = """SELECT refs.query_smiles, refs.group_id, refs.molecule_id, records.record_key, records.payload
                 FROM refs JOIN records ON refs.record_id=records.external_record_id
                 ORDER BY refs.query_smiles, refs.group_id, refs.molecule_id, refs.record_id"""
        assignments = duplicates = 0
        for query_smiles, level, molecule_id, record_key, payload_json in connection.execute(sql):
            record = json.loads(payload_json)
            prompt = renderer.render(record, str(query_smiles))
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            model = MODELS[level]
            identity = {
                "prompt_hash": prompt_hash, "model": model["model"], "model_revision": model["revision"],
                "scoring_contract_version": SCORING_CONTRACT_VERSION,
                "template_hash": renderer.template_hash, "projection_hash": renderer.projection_hash,
            }
            cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            query_id = _dimension(connection, "queries", "query_id", "query_smiles", str(query_smiles))
            group_key = _dimension(connection, "groups_dim", "group_key", "group_id", str(level))
            molecule_key = _dimension(connection, "molecules", "molecule_key", "molecule_chembl_id", str(molecule_id))
            inserted = connection.execute("INSERT OR IGNORE INTO prompt_tasks(cache_key,prompt) VALUES (?,?)", (cache_key, prompt)).rowcount
            duplicates += int(not inserted)
            prompt_key = int(connection.execute("SELECT prompt_key FROM prompt_tasks WHERE cache_key=?", (cache_key,)).fetchone()[0])
            connection.execute("INSERT INTO prompt_assignments VALUES (?,?,?,?,?)", (query_id, group_key, molecule_key, record_key, prompt_key))
            assignments += 1
            if assignments % 100_000 == 0:
                connection.commit()
                print(f"rendered {assignments} assignments", flush=True)
        connection.commit()

        input_paths = {
            "stage3_records": STAGE3, "stage3_manifest": STAGE3_MANIFEST,
            "source_contract": SOURCE_CONTRACT, "level_mapping": LEVEL_MAPPING,
            "level_manifest": LEVEL_MANIFEST, "queries": paths["queries"],
            "prompt_template": PROMPT_ROOT / "prompt.jinja", **dataset_files,
        }
        version = {
            "schema_version": COMPACT_CACHE_SCHEMA_VERSION, "status": "prepared",
            "profile": profile, "task_id": TASK_ID, "subset": subset,
            "models": {level: MODELS[level] for level in levels}, "levels_in_cache": list(levels),
            "scoring_contract_version": SCORING_CONTRACT_VERSION,
            "backbone_dtype": BACKBONE_DTYPE, "logit_extraction_dtype": LOGIT_EXTRACTION_DTYPE,
            "template_hash": renderer.template_hash, "projection_hash": renderer.projection_hash,
            "training_oral_projection_sha256": TRAINING_ORAL_PROJECTION_SHA256,
            "candidate_contract": "v10_uid_level_then_morgan_top75_all_parent_forms_scaffold_disjoint.oral_v25",
            "record_scope": RECORD_SCOPE,
            "query_context_policy": "copy_retrieval_non_result_non_identity_source_context.oral_v25",
            "neighbor_identity_policy": "scaffold_disjoint", "morgan_pool_size": pool_size,
            "morgan_fingerprint": {"radius": 2, "bits": 2048, "similarity": "Tanimoto"},
            "n_query_rows": len(queries), "n_unique_query_parents": int(connection.execute("SELECT COUNT(*) FROM queries").fetchone()[0]),
            "n_catalog_records": len(wanted), "n_score_assignments": assignments,
            "n_prompt_scores": int(connection.execute("SELECT COUNT(*) FROM prompt_tasks").fetchone()[0]),
            "n_duplicate_prompt_tasks_collapsed": duplicates, "level_assignment_counts": candidate_counts,
            **record_audit, "inputs": {name: {"path": str(path.resolve()), "sha256": file_sha256(path)} for name, path in input_paths.items()},
            "cache": str(paths["cache"].resolve()),
        }
        for key in ("schema_version", "status", "task_id", "profile", "subset", "models",
                    "scoring_contract_version", "backbone_dtype", "logit_extraction_dtype",
                    "template_hash", "projection_hash", "candidate_contract", "record_scope"):
            connection.execute("INSERT INTO cache_metadata VALUES (?,?)", (key, json.dumps(version[key], sort_keys=True)))
        connection.execute("DROP TABLE refs")
        connection.commit()
    finally:
        connection.close()
    _write_json(paths["version"], version)
    return version


def _level_prompt_rows(connection: sqlite3.Connection, level: str) -> list[tuple[int, str, str]]:
    return connection.execute("""
        SELECT DISTINCT p.prompt_key, p.cache_key, p.prompt
        FROM prompt_tasks p JOIN prompt_assignments a USING(prompt_key)
        JOIN groups_dim g USING(group_key) WHERE g.group_id=?
        ORDER BY length(p.prompt), p.prompt_key
    """, (level,)).fetchall()


def _runtime_metadata() -> dict[str, Any]:
    import torch

    packages = {}
    for name in ("transformers", "tokenizers", "huggingface-hub", "flash-attn", "rdkit"):
        try:
            packages[name] = package_version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "packages": packages,
    }


def score(subset: str, level: str, *, device: int, batch_size: int = 128,
          num_shards: int = 1, shard_index: int = 0) -> dict[str, Any]:
    if level not in MODELS or batch_size < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid Oral V25 level or batch size")
    single_level = level if level in {"L2", "L3"} else None
    _, levels = _profile_levels(single_level)
    paths = cache_paths(subset, single_level=single_level)
    version = json.loads(paths["version"].read_text())
    if version.get("status") != "prepared" or version.get("models") != {key: MODELS[key] for key in levels}:
        raise ValueError("Oral V25 cache is not prepared with the pinned models")
    connection = sqlite3.connect(f"file:{paths['cache'].resolve()}?mode=ro", uri=True)
    rows = _level_prompt_rows(connection, level)[shard_index::num_shards]
    connection.close()
    paths["journals"].mkdir(exist_ok=True)
    suffix = "" if num_shards == 1 else f"-{shard_index:02d}-of-{num_shards:02d}"
    journal = paths["journals"] / f"scores-{level}{suffix}.jsonl"
    completed = _read_jsonl(journal) if journal.exists() else []
    completed_keys = [row["cache_key"] for row in completed]
    expected_keys = {row[1] for row in rows}
    if len(completed_keys) != len(set(completed_keys)) or not set(completed_keys) <= expected_keys:
        raise ValueError(f"invalid Oral V25 score journal: {journal}")
    completed_keys = set(completed_keys)
    pending = [row for row in rows if row[1] not in completed_keys]
    model_info = MODELS[level]
    snapshot = resolve_model_snapshot(model_info["model"], model_info["revision"], local_files_only=True)
    model, tokenizer = load_model(snapshot, device=device)
    renderer = V25OralPromptRenderer()
    with journal.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            tasks = [PromptTask(
                cache_key=key, prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(), prompt=prompt,
                task_id=TASK_ID, query_smiles="", group_id=level, molecule_id="", record_id="",
                model=model_info["model"], model_revision=model_info["revision"],
                scoring_contract_version=SCORING_CONTRACT_VERSION,
                template_hash=renderer.template_hash, projection_hash=renderer.projection_hash,
            ) for _, key, prompt in batch]
            for result in score_prompt_batch(model, tokenizer, tasks, device=device):
                handle.write(json.dumps({"cache_key": result.cache_key, "transfer_probability": result.transfer_probability}, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(f"{subset} {level}: {len(completed) + offset + len(batch)}/{len(rows)}", flush=True)
    return {"status": "complete", "subset": subset, "level": level, "n_scores": len(rows),
            "num_shards": num_shards, "shard_index": shard_index, "journal": str(journal)}


def finalize(subset: str, *, l2_only: bool = False, single_level: str | None = None) -> dict[str, Any]:
    _, levels = _profile_levels(single_level, l2_only=l2_only)
    paths = cache_paths(subset, l2_only=l2_only, single_level=single_level)
    version = json.loads(paths["version"].read_text())
    if version.get("status") != "prepared":
        raise ValueError("Oral V25 cache is not prepared")
    connection = sqlite3.connect(paths["cache"])
    try:
        connection.execute("CREATE TEMP TABLE prompt_scores(prompt_key INTEGER PRIMARY KEY, transfer_probability REAL NOT NULL)")
        for level in levels:
            expected = _level_prompt_rows(connection, level)
            journal = paths["journals"] / f"scores-{level}.jsonl"
            observed = _read_jsonl(journal) if journal.exists() else []
            for shard in sorted(paths["journals"].glob(f"scores-{level}-*-of-*.jsonl")):
                observed.extend(_read_jsonl(shard))
            scores = {row["cache_key"]: float(row["transfer_probability"]) for row in observed}
            if len(scores) != len(observed) or set(scores) != {row[1] for row in expected}:
                raise ValueError(f"incomplete Oral V25 score journal: {journal}")
            if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in scores.values()):
                raise ValueError(f"invalid probability in Oral V25 score journal: {journal}")
            connection.executemany("INSERT INTO prompt_scores VALUES (?,?)", ((key, scores[cache_key]) for key, cache_key, _ in expected))
        connection.executescript("""
            CREATE TABLE scores(score_key INTEGER PRIMARY KEY, transfer_probability REAL NOT NULL);
            INSERT INTO scores SELECT prompt_key, transfer_probability FROM prompt_scores;
            CREATE TABLE assignments(query_id INTEGER NOT NULL, group_key INTEGER NOT NULL,
                molecule_key INTEGER NOT NULL, record_key INTEGER NOT NULL, score_key INTEGER NOT NULL,
                PRIMARY KEY(query_id, group_key, molecule_key, record_key)) WITHOUT ROWID;
            INSERT INTO assignments SELECT query_id, group_key, molecule_key, record_key, prompt_key FROM prompt_assignments;
            DROP TABLE prompt_assignments; DROP TABLE prompt_scores; DROP TABLE prompt_tasks;
        """)
        level_assignment_counts = {
            str(level): int(count)
            for level, count in connection.execute(
                "SELECT g.group_id, COUNT(*) FROM assignments a "
                "JOIN groups_dim g USING(group_key) GROUP BY g.group_id"
            )
        }
        connection.execute("UPDATE cache_metadata SET value=? WHERE key='status'", (json.dumps("complete"),))
        connection.execute(
            "UPDATE cache_metadata SET value=? WHERE key='record_scope'",
            (json.dumps(RECORD_SCOPE),),
        )
        connection.commit()
        n_scores = int(connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0])
    finally:
        connection.close()
    connection = sqlite3.connect(paths["cache"])
    connection.execute("VACUUM")
    quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
    connection.close()
    if quick_check != "ok":
        raise ValueError(f"Oral V25 SQLite quick check failed: {quick_check}")
    version.update(
        status="complete",
        record_scope=RECORD_SCOPE,
        n_cached_scores=n_scores,
        level_candidate_record_visits=version["level_assignment_counts"],
        level_assignment_counts=level_assignment_counts,
        cache_quick_check=quick_check,
        cache_sha256=file_sha256(paths["cache"]),
        implementation_sha256=file_sha256(Path(__file__)),
        scoring_runtime=_runtime_metadata(),
    )
    _write_json(paths["version"], version)
    for journal in paths["journals"].glob("*.jsonl"):
        journal.unlink()
    paths["journals"].rmdir()
    return version


def load_top_ranked_records(
    queries: Mapping[str, str], *, subset: str, levels: Sequence[str] = LEVELS,
    limit: int | Mapping[str, int] = 50, workers: int = 8,
    exclude_record_ids_by_query: Mapping[str, Iterable[str]] | None = None,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    """Read highest-probability Oral V25 records for frozen valid/test queries."""
    requested = tuple(dict.fromkeys(levels))
    separate = set(requested) & {"L2", "L3"}
    if separate and len(requested) != 1:
        raise ValueError("Load the separate L2 cache or L3 cache independently of other levels")
    single_level = requested[0] if separate else None
    profile, cache_levels = _profile_levels(single_level)
    paths = cache_paths(subset, single_level=single_level)
    version = json.loads(paths["version"].read_text())
    expected = {"status": "complete", "profile": profile, "task_id": TASK_ID, "subset": subset,
                "models": {level: MODELS[level] for level in cache_levels}, "neighbor_identity_policy": "scaffold_disjoint"}
    for key, value in expected.items():
        if version.get(key) != value:
            raise ValueError(f"Oral V25 cache has wrong {key}: {version.get(key)!r}")
    if version.get("cache_sha256") != file_sha256(paths["cache"]) or version.get("cache_quick_check") != "ok":
        raise ValueError("Oral V25 cache integrity mismatch")
    accepted_builders = {file_sha256(Path(__file__))}
    if profile in AUDITED_BUILDERS:
        accepted_builders.add(AUDITED_BUILDERS[profile])
    if version.get("implementation_sha256") not in accepted_builders:
        raise ValueError("Oral V25 cache builder changed")
    for item in version["inputs"].values():
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Oral V25 cache input changed: {item['path']}")
    frozen = {str(row["benchmark_row_id"]): row for row in _read_jsonl(paths["queries"])}
    normalized = {}
    for query_id, smiles in queries.items():
        row = frozen.get(str(query_id))
        if row is None or str(row["drug"]) != str(smiles):
            raise ValueError(f"query {query_id} differs from the frozen {subset} ledger")
        normalized[str(query_id)] = str(row["molecule_identity"]["parent_smiles"])
    limits = {level: int(limit.get(level, 0) if isinstance(limit, Mapping) else limit) for level in requested}
    if not requested or not set(requested) <= set(cache_levels) or any(value < 1 for value in limits.values()):
        raise ValueError("invalid Oral V25 record levels or limits")

    def load_one(item: tuple[str, str]) -> tuple[str, dict[str, dict[str, Any]]]:
        query_id, parent_smiles = item
        seen = set(exclude_record_ids_by_query.get(query_id, ())) if exclude_record_ids_by_query else set()
        connection = sqlite3.connect(f"file:{paths['cache'].resolve()}?mode=ro", uri=True)
        result = {}
        try:
            for level in requested:
                rows = connection.execute("""
                    SELECT r.external_record_id,m.molecule_chembl_id,s.transfer_probability,r.payload
                    FROM assignments a JOIN queries q USING(query_id) JOIN groups_dim g USING(group_key)
                    JOIN records r USING(record_key) JOIN molecules m USING(molecule_key) JOIN scores s USING(score_key)
                    WHERE q.query_smiles=? AND g.group_id=?
                    ORDER BY s.transfer_probability DESC,r.external_record_id
                """, (parent_smiles, level)).fetchall()
                available = len(rows)
                selected = [row for row in rows if str(row[0]) not in seen][:limits[level]]
                if len(selected) != limits[level]:
                    raise ValueError(f"query {query_id} has only {len(selected)} unseen {level} records")
                records = []
                for record_id, molecule_id, probability, payload_json in selected:
                    seen.add(str(record_id))
                    records.append({"record_id": str(record_id), "reference_molecule_id": str(molecule_id),
                                    "transfer_likelihood": float(probability), "payload": json.loads(payload_json)})
                result[level] = {"available_record_count": available, "records": records}
            return query_id, result
        finally:
            connection.close()

    work = list(normalized.items())
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(work))) as pool:
        selected = dict(pool.map(load_one, work))
    return selected, {"profile": version["profile"], "subset": subset, "version": str(paths["version"]),
                      "version_sha256": file_sha256(paths["version"]), "cache": str(paths["cache"]),
                      "cache_sha256": version["cache_sha256"], "models": version["models"], "levels": list(requested),
                      "records_per_level": limits, "n_query_rows": len(queries),
                      "n_unique_query_parents": len(set(normalized.values())),
                      "candidate_contract": version["candidate_contract"], "record_scope": version["record_scope"],
                      "neighbor_identity_policy": "scaffold_disjoint", "assay_transfer_scores_used": True,
                      "inputs": version["inputs"]}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--subset", choices=("valid", "test"), required=True)
    prepare_parser.add_argument("--pool-size", type=int, default=POOL_SIZE)
    prepare_parser.add_argument("--l2-only", action="store_true")
    prepare_parser.add_argument("--single-level", choices=("L2", "L3"))
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--subset", choices=("valid", "test"), required=True)
    score_parser.add_argument("--level", choices=tuple(MODELS), required=True)
    score_parser.add_argument("--device", type=int, required=True)
    score_parser.add_argument("--batch-size", type=int, default=128)
    score_parser.add_argument("--num-shards", type=int, default=1)
    score_parser.add_argument("--shard-index", type=int, default=0)
    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("--subset", choices=("valid", "test"), required=True)
    finalize_parser.add_argument("--l2-only", action="store_true")
    finalize_parser.add_argument("--single-level", choices=("L2", "L3"))
    args = parser.parse_args(argv)
    result = (prepare(args.subset, args.pool_size, l2_only=args.l2_only, single_level=args.single_level) if args.command == "prepare" else
              score(args.subset, args.level, device=args.device, batch_size=args.batch_size,
                    num_shards=args.num_shards, shard_index=args.shard_index) if args.command == "score" else
              finalize(args.subset, l2_only=args.l2_only, single_level=args.single_level))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
