"""Build current Stage 3 progressive record-transfer score caches."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
from rdkit import DataStructs

from predict.retrieval.assay_reranking.runtime import (
    BACKBONE_DTYPE,
    CACHE_ROOT,
    COMPACT_CACHE_SCHEMA_VERSION,
    LOGIT_EXTRACTION_DTYPE,
    SCORING_CONTRACT_VERSION,
    PromptTask,
    file_sha256,
    load_model,
    model_profile,
    resolve_model_snapshot,
    score_prompt_batch,
)
from predict.retrieval.assay_reranking.v19_1 import (
    ASSET_ROOT,
    PROJECTION_SHA256,
    TRAINING_PROMPT_SHA256,
    V191PromptRenderer,
)
from predict.retrieval.policies import (
    decide_candidate,
    normalize_molecule_identity,
    standardize_smiles_and_fp,
)


TASK_CONFIGS = {
    "bbb_martins": {
        "cache_profile": "v19_1_bbb_stage3_progressive_top75_scaffold_disjoint",
        "gold_task": "BBB_Martins",
        "levels": ("L2", "L3", "L4", "L5"),
        "extension_file": "direct_bbb_extension.json",
        "extension_source": "direct_bbb",
        "level_contract": {
            "L1": "current vote-ledger source records; globally excluded here",
            "L2": "current near-direct plus current non-voting former L1",
            "L3": "current passive permeability",
            "L4": "current efflux transport",
            "L5": "current influx transport",
        },
    },
    "bioavailability_ma": {
        "cache_profile": "v19_1_bioavailability_ma_stage3_progressive_top75_scaffold_disjoint",
        "gold_task": "Bioavailability_Ma",
        "levels": ("L2", "L3", "L4", "L5", "L6"),
        "extension_file": "direct_bioavailability_extension.json",
        "extension_source": "hf_bioavailability",
        "level_contract": {
            "L1": "current direct oral-bioavailability voters; globally excluded here",
            "L2": "current nondirect oral bioavailability plus non-voting former L1",
            "L3": "current oral AUC/Cmax exposure",
            "L4": "current intestinal absorption and permeability",
            "L5": "current gut-wall efflux and intestinal metabolism",
            "L6": "current hepatic clearance and metabolic stability",
        },
    },
    "skin_reaction": {
        "cache_profile": "v19_1_skin_reaction_stage3_progressive_top75_scaffold_disjoint",
        "gold_task": "Skin_Reaction",
        "levels": ("L3", "L4"),
        "extension_file": "direct_skin_reaction_extension.json",
        "extension_source": "direct_skin_reaction",
        "level_contract": {
            "L1": "current sensitization/contact-allergy voters; existing V9 cache",
            "L2": "current non-voting skin outcomes; excluded here",
            "L3": "current sensitization AOP evidence",
            "L4": "current phototoxicity, irritation, corrosion, and local skin damage",
        },
    },
}
POPCOUNT = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)


class ProgressiveV191PromptRenderer(V191PromptRenderer):
    """Use V19.1 with the task's explicit out-of-domain direct-source binding."""

    def __init__(self, task_id: str = "bbb_martins") -> None:
        config = TASK_CONFIGS[task_id]
        super().__init__(task_id)
        extension_path = ASSET_ROOT / str(config["extension_file"])
        extension = json.loads(extension_path.read_text(encoding="utf-8"))
        if extension.get("base_projection_sha256") != PROJECTION_SHA256:
            raise ValueError("Direct-source extension targets another V19.1 projection")
        self.projection["labels"].update(extension["labels"])
        self.projection["tasks"][task_id][str(config["extension_source"])] = extension[
            "binding"
        ]
        merged = json.dumps(
            self.projection, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        self.projection_hash = hashlib.sha256(merged).hexdigest()


def _paths(task_id: str) -> dict[str, Path]:
    config = TASK_CONFIGS[task_id]
    profile = str(config["cache_profile"])
    output_root = CACHE_ROOT / profile / task_id / "scaffold/valid"
    return {
        "records": Path(
            f"data/evidence_libraries/{task_id}/v7/03_pair_buckets/records.parquet"
        ),
        "mapping": Path(
            f"data/artifacts/evidence_library_assets/{task_id}/construction_assets/"
            "progressive_level_mapping_v1/record_levels.parquet"
        ),
        "mapping_manifest": Path(
            f"data/artifacts/evidence_library_assets/{task_id}/construction_assets/"
            "progressive_level_mapping_v1/manifest.json"
        ),
        "queries": Path(
            f"data/gold_labels/{config['gold_task']}/v1/scaffold/"
            "valid_molecule_condition_labels.jsonl"
        ),
        "extension": ASSET_ROOT / str(config["extension_file"]),
        "cache": output_root / "scores.sqlite3",
        "version": output_root / "VERSION.json",
        "journals": output_root / ".scores",
    }


def load_top_ranked_records(
    task_id: str,
    queries: Mapping[str, str],
    *,
    levels: Sequence[str] = ("L3", "L4", "L5"),
    limit: int = 50,
    workers: int = 8,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    """Read the highest-scored Stage 3 records from one finalized cache.

    ``queries`` maps frozen benchmark row IDs to their current split SMILES. The
    cache's pinned query ledger supplies the exact parent-SMILES spelling used as
    the SQLite key. Every cache and source hash is checked before records are read.
    """
    if task_id not in TASK_CONFIGS:
        raise ValueError(f"unsupported progressive cache task: {task_id}")
    if limit < 1 or workers < 1:
        raise ValueError("limit and workers must be positive")
    if not queries:
        raise ValueError("at least one query is required")
    requested_levels = tuple(dict.fromkeys(str(level) for level in levels))
    if not requested_levels:
        raise ValueError("at least one progressive level is required")

    paths = _paths(task_id)
    version = json.loads(paths["version"].read_text(encoding="utf-8"))
    expected = {
        "schema_version": COMPACT_CACHE_SCHEMA_VERSION,
        "status": "complete",
        "profile": TASK_CONFIGS[task_id]["cache_profile"],
        "task_id": task_id,
        "model": model_profile(task_id, "indirect")["model"],
        "model_revision": model_profile(task_id, "indirect")["revision"],
        "candidate_contract": "current_stage3_level_then_morgan_top75_scaffold_disjoint.v1",
        "record_scope": "normalized_v7_stage3_with_current_vote_pure_level_mapping.v1",
        "neighbor_identity_policy": "scaffold_disjoint",
        "cache_quick_check": "ok",
    }
    for field, value in expected.items():
        if version.get(field) != value:
            raise ValueError(
                f"{task_id} progressive cache has wrong {field}: {version.get(field)!r}"
            )
    if not set(requested_levels) <= set(version.get("levels_in_cache") or []):
        raise ValueError(f"{task_id} progressive cache lacks {requested_levels}")
    if version.get("cache_sha256") != file_sha256(paths["cache"]):
        raise ValueError(f"{task_id} progressive cache hash disagrees with VERSION.json")
    for source, expected_hash in (version.get("inputs") or {}).items():
        source_path = Path(source)
        if not source_path.is_file() or file_sha256(source_path) != expected_hash:
            raise ValueError(f"{task_id} progressive cache input changed: {source_path}")

    frozen_queries = {
        str(row["benchmark_row_id"]): row for row in _read_jsonl(paths["queries"])
    }
    normalized_queries: dict[str, str] = {}
    for query_id, smiles in queries.items():
        frozen = frozen_queries.get(str(query_id))
        if frozen is None:
            raise ValueError(f"query {query_id} is absent from the frozen cache ledger")
        if str(frozen.get("drug") or "") != str(smiles):
            raise ValueError(f"query {query_id} SMILES differs from the frozen cache ledger")
        parent_smiles = str(
            (frozen.get("molecule_identity") or {}).get("parent_smiles") or ""
        )
        if not parent_smiles:
            raise ValueError(f"query {query_id} has no frozen parent SMILES")
        normalized_queries[str(query_id)] = parent_smiles

    sql = """
        SELECT r.external_record_id, m.molecule_chembl_id,
               s.transfer_probability, r.payload, COUNT(*) OVER ()
        FROM assignments AS a
        JOIN queries AS q USING(query_id)
        JOIN groups_dim AS g USING(group_key)
        JOIN records AS r USING(record_key)
        JOIN molecules AS m USING(molecule_key)
        JOIN scores AS s USING(score_key)
        WHERE q.query_smiles = ? AND g.group_id = ?
        ORDER BY s.transfer_probability DESC, r.external_record_id
        LIMIT ?
    """

    def load_one(item: tuple[str, str]) -> tuple[str, dict[str, dict[str, Any]]]:
        query_id, parent_smiles = item
        connection = sqlite3.connect(
            f"file:{paths['cache'].resolve()}?mode=ro", uri=True
        )
        try:
            result: dict[str, dict[str, Any]] = {}
            for level in requested_levels:
                rows = connection.execute(sql, (parent_smiles, level, limit)).fetchall()
                if len(rows) != limit:
                    raise ValueError(
                        f"{task_id} query {query_id} has {len(rows)} {level} records; "
                        f"exactly {limit} are required"
                    )
                records = []
                for record_id, molecule_id, score_value, payload_json, available in rows:
                    payload = json.loads(payload_json)
                    if (
                        str(payload.get("record_id") or "") != str(record_id)
                        or str(payload.get("progressive_level") or "") != level
                    ):
                        raise ValueError(
                            f"{task_id} cache payload disagrees with {record_id} at {level}"
                        )
                    records.append(
                        {
                            "record_id": str(record_id),
                            "reference_molecule_id": str(molecule_id),
                            "transfer_likelihood": float(score_value),
                            "payload": payload,
                        }
                    )
                result[level] = {
                    "available_record_count": int(rows[0][4]),
                    "records": records,
                }
            return query_id, result
        finally:
            connection.close()

    unique_queries = dict.fromkeys(normalized_queries.values())
    by_parent: dict[str, dict[str, dict[str, Any]]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(workers, len(unique_queries))
    ) as pool:
        for parent_smiles, result in pool.map(
            load_one, ((smiles, smiles) for smiles in unique_queries)
        ):
            by_parent[parent_smiles] = result
    selected = {
        query_id: by_parent[parent_smiles]
        for query_id, parent_smiles in normalized_queries.items()
    }
    return selected, {
        "version": str(paths["version"]),
        "version_sha256": file_sha256(paths["version"]),
        "cache": str(paths["cache"]),
        "cache_sha256": version["cache_sha256"],
        "profile": version["profile"],
        "model": version["model"],
        "model_revision": version["model_revision"],
        "levels": list(requested_levels),
        "records_per_level": limit,
        "n_query_rows": len(queries),
        "n_unique_query_parents": len(unique_queries),
        "candidate_contract": version["candidate_contract"],
        "record_scope": version["record_scope"],
        "neighbor_identity_policy": version["neighbor_identity_policy"],
        "inputs": dict(version.get("inputs") or {}),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _level_by_record(
    mapping_path: Path, levels_in_cache: Sequence[str]
) -> tuple[dict[str, str], Counter]:
    mapping = pq.read_table(
        mapping_path,
        columns=[
            "canonical_record_id",
            "progressive_level",
            "source_id",
            "finite_scalar_value_present",
            "score_cache_candidate_eligible",
            "scoring_domain_flags",
        ],
    ).to_pylist()
    levels: dict[str, str] = {}
    counts: Counter = Counter()
    for row in mapping:
        level = str(row.get("progressive_level") or "")
        if level not in {"L1", *levels_in_cache}:
            continue
        record_id = str(row["canonical_record_id"])
        if record_id in levels:
            raise ValueError(f"duplicate mapping record: {record_id}")
        levels[record_id] = level
        counts[
            (level, str(row["source_id"]), bool(row["finite_scalar_value_present"]))
        ] += 1
    return levels, counts


def _tanimoto_vector(query_fp: Any, packed: np.ndarray, bit_counts: np.ndarray) -> np.ndarray:
    query = np.frombuffer(DataStructs.BitVectToBinaryText(query_fp), dtype=np.uint8)
    intersection = POPCOUNT[np.bitwise_and(packed, query)].sum(axis=1, dtype=np.uint16)
    union = bit_counts.astype(np.int32) + int(POPCOUNT[query].sum()) - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros(len(packed), dtype=np.float64),
        where=union != 0,
    )


def _open_build_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=OFF")
    connection.executescript(
        """
        CREATE TABLE cache_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE queries(query_id INTEGER PRIMARY KEY, query_smiles TEXT UNIQUE NOT NULL);
        CREATE TABLE groups_dim(group_key INTEGER PRIMARY KEY, group_id TEXT UNIQUE NOT NULL);
        CREATE TABLE molecules(
            molecule_key INTEGER PRIMARY KEY, molecule_chembl_id TEXT UNIQUE NOT NULL
        );
        CREATE TABLE records(
            record_key INTEGER PRIMARY KEY,
            external_record_id TEXT UNIQUE NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE TABLE refs(
            record_id TEXT NOT NULL,
            query_smiles TEXT NOT NULL,
            group_id TEXT NOT NULL,
            molecule_id TEXT NOT NULL,
            PRIMARY KEY(record_id, query_smiles, group_id, molecule_id)
        ) WITHOUT ROWID;
        CREATE TABLE prompt_tasks(
            prompt_key INTEGER PRIMARY KEY,
            cache_key TEXT UNIQUE NOT NULL,
            prompt TEXT NOT NULL
        );
        CREATE TABLE prompt_assignments(
            query_id INTEGER NOT NULL,
            group_key INTEGER NOT NULL,
            molecule_key INTEGER NOT NULL,
            record_key INTEGER NOT NULL,
            prompt_key INTEGER NOT NULL,
            PRIMARY KEY(query_id, group_key, molecule_key, record_key)
        ) WITHOUT ROWID;
        """
    )
    return connection


def _dimension_key(
    connection: sqlite3.Connection,
    table: str,
    key_column: str,
    value_column: str,
    value: str,
) -> int:
    connection.execute(
        f"INSERT OR IGNORE INTO {table}({value_column}) VALUES (?)", (value,)
    )
    return int(
        connection.execute(
            f"SELECT {key_column} FROM {table} WHERE {value_column}=?", (value,)
        ).fetchone()[0]
    )


def _prepare_candidates(
    connection: sqlite3.Connection,
    paths: Mapping[str, Path],
    records_by_level: Mapping[str, Mapping[int, list[str]]],
    molecules: Sequence[Mapping[str, Any]],
    packed: np.ndarray,
    pool_size: int,
    levels_in_cache: Sequence[str],
) -> tuple[int, Counter]:
    molecule_ids = [str(row["molecule_id"]) for row in molecules]
    bit_counts = POPCOUNT[packed].sum(axis=1, dtype=np.uint16)
    level_sets = {level: set(rows) for level, rows in records_by_level.items()}
    queries = _read_jsonl(paths["queries"])
    counts: Counter = Counter()
    for query_index, query in enumerate(queries):
        stored_identity = query.get("molecule_identity") or {}
        query_smiles = str(stored_identity.get("parent_smiles") or "")
        if not query_smiles:
            raise ValueError(f"Query {query_index} lacks its frozen parent SMILES")
        current_identity = normalize_molecule_identity(str(query["drug"]))
        frozen_parent = str(stored_identity.get("parent_inchi_key") or query_smiles)
        current_parent = (
            current_identity.parent_inchi_key or current_identity.parent_smiles
        )
        if current_parent != frozen_parent:
            raise ValueError(f"Query {query_index} parent identity drifted from its gold row")
        _, _, query_fp = standardize_smiles_and_fp(query_smiles)
        if query_fp is None:
            raise ValueError(f"Invalid query SMILES at row {query_index}")
        query_identity = normalize_molecule_identity(query_smiles)
        similarities = _tanimoto_vector(query_fp, packed, bit_counts)
        ranked = sorted(
            range(len(molecules)),
            key=lambda index: (-float(similarities[index]), molecule_ids[index]),
        )
        selected = {level: 0 for level in levels_in_cache}
        for molecule_index in ranked:
            relevant = [
                level
                for level in levels_in_cache
                if selected[level] < pool_size and molecule_index in level_sets[level]
            ]
            if not relevant:
                if all(selected[level] == pool_size for level in levels_in_cache):
                    break
                continue
            if decide_candidate(
                query_identity, molecules[molecule_index], "scaffold_disjoint"
            ).excluded:
                for level in relevant:
                    counts[(level, "identity_excluded")] += 1
                continue
            for level in relevant:
                record_ids = records_by_level[level][molecule_index]
                connection.executemany(
                    "INSERT OR IGNORE INTO refs VALUES (?, ?, ?, ?)",
                    (
                        (record_id, query_smiles, level, molecule_ids[molecule_index])
                        for record_id in record_ids
                    ),
                )
                selected[level] += 1
                counts[(level, "candidate_molecules")] += 1
                counts[(level, "record_assignments")] += len(record_ids)
        if any(selected[level] != pool_size for level in levels_in_cache):
            raise ValueError(f"Query {query_index} lacks {pool_size} candidates: {selected}")
        if (query_index + 1) % 25 == 0:
            connection.commit()
            print(f"prepared candidates for {query_index + 1}/{len(queries)} queries", flush=True)
    connection.commit()
    return len(queries), counts


def _record_payload(
    renderer: ProgressiveV191PromptRenderer,
    row: Mapping[str, Any],
    level: str,
    parent_smiles: str,
    task_id: str,
) -> dict[str, Any]:
    source_id = str(row.get("source_id") or "")
    fields = renderer.prompt_fields(source_id)
    source_fields = {field: row.get(field) for field in fields}
    return {
        "record_id": str(row["canonical_record_id"]),
        "task_id": task_id,
        "source_id": source_id,
        "progressive_level": level,
        "canonical_smiles": parent_smiles,
        "source_canonical_smiles": str(row.get("canonical_smiles") or ""),
        "measurement_kind": "numeric" if row.get("finite_scalar_value") is not None else "nonnumeric",
        "training_measurement_kind_supported": row.get("finite_scalar_value") is not None,
        "source_contract": {
            "source_id": source_id,
            "record_contract_version": "starling_record_contract.v7",
        },
        "source_fields": source_fields,
        **source_fields,
    }


def _hydrate_and_render(
    connection: sqlite3.Connection,
    paths: Mapping[str, Path],
    renderer: ProgressiveV191PromptRenderer,
    level_by_record: Mapping[str, str],
    record_parent_smiles: Mapping[str, str],
    task_id: str,
) -> tuple[int, int, int]:
    wanted = {
        str(row[0]) for row in connection.execute("SELECT DISTINCT record_id FROM refs")
    }
    found = 0
    parquet = pq.ParquetFile(paths["records"])
    for batch in parquet.iter_batches(batch_size=16_384):
        rows = []
        for raw in batch.to_pylist():
            record_id = str(raw["canonical_record_id"])
            if record_id not in wanted:
                continue
            payload = _record_payload(
                renderer,
                raw,
                level_by_record[record_id],
                record_parent_smiles[record_id],
                task_id,
            )
            rows.append(
                (record_id, json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
            )
        if rows:
            connection.executemany(
                "INSERT OR REPLACE INTO records(external_record_id, payload) VALUES (?, ?)",
                rows,
            )
            connection.commit()
            found += len(rows)
    if found != len(wanted):
        raise ValueError(f"Record hydration mismatch: expected={len(wanted)}, found={found}")

    profile = model_profile(task_id, "indirect")
    assignments = duplicates = 0
    query = """
        SELECT refs.query_smiles, refs.group_id, refs.molecule_id,
               records.record_key, records.payload
        FROM refs JOIN records ON refs.record_id = records.external_record_id
        ORDER BY refs.query_smiles, refs.group_id, refs.molecule_id, refs.record_id
    """
    for query_smiles, group_id, molecule_id, record_key, payload_json in connection.execute(query):
        record = json.loads(payload_json)
        prompt = renderer.render(record, str(query_smiles))
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        identity = {
            "prompt_hash": prompt_hash,
            "model": profile["model"],
            "model_revision": profile["revision"],
            "scoring_contract_version": SCORING_CONTRACT_VERSION,
            "template_hash": renderer.template_hash,
            "projection_hash": renderer.projection_hash,
        }
        cache_key = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        query_id = _dimension_key(
            connection, "queries", "query_id", "query_smiles", str(query_smiles)
        )
        group_key = _dimension_key(
            connection, "groups_dim", "group_key", "group_id", str(group_id)
        )
        molecule_key = _dimension_key(
            connection, "molecules", "molecule_key", "molecule_chembl_id", str(molecule_id)
        )
        inserted = connection.execute(
            "INSERT OR IGNORE INTO prompt_tasks(cache_key, prompt) VALUES (?, ?)",
            (cache_key, prompt),
        ).rowcount
        duplicates += int(not inserted)
        prompt_key = int(
            connection.execute(
                "SELECT prompt_key FROM prompt_tasks WHERE cache_key=?", (cache_key,)
            ).fetchone()[0]
        )
        connection.execute(
            "INSERT OR IGNORE INTO prompt_assignments VALUES (?, ?, ?, ?, ?)",
            (query_id, group_key, molecule_key, int(record_key), prompt_key),
        )
        assignments += 1
        if assignments % 100_000 == 0:
            connection.commit()
            print(f"rendered {assignments} score assignments", flush=True)
    connection.commit()
    prompts = int(connection.execute("SELECT COUNT(*) FROM prompt_tasks").fetchone()[0])
    return found, assignments, duplicates


def prepare(task_id: str = "bbb_martins", pool_size: int = 75) -> dict[str, Any]:
    config = TASK_CONFIGS[task_id]
    levels_in_cache = tuple(config["levels"])
    paths = _paths(task_id)
    required = [
        paths[key]
        for key in (
            "records", "mapping", "mapping_manifest", "queries",
        )
    ] + [
        ASSET_ROOT / "prompt.jinja",
        ASSET_ROOT / "prompt_projection.json",
        paths["extension"],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing progressive cache input(s): " + ", ".join(missing))
    output_root = paths["cache"].parent
    output_root.mkdir(parents=True, exist_ok=True)
    if paths["cache"].exists() or paths["version"].exists():
        raise FileExistsError(f"Progressive cache already exists: {output_root}")

    renderer = ProgressiveV191PromptRenderer(task_id)
    level_by_record, level_counts = _level_by_record(
        paths["mapping"], levels_in_cache
    )
    mapping = pq.read_table(
        paths["mapping"],
        columns=["canonical_record_id", "score_cache_candidate_eligible"],
    ).to_pylist()
    eligible = {
        str(row["canonical_record_id"])
        for row in mapping
        if bool(row["score_cache_candidate_eligible"])
    }
    stage3 = pq.read_table(
        paths["records"], columns=["canonical_record_id", "canonical_smiles"]
    ).to_pylist()
    molecule_index: dict[str, int] = {}
    molecules: list[dict[str, Any]] = []
    fingerprints: list[np.ndarray] = []
    record_parent_smiles: dict[str, str] = {}
    records_by_level: dict[str, dict[int, list[str]]] = {
        level: defaultdict(list) for level in levels_in_cache
    }
    for row in stage3:
        record_id = str(row["canonical_record_id"])
        if record_id not in eligible:
            continue
        level = level_by_record[record_id]
        identity = normalize_molecule_identity(str(row.get("canonical_smiles") or ""))
        if identity.status != "ok" or not identity.parent_smiles:
            raise ValueError(f"cannot normalize Stage 3 parent for record {record_id}")
        molecule_id = identity.parent_inchi_key or identity.parent_smiles
        if molecule_id not in molecule_index:
            index = len(molecules)
            _, _, fingerprint = standardize_smiles_and_fp(identity.parent_smiles)
            if fingerprint is None:
                raise ValueError(f"cannot fingerprint Stage 3 parent for record {record_id}")
            molecule_index[molecule_id] = index
            molecules.append(
                {
                    "molecule_id": molecule_id,
                    "canonical_smiles": identity.parent_smiles,
                    "molecule_identity": identity.to_dict(),
                }
            )
            fingerprints.append(
                np.frombuffer(
                    DataStructs.BitVectToBinaryText(fingerprint), dtype=np.uint8
                )
            )
        index = molecule_index[molecule_id]
        records_by_level[level][index].append(record_id)
        record_parent_smiles[record_id] = identity.parent_smiles
    if set(record_parent_smiles) != eligible:
        missing = next(iter(eligible - set(record_parent_smiles)), None)
        raise ValueError(f"Stage 3 mapping coverage mismatch; first missing={missing}")
    packed = np.stack(fingerprints)
    connection = _open_build_db(paths["cache"])
    try:
        n_queries, candidate_counts = _prepare_candidates(
            connection,
            paths,
            records_by_level,
            molecules,
            packed,
            pool_size,
            levels_in_cache,
        )
        n_records, n_assignments, n_duplicates = _hydrate_and_render(
            connection,
            paths,
            renderer,
            level_by_record,
            record_parent_smiles,
            task_id,
        )
        n_prompts = int(
            connection.execute("SELECT COUNT(*) FROM prompt_tasks").fetchone()[0]
        )
        version = {
            "schema_version": COMPACT_CACHE_SCHEMA_VERSION,
            "status": "prepared",
            "profile": config["cache_profile"],
            "task_id": task_id,
            "model": model_profile(task_id, "indirect")["model"],
            "model_revision": model_profile(task_id, "indirect")["revision"],
            "scoring_contract_version": SCORING_CONTRACT_VERSION,
            "backbone_dtype": BACKBONE_DTYPE,
            "logit_extraction_dtype": LOGIT_EXTRACTION_DTYPE,
            "template_hash": renderer.template_hash,
            "training_artifact_template_hash": TRAINING_PROMPT_SHA256,
            "training_template_difference": (
                "one terminal newline; rendered prompts are identical after strip"
            ),
            "projection_hash": renderer.projection_hash,
            "template_profile": "v19_1_retrieval_context_copy",
            "candidate_contract": "current_stage3_level_then_morgan_top75_scaffold_disjoint.v1",
            "record_scope": "normalized_v7_stage3_with_current_vote_pure_level_mapping.v1",
            "revised_level_contract": config["level_contract"],
            "levels_in_cache": list(levels_in_cache),
            "l1_cache": str(
                CACHE_ROOT
                / f"v9_direct_gold_morgan100/{task_id}/scaffold/valid/rankings.parquet"
            ),
            "l1_morgan_width": pool_size,
            "assay_transfer_initial_morgan_filter": pool_size,
            "min_similarity": 0.0,
            "neighbor_identity_policy": "scaffold_disjoint",
            "global_source_exclusion_policy": "L1 records only",
            "unresolved_endpoint_policy": "exclude unresolved-endpoint records",
            "unresolved_parent_identity_policy": (
                "exclude records that cannot be Morgan-ranked or scaffold-filtered"
            ),
            "other_out_of_domain_policy": "retain and flag in the level mapping",
            "fresh_stage3_parent_molecules": len(molecules),
            "n_queries": n_queries,
            "n_unique_query_smiles": int(
                connection.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
            ),
            "n_catalog_records": n_records,
            "n_score_assignments": n_assignments,
            "n_prompt_scores": n_prompts,
            "n_duplicate_prompt_tasks_collapsed": n_duplicates,
            "level_record_counts": {
                level: {
                    "total": sum(
                        count for (current, _, _), count in level_counts.items() if current == level
                    ),
                    "numeric": sum(
                        count
                        for (current, _, numeric), count in level_counts.items()
                        if current == level and numeric
                    ),
                }
                for level in ("L1", *levels_in_cache)
            },
            "level_candidate_counts": {
                level: {
                    name: int(candidate_counts[(level, name)])
                    for name in (
                        "candidate_molecules",
                        "record_assignments",
                        "identity_excluded",
                    )
                }
                for level in levels_in_cache
            },
            "inputs": {str(path): file_sha256(path) for path in required},
            "cache": str(paths["cache"]),
        }
        metadata_keys = (
            "schema_version", "status", "task_id", "profile", "model", "model_revision",
            "scoring_contract_version", "backbone_dtype", "logit_extraction_dtype",
            "template_hash", "projection_hash", "candidate_contract", "record_scope",
        )
        connection.executemany(
            "INSERT INTO cache_metadata(key, value) VALUES (?, ?)",
            ((key, json.dumps(version[key], sort_keys=True)) for key in metadata_keys),
        )
        connection.execute("DROP TABLE refs")
        connection.commit()
    finally:
        connection.close()
    _write_json(paths["version"], version)
    return version


def _shard_rows(
    connection: sqlite3.Connection, shard_index: int, num_shards: int
) -> list[tuple[int, str, str]]:
    return connection.execute(
        """
        SELECT prompt_key, cache_key, prompt FROM prompt_tasks
        WHERE (prompt_key - 1) % ? = ? ORDER BY prompt_key
        """,
        (num_shards, shard_index),
    ).fetchall()


def score(
    *,
    task_id: str = "bbb_martins",
    shard_index: int,
    num_shards: int,
    device: int,
    batch_size: int = 128,
) -> dict[str, Any]:
    paths = _paths(task_id)
    version = json.loads(paths["version"].read_text())
    if version.get("status") != "prepared":
        raise ValueError("Progressive cache is not prepared")
    connection = sqlite3.connect(f"file:{paths['cache'].resolve()}?mode=ro", uri=True)
    rows = _shard_rows(connection, shard_index, num_shards)
    connection.close()
    paths["journals"].mkdir(exist_ok=True)
    journal = paths["journals"] / f"scores-{shard_index:02d}-of-{num_shards:02d}.jsonl"
    completed = _read_jsonl(journal) if journal.exists() else []
    if [row["cache_key"] for row in completed] != [row[1] for row in rows[: len(completed)]]:
        raise ValueError(f"Score journal is not an exact shard prefix: {journal}")
    profile = model_profile(task_id, "indirect")
    renderer = ProgressiveV191PromptRenderer(task_id)
    snapshot = resolve_model_snapshot(
        str(profile["model"]), str(profile["revision"]), local_files_only=True
    )
    model, tokenizer = load_model(snapshot, device=device)
    with journal.open("a", encoding="utf-8") as handle:
        for offset in range(len(completed), len(rows), batch_size):
            batch_rows = rows[offset : offset + batch_size]
            tasks = [
                PromptTask(
                    cache_key=cache_key,
                    prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
                    prompt=prompt,
                    task_id=task_id,
                    query_smiles="",
                    group_id="",
                    molecule_id="",
                    record_id="",
                    model=str(profile["model"]),
                    model_revision=str(profile["revision"]),
                    scoring_contract_version=SCORING_CONTRACT_VERSION,
                    template_hash=renderer.template_hash,
                    projection_hash=renderer.projection_hash,
                )
                for _, cache_key, prompt in batch_rows
            ]
            for result in score_prompt_batch(model, tokenizer, tasks, device=device):
                handle.write(
                    json.dumps(
                        {
                            "cache_key": result.cache_key,
                            "transfer_probability": result.transfer_probability,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
            print(
                f"shard {shard_index + 1}/{num_shards}: "
                f"{min(offset + len(tasks), len(rows))}/{len(rows)}",
                flush=True,
            )
    return {"status": "complete", "journal": str(journal), "n_scores": len(rows)}


def finalize(num_shards: int, task_id: str = "bbb_martins") -> dict[str, Any]:
    paths = _paths(task_id)
    version = json.loads(paths["version"].read_text())
    connection = sqlite3.connect(paths["cache"])
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        "CREATE TABLE prompt_scores(prompt_key INTEGER PRIMARY KEY, transfer_probability REAL NOT NULL)"
    )
    try:
        for shard_index in range(num_shards):
            expected = _shard_rows(connection, shard_index, num_shards)
            journal = paths["journals"] / f"scores-{shard_index:02d}-of-{num_shards:02d}.jsonl"
            observed = _read_jsonl(journal) if journal.exists() else []
            if [row["cache_key"] for row in observed] != [row[1] for row in expected]:
                raise ValueError(f"Incomplete or mismatched score journal: {journal}")
            connection.executemany(
                "INSERT INTO prompt_scores VALUES (?, ?)",
                (
                    (prompt_key, float(row["transfer_probability"]))
                    for (prompt_key, _, _), row in zip(expected, observed)
                ),
            )
            connection.commit()
        expected_count = int(
            connection.execute("SELECT COUNT(*) FROM prompt_tasks").fetchone()[0]
        )
        observed_count = int(
            connection.execute("SELECT COUNT(*) FROM prompt_scores").fetchone()[0]
        )
        if expected_count != observed_count:
            raise ValueError(
                f"Prompt score coverage mismatch: expected={expected_count}, found={observed_count}"
            )
        connection.executescript(
            """
            CREATE TABLE scores(
                score_key INTEGER PRIMARY KEY, transfer_probability REAL NOT NULL
            );
            INSERT INTO scores SELECT prompt_key, transfer_probability FROM prompt_scores;
            CREATE TABLE assignments(
                query_id INTEGER NOT NULL,
                group_key INTEGER NOT NULL,
                molecule_key INTEGER NOT NULL,
                record_key INTEGER NOT NULL,
                score_key INTEGER NOT NULL,
                PRIMARY KEY(query_id, group_key, molecule_key, record_key)
            ) WITHOUT ROWID;
            INSERT INTO assignments
                SELECT query_id, group_key, molecule_key, record_key, prompt_key
                FROM prompt_assignments;
            DROP TABLE prompt_assignments;
            DROP TABLE prompt_scores;
            DROP TABLE prompt_tasks;
            """
        )
        connection.execute(
            "UPDATE cache_metadata SET value=? WHERE key='status'", (json.dumps("complete"),)
        )
        connection.commit()
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
    finally:
        connection.close()
    for journal in paths["journals"].glob("*.jsonl"):
        journal.unlink()
    paths["journals"].rmdir()
    connection = sqlite3.connect(paths["cache"])
    connection.execute("VACUUM")
    connection.close()
    version.update(
        status="complete",
        n_cached_scores=observed_count,
        cache_quick_check=quick_check,
        cache_sha256=file_sha256(paths["cache"]),
    )
    _write_json(paths["version"], version)
    return version


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--task", choices=sorted(TASK_CONFIGS), default="bbb_martins")
    prepare_parser.add_argument("--pool-size", type=int, default=75)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--task", choices=sorted(TASK_CONFIGS), default="bbb_martins")
    score_parser.add_argument("--shard-index", type=int, required=True)
    score_parser.add_argument("--num-shards", type=int, required=True)
    score_parser.add_argument("--device", type=int, required=True)
    score_parser.add_argument("--batch-size", type=int, default=128)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--task", choices=sorted(TASK_CONFIGS), default="bbb_martins")
    finalize_parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(task_id=args.task, pool_size=args.pool_size)
    elif args.command == "score":
        result = score(
            task_id=args.task,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        result = finalize(args.num_shards, task_id=args.task)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
