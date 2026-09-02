"""Build the BBB historical progressive L2-L5 transfer-score cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
    V191PromptRenderer,
)
from predict.retrieval.policies import (
    decide_candidate,
    normalize_molecule_identity,
    standardize_smiles_and_fp,
)


PROFILE_NAME = "v19_1_bbb_historical_progressive_top75"
TASK_ID = "bbb_martins"
LEVELS = ("L2", "L3", "L4", "L5")
LEVEL_BY_HISTORICAL_GROUP = {
    "Tier 1.starling_direct_bbb_evidence": "L2",
    "Proxy.central_functional_access": "L2",
    "Mechanism.passive_permeability": "L3",
    "Mechanism.efflux_transport": "L4",
    "Mechanism.influx_transport": "L5",
}
EXTENSION_PATH = ASSET_ROOT / "direct_bbb_extension.json"
EVIDENCE_ROOT = Path(
    "outputs/paper/molecular_evidence_agent_starling_scaffold_conditioned_benchmark/"
    "evidence/bbb_starling_v7"
)
OVERLAY_ROOT = Path(
    "outputs/paper/starling_conditioned_assay_family_curve_v1/source_overlays/"
    "bbb_source_family_purity_v5"
)
QUERY_PATH = Path(
    "data/gold_labels/BBB_Martins/v1/scaffold/valid_molecule_condition_labels.jsonl"
)
OUTPUT_ROOT = CACHE_ROOT / PROFILE_NAME / TASK_ID / "scaffold/valid"
POPCOUNT = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)


def revised_level(retrieval_source_id: str, historical_group: str | None) -> str:
    """Keep gold votes in L1; move historical non-voting L1 into L2."""
    if retrieval_source_id == "direct_vote":
        return "L1"
    try:
        return LEVEL_BY_HISTORICAL_GROUP[str(historical_group)]
    except KeyError as exc:
        raise ValueError(f"Unmapped historical BBB group: {historical_group!r}") from exc


class ProgressiveV191PromptRenderer(V191PromptRenderer):
    """Use the frozen V19.1 template with the historical direct-source fields."""

    def __init__(self) -> None:
        super().__init__(TASK_ID)
        extension = json.loads(EXTENSION_PATH.read_text(encoding="utf-8"))
        if extension.get("base_projection_sha256") != PROJECTION_SHA256:
            raise ValueError("Direct-BBB extension targets another V19.1 projection")
        self.projection["labels"].update(extension["labels"])
        self.projection["tasks"][TASK_ID]["direct_bbb"] = extension["binding"]
        merged = json.dumps(
            self.projection, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        self.projection_hash = hashlib.sha256(merged).hexdigest()


def _paths() -> dict[str, Path]:
    return {
        "records": EVIDENCE_ROOT / "06_records/records.parquet",
        "bridge": EVIDENCE_ROOT / "07_molecule_evidence/molecule_family_records.parquet",
        "molecules": EVIDENCE_ROOT / "08_neighbor_index/molecules.parquet",
        "memberships": EVIDENCE_ROOT / "08_neighbor_index/group_membership.parquet",
        "fingerprints": EVIDENCE_ROOT / "08_neighbor_index/fingerprints.npz",
        "overlay": OVERLAY_ROOT / "records.parquet",
        "overlay_manifest": OVERLAY_ROOT / "manifest.json",
        "queries": QUERY_PATH,
        "cache": OUTPUT_ROOT / "scores.sqlite3",
        "version": OUTPUT_ROOT / "VERSION.json",
        "journals": OUTPUT_ROOT / ".scores",
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _level_by_record(records_path: Path, overlay_path: Path) -> tuple[dict[str, str], Counter]:
    overlay = pq.read_table(
        overlay_path, columns=["canonical_record_id", "group_id"]
    ).to_pydict()
    historical_group = dict(
        zip(map(str, overlay["canonical_record_id"]), overlay["group_id"])
    )
    records = pq.read_table(
        records_path,
        columns=[
            "canonical_record_id",
            "retrieval_source_id",
            "source_id",
            "finite_scalar_value",
        ],
    ).to_pydict()
    levels: dict[str, str] = {}
    counts: Counter = Counter()
    for record_id, retrieval_source, source_id, scalar in zip(
        records["canonical_record_id"],
        records["retrieval_source_id"],
        records["source_id"],
        records["finite_scalar_value"],
    ):
        record_id = str(record_id)
        level = revised_level(str(retrieval_source), historical_group.get(record_id))
        levels[record_id] = level
        counts[(level, str(source_id), scalar is not None)] += 1
    return levels, counts


def _records_by_level_molecule(
    paths: Mapping[str, Path], level_by_record: Mapping[str, str]
) -> dict[str, dict[int, list[str]]]:
    bridge = pq.read_table(
        paths["bridge"], columns=["evidence_id", "canonical_record_id"]
    ).to_pydict()
    evidence_records: dict[str, list[str]] = defaultdict(list)
    for evidence_id, record_id in zip(
        bridge["evidence_id"], bridge["canonical_record_id"]
    ):
        evidence_records[str(evidence_id)].append(str(record_id))

    memberships = pq.read_table(
        paths["memberships"], columns=["molecule_index", "evidence_id"]
    ).to_pydict()
    output: dict[str, dict[int, list[str]]] = {
        level: defaultdict(list) for level in LEVELS
    }
    seen: set[str] = set()
    for molecule_index, evidence_id in zip(
        memberships["molecule_index"], memberships["evidence_id"]
    ):
        for record_id in evidence_records[str(evidence_id)]:
            level = level_by_record[record_id]
            if level == "L1":
                continue
            output[level][int(molecule_index)].append(record_id)
            seen.add(record_id)
    expected = {record_id for record_id, level in level_by_record.items() if level != "L1"}
    if seen != expected:
        missing = next(iter(expected - seen), None)
        raise ValueError(
            f"Stage 07/08 does not cover the revised levels: "
            f"expected={len(expected)}, found={len(seen)}, first_missing={missing}"
        )
    return output


def _runtime_molecule(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "canonical_smiles": str(row.get("canonical_smiles") or ""),
        "molecule_identity": {
            "parent_smiles": row.get("parent_smiles"),
            "parent_inchi_key": row.get("parent_inchi_key"),
            "parent_connectivity_key": row.get("parent_connectivity_key"),
            "component_parent_inchi_keys": row.get("component_parent_inchi_keys") or [],
            "status": row.get("identity_status"),
        },
    }


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
    pool_size: int,
) -> tuple[int, Counter]:
    molecule_rows = pq.read_table(paths["molecules"]).to_pylist()
    molecules = [_runtime_molecule(row) for row in molecule_rows]
    molecule_ids = [str(row["molecule_id"]) for row in molecule_rows]
    with np.load(paths["fingerprints"], allow_pickle=False) as payload:
        packed = np.asarray(payload["packed_fingerprints"], dtype=np.uint8)
    bit_counts = POPCOUNT[packed].sum(axis=1, dtype=np.uint16)
    level_sets = {level: set(rows) for level, rows in records_by_level.items()}
    queries = _read_jsonl(paths["queries"])
    counts: Counter = Counter()
    for query_index, query in enumerate(queries):
        query_smiles = str(query["drug"])
        _, _, query_fp = standardize_smiles_and_fp(query_smiles)
        if query_fp is None:
            raise ValueError(f"Invalid query SMILES at row {query_index}")
        query_identity = normalize_molecule_identity(query_smiles)
        similarities = _tanimoto_vector(query_fp, packed, bit_counts)
        ranked = sorted(
            range(len(molecules)),
            key=lambda index: (-float(similarities[index]), molecule_ids[index]),
        )
        selected = {level: 0 for level in LEVELS}
        for molecule_index in ranked:
            relevant = [
                level
                for level in LEVELS
                if selected[level] < pool_size and molecule_index in level_sets[level]
            ]
            if not relevant:
                if all(selected[level] == pool_size for level in LEVELS):
                    break
                continue
            if decide_candidate(
                query_identity, molecules[molecule_index], "parent_disjoint"
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
        if any(selected[level] != pool_size for level in LEVELS):
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
) -> dict[str, Any]:
    source_id = str(row.get("source_id") or "")
    fields = renderer.prompt_fields(source_id)
    source_fields = {field: row.get(field) for field in fields}
    return {
        "record_id": str(row["canonical_record_id"]),
        "task_id": TASK_ID,
        "source_id": source_id,
        "historical_progressive_level": level,
        "canonical_smiles": str(row.get("canonical_smiles") or ""),
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
            payload = _record_payload(renderer, raw, level_by_record[record_id])
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

    profile = model_profile(TASK_ID, "indirect")
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


def prepare(pool_size: int = 75) -> dict[str, Any]:
    paths = _paths()
    required = [
        paths[key]
        for key in (
            "records", "bridge", "molecules", "memberships", "fingerprints",
            "overlay", "overlay_manifest", "queries",
        )
    ] + [ASSET_ROOT / "prompt.jinja", ASSET_ROOT / "prompt_projection.json", EXTENSION_PATH]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing progressive cache input(s): " + ", ".join(missing))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if paths["cache"].exists() or paths["version"].exists():
        raise FileExistsError(f"Progressive cache already exists: {OUTPUT_ROOT}")

    renderer = ProgressiveV191PromptRenderer()
    level_by_record, level_counts = _level_by_record(paths["records"], paths["overlay"])
    records_by_level = _records_by_level_molecule(paths, level_by_record)
    connection = _open_build_db(paths["cache"])
    try:
        n_queries, candidate_counts = _prepare_candidates(
            connection, paths, records_by_level, pool_size
        )
        n_records, n_assignments, n_duplicates = _hydrate_and_render(
            connection, paths, renderer, level_by_record
        )
        n_prompts = int(
            connection.execute("SELECT COUNT(*) FROM prompt_tasks").fetchone()[0]
        )
        version = {
            "schema_version": COMPACT_CACHE_SCHEMA_VERSION,
            "status": "prepared",
            "profile": PROFILE_NAME,
            "task_id": TASK_ID,
            "model": model_profile(TASK_ID, "indirect")["model"],
            "model_revision": model_profile(TASK_ID, "indirect")["revision"],
            "scoring_contract_version": SCORING_CONTRACT_VERSION,
            "backbone_dtype": BACKBONE_DTYPE,
            "logit_extraction_dtype": LOGIT_EXTRACTION_DTYPE,
            "template_hash": renderer.template_hash,
            "projection_hash": renderer.projection_hash,
            "template_profile": "v19_1_retrieval_context_copy",
            "candidate_contract": "revised_historical_level_then_morgan_top75_parent_disjoint.v1",
            "record_scope": "normalized_v7_stage06_with_historical_v5_row_family_overlay",
            "revised_level_contract": {
                "L1": "conditioned benchmark direct_vote rows; existing V9 cache",
                "L2": "historical L2 plus non-voting historical L1",
                "L3": "historical passive permeability",
                "L4": "historical efflux transport",
                "L5": "historical influx transport",
            },
            "levels_in_cache": list(LEVELS),
            "l1_cache": str(
                CACHE_ROOT
                / "v9_direct_gold_morgan100/bbb_martins/scaffold/valid/rankings.parquet"
            ),
            "l1_morgan_width": pool_size,
            "assay_transfer_initial_morgan_filter": pool_size,
            "min_similarity": 0.0,
            "neighbor_identity_policy": "parent_disjoint",
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
                for level in ("L1", *LEVELS)
            },
            "level_candidate_counts": {
                level: {
                    name: int(candidate_counts[(level, name)])
                    for name in ("candidate_molecules", "record_assignments", "identity_excluded")
                }
                for level in LEVELS
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
    *, shard_index: int, num_shards: int, device: int, batch_size: int = 128
) -> dict[str, Any]:
    paths = _paths()
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
    profile = model_profile(TASK_ID, "indirect")
    renderer = ProgressiveV191PromptRenderer()
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
                    task_id=TASK_ID,
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


def finalize(num_shards: int) -> dict[str, Any]:
    paths = _paths()
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
    prepare_parser.add_argument("--pool-size", type=int, default=75)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--shard-index", type=int, required=True)
    score_parser.add_argument("--num-shards", type=int, required=True)
    score_parser.add_argument("--device", type=int, required=True)
    score_parser.add_argument("--batch-size", type=int, default=128)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(pool_size=args.pool_size)
    elif args.command == "score":
        result = score(
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        result = finalize(args.num_shards)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
