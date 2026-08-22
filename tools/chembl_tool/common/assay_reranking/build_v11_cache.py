#!/usr/bin/env python3
"""Prepare, score, and verify task-specific v11 assay-transfer caches."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import queue
import sqlite3
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rdkit import DataStructs

from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.retrieval_policy import decide_candidate
from tools.chembl_tool.common.task_workflows.evidence_library import (
    standardize_smiles_and_fp,
)
from tools.chembl_tool.common.assay_reranking.v11 import (
    ASSET_ROOT,
    BACKBONE_DTYPE,
    CACHE_SCHEMA_VERSION,
    COMPACT_CACHE_SCHEMA_VERSION,
    CANDIDATE_SCHEMA_VERSION,
    CATALOG_SCHEMA_VERSION,
    DEMAND_SCHEMA_VERSION,
    LOGIT_EXTRACTION_DTYPE,
    PROFILE_NAME,
    PROJECTION_PATH,
    QUERY_CONTEXT_POLICY,
    SCORING_CONTRACT_VERSION,
    TEMPLATE_PROFILE,
    PromptScore,
    PromptTask,
    V11PromptRenderer,
    V11ScoreCache,
    build_prompt_task,
    file_sha256,
    model_profile,
    prompt_task_from_dict,
    prompt_task_to_dict,
    require_immutable_revision,
    verify_vendored_assets,
)


SUPPORTED_TASKS = ("bioavailability_ma", "skin_reaction", "bbb_martins")
DEFAULT_TASKS = ("bioavailability_ma", "skin_reaction")
REQUIRED_TRANSFORMERS_VERSION = "4.57.6"
RECORD_SCOPES = ("all", "labelable_direct", "numeric_direct")
ALL_RECORD_SCOPE = "all_stage07_retrieval_eligible_records"
POPCOUNT = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)


def _log(message: str) -> None:
    print(f"[v11_assay_transfer_cache] {message}", file=sys.stderr, flush=True)


def _task_paths(args: argparse.Namespace, task_id: str) -> dict[str, Path]:
    split = args.benchmark_split
    subset = args.evaluation_subset
    profile = model_profile(task_id)
    artifact_root = args.task_evidence_views.get(task_id)
    query_path = args.task_query_jsonls.get(task_id)
    target = args.task_cache_dirs.get(task_id)
    if artifact_root is None:
        artifact_root = (
            Path("outputs/chembl_tool/tasks")
            / task_id
            / "evidence_library/starling_normalized_v7"
        )
        records = artifact_root / "06_remove_heldout_overlap" / split / "records.parquet"
        evidence = artifact_root / "07_molecule_evidence" / split
        index = artifact_root / "08_neighbor_index" / split
    else:
        records = artifact_root / "06_records/records.parquet"
        evidence = artifact_root / "07_molecule_evidence"
        index = artifact_root / "08_neighbor_index"
    if query_path is None:
        query_path = (
            Path("data/processed_starling")
            / str(profile["dataset_name"])
            / split
            / f"{subset}.jsonl"
        )
    if target is None:
        target = (
            Path(args.output_root)
            / task_id
            / "evidence_library/assay_transfer_rerank"
            / PROFILE_NAME
            / split
            / subset
        )
    return {
        "artifact_root": artifact_root,
        "records": records,
        "evidence": evidence,
        "bridge": evidence / "molecule_family_records.parquet",
        "index": index,
        "queries": query_path,
        "output": target,
        "cache": target / "scores.sqlite3",
        "version": target / "VERSION.json",
    }


def _required_inputs(paths: Mapping[str, Path]) -> list[Path]:
    inputs = [
        paths["records"],
        paths["bridge"],
        paths["evidence"] / "manifest.json",
        paths["index"] / "manifest.json",
        paths["index"] / "molecules.parquet",
        paths["index"] / "group_membership.parquet",
        paths["index"] / "fingerprints.npz",
        paths["queries"],
    ]
    root_manifest = paths["artifact_root"] / "manifest.json"
    if root_manifest.is_file():
        inputs.append(root_manifest)
    return inputs


def _check_inputs(paths: Mapping[str, Path]) -> None:
    missing = [str(path) for path in _required_inputs(paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing v11 cache input(s): " + ", ".join(missing))


def _input_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {str(path): file_sha256(path) for path in _required_inputs(paths)}


def _check_source_stability(paths: Mapping[str, Path]) -> None:
    transactions = sorted(paths["artifact_root"].glob(".downstream-build-*"))
    if transactions:
        raise RuntimeError(
            "Normalized-v7 source has an active downstream transaction: "
            + ", ".join(str(path) for path in transactions)
        )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _experiment_config(task_id: str) -> Any:
    module = importlib.import_module(
        f"tools.chembl_tool.tasks.{task_id}.experiment_config"
    )
    return module.get_source_config("starling")


def _runtime_molecule(row: Mapping[str, Any]) -> dict[str, Any]:
    def optional(value: Any) -> Any:
        return None if isinstance(value, float) and np.isnan(value) else value

    components = optional(row.get("component_parent_inchi_keys")) or []
    if isinstance(components, np.ndarray):
        components = components.tolist()
    return {
        "molecule_chembl_id": str(row.get("molecule_id") or ""),
        "canonical_smiles": str(row.get("canonical_smiles") or ""),
        "standard_inchi_key": str(row.get("standard_inchi_key") or ""),
        "molecule_identity": {
            "parent_smiles": optional(row.get("parent_smiles")),
            "parent_inchi_key": optional(row.get("parent_inchi_key")),
            "parent_connectivity_key": optional(row.get("parent_connectivity_key")),
            "component_parent_inchi_keys": list(components),
            "status": optional(row.get("identity_status")),
        },
    }


def _load_lightweight_index(index_dir: Path) -> dict[str, Any]:
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    molecules_rows = pq.read_table(index_dir / "molecules.parquet").to_pylist()
    molecules = [_runtime_molecule(row) for row in molecules_rows]
    memberships = pq.read_table(index_dir / "group_membership.parquet").to_pylist()
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    evidence_by_index_group: dict[tuple[int, str], str] = {}
    for row in memberships:
        group = str(row["group_id"])
        index = int(row["molecule_index"])
        group_to_indices[group].append(index)
        evidence_by_index_group[(index, group)] = str(row["evidence_id"])
    with np.load(index_dir / "fingerprints.npz", allow_pickle=False) as payload:
        packed = np.asarray(payload["packed_fingerprints"], dtype=np.uint8)
        size = int(payload["fingerprint_size"][0])
        bitorder = str(payload["bitorder"][0])
    if packed.shape != (len(molecules), 256) or size != 2048 or bitorder != "little":
        raise ValueError("Unexpected compact Morgan fingerprint contract")
    return {
        "manifest": manifest,
        "molecules": molecules,
        "packed": packed,
        "bit_counts": POPCOUNT[packed].sum(axis=1, dtype=np.uint16),
        "group_to_indices": {
            group: sorted(indices) for group, indices in group_to_indices.items()
        },
        "evidence_by_index_group": evidence_by_index_group,
    }


def _tanimoto_vector(query_fp: Any, index: Mapping[str, Any]) -> np.ndarray:
    query = np.frombuffer(DataStructs.BitVectToBinaryText(query_fp), dtype=np.uint8)
    if query.shape != (256,):
        raise ValueError("Query Morgan fingerprint is not 2048 bits")
    packed = index["packed"]
    intersection = POPCOUNT[np.bitwise_and(packed, query)].sum(axis=1, dtype=np.uint16)
    query_count = int(POPCOUNT[query].sum(dtype=np.uint16))
    union = index["bit_counts"].astype(np.int32) + query_count - intersection.astype(np.int32)
    return np.divide(
        intersection,
        union,
        out=np.zeros(len(packed), dtype=np.float64),
        where=union != 0,
    )


def _load_bridge(path: Path) -> dict[str, tuple[str, ...]]:
    rows = pq.read_table(
        path,
        columns=["evidence_id", "canonical_record_id", "record_order"],
    ).to_pylist()
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["evidence_id"])].append(
            (int(row.get("record_order") or 0), str(row["canonical_record_id"]))
        )
    return {
        evidence_id: tuple(record_id for _, record_id in sorted(values))
        for evidence_id, values in grouped.items()
    }


def _candidate_rows(
    *,
    task_id: str,
    queries: list[dict[str, Any]],
    index: Mapping[str, Any],
    bridge: Mapping[str, tuple[str, ...]],
    initial_pool: int,
    min_similarity: float,
    identity_policy: str,
) -> Iterator[dict[str, Any]]:
    config = _experiment_config(task_id)
    available_groups = set(index["group_to_indices"])
    specs = config.mechanism_groups
    for query_index, query in enumerate(queries):
        query_smiles = str(query.get("drug") or "")
        if not query_smiles:
            raise ValueError(f"Query {query_index} has no drug SMILES")
        canonical, _, query_fp = standardize_smiles_and_fp(query_smiles)
        if query_fp is None:
            raise ValueError(f"Query {query_index} has invalid SMILES")
        query_identity = normalize_molecule_identity(query_smiles)
        similarities = _tanimoto_vector(query_fp, index)
        for spec in specs:
            source_groups = spec.resolve(available_groups)
            candidate_indices = sorted(
                {
                    molecule_index
                    for source_group in source_groups
                    for molecule_index in index["group_to_indices"].get(source_group, [])
                }
            )
            ranked = sorted(
                (
                    (float(similarities[molecule_index]), molecule_index)
                    for molecule_index in candidate_indices
                    if float(similarities[molecule_index]) >= min_similarity
                ),
                key=lambda item: (
                    -item[0],
                    index["molecules"][item[1]]["molecule_chembl_id"],
                ),
            )
            candidates = []
            excluded = 0
            examined = 0
            for structural_rank, (similarity, molecule_index) in enumerate(ranked, start=1):
                examined += 1
                molecule = index["molecules"][molecule_index]
                decision = decide_candidate(query_identity, molecule, identity_policy)
                if decision.excluded:
                    excluded += 1
                    continue
                matched_groups = [
                    group
                    for group in source_groups
                    if (molecule_index, group) in index["evidence_by_index_group"]
                ]
                record_ids: list[str] = []
                for group in matched_groups:
                    evidence_id = index["evidence_by_index_group"][(molecule_index, group)]
                    try:
                        record_ids.extend(bridge[evidence_id])
                    except KeyError as exc:
                        raise ValueError(
                            f"Stage 08 membership references missing Stage 07 evidence {evidence_id}"
                        ) from exc
                unique_ids = list(dict.fromkeys(record_ids))
                if not unique_ids:
                    raise ValueError(
                        f"Stage 08 candidate has no Stage 07 records: "
                        f"{molecule['molecule_chembl_id']}/{spec.group_id}"
                    )
                candidates.append(
                    {
                        "molecule_id": molecule["molecule_chembl_id"],
                        "canonical_smiles": molecule["canonical_smiles"],
                        "structural_rank": structural_rank,
                        "similarity": round(similarity, 8),
                        "source_group_ids": list(matched_groups),
                        "record_ids": unique_ids,
                    }
                )
                if len(candidates) == initial_pool:
                    break
            yield {
                "record_type": "candidate_group",
                "query_index": query_index,
                "query_smiles": query_smiles,
                "query_canonical_smiles": canonical,
                "group_id": spec.group_id,
                "source_group_ids": list(source_groups),
                "morgan_candidates_examined": examined,
                "n_identity_excluded": excluded,
                "candidates": candidates,
            }
        if (query_index + 1) % 25 == 0:
            _log(f"{task_id}: prepared Morgan candidates for {query_index + 1}/{len(queries)} queries")


def _open_prepare_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=OFF")
    connection.executescript(
        """
        CREATE TABLE refs (
            record_id TEXT NOT NULL,
            query_smiles TEXT NOT NULL,
            group_id TEXT NOT NULL,
            molecule_id TEXT NOT NULL,
            PRIMARY KEY(record_id, query_smiles, group_id, molecule_id)
        ) WITHOUT ROWID;
        CREATE TABLE records (
            record_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )
    return connection


def _publish_jsonl(header: Mapping[str, Any], rows_path: Path, output: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        target.write(json.dumps(dict(header), ensure_ascii=False, sort_keys=True) + "\n")
        with rows_path.open("r", encoding="utf-8") as source:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    temporary.replace(output)


def _source_projection(task_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
    policy = importlib.import_module(
        f"tools.chembl_tool.tasks.{task_id}.starling_policy"
    ).POLICY
    profile = policy.compact_profile_for_contract("starling_record_contract.v7")
    return profile.llm_source_projection(record)


def _eligible_record_ids(task_id: str, path: Path, scope: str) -> set[str] | None:
    if scope == "all":
        return None

    if task_id == "bbb_martins":
        from tools.chembl_tool.tasks.bbb_martins.experimental_meaningful_cns_access_benchmark import (
            label_record,
        )
    elif task_id == "skin_reaction":
        from tools.chembl_tool.tasks.skin_reaction.starling_benchmark import label_record
    else:
        from tools.chembl_tool.common.starling.benchmark_dataset import (
            has_reported_text,
            parse_numeric_interval,
        )
        from tools.chembl_tool.tasks.bioavailability_ma.canonical_source import (
            DIRECT_REPORT_TYPES,
        )
        from tools.chembl_tool.tasks.bioavailability_ma.starling_benchmark import (
            is_human_context,
            label_bioavailability_value,
        )

    eligible = set()
    direct_sources = {
        "bbb_martins": {"direct_bbb"},
        "skin_reaction": {"direct_skin_reaction"},
        "bioavailability_ma": {"hf_bioavailability", "oral_exposure"},
    }[task_id]
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=16_384):
        for record in batch.to_pylist():
            source_id = str(record.get("source_id") or "")
            if source_id not in direct_sources:
                continue
            fields = _source_projection(task_id, record)["source_fields"]
            if task_id == "bbb_martins":
                source_index = fields.get("source_index")
                label, _ = label_record(
                    {
                        **fields,
                        "quant_metric": fields.get("endpoint_name"),
                        "quant_value": fields.get("measurement_text"),
                        "quant_units": fields.get("unit_text"),
                    },
                    source_index=(
                        int(source_index) if source_index is not None else None
                    ),
                )
            elif task_id == "skin_reaction":
                label, _ = label_record(
                    {
                        "reaction_type": fields.get("endpoint_name"),
                        "outcome_label": fields.get("outcome_label"),
                    }
                )
            else:
                report_type = str(
                    fields.get("bioavailability_report_type") or ""
                ).strip().lower()
                if source_id == "hf_bioavailability" and report_type not in DIRECT_REPORT_TYPES:
                    continue
                population = fields.get("species_or_population") or fields.get(
                    "study_context"
                )
                if not is_human_context(population) or has_reported_text(
                    fields.get("qualifying_conditions")
                ):
                    continue
                measurement = str(fields.get("measurement_text") or "").strip()
                unit = str(fields.get("unit_text") or "").strip()
                value_text = f"{measurement} {unit}".strip()
                label, _ = label_bioavailability_value(value_text)
            if label is None:
                continue
            if scope == "numeric_direct":
                interval = parse_numeric_interval(
                    value_text, fraction_to_percent=True
                )
                if not (
                    interval is not None
                    and interval.method in {"reported_point", "reported_mean_plus_minus"}
                    and math.isfinite(interval.lower)
                    and math.isfinite(interval.upper)
                ):
                    continue
            eligible.add(str(record["canonical_record_id"]))
    return eligible


def _catalog_record(
    task_id: str,
    record: Mapping[str, Any],
    renderer: V11PromptRenderer,
    training_kinds: set[str],
) -> dict[str, Any]:
    source_id = str(record.get("source_id") or "")
    config = renderer.projection["tasks"][task_id]["sources"].get(source_id)
    if config is None:
        raise ValueError(f"Stage 07 retained source absent from v11 projection: {task_id}/{source_id}")
    prompt_fields = [*config["both"], *config.get("retrieval_only", [])]
    if record.get("processed_gold_voting_record_key"):
        source_contract = {
            "source_id": source_id,
            "source_name": record.get("source_name"),
            "record_contract_version": "processed_starling_gold.v1",
        }
        source_fields = {field: record.get(field) for field in prompt_fields}
    else:
        projection = _source_projection(task_id, record)
        source_contract = {
            key: value for key, value in projection.items() if key != "source_fields"
        }
        source_fields = projection["source_fields"]
    return {
        "record_type": "assay_record",
        "record_id": str(record["canonical_record_id"]),
        "task_id": task_id,
        "source_id": source_id,
        "group_id": str(record.get("group_id") or ""),
        "canonical_smiles": str(record.get("canonical_smiles") or ""),
        "measurement_kind": str(record.get("measurement_kind") or ""),
        "training_measurement_kind_supported": (
            str(record.get("measurement_kind") or "") in training_kinds
        ),
        "source_contract": source_contract,
        "source_fields": source_fields,
        "processed_gold_voting_record_key": record.get(
            "processed_gold_voting_record_key"
        ),
        "processed_gold_lineage": record.get("processed_gold_lineage"),
        "processed_gold_split": record.get("processed_gold_split"),
        "record_vote": record.get("record_vote"),
        "molecule_Y": record.get("molecule_Y"),
        **{field: record.get(field) for field in prompt_fields},
    }


def _hydrate_catalog(
    *,
    task_id: str,
    records_path: Path,
    connection: sqlite3.Connection,
    catalog_rows_path: Path,
    renderer: V11PromptRenderer,
    training_kinds: set[str],
) -> int:
    wanted_count = int(connection.execute("SELECT COUNT(DISTINCT record_id) FROM refs").fetchone()[0])
    wanted = {
        str(row[0]) for row in connection.execute("SELECT DISTINCT record_id FROM refs")
    }
    parquet = pq.ParquetFile(records_path)
    found = 0
    with catalog_rows_path.open("w", encoding="utf-8") as output:
        for batch_number, batch in enumerate(parquet.iter_batches(batch_size=16_384), start=1):
            ids = batch.column(batch.schema.get_field_index("canonical_record_id"))
            mask = pa.array(
                [str(record_id) in wanted for record_id in ids.to_pylist()],
                type=pa.bool_(),
            )
            if not pc.any(mask).as_py():
                continue
            selected = batch.filter(mask)
            rows_to_insert = []
            for raw in selected.to_pylist():
                payload = _catalog_record(task_id, raw, renderer, training_kinds)
                encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
                output.write(encoded + "\n")
                rows_to_insert.append((payload["record_id"], encoded))
            if rows_to_insert:
                connection.executemany(
                    "INSERT OR REPLACE INTO records(record_id, payload) VALUES (?, ?)",
                    rows_to_insert,
                )
                connection.commit()
                found += len(rows_to_insert)
            if batch_number % 10 == 0:
                _log(f"{task_id}: hydrated {found}/{wanted_count} demanded Stage 07 records")
    if found != wanted_count:
        missing = connection.execute(
            "SELECT record_id FROM refs EXCEPT SELECT record_id FROM records LIMIT 1"
        ).fetchone()
        raise ValueError(
            f"Stage 07/06 record coverage mismatch: expected={wanted_count}, found={found}, "
            f"first_missing={missing[0] if missing else 'unknown'}"
        )
    return found


def _write_demand(
    *,
    task_id: str,
    connection: sqlite3.Connection,
    rows_path: Path,
    renderer: V11PromptRenderer,
    model: str,
    revision: str,
) -> tuple[int, int]:
    query = """
        SELECT refs.query_smiles, refs.group_id, refs.molecule_id, records.payload
        FROM refs JOIN records USING(record_id)
        ORDER BY refs.query_smiles, refs.record_id
    """
    count = duplicate_count = 0
    seen_cache_keys: set[str] = set()
    with rows_path.open("w", encoding="utf-8") as output:
        for query_smiles, group_id, molecule_id, payload_json in connection.execute(query):
            record = json.loads(payload_json)
            task = build_prompt_task(
                renderer,
                record,
                query_smiles=str(query_smiles),
                group_id=str(group_id),
                molecule_id=str(molecule_id),
                model=model,
                model_revision=revision,
            )
            if task.cache_key in seen_cache_keys:
                duplicate_count += 1
                continue
            seen_cache_keys.add(task.cache_key)
            output.write(
                json.dumps(
                    {"record_type": "prompt_task", **prompt_task_to_dict(task)},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            count += 1
            if count % 100_000 == 0:
                _log(f"{task_id}: rendered {count} unique v11 prompt tasks")
    return count, duplicate_count


def _open_compact_build_db(path: Path) -> sqlite3.Connection:
    path.unlink(missing_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
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
        CREATE TABLE prompt_scores(
            prompt_key INTEGER PRIMARY KEY,
            transfer_probability REAL NOT NULL
        );
        """
    )
    return connection


def _dimension_key(
    connection: sqlite3.Connection, table: str, key_column: str, value_column: str, value: str
) -> int:
    connection.execute(
        f"INSERT OR IGNORE INTO {table}({value_column}) VALUES (?)", (value,)
    )
    row = connection.execute(
        f"SELECT {key_column} FROM {table} WHERE {value_column}=?", (value,)
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to resolve {table} dimension")
    return int(row[0])


def _winning_record_payload(renderer: V11PromptRenderer, record: Mapping[str, Any]) -> dict:
    return {
        "record_id": str(record["record_id"]),
        "canonical_endpoint_key": renderer.canonical_endpoint_key(record),
        "canonical_smiles": record.get("canonical_smiles"),
        "measurement_kind": record.get("measurement_kind"),
        "training_measurement_kind_supported": record.get(
            "training_measurement_kind_supported"
        ),
        "source_contract": record.get("source_contract"),
        "source_fields": record.get("source_fields"),
        "processed_gold_voting_record_key": record.get(
            "processed_gold_voting_record_key"
        ),
        "processed_gold_lineage": record.get("processed_gold_lineage"),
        "processed_gold_split": record.get("processed_gold_split"),
        "record_vote": record.get("record_vote"),
        "molecule_Y": record.get("molecule_Y"),
    }


def _materialize_compact_demand(
    *, task_id: str, source: sqlite3.Connection, target: sqlite3.Connection,
    renderer: V11PromptRenderer, model: str, revision: str,
) -> tuple[int, int, int]:
    query = """
        SELECT refs.query_smiles, refs.group_id, refs.molecule_id, records.payload
        FROM refs JOIN records USING(record_id)
        ORDER BY refs.query_smiles, refs.group_id, refs.molecule_id, refs.record_id
    """
    assignments = duplicates = 0
    for query_smiles, group_id, molecule_id, payload_json in source.execute(query):
        record = json.loads(payload_json)
        task = build_prompt_task(
            renderer, record, query_smiles=str(query_smiles), group_id=str(group_id),
            molecule_id=str(molecule_id), model=model, model_revision=revision,
        )
        query_id = _dimension_key(target, "queries", "query_id", "query_smiles", str(query_smiles))
        group_key = _dimension_key(target, "groups_dim", "group_key", "group_id", str(group_id))
        molecule_key = _dimension_key(
            target, "molecules", "molecule_key", "molecule_chembl_id", str(molecule_id)
        )
        record_id = str(record["record_id"])
        payload = json.dumps(
            _winning_record_payload(renderer, record), ensure_ascii=False, sort_keys=True
        )
        target.execute(
            "INSERT OR IGNORE INTO records(external_record_id, payload) VALUES (?, ?)",
            (record_id, payload),
        )
        record_key = int(target.execute(
            "SELECT record_key FROM records WHERE external_record_id=?", (record_id,)
        ).fetchone()[0])
        inserted = target.execute(
            "INSERT OR IGNORE INTO prompt_tasks(cache_key, prompt) VALUES (?, ?)",
            (task.cache_key, task.prompt),
        ).rowcount
        if not inserted:
            duplicates += 1
        prompt_key = int(target.execute(
            "SELECT prompt_key FROM prompt_tasks WHERE cache_key=?", (task.cache_key,)
        ).fetchone()[0])
        target.execute(
            "INSERT OR IGNORE INTO prompt_assignments VALUES (?, ?, ?, ?, ?)",
            (query_id, group_key, molecule_key, record_key, prompt_key),
        )
        assignments += 1
        if assignments % 100_000 == 0:
            target.commit()
            _log(f"{task_id}: materialized {assignments} compact score assignments")
    target.commit()
    prompts = int(target.execute("SELECT COUNT(*) FROM prompt_tasks").fetchone()[0])
    return assignments, prompts, duplicates


def _metadata_rows(version: Mapping[str, Any]) -> list[tuple[str, str]]:
    keys = (
        "schema_version", "status", "task_id", "profile", "model", "model_revision",
        "scoring_contract_version", "backbone_dtype", "logit_extraction_dtype",
        "template_hash", "projection_hash", "candidate_contract", "record_scope",
    )
    return [(key, json.dumps(version[key], sort_keys=True)) for key in keys]


def prepare_task(args: argparse.Namespace, task_id: str) -> dict[str, Any]:
    if args.benchmark_split != "scaffold":
        raise ValueError("V11 with-categorical models reserve scaffold valid+test only")
    source_hashes = verify_vendored_assets()
    profile = dict(model_profile(task_id))
    if args.assay_transfer_model:
        profile["model"] = args.assay_transfer_model
        profile["revision"] = args.assay_transfer_model_revision
    local_model = None
    if args.assay_transfer_local_model_dir:
        model_path = args.assay_transfer_local_model_dir.resolve()
        model_hash = _directory_sha256(model_path)
        local_model = {"path": str(model_path), "sha256": model_hash}
        profile["model"] = f"local:{model_path.name}"
        # Prompt cache keys require a 40-character immutable revision. The
        # complete SHA-256 remains in local_model and is rechecked at scoring.
        profile["revision"] = model_hash[:40]
    paths = _task_paths(args, task_id)
    _check_inputs(paths)
    _check_source_stability(paths)
    paper_direct_contract = None
    if args.record_scope != "all":
        root_manifest_path = paths["artifact_root"] / "manifest.json"
        root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
        heldout_filter = root_manifest.get("heldout_filter") or {}
        if (
            root_manifest.get("heldout_filter_mode") != "all_scaffolds"
            or not heldout_filter.get("zero_parent_overlap")
            or not heldout_filter.get("zero_heldout_scaffold_overlap")
        ):
            raise ValueError(
                "Direct caches require all-source held-out scaffold filtering"
            )
        paper_direct_contract = root_manifest.get("paper_direct_contract")
        if task_id == "bioavailability_ma" and not paper_direct_contract:
            raise ValueError(
                "Bioavailability direct caches require the canonical paper direct contract"
            )
    input_hashes_before = _input_hashes(paths)
    record_scope = ALL_RECORD_SCOPE if args.record_scope == "all" else args.record_scope
    paths["output"].mkdir(parents=True, exist_ok=True)
    if paths["version"].is_file() and paths["cache"].is_file() and not args.force_prepare:
        existing = json.loads(paths["version"].read_text(encoding="utf-8"))
        if (
            existing.get("inputs") == input_hashes_before
            and existing.get("record_scope") == record_scope
        ):
            _log(f"{task_id}: reusing existing {existing.get('status')} compact cache")
            return existing
        raise ValueError(f"{task_id}: existing cache contract differs; use --force-prepare")
    renderer = V11PromptRenderer(task_id)
    queries = _read_jsonl(paths["queries"])
    index = _load_lightweight_index(paths["index"])
    bridge = _load_bridge(paths["bridge"])
    eligible_record_ids = _eligible_record_ids(task_id, paths["records"], args.record_scope)
    if eligible_record_ids is not None:
        config = _experiment_config(task_id)
        available_groups = set(index["group_to_indices"])
        direct_groups = {
            source_group
            for spec in config.direct_groups
            for source_group in spec.resolve(available_groups)
        }
        bridge = {
            evidence_id: selected
            for evidence_id, record_ids in bridge.items()
            if (selected := tuple(
                record_id for record_id in record_ids if record_id in eligible_record_ids
            ))
        }
        retained_evidence = set(bridge)
        index["evidence_by_index_group"] = {
            key: evidence_id
            for key, evidence_id in index["evidence_by_index_group"].items()
            if evidence_id in retained_evidence and key[1] in direct_groups
        }
        groups: dict[str, list[int]] = defaultdict(list)
        for molecule_index, group_id in index["evidence_by_index_group"]:
            groups[group_id].append(molecule_index)
        index["group_to_indices"] = {
            group_id: sorted(set(indices)) for group_id, indices in groups.items()
        }
    work_db = paths["output"] / ".prepare.sqlite3"
    catalog_tmp = paths["output"] / ".catalog_rows.tmp"
    work_db.unlink(missing_ok=True)
    source = _open_prepare_db(work_db)
    target = _open_compact_build_db(paths["cache"])
    n_groups = n_candidates = n_refs = n_identity_excluded = 0
    try:
        for row in _candidate_rows(
            task_id=task_id, queries=queries, index=index, bridge=bridge,
            initial_pool=args.assay_transfer_initial_morgan_filter,
            min_similarity=args.min_similarity, identity_policy=args.neighbor_identity_policy,
        ):
            n_groups += 1
            n_candidates += len(row["candidates"])
            n_identity_excluded += int(row["n_identity_excluded"])
            inserts = [
                (record_id, row["query_smiles"], row["group_id"], candidate["molecule_id"])
                for candidate in row["candidates"] for record_id in candidate["record_ids"]
            ]
            source.executemany(
                "INSERT OR IGNORE INTO refs VALUES (?, ?, ?, ?)", inserts
            )
            n_refs += len(inserts)
            if n_groups % 100 == 0:
                source.commit()
        source.commit()
        n_records = _hydrate_catalog(
            task_id=task_id, records_path=paths["records"], connection=source,
            catalog_rows_path=catalog_tmp, renderer=renderer,
            training_kinds=set(profile["training_measurement_kinds"]),
        )
        n_assignments, n_prompts, n_duplicates = _materialize_compact_demand(
            task_id=task_id, source=source, target=target, renderer=renderer,
            model=str(profile["model"]), revision=str(profile["revision"]),
        )
        input_hashes_after = _input_hashes(paths)
        if input_hashes_after != input_hashes_before:
            raise RuntimeError("Lineage inputs changed during compact cache preparation")
        version = {
            "schema_version": COMPACT_CACHE_SCHEMA_VERSION,
            "status": "prepared", "profile": PROFILE_NAME, "task_id": task_id,
            "model": profile["model"], "model_revision": profile["revision"],
            "local_model": local_model,
            "template_profile": TEMPLATE_PROFILE, "template_hash": renderer.template_hash,
            "projection_hash": renderer.projection_hash, "projection_path": str(PROJECTION_PATH),
            "prompt_source_revision": "ee303afdc972fbcef820b6a39076032917b7346f",
            "prompt_asset_hashes": source_hashes,
            "scoring_contract_version": SCORING_CONTRACT_VERSION,
            "backbone_dtype": BACKBONE_DTYPE,
            "logit_extraction_dtype": LOGIT_EXTRACTION_DTYPE,
            "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
            "benchmark_split": args.benchmark_split,
            "evaluation_subset": args.evaluation_subset,
            "candidate_contract": (
                "tanimoto_identity_exclusion_then_eligible_pool.v1"
                if args.record_scope == "all"
                else "record_scope_then_tanimoto_identity_exclusion_then_eligible_pool.v1"
            ),
            "assay_transfer_initial_morgan_filter": args.assay_transfer_initial_morgan_filter,
            "min_similarity": args.min_similarity,
            "neighbor_identity_policy": args.neighbor_identity_policy,
            "record_scope": record_scope,
            "paper_direct_contract": paper_direct_contract,
            "n_record_scope_records": (
                None if eligible_record_ids is None else len(eligible_record_ids)
            ),
            "n_record_scope_molecules": (
                None
                if eligible_record_ids is None
                else len({key[0] for key in index["evidence_by_index_group"]})
            ),
            "stage04_or_stage05_filter_applied": False,
            "source_stability_check": "pass",
            "n_queries": len(queries), "n_candidate_groups": n_groups,
            "n_candidates": n_candidates, "n_identity_excluded": n_identity_excluded,
            "n_catalog_records": n_records, "n_score_assignments": n_assignments,
            "n_prompt_scores": n_prompts,
            "n_prompt_tasks_before_cache_key_deduplication": n_prompts + n_duplicates,
            "n_duplicate_prompt_tasks_collapsed": n_duplicates,
            "n_cached_scores": 0, "cache": str(paths["cache"]),
            "inputs": input_hashes_before,
        }
        target.executemany(
            "INSERT OR REPLACE INTO cache_metadata(key, value) VALUES (?, ?)",
            _metadata_rows(version),
        )
        target.commit()
        _write_json_atomic(paths["version"], version)
        return version
    finally:
        source.close()
        target.close()
        catalog_tmp.unlink(missing_ok=True)
        work_db.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            Path(str(work_db) + suffix).unlink(missing_ok=True)


def _iter_demand(path: Path) -> Iterator[PromptTask]:
    with path.open(encoding="utf-8") as handle:
        header = json.loads(next(handle))
        if header.get("schema_version") != DEMAND_SCHEMA_VERSION:
            raise ValueError("Unexpected prompt-demand schema")
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                payload.pop("record_type", None)
                yield prompt_task_from_dict(payload)


def _iter_missing_batches(
    demand_path: Path, cache: V11ScoreCache, batch_size: int
) -> Iterator[list[PromptTask]]:
    pending: list[PromptTask] = []
    lookup_chunk: list[PromptTask] = []
    seen_cache_keys: set[str] = set()
    for task in _iter_demand(demand_path):
        if task.cache_key in seen_cache_keys:
            continue
        seen_cache_keys.add(task.cache_key)
        lookup_chunk.append(task)
        if len(lookup_chunk) < 500:
            continue
        cached = cache.lookup(lookup_chunk)
        for item in lookup_chunk:
            if item.cache_key not in cached:
                pending.append(item)
                if len(pending) == batch_size:
                    yield pending
                    pending = []
        lookup_chunk = []
    if lookup_chunk:
        cached = cache.lookup(lookup_chunk)
        for item in lookup_chunk:
            if item.cache_key not in cached:
                pending.append(item)
                if len(pending) == batch_size:
                    yield pending
                    pending = []
    if pending:
        yield pending


def _first_divergent_index(left: list[int], right: list[int]) -> int:
    for index in range(min(len(left), len(right))):
        if left[index] != right[index]:
            return index
    raise ValueError("A/B completions did not diverge in tokenization")


def _selected_fp32_head_logits(
    hidden_states: Any,
    output_head: Any,
    a_tokens: list[int],
    b_tokens: list[int],
    torch: Any,
) -> tuple[Any, Any]:
    """Compute only the A/B output-head logits with FP32 accumulation."""
    if hidden_states.ndim != 2:
        raise ValueError("Expected one final hidden vector per prompt")
    if len(a_tokens) != hidden_states.shape[0] or len(b_tokens) != hidden_states.shape[0]:
        raise ValueError("A/B token count does not match the prompt batch")
    weight = output_head.weight
    if weight.ndim != 2 or weight.shape[1] != hidden_states.shape[1]:
        raise ValueError("Output-head and hidden-state dimensions do not match")
    device = hidden_states.device
    a_index = torch.as_tensor(a_tokens, dtype=torch.long, device=device)
    b_index = torch.as_tensor(b_tokens, dtype=torch.long, device=device)
    hidden_fp32 = hidden_states.to(dtype=torch.float32)
    a_weight = weight.index_select(0, a_index).to(dtype=torch.float32)
    b_weight = weight.index_select(0, b_index).to(dtype=torch.float32)
    logit_a = (hidden_fp32 * a_weight).sum(dim=-1, dtype=torch.float32)
    logit_b = (hidden_fp32 * b_weight).sum(dim=-1, dtype=torch.float32)
    bias = getattr(output_head, "bias", None)
    if bias is not None:
        logit_a = logit_a + bias.index_select(0, a_index).to(dtype=torch.float32)
        logit_b = logit_b + bias.index_select(0, b_index).to(dtype=torch.float32)
    return logit_a, logit_b


def _score_prompt_batch(
    model: Any, tokenizer: Any, tasks: list[PromptTask], device: int, torch: Any
) -> list[PromptScore]:
    prefixes: list[list[int]] = []
    a_tokens: list[int] = []
    b_tokens: list[int] = []
    for task in tasks:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": task.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        a_ids = tokenizer(prompt + "(A)", add_special_tokens=False).input_ids
        b_ids = tokenizer(prompt + "(B)", add_special_tokens=False).input_ids
        prefix = _first_divergent_index(a_ids, b_ids)
        prefixes.append(a_ids[:prefix])
        a_tokens.append(int(a_ids[prefix]))
        b_tokens.append(int(b_ids[prefix]))
    encoded = tokenizer.pad({"input_ids": prefixes}, padding=True, return_tensors="pt")
    encoded = {key: value.to(f"cuda:{device}") for key, value in encoded.items()}
    decoder = getattr(model, "model", None)
    if decoder is None:
        raise ValueError("Causal LM does not expose its decoder as model.model")
    hidden = decoder(**encoded, use_cache=False, return_dict=True).last_hidden_state
    positions = torch.arange(encoded["attention_mask"].shape[1], device=hidden.device)
    last = (encoded["attention_mask"] * positions).max(dim=1).values
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    final_hidden = hidden[rows, last]
    logit_a_tensor, logit_b_tensor = _selected_fp32_head_logits(
        final_hidden,
        model.get_output_embeddings(),
        a_tokens,
        b_tokens,
        torch,
    )
    probabilities = torch.softmax(
        torch.stack((logit_a_tensor, logit_b_tensor), dim=-1), dim=-1
    )[:, 0]
    output = []
    for index, task in enumerate(tasks):
        logit_a = float(logit_a_tensor[index].item())
        logit_b = float(logit_b_tensor[index].item())
        output.append(
            PromptScore(
                cache_key=task.cache_key,
                logp_transfer=logit_a,
                logp_not_transfer=logit_b,
                transfer_probability=float(probabilities[index].item()),
            )
        )
    return output


def _gpu_worker(
    device: int,
    snapshot: str,
    dtype_name: str,
    work_queue: Any,
    result_queue: Any,
) -> None:
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if transformers.__version__ != REQUIRED_TRANSFORMERS_VERSION:
            raise RuntimeError(
                "Pinned InternS1 tokenizer requires transformers "
                f"{REQUIRED_TRANSFORMERS_VERSION}; observed {transformers.__version__}"
            )
        torch.cuda.set_device(device)
        if dtype_name != BACKBONE_DTYPE:
            raise ValueError(f"V11 backbone dtype is frozen to {BACKBONE_DTYPE}")
        dtype = torch.bfloat16
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot, trust_remote_code=True, local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
        ).to(f"cuda:{device}")
        model.eval()
        model.config.use_cache = False
        result_queue.put({"type": "ready", "device": device})
        while True:
            item = work_queue.get()
            if item is None:
                break
            tasks = [prompt_task_from_dict(payload) for payload in item["tasks"]]
            with torch.inference_mode():
                scores = _score_prompt_batch(model, tokenizer, tasks, device, torch)
            result_queue.put(
                {
                    "type": "result",
                    "device": device,
                    "batch_id": item["batch_id"],
                    "scores": [score.__dict__ for score in scores],
                }
            )
    except BaseException as exc:
        result_queue.put(
            {
                "type": "error",
                "device": device,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise


def _parse_devices(value: str) -> list[int]:
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not devices or any(device < 0 for device in devices):
        raise ValueError("--devices must contain nonnegative CUDA device indices")
    return devices


def _resolve_snapshot(model: str, revision: str, local_files_only: bool) -> str:
    from huggingface_hub import HfApi, snapshot_download

    immutable = require_immutable_revision(revision)
    if not local_files_only:
        observed = require_immutable_revision(HfApi().model_info(model, revision=revision).sha)
        if observed != immutable:
            raise ValueError(f"Configured model revision changed: expected={immutable}, observed={observed}")
    return snapshot_download(
        repo_id=model,
        revision=immutable,
        local_files_only=local_files_only,
    )


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Local model directory contains no files: {path}")
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _run_workers(
    *,
    batches: Iterable[list[PromptTask]],
    cache: Any,
    snapshot: str,
    devices: list[int],
    dtype: str,
) -> int:
    context = mp.get_context("spawn")
    work_queue = context.Queue(maxsize=max(2, len(devices) * 2))
    result_queue = context.Queue()
    workers = [
        context.Process(
            target=_gpu_worker,
            args=(device, snapshot, dtype, work_queue, result_queue),
            name=f"v11-assay-transfer-{worker_index}-gpu-{device}",
        )
        for worker_index, device in enumerate(devices)
    ]
    for worker in workers:
        worker.start()
    iterator = enumerate(batches)
    in_flight: dict[int, list[PromptTask]] = {}
    completed = 0
    next_report = 10_000
    try:
        ready = 0
        while ready < len(workers):
            message = result_queue.get(timeout=600)
            if message.get("type") == "error":
                raise RuntimeError(message["traceback"])
            if message.get("type") != "ready":
                raise RuntimeError(f"Unexpected worker startup message: {message}")
            ready += 1

        def submit() -> bool:
            try:
                batch_id, tasks = next(iterator)
            except StopIteration:
                return False
            in_flight[batch_id] = tasks
            work_queue.put(
                {
                    "batch_id": batch_id,
                    "tasks": [prompt_task_to_dict(task) for task in tasks],
                }
            )
            return True

        for _ in range(len(workers) * 2):
            if not submit():
                break
        while in_flight:
            try:
                message = result_queue.get(timeout=120)
            except queue.Empty as exc:
                failed = [worker for worker in workers if worker.exitcode not in (None, 0)]
                if failed:
                    raise RuntimeError(f"V11 scoring worker failed: {failed[0].exitcode}") from exc
                continue
            if message.get("type") == "error":
                raise RuntimeError(message["traceback"])
            batch_id = int(message["batch_id"])
            tasks = in_flight.pop(batch_id)
            scores = [PromptScore(**payload) for payload in message["scores"]]
            cache.write_batch(tasks, scores)
            completed += len(tasks)
            if completed >= next_report:
                _log(f"scored {completed} missing prompts")
                next_report += 10_000
            submit()
        for _ in workers:
            work_queue.put(None)
        for worker in workers:
            worker.join(timeout=60)
        failed = [worker for worker in workers if worker.exitcode not in (None, 0)]
        if failed:
            raise RuntimeError(f"V11 scoring workers exited nonzero: {failed}")
        return completed
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)


class _CompactBuildWriter:
    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path, timeout=60.0)
        self.connection.execute("PRAGMA journal_mode=WAL")

    def write_batch(
        self, tasks: Iterable[PromptTask], scores: Iterable[PromptScore]
    ) -> int:
        tasks_by_key = {task.cache_key: task for task in tasks}
        rows = []
        for score in scores:
            if score.cache_key not in tasks_by_key:
                raise ValueError(f"Score has no matching compact prompt: {score.cache_key}")
            prompt_key = self.connection.execute(
                "SELECT prompt_key FROM prompt_tasks WHERE cache_key=?", (score.cache_key,)
            ).fetchone()
            if prompt_key is None:
                raise ValueError(f"Unknown compact prompt cache key: {score.cache_key}")
            rows.append((int(prompt_key[0]), float(score.transfer_probability)))
        self.connection.executemany(
            "INSERT OR IGNORE INTO prompt_scores(prompt_key, transfer_probability) VALUES (?, ?)",
            rows,
        )
        self.connection.commit()
        return len(rows)

    def close(self) -> None:
        self.connection.close()


def _iter_compact_missing_batches(
    path: Path, *, task_id: str, model: str, revision: str, batch_size: int
) -> Iterator[list[PromptTask]]:
    renderer = V11PromptRenderer(task_id)
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        cursor = connection.execute(
            """
            SELECT p.cache_key, p.prompt
            FROM prompt_tasks p LEFT JOIN prompt_scores s USING(prompt_key)
            WHERE s.prompt_key IS NULL ORDER BY p.prompt_key
            """
        )
        batch = []
        for cache_key, prompt in cursor:
            batch.append(
                PromptTask(
                    cache_key=str(cache_key), prompt_hash="", prompt=str(prompt),
                    task_id=task_id, query_smiles="", group_id="", molecule_id="",
                    record_id="", model=model, model_revision=revision,
                    scoring_contract_version=SCORING_CONTRACT_VERSION,
                    template_hash=renderer.template_hash,
                    projection_hash=renderer.projection_hash,
                )
            )
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    finally:
        connection.close()


def _finalize_compact_cache(paths: Mapping[str, Path], version: dict[str, Any]) -> None:
    connection = sqlite3.connect(paths["cache"])
    try:
        expected = int(version["n_prompt_scores"])
        scored = int(connection.execute("SELECT COUNT(*) FROM prompt_scores").fetchone()[0])
        if scored != expected:
            raise ValueError(f"Cannot finalize incomplete compact cache: {scored}/{expected}")
        connection.executescript(
            """
            CREATE TABLE scores(
                score_key INTEGER PRIMARY KEY,
                transfer_probability REAL NOT NULL
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
            CREATE INDEX assignments_lookup
                ON assignments(query_id, group_key, molecule_key);
            DROP TABLE prompt_assignments;
            DROP TABLE prompt_scores;
            DROP TABLE prompt_tasks;
            """
        )
        version["status"] = "complete"
        version["n_cached_scores"] = scored
        version["rendered_prompts_retained"] = False
        connection.executemany(
            "INSERT OR REPLACE INTO cache_metadata(key, value) VALUES (?, ?)",
            _metadata_rows(version),
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("VACUUM")
    finally:
        connection.close()


def score_task(args: argparse.Namespace, task_id: str) -> dict[str, Any]:
    paths = _task_paths(args, task_id)
    version = json.loads(paths["version"].read_text(encoding="utf-8"))
    if version.get("status") == "complete":
        return verify_task(args, task_id)
    expected_precision = {
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "backbone_dtype": BACKBONE_DTYPE,
        "logit_extraction_dtype": LOGIT_EXTRACTION_DTYPE,
    }
    mismatches = {
        key: {"expected": expected, "observed": version.get(key)}
        for key, expected in expected_precision.items()
        if version.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "Prepared v11 cache has incompatible precision provenance: "
            + json.dumps(mismatches, sort_keys=True)
        )
    local_model = version.get("local_model")
    if local_model:
        snapshot = str(Path(local_model["path"]).resolve())
        if _directory_sha256(Path(snapshot)) != local_model["sha256"]:
            raise ValueError("Local assay-transfer model content changed after preparation")
    else:
        snapshot = _resolve_snapshot(
            str(version["model"]), str(version["model_revision"]), args.local_files_only
        )
    cache = _CompactBuildWriter(paths["cache"])
    try:
        scored = _run_workers(
            batches=_iter_compact_missing_batches(
                paths["cache"], task_id=task_id, model=str(version["model"]),
                revision=str(version["model_revision"]), batch_size=args.batch_size
            ),
            cache=cache,
            snapshot=snapshot,
            devices=_parse_devices(args.devices),
            dtype=args.dtype,
        )
    finally:
        cache.close()
    _finalize_compact_cache(paths, version)
    version["cache_sha256"] = file_sha256(paths["cache"])
    _write_json_atomic(paths["version"], version)
    result = verify_task(args, task_id)
    result["n_scored_this_run"] = scored
    return result


def verify_task(args: argparse.Namespace, task_id: str) -> dict[str, Any]:
    paths = _task_paths(args, task_id)
    version = json.loads(paths["version"].read_text(encoding="utf-8"))
    local_model = version.get("local_model")
    if local_model and _directory_sha256(Path(local_model["path"])) != local_model["sha256"]:
        raise ValueError("Local assay-transfer model content differs from VERSION.json")
    expected = int(version["n_prompt_scores"])
    connection = sqlite3.connect(f"file:{paths['cache'].resolve()}?mode=ro", uri=True)
    try:
        tables = {
            str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if version.get("status") != "complete":
            cached = int(connection.execute("SELECT COUNT(*) FROM prompt_scores").fetchone()[0])
            return {**version, "n_cached_scores": cached, "cache_quick_check": quick_check}
        forbidden = {"prompt_tasks", "prompt_assignments", "prompt_scores"} & tables
        if forbidden:
            raise ValueError(f"Final compact cache retained build tables: {sorted(forbidden)}")
        cache_count = int(connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0])
        assignments = int(connection.execute("SELECT COUNT(*) FROM assignments").fetchone()[0])
        orphans = int(connection.execute(
            "SELECT COUNT(*) FROM assignments a LEFT JOIN scores s USING(score_key) "
            "WHERE s.score_key IS NULL"
        ).fetchone()[0])
    finally:
        connection.close()
    if cache_count != expected or assignments != int(version["n_score_assignments"]) or orphans or quick_check != "ok":
        raise ValueError(
            "Incomplete compact v11 cache: "
            f"scores={cache_count}/{expected}, assignments={assignments}, "
            f"orphans={orphans}, quick_check={quick_check}"
        )
    observed_hash = file_sha256(paths["cache"])
    if version.get("cache_sha256") != observed_hash:
        raise ValueError("Compact v11 cache hash differs from VERSION.json")
    version.update(n_cached_scores=cache_count, cache_quick_check=quick_check)
    _write_json_atomic(paths["version"], version)
    return version


def _task_path_map(values: list[str], flag: str) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        task_id, separator, raw_path = value.partition("=")
        if not separator or task_id not in SUPPORTED_TASKS or not raw_path:
            raise ValueError(f"{flag} requires TASK=PATH with a supported task")
        if task_id in output:
            raise ValueError(f"{flag} repeats task {task_id}")
        output[task_id] = Path(raw_path)
    return output


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=SUPPORTED_TASKS, default=list(DEFAULT_TASKS))
    parser.add_argument("--benchmark-split", choices=["scaffold", "random"], default="scaffold")
    parser.add_argument("--evaluation-subset", choices=["valid", "test"], default="valid")
    parser.add_argument("--phase", choices=["prepare", "score", "verify", "all"], default="prepare")
    parser.add_argument("--output-root", default="outputs/chembl_tool/tasks")
    parser.add_argument("--task-evidence-view", action="append", default=[], metavar="TASK=PATH")
    parser.add_argument("--task-query-jsonl", action="append", default=[], metavar="TASK=PATH")
    parser.add_argument("--task-cache-dir", action="append", default=[], metavar="TASK=PATH")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--record-scope", choices=RECORD_SCOPES, default="all")
    parser.add_argument("--assay-transfer-initial-morgan-filter", type=int, default=50)
    parser.add_argument("--min-similarity", type=float, default=0.0)
    parser.add_argument(
        "--neighbor-identity-policy",
        choices=["operational", "parent_disjoint"],
        default="parent_disjoint",
    )
    parser.add_argument("--devices", default="0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dtype", choices=[BACKBONE_DTYPE], default=BACKBONE_DTYPE)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--assay-transfer-model", default=None)
    parser.add_argument("--assay-transfer-model-revision", default=None)
    parser.add_argument("--assay-transfer-local-model-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.assay_transfer_initial_morgan_filter <= 0:
        parser.error("--assay-transfer-initial-morgan-filter must be positive")
    if not 0.0 <= args.min_similarity <= 1.0:
        parser.error("--min-similarity must be between 0 and 1")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.record_scope == "numeric_direct" and args.tasks != ["bioavailability_ma"]:
        parser.error("--record-scope numeric_direct requires only bioavailability_ma")
    if bool(args.assay_transfer_model) != bool(args.assay_transfer_model_revision):
        parser.error("model and model revision overrides must be provided together")
    if args.assay_transfer_model and len(args.tasks) != 1:
        parser.error("model overrides require exactly one task")
    if args.assay_transfer_local_model_dir and (
        len(args.tasks) != 1 or args.assay_transfer_model
    ):
        parser.error("a local model requires exactly one task and no model override")
    try:
        args.task_evidence_views = _task_path_map(
            args.task_evidence_view, "--task-evidence-view"
        )
        args.task_query_jsonls = _task_path_map(
            args.task_query_jsonl, "--task-query-jsonl"
        )
        args.task_cache_dirs = _task_path_map(args.task_cache_dir, "--task-cache-dir")
    except ValueError as exc:
        parser.error(str(exc))
    configured = set(args.task_evidence_views) | set(args.task_query_jsonls) | set(args.task_cache_dirs)
    for task_id in configured:
        if not all(
            task_id in mapping
            for mapping in (args.task_evidence_views, args.task_query_jsonls, args.task_cache_dirs)
        ):
            parser.error(
                f"Explicit lineage paths for {task_id} require evidence view, query JSONL, and cache dir"
            )
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    results = {}
    for task_id in args.tasks:
        _log(f"{task_id}: phase={args.phase}")
        if args.phase in {"prepare", "all"}:
            results[task_id] = prepare_task(args, task_id)
        if args.phase in {"score", "all"}:
            results[task_id] = score_task(args, task_id)
        elif args.phase == "verify":
            results[task_id] = verify_task(args, task_id)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
