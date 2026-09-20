"""Publish Gold-v1 plus TDC-train L1 rankings for TDC valid/test queries.

The cache stores ranked parent/context UIDs only. Gold contexts retain their
published physical voter membership and hydrate from the active evidence
projection; TDC contexts hydrate from the existing TDC projection.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import DataStructs

from data.processing.gold_labels.conditioned_benchmark import task_root, tdc_task_root
from predict.retrieval.policies import standardize_smiles_and_fp
from predict.utils.json import read_jsonl, sha256_file

from .build_ranked_retrieval import _digest
from .build_ranked_uid_retrieval import _schema, _write_complete
from .ranked_uid_retrieval import CAPACITY, SCHEMA_VERSION
from .runtime import (
    BACKBONE_DTYPE, LOGIT_EXTRACTION_DTYPE, SCORING_CONTRACT_VERSION,
    PromptTask, cache_profile_root, load_model, resolve_model_snapshot,
    score_prompt_batch,
)
from .score_tdc_ranked_retrieval import _prompt_task
from .v9 import V9PromptRenderer, model_profile, reference_provenance


PROFILE = "ranked_level_retrieval_tdc_v1_gold_v1_mixed_l1_assay_v10_3_best_v1"
TASKS = ("bbb_martins", "bioavailability_ma")
SUBSETS = ("valid", "test")
LINEAGE = "v10_3_best"
ROOT = Path(__file__).resolve().parents[3]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _projection_entry(index_path: Path) -> dict[str, Any]:
    index = _json(index_path)
    entry = dict(index["evidence"])
    manifest = (index_path.parent / entry["manifest"]).resolve()
    entry["manifest"] = str(manifest)
    if sha256_file(manifest) != entry["manifest_sha256"]:
        raise ValueError(f"Evidence manifest differs from release index: {manifest}")
    return entry


def _sources(task: str) -> tuple[list[dict[str, Any]], dict[str, list[str]], list[dict[str, Any]]]:
    gold_path = task_root(task, "v1") / "train_molecule_condition_labels.jsonl"
    tdc_path = tdc_task_root(task) / "train_molecule_condition_labels.jsonl"
    gold = read_jsonl(gold_path)
    tdc = read_jsonl(tdc_path)
    membership = pq.read_table(
        task_root(task, "v1") / "voter_membership.parquet",
        columns=["benchmark_row_id", "source_row_uid", "vote_id", "physical_member_index"],
    ).to_pylist()
    ordered: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for row in membership:
        ordered[str(row["benchmark_row_id"])].append((
            str(row["vote_id"]), int(row["physical_member_index"]),
            str(row["source_row_uid"]),
        ))
    members = {key: [row[2] for row in sorted(values)] for key, values in ordered.items()}
    contexts = []
    for source, rows in (("gold_v1", gold), ("tdc_v1", tdc)):
        for row in rows:
            context_id = str(row["benchmark_row_id"])
            identity = row["molecule_identity"]
            contexts.append({
                **row,
                "context_id": context_id,
                "source_kind": source,
                "parent_id": str(row["molecule_identity_key"]),
                "parent_smiles": str(identity["parent_smiles"]),
                "members": members[context_id] if source == "gold_v1"
                else list(map(str, row["source_record_ids"])),
            })
    if len({row["context_id"] for row in contexts}) != len(contexts):
        raise ValueError("Gold-v1 and TDC context IDs overlap")
    return contexts, members, [{"path": str(gold_path), "sha256": sha256_file(gold_path)}, {"path": str(tdc_path), "sha256": sha256_file(tdc_path)}]


def _reused_scores(task: str) -> dict[str, float]:
    root = cache_profile_root("ranked_level_retrieval_tdc_v1_assay_v10_3_best_v1") / task
    scores: dict[str, float] = {}
    for subset in SUBSETS:
        manifest = _json(root / "scaffold" / subset / "L1/VERSION.json")
        with sqlite3.connect(root / "scaffold" / subset / "L1" / manifest["database"]) as connection:
            for key, value in connection.execute(
                "SELECT score_key,assay_transfer_score FROM rankings "
                "WHERE score_key IS NOT NULL AND assay_transfer_score IS NOT NULL"
            ):
                scores[str(key)] = float(value)
    return scores


def prepare(task: str, output_root: Path) -> dict[str, Any]:
    destination = output_root / task
    if destination.exists():
        raise FileExistsError(f"Refusing to replace mixed L1 cache: {destination}")
    contexts, _, source_inputs = _sources(task)
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for context in contexts:
        by_parent[context["parent_id"]].append(context)
    parents = sorted(by_parent)
    fingerprints = [standardize_smiles_and_fp(by_parent[parent][0]["parent_smiles"])[2] for parent in parents]
    if any(value is None for value in fingerprints):
        raise ValueError("Mixed L1 contains an invalid parent fingerprint")
    renderer, profile = V9PromptRenderer(task), model_profile(task, LINEAGE)
    reused = _reused_scores(task)
    prompt_rows: dict[str, dict[str, Any]] = {}
    assignments = []
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{task}.", dir=output_root) as temporary:
        root = Path(temporary)
        selected_contexts: set[str] = set()
        for subset in SUBSETS:
            query_path = tdc_task_root(task) / f"{subset}_molecule_condition_labels.jsonl"
            queries = read_jsonl(query_path)
            target = root / "scaffold" / subset / "L1"
            target.mkdir(parents=True)
            database = target / "rankings.sqlite3"
            connection = sqlite3.connect(database)
            _schema(connection, "L1")
            ranking_rows, query_ledger, query_counts = [], [], {}
            for query in queries:
                query_id = str(query["benchmark_row_id"])
                query_parent = str(query["molecule_identity_key"])
                query_smiles = str(query["molecule_identity"]["parent_smiles"])
                query_fp = standardize_smiles_and_fp(query_smiles)[2]
                similarities = DataStructs.BulkTanimotoSimilarity(query_fp, fingerprints)
                candidates = sorted(
                    (
                        (float(similarity), parent)
                        for similarity, parent in zip(similarities, parents)
                        if parent != query_parent and (
                            not query.get("bemis_murcko_scaffold")
                            or str(by_parent[parent][0].get("bemis_murcko_scaffold") or "")
                            != str(query["bemis_murcko_scaffold"])
                        )
                    ),
                    key=lambda row: (-row[0], row[1]),
                )[:CAPACITY]
                if len(candidates) != CAPACITY:
                    raise ValueError(f"Insufficient mixed L1 parents: {task}/{subset}/{query_id}")
                query_ledger.append((query_id, str(query["drug"]), query_parent, query_smiles))
                member_count = 0
                for rank, (similarity, parent) in enumerate(candidates, 1):
                    parent_contexts = sorted(
                        by_parent[parent],
                        key=lambda row: (0 if row["source_kind"] == "gold_v1" else 1, row["context_id"]),
                    )
                    morgan_context = parent_contexts[0]
                    selected_contexts.update(row["context_id"] for row in parent_contexts)
                    member_count += len(morgan_context["members"])
                    ranking_rows.append((
                        query_id, parent, parent, morgan_context["parent_smiles"], similarity,
                        rank, 1, rank, None, None, morgan_context["context_id"], None,
                        len(morgan_context["members"]), None, None,
                    ))
                    for context in parent_contexts:
                        prompt = _prompt_task(task, renderer, profile, context, query)
                        value = reused.get(prompt.cache_key)
                        if value is None:
                            prompt_rows.setdefault(prompt.cache_key, asdict(prompt))
                        assignments.append({
                            "subset": subset, "benchmark_row_id": query_id,
                            "parent_id": parent, "context_id": context["context_id"],
                            "score_key": prompt.cache_key,
                            "reused_score": value,
                        })
                query_counts[query_id] = {
                    "candidate_parents": CAPACITY,
                    "morgan_candidate_records": member_count,
                    "assay_candidate_records": 0,
                }
            connection.executemany("INSERT INTO queries VALUES (?,?,?,?)", query_ledger)
            connection.executemany(
                "INSERT INTO rankings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ranking_rows
            )
            context_map = {row["context_id"]: row for row in contexts}
            used = sorted({row["context_id"] for row in contexts if row["context_id"] in selected_contexts})
            connection.executemany("INSERT OR IGNORE INTO contexts VALUES (?,?,?,?)", [
                (context_id, context_map[context_id]["parent_id"],
                 str(context_map[context_id].get("condition_group") or ""),
                 int(context_map[context_id]["Y"])) for context_id in used
            ])
            connection.executemany("INSERT OR IGNORE INTO context_records VALUES (?,?,?)", [
                (context_id, uid, rank)
                for context_id in used
                for rank, uid in enumerate(context_map[context_id]["members"], 1)
            ])
            connection.commit()
            identity = {
                "schema_version": SCHEMA_VERSION, "status": "prepared",
                "task_id": task, "subset": subset, "level": "L1", "pool": "fixed",
                "capacity": CAPACITY, "parent_capacity": CAPACITY,
                "label_release": {"benchmark": "tdc", "version": "v1"},
                "l1_sources": ["gold_v1", "tdc_v1"],
                "neighbor_identity_policy": "scaffold_disjoint",
                "shared_candidate_universe": True, "query_count": len(queries),
                "stored_rows": len(ranking_rows), "query_counts": query_counts,
                "model": profile, "assay_transfer_status": "prepared",
                "inputs": {"query_sha256": sha256_file(query_path), "candidate_sources": source_inputs},
                "database": database.name,
            }
            _write_json(target / "VERSION.json", identity)
            connection.close()
        build = root / ".build"
        build.mkdir()
        pq.write_table(pa.Table.from_pylist([prompt_rows[key] for key in sorted(prompt_rows)]), build / "prompts.parquet", compression="zstd")
        pq.write_table(pa.Table.from_pylist(assignments), build / "assignments.parquet", compression="zstd")
        prepared = {
            "status": "prepared", "task_id": task, "profile": PROFILE,
            "model": profile, "prompt_count": len(prompt_rows),
            "prompts_sha256": sha256_file(build / "prompts.parquet"),
            "assignments_sha256": sha256_file(build / "assignments.parquet"),
            "exact_reuse": sum(row["reused_score"] is not None for row in assignments),
        }
        _write_json(build / "PREPARED.json", prepared)
        os.replace(root, destination)
    return prepared


def score(task: str, output_root: Path, shard_index: int, num_shards: int, device: int, batch_size: int) -> dict[str, Any]:
    root = output_root / task
    prepared = _json(root / ".build/PREPARED.json")
    rows = pq.read_table(root / ".build/prompts.parquet").to_pylist()[shard_index::num_shards]
    tasks = [PromptTask(**row) for row in rows]
    journal_dir = root / ".build/scores"
    journal_dir.mkdir(exist_ok=True)
    journal = journal_dir / f"{shard_index:02d}-of-{num_shards:02d}.jsonl"
    done = read_jsonl(journal) if journal.exists() else []
    if [row["cache_key"] for row in done] != [row.cache_key for row in tasks[:len(done)]]:
        raise ValueError("Score journal is not an exact prompt prefix")
    renderer, profile = V9PromptRenderer(task), model_profile(task, LINEAGE)
    snapshot = resolve_model_snapshot(profile["model"], profile["revision"], local_files_only=True)
    model, tokenizer = load_model(snapshot, device=device)
    with journal.open("a", encoding="utf-8") as handle:
        for offset in range(len(done), len(tasks), batch_size):
            for result in score_prompt_batch(model, tokenizer, tasks[offset:offset + batch_size], device=device):
                handle.write(json.dumps(asdict(result), sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush(); os.fsync(handle.fileno())
    return {"task": task, "shard": shard_index, "completed": len(tasks), "template": renderer.template_hash}


def finalize(task: str, output_root: Path, num_shards: int) -> dict[str, Any]:
    root = output_root / task
    prepared = _json(root / ".build/PREPARED.json")
    scores: dict[str, float] = {}
    for shard in range(num_shards):
        for row in read_jsonl(root / ".build/scores" / f"{shard:02d}-of-{num_shards:02d}.jsonl"):
            key, value = str(row["cache_key"]), float(row["transfer_probability"])
            if key in scores or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid duplicate assay score: {key}")
            scores[key] = value
    prompts = {str(row["cache_key"]) for row in pq.read_table(root / ".build/prompts.parquet", columns=["cache_key"]).to_pylist()}
    if set(scores) != prompts:
        raise ValueError(f"Fresh score mismatch: missing={len(prompts-set(scores))}")
    assignments = pq.read_table(root / ".build/assignments.parquet").to_pylist()
    by_split_query_parent: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in assignments:
        row["score"] = float(row["reused_score"] if row["reused_score"] is not None else scores[row["score_key"]])
        by_split_query_parent[(row["subset"], row["benchmark_row_id"], row["parent_id"])].append(row)
    index_splits = {}
    for subset in SUBSETS:
        level = root / "scaffold" / subset / "L1"
        manifest_path = level / "VERSION.json"
        manifest = _json(manifest_path)
        database = level / manifest["database"]
        with sqlite3.connect(database) as connection:
            queries = [row[0] for row in connection.execute("SELECT benchmark_row_id FROM queries")]
            for query_id in queries:
                updates = []
                for parent_id, in connection.execute("SELECT parent_id FROM rankings WHERE benchmark_row_id=?", (query_id,)):
                    chosen = min(
                        by_split_query_parent[(subset, query_id, parent_id)],
                        key=lambda row: (-row["score"], row["context_id"]),
                    )
                    members = connection.execute(
                        "SELECT COUNT(*) FROM context_records WHERE context_id=?", (chosen["context_id"],)
                    ).fetchone()[0]
                    updates.append((chosen["score"], chosen["context_id"], members, chosen["score_key"], query_id, parent_id))
                connection.executemany(
                    "UPDATE rankings SET assay_transfer_score=?,assay_context_id=?,assay_member_count=?,score_key=? WHERE benchmark_row_id=? AND parent_id=?",
                    updates,
                )
                ordered = connection.execute(
                    "SELECT item_id FROM rankings WHERE benchmark_row_id=? ORDER BY assay_transfer_score DESC,item_id", (query_id,)
                ).fetchall()
                connection.executemany(
                    "UPDATE rankings SET assay_rank=? WHERE benchmark_row_id=? AND item_id=?",
                    [(rank, query_id, item_id) for rank, (item_id,) in enumerate(ordered, 1)],
                )
            counts = dict(manifest["query_counts"])
            for query_id in queries:
                counts[query_id]["assay_candidate_records"] = connection.execute(
                    "SELECT SUM(assay_member_count) FROM rankings WHERE benchmark_row_id=?", (query_id,)
                ).fetchone()[0]
            identity = {key: value for key, value in manifest.items() if key not in {"schema_version", "status", "database", "content_id", "database_sha256", "ranking_rows_sha256"}}
            identity.update({
                "query_counts": counts, "assay_transfer_status": "complete",
                "scoring_contract_version": SCORING_CONTRACT_VERSION,
                "backbone_dtype": BACKBONE_DTYPE, "logit_extraction_dtype": LOGIT_EXTRACTION_DTYPE,
                "reference_provenance": reference_provenance(task, LINEAGE),
            })
            final = _write_complete(connection, database, task=task, subset=subset, level="L1", identity=identity, target=level)
        index_splits[subset] = {"levels": {"L1": {
            "manifest": str(manifest_path.relative_to(root)),
            "manifest_sha256": sha256_file(manifest_path), "content_id": final["content_id"],
        }}}
    gold_index = ROOT / "data/caches/assay_reranking/active/ranked_level_retrieval_v3" / task / "RELEASE_INDEX.json"
    tdc_index = cache_profile_root("ranked_level_retrieval_tdc_v1_assay_v10_3_best_v1") / task / "RELEASE_INDEX.json"
    evidence_sources = [_projection_entry(path) for path in (gold_index, tdc_index)]
    index = {
        "schema_version": "ranked_uid_task_release_index.v1", "selection_contract": SCHEMA_VERSION,
        "profile": PROFILE, "task_id": task, "status": "complete",
        "label_release": {"benchmark": "tdc", "version": "v1"}, "l1_sources": ["gold_v1", "tdc_v1"],
        "pool": "all", "parent_capacity": CAPACITY, "levels_independent": True,
        "ranking_modes": ["morgan", "assay-transfer"], "assay_transfer_status": "complete",
        "neighbor_identity_policy_by_level": {"L1": "scaffold_disjoint"},
        "evidence_sources": evidence_sources, "splits": index_splits,
    }
    _write_json(root / "RELEASE_INDEX.json", index)
    report = validate(task, output_root)
    _write_json(root / "VALIDATION.json", report)
    shutil.rmtree(root / ".build")
    return report


def validate(task: str, output_root: Path) -> dict[str, Any]:
    root = output_root / task
    index = _json(root / "RELEASE_INDEX.json")
    if index.get("status") != "complete" or len(index.get("evidence_sources") or []) != 2:
        raise ValueError("Mixed L1 release is incomplete")
    report = {"status": "complete", "task_id": task, "profile": PROFILE, "splits": {}}
    for subset in SUBSETS:
        manifest_path = root / index["splits"][subset]["levels"]["L1"]["manifest"]
        manifest = _json(manifest_path)
        with sqlite3.connect(manifest_path.with_name(manifest["database"])) as connection:
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise ValueError("Mixed L1 database failed integrity")
            queries = connection.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
            bad = connection.execute(
                "SELECT COUNT(*) FROM rankings WHERE assay_rank IS NULL OR assay_transfer_score NOT BETWEEN 0 AND 1 OR assay_context_id IS NULL"
            ).fetchone()[0]
            dense = connection.execute(
                "SELECT COUNT(*) FROM (SELECT benchmark_row_id FROM rankings GROUP BY benchmark_row_id HAVING COUNT(*)=100 AND MIN(assay_rank)=1 AND MAX(assay_rank)=100 AND COUNT(DISTINCT assay_rank)=100)"
            ).fetchone()[0]
            if bad or dense != queries:
                raise ValueError(f"Mixed L1 ranks are incomplete: {task}/{subset}")
        report["splits"][subset] = {"queries": queries, "ranking_rows": manifest["stored_rows"]}
    report["content_id"] = _digest(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "score", "finalize", "validate"))
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--output-root", type=Path, default=cache_profile_root(PROFILE))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.command == "prepare": result = prepare(args.task, args.output_root)
    elif args.command == "score": result = score(args.task, args.output_root, args.shard_index, args.num_shards, args.device, args.batch_size)
    elif args.command == "finalize": result = finalize(args.task, args.output_root, args.num_shards)
    else: result = validate(args.task, args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
