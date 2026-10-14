"""Publish TDC-v2 L1 query-pure ranks using exact pinned v1 scores."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

from data.processing.gold_labels.conditioned_benchmark import tdc_task_root
from predict.utils.json import read_jsonl, sha256_file

from .build_ranked_uid_retrieval import _write_complete
from .runtime import SCORING_CONTRACT_VERSION, cache_profile_root
from .score_tdc_ranked_retrieval import (
    DirectL1PromptRenderer, _model_profile, _profile, _prompt_task, _write_json,
)


TASKS = (
    "bbb_martins", "bioavailability_ma", "skin_reaction",
    "ames", "dili", "carcinogens",
)
COMPLETE_TASKS = frozenset(("bioavailability_ma", "skin_reaction"))
QUERY_PURE_POLICY = {
    "schema_version": "tdc_v2_l1_query_pure.v1",
    "complete_order": "assay_transfer",
    "incomplete_order": "morgan",
    "completeness": "all_100_parents_and_all_train_contexts",
}
ROOT = Path(__file__).resolve().parents[3]


def _index(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise ValueError(f"Incomplete cache release: {path}")
    return value


def _validate(base: Path, result: Path) -> None:
    frozen = (
        "benchmark_row_id,item_id,parent_id,parent_smiles,morgan_similarity,"
        "parent_morgan_rank,within_parent_rank,morgan_rank,morgan_context_id,morgan_member_count"
    )
    index = _index(result / "RELEASE_INDEX.json")
    for subset in ("valid", "test"):
        old_level = base / "scaffold" / subset / "L1"
        new_level = result / "scaffold" / subset / "L1"
        entry = index["splits"][subset]["levels"]["L1"]
        if sha256_file(new_level / "VERSION.json") != entry["manifest_sha256"]:
            raise ValueError(f"Published manifest hash differs: {subset}")
        old_manifest, new_manifest = _index(old_level / "VERSION.json"), _index(new_level / "VERSION.json")
        with sqlite3.connect(old_level / old_manifest["database"]) as old, sqlite3.connect(new_level / new_manifest["database"]) as new:
            if new.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise ValueError(f"Assay database failed integrity: {subset}")
            for table, columns, order in (
                ("rankings", frozen, "benchmark_row_id,item_id"),
                ("queries", "*", "benchmark_row_id"),
                ("contexts", "*", "context_id"),
                ("context_records", "*", "context_id,within_context_rank"),
            ):
                query = f"SELECT {columns} FROM {table} ORDER BY {order}"
                if old.execute(query).fetchall() != new.execute(query).fetchall():
                    raise ValueError(f"Frozen {table} changed: {subset}")
            bad = new.execute(
                "SELECT COUNT(*) FROM rankings WHERE "
                "(assay_transfer_score IS NULL)!=(assay_rank IS NULL) "
                "OR (assay_transfer_score IS NULL)!=(score_key IS NULL) "
                "OR (assay_transfer_score IS NULL)!=(assay_context_id IS NULL) "
                "OR (assay_transfer_score IS NULL)!=(assay_member_count IS NULL) "
                "OR (assay_transfer_score IS NOT NULL AND "
                "(assay_transfer_score<0 OR assay_transfer_score>1 "
                "OR assay_context_id!=morgan_context_id "
                "OR assay_member_count!=morgan_member_count))"
            ).fetchone()[0]
            queries = new.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
            ranked = new.execute(
                "SELECT COUNT(*) FROM (SELECT benchmark_row_id FROM rankings "
                "GROUP BY benchmark_row_id HAVING MIN(assay_rank)=1 "
                "AND MAX(assay_rank)=100 AND COUNT(DISTINCT assay_rank)=100)"
            ).fetchone()[0]
            unranked = new.execute(
                "SELECT COUNT(*) FROM (SELECT benchmark_row_id FROM rankings "
                "GROUP BY benchmark_row_id HAVING COUNT(assay_rank)=0)"
            ).fetchone()[0]
            if bad or ranked + unranked != queries:
                raise ValueError(f"Mixed or incomplete query ranks: {subset}")
            expected_ranked = index.get("query_pure_summary", {}).get(subset, {}).get("assay_pure_queries", queries)
            if ranked != expected_ranked:
                raise ValueError(f"Assay-pure query census differs: {subset}")


def publish(task: str) -> dict:
    if task not in TASKS:
        raise ValueError(f"Unsupported TDC task: {task}")
    old_root = cache_profile_root(_profile(task, "tdc-v1")) / task
    base = ROOT / "data/caches/assay_reranking/active/flat_v5/tdc_v2" / task / "l1/morgan/tdc_v2/original"
    target = base.parent.parent.parent / (
        "assay_transfer/reused_task_best_v1/original" if task in COMPLETE_TASKS
        else "query_pure/reused_task_best_v1/original"
    )
    if target.exists():
        raise FileExistsError(f"Refusing to replace assay cache: {target}")
    old_index_path, base_index_path = old_root / "RELEASE_INDEX.json", base / "RELEASE_INDEX.json"
    old_index, base_index = _index(old_index_path), _index(base_index_path)
    if old_index.get("ranking_modes") != ["morgan", "assay-transfer"] or base_index.get("ranking_modes") != ["morgan"]:
        raise ValueError("Unexpected source ranking modes")
    if base_index.get("label_release") != {"benchmark": "tdc", "version": "v2"}:
        raise ValueError("Base cache is not TDC v2")
    renderer, model = DirectL1PromptRenderer(task, "tdc-v1"), _model_profile(task, "tdc-v1")
    scores: dict[str, float] = {}
    projection_hashes = set()
    for subset in ("valid", "test"):
        level = old_root / "scaffold" / subset / "L1"
        manifest = _index(level / "VERSION.json")
        if (manifest.get("model") != model or
            manifest.get("scoring_contract_version") != SCORING_CONTRACT_VERSION or
            manifest["prompt_assets"]["template_sha256"] != renderer.template_hash):
            raise ValueError(f"Old assay score contract differs: {task}/{subset}")
        projection_hashes.add(manifest["prompt_assets"]["projection_sha256"])
        with sqlite3.connect(level / manifest["database"]) as connection:
            for key, score in connection.execute(
                "SELECT score_key,assay_transfer_score FROM rankings"
            ):
                if key in scores and scores[key] != score:
                    raise ValueError(f"Conflicting old assay score: {key}")
                scores[str(key)] = float(score)
    if len(projection_hashes) != 1:
        raise ValueError("Old projection hashes differ between splits")
    renderer.projection_hash = projection_hashes.pop()
    train_rows = read_jsonl(tdc_task_root(task) / "train_molecule_condition_labels.jsonl")
    train: dict[str, list[dict]] = {}
    for row in train_rows:
        parent = str(row["molecule_identity_key"])
        train.setdefault(parent, []).append(row)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{task}.", dir=target.parent) as scratch:
        result_root = Path(scratch) / "original"
        shutil.copytree(base, result_root)
        result_index = _index(result_root / "RELEASE_INDEX.json")
        counts = {}
        for subset in ("valid", "test"):
            level = result_root / "scaffold" / subset / "L1"
            manifest_path = level / "VERSION.json"
            manifest = _index(manifest_path)
            queries = {
                str(row["benchmark_row_id"]): row
                for row in read_jsonl(tdc_task_root(task) / f"{subset}_molecule_condition_labels.jsonl")
            }
            database = level / manifest["database"]
            with sqlite3.connect(database) as connection:
                rows = connection.execute(
                    "SELECT benchmark_row_id,item_id,morgan_member_count FROM rankings"
                ).fetchall()
                updates: dict[str, list[tuple[float, str, str, str]]] = {}
                incomplete: set[str] = set()
                for query_id, parent, members in rows:
                    known = train[parent]
                    if members != sum(len(context["source_record_ids"]) for context in known):
                        raise ValueError(f"Physical train membership differs: {task}/{parent}")
                    keys = [
                        _prompt_task(task, renderer, model, context, queries[query_id]).cache_key
                        for context in known
                    ]
                    if any(key not in scores for key in keys):
                        incomplete.add(query_id)
                        continue
                    best = max(keys, key=lambda key: (scores[key], key))
                    updates.setdefault(query_id, []).append((scores[best], best, query_id, parent))
                query_ids = [row[0] for row in connection.execute("SELECT benchmark_row_id FROM queries")]
                pure = set(query_ids) - incomplete
                scored_updates = [row for query_id in pure for row in updates.get(query_id, [])]
                if len(scored_updates) != 100 * len(pure):
                    raise ValueError(f"Assay-pure L1 query has fewer than 100 parents: {task}/{subset}")
                connection.executemany(
                    "UPDATE rankings SET assay_transfer_score=?,score_key=?,"
                    "assay_context_id=morgan_context_id,assay_member_count=morgan_member_count "
                    "WHERE benchmark_row_id=? AND item_id=?", scored_updates,
                )
                for query_id in pure:
                    ordered = connection.execute(
                        "SELECT item_id FROM rankings WHERE benchmark_row_id=? "
                        "ORDER BY assay_transfer_score DESC,item_id", (query_id,),
                    ).fetchall()
                    connection.executemany(
                        "UPDATE rankings SET assay_rank=? WHERE benchmark_row_id=? AND item_id=?",
                        [(rank, query_id, item_id) for rank, (item_id,) in enumerate(ordered, 1)],
                    )
                connection.commit()
                query_counts = dict(manifest["query_counts"])
                for query_id in pure:
                    query_counts[query_id]["assay_candidate_records"] = query_counts[query_id]["morgan_candidate_records"]
                identity = {key: value for key, value in manifest.items() if key not in {
                    "schema_version", "status", "content_id", "database", "database_sha256", "ranking_rows_sha256",
                }}
                identity.update({
                    "query_counts": query_counts,
                    "assay_transfer_status": "complete" if not incomplete else "query_pure_partial",
                    "score_count": len(scored_updates), "model": model,
                    "scoring_contract_version": SCORING_CONTRACT_VERSION,
                    "prompt_assets": {"template_sha256": renderer.template_hash,
                                      "projection_sha256": renderer.projection_hash},
                    "score_reuse": {"method": "exact_v1_score_key",
                                    "source_release_index_sha256": sha256_file(old_index_path)},
                })
                complete = _write_complete(
                    connection, database, task=task, subset=subset, level="L1",
                    identity=identity, target=level,
                )
            result_index["splits"][subset]["levels"]["L1"].update({
                "content_id": complete["content_id"], "manifest_sha256": sha256_file(manifest_path),
            })
            counts[subset] = {
                "queries": len(query_ids), "assay_pure_queries": len(pure),
                "morgan_pure_queries": len(incomplete), "scores_reused": len(scored_updates),
            }
        result_index.update({
            "profile": f"flat_v5/tdc_v2/{task}/l1/{'assay_transfer' if task in COMPLETE_TASKS else 'query_pure'}/reused_task_best_v1/original",
            "ranking_modes": ["morgan", "assay-transfer"] if task in COMPLETE_TASKS else ["morgan", "query-pure"],
            "assay_transfer_status": "complete" if task in COMPLETE_TASKS else "query_pure_partial",
            "model": model, "scoring_contract_version": SCORING_CONTRACT_VERSION,
            "base_release_index_sha256": sha256_file(base_index_path),
            "score_source_release_index_sha256": sha256_file(old_index_path),
            "query_pure_policy": QUERY_PURE_POLICY,
            "query_pure_summary": counts,
        })
        _write_json(result_root / "RELEASE_INDEX.json", result_index)
        _validate(base, result_root)
        os.replace(result_root, target)
    return {"task": task, "status": "complete", "target": str(target), "splits": counts,
            "release_index_sha256": sha256_file(target / "RELEASE_INDEX.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.task), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
