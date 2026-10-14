"""Publish exact frozen assay scores without changing TDC-v2 query-pure caches."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import tempfile
from pathlib import Path

import yaml

from data.processing.gold_labels.conditioned_benchmark import tdc_task_root
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic

from .build_tdc_v2_query_pure_l2plus import SOURCE_PROFILES, TASKS
from .publish_tdc_v2_reused_l1 import _profile
from .runtime import SCORING_CONTRACT_VERSION, cache_profile_root
from .score_tdc_ranked_retrieval import DirectL1PromptRenderer, _model_profile, _prompt_task


ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "predict/retrieval/assay_reranking/ranked_level_retrieval_tdc_v2_query_pure_v1.yaml"
OUTPUT = ROOT / "data/caches/assay_reranking/active/tdc_v2_exact_scores_v1"
LEVELS = {
    "bbb_martins": ("L2", "L3", "L4", "L5"),
    "bioavailability_ma": ("L2", "L3", "L4", "L5", "L6"),
    "skin_reaction": ("L2", "L3"),
    "ames": ("L2", "L3", "L4", "L5"),
    "dili": ("L2", "L3", "L4", "L5", "L6", "L7"),
    "carcinogens": ("L2", "L3", "L4", "L5", "L6", "L7"),
}


def _level(index_path: Path, subset: str, level: str, inputs: dict[str, str]) -> tuple[Path, dict]:
    index = json.loads(index_path.read_text())
    if index.get("status") != "complete":
        raise ValueError(f"Incomplete score input: {index_path}")
    entry = index["splits"][subset]["levels"][level]
    path = (index_path.parent / entry["manifest"]).resolve()
    if sha256_file(path) != entry["manifest_sha256"]:
        raise ValueError(f"Score input manifest changed: {path}")
    manifest = json.loads(path.read_text())
    database = path.with_name(manifest["database"])
    if sha256_file(database) != manifest["database_sha256"]:
        raise ValueError(f"Score input database changed: {database}")
    inputs[str(index_path)] = sha256_file(index_path)
    inputs[str(path)] = entry["manifest_sha256"]
    inputs[str(database)] = manifest["database_sha256"]
    return database, manifest


def _reader(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)


def _record(db: sqlite3.Connection, key: tuple[str, str, str, str], score: float, score_key: str) -> None:
    if not math.isfinite(score) or not 0 <= score <= 1 or not score_key:
        raise ValueError(f"Invalid exact assay score: {key}")
    db.execute("INSERT INTO scores VALUES (?,?,?,?,?,?)", (*key, score, score_key))


def _direct(task: str, index: Path, db: sqlite3.Connection, inputs: dict[str, str]) -> dict[str, int]:
    old_index = cache_profile_root(_profile(task, "tdc-v1")) / task / "RELEASE_INDEX.json"
    renderer = DirectL1PromptRenderer(task, "tdc-v1")
    model = _model_profile(task, "tdc-v1")
    scores: dict[str, float] = {}
    projections = set()
    for subset in ("valid", "test"):
        path, manifest = _level(old_index, subset, "L1", inputs)
        if (manifest.get("model") != model or
                manifest.get("scoring_contract_version") != SCORING_CONTRACT_VERSION or
                manifest["prompt_assets"]["template_sha256"] != renderer.template_hash):
            raise ValueError(f"Frozen L1 score contract changed: {task}/{subset}")
        projections.add(manifest["prompt_assets"]["projection_sha256"])
        with _reader(path) as source:
            for key, value in source.execute("SELECT score_key,assay_transfer_score FROM rankings"):
                if key in scores and scores[key] != value:
                    raise ValueError(f"Conflicting frozen L1 score: {key}")
                scores[str(key)] = float(value)
    if len(projections) != 1:
        raise ValueError("Frozen L1 score projection differs across splits")
    renderer.projection_hash = projections.pop()
    train_path = tdc_task_root(task) / "train_molecule_condition_labels.jsonl"
    inputs[str(train_path.resolve())] = sha256_file(train_path)
    train: dict[str, list[dict]] = {}
    for row in read_jsonl(train_path):
        train.setdefault(str(row["molecule_identity_key"]), []).append(row)
    counts = {}
    for subset in ("valid", "test"):
        query_path = tdc_task_root(task) / f"{subset}_molecule_condition_labels.jsonl"
        inputs[str(query_path.resolve())] = sha256_file(query_path)
        queries = {str(row["benchmark_row_id"]): row for row in read_jsonl(query_path)}
        path, _ = _level(index, subset, "L1", inputs)
        count = 0
        with _reader(path) as source:
            for query_id, parent, context_id, members, frozen_score, frozen_key in source.execute(
                "SELECT benchmark_row_id,parent_id,morgan_context_id,morgan_member_count,"
                "assay_transfer_score,score_key FROM rankings"):
                contexts = train[parent]
                if members != sum(len(row["source_record_ids"]) for row in contexts):
                    raise ValueError(f"L1 physical membership changed: {task}/{parent}")
                keys = [_prompt_task(task, renderer, model, row, queries[query_id]).cache_key
                        for row in contexts]
                if any(key not in scores for key in keys):
                    if frozen_score is not None:
                        raise ValueError(f"Published L1 score lacks exact key: {query_id}/{parent}")
                    continue
                best = max(keys, key=lambda key: (scores[key], key))
                score = scores[best]
                if frozen_score is not None and (score != frozen_score or best != frozen_key):
                    raise ValueError(f"Published L1 score differs: {query_id}/{parent}")
                _record(db, (query_id, "L1", context_id, parent), score, best)
                count += 1
        counts[f"{subset}/L1"] = count
    return counts


def _indirect(task: str, index: Path, db: sqlite3.Connection, inputs: dict[str, str]) -> dict[str, int]:
    old_index = cache_profile_root(SOURCE_PROFILES[task]) / task / "RELEASE_INDEX.json"
    counts = {}
    for level in LEVELS[task]:
        old = []
        for subset in ("valid", "test"):
            path, manifest = _level(old_index, subset, level, inputs)
            reader = _reader(path)
            identities = {}
            for query_id, drug, parent, smiles in reader.execute("SELECT * FROM queries"):
                identity = (drug, parent, smiles)
                if identity in identities:
                    raise ValueError(f"Duplicate frozen query identity: {task}/{level}")
                identities[identity] = query_id
            old.append((reader, identities, manifest))
        try:
            if old[0][2].get("model") != old[1][2].get("model"):
                raise ValueError(f"Frozen L2+ score model differs: {task}/{level}")
            for subset in ("valid", "test"):
                path, _ = _level(index, subset, level, inputs)
                count = 0
                with _reader(path) as source:
                    for query_id, drug, parent, smiles in source.execute("SELECT * FROM queries"):
                        known: dict[tuple[str, str], tuple[float, str]] = {}
                        for reader, identities, _ in old:
                            old_id = identities.get((drug, parent, smiles))
                            if old_id is None:
                                continue
                            for uid, row_parent, score, key in reader.execute(
                                "SELECT item_id,parent_id,assay_transfer_score,score_key "
                                "FROM rankings WHERE benchmark_row_id=? AND assay_transfer_score IS NOT NULL",
                                (old_id,)):
                                item = (uid, row_parent)
                                if item in known and known[item] != (score, key):
                                    raise ValueError(f"Conflicting frozen L2+ score: {task}/{level}/{item}")
                                known[item] = (score, key)
                        for uid, row_parent, frozen_score, frozen_key in source.execute(
                            "SELECT item_id,parent_id,assay_transfer_score,score_key "
                            "FROM rankings WHERE benchmark_row_id=?", (query_id,)):
                            value = known.get((uid, row_parent))
                            if value is None:
                                if frozen_score is not None:
                                    raise ValueError(f"Published L2+ score lacks exact source: {query_id}/{uid}")
                                continue
                            score, key = value
                            if frozen_score is not None and (score != frozen_score or key != frozen_key):
                                raise ValueError(f"Published L2+ score differs: {query_id}/{uid}")
                            _record(db, (query_id, level, uid, row_parent), float(score), str(key))
                            count += 1
                counts[f"{subset}/{level}"] = count
        finally:
            for reader, _, _ in old:
                reader.close()
    return counts


def build(task: str, *, bundle_path: Path = BUNDLE, output_root: Path = OUTPUT) -> Path:
    if task not in TASKS:
        raise ValueError(f"Unsupported TDC task: {task}")
    bundle_path, output_root = bundle_path.resolve(), output_root.resolve()
    target = output_root / task
    if target.exists():
        raise FileExistsError(f"Refusing to replace exact-score overlay: {target}")
    bundle = yaml.safe_load(bundle_path.read_text())
    entry = bundle["caches"][task]
    direct = (bundle_path.parent / entry["L1"]).resolve()
    indirect = (bundle_path.parent / entry["later"]).resolve()
    inputs = {str(bundle_path): sha256_file(bundle_path)}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{task}.", dir=target.parent) as scratch:
        stage = Path(scratch) / task
        stage.mkdir()
        database = stage / "scores.sqlite3"
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE scores (query_id TEXT NOT NULL, level TEXT NOT NULL, "
                               "item_id TEXT NOT NULL, parent_id TEXT NOT NULL, score REAL NOT NULL, "
                               "score_key TEXT NOT NULL, PRIMARY KEY(query_id,level,item_id))")
            counts = {**_direct(task, direct, connection, inputs),
                      **_indirect(task, indirect, connection, inputs)}
            connection.commit()
            total = connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
            if total != sum(counts.values()) or connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Exact-score overlay failed integrity checks")
        write_json_atomic(stage / "manifest.json", {
            "schema_version": "tdc_v2_exact_scores.v1", "status": "complete", "task": task,
            "inputs": inputs, "counts": counts, "score_count": total,
            "database": database.name, "database_sha256": sha256_file(database),
        })
        stage.rename(target)
    return target / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(build(args.task, bundle_path=args.bundle, output_root=args.output_root))


if __name__ == "__main__":
    main()
