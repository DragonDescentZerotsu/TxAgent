"""Add pinned direct-model assay ranks to immutable L1 Morgan caches."""

from __future__ import annotations

import argparse
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
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from data.processing.gold_labels.conditioned_benchmark import tdc_task_root
from predict.utils.json import read_jsonl, sha256_file

from .build_ranked_uid_retrieval import _write_complete
from .runtime import (
    BACKBONE_DTYPE,
    LOGIT_EXTRACTION_DTYPE,
    SCORING_CONTRACT_VERSION,
    PromptTask,
    cache_profile_root,
    model_profile as runtime_model_profile,
    load_model,
    resolve_model_snapshot,
    score_prompt_batch,
)
from .v9 import V9PromptRenderer, _cache_key, _record_value, model_profile, reference_provenance


BASE_PROFILE = "ranked_level_retrieval_tdc_v1"
PROFILE = "ranked_level_retrieval_tdc_v1_assay_v10_3_best_v1"
SKIN_PROFILE = "ranked_level_retrieval_tdc_v1_assay_skin_v9_v1"
GOLD_PROFILE = "ranked_level_retrieval_gold_v1_addon_l1_assay_safety_best_v1"
TDC_PROFILE = "ranked_level_retrieval_tdc_v1_l1_assay_task_best_v1"
GOLD_BASE_PROFILE = "ranked_level_retrieval_gold_v1_addon_v2"
GOLD_TASKS = {"ames": "Ames", "dili": "DILI", "carcinogens": "Carcinogens"}
TDC_TASKS = ("bbb_martins", "bioavailability_ma", "skin_reaction")
TASKS = (*TDC_TASKS, *GOLD_TASKS)
SUBSETS = ("valid", "test")
ASSET_ROOT = Path(__file__).with_name("prompts") / "direct_l1_task_best_v1"
CONFIG_PATH = ASSET_ROOT / "config.json"


def _lineage(task: str) -> str:
    return "v9" if task == "skin_reaction" else "v10_3_best"


def _profile(task: str, release: str = "legacy") -> str:
    if release == "gold-v1":
        return GOLD_PROFILE
    if release == "tdc-v1":
        return TDC_PROFILE
    return SKIN_PROFILE if task == "skin_reaction" else PROFILE


def _base_profile(release: str) -> str:
    return GOLD_BASE_PROFILE if release == "gold-v1" else BASE_PROFILE


def _model_profile(task: str, release: str) -> dict[str, Any]:
    return (
        model_profile(task, _lineage(task))
        if release == "legacy"
        else runtime_model_profile(task, "direct_task_best")
    )


class DirectL1PromptRenderer:
    def __init__(self, task: str, release: str):
        self.task_id = task
        self.release = release
        self.config = _json(CONFIG_PATH)
        section = "gold" if release == "gold-v1" else "tdc"
        self.task = dict(self.config[section][task])
        self.template = "gold.jinja" if release == "gold-v1" else "tdc.jinja"
        self.template_hash = sha256_file(ASSET_ROOT / self.template)
        self.projection_hash = sha256_file(CONFIG_PATH)
        self.environment = Environment(
            loader=FileSystemLoader(str(ASSET_ROOT)),
            undefined=StrictUndefined,
            autoescape=False,
        )

    @staticmethod
    def _context(row: dict[str, Any]) -> str | None:
        group = str(row.get("condition_group") or "")
        if group == "no_reported_external_condition":
            return None
        atoms = row.get("condition_atoms") or group.split("+")
        return "\n".join(
            f"- {key.replace('_', ' ').capitalize()}: {value.replace('_', ' ')}"
            for key, value in (str(atom).split("=", 1) for atom in atoms)
        )

    def render(self, known: dict[str, Any], query: dict[str, Any]) -> str:
        values = {
            **self.task,
            "a_smiles": str(known["drug"]),
            "b_smiles": str(query["drug"]),
        }
        if self.release == "gold-v1":
            values.update(
                a_context=self._context(known),
                b_context=self._context(query),
                known_value=f"{100.0 * float(known['voter_mean']):.1f}%",
            )
        else:
            values["known_outcome"] = (
                self.task["positive"] if int(known["Y"]) else self.task["negative"]
            )
        return self.environment.get_template(self.template).render(**values).strip()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _prompt_task(task: str, renderer: Any, profile: dict[str, Any], known: dict[str, Any], query: dict[str, Any]) -> PromptTask:
    if isinstance(renderer, V9PromptRenderer):
        prompt = renderer.render(
            {
                "smiles": known["drug"],
                "value": _record_value(known),
                "condition_group": known.get("condition_group"),
                "condition_atoms": known.get("condition_atoms") or [],
            },
            {
                "smiles": query["drug"],
                "condition_group": query.get("condition_group"),
                "condition_atoms": query.get("condition_atoms") or [],
            },
        )
    else:
        prompt = renderer.render(known, query)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    return PromptTask(
        cache_key=_cache_key(prompt_hash, profile, renderer),
        prompt_hash=prompt_hash,
        prompt=prompt,
        task_id=task,
        query_smiles=str(query["drug"]),
        group_id=str(query["benchmark_row_id"]),
        molecule_id=str(known["molecule_identity_key"]),
        record_id=str(known["benchmark_row_id"]),
        model=str(profile["model"]),
        model_revision=str(profile["revision"]),
        scoring_contract_version=SCORING_CONTRACT_VERSION,
        template_hash=renderer.template_hash,
        projection_hash=renderer.projection_hash,
    )


def _rows(task: str, release: str, subset: str) -> tuple[Path, list[dict[str, Any]]]:
    if release == "gold-v1":
        root = Path(__file__).resolve().parents[3] / "data/gold_labels" / GOLD_TASKS[task] / "v1/scaffold"
        path = root / f"{subset}.jsonl"
        rows = read_jsonl(path)
        if subset == "train":
            voters = {
                str(row["benchmark_row_id"]): float(row["voter_mean"])
                for row in pq.read_table(
                    root / "gold_label_record_index.parquet",
                    columns=["benchmark_row_id", "voter_mean", "split"],
                    filters=[("split", "=", "train")],
                ).to_pylist()
            }
            rows = [{**row, "voter_mean": voters[str(row["benchmark_row_id"])]} for row in rows]
        return path, rows
    root = tdc_task_root(task)
    path = root / f"{subset}_molecule_condition_labels.jsonl"
    return path, read_jsonl(path)


def _provenance(task: str, release: str, profile: dict[str, Any]) -> dict[str, Any]:
    if release == "legacy":
        return reference_provenance(task, _lineage(task))
    train_path, _ = _rows(task, release, "train")
    value = {
        "benchmark": release,
        "candidate_source": "frozen_train_contexts",
        "train_sha256": sha256_file(train_path),
        "model": profile,
        "prompt_contract": profile["prompt_profile"],
        "dataset_revision": _json(CONFIG_PATH)["upstream"][task]["dataset_revision"],
    }
    if release == "gold-v1":
        value["gold_record_index_sha256"] = sha256_file(
            train_path.with_name("gold_label_record_index.parquet")
        )
    return value


def prepare(task: str, output_root: Path, release: str = "legacy") -> dict[str, Any]:
    if release == "gold-v1" and task not in GOLD_TASKS or release != "gold-v1" and task not in TDC_TASKS:
        raise ValueError(f"{task} is not supported by {release}")
    lineage, profile_name = _lineage(task), _profile(task, release)
    base_profile = _base_profile(release)
    source = cache_profile_root(base_profile) / task
    target = output_root / task
    if target.exists():
        raise FileExistsError(f"Refusing to replace prepared cache: {target}")
    base_index = _json(source / "RELEASE_INDEX.json")
    if (
        base_index.get("ranking_modes") != ["morgan"]
        or base_index.get("assay_transfer_status") not in {None, "not_computed"}
    ):
        raise ValueError(f"Unexpected base cache contract: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{task}.", dir=target.parent) as temporary:
        root = Path(temporary) / task
        shutil.copytree(source, root)
        train_path, train_rows = _rows(task, release, "train")
        train = {str(row["benchmark_row_id"]): row for row in train_rows}
        renderer = (
            V9PromptRenderer(task)
            if release == "legacy"
            else DirectL1PromptRenderer(task, release)
        )
        profile = _model_profile(task, release)
        prompts: dict[str, dict[str, Any]] = {}
        base_files = {}
        for subset in SUBSETS:
            query_path, query_rows = _rows(task, release, subset)
            queries = {str(row["benchmark_row_id"]): row for row in query_rows}
            level = root / "scaffold" / subset / "L1"
            manifest_path = level / "VERSION.json"
            manifest = _json(manifest_path)
            database = level / str(manifest["database"])
            base_files[subset] = {"manifest_sha256": sha256_file(source / "scaffold" / subset / "L1" / "VERSION.json"), "database_sha256": sha256_file(source / "scaffold" / subset / "L1" / str(manifest["database"]))}
            with sqlite3.connect(database) as connection:
                if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                    raise ValueError(f"Base database failed integrity: {task}/{subset}")
                rows = connection.execute("SELECT benchmark_row_id,item_id,morgan_context_id,assay_transfer_score,assay_rank,assay_context_id,score_key FROM rankings").fetchall()
                if any(any(value is not None for value in row[3:]) for row in rows):
                    raise ValueError(f"Base cache already has assay data: {task}/{subset}")
                updates = []
                for query_id, item_id, context_id, *_ in rows:
                    if str(item_id) != str(train[str(context_id)]["molecule_identity_key"]):
                        raise ValueError("Frozen ranking context no longer matches TDC train")
                    prompt_task = _prompt_task(task, renderer, profile, train[str(context_id)], queries[str(query_id)])
                    prompts.setdefault(prompt_task.cache_key, asdict(prompt_task))
                    updates.append((prompt_task.cache_key, query_id, item_id))
                connection.executemany("UPDATE rankings SET score_key=? WHERE benchmark_row_id=? AND item_id=?", updates)
                connection.commit()
            manifest.update({
                "status": "prepared",
                "model": profile,
                "assay_transfer_status": "prepared",
                "assay_transfer_lineage": lineage if release == "legacy" else release,
                "scoring_contract_version": SCORING_CONTRACT_VERSION,
                "backbone_dtype": BACKBONE_DTYPE,
                "logit_extraction_dtype": LOGIT_EXTRACTION_DTYPE,
                "prompt_assets": {"template_sha256": renderer.template_hash, "projection_sha256": renderer.projection_hash},
                "reference_provenance": _provenance(task, release, profile),
                "base_profile": base_profile,
                "base_manifest_sha256": base_files[subset]["manifest_sha256"],
                "base_database_sha256": base_files[subset]["database_sha256"],
            })
            _write_json(manifest_path, manifest)
        build = root / ".build"
        build.mkdir()
        prompt_path = build / "prompts.parquet"
        pq.write_table(pa.Table.from_pylist([prompts[key] for key in sorted(prompts)]), prompt_path, compression="zstd")
        _write_json(build / "PREPARED.json", {"status": "prepared", "task_id": task, "release": release, "profile": profile_name, "base_profile": base_profile, "prompt_count": len(prompts), "prompts_sha256": sha256_file(prompt_path), "base_files": base_files, "model": profile, "inputs": {"train": str(train_path), "train_sha256": sha256_file(train_path)}})
        index = _json(root / "RELEASE_INDEX.json")
        index.update({"profile": profile_name, "status": "prepared", "assay_transfer_status": "prepared"})
        _write_json(root / "RELEASE_INDEX.json", index)
        os.replace(root, target)
    return {"task": task, "prompt_count": len(prompts), "target": str(target)}


def score(task: str, output_root: Path, shard_index: int, num_shards: int, device: int, batch_size: int) -> dict[str, Any]:
    root = output_root / task
    prepared = _json(root / ".build/PREPARED.json")
    prompt_path = root / ".build/prompts.parquet"
    if sha256_file(prompt_path) != prepared["prompts_sha256"]:
        raise ValueError("Prepared prompt table changed")
    rows = pq.read_table(prompt_path).to_pylist()
    selected = [PromptTask(**row) for index, row in enumerate(rows) if index % num_shards == shard_index]
    score_dir = root / ".build/scores"
    score_dir.mkdir(exist_ok=True)
    journal = score_dir / f"{shard_index:02d}-of-{num_shards:02d}.jsonl"
    completed: list[dict[str, Any]] = []
    if journal.exists():
        completed = read_jsonl(journal)
        expected = [row.cache_key for row in selected[:len(completed)]]
        if [str(row["cache_key"]) for row in completed] != expected:
            raise ValueError("Score journal is not an exact prompt prefix")
    if len(completed) == len(selected):
        return {"task": task, "shard": shard_index, "completed": len(completed)}
    profile = dict(prepared["model"])
    snapshot = resolve_model_snapshot(profile["model"], profile["revision"], local_files_only=True)
    model, tokenizer = load_model(snapshot, device=device)
    with journal.open("a", encoding="utf-8") as handle:
        for offset in range(len(completed), len(selected), batch_size):
            batch = selected[offset:offset + batch_size]
            for result in score_prompt_batch(model, tokenizer, batch, device=device):
                row = asdict(result)
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return {"task": task, "shard": shard_index, "completed": len(selected)}


def finalize(task: str, output_root: Path, num_shards: int) -> dict[str, Any]:
    root = output_root / task
    prepared = _json(root / ".build/PREPARED.json")
    profile_name = str(prepared["profile"])
    prompt_path = root / ".build/prompts.parquet"
    if sha256_file(prompt_path) != prepared["prompts_sha256"]:
        raise ValueError("Prepared prompt table changed")
    expected = {str(row["cache_key"]) for row in pq.read_table(prompt_path, columns=["cache_key"]).to_pylist()}
    scores: dict[str, float] = {}
    for shard in range(num_shards):
        path = root / ".build/scores" / f"{shard:02d}-of-{num_shards:02d}.jsonl"
        if not path.is_file():
            raise ValueError(f"Missing score journal: {path}")
        for row in read_jsonl(path):
            key, value = str(row["cache_key"]), float(row["transfer_probability"])
            if key in scores or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid or duplicate score: {key}")
            scores[key] = value
    if set(scores) != expected:
        raise ValueError(f"Score-key mismatch: missing={len(expected-set(scores))} extra={len(set(scores)-expected)}")
    index = _json(root / "RELEASE_INDEX.json")
    for subset in SUBSETS:
        level = root / "scaffold" / subset / "L1"
        manifest_path = level / "VERSION.json"
        manifest = _json(manifest_path)
        database = level / str(manifest["database"])
        with sqlite3.connect(database) as connection:
            rows = connection.execute("SELECT benchmark_row_id,item_id,score_key FROM rankings").fetchall()
            connection.executemany("UPDATE rankings SET assay_transfer_score=?,assay_context_id=morgan_context_id,assay_member_count=morgan_member_count WHERE benchmark_row_id=? AND item_id=?", [(scores[str(key)], query_id, item_id) for query_id, item_id, key in rows])
            query_ids = [row[0] for row in connection.execute("SELECT benchmark_row_id FROM queries")]
            for query_id in query_ids:
                ordered = connection.execute("SELECT item_id FROM rankings WHERE benchmark_row_id=? ORDER BY assay_transfer_score DESC,item_id", (query_id,)).fetchall()
                connection.executemany("UPDATE rankings SET assay_rank=? WHERE benchmark_row_id=? AND item_id=?", [(rank, query_id, item_id) for rank, (item_id,) in enumerate(ordered, 1)])
            connection.commit()
            query_counts = dict(manifest["query_counts"])
            for counts in query_counts.values():
                counts["assay_candidate_records"] = counts["morgan_candidate_records"]
            identity = {key: value for key, value in manifest.items() if key not in {"schema_version", "status", "content_id", "database", "database_sha256", "ranking_rows_sha256"}}
            identity.update({"query_counts": query_counts, "assay_transfer_status": "complete", "score_count": len(rows)})
            final_manifest = _write_complete(connection, database, task=task, subset=subset, level="L1", identity=identity, target=level)
        index["splits"][subset]["levels"]["L1"].update({"manifest_sha256": sha256_file(manifest_path), "content_id": final_manifest["content_id"]})
    index.update({"profile": profile_name, "base_profile": prepared["base_profile"], "status": "complete", "ranking_modes": ["morgan", "assay-transfer"], "assay_transfer_status": "complete", "model": prepared["model"], "scoring_contract_version": SCORING_CONTRACT_VERSION})
    _write_json(root / "RELEASE_INDEX.json", index)
    report = validate(task, output_root)
    _write_json(root / "VALIDATION.json", report)
    shutil.rmtree(root / ".build")
    return report


def validate(task: str, output_root: Path) -> dict[str, Any]:
    root = output_root / task
    index = _json(root / "RELEASE_INDEX.json")
    base = cache_profile_root(str(index.get("base_profile") or BASE_PROFILE)) / task
    if index.get("ranking_modes") != ["morgan", "assay-transfer"] or index.get("assay_transfer_status") != "complete":
        raise ValueError("Assay-transfer release is not complete")
    result = {"status": "complete", "task_id": task, "profile": index["profile"], "splits": {}}
    immutable = "benchmark_row_id,item_id,parent_id,parent_smiles,morgan_similarity,parent_morgan_rank,within_parent_rank,morgan_rank,morgan_context_id,morgan_member_count"
    for subset in SUBSETS:
        for level_name, entry in index["splits"][subset]["levels"].items():
            if level_name == "L1":
                continue
            relative_manifest = Path(str(entry["manifest"]))
            if (root / relative_manifest).read_bytes() != (base / relative_manifest).read_bytes():
                raise ValueError(f"Frozen {level_name} manifest changed")
            manifest = _json(root / relative_manifest)
            database = relative_manifest.with_name(str(manifest["database"]))
            if (root / database).read_bytes() != (base / database).read_bytes():
                raise ValueError(f"Frozen {level_name} database changed")
        manifest_path = root / "scaffold" / subset / "L1/VERSION.json"
        manifest = _json(manifest_path)
        entry = index["splits"][subset]["levels"]["L1"]
        if sha256_file(manifest_path) != entry["manifest_sha256"]:
            raise ValueError("Release index manifest hash mismatch")
        database = manifest_path.with_name(str(manifest["database"]))
        base_manifest = _json(base / "scaffold" / subset / "L1/VERSION.json")
        base_database = base / "scaffold" / subset / "L1" / str(base_manifest["database"])
        with sqlite3.connect(database) as current, sqlite3.connect(base_database) as original:
            if current.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise ValueError("Ranking database failed integrity")
            if current.execute(f"SELECT {immutable} FROM rankings ORDER BY benchmark_row_id,item_id").fetchall() != original.execute(f"SELECT {immutable} FROM rankings ORDER BY benchmark_row_id,item_id").fetchall():
                raise ValueError("Frozen Morgan candidate universe changed")
            if current.execute("SELECT * FROM queries ORDER BY benchmark_row_id").fetchall() != original.execute("SELECT * FROM queries ORDER BY benchmark_row_id").fetchall() or current.execute("SELECT * FROM contexts ORDER BY context_id").fetchall() != original.execute("SELECT * FROM contexts ORDER BY context_id").fetchall() or current.execute("SELECT * FROM context_records ORDER BY context_id,within_context_rank").fetchall() != original.execute("SELECT * FROM context_records ORDER BY context_id,within_context_rank").fetchall():
                raise ValueError("Frozen query/context ledgers changed")
            bad = current.execute("SELECT COUNT(*) FROM rankings WHERE assay_transfer_score IS NULL OR assay_transfer_score < 0 OR assay_transfer_score > 1 OR assay_rank IS NULL OR assay_context_id != morgan_context_id OR assay_member_count != morgan_member_count OR score_key IS NULL").fetchone()[0]
            rows = current.execute("SELECT COUNT(*) FROM rankings").fetchone()[0]
            query_count = current.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
            ranks = current.execute("SELECT COUNT(*) FROM (SELECT benchmark_row_id FROM rankings GROUP BY benchmark_row_id HAVING MIN(assay_rank)=1 AND MAX(assay_rank)=100 AND COUNT(DISTINCT assay_rank)=100)").fetchone()[0]
            if bad or ranks != query_count:
                raise ValueError("Assay scores or dense ranks are incomplete")
        result["splits"][subset] = {"query_count": query_count, "ranking_rows": rows, "database_sha256": sha256_file(database)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "score", "finalize", "validate"))
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--release", choices=("legacy", "gold-v1", "tdc-v1"), default="legacy")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    args.output_root = args.output_root or cache_profile_root(_profile(args.task, args.release))
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be within --num-shards")
    function = {"prepare": prepare, "score": score, "finalize": finalize, "validate": validate}[args.command]
    if args.command == "score":
        result = function(args.task, args.output_root, args.shard_index, args.num_shards, args.device, args.batch_size)
    elif args.command == "finalize":
        result = function(args.task, args.output_root, args.num_shards)
    else:
        result = (
            function(args.task, args.output_root, args.release)
            if args.command == "prepare"
            else function(args.task, args.output_root)
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
