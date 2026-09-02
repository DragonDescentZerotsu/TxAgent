"""Render V9 prompts and build the strict current-gold Morgan-100 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from rdkit import DataStructs

from predict.retrieval.assay_reranking.runtime import (
    CACHE_ROOT,
    CachedAssayReranker,
    PromptTask,
    SCORING_CONTRACT_VERSION,
    file_sha256,
    load_model,
    model_profile as load_model_profile,
    probability_from_logits,
    resolve_model_snapshot,
    score_prompt_batch,
)
from predict.retrieval.policies import (
    decide_candidate,
    normalize_molecule_identity,
    standardize_smiles_and_fp,
)


ASSET_ROOT = Path(__file__).with_name("prompts") / "v9"
CONFIG_PATH = ASSET_ROOT / "prompt_config.json"
PROFILE_NAME = "v9_direct_gold_top75"
RANKING_PROFILE_NAME = "v9_direct_gold_morgan100"
TEMPLATE_PROFILE = "v9_context_conditioned"
QUERY_CONTEXT_POLICY = "copy_gold_condition_context_value_hidden.v9"
RANKING_SCHEMA_VERSION = "context_conditioned_gold_valid_ranking_cache.v2"
OLD_RANKING_ROOT = Path(
    "/vast/projects/myatskar/design-documents/joseph/therapeutic-tuning/results/"
    "starling_benchmark/2026-08-31/context_conditioned_v1_gold_valid_top75_direct"
)
GOLD_TASK_NAMES = {
    "bbb_martins": "BBB_Martins",
    "bioavailability_ma": "Bioavailability_Ma",
    "skin_reaction": "Skin_Reaction",
}


def model_profile(task_id: str) -> dict[str, Any]:
    return load_model_profile(task_id, "direct")


def default_cache_paths(
    task_id: str, *, split: str = "scaffold", subset: str = "valid"
) -> dict[str, str]:
    model_profile(task_id)
    root = CACHE_ROOT / PROFILE_NAME / task_id / split / subset
    return {
        "catalog": "",
        "candidate_manifest": str(root / "candidates.jsonl"),
        "cache": str(root / "scores.sqlite3"),
        "version": str(root / "VERSION.json"),
    }


def ranking_cache_dir(
    task_id: str,
    *,
    pool_size: int = 100,
    split: str = "scaffold",
    subset: str = "valid",
) -> Path:
    model_profile(task_id)
    if pool_size <= 0:
        raise ValueError("Morgan pool size must be positive")
    profile = f"v9_direct_gold_morgan{pool_size}"
    return CACHE_ROOT / profile / task_id / split / subset


def verify_vendored_assets() -> dict[str, str]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = str(config["_provenance"]["prompt_sha256"])
    observed = file_sha256(ASSET_ROOT / "prompt.jinja")
    if observed != expected:
        raise ValueError(
            "Vendored V9 prompt hash mismatch: "
            + json.dumps({"expected": expected, "observed": observed}, sort_keys=True)
        )
    return {"prompt.jinja": observed, "prompt_config.json": file_sha256(CONFIG_PATH)}


class V9PromptRenderer:
    """Fill one V9 train-neighbor to val/test-query transfer prompt."""

    def __init__(self, task_id: str):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        try:
            self.task = dict(config["tasks"][task_id])
        except KeyError as exc:
            raise ValueError(f"Unknown V9 direct task: {task_id}") from exc
        self.task_id = task_id
        self.template_hash = file_sha256(ASSET_ROOT / "prompt.jinja")
        self.projection_hash = file_sha256(CONFIG_PATH)
        self.environment = Environment(
            loader=FileSystemLoader(str(ASSET_ROOT)),
            undefined=StrictUndefined,
            autoescape=False,
        )

    def render(
        self, known: Mapping[str, Any], query: Mapping[str, Any]
    ) -> str:
        """Render a known train condition and an unlabeled val/test condition."""
        value = float(known["value"])
        if not 0.0 <= value <= 1.0:
            raise ValueError("V9 direct vote-mean values must be between zero and one")
        return self.environment.get_template("prompt.jinja").render(
            **self.task,
            a_smiles=str(known["smiles"]),
            b_smiles=str(query["smiles"]),
            a_context=_context_text(known),
            b_context=_context_text(query),
            known_value=f"{100.0 * value:.1f}%",
        ).strip()


def _context_text(row: Mapping[str, Any]) -> str | None:
    atoms = row.get("condition_atoms") or []
    if not atoms or row.get("condition_group") == "no_reported_external_condition":
        return None
    labels = {
        "barrier_state": "Barrier state",
        "co_treatment": "Co-treatment",
        "disease": "Disease",
        "prandial_state": "Prandial state",
        "release_profile": "Release profile",
    }
    output = []
    for atom in atoms:
        key, value = str(atom).split("=", 1)
        label = labels.get(key, key.replace("_", " ").capitalize())
        output.append(f"- {label}: {value.replace('_', ' ')}")
    return "\n".join(output)


def prefilter_train_neighbors(
    query_rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    *,
    pool_size: int,
    query_smiles_field: str = "drug",
    train_smiles_field: str = "drug",
) -> list[dict[str, Any]]:
    """Return top parent-disjoint train rows for every val/test query."""
    if pool_size <= 0:
        raise ValueError("Morgan pool size must be positive")
    train = _fingerprints(train_rows, train_smiles_field)
    output = []
    for query_index, row in enumerate(query_rows):
        query_smiles = str(row.get(query_smiles_field) or "")
        _, _, query_fp = standardize_smiles_and_fp(query_smiles)
        if query_fp is None:
            raise ValueError(f"Query row {query_index} has invalid SMILES")
        similarities = DataStructs.BulkTanimotoSimilarity(
            query_fp, [item["fingerprint"] for item in train]
        )
        ranked = sorted(
            zip(similarities, train),
            key=lambda item: (-float(item[0]), item[1]["index"]),
        )
        candidates = []
        for similarity, item in ranked:
            decision = decide_candidate(
                normalize_molecule_identity(query_smiles),
                {"canonical_smiles": item["smiles"]},
                "parent_disjoint",
            )
            if decision.excluded:
                continue
            candidates.append(
                {
                    "train_index": item["index"],
                    "train_smiles": item["smiles"],
                    "similarity": round(float(similarity), 8),
                }
            )
            if len(candidates) == pool_size:
                break
        output.append(
            {
                "query_index": query_index,
                "query_smiles": query_smiles,
                "candidates": candidates,
            }
        )
    return output


def _fingerprints(
    rows: Sequence[Mapping[str, Any]], smiles_field: str
) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(rows):
        smiles = str(row.get(smiles_field) or "")
        canonical, _, fingerprint = standardize_smiles_and_fp(smiles)
        if fingerprint is None:
            raise ValueError(f"Train row {index} has invalid SMILES")
        output.append({"index": index, "smiles": canonical, "fingerprint": fingerprint})
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(path)


def _gold_paths(task_id: str) -> tuple[Path, Path]:
    try:
        task_name = GOLD_TASK_NAMES[task_id]
    except KeyError as exc:
        raise ValueError(f"Unknown V9 direct task: {task_id}") from exc
    root = Path("data/gold_labels") / task_name / "v1" / "scaffold"
    return (
        root / "train_molecule_condition_labels.jsonl",
        root / "valid_molecule_condition_labels.jsonl",
    )


def _record_value(row: Mapping[str, Any]) -> float:
    counts = row.get("label_counts") or {}
    positive = int(counts.get("1", 0))
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        raise ValueError(f"Gold row has no label votes: {row.get('benchmark_row_id')}")
    return positive / total


def _cache_key(prompt_hash: str, profile: Mapping[str, Any], renderer: Any) -> str:
    identity = {
        "prompt_hash": prompt_hash,
        "model": profile["model"],
        "model_revision": profile["revision"],
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "template_hash": renderer.template_hash,
        "projection_hash": renderer.projection_hash,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _current_candidates(
    task_id: str, pool_size: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build top-N distinct train parents, then expand their condition rows."""
    from rdkit import DataStructs

    train_path, query_path = _gold_paths(task_id)
    train_rows, query_rows = _read_jsonl(train_path), _read_jsonl(query_path)
    renderer, profile = V9PromptRenderer(task_id), model_profile(task_id)
    parent_rows: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    parent_fps: dict[str, Any] = {}
    parent_first: dict[str, int] = {}
    for index, row in enumerate(train_rows):
        parent = str(row["molecule_identity_key"])
        canonical, _, fingerprint = standardize_smiles_and_fp(str(row["drug"]))
        if fingerprint is None:
            raise ValueError(f"Invalid training SMILES at row {index}")
        parent_rows.setdefault(parent, []).append((index, row))
        parent_fps.setdefault(parent, fingerprint)
        parent_first.setdefault(parent, index)

    parents = sorted(parent_rows, key=parent_first.__getitem__)
    fingerprints = [parent_fps[parent] for parent in parents]
    output: list[dict[str, Any]] = []
    for query_index, query in enumerate(query_rows):
        query_parent = str(query["molecule_identity_key"])
        _, _, query_fp = standardize_smiles_and_fp(str(query["drug"]))
        if query_fp is None:
            raise ValueError(f"Invalid query SMILES at row {query_index}")
        similarities = DataStructs.BulkTanimotoSimilarity(query_fp, fingerprints)
        ranked = sorted(
            (
                (float(similarity), parent_first[parent], parent)
                for similarity, parent in zip(similarities, parents)
                if parent != query_parent
            ),
            key=lambda item: (-item[0], item[1]),
        )[:pool_size]
        if len(ranked) != pool_size:
            raise ValueError(
                f"{task_id} query {query['benchmark_row_id']} has only {len(ranked)} parents"
            )
        for parent_rank, (similarity, _, parent) in enumerate(ranked):
            contexts = parent_rows[parent]
            for context_index, (_, known) in enumerate(contexts):
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
                prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
                output.append({
                    "task_id": GOLD_TASK_NAMES[task_id],
                    "query_record_id": str(query["benchmark_row_id"]),
                    "retrieval_record_id": str(known["benchmark_row_id"]),
                    "query_molecule_identity_key": query_parent,
                    "retrieval_molecule_identity_key": parent,
                    "query_smiles": str(query["drug"]),
                    "retrieval_smiles": str(known["drug"]),
                    "query_condition_group": str(query.get("condition_group") or ""),
                    "retrieval_condition_group": str(known.get("condition_group") or ""),
                    "query_condition_scope": str(query.get("condition_scope") or ""),
                    "retrieval_condition_scope": str(known.get("condition_scope") or ""),
                    "query_condition_atoms": list(query.get("condition_atoms") or []),
                    "retrieval_condition_atoms": list(known.get("condition_atoms") or []),
                    "query_source_record_count": int(query.get("source_record_count") or 0),
                    "retrieval_source_record_count": int(known.get("source_record_count") or 0),
                    "query_agreement_fraction": float(query.get("agreement_fraction") or 0),
                    "retrieval_agreement_fraction": float(known.get("agreement_fraction") or 0),
                    "query_gold_Y": int(query["Y"]),
                    "retrieval_gold_Y": int(known["Y"]),
                    "morgan_tanimoto_similarity": similarity,
                    "ranking_candidate_parent_count": pool_size,
                    "retrieval_parent_rank": parent_rank,
                    "retrieval_parent_context_index": context_index,
                    "retrieval_parent_context_count": len(contexts),
                    "prompt": prompt,
                    "prompt_hash": prompt_hash,
                    "cache_key": _cache_key(prompt_hash, profile, renderer),
                    "model_score": None,
                    "prob_transfer": None,
                    "score_origin": None,
                })
    return output, {
        "train": str(train_path.resolve()),
        "train_sha256": file_sha256(train_path),
        "valid": str(query_path.resolve()),
        "valid_sha256": file_sha256(query_path),
        "n_train_rows": len(train_rows),
        "n_train_parents": len(parent_rows),
        "n_queries": len(query_rows),
    }


def _validated_reuse(
    task_id: str, reuse_root: Path
) -> tuple[dict[tuple[str, str, str], tuple[float, float]], dict[str, Any]]:
    import pyarrow.parquet as pq

    root = reuse_root / task_id
    version_path = root / "scaffold" / "valid" / "VERSION.json"
    if version_path.is_file():
        root = version_path.parent
        manifest_path = version_path
    else:
        manifest_path = root / "manifest.json"
    rankings_path = root / "rankings.parquet"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = model_profile(task_id)
    expected_models = (
        profile,
        {"id": profile["model"], "revision": profile["revision"]},
    )
    if manifest.get("status") != "complete" or manifest.get("model") not in expected_models:
        raise ValueError(f"Old {task_id} cache is incomplete or uses another model")
    expected_hash = manifest.get("rankings_sha256") or (
        manifest.get("output") or {}
    ).get("rankings_sha256")
    if expected_hash != file_sha256(rankings_path):
        raise ValueError(f"Old {task_id} rankings hash mismatch")
    schema = str(manifest.get("schema_version") or "")
    if schema == RANKING_SCHEMA_VERSION:
        if manifest.get("prompt_assets") != verify_vendored_assets():
            raise ValueError(f"Reusable {task_id} cache uses different prompt assets")
        prompt_column = "prompt_hash"
    elif schema == "context_conditioned_gold_valid_ranking_cache.v1":
        prompt_column = "prompt"
    else:
        raise ValueError(f"Unsupported reusable V9 cache schema: {schema}")
    rows = pq.read_table(
        rankings_path,
        columns=[
            "query_record_id", "retrieval_record_id", prompt_column,
            "model_score", "prob_transfer",
        ],
    ).to_pylist()
    reuse: dict[tuple[str, str, str], tuple[float, float]] = {}
    for row in rows:
        score, probability = float(row["model_score"]), float(row["prob_transfer"])
        if abs(probability_from_logits(score, 0.0) - probability) > 1e-8:
            raise ValueError("Old V9 probability does not match its score margin")
        prompt_hash = (
            str(row["prompt_hash"])
            if prompt_column == "prompt_hash"
            else hashlib.sha256(str(row["prompt"]).encode()).hexdigest()
        )
        key = (str(row["query_record_id"]), str(row["retrieval_record_id"]), prompt_hash)
        if key in reuse and reuse[key] != (score, probability):
            raise ValueError(f"Conflicting reusable V9 score: {key[:2]}")
        reuse[key] = (score, probability)
    return reuse, {
        "root": str(root),
        "manifest_sha256": file_sha256(manifest_path),
        "rankings_sha256": file_sha256(rankings_path),
        "n_rows": len(rows),
        "schema_version": schema,
        "score_origin": "exact_cache_reuse",
    }


def prepare_ranking_cache(
    task_id: str, *, pool_size: int = 100, reuse_root: Path = OLD_RANKING_ROOT
) -> dict[str, Any]:
    """Create the fresh candidate universe and seed only exact old scores."""
    root = ranking_cache_dir(task_id, pool_size=pool_size)
    build_dir = root / ".build"
    if root.joinpath("VERSION.json").exists():
        raise FileExistsError(f"Finalized V9 cache already exists: {root}")
    if build_dir.exists() and any(build_dir.glob("scores-*.jsonl")):
        raise ValueError("Cannot replace candidates after scoring has started")
    candidates, inputs = _current_candidates(task_id, pool_size)
    reuse, reused_source = _validated_reuse(task_id, reuse_root)
    reused = 0
    for row in candidates:
        key = (row["query_record_id"], row["retrieval_record_id"], row["prompt_hash"])
        if key in reuse:
            row["model_score"], row["prob_transfer"] = reuse[key]
            row["score_origin"] = reused_source["score_origin"]
            reused += 1
    candidates_path = build_dir / "candidates.parquet"
    _write_parquet(candidates_path, candidates)
    receipt = {
        "schema_version": RANKING_SCHEMA_VERSION,
        "status": "prepared",
        "task_id": task_id,
        "model": model_profile(task_id),
        "prompt_assets": verify_vendored_assets(),
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "candidate_policy": (
            f"morgan_top{pool_size}_distinct_gold_train_parents_then_all_context_rows"
        ),
        "morgan_pool_size": pool_size,
        "inputs": inputs,
        "reuse_source": reused_source,
        "n_candidates": len(candidates),
        "n_exact_reused": reused,
        "n_to_score": len(candidates) - reused,
        "candidates_sha256": file_sha256(candidates_path),
    }
    _write_json(build_dir / "BUILD.json", receipt)
    return receipt


def _missing_tasks(
    task_id: str, pool_size: int
) -> tuple[list[PromptTask], dict[str, Any]]:
    import pyarrow.parquet as pq

    root = ranking_cache_dir(task_id, pool_size=pool_size)
    build = json.loads((root / ".build/BUILD.json").read_text(encoding="utf-8"))
    candidates_path = root / ".build/candidates.parquet"
    if build["candidates_sha256"] != file_sha256(candidates_path):
        raise ValueError("Prepared V9 candidates hash mismatch")
    profile, renderer = model_profile(task_id), V9PromptRenderer(task_id)
    by_key: dict[str, PromptTask] = {}
    for row in pq.read_table(candidates_path).to_pylist():
        if row["model_score"] is not None:
            continue
        task = PromptTask(
            cache_key=str(row["cache_key"]), prompt_hash=str(row["prompt_hash"]),
            prompt=str(row["prompt"]), task_id=task_id,
            query_smiles=str(row["query_smiles"]), group_id="direct_gold",
            molecule_id=str(row["retrieval_molecule_identity_key"]),
            record_id=str(row["retrieval_record_id"]), model=str(profile["model"]),
            model_revision=str(profile["revision"]),
            scoring_contract_version=SCORING_CONTRACT_VERSION,
            template_hash=renderer.template_hash, projection_hash=renderer.projection_hash,
        )
        previous = by_key.setdefault(task.cache_key, task)
        if previous.prompt != task.prompt:
            raise ValueError("V9 cache-key collision")
    return sorted(by_key.values(), key=lambda item: item.cache_key), build


def score_ranking_cache(
    task_id: str,
    *,
    pool_size: int = 100,
    shard_index: int,
    num_shards: int,
    batch_size: int = 64,
    device: int = 0,
) -> dict[str, Any]:
    """Resume one deterministic GPU shard of missing prompt scores."""
    if not 0 <= shard_index < num_shards or batch_size <= 0:
        raise ValueError("Invalid shard or batch size")
    tasks, build = _missing_tasks(task_id, pool_size)
    shard = tasks[shard_index::num_shards]
    journal = (
        ranking_cache_dir(task_id, pool_size=pool_size)
        / ".build"
        / f"scores-{shard_index:02d}-of-{num_shards:02d}.jsonl"
    )
    completed = _read_jsonl(journal) if journal.exists() else []
    if [row["cache_key"] for row in completed] != [
        task.cache_key for task in shard[: len(completed)]
    ]:
        raise ValueError(f"Score journal is not an exact shard prefix: {journal}")
    if len(completed) == len(shard):
        return {
            "status": "complete",
            "task_id": task_id,
            "scored": len(shard),
            "journal": str(journal),
        }
    profile = build["model"]
    snapshot = resolve_model_snapshot(profile["model"], profile["revision"], local_files_only=True)
    model, tokenizer = load_model(snapshot, device=device)
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8") as handle:
        for offset in range(len(completed), len(shard), batch_size):
            batch = shard[offset:offset + batch_size]
            for score in score_prompt_batch(model, tokenizer, batch, device=device):
                row = {
                    "cache_key": score.cache_key,
                    "model_score": score.logp_transfer - score.logp_not_transfer,
                    "prob_transfer": score.transfer_probability,
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(
                f"{task_id} shard {shard_index + 1}/{num_shards}: "
                f"{min(offset + len(batch), len(shard))}/{len(shard)}",
                flush=True,
            )
    return {"status": "complete", "task_id": task_id, "scored": len(shard), "journal": str(journal)}


def finalize_ranking_cache(
    task_id: str, *, pool_size: int = 100, num_shards: int
) -> dict[str, Any]:
    """Merge complete journals, publish atomically, then remove build files."""
    import pyarrow.parquet as pq

    root = ranking_cache_dir(task_id, pool_size=pool_size)
    tasks_and_build = _missing_tasks(task_id, pool_size)
    tasks, build = tasks_and_build
    scored: dict[str, tuple[float, float]] = {}
    for shard_index in range(num_shards):
        expected = tasks[shard_index::num_shards]
        path = root / ".build" / f"scores-{shard_index:02d}-of-{num_shards:02d}.jsonl"
        rows = _read_jsonl(path) if path.exists() else []
        if [row["cache_key"] for row in rows] != [task.cache_key for task in expected]:
            raise ValueError(f"Incomplete or mismatched score journal: {path}")
        scored.update({
            row["cache_key"]: (
                float(row["model_score"]),
                float(row["prob_transfer"]),
            )
            for row in rows
        })
    rows = pq.read_table(root / ".build/candidates.parquet").to_pylist()
    for row in rows:
        if row["model_score"] is None:
            row["model_score"], row["prob_transfer"] = scored[row["cache_key"]]
            row["score_origin"] = "fresh_v9_inference"
    rows.sort(key=lambda row: (
        row["query_record_id"], -float(row["model_score"]),
        int(row["retrieval_parent_rank"]), int(row["retrieval_parent_context_index"]),
        row["retrieval_record_id"],
    ))
    previous, rank = None, 0
    for row in rows:
        if row["query_record_id"] != previous:
            previous, rank = row["query_record_id"], 0
        row["model_rank"] = rank
        rank += 1
        row.pop("prompt")
        row.pop("cache_key")
    rankings_path = root / "rankings.parquet"
    _write_parquet(rankings_path, rows)
    train_path, _ = _gold_paths(task_id)
    training_record_ids = sorted(
        str(row["benchmark_row_id"]) for row in _read_jsonl(train_path)
    )
    version = {
        **build,
        "status": "complete",
        "n_fresh_scores": len(scored),
        "n_final_rows": len(rows),
        "n_final_queries": len({row["query_record_id"] for row in rows}),
        "query_record_ids": sorted({row["query_record_id"] for row in rows}),
        "training_record_ids": training_record_ids,
        "retrieval_record_ids": sorted({row["retrieval_record_id"] for row in rows}),
        "rankings": str(rankings_path.resolve()),
        "rankings_sha256": file_sha256(rankings_path),
    }
    _write_json(root / "VERSION.json", version)
    for path in sorted((root / ".build").iterdir()):
        path.unlink()
    (root / ".build").rmdir()
    return version


class V9CachedAssayReranker(CachedAssayReranker):
    """Read a finalized V9 direct score cache; never score during prediction."""

    def __init__(
        self,
        *,
        task_id: str,
        cache_path: str | Path,
        model: str | None = None,
        model_revision: str | None = None,
        allow_missing: bool = False,
        cache_mode: str = "read_only",
        **kwargs: Any,
    ):
        if cache_mode != "read_only":
            raise ValueError("V9 inference only accepts a finalized read-only cache")
        profile = model_profile(task_id)
        super().__init__(
            task_id=task_id,
            cache_path=cache_path,
            model=model or str(profile["model"]),
            model_revision=model_revision or str(profile["revision"]),
            renderer=V9PromptRenderer(task_id),
            profile_name=PROFILE_NAME,
            template_profile=TEMPLATE_PROFILE,
            query_context_policy=QUERY_CONTEXT_POLICY,
            allow_missing=allow_missing,
            **kwargs,
        )


def preflight_cache_coverage(
    *,
    records: list[dict[str, Any]],
    indices: list[int],
    smiles_field: str,
    cache_path: str,
    model: str,
    model_revision: str,
    task_id: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Fail unless a finalized cache contains assignments for every requested query."""
    reranker = V9CachedAssayReranker(
        task_id=task_id,
        cache_path=cache_path,
        model=model,
        model_revision=model_revision,
        **kwargs,
    )
    try:
        queries = {
            str(records[index].get(smiles_field) or "").strip()
            for index in indices
        }
        queries.discard("")
        placeholders = ",".join("?" for _ in queries)
        found = set()
        if queries:
            rows = reranker.cache.connection.execute(
                f"SELECT query_smiles FROM queries WHERE query_smiles IN ({placeholders})",
                sorted(queries),
            )
            found = {str(row[0]) for row in rows}
        missing = sorted(queries - found)
        if missing:
            raise ValueError(
                f"V9 cache is missing {len(missing)} of {len(queries)} requested queries"
            )
        if not reranker.cache.metadata.get("candidate_contract"):
            raise ValueError("V9 cache metadata has no frozen candidate contract")
        score_count = int(
            reranker.cache.connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        )
        return {
            "status": "pass",
            "n_queries": len(queries),
            "n_cache_scores": score_count,
            "provenance": reranker.provenance(),
        }
    finally:
        reranker.cache.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--task", choices=sorted(GOLD_TASK_NAMES), required=True)
    prepare.add_argument("--pool-size", type=int, default=100)
    prepare.add_argument(
        "--reuse-root", "--old-root", dest="reuse_root", type=Path,
        default=OLD_RANKING_ROOT,
    )
    score = subparsers.add_parser("score")
    score.add_argument("--task", choices=sorted(GOLD_TASK_NAMES), required=True)
    score.add_argument("--pool-size", type=int, default=100)
    score.add_argument("--shard-index", type=int, required=True)
    score.add_argument("--num-shards", type=int, required=True)
    score.add_argument("--batch-size", type=int, default=64)
    score.add_argument("--device", type=int, default=0)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--task", choices=sorted(GOLD_TASK_NAMES), required=True)
    finalize.add_argument("--pool-size", type=int, default=100)
    finalize.add_argument("--num-shards", type=int, required=True)
    download = subparsers.add_parser("download")
    download.add_argument("--task", choices=sorted(GOLD_TASK_NAMES), required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_ranking_cache(
            args.task, pool_size=args.pool_size, reuse_root=args.reuse_root
        )
    elif args.command == "score":
        result = score_ranking_cache(
            args.task, pool_size=args.pool_size,
            shard_index=args.shard_index, num_shards=args.num_shards,
            batch_size=args.batch_size, device=args.device,
        )
    elif args.command == "finalize":
        result = finalize_ranking_cache(
            args.task, pool_size=args.pool_size, num_shards=args.num_shards
        )
    else:
        profile = model_profile(args.task)
        result = {
            "task_id": args.task,
            "snapshot": resolve_model_snapshot(profile["model"], profile["revision"]),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
