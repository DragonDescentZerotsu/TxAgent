"""Precompute cached assay-transfer likelihoods with one full model replica per GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import queue
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

from tools.chembl_tool.common.experiment_retrieval import retrieve_experiment_view
from tools.chembl_tool.tasks.bioavailability_ma.reranking.assay_transfer_rerank import (
    ASSAY_TRANSFER_MODEL,
    ASSAY_TRANSFER_MODEL_REVISION,
    DEFAULT_TEMPLATE_PROFILE,
    DEFAULT_CACHE,
    DEFAULT_CATALOG,
    TEMPLATE_PROFILES,
    V6_5_NO_QUERY_EXTRA_DETAILS_TEMPLATE_PROFILE,
    V6_5_TEMPLATE_PROFILE,
    AssayTransferCachedReranker,
    PromptScore,
    PromptTask,
    probability_from_log_likelihoods,
    prompt_task_from_dict,
    prompt_task_to_dict,
    require_immutable_revision,
)
from tools.chembl_tool.tasks.bioavailability_ma.reranking.build_assay_transfer_rerank_catalog import (
    DEFAULT_CANDIDATE_MANIFEST,
    build_candidate_scoped_catalog,
    build_manifest_for_prebuilt_catalog,
)
from tools.chembl_tool.tasks.bioavailability_ma.experiment_config import get_source_config
from tools.chembl_tool.tasks.bioavailability_ma.retrieve_neighbors import load_index


DEFAULT_INPUT = "data/processed/Bioavailability_Ma/valid.jsonl"
DEFAULT_INDEX = (
    "outputs/paper/molecular_evidence_agent/evidence/"
    "bioavailability_starling_five_source_full/starling_factor_neighbor_index.pkl"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.reuse_prebuilt_catalog and args.reuse_frozen_flat_artifacts:
        raise SystemExit(
            "--reuse-prebuilt-catalog and --reuse-frozen-flat-artifacts are mutually exclusive"
        )
    if args.retrieval_source == "starling_in_distribution":
        if not args.reuse_prebuilt_catalog:
            raise SystemExit(
                "starling_in_distribution requires --reuse-prebuilt-catalog"
            )
        if args.assay_transfer_template_profile not in {
            V6_5_TEMPLATE_PROFILE,
            V6_5_NO_QUERY_EXTRA_DETAILS_TEMPLATE_PROFILE,
        }:
            raise SystemExit(
                "starling_in_distribution requires a v6.5 query-context-copy template profile"
            )
    if not args.reuse_frozen_flat_artifacts and _catalog_is_flat(Path(args.rerank_catalog)):
        raise SystemExit(
            "The supplied flat catalog and condition manifest are immutable. Pass "
            "--reuse-frozen-flat-artifacts to score missing condition prompts without rebuilding them, "
            "or use precompute_flat_assay_transfer_cache --prepare/--infer for the full cache."
        )
    if args.rerank_workers_per_device != 1:
        raise SystemExit("--rerank-workers-per-device is currently fixed to 1")
    devices = _parse_devices(args.rerank_devices)

    records = _read_jsonl(Path(args.input_jsonl))
    indices = _select_indices(args, len(records))
    index = load_index(Path(args.index))
    if args.reuse_frozen_flat_artifacts:
        if not Path(args.rerank_catalog).is_file() or not Path(args.candidate_manifest).is_file():
            raise SystemExit("Frozen flat catalog and candidate manifest must both exist")
        frozen = {
            "catalog": args.rerank_catalog,
            "candidate_manifest": args.candidate_manifest,
        }
        print("[assay_transfer_precompute] reusing immutable flat catalog and candidate manifest", file=sys.stderr, flush=True)
    elif args.reuse_prebuilt_catalog:
        if not Path(args.rerank_catalog).is_file():
            raise SystemExit("--reuse-prebuilt-catalog requires an existing catalog")
        frozen = build_manifest_for_prebuilt_catalog(
            records=records,
            indices=indices,
            smiles_field=args.smiles_field,
            index=index,
            catalog_path=Path(args.rerank_catalog),
            manifest_output=Path(args.candidate_manifest),
            experiment_mode=args.experiment_mode,
            top_k_per_group=args.top_k_per_group,
            min_similarity=args.min_similarity,
            neighbor_identity_policy=args.neighbor_identity_policy,
            initial_morgan_filter=args.assay_transfer_initial_morgan_filter,
            index_path=args.index,
            condition_id=args.condition_id,
            template_profile=args.assay_transfer_template_profile,
        )
        print(
            f"[assay_transfer_precompute] froze prebuilt-catalog manifest "
            f"queries={frozen['n_queries']} groups={frozen['n_groups']} "
            f"candidates={frozen['n_candidates']}",
            file=sys.stderr,
            flush=True,
        )
    else:
        frozen = build_candidate_scoped_catalog(
            records=records,
            indices=indices,
            smiles_field=args.smiles_field,
            index=index,
            output=Path(args.rerank_catalog),
            manifest_output=Path(args.candidate_manifest),
            experiment_mode=args.experiment_mode,
            top_k_per_group=args.top_k_per_group,
            min_similarity=args.min_similarity,
            neighbor_identity_policy=args.neighbor_identity_policy,
            initial_morgan_filter=args.assay_transfer_initial_morgan_filter,
            index_path=args.index,
            condition_id=args.condition_id,
        )
        print(
            f"[assay_transfer_precompute] froze candidates queries={frozen['n_queries']} "
            f"groups={frozen['n_candidate_groups']} records={frozen['n_records']}",
            file=sys.stderr,
            flush=True,
        )
    if args.prepare_only:
        print(json.dumps({"status": "prepared", **frozen}, indent=2))
        return 0
    _check_free_vram(devices, args.rerank_min_free_vram_gib)
    snapshot_path, immutable_revision = resolve_model_snapshot(
        args.assay_transfer_model,
        args.assay_transfer_model_revision,
        force_download=args.force_model_download,
        local_files_only=args.local_files_only,
    )
    args.assay_transfer_model_revision = immutable_revision

    reranker, tasks = collect_prompt_tasks(
        records=records,
        indices=indices,
        smiles_field=args.smiles_field,
        index_path=args.index,
        catalog_path=args.rerank_catalog,
        candidate_manifest_path=args.candidate_manifest,
        cache_path=args.rerank_cache,
        model=args.assay_transfer_model,
        model_revision=immutable_revision,
        experiment_mode=args.experiment_mode,
        top_k_per_group=args.top_k_per_group,
        min_similarity=args.min_similarity,
        neighbor_identity_policy=args.neighbor_identity_policy,
        initial_morgan_filter=args.assay_transfer_initial_morgan_filter,
        force_rescore=args.force_rescore,
        template_profile=args.assay_transfer_template_profile,
        retrieval_source=args.retrieval_source,
    )
    print(
        f"[assay_transfer_precompute] queries={len(indices)} prompts_to_score={len(tasks)} "
        f"devices={devices} batch_size_per_gpu={args.rerank_batch_size}",
        file=sys.stderr,
        flush=True,
    )
    if not tasks:
        payload = {
            "status": "complete",
            "n_scored": 0,
            "n_prompt_task_references": reranker.prompt_task_reference_count,
            "provenance": reranker.provenance(),
        }
        _write_version_manifest(args, reranker, payload, len(indices))
        print(json.dumps(payload, indent=2))
        reranker.cache.close()
        return 0

    try:
        summary = run_spawned_workers(
            tasks,
            cache=reranker.cache,
            snapshot_path=snapshot_path,
            devices=devices,
            batch_size=args.rerank_batch_size,
            dtype=args.rerank_dtype,
        )
        payload = {
            "status": "complete",
            **summary,
            "n_prompt_task_references": reranker.prompt_task_reference_count,
            "provenance": reranker.provenance(),
        }
        _write_version_manifest(args, reranker, payload, len(indices))
    finally:
        reranker.cache.close()
    print(json.dumps(payload, indent=2))
    return 0


def _write_version_manifest(
    args: argparse.Namespace,
    reranker: AssayTransferCachedReranker,
    summary: dict[str, Any],
    n_queries: int,
) -> None:
    if not args.cache_version_manifest:
        return
    provenance = reranker.provenance()
    n_scores = int(
        reranker.cache.connection.execute("SELECT COUNT(*) FROM prompt_scores").fetchone()[0]
    )
    payload = {
        "status": "complete",
        **provenance,
        "n_queries": n_queries,
        "n_prompt_scores": n_scores,
        "n_prompt_task_references": int(summary.get("n_prompt_task_references") or 0),
        "input_jsonl": args.input_jsonl,
        "input_sha256": _file_sha256(Path(args.input_jsonl)),
        "index": args.index,
        "index_sha256": _file_sha256(Path(args.index)),
        "catalog": args.rerank_catalog,
        "catalog_sha256": _file_sha256(Path(args.rerank_catalog)),
        "candidate_manifest": args.candidate_manifest,
        "retrieval_source": args.retrieval_source,
        "experiment_mode": args.experiment_mode,
        "neighbor_identity_policy": args.neighbor_identity_policy,
        "min_similarity": args.min_similarity,
        "assay_transfer_initial_morgan_filter": args.assay_transfer_initial_morgan_filter,
        "devices": _parse_devices(args.rerank_devices),
        "dtype": args.rerank_dtype,
        "batch_size_per_gpu": args.rerank_batch_size,
    }
    output = Path(args.cache_version_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_prompt_tasks(
    *,
    records: list[dict[str, Any]],
    indices: list[int],
    smiles_field: str,
    index_path: str,
    catalog_path: str,
    candidate_manifest_path: str | None,
    cache_path: str,
    model: str,
    model_revision: str,
    experiment_mode: str,
    top_k_per_group: int,
    min_similarity: float,
    neighbor_identity_policy: str,
    initial_morgan_filter: int,
    force_rescore: bool,
    template_profile: str = DEFAULT_TEMPLATE_PROFILE,
    retrieval_source: str = "starling",
) -> tuple[AssayTransferCachedReranker, list[PromptTask]]:
    if experiment_mode not in {"direct", "full_flat", "full_mechanism"}:
        raise ValueError("Assay-transfer reranking requires direct, full_flat, or full_mechanism mode")
    index = load_index(Path(index_path))
    config = get_source_config(retrieval_source)
    reranker = AssayTransferCachedReranker(
        catalog_path=catalog_path,
        cache_path=cache_path,
        cache_mode="read_write",
        model=model,
        model_revision=model_revision,
        allow_missing=True,
        candidate_manifest_path=candidate_manifest_path,
        template_profile=template_profile,
    )
    for ordinal, index_value in enumerate(indices, start=1):
        query_smiles = str(records[index_value].get(smiles_field) or "")
        if not query_smiles:
            raise ValueError(f"Input record {index_value} has no `{smiles_field}` value")
        retrieve_experiment_view(
            query_smiles,
            index,
            mode=experiment_mode,
            config=config,
            top_k_per_group=top_k_per_group,
            min_similarity=min_similarity,
            neighbor_identity_policy=neighbor_identity_policy,
            reranker=reranker,
            assay_transfer_initial_morgan_filter=initial_morgan_filter,
        )
        if ordinal % 25 == 0:
            print(
                f"[assay_transfer_precompute] prepared {ordinal}/{len(indices)} queries; "
                f"unique_prompts={len(reranker.seen_tasks)} misses={len(reranker.missing_tasks)}",
                file=sys.stderr,
                flush=True,
            )
    selected = reranker.seen_tasks if force_rescore else reranker.missing_tasks
    return reranker, [selected[key] for key in sorted(selected)]


def preflight_cache_coverage(
    *,
    records: list[dict[str, Any]],
    indices: list[int],
    smiles_field: str,
    index_path: str,
    catalog_path: str,
    candidate_manifest_path: str | None = None,
    cache_path: str,
    model: str,
    model_revision: str,
    experiment_mode: str,
    top_k_per_group: int,
    min_similarity: float,
    neighbor_identity_policy: str,
    initial_morgan_filter: int,
    require_selected_scores: bool = False,
    template_profile: str = DEFAULT_TEMPLATE_PROFILE,
    expected_score_count: int = 0,
    cache_version_path: str = "",
    retrieval_source: str = "starling",
    assay_transfer_min_score: float | None = None,
) -> dict[str, Any]:
    from tools.chembl_tool.tasks.bioavailability_ma.reranking.assay_transfer_prompt_policy import (
        prepare_assay_transfer_selected_neighbors,
    )

    index = load_index(Path(index_path))
    config = get_source_config(retrieval_source)
    reranker = AssayTransferCachedReranker(
        catalog_path=catalog_path,
        cache_path=cache_path,
        cache_mode="read_only",
        model=model,
        model_revision=model_revision,
        candidate_manifest_path=candidate_manifest_path,
        template_profile=template_profile,
    )
    n_unscoreable_selected_dropped = 0
    n_below_min_score_dropped = 0
    n_scoreable_selected = 0
    try:
        for index_value in indices:
            query_smiles = str(records[index_value].get(smiles_field) or "")
            retrieval = retrieve_experiment_view(
                query_smiles,
                index,
                mode=experiment_mode,
                config=config,
                top_k_per_group=top_k_per_group,
                min_similarity=min_similarity,
                neighbor_identity_policy=neighbor_identity_policy,
                reranker=reranker,
                assay_transfer_initial_morgan_filter=initial_morgan_filter,
                assay_transfer_min_score=assay_transfer_min_score,
            )
            prepare_assay_transfer_selected_neighbors(
                retrieval, expose_scores=require_selected_scores
            )
            coverage = retrieval.get("coverage") or {}
            n_unscoreable_selected_dropped += int(
                coverage.get("n_unscoreable_selected_dropped") or 0
            )
            n_below_min_score_dropped += int(
                coverage.get("n_below_assay_transfer_min_score_dropped") or 0
            )
            n_scoreable_selected += int(coverage.get("n_neighbors_total") or 0)
        quick_check = str(reranker.cache.connection.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check != "ok":
            raise ValueError(f"Assay-transfer cache quick_check failed: {quick_check}")
        total_cache_rows = int(
            reranker.cache.connection.execute("SELECT COUNT(*) FROM prompt_scores").fetchone()[0]
        )
        if expected_score_count:
            if len(reranker.seen_tasks) != expected_score_count:
                raise ValueError(
                    "Assay-transfer prompt demand count mismatch: "
                    f"expected={expected_score_count}, observed={len(reranker.seen_tasks)}"
                )
            if total_cache_rows != expected_score_count:
                raise ValueError(
                    "Assay-transfer cache row count mismatch: "
                    f"expected={expected_score_count}, observed={total_cache_rows}"
                )
        version_validation = _validate_cache_version_manifest(
            path=cache_version_path,
            provenance=reranker.provenance(),
            total_cache_rows=total_cache_rows,
            expected_score_count=expected_score_count,
        )
        result = {
            "status": "complete",
            "n_queries": len(indices),
            "n_unique_prompt_scores": len(reranker.seen_tasks),
            "n_prompt_task_references": reranker.prompt_task_reference_count,
            "cache_quick_check": quick_check,
            "n_cache_rows": total_cache_rows,
            "cache_version_validation": version_validation,
            "provenance": reranker.provenance(),
            "assay_transfer_min_score": assay_transfer_min_score,
        }
        result.update(
            {
                "n_scoreable_selected": n_scoreable_selected,
                "n_unscoreable_selected_dropped": n_unscoreable_selected_dropped,
                "n_below_assay_transfer_min_score_dropped": n_below_min_score_dropped,
            }
        )
        return result
    finally:
        reranker.cache.close()


def _validate_cache_version_manifest(
    *,
    path: str,
    provenance: dict[str, Any],
    total_cache_rows: int,
    expected_score_count: int,
) -> dict[str, Any]:
    if not path:
        return {"status": "not_requested"}
    version_path = Path(path)
    if not version_path.is_file():
        raise FileNotFoundError(f"Assay-transfer cache VERSION manifest does not exist: {path}")
    payload = json.loads(version_path.read_text(encoding="utf-8"))
    expected = {
        "model": provenance["model"],
        "model_revision": provenance["model_revision"],
        "scoring_contract_version": provenance["scoring_contract_version"],
        "template_hash": provenance["template_hash"],
        "template_profile": provenance["template_profile"],
        "query_context_policy": provenance["query_context_policy"],
        "catalog_version": provenance["catalog_version"],
        "candidate_manifest_sha256": provenance["candidate_manifest_sha256"],
        "n_prompt_scores": expected_score_count or total_cache_rows,
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if payload.get("status") != "complete":
        mismatches["status"] = {"expected": "complete", "observed": payload.get("status")}
    if mismatches:
        raise ValueError(
            "Assay-transfer cache VERSION manifest mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {"status": "pass", "path": str(version_path), "validated_fields": sorted(expected)}


def run_spawned_workers(
    tasks: list[PromptTask],
    *,
    cache: Any,
    snapshot_path: str,
    devices: list[int],
    batch_size: int,
    dtype: str,
    worker_target: Any = None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("rerank batch size must be positive")
    target = worker_target or _gpu_worker
    batches = [tasks[offset : offset + batch_size] for offset in range(0, len(tasks), batch_size)]
    context = mp.get_context("spawn")
    prefetch_depth = 2
    work_queue = context.Queue(maxsize=max(2, len(devices) * (prefetch_depth + 1)))
    result_queue = context.Queue()
    workers = [
        context.Process(
            target=target,
            args=(local_device, snapshot_path, dtype, work_queue, result_queue),
            name=f"assay-transfer-gpu-{local_device}",
        )
        for local_device in devices
    ]
    for worker in workers:
        worker.start()
    started = time.monotonic()
    completed = 0
    committed = 0
    peak_vram: dict[str, float] = {}
    try:
        ready = set()
        while len(ready) < len(workers):
            message = _next_worker_message(result_queue, workers)
            if message.get("type") == "error":
                raise RuntimeError(_worker_error_text(message))
            if message.get("type") != "ready":
                raise RuntimeError(f"Unexpected worker startup message: {message}")
            ready.add(int(message["device"]))

        next_batch = 0
        in_flight = 0
        for _ in range(len(workers) * prefetch_depth):
            if next_batch >= len(batches):
                break
            work_queue.put({"batch_id": next_batch, "tasks": [prompt_task_to_dict(task) for task in batches[next_batch]]})
            next_batch += 1
            in_flight += 1

        while in_flight:
            message = _next_worker_message(result_queue, workers)
            if message.get("type") == "error":
                raise RuntimeError(_worker_error_text(message))
            if message.get("type") != "result":
                raise RuntimeError(f"Unexpected worker message: {message}")
            batch_id = int(message["batch_id"])
            batch_tasks = batches[batch_id]
            scores = [PromptScore(**payload) for payload in message["scores"]]
            if {score.cache_key for score in scores} != {task.cache_key for task in batch_tasks}:
                raise RuntimeError(f"Worker returned mismatched score keys for batch {batch_id}")
            committed += cache.write_batch(batch_tasks, scores)
            completed += len(batch_tasks)
            in_flight -= 1
            peak_vram[str(message["device"])] = max(
                peak_vram.get(str(message["device"]), 0.0), float(message.get("peak_vram_gib") or 0.0)
            )
            if next_batch < len(batches):
                work_queue.put({"batch_id": next_batch, "tasks": [prompt_task_to_dict(task) for task in batches[next_batch]]})
                next_batch += 1
                in_flight += 1
            print(
                f"[assay_transfer_precompute] scored={completed}/{len(tasks)} "
                f"elapsed_s={time.monotonic() - started:.1f}",
                file=sys.stderr,
                flush=True,
            )
        for _ in workers:
            work_queue.put(None)
        for worker in workers:
            worker.join(timeout=60)
        bad = [worker for worker in workers if worker.exitcode not in (0, None)]
        if bad:
            raise RuntimeError(f"Assay-transfer workers exited nonzero: {[(w.name, w.exitcode) for w in bad]}")
    except BaseException:
        _stop_workers(workers)
        raise
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)
        work_queue.close()
        result_queue.close()
    return {
        "n_scored": completed,
        "n_cache_rows_committed": committed,
        "n_workers": len(workers),
        "devices": devices,
        "peak_vram_gib_by_device": peak_vram,
        "elapsed_s": round(time.monotonic() - started, 3),
    }


def _gpu_worker(
    local_device: int,
    snapshot_path: str,
    dtype_name: str,
    work_queue: Any,
    result_queue: Any,
) -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.cuda.set_device(local_device)
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_name)
        if dtype is None:
            raise ValueError(f"Unsupported rerank dtype: {dtype_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot_path, trust_remote_code=True, local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            snapshot_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
        )
        model.to(f"cuda:{local_device}")
        model.eval()
        model.config.use_cache = False
        torch.cuda.reset_peak_memory_stats(local_device)
        result_queue.put({"type": "ready", "device": local_device})
        while True:
            item = work_queue.get()
            if item is None:
                break
            tasks = [prompt_task_from_dict(payload) for payload in item["tasks"]]
            with torch.inference_mode():
                scores = _score_prompt_batch(model, tokenizer, tasks, local_device, torch)
            result_queue.put(
                {
                    "type": "result",
                    "device": local_device,
                    "batch_id": item["batch_id"],
                    "scores": [score.__dict__ for score in scores],
                    "peak_vram_gib": torch.cuda.max_memory_allocated(local_device) / (1024**3),
                }
            )
    except BaseException as error:
        result_queue.put(
            {
                "type": "error",
                "device": local_device,
                "error_type": type(error).__name__,
                "message": str(error),
                "cuda_oom": "out of memory" in str(error).lower(),
                "traceback": traceback.format_exc(),
            }
        )
        raise


def _score_prompt_batch(model: Any, tokenizer: Any, tasks: list[PromptTask], device: int, torch: Any) -> list[PromptScore]:
    """Mirror training evaluation: softmax the first divergent A/B token logits."""
    option_a, option_b = "(A)", "(B)"
    encoded_prefixes: list[list[int]] = []
    a_token_ids: list[int] = []
    b_token_ids: list[int] = []
    for task in tasks:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": task.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        a_ids = tokenizer(prompt + option_a, add_special_tokens=False).input_ids
        b_ids = tokenizer(prompt + option_b, add_special_tokens=False).input_ids
        prefix_len = _first_divergent_index(a_ids, b_ids)
        if prefix_len <= 0:
            raise ValueError("A/B candidate tokenization diverged before the prompt prefix")
        encoded_prefixes.append(a_ids[:prefix_len])
        a_token_ids.append(int(a_ids[prefix_len]))
        b_token_ids.append(int(b_ids[prefix_len]))
    encoded = tokenizer.pad(
        {"input_ids": encoded_prefixes},
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(f"cuda:{device}") for key, value in encoded.items()}
    logits = model(**encoded, use_cache=False).logits
    positions = torch.arange(encoded["attention_mask"].shape[1], device=logits.device)
    last_positions = (encoded["attention_mask"] * positions).max(dim=1).values
    rows = torch.arange(logits.shape[0], device=logits.device)
    next_logits = logits[rows, last_positions]
    scores = []
    for index, task in enumerate(tasks):
        logit_a = float(next_logits[index, a_token_ids[index]].float().item())
        logit_b = float(next_logits[index, b_token_ids[index]].float().item())
        scores.append(
            PromptScore(
                cache_key=task.cache_key,
                logp_transfer=logit_a,
                logp_not_transfer=logit_b,
                transfer_probability=probability_from_log_likelihoods(logit_a, logit_b),
            )
        )
    return scores


def _first_divergent_index(left: list[int], right: list[int]) -> int:
    for index in range(min(len(left), len(right))):
        if left[index] != right[index]:
            return index
    raise ValueError("A/B candidate completions did not diverge in tokenization")


def resolve_model_snapshot(
    model: str,
    revision: str,
    *,
    force_download: bool,
    local_files_only: bool,
) -> tuple[str, str]:
    from huggingface_hub import HfApi, snapshot_download

    if local_files_only:
        immutable_revision = require_immutable_revision(revision)
    else:
        immutable_revision = require_immutable_revision(HfApi().model_info(model, revision=revision).sha)
    path = snapshot_download(
        repo_id=model,
        revision=immutable_revision,
        force_download=force_download,
        local_files_only=local_files_only,
    )
    return path, immutable_revision


def _check_free_vram(devices: list[int], minimum_gib: float) -> None:
    import torch

    visible_count = torch.cuda.device_count()
    for local_device in devices:
        if local_device < 0 or local_device >= visible_count:
            raise RuntimeError(f"Requested local CUDA device {local_device} is not visible")
        free_bytes, _ = torch.cuda.mem_get_info(local_device)
        free_gib = float(free_bytes) / (1024.0**3)
        if free_gib < minimum_gib:
            raise RuntimeError(
                f"Local CUDA device {local_device} has {free_gib:.2f} GiB free; "
                f"at least {minimum_gib:.2f} GiB is required"
            )


def _next_worker_message(result_queue: Any, workers: list[Any]) -> dict[str, Any]:
    while True:
        try:
            return result_queue.get(timeout=2.0)
        except queue.Empty:
            failed = [worker for worker in workers if worker.exitcode not in (None, 0)]
            if failed:
                raise RuntimeError(f"Assay-transfer worker died: {[(w.name, w.exitcode) for w in failed]}")


def _worker_error_text(message: dict[str, Any]) -> str:
    suffix = " (CUDA OOM)" if message.get("cuda_oom") else ""
    return (
        f"Assay-transfer worker {message.get('device')} failed{suffix}: "
        f"{message.get('error_type')}: {message.get('message')}\n{message.get('traceback', '')}"
    )


def _stop_workers(workers: Iterable[Any]) -> None:
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
    for worker in workers:
        worker.join(timeout=10)


def _parse_devices(value: str) -> list[int]:
    devices = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not devices or len(devices) != len(set(devices)):
        raise ValueError(f"Invalid --rerank-devices value: {value!r}")
    return devices


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _catalog_is_flat(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            metadata = json.loads(line)
            return bool(metadata.get("conditions")) or str(metadata.get("source_mode") or "").startswith("flat_")
    return False


def _select_indices(args: argparse.Namespace, length: int) -> list[int]:
    if args.indices:
        indices = []
        for token in args.indices:
            if "-" in token:
                start, end = token.split("-", 1)
                indices.extend(range(int(start), int(end) + 1))
            else:
                indices.append(int(token))
    else:
        end = length if args.limit <= 0 else min(length, args.start + args.limit)
        indices = list(range(args.start, end))
    indices = sorted(set(indices))
    if any(index < 0 or index >= length for index in indices):
        raise ValueError(f"Query indices out of range for {length} records: {indices}")
    return indices


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT)
    parser.add_argument("--smiles-field", default="drug")
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument(
        "--retrieval-source",
        choices=["starling", "starling_in_distribution"],
        default="starling",
    )
    parser.add_argument("--experiment-mode", choices=["direct", "full_flat", "full_mechanism"], default="full_mechanism")
    parser.add_argument("--neighbor-identity-policy", choices=["operational", "parent_disjoint"], default="operational")
    parser.add_argument("--top-k-per-group", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.3)
    parser.add_argument("--indices", nargs="*", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retrieval-reranker", choices=["assay_transfer"], default="assay_transfer")
    parser.add_argument("--assay-transfer-initial-morgan-filter", type=int, default=100)
    parser.add_argument("--rerank-catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--candidate-manifest", default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--rerank-cache", default=DEFAULT_CACHE)
    parser.add_argument(
        "--cache-version-manifest",
        default="",
        help="Write immutable cache and retrieval provenance after successful completion.",
    )
    parser.add_argument("--rerank-cache-mode", choices=["read_write"], default="read_write")
    parser.add_argument("--assay-transfer-model", default=ASSAY_TRANSFER_MODEL)
    parser.add_argument("--assay-transfer-model-revision", default=ASSAY_TRANSFER_MODEL_REVISION)
    parser.add_argument(
        "--assay-transfer-template-profile",
        choices=TEMPLATE_PROFILES,
        default=DEFAULT_TEMPLATE_PROFILE,
    )
    parser.add_argument("--force-model-download", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force-rescore", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Freeze and validate the catalog/manifest inputs without loading the model or requiring GPUs.",
    )
    parser.add_argument(
        "--reuse-frozen-flat-artifacts",
        action="store_true",
        help="Use an existing immutable flat catalog/condition manifest and score only missing prompts.",
    )
    parser.add_argument(
        "--reuse-prebuilt-catalog",
        action="store_true",
        help="Keep an existing source-native catalog immutable and freeze only its condition manifest.",
    )
    parser.add_argument(
        "--condition-id",
        default="validation__starling_in_distribution__parent_disjoint__r100_c100_min0__v6_5",
    )
    parser.add_argument("--rerank-devices", default="0,1")
    parser.add_argument("--rerank-workers-per-device", type=int, default=1)
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--rerank-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--rerank-min-free-vram-gib", type=float, default=64.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
