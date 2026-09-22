"""Run cache-matched flat validation conditions through one prompt pool.

The launcher materializes one cache-matched retrieval batch per condition and
task, verifies the frozen single-molecule branches, and prepares the canonical
task batches. Throughput mode immediately hands runnable calls to
``branches.scheduler.run_prepared_prompt_pool`` with one shared provider pool;
live mode is the explicit pilot/review path.
Normal task batch artifacts, traces, predictions, metrics, and reports remain
the outputs; this module owns only matrix coordination and its root receipts.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from urllib.request import urlopen

from data.processing.gold_labels.conditioned_benchmark import split_path, tdc_split_path
from data.processing.llm_api import (
    DEFAULT_ENV_FILE,
    provider_from_base_url,
)
from predict.harnesses.branches import flat
from predict.harnesses.branches.runner import collect_completed_item
from predict.harnesses.branches.scheduler import (
    BatchCommand,
    prepare_batch_commands,
    run_prepared_prompt_pool,
)
from predict.harnesses.branches.runtime import (
    _source_run_dir,
    collect_stage_result,
    execute_stage,
    load_stage_state,
    prepare_stage_item,
    ready_stage_jobs,
)
from predict.harnesses.branches.inference import load_frozen_single_analysis
from predict.harnesses.branches.prompt import attach_external_condition
from predict.llm_io.response import validated_branch_content
from predict.api_client.pool import (
    DEFAULT_PROVIDER_POOL_CONFIG,
    OpenAIProviderPool,
    ProviderPoolConfig,
    build_provider_pool,
    load_provider_pool_config,
    preflight_provider_models,
    primary_capacity,
    select_healthy_providers,
)
from predict.retrieval.assay_reranking.cache_matched import (
    DEFAULT_CACHE_BUNDLE,
    DEFAULT_GOLD_CONTEXT_MAPPING,
)
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic


MATRIX_VERSION = "joseph_flat_matrix.v4"
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEFAULT_PROVIDER_CONFIG = DEFAULT_PROVIDER_POOL_CONFIG
TASKS = (
    "bbb_martins", "bioavailability_ma", "skin_reaction",
    "ames", "dili", "carcinogens",
)
TASK_PROMPT_PROFILE_OPTIONS = {
    "bbb_martins": "--bbb-prompt-profile",
    "bioavailability_ma": "--bioavailability-prompt-profile",
    "skin_reaction": "--skin-prompt-profile",
    "ames": "--ames-prompt-profile",
    "dili": "--dili-prompt-profile",
    "carcinogens": "--carcinogens-prompt-profile",
}
DIRECT_GRID_TASKS = ("bbb_martins", "bioavailability_ma", "skin_reaction")
RECORDS_PER_LEVEL = 10
MAX_TOKENS = 262_144
GRID_SCREEN_SCHEMA = "record_selection_lambda_screen.v1"
DIRECT_GRID_SCHEMA = "gold_submodular_selection_grid.v1"
DIRECT_GRID_SCHEMAS = {
    DIRECT_GRID_SCHEMA,
    "direct_context_selection_grid.v2",
}
DEFAULT_CONDITIONS = ("morgan:all",)
CONDITION_CHOICES = tuple(
    f"{mode}:{pool}" for mode in flat.VARIANTS for pool in flat.RECORD_POOLS
)
SINGLE_ROOT = Path(
    "outputs/paper/starling_conditioned_gold_l1_deepseek_v4_flash_nvfp4_query_prior/"
    "runs_deployment_visible_parent_disjoint"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/paper/assay_transfer_harness/joseph/"
    "joseph_flat_v2_morgan_valid_records10_cache_v3_deepseek_v4_flash"
)


def endpoint_allocations(
    parallelism: int,
    provider_pool_config: Path = DEFAULT_PROVIDER_CONFIG,
    *,
    config: ProviderPoolConfig | None = None,
) -> list[tuple[str, int]]:
    """Allocate the requested budget across configured primary providers."""
    config = config or load_provider_pool_config(provider_pool_config)
    priority = min(spec.priority for spec in config.providers)
    providers = [spec for spec in config.providers if spec.priority == priority]
    if not 1 <= parallelism <= sum(spec.max_inflight for spec in providers):
        raise ValueError(
            f"Provider capacity is {sum(spec.max_inflight for spec in providers)}, not {parallelism}"
        )
    allocations = [0] * len(providers)
    for _ in range(parallelism):
        index = min(
            (i for i, spec in enumerate(providers) if allocations[i] < spec.max_inflight),
            key=allocations.__getitem__,
        )
        allocations[index] += 1
    return [
        (spec.base_url, allocations[index])
        for index, spec in enumerate(providers)
        if allocations[index]
    ]


def sample_provider_loads(
    config: ProviderPoolConfig,
    *,
    samples: int,
    interval_s: float,
) -> list[dict[str, Any]]:
    """Sample current SGLang running plus waiting work on every endpoint."""
    if samples < 1 or interval_s < 0:
        raise ValueError("load samples must be positive and interval non-negative")
    receipts = []
    for sample_index in range(samples):
        observed_at = time.time()
        for provider in config.providers:
            with urlopen(provider.base_url.rstrip("/") + "/loads", timeout=10) as response:
                payload = json.load(response)
            loads = payload.get("loads") or []
            running = sum(int(row.get("num_running_reqs", 0)) for row in loads)
            waiting = sum(int(row.get("num_waiting_reqs", 0)) for row in loads)
            receipts.append({
                "sample": sample_index + 1,
                "observed_at_unix": observed_at,
                "provider": provider.name,
                "base_url": provider.base_url,
                "running": running,
                "waiting": waiting,
                "total": running + waiting,
            })
        if sample_index + 1 < samples:
            time.sleep(interval_s)
    return receipts


def top_up_provider_config(
    config: ProviderPoolConfig,
    load_samples: list[dict[str, Any]],
    *,
    target_total: int,
) -> tuple[ProviderPoolConfig, list[dict[str, Any]]]:
    """Cap this launcher so observed server work plus new work is at most target."""
    if target_total < 1:
        raise ValueError("target endpoint load must be positive")
    allocations = []
    active = []
    for provider in config.providers:
        if target_total > provider.max_inflight:
            raise ValueError(
                f"target {target_total} exceeds configured cap {provider.max_inflight} "
                f"for {provider.name}"
            )
        observations = [
            int(row["total"]) for row in load_samples if row["provider"] == provider.name
        ]
        if not observations:
            raise ValueError(f"No load samples recorded for {provider.name}")
        largest = max(observations)
        slots = max(0, target_total - largest)
        allocations.append({
            "provider": provider.name,
            "base_url": provider.base_url,
            "target_total": target_total,
            "largest_observed_running_plus_waiting": largest,
            "launcher_slots": slots,
        })
        if slots:
            active.append(replace(provider, max_inflight=slots))
    if not active:
        raise ValueError("Every endpoint is already at or above the requested total load")
    return replace(config, providers=tuple(active)), allocations


def load_preselected_grid(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != GRID_SCREEN_SCHEMA or payload.get("status") != "complete":
        raise ValueError(f"Invalid completed lambda screen manifest: {path}")
    profiles = payload.get("selected_profiles") or []
    if not profiles or len({row.get("name") for row in profiles}) != len(profiles):
        raise ValueError("Lambda screen must contain one or more unique profiles")
    for profile in profiles:
        for task in TASKS:
            entry = (profile.get("task_manifests") or {}).get(task) or {}
            manifest_path = Path(str(entry.get("path") or ""))
            if not manifest_path.is_absolute():
                manifest_path = path.parent / manifest_path
            if not manifest_path.is_file() or sha256_file(manifest_path) != entry.get("sha256"):
                raise ValueError(f"Selected UID manifest mismatch for {profile.get('name')} {task}")
            entry["path"] = str(manifest_path.resolve())
    return profiles


def load_preselected_direct_grid(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") not in DIRECT_GRID_SCHEMAS
        or payload.get("status") != "complete"
        or payload.get("kind") != "direct"
    ):
        raise ValueError(f"Invalid completed direct selection grid: {path}")
    profiles = payload.get("profiles") or []
    if len(profiles) != int(payload.get("profile_count", -1)) or len({
        row.get("name") for row in profiles
    }) != len(profiles):
        raise ValueError("Direct grid profile count or names are invalid")
    tasks = tuple(payload.get("tasks") or DIRECT_GRID_TASKS)
    if not tasks or len(tasks) != len(set(tasks)) or set(tasks) - set(TASKS):
        raise ValueError("Direct grid task list is invalid")
    for profile in profiles:
        subsets = set()
        for task in tasks:
            entry = (profile.get("task_manifests") or {}).get(task) or {}
            manifest = Path(str(entry.get("path") or ""))
            if not manifest.is_absolute():
                manifest = path.parent / manifest
            if not manifest.is_file() or sha256_file(manifest) != entry.get("sha256"):
                raise ValueError(f"Direct manifest mismatch for {profile.get('name')} {task}")
            subsets.add(json.loads(manifest.read_text(encoding="utf-8")).get("subset"))
            entry["path"] = str(manifest.resolve())
        if len(subsets) != 1 or None in subsets:
            raise ValueError(f"Direct profile mixes evaluation subsets: {profile.get('name')}")
        profile["subset"] = subsets.pop()
        profile["tasks"] = tasks
    if len({profile["subset"] for profile in profiles}) != 1:
        raise ValueError("Direct grid profiles mix evaluation subsets")
    return profiles


def provider_client(
    parallelism: int,
    *,
    pilot: bool = False,
    provider_pool_config: Path = DEFAULT_PROVIDER_CONFIG,
    max_tokens: int = MAX_TOKENS,
    timeout_s: int = 900,
    config: ProviderPoolConfig | None = None,
) -> OpenAIProviderPool:
    """Build the shared flat-default reasoning client with bounded transports."""
    config = config or load_provider_pool_config(provider_pool_config)
    if parallelism > primary_capacity(config):
        raise ValueError(
            f"Provider priority tier capacity is {primary_capacity(config)}, not {parallelism}"
        )
    allocations = dict(endpoint_allocations(
        parallelism, provider_pool_config, config=config
    ))
    if pilot:
        first = config.providers[0]
        config = replace(config, providers=(replace(first, max_inflight=1),))
    else:
        config = replace(
            config,
            providers=tuple(
                replace(spec, max_inflight=allocations[spec.base_url])
                for spec in config.providers
                if spec.base_url in allocations
            ),
        )
    provider_kinds = {provider_from_base_url(spec.base_url) for spec in config.providers}
    if len(provider_kinds) != 1:
        raise ValueError("Flat matrix providers must share one reasoning transport mode")
    local = provider_kinds == {"local"}

    return build_provider_pool(
        config,
        env_file=DEFAULT_ENV_FILE,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        temperature=0.0,
        tool_service_url="http://127.0.0.1:8765",
        enable_group_tools=False,
        max_tool_rounds=0,
        reasoning_effort="high" if local else "",
        enable_thinking=local,
        transport_max_retries=0,
    )


def _single_batch(task: str) -> Path:
    return (SINGLE_ROOT / task / f"{task}__none").resolve()


def verify_single_reuse(task: str, detailed_input: Path) -> dict[str, Any]:
    """Verify compact/detailed row identity and every frozen single checkpoint."""
    receipt_path = Path(
        "data/artifacts/gold_labels/conditioned_benchmark/migration_receipt.json"
    ).resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    split_receipt = receipt["tasks"][task]["splits"]["valid"]
    compact_input = split_path(task, "valid").resolve()
    if (
        sha256_file(compact_input) != split_receipt["canonical_sha256"]
        or not split_receipt["byte_identical"]
    ):
        raise ValueError(f"Active compact validation split differs from its receipt: {task}")
    compact, detailed = read_jsonl(compact_input), read_jsonl(detailed_input)
    keys = ("benchmark_row_id", "drug", "Y", "condition_group")
    if [tuple(row.get(key) for key in keys) for row in compact] != [
        tuple(row.get(key) for key in keys) for row in detailed
    ]:
        raise ValueError(f"Detailed validation split disagrees with compact rows: {task}")

    source = _single_batch(task)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    expected_profile = flat.prompt_assets(flat.JOSEPH_PROMPT_VERSION)["tasks"][task][
        "prompt_profile"
    ]
    expected = {
        "n_items": len(detailed),
        "indices": list(range(len(detailed))),
        "task_prompt_profile": expected_profile,
        "identity_blind": False,
        "neighbor_identity_policy": "parent_disjoint",
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Frozen single batch mismatch for {task}: {mismatches}")
    expected_legacy_input = (
        Path(receipt["tasks"][task]["canonical_root"]) / "valid.jsonl"
    )
    if Path(manifest["input_jsonl"]) != expected_legacy_input:
        raise ValueError(f"Frozen single batch has unexpected input lineage: {task}")

    hashes: list[str] = []
    for index in range(len(detailed)):
        run_dir = source / "runs" / f"{source.name}_idx{index:05d}"
        output_path = run_dir / "single_molecule_reasoning_output.json"
        if load_frozen_single_analysis(str(run_dir)) is None:
            raise ValueError(f"Frozen single output is missing or invalid: {output_path}")
        hashes.append(sha256_file(output_path))
    return {
        "task": task,
        "source_batch": str(source),
        "rows": len(detailed),
        "compact_input": str(compact_input),
        "compact_sha256": sha256_file(compact_input),
        "detailed_input": str(detailed_input),
        "detailed_sha256": sha256_file(detailed_input),
        "migration_receipt": str(receipt_path),
        "migration_receipt_sha256": sha256_file(receipt_path),
        "single_outputs_sha256": hashlib.sha256(
            "\n".join(hashes).encode("utf-8")
        ).hexdigest(),
        "prompt_profile": expected_profile,
        "status": "verified",
    }


def _selection_args(
    task: str,
    root: Path,
    *,
    limit: int,
    reranking: str = flat.MORGAN_VARIANT,
    record_pool: str = "all",
    records_per_level: int = RECORDS_PER_LEVEL,
    context_width: int | None = None,
    min_contrast: int = 0,
    context_v5: bool = False,
    context_v6: bool = False,
    all_levels: bool = False,
    flat_preselected_uids: Path | None = None,
    flat_preselected_contexts: Path | None = None,
    context_v5_six_tasks: bool = False,
    six_task_prompt_version: str = flat.CONTEXT_V5_SIX_TASKS_PROMPT_VERSION,
    batch_id: str | None = None,
    evaluation_subset: str = "valid",
    prior_root: Path = flat.DEFAULT_QUERY_PRIOR_ROOT,
    assay_transfer_cache: Path = DEFAULT_CACHE_BUNDLE,
    benchmark: str = "gold",
) -> argparse.Namespace:
    input_jsonl = (
        tdc_split_path(task, evaluation_subset)
        if benchmark == "tdc" else split_path(task, evaluation_subset)
    ).with_name(
        f"{evaluation_subset}_molecule_condition_labels.jsonl"
    ).resolve()
    context_v4 = (
        context_width is not None and not context_v5 and not context_v6
        and not context_v5_six_tasks
    )
    context_prompt = context_v4 or context_v5 or context_v6 or context_v5_six_tasks
    return argparse.Namespace(
        harness_version=(
            flat.JOSEPH_PROMPT_HARNESSES[six_task_prompt_version]
            if context_v5_six_tasks
            else flat.CONTEXT_V6_HARNESS_VERSION if context_v6
            else flat.CONTEXT_V5_HARNESS_VERSION if context_v5
            else flat.CONTEXT_V4_HARNESS_VERSION if context_v4
            else flat.PUBLIC_HARNESS_VERSION
        ),
        prompt_version=(
            six_task_prompt_version if context_v5_six_tasks
            else flat.CONTEXT_V6_PROMPT_VERSION if context_v6
            else flat.CONTEXT_V5_PROMPT_VERSION if context_v5
            else flat.CONTEXT_V4_PROMPT_VERSION if context_v4
            else flat.JOSEPH_PROMPT_VERSION
        ),
        task=task,
        benchmark=benchmark,
        reranking=reranking,
        assay_transfer_cache=assay_transfer_cache.resolve(),
        record_pool=record_pool,
        evaluation_subset=evaluation_subset,
        input_jsonl=input_jsonl,
        evidence_library=(Path("data/evidence_libraries") / task / "v10").resolve(),
        level_mapper=Path("data/evidence_libraries/level_mappings.v1.json").resolve(),
        gold_context_mapping=DEFAULT_GOLD_CONTEXT_MAPPING.resolve(),
        allow_frozen_l1_vote_scores=task == "bbb_martins",
        l1_molecules=10,
        l1_records_per_molecule=10,
        records_per_level=records_per_level,
        ranking_tie_seed=0,
        max_level=flat.TASKS[task] if all_levels else 1 if context_prompt else 0,
        layout="level-grouped" if context_v5 or context_v5_six_tasks else "global",
        query_prior="cached",
        prior_root=prior_root.resolve(),
        l1_min_contrast=min_contrast,
        morgan_primary_parent_width=context_width or 100,
        molecule_description_mode="none",
        molecule_description_cache_version="v1",
        flat_preselected_uids=flat_preselected_uids,
        flat_preselected_contexts=flat_preselected_contexts,
        record_limits_by_level={
            f"L{level}": records_per_level
            for level in range(2, flat.TASKS[task] + 1)
        },
        indices=None,
        start=0,
        limit=limit,
        batch_root=(root / task).resolve(),
        batch_id=batch_id or (
            f"{task}__{'all_levels_' if all_levels else 'l1_'}k10_w{context_width}_m{min_contrast}"
            if context_v5 else
            f"{task}__k10_w{context_width}_m{min_contrast}"
            if context_v4 else
            f"{task}__morgan_records10"
            if (
                reranking == flat.MORGAN_VARIANT
                and record_pool == "assay-transfer-trained"
                and records_per_level == RECORDS_PER_LEVEL
            )
            else f"{task}__{reranking}_{record_pool}_records{records_per_level}"
        ),
    )


def _batch_command(
    args: argparse.Namespace,
    source: Path,
    *,
    model: str = MODEL,
    base_url: str = "",
    api_key_env: str = "",
    trace_root: Path = Path("outputs/paper/live"),
    execution_mode: str = "live",
    timeout_s: int = 900,
    max_tokens: int = MAX_TOKENS,
    reasoning_effort: str = "high",
    enable_thinking: bool = True,
) -> BatchCommand:
    if not base_url:
        default = load_provider_pool_config(DEFAULT_PROVIDER_CONFIG).providers[0]
        base_url, api_key_env = default.base_url, default.api_key_env
    task = args.task
    batch_id = args.batch_id
    context_v4 = args.prompt_version in flat.CONTEXT_PROMPT_VERSIONS
    command = [
        sys.executable,
        "-m",
        f"predict.harnesses.branches.tasks.{task}.contract",
        "--experiment-mode",
        "full_flat",
        "--input-jsonl",
        str(args.input_jsonl),
        "--batch-root",
        str(args.batch_root),
        "--batch-id",
        batch_id,
        "--retrieval-replay-source-batch",
        str(source),
        "--flat-selection-manifest",
        str(source / "manifest.json"),
        "--flat-prompt-version",
        args.prompt_version,
        "--flat-reranking",
        args.reranking,
        "--retrieval-strategy",
        "morgan_fingerprint",
        "--neighbor-identity-policy",
        "scaffold_disjoint",
        "--min-similarity",
        "0",
        "--disable-flat-tools",
        "--model",
        model,
        "--base-url",
        base_url,
        "--api-key-env",
        api_key_env,
        "--timeout-s",
        str(timeout_s),
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        "0",
        "--reasoning-effort",
        reasoning_effort,
        "--enable-thinking" if enable_thinking else "--disable-thinking",
        "--skip-existing",
        "--no-stream-logs",
        "--trace-root",
        str(trace_root),
        "--execution-mode",
        execution_mode,
    ]
    if args.indices:
        command.extend(["--indices", *args.indices])
    else:
        command.extend(["--start", "0", "--limit", str(args.limit)])
    if context_v4:
        command.extend([
            TASK_PROMPT_PROFILE_OPTIONS[task],
            flat.prompt_assets(args.prompt_version)["tasks"][task]["prompt_profile"],
        ])
        command.extend([
            "--flat-layout", args.layout,
            "--flat-query-prior", args.query_prior,
            "--single-analysis-source-batch",
            str(args.prior_root / task / f"{task}__none"),
        ])
    else:
        command.extend(["--single-analysis-source-batch", str(_single_batch(task))])
    return BatchCommand(batch_id, command)


def _provider_execution(
    config: ProviderPoolConfig,
    timeout_s: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    models = {spec.model for spec in config.providers}
    if len(models) != 1:
        raise ValueError("Flat matrix providers must request one exact model alias")
    provider_kinds = {provider_from_base_url(spec.base_url) for spec in config.providers}
    if len(provider_kinds) != 1:
        raise ValueError("Flat matrix providers must share one reasoning transport mode")
    local = provider_kinds == {"local"}

    receipts = []
    for spec in config.providers:
        provider = provider_from_base_url(spec.base_url)
        receipts.append(
            {
                **spec.public_dict(),
                "provider": provider,
                "credential_env": spec.api_key_env,
                "status": "configured",
            }
        )
    first = config.providers[0]
    return (
        {
            "model": first.model,
            "base_url": first.base_url,
            "api_key_env": first.api_key_env,
            "timeout_s": first.timeout_s or timeout_s,
            "reasoning_effort": "high" if local else "",
            "enable_thinking": local,
        },
        receipts,
    )


def _run_pilot(prepared_by_name: dict[str, Any], client: OpenAIProviderPool) -> None:
    """Complete the first three queries in each canonical batch."""
    for prepared in prepared_by_name.values():
        for item in prepared.items[:3]:
            if collect_completed_item(prepared, item) is not None:
                continue
            result = prepare_stage_item(prepared, item)
            if result.get("status") != "ok":
                raise RuntimeError(f"Pilot preparation failed: {result}")
            state = load_stage_state(prepared, item)
            if state is None:
                raise RuntimeError(f"Pilot has no resumable stage state: {prepared.batch_id}")
            while collect_stage_result(state).get("status") != "ok":
                jobs = ready_stage_jobs(state)
                if not jobs:
                    raise RuntimeError(f"Pilot dependencies stalled: {prepared.batch_id}")
                for job in jobs:
                    output = execute_stage(job, client)
                    if output.get("status") != "ok":
                        raise RuntimeError(
                            f"Pilot {job.stage} failed for {prepared.batch_id}: {output}"
                        )


def _materialize_direct_request_review(
    prepared_by_name: dict[str, Any],
    *,
    root: Path,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Render exact direct-grid requests without creating an inference client."""
    rows: list[dict[str, Any]] = []
    for batch_id, prepared in sorted(prepared_by_name.items()):
        if prepared.args.flat_prompt_version not in {
            flat.CONTEXT_V5_SIX_TASKS_PROMPT_VERSION,
            flat.CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
        }:
            raise ValueError(f"Unexpected direct-grid prompt version for {batch_id}")
        retrieval_batch = Path(prepared.args.retrieval_replay_source_batch)
        prior_batch = Path(prepared.args.single_analysis_source_batch)
        for item in prepared.items:
            retrieval_path = (
                retrieval_batch / "runs"
                / f"{retrieval_batch.name}_idx{item.index:05d}" / "retrieval.json"
            )
            prior_path = _source_run_dir(
                prior_batch, item.index
            ) / "single_molecule_reasoning_output.json"
            retrieval = attach_external_condition(
                json.loads(retrieval_path.read_text(encoding="utf-8")), item.record
            )
            prior = validated_branch_content(
                json.loads(prior_path.read_text(encoding="utf-8"))
            )
            messages, metadata = flat.build_flat_context_request(
                retrieval,
                task_id=prepared.config.pipeline_module.split(".")[-2],
                task_prompt_profile=str(prepared.args.task_prompt_profile),
                layout=str(prepared.args.flat_layout),
                reranking=str(prepared.args.flat_reranking),
                query_prior=prior,
                prompt_version=str(prepared.args.flat_prompt_version),
            )
            request = {
                "schema_version": "joseph_flat_context_request.v1",
                "messages": messages,
                "message_char_count": sum(len(message["content"]) for message in messages),
                **metadata,
            }
            run_id = f"{prepared.batch_id}_idx{item.index:05d}"
            request_path = prepared.batch_run_root / run_id / "request.json"
            write_json_atomic(request_path, request)
            rows.append({
                "batch_id": batch_id,
                "task": prepared.config.pipeline_module.split(".")[-2],
                "query_index": item.index,
                "benchmark_row_id": item.record["benchmark_row_id"],
                "request_path": str(request_path),
                "request_sha256": sha256_file(request_path),
                "retrieval_sha256": sha256_file(retrieval_path),
                "query_prior_sha256": sha256_file(prior_path),
                "message_char_count": request["message_char_count"],
            })
    index_path = root / "request_review.tsv"
    fieldnames = tuple(rows[0]) if rows else ()
    temporary = index_path.with_suffix(".tsv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, index_path)
    summary = {
        "schema_version": "direct_prompt_review.v1",
        "status": "prepared_awaiting_prompt_review",
        "request_count": len(rows),
        "batch_count": len(prepared_by_name),
        "model": model,
        "reasoning_effort": "high",
        "thinking": {"type": "enabled"},
        "max_tokens": max_tokens,
        "request_index": str(index_path),
        "request_index_sha256": sha256_file(index_path),
        "prompt_assets": flat.prompt_asset_manifest(
            next(iter(prepared_by_name.values())).args.flat_prompt_version
        ),
    }
    write_json_atomic(root / "request_review.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--study", default="")
    parser.add_argument("--method", default="")
    parser.add_argument(
        "--results-root", type=Path,
        default=Path("outputs/paper/assay_transfer_harness/joseph"),
    )
    parser.add_argument(
        "--parallelism", type=int, default=None,
        help="Required global outstanding-request budget for inference.",
    )
    parser.add_argument("--preparation-workers", type=int, default=8)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITION_CHOICES,
        default=DEFAULT_CONDITIONS,
    )
    parser.add_argument("--records-per-level", type=int, default=RECORDS_PER_LEVEL)
    parser.add_argument("--context-v4-grid", action="store_true")
    parser.add_argument("--context-v5-grid", action="store_true")
    parser.add_argument("--context-v5-l1", action="store_true")
    parser.add_argument("--context-v5-all-level", action="store_true")
    parser.add_argument("--preselected-grid-manifest", type=Path)
    parser.add_argument("--preselected-direct-grid-manifest", type=Path)
    parser.add_argument(
        "--preselected-direct-harness-version",
        choices=(
            flat.CONTEXT_V5_SIX_TASKS_HARNESS_VERSION,
            flat.CONTEXT_V5_SIX_TASKS_UPSTREAM_HARNESS_VERSION,
        ),
        default=flat.CONTEXT_V5_SIX_TASKS_HARNESS_VERSION,
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS)
    parser.add_argument(
        "--evaluation-subset", choices=("valid", "test")
    )
    parser.add_argument(
        "--evaluation-subsets", nargs="+", choices=("valid", "test")
    )
    parser.add_argument("--benchmark", choices=("gold", "tdc"), default="gold")
    parser.add_argument(
        "--prior-root", type=Path, default=flat.DEFAULT_QUERY_PRIOR_ROOT
    )
    parser.add_argument(
        "--prior-root-by-subset", action="append", default=[], metavar="SUBSET=PATH"
    )
    parser.add_argument(
        "--assay-transfer-cache", type=Path, default=DEFAULT_CACHE_BUNDLE
    )
    parser.add_argument(
        "--preselected-harness-version",
        choices=(flat.CONTEXT_V5_HARNESS_VERSION, flat.CONTEXT_V6_HARNESS_VERSION),
        default=flat.CONTEXT_V6_HARNESS_VERSION,
    )
    parser.add_argument("--grid-run-id", default="")
    parser.add_argument(
        "--morgan-primary-parent-widths", nargs="+", type=int,
        choices=(15, 25, 50, 100), default=(15, 25, 50),
    )
    parser.add_argument(
        "--l1-min-contrasts", nargs="+", type=int, default=(0, 1, 2),
    )
    parser.add_argument("--provider-pool-config", type=Path, default=DEFAULT_PROVIDER_CONFIG)
    parser.add_argument("--trace-root", type=Path, default=Path("outputs/paper/live"))
    parser.add_argument(
        "--execution-mode", choices=("live", "throughput"), default="throughput"
    )
    parser.add_argument("--continue-after-pilot", action="store_true")
    parser.add_argument("--live-run-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--request-timeout-s", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--target-total-load-per-endpoint", type=int)
    parser.add_argument("--load-samples", type=int, default=6)
    parser.add_argument("--load-sample-interval-s", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-pilots", action="store_true")
    args = parser.parse_args(raw_argv)
    if sum((
        args.context_v4_grid, args.context_v5_grid, args.context_v5_l1,
        args.context_v5_all_level,
        args.preselected_grid_manifest is not None,
        args.preselected_direct_grid_manifest is not None,
    )) > 1:
        parser.error("context matrix modes are mutually exclusive")
    if args.evaluation_subset and args.evaluation_subsets:
        parser.error("use either --evaluation-subset or --evaluation-subsets")
    evaluation_subsets = tuple(args.evaluation_subsets or (args.evaluation_subset or "valid",))
    if args.preselected_grid_manifest and (
        args.records_per_level != 10
        or args.morgan_primary_parent_widths != (15, 25, 50)
        or args.l1_min_contrasts != (0, 1, 2)
    ):
        parser.error("the preselected grid has fixed K10/W25/M0 settings")
    if args.preselected_direct_grid_manifest and len(evaluation_subsets) != 1:
        parser.error("a direct selection grid must target exactly one evaluation subset")
    if "--preselected-harness-version" in raw_argv and not args.preselected_grid_manifest:
        parser.error("--preselected-harness-version requires --preselected-grid-manifest")
    if (
        "--preselected-direct-harness-version" in raw_argv
        and not args.preselected_direct_grid_manifest
    ):
        parser.error(
            "--preselected-direct-harness-version requires --preselected-direct-grid-manifest"
        )
    if args.parallelism is not None and args.parallelism < 1:
        parser.error("--parallelism must be positive")
    if min(
        args.preparation_workers,
        args.records_per_level,
        args.request_timeout_s,
        args.max_tokens,
    ) < 1 or args.limit < 0:
        parser.error("worker, record, token, and timeout values must be positive")
    if args.load_samples < 1 or args.load_sample_interval_s < 0:
        parser.error("load samples must be positive and sample interval non-negative")
    if args.target_total_load_per_endpoint is not None and args.parallelism is not None:
        parser.error("top-up allocation derives parallelism; do not also pass --parallelism")
    if args.target_total_load_per_endpoint is not None and args.prepare_only:
        parser.error("top-up allocation is an immediate pre-launch operation")
    if len(args.conditions) != len(set(args.conditions)):
        parser.error("--conditions cannot contain duplicates")
    if len(evaluation_subsets) != len(set(evaluation_subsets)):
        parser.error("evaluation subsets cannot contain duplicates")
    if len(args.morgan_primary_parent_widths) != len(set(args.morgan_primary_parent_widths)):
        parser.error("--morgan-primary-parent-widths cannot contain duplicates")
    if (len(args.l1_min_contrasts) != len(set(args.l1_min_contrasts))
            or min(args.l1_min_contrasts) < 0):
        parser.error("--l1-min-contrasts must be unique and non-negative")
    if (args.context_v5_l1 or args.context_v5_all_level) and len(args.l1_min_contrasts) != 1:
        parser.error("a single full-flat-v5 run requires exactly one --l1-min-contrasts value")
    conditions = [tuple(value.split(":", 1)) for value in args.conditions]
    args.provider_pool_config = args.provider_pool_config.resolve()
    args.trace_root = args.trace_root.resolve()
    args.prior_root = args.prior_root.resolve()
    args.assay_transfer_cache = args.assay_transfer_cache.resolve()
    prior_roots = {subset: args.prior_root for subset in evaluation_subsets}
    for value in args.prior_root_by_subset:
        subset, separator, path = value.partition("=")
        if not separator or subset not in {"valid", "test"} or not path:
            parser.error("--prior-root-by-subset must be valid=PATH or test=PATH")
        prior_roots[subset] = Path(path).resolve()
    if args.preselected_grid_manifest:
        args.preselected_grid_manifest = args.preselected_grid_manifest.resolve()
    if args.preselected_direct_grid_manifest:
        args.preselected_direct_grid_manifest = args.preselected_direct_grid_manifest.resolve()
    declared_direct_tasks = (
        json.loads(args.preselected_direct_grid_manifest.read_text()).get("tasks") or []
        if args.preselected_direct_grid_manifest else []
    )
    candidate_config = load_provider_pool_config(args.provider_pool_config)
    args.requested_parallelism = args.parallelism
    args.endpoint_selection = None
    args.load_receipt = []
    args.top_up_allocations = []
    late_top_up = False
    if args.prepare_only:
        args.parallelism = args.parallelism or 1
        provider_config = candidate_config
    else:
        if args.target_total_load_per_endpoint is not None:
            late_top_up = True
            provider_config = candidate_config
            args.parallelism = args.preparation_workers
        elif args.parallelism is None:
            parser.error("full-batch inference requires explicit --parallelism")
        else:
            selection = select_healthy_providers(candidate_config, args.parallelism)
            args.endpoint_selection = selection.public_dict()
            args.parallelism = selection.effective_parallelism
            provider_config = selection.config

    root = args.output_root.resolve()
    args.results_root = args.results_root.resolve()
    if bool(args.study) != bool(args.method):
        parser.error("--study and --method must be supplied together")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "launcher.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare_only:
            execution = {
                "model": MODEL,
                "base_url": "",
                "api_key_env": "",
                "timeout_s": args.request_timeout_s,
                "reasoning_effort": "high",
                "enable_thinking": True,
            }
            endpoint_receipts = []
        else:
            execution, endpoint_receipts = _provider_execution(
                provider_config,
                args.request_timeout_s,
            )
        commands = []
        selected_tasks = tuple(dict.fromkeys(args.tasks or (
            declared_direct_tasks or TASKS[:2]
        )))
        context_matrix = (
            args.context_v4_grid or args.context_v5_grid
            or args.context_v5_l1 or args.context_v5_all_level
            or args.preselected_grid_manifest
            or args.preselected_direct_grid_manifest
        )
        single_receipts = [] if context_matrix else [
            verify_single_reuse(
                task,
                split_path(task, "valid").with_name(
                    "valid_molecule_condition_labels.jsonl"
                ).resolve(),
            )
            for task in selected_tasks
        ]
        selection_receipts = []
        variants = (
            [
                ("assay-transfer-contrastive", "all", width, contrast, False)
                for width in args.morgan_primary_parent_widths
                for contrast in args.l1_min_contrasts
            ] + [("assay-transfer-contrastive", "all", 25, 0, True)]
            if args.context_v5_grid else
            [
                ("assay-transfer-contrastive", "all", width, contrast, False)
                for width in args.morgan_primary_parent_widths
                for contrast in args.l1_min_contrasts
            ]
            if args.context_v4_grid else
            [("assay-transfer-contrastive", "all", 25, args.l1_min_contrasts[0], False)]
            if args.context_v5_l1 else
            [("assay-transfer-contrastive", "all", 25, args.l1_min_contrasts[0], True)]
            if args.context_v5_all_level else
            [(reranking, record_pool, None, 0, False) for reranking, record_pool in conditions]
        )
        grid_profiles = (
            load_preselected_grid(args.preselected_grid_manifest)
            if args.preselected_grid_manifest else []
        )
        direct_grid_profiles = (
            load_preselected_direct_grid(args.preselected_direct_grid_manifest)
            if args.preselected_direct_grid_manifest else []
        )
        direct_grid_subset = (
            "valid" if direct_grid_profiles
            and direct_grid_profiles[0]["subset"] == "valid_small"
            else direct_grid_profiles[0]["subset"] if direct_grid_profiles else None
        )
        if direct_grid_profiles and direct_grid_subset != evaluation_subsets[0]:
            parser.error(
                "direct selection grid subset does not match --evaluation-subset: "
                f"{direct_grid_profiles[0]['subset']} != {evaluation_subsets[0]}"
            )
        preselected_v5 = bool(
            grid_profiles
            and args.preselected_harness_version == flat.CONTEXT_V5_HARNESS_VERSION
        )
        preselected_v6 = bool(
            grid_profiles
            and args.preselected_harness_version == flat.CONTEXT_V6_HARNESS_VERSION
        )
        grid_run_id = args.grid_run_id or time.strftime("k10_m0_%Y%m%d_%H%M%S")
        work = (
            [
                {
                    "profile": profile["name"], "task": task,
                    "subset": evaluation_subsets[0],
                    "manifest": Path(profile["task_manifests"][task]["path"]).resolve(),
                    "direct": True,
                    "root": root / profile["name"] / "with_query_prior" / grid_run_id,
                }
                for profile in direct_grid_profiles
                for task in selected_tasks
            ]
            if direct_grid_profiles else
            [
                {
                    "profile": profile["name"],
                    "task": task,
                    "subset": subset,
                    "manifest": Path(profile["task_manifests"][task]["path"]).resolve(),
                    "root": (
                        root / profile["name"] / "with_query_prior" / grid_run_id
                        / subset if len(evaluation_subsets) > 1
                        else root / profile["name"] / "with_query_prior" / grid_run_id
                    ),
                    "direct": False,
                }
                for profile in grid_profiles
                for subset in evaluation_subsets
                for task in selected_tasks
            ]
            if grid_profiles else
            [
                {
                    "profile": "", "task": task, "subset": subset, "manifest": None,
                    "direct": False,
                    "root": root / subset if len(evaluation_subsets) > 1 else root,
                    "variant": (reranking, record_pool, width, contrast, all_levels),
                }
                for reranking, record_pool, width, contrast, all_levels in variants
                for subset in evaluation_subsets
                for task in selected_tasks
            ]
        )
        for item in work:
            task = item["task"]
            subset = item["subset"]
            if item["profile"]:
                reranking, record_pool, width, contrast, all_levels = (
                    "assay-transfer-contrastive", "all", 25, 0, True
                )
            else:
                reranking, record_pool, width, contrast, all_levels = item["variant"]
            selection_args = _selection_args(
                task,
                item["root"],
                limit=args.limit,
                reranking=reranking,
                record_pool=record_pool,
                records_per_level=args.records_per_level,
                context_width=width,
                min_contrast=contrast,
                context_v5=(
                    args.context_v5_grid or args.context_v5_l1
                    or args.context_v5_all_level or preselected_v5
                ),
                context_v6=preselected_v6,
                context_v5_six_tasks=item["direct"],
                six_task_prompt_version=flat.JOSEPH_HARNESS_PROMPTS[
                    args.preselected_direct_harness_version
                ],
                all_levels=all_levels and not item["direct"],
                flat_preselected_uids=(None if item["direct"] else item["manifest"]),
                flat_preselected_contexts=(item["manifest"] if item["direct"] else None),
                evaluation_subset=subset,
                prior_root=prior_roots[subset],
                assay_transfer_cache=args.assay_transfer_cache,
                benchmark=args.benchmark,
                batch_id=(
                    f"{task}__{item['profile']}" if item["profile"] else None
                ),
            )
            if len(evaluation_subsets) > 1:
                prefix = f"{task}__"
                selection_args.batch_id = (
                    f"{prefix}{subset}__{selection_args.batch_id.removeprefix(prefix)}"
                )
            source, selection = flat._materialize_cache_matched_retrievals(
                selection_args
            )
            selection_receipts.append(
                {
                    "task": task,
                    "evaluation_subset": subset,
                    "reranking": reranking,
                    "record_pool": record_pool,
                    "morgan_primary_parent_width": width,
                    "l1_min_contrast": contrast,
                    "profile": item["profile"],
                    "preselected_uids": (
                        "" if item["direct"] else str(item["manifest"] or "")
                    ),
                    "preselected_contexts": (
                        str(item["manifest"] or "") if item["direct"] else ""
                    ),
                    "path": str(source / "manifest.json"),
                    "sha256": sha256_file(source / "manifest.json"),
                    "selection_contract_sha256": selection[
                        "selection_contract_sha256"
                    ],
                }
            )
            commands.append(
                _batch_command(
                    selection_args,
                    source,
                    trace_root=args.trace_root,
                    execution_mode=args.execution_mode,
                    max_tokens=args.max_tokens,
                    **execution,
                )
            )

        prepared = prepare_batch_commands(
            commands,
            max_workers=args.preparation_workers,
            max_stage_requeues=0,
        )
        request_review = None
        if args.prepare_only and args.preselected_direct_grid_manifest:
            request_review = _materialize_direct_request_review(
                prepared,
                root=root,
                model=MODEL,
                max_tokens=args.max_tokens,
            )
            expected_batches = len(direct_grid_profiles) * len(selected_tasks)
            expected_requests = sum(len(batch.items) for batch in prepared.values())
            if (
                request_review["batch_count"] != expected_batches
                or request_review["request_count"] != expected_requests
            ):
                raise ValueError(
                    "Direct request review is incomplete: "
                    f"{request_review['batch_count']} batches and "
                    f"{request_review['request_count']} requests"
                )
        live_runs = {}
        subset_by_batch = {
            command.experiment_name: command.experiment_name.split("__", 2)[1]
            if len(evaluation_subsets) > 1 else evaluation_subsets[0]
            for command in commands
        }
        if not args.prepare_only:
            from predict.live import (
                _refresh_catalog as refresh_live_catalog,
                create_run,
                update_run,
            )

            command_by_batch = {command.experiment_name: command.command for command in commands}
            for batch_id, batch in prepared.items():
                task, _, condition = batch_id.partition("__")
                method = f"flat_{condition or batch_id}"
                command = command_by_batch[batch_id]
                run_dir = create_run(
                    root=args.trace_root,
                    dataset=task,
                    method=method,
                    command=command,
                    metadata={
                        "harness": "flat",
                        "prompt_version": batch.args.flat_prompt_version,
                        "evaluation_subset": subset_by_batch[batch_id],
                        "pilot_size": 3,
                        "output_root": str(batch.batch_dir),
                        "matrix_output_root": str(root),
                    },
                    requested_id=args.live_run_id,
                    execution_mode=args.execution_mode,
                    refresh_catalog=False,
                )
                batch.args.trace_root = str(args.trace_root)
                batch.args.live_run_id = run_dir.name
                batch.args.live_method = method
                live_runs[batch_id] = run_dir
                update_run(
                    run_dir,
                    resume_command=[*command, "--live-run-id", run_dir.name],
                    refresh_catalog=False,
                )
            if args.execution_mode == "live":
                refresh_live_catalog(args.trace_root)
        if late_top_up:
            candidate_failovers = candidate_config.max_failovers
            selection = select_healthy_providers(
                candidate_config, primary_capacity(candidate_config),
            )
            candidate_config = replace(
                selection.config, max_failovers=candidate_failovers,
            )
            model_checks = list(selection.checks)
            args.load_receipt = sample_provider_loads(
                candidate_config,
                samples=args.load_samples,
                interval_s=args.load_sample_interval_s,
            )
            provider_config, args.top_up_allocations = top_up_provider_config(
                candidate_config,
                args.load_receipt,
                target_total=args.target_total_load_per_endpoint,
            )
            args.parallelism = primary_capacity(provider_config)
            args.requested_parallelism = args.parallelism
            args.endpoint_selection = {
                "checks": model_checks,
                "mode": "top_up_to_total_running_plus_waiting",
                "load_samples": args.load_receipt,
                "allocations": args.top_up_allocations,
            }
            execution, endpoint_receipts = _provider_execution(
                provider_config, args.request_timeout_s,
            )
        manifest = {
            "version": MATRIX_VERSION,
            "status": (
                "prepared_awaiting_prompt_review"
                if args.prepare_only and args.preselected_direct_grid_manifest
                else "prepared" if args.prepare_only else "running"
            ),
            "tasks": list(selected_tasks),
            "benchmark": args.benchmark,
            "evaluation_subsets": list(evaluation_subsets),
            "harness_version": (
                args.preselected_direct_harness_version
                if args.preselected_direct_grid_manifest
                else args.preselected_harness_version if args.preselected_grid_manifest
                else flat.CONTEXT_V5_HARNESS_VERSION
                if args.context_v5_grid or args.context_v5_l1 or args.context_v5_all_level
                else flat.CONTEXT_V4_HARNESS_VERSION
                if args.context_v4_grid else flat.PUBLIC_HARNESS_VERSION
            ),
            "prompt_version": (
                flat.JOSEPH_HARNESS_PROMPTS[args.preselected_direct_harness_version]
                if args.preselected_direct_grid_manifest
                else flat.JOSEPH_HARNESS_PROMPTS[args.preselected_harness_version]
                if args.preselected_grid_manifest
                else flat.CONTEXT_V5_PROMPT_VERSION
                if args.context_v5_grid or args.context_v5_l1 or args.context_v5_all_level
                else flat.CONTEXT_V4_PROMPT_VERSION
                if args.context_v4_grid else flat.JOSEPH_PROMPT_VERSION
            ),
            "conditions": [
                {"reranking": reranking, "record_pool": record_pool}
                for reranking, record_pool in conditions
            ],
            "records_per_level": args.records_per_level,
            "context_v4_grid": args.context_v4_grid,
            "context_v5_grid": args.context_v5_grid,
            "context_v5_l1": args.context_v5_l1,
            "context_v5_all_level": args.context_v5_all_level,
            "preselected_grid_manifest": (
                {
                    "path": str(args.preselected_grid_manifest),
                    "sha256": sha256_file(args.preselected_grid_manifest),
                    "profiles": len(grid_profiles),
                    "run_id": grid_run_id,
                }
                if args.preselected_grid_manifest else None
            ),
            "preselected_direct_grid_manifest": (
                {
                    "path": str(args.preselected_direct_grid_manifest),
                    "sha256": sha256_file(args.preselected_direct_grid_manifest),
                    "profiles": len(direct_grid_profiles),
                    "run_id": grid_run_id,
                }
                if args.preselected_direct_grid_manifest else None
            ),
            "morgan_primary_parent_widths": (
                [25] if args.context_v5_l1 else args.morgan_primary_parent_widths
                if args.context_v4_grid or args.context_v5_grid
                else [25] if args.context_v5_all_level or args.preselected_grid_manifest else []
            ),
            "l1_min_contrasts": (
                list(args.l1_min_contrasts) if args.context_v5_l1 else args.l1_min_contrasts
                if args.context_v4_grid or args.context_v5_grid
                else list(args.l1_min_contrasts) if args.context_v5_all_level
                else [0] if args.preselected_grid_manifest else []
            ),
            "l1_molecules": 10,
            "l1_records_per_molecule": 10,
            "max_level_by_task": (
                {task: 1 for task in selected_tasks}
                if args.context_v4_grid or args.context_v5_l1 else
                {task: (1 if args.preselected_direct_grid_manifest else flat.TASKS[task])
                 for task in selected_tasks}
            ),
            "ranking_tie_seed": 0,
            "model": execution["model"],
            "parallelism": args.parallelism,
            "requested_parallelism": args.requested_parallelism,
            "provider_pool_config": {
                "path": str(args.provider_pool_config),
                "sha256": sha256_file(args.provider_pool_config),
                "config": load_provider_pool_config(
                    args.provider_pool_config
                ).public_dict(),
                "selection": args.endpoint_selection,
            },
            "assay_transfer_cache": {
                "path": str(args.assay_transfer_cache),
                "sha256": sha256_file(args.assay_transfer_cache),
            },
            "prior_roots": {subset: str(path) for subset, path in prior_roots.items()},
            "endpoint_allocations": dict(
                endpoint_allocations(
                    args.parallelism, args.provider_pool_config,
                    config=provider_config,
                )
            ) if not args.prepare_only else {},
            "load_samples": args.load_receipt,
            "top_up_allocations": args.top_up_allocations,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "reasoning_effort": execution["reasoning_effort"],
            "thinking": (
                {"type": "enabled"} if execution["enable_thinking"] else None
            ),
            "request_timeout_s": execution["timeout_s"],
            "transport_max_retries": 0,
            "provider_failovers": provider_config.max_failovers,
            "endpoint_preflight": endpoint_receipts,
            "single_reuse": single_receipts,
            "selections": selection_receipts,
            "request_review": request_review,
            "limit": args.limit,
            "pilot_queries_per_batch": (
                0
                if args.prepare_only or args.skip_pilots or args.execution_mode == "throughput"
                else 3
            ),
            "pilots_skipped": args.skip_pilots or args.execution_mode == "throughput",
            "execution_mode": args.execution_mode,
            "code": {
                "matrix": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
                "flat": {"path": str(Path(flat.__file__).resolve()), "sha256": sha256_file(Path(flat.__file__))},
            },
        }
        write_json_atomic(root / "matrix.json", manifest)
        organized_runs = {}
        if args.study and direct_grid_profiles:
            for item in work:
                organized_runs[item["profile"]] = item["root"]
            for profile, run_root in organized_runs.items():
                write_json_atomic(run_root / "run.json", {
                    "schema_version": "organized_study_run.v1",
                    "study": args.study, "method": profile,
                    "run_id": run_root.name, "batch_id": root.name,
                    "status": manifest["status"], "metric_status": "pending",
                    "prompt_version": manifest["prompt_version"],
                    "harness_version": manifest["harness_version"],
                    "reranking": "assay-transfer-contrastive",
                    "query_prior": "with_query_prior", "l1_molecules": 10,
                    "l1_min_contrast": 0, "morgan_primary_parent_width": 25,
                    "tasks": list(selected_tasks),
                    "evaluation_subset": list(evaluation_subsets),
                    "matrix_json": str(root / "matrix.json"),
                    "matrix_json_sha256": sha256_file(root / "matrix.json"),
                })
        elif args.study:
            write_json_atomic(root / "run.json", {
                "schema_version": "organized_study_run.v1",
                "study": args.study, "method": args.method,
                "run_id": root.name, "batch_id": root.name,
                "status": manifest["status"], "metric_status": "pending",
                "prompt_version": manifest["prompt_version"],
                "harness_version": manifest["harness_version"],
                "reranking": "assay-transfer-contrastive",
                "query_prior": "with_query_prior", "l1_molecules": 10,
                "l1_min_contrast": args.l1_min_contrasts[0],
                "morgan_primary_parent_width": 25,
                "tasks": list(selected_tasks), "evaluation_subset": list(evaluation_subsets),
                "matrix_json": str(root / "matrix.json"),
                "matrix_json_sha256": sha256_file(root / "matrix.json"),
            })
            from predict.harnesses.progressive.matrix import _refresh_results_catalog
            _refresh_results_catalog(args.results_root)
        if args.prepare_only:
            return 0

        pilot_client = None
        if not args.skip_pilots and args.execution_mode == "live":
            pilot_client = provider_client(
                args.parallelism,
                pilot=True,
                provider_pool_config=args.provider_pool_config,
                config=provider_config,
                max_tokens=args.max_tokens,
                timeout_s=args.request_timeout_s,
            )
            _run_pilot(prepared, pilot_client)
            if not args.continue_after_pilot:
                from predict.live import update_run

                for run_dir in live_runs.values():
                    update_run(
                        run_dir, status="awaiting_review", refresh_catalog=False,
                    )
                refresh_live_catalog(args.trace_root)
                manifest["status"] = "awaiting_review"
                write_json_atomic(root / "matrix.json", manifest)
                return 0
        client = provider_client(
            args.parallelism,
            provider_pool_config=args.provider_pool_config,
            config=provider_config,
            max_tokens=args.max_tokens,
            timeout_s=args.request_timeout_s,
        )
        failed = run_prepared_prompt_pool(
            prepared,
            max_workers=args.parallelism,
            max_stage_requeues=0,
            preparation_workers=args.preparation_workers,
            stage_client=client,
        )
        completion = {
            "status": "complete" if not failed else "incomplete",
            "failed_batches": failed,
            "pilot_provider_pool": (
                pilot_client.snapshot() if pilot_client is not None else None
            ),
            "provider_pool": client.snapshot(),
        }
        write_json_atomic(root / "completion.json", completion)
        manifest["status"] = completion["status"]
        write_json_atomic(root / "matrix.json", manifest)
        if args.study:
            run_paths = (
                [run_root / "run.json" for run_root in organized_runs.values()]
                if organized_runs else [root / "run.json"]
            )
            for run_path in run_paths:
                run = json.loads(run_path.read_text())
                run["status"] = completion["status"]
                run["metric_status"] = completion["status"]
                run["matrix_json_sha256"] = sha256_file(root / "matrix.json")
                write_json_atomic(run_path, run)
            from predict.harnesses.progressive.matrix import _refresh_results_catalog
            _refresh_results_catalog(args.results_root)
        from predict.live import update_run

        for run_dir in live_runs.values():
            update_run(
                run_dir,
                status="failed" if failed else "complete",
                refresh_catalog=False,
            )
        if args.execution_mode == "live":
            refresh_live_catalog(args.trace_root)
        return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
