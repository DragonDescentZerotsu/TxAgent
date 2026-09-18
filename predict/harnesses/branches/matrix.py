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
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

from data.processing.gold_labels.conditioned_benchmark import split_path
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
    collect_stage_result,
    execute_stage,
    load_stage_state,
    prepare_stage_item,
    ready_stage_jobs,
)
from predict.harnesses.branches.inference import load_frozen_single_analysis
from predict.api_client.pool import (
    DEFAULT_PROVIDER_POOL_CONFIG,
    OpenAIProviderPool,
    ProviderPoolConfig,
    build_provider_pool,
    load_provider_pool_config,
    primary_capacity,
    select_healthy_providers,
)
from predict.retrieval.assay_reranking.cache_matched import (
    DEFAULT_CACHE_BUNDLE,
    DEFAULT_GOLD_CONTEXT_MAPPING,
)
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic


MATRIX_VERSION = "joseph_flat_matrix.v3"
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEFAULT_PROVIDER_CONFIG = DEFAULT_PROVIDER_POOL_CONFIG
TASKS = ("bbb_martins", "bioavailability_ma")
RECORDS_PER_LEVEL = 10
MAX_TOKENS = 262_144
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
) -> argparse.Namespace:
    input_jsonl = split_path(task, "valid").with_name(
        "valid_molecule_condition_labels.jsonl"
    ).resolve()
    return argparse.Namespace(
        harness_version=flat.PUBLIC_HARNESS_VERSION,
        prompt_version=flat.JOSEPH_PROMPT_VERSION,
        task=task,
        reranking=reranking,
        assay_transfer_cache=DEFAULT_CACHE_BUNDLE.resolve(),
        record_pool=record_pool,
        evaluation_subset="valid",
        input_jsonl=input_jsonl,
        evidence_library=(Path("data/evidence_libraries") / task / "v10").resolve(),
        level_mapper=Path("data/evidence_libraries/level_mappings.v1.json").resolve(),
        gold_context_mapping=DEFAULT_GOLD_CONTEXT_MAPPING.resolve(),
        allow_frozen_l1_vote_scores=task == "bbb_martins",
        l1_molecules=10,
        l1_records_per_molecule=10,
        records_per_level=records_per_level,
        ranking_tie_seed=0,
        max_level=0,
        indices=None,
        start=0,
        limit=limit,
        batch_root=(root / task).resolve(),
        batch_id=(
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
    command = [
        sys.executable,
        "-m",
        "predict.harnesses.branches",
        "--organization",
        "flat",
        "--task",
        task,
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
        flat.JOSEPH_PROMPT_VERSION,
        "--flat-reranking",
        args.reranking,
        "--retrieval-strategy",
        "morgan_fingerprint",
        "--neighbor-identity-policy",
        "scaffold_disjoint",
        "--min-similarity",
        "0",
        "--disable-flat-tools",
        "--single-analysis-source-batch",
        str(_single_batch(task)),
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
        "--start",
        "0",
        "--limit",
        str(args.limit),
    ]
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


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
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
    parser.add_argument("--provider-pool-config", type=Path, default=DEFAULT_PROVIDER_CONFIG)
    parser.add_argument("--trace-root", type=Path, default=Path("outputs/paper/live"))
    parser.add_argument(
        "--execution-mode", choices=("live", "throughput"), default="throughput"
    )
    parser.add_argument("--continue-after-pilot", action="store_true")
    parser.add_argument("--live-run-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--request-timeout-s", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-pilots", action="store_true")
    args = parser.parse_args(raw_argv)
    if args.parallelism is not None and args.parallelism < 1:
        parser.error("--parallelism must be positive")
    if min(
        args.preparation_workers,
        args.records_per_level,
        args.request_timeout_s,
        args.max_tokens,
    ) < 1 or args.limit < 0:
        parser.error("worker, record, token, and timeout values must be positive")
    if len(args.conditions) != len(set(args.conditions)):
        parser.error("--conditions cannot contain duplicates")
    conditions = [tuple(value.split(":", 1)) for value in args.conditions]
    args.provider_pool_config = args.provider_pool_config.resolve()
    args.trace_root = args.trace_root.resolve()
    candidate_config = load_provider_pool_config(args.provider_pool_config)
    args.requested_parallelism = args.parallelism
    args.endpoint_selection = None
    if args.prepare_only:
        args.parallelism = args.parallelism or 1
        provider_config = candidate_config
    else:
        if args.parallelism is None:
            parser.error("full-batch inference requires explicit --parallelism")
        selection = select_healthy_providers(candidate_config, args.parallelism)
        args.endpoint_selection = selection.public_dict()
        args.parallelism = selection.effective_parallelism
        provider_config = selection.config

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "launcher.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare_only:
            execution = {
                "model": MODEL,
                "base_url": "",
                "api_key_env": "",
                "timeout_s": args.request_timeout_s,
                "reasoning_effort": "",
                "enable_thinking": False,
            }
            endpoint_receipts = []
        else:
            execution, endpoint_receipts = _provider_execution(
                provider_config,
                args.request_timeout_s,
            )
        commands = []
        single_receipts = [
            verify_single_reuse(
                task,
                split_path(task, "valid").with_name(
                    "valid_molecule_condition_labels.jsonl"
                ).resolve(),
            )
            for task in TASKS
        ]
        selection_receipts = []
        for reranking, record_pool in conditions:
            for task in TASKS:
                selection_args = _selection_args(
                    task,
                    root,
                    limit=args.limit,
                    reranking=reranking,
                    record_pool=record_pool,
                    records_per_level=args.records_per_level,
                )
                source, selection = flat._materialize_cache_matched_retrievals(
                    selection_args
                )
                selection_receipts.append(
                    {
                        "task": task,
                        "reranking": reranking,
                        "record_pool": record_pool,
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
            max_workers=args.parallelism,
            max_stage_requeues=0,
        )
        live_runs = {}
        if not args.prepare_only:
            from predict.live import create_run, update_run

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
                        "prompt_version": flat.JOSEPH_PROMPT_VERSION,
                        "evaluation_subset": "valid",
                        "pilot_size": 3,
                        "output_root": str(batch.batch_dir),
                        "matrix_output_root": str(root),
                    },
                    requested_id=args.live_run_id,
                    execution_mode=args.execution_mode,
                )
                batch.args.trace_root = str(args.trace_root)
                batch.args.live_run_id = run_dir.name
                batch.args.live_method = method
                live_runs[batch_id] = run_dir
                update_run(
                    run_dir,
                    resume_command=[*command, "--live-run-id", run_dir.name],
                )
        manifest = {
            "version": MATRIX_VERSION,
            "status": "prepared" if args.prepare_only else "running",
            "tasks": list(TASKS),
            "evaluation_subset": "valid",
            "harness_version": flat.PUBLIC_HARNESS_VERSION,
            "prompt_version": flat.JOSEPH_PROMPT_VERSION,
            "conditions": [
                {"reranking": reranking, "record_pool": record_pool}
                for reranking, record_pool in conditions
            ],
            "records_per_level": args.records_per_level,
            "l1_molecules": 10,
            "l1_records_per_molecule": 10,
            "max_level_by_task": {"bbb_martins": 5, "bioavailability_ma": 6},
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
            "endpoint_allocations": dict(
                endpoint_allocations(
                    args.parallelism, args.provider_pool_config,
                    config=provider_config,
                )
            ) if not args.prepare_only else {},
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
                    update_run(run_dir, status="awaiting_review")
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
        from predict.live import update_run

        for run_dir in live_runs.values():
            update_run(
                run_dir,
                status="failed" if failed else "complete",
            )
        return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
