"""Run the approved four-task measurement extraction and publish its mappings.

Skin consumes paid capacity first. The two independent OpenAI ledgers are then
shared across Ames, DILI, and Carcinogens in proportion to their current Stage 1
extraction-row counts. Any remaining rows are completed by the reviewed local
provider pool. This is a full-throughput workflow; it has no pilot path.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import socket
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from data.processing.evidence_library.versions.v10 import (
    build_measurement_resolution_mapping as runner,
)
from data.processing.llm_api import DEFAULT_ENV_FILE, resolve_api_key
from data.processing.paths import REPO_ROOT


TASKS = ("skin_reaction", "ames", "dili", "carcinogens")
LOCAL_TASKS = ("dili", "carcinogens")
PROPORTIONAL_TASKS = ("ames", "dili", "carcinogens")
OPENAI_KEYS = ("OPENAI_API_KEY_ONE", "OPENAI_API_KEY_TWO")
OPENAI_MODEL = "gpt-5.4-mini"
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
EXECUTION_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
PROVIDER_CONFIG = Path(
    "predict/api_client/providers/measurement_resolution_v10_three_endpoint_128.json"
)
LOCAL_PROVIDER_CONFIG = Path(
    "predict/api_client/providers/measurement_resolution_v10_dgx005_50002_550.json"
)
EXPECTED_PROVIDERS = {
    "dgx005_50001": ("http://dgx005:50001/v1", 128),
    "dgx011_50001": ("http://dgx011:50001/v1", 128),
    "dgx014_50002": ("http://dgx014:50002/v1", 128),
}
LOCAL_EXPECTED_PROVIDERS = {
    "dgx005_50002": ("http://dgx005:50002/v1", 550),
}
LOCAL_PARALLELISM = sum(capacity for _, capacity in LOCAL_EXPECTED_PROVIDERS.values())
INITIAL_MAX_COMPLETION_TOKENS = 8_192
RETRY_MAX_COMPLETION_TOKENS = 65_536
EXTENDED_RETRY_MAX_COMPLETION_TOKENS = 524_288
LOCAL_RETRY_MAX_ATTEMPTS = 3
DILI_POOL_CONFIG_MODULE = (
    "data.processing.evidence_library.versions.v10.tasks.dili."
    "starling_measurement_resolution_dgx005_50002_550"
)
EXECUTION_TIMEOUT_S = 3_600
DURABLE_RUNS_ROOT = (
    REPO_ROOT / "outputs/chembl_tool/measurement_resolution_generation"
)
LEDGER_MAX_TOKENS = 10_000_000
ACCEPTED_PAID_EXITS = {
    0,
    runner.PHASE_BUDGET_EXHAUSTED_EXIT_CODE,
    runner.BUDGET_EXHAUSTED_EXIT_CODE,
    runner.CREDENTIAL_UNAVAILABLE_EXIT_CODE,
    runner.RETRY_LIMIT_EXIT_CODE,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument(
        "--canonical-cache-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--provider-config", type=Path, default=PROVIDER_CONFIG)
    parser.add_argument("--expected-host", default="epyc-1-6")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="skip paid phases and run selected tasks on one interleaved local queue",
    )
    parser.add_argument(
        "--retry-only",
        action="store_true",
        help="refuse new rows and submit only failed rows already present in the cache",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=list(TASKS),
    )
    parser.add_argument(
        "--retry-max-completion-tokens",
        type=int,
        default=RETRY_MAX_COMPLETION_TOKENS,
    )
    parser.add_argument("--request-timeout-s", type=float, default=EXECUTION_TIMEOUT_S)
    parser.add_argument("--dili-retry-batch-size", type=int, default=20)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _validate_provider_config(
    path: Path,
    expected_providers: dict[str, tuple[str, int]] = EXPECTED_PROVIDERS,
    expected_model: str = DEEPSEEK_MODEL,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    providers = {
        str(item["name"]): (
            str(item["base_url"]),
            int(item["max_inflight"]),
            str(item["model"]),
        )
        for item in payload.get("providers") or ()
    }
    expected = {
        name: (base_url, capacity, expected_model)
        for name, (base_url, capacity) in expected_providers.items()
    }
    if providers != expected:
        raise ValueError(
            f"provider matrix mismatch: expected={expected}, found={providers}"
        )


def _preflight(args: argparse.Namespace) -> None:
    host = socket.gethostname().split(".", 1)[0]
    if host != args.expected_host:
        raise RuntimeError(f"run must execute on {args.expected_host}, found {host}")
    staging = args.staging_root.resolve()
    durable = (DURABLE_RUNS_ROOT / args.run_id).resolve()
    if staging != durable or args.canonical_cache_root.resolve() != durable:
        raise ValueError(f"staging and canonical roots must equal {durable}")
    if not args.env_file.is_file():
        raise FileNotFoundError(args.env_file)
    for credential in OPENAI_KEYS:
        resolve_api_key(
            "openai", env_file=args.env_file, credential_env=credential
        )
    _validate_provider_config(args.provider_config)


def _preflight_local(args: argparse.Namespace) -> None:
    host = socket.gethostname().split(".", 1)[0]
    if host != args.expected_host:
        raise RuntimeError(f"run must execute on {args.expected_host}, found {host}")
    staging = args.staging_root.resolve()
    durable = (DURABLE_RUNS_ROOT / args.run_id).resolve()
    if staging != durable or args.canonical_cache_root.resolve() != durable:
        raise ValueError(f"staging and canonical roots must equal {durable}")
    if tuple(args.tasks) != LOCAL_TASKS:
        raise ValueError(f"local combined queue requires tasks={LOCAL_TASKS}")
    if args.retry_max_completion_tokens not in {
        RETRY_MAX_COMPLETION_TOKENS,
        EXTENDED_RETRY_MAX_COMPLETION_TOKENS,
    }:
        raise ValueError(
            "retry token ceiling must be one of "
            f"{RETRY_MAX_COMPLETION_TOKENS}, {EXTENDED_RETRY_MAX_COMPLETION_TOKENS}"
        )
    if args.request_timeout_s <= 0:
        raise ValueError("request timeout must be positive")
    if not 1 <= args.dili_retry_batch_size <= 20:
        raise ValueError("DILI retry batch size must be between 1 and 20")
    if not args.retry_only:
        raise ValueError("OpenRouter successor execution requires --retry-only")
    _validate_provider_config(
        args.provider_config, LOCAL_EXPECTED_PROVIDERS, EXECUTION_MODEL
    )


def _ledger_path(root: Path, credential: str) -> Path:
    return root / "ledgers" / f"{credential.lower()}.json"


def _ledger_state(path: Path, epoch: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "max_tokens": LEDGER_MAX_TOKENS,
            "input_tokens": 0,
            "output_tokens": 0,
            "conservative_unreported_tokens": 0,
            "status": "active",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload["epochs"][epoch])


def _spent(state: dict[str, Any]) -> int:
    return sum(
        int(state.get(key) or 0)
        for key in (
            "input_tokens",
            "output_tokens",
            "conservative_unreported_tokens",
        )
    )


def _remaining(state: dict[str, Any]) -> int:
    if state.get("status") != "active":
        return 0
    return max(0, int(state["max_tokens"]) - _spent(state))


def _mapping_path(root: Path, task: str) -> Path:
    return root / "mappings" / task / "measurement_resolution.parquet"


def _common_args(args: argparse.Namespace, task: str) -> list[str]:
    return [
        "--task",
        task,
        "--cache-dir",
        str(args.staging_root / "cache"),
        "--mapping-path",
        str(_mapping_path(args.staging_root, task)),
        "--env-file",
        str(args.env_file),
        "--max-completion-tokens",
        "8192",
    ]


def _run_paid(
    args: argparse.Namespace,
    task: str,
    credential: str,
    phase_budget: int | None = None,
) -> int:
    argv = _common_args(args, task) + [
        "--model",
        OPENAI_MODEL,
        "--base-url",
        OPENAI_BASE_URL,
        "--provider",
        "openai",
        "--api-key-env",
        credential,
        "--token-ledger",
        str(_ledger_path(args.staging_root, credential)),
        "--budget-epoch",
        args.run_id,
        "--budget-max-tokens",
        str(LEDGER_MAX_TOKENS),
        "--workers",
        "8",
        "--defer-publication",
    ]
    if phase_budget is not None:
        argv += ["--phase-budget-max-tokens", str(phase_budget)]
    result = runner.main(argv)
    if result not in ACCEPTED_PAID_EXITS:
        raise RuntimeError(
            f"paid phase failed for {task}/{credential} with exit {result}"
        )
    return result


def _extraction_count(task: str) -> int:
    path = Path(runner.TaskConfig(task).DEFAULT_CLEANED_RECORDS)
    count = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=250_000, columns=["measurement_resolution_route"]
    ):
        count += sum(value == "extract" for value in batch.column(0).to_pylist())
    return count


def _proportional_quotas(total: int, counts: dict[str, int]) -> dict[str, int]:
    denominator = sum(counts.values())
    quotas: dict[str, int] = {}
    assigned = 0
    for task in PROPORTIONAL_TASKS[:-1]:
        quota = total * counts[task] // denominator
        quotas[task] = quota
        assigned += quota
    quotas[PROPORTIONAL_TASKS[-1]] = total - assigned
    return quotas


def _paid_usage(cache_path: Path) -> int:
    """Return reported GPT usage from a resumable task cache."""
    if not cache_path.is_file():
        return 0
    request_models: dict[str, str] = {}
    total = 0
    with cache_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            request_id = str(event.get("request_id") or "")
            if event.get("status") == "submitted":
                request_models[request_id] = str(event.get("model") or "")
                continue
            if (
                event.get("status") != "terminal"
                or request_models.get(request_id) != OPENAI_MODEL
            ):
                continue
            usage = event.get("usage") or {}
            total += int(usage.get("input_tokens") or 0)
            total += int(usage.get("output_tokens") or 0)
    return total


def _run_skin_first(args: argparse.Namespace) -> None:
    for credential in OPENAI_KEYS:
        if _remaining(
            _ledger_state(_ledger_path(args.staging_root, credential), args.run_id)
        ) == 0:
            continue
        if _run_paid(args, "skin_reaction", credential) == 0:
            return


def _run_proportional_paid(args: argparse.Namespace) -> dict[str, Any]:
    counts = {task: _extraction_count(task) for task in PROPORTIONAL_TASKS}
    ledger_states = {
        credential: _ledger_state(
            _ledger_path(args.staging_root, credential), args.run_id
        )
        for credential in OPENAI_KEYS
    }
    available = sum(_remaining(state) for state in ledger_states.values())
    paid_capacity = sum(int(state["max_tokens"]) for state in ledger_states.values())
    skin_usage = _paid_usage(
        args.staging_root / "cache" / "skin_reaction" / "requests.jsonl"
    )
    target_quotas = _proportional_quotas(
        max(0, paid_capacity - skin_usage), counts
    )
    spent_by_task = {
        task: _paid_usage(
            args.staging_root / "cache" / task / "requests.jsonl"
        )
        for task in PROPORTIONAL_TASKS
    }
    quotas = {
        task: max(0, target_quotas[task] - spent_by_task[task])
        for task in PROPORTIONAL_TASKS
    }
    # Reported cache usage can be slightly below ledger charges when a provider
    # omits usage. Give any resulting remainder to the final task instead of
    # leaving paid capacity idle.
    quotas[PROPORTIONAL_TASKS[-1]] += max(0, available - sum(quotas.values()))
    for task in PROPORTIONAL_TASKS:
        task_remaining = quotas[task]
        for credential in OPENAI_KEYS:
            ledger = _ledger_path(args.staging_root, credential)
            before = _ledger_state(ledger, args.run_id)
            allowance = min(task_remaining, _remaining(before))
            if allowance < 1:
                continue
            result = _run_paid(args, task, credential, allowance)
            after = _ledger_state(ledger, args.run_id)
            task_remaining = max(0, task_remaining - (_spent(after) - _spent(before)))
            if result == 0 or task_remaining == 0:
                break
    return {
        "extraction_rows": counts,
        "paid_token_target_quotas": target_quotas,
        "paid_tokens_already_spent": spent_by_task,
        "paid_token_remaining_quotas": quotas,
    }


def _run_fallback(args: argparse.Namespace, task: str) -> None:
    argv = _common_args(args, task) + [
        "--model",
        DEEPSEEK_MODEL,
        "--provider-pool-config",
        str(args.provider_config),
        "--parallelism",
        str(sum(capacity for _, capacity in EXPECTED_PROVIDERS.values())),
        "--no-token-ledger",
        "--require-complete",
        "--retry-failed",
        "--retry-max-attempts",
        "3",
    ]
    for _ in range(3):
        result = runner.main(argv)
        if result == 0:
            return
        if result not in {runner.RETRY_LIMIT_EXIT_CODE, 3}:
            raise RuntimeError(f"local fallback failed for {task} with exit {result}")
    raise RuntimeError(f"local fallback did not complete {task} after three passes")


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _publish_cache(source: Path, destination: Path) -> dict[str, str]:
    source_hashes = _tree_hashes(source)
    if destination.exists():
        if _tree_hashes(destination) != source_hashes:
            raise FileExistsError(f"canonical cache differs: {destination}")
        return source_hashes
    temporary = destination.with_name(destination.name + ".publishing")
    if temporary.exists():
        raise FileExistsError(f"stale publication directory: {temporary}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, temporary)
    if _tree_hashes(temporary) != source_hashes:
        raise IOError("cache publication hash verification failed")
    os.replace(temporary, destination)
    return source_hashes


def _publish_mapping(task: str, source: Path) -> dict[str, Any]:
    config = runner.TaskConfig(task)
    destination = Path(config.DEFAULT_MAPPING_PATH)
    source_manifest = source.with_suffix(".manifest.json")
    destination_manifest = destination.with_suffix(".manifest.json")
    if destination.exists() or destination_manifest.exists():
        if (
            destination.is_file()
            and destination_manifest.is_file()
            and _sha256(destination) == _sha256(source)
        ):
            config.validate_mapping_provenance(destination)
            return {
                "path": str(destination),
                "sha256": _sha256(destination),
                "rows": pq.ParquetFile(destination).metadata.num_rows,
            }
        raise FileExistsError(f"refusing to overwrite canonical mapping: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.publishing")
    shutil.copy2(source, temporary)
    if _sha256(temporary) != _sha256(source):
        raise IOError(f"mapping publication hash verification failed: {task}")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["mapping_path"] = str(destination)
    manifest["mapping_sha256"] = _sha256(temporary)
    temporary_manifest = destination.with_suffix(".manifest.json.publishing")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)
    os.replace(temporary_manifest, destination_manifest)
    config.validate_mapping_provenance(destination)
    return {
        "path": str(destination),
        "sha256": _sha256(destination),
        "rows": pq.ParquetFile(destination).metadata.num_rows,
    }


def _local_runner_args(args: argparse.Namespace, task: str) -> argparse.Namespace:
    argv = [
        "--task",
        task,
        "--cache-dir",
        str(args.staging_root / "cache"),
        "--mapping-path",
        str(_mapping_path(args.staging_root, task)),
        "--model",
        EXECUTION_MODEL,
        "--provider-pool-config",
        str(args.provider_config),
        "--parallelism",
        str(LOCAL_PARALLELISM),
        "--no-token-ledger",
        "--require-complete",
        "--max-completion-tokens",
        str(INITIAL_MAX_COMPLETION_TOKENS),
        "--retry-max-completion-tokens",
        str(args.retry_max_completion_tokens),
        "--request-timeout-s",
        str(args.request_timeout_s),
    ]
    if task == "dili":
        argv += ["--config-module", DILI_POOL_CONFIG_MODULE]
    return runner.parse_args(argv)


def _prepare_local_task(args: argparse.Namespace, task: str) -> dict[str, Any]:
    task_args = _local_runner_args(args, task)
    runner._validate_task_args(task_args)
    prepared = runner._prepare_plan(task_args)
    (
        config,
        records_path,
        mapping_path,
        base_mapping_path,
        profile_path,
        endpoint_profile,
        profile_digest,
        candidates,
        base_assignments,
        delta_candidates,
        planned_batches,
    ) = prepared
    runner._validate_inference_args(task_args)
    prompt_manifest = runner._reviewed_prompt(config)
    request_extra_body = getattr(config.module, "REQUEST_EXTRA_BODY", None)
    cache_path, cache = runner._run_cache(
        task_args,
        config,
        records_path,
        profile_digest,
        prompt_manifest,
        request_extra_body,
    )
    batches, early_exit = runner._remaining_batches(
        task_args,
        config,
        delta_candidates,
        planned_batches,
        cache,
        endpoint_profile,
        profile_digest,
    )
    if early_exit not in (None, runner.RETRY_LIMIT_EXIT_CODE):
        raise RuntimeError(f"cannot prepare {task}: exit={early_exit}")
    return {
        "task": task,
        "args": task_args,
        "config": config,
        "records_path": records_path,
        "mapping_path": mapping_path,
        "base_mapping_path": base_mapping_path,
        "profile_path": profile_path,
        "endpoint_profile": endpoint_profile,
        "profile_digest": profile_digest,
        "candidates": candidates,
        "base_assignments": base_assignments,
        "delta_candidates": delta_candidates,
        "batches": batches,
        "prompt_manifest": prompt_manifest,
        "request_extra_body": request_extra_body,
        "cache_path": cache_path,
        "cache": cache,
    }


def _interleaved_batches(contexts: list[dict[str, Any]]) -> deque[tuple[dict[str, Any], Any]]:
    queues = [(context, deque(context["batches"])) for context in contexts]
    output: deque[tuple[dict[str, Any], Any]] = deque()
    while queues:
        for context, queue in queues:
            if queue:
                output.append((context, queue.popleft()))
        queues = [(context, queue) for context, queue in queues if queue]
    return output


def _status(
    path: Path,
    *,
    args: argparse.Namespace,
    state: str,
    phase: str,
    contexts: list[dict[str, Any]],
    pool: Any,
    completed_batches: dict[str, int],
    total_batches: dict[str, int],
    in_flight: dict[Any, tuple[dict[str, Any], Any]],
    queued_counts: dict[str, int],
    error: str | None = None,
) -> None:
    inflight_counts = {
        task: sum(1 for context, _ in in_flight.values() if context["task"] == task)
        for task in args.tasks
    }
    _atomic_json(
        path,
        {
            "run_id": args.run_id,
            "state": state,
            "phase": phase,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "tasks": {
                context["task"]: {
                    "candidate_rows": len(context["delta_candidates"]),
                    "assigned_rows": len(context["cache"].assignments),
                    "completed_batches": completed_batches[context["task"]],
                    "total_batches": total_batches[context["task"]],
                    "in_flight_batches": inflight_counts[context["task"]],
                    "queued_batches": queued_counts[context["task"]],
                }
                for context in contexts
            },
            "provider_pool": pool.snapshot(),
            "error": error,
        },
    )


def _run_interleaved_pass(
    args: argparse.Namespace,
    contexts: list[dict[str, Any]],
    pool: Any,
    status_path: Path,
    phase: str,
) -> None:
    queued = _interleaved_batches(contexts)
    total_batches = {
        context["task"]: len(context["batches"]) for context in contexts
    }
    completed_batches = {task: 0 for task in args.tasks}
    queued_counts = dict(total_batches)
    in_flight: dict[Any, tuple[dict[str, Any], Any]] = {}
    last_status = 0.0
    completions_since_status = 0
    with ThreadPoolExecutor(max_workers=LOCAL_PARALLELISM) as executor:
        while queued or in_flight:
            while queued and len(in_flight) < LOCAL_PARALLELISM:
                context, batch = queued.popleft()
                queued_counts[context["task"]] -= 1
                future = runner._submit_request(
                    executor,
                    batch,
                    context["cache"],
                    None,
                    "",
                    EXECUTION_MODEL,
                    "provider_pool",
                    context["config"],
                    context["llm"],
                    {},
                    args.retry_max_completion_tokens,
                )
                in_flight[future] = (context, batch)
            done, _ = wait(
                tuple(in_flight), timeout=30, return_when=FIRST_COMPLETED
            )
            for future in done:
                context, batch = in_flight.pop(future)
                runner._complete_request(
                    future, batch, context["cache"], None
                )
                completed_batches[context["task"]] += 1
                completions_since_status += 1
            now = time.monotonic()
            if (
                completions_since_status >= 100
                or now - last_status >= 30
                or not in_flight
            ):
                _status(
                    status_path,
                    args=args,
                    state="running",
                    phase=phase,
                    contexts=contexts,
                    pool=pool,
                    completed_batches=completed_batches,
                    total_batches=total_batches,
                    in_flight=in_flight,
                    queued_counts=queued_counts,
                )
                last_status = now
                completions_since_status = 0


def _retry_ids(cache: Any, retry_model: str, event_start: int = 0) -> set[str]:
    retryable = runner.retryable_assignment_ids(cache)
    attempts: dict[str, int] = {}
    for event in cache.events[event_start:]:
        if event.get("status") != "submitted" or event.get("model") != retry_model:
            continue
        for record_id in event.get("row_ids") or ():
            key = str(record_id)
            attempts[key] = attempts.get(key, 0) + 1
    return {
        record_id
        for record_id in retryable
        if attempts.get(record_id, 0) < LOCAL_RETRY_MAX_ATTEMPTS
    }


def _retry_epoch_start(cache: Any, run_id: str) -> int:
    for index, event in enumerate(cache.events):
        if event.get("status") == "retry_epoch" and event.get("run_id") == run_id:
            return index + 1
    cache._append(
        {"cache_version": runner.CACHE_VERSION, "status": "retry_epoch", "run_id": run_id}
    )
    return len(cache.events)


def _plan_retries(context: dict[str, Any]) -> None:
    retry_ids = _retry_ids(
        context["cache"], context["args"].model, context["retry_epoch_start"]
    )
    if not retry_ids:
        context["batches"] = []
        return
    context["cache"].allow_retry(retry_ids)
    candidates = [
        row
        for row in context["delta_candidates"]
        if str(row["id"]) in retry_ids
    ]
    batches = runner.plan_batches(
        candidates,
        context["config"],
        attempted=context["cache"].attempted,
        endpoint_profile=context["endpoint_profile"],
        profile_digest=context["profile_digest"],
        model=context["args"].model,
        max_completion_tokens=RETRY_MAX_COMPLETION_TOKENS,
    )
    if context["task"] == "dili" and context["retry_batch_size"] < 20:
        batch_size = context["retry_batch_size"]
        batches = [
            runner.RequestBatch(
                request_id=f"{batch.request_id}_part_{offset // batch_size + 1}",
                source_id=batch.source_id,
                rows=batch.rows[offset : offset + batch_size],
                prompt=batch.prompt,
                max_completion_tokens=batch.max_completion_tokens,
                payload_rows=(
                    batch.api_rows[offset : offset + batch_size]
                    if batch.payload_rows is not None
                    else None
                ),
            )
            for batch in batches
            for offset in range(0, len(batch.rows), batch_size)
        ]
    context["batches"] = batches


def _materialize_local_task(context: dict[str, Any]) -> None:
    result = runner._publish_run(
        context["args"],
        context["candidates"],
        runner.SubmissionCache(context["cache_path"]),
        context["config"],
        context["mapping_path"],
        context["records_path"],
        context["profile_path"],
        context["args"].model,
        "provider_pool",
        context["base_assignments"],
        context["base_mapping_path"],
        context["request_extra_body"],
        context["prompt_manifest"],
    )
    if result != 0:
        raise RuntimeError(
            f"materialization failed for {context['task']} with exit {result}"
        )


def _run_local_interleaved(args: argparse.Namespace) -> int:
    _preflight_local(args)
    args.staging_root.mkdir(parents=True, exist_ok=True)
    status_path = args.staging_root / "status.json"
    contexts = [_prepare_local_task(args, task) for task in args.tasks]
    for context in contexts:
        context["retry_batch_size"] = (
            args.dili_retry_batch_size
            if context["task"] == "dili"
            else context["config"].BATCH_SIZE
        )
    if args.retry_only:
        unattempted = {
            context["task"]: len(context["batches"])
            for context in contexts
            if context["batches"]
        }
        if unattempted:
            raise RuntimeError(
                f"retry-only cache contains previously unattempted batches: {unattempted}"
            )
    seed_hashes = {
        context["task"]: _tree_hashes(Path(context["cache_path"]).parent)
        for context in contexts
    }
    for context in contexts:
        context["retry_epoch_start"] = _retry_epoch_start(
            context["cache"], args.run_id
        )
    pool, pool_receipt = runner._build_measurement_provider_pool(
        contexts[0]["args"], "low"
    )
    if (
        pool_receipt["effective_parallelism"] != LOCAL_PARALLELISM
        or any(check["status"] != "healthy" for check in pool_receipt["checks"])
    ):
        raise RuntimeError(f"not all reviewed providers are healthy: {pool_receipt}")
    for context in contexts:
        _atomic_json(
            Path(context["cache_path"]).parent / "provider_pool_receipt.json",
            pool_receipt,
        )
        context["llm"] = runner._provider_pool_llm(
            context["args"], context["cache"], "low", pool
        )
    try:
        if not args.retry_only:
            _run_interleaved_pass(args, contexts, pool, status_path, "initial")
        for attempt in range(1, LOCAL_RETRY_MAX_ATTEMPTS + 1):
            for context in contexts:
                _plan_retries(context)
            if not any(context["batches"] for context in contexts):
                break
            _run_interleaved_pass(
                args, contexts, pool, status_path, f"retry_{attempt}"
            )
        unresolved = {
            context["task"]: len(
                runner.retryable_assignment_ids(context["cache"])
            )
            for context in contexts
        }
        if any(unresolved.values()):
            raise RuntimeError(f"retry ceiling exhausted: {unresolved}")
        for context in contexts:
            _materialize_local_task(context)
        cache_hashes = {
            context["task"]: _publish_cache(
                Path(context["cache_path"]).parent,
                args.canonical_cache_root / "cache" / context["task"],
            )
            for context in contexts
        }
        mappings = {
            context["task"]: _publish_mapping(
                context["task"], context["mapping_path"]
            )
            for context in contexts
        }
        receipt = {
            "run_id": args.run_id,
            "status": "complete",
            "host": socket.gethostname(),
            "staging_root": str(args.staging_root),
            "canonical_cache_root": str(args.canonical_cache_root),
            "tasks": list(args.tasks),
            "provider_config": str(args.provider_config),
            "provider_config_sha256": _sha256(args.provider_config),
            "provider_pool": pool_receipt,
            "initial_max_completion_tokens": (
                None if args.retry_only else INITIAL_MAX_COMPLETION_TOKENS
            ),
            "retry_max_completion_tokens": args.retry_max_completion_tokens,
            "request_timeout_s": args.request_timeout_s,
            "max_submissions_per_row": LOCAL_RETRY_MAX_ATTEMPTS,
            "dili_retry_batch_size": args.dili_retry_batch_size,
            "seed_cache_hashes": seed_hashes,
            "cache_file_hashes": cache_hashes,
            "mappings": mappings,
        }
        _atomic_json(args.canonical_cache_root / "publication_receipt.json", receipt)
        _atomic_json(status_path, receipt)
        return 0
    except BaseException as error:
        _atomic_json(
            status_path,
            {
                "run_id": args.run_id,
                "status": "failed",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.canonical_cache_root is None:
        args.canonical_cache_root = Path(
            "outputs/chembl_tool/measurement_resolution_generation"
        ) / args.run_id
    if args.local_only:
        return _run_local_interleaved(args)
    _preflight(args)
    args.staging_root.mkdir(parents=True, exist_ok=True)
    schedule_path = args.staging_root / "schedule.json"
    _atomic_json(schedule_path, {"run_id": args.run_id, "status": "running"})
    _run_skin_first(args)
    allocation = _run_proportional_paid(args)
    for task in TASKS:
        _run_fallback(args, task)
    cache_hashes = _publish_cache(
        args.staging_root / "cache", args.canonical_cache_root / "cache"
    )
    ledger_hashes = _publish_cache(
        args.staging_root / "ledgers", args.canonical_cache_root / "ledgers"
    )
    mappings = {
        task: _publish_mapping(task, _mapping_path(args.staging_root, task))
        for task in TASKS
    }
    receipt = {
        "run_id": args.run_id,
        "status": "complete",
        "host": socket.gethostname(),
        "staging_root": str(args.staging_root),
        "canonical_cache_root": str(args.canonical_cache_root),
        "provider_config": str(args.provider_config),
        "provider_config_sha256": _sha256(args.provider_config),
        "provider_matrix": EXPECTED_PROVIDERS,
        "controller_sha256": _sha256(Path(__file__)),
        "runner_sha256": _sha256(Path(runner.__file__)),
        "openai_credentials": list(OPENAI_KEYS),
        "ledger_max_tokens_per_credential": LEDGER_MAX_TOKENS,
        **allocation,
        "cache_file_hashes": cache_hashes,
        "ledger_file_hashes": ledger_hashes,
        "mappings": mappings,
    }
    _atomic_json(args.canonical_cache_root / "publication_receipt.json", receipt)
    _atomic_json(schedule_path, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
