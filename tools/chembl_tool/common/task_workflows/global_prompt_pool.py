"""Matrix-wide ready pool for resumable molecular reasoning work.

The pool removes condition-level concurrency partitions while delegating prompt
construction, validation, trace serialization, and metric generation to the
existing task batch/pipeline contracts.  Incomplete runs use stage checkpoints,
so successful single/group branches survive retries and only the missing branch
work plus dependent final synthesis is executed again.
"""

from __future__ import annotations

import concurrent.futures
from collections import deque
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any

from tools.chembl_tool.common.task_workflows.reasoning_batch import (
    BatchItem,
    PreparedBatch,
    _parse_args,
    _resolve_item_future,
    _write_json,
    collect_completed_item,
    finalize_batch,
    prepare_batch,
)
from tools.chembl_tool.common.task_workflows.reasoning_stage_runtime import (
    FINAL_STAGE,
    SINGLE_STAGE,
    StageJob,
    StageState,
    collect_stage_result,
    execute_stage,
    load_stage_state,
    prepare_stage_item,
    ready_stage_jobs,
)


SCHEDULER_VERSION = "global_prompt_ready_pool.v1"


@dataclass(frozen=True)
class BatchCommand:
    experiment_name: str
    command: list[str]


@dataclass(frozen=True)
class PoolJob:
    experiment_name: str
    prepared: PreparedBatch
    item: BatchItem
    attempt: int = 1


@dataclass(frozen=True)
class ActiveJob:
    kind: str
    job: PoolJob | StageJob


@dataclass
class _PoolRuntime:
    results: dict[str, list[dict[str, Any]]]
    stage_states: dict[tuple[str, int], StageState]
    stage_experiments: dict[tuple[str, int], str]
    single_dependency_waiters: dict[tuple[str, int], set[tuple[str, int]]]
    terminal_stage_states: set[tuple[str, int]]
    finalized_stage_states: set[tuple[str, int]]
    completed_jobs: int
    total_jobs: int


def run_global_prompt_pool(
    commands: list[BatchCommand],
    *,
    max_workers: int,
    max_stage_requeues: int = 0,
    preparation_workers: int = 8,
) -> list[dict[str, Any]]:
    """Parse matrix batch commands, then use the canonical prepared-batch pool."""
    prepared_by_name = {
        spec.experiment_name: _prepare_command(
            spec,
            max_workers=max_workers,
            max_stage_requeues=max_stage_requeues,
        )
        for spec in commands
    }
    return run_prepared_prompt_pool(
        prepared_by_name,
        max_workers=max_workers,
        max_stage_requeues=max_stage_requeues,
        preparation_workers=preparation_workers,
    )


def run_prepared_prompt_pool(
    prepared_by_name: dict[str, PreparedBatch],
    *,
    max_workers: int,
    max_stage_requeues: int = 0,
    preparation_workers: int = 8,
) -> list[dict[str, Any]]:
    """Run prepared task/condition batches through one global ready queue."""
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if max_stage_requeues < 0:
        raise ValueError("max_stage_requeues must be non-negative")
    if preparation_workers < 1:
        raise ValueError("preparation_workers must be positive")
    effective_preparation_workers = min(max_workers, preparation_workers)

    for prepared in prepared_by_name.values():
        _mark_scheduler_manifest(
            prepared,
            max_workers=max_workers,
            max_stage_requeues=max_stage_requeues,
            preparation_workers=effective_preparation_workers,
        )
    runtime, pending, initial_stage_jobs = _initialize_pool_runtime(prepared_by_name)
    pending = _round_robin_jobs(pending)
    _log(
        f"scheduler={SCHEDULER_VERSION} batches={len(prepared_by_name)} "
        f"retrieval_preparations={len(pending)} stage_jobs={len(initial_stage_jobs)} "
        f"restored={runtime.completed_jobs} max_workers={max_workers} "
        f"preparation_workers={preparation_workers}"
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as prompt_executor, concurrent.futures.ThreadPoolExecutor(
        max_workers=preparation_workers
    ) as preparation_executor:
        futures: dict[
            concurrent.futures.Future[dict[str, Any]],
            ActiveJob,
        ] = {}
        active_stage_keys: set[tuple[str, int, str, str]] = set()
        for job in pending:
            futures[
                preparation_executor.submit(
                    prepare_stage_item,
                    job.prepared,
                    job.item,
                )
            ] = ActiveJob("prepare", job)
        for job in initial_stage_jobs:
            _submit_stage_job(prompt_executor, futures, active_stage_keys, job)

        while futures:
            done, _ = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                active = futures.pop(future)
                if active.kind == "prepare":
                    _handle_preparation_completion(
                        active,
                        future,
                        runtime=runtime,
                        preparation_executor=preparation_executor,
                        prompt_executor=prompt_executor,
                        futures=futures,
                        active_stage_keys=active_stage_keys,
                        max_stage_requeues=max_stage_requeues,
                    )
                    continue
                _handle_stage_completion(
                    active,
                    future,
                    runtime=runtime,
                    prompt_executor=prompt_executor,
                    futures=futures,
                    active_stage_keys=active_stage_keys,
                    max_stage_requeues=max_stage_requeues,
                )

        _collect_unfinalized_stage_results(runtime)

    failed: list[dict[str, Any]] = []
    for experiment_name, prepared in prepared_by_name.items():
        returncode = finalize_batch(prepared, runtime.results[experiment_name])
        if returncode:
            failed.append(
                {"experiment": experiment_name, "returncode": returncode}
            )
    return failed


def _initialize_pool_runtime(
    prepared_by_name: dict[str, PreparedBatch],
) -> tuple[_PoolRuntime, list[PoolJob], list[StageJob]]:
    results: dict[str, list[dict[str, Any]]] = {
        name: [] for name in prepared_by_name
    }
    runtime = _PoolRuntime(
        results=results,
        stage_states={},
        stage_experiments={},
        single_dependency_waiters={},
        terminal_stage_states=set(),
        finalized_stage_states=set(),
        completed_jobs=0,
        total_jobs=sum(len(prepared.items) for prepared in prepared_by_name.values()),
    )
    pending: list[PoolJob] = []
    initial_stage_jobs: list[StageJob] = []
    for name, prepared in prepared_by_name.items():
        for item in prepared.items:
            complete = collect_completed_item(prepared, item)
            if complete is not None:
                results[name].append(complete)
                runtime.completed_jobs += 1
                continue
            state = (
                load_stage_state(prepared, item)
                if prepared.args.skip_existing
                else None
            )
            if state is None:
                pending.append(PoolJob(name, prepared, item))
                continue
            _register_runtime_stage_state(runtime, state, name)
            initial_stage_jobs.extend(ready_stage_jobs(state))
    return runtime, pending, initial_stage_jobs


def _register_runtime_stage_state(
    runtime: _PoolRuntime,
    state: StageState,
    experiment_name: str,
) -> None:
    _register_stage_state(
        state,
        experiment_name,
        runtime.stage_states,
        runtime.stage_experiments,
        runtime.single_dependency_waiters,
    )


def _record_result(
    runtime: _PoolRuntime,
    experiment_name: str,
    query_index: int,
    result: dict[str, Any],
) -> None:
    runtime.results[experiment_name].append(result)
    runtime.completed_jobs += 1
    _log_progress(
        runtime.completed_jobs,
        runtime.total_jobs,
        experiment_name,
        query_index,
        result.get("status", "error"),
    )


def _handle_preparation_completion(
    active: ActiveJob,
    future: concurrent.futures.Future[dict[str, Any]],
    *,
    runtime: _PoolRuntime,
    preparation_executor: concurrent.futures.ThreadPoolExecutor,
    prompt_executor: concurrent.futures.ThreadPoolExecutor,
    futures: dict[concurrent.futures.Future[dict[str, Any]], ActiveJob],
    active_stage_keys: set[tuple[str, int, str, str]],
    max_stage_requeues: int,
) -> None:
    job = active.job
    assert isinstance(job, PoolJob)
    result = _resolve_item_future(job.prepared, job.item, future)
    if result.get("status") == "ok":
        state = load_stage_state(job.prepared, job.item)
        if state is not None:
            _register_runtime_stage_state(runtime, state, job.experiment_name)
            for stage_job in ready_stage_jobs(state):
                _submit_stage_job(
                    prompt_executor,
                    futures,
                    active_stage_keys,
                    stage_job,
                )
            return
        result = {
            **result,
            "status": "error",
            "returncode": 1,
            "error": "prepare-only completed without a resumable stage manifest",
        }
    if job.attempt <= max_stage_requeues:
        retry = PoolJob(
            job.experiment_name,
            job.prepared,
            job.item,
            attempt=job.attempt + 1,
        )
        futures[
            preparation_executor.submit(
                prepare_stage_item,
                retry.prepared,
                retry.item,
            )
        ] = ActiveJob("prepare", retry)
        _log(
            f"requeue preparation experiment={job.experiment_name} "
            f"index={job.item.index} attempt={retry.attempt}"
        )
        return
    _record_result(runtime, job.experiment_name, job.item.index, result)


def _handle_stage_completion(
    active: ActiveJob,
    future: concurrent.futures.Future[dict[str, Any]],
    *,
    runtime: _PoolRuntime,
    prompt_executor: concurrent.futures.ThreadPoolExecutor,
    futures: dict[concurrent.futures.Future[dict[str, Any]], ActiveJob],
    active_stage_keys: set[tuple[str, int, str, str]],
    max_stage_requeues: int,
) -> None:
    stage_job = active.job
    assert isinstance(stage_job, StageJob)
    active_stage_keys.discard(stage_job.key)
    try:
        stage_output = future.result()
    except Exception as exc:  # noqa: BLE001 - persist other ready work.
        stage_output = {"status": "error", "error": str(exc)}
    state = stage_job.state
    state_key = state.key
    if stage_output.get("status") != "ok":
        _handle_stage_failure(
            stage_job,
            stage_output,
            runtime=runtime,
            prompt_executor=prompt_executor,
            futures=futures,
            active_stage_keys=active_stage_keys,
            max_stage_requeues=max_stage_requeues,
        )
        return
    if stage_job.stage == FINAL_STAGE:
        experiment_name = runtime.stage_experiments[state_key]
        runtime.finalized_stage_states.add(state_key)
        _record_result(
            runtime,
            experiment_name,
            state.item.index,
            collect_stage_result(state),
        )
        return
    wake_state_keys = {state_key}
    if stage_job.stage == SINGLE_STAGE:
        wake_state_keys.update(
            runtime.single_dependency_waiters.get(state_key, set())
        )
    _submit_ready_stage_keys(
        prompt_executor,
        futures,
        active_stage_keys,
        runtime.stage_states,
        runtime.terminal_stage_states,
        runtime.finalized_stage_states,
        wake_state_keys,
    )


def _handle_stage_failure(
    stage_job: StageJob,
    stage_output: dict[str, Any],
    *,
    runtime: _PoolRuntime,
    prompt_executor: concurrent.futures.ThreadPoolExecutor,
    futures: dict[concurrent.futures.Future[dict[str, Any]], ActiveJob],
    active_stage_keys: set[tuple[str, int, str, str]],
    max_stage_requeues: int,
) -> None:
    state = stage_job.state
    error_text = str(
        stage_output.get("error")
        or stage_output.get("status")
        or "unknown stage failure"
    )
    _log(
        f"stage failure batch={state.prepared.batch_id} "
        f"index={state.item.index} stage={stage_job.stage} "
        f"group={stage_job.group_id} error={error_text[:500]}"
    )
    if stage_job.attempt <= max_stage_requeues:
        retry = StageJob(
            state,
            stage_job.stage,
            stage_job.group_id,
            attempt=stage_job.attempt + 1,
        )
        _submit_stage_job(
            prompt_executor,
            futures,
            active_stage_keys,
            retry,
        )
        _log(
            f"requeue stage batch={state.prepared.batch_id} "
            f"index={state.item.index} stage={stage_job.stage} "
            f"group={stage_job.group_id} attempt={retry.attempt}"
        )
        return
    runtime.terminal_stage_states.add(state.key)


def _collect_unfinalized_stage_results(runtime: _PoolRuntime) -> None:
    for state_key, state in runtime.stage_states.items():
        if state_key in runtime.finalized_stage_states:
            continue
        experiment_name = runtime.stage_experiments[state_key]
        _record_result(
            runtime,
            experiment_name,
            state.item.index,
            collect_stage_result(state),
        )


def _submit_stage_job(
    executor: concurrent.futures.ThreadPoolExecutor,
    futures: dict[concurrent.futures.Future[dict[str, Any]], ActiveJob],
    active_stage_keys: set[tuple[str, int, str, str]],
    job: StageJob,
) -> None:
    if job.key in active_stage_keys:
        return
    active_stage_keys.add(job.key)
    futures[executor.submit(execute_stage, job)] = ActiveJob("stage", job)


def _submit_ready_stage_keys(
    executor: concurrent.futures.ThreadPoolExecutor,
    futures: dict[concurrent.futures.Future[dict[str, Any]], ActiveJob],
    active_stage_keys: set[tuple[str, int, str, str]],
    stage_states: dict[tuple[str, int], StageState],
    terminal_stage_states: set[tuple[str, int]],
    finalized_stage_states: set[tuple[str, int]],
    state_keys: set[tuple[str, int]],
) -> None:
    """Wake only the changed sample and its sample-level dependents."""
    for state_key in state_keys:
        if state_key in terminal_stage_states or state_key in finalized_stage_states:
            continue
        state = stage_states.get(state_key)
        if state is None:
            continue
        for ready_job in ready_stage_jobs(state):
            _submit_stage_job(
                executor,
                futures,
                active_stage_keys,
                ready_job,
            )


def _register_stage_state(
    state: StageState,
    experiment_name: str,
    stage_states: dict[tuple[str, int], StageState],
    stage_experiments: dict[tuple[str, int], str],
    single_dependency_waiters: dict[
        tuple[str, int], set[tuple[str, int]]
    ],
) -> None:
    stage_states[state.key] = state
    stage_experiments[state.key] = experiment_name
    source_batch = str(
        getattr(state.prepared.args, "single_analysis_source_batch", "") or ""
    )
    if source_batch:
        source_key = (str(Path(source_batch).resolve()), state.item.index)
        single_dependency_waiters.setdefault(source_key, set()).add(state.key)


def _log_progress(
    completed: int,
    total: int,
    experiment: str,
    index: int,
    status: str,
) -> None:
    _log(
        f"progress={completed}/{total} experiment={experiment} "
        f"index={index} status={status}"
    )


def _prepare_command(
    spec: BatchCommand,
    *,
    max_workers: int,
    max_stage_requeues: int,
) -> PreparedBatch:
    command = spec.command
    if len(command) < 4 or command[1] != "-m":
        raise ValueError(
            f"Expected a Python -m batch command for {spec.experiment_name}: {command}"
        )
    module = importlib.import_module(command[2])
    config = getattr(module, "CONFIG", None)
    if config is None:
        raise ValueError(f"Batch module does not export CONFIG: {command[2]}")
    args = _parse_args(config, command[3:])
    args.parallelism = max_workers
    return prepare_batch(config, args)


def _mark_scheduler_manifest(
    prepared: PreparedBatch,
    *,
    max_workers: int,
    max_stage_requeues: int,
    preparation_workers: int,
) -> None:
    prepared.batch_dir.mkdir(parents=True, exist_ok=True)
    prepared.manifest["scheduler"] = {
        "version": SCHEDULER_VERSION,
        "global_max_workers": max_workers,
        "max_stage_requeues": max_stage_requeues,
        "retrieval_preparation_workers": preparation_workers,
        "resource_partition": "shared_across_tasks_and_conditions",
        "artifact_contract": "legacy_batch_and_per_run_paths_unchanged",
    }
    _write_json(prepared.batch_dir / "manifest.json", prepared.manifest)


def _round_robin_jobs(jobs: list[PoolJob]) -> list[PoolJob]:
    """Interleave batches for fairness without reserving per-batch capacity."""
    by_name: dict[str, deque[PoolJob]] = {}
    for job in jobs:
        by_name.setdefault(job.experiment_name, deque()).append(job)
    ordered: list[PoolJob] = []
    names = list(by_name)
    while names:
        next_names: list[str] = []
        for name in names:
            queue = by_name[name]
            ordered.append(queue.popleft())
            if queue:
                next_names.append(name)
        names = next_names
    return ordered


def _log(message: str) -> None:
    print(f"[global_prompt_pool] {message}", file=sys.stderr, flush=True)
