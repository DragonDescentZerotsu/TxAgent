"""Persistent single/group/final stage execution for the global prompt pool."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from tools.chembl_tool.common.identity_blind import (
    prepare_reasoning_retrieval,
    sanitize_identity_blind_branch_outputs,
)
from tools.chembl_tool.common.final_evidence_surface import SUMMARY_ONLY
from tools.chembl_tool.common.final_decision_prior import STANDARD_FINAL_DECISION
from tools.chembl_tool.common.openai_reasoning_client import (
    OpenAICompatibleClient,
    ToolServiceClient,
)
from tools.chembl_tool.common.json_utils import (
    atomic_output_path,
    write_json_atomic as _write_json_atomic,
    write_jsonl_atomic as _write_jsonl_atomic,
)
from tools.chembl_tool.common.reasoning_calls import load_frozen_single_analysis
from tools.chembl_tool.common.retrieval_ablation import load_reusable_group_outputs
from tools.chembl_tool.common.task_workflows.reasoning_batch import (
    BatchItem,
    PreparedBatch,
    _collect_result,
    _exclusive_run_lock,
    _prepare_final_only_run_dir,
    _read_json,
    _read_jsonl,
    _result_is_complete,
    _run_subprocess_with_logs,
    _single_run_command,
    _tail_text,
    _write_json,
)


STAGE_RUNTIME_VERSION = "reasoning_prompt_stage.v1"
SINGLE_STAGE = "single"
GROUP_STAGE = "group"
FINAL_STAGE = "final"


@dataclass
class StageState:
    prepared: PreparedBatch
    item: BatchItem
    run_id: str
    run_dir: Path
    retrieval: dict[str, Any]
    expected_group_ids: tuple[str, ...]
    _context: dict[str, Any] | None = None
    _context_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def key(self) -> tuple[str, int]:
        # batch_id is only unique inside one batch root. Matrix schedulers may
        # legitimately run the same condition over multiple isolated folds.
        return (str(self.prepared.batch_dir.resolve()), self.item.index)


@dataclass(frozen=True)
class StageJob:
    state: StageState
    stage: str
    group_id: str = ""
    attempt: int = 1

    @property
    def key(self) -> tuple[str, int, str, str]:
        return (*self.state.key, self.stage, self.group_id)


def prepare_stage_item(
    prepared: PreparedBatch,
    item: BatchItem,
) -> dict[str, Any]:
    """Materialize retrieval and the canonical manifest without issuing LLM calls."""
    run_id = f"{prepared.batch_id}_idx{item.index:05d}"
    run_dir = prepared.batch_run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = prepared.logs_dir / f"{run_id}.stdout.log"
    stderr_path = prepared.logs_dir / f"{run_id}.stderr.log"
    started = time.monotonic()
    with _exclusive_run_lock(run_dir):
        if not getattr(prepared.args, "skip_existing", False):
            _clear_previous_run_artifacts(run_dir)
        if getattr(prepared.args, "final_only_source_batch", ""):
            _prepare_final_only_run_dir(
                prepared.args,
                item.index,
                run_id,
                run_dir,
            )
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            returncode = 0
        else:
            command = _single_run_command(
                prepared.config,
                prepared.args,
                item.index,
                run_id,
                prepared.batch_run_root,
            )
            command.append("--prepare-only")
            returncode = _run_subprocess_with_logs(
                command,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                stream_logs=prepared.args.stream_logs,
                prefix=f"idx{item.index:05d}:prepare",
            )
            if returncode == 0:
                _initialize_run_manifest(prepared, item, run_id, run_dir)
    result: dict[str, Any] = {
        "query_index": item.index,
        "run_id": run_id,
        "status": "ok" if returncode == 0 else "error",
        "returncode": returncode,
        "latency_s": round(time.monotonic() - started, 3),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    if returncode != 0:
        result["error"] = _tail_text(stderr_path, stdout_path)
    return result


def _clear_previous_run_artifacts(run_dir: Path) -> None:
    for name in (
        "retrieval.json",
        "single_molecule_reasoning_output.json",
        "group_reasoning_outputs.jsonl",
        "group_reasoning_outputs_raw.jsonl",
        "final_reasoning_output.json",
        "trace_messages.jsonl",
        "manifest.json",
        "reuse.json",
        "stage_resume.json",
    ):
        (run_dir / name).unlink(missing_ok=True)


def load_stage_state(
    prepared: PreparedBatch,
    item: BatchItem,
) -> StageState | None:
    """Load a resumable run, or return None when a full pipeline seed is required."""
    run_id = f"{prepared.batch_id}_idx{item.index:05d}"
    run_dir = prepared.batch_run_root / run_id
    retrieval_path = run_dir / "retrieval.json"
    manifest_path = run_dir / "manifest.json"
    if not retrieval_path.exists() or not manifest_path.exists():
        return None
    try:
        retrieval = _read_json(retrieval_path)
        manifest = _read_json(manifest_path)
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if retrieval.get("status") != "ok":
        return None
    expected_group_ids = [
        str(group.get("group_id") or "")
        for group in retrieval.get("groups") or []
        if group.get("neighbors") and str(group.get("group_id") or "")
    ]
    max_groups = int(getattr(prepared.args, "max_groups", 0) or 0)
    if max_groups:
        expected_group_ids = expected_group_ids[:max_groups]
    expected_count = manifest.get("n_groups_with_neighbors")
    if expected_count is None or int(expected_count) != len(expected_group_ids):
        return None
    return StageState(
        prepared=prepared,
        item=item,
        run_id=run_id,
        run_dir=run_dir,
        retrieval=retrieval,
        expected_group_ids=tuple(expected_group_ids),
    )


def ready_stage_jobs(state: StageState) -> list[StageJob]:
    """Return independent ready branches, or final once every dependency is valid."""
    result = _collect_state_result(state)
    if _result_is_complete(result):
        return []
    jobs: list[StageJob] = []
    if result.get("single_status") != "ok" and _single_stage_dependency_ready(state):
        jobs.append(StageJob(state, SINGLE_STAGE))
    group_status = _group_status_by_id(state.run_dir)
    for group_id in state.expected_group_ids:
        if group_status.get(group_id) != "ok":
            jobs.append(StageJob(state, GROUP_STAGE, group_id))
    if not jobs and _earlier_stages_ready(state):
        jobs.append(StageJob(state, FINAL_STAGE))
    return jobs


def _single_stage_dependency_ready(state: StageState) -> bool:
    source_batch = str(
        getattr(state.prepared.args, "single_analysis_source_batch", "") or ""
    )
    if not source_batch:
        return True
    source_dir = _source_run_dir(Path(source_batch), state.item.index)
    return (source_dir / "single_molecule_reasoning_output.json").exists()


def execute_stage(job: StageJob) -> dict[str, Any]:
    if job.stage == SINGLE_STAGE:
        return _execute_single(job.state)
    if job.stage == GROUP_STAGE:
        return _execute_group(job.state, job.group_id)
    if job.stage == FINAL_STAGE:
        return _execute_final(job.state)
    raise ValueError(f"Unknown reasoning stage: {job.stage}")


def collect_stage_result(state: StageState) -> dict[str, Any]:
    result = _collect_state_result(state)
    complete = _result_is_complete(result)
    result.update(
        {
            "status": "ok" if complete else "error",
            "returncode": 0 if complete else 1,
            "latency_s": 0.0,
            "stdout_log": str(
                state.prepared.logs_dir / f"{state.run_id}.stdout.log"
            ),
            "stderr_log": str(
                state.prepared.logs_dir / f"{state.run_id}.stderr.log"
            ),
        }
    )
    return result


def _initialize_run_manifest(
    prepared: PreparedBatch,
    item: BatchItem,
    run_id: str,
    run_dir: Path,
) -> None:
    """Write the same per-run manifest fields as the task pipelines."""
    retrieval_path = run_dir / "retrieval.json"
    if not retrieval_path.exists():
        raise RuntimeError(f"prepare-only did not write retrieval.json: {run_dir}")
    retrieval = _read_json(retrieval_path)
    if retrieval.get("status") != "ok":
        raise RuntimeError(f"prepare-only retrieval is invalid: {run_dir}")
    module = importlib.import_module(prepared.config.pipeline_module)
    args = prepared.args
    groups = [group for group in retrieval.get("groups") or [] if group.get("neighbors")]
    if args.max_groups:
        groups = groups[: args.max_groups]
    raw_group_path = run_dir / "group_reasoning_outputs_raw.jsonl"
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "input_jsonl": args.input_jsonl,
        "query_index": item.index,
        "smiles_field": args.smiles_field,
        "experiment_mode": args.experiment_mode,
        "retrieval_source": args.retrieval_source,
        "neighbor_identity_policy": args.neighbor_identity_policy,
        "neighbor_selector": args.neighbor_selector,
        "neighbor_context_profile": args.neighbor_context_profile,
        "final_evidence_surface": getattr(args, "final_evidence_surface", SUMMARY_ONLY),
        "final_decision_profile": getattr(
            args,
            "final_decision_profile",
            STANDARD_FINAL_DECISION,
        ),
        "task_prompt_profile": getattr(args, "task_prompt_profile", ""),
        "retrieval_replay_source_run_dir": _configured_source_run_dir(
            args.retrieval_replay_source_batch,
            item.index,
        ),
        "prefetched_tool_replay_source_run_dir": _configured_source_run_dir(
            args.prefetched_tool_replay_source_batch,
            item.index,
        ),
        "identity_blind": args.identity_blind,
        "harness_prefetch_tools": args.identity_blind or args.harness_prefetch_tools,
        "neighbor_index": args.index if args.experiment_mode != "none" else "",
        "retrieval_evidence_source": retrieval.get("evidence_source", {}),
        "model": args.model,
        "base_url": args.base_url,
        "tool_service_url": args.tool_service_url,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "thinking": {"type": "enabled"} if args.enable_thinking else {"type": "disabled"},
        "group_tools_enabled": not args.disable_group_tools,
        "tool_execution_mode": (
            "harness_prefetch"
            if args.identity_blind or args.harness_prefetch_tools
            else "llm_function_call"
        ),
        "single_analysis_source_run_dir": _configured_source_run_dir(
            args.single_analysis_source_batch,
            item.index,
        ),
        "group_analysis_source_run_dir": _configured_source_run_dir(
            args.group_analysis_source_batch,
            item.index,
        ),
        "chembl_exact_context_enabled": False,
        "chembl_sqlite": str(getattr(module, "DEFAULT_CHEMBL_SQLITE", "")),
        "group_tool_names": [
            tool["function"]["name"]
            for tool in getattr(module, "GROUP_REASONING_TOOLS", [])
        ]
        if not args.disable_group_tools
        else [],
        "max_tool_rounds": args.max_tool_rounds,
        "top_k_per_group": args.top_k_per_group,
        "min_similarity": args.min_similarity,
        "n_groups_with_neighbors": len(groups),
        "expected_group_ids": [str(group.get("group_id") or "") for group in groups],
        "paths": {
            "retrieval": str(retrieval_path),
            "single_molecule_reasoning_output": str(
                run_dir / "single_molecule_reasoning_output.json"
            ),
            "group_reasoning_outputs": str(
                run_dir / "group_reasoning_outputs.jsonl"
            ),
            "group_reasoning_outputs_raw": (
                str(raw_group_path) if args.identity_blind else ""
            ),
            "final_reasoning_output": str(run_dir / "final_reasoning_output.json"),
            "trace_messages": str(run_dir / "trace_messages.jsonl"),
        },
        "query_label_for_eval_only": item.record.get(args.label_field),
        "stage_pool": {
            "version": STAGE_RUNTIME_VERSION,
            "artifact_contract": "legacy_paths_unchanged",
            "prepared_without_llm": True,
            "events": [],
        },
    }
    tier1_defaults = getattr(module, "DEFAULT_TIER1_REPLACEMENT_GROUPS", None)
    if tier1_defaults is not None:
        manifest.update(
            {
                "tier1_replacement_index": args.tier1_replacement_index,
                "tier1_replacement_groups": (
                    args.tier1_replacement_groups or tier1_defaults
                    if args.tier1_replacement_index
                    else []
                ),
                "groups": args.groups or [],
            }
        )
    _write_json_atomic(run_dir / "manifest.json", manifest)
    if not groups:
        _write_jsonl_atomic(run_dir / "group_reasoning_outputs.jsonl", [])
        if args.identity_blind:
            _write_jsonl_atomic(raw_group_path, [])
    _hydrate_configured_branch_reuse(prepared, item, retrieval, run_dir)


def _configured_source_run_dir(source_batch_value: str, query_index: int) -> str:
    if not source_batch_value:
        return ""
    return str(_source_run_dir(Path(source_batch_value), query_index))


def _hydrate_configured_branch_reuse(
    prepared: PreparedBatch,
    item: BatchItem,
    retrieval: dict[str, Any],
    run_dir: Path,
) -> None:
    args = prepared.args
    single_source = _configured_source_run_dir(
        args.single_analysis_source_batch,
        item.index,
    )
    if single_source and (
        Path(single_source) / "single_molecule_reasoning_output.json"
    ).exists():
        single_output = load_frozen_single_analysis(single_source)
        if single_output is not None:
            _write_json_atomic(
                run_dir / "single_molecule_reasoning_output.json",
                single_output,
            )
    group_source = _configured_source_run_dir(
        args.group_analysis_source_batch,
        item.index,
    )
    if group_source and Path(group_source).exists():
        group_outputs = load_reusable_group_outputs(
            group_source,
            retrieval,
            target_neighbor_context_profile=args.neighbor_context_profile,
        )
        if group_outputs:
            _write_jsonl_atomic(
                run_dir / "group_reasoning_outputs.jsonl",
                group_outputs,
            )


def _execute_single(state: StageState) -> dict[str, Any]:
    source_batch = str(state.prepared.args.single_analysis_source_batch or "")
    if source_batch:
        source_dir = _source_run_dir(Path(source_batch), state.item.index)
        output = load_frozen_single_analysis(str(source_dir))
        if output is None:
            raise RuntimeError(f"Frozen single stage was not loaded: {source_dir}")
    else:
        context = _stage_context(state)
        module = context["module"]
        client = _make_client(state)
        output = module._reason_single_molecule(
            client,
            module._llm_query_payload(context["reasoning_retrieval"]["query"]),
            module._clean_query_chembl_context(
                context["reasoning_retrieval"].get("query_chembl_context") or {}
            ),
            **_task_prompt_kwargs(state),
        )
    with _exclusive_run_lock(state.run_dir):
        _write_json_atomic(
            state.run_dir / "single_molecule_reasoning_output.json",
            output,
        )
        _invalidate_dependent_final(state.run_dir)
        _record_stage_event(state, SINGLE_STAGE, output.get("status", "error"))
    return output


def _execute_group(state: StageState, group_id: str) -> dict[str, Any]:
    context = _stage_context(state)
    module = context["module"]
    group = next(
        (
            row
            for row in context["reasoning_retrieval"].get("groups") or []
            if str(row.get("group_id") or "") == group_id
        ),
        None,
    )
    if group is None or not group.get("neighbors"):
        raise ValueError(f"Reasoning group is not available: {group_id}")
    client = _make_client(state)
    raw_output = module._reason_one_group(
        client,
        module._llm_query_payload(context["reasoning_retrieval"]["query"]),
        group,
        **_task_prompt_kwargs(state),
    )
    canonical_output = raw_output
    if state.prepared.args.identity_blind:
        canonical_output = sanitize_identity_blind_branch_outputs(
            [raw_output],
            state.retrieval,
        )[0]
    with _exclusive_run_lock(state.run_dir):
        if state.prepared.args.identity_blind:
            _merge_group_output(
                state.run_dir / "group_reasoning_outputs_raw.jsonl",
                raw_output,
            )
        _merge_group_output(
            state.run_dir / "group_reasoning_outputs.jsonl",
            canonical_output,
        )
        _invalidate_dependent_final(state.run_dir)
        _record_stage_event(
            state,
            GROUP_STAGE,
            canonical_output.get("status", "error"),
            group_id=group_id,
        )
    return canonical_output


def _execute_final(state: StageState) -> dict[str, Any]:
    if not _earlier_stages_ready(state):
        raise RuntimeError(f"Final dependencies are incomplete: {state.run_id}")
    context = _stage_context(state)
    module = context["module"]
    single_output = _read_json(
        state.run_dir / "single_molecule_reasoning_output.json"
    )
    group_outputs = _canonical_group_outputs(state)
    client = _make_client(state)
    final_surface = getattr(
        state.prepared.args,
        "final_evidence_surface",
        SUMMARY_ONLY,
    )
    final_kwargs = (
        {"final_evidence_surface": final_surface}
        if state.prepared.config.supports_shared_retrieval_contract
        and final_surface != SUMMARY_ONLY
        else {}
    )
    final_kwargs.update(_task_prompt_kwargs(state))
    final_decision_profile = getattr(
        state.prepared.args,
        "final_decision_profile",
        STANDARD_FINAL_DECISION,
    )
    if final_decision_profile != STANDARD_FINAL_DECISION:
        final_kwargs["final_decision_profile"] = final_decision_profile
    final_output = module._run_final_reasoning(
        client,
        context["reasoning_retrieval"],
        single_output,
        group_outputs,
        **final_kwargs,
    )
    final_path = state.run_dir / "final_reasoning_output.json"
    trace_path = state.run_dir / "trace_messages.jsonl"
    with _exclusive_run_lock(state.run_dir):
        # Publish the completion marker last. If trace serialization fails, a
        # resumable run must not look final-complete while its trace is absent.
        with atomic_output_path(final_path) as final_temp:
            _write_json(final_temp, final_output)
            if state.prepared.args.save_trace:
                with atomic_output_path(trace_path) as trace_temp:
                    module._write_trace_jsonl(
                        trace_temp,
                        query_record=state.item.record,
                        query_index=state.item.index,
                        smiles=(
                            "[identity_blind]"
                            if state.prepared.args.identity_blind
                            else str(
                                state.item.record.get(
                                    state.prepared.args.smiles_field
                                )
                                or ""
                            )
                        ),
                        single_output=single_output,
                        group_outputs=group_outputs,
                        final_output=final_output,
                    )
            else:
                trace_path.unlink(missing_ok=True)
        _record_stage_event(state, FINAL_STAGE, final_output.get("status", "error"))
    return final_output


def _task_prompt_kwargs(state: StageState) -> dict[str, str]:
    if not state.prepared.config.prompt_profile_option:
        return {}
    return {"prompt_profile": str(state.prepared.args.task_prompt_profile)}


def _invalidate_dependent_final(run_dir: Path) -> None:
    """A changed prerequisite makes the existing final and combined trace stale."""
    for name in ("final_reasoning_output.json", "trace_messages.jsonl"):
        (run_dir / name).unlink(missing_ok=True)


def _stage_context(state: StageState) -> dict[str, Any]:
    with state._context_lock:
        if state._context is not None:
            return state._context
        module = importlib.import_module(state.prepared.config.pipeline_module)
        load_env = getattr(module, "_load_env", None)
        if load_env is not None:
            load_env(Path(state.prepared.args.env_file))
        tool_service = ToolServiceClient(
            state.prepared.args.tool_service_url,
            timeout_s=state.prepared.args.timeout_s,
        )
        prefetched_replay = _prefetched_replay_run_dir(state)
        reasoning_retrieval = prepare_reasoning_retrieval(
            state.retrieval,
            tool_service,
            identity_blind=state.prepared.args.identity_blind,
            harness_prefetch_tools=state.prepared.args.harness_prefetch_tools,
            prefetched_tool_replay_run_dir=prefetched_replay,
            neighbor_context_profile=state.prepared.args.neighbor_context_profile,
        )
        state._context = {
            "module": module,
            "reasoning_retrieval": reasoning_retrieval,
        }
        return state._context


def _make_client(state: StageState) -> OpenAICompatibleClient:
    args = state.prepared.args
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {args.api_key_env}")
    return OpenAICompatibleClient(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tool_service_url=args.tool_service_url,
        enable_group_tools=not args.disable_group_tools,
        max_tool_rounds=args.max_tool_rounds,
        reasoning_effort=args.reasoning_effort,
        enable_thinking=args.enable_thinking,
    )


def _prefetched_replay_run_dir(state: StageState) -> str:
    source_batch = str(state.prepared.args.prefetched_tool_replay_source_batch or "")
    if source_batch:
        return str(_source_run_dir(Path(source_batch), state.item.index))
    manifest_path = state.run_dir / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    source = str(manifest.get("prefetched_tool_replay_source_run_dir") or "")
    return "" if source == str(state.run_dir) else source


def _source_run_dir(source_batch: Path, query_index: int) -> Path:
    source_run_id = f"{source_batch.name}_idx{query_index:05d}"
    return source_batch / "runs" / source_run_id


def _group_status_by_id(run_dir: Path) -> dict[str, str]:
    path = run_dir / "group_reasoning_outputs.jsonl"
    if not path.exists():
        return {}
    try:
        rows = _read_jsonl(path)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return {
        str(row.get("group_id") or ""): str(row.get("status") or "")
        for row in rows
    }


def _earlier_stages_ready(state: StageState) -> bool:
    single_path = state.run_dir / "single_molecule_reasoning_output.json"
    if not single_path.exists():
        return False
    try:
        if _read_json(single_path).get("status") != "ok":
            return False
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    statuses = _group_status_by_id(state.run_dir)
    return all(statuses.get(group_id) == "ok" for group_id in state.expected_group_ids)


def _canonical_group_outputs(state: StageState) -> list[dict[str, Any]]:
    path = state.run_dir / "group_reasoning_outputs.jsonl"
    rows = _read_jsonl(path) if path.exists() else []
    by_id = {str(row.get("group_id") or ""): row for row in rows}
    # The original task pipelines sort successful branches before final
    # synthesis. Preserve that prompt ordering exactly across staged resumes.
    return [by_id[group_id] for group_id in sorted(state.expected_group_ids)]


def _merge_group_output(path: Path, output: dict[str, Any]) -> None:
    rows = _read_jsonl(path) if path.exists() else []
    by_id = {str(row.get("group_id") or ""): row for row in rows}
    by_id[str(output.get("group_id") or "")] = output
    _write_jsonl_atomic(path, [by_id[key] for key in sorted(by_id)])


def _record_stage_event(
    state: StageState,
    stage: str,
    status: str,
    *,
    group_id: str = "",
) -> None:
    manifest_path = state.run_dir / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    stage_pool = manifest.setdefault(
        "stage_pool",
        {
            "version": STAGE_RUNTIME_VERSION,
            "artifact_contract": "legacy_paths_unchanged",
            "events": [],
        },
    )
    events = stage_pool.setdefault("events", [])
    events.append(
        {
            "stage": stage,
            "group_id": group_id,
            "status": status,
        }
    )
    if len(events) > 100:
        del events[:-100]
    _write_json_atomic(manifest_path, manifest)


def _collect_state_result(state: StageState) -> dict[str, Any]:
    return _collect_result(
        state.prepared.config,
        state.prepared.args,
        state.item,
        state.run_id,
        state.run_dir,
    )
