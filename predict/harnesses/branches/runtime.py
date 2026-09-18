"""Execute persistent single, group, and final model stages.

``prepare_stage_item`` first invokes the task pipeline in preparation-only mode
to materialize retrieval and a run manifest. ``ready_stage_jobs`` examines the
resulting checkpoints and exposes only dependency-ready work to the global
scheduler. ``execute_stage`` runs exactly one single, group, or final model
call. Each successful stage is written before its dependents become eligible,
which makes interrupted batches resumable without repeating completed calls.

The runtime consumes already organized retrieval groups. It may prefetch tools,
sanitize identity-blind payloads, and call task prompt functions, but it must
not decide whether the experiment uses direct, flat, or mechanism-family
evidence.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import importlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from predict.harnesses.branches.visibility import (
    expose_neighbor_smiles_only,
    prepare_reasoning_retrieval,
    query_without_prefetched_tools,
    sanitize_identity_blind_branch_outputs,
)
from predict.harnesses.branches.reasoning.final_evidence import SUMMARY_ONLY
from predict.harnesses.branches.reasoning.final_decision import STANDARD_FINAL_DECISION
from predict.api_client.client import OpenAICompatibleClient
from predict.tools.client import ToolServiceClient
from predict.traces.io import write_trace
from predict.utils.json import (
    atomic_output_path,
    write_json_atomic as _write_json_atomic,
    write_jsonl_atomic as _write_jsonl_atomic,
)
from predict.harnesses.branches.inference import load_frozen_single_analysis
from predict.harnesses.branches.flat import (
    CONTEXT_V4_PROMPT_VERSION,
    CONTEXT_PROMPT_VERSIONS,
    CONTEXT_V5_PROMPT_VERSION,
    build_flat_context_request,
    derive_flat_claim_evidence,
    flat_context_validation,
)
from predict.harnesses.branches.prompt import attach_external_condition
from predict.harnesses.branches.analogous_flat_prompt import (
    PROMPT_IDENTITY_VIEW,
    prompt_provenance as analogous_flat_prompt_provenance,
    reason_final as reason_analogous_flat_final,
    reason_group as reason_analogous_flat_group,
)
from predict.llm_io.response import (
    call_with_json_validation,
    structured_response_is_valid,
    validated_branch_content,
)
from predict.harnesses.branches.artifacts import load_reusable_group_outputs
from predict.harnesses.branches.runner import (
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
            synchronize_configured_single_reuse(
                prepared,
                item,
                run_dir,
            )
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
        "request.json",
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
    expected_group_ids = [] if _flat_one_call_args(prepared.args) else [
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
    state = StageState(
        prepared=prepared,
        item=item,
        run_id=run_id,
        run_dir=run_dir,
        retrieval=retrieval,
        expected_group_ids=tuple(expected_group_ids),
    )
    _export_existing_traces(state)
    return state


def ready_stage_jobs(state: StageState) -> list[StageJob]:
    """Return independent ready branches, or final once every dependency is valid."""
    result = _collect_state_result(state)
    if _result_is_complete(result):
        return []
    jobs: list[StageJob] = []
    expected_single_status = _expected_single_status(state)
    if (
        expected_single_status == "ok"
        and result.get("single_status") != "ok"
        and _single_stage_dependency_ready(state)
    ):
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


def execute_stage(
    job: StageJob,
    stage_client: Any | None = None,
) -> dict[str, Any]:
    if job.stage == SINGLE_STAGE:
        return _execute_single(job.state)
    if job.stage == GROUP_STAGE:
        return _execute_group(job.state, job.group_id, stage_client)
    if job.stage == FINAL_STAGE:
        return _execute_final(job.state, stage_client)
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
    analogous_reasoning_only = bool(
        getattr(args, "analogous_reasoning_only", False)
    )
    flat_one_call = _flat_one_call_args(args)
    groups = [] if flat_one_call else [
        group for group in retrieval.get("groups") or [] if group.get("neighbors")
    ]
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
        "flat_prompt_version": getattr(args, "flat_prompt_version", ""),
        "harness_version": prepared.manifest.get("harness_version", ""),
        "flat_reranking": prepared.manifest.get("flat_reranking", ""),
        "evaluation_subset": prepared.manifest.get("evaluation_subset", ""),
        "record_pool": prepared.manifest.get("record_pool", ""),
        "cache_pool": prepared.manifest.get("cache_pool", ""),
        "evidence_projection": prepared.manifest.get("evidence_projection", ""),
        "extra_details_policy": prepared.manifest.get("extra_details_policy", ""),
        "molecule_name_visible": prepared.manifest.get("molecule_name_visible", ""),
        "allow_frozen_l1_vote_scores": prepared.manifest.get(
            "allow_frozen_l1_vote_scores", False
        ),
        "flat_selection_contract_sha256": prepared.manifest.get(
            "flat_selection_contract_sha256", ""
        ),
        "flat_selection_manifest": prepared.manifest.get(
            "flat_selection_manifest", {}
        ),
        "group_prompt_version": prepared.manifest.get("group_prompt_version", ""),
        "group_prompt_provenance": prepared.manifest.get(
            "group_prompt_provenance", {}
        ),
        "retrieval_replay_source_run_dir": _configured_source_run_dir(
            args.retrieval_replay_source_batch,
            item.index,
        ),
        "prefetched_tool_replay_source_run_dir": _configured_source_run_dir(
            args.prefetched_tool_replay_source_batch,
            item.index,
        ),
        "identity_blind": args.identity_blind,
        "disable_flat_tools": bool(getattr(args, "disable_flat_tools", False)),
        "analogous_reasoning_only": analogous_reasoning_only,
        "flat_one_call": flat_one_call,
        "flat_layout": getattr(args, "flat_layout", ""),
        "flat_query_prior": getattr(args, "flat_query_prior", ""),
        "single_branch_execution": (
            "omitted"
            if analogous_reasoning_only
            or flat_one_call and args.flat_query_prior == "none"
            else "reused"
            if flat_one_call
            else "executed_or_reused"
        ),
        "single_branch_omission_reason": (
            "analogous_reasoning_only"
            if analogous_reasoning_only
            else "query_prior_disabled"
            if flat_one_call and args.flat_query_prior == "none"
            else ""
        ),
        "query_tool_execution": (
            "omitted" if analogous_reasoning_only or flat_one_call else "enabled"
        ),
        "group_query_tool_instruction_policy": (
            "omitted.v1"
            if analogous_reasoning_only or getattr(args, "disable_flat_tools", False)
            else "standard.v1"
        ),
        "final_prompt_provenance": (
            prepared.config.final_prompt_provenance(
                analogous_reasoning_only=analogous_reasoning_only
            )
            if prepared.config.final_prompt_provenance is not None
            and not _analogous_flat_args(args)
            else {}
        ),
        "analogous_flat_prompt_provenance": (
            analogous_flat_prompt_provenance(_task_id(prepared))
            if _analogous_flat_args(args)
            else {}
        ),
        "prompt_identity_view": (
            PROMPT_IDENTITY_VIEW if _analogous_flat_args(args) else "identity_blind"
            if args.identity_blind else "deployment_visible"
        ),
        "harness_prefetch_tools": (
            False
            if analogous_reasoning_only or flat_one_call
            else args.identity_blind or args.harness_prefetch_tools
        ),
        "neighbor_index": args.index if args.experiment_mode != "none" else "",
        "retrieval_evidence_source": retrieval.get("evidence_source", {}),
        "model": args.model,
        "base_url": args.base_url,
        "tool_service_url": args.tool_service_url,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "thinking": {"type": "enabled"} if args.enable_thinking else {"type": "disabled"},
        "group_tools_enabled": (
            not args.disable_group_tools
            and not getattr(args, "disable_flat_tools", False)
            and not analogous_reasoning_only
        ),
        "tool_execution_mode": (
            "omitted"
            if analogous_reasoning_only or flat_one_call
            else "harness_prefetch_query_only"
            if getattr(args, "disable_flat_tools", False)
            else "harness_prefetch"
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
        and not getattr(args, "disable_flat_tools", False)
        and not analogous_reasoning_only
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
    if analogous_reasoning_only or flat_one_call and args.flat_query_prior == "none":
        _write_json_atomic(
            run_dir / "single_molecule_reasoning_output.json",
            {
                "analysis_id": "single_molecule",
                "status": "omitted",
                "reason": (
                    "analogous_reasoning_only"
                    if analogous_reasoning_only
                    else "query_prior_disabled"
                ),
            },
        )
    if not groups:
        _write_jsonl_atomic(run_dir / "group_reasoning_outputs.jsonl", [])
        if args.identity_blind:
            _write_jsonl_atomic(raw_group_path, [])
    _hydrate_configured_branch_reuse(prepared, item, retrieval, run_dir)


def _configured_source_run_dir(source_batch_value: str, query_index: int) -> str:
    if not source_batch_value:
        return ""
    return str(_source_run_dir(Path(source_batch_value), query_index))


def synchronize_configured_single_reuse(
    prepared: PreparedBatch,
    item: BatchItem,
    run_dir: Path,
) -> None:
    """Freeze a configured single checkpoint before completion detection."""
    source_dir = _configured_source_run_dir(
        getattr(prepared.args, "single_analysis_source_batch", ""),
        item.index,
    )
    if not source_dir:
        return
    if not run_dir.is_dir() or not (
        Path(source_dir) / "single_molecule_reasoning_output.json"
    ).is_file():
        return
    source_output = load_frozen_single_analysis(source_dir)
    if source_output is None:
        return
    target_path = run_dir / "single_molecule_reasoning_output.json"
    target_output = _read_json(target_path) if target_path.is_file() else None
    source_visible = (source_output.get("status"), validated_branch_content(source_output))
    target_visible = (
        (target_output or {}).get("status"),
        validated_branch_content(target_output or {}),
    )
    _write_json_atomic(target_path, source_output)
    if target_visible != source_visible:
        _invalidate_dependent_final(run_dir)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        manifest["single_analysis_source_run_dir"] = source_dir
        _write_json_atomic(manifest_path, manifest)


def _hydrate_configured_branch_reuse(
    prepared: PreparedBatch,
    item: BatchItem,
    retrieval: dict[str, Any],
    run_dir: Path,
) -> None:
    args = prepared.args
    synchronize_configured_single_reuse(prepared, item, run_dir)
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
    if _analogous_reasoning_only(state):
        raise RuntimeError("The single stage is omitted in analogous-reasoning-only mode")
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
    _write_stage_trace(
        state,
        SINGLE_STAGE,
        output,
        state.run_dir / "single_molecule_reasoning_output.json",
    )
    return output


def _execute_group(
    state: StageState,
    group_id: str,
    stage_client: Any | None = None,
) -> dict[str, Any]:
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
    client = stage_client or _make_client(state)
    group_kwargs: dict[str, Any] = {}
    if _versioned_flat(state):
        group_kwargs = {
            "flat_prompt_version": state.prepared.args.flat_prompt_version,
            "retrieval_strategy": state.prepared.args.retrieval_strategy,
            "flat_reranking": getattr(state.prepared.args, "flat_reranking", ""),
        }
    elif (
        state.prepared.config.supports_analogous_reasoning_only
        and not _analogous_flat(state)
    ):
        group_kwargs = {
            "include_assay_transfer_score": module.scored_neighbors_prompt_enabled(
                context["reasoning_retrieval"]
            ),
            "prompt_format": state.prepared.args.group_prompt_format,
            "prompt_options": {
                "prompt_min_similarity": (
                    state.prepared.args.group_prompt_min_similarity
                    if state.prepared.args.group_prompt_min_similarity is not None
                    else state.prepared.args.min_similarity
                ),
                "instructions_file": (
                    state.prepared.args.group_prompt_instructions_file or None
                ),
                "output_schema_profile": state.prepared.args.group_output_schema,
                "presentation_style": state.prepared.args.presentation_style,
                "prompt_version": state.prepared.args.group_prompt_version,
                "omit_query_tools": (
                    _analogous_reasoning_only(state)
                    or getattr(state.prepared.args, "disable_flat_tools", False)
                ),
            },
        }
    group_kwargs.update(_task_prompt_kwargs(state))
    group_query = context["reasoning_retrieval"]["query"]
    if getattr(state.prepared.args, "disable_flat_tools", False):
        group_query = query_without_prefetched_tools(group_query)
    group_query_payload = getattr(
        module,
        "_llm_evidence_query_payload",
        module._llm_query_payload,
    )
    if _analogous_flat(state):
        raw_output = reason_analogous_flat_group(
            client,
            group,
            task_id=_task_id(state.prepared),
        )
    else:
        raw_output = module._reason_one_group(
            client,
            group_query_payload(group_query),
            group,
            **group_kwargs,
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
    _write_stage_trace(
        state,
        f"group_{group_id}",
        canonical_output,
        state.run_dir / "group_reasoning_outputs.jsonl",
    )
    return canonical_output


def _execute_final(
    state: StageState,
    stage_client: Any | None = None,
) -> dict[str, Any]:
    if not _earlier_stages_ready(state):
        raise RuntimeError(f"Final dependencies are incomplete: {state.run_id}")
    if _flat_one_call(state):
        return _execute_flat_context_final(state, stage_client)
    context = _stage_context(state)
    module = context["module"]
    single_output = _read_json(
        state.run_dir / "single_molecule_reasoning_output.json"
    )
    group_outputs = _canonical_group_outputs(state)
    client = stage_client or _make_client(state)
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
    if _analogous_reasoning_only(state):
        final_kwargs["analogous_reasoning_only"] = True
    if _analogous_flat(state):
        final_output = reason_analogous_flat_final(
            client,
            group_outputs,
            task_id=_task_id(state.prepared),
        )
    else:
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
    _write_stage_trace(state, FINAL_STAGE, final_output, final_path)
    return final_output


def _execute_flat_context_final(
    state: StageState,
    stage_client: Any | None,
) -> dict[str, Any]:
    """Issue the context-v4 flat harness's sole model request."""
    retrieval = attach_external_condition(deepcopy(state.retrieval), state.item.record)
    single_output = _read_json(
        state.run_dir / "single_molecule_reasoning_output.json"
    )
    query_prior = (
        validated_branch_content(single_output)
        if state.prepared.args.flat_query_prior == "cached"
        else {}
    )
    messages, prompt_metadata = build_flat_context_request(
        retrieval,
        task_id=_task_id(state.prepared),
        task_prompt_profile=str(state.prepared.args.task_prompt_profile),
        layout=str(state.prepared.args.flat_layout),
        reranking=str(state.prepared.args.flat_reranking),
        query_prior=query_prior,
        prompt_version=str(state.prepared.args.flat_prompt_version),
    )
    request = {
        "schema_version": "joseph_flat_context_request.v1",
        "messages": messages,
        "message_char_count": sum(len(message["content"]) for message in messages),
        **prompt_metadata,
    }
    _write_json_atomic(state.run_dir / "request.json", request)
    response = call_with_json_validation(
        (stage_client or _make_client(state)).chat_json,
        messages,
        branch_name="flat-context-final",
        **flat_context_validation(
            _task_id(state.prepared),
            task_prompt_profile=str(state.prepared.args.task_prompt_profile),
            prompt_version=str(state.prepared.args.flat_prompt_version),
            reference_index=prompt_metadata["reasoning_reference_index"],
        ),
    )
    final_output = {
        "status": "ok" if structured_response_is_valid(response) else "error",
        "llm": response,
        "prompt": prompt_metadata,
    }
    if (
        state.prepared.args.flat_prompt_version == CONTEXT_V5_PROMPT_VERSION
        and final_output["status"] == "ok"
    ):
        final_output["claim_evidence"] = derive_flat_claim_evidence(
            response["content"]
        )
    final_path = state.run_dir / "final_reasoning_output.json"
    trace_path = state.run_dir / "trace_messages.jsonl"
    module = importlib.import_module(state.prepared.config.pipeline_module)
    with _exclusive_run_lock(state.run_dir):
        with atomic_output_path(final_path) as final_temp:
            _write_json(final_temp, final_output)
            if state.prepared.args.save_trace:
                with atomic_output_path(trace_path) as trace_temp:
                    module._write_trace_jsonl(
                        trace_temp,
                        query_record=state.item.record,
                        query_index=state.item.index,
                        smiles=str(state.item.record.get(state.prepared.args.smiles_field) or ""),
                        single_output=single_output,
                        group_outputs=[],
                        final_output=final_output,
                    )
            else:
                trace_path.unlink(missing_ok=True)
        _record_stage_event(state, FINAL_STAGE, final_output.get("status", "error"))
    _write_stage_trace(state, FINAL_STAGE, final_output, final_path)
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
            include_query_tools=not _analogous_reasoning_only(state),
            include_neighbor_tools=(
                not getattr(state.prepared.args, "disable_flat_tools", False)
                and not _analogous_reasoning_only(state)
            ),
        )
        if _analogous_flat(state):
            reasoning_retrieval = expose_neighbor_smiles_only(
                reasoning_retrieval,
                state.retrieval,
            )
        state._context = {
            "module": module,
            "reasoning_retrieval": reasoning_retrieval,
        }
        return state._context


def _make_client(state: StageState) -> OpenAICompatibleClient:
    from data.processing.llm_api import openai_compatible_client

    args = state.prepared.args
    transport, _ = openai_compatible_client(
        base_url=args.base_url,
        env_file=args.env_file,
        credential_env=args.api_key_env or None,
        max_connections=1,
        timeout_s=args.timeout_s,
        max_retries=getattr(args, "transport_max_retries", 0),
    )
    return OpenAICompatibleClient(
        api_key="loaded-by-shared-client",
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tool_service_url=args.tool_service_url,
        enable_group_tools=(
            not args.disable_group_tools
            and not getattr(args, "disable_flat_tools", False)
            and not _analogous_reasoning_only(state)
        ),
        max_tool_rounds=args.max_tool_rounds,
        reasoning_effort=args.reasoning_effort,
        enable_thinking=args.enable_thinking,
        transport_max_retries=getattr(args, "transport_max_retries", 0),
        openai_client=transport,
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
        if _read_json(single_path).get("status") != _expected_single_status(state):
            return False
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    statuses = _group_status_by_id(state.run_dir)
    return all(statuses.get(group_id) == "ok" for group_id in state.expected_group_ids)


def _expected_single_status(state: StageState) -> str:
    if _flat_one_call(state) and state.prepared.args.flat_query_prior == "none":
        return "omitted"
    return "omitted" if _analogous_reasoning_only(state) else "ok"


def _analogous_reasoning_only(state: StageState) -> bool:
    return bool(getattr(state.prepared.args, "analogous_reasoning_only", False))


def _flat_one_call(state: StageState) -> bool:
    return _flat_one_call_args(state.prepared.args)


def _flat_one_call_args(args: Any) -> bool:
    return (
        str(getattr(args, "flat_prompt_version", ""))
        in CONTEXT_PROMPT_VERSIONS
        and str(getattr(args, "experiment_mode", "")) == "full_flat"
    )


def _analogous_flat(state: StageState) -> bool:
    return _analogous_flat_args(state.prepared.args)


def _analogous_flat_args(args: Any) -> bool:
    return bool(getattr(args, "analogous_reasoning_only", False)) and str(
        getattr(args, "experiment_mode", "")
    ) == "full_flat"


def _versioned_flat(state: StageState) -> bool:
    return _versioned_flat_args(state.prepared.args)


def _versioned_flat_args(args: Any) -> bool:
    return bool(getattr(args, "flat_prompt_version", "")) and str(
        getattr(args, "experiment_mode", "")
    ) == "full_flat"


def _task_id(prepared: PreparedBatch) -> str:
    return prepared.config.pipeline_module.split(".")[-2]


def _write_stage_trace(
    state: StageState,
    stage: str,
    output: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    mode = str(getattr(state.prepared.args, "experiment_mode", "native"))
    harness = {"full_flat": "flat", "full_mechanism": "full"}.get(mode, mode)
    write_trace(
        trace_root=getattr(state.prepared.args, "trace_root", "outputs/paper/live"),
        experiment_id=state.prepared.batch_id,
        task=_task_id(state.prepared),
        harness=harness,
        sample_id=f"query_idx{state.item.index:05d}",
        stage=stage,
        checkpoint_path=checkpoint_path,
        output=output,
        run_id=getattr(state.prepared.args, "live_run_id", ""),
        method=getattr(state.prepared.args, "live_method", harness),
    )


def _export_existing_traces(state: StageState) -> None:
    single_path = state.run_dir / "single_molecule_reasoning_output.json"
    if single_path.exists():
        _write_stage_trace(state, SINGLE_STAGE, _read_json(single_path), single_path)
    group_path = state.run_dir / "group_reasoning_outputs.jsonl"
    if group_path.exists():
        for output in _read_jsonl(group_path):
            _write_stage_trace(
                state,
                f"group_{output.get('group_id') or 'unknown'}",
                output,
                group_path,
            )
    final_path = state.run_dir / "final_reasoning_output.json"
    if final_path.exists():
        _write_stage_trace(state, FINAL_STAGE, _read_json(final_path), final_path)


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
