"""Execute resumable FINAL requests and native single-molecule query priors."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import importlib
import json
import os
from pathlib import Path
import shutil
from functools import lru_cache
import threading
import time
from typing import Any

SUMMARY_ONLY = "summary_only"
STANDARD_FINAL_DECISION = "standard"

from predict.harnesses.branches.visibility import (
    expose_neighbor_smiles_only,
    prepare_reasoning_retrieval,
    query_without_prefetched_tools,
    sanitize_identity_blind_branch_outputs,
)
from predict.api_client.client import OpenAICompatibleClient
from predict.tools.client import ToolServiceClient
from predict.traces.io import write_trace
from predict.utils.json import (
    atomic_output_path,
    sha256_file,
    write_json_atomic as _write_json_atomic,
    write_jsonl_atomic as _write_jsonl_atomic,
)
from predict.harnesses.branches.inference import load_frozen_single_analysis
from predict.harnesses.branches.flat import (
    CONTEXT_V4_PROMPT_VERSION,
    CONTEXT_CLAIMS_PROMPT_VERSIONS,
    CONTEXT_PROMPT_VERSIONS,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_V3_PROMPT_VERSION,
    FINAL_FULL_FLAT_PROMPT_VERSION,
    FINAL_FULL_FLAT_TDC_CARC_PROMPT_VERSION,
    FINAL_FULL_FLAT_TDC_CARC_STRONG_PROMPT_VERSION, FINAL_FULL_FLAT_TDC_CARC_POSITIVE_GATE_PROMPT_VERSION,
    TDC_SKIN_LLNA_PROMPT_VERSION,
    TDC_SKIN_LLNA_SCORES_PROMPT_VERSION,
    TDC_V2_MIXED_PROMPT_VERSION,
    DILI_TDC_V2_MIXED_V3_PROMPT_VERSION,
    DILI_TDC_V2_POLARIZED_PROMPT_VERSION,
    build_flat_context_request,
    derive_flat_claim_evidence,
    flat_context_validation,
    native_flat_prediction,
)
from predict.harnesses.branches.prompt import attach_external_condition
from predict.harnesses.branches.manifests import read_run, read_record, write_batch, run_directory, canonical_location
from predict.utils.response import (
    call_with_json_validation,
    structured_response_is_valid,
    validated_branch_content,
    validate_query_prior_output,
)
from predict.harnesses.branches.artifacts import (
    load_reusable_group_outputs,
    write_trace_jsonl,
)
from predict.harnesses.branches.runner import (
    BatchItem,
    PreparedBatch,
    _collect_result,
    _exclusive_run_lock,
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
        synchronize_configured_single_reuse(
            prepared,
            item,
            run_dir,
        )
        replay_batch = getattr(
            prepared.args, "retrieval_replay_source_batch", ""
        )
        if _flat_one_call_args(prepared.args) and replay_batch:
            source = _source_run_dir(Path(replay_batch), item.index)
            shutil.copyfile(source / "retrieval.json", run_dir / "retrieval.json")
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
    if not retrieval_path.exists() or not ((prepared.batch_dir / "manifest.json").exists() or manifest_path.exists()):
        return None
    try:
        retrieval = _read_json(retrieval_path)
        manifest = read_run(run_dir, batch=prepared.manifest)
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if retrieval.get("status") != "ok":
        return None
    expected_group_ids = []
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
        return _execute_single(job.state, stage_client)
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
    """Initialize query checkpoints without copying the batch contract."""
    retrieval_path = run_dir / "retrieval.json"
    retrieval = _read_json(retrieval_path)
    if retrieval.get("status") != "ok":
        raise RuntimeError(f"prepare-only retrieval is invalid: {run_dir}")
    args = prepared.args
    flat_one_call = _flat_one_call_args(args)
    groups = []
    raw_group_path = run_dir / "group_reasoning_outputs_raw.jsonl"
    batch_path = prepared.batch_dir / "manifest.json"
    if not batch_path.exists():
        write_batch(batch_path, dict(vars(args), batch_id=prepared.batch_id, flat_one_call=flat_one_call),
                    records=[(row.index, row.record) for row in prepared.items])
    _write_json_atomic(run_dir / "stage_resume.json", {
        "n_groups_with_neighbors": len(groups), "expected_group_ids": [],
        "stage_pool": {"version": STAGE_RUNTIME_VERSION,
                       "prepared_without_llm": True, "events": []},
    })
    if flat_one_call and args.flat_query_prior == "none":
        _write_json_atomic(
            run_dir / "single_molecule_reasoning_output.json",
            {
                "analysis_id": "single_molecule",
                "status": "omitted",
                "reason": (
                    "query_prior_disabled"
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
    if not run_dir.is_dir() or getattr(prepared.args, "flat_query_prior", "cached") == "none":
        return
    target_path = run_dir / "single_molecule_reasoning_output.json"
    source_path = Path(source_dir) / target_path.name
    if target_path.is_file():
        require_matching_query_prior(target_path, source_path, item.record.get("benchmark_row_id", item.index))
        return
    if not source_path.is_file():
        return
    manifest_path = run_dir / "stage_resume.json"
    source_manifest = Path(source_dir).parent.parent / "manifest.json"
    source_uid = read_record(source_manifest).get("run_uid", "") if source_manifest.exists() else ""
    source_output = load_frozen_single_analysis(source_dir)
    if source_output is None:
        return
    if validated_branch_content(source_output) is None:
        raise ValueError(f"Requested query prior is invalid: {source_path}")
    _write_json_atomic(target_path, source_output)
    if source_uid:
        checkpoint = _read_json(manifest_path) if manifest_path.exists() else {}
        checkpoint["single_analysis_source_run_uid"] = source_uid
        _write_json_atomic(manifest_path, checkpoint)


def require_matching_query_prior(saved: Path, requested: Path, query_id: Any) -> Any:
    """Validate model-visible prior content without modifying either checkpoint."""
    try:
        old_document, current_document = _read_json(saved), _read_json(requested)
        old = validated_branch_content(validate_query_prior_output(old_document))
        current = validated_branch_content(validate_query_prior_output(current_document))
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"Cannot validate query prior for {query_id}: saved={saved}, requested={requested}") from error
    if old is None or current is None or old != current:
        raise ValueError(
            f"Query prior changed or is invalid for {query_id}: saved={saved}, requested={requested}. "
            "Use a new run identity; the saved checkpoint was left unchanged."
        )
    return current



def _hydrate_configured_branch_reuse(prepared: PreparedBatch, item: BatchItem,
                                    retrieval: dict[str, Any], run_dir: Path) -> None:
    synchronize_configured_single_reuse(prepared, item, run_dir)


def _execute_single(
    state: StageState,
    stage_client: Any | None = None,
) -> dict[str, Any]:
    source_batch = str(state.prepared.args.single_analysis_source_batch or "")
    if source_batch:
        source_dir = _source_run_dir(Path(source_batch), state.item.index)
        saved = state.run_dir / "single_molecule_reasoning_output.json"
        if saved.is_file():
            require_matching_query_prior(saved, source_dir / saved.name,
                                         state.item.record.get("benchmark_row_id", state.item.index))
        output = load_frozen_single_analysis(str(source_dir))
        if output is None:
            raise RuntimeError(f"Frozen single stage was not loaded: {source_dir}")
    else:
        context = _stage_context(state)
        module = context["module"]
        client = stage_client or _make_client(state)
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


def _execute_final(state: StageState, stage_client: Any | None = None) -> dict[str, Any]:
    if not _earlier_stages_ready(state):
        raise RuntimeError(f"Final dependencies are incomplete: {state.run_id}")
    return _execute_flat_context_final(state, stage_client)


def _execute_flat_context_final(
    state: StageState,
    stage_client: Any | None,
) -> dict[str, Any]:
    """Issue the flat harness's sole FINAL request."""
    retrieval = attach_external_condition(deepcopy(state.retrieval), state.item.record)
    single_output = _read_json(
        state.run_dir / "single_molecule_reasoning_output.json"
    )
    query_prior = (
        validated_branch_content(validate_query_prior_output(single_output))
        if state.prepared.args.flat_query_prior == "cached"
        else {}
    )
    replay_root = str(getattr(state.prepared.args, "flat_request_replay_root", "") or "")
    if replay_root:
        request = _read_json(Path(replay_root) / f"{state.item.index:05d}" / "request.json")
        if request.get("schema_version") != "joseph_flat_context_request.v1":
            raise ValueError("invalid frozen flat request")
        messages = request["messages"]
        prompt_metadata = {key: value for key, value in request.items()
                           if key not in {"schema_version", "messages", "message_char_count"}}
    else:
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
        state.prepared.args.flat_prompt_version in {
            CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
            CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_PROMPT_VERSION,
            CONTEXT_V5_SIX_TASKS_UPSTREAM_V3_PROMPT_VERSION,
            TDC_SKIN_LLNA_PROMPT_VERSION,
            TDC_SKIN_LLNA_SCORES_PROMPT_VERSION,
            TDC_V2_MIXED_PROMPT_VERSION,
            DILI_TDC_V2_MIXED_V3_PROMPT_VERSION,
            DILI_TDC_V2_POLARIZED_PROMPT_VERSION,
        }
        and final_output["status"] == "ok"
    ):
        portable = str(response["content"]["final_prediction"])
        native, mapping_sha256 = native_flat_prediction(
            _task_id(state.prepared),
            portable,
            prompt_version=str(state.prepared.args.flat_prompt_version),
        )
        response["content"]["final_prediction"] = native
        final_output["prediction_mapping"] = {
            "schema_version": "portable_prediction_mapping.v1",
            "task": _task_id(state.prepared),
            "portable_prediction": portable,
            "native_prediction": native,
            "mapping_sha256": mapping_sha256,
        }
    if (
        state.prepared.args.flat_prompt_version in CONTEXT_CLAIMS_PROMPT_VERSIONS
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
                    trace_kwargs = {
                        "query_record": state.item.record,
                        "query_index": state.item.index,
                        "smiles": str(
                            state.item.record.get(state.prepared.args.smiles_field) or ""
                        ),
                        "single_output": single_output,
                        "group_outputs": [],
                        "final_output": final_output,
                    }
                    if state.prepared.args.flat_prompt_version in {
                        CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
                        CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_PROMPT_VERSION,
                        CONTEXT_V5_SIX_TASKS_UPSTREAM_V3_PROMPT_VERSION,
                        FINAL_FULL_FLAT_PROMPT_VERSION,
                        FINAL_FULL_FLAT_TDC_CARC_PROMPT_VERSION,
                        FINAL_FULL_FLAT_TDC_CARC_STRONG_PROMPT_VERSION, FINAL_FULL_FLAT_TDC_CARC_POSITIVE_GATE_PROMPT_VERSION,
                        TDC_SKIN_LLNA_PROMPT_VERSION,
                        TDC_SKIN_LLNA_SCORES_PROMPT_VERSION,
                        TDC_V2_MIXED_PROMPT_VERSION,
                        DILI_TDC_V2_MIXED_V3_PROMPT_VERSION,
                        DILI_TDC_V2_POLARIZED_PROMPT_VERSION,
                    }:
                        write_trace_jsonl(
                            trace_temp,
                            prediction_field="final_prediction",
                            **trace_kwargs,
                        )
                    else:
                        module._write_trace_jsonl(trace_temp, **trace_kwargs)
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
            include_query_tools=True,
            include_neighbor_tools=(
                not getattr(state.prepared.args, "disable_flat_tools", False)
            ),
        )
        state._context = {
            "module": module,
            "reasoning_retrieval": reasoning_retrieval,
        }
        return state._context


def _make_client(state: StageState) -> OpenAICompatibleClient:
    from predict.api_client.transport import openai_compatible_client

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
    manifest_path = state.run_dir / "stage_resume.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    source = str(manifest.get("prefetched_tool_replay_source_run_dir") or "")
    return "" if source == str(state.run_dir) else source


@lru_cache(maxsize=None)
def _query_prior_overlay(source_batch: Path) -> tuple[Path, ...] | None:
    manifest_path = source_batch / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != "branch_query_prior_overlay.v1":
        return None
    input_path = Path(str(manifest.get("input_jsonl") or ""))
    if not input_path.is_file():
        raise ValueError(f"Query-prior overlay input hash mismatch: {source_batch}")
    sources = manifest.get("sources") or []
    if len(sources) != int(manifest.get("n_items", -1)):
        raise ValueError(f"Query-prior overlay source count mismatch: {source_batch}")
    output = []
    for index, source in enumerate(sources):
        if int(source.get("target_index", -1)) != index:
            raise ValueError(f"Query-prior overlay indices are not contiguous: {source_batch}")
        run_dir = Path(str(source.get("run_dir") or ""))
        output.append(run_dir)
    return tuple(output)


def _source_run_dir(source_batch: Path, query_index: int) -> Path:
    source_batch = canonical_location(source_batch)
    overlay = _query_prior_overlay(source_batch.resolve())
    if overlay is not None:
        try:
            return canonical_location(overlay[query_index])
        except IndexError as error:
            raise ValueError(f"Query-prior overlay lacks index {query_index}") from error
    return run_directory(source_batch, query_index)


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
    return "ok"


def _flat_one_call(state: StageState) -> bool:
    return _flat_one_call_args(state.prepared.args)


def _flat_one_call_args(args: Any) -> bool:
    return (
        str(getattr(args, "flat_prompt_version", ""))
        in CONTEXT_PROMPT_VERSIONS
        and str(getattr(args, "experiment_mode", "")) == "full_flat"
    )


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
    experiment = state.prepared.batch_id
    if getattr(state.prepared.args, "run_name", ""):
        metadata = state.prepared.manifest
        benchmark = metadata.get("benchmark") or metadata["benchmark_release"]
        experiment = "__".join((
            state.prepared.args.run_name, state.prepared.args.flat_prompt_version,
            str(benchmark), str(metadata["evaluation_subset"]), experiment,
        ))
    write_trace(
        trace_root=getattr(state.prepared.args, "trace_root", "outputs/paper/live"),
        experiment_id=experiment,
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


def _record_stage_event(
    state: StageState,
    stage: str,
    status: str,
    *,
    group_id: str = "",
) -> None:
    manifest_path = state.run_dir / "stage_resume.json"
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
