"""Launch fold-local none/direct agent batches through one global prompt pool."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Iterable

from tools.chembl_tool.common.coverage_reasoning import STANDARD_NEIGHBOR_CONTEXT
from tools.chembl_tool.common.json_utils import write_json_atomic
from tools.chembl_tool.common.neighbor_selection import SIMILARITY_SELECTOR
from tools.chembl_tool.common.task_workflows.global_prompt_pool import (
    BatchCommand,
    SCHEDULER_VERSION,
    run_global_prompt_pool,
)
from tools.chembl_tool.paper_experiments.molecular_evidence_agent import (
    IDENTITY_BLIND,
    PARENT_DISJOINT,
    Experiment,
    _command,
    ensure_endpoint_api_key,
)

from .contract import TaskSpec, direct_index_path, fold_root


DEFAULT_BASE_URL = "http://127.0.0.1:9001/v1"
DEFAULT_MODEL = "gpt-oss-120b"
DEFAULT_API_KEY_ENV = "GPT_OSS_LOCAL_API_KEY"
DEFAULT_REASONING_EFFORT = ""
MAX_ENDPOINT_CONCURRENCY = 512


def run_agent_oof(
    specs: Iterable[TaskSpec],
    *,
    output_root: str | Path,
    folds: Iterable[int],
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    parallelism: int = 128,
    timeout_s: int = 300,
    max_stage_requeues: int = 1,
    limit: int = 0,
    manifest_only: bool = False,
) -> dict[str, Any]:
    """Run all selected folds in one scheduler without outer condition fan-out."""
    if not 1 <= parallelism <= MAX_ENDPOINT_CONCURRENCY:
        raise ValueError(
            f"parallelism must be in [1, {MAX_ENDPOINT_CONCURRENCY}], got {parallelism}"
        )
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if max_stage_requeues < 0:
        raise ValueError("max_stage_requeues must be non-negative")

    selected_folds = sorted(set(int(fold) for fold in folds))
    commands: list[BatchCommand] = []
    manifest_rows: list[dict[str, Any]] = []
    for spec in specs:
        for fold in selected_folds:
            current_root = fold_root(output_root, spec.task, fold)
            query_path = current_root / "agent_input" / "valid.jsonl"
            index_path = direct_index_path(output_root, spec, fold)
            for path in (query_path, index_path):
                if not path.exists():
                    raise FileNotFoundError(path)
            paper_root = current_root / "agent"
            common = dict(
                task=spec.task,
                batch_module=spec.batch_module,
                input_jsonl=str(query_path),
                index=str(index_path),
            )
            experiments = (
                Experiment(
                    name=f"{spec.task}__none",
                    mode="none",
                    source="starling",
                    **common,
                ),
                Experiment(
                    name=spec.direct_condition,
                    mode="direct",
                    source="starling",
                    **common,
                ),
            )
            args = argparse.Namespace(
                split="valid",
                paper_root=paper_root,
                python_executable=sys.executable,
                base_url=base_url,
                model=model,
                api_key_env=api_key_env,
                reasoning_effort=reasoning_effort,
                visibility_mode=IDENTITY_BLIND,
                neighbor_identity_policy=PARENT_DISJOINT,
                fresh_parent_disjoint=True,
                neighbor_selector=SIMILARITY_SELECTOR,
                neighbor_context_profile=STANDARD_NEIGHBOR_CONTEXT,
                parallelism=parallelism,
                timeout_s=timeout_s,
                max_stage_requeues=max_stage_requeues,
                limit=limit,
                single_analysis_root="",
                group_analysis_root="",
            )
            for experiment in experiments:
                command = _command(experiment, args)
                scheduler_name = f"{spec.task}.fold_{fold:02d}.{experiment.name}"
                commands.append(BatchCommand(scheduler_name, command))
                manifest_rows.append(
                    {
                        "scheduler_name": scheduler_name,
                        "task": spec.task,
                        "fold": fold,
                        "condition": experiment.name,
                        "mode": experiment.mode,
                        "input_jsonl": str(query_path),
                        "index": str(index_path),
                        "paper_root": str(paper_root),
                        "command": _redact_command(command, api_key_env),
                    }
                )

    manifest = {
        "schema_version": "router_oof_agent_matrix.v1",
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "reasoning_effort": reasoning_effort,
        "visibility_mode": IDENTITY_BLIND,
        "neighbor_identity_policy": PARENT_DISJOINT,
        "scheduler": {
            "version": SCHEDULER_VERSION,
            "global_parallelism": parallelism,
            "endpoint_concurrency_cap": MAX_ENDPOINT_CONCURRENCY,
            "max_stage_requeues": max_stage_requeues,
        },
        "limit_per_batch": limit,
        "manifest_only": manifest_only,
        "batches": manifest_rows,
    }
    output_path = Path(output_root) / "agent_matrix_manifest.json"
    write_json_atomic(output_path, manifest)
    if manifest_only:
        return {**manifest, "manifest_path": str(output_path), "failed": []}

    ensure_endpoint_api_key(api_key_env, base_url)
    failed = run_global_prompt_pool(
        commands,
        max_workers=parallelism,
        max_stage_requeues=max_stage_requeues,
    )
    result = {**manifest, "manifest_path": str(output_path), "failed": failed}
    write_json_atomic(Path(output_root) / "agent_matrix_result.json", result)
    return result


def _redact_command(command: list[str], api_key_env: str) -> list[str]:
    """The command contains only an environment-variable name, never its value."""
    if os.environ.get(api_key_env) and os.environ[api_key_env] in command:
        raise AssertionError("API key value unexpectedly appeared in the command")
    return list(command)
