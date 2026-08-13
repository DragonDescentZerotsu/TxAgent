"""Run the matched train-label direct-agent valid experiment in one prompt pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from tools.chembl_tool.common.json_utils import write_json_atomic
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
    experiment_run_root,
)
from tools.chembl_tool.tasks.bioavailability_ma.prompt_profiles import (
    BIOAVAILABILITY_PROMPT_PROFILES,
    LEGACY_BIOAVAILABILITY_V1,
)
from tools.chembl_tool.tasks.bbb_martins.prompt_profiles import (
    BBB_PROMPT_PROFILES,
    MEANINGFUL_CNS_ACCESS_V1,
)

from .contract import DEFAULT_OUTPUT_ROOT, SCHEMA_VERSION, replay_batch, selected_specs


DEFAULT_BASE_URL = "http://127.0.0.1:9001/v1"
DEFAULT_MODEL = "gpt-oss-120b"
DEFAULT_API_KEY_ENV = "GPT_OSS_LOCAL_API_KEY"


def run_experiment(
    *,
    output_root: str | Path,
    tasks: list[str] | None = None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    reasoning_effort: str = "",
    parallelism: int = 128,
    timeout_s: int = 300,
    max_stage_requeues: int = 1,
    limit: int = 0,
    manifest_only: bool = False,
    bbb_prompt_profile: str = MEANINGFUL_CNS_ACCESS_V1,
    bioavailability_prompt_profile: str = LEGACY_BIOAVAILABILITY_V1,
    specs_override: list[Any] | None = None,
    single_analysis_root_overrides: dict[str, Path] | None = None,
    retrieval_contract: str = "exact formal Morgan KNN top-3 with train Y visible",
) -> dict[str, Any]:
    root = Path(output_root)
    specs = specs_override or selected_specs(tasks)
    single_analysis_root_overrides = single_analysis_root_overrides or {}
    commands: list[BatchCommand] = []
    rows = []
    dependency_rows = []
    for spec in specs:
        replay = replay_batch(root, spec)
        if not (replay / "manifest.json").exists():
            raise FileNotFoundError(
                f"Missing materialized retrievals for {spec.task}: {replay / 'manifest.json'}"
            )
        experiment = Experiment(
            name=spec.condition,
            task=spec.task,
            batch_module=spec.batch_module,
            input_jsonl=str(spec.input_jsonl),
            index=str(spec.retrieval_index),
            mode="direct",
            source="starling",
        )
        prompt_profile_args = _prompt_profile_args(
            spec.task,
            spec.prompt_profile_args,
            bioavailability_prompt_profile,
            bbb_prompt_profile=bbb_prompt_profile,
        )
        fresh_single = spec.task not in single_analysis_root_overrides and _fresh_single_required(
            spec.task,
            bbb_prompt_profile=bbb_prompt_profile,
            bioavailability_prompt_profile=bioavailability_prompt_profile,
        )
        single_root = single_analysis_root_overrides.get(spec.task) or (
            experiment_run_root(
                IDENTITY_BLIND,
                PARENT_DISJOINT,
                paper_root=root,
            )
            if fresh_single
            else spec.single_source_batch.parents[1]
        )
        args = SimpleNamespace(
            split="valid",
            paper_root=root,
            python_executable=sys.executable,
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            reasoning_effort=reasoning_effort,
            visibility_mode=IDENTITY_BLIND,
            neighbor_identity_policy=PARENT_DISJOINT,
            fresh_parent_disjoint=True,
            neighbor_selector="similarity",
            neighbor_context_profile="standard",
            parallelism=parallelism,
            timeout_s=timeout_s,
            max_stage_requeues=max_stage_requeues,
            limit=limit,
            single_analysis_root=str(single_root),
            group_analysis_root="",
        )
        if fresh_single:
            none_experiment = Experiment(
                name=f"{spec.task}__none",
                task=spec.task,
                batch_module=spec.batch_module,
                input_jsonl=str(spec.input_jsonl),
                index=str(spec.retrieval_index),
                mode="none",
                source="starling",
            )
            none_args = SimpleNamespace(**vars(args))
            none_args.single_analysis_root = ""
            none_command = _command(none_experiment, none_args)
            none_command.extend(prompt_profile_args)
            commands.append(
                BatchCommand(f"{spec.task}.{none_experiment.name}", none_command)
            )
            dependency_rows.append(
                {
                    "task": spec.task,
                    "condition": none_experiment.name,
                    "purpose": "same-profile single-molecule dependency",
                    "command": none_command,
                }
            )
        command = _command(experiment, args)
        command.extend(["--retrieval-replay-source-batch", str(replay)])
        command.extend(prompt_profile_args)
        scheduler_name = f"{spec.task}.{spec.condition}"
        commands.append(BatchCommand(scheduler_name, command))
        rows.append(
            {
                "task": spec.task,
                "condition": spec.condition,
                "input_jsonl": str(spec.input_jsonl),
                "knn_predictions": str(spec.knn_predictions),
                "retrieval_replay_source_batch": str(replay),
                "single_analysis_source_batch": str(
                    single_root / spec.task / f"{spec.task}__none"
                ),
                "task_prompt_profile_args": list(prompt_profile_args),
                "command": command,
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_subset": "valid",
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "reasoning_effort": reasoning_effort,
        "visibility_mode": IDENTITY_BLIND,
        "neighbor_identity_policy": PARENT_DISJOINT,
        "retrieval_contract": retrieval_contract,
        "external_starling_records_visible": False,
        "scheduler": {
            "version": SCHEDULER_VERSION,
            "parallelism": parallelism,
            "max_stage_requeues": max_stage_requeues,
        },
        "limit": limit,
        "manifest_only": manifest_only,
        "bbb_prompt_profile": bbb_prompt_profile,
        "bioavailability_prompt_profile": bioavailability_prompt_profile,
        "dependency_batches": dependency_rows,
        "batches": rows,
    }
    write_json_atomic(root / "experiment_manifest.json", manifest)
    if manifest_only:
        return manifest

    ensure_endpoint_api_key(api_key_env, base_url)
    failed = run_global_prompt_pool(
        commands,
        max_workers=parallelism,
        max_stage_requeues=max_stage_requeues,
    )
    result = {**manifest, "failed": failed}
    write_json_atomic(root / "experiment_result.json", result)
    return result


def _prompt_profile_args(
    task: str,
    frozen_args: tuple[str, ...],
    bioavailability_prompt_profile: str,
    bbb_prompt_profile: str = MEANINGFUL_CNS_ACCESS_V1,
) -> tuple[str, ...]:
    if task == "bbb_martins":
        return ("--bbb-prompt-profile", bbb_prompt_profile)
    if task != "bioavailability_ma":
        return frozen_args
    return (
        "--bioavailability-prompt-profile",
        bioavailability_prompt_profile,
    )


def _fresh_single_required(
    task: str,
    *,
    bbb_prompt_profile: str,
    bioavailability_prompt_profile: str,
) -> bool:
    if task == "bbb_martins":
        return bbb_prompt_profile != MEANINGFUL_CNS_ACCESS_V1
    if task == "bioavailability_ma":
        return bioavailability_prompt_profile != LEGACY_BIOAVAILABILITY_V1
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--parallelism", type=int, default=128)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--max-stage-requeues", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument(
        "--bbb-prompt-profile",
        choices=BBB_PROMPT_PROFILES,
        default=MEANINGFUL_CNS_ACCESS_V1,
    )
    parser.add_argument(
        "--bioavailability-prompt-profile",
        choices=BIOAVAILABILITY_PROMPT_PROFILES,
        default=LEGACY_BIOAVAILABILITY_V1,
    )
    args = parser.parse_args(argv)
    result = run_experiment(
        output_root=args.output_root,
        tasks=args.tasks,
        base_url=args.base_url,
        model=args.model,
        api_key_env=args.api_key_env,
        reasoning_effort=args.reasoning_effort,
        parallelism=args.parallelism,
        timeout_s=args.timeout_s,
        max_stage_requeues=args.max_stage_requeues,
        limit=args.limit,
        manifest_only=args.manifest_only,
        bbb_prompt_profile=args.bbb_prompt_profile,
        bioavailability_prompt_profile=args.bioavailability_prompt_profile,
    )
    print(json.dumps(result, indent=2), flush=True)
    return int(bool(result.get("failed")))


if __name__ == "__main__":
    raise SystemExit(main())
