"""Run the train/valid diagnostic final-evidence-surface ablation.

This runner never re-runs retrieval, single-molecule reasoning, or group
reasoning.  It copies the frozen ``starling_full_flat`` source artifacts into
an isolated lineage and schedules only the final stage with an explicit
versioned evidence surface.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlparse

from tools.chembl_tool.common.final_evidence_surface import (
    CARDS_ONLY,
    FINAL_EVIDENCE_CARD_CONTRACT_VERSION,
    SUMMARY_PLUS_CARDS,
)
from tools.chembl_tool.common.json_utils import sha256_file, write_json_atomic
from tools.chembl_tool.common.task_workflows.global_prompt_pool import (
    BatchCommand,
    SCHEDULER_VERSION,
    run_global_prompt_pool,
)
from tools.chembl_tool.paper_experiments.starling_benchmark_matrix import (
    experiments_for_starling_benchmark,
)


DATASET_LINEAGE = "record_supported_v2"
BENCHMARK_SPLIT = "scaffold"
EVALUATION_SUBSET = "valid"
CONTROL_CONDITION_SUFFIX = "__starling_full_flat"
TASKS = ("bbb_martins", "bioavailability_ma", "skin_reaction")
SURFACES = (SUMMARY_PLUS_CARDS, CARDS_ONLY)

DEFAULT_DATA_ROOT = Path("data/processed_starling_record_supported_v2")
DEFAULT_CANONICAL_PAPER_ROOT = Path(
    "outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2"
)
DEFAULT_SOURCE_RUN_ROOT = Path(
    "outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2_valid_gpt_oss_120b/"
    "runs_identity_blind_parent_disjoint"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/paper/final_evidence_surface_record_supported_v2_valid_gpt_oss_120b"
)
EXPECTED_TASK_ROWS = {
    "bbb_martins": 500,
    "bioavailability_ma": 209,
    "skin_reaction": 245,
}
DEFAULT_API_KEY_ENV = "GPT_OSS_LOCAL_API_KEY"
LOCAL_GPT_OSS_API_KEY = "EMPTY"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    experiments = experiments_for_starling_benchmark(
        BENCHMARK_SPLIT,
        evaluation_subset=EVALUATION_SUBSET,
        data_root=args.benchmark_data_root,
        canonical_paper_root=args.canonical_paper_root,
    )
    controls = {
        experiment.task: experiment
        for experiment in experiments
        if experiment.name.endswith(CONTROL_CONDITION_SUFFIX)
    }
    selected_tasks = list(dict.fromkeys(args.tasks))
    selected_surfaces = list(dict.fromkeys(args.surfaces))
    missing_controls = sorted(set(selected_tasks) - set(controls))
    if missing_controls:
        raise SystemExit(
            "Missing Starling full-flat experiment definitions: "
            + ", ".join(missing_controls)
        )

    source_audit = {
        task: _validate_source_batch(
            Path(args.source_run_root) / task / controls[task].name,
            expected_rows=EXPECTED_TASK_ROWS[task],
            allow_partial=bool(args.limit or args.indices),
        )
        for task in selected_tasks
    }
    runtime_contract = _resolve_runtime_contract(source_audit)
    _apply_and_validate_runtime_contract(args, runtime_contract)
    manifest = {
        "schema_version": "starling.final_evidence_surface_experiment.v1",
        "dataset_lineage": DATASET_LINEAGE,
        "benchmark_split": BENCHMARK_SPLIT,
        "evaluation_subset": EVALUATION_SUBSET,
        "visibility_mode": "identity_blind",
        "neighbor_identity_policy": "parent_disjoint",
        "control_condition": "starling_full_flat_summary_only",
        "control_source_run_root": str(args.source_run_root),
        "source_audit": source_audit,
        "card_contract_version": FINAL_EVIDENCE_CARD_CONTRACT_VERSION,
        "surfaces": selected_surfaces,
        "tasks": selected_tasks,
        "model": args.model,
        "base_url": args.base_url,
        "reasoning_effort": args.reasoning_effort,
        "temperature": 0.0,
        "max_tokens": 20480,
        "parallelism": args.parallelism,
        "scheduler": {"name": "global_prompt_pool", "version": SCHEDULER_VERSION},
        "reuse_contract": {
            "retrieval": "copied_from_frozen_source_batch",
            "single": "copied_from_frozen_source_batch",
            "group": "copied_from_frozen_source_batch",
            "final": "fresh",
        },
        "limit": args.limit,
        "indices": args.indices or [],
        "output_root": str(args.output_root),
    }
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "experiment_manifest.json"
    write_json_atomic(manifest_path, manifest)
    if args.manifest_only:
        print(json.dumps({"experiment_manifest": str(manifest_path)}, indent=2))
        return 0

    _ensure_endpoint_api_key(args.api_key_env, args.base_url, args.model)
    commands = [
        BatchCommand(
            f"{task}:{surface}",
            _batch_command(controls[task], task, surface, args),
        )
        for surface in selected_surfaces
        for task in selected_tasks
    ]
    failed = run_global_prompt_pool(
        commands,
        max_workers=args.parallelism,
        max_stage_requeues=args.max_stage_requeues,
    )
    if failed:
        print(json.dumps({"failed": failed}, indent=2), file=sys.stderr)
        return 1
    return 0


def _batch_command(
    experiment: Any,
    task: str,
    surface: str,
    args: argparse.Namespace,
) -> list[str]:
    source_batch = Path(args.source_run_root) / task / experiment.name
    batch_root = Path(args.output_root) / "runs_identity_blind_parent_disjoint" / surface / task
    batch_id = f"{experiment.name}__{surface}"
    command = [
        args.python_executable,
        "-m",
        experiment.batch_module,
        "--input-jsonl",
        experiment.input_jsonl,
        "--index",
        experiment.index,
        "--batch-root",
        str(batch_root),
        "--batch-id",
        batch_id,
        "--experiment-mode",
        experiment.mode,
        "--retrieval-source",
        experiment.source,
        "--neighbor-identity-policy",
        "parent_disjoint",
        "--final-only-source-batch",
        str(source_batch),
        "--final-evidence-surface",
        surface,
        "--identity-blind",
        "--api-key-env",
        args.api_key_env,
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--disable-thinking",
        "--reasoning-effort",
        args.reasoning_effort,
        "--temperature",
        "0",
        "--max-tokens",
        "20480",
        "--timeout-s",
        str(args.timeout_s),
        "--max-tool-rounds",
        "3",
        "--parallelism",
        str(args.parallelism),
        "--no-stream-logs",
        "--no-combine-traces",
        "--skip-existing",
    ]
    if args.limit:
        command.extend(["--limit", str(args.limit)])
    if args.indices:
        command.append("--indices")
        command.extend(args.indices)
    return command


def _validate_source_batch(
    source_batch: Path,
    *,
    expected_rows: int,
    allow_partial: bool,
) -> dict[str, Any]:
    required = (
        source_batch / "manifest.json",
        source_batch / "metrics.json",
        source_batch / "predictions.jsonl",
        source_batch / "runs",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing frozen source artifacts:\n" + "\n".join(missing))
    metrics = json.loads((source_batch / "metrics.json").read_text(encoding="utf-8"))
    if int(metrics.get("n_failed_runs") or 0) != 0:
        raise SystemExit(f"Frozen source batch has failed runs: {source_batch}")
    if not allow_partial and int(metrics.get("n_successful") or 0) != expected_rows:
        raise SystemExit(
            f"Frozen source batch is incomplete: {source_batch} "
            f"successful={metrics.get('n_successful')} expected={expected_rows}"
        )
    source_runs = sorted(path for path in (source_batch / "runs").iterdir() if path.is_dir())
    if not source_runs:
        raise SystemExit(f"Frozen source batch has no run directories: {source_batch}")
    runtime_contracts: set[str] = set()
    for run_dir in source_runs:
        run_manifest_path = run_dir / "manifest.json"
        if not run_manifest_path.exists():
            raise SystemExit(f"Frozen source run lacks manifest: {run_dir}")
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        contract = {
            "model": str(run_manifest.get("model") or ""),
            "base_url": str(run_manifest.get("base_url") or ""),
            "reasoning_effort": str(run_manifest.get("reasoning_effort") or ""),
            "temperature": float(run_manifest.get("temperature") or 0.0),
            "thinking_type": str((run_manifest.get("thinking") or {}).get("type") or ""),
            "identity_blind": bool(run_manifest.get("identity_blind")),
            "neighbor_identity_policy": str(
                run_manifest.get("neighbor_identity_policy") or ""
            ),
            "tool_execution_mode": str(run_manifest.get("tool_execution_mode") or ""),
        }
        runtime_contracts.add(json.dumps(contract, sort_keys=True))
    if len(runtime_contracts) != 1:
        raise SystemExit(f"Frozen source batch has mixed runtime contracts: {source_batch}")
    runtime_contract = json.loads(next(iter(runtime_contracts)))
    required_contract = {
        "temperature": 0.0,
        "thinking_type": "disabled",
        "identity_blind": True,
        "neighbor_identity_policy": "parent_disjoint",
        "tool_execution_mode": "harness_prefetch",
    }
    mismatches = {
        key: {"actual": runtime_contract.get(key), "required": value}
        for key, value in required_contract.items()
        if runtime_contract.get(key) != value
    }
    if mismatches:
        raise SystemExit(
            f"Frozen source batch violates the matched ablation contract: "
            f"{source_batch} {json.dumps(mismatches, sort_keys=True)}"
        )
    return {
        "source_batch": str(source_batch),
        "expected_rows": expected_rows,
        "n_successful": metrics.get("n_successful"),
        "n_failed_runs": metrics.get("n_failed_runs"),
        "macro_f1": metrics.get("macro_f1"),
        "manifest_sha256": sha256_file(source_batch / "manifest.json"),
        "metrics_sha256": sha256_file(source_batch / "metrics.json"),
        "predictions_sha256": sha256_file(source_batch / "predictions.jsonl"),
        "runtime_contract": runtime_contract,
        "n_source_run_manifests_checked": len(source_runs),
    }


def _resolve_runtime_contract(source_audit: dict[str, dict[str, Any]]) -> dict[str, Any]:
    contracts = {
        json.dumps(audit["runtime_contract"], sort_keys=True)
        for audit in source_audit.values()
    }
    if len(contracts) != 1:
        raise SystemExit("Selected tasks do not share one frozen source runtime contract.")
    return json.loads(next(iter(contracts)))


def _apply_and_validate_runtime_contract(
    args: argparse.Namespace,
    source_contract: dict[str, Any],
) -> None:
    for attribute in ("model", "base_url", "reasoning_effort"):
        source_value = str(source_contract.get(attribute) or "")
        requested = getattr(args, attribute)
        if requested is None:
            setattr(args, attribute, source_value)
        elif str(requested) != source_value:
            raise SystemExit(
                f"--{attribute.replace('_', '-')}={requested!r} does not match "
                f"the frozen source value {source_value!r}."
            )


def _ensure_endpoint_api_key(api_key_env: str, base_url: str, model: str) -> None:
    if os.environ.get(api_key_env):
        return
    hostname = (urlparse(base_url).hostname or "").lower()
    if hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit(f"Missing required API key environment variable: {api_key_env}")
    os.environ[api_key_env] = (
        LOCAL_GPT_OSS_API_KEY if model == "gpt-oss-120b" else "local"
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--surfaces", nargs="+", choices=SURFACES, default=list(SURFACES))
    parser.add_argument("--benchmark-data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--canonical-paper-root", default=str(DEFAULT_CANONICAL_PAPER_ROOT))
    parser.add_argument("--source-run-root", default=str(DEFAULT_SOURCE_RUN_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--parallelism", type=int, default=128)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--max-stage-requeues", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--indices", nargs="*", default=None)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args(argv)
    if args.parallelism < 1 or args.parallelism > 512:
        parser.error("--parallelism must be in 1..512")
    if args.max_stage_requeues < 0:
        parser.error("--max-stage-requeues must be non-negative")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
