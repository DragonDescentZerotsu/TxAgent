"""Run the complete Starling MiniMol-retrieval GLM agent experiment.

This is a resumable orchestration entrypoint. It delegates retrieval, reasoning,
parent-disjoint materialization, statistics, and plotting to the existing
paper-experiment modules rather than implementing experiment logic itself.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from tools.chembl_tool.paper_experiments.build_starling_benchmark_indices import (
    BENCHMARK_SPLITS,
)
from tools.chembl_tool.paper_experiments.minimol_retrieval_contract import (
    DEFAULT_FEATURE_ROOT,
    paper_root_for_minimol_retrieval,
)
from tools.chembl_tool.paper_experiments.molecular_evidence_agent import (
    GLM_API_KEY_ENV,
    GLM_BASE_URL,
    GLM_MODEL,
    GLM_REASONING_EFFORT,
    ensure_endpoint_api_key,
)
from tools.chembl_tool.paper_experiments.starling_benchmark_matrix import (
    experiments_for_starling_benchmark,
)


DEFAULT_OUTPUT = Path("outputs/paper/minimol_retrieval_agent_results")
POLICIES = ("operational", "parent_disjoint")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not (DEFAULT_FEATURE_ROOT / "summary.json").is_file():
        raise SystemExit(
            "MiniMol retrieval feature store is missing; run "
            "build_minimol_retrieval_features first."
        )
    ensure_endpoint_api_key(args.api_key_env, args.base_url)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "experiment_state.json"
    state: dict[str, Any] = {
        "type": "minimol_retrieval_agent_experiment_state.v1",
        "status": "running",
        "parallelism": args.parallelism,
        "group_workers": args.group_workers,
        "model": args.model,
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "reasoning_effort": args.reasoning_effort,
        "completed_phases": [],
    }
    _write_json_atomic(state_path, state)

    try:
        for policy in POLICIES:
            if policy == "parent_disjoint":
                _run_parent_disjoint_materialization(args)
                state["completed_phases"].append("parent_disjoint_materialization")
                _write_json_atomic(state_path, state)
            _run_matrix(policy, args)
            state["completed_phases"].append(policy)
            _write_json_atomic(state_path, state)

        _run_parent_disjoint_summaries(args)
        state["completed_phases"].append("parent_disjoint_summaries")
        _write_json_atomic(state_path, state)

        _run_final_summary_and_plot(args)
        state["completed_phases"].append("final_summary_and_plot")
        state["status"] = "completed"
        _write_json_atomic(state_path, state)
    except BaseException as exc:
        state["status"] = "failed"
        state["error_type"] = type(exc).__name__
        state["error"] = str(exc)
        _write_json_atomic(state_path, state)
        raise
    return 0


def _run_matrix(policy: str, args: argparse.Namespace) -> None:
    api_key_env = getattr(args, "api_key_env", GLM_API_KEY_ENV)
    base_url = getattr(args, "base_url", GLM_BASE_URL)
    model = getattr(args, "model", GLM_MODEL)
    reasoning_effort = getattr(args, "reasoning_effort", GLM_REASONING_EFFORT)
    commands: list[tuple[str, list[str]]] = []
    for split in BENCHMARK_SPLITS:
        by_task: dict[str, list[str]] = {}
        for experiment in experiments_for_starling_benchmark(
            split,
            retrieval_feature="minimol",
        ):
            if experiment.mode == "none":
                continue
            by_task.setdefault(experiment.task, []).append(experiment.name)
        for task, experiments in sorted(by_task.items()):
            commands.append(
                (
                    f"{policy}/{split}_{task}.log",
                    [
                        args.python_executable,
                        "-u",
                        "-m",
                        "tools.chembl_tool.paper_experiments.starling_benchmark_matrix",
                        "--benchmark-split",
                        split,
                        "--retrieval-feature",
                        "minimol",
                        "--visibility-mode",
                        "deployment_visible",
                        "--neighbor-identity-policy",
                        policy,
                        "--experiments",
                        *experiments,
                        "--parallelism",
                        str(args.parallelism),
                        "--group-workers",
                        str(args.group_workers),
                        "--api-key-env",
                        api_key_env,
                        "--base-url",
                        base_url,
                        "--model",
                        model,
                        "--reasoning-effort",
                        reasoning_effort,
                    ],
                )
            )
    _run_commands(commands, log_root=args.output_dir / "launcher_logs")


def _run_parent_disjoint_materialization(args: argparse.Namespace) -> None:
    commands = [
        (
            f"parent_disjoint_materialization/{split}.log",
            [
                args.python_executable,
                "-u",
                "-m",
                "tools.chembl_tool.paper_experiments.parent_disjoint_ablation",
                "--benchmark-split",
                split,
                "--retrieval-feature",
                "minimol",
                "--materialize",
            ],
        )
        for split in BENCHMARK_SPLITS
    ]
    _run_commands(commands, log_root=args.output_dir / "launcher_logs")


def _run_parent_disjoint_summaries(args: argparse.Namespace) -> None:
    commands = []
    for split in BENCHMARK_SPLITS:
        paper_root = paper_root_for_minimol_retrieval(split)
        commands.append(
            (
                f"parent_disjoint_summary/{split}.log",
                [
                    args.python_executable,
                    "-u",
                    "-m",
                    "tools.chembl_tool.paper_experiments.summarize_parent_disjoint_results",
                    "--operational-root",
                    str(paper_root / "runs_deployment_visible"),
                    "--parent-disjoint-root",
                    str(paper_root / "runs_deployment_visible_parent_disjoint"),
                    "--output-dir",
                    str(paper_root / "analysis/parent_disjoint_ablation"),
                ],
            )
        )
    _run_commands(commands, log_root=args.output_dir / "launcher_logs")


def _run_final_summary_and_plot(args: argparse.Namespace) -> None:
    figures = args.output_dir / "figures"
    commands = [
        (
            "final/summarize.log",
            [
                args.python_executable,
                "-u",
                "-m",
                "tools.chembl_tool.paper_experiments.summarize_minimol_retrieval_agent",
                "--output-dir",
                str(args.output_dir),
            ],
        ),
        (
            "final/plot.log",
            [
                args.python_executable,
                "-u",
                "-m",
                "tools.chembl_tool.paper_experiments.plot_minimol_retrieval_agent",
                "--metrics",
                str(args.output_dir / "condition_results.tsv"),
                "--output",
                str(figures / "minimol_vs_morgan_agent_retrieval.svg"),
                "--png-output",
                str(figures / "minimol_vs_morgan_agent_retrieval_highres.png"),
            ],
        ),
        (
            "final/plot_all_results.log",
            [
                args.python_executable,
                "-u",
                "-m",
                "tools.chembl_tool.paper_experiments.plot_starling_with_minimol_agent",
                "--output",
                str(figures / "starling_benchmark_with_minimol_agent.svg"),
                "--data-output",
                str(args.output_dir / "all_results_comparison.tsv"),
                "--png-output",
                str(figures / "starling_benchmark_with_minimol_agent_highres.png"),
            ],
        ),
    ]
    # Both plots consume completed summaries, so these commands are intentionally serial.
    for command in commands:
        _run_commands([command], log_root=args.output_dir / "launcher_logs")


def _run_commands(
    commands: list[tuple[str, list[str]]],
    *,
    log_root: Path,
) -> None:
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        futures = {
            executor.submit(_run_command, command, log_root / relative_log): relative_log
            for relative_log, command in commands
        }
        for future in as_completed(futures):
            relative_log = futures[future]
            return_code = future.result()
            if return_code:
                failures.append(f"{relative_log}: exit {return_code}")
    if failures:
        raise RuntimeError("Experiment subprocess failures: " + "; ".join(failures))


def _run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[orchestrator] command={json.dumps(command)}\n")
        handle.flush()
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        handle.write(f"[orchestrator] exit={completed.returncode}\n")
        handle.flush()
        return completed.returncode


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--api-key-env", default=GLM_API_KEY_ENV)
    parser.add_argument("--base-url", default=GLM_BASE_URL)
    parser.add_argument("--model", default=GLM_MODEL)
    parser.add_argument("--reasoning-effort", default=GLM_REASONING_EFFORT)
    parser.add_argument("--parallelism", type=int, default=8)
    parser.add_argument("--group-workers", type=int, default=4)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
