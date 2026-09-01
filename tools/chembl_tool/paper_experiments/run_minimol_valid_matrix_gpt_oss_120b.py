"""Run the current-lineage Starling valid MiniMol matrix serially by task.

This launcher is intentionally narrow: it reuses frozen 120B single analyses,
runs every retrieval-bearing condition with one shared endpoint consumer at a
time, and records resumable task-level state outside the result directories.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from tools.chembl_tool.common.json_utils import write_json_atomic


DEFAULT_OUTPUT_ROOT = Path(
    "outputs/paper/"
    "molecular_evidence_agent_scaffold_current_latest_valid_gpt_oss_120b_"
    "minimol_top5_nothreshold"
)
DEFAULT_CANONICAL_ROOT = Path(
    "outputs/paper/"
    "molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1_"
    "minimol_retrieval"
)


@dataclass(frozen=True)
class TaskPhase:
    task: str
    data_root: Path
    lineage: str
    feature_root: Path
    single_analysis_root: Path
    experiments: tuple[str, ...]


PHASES = (
    TaskPhase(
        task="bbb_martins",
        data_root=Path("data/processed_starling_experimental_meaningful_cns_access_v2"),
        lineage="experimental_meaningful_cns_access_v2",
        feature_root=Path(
            "outputs/paper/minimol_retrieval_features_scaffold_current_latest_valid_bbb_v2"
        ),
        single_analysis_root=Path(
            "outputs/paper/"
            "molecular_evidence_agent_starling_scaffold_"
            "experimental_meaningful_cns_access_v2_valid_gpt_oss_120b/"
            "runs_identity_blind_parent_disjoint"
        ),
        experiments=(
            "bbb_martins__chembl_direct",
            "bbb_martins__chembl_full_flat",
            "bbb_martins__chembl_full_mechanism",
            "bbb_martins__starling_direct",
            "bbb_martins__starling_full_flat",
            "bbb_martins__starling_full_mechanism",
        ),
    ),
    TaskPhase(
        task="bioavailability_ma",
        data_root=Path("data/processed_starling_record_supported_v2"),
        lineage="record_supported_v2",
        feature_root=Path(
            "outputs/paper/minimol_retrieval_features_scaffold_current_latest_valid_bio_skin_v2"
        ),
        single_analysis_root=Path(
            "outputs/paper/"
            "molecular_evidence_agent_starling_scaffold_record_supported_v2_valid_"
            "gpt_oss_120b_bugfix_v2/runs_identity_blind_parent_disjoint"
        ),
        experiments=(
            "bioavailability_ma__chembl_direct",
            "bioavailability_ma__starling_direct_numeric",
            "bioavailability_ma__starling_direct_full",
            "bioavailability_ma__chembl_full_flat",
            "bioavailability_ma__chembl_full_mechanism",
            "bioavailability_ma__starling_full_flat",
            "bioavailability_ma__starling_full_mechanism",
        ),
    ),
    TaskPhase(
        task="skin_reaction",
        data_root=Path("data/processed_starling_record_supported_v2"),
        lineage="record_supported_v2",
        feature_root=Path(
            "outputs/paper/minimol_retrieval_features_scaffold_current_latest_valid_bio_skin_v2"
        ),
        single_analysis_root=Path(
            "outputs/paper/"
            "molecular_evidence_agent_starling_scaffold_record_supported_v2_valid_"
            "gpt_oss_120b_bugfix_v2/runs_identity_blind_parent_disjoint"
        ),
        experiments=(
            "skin_reaction__chembl_direct",
            "skin_reaction__chembl_full_flat",
            "skin_reaction__chembl_full_mechanism",
            "skin_reaction__starling_direct",
            "skin_reaction__starling_full_flat",
            "skin_reaction__starling_full_mechanism",
        ),
    ),
)

STARLING_CORE_EXPERIMENTS = {
    "bbb_martins": (
        "bbb_martins__starling_direct",
        "bbb_martins__starling_full_flat",
    ),
    "bioavailability_ma": (
        "bioavailability_ma__starling_direct_full",
        "bioavailability_ma__starling_full_flat",
    ),
    "skin_reaction": (
        "skin_reaction__starling_direct",
        "skin_reaction__starling_full_flat",
    ),
}


def phases_for_profile(profile: str) -> tuple[TaskPhase, ...]:
    if profile == "all":
        return PHASES
    if profile == "starling_core":
        return tuple(
            replace(phase, experiments=STARLING_CORE_EXPERIMENTS[phase.task])
            for phase in PHASES
        )
    raise ValueError(f"Unknown condition profile: {profile}")


def command_for_phase(phase: TaskPhase, args: argparse.Namespace) -> list[str]:
    return [
        args.python_executable,
        "-u",
        "-m",
        "tools.chembl_tool.paper_experiments.starling_benchmark_matrix",
        "--benchmark-split",
        "scaffold",
        "--evaluation-subset",
        "valid",
        "--retrieval-feature",
        "minimol",
        "--minimol-feature-root",
        str(phase.feature_root),
        "--benchmark-data-root",
        str(phase.data_root),
        "--benchmark-lineage",
        phase.lineage,
        "--canonical-paper-root",
        str(args.canonical_paper_root),
        "--output-root",
        str(args.output_root),
        "--single-analysis-root",
        str(phase.single_analysis_root),
        "--visibility-mode",
        "identity_blind",
        "--neighbor-identity-policy",
        "parent_disjoint",
        "--top-k-per-group",
        "5",
        "--min-similarity",
        "0.0",
        "--parallelism",
        str(args.parallelism),
        "--timeout-s",
        str(args.timeout_s),
        "--api-key-env",
        args.api_key_env,
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--reasoning-effort",
        "",
        "--experiments",
        *phase.experiments,
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_root = args.output_root / "launcher_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / "current_minimol_valid_matrix_state.json"
    completed = (
        _completed_phases(state_path, condition_profile=args.condition_profile)
        if args.resume
        else []
    )
    state: dict[str, Any] = {
        "type": "current_minimol_valid_matrix_state.v1",
        "status": "running",
        "completed_phases": completed,
        "active_phase": None,
        "condition_profile": args.condition_profile,
        "top_k_per_group": 5,
        "min_similarity": 0.0,
        "parallelism": args.parallelism,
        "base_url": args.base_url,
        "model": args.model,
    }
    write_json_atomic(state_path, state)
    env = os.environ.copy()
    env[args.api_key_env] = args.api_key

    for phase in phases_for_profile(args.condition_profile):
        if phase.task in completed:
            continue
        state["active_phase"] = phase.task
        write_json_atomic(state_path, state)
        command = command_for_phase(phase, args)
        log_path = log_root / f"{phase.task}.log"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n[orchestrator] command=" + json.dumps(command) + "\n")
            handle.flush()
            result = subprocess.run(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
            handle.write(f"[orchestrator] exit={result.returncode}\n")
            handle.flush()
        if result.returncode:
            state.update(status="failed", return_code=result.returncode)
            write_json_atomic(state_path, state)
            return result.returncode
        completed.append(phase.task)
        state["completed_phases"] = completed
        state["active_phase"] = None
        write_json_atomic(state_path, state)

    state.update(status="completed", active_phase=None)
    write_json_atomic(state_path, state)
    return 0


def _completed_phases(state_path: Path, *, condition_profile: str) -> list[str]:
    if not state_path.is_file():
        return []
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if payload.get("condition_profile") != condition_profile:
        return []
    return [str(value) for value in payload.get("completed_phases") or []]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--canonical-paper-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--parallelism", type=int, default=128)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--api-key-env", default="GPT_OSS_LOCAL_API_KEY")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--base-url", default="http://127.0.0.1:9001/v1")
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument(
        "--condition-profile",
        choices=("all", "starling_core"),
        default="all",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
