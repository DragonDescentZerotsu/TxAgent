"""Run the frozen GLM-5.2 molecular-evidence-agent experiment matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PAPER_ROOT = Path("outputs/paper/molecular_evidence_agent")
GLM_BASE_URL = "https://litellm.parcc.upenn.edu/v1"
GLM_MODEL = "zai-org/GLM-5.2-FP8"
IDENTITY_BLIND = "identity_blind"
DEPLOYMENT_VISIBLE = "deployment_visible"
DEPLOYMENT_VISIBLE_PREFETCHED = "deployment_visible_prefetched"
VISIBILITY_MODES = (IDENTITY_BLIND, DEPLOYMENT_VISIBLE_PREFETCHED, DEPLOYMENT_VISIBLE)
OPERATIONAL = "operational"
PARENT_DISJOINT = "parent_disjoint"
NEIGHBOR_IDENTITY_POLICIES = (OPERATIONAL, PARENT_DISJOINT)
DATA_SPLITS = ("test", "valid")


@dataclass(frozen=True)
class Experiment:
    name: str
    task: str
    batch_module: str
    input_jsonl: str
    index: str
    mode: str
    source: str


def _task_experiments(
    task: str,
    data_name: str,
    index: str,
    *,
    include_starling_direct: str = "",
    include_starling_full: str = "",
) -> list[Experiment]:
    module = f"tools.chembl_tool.tasks.{task}.run_reasoning_batch"
    input_jsonl = f"data/processed/{data_name}/test.jsonl"
    experiments = [
        Experiment(f"{task}__none", task, module, input_jsonl, index, "none", "chembl"),
        Experiment(f"{task}__chembl_direct", task, module, input_jsonl, index, "direct", "chembl"),
        Experiment(f"{task}__chembl_full_flat", task, module, input_jsonl, index, "full_flat", "chembl"),
        Experiment(
            f"{task}__chembl_full_mechanism",
            task,
            module,
            input_jsonl,
            index,
            "full_mechanism",
            "chembl",
        ),
    ]
    if include_starling_direct:
        experiments.append(
            Experiment(
                f"{task}__starling_direct",
                task,
                module,
                input_jsonl,
                include_starling_direct,
                "direct",
                "starling",
            )
        )
    if include_starling_full:
        experiments.extend(
            [
                Experiment(
                    f"{task}__starling_full_flat",
                    task,
                    module,
                    input_jsonl,
                    include_starling_full,
                    "full_flat",
                    "starling",
                ),
                Experiment(
                    f"{task}__starling_full_mechanism",
                    task,
                    module,
                    input_jsonl,
                    include_starling_full,
                    "full_mechanism",
                    "starling",
                ),
            ]
        )
    return experiments


EXPERIMENTS = [
    *_task_experiments(
        "bbb_martins",
        "BBB_Martins",
        "outputs/chembl_tool/tasks/bbb_martins/evidence_library/bbb_neighbor_index.pkl",
        include_starling_direct=(
            "outputs/paper/molecular_evidence_agent/evidence/bbb_starling/all/"
            "starling_bbb_neighbor_index.pkl"
        ),
        include_starling_full=(
            "outputs/paper/molecular_evidence_agent/evidence/bbb_starling_full/"
            "starling_bbb_neighbor_index.pkl"
        ),
    ),
    *_task_experiments(
        "skin_reaction",
        "Skin_Reaction",
        "outputs/chembl_tool/tasks/skin_reaction/evidence_library/skin_reaction_neighbor_index.pkl",
        include_starling_direct=(
            "outputs/paper/molecular_evidence_agent/evidence/skin_reaction_starling_full/"
            "starling_skin_reaction_neighbor_index.pkl"
        ),
        include_starling_full=(
            "outputs/paper/molecular_evidence_agent/evidence/skin_reaction_starling_full/"
            "starling_skin_reaction_neighbor_index.pkl"
        ),
    ),
    *_task_experiments(
        "clintox",
        "ClinTox",
        "outputs/chembl_tool/tasks/clintox/evidence_library/clintox_neighbor_index.pkl",
    ),
    Experiment(
        "bioavailability_ma__none",
        "bioavailability_ma",
        "tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch",
        "data/processed/Bioavailability_Ma/test.jsonl",
        "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/bioavailability_neighbor_index.pkl",
        "none",
        "chembl",
    ),
    Experiment(
        "bioavailability_ma__chembl_direct",
        "bioavailability_ma",
        "tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch",
        "data/processed/Bioavailability_Ma/test.jsonl",
        "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/bioavailability_neighbor_index.pkl",
        "direct",
        "chembl",
    ),
    Experiment(
        "bioavailability_ma__starling_direct_numeric",
        "bioavailability_ma",
        "tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch",
        "data/processed/Bioavailability_Ma/test.jsonl",
        str(PAPER_ROOT / "evidence/bioavailability_starling_direct_numeric/starling_factor_neighbor_index.pkl"),
        "direct",
        "starling",
    ),
    Experiment(
        "bioavailability_ma__starling_direct_full",
        "bioavailability_ma",
        "tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch",
        "data/processed/Bioavailability_Ma/test.jsonl",
        str(PAPER_ROOT / "evidence/bioavailability_starling_full/starling_factor_neighbor_index.pkl"),
        "direct",
        "starling",
    ),
    Experiment(
        "bioavailability_ma__chembl_full_flat",
        "bioavailability_ma",
        "tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch",
        "data/processed/Bioavailability_Ma/test.jsonl",
        "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/bioavailability_neighbor_index.pkl",
        "full_flat",
        "chembl",
    ),
    Experiment(
        "bioavailability_ma__chembl_full_mechanism",
        "bioavailability_ma",
        "tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch",
        "data/processed/Bioavailability_Ma/test.jsonl",
        "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/bioavailability_neighbor_index.pkl",
        "full_mechanism",
        "chembl",
    ),
    Experiment(
        "bioavailability_ma__starling_full_flat",
        "bioavailability_ma",
        "tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch",
        "data/processed/Bioavailability_Ma/test.jsonl",
        str(PAPER_ROOT / "evidence/bioavailability_starling_full/starling_factor_neighbor_index.pkl"),
        "full_flat",
        "starling",
    ),
    Experiment(
        "bioavailability_ma__starling_full_mechanism",
        "bioavailability_ma",
        "tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch",
        "data/processed/Bioavailability_Ma/test.jsonl",
        str(PAPER_ROOT / "evidence/bioavailability_starling_full/starling_factor_neighbor_index.pkl"),
        "full_mechanism",
        "starling",
    ),
]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    all_experiments = experiments_for_split(args.split)
    selected = _select_experiments(args.experiments, experiments=all_experiments)
    selected = _prepare_policy_selection(selected, args)
    if args.list:
        print("\n".join(experiment.name for experiment in selected))
        return 0
    _require_parent_disjoint_reuse_plans(selected, args)

    manifest = {
        "model": GLM_MODEL,
        "base_url": GLM_BASE_URL,
        "api_key_env": args.api_key_env,
        "visibility_mode": args.visibility_mode,
        "identity_blind": args.visibility_mode == IDENTITY_BLIND,
        "visibility_contract": _visibility_contract(args.visibility_mode),
        "neighbor_identity_policy": args.neighbor_identity_policy,
        "data_split": args.split,
        "paper_root": str(_paper_root_from_args(args)),
        "temperature": 0.0,
        "max_tokens": 20480,
        "transport_max_retries": 2,
        "experiments": [experiment.__dict__ for experiment in all_experiments],
        "selected_experiments": [experiment.name for experiment in selected],
    }
    paper_root = _paper_root_from_args(args)
    paper_root.mkdir(parents=True, exist_ok=True)
    if args.experiments:
        selected_tasks = "_".join(sorted({experiment.task for experiment in selected}))
        policy_suffix = "" if args.neighbor_identity_policy == OPERATIONAL else f"_{args.neighbor_identity_policy}"
        matrix_path = paper_root / f"experiment_matrix_{args.visibility_mode}{policy_suffix}_{selected_tasks}.json"
    else:
        policy_suffix = "" if args.neighbor_identity_policy == OPERATIONAL else f"_{args.neighbor_identity_policy}"
        matrix_path = (
            paper_root / "experiment_matrix.json"
            if args.visibility_mode == IDENTITY_BLIND and not policy_suffix
            else paper_root / f"experiment_matrix_{args.visibility_mode}{policy_suffix}.json"
        )
    matrix_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    failed = []
    for experiment in selected:
        command = _command(experiment, args)
        print(f"[paper_matrix] starting {experiment.name}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            failed.append({"experiment": experiment.name, "returncode": completed.returncode})
    if failed:
        print(json.dumps({"failed": failed}, indent=2), file=sys.stderr)
        return 1
    return 0


def _command(experiment: Experiment, args: argparse.Namespace) -> list[str]:
    visibility_mode = getattr(args, "visibility_mode", IDENTITY_BLIND)
    neighbor_identity_policy = getattr(args, "neighbor_identity_policy", OPERATIONAL)
    split = getattr(args, "split", "test")
    experiment = experiment_for_split(experiment, split)
    paper_root = _paper_root_from_args(args)
    batch_root = experiment_run_root(
        visibility_mode,
        neighbor_identity_policy,
        paper_root=paper_root,
    ) / experiment.task
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
        experiment.name,
        "--experiment-mode",
        experiment.mode,
        "--retrieval-source",
        experiment.source,
        "--neighbor-identity-policy",
        neighbor_identity_policy,
        "--api-key-env",
        args.api_key_env,
        "--base-url",
        GLM_BASE_URL,
        "--model",
        GLM_MODEL,
        "--disable-thinking",
        "--reasoning-effort",
        "",
        "--temperature",
        "0",
        "--max-tokens",
        "20480",
        "--timeout-s",
        "300",
        "--max-tool-rounds",
        "3",
        "--parallelism",
        str(args.parallelism),
        "--group-workers",
        str(args.group_workers),
        "--no-stream-logs",
        "--no-combine-traces",
        "--skip-existing",
    ]
    if visibility_mode == IDENTITY_BLIND:
        command.append("--identity-blind")
    elif visibility_mode == DEPLOYMENT_VISIBLE_PREFETCHED:
        command.append("--harness-prefetch-tools")
        if experiment.mode != "none":
            blind_batch = (
                experiment_run_root(IDENTITY_BLIND, paper_root=paper_root)
                / experiment.task
                / experiment.name
            )
            command.extend(["--retrieval-replay-source-batch", str(blind_batch)])
            command.extend(["--prefetched-tool-replay-source-batch", str(blind_batch)])
    if experiment.mode != "none":
        single_root = experiment_run_root(visibility_mode, OPERATIONAL, paper_root=paper_root)
        command.extend(
            [
                "--single-analysis-source-batch",
                str(single_root / experiment.task / f"{experiment.task}__none")
                if neighbor_identity_policy == PARENT_DISJOINT
                else str(batch_root / f"{experiment.task}__none"),
            ]
        )
        if neighbor_identity_policy == PARENT_DISJOINT:
            command.extend(
                [
                    "--group-analysis-source-batch",
                    str(single_root / experiment.task / experiment.name),
                ]
            )
    return command


def experiment_run_root(
    visibility_mode: str,
    neighbor_identity_policy: str = OPERATIONAL,
    *,
    paper_root: Path = PAPER_ROOT,
) -> Path:
    """Return the stable batch root for one paper visibility regime."""
    if neighbor_identity_policy == PARENT_DISJOINT:
        if visibility_mode != DEPLOYMENT_VISIBLE:
            raise ValueError("Parent-disjoint paper ablation is defined for deployment-visible agentic runs.")
        return paper_root / "runs_deployment_visible_parent_disjoint"
    if neighbor_identity_policy != OPERATIONAL:
        raise ValueError(f"Unknown neighbor identity policy: {neighbor_identity_policy}")
    if visibility_mode == IDENTITY_BLIND:
        return paper_root / "runs"
    if visibility_mode == DEPLOYMENT_VISIBLE:
        return paper_root / "runs_deployment_visible"
    if visibility_mode == DEPLOYMENT_VISIBLE_PREFETCHED:
        return paper_root / "runs_deployment_visible_prefetched"
    raise ValueError(f"Unknown visibility mode: {visibility_mode}")


def experiment_result_name(experiment_name: str, visibility_mode: str) -> str:
    """Keep frozen blind names stable and qualify deployment-visible results."""
    if visibility_mode == IDENTITY_BLIND:
        return experiment_name
    return f"{visibility_mode}__{experiment_name}"


def _visibility_contract(visibility_mode: str) -> dict[str, Any]:
    if visibility_mode == IDENTITY_BLIND:
        return {
            "query_structure": "hidden_from_llm",
            "query_name": "not_provided",
            "neighbor_structure": "hidden_from_llm",
            "neighbor_name": "hidden_from_llm",
            "tool_execution": "harness_prefetch",
        }
    contract = {
        "query_structure": "visible_to_llm",
        "query_name": "not_provided",
        "neighbor_structure": "visible_to_llm",
        "neighbor_name": "visible_when_present_in_source_evidence",
    }
    contract["tool_execution"] = (
        "harness_prefetch" if visibility_mode == DEPLOYMENT_VISIBLE_PREFETCHED else "llm_function_call"
    )
    return contract


def paper_root_for_split(split: str) -> Path:
    """Keep validation artifacts isolated from the frozen test-set paper roots."""
    if split not in DATA_SPLITS:
        raise ValueError(f"Unknown data split: {split}")
    if split == "test":
        return PAPER_ROOT
    return PAPER_ROOT.with_name(f"{PAPER_ROOT.name}_{split}")


def experiment_for_split(experiment: Experiment, split: str) -> Experiment:
    if split not in DATA_SPLITS:
        raise ValueError(f"Unknown data split: {split}")
    input_path = Path(experiment.input_jsonl)
    if input_path.name == f"{split}.jsonl":
        return experiment
    if input_path.name != "test.jsonl":
        raise ValueError(f"Frozen experiment input is not a test split: {input_path}")
    return replace(experiment, input_jsonl=str(input_path.with_name(f"{split}.jsonl")))


def experiments_for_split(split: str) -> list[Experiment]:
    return [experiment_for_split(experiment, split) for experiment in EXPERIMENTS]


def _paper_root_from_args(args: argparse.Namespace) -> Path:
    explicit = getattr(args, "paper_root", None)
    return Path(explicit) if explicit else paper_root_for_split(getattr(args, "split", "test"))


def _select_experiments(
    names: list[str],
    *,
    experiments: list[Experiment] = EXPERIMENTS,
) -> list[Experiment]:
    if not names:
        return list(experiments)
    by_name = {experiment.name: experiment for experiment in experiments}
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise SystemExit(f"Unknown experiments: {', '.join(missing)}")
    return [by_name[name] for name in names]


def _prepare_policy_selection(
    selected: list[Experiment],
    args: argparse.Namespace,
) -> list[Experiment]:
    """Keep the primary parent-disjoint run analog-only and reuse query-only baselines."""
    if args.neighbor_identity_policy != PARENT_DISJOINT:
        return selected
    if args.visibility_mode != DEPLOYMENT_VISIBLE:
        raise SystemExit("parent_disjoint requires --visibility-mode deployment_visible")
    selected_none = [experiment.name for experiment in selected if experiment.mode == "none"]
    if args.experiments and selected_none:
        raise SystemExit(
            "Query-only conditions are policy-invariant and must remain in the operational root; "
            f"remove from parent-disjoint selection: {', '.join(selected_none)}"
        )
    return [experiment for experiment in selected if experiment.mode != "none"]


def _require_parent_disjoint_reuse_plans(
    selected: list[Experiment],
    args: argparse.Namespace,
) -> None:
    """Prevent an accidental full rerun when selective parent reuse was not planned."""
    if args.neighbor_identity_policy != PARENT_DISJOINT:
        return
    target_root = experiment_run_root(
        DEPLOYMENT_VISIBLE,
        PARENT_DISJOINT,
        paper_root=_paper_root_from_args(args),
    )
    missing = [
        experiment.name
        for experiment in selected
        if not (target_root / experiment.task / experiment.name / "reuse_plan.json").exists()
    ]
    if missing:
        raise SystemExit(
            "Missing parent-disjoint reuse plans. Run "
            "python -m tools.chembl_tool.paper_experiments.parent_disjoint_ablation "
            f"--materialize first. Missing: {', '.join(missing)}"
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="*", default=[])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--api-key-env", default="GLM_API_KEY")
    parser.add_argument("--visibility-mode", choices=VISIBILITY_MODES, default=DEPLOYMENT_VISIBLE)
    parser.add_argument("--split", choices=DATA_SPLITS, default="test")
    parser.add_argument(
        "--paper-root",
        default="",
        help="Override the split-specific output root (default: frozen test root or a sibling _valid root).",
    )
    parser.add_argument(
        "--neighbor-identity-policy",
        choices=NEIGHBOR_IDENTITY_POLICIES,
        default=PARENT_DISJOINT,
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--parallelism", type=int, default=8)
    parser.add_argument("--group-workers", type=int, default=8)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
