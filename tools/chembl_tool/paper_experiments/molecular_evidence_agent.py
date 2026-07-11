"""Run the frozen GLM-5.2 molecular-evidence-agent experiment matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys


PAPER_ROOT = Path("outputs/paper/molecular_evidence_agent")
GLM_BASE_URL = "https://litellm.parcc.upenn.edu/v1"
GLM_MODEL = "zai-org/GLM-5.2-FP8"


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
    ),
    *_task_experiments(
        "skin_reaction",
        "Skin_Reaction",
        "outputs/chembl_tool/tasks/skin_reaction/evidence_library/skin_reaction_neighbor_index.pkl",
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
    selected = _select_experiments(args.experiments)
    if args.list:
        print("\n".join(experiment.name for experiment in selected))
        return 0

    manifest = {
        "model": GLM_MODEL,
        "base_url": GLM_BASE_URL,
        "api_key_env": args.api_key_env,
        "identity_blind": True,
        "temperature": 0.0,
        "max_tokens": 20480,
        "transport_max_retries": 2,
        "experiments": [experiment.__dict__ for experiment in EXPERIMENTS],
        "selected_experiments": [experiment.name for experiment in selected],
    }
    PAPER_ROOT.mkdir(parents=True, exist_ok=True)
    (PAPER_ROOT / "experiment_matrix.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

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
    command = [
        args.python_executable,
        "-m",
        experiment.batch_module,
        "--input-jsonl",
        experiment.input_jsonl,
        "--index",
        experiment.index,
        "--batch-root",
        str(PAPER_ROOT / "runs" / experiment.task),
        "--batch-id",
        experiment.name,
        "--experiment-mode",
        experiment.mode,
        "--retrieval-source",
        experiment.source,
        "--identity-blind",
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
        "--skip-existing",
    ]
    if experiment.mode != "none":
        command.extend(
            [
                "--single-analysis-source-batch",
                str(PAPER_ROOT / "runs" / experiment.task / f"{experiment.task}__none"),
            ]
        )
    return command


def _select_experiments(names: list[str]) -> list[Experiment]:
    if not names:
        return list(EXPERIMENTS)
    by_name = {experiment.name: experiment for experiment in EXPERIMENTS}
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise SystemExit(f"Unknown experiments: {', '.join(missing)}")
    return [by_name[name] for name in names]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="*", default=[])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--api-key-env", default="GLM_API_KEY")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--parallelism", type=int, default=8)
    parser.add_argument("--group-workers", type=int, default=8)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
