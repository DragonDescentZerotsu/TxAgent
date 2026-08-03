"""Build held-out Starling binary benchmarks for tasks with direct evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.benchmark_dataset import build_benchmark_dataset
from tools.chembl_tool.tasks.bbb_martins.starling_benchmark import (
    SOURCE_REVISION as BBB_REVISION,
)
from tools.chembl_tool.tasks.bbb_martins.starling_benchmark import (
    load_label_decisions as load_bbb,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_benchmark import (
    load_label_decisions as load_bioavailability,
)
from tools.chembl_tool.tasks.bioavailability_ma.canonical_source import DIRECT_CLAIMS_PATH
from tools.chembl_tool.tasks.skin_reaction.starling_benchmark import (
    load_label_decisions as load_skin_reaction,
)


TASK_OUTPUT_NAMES = {
    "bbb_martins": "BBB_Martins",
    "bioavailability_ma": "Bioavailability_Ma",
    "skin_reaction": "Skin_Reaction",
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.output_root)
    if args.summarize_existing:
        summaries = _collect_summaries(root, {})
        _write_root_summary(root, summaries)
        print(json.dumps(summaries, ensure_ascii=False, indent=2, default=str), flush=True)
        return 0

    tasks = list(TASK_OUTPUT_NAMES) if args.tasks == ["all"] else args.tasks
    summaries: dict[str, Any] = {}
    for task in tasks:
        decisions, source_metadata = _load_task(task, args)
        output_name = TASK_OUTPUT_NAMES[task]
        print(f"[starling-benchmark] building {output_name}", flush=True)
        summaries[task] = build_benchmark_dataset(
            task_name=output_name,
            decisions=decisions,
            source_metadata=source_metadata,
            output_dir=Path(args.output_root) / output_name,
            max_eval_size=args.max_eval_size,
            valid_fraction=args.valid_fraction,
            test_fraction=args.test_fraction,
            agreement_threshold=args.agreement_threshold,
            seed=args.seed,
            max_rejection_examples=args.max_rejection_examples,
        )
        print(
            f"[starling-benchmark] {output_name}: "
            f"random_valid={summaries[task]['splits']['random']['n_valid']:,} "
            f"random_test={summaries[task]['splits']['random']['n_test']:,} "
            f"scaffold_valid={summaries[task]['splits']['scaffold']['n_valid']:,} "
            f"scaffold_test={summaries[task]['splits']['scaffold']['n_test']:,}",
            flush=True,
        )

    summaries = _collect_summaries(root, summaries)
    _write_root_summary(root, summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


def _load_task(task: str, args: argparse.Namespace):
    if task == "bbb_martins":
        return load_bbb(revision=args.bbb_revision, max_rows=args.max_rows_per_source)
    if task == "bioavailability_ma":
        return load_bioavailability(
            source_path=args.bioavailability_source,
            max_rows=args.max_rows_per_source,
        )
    if task == "skin_reaction":
        return load_skin_reaction(max_rows=args.max_rows_per_source)
    raise ValueError(f"unsupported task: {task}")


def _collect_summaries(
    root: Path,
    current: dict[str, Any],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for task, output_name in TASK_OUTPUT_NAMES.items():
        if task in current:
            summaries[task] = current[task]
            continue
        summary_path = root / output_name / "summary.json"
        if summary_path.exists():
            summaries[task] = json.loads(summary_path.read_text(encoding="utf-8"))
    return summaries


def _write_root_summary(root: Path, summaries: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["all", *TASK_OUTPUT_NAMES],
        default=["all"],
    )
    parser.add_argument("--output-root", default="data/processed_starling")
    parser.add_argument("--max-eval-size", type=int, default=500)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--agreement-threshold", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--bbb-revision", default=BBB_REVISION)
    parser.add_argument("--bioavailability-source", default=str(DIRECT_CLAIMS_PATH))
    parser.add_argument("--max-rows-per-source", type=int, default=0)
    parser.add_argument("--max-rejection-examples", type=int, default=20)
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Rebuild only the root summary from existing task summaries.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
