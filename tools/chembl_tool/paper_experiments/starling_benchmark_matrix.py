"""Run the frozen paper pipeline on Starling random/scaffold benchmark tests."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from tools.chembl_tool.common.coverage_reasoning import (
    COVERAGE_MMP_LEDGER_NEIGHBOR_CONTEXT,
    NEIGHBOR_CONTEXT_PROFILES,
    STANDARD_NEIGHBOR_CONTEXT,
)
from tools.chembl_tool.common.neighbor_selection import (
    NEIGHBOR_SELECTORS,
    SIMILARITY_SELECTOR,
)
from tools.chembl_tool.common.json_utils import (
    write_json_atomic as _write_json_atomic,
)
from tools.chembl_tool.common.task_workflows.global_prompt_pool import (
    BatchCommand,
    SCHEDULER_VERSION as GLOBAL_PROMPT_POOL_VERSION,
    run_global_prompt_pool,
)

from .build_starling_benchmark_indices import (
    BENCHMARK_LINEAGE,
    BENCHMARK_SPLITS,
    paper_root_for_benchmark_split,
)
from .molecular_evidence_agent import (
    DEPLOYMENT_VISIBLE,
    EXPERIMENTS,
    GLM_API_KEY_ENV,
    GLM_BASE_URL,
    GLM_MODEL,
    GLM_REASONING_EFFORT,
    IDENTITY_BLIND,
    NEIGHBOR_IDENTITY_POLICIES,
    PARENT_DISJOINT,
    VISIBILITY_MODES,
    Experiment,
    _command,
    _prepare_policy_selection,
    _require_parent_disjoint_reuse_plans,
    _select_experiments,
    _visibility_contract,
    ensure_endpoint_api_key,
    experiment_run_root,
)
from .minimol_retrieval_contract import (
    MINIMOL_RETRIEVAL_FEATURE,
    MORGAN_RETRIEVAL_FEATURE,
    RETRIEVAL_FEATURES,
    descriptor_path_for_experiment,
    paper_root_for_minimol_retrieval,
)


GLOBAL_PROMPT_POOL_SCHEDULER = "global_prompt_pool"


TASK_DATA_NAMES = {
    "bbb_martins": "BBB_Martins",
    "bioavailability_ma": "Bioavailability_Ma",
    "skin_reaction": "Skin_Reaction",
}
DEFAULT_BENCHMARK_DATA_ROOT = Path("data/processed_starling")
EVALUATION_SUBSETS = ("valid", "test")
DEFAULT_ENDPOINT_CONCURRENCY_BUDGET = 512
DEFAULT_LAUNCHER_PARALLELISM = 128


def experiments_for_starling_benchmark(
    split: str,
    *,
    evaluation_subset: str = "test",
    retrieval_feature: str = MORGAN_RETRIEVAL_FEATURE,
) -> list[Experiment]:
    """Replace only benchmark inputs and held-out-filtered Starling indices."""
    if evaluation_subset not in EVALUATION_SUBSETS:
        raise ValueError(f"Unknown evaluation subset: {evaluation_subset}")
    if retrieval_feature not in RETRIEVAL_FEATURES:
        raise ValueError(f"Unknown retrieval feature: {retrieval_feature}")
    paper_root = paper_root_for_benchmark_split(split)
    experiments: list[Experiment] = []
    for experiment in EXPERIMENTS:
        data_name = TASK_DATA_NAMES.get(experiment.task)
        if data_name is None:
            continue
        input_jsonl = (
            f"data/processed_starling/{data_name}/{split}/{evaluation_subset}.jsonl"
        )
        index = experiment.index
        if experiment.source == "starling":
            index = str(_starling_index_path(experiment, paper_root))
        updated = replace(experiment, input_jsonl=input_jsonl, index=index)
        if retrieval_feature == MINIMOL_RETRIEVAL_FEATURE and updated.mode != "none":
            updated = replace(
                updated,
                index=str(descriptor_path_for_experiment(split, updated.name)),
            )
        experiments.append(updated)
    return experiments


def _starling_index_path(experiment: Experiment, paper_root: Path) -> Path:
    if experiment.task == "bbb_martins":
        name = "bbb_starling_v7"
    elif experiment.task == "skin_reaction":
        name = "skin_reaction_starling_v7"
    elif experiment.name.endswith("__starling_direct_numeric"):
        name = "bioavailability_starling_v7_direct_numeric"
    else:
        name = "bioavailability_starling_v7"
    return paper_root / "evidence" / name / "08_neighbor_index"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_concurrency(args)
    _validate_retrieval_ablation_args(args)
    args.fresh_parent_disjoint = (
        args.visibility_mode in {IDENTITY_BLIND, DEPLOYMENT_VISIBLE}
        and args.neighbor_identity_policy == PARENT_DISJOINT
    )
    if (
        args.retrieval_feature == MINIMOL_RETRIEVAL_FEATURE
        and args.visibility_mode not in {IDENTITY_BLIND, DEPLOYMENT_VISIBLE}
    ):
        raise SystemExit(
            "The MiniMol retrieval-feature ablation is frozen for "
            "deployment_visible only."
        )
    experiments = experiments_for_starling_benchmark(
        args.benchmark_split,
        evaluation_subset=args.evaluation_subset,
        retrieval_feature=args.retrieval_feature,
    )
    selected = _select_experiments(args.experiments, experiments=experiments)
    selected = _prepare_policy_selection(selected, args)
    if args.retrieval_feature == MINIMOL_RETRIEVAL_FEATURE:
        requested_none = [experiment.name for experiment in selected if experiment.mode == "none"]
        if args.experiments and requested_none:
            raise SystemExit(
                "MiniMol retrieval leaves query-only conditions unchanged; reuse the frozen "
                "Morgan-root none batches instead of selecting: "
                + ", ".join(requested_none)
            )
        selected = [experiment for experiment in selected if experiment.mode != "none"]
    if args.list:
        print("\n".join(experiment.name for experiment in selected))
        return 0

    canonical_paper_root = (
        paper_root_for_minimol_retrieval(args.benchmark_split)
        if args.retrieval_feature == MINIMOL_RETRIEVAL_FEATURE
        else paper_root_for_benchmark_split(args.benchmark_split)
    )
    paper_root = _paper_root_for_evaluation_subset(
        canonical_paper_root,
        args.evaluation_subset,
        output_root=args.output_root,
    )
    args.paper_root = str(paper_root)
    args.split = args.evaluation_subset
    if args.retrieval_feature == MINIMOL_RETRIEVAL_FEATURE:
        frozen_morgan_paper_root = _paper_root_for_evaluation_subset(
            paper_root_for_benchmark_split(args.benchmark_split),
            args.evaluation_subset,
        )
        frozen_morgan_root = experiment_run_root(
            IDENTITY_BLIND if args.fresh_parent_disjoint else DEPLOYMENT_VISIBLE,
            PARENT_DISJOINT if args.fresh_parent_disjoint else "operational",
            paper_root=frozen_morgan_paper_root,
        )
        if not args.single_analysis_root:
            args.single_analysis_root = str(frozen_morgan_root)
        if not args.fresh_parent_disjoint:
            args.group_analysis_root = str(frozen_morgan_root)
    _validate_inputs(selected)
    _require_parent_disjoint_reuse_plans(selected, args)
    benchmark_provenance = _benchmark_provenance(
        args.benchmark_split,
        experiments,
    )

    manifest: dict[str, Any] = {
        "benchmark_source": "starling",
        "dataset_lineage": BENCHMARK_LINEAGE,
        "benchmark_split": args.benchmark_split,
        "evaluation_subset": args.evaluation_subset,
        "retrieval_feature": args.retrieval_feature,
        "model": args.model,
        "served_model": args.model,
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "reasoning_effort": args.reasoning_effort,
        "reasoning": {
            "disable_thinking_flag": True,
            "reasoning_effort": args.reasoning_effort,
            "provider_reasoning_preserved": True,
        },
        "visibility_mode": args.visibility_mode,
        "visibility_contract": _visibility_contract(args.visibility_mode),
        "neighbor_identity_policy": args.neighbor_identity_policy,
        "neighbor_selector": args.neighbor_selector,
        "neighbor_context_profile": args.neighbor_context_profile,
        "exclude_nondirect_bioavailability_records": (
            bool(getattr(args, "exclude_nondirect_bioavailability_records", False))
        ),
        "nondirect_bioavailability_evidence_policy": {
            "default_policy": "include",
            "runtime_exclusion": bool(
                getattr(args, "exclude_nondirect_bioavailability_records", False)
            ),
            "filter_stage": "before_neighbor_ranking_and_top_k",
            "scope": "bioavailability_ma Starling retrieval conditions only",
        },
        "paper_root": str(paper_root),
        "canonical_paper_root": str(canonical_paper_root),
        "single_analysis_root": str(getattr(args, "single_analysis_root", "")),
        "group_analysis_root": str(getattr(args, "group_analysis_root", "")),
        "temperature": 0.0,
        "max_tokens": 20480,
        "timeout_s": args.timeout_s,
        "endpoint_concurrency_budget": DEFAULT_ENDPOINT_CONCURRENCY_BUDGET,
        "parallelism": args.parallelism,
        "global_pool_parallelism": args.parallelism,
        "scheduler": {
            "name": GLOBAL_PROMPT_POOL_SCHEDULER,
            "version": GLOBAL_PROMPT_POOL_VERSION,
            "max_stage_requeues": getattr(args, "max_stage_requeues", 0),
        },
        "effective_concurrency": args.parallelism,
        "fresh_parent_disjoint": args.fresh_parent_disjoint,
        "operational_staging_used": (
            args.neighbor_identity_policy == PARENT_DISJOINT
            and not args.fresh_parent_disjoint
        ),
        "limit": args.limit,
        "benchmark_provenance": benchmark_provenance,
        "experiments": [
            {
                **experiment.__dict__,
                "input_jsonl_sha256": benchmark_provenance[experiment.task][
                    f"{args.evaluation_subset}_jsonl_sha256"
                ],
                "effective_neighbor_identity_policy": (
                    "not_applicable"
                    if experiment.mode == "none"
                    else args.neighbor_identity_policy
                ),
                "benchmark_provenance_ref": experiment.task,
            }
            for experiment in experiments
        ],
        "selected_experiments": [experiment.name for experiment in selected],
    }
    paper_root.mkdir(parents=True, exist_ok=True)
    matrix_path = _matrix_manifest_path(paper_root, args, selected)
    _write_json_atomic(matrix_path, manifest)
    if args.manifest_only:
        print(json.dumps({"matrix_manifest": str(matrix_path)}, indent=2))
        return 0

    ensure_endpoint_api_key(args.api_key_env, args.base_url)
    failed = _run_selected_experiments(selected, args)
    if failed:
        print(json.dumps({"failed": failed}, indent=2), file=sys.stderr)
        return 1
    return 0


def _run_selected_experiments(
    selected: list[Experiment],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Share one ready queue and resolve frozen-single dependencies per sample."""
    commands = [
        BatchCommand(experiment.name, _command(experiment, args))
        for experiment in selected
    ]
    return run_global_prompt_pool(
        commands,
        max_workers=args.parallelism,
        max_stage_requeues=getattr(args, "max_stage_requeues", 0),
    )


def _paper_root_for_evaluation_subset(
    canonical_root: Path,
    evaluation_subset: str,
    *,
    output_root: str | Path | None = None,
) -> Path:
    """Resolve an isolated result root without changing canonical evidence paths."""
    if evaluation_subset not in EVALUATION_SUBSETS:
        raise ValueError(f"Unknown evaluation subset: {evaluation_subset}")
    if output_root:
        return Path(output_root)
    if evaluation_subset == "test":
        return canonical_root
    return canonical_root.with_name(f"{canonical_root.name}_valid")


def _validate_concurrency(args: argparse.Namespace) -> None:
    """Enforce the endpoint-wide request budget for the single-launcher protocol."""
    if args.parallelism < 1:
        raise SystemExit("--parallelism must be positive")
    if getattr(args, "max_stage_requeues", 0) < 0:
        raise SystemExit("--max-stage-requeues must be non-negative")
    requested = args.parallelism
    if requested > DEFAULT_ENDPOINT_CONCURRENCY_BUDGET:
        raise SystemExit(
            "Requested concurrency exceeds the frozen endpoint budget: "
            f"{requested} > "
            f"{DEFAULT_ENDPOINT_CONCURRENCY_BUDGET}"
        )


def _validate_retrieval_ablation_args(args: argparse.Namespace) -> None:
    """Keep experimental selectors/prompts out of canonical result roots."""
    nonstandard = (
        args.neighbor_selector != SIMILARITY_SELECTOR
        or args.neighbor_context_profile != STANDARD_NEIGHBOR_CONTEXT
    )
    if nonstandard and not args.output_root:
        raise SystemExit(
            "Non-standard neighbor selection/context requires an explicit --output-root."
        )
    if (
        bool(getattr(args, "exclude_nondirect_bioavailability_records", False))
        and not args.output_root
    ):
        raise SystemExit(
            "--exclude-nondirect-bioavailability-records requires an explicit "
            "--output-root to prevent --skip-existing reuse."
        )
    if args.retrieval_feature != MORGAN_RETRIEVAL_FEATURE and nonstandard:
        raise SystemExit(
            "Coverage selection/context currently requires --retrieval-feature morgan."
        )
    if (
        args.neighbor_context_profile == COVERAGE_MMP_LEDGER_NEIGHBOR_CONTEXT
        and args.visibility_mode == IDENTITY_BLIND
    ):
        raise SystemExit(
            "coverage_mmp_ledger is visible-only; choose deployment_visible or "
            "deployment_visible_prefetched."
        )


def _validate_inputs(experiments: list[Experiment]) -> None:
    missing: set[str] = set()
    for experiment in experiments:
        required = [experiment.input_jsonl]
        # Query-only ``none`` never loads or consults a retrieval index.  Its
        # inherited placeholder path must not block manifest validation.
        if experiment.mode != "none":
            required.append(experiment.index)
        for value in required:
            if not Path(value).exists():
                missing.add(value)
    if missing:
        raise SystemExit("Missing benchmark inputs:\n" + "\n".join(sorted(missing)))


def _benchmark_provenance(
    split: str,
    experiments: list[Experiment],
    *,
    data_root: Path = DEFAULT_BENCHMARK_DATA_ROOT,
) -> dict[str, dict[str, Any]]:
    provenance: dict[str, dict[str, Any]] = {}
    for task in sorted({experiment.task for experiment in experiments}):
        data_name = TASK_DATA_NAMES[task]
        task_dir = data_root / data_name
        split_dir = task_dir / split
        task_summary_path = task_dir / "summary.json"
        split_summary_path = split_dir / "summary.json"
        valid_path = split_dir / "valid.jsonl"
        test_path = split_dir / "test.jsonl"
        valid_labels_path = split_dir / "valid_molecule_labels.jsonl"
        test_labels_path = split_dir / "test_molecule_labels.jsonl"
        heldout_path = split_dir / "heldout_molecule_labels.jsonl"
        required = (
            task_summary_path,
            split_summary_path,
            valid_path,
            test_path,
            valid_labels_path,
            test_labels_path,
            heldout_path,
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise SystemExit(
                "Missing Starling benchmark provenance artifacts:\n"
                + "\n".join(missing)
            )
        task_summary = json.loads(task_summary_path.read_text(encoding="utf-8"))
        split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))
        provenance[task] = {
            "data_name": data_name,
            "protocol_version": task_summary.get("protocol_version"),
            "identity_normalizer_version": task_summary.get(
                "identity_normalizer_version"
            ),
            "seed": task_summary.get("seed"),
            "parent_label_policy": task_summary.get("parent_label_policy"),
            "split_size_policy": task_summary.get("split_size_policy"),
            "source_metadata": task_summary.get("source_metadata"),
            "split_summary": split_summary,
            "task_summary_path": str(task_summary_path),
            "task_summary_sha256": _sha256_file(task_summary_path),
            "split_summary_path": str(split_summary_path),
            "split_summary_sha256": _sha256_file(split_summary_path),
            "valid_jsonl": str(valid_path),
            "valid_jsonl_sha256": _sha256_file(valid_path),
            "test_jsonl": str(test_path),
            "test_jsonl_sha256": _sha256_file(test_path),
            "valid_molecule_labels_jsonl": str(valid_labels_path),
            "valid_molecule_labels_sha256": _sha256_file(valid_labels_path),
            "test_molecule_labels_jsonl": str(test_labels_path),
            "test_molecule_labels_sha256": _sha256_file(test_labels_path),
            "heldout_molecule_labels_jsonl": str(heldout_path),
            "heldout_molecule_labels_sha256": _sha256_file(heldout_path),
        }
    return provenance


def _matrix_manifest_path(
    paper_root: Path,
    args: argparse.Namespace,
    selected: list[Experiment],
) -> Path:
    suffix = f"{args.visibility_mode}_{args.neighbor_identity_policy}"
    if not args.experiments:
        return paper_root / f"experiment_matrix_{suffix}.json"
    names = [experiment.name for experiment in selected]
    tasks = "_".join(sorted({experiment.task for experiment in selected}))
    selection_hash = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()[:12]
    return paper_root / f"experiment_matrix_{suffix}_{tasks}_{selection_hash}.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-split", choices=BENCHMARK_SPLITS, required=True)
    parser.add_argument(
        "--evaluation-subset",
        choices=EVALUATION_SUBSETS,
        default="valid",
        help="Run v4 validation first; select test only after the settings are frozen.",
    )
    parser.add_argument(
        "--retrieval-feature",
        choices=RETRIEVAL_FEATURES,
        default=MORGAN_RETRIEVAL_FEATURE,
    )
    parser.add_argument("--experiments", nargs="*", default=[])
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Validate inputs and write the matrix manifest without launching conditions.",
    )
    parser.add_argument("--api-key-env", default=GLM_API_KEY_ENV)
    parser.add_argument("--base-url", default=GLM_BASE_URL)
    parser.add_argument("--model", default=GLM_MODEL)
    parser.add_argument(
        "--output-root",
        default="",
        help=(
            "Explicit model-specific result root. Canonical benchmark evidence "
            "indices remain under the split lineage root."
        ),
    )
    parser.add_argument("--reasoning-effort", default=GLM_REASONING_EFFORT)
    parser.add_argument("--visibility-mode", choices=VISIBILITY_MODES, default=IDENTITY_BLIND)
    parser.add_argument(
        "--neighbor-identity-policy",
        choices=NEIGHBOR_IDENTITY_POLICIES,
        default=PARENT_DISJOINT,
    )
    parser.add_argument(
        "--neighbor-selector",
        choices=NEIGHBOR_SELECTORS,
        default=SIMILARITY_SELECTOR,
    )
    parser.add_argument(
        "--neighbor-context-profile",
        choices=NEIGHBOR_CONTEXT_PROFILES,
        default=STANDARD_NEIGHBOR_CONTEXT,
    )
    parser.add_argument(
        "--exclude-nondirect-bioavailability-records",
        action="store_true",
        help=(
            "Bioavailability Starling only: exclude retained relative/apparent "
            "HF records before neighbor ranking and top-k selection."
        ),
    )
    parser.add_argument(
        "--single-analysis-root",
        default="",
        help="Optional frozen none-condition run root reused by retrieval conditions.",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--parallelism",
        type=int,
        default=DEFAULT_LAUNCHER_PARALLELISM,
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=300,
        help="Per OpenAI-compatible request timeout; raise only for targeted repairs.",
    )
    parser.add_argument(
        "--max-stage-requeues",
        type=int,
        default=0,
        help="Immediate stage-level retries after the built-in structured-output attempts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run at most this many test rows per selected condition; 0 runs the full test.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
