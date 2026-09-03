"""Run and audit the BBB/Bio final-only train-ratio tie-break experiment."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

from tools.chembl_tool.common.final_decision_prior import TRAIN_RATIO_TIEBREAK_V1, TrainRatioPrior
from tools.chembl_tool.common.json_utils import (
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.common.task_workflows.global_prompt_pool import (
    BatchCommand,
    SCHEDULER_VERSION,
    run_global_prompt_pool,
)
from tools.chembl_tool.paper_experiments.run_final_evidence_surface_experiment import (
    _apply_and_validate_runtime_contract,
    _ensure_endpoint_api_key,
    _resolve_runtime_contract,
    _validate_source_batch,
)
from tools.chembl_tool.paper_experiments.train_ratio_prior_analysis import (
    TrainRatioAnalysisSpec,
    analyze_train_ratio_prior,
    render_train_ratio_report,
)
from tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline import (
    TRAIN_RATIO_PRIOR as BBB_TRAIN_PRIOR,
)
from tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_pipeline import (
    TRAIN_RATIO_PRIOR as BIO_TRAIN_PRIOR,
)
from tools.chembl_tool.tasks.bioavailability_ma.prompt_profiles import (
    LEGACY_BIOAVAILABILITY_V1,
)


SCHEMA_VERSION = "starling.train_ratio_prior_experiment.v1"
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/paper/train_ratio_tiebreak_v1_scaffold_valid_gpt_oss_120b"
)
DEFAULT_API_KEY_ENV = "GPT_OSS_LOCAL_API_KEY"


@dataclass(frozen=True)
class TaskSpec:
    task: str
    batch_module: str
    prediction_field: str
    source_batch: Path
    train_jsonl: Path
    expected_valid_rows: int
    prior: TrainRatioPrior
    prompt_profile_args: tuple[str, ...] = ()

    @property
    def candidate_batch_id(self) -> str:
        return f"{self.task}__starling_full_flat__{TRAIN_RATIO_TIEBREAK_V1}"


TASK_SPECS = {
    "bbb_martins": TaskSpec(
        task="bbb_martins",
        batch_module="tools.chembl_tool.tasks.bbb_martins.run_reasoning_batch",
        prediction_field="bbb_prediction",
        source_batch=Path(
            "outputs/paper/molecular_evidence_agent_starling_scaffold_"
            "experimental_meaningful_cns_access_v2_valid_gpt_oss_120b/"
            "runs_identity_blind_parent_disjoint/bbb_martins/"
            "bbb_martins__starling_full_flat"
        ),
        train_jsonl=Path(
            "data/gold_labels/legacy/processed_starling_experimental_meaningful_cns_access_v2/"
            "BBB_Martins/scaffold/train.jsonl"
        ),
        expected_valid_rows=366,
        prior=BBB_TRAIN_PRIOR,
    ),
    "bioavailability_ma": TaskSpec(
        task="bioavailability_ma",
        batch_module=(
            "tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch"
        ),
        prediction_field="bioavailability_prediction",
        source_batch=Path(
            "outputs/paper/molecular_evidence_agent_starling_scaffold_"
            "record_supported_v2_valid_gpt_oss_120b_bugfix_v2/"
            "runs_identity_blind_parent_disjoint/bioavailability_ma/"
            "bioavailability_ma__starling_full_flat"
        ),
        train_jsonl=Path(
            "data/gold_labels/legacy/processed_starling_record_supported_v2/"
            "Bioavailability_Ma/scaffold/train.jsonl"
        ),
        expected_valid_rows=209,
        prior=BIO_TRAIN_PRIOR,
        prompt_profile_args=(
            "--bioavailability-prompt-profile",
            LEGACY_BIOAVAILABILITY_V1,
        ),
    ),
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    specs = [TASK_SPECS[task] for task in dict.fromkeys(args.tasks)]
    source_audit = {
        spec.task: _validate_source_batch(
            spec.source_batch,
            expected_rows=spec.expected_valid_rows,
            allow_partial=False,
        )
        for spec in specs
    }
    train_prior_audit = {
        spec.task: _validate_train_prior(spec) for spec in specs
    }
    runtime_contract = _resolve_runtime_contract(source_audit)
    _apply_and_validate_runtime_contract(args, runtime_contract)
    output_root = Path(args.output_root)
    candidate_batches = {
        spec.task: _candidate_batch_path(output_root, spec) for spec in specs
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "profile": TRAIN_RATIO_TIEBREAK_V1,
        "tasks": [spec.task for spec in specs],
        "evaluation_subset": "valid",
        "benchmark_split": "scaffold",
        "visibility_mode": "identity_blind",
        "neighbor_identity_policy": "parent_disjoint",
        "source_condition": "starling_full_flat",
        "task_prompt_profile_args": {
            spec.task: list(spec.prompt_profile_args) for spec in specs
        },
        "source_audit": source_audit,
        "train_prior_audit": train_prior_audit,
        "candidate_batches": {
            task: str(path) for task, path in candidate_batches.items()
        },
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
        "promotion_gate": {
            "macro_f1_delta": "positive",
            "macro_f1_bootstrap_95ci_lower": "> 0",
            "accuracy": "non_decreasing",
        },
        "output_root": str(output_root),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_or_validate_manifest(output_root / "experiment_manifest.json", manifest)
    if args.manifest_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    if not args.analyze_only:
        _ensure_endpoint_api_key(args.api_key_env, args.base_url, args.model)
        commands = [
            BatchCommand(spec.task, _batch_command(spec, args)) for spec in specs
        ]
        failed = run_global_prompt_pool(
            commands,
            max_workers=args.parallelism,
            max_stage_requeues=args.max_stage_requeues,
        )
        if failed:
            print(json.dumps({"failed": failed}, indent=2), file=sys.stderr)
            return 1

    summary, sample_rows = analyze_train_ratio_prior(
        [
            TrainRatioAnalysisSpec(
                task=spec.task,
                prediction_field=spec.prediction_field,
                source_batch=spec.source_batch,
                candidate_batch=candidate_batches[spec.task],
                prior=spec.prior,
            )
            for spec in specs
        ]
    )
    analysis_root = output_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(analysis_root / "summary.json", summary)
    write_jsonl_atomic(analysis_root / "sample_audit.jsonl", sample_rows)
    (analysis_root / "report.md").write_text(
        render_train_ratio_report(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _batch_command(spec: TaskSpec, args: argparse.Namespace) -> list[str]:
    source_manifest = _read_json(spec.source_batch / "manifest.json")
    source_run_manifest = _representative_run_manifest(spec.source_batch)
    batch_root = (
        Path(args.output_root)
        / "runs_identity_blind_parent_disjoint"
        / spec.task
    )
    command = [
        args.python_executable,
        "-m",
        spec.batch_module,
        "--input-jsonl",
        str(source_manifest["input_jsonl"]),
        "--index",
        str(source_run_manifest["neighbor_index"]),
        "--batch-root",
        str(batch_root),
        "--batch-id",
        spec.candidate_batch_id,
        "--experiment-mode",
        str(source_manifest["experiment_mode"]),
        "--retrieval-source",
        str(source_manifest["retrieval_source"]),
        "--neighbor-identity-policy",
        "parent_disjoint",
        "--final-only-source-batch",
        str(spec.source_batch),
        "--final-decision-profile",
        TRAIN_RATIO_TIEBREAK_V1,
        "--identity-blind",
        "--api-key-env",
        args.api_key_env,
        "--base-url",
        args.base_url,
        "--tool-service-url",
        str(source_run_manifest["tool_service_url"]),
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
        str(source_run_manifest["max_tool_rounds"]),
        "--parallelism",
        str(args.parallelism),
        "--no-stream-logs",
        "--no-combine-traces",
        "--skip-existing",
    ]
    command.extend(spec.prompt_profile_args)
    return command


def _validate_train_prior(spec: TaskSpec) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in spec.train_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = [int(row["Y"]) for row in rows]
    observed = Counter(labels)
    expected = {0: spec.prior.negative_count, 1: spec.prior.positive_count}
    if set(observed) != {0, 1} or dict(observed) != expected:
        raise ValueError(
            f"Frozen train prior mismatch for {spec.task}: "
            f"observed={dict(observed)} expected={expected}"
        )
    return {
        "train_jsonl": str(spec.train_jsonl),
        "sha256": sha256_file(spec.train_jsonl),
        "dataset_lineage": spec.prior.dataset_lineage,
        "split": spec.prior.split,
        "negative_count": spec.prior.negative_count,
        "positive_count": spec.prior.positive_count,
        "total": spec.prior.total,
        "positive_fraction": spec.prior.positive_fraction,
    }


def _candidate_batch_path(output_root: Path, spec: TaskSpec) -> Path:
    return (
        output_root
        / "runs_identity_blind_parent_disjoint"
        / spec.task
        / spec.candidate_batch_id
    )


def _representative_run_manifest(source_batch: Path) -> dict[str, Any]:
    run_dirs = sorted(path for path in (source_batch / "runs").iterdir() if path.is_dir())
    if not run_dirs:
        raise ValueError(f"No source runs: {source_batch}")
    return _read_json(run_dirs[0] / "manifest.json")


def _write_or_validate_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists() and _read_json(path) != manifest:
        raise ValueError(
            f"Existing experiment manifest differs; use a new output root: {path}"
        )
    write_json_atomic(path, manifest)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASK_SPECS), default=list(TASK_SPECS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--parallelism", type=int, default=128)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--max-stage-requeues", type=int, default=0)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.parallelism <= 512:
        parser.error("--parallelism must be in 1..512")
    if args.max_stage_requeues < 0:
        parser.error("--max-stage-requeues must be non-negative")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
