"""Run the frozen conditioned cumulative-assay-family curve on scaffold-valid."""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

from tools.chembl_tool.common.evidence_contract import ASSAY_RAW_CARD_PROMPT_PROFILE
from tools.chembl_tool.common.json_utils import sha256_file, write_json_atomic
from tools.chembl_tool.common.reasoning_payload import (
    EXTERNAL_CONDITION_RENDERER_VERSION,
)
from data.processing.gold_labels.conditioned_benchmark import split_path
from tools.chembl_tool.common.task_workflows.global_prompt_pool import (
    BatchCommand,
    SCHEDULER_VERSION,
    prepare_batch_commands,
    run_prepared_prompt_pool,
)
from tools.chembl_tool.common.task_workflows.reasoning_batch import (
    collect_completed_item,
)
from tools.chembl_tool.common.task_workflows.reasoning_stage_runtime import (
    load_stage_state,
    prepare_stage_item,
)
from tools.chembl_tool.paper_experiments.carry_forward_assay_prefix import (
    carry_forward_batch,
)


ROOT = Path("outputs/paper/starling_conditioned_assay_family_curve_v1")
HISTORICAL_OUTPUT_ROOT = ROOT / "scaffold_valid_epyc_deepseek_v4_flash_0731"
OUTPUT_ROOT = ROOT / "scaffold_valid_epyc_deepseek_v4_flash_0731_bio_nondirect_context_v1"
REPLAY_ROOT = ROOT / "replay_batches" / "valid"
BIOAVAILABILITY_REPLAY_ROOT = ROOT / "replay_batches" / "valid_nondirect_context_v1"
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
BASE_URL = "http://127.0.0.1:50001/v1"
REFERENCE_POOL = "direct_only_heldout_filtered"
IDENTITY_POLICY = "scaffold_disjoint"


@dataclass(frozen=True)
class TaskSpec:
    module: str
    input_jsonl: Path
    index: Path
    family_manifest: Path
    conditioned: bool
    replay_root: Path | None = None


TASKS = {
    "bbb_martins": TaskSpec(
        "tools.chembl_tool.tasks.bbb_martins.run_reasoning_batch",
        split_path("bbb_martins", "valid"),
        ROOT / "indices/bbb_martins/raw_v3/assay_neighbor_index.pkl",
        ROOT / "family_catalogs/bbb_martins/manifest.json",
        True,
    ),
    "bioavailability_ma": TaskSpec(
        "tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch",
        split_path("bioavailability_ma", "valid"),
        ROOT / "indices/bioavailability_ma/raw_v3/assay_neighbor_index.pkl",
        ROOT / "family_catalogs/bioavailability_ma/manifest.json",
        True,
    ),
    "skin_reaction": TaskSpec(
        "tools.chembl_tool.tasks.skin_reaction.run_reasoning_batch",
        split_path("skin_reaction", "valid"),
        ROOT / "indices/skin_reaction/raw_v3/assay_neighbor_index.pkl",
        ROOT / "family_catalogs/skin_reaction/manifest.json",
        True,
    ),
}

BIOAVAILABILITY_NONDIRECT_CONTEXT_SPEC = TaskSpec(
    "tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch",
    split_path("bioavailability_ma", "valid"),
    ROOT
    / "indices/bioavailability_ma/raw_v3_nondirect_context_v1/assay_neighbor_index.pkl",
    ROOT
    / "family_catalogs/bioavailability_ma_nondirect_context_v1/manifest.json",
    True,
    BIOAVAILABILITY_REPLAY_ROOT / "bioavailability_ma",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _n_rows(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def _levels(task: str) -> list[dict[str, Any]]:
    levels = list(_read_json(TASKS[task].family_manifest).get("levels") or [])
    if not levels or [row["level"] for row in levels] != list(range(1, len(levels) + 1)):
        raise ValueError(f"{task} family catalog has a non-contiguous level plan")
    return levels


def _validate_inputs(tasks: list[str]) -> None:
    for task in tasks:
        spec = TASKS[task]
        for path in (spec.input_jsonl, spec.index, spec.family_manifest):
            if not path.is_file():
                raise FileNotFoundError(path)
        family = _read_json(spec.family_manifest)
        expected_overlap_policy = (
            "assign each physical assay to its earliest cumulative family level"
        )
        if family.get("overlap_policy") != expected_overlap_policy:
            raise ValueError(f"{task} family overlap policy is not frozen")
        index_manifest = _read_json(spec.index.with_name("manifest.json"))
        expected = {
            "reference_pool": REFERENCE_POOL,
            "neighbor_identity_policy_default": IDENTITY_POLICY,
            "evidence_prompt_profile": ASSAY_RAW_CARD_PROMPT_PROFILE,
            "support_text_policy": "complete_representative_support",
            "n_direct_heldout_records_after_filter": 0,
        }
        for field, value in expected.items():
            if index_manifest.get(field) != value:
                raise ValueError(
                    f"{task} index has wrong {field}: {index_manifest.get(field)!r}"
                )
        heldout_hash = index_manifest.get("heldout_molecules_jsonl_sha256") or index_manifest.get(
            "heldout_molecules_sha256"
        )
        if not heldout_hash:
            raise ValueError(f"{task} index manifest lacks held-out lineage hash")
        final_assays = int(_levels(task)[-1]["cumulative_physical_assays"])
        if int(index_manifest.get("n_assays", -1)) != final_assays:
            raise ValueError(f"{task} family catalog and raw index disagree")


def _replay_root(task: str) -> Path:
    return TASKS[task].replay_root or REPLAY_ROOT / task


def _replay_is_complete(task: str) -> bool:
    spec = TASKS[task]
    expected_rows = _n_rows(spec.input_jsonl)
    input_hash = sha256_file(spec.input_jsonl)
    index_hash = sha256_file(spec.index)
    for level in _levels(task):
        prefix = int(level["cumulative_physical_assays"])
        manifest_path = _replay_root(task) / f"assay_flat_top{prefix}" / "manifest.json"
        if not manifest_path.is_file():
            return False
        manifest = _read_json(manifest_path)
        if (
            int(manifest.get("n_items", -1)) != expected_rows
            or manifest.get("input_jsonl_sha256") != input_hash
            or manifest.get("index_sha256") != index_hash
            or manifest.get("neighbor_identity_policy") != IDENTITY_POLICY
            or int(manifest.get("top_k_per_assay", -1)) != 3
            or float(manifest.get("min_similarity", -1)) != 0.3
        ):
            return False
        if spec.conditioned and (
            manifest.get("condition_renderer") != EXTERNAL_CONDITION_RENDERER_VERSION
            or manifest.get("condition_visibility") != "flat_evidence_and_final_only"
        ):
            return False
    return True


def _materialize_task(task: str, python_executable: str) -> None:
    if _replay_is_complete(task):
        print(f"[{task}] replay batches already complete", flush=True)
        return
    spec = TASKS[task]
    prefixes = [
        str(level["cumulative_physical_assays"]) for level in _levels(task)
    ]
    command = [
        python_executable,
        "-m",
        "tools.chembl_tool.common.assay_retrieval",
        "materialize-batch",
        "--index",
        str(spec.index),
        "--input-jsonl",
        str(spec.input_jsonl),
        "--prefixes",
        *prefixes,
        "--top-k-per-assay",
        "3",
        "--min-similarity",
        "0.3",
        "--neighbor-identity-policy",
        IDENTITY_POLICY,
        "--output-root",
        str(_replay_root(task)),
    ]
    if spec.conditioned:
        command.extend(["--condition-field", "condition_group"])
    subprocess.run(command, check=True)
    if not _replay_is_complete(task):
        raise RuntimeError(f"{task} replay materialization did not pass its manifest gate")


def materialize_replays(
    tasks: list[str],
    *,
    python_executable: str,
    workers: int,
) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        futures = {
            pool.submit(_materialize_task, task, python_executable): task for task in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            future.result()


def _batch_dir(output_root: Path, task: str, level: int, prefix: int = 0) -> Path:
    name = "none" if level == 0 else f"family_level_{level}_top{prefix}"
    return output_root / task / name


def _batch_command(
    args: argparse.Namespace,
    task: str,
    *,
    level: int,
    prefix: int = 0,
) -> BatchCommand:
    spec = TASKS[task]
    batch_dir = _batch_dir(Path(args.output_root), task, level, prefix)
    command = [
        args.python_executable,
        "-m",
        spec.module,
        "--input-jsonl",
        str(spec.input_jsonl),
        "--index",
        str(spec.index),
        "--experiment-mode",
        "none" if level == 0 else "full_flat",
        "--retrieval-source",
        "starling",
        "--neighbor-identity-policy",
        IDENTITY_POLICY,
        "--identity-blind",
        "--batch-root",
        str(batch_dir.parent),
        "--batch-id",
        batch_dir.name,
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--api-key-env",
        args.api_key_env,
        "--tool-service-url",
        args.tool_service_url,
        "--top-k-per-group",
        "3",
        "--min-similarity",
        "0.3",
        "--max-tokens",
        str(args.max_tokens),
        "--timeout-s",
        str(args.timeout_s),
        "--disable-thinking",
        "--reasoning-effort",
        "",
        "--max-stage-requeues",
        str(args.max_stage_requeues),
        "--skip-existing",
        "--no-stream-logs",
    ]
    if level > 0:
        replay = _replay_root(task) / f"assay_flat_top{prefix}"
        command.extend(["--retrieval-replay-source-batch", str(replay)])
        single_source = (
            Path(args.single_analysis_source_batch)
            if args.single_analysis_source_batch
            else _batch_dir(Path(args.output_root), task, 0)
        )
        command.extend(
            [
                "--single-analysis-source-batch",
                str(single_source),
            ]
        )
    if args.limit:
        command.extend(["--limit", str(args.limit)])
    if args.indices:
        command.append("--indices")
        command.extend(args.indices)
    name = f"{task}__none" if level == 0 else f"{task}__family_level_{level}_top{prefix}"
    return BatchCommand(name, command)


def _prepare_missing(prepared_by_name: dict[str, Any], workers: int) -> None:
    jobs = []
    for prepared in prepared_by_name.values():
        for item in prepared.items:
            if collect_completed_item(prepared, item) is not None:
                continue
            if load_stage_state(prepared, item) is None:
                jobs.append((prepared, item))
    if not jobs:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = [pool.submit(prepare_stage_item, prepared, item) for prepared, item in jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result.get("status") != "ok":
                raise RuntimeError(f"retrieval preparation failed: {result}")


def _round_commands(args: argparse.Namespace, level: int) -> list[BatchCommand]:
    commands = []
    for task in args.tasks:
        if level == 0:
            if args.single_analysis_source_batch:
                continue
            commands.append(_batch_command(args, task, level=0))
            continue
        levels = _levels(task)
        if level <= len(levels):
            prefix = int(levels[level - 1]["cumulative_physical_assays"])
            commands.append(_batch_command(args, task, level=level, prefix=prefix))
    return commands


def _carry_forward_round(
    args: argparse.Namespace,
    level: int,
) -> dict[str, Any]:
    receipts = {}
    if level <= 1:
        return receipts
    for task in args.tasks:
        levels = _levels(task)
        if level > len(levels):
            continue
        previous_prefix = int(levels[level - 2]["cumulative_physical_assays"])
        prefix = int(levels[level - 1]["cumulative_physical_assays"])
        source = _batch_dir(Path(args.output_root), task, level - 1, previous_prefix)
        target = _batch_dir(Path(args.output_root), task, level, prefix)
        receipts[task] = carry_forward_batch(source, target)
    return receipts


def _experiment_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment": "starling_conditioned_assay_family_curve.v1",
        "scheduler": SCHEDULER_VERSION,
        "evaluation_subset": "valid",
        "tasks": args.tasks,
        "model": args.model,
        "base_url": args.base_url,
        "parallelism": args.parallelism,
        "max_tokens": args.max_tokens,
        "timeout_s": args.timeout_s,
        "visibility_mode": "identity_blind",
        "reference_pool": REFERENCE_POOL,
        "bioavailability_source_revision": args.bioavailability_source_revision,
        "neighbor_identity_policy": IDENTITY_POLICY,
        "top_k_per_assay": 3,
        "min_similarity": 0.3,
        "condition_visibility": "flat_evidence_and_final_only",
        "condition_renderer": EXTERNAL_CONDITION_RENDERER_VERSION,
        "null_condition_policy": "omit_condition_from_prompt",
        "single_reuse": (
            "one external aligned none batch reused by every family level"
            if args.single_analysis_source_batch
            else "one local aligned none batch reused by every family level"
        ),
        "single_analysis_source_batch": args.single_analysis_source_batch,
        "level_reuse": "adjacent evidence-equivalent carry-forward before each round",
        "assay_overlap_policy": "earliest cumulative level; physical assay remains atomic",
        "levels_by_task": {task: _levels(task) for task in args.tasks},
        "inputs": {
            task: {
                "valid_jsonl": str(TASKS[task].input_jsonl),
                "valid_jsonl_sha256": sha256_file(TASKS[task].input_jsonl),
                "index": str(TASKS[task].index),
                "index_sha256": sha256_file(TASKS[task].index),
                "index_manifest_sha256": sha256_file(
                    TASKS[task].index.with_name("manifest.json")
                ),
                "family_manifest": str(TASKS[task].family_manifest),
                "family_manifest_sha256": sha256_file(TASKS[task].family_manifest),
                "n_valid_rows": _n_rows(TASKS[task].input_jsonl),
            }
            for task in args.tasks
        },
        "rounds": [],
    }


def run(args: argparse.Namespace) -> int:
    _validate_inputs(args.tasks)
    materialize_replays(
        args.tasks,
        python_executable=args.python_executable,
        workers=args.materialize_workers,
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "experiment_manifest.json"
    manifest = _experiment_manifest(args)
    if manifest_path.is_file():
        previous = _read_json(manifest_path)
        for field in (
            "experiment",
            "model",
            "base_url",
            "visibility_mode",
            "reference_pool",
            "bioavailability_source_revision",
            "neighbor_identity_policy",
            "condition_renderer",
        ):
            if (
                field == "bioavailability_source_revision"
                and previous.get(field) is None
                and manifest.get(field) == "historical_missing_nondirect"
            ):
                continue
            if previous.get(field) != manifest.get(field):
                raise ValueError(f"cannot resume with changed {field}")
        if previous.get("inputs") != manifest.get("inputs"):
            raise ValueError(
                "cannot resume with changed inputs; choose a new --output-root"
            )
        manifest["rounds"] = list(previous.get("rounds") or [])
    write_json_atomic(manifest_path, manifest)

    max_level = max(len(_levels(task)) for task in args.tasks)
    for level in range(0, max_level + 1):
        commands = _round_commands(args, level)
        if not commands:
            continue
        prepared = prepare_batch_commands(
            commands,
            max_workers=args.parallelism,
            max_stage_requeues=args.max_stage_requeues,
        )
        _prepare_missing(prepared, args.preparation_workers)
        reuse = _carry_forward_round(args, level)
        round_record = {
            "level": level,
            "conditions": list(prepared),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "carry_forward": reuse,
        }
        manifest["rounds"] = [
            row for row in manifest["rounds"] if int(row.get("level", -1)) != level
        ] + [round_record]
        write_json_atomic(manifest_path, manifest)
        if args.prepare_only:
            continue
        failed = run_prepared_prompt_pool(
            prepared,
            max_workers=args.parallelism,
            max_stage_requeues=args.max_stage_requeues,
        )
        round_record["finished_at"] = datetime.now(timezone.utc).isoformat()
        round_record["failed"] = failed
        write_json_atomic(manifest_path, manifest)
        if failed:
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--tool-service-url", default="http://127.0.0.1:8765")
    parser.add_argument("--parallelism", type=int, default=128)
    parser.add_argument("--max-stage-requeues", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=20_480)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--materialize-workers", type=int, default=4)
    parser.add_argument("--preparation-workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--indices", nargs="*", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--bioavailability-source-revision",
        choices=("historical_missing_nondirect", "nondirect_context_v1"),
        default=None,
        help=(
            "Select the versioned Bioavailability assay source contract. Fresh roots "
            "default to nondirect_context_v1; historical roots remain resumable."
        ),
    )
    parser.add_argument(
        "--single-analysis-source-batch",
        default="",
        help=(
            "Reuse a complete compatible none batch for the single-molecule branch "
            "and omit the local level-0 run. Intended for a one-task correction run."
        ),
    )
    parser.add_argument(
        "--python-executable",
        default="/data1/tianang/anaconda3/envs/vllm/bin/python",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.parallelism <= 128:
        parser.error("--parallelism must be between 1 and 128")
    if args.materialize_workers < 1:
        parser.error("--materialize-workers must be positive")
    if not 1 <= args.preparation_workers <= 64:
        parser.error("--preparation-workers must be between 1 and 64")
    if args.single_analysis_source_batch:
        if len(args.tasks) != 1:
            parser.error("--single-analysis-source-batch requires exactly one task")
        if not Path(args.single_analysis_source_batch).is_dir():
            parser.error("--single-analysis-source-batch must be an existing directory")
    if args.bioavailability_source_revision is None:
        manifest_path = Path(args.output_root) / "experiment_manifest.json"
        historical_resume = False
        if manifest_path.is_file():
            previous = _read_json(manifest_path)
            bio_index = str(
                ((previous.get("inputs") or {}).get("bioavailability_ma") or {}).get(
                    "index"
                )
                or ""
            )
            historical_resume = bio_index.endswith(
                "indices/bioavailability_ma/raw_v3/assay_neighbor_index.pkl"
            )
        args.bioavailability_source_revision = (
            "historical_missing_nondirect"
            if historical_resume
            else "nondirect_context_v1"
        )
    if args.bioavailability_source_revision == "nondirect_context_v1":
        TASKS["bioavailability_ma"] = BIOAVAILABILITY_NONDIRECT_CONTEXT_SPEC
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
