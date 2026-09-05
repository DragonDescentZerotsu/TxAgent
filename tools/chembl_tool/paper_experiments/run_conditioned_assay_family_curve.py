"""Run cumulative full-flat levels, optionally matched to frozen progressive inputs."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import fcntl
import os
import importlib
from pathlib import Path
import shutil
import subprocess
from typing import Any

from tools.chembl_tool.common.evidence_contract import ASSAY_RAW_CARD_PROMPT_PROFILE
from tools.chembl_tool.common.json_utils import atomic_output_path, sha256_file, write_json_atomic
from tools.chembl_tool.common.reasoning_payload import (
    EXTERNAL_CONDITION_RENDERER_VERSION,
    load_env_file,
)
from tools.chembl_tool.common.starling.conditioned_benchmark import split_path
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
    "clintox": TaskSpec(
        "tools.chembl_tool.tasks.clintox.run_reasoning_batch",
        split_path("clintox", "valid"),
        ROOT / "indices/clintox/raw_v3/assay_neighbor_index.pkl",
        ROOT / "family_catalogs/clintox/manifest.json",
        False,
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
            "source-native assay ids are disjoint; assign each to one family level"
            if task == "clintox"
            else "assign each physical assay to its earliest cumulative family level"
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


def _copy_matched_prepared(source: Path, target: Path, expected_hash: str) -> None:
    """Publish frozen inputs atomically; never replace altered resume inputs."""
    if target.exists():
        if sha256_file(target) != expected_hash:
            raise ValueError(f"changed matched prepared input; use a fresh output root: {target}")
        return
    with atomic_output_path(target) as temporary:
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != expected_hash:
            raise ValueError(f"prepared input changed while copying: {source}")


def _validate_matched_task(runtime, source, task):
    spec = runtime._progressive_task_specs(source["split_scheme"], source["evaluation_subset"])[task]
    inputs = source["inputs"][task]
    for field, path in (("input_sha256", spec.input_jsonl), ("index_sha256", spec.index),
                        ("family_manifest_sha256", spec.family_manifest)):
        if sha256_file(path) != inputs[field]:
            raise ValueError(f"stale matched source: {task} {field}")
    if runtime._levels(task) != inputs["levels"]:
        raise ValueError(f"changed level definitions: {task}")
    runtime._heldout_filter_validation(
        task=task, split_scheme=source["split_scheme"],
        index_manifest=_read_json(spec.index.with_name("manifest.json")),
        heldout_path=runtime.task_root(task, source["split_scheme"]) / "heldout_molecule_condition_labels.jsonl",
    )
    return spec


def _matched_roots(args):
    source, output = Path(args.matched_progressive_root).resolve(), Path(args.output_root).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("matched output root must be outside its source run")
    return source, output


def _validate_matched_source(runtime, args, source, *, refresh_priors=False):
    expected = {
        "experiment": runtime.PROGRESSIVE_PROTOCOL_VERSION,
        "visibility_mode": "deployment_visible_prefetched",
        "prompt_profile": "progressive_compact_tools_short_aliases.v2",
        "temperature": 0.0, "thinking": "provider_default", "reasoning_effort": "omitted",
        "max_tokens": args.max_tokens, "tool_prefetch_complete": True,
    }
    if not refresh_priors:
        expected["model"] = args.model
    for field, value in expected.items():
        if source.get(field) != value:
            raise ValueError(f"unsupported matched source {field}")
    if args.limit or args.indices:
        raise ValueError("matched runs use the source's exact evaluation indices")


def _replace_matched_prior(prepared, prior):
    query_prior, tool, none, prior_index = prior
    if tool != prepared["query_tool_summary"]:
        raise ValueError("refreshed prior changed frozen query tools")
    return {**prepared, "query_prior": query_prior, "reused_none_final": none,
            "reused_single_source_index": prior_index}


def _refresh_matched_priors(args, source_root: Path, output_root: Path) -> int:
    """Run fresh single/None branches over frozen retrieval and tool outputs."""
    from tools.chembl_tool.common.task_workflows import reasoning_batch as batch
    from tools.chembl_tool.common.task_workflows import reasoning_stage_runtime as stages
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_progressive_curve as runtime

    source = _read_json(source_root / "experiment_manifest.json")
    _validate_matched_source(runtime, args, source, refresh_priors=True)
    receipt = output_root / "completion.json"
    if receipt.exists():
        for relative, digest in _read_json(receipt)["sha256"].items():
            if sha256_file(output_root / relative) != digest:
                raise ValueError(f"changed refreshed prior artifact: {relative}")
        return 0
    batches, queries = {}, []
    provider = runtime._resolve_provider_pool_config(args)
    if len(provider.providers) != 1:
        raise ValueError("query-prior refresh requires one compatible routing endpoint")
    endpoint = provider.providers[0]
    if endpoint.max_inflight > args.endpoint_concurrency_budget:
        raise ValueError("provider capacity exceeds endpoint budget")
    for task in args.tasks or source["tasks"]:
        _validate_matched_task(runtime, source, task)
        source_batch = source_root / "single_cache" / task / "none"
        original = _read_json(source_batch / "manifest.json")
        if sha256_file(source_batch / "manifest.json") != source["inputs"][task]["single_source_manifest_sha256"]:
            raise ValueError(f"changed source single manifest: {task}")
        config = importlib.import_module(f"tools.chembl_tool.tasks.{task}.run_reasoning_batch").CONFIG
        command = ["--input-jsonl", source["inputs"][task]["input_jsonl"],
                   "--index", source["inputs"][task]["index"], "--experiment-mode", "none",
                   "--retrieval-source", "starling", "--neighbor-identity-policy", source["neighbor_identity_policy"],
                   "--harness-prefetch-tools", "--retrieval-replay-source-batch", str(source_batch),
                   "--prefetched-tool-replay-source-batch", str(source_batch),
                   "--batch-root", str(output_root / task), "--batch-id", "none",
                   "--model", args.model, "--base-url", endpoint.base_url, "--api-key-env", endpoint.api_key_env,
                   "--env-file", str(args.env_file), "--tool-service-url", args.tool_service_url,
                   "--max-tokens", str(args.max_tokens), "--timeout-s", str(args.timeout_s),
                   "--parallelism", str(args.parallelism_per_task or args.parallelism),
                   "--disable-thinking", "--reasoning-effort", "", "--skip-existing", "--no-stream-logs",
                   "--request-extra-body-json", json.dumps(dict(endpoint.request_extra_body or {}))]
        if config.prompt_profile_option:
            command += [config.prompt_profile_option, original["task_prompt_profile"]]
        prepared = batch.prepare_batch(config, batch._parse_args(config, command))
        if prepared.manifest["indices"] != source["evaluation_indices_by_task"][task]:
            raise ValueError(f"query-prior refresh requires the source's complete evaluation cohort: {task}")
        prepared.manifest["scheduler"] = {
            "version": "shared_query_pool_sequential_single_final.v1",
            "global_max_workers": args.parallelism,
            "parallelism_per_task": args.parallelism_per_task,
            "max_stage_requeues": args.max_stage_requeues,
            "validation_retry_policy": "legacy single/None validation, no speculative race",
        }
        write_json_atomic(prepared.batch_dir / "manifest.json", prepared.manifest)
        batches[task] = prepared
        for item in prepared.items:
            run_id = f"none_idx{item.index:05d}"
            target = prepared.batch_run_root / run_id
            frozen = source_batch / "runs" / run_id / "retrieval.json"
            _copy_matched_prepared(frozen, target / "retrieval.json", sha256_file(frozen))
            if not (target / "manifest.json").exists():
                stages._initialize_run_manifest(prepared, item, run_id, target)
            queries.append(runtime.PreparedQuery(task, item.index, target))

    def run_prior(_args, query, _client):
        prepared = batches[query.task]
        state = stages.load_stage_state(prepared, prepared.items[query.index])
        if state is None or state.expected_group_ids:
            raise ValueError(f"invalid frozen None state: {query.query_dir}")
        while jobs := stages.ready_stage_jobs(state):
            for job in jobs:
                if stages.execute_stage(job).get("status") != "ok":
                    return {"task": query.task, "index": query.index, "status": "error"}
        result = stages.collect_stage_result(state)
        return {"task": query.task, "index": query.index, "status": result["status"]}

    prior_args = copy.copy(args)
    prior_args.tasks = list(batches)
    prior_args.retry_race_width = 1
    failed = runtime._run_query_rounds(prior_args, queries, None, output_root, run_query=run_prior)
    for prepared in batches.values():
        results = [stages.collect_stage_result(stages.load_stage_state(prepared, item)) for item in prepared.items]
        batch.finalize_batch(prepared, results)
    if not failed:
        artifacts = [path for task in batches for path in (output_root / task / "none").rglob("*.json")]
        write_json_atomic(receipt, {"model": args.model, "sha256": {
            str(path.relative_to(output_root)): sha256_file(path) for path in artifacts}})
    return failed


def _run_matched_curve(args: argparse.Namespace) -> int:
    """Replay frozen cards/tools with fresh full-flat or progressive level outputs."""
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_progressive_curve as runtime

    source_root, output_root = _matched_roots(args)
    source_path = source_root / "experiment_manifest.json"
    source = _read_json(source_path)
    independent = args.matched_organizations[0] == "full_flat"
    _validate_matched_source(runtime, args, source, refresh_priors=bool(getattr(args, "matched_single_root", "")))
    args.tasks = args.tasks or source["tasks"]
    print(f"[matched {args.matched_organizations[0]}] validating frozen inputs for {', '.join(args.tasks)}", flush=True)
    records, queries, hashes = {}, [], {}
    prior_root = Path(args.matched_single_root) if getattr(args, "matched_single_root", "") else None
    priors = {}
    for task in args.tasks:
        spec = _validate_matched_task(runtime, source, task)
        records[task] = runtime.read_jsonl(spec.input_jsonl)
        if prior_root is not None:
            prior_manifest = _read_json(prior_root / task / "none/manifest.json")
            if prior_manifest["model"] != args.model or prior_manifest["indices"] != source["evaluation_indices_by_task"][task]:
                raise ValueError(f"incompatible refreshed query priors: {task}")
        for index in source["evaluation_indices_by_task"][task]:
            query_dir = runtime._query_dir(output_root, task, index)
            queries.append(runtime.PreparedQuery(task, index, query_dir))
            if prior_root is not None:
                prior = runtime._load_reused_query_prior(task, records[task][index], prior_root)
            for level in runtime._levels(task):
                relative = runtime._query_dir(Path(), task, index) / "levels" / f"level_{level['level']}" / "prepared.json"
                prepared = _read_json(source_root / relative)
                if not prepared.get("tool_prefetch_complete"):
                    raise ValueError(f"incomplete source tools: {relative}")
                if prior_root is not None:
                    _replace_matched_prior(prepared, prior)
                    priors[str(relative)] = prior
                hashes[str(relative)] = sha256_file(source_root / relative)
    manifest = {field: source.get(field) for field in runtime._RESUME_INVARIANT_FIELDS}
    manifest.update(
        experiment="conditioned_assay_matched_full_flat.v1" if independent else runtime.PROGRESSIVE_PROTOCOL_VERSION,
        prompt_profile="independent_cumulative_tools_short_aliases.v1" if independent else source["prompt_profile"],
        tasks=args.tasks, model=args.model, base_url=args.base_url,
        parallelism=args.parallelism,
        matched_progressive_root=str(source_root),
        matched_source_manifest_sha256=sha256_file(source_path),
        prepared_sha256=hashes,
        reasoning_organization="independent full-flat at every cumulative level" if independent else "append-only progressive",
        prior_state_used=not independent,
        level_output_reuse=False,
    )
    if prior_root is not None:
        manifest.update(model_identity=runtime._model_identity(args.model),
                        query_prior_source_root=str(prior_root),
                        refreshed_single_root=str(prior_root),
                        refreshed_single_manifest_sha256={task: sha256_file(prior_root / task / "none/manifest.json") for task in args.tasks})
        manifest["inputs"] = copy.deepcopy(manifest["inputs"])
        for task in args.tasks:
            manifest["inputs"][task]["single_source_manifest_sha256"] = manifest["refreshed_single_manifest_sha256"][task]
        manifest["source_prepared_sha256"] = manifest.pop("prepared_sha256")
    provider_config = runtime._resolve_provider_pool_config(args)
    if sum(spec.max_inflight for spec in provider_config.providers) > args.endpoint_concurrency_budget:
        raise ValueError("provider capacity exceeds endpoint budget")
    if args.provider_pool_config:
        manifest["provider_pool"] = provider_config.public_dict()
    manifest_path = output_root / "experiment_manifest.json"
    if manifest_path.exists():
        previous = _read_json(manifest_path)
        for field, value in manifest.items():
            if field not in {"base_url", "parallelism"} and previous.get(field) != value:
                raise ValueError(f"cannot resume matched full-flat with changed {field}")
    manifest.update(parallelism_per_task=args.parallelism_per_task,
                    endpoint_concurrency_budget=args.endpoint_concurrency_budget,
                    retry_race_width=args.retry_race_width)
    for field in ("min_similarity", "condition_policy"):
        manifest[field] = source.get(field)
    manifest["model_identity"] = runtime._model_identity(args.model)
    manifest["provider_pool"] = provider_config.public_dict()
    write_json_atomic(manifest_path, {**manifest, "status": "preparing"})
    for relative, expected_hash in hashes.items():
        if relative in priors:
            target = output_root / relative
            if sha256_file(source_root / relative) != expected_hash:
                raise ValueError(f"changed source prepared input: {relative}")
            refreshed = _replace_matched_prior(_read_json(source_root / relative), priors[relative])
            if target.exists() and _read_json(target) != refreshed:
                raise ValueError(f"changed refreshed prepared input: {target}")
            write_json_atomic(target, refreshed)
        else:
            _copy_matched_prepared(source_root / relative, output_root / relative, expected_hash)
    if prior_root is not None:
        manifest["prepared_sha256"] = {relative: sha256_file(output_root / relative) for relative in hashes}
        write_json_atomic(manifest_path, {**manifest, "status": "running"})
    args.independent_levels = independent
    args.transport_max_retries = 0
    client = None if args.prepare_only else runtime._make_client(args, provider_config)
    try:
        failed = runtime._run_query_rounds(args, queries, client, output_root)
    finally:
        if isinstance(client, runtime.ParallelRetryClient):
            client.close()
    if not args.prepare_only:
        for task in args.tasks:
            runtime._summarize_task(task=task, records=records[task],
                                    indices=source["evaluation_indices_by_task"][task], output_root=output_root)
    manifest.update(status="failed" if failed else ("prepared" if args.prepare_only else "complete"),
                    n_failed_queries=failed, n_prepared_levels=len(hashes))
    write_json_atomic(manifest_path, manifest)
    print(json.dumps({key: manifest[key] for key in ("status", "n_failed_queries", "n_prepared_levels")}))
    return int(bool(failed))


def _run_matched_suite(args: argparse.Namespace) -> int:
    """Run replicates sequentially; each run owns the single shared endpoint budget."""
    source, root = _matched_roots(args)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = [
            {"replicate": replicate, "organization": organization,
             "root": str(root / f"replicate_{replicate:02d}" / organization), "status": "queued"}
            for replicate in args.replicate_ids
            for organization in args.matched_organizations
        ]
        config = {"source": str(source), "source_manifest_sha256": sha256_file(source / "experiment_manifest.json"),
                  "replicate_ids": args.replicate_ids, "organizations": args.matched_organizations,
                  "model": args.model, "max_tokens": args.max_tokens, "tasks": args.tasks,
                  "parallelism": args.parallelism, "parallelism_per_task": args.parallelism_per_task,
                  "endpoint_concurrency_budget": args.endpoint_concurrency_budget,
                  "retry_race_width": args.retry_race_width, "max_stage_requeues": args.max_stage_requeues,
                  "retry_delay_s": args.retry_delay_s, "base_url": args.base_url, "timeout_s": args.timeout_s}
        if args.refresh_query_priors:
            config["refresh_query_priors"] = True
        if args.provider_pool_config:
            from tools.chembl_tool.common.openai_provider_pool import load_provider_pool_config
            config["provider_pool"] = load_provider_pool_config(args.provider_pool_config).public_dict()
        manifest_path = root / "suite_manifest.json"
        if manifest_path.exists() and _read_json(manifest_path) != config:
            raise ValueError("Cannot resume suite with a changed configuration")
        write_json_atomic(manifest_path, config)
        status = {"phase": "running", "pid": os.getpid(), "jobs": jobs,
                  "mode": "prepare_only" if args.prepare_only else "inference"}

        def publish():
            status["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_json_atomic(root / "suite_status.json", status)

        publish()
        if args.refresh_query_priors:
            status["phase"] = "query_priors"
            status["active_root"] = str(root / "single_cache")
            publish()
            try:
                failed = _refresh_matched_priors(args, source, root / "single_cache")
            except Exception as exc:
                failed = True
                status.update(error_type=type(exc).__name__, error=str(exc))
            if failed:
                status["phase"] = "needs_attention"
                publish()
                return 1
            args.matched_single_root = str(root / "single_cache")
            status["phase"] = "running"
        for job in jobs:
            job["status"] = "running"
            status["active_root"] = job["root"]
            publish()
            child = copy.copy(args)
            child.output_root = job["root"]
            child.matched_organizations = [job["organization"]]
            try:
                code = _run_matched_curve(child)
                job["status"] = "needs_attention" if code else ("prepared" if args.prepare_only else "complete")
                job["exit_code"] = code
            except Exception as exc:
                job.update(status="needs_attention", error_type=type(exc).__name__, error=str(exc))
            publish()
        failed = any(job["status"] == "needs_attention" for job in jobs)
        status.update(phase="needs_attention" if failed else ("prepared" if args.prepare_only else "complete"), active_root=None)
        publish()
        return int(failed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=None)
    parser.add_argument("--matched-progressive-root", default="",
                        help="Run independent full-flat on this frozen run's exact cumulative cards/tools.")
    parser.add_argument("--matched-organizations", nargs="+", choices=("full_flat", "progressive"), default=["full_flat"])
    parser.add_argument("--replicate-ids", nargs="+", type=int,
                        help="Sequential fresh replicate directories under output-root; existing checkpoints resume.")
    parser.add_argument("--retry-race-width", type=int, default=1)
    parser.add_argument("--refresh-query-priors", action="store_true",
                        help="Before a matched suite, refresh single/None with the selected model over frozen source tools.")
    parser.add_argument("--provider-pool-config", default="")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--tool-service-url", default="http://127.0.0.1:8765")
    parser.add_argument("--parallelism", type=int, default=128)
    parser.add_argument("--parallelism-per-task", type=int, default=0,
                        help="Matched full-flat per-task cap within the shared pool; 0 uses the global cap.")
    parser.add_argument("--endpoint-concurrency-budget", type=int, default=512,
                        help="Matched full-flat endpoint cap; increase only for an authorized run.")
    parser.add_argument("--max-stage-requeues", type=int, default=3)
    parser.add_argument("--retry-delay-s", type=int, default=60,
                        help="Matched full-flat retry cooldown; doubles up to 900 seconds.")
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
    load_env_file(Path(args.env_file))
    if args.refresh_query_priors and not (args.matched_progressive_root and args.replicate_ids):
        parser.error("--refresh-query-priors requires a matched suite")
    if args.refresh_query_priors and args.prepare_only:
        parser.error("--refresh-query-priors requires inference")
    budget = args.endpoint_concurrency_budget if args.matched_progressive_root else 128
    if not 1 <= args.parallelism <= budget:
        parser.error(f"--parallelism must be between 1 and the endpoint budget {budget}")
    if not 0 <= args.parallelism_per_task <= args.parallelism:
        parser.error("--parallelism-per-task must be between 0 and --parallelism")
    if args.parallelism_per_task and not args.matched_progressive_root:
        parser.error("--parallelism-per-task requires --matched-progressive-root")
    if args.max_stage_requeues < 0 or args.retry_delay_s < 0:
        parser.error("retry count and delay must be non-negative")
    if not 1 <= args.retry_race_width <= (args.parallelism_per_task or args.parallelism):
        parser.error("retry race width must fit the task request budget")
    if len(args.matched_organizations) != len(set(args.matched_organizations)):
        parser.error("matched organizations must be unique")
    if args.replicate_ids and (min(args.replicate_ids) < 1 or len(set(args.replicate_ids)) != len(args.replicate_ids)):
        parser.error("replicate ids must be unique positive integers")
    if not args.matched_progressive_root and (args.replicate_ids or args.matched_organizations != ["full_flat"] or args.retry_race_width > 1):
        parser.error("replicates, organizations and racing require --matched-progressive-root")
    if len(args.matched_organizations) > 1 and not args.replicate_ids:
        parser.error("multiple organizations require --replicate-ids")
    if args.materialize_workers < 1:
        parser.error("--materialize-workers must be positive")
    if not 1 <= args.preparation_workers <= 64:
        parser.error("--preparation-workers must be between 1 and 64")
    if args.matched_progressive_root:
        if Path(args.output_root) == OUTPUT_ROOT:
            parser.error("--matched-progressive-root requires an explicit --output-root")
        return _run_matched_suite(args) if args.replicate_ids else _run_matched_curve(args)
    args.tasks = args.tasks or list(TASKS)
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
