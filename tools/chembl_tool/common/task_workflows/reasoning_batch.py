"""Shared batch runner for molecule-level reasoning tasks."""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import gc
import hashlib
from contextlib import contextmanager
import fcntl
import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tools.chembl_tool.common.assay_transfer_selection import (
    ASSAY_TRANSFER_DIVERSITY_MODES,
    ASSAY_TRANSFER_DIVERSITY_NONE,
    ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
    ASSAY_TRANSFER_RECORDS_PER_MOLECULE_MAX,
    ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    ASSAY_TRANSFER_SELECTION_UNITS,
    validate_assay_transfer_diversity,
    validate_assay_transfer_records_per_molecule,
)
from tools.chembl_tool.common.cli.retrieval_args import add_retrieval_strategy_args
from tools.chembl_tool.common.final_evidence_surface import (
    SUMMARY_ONLY,
    add_final_evidence_surface_argument,
)
from tools.chembl_tool.common.final_decision_prior import (
    GENERAL_FINAL_DECISION_PROFILES,
    STANDARD_FINAL_DECISION,
    add_final_decision_profile_argument,
)
from tools.chembl_tool.common.coverage_reasoning import (
    COVERAGE_MMP_LEDGER_NEIGHBOR_CONTEXT,
    NEIGHBOR_CONTEXT_PROFILES,
    STANDARD_NEIGHBOR_CONTEXT,
)
from tools.chembl_tool.common.neighbor_selection import (
    NEIGHBOR_SELECTORS,
    SIMILARITY_SELECTOR,
)
from tools.chembl_tool.common.experiment_retrieval import ASSAY_TRANSFER_TOOL_STRATEGY
from tools.chembl_tool.common.retrieval_policy import NEIGHBOR_IDENTITY_POLICIES
from tools.chembl_tool.common.prompt_profile import (
    prompt_profile_from_manifest,
    require_matching_prompt_profiles,
)
from tools.chembl_tool.common.task_workflows.analogous_flat_prompt import (
    PROMPT_IDENTITY_VIEW,
    prompt_provenance as analogous_flat_prompt_provenance,
)


@dataclass(frozen=True)
class BatchConfig:
    description: str
    default_input: str
    default_batch_root: str
    default_index: str
    default_model: str
    batch_id_prefix: str
    pipeline_module: str
    log_prefix: str
    report_title: str
    prediction_field: str
    canonical_positive: str
    canonical_negative: str
    positive_predictions: frozenset[str]
    negative_predictions: frozenset[str]
    rerank_preflight: Callable[..., dict[str, Any]] | None = None
    supports_assay_transfer_scores: bool = False
    # True for tasks wired to the unified --retrieval-strategy CLI.
    # Every task receives the strict --morgan-neighbor-selector flag.
    supports_retrieval_strategy: bool = False
    group_prompt_formats: tuple[str, ...] = ()
    default_group_prompt_format: str = ""
    group_output_schemas: tuple[str, ...] = ()
    default_group_output_schema: str = ""
    group_prompt_versions: tuple[str, ...] = ()
    default_group_prompt_version: str = ""
    group_prompt_provenance: Callable[..., dict[str, Any]] | None = None
    supports_shared_retrieval_contract: bool = True
    supports_nondirect_bioavailability_filter: bool = False
    supports_analogous_reasoning_only: bool = False
    analogous_reasoning_modes: tuple[str, ...] = ("full_mechanism",)
    final_prompt_provenance: Callable[..., dict[str, Any]] | None = None
    assay_transfer_profile_default: str = "legacy_bio"
    rerank_catalog_default: str = (
        "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
        "assay_transfer_rerank/flat_v2/catalog.jsonl"
    )
    rerank_cache_default: str = (
        "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
        "assay_transfer_rerank/flat_v2/scores.sqlite3"
    )
    rerank_candidate_manifest_default: str = ""
    rerank_version_manifest_default: str = ""
    assay_transfer_model_default: str = "jiosephlee/assay-transfer-tool"
    assay_transfer_model_revision_default: str = (
        "9515603b1a5c4586e41c221dcdbc5e7487c0c3f5"
    )
    assay_transfer_template_profile_default: str = "legacy_v3"
    v11_rerank_catalog_default: str = ""
    v11_rerank_cache_default: str = ""
    v11_rerank_candidate_manifest_default: str = ""
    v11_rerank_version_manifest_default: str = ""
    v11_assay_transfer_model_default: str = ""
    v11_assay_transfer_model_revision_default: str = ""
    v11_index_default: str = ""
    supports_final_decision_profiles: bool = False
    final_decision_profile_choices: tuple[str, ...] = GENERAL_FINAL_DECISION_PROFILES
    prompt_profile_option: str = ""
    prompt_profile_choices: tuple[str, ...] = ()
    default_prompt_profile: str = ""
    historical_prompt_profile: str = ""
    default_api_key_env: str = "DEEPSEEK_API_KEY"
    default_base_url: str = "https://api.deepseek.com"


@dataclass(frozen=True)
class BatchItem:
    index: int
    record: dict[str, Any]


@dataclass
class PreparedBatch:
    """One batch prepared for either local or matrix-wide scheduling."""

    config: BatchConfig
    args: argparse.Namespace
    batch_id: str
    batch_dir: Path
    logs_dir: Path
    batch_run_root: Path
    items: list[BatchItem]
    manifest: dict[str, Any]


def main(config: BatchConfig, argv: list[str] | None = None) -> int:
    args = _parse_args(config, argv)
    prepared = prepare_batch(config, args)
    # Local import avoids a module cycle while making the task CLI and matrix
    # launcher use exactly the same prompt scheduler and checkpoint semantics.
    from tools.chembl_tool.common.task_workflows.global_prompt_pool import (
        run_prepared_prompt_pool,
    )

    failed = run_prepared_prompt_pool(
        {prepared.batch_id: prepared},
        max_workers=args.parallelism,
        max_stage_requeues=args.max_stage_requeues,
    )
    return 1 if failed else 0


def prepare_batch(config: BatchConfig, args: argparse.Namespace) -> PreparedBatch:
    """Materialize stable batch metadata without choosing a scheduling policy."""
    _validate_analogous_reasoning_only(config, args)
    requested_top_k_per_group = args.top_k_per_group
    is_assay_transfer = args.retrieval_strategy == ASSAY_TRANSFER_TOOL_STRATEGY
    _validate_assay_transfer_scores(config, args)
    group_prompt_instruction_provenance = _group_prompt_instruction_provenance(
        args.group_prompt_instructions_file
    )
    if group_prompt_instruction_provenance:
        args.group_prompt_instructions_file = str(
            group_prompt_instruction_provenance["path"]
        )
        args.group_prompt_instructions_sha256 = str(
            group_prompt_instruction_provenance["sha256"]
        )
    else:
        args.group_prompt_instructions_sha256 = ""
    group_prompt_provenance = (
        config.group_prompt_provenance(
            args.group_prompt_format,
            prompt_version=args.group_prompt_version,
            instructions_file=args.group_prompt_instructions_file or None,
            output_schema_profile=args.group_output_schema,
        )
        if config.group_prompt_provenance is not None
        and args.group_prompt_format in {"morganfingerprint", "assay_transfer_tool"}
        and not _is_analogous_flat(args)
        else {}
    )
    records = _read_jsonl(Path(args.input_jsonl))
    indices = _select_indices(args, len(records))
    batch_id = args.batch_id or time.strftime(f"{config.batch_id_prefix}_%Y%m%d_%H%M%S")
    batch_dir = _ensure_dir(Path(args.batch_root) / batch_id)
    _validate_group_prompt_contract(batch_dir, group_prompt_provenance)
    _validate_final_decision_profile_contract(args, batch_dir)
    _validate_prompt_profile_contract(config, args, batch_dir)
    logs_dir = _ensure_dir(batch_dir / "logs")
    batch_run_root = _ensure_dir(batch_dir / "runs")

    items = [BatchItem(index=i, record=records[i]) for i in indices]
    rerank_preflight = {"status": "not_requested"}
    if args.retrieval_strategy == ASSAY_TRANSFER_TOOL_STRATEGY:
        if config.rerank_preflight is None:
            raise SystemExit(f"Pipeline {config.pipeline_module} does not support assay-transfer reranking")
        if args.retrieval_source not in {"starling", "starling_in_distribution"}:
            raise SystemExit(
                "assay_transfer reranking is enabled only for Starling retrieval"
            )
        if args.rerank_cache_mode != "read_only":
            raise SystemExit("reasoning batches require --rerank-cache-mode read_only")
        if args.reuse_existing_rerank_preflight and args.rerank_preflight_source_batch:
            raise SystemExit(
                "--reuse-existing-rerank-preflight and "
                "--rerank-preflight-source-batch are mutually exclusive"
            )
        if args.rerank_preflight_source_batch:
            source_batch = Path(args.rerank_preflight_source_batch)
            source_manifest = _read_json(source_batch / "manifest.json")
            _validate_reused_rerank_preflight(source_manifest, args, indices)
            rerank_preflight = dict(
                source_manifest.get("rerank_cache_preflight") or {}
            )
            rerank_preflight["reused_from_batch"] = str(source_batch)
            _log(
                config,
                "reusing matched assay-transfer cache preflight from "
                f"{source_batch}",
            )
        elif args.reuse_existing_rerank_preflight:
            existing_manifest = _read_json(batch_dir / "manifest.json")
            _validate_reused_rerank_preflight(existing_manifest, args, indices)
            rerank_preflight = existing_manifest.get("rerank_cache_preflight") or {}
            _log(config, "reusing complete assay-transfer cache preflight from existing batch manifest")
        else:
            _log(config, "running read-only assay-transfer cache coverage preflight")
            rerank_preflight_kwargs = dict(
                records=records,
                indices=indices,
                smiles_field=args.smiles_field,
                index_path=args.index,
                catalog_path=args.rerank_catalog,
                candidate_manifest_path=args.rerank_candidate_manifest,
                cache_path=args.rerank_cache,
                model=args.assay_transfer_model,
                model_revision=args.assay_transfer_model_revision,
                experiment_mode=args.experiment_mode,
                top_k_per_group=args.top_k_per_group,
                min_similarity=args.min_similarity,
                neighbor_identity_policy=args.neighbor_identity_policy,
                initial_morgan_filter=args.assay_transfer_initial_morgan_filter,
                require_selected_scores=args.enable_assay_transfer_scores,
                template_profile=args.assay_transfer_template_profile,
                expected_score_count=args.rerank_expected_score_count,
                cache_version_path=args.rerank_cache_version_manifest,
                retrieval_source=args.retrieval_source,
                assay_transfer_profile=args.assay_transfer_profile,
                assay_transfer_min_score=args.assay_transfer_min_score,
                assay_transfer_diversity_mode=args.assay_transfer_diversity_mode,
                assay_transfer_diversity_score_slack=args.assay_transfer_diversity_score_slack,
                assay_transfer_selection_unit=args.assay_transfer_selection_unit,
                assay_transfer_records_per_molecule=(
                    args.assay_transfer_records_per_molecule
                ),
            )
            if config.supports_nondirect_bioavailability_filter:
                rerank_preflight_kwargs[
                    "exclude_nondirect_bioavailability_records"
                ] = bool(
                    getattr(args, "exclude_nondirect_bioavailability_records", False)
                )
            rerank_preflight = config.rerank_preflight(**rerank_preflight_kwargs)
            _release_preflight_memory()
    manifest = {
        "batch_id": batch_id,
        "input_jsonl": args.input_jsonl,
        "smiles_field": args.smiles_field,
        "label_field": args.label_field,
        "n_items": len(items),
        "indices": indices,
        "parallelism": args.parallelism,
        # Retained as a compatibility field for existing viewers/manifests.
        # Prompt fan-out is now exclusively governed by the global pool.
        "group_workers": 1,
        "save_trace": args.save_trace,
        "combine_traces": args.combine_traces,
        "stream_logs": args.stream_logs,
        "model": args.model,
        "experiment_mode": args.experiment_mode,
        "retrieval_source": args.retrieval_source,
        "exclude_nondirect_bioavailability_records": (
            bool(getattr(args, "exclude_nondirect_bioavailability_records", False))
        ),
        "retrieval_strategy": args.retrieval_strategy,
        "retrieval_reranker": "assay_transfer" if is_assay_transfer else "none",
        "assay_transfer_profile": args.assay_transfer_profile,
        "enable_assay_transfer_scores": args.enable_assay_transfer_scores,
        "assay_transfer_min_score": args.assay_transfer_min_score,
        "assay_transfer_diversity_mode": args.assay_transfer_diversity_mode,
        "assay_transfer_diversity_score_slack": args.assay_transfer_diversity_score_slack,
        "assay_transfer_selection_unit": args.assay_transfer_selection_unit,
        "assay_transfer_records_per_molecule": (
            args.assay_transfer_records_per_molecule
        ),
        "assay_transfer_template_profile": args.assay_transfer_template_profile,
        "group_prompt_format": args.group_prompt_format,
        "group_prompt_version": (
            "analogous_flat_v1" if _is_analogous_flat(args) else args.group_prompt_version
        ),
        "group_prompt_provenance": group_prompt_provenance,
        "group_output_schema": (
            "analogous_flat_minimal_v1"
            if _is_analogous_flat(args)
            else args.group_output_schema
        ),
        "group_prompt_instructions_file": group_prompt_instruction_provenance.get(
            "path", ""
        ),
        "group_prompt_instructions_sha256": group_prompt_instruction_provenance.get(
            "sha256", ""
        ),
        "group_prompt_instruction_count": group_prompt_instruction_provenance.get(
            "instruction_count", 0
        ),
        "group_evidence_presentation": (
            "analogous_flat_minimal.v1"
            if _is_analogous_flat(args)
            else "minimal_evidence.v1"
            if args.group_prompt_format in {"morganfingerprint", "assay_transfer_tool"}
            else "legacy"
        ),
        "llm_neighbor_score_policy": (
            "assay_transfer_scored_neighbors.v1" if args.enable_assay_transfer_scores else ""
        ),
        "assay_transfer_initial_morgan_filter": args.assay_transfer_initial_morgan_filter,
        "rerank_catalog": args.rerank_catalog if is_assay_transfer else "",
        "rerank_cache": args.rerank_cache if is_assay_transfer else "",
        "rerank_candidate_manifest": (
            args.rerank_candidate_manifest if is_assay_transfer else ""
        ),
        "rerank_cache_preflight": rerank_preflight,
        "rerank_expected_score_count": args.rerank_expected_score_count,
        "rerank_cache_version_manifest": args.rerank_cache_version_manifest,
        "rerank_preflight_reused": bool(
            args.reuse_existing_rerank_preflight
            or args.rerank_preflight_source_batch
        ),
        "rerank_preflight_source_batch": args.rerank_preflight_source_batch,
        "min_similarity": args.min_similarity,
        "neighbor_identity_policy": args.neighbor_identity_policy,
        "morgan_neighbor_selector": args.morgan_neighbor_selector,
        "neighbor_selector": args.neighbor_selector,
        "neighbor_context_profile": args.neighbor_context_profile,
        "final_evidence_surface": getattr(args, "final_evidence_surface", SUMMARY_ONLY),
        "final_decision_profile": getattr(
            args,
            "final_decision_profile",
            STANDARD_FINAL_DECISION,
        ),
        "task_prompt_profile": getattr(args, "task_prompt_profile", ""),
        "identity_blind": args.identity_blind,
        "disable_flat_tools": args.disable_flat_tools,
        "group_tools_enabled": (
            not args.disable_group_tools
            and not args.disable_flat_tools
            and not args.analogous_reasoning_only
        ),
        "flat_group_tool_policy": (
            "omitted.v1"
            if args.disable_flat_tools or args.analogous_reasoning_only
            else "standard.v1"
        ),
        "analogous_reasoning_only": args.analogous_reasoning_only,
        "single_branch_execution": (
            "omitted" if args.analogous_reasoning_only else "executed_or_reused"
        ),
        "single_branch_omission_reason": (
            "analogous_reasoning_only" if args.analogous_reasoning_only else ""
        ),
        "query_tool_execution": (
            "omitted" if args.analogous_reasoning_only else "enabled"
        ),
        "group_query_tool_instruction_policy": (
            "omitted.v1" if args.analogous_reasoning_only else "standard.v1"
        ),
        "final_prompt_provenance": (
            config.final_prompt_provenance(
                analogous_reasoning_only=args.analogous_reasoning_only
            )
            if config.final_prompt_provenance is not None and not _is_analogous_flat(args)
            else {}
        ),
        "analogous_flat_prompt_provenance": (
            analogous_flat_prompt_provenance(config.pipeline_module.split(".")[-2])
            if _is_analogous_flat(args)
            else {}
        ),
        "harness_prefetch_tools": (
            False
            if args.analogous_reasoning_only
            else args.identity_blind or args.harness_prefetch_tools
        ),
        "visibility_mode": (
            PROMPT_IDENTITY_VIEW
            if _is_analogous_flat(args)
            else "identity_blind"
            if args.identity_blind
            else "deployment_visible_prefetched"
            if args.harness_prefetch_tools
            else "deployment_visible"
        ),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k_per_group_requested": requested_top_k_per_group,
        "top_k_per_group": args.top_k_per_group,
        "transport_max_retries": 2,
        "single_analysis_source_batch": args.single_analysis_source_batch,
        "group_analysis_source_batch": args.group_analysis_source_batch,
        "retrieval_replay_source_batch": args.retrieval_replay_source_batch,
        "prefetched_tool_replay_source_batch": args.prefetched_tool_replay_source_batch,
        "tier1_replacement_index": args.tier1_replacement_index,
        "tier1_replacement_groups": args.tier1_replacement_groups or [],
        "final_only_source_batch": args.final_only_source_batch,
        "final_only_groups": args.final_only_groups or [],
        "started_at": _now(),
        "paths": {
            "batch_dir": str(batch_dir),
            "predictions": str(batch_dir / "predictions.jsonl"),
            "metrics": str(batch_dir / "metrics.json"),
            "report": str(batch_dir / "report.md"),
            "molecule_runs": str(batch_run_root),
            "combined_trace": str(batch_dir / "trace_messages.jsonl"),
            "logs": str(logs_dir),
        },
    }
    _write_json(batch_dir / "manifest.json", manifest)

    _log(config, f"batch_id={batch_id}")
    _log(config, f"items={len(items)} parallelism={args.parallelism}")
    return PreparedBatch(
        config=config,
        args=args,
        batch_id=batch_id,
        batch_dir=batch_dir,
        logs_dir=logs_dir,
        batch_run_root=batch_run_root,
        items=items,
        manifest=manifest,
    )


def _resolve_item_future(
    prepared: PreparedBatch,
    item: BatchItem,
    future: concurrent.futures.Future[dict[str, Any]],
) -> dict[str, Any]:
    try:
        result = future.result()
    except Exception as exc:  # noqa: BLE001 - a batch must keep making progress.
        result = _error_result(
            prepared.config,
            prepared.args,
            item,
            prepared.batch_id,
            str(exc),
        )
    return result


def collect_completed_item(
    prepared: PreparedBatch,
    item: BatchItem,
) -> dict[str, Any] | None:
    """Return a complete existing result without consuming a worker slot."""
    if not prepared.args.skip_existing:
        return None
    run_id = f"{prepared.batch_id}_idx{item.index:05d}"
    run_dir = prepared.batch_run_root / run_id
    if not _has_resume_artifacts(run_dir):
        return None
    result = _collect_result(
        prepared.config,
        prepared.args,
        item,
        run_id,
        run_dir,
    )
    if not _result_is_complete(result):
        return None
    stdout_path = prepared.logs_dir / f"{run_id}.stdout.log"
    stderr_path = prepared.logs_dir / f"{run_id}.stderr.log"
    result.update(
        {
            "status": "ok",
            "returncode": 0,
            "latency_s": 0.0,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }
    )
    return result


def finalize_batch(prepared: PreparedBatch, results: list[dict[str, Any]]) -> int:
    """Write the same predictions/metrics/report contract as the legacy runner."""
    config = prepared.config
    args = prepared.args
    batch_dir = prepared.batch_dir
    manifest = prepared.manifest

    results.sort(key=lambda row: row["query_index"])
    if args.save_trace and args.combine_traces:
        _combine_traces(batch_dir / "trace_messages.jsonl", results)

    metrics = compute_metrics(config, results)
    metrics["finished_at"] = _now()
    _write_jsonl(batch_dir / "predictions.jsonl", results)
    _write_json(batch_dir / "metrics.json", metrics)
    _write_report(config, batch_dir / "report.md", manifest, metrics, results)

    manifest["finished_at"] = metrics["finished_at"]
    _write_json(batch_dir / "manifest.json", manifest)
    print(json.dumps({"manifest": manifest, "metrics": metrics}, ensure_ascii=False, indent=2), flush=True)
    return 0 if metrics["n_failed_runs"] == 0 else 1


def _release_preflight_memory() -> None:
    """Return large, temporary retrieval-audit allocations before fan-out."""
    gc.collect()
    try:
        libc = ctypes.CDLL(None)
        malloc_trim = libc.malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)
    except (AttributeError, OSError):
        pass


def _validate_reused_rerank_preflight(
    manifest: dict[str, Any], args: argparse.Namespace, indices: list[int]
) -> None:
    preflight = manifest.get("rerank_cache_preflight") or {}
    expected = {
        "indices": indices,
        "experiment_mode": args.experiment_mode,
        "retrieval_source": args.retrieval_source,
        "exclude_nondirect_bioavailability_records": (
            bool(getattr(args, "exclude_nondirect_bioavailability_records", False))
        ),
        "retrieval_reranker": (
            "assay_transfer"
            if args.retrieval_strategy == ASSAY_TRANSFER_TOOL_STRATEGY
            else "none"
        ),
        "enable_assay_transfer_scores": args.enable_assay_transfer_scores,
        "assay_transfer_min_score": args.assay_transfer_min_score,
        "assay_transfer_diversity_mode": args.assay_transfer_diversity_mode,
        "assay_transfer_diversity_score_slack": args.assay_transfer_diversity_score_slack,
        "assay_transfer_selection_unit": args.assay_transfer_selection_unit,
        "assay_transfer_records_per_molecule": (
            args.assay_transfer_records_per_molecule
        ),
        "assay_transfer_template_profile": args.assay_transfer_template_profile,
        "group_prompt_format": args.group_prompt_format,
        "assay_transfer_initial_morgan_filter": args.assay_transfer_initial_morgan_filter,
        "rerank_catalog": args.rerank_catalog,
        "rerank_cache": args.rerank_cache,
        "rerank_candidate_manifest": args.rerank_candidate_manifest,
        "rerank_expected_score_count": args.rerank_expected_score_count,
        "rerank_cache_version_manifest": args.rerank_cache_version_manifest,
        "neighbor_identity_policy": args.neighbor_identity_policy,
        "top_k_per_group": args.top_k_per_group,
        "min_similarity": args.min_similarity,
    }
    mismatches = [
        key for key, expected_value in expected.items() if manifest.get(key) != expected_value
    ]
    if (
        "assay_transfer_profile" in manifest
        or args.assay_transfer_profile == "v11_with_categorical"
    ) and manifest.get("assay_transfer_profile") != args.assay_transfer_profile:
        mismatches.append("assay_transfer_profile")
    provenance = preflight.get("provenance") or {}
    if provenance.get("model") != args.assay_transfer_model:
        mismatches.append("assay_transfer_model")
    if provenance.get("model_revision") != args.assay_transfer_model_revision:
        mismatches.append("assay_transfer_model_revision")
    if preflight.get("status") != "complete" or int(preflight.get("n_queries") or -1) != len(indices):
        mismatches.append("rerank_cache_preflight")
    if mismatches:
        raise SystemExit(
            "--reuse-existing-rerank-preflight does not match the current run: "
            + ", ".join(sorted(set(mismatches)))
        )


@contextmanager
def _exclusive_run_lock(run_dir: Path):
    """Serialize writers for one sample-condition across resume launchers."""
    lock_path = run_dir / ".run.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _prepare_final_only_run_dir(args: argparse.Namespace, query_index: int, run_id: str, run_dir: Path) -> None:
    source_batch_dir = Path(args.final_only_source_batch)
    if not source_batch_dir.exists():
        raise FileNotFoundError(f"Final-only source batch does not exist: {source_batch_dir}")
    source_runs_dir = source_batch_dir / "runs"
    matches = sorted(source_runs_dir.glob(f"*_idx{query_index:05d}"))
    if not matches:
        raise FileNotFoundError(f"No source run for query index {query_index}: {source_runs_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous source runs for query index {query_index}: {matches}")
    source_run_dir = matches[0]
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("retrieval.json", "single_molecule_reasoning_output.json", "group_reasoning_outputs.jsonl"):
        source_path = source_run_dir / name
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source artifact for final-only rerun: {source_path}")
        shutil.copy2(source_path, run_dir / name)

    manifest_path = source_run_dir / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    manifest["run_id"] = run_id
    manifest["final_only_source_run_dir"] = str(source_run_dir)
    manifest["final_only_source_batch"] = str(source_batch_dir)
    manifest["final_evidence_surface"] = getattr(
        args,
        "final_evidence_surface",
        SUMMARY_ONLY,
    )
    manifest["final_decision_profile"] = getattr(
        args,
        "final_decision_profile",
        STANDARD_FINAL_DECISION,
    )
    if args.final_only_groups:
        filter_audit = _filter_final_only_run_artifacts(run_dir, args.final_only_groups)
        manifest["final_only_group_filter"] = filter_audit
        manifest["n_groups_with_neighbors"] = filter_audit["n_retained_groups_with_neighbors"]
    manifest.setdefault("paths", {})
    manifest["paths"].update(
        {
            "retrieval": str(run_dir / "retrieval.json"),
            "single_molecule_reasoning_output": str(run_dir / "single_molecule_reasoning_output.json"),
            "group_reasoning_outputs": str(run_dir / "group_reasoning_outputs.jsonl"),
            "final_reasoning_output": str(run_dir / "final_reasoning_output.json"),
            "trace_messages": str(run_dir / "trace_messages.jsonl"),
        }
    )
    _write_json(run_dir / "manifest.json", manifest)


def _filter_final_only_run_artifacts(run_dir: Path, requested_group_ids: list[str]) -> dict[str, Any]:
    """Restrict copied retrieval/group artifacts before a final-only rerun."""
    requested = list(dict.fromkeys(str(group_id) for group_id in requested_group_ids if group_id))
    if not requested:
        raise ValueError("Final-only group filtering requires at least one group id.")
    requested_set = set(requested)

    retrieval_path = run_dir / "retrieval.json"
    group_path = run_dir / "group_reasoning_outputs.jsonl"
    retrieval = _read_json(retrieval_path)
    groups = list(retrieval.get("groups") or [])
    available = {str(group.get("group_id") or "") for group in groups}
    missing = sorted(requested_set - available)
    if missing:
        raise ValueError(
            "Final-only group ids are absent from copied retrieval: "
            + ", ".join(missing)
        )

    retained_groups = [
        group for group in groups if str(group.get("group_id") or "") in requested_set
    ]
    outputs = _read_jsonl(group_path)
    retained_outputs = [
        output
        for output in outputs
        if str(output.get("group_id") or "") in requested_set
    ]
    retained_output_ids = [str(output.get("group_id") or "") for output in retained_outputs]
    expected_output_ids = [
        str(group.get("group_id") or "")
        for group in retained_groups
        if group.get("neighbors")
    ]
    if retained_output_ids != expected_output_ids:
        raise ValueError(
            "Copied group outputs do not match retained retrieval groups with neighbors: "
            f"expected={expected_output_ids}, found={retained_output_ids}"
        )

    coverage = dict(retrieval.get("coverage") or {})
    coverage.update(
        {
            "n_groups": len(retained_groups),
            "n_groups_with_neighbors": len(expected_output_ids),
            "n_neighbors_total": sum(
                len(group.get("neighbors") or []) for group in retained_groups
            ),
        }
    )
    retrieval["groups"] = retained_groups
    retrieval["coverage"] = coverage
    retrieval.setdefault("experiment", {})["final_only_group_filter"] = {
        "requested_group_ids": requested,
        "source_group_ids": [str(group.get("group_id") or "") for group in groups],
        "retained_group_ids": [str(group.get("group_id") or "") for group in retained_groups],
    }
    _write_json(retrieval_path, retrieval)
    _write_jsonl(group_path, retained_outputs)
    return {
        "requested_group_ids": requested,
        "source_group_ids": [str(group.get("group_id") or "") for group in groups],
        "retained_group_ids": [str(group.get("group_id") or "") for group in retained_groups],
        "n_source_group_outputs": len(outputs),
        "n_retained_group_outputs": len(retained_outputs),
        "n_retained_groups_with_neighbors": len(expected_output_ids),
    }


def _single_run_command(
    config: BatchConfig,
    args: argparse.Namespace,
    query_index: int,
    run_id: str,
    run_root: Path,
) -> list[str]:
    command = [
        args.python_executable,
        "-m",
        config.pipeline_module,
        "--input-jsonl",
        args.input_jsonl,
        "--query-index",
        str(query_index),
        "--smiles-field",
        args.smiles_field,
        "--index",
        args.index,
        "--out-root",
        str(run_root),
        "--run-id",
        run_id,
        "--env-file",
        args.env_file,
        "--api-key-env",
        args.api_key_env,
        "--base-url",
        args.base_url,
        "--tool-service-url",
        args.tool_service_url,
        "--model",
        args.model,
        "--timeout-s",
        str(args.timeout_s),
        "--max-tokens",
        str(args.max_tokens),
        "--max-tool-rounds",
        str(args.max_tool_rounds),
        "--reasoning-effort",
        args.reasoning_effort,
        "--top-k-per-group",
        str(args.top_k_per_group),
        "--min-similarity",
        str(args.min_similarity),
    ]
    if config.supports_shared_retrieval_contract:
        command.extend(
            [
                "--experiment-mode",
                args.experiment_mode,
                "--retrieval-source",
                args.retrieval_source,
                "--neighbor-identity-policy",
                args.neighbor_identity_policy,
                "--morgan-neighbor-selector",
                args.neighbor_selector,
                "--neighbor-context-profile",
                args.neighbor_context_profile,
            ]
        )
    if (
        config.supports_nondirect_bioavailability_filter
        and bool(getattr(args, "exclude_nondirect_bioavailability_records", False))
    ):
        command.append("--exclude-nondirect-bioavailability-records")
    if args.assay_transfer_min_score is not None:
        command.extend(
            ["--assay-transfer-min-score", str(args.assay_transfer_min_score)]
        )
    if args.assay_transfer_diversity_mode != ASSAY_TRANSFER_DIVERSITY_NONE:
        command.extend(
            [
                "--assay-transfer-diversity-mode",
                args.assay_transfer_diversity_mode,
                "--assay-transfer-diversity-score-slack",
                str(args.assay_transfer_diversity_score_slack),
            ]
        )
    if args.assay_transfer_selection_unit != ASSAY_TRANSFER_SELECTION_SCORED_RECORD:
        command.extend(
            ["--assay-transfer-selection-unit", args.assay_transfer_selection_unit]
        )
    if (
        args.assay_transfer_records_per_molecule
        != ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT
    ):
        command.extend(
            [
                "--assay-transfer-records-per-molecule",
                str(args.assay_transfer_records_per_molecule),
            ]
        )
    if args.group_prompt_format:
        command.extend(["--group-prompt-format", args.group_prompt_format])
    if args.group_prompt_version:
        command.extend(["--group-prompt-version", args.group_prompt_version])
    if args.group_output_schema:
        command.extend(["--group-output-schema", args.group_output_schema])
    if args.group_prompt_instructions_file:
        command.extend(
            [
                "--group-prompt-instructions-file",
                args.group_prompt_instructions_file,
                "--group-prompt-instructions-sha256",
                args.group_prompt_instructions_sha256,
            ]
        )
    if args.group_prompt_min_similarity is not None:
        command.extend(
            ["--group-prompt-min-similarity", str(args.group_prompt_min_similarity)]
        )
    if config.group_prompt_formats and getattr(args, "presentation_style", "legacy") != "legacy":
        command.extend(["--presentation-style", args.presentation_style])
    if config.supports_shared_retrieval_contract:
        final_surface = getattr(args, "final_evidence_surface", SUMMARY_ONLY)
        if final_surface != SUMMARY_ONLY:
            command.extend(["--final-evidence-surface", final_surface])
        final_decision_profile = getattr(args, "final_decision_profile", STANDARD_FINAL_DECISION)
        if config.supports_final_decision_profiles and final_decision_profile != STANDARD_FINAL_DECISION:
            command.extend(["--final-decision-profile", final_decision_profile])
    if config.prompt_profile_option:
        command.extend(
            [
                config.prompt_profile_option,
                str(args.task_prompt_profile),
            ]
        )
    if not args.enable_thinking:
        command.append("--disable-thinking")
    else:
        command.append("--enable-thinking")
    if args.max_groups:
        command.extend(["--max-groups", str(args.max_groups)])
    if args.groups:
        command.append("--groups")
        command.extend(args.groups)
    if args.tier1_replacement_index and config.supports_shared_retrieval_contract:
        command.extend(["--tier1-replacement-index", args.tier1_replacement_index])
        if args.tier1_replacement_groups:
            command.append("--tier1-replacement-groups")
            command.extend(args.tier1_replacement_groups)
    if args.disable_group_tools:
        command.append("--disable-group-tools")
    if args.disable_flat_tools:
        command.append("--disable-flat-tools")
    if args.analogous_reasoning_only:
        command.append("--analogous-reasoning-only")
    if args.identity_blind and config.supports_shared_retrieval_contract:
        command.append("--identity-blind")
    elif args.harness_prefetch_tools and config.supports_shared_retrieval_contract:
        command.append("--harness-prefetch-tools")
    if config.supports_retrieval_strategy:
        command.extend(["--retrieval-strategy", args.retrieval_strategy])
    if args.retrieval_strategy == ASSAY_TRANSFER_TOOL_STRATEGY:
        command.extend(
            [
                "--assay-transfer-initial-morgan-filter",
                str(args.assay_transfer_initial_morgan_filter),
                "--assay-transfer-profile",
                args.assay_transfer_profile,
                "--rerank-catalog",
                args.rerank_catalog,
                "--rerank-cache",
                args.rerank_cache,
                "--rerank-candidate-manifest",
                args.rerank_candidate_manifest,
                "--rerank-cache-mode",
                args.rerank_cache_mode,
                "--assay-transfer-model",
                args.assay_transfer_model,
                "--assay-transfer-model-revision",
                args.assay_transfer_model_revision,
                "--assay-transfer-template-profile",
                args.assay_transfer_template_profile,
            ]
        )
    if args.enable_assay_transfer_scores:
        command.append("--enable-assay-transfer-scores")
    if args.single_analysis_source_batch and config.supports_shared_retrieval_contract:
        source_batch = Path(args.single_analysis_source_batch)
        source_run_id = f"{source_batch.name}_idx{query_index:05d}"
        command.extend(
            [
                "--single-analysis-source-run-dir",
                str(source_batch / "runs" / source_run_id),
            ]
        )
    if args.group_analysis_source_batch and config.supports_shared_retrieval_contract:
        source_batch = Path(args.group_analysis_source_batch)
        source_run_id = f"{source_batch.name}_idx{query_index:05d}"
        command.extend(
            [
                "--group-analysis-source-run-dir",
                str(source_batch / "runs" / source_run_id),
            ]
        )
    if args.retrieval_replay_source_batch and config.supports_shared_retrieval_contract:
        source_batch = Path(args.retrieval_replay_source_batch)
        source_run_id = f"{source_batch.name}_idx{query_index:05d}"
        command.extend(
            [
                "--retrieval-replay-run-dir",
                str(source_batch / "runs" / source_run_id),
            ]
        )
    if args.prefetched_tool_replay_source_batch and config.supports_shared_retrieval_contract:
        source_batch = Path(args.prefetched_tool_replay_source_batch)
        source_run_id = f"{source_batch.name}_idx{query_index:05d}"
        command.extend(
            [
                "--prefetched-tool-replay-run-dir",
                str(source_batch / "runs" / source_run_id),
            ]
        )
    return command


def _has_resume_artifacts(run_dir: Path) -> bool:
    return any(
        (run_dir / name).exists()
        for name in (
            "retrieval.json",
            "single_molecule_reasoning_output.json",
            "group_reasoning_outputs.jsonl",
            "final_reasoning_output.json",
            "manifest.json",
        )
    )


def _run_subprocess_with_logs(
    command: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    stream_logs: bool,
    prefix: str,
) -> int:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        stdout_thread = threading.Thread(
            target=_copy_stream,
            args=(process.stdout, stdout_file, False, prefix, "stdout"),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_copy_stream,
            args=(process.stderr, stderr_file, stream_logs, prefix, "stderr"),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        returncode = process.wait()
        stdout_thread.join()
        stderr_thread.join()
    return returncode


def _copy_stream(stream: Any, log_file: Any, echo: bool, prefix: str, stream_name: str) -> None:
    if stream is None:
        return
    for line in stream:
        log_file.write(line)
        log_file.flush()
        if echo:
            print(f"[{prefix} {stream_name}] {line}", end="", file=sys.stderr, flush=True)


def _tail_text(*paths: Path, limit: int = 4000) -> str:
    chunks = []
    for path in paths:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            if text:
                chunks.append(f"==> {path}\n{text[-limit:]}")
    return "\n".join(chunks) or "subprocess failed"


def _collect_result(
    config: BatchConfig,
    args: argparse.Namespace,
    item: BatchItem,
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    final_path = run_dir / "final_reasoning_output.json"
    single_path = run_dir / "single_molecule_reasoning_output.json"
    group_path = run_dir / "group_reasoning_outputs.jsonl"
    manifest_path = run_dir / "manifest.json"
    final_output = _read_json(final_path) if final_path.exists() else {}
    single_output = _read_json(single_path) if single_path.exists() else {}
    group_outputs = _read_jsonl(group_path) if group_path.exists() else []
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    reuse_path = run_dir / "reuse.json"
    reuse = _read_json(reuse_path) if reuse_path.exists() else {}
    content = ((final_output.get("llm") or {}).get("content") or {}) if isinstance(final_output, dict) else {}
    prediction = _normalize_prediction(config, content.get(config.prediction_field))
    pred_label = prediction_to_label(config, prediction)
    true_label = _parse_label(item.record.get(args.label_field))
    correct = bool(pred_label == true_label) if pred_label is not None and true_label is not None else False
    return {
        "query_index": item.index,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "smiles": item.record.get(args.smiles_field, ""),
        "label": true_label,
        config.prediction_field: prediction,
        "pred_label": pred_label,
        "confidence": content.get("confidence"),
        "evidence_state": content.get("evidence_state"),
        "prior_used": content.get("prior_used"),
        "correct": correct,
        "final_status": (final_output.get("status") if isinstance(final_output, dict) else None),
        "single_status": (single_output.get("status") if isinstance(single_output, dict) else None),
        "analogous_reasoning_only": bool(manifest.get("analogous_reasoning_only")),
        "n_group_outputs": len(group_outputs),
        "n_failed_group_outputs": sum(row.get("status") != "ok" for row in group_outputs),
        "n_groups_with_neighbors": manifest.get("n_groups_with_neighbors"),
        "expected_group_ids": manifest.get("expected_group_ids"),
        "group_ids": [str(row.get("group_id") or "") for row in group_outputs],
        "trace_messages": str(run_dir / "trace_messages.jsonl") if (run_dir / "trace_messages.jsonl").exists() else "",
        "final_reasoning_output": str(final_path) if final_path.exists() else "",
        "final_summary": content.get("final_summary", ""),
        "reused_from": reuse.get("reused_from", ""),
        "reuse_reason": reuse.get("reuse_reason", ""),
    }


def _result_is_complete(result: dict[str, Any]) -> bool:
    if result.get("final_status") != "ok" or result.get("pred_label") is None:
        return False
    acceptable_single_status = (
        "omitted" if result.get("analogous_reasoning_only") else "ok"
    )
    if result.get("single_status") != acceptable_single_status:
        return False
    expected_groups = result.get("n_groups_with_neighbors")
    if expected_groups is not None and int(result.get("n_group_outputs") or 0) != int(expected_groups):
        return False
    expected_group_ids = result.get("expected_group_ids")
    if expected_group_ids is not None:
        expected_ids = [str(group_id) for group_id in expected_group_ids]
        actual_ids = [str(group_id) for group_id in result.get("group_ids") or []]
        if len(actual_ids) != len(expected_ids) or set(actual_ids) != set(expected_ids):
            return False
    return int(result.get("n_failed_group_outputs") or 0) == 0


def compute_metrics(config: BatchConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [row for row in rows if row.get("label") in (0, 1)]
    successful = [row for row in evaluable if row.get("status") == "ok"]
    correct = sum(1 for row in successful if row.get("correct"))
    per_class = {str(label): _class_metrics(successful, label) for label in (0, 1)}
    macro_f1 = round(sum(per_class[str(label)]["f1"] for label in (0, 1)) / 2, 6) if successful else 0.0
    prediction_distribution: dict[str, int] = {}
    for row in successful:
        prediction = str(row.get(config.prediction_field) or "missing")
        prediction_distribution[prediction] = prediction_distribution.get(prediction, 0) + 1
    confusion_matrix = {
        "tn": sum(1 for row in successful if row.get("label") == 0 and row.get("pred_label") == 0),
        "fp": sum(1 for row in successful if row.get("label") == 0 and row.get("pred_label") == 1),
        "fn": sum(1 for row in successful if row.get("label") == 1 and row.get("pred_label") == 0),
        "tp": sum(1 for row in successful if row.get("label") == 1 and row.get("pred_label") == 1),
    }
    positive_class = per_class["1"]
    return {
        "n_total": len(rows),
        "n_evaluable": len(evaluable),
        "n_successful": len(successful),
        "n_failed_runs": sum(1 for row in rows if row.get("status") != "ok"),
        "accuracy": _safe_div(correct, len(successful)),
        "accuracy_including_failed": _safe_div(correct, len(evaluable)),
        "macro_f1": macro_f1,
        "per_class": per_class,
        "positive_class_precision": positive_class["precision"],
        "positive_class_recall": positive_class["recall"],
        "positive_class_f1": positive_class["f1"],
        "confusion_matrix": confusion_matrix,
        "prediction_distribution": prediction_distribution,
    }


def _class_metrics(rows: list[dict[str, Any]], label: int) -> dict[str, Any]:
    tp = sum(1 for row in rows if row.get("label") == label and row.get("pred_label") == label)
    fp = sum(1 for row in rows if row.get("label") != label and row.get("pred_label") == label)
    fn = sum(1 for row in rows if row.get("label") == label and row.get("pred_label") != label)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def prediction_to_label(config: BatchConfig, prediction: str | None) -> int | None:
    normalized = str(prediction or "").strip().lower()
    if normalized in config.positive_predictions or normalized == config.canonical_positive:
        return 1
    if normalized in config.negative_predictions or normalized == config.canonical_negative:
        return 0
    return None


def _normalize_prediction(config: BatchConfig, value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in config.positive_predictions:
        return config.canonical_positive
    if text in config.negative_predictions:
        return config.canonical_negative
    return text or "missing"


def _parse_label(value: Any) -> int | None:
    try:
        label = int(value)
    except (TypeError, ValueError):
        return None
    return label if label in (0, 1) else None


def _safe_div(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _select_indices(args: argparse.Namespace, n_records: int) -> list[int]:
    if args.indices:
        indices = []
        for token in args.indices:
            indices.extend(_parse_index_token(token))
    else:
        end = n_records if args.limit <= 0 else min(n_records, args.start + args.limit)
        indices = list(range(args.start, end))
    bad = [index for index in indices if index < 0 or index >= n_records]
    if bad:
        raise SystemExit(f"Query indices out of range 0..{n_records - 1}: {bad}")
    return sorted(dict.fromkeys(indices))


def _parse_index_token(token: str) -> list[int]:
    if "-" not in token:
        return [int(token)]
    start, end = token.split("-", 1)
    return list(range(int(start), int(end) + 1))


def _error_result(
    config: BatchConfig,
    args: argparse.Namespace,
    item: BatchItem,
    batch_id: str,
    error: str,
) -> dict[str, Any]:
    run_id = f"{batch_id}_idx{item.index:05d}"
    return {
        "query_index": item.index,
        "run_id": run_id,
        "status": "error",
        "smiles": item.record.get(args.smiles_field, ""),
        "label": _parse_label(item.record.get(args.label_field)),
        config.prediction_field: "missing",
        "pred_label": None,
        "correct": False,
        "error": error,
    }


def _combine_traces(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            trace = row.get("trace_messages")
            if not trace:
                continue
            trace_path = Path(trace)
            if trace_path.exists():
                out.write(trace_path.read_text(encoding="utf-8"))


def _write_report(
    config: BatchConfig,
    path: Path,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        f"# {config.report_title}: {manifest['batch_id']}",
        "",
        f"- input_jsonl: `{manifest['input_jsonl']}`",
        f"- n_items: {metrics['n_total']}",
        f"- n_successful: {metrics['n_successful']}",
        f"- n_failed_runs: {metrics['n_failed_runs']}",
        f"- accuracy: {metrics['accuracy']}",
        f"- accuracy_including_failed: {metrics['accuracy_including_failed']}",
        f"- macro_f1: {metrics['macro_f1']}",
        f"- positive_class_precision: {metrics['positive_class_precision']}",
        f"- positive_class_recall: {metrics['positive_class_recall']}",
        f"- positive_class_f1: {metrics['positive_class_f1']}",
        f"- confusion_matrix: `{json.dumps(metrics['confusion_matrix'], ensure_ascii=False)}`",
        f"- prediction_distribution: `{json.dumps(metrics['prediction_distribution'], ensure_ascii=False)}`",
        "",
        "| index | label | prediction | pred_label | correct | confidence | run_id |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {query_index} | {label} | {prediction} | {pred_label} | {correct} | {confidence} | {run_id} |".format(
                **{
                    "prediction": row.get(config.prediction_field, ""),
                    **{key: row.get(key, "") for key in [
                        "query_index",
                        "label",
                        "pred_label",
                        "correct",
                        "confidence",
                        "run_id",
                    ]},
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _log(config: BatchConfig, message: str) -> None:
    print(f"[{config.log_prefix}] {message}", file=sys.stderr, flush=True)


def _parse_args(config: BatchConfig, argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=config.description)
    parser.add_argument("--input-jsonl", default=config.default_input)
    parser.add_argument("--smiles-field", default="drug")
    parser.add_argument("--label-field", default="Y")
    parser.add_argument("--index", default=config.default_index)
    parser.add_argument(
        "--experiment-mode",
        choices=["none", "direct", "full_flat", "full_mechanism", "native"],
        default="native",
    )
    parser.add_argument("--retrieval-source", default="chembl")
    if config.supports_nondirect_bioavailability_filter:
        parser.add_argument(
            "--exclude-nondirect-bioavailability-records",
            action="store_true",
            help=(
                "Exclude retained relative/apparent Bioavailability HF records "
                "before neighbor ranking and top-k selection."
            ),
        )
    else:
        parser.set_defaults(exclude_nondirect_bioavailability_records=False)
    parser.add_argument(
        "--neighbor-identity-policy",
        choices=NEIGHBOR_IDENTITY_POLICIES,
        default="operational",
    )
    parser.add_argument("--identity-blind", action="store_true")
    if config.supports_analogous_reasoning_only:
        parser.add_argument(
            "--analogous-reasoning-only",
            "--analogous_reasoning_only",
            dest="analogous_reasoning_only",
            action="store_true",
            help=(
                "Bioavailability full_mechanism only: omit the single-molecule "
                "branch and every model-facing query tool."
            ),
        )
    else:
        parser.set_defaults(analogous_reasoning_only=False)
    parser.add_argument(
        "--single-analysis-source-batch",
        default="",
        help="Reuse each query's frozen single-molecule branch from another batch.",
    )
    parser.add_argument(
        "--group-analysis-source-batch",
        default="",
        help="Reuse independent group branches with identical LLM-visible retrieval input.",
    )
    parser.add_argument(
        "--retrieval-replay-source-batch",
        default="",
        help="Reuse each query's frozen retrieval.json from another batch.",
    )
    parser.add_argument(
        "--prefetched-tool-replay-source-batch",
        default="",
        help="Reuse each query's frozen harness-prefetched tool outputs from another batch.",
    )
    parser.add_argument("--batch-root", default=config.default_batch_root)
    parser.add_argument("--batch-id", default="")
    parser.add_argument(
        "--final-only-source-batch",
        default="",
        help="Existing batch directory whose retrieval/single/group artifacts should be reused for final-only reruns.",
    )
    parser.add_argument(
        "--final-only-groups",
        nargs="*",
        default=None,
        help=(
            "Optional exact group ids retained from --final-only-source-batch before final synthesis. "
            "This filters copied retrieval and group outputs without rerunning earlier reasoning stages."
        ),
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--indices", nargs="*", default=None, help="Indices or inclusive ranges, e.g. 0 3 5-8.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all records from --start.")
    parser.add_argument("--parallelism", type=int, default=1, help="Number of molecules to run concurrently.")
    parser.add_argument(
        "--max-stage-requeues",
        type=int,
        default=0,
        help="Immediate retries for a failed single, group, or final prompt stage.",
    )
    parser.add_argument("--save-trace", dest="save_trace", action="store_true", default=True)
    parser.add_argument("--no-save-trace", dest="save_trace", action="store_false")
    parser.add_argument("--combine-traces", dest="combine_traces", action="store_true", default=True)
    parser.add_argument("--no-combine-traces", dest="combine_traces", action="store_false")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--reuse-existing-rerank-preflight",
        action="store_true",
        help="Reuse a complete same-size assay-transfer audit from this batch's existing manifest.",
    )
    parser.add_argument(
        "--rerank-preflight-source-batch",
        default="",
        help=(
            "Reuse a validated assay-transfer cache preflight from a matched source "
            "batch instead of repeating the deterministic full-query scan."
        ),
    )
    parser.add_argument("--stream-logs", dest="stream_logs", action="store_true", default=True)
    parser.add_argument("--no-stream-logs", dest="stream_logs", action="store_false")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--api-key-env", default=config.default_api_key_env)
    parser.add_argument("--base-url", default=config.default_base_url)
    parser.add_argument("--tool-service-url", default="http://127.0.0.1:8765")
    parser.add_argument("--model", default=config.default_model)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=20480)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tool-rounds", type=int, default=10)
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        help="OpenAI-compatible reasoning_effort value. Use an empty string to omit this parameter.",
    )
    parser.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", default=True)
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--top-k-per-group", type=int, default=3)
    parser.add_argument(
        "--enable-assay-transfer-scores",
        action="store_true",
        help="Bioavailability-only variable-k assay-transfer score visibility policy.",
    )
    parser.add_argument("--min-similarity", type=float, default=0.3)
    add_retrieval_strategy_args(parser)
    parser.add_argument(
        "--assay-transfer-min-score",
        type=float,
        default=None,
        help="Optional inclusive cached transfer-probability floor applied before final top-k.",
    )
    if config.supports_retrieval_strategy:
        parser.add_argument(
            "--assay-transfer-diversity-mode",
            choices=ASSAY_TRANSFER_DIVERSITY_MODES,
            default=ASSAY_TRANSFER_DIVERSITY_NONE,
        )
        parser.add_argument(
            "--assay-transfer-diversity-score-slack",
            type=float,
            default=0.0,
        )
    else:
        parser.set_defaults(
            assay_transfer_diversity_mode=ASSAY_TRANSFER_DIVERSITY_NONE,
            assay_transfer_diversity_score_slack=0.0,
        )
    parser.add_argument(
        "--assay-transfer-profile",
        choices=["legacy_bio", "v11_with_categorical"],
        default=config.assay_transfer_profile_default,
    )
    parser.add_argument(
        "--assay-transfer-selection-unit",
        choices=ASSAY_TRANSFER_SELECTION_UNITS,
        default=ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    )
    parser.add_argument(
        "--assay-transfer-records-per-molecule",
        type=int,
        default=ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
        help=(
            "Maximum endpoint-distinct assay records shown inside each selected "
            f"unique molecule (1-{ASSAY_TRANSFER_RECORDS_PER_MOLECULE_MAX}). "
            "Values above 1 require unique_molecule selection."
        ),
    )
    parser.add_argument(
        "--rerank-catalog",
        default=config.rerank_catalog_default,
    )
    parser.add_argument(
        "--rerank-cache",
        default=config.rerank_cache_default,
    )
    parser.add_argument(
        "--rerank-candidate-manifest",
        default=config.rerank_candidate_manifest_default,
        help="Exact flat_v2 condition manifest; required with a flat assay-transfer catalog.",
    )
    parser.add_argument("--rerank-cache-mode", choices=["read_only", "read_write"], default="read_only")
    parser.add_argument(
        "--assay-transfer-model", default=config.assay_transfer_model_default
    )
    parser.add_argument(
        "--assay-transfer-model-revision",
        default=config.assay_transfer_model_revision_default,
    )
    parser.add_argument(
        "--assay-transfer-template-profile",
        choices=[
            "legacy_v3",
            "v6_5_query_context_copy",
            "v6_5_query_context_copy_no_extra_details",
            "v11_query_context_copy",
        ],
        default=config.assay_transfer_template_profile_default,
    )
    parser.add_argument(
        "--group-prompt-format",
        choices=list(config.group_prompt_formats) or None,
        default=config.default_group_prompt_format,
    )
    parser.add_argument("--group-prompt-min-similarity", type=float, default=None)
    parser.add_argument(
        "--presentation-style",
        choices=["legacy", "full"],
        default="legacy",
        help="Record presentation style for text group prompts: legacy or per-source full.",
    )
    parser.add_argument(
        "--group-output-schema",
        choices=list(config.group_output_schemas) or None,
        default=config.default_group_output_schema,
    )
    if config.group_prompt_versions:
        parser.add_argument(
            "--group-prompt-version",
            choices=config.group_prompt_versions,
            default=config.default_group_prompt_version,
        )
    else:
        parser.set_defaults(group_prompt_version="")
    parser.add_argument(
        "--group-prompt-instructions-file",
        default="",
        help="Optional UTF-8 instruction file for a non-legacy text group prompt format.",
    )
    parser.add_argument(
        "--rerank-expected-score-count",
        type=int,
        default=0,
        help="Require both prompt demand and SQLite row count to equal this value during preflight.",
    )
    parser.add_argument(
        "--rerank-cache-version-manifest",
        default=config.rerank_version_manifest_default,
        help="Optional VERSION.json whose model, template, catalog, manifest, and count must match preflight.",
    )
    parser.add_argument(
        "--neighbor-context-profile",
        choices=NEIGHBOR_CONTEXT_PROFILES,
        default=STANDARD_NEIGHBOR_CONTEXT,
    )
    add_final_evidence_surface_argument(parser)
    if config.supports_final_decision_profiles:
        add_final_decision_profile_argument(
            parser,
            choices=config.final_decision_profile_choices,
        )
    else:
        parser.set_defaults(final_decision_profile=STANDARD_FINAL_DECISION)
    if config.prompt_profile_option:
        parser.add_argument(
            config.prompt_profile_option,
            dest="task_prompt_profile",
            choices=config.prompt_profile_choices,
            default=config.default_prompt_profile,
        )
    else:
        parser.set_defaults(task_prompt_profile="")
    parser.add_argument("--groups", nargs="*", default=None, help="Optional exact Tier.endpoint_group ids to reason over.")
    parser.add_argument(
        "--tier1-replacement-index",
        default="",
        help="Optional pipeline-specific neighbor index used to replace base Tier 1 retrieval groups.",
    )
    parser.add_argument(
        "--tier1-replacement-groups",
        nargs="*",
        default=None,
        help="Optional group ids to retrieve from --tier1-replacement-index.",
    )
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--disable-group-tools", action="store_true")
    parser.add_argument(
        "--disable-flat-tools",
        action="store_true",
        help=(
            "full_flat only: omit neighbor comparison tools from the flat group branch; "
            "the single-molecule branch still receives molecule_properties"
        ),
    )
    parser.add_argument("--harness-prefetch-tools", action="store_true")
    args = parser.parse_args(argv)
    if args.assay_transfer_profile == "v11_with_categorical":
        custom_v11_cache = args.rerank_cache != config.rerank_cache_default
        if args.assay_transfer_initial_morgan_filter == 100:
            args.assay_transfer_initial_morgan_filter = 50
        if args.min_similarity == 0.3:
            args.min_similarity = 0.0
        if not custom_v11_cache and args.rerank_catalog == config.rerank_catalog_default:
            args.rerank_catalog = config.v11_rerank_catalog_default
        if args.rerank_cache == config.rerank_cache_default:
            args.rerank_cache = config.v11_rerank_cache_default
        if (
            not custom_v11_cache
            and args.rerank_candidate_manifest == config.rerank_candidate_manifest_default
        ):
            args.rerank_candidate_manifest = config.v11_rerank_candidate_manifest_default
        if args.rerank_cache_version_manifest == config.rerank_version_manifest_default:
            args.rerank_cache_version_manifest = config.v11_rerank_version_manifest_default
        if args.assay_transfer_model == config.assay_transfer_model_default:
            args.assay_transfer_model = config.v11_assay_transfer_model_default
        if args.assay_transfer_model_revision == config.assay_transfer_model_revision_default:
            args.assay_transfer_model_revision = (
                config.v11_assay_transfer_model_revision_default
            )
        if args.assay_transfer_template_profile == config.assay_transfer_template_profile_default:
            args.assay_transfer_template_profile = "v11_query_context_copy"
        if (
            args.retrieval_strategy == ASSAY_TRANSFER_TOOL_STRATEGY
            and args.index == config.default_index
            and config.v11_index_default
        ):
            args.index = config.v11_index_default
    # Keep the branch's retrieval-strategy attribute and main's scheduler/runtime
    # attribute synchronized. Both CLI spellings are aliases in retrieval_args.py.
    args.neighbor_selector = args.morgan_neighbor_selector
    args.group_prompt_instructions_sha256 = ""
    args.groups = _normalize_group_args(args.groups)
    args.tier1_replacement_groups = _normalize_group_args(args.tier1_replacement_groups)
    args.final_only_groups = _normalize_group_args(args.final_only_groups)
    if args.exclude_nondirect_bioavailability_records and (
        not args.retrieval_source.startswith("starling")
        or args.experiment_mode not in {"direct", "full_flat", "full_mechanism"}
    ):
        parser.error(
            "--exclude-nondirect-bioavailability-records requires a Starling "
            "direct, full_flat, or full_mechanism retrieval run"
        )
    if args.final_only_groups and not args.final_only_source_batch:
        parser.error("--final-only-groups requires --final-only-source-batch")
    if args.max_stage_requeues < 0:
        parser.error("--max-stage-requeues must be non-negative")
    if args.disable_flat_tools and args.experiment_mode != "full_flat":
        parser.error("--disable-flat-tools requires --experiment-mode full_flat")
    if not config.supports_shared_retrieval_contract:
        unsupported = []
        if args.experiment_mode != "native":
            unsupported.append("--experiment-mode")
        if args.retrieval_source != "chembl":
            unsupported.append("--retrieval-source")
        if args.neighbor_identity_policy != "operational":
            unsupported.append("--neighbor-identity-policy")
        if args.neighbor_selector != SIMILARITY_SELECTOR:
            unsupported.append("--neighbor-selector")
        if args.neighbor_context_profile != STANDARD_NEIGHBOR_CONTEXT:
            unsupported.append("--neighbor-context-profile")
        if getattr(args, "final_evidence_surface", SUMMARY_ONLY) != SUMMARY_ONLY:
            unsupported.append("--final-evidence-surface")
        if args.identity_blind:
            unsupported.append("--identity-blind")
        if args.harness_prefetch_tools:
            unsupported.append("--harness-prefetch-tools")
        if any(
            (
                args.single_analysis_source_batch,
                args.group_analysis_source_batch,
                args.retrieval_replay_source_batch,
                args.prefetched_tool_replay_source_batch,
                args.tier1_replacement_index,
            )
        ):
            unsupported.append("branch/retrieval reuse")
        if unsupported:
            parser.error(
                f"{config.pipeline_module} does not support: "
                + ", ".join(unsupported)
            )
    return args


def _validate_analogous_reasoning_only(
    config: BatchConfig,
    args: argparse.Namespace,
) -> None:
    if not args.analogous_reasoning_only:
        return
    if not config.supports_analogous_reasoning_only:
        raise SystemExit(
            f"Pipeline {config.pipeline_module} does not support --analogous-reasoning-only"
        )
    if args.experiment_mode not in config.analogous_reasoning_modes:
        raise SystemExit(
            "--analogous-reasoning-only requires --experiment-mode "
            + " or ".join(config.analogous_reasoning_modes)
        )
    if args.experiment_mode == "full_flat":
        if not args.identity_blind:
            raise SystemExit(
                "flat --analogous-reasoning-only requires --identity-blind"
            )
        if args.retrieval_strategy not in {
            ASSAY_TRANSFER_TOOL_STRATEGY,
            "morgan_fingerprint",
        }:
            raise SystemExit(
                "flat --analogous-reasoning-only requires --retrieval-strategy "
                "assay_transfer_tool or morgan_fingerprint"
            )
        if (
            args.retrieval_strategy == ASSAY_TRANSFER_TOOL_STRATEGY
            and not args.enable_assay_transfer_scores
        ):
            raise SystemExit(
                "flat --analogous-reasoning-only requires "
                "--enable-assay-transfer-scores for assay_transfer_tool retrieval"
            )
    incompatible = {
        "--single-analysis-source-batch": args.single_analysis_source_batch,
        "--group-analysis-source-batch": args.group_analysis_source_batch,
        "--prefetched-tool-replay-source-batch": args.prefetched_tool_replay_source_batch,
        "--final-only-source-batch": args.final_only_source_batch,
    }
    used = [name for name, value in incompatible.items() if value]
    if used:
        raise SystemExit(
            "--analogous-reasoning-only requires fresh reasoning and cannot use: "
            + ", ".join(used)
        )
    if args.neighbor_context_profile == COVERAGE_MMP_LEDGER_NEIGHBOR_CONTEXT:
        raise SystemExit(
            "--analogous-reasoning-only cannot use --neighbor-context-profile "
            "coverage_mmp_ledger because it invokes query comparison tools"
        )
    if getattr(args, "final_evidence_surface", SUMMARY_ONLY) != SUMMARY_ONLY:
        raise SystemExit(
            "--analogous-reasoning-only requires --final-evidence-surface summary_only"
        )
    if getattr(args, "final_decision_profile", STANDARD_FINAL_DECISION) != STANDARD_FINAL_DECISION:
        raise SystemExit(
            "--analogous-reasoning-only requires --final-decision-profile standard"
        )


def _is_analogous_flat(args: argparse.Namespace) -> bool:
    return bool(args.analogous_reasoning_only) and args.experiment_mode == "full_flat"


def _validate_assay_transfer_scores(config: BatchConfig, args: argparse.Namespace) -> None:
    is_assay_transfer = args.retrieval_strategy == ASSAY_TRANSFER_TOOL_STRATEGY
    if args.assay_transfer_profile == "v11_with_categorical":
        required = {
            "rerank cache": args.rerank_cache,
            "model": args.assay_transfer_model,
            "model revision": args.assay_transfer_model_revision,
        }
        missing = [name for name, value in required.items() if not str(value or "").strip()]
        if missing:
            raise SystemExit(
                "V11 assay-transfer defaults are not configured for this task: "
                + ", ".join(missing)
            )
        if args.assay_transfer_template_profile != "v11_query_context_copy":
            raise SystemExit(
                "--assay-transfer-profile v11_with_categorical requires "
                "--assay-transfer-template-profile v11_query_context_copy"
            )
    elif args.assay_transfer_template_profile == "v11_query_context_copy":
        raise SystemExit(
            "--assay-transfer-template-profile v11_query_context_copy requires "
            "--assay-transfer-profile v11_with_categorical"
        )
    # --retrieval-strategy is the source of truth; it locks the compatible group-prompt-format.
    if is_assay_transfer:
        if args.group_prompt_format != "assay_transfer_tool":
            raise SystemExit(
                "--retrieval-strategy assay_transfer_tool requires "
                "--group-prompt-format assay_transfer_tool"
            )
        if args.morgan_neighbor_selector != SIMILARITY_SELECTOR:
            raise SystemExit(
                "--morgan-neighbor-selector applies only to "
                "--retrieval-strategy morgan_fingerprint"
            )
    else:  # morgan_fingerprint
        if args.group_prompt_format == "assay_transfer_tool":
            raise SystemExit(
                "--group-prompt-format assay_transfer_tool requires "
                "--retrieval-strategy assay_transfer_tool"
            )
        if args.assay_transfer_min_score is not None:
            raise SystemExit(
                "--assay-transfer-min-score requires --retrieval-strategy assay_transfer_tool"
            )
        if args.assay_transfer_diversity_mode != ASSAY_TRANSFER_DIVERSITY_NONE:
            raise SystemExit(
                "--assay-transfer-diversity-mode requires --retrieval-strategy assay_transfer_tool"
            )
    if args.assay_transfer_min_score is not None and not 0.0 <= args.assay_transfer_min_score <= 1.0:
        raise SystemExit("--assay-transfer-min-score must be between 0 and 1 inclusive")
    try:
        validate_assay_transfer_diversity(
            mode=args.assay_transfer_diversity_mode,
            score_slack=args.assay_transfer_diversity_score_slack,
        )
        validate_assay_transfer_records_per_molecule(
            args.assay_transfer_records_per_molecule,
            selection_unit=args.assay_transfer_selection_unit,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if (
        args.assay_transfer_records_per_molecule
        > ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT
        and not is_assay_transfer
    ):
        raise SystemExit(
            "--assay-transfer-records-per-molecule greater than 1 requires "
            "--retrieval-strategy assay_transfer_tool"
        )
    if args.group_prompt_format and not config.group_prompt_formats:
        raise SystemExit(f"Pipeline {config.pipeline_module} does not support --group-prompt-format")
    if args.group_output_schema and not config.group_output_schemas:
        raise SystemExit(
            f"Pipeline {config.pipeline_module} does not support --group-output-schema"
        )
    if (
        args.group_output_schema == "assay-transfer"
        and args.group_prompt_format != "assay_transfer_tool"
    ):
        raise SystemExit(
            "--group-output-schema assay-transfer requires "
            "--group-prompt-format assay_transfer_tool"
        )
    if args.group_prompt_instructions_file and args.group_prompt_format == "legacy":
        raise SystemExit(
            "--group-prompt-instructions-file requires a non-legacy text group prompt format"
        )
    if (
        args.group_prompt_instructions_file
        and args.group_prompt_version
        and args.group_prompt_version != "legacy_unversioned"
    ):
        raise SystemExit(
            "A named --group-prompt-version is immutable and cannot be combined with "
            "--group-prompt-instructions-file"
        )
    if not args.enable_assay_transfer_scores:
        return
    if not config.supports_assay_transfer_scores:
        raise SystemExit(f"Pipeline {config.pipeline_module} does not support --enable-assay-transfer-scores")
    if args.experiment_mode not in {"full_flat", "full_mechanism"}:
        raise SystemExit(
            "--enable-assay-transfer-scores requires --experiment-mode "
            "full_flat or full_mechanism"
        )
    if args.retrieval_source not in {"starling", "starling_in_distribution"}:
        raise SystemExit(
            "--enable-assay-transfer-scores requires --retrieval-source "
            "starling or starling_in_distribution"
        )
    if not is_assay_transfer:
        raise SystemExit(
            "--enable-assay-transfer-scores requires --retrieval-strategy assay_transfer_tool"
        )


def _validate_prompt_profile_contract(
    config: BatchConfig,
    args: argparse.Namespace,
    batch_dir: Path,
) -> None:
    """Prevent prompt-profile changes from silently reusing old branches."""
    if not config.prompt_profile_option:
        return
    target = str(args.task_prompt_profile)
    existing_manifest = batch_dir / "manifest.json"
    if existing_manifest.exists():
        existing = prompt_profile_from_manifest(
            json.loads(existing_manifest.read_text(encoding="utf-8")),
            historical_profile=config.historical_prompt_profile,
        )
        if existing != target:
            raise ValueError(
                f"Batch prompt profile mismatch: existing={existing!r} target={target!r} "
                f"batch={batch_dir}"
            )
    require_matching_prompt_profiles(
        target_profile=target,
        source_dirs=(
            args.single_analysis_source_batch,
            args.group_analysis_source_batch,
            args.final_only_source_batch,
        ),
        historical_profile=config.historical_prompt_profile,
    )


def _validate_group_prompt_contract(
    batch_dir: Path,
    target: dict[str, Any],
) -> None:
    """Prevent a named text prompt from sharing a batch with another prompt hash."""
    if not target:
        return
    manifest_path = batch_dir / "manifest.json"
    if not manifest_path.exists():
        return
    existing = _read_json(manifest_path).get("group_prompt_provenance") or {}
    named_target = target.get("prompt_version") != "legacy_unversioned"
    if (named_target and existing != target) or (existing and existing != target):
        raise ValueError(
            "Batch group-prompt provenance mismatch: "
            f"existing={existing.get('prompt_version')!r}/"
            f"{existing.get('template_sha256')!r}, target="
            f"{target.get('prompt_version')!r}/{target.get('template_sha256')!r}, "
            f"batch={batch_dir}"
        )


def _validate_final_decision_profile_contract(
    args: argparse.Namespace,
    batch_dir: Path,
) -> None:
    """Keep the final-only prior opt-in and prevent mixed batch lineages."""
    target = str(
        getattr(args, "final_decision_profile", STANDARD_FINAL_DECISION)
    )
    if target != STANDARD_FINAL_DECISION and not args.final_only_source_batch:
        raise ValueError(
            f"{target} is a final-only experiment and requires "
            "--final-only-source-batch"
        )
    existing_manifest = batch_dir / "manifest.json"
    if not existing_manifest.exists():
        return
    existing = str(
        json.loads(existing_manifest.read_text(encoding="utf-8")).get(
            "final_decision_profile"
        )
        or STANDARD_FINAL_DECISION
    )
    if existing != target:
        raise ValueError(
            "Batch final-decision profile mismatch: "
            f"existing={existing!r} target={target!r} batch={batch_dir}"
        )


def _normalize_group_args(groups: list[str] | None) -> list[str] | None:
    if not groups:
        return groups
    normalized: list[str] = []
    i = 0
    while i < len(groups):
        group = groups[i]
        if group in {"Tier", "Starling", "Combined"} and i + 1 < len(groups) and "." in groups[i + 1]:
            normalized.append(f"{group} {groups[i + 1]}")
            i += 2
        else:
            normalized.append(group)
            i += 1
    return normalized


def _group_prompt_instruction_provenance(path_value: str) -> dict[str, Any]:
    if not path_value:
        return {}
    path = Path(path_value)
    try:
        resolved = path.expanduser().resolve(strict=True)
        raw = resolved.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise SystemExit(
            f"Cannot read group prompt instructions file {path}: {exc}"
        ) from exc
    instructions = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not instructions:
        raise SystemExit(
            f"Group prompt instructions file has no instruction lines: {resolved}"
        )
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "instruction_count": len(instructions),
    }
