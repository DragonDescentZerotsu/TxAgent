"""Run either progressive evidence profile through one resumable LLM loop.

``standard`` progressively reveals mechanism-family molecule cards and their
analog-comparison tools. ``context_records`` selects contexts and later records
with either frozen assay-transfer ranks or Morgan similarity; it never invokes
per-analog tools. Both profiles keep
earlier evidence visible, carry forward a structured reasoning state, and
checkpoint every query level.

The historical cumulative-family and geometric assay-prefix launchers remain
unchanged.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
from functools import lru_cache
from typing import Any, Mapping

import pyarrow.parquet as pq

from predict.harnesses.progressive.retrieval import (
    ASSAY_MECHANISM_TAGGED_PROMPT_PROFILE,
    build_family_molecule_prefix_view,
    retrieve_family_molecule_prefixes,
)
from predict.harnesses.progressive.context_records import profile as context_records
from predict.retrieval.retrieve import load_index
from predict.retrieval.assay_reranking.runtime import CACHE_ROOT
from predict.retrieval.assay_reranking.progressive_levels import (
    LUNA_RELEVANCE_CACHE_PROFILE,
    V21_CACHE_PROFILE,
    load_top_ranked_records,
    load_v21_molecule_card_records,
)
from predict.retrieval.assay_reranking.v9 import (
    RANKING_PROFILE_NAME,
    RANKING_SCHEMA_VERSION,
    model_profile as v9_model_profile,
    verify_vendored_assets as verify_v9_assets,
)
from predict.utils.json import (
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from predict.llm_engine.client import OpenAICompatibleClient
from predict.llm_engine.pool import (
    OpenAIProviderPool,
    ProviderPoolConfig,
    ProviderPoolExhausted,
    ProviderSpec,
    load_env_file,
    load_provider_pool_config,
)
from predict.harnesses.progressive.state import (
    MOLECULE_CARD_CONTRACT_PATH,
    PROGRESSIVE_PROTOCOL_VERSION,
    ProgressiveTaskContract,
    append_evidence,
    attach_analog_tool_summaries,
    build_progressive_messages,
    card_alias_maps,
    card_ids,
    extract_cumulative_evidence,
    molecule_card_contract,
    progressive_state_errors,
    restore_card_ids,
    select_initial_evidence,
    select_progressive_delta,
    stable_analog_id,
    state_from_content,
)
from predict.tools.client import ToolServiceClient
from predict.tools.prefetch import invoke_with_retry
from predict.traces.io import DEFAULT_TRACE_ROOT, write_trace
from predict.llm_io.query import external_condition_sentence
from predict.llm_io.response import (
    call_with_json_validation,
    structured_response_is_valid,
)
from data.processing.gold_labels.conditioned_benchmark import split_path
from predict.harnesses.progressive.tasks import bbb_martins as bbb_config
from predict.harnesses.progressive.tasks import bioavailability_ma as bio_config
from predict.harnesses.progressive.tasks import skin_reaction as skin_config


MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
BASE_URL = "http://epyc-3-6:50000/v1"
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/paper/starling_conditioned_assay_progressive_visible_v10_bio_legacy_gold_vote_pure_v1/"
    "scaffold_valid_deepseek_v4_flash_0731"
)
DEFAULT_SINGLE_CACHE_ROOT = Path(
    "outputs/paper/starling_conditioned_gold_l1_deepseek_v4_flash_nvfp4_query_prior/"
    "runs_deployment_visible_parent_disjoint"
)
DEFAULT_V9_RANKING_ROOT = CACHE_ROOT / RANKING_PROFILE_NAME
CONTEXT_L2_REUSE_ROOTS = {
    "bbb_martins": Path(
        "outputs/paper/assay_transfer_harness/starling_context_record_progressive_v4/"
        "scaffold_valid_deepseek_v4_flash_0731_epyc36_with_prior_cap10"
    ),
    "bioavailability_ma": Path(
        "outputs/paper/assay_transfer_harness/starling_context_record_progressive_v3/"
        "scaffold_valid_deepseek_v4_flash_0731_epyc36_with_prior_cap10"
    ),
    "skin_reaction": Path(
        "outputs/paper/assay_transfer_harness/starling_context_record_progressive_v6/"
        "scaffold_valid_deepseek_v4_flash_0731_epyc36_with_prior_cap10"
    ),
}
TASK_NAMES = ("bbb_martins", "bioavailability_ma", "skin_reaction")
REFERENCE_POOL = "direct_only_heldout_filtered"
IDENTITY_POLICY = "scaffold_disjoint"
MAX_ENDPOINT_CONCURRENCY_BUDGET = 512
COMPLETE_LEVEL_STATUSES = {"ok", "carried_forward", "reused_none", "reused"}

_MODEL_IDENTITY_ALIASES = {
    "deepseek-ai/deepseek-v4-flash-0731": "deepseek-v4-flash-0731",
    "deepseek/deepseek-v4-flash-0731": "deepseek-v4-flash-0731",
    "deepseek/deepseek-v4-flash": "deepseek-v4-flash-0731",
    "deepseek-v4-flash": "deepseek-v4-flash-0731",
    "nvidia/deepseek-v4-flash-nvfp4": "deepseek-v4-flash-0731",
}


def _model_identity(model: str) -> str:
    normalized = str(model or "").strip().lower()
    return _MODEL_IDENTITY_ALIASES.get(normalized, normalized)


_RESUME_INVARIANT_FIELDS = (
    "experiment",
    "profile",
    "tasks",
    "visibility_mode",
    "reference_pool",
    "neighbor_identity_policy",
    "selection",
    "candidate_generation",
    "prompt_profile",
    "prompt_template",
    "molecule_card_contract",
    "max_tokens",
    "temperature",
    "thinking",
    "reasoning_effort",
    "tool_prefetch_complete",
    "evaluation_indices_by_task",
    "inputs",
    "l2_reuse",
)


def _execution_provider(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": str(manifest.get("model") or ""),
        "model_identity": _model_identity(str(manifest.get("model") or "")),
        "base_url": str(manifest.get("base_url") or ""),
        "transport_max_retries": int(manifest.get("transport_max_retries") or 0),
    }


def _execution_provider_from_spec(
    spec: ProviderSpec,
    *,
    transport_max_retries: int,
) -> dict[str, Any]:
    return {
        "name": spec.name,
        "model": spec.model,
        "model_identity": _model_identity(spec.model),
        "base_url": spec.base_url,
        "api_key_env": spec.api_key_env,
        "max_inflight": spec.max_inflight,
        "timeout_s": spec.timeout_s,
        "transport_max_retries": transport_max_retries,
    }


def _merge_resume_manifest(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    for field in _RESUME_INVARIANT_FIELDS:
        if previous.get(field) != current.get(field):
            raise ValueError(f"cannot resume progressive run with changed {field}")
    if _model_identity(str(previous.get("model") or "")) != _model_identity(
        str(current.get("model") or "")
    ):
        raise ValueError("cannot resume progressive run with changed model identity")

    merged = dict(current)
    for field in ("model", "base_url", "transport_max_retries", "started_at"):
        if field in previous:
            merged[field] = previous[field]
    merged["model_identity"] = _model_identity(str(merged.get("model") or ""))
    providers = list(previous.get("execution_providers") or [])
    if not providers:
        providers.append(_execution_provider(previous))
    current_providers = list(current.get("execution_providers") or [])
    if not current_providers:
        current_providers.append(_execution_provider(current))
    for current_provider in current_providers:
        if current_provider not in providers:
            providers.append(current_provider)
    merged["execution_providers"] = providers
    merged["last_resumed_at"] = _now()
    return merged


TASK_CONFIGS = {
    "bbb_martins": bbb_config,
    "bioavailability_ma": bio_config,
    "skin_reaction": skin_config,
}

SOURCE_PURITY_ROOT = Path(
    "outputs/paper/starling_conditioned_assay_family_curve_v1"
)


@dataclass(frozen=True)
class ProgressiveTaskSpec:
    input_jsonl: Path
    index: Path
    family_manifest: Path


PROGRESSIVE_TASKS = {
    "bbb_martins": ProgressiveTaskSpec(
        split_path("bbb_martins", "valid"),
        SOURCE_PURITY_ROOT
        / "indices/bbb_martins/"
        "mechanism_tagged_v4_source_purity_v5/assay_neighbor_index.pkl",
        SOURCE_PURITY_ROOT
        / "family_catalogs_mechanism_tagged_v1/"
        "bbb_martins_source_purity_v5/manifest.json",
    ),
    "bioavailability_ma": ProgressiveTaskSpec(
        split_path("bioavailability_ma", "valid"),
        SOURCE_PURITY_ROOT
        / "indices/bioavailability_ma/"
        "mechanism_tagged_v4_legacy_record_supported_v2_vote_pure_v1/assay_neighbor_index.pkl",
        SOURCE_PURITY_ROOT
        / "family_catalogs/"
        "bioavailability_ma_legacy_record_supported_v2_vote_pure_v1/manifest.json",
    ),
    "skin_reaction": ProgressiveTaskSpec(
        split_path("skin_reaction", "valid"),
        SOURCE_PURITY_ROOT
        / "indices/skin_reaction/"
        "mechanism_tagged_v4_source_purity_v1/assay_neighbor_index.pkl",
        SOURCE_PURITY_ROOT
        / "family_catalogs_mechanism_tagged_v1/"
        "skin_reaction_source_purity_v1/manifest.json",
    ),
}

V7_PROGRESSIVE_GROUPS = {
    "bbb_martins": (
        ("retrieval_source_id=direct_vote", "direct_brain_exposure"),
        ("retrieval_source_id=direct_residual", "direct_residual"),
        ("Mechanism.passive_permeability", "passive_permeability"),
        ("Mechanism.efflux_transport", "efflux_transport"),
        ("Mechanism.influx_transport", "influx_transport"),
    ),
    "bioavailability_ma": (
        ("Observed.direct_oral_bioavailability", "direct_oral_bioavailability"),
        ("Observed.nondirect_oral_bioavailability", "nondirect_oral_bioavailability"),
        ("Observed.oral_auc_cmax_exposure", "oral_auc_cmax_exposure"),
        ("Fa.absorption_solubility_permeability", "fa"),
        ("Fg.gut_wall_efflux_intestinal_metabolism", "fg"),
        ("Fh.hepatic_clearance_metabolic_stability", "fh"),
    ),
    "skin_reaction": (
        ("Direct.skin_reaction", "direct_skin_sensitization"),
        ("Mechanism.sensitization_aop", "sensitisation_aop"),
    ),
}

BBB_V7_LEVEL_DESCRIPTIONS = {
    1: "Gold-swapped direct benchmark votes; closest to the benchmark label.",
    2: "Additional direct-source residual evidence not represented by a gold vote.",
    3: "Passive permeability evidence; indirect and conditional on transferability.",
    4: "Efflux-transporter evidence with directional claims kept distinct.",
    5: "Influx or uptake-transporter evidence dependent on transporter context.",
}

TASK_DATA_DIRECTORIES = {
    "bbb_martins": "BBB_Martins",
    "bioavailability_ma": "Bioavailability_Ma",
    "skin_reaction": "Skin_Reaction",
}


def _benchmark_scaffold_root(root: Path, task: str) -> Path:
    task_root = root / TASK_DATA_DIRECTORIES[task]
    current = task_root / "CURRENT"
    if current.is_file():
        task_root /= current.read_text(encoding="utf-8").strip()
    return task_root / "scaffold"


@dataclass(frozen=True)
class PreparedQuery:
    task: str
    index: int
    query_dir: Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _levels(task: str, max_level: int = 0) -> list[dict[str, Any]]:
    manifest = _read_json(PROGRESSIVE_TASKS[task].family_manifest)
    levels = [dict(row) for row in manifest.get("levels") or []]
    if not levels and task in V7_PROGRESSIVE_GROUPS:
        levels = [
            {"level": level, "source_group_id": group_id, "endpoint_group": endpoint}
            for level, (group_id, endpoint) in enumerate(
                V7_PROGRESSIVE_GROUPS[task], start=1
            )
        ]
    for row in levels:
        level = int(row["level"])
        endpoint = str(row.get("endpoint_group") or "")
        endpoint_descriptions = getattr(
            TASK_CONFIGS[task], "PROGRESSIVE_ASSAY_ENDPOINT_DESCRIPTIONS", {}
        )
        row["description"] = (
            BBB_V7_LEVEL_DESCRIPTIONS.get(level)
            if task == "bbb_martins" and PROGRESSIVE_TASKS[task].index.is_dir()
            else endpoint_descriptions.get(endpoint)
            or TASK_CONFIGS[task].PROGRESSIVE_ASSAY_LEVEL_DESCRIPTIONS[level]
        )
    if [int(row["level"]) for row in levels] != list(range(1, len(levels) + 1)):
        raise ValueError(f"{task} has a non-contiguous family-level catalog")
    return levels[:max_level] if max_level else levels


def _run_levels(args: argparse.Namespace, task: str) -> list[dict[str, Any]]:
    """Dispatch only the level catalog; each profile owns its level meaning."""
    if args.profile == "context_records":
        return context_records.levels(
            args.max_level,
            task=task,
            include_indirect=_record_cache_enabled(args),
        )
    return _levels(task, args.max_level)


def _record_cache_enabled(args: argparse.Namespace) -> bool:
    return (
        args.context_record_l3_l5
        or args.assay_transfer_cache_profile == V21_CACHE_PROFILE
    )


def _record_level_names(args: argparse.Namespace, task: str) -> tuple[str, ...]:
    if (
        args.assay_transfer_cache_profile == V21_CACHE_PROFILE
        and args.v21_selection_mode == "record_only"
    ):
        return context_records.record_level_names(task, first_level=1)
    return context_records.indirect_level_names(task)


def _indirect_record_limits(
    args: argparse.Namespace, task: str
) -> dict[str, int]:
    levels = _record_level_names(args, task)
    limits = {level: args.indirect_record_limit_per_level for level in levels}
    if (
        args.assay_transfer_cache_profile == V21_CACHE_PROFILE
        and args.v21_selection_mode == "record_only"
    ):
        limits["L1"] = args.record_limit_per_context_level
        limits["L2"] = args.l2_record_limit_per_context
    if args.indirect_final_level_record_limit:
        limits[levels[-1]] = args.indirect_final_level_record_limit
    elif (
        args.assay_transfer_cache_profile == LUNA_RELEVANCE_CACHE_PROFILE
        and task == "bbb_martins"
    ):
        limits[levels[-1]] = 25
    return limits


def _run_protocol(args: argparse.Namespace) -> str:
    return (
        (
            context_records.INDIRECT_PROTOCOL_VERSION
            if _record_cache_enabled(args)
            else context_records.PROTOCOL_VERSION
        )
        if args.profile == "context_records"
        else PROGRESSIVE_PROTOCOL_VERSION
    )


def _context_prompt_version(args: argparse.Namespace, task: str) -> str:
    return context_records.resolve_prompt_version(
        task,
        args.assay_transfer_prompt_version,
        ranking=args.context_ranking,
    )


def _validate_context_l2_reuse_source(
    args: argparse.Namespace,
    task: str,
    input_path: Path,
    selected_indices: list[int],
) -> dict[str, Any]:
    try:
        root = CONTEXT_L2_REUSE_ROOTS[task]
    except KeyError as exc:
        raise ValueError(f"no frozen L2 reuse source for {task}") from exc
    manifest_path = root / "experiment_manifest.json"
    manifest = _read_json(manifest_path)
    prompt_version = _context_prompt_version(args, task)
    prompt = context_records.prompt_profile(prompt_version)
    expected = {
        "experiment": context_records.PROTOCOL_VERSION,
        "profile": "context_records",
        "model_identity": _model_identity(args.model),
        "query_prior_mode": "fresh",
        "max_level": 2,
        "n_failed_queries": 0,
        "prompt_profile": prompt["name"],
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"{task} L2 reuse source has wrong {field}: {manifest.get(field)!r}"
            )
    if task not in (manifest.get("tasks") or []):
        raise ValueError(f"{task} is absent from its L2 reuse source")
    if not manifest.get("finished_at"):
        raise ValueError(f"{task} L2 reuse source is unfinished")
    if (manifest.get("selection") or {}).get(
        "record_limit_per_context_level"
    ) != args.record_limit_per_context_level:
        raise ValueError(f"{task} L2 reuse source uses a different record cap")
    task_inputs = (manifest.get("inputs") or {}).get(task) or {}
    if task_inputs.get("input_sha256") != sha256_file(input_path):
        raise ValueError(f"{task} L2 reuse source uses a different valid split")
    if (manifest.get("prompt_template") or {}).get("sha256") != sha256_file(
        prompt["template"]
    ):
        raise ValueError(f"{task} L2 reuse source uses a different prompt template")
    available_indices = set(
        (manifest.get("evaluation_indices_by_task") or {}).get(task) or []
    )
    if not set(selected_indices) <= available_indices:
        raise ValueError(f"{task} L2 reuse source lacks selected query indices")
    return {
        "root": str(root),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "prompt_version": prompt_version,
        "levels": [1, 2],
        "reuse_policy": "semantic_prepared_match_then_copy_frozen_state.v1",
    }


_L2_REUSE_PREPARED_FIELDS = (
    "profile",
    "query_prior_mode",
    "l1_source",
    "l1_ranking",
    "task",
    "query_index",
    "benchmark_row_id",
    "molecule_identity_key",
    "condition_group",
    "reused_single_source_index",
    "query_smiles",
    "condition_sentence",
    "query_prior",
    "query_tool_summary",
    "reused_none_final",
    "level",
    "level_definition",
    "new_card_ids",
    "n_active_cards",
    "should_call_model",
)


def _reuse_context_l1_l2_outputs(
    *,
    task: str,
    query_index: int,
    query_dir: Path,
    source_root: Path,
    prompt_version: str,
    record_limit: int,
    l2_record_limit: int,
) -> None:
    """Copy frozen L1/L2 states only after their visible prompts match."""
    source_query_dir = _query_dir(source_root, task, query_index)
    prior_state: Mapping[str, Any] | None = None
    contract = _task_contract(task)
    for level in (1, 2):
        level_dir = query_dir / "levels" / f"level_{level}"
        source_level_dir = source_query_dir / "levels" / f"level_{level}"
        prepared = _read_json(level_dir / "prepared.json")
        source_prepared = _read_json(source_level_dir / "prepared.json")
        for field in _L2_REUSE_PREPARED_FIELDS:
            if prepared.get(field) != source_prepared.get(field):
                raise ValueError(
                    f"{task} query {query_index} L{level} cannot be reused: "
                    f"prepared {field} differs"
                )
        source_output_path = source_level_dir / "output.json"
        source_request_path = source_level_dir / "request.json"
        source_output = _read_json(source_output_path)
        if source_output.get("status") not in COMPLETE_LEVEL_STATUSES or not isinstance(
            source_output.get("state"), Mapping
        ):
            raise ValueError(
                f"{task} query {query_index} L{level} reuse output is not complete"
            )
        _, alias_to_card_id = card_alias_maps(
            prepared["active_evidence"]
        )
        if source_output.get("status") == "carried_forward":
            expected_state = {
                **(prior_state or {}),
                "level": level,
                "revision_action": "keep",
            }
            if prepared["should_call_model"] or source_output["state"] != expected_state:
                raise ValueError(
                    f"{task} query {query_index} L{level} cannot reuse "
                    "an inconsistent carried-forward state"
                )
        else:
            current_messages = context_records.build_messages(
                contract=contract,
                current_level=level,
                query_smiles=prepared["query_smiles"],
                condition_sentence=prepared["condition_sentence"],
                query_prior=prepared["query_prior"] or None,
                query_tool_summary=prepared.get("query_tool_summary") or {},
                active=prepared["active_evidence"],
                prior_state=prior_state,
                prompt_version=prompt_version,
                record_limit=record_limit,
                l2_record_limit=l2_record_limit,
                include_indirect=False,
            )
            source_request = _read_json(source_request_path)
            if (
                source_request.get("messages") != current_messages
                or source_request.get("card_alias_map") != alias_to_card_id
            ):
                raise ValueError(
                    f"{task} query {query_index} L{level} cannot be reused: "
                    "model-visible request differs"
                )
        reused_output = dict(source_output)
        reused_output.update(
            {
                "status": "reused",
                "model_called": False,
                "source_model_called": source_output.get("model_called") is True,
                "reused_from": str(source_output_path),
                "reused_output_sha256": sha256_file(source_output_path),
                "reused_at": _now(),
            }
        )
        write_json_atomic(level_dir / "output.json", reused_output)
        if source_request_path.is_file():
            shutil.copy2(source_request_path, level_dir / "request.json")
        prior_state = source_output["state"]


def _load_progressive_index(task: str) -> dict[str, Any]:
    path = PROGRESSIVE_TASKS[task].index
    if not path.is_dir():
        with path.open("rb") as handle:
            return pickle.load(handle)
    index = load_index(path)
    group_levels = {
        group_id: (level, endpoint)
        for level, (group_id, endpoint) in enumerate(
            V7_PROGRESSIVE_GROUPS[task], start=1
        )
        if not group_id.startswith("retrieval_source_id=")
    }
    for molecule_groups in index.get("evidence_by_molecule_group", {}).values():
        for group_id, rows in molecule_groups.items():
            split_bbb_direct = (
                task == "bbb_martins"
                and group_id == "Tier 1.starling_direct_bbb_evidence"
            )
            if not split_bbb_direct and group_id not in group_levels:
                raise ValueError(f"{task} compact index has unmapped group {group_id!r}")
            for row in rows:
                for example in row.get("source_record_examples") or []:
                    if split_bbb_direct:
                        source_id = str(
                            (example.get("source_contract") or {}).get("source_id") or ""
                        )
                        try:
                            level = {
                                "conditioned_benchmark_gold": 1,
                                "direct_bbb": 2,
                            }[source_id]
                        except KeyError as exc:
                            raise ValueError(
                                f"BBB direct card has unexpected source {source_id!r}"
                            ) from exc
                        endpoint = "direct_brain_exposure" if level == 1 else "direct_residual"
                    else:
                        level, endpoint = group_levels[group_id]
                    example["evidence_family"] = endpoint
                    example["evidence_family_level"] = level
    index["source"] = {
        **dict(index.get("source") or {}),
        "evidence_prompt_profile": ASSAY_MECHANISM_TAGGED_PROMPT_PROFILE,
    }
    return index


def _configure_v7_paths(args: argparse.Namespace) -> None:
    if not args.evidence_root:
        return
    benchmark_root = Path(args.benchmark_data_root)
    evidence_root = Path(args.evidence_root)
    task_directories = {
        "bbb_martins": "BBB_Martins",
        "bioavailability_ma": "Bioavailability_Ma",
        "skin_reaction": "Skin_Reaction",
    }
    index_names = {
        "bbb_martins": "bbb_starling_v7",
        "bioavailability_ma": "bioavailability_starling_v7",
        "skin_reaction": "skin_reaction_starling_v7",
    }
    for task in args.tasks:
        if task not in index_names:
            raise ValueError(f"compact normalized-v7 progressive mode does not support {task}")
        index_dir = evidence_root / index_names[task] / "08_neighbor_index"
        PROGRESSIVE_TASKS[task] = ProgressiveTaskSpec(
            _benchmark_scaffold_root(benchmark_root, task) / "valid.jsonl",
            index_dir,
            index_dir / "manifest.json",
        )


def _task_contract(task: str) -> ProgressiveTaskContract:
    try:
        return TASK_CONFIGS[task].get_progressive_task_contract()
    except KeyError as exc:
        raise ValueError(f"unsupported progressive task: {task}") from exc


def _query_dir(output_root: Path, task: str, query_index: int) -> Path:
    return output_root / task / "queries" / f"query_idx{query_index:05d}"


def _write_level_trace(
    args: argparse.Namespace,
    prepared_query: PreparedQuery,
    level: int,
    output_path: Path,
    output: Mapping[str, Any],
) -> None:
    write_trace(
        trace_root=args.trace_root,
        experiment_id=Path(args.output_root).name,
        task=prepared_query.task,
        harness="progressive",
        sample_id=f"query_idx{prepared_query.index:05d}",
        stage=f"level_{level:02d}",
        checkpoint_path=output_path,
        output=output,
    )


def _single_batch_dir(task: str, single_root: Path) -> Path:
    for name in ("none", f"{task}__none"):
        path = single_root / task / name
        if path.is_dir():
            return path
    return single_root / task / "none"


def _source_run_dir(task: str, query_index: int, single_root: Path) -> Path:
    runs = _single_batch_dir(task, single_root) / "runs"
    current = runs / f"{task}__none_idx{query_index:05d}"
    archived = runs / f"none_idx{query_index:05d}"
    return current if current.is_dir() else archived


def _stable_query_key(record: Mapping[str, Any]) -> tuple[str, str]:
    parent = str(record.get("molecule_identity_key") or "").strip()
    condition = str(record.get("condition_group") or "").strip()
    if not parent:
        raise ValueError("conditioned benchmark row lacks molecule_identity_key")
    return parent, condition


@lru_cache(maxsize=None)
def _single_source_index(task: str, single_root_text: str) -> dict[tuple[str, str], int]:
    single_root = Path(single_root_text)
    manifest = _read_json(_single_batch_dir(task, single_root) / "manifest.json")
    input_path = Path(str(manifest.get("input_jsonl") or ""))
    if input_path.is_file():
        records = read_jsonl(input_path)
    else:
        # The old compatibility path was removed when the active gold releases
        # moved under data/gold_labels. Reuse remains fail-closed by checking
        # every saved run's query SMILES against the current ordered valid set.
        current_path = PROGRESSIVE_TASKS[task].input_jsonl
        records = read_jsonl(current_path)
        for index, record in enumerate(records):
            retrieval_path = _source_run_dir(task, index, single_root) / "retrieval.json"
            if not retrieval_path.is_file():
                raise FileNotFoundError(retrieval_path)
            query = _read_json(retrieval_path).get("query") or {}
            if str(query.get("input_smiles") or "") != str(record.get("drug") or ""):
                raise ValueError(
                    f"reusable single query {index} differs from current valid gold"
                )
    if len(records) != int(manifest.get("n_items") or -1):
        raise ValueError(f"reusable single input count disagrees with manifest: {input_path}")
    mapping: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        key = _stable_query_key(record)
        if key in mapping:
            raise ValueError(f"duplicate reusable single identity: {key}")
        mapping[key] = index
    return mapping


def _gold_l1_candidates(
    *,
    task: str,
    records: list[dict[str, Any]],
    ranking_root: Path,
    ranking: str,
    benchmark_root: Path,
    min_similarity: float,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    task_cache = ranking_root / task / "scaffold" / "valid"
    manifest_path = task_cache / "VERSION.json"
    rankings_path = task_cache / "rankings.parquet"
    cache_manifest = _read_json(manifest_path)
    if cache_manifest.get("schema_version") != RANKING_SCHEMA_VERSION:
        raise ValueError(f"{task} has an incompatible V9 ranking cache")
    if cache_manifest.get("status") != "complete":
        raise ValueError(f"{task} V9 ranking cache is incomplete")
    if cache_manifest.get("rankings_sha256") != sha256_file(rankings_path):
        raise ValueError(f"{task} V9 rankings hash disagrees with manifest")
    if cache_manifest.get("model") != v9_model_profile(task):
        raise ValueError(f"{task} V9 model lineage is not the pinned direct model")
    if cache_manifest.get("prompt_assets") != verify_v9_assets():
        raise ValueError(f"{task} V9 prompt assets differ from the cache build")

    stage6_dir = PROGRESSIVE_TASKS[task].index.parent / "06_records"
    stage6_manifest_path = stage6_dir / "manifest.json"
    stage6_records_path = stage6_dir / "records.parquet"
    stage6_manifest = _read_json(stage6_manifest_path)
    stage6_records_sha256 = sha256_file(stage6_records_path)
    if (stage6_manifest.get("records_file") or {}).get("sha256") != stage6_records_sha256:
        raise ValueError(f"{task} Stage 06 records hash disagrees with manifest")

    data_dir = _benchmark_scaffold_root(benchmark_root, task)
    train_rows = read_jsonl(data_dir / "train_molecule_condition_labels.jsonl")
    train_by_id = {str(row["benchmark_row_id"]): row for row in train_rows}
    train_ids = set(train_by_id)

    ranking_rows = pq.read_table(rankings_path).to_pylist()
    current_by_id = {str(row["benchmark_row_id"]): row for row in records}
    current_ids = set(current_by_id)
    cache_query_ids = {str(row["query_record_id"]) for row in ranking_rows}
    if cache_query_ids != current_ids:
        raise ValueError(
            f"{task} V9 query set differs from current valid: "
            f"missing={len(current_ids - cache_query_ids)}, "
            f"extra={len(cache_query_ids - current_ids)}"
        )
    retrieval_ids = {str(row["retrieval_record_id"]) for row in ranking_rows}
    if set(cache_manifest.get("training_record_ids") or []) != train_ids:
        raise ValueError(
            f"{task} V9 source training set differs from current train"
        )
    if not retrieval_ids <= train_ids:
        raise ValueError(f"{task} V9 cache contains stale retrieval records")

    family = _levels(task, 1)[0]["endpoint_group"]
    candidates: dict[str, dict[str, dict[str, Any]]] = {
        query_id: {} for query_id in current_ids
    }
    for row in ranking_rows:
        query_id = str(row["query_record_id"])
        if query_id not in candidates:
            raise ValueError(f"{task} V9 cache contains an unexpected query: {query_id}")
        current = current_by_id[query_id]
        if (
            str(current.get("molecule_identity_key")) != str(row["query_molecule_identity_key"])
            or str(current.get("condition_group")) != str(row["query_condition_group"])
        ):
            raise ValueError(f"{task} V9 row disagrees with current query identity: {query_id}")
        record_id = str(row["retrieval_record_id"])
        training = train_by_id[record_id]
        if (
            str(training.get("molecule_identity_key"))
            != str(row["retrieval_molecule_identity_key"])
            or str(training.get("condition_group")) != str(row["retrieval_condition_group"])
        ):
            raise ValueError(f"{task} V9 row disagrees with current gold identity: {record_id}")
        if float(row["morgan_tanimoto_similarity"]) < min_similarity:
            continue
        parent = str(row["retrieval_molecule_identity_key"])
        analog_id = stable_analog_id(
            {"standard_inchi_key": parent, "canonical_smiles": row["retrieval_smiles"]}
        )
        analog_rank = (
            int(row["model_rank"])
            if ranking == "v9"
            else int(row["retrieval_parent_rank"])
        )
        analog = candidates[query_id].setdefault(
            analog_id,
            {
                "analog_id": analog_id,
                "canonical_smiles": str(row["retrieval_smiles"]),
                "similarity": float(row["morgan_tanimoto_similarity"]),
                "molecule_relation": "structural_analog",
                "_selection_rank": analog_rank,
                "cards": {},
            },
        )
        analog["_selection_rank"] = min(int(analog["_selection_rank"]), analog_rank)
        card_rank = (
            int(row["model_rank"])
            if ranking == "v9"
            else int(row["retrieval_parent_context_index"])
        )
        card_id = "card_" + hashlib.sha256(
            f"{task}:conditioned_gold:{record_id}".encode("utf-8")
        ).hexdigest()[:16]
        label_counts = training.get("label_counts") or {}
        vote_total = sum(int(value) for value in label_counts.values())
        positive_fraction = int(label_counts.get("1", 0)) / vote_total
        card = {
            "card_id": card_id,
            "evidence_family": family,
            "assay_context": "conditioned benchmark training label",
            "endpoint": _task_contract(task).endpoint_name,
            "reported_value": (
                f"Frozen label={int(training['Y'])}; "
                f"positive-vote fraction={100.0 * positive_fraction:.1f}%"
            ),
            "reported_unit": "",
            "qualifying_conditions": (
                ""
                if training.get("condition_group") == "no_reported_external_condition"
                else str(training.get("condition_group") or "")
            ),
            "support_text": "Frozen conditioned benchmark training outcome.",
            "_assay_key": str(training.get("condition_group") or record_id),
            "_selection_rank": card_rank,
        }
        if ranking == "v9":
            card["transfer_likelihood"] = round(float(row["prob_transfer"]), 2)
        analog["cards"][card_id] = card

    if min_similarity == 0 and any(not rows for rows in candidates.values()):
        raise ValueError(f"{task} has a current valid query without gold L1 candidates")
    return candidates, {
        "schema_version": cache_manifest["schema_version"],
        "ranking": ranking,
        "min_similarity": min_similarity,
        "n_queries_without_candidates": sum(not rows for rows in candidates.values()),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "rankings": str(rankings_path),
        "rankings_sha256": sha256_file(rankings_path),
        "model": cache_manifest.get("model"),
        "n_current_queries": len(current_ids),
        "n_extra_cache_queries": 0,
        "n_stale_retrieval_records": 0,
        "n_gold_training_rows": len(train_ids),
        "gold_card_source": "conditioned_benchmark_train_labels",
        "stage6_manifest": str(stage6_manifest_path),
        "stage6_manifest_sha256": sha256_file(stage6_manifest_path),
        "stage6_records_sha256": stage6_records_sha256,
    }


def _load_reused_query_prior(
    task: str,
    record: Mapping[str, Any],
    single_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    key = _stable_query_key(record)
    try:
        source_index = _single_source_index(task, str(single_root))[key]
    except KeyError as exc:
        raise ValueError(f"no reusable single branch for stable query identity: {key}") from exc
    run_dir = _source_run_dir(task, source_index, single_root)
    single = _read_json(run_dir / "single_molecule_reasoning_output.json")
    none_final = _read_json(run_dir / "final_reasoning_output.json")
    if single.get("status") != "ok" or none_final.get("status") != "ok":
        raise ValueError(f"incomplete visible none/single source: {run_dir}")
    single_llm = single.get("llm") or {}
    if not structured_response_is_valid(single_llm):
        raise ValueError(f"invalid reused single output: {run_dir}")
    query_tool_results = single_llm.get("tool_results") or []
    query_tool_summary = query_tool_results[0] if query_tool_results else {}
    return (
        dict(single_llm.get("content") or {}),
        dict(query_tool_summary),
        none_final,
        source_index,
    )


def _prefetch_analog_tools(
    *,
    query_smiles: str,
    analogs: Mapping[str, Mapping[str, Any]],
    tool_service_url: str,
    timeout_s: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    order: list[str] = []
    for analog in sorted(analogs.values(), key=lambda row: str(row["analog_id"])):
        analog_id = str(analog["analog_id"])
        reference_smiles = str(analog.get("canonical_smiles") or "")
        order.append(analog_id)
        calls.extend(
            [
                (
                    "mmp_structure_compare",
                    {
                        "query_smiles": query_smiles,
                        "reference_smiles": reference_smiles,
                        "max_mmp_alternatives": 5,
                        "mcs_timeout_s": 5,
                    },
                ),
                (
                    "properties_compare",
                    {
                        "query_smiles": query_smiles,
                        "reference_smiles": reference_smiles,
                        "logd_ph": 7.4,
                    },
                ),
            ]
        )
    client = ToolServiceClient(tool_service_url, timeout_s=timeout_s)
    results = invoke_with_retry(client, calls)
    # A transient service failure gets one cheap retry. Molecule-specific tool
    # failures remain visible as unavailable evidence instead of discarding the
    # entire query; the full error receipt is retained outside the model prompt.
    failures = []
    for position, result in enumerate(results):
        if result.get("status") == "ok":
            continue
        analog_id = order[position // 2]
        failures.append({"analog_id": analog_id, **dict(result)})
        results[position] = {
            "tool_name": result.get("tool_name"),
            "status": "error",
            "content": (
                f"[{result.get('tool_name')}]\n"
                "Comparison unavailable for this analog; do not infer a missing result."
            ),
        }
    summaries = {
        analog_id: results[position * 2 : position * 2 + 2]
        for position, analog_id in enumerate(order)
    }
    return summaries, failures


def _prepare_query(
    *,
    task: str,
    query_index: int,
    record: Mapping[str, Any],
    index: Mapping[str, Any] | None,
    levels: list[dict[str, Any]],
    gold_l1_candidates: Mapping[str, Mapping[str, Any]] | None,
    output_root: Path,
    single_root: Path,
    query_prior_mode: str,
    l1_source: str,
    l1_ranking: str,
    tool_service_url: str,
    timeout_s: int,
    prefetch_tools: bool,
) -> PreparedQuery:
    query_dir = _query_dir(output_root, task, query_index)
    complete_path = query_dir / "prepared_manifest.json"
    if complete_path.is_file():
        manifest = _read_json(complete_path)
        if (
            manifest.get("status") == "ok"
            and manifest.get("protocol") == PROGRESSIVE_PROTOCOL_VERSION
            and bool(manifest.get("tool_prefetch_complete")) is prefetch_tools
            and manifest.get("query_prior_mode") == query_prior_mode
            and manifest.get("l1_source") == l1_source
            and manifest.get("l1_ranking") == l1_ranking
        ):
            return PreparedQuery(task, query_index, query_dir)

    query_smiles = str(record.get("drug") or "")
    if query_prior_mode == "fresh":
        query_prior, query_tool_summary, none_final, single_source_index = _load_reused_query_prior(
            task, record, single_root
        )
    else:
        query_prior, query_tool_summary, none_final, single_source_index = {}, {}, {}, None
    cumulative_by_level: dict[int, dict[str, dict[str, Any]]] = {}
    retrieval_audits: dict[int, dict[str, Any]] = {}
    if gold_l1_candidates is not None:
        cumulative = {
            key: json.loads(json.dumps(value, ensure_ascii=False))
            for key, value in gold_l1_candidates.items()
        }
        cumulative_by_level[1] = cumulative
        retrieval_audits[1] = {
            "candidate_source": "conditioned_gold_training_labels",
            "ranking": l1_ranking,
            "n_cumulative_visible_molecules": len(cumulative),
            "n_cumulative_visible_cards": sum(
                len(row.get("cards") or {}) for row in cumulative.values()
            ),
        }
        if len(levels) > 1:
            if index is None:
                raise ValueError("gold L1 with later levels requires a normalized-v7 index")
            level_ids = [int(row["level"]) for row in levels]
            retrievals = retrieve_family_molecule_prefixes(
                query_smiles,
                index,
                levels=level_ids,
                min_similarity=0.3,
                neighbor_identity_policy=IDENTITY_POLICY,
            )
            for level_row in levels[1:]:
                level = int(level_row["level"])
                retrieval = retrievals[level]
                if retrieval.get("status") != "ok":
                    raise RuntimeError(f"{task} query {query_index} level {level} retrieval failed")
                cumulative = json.loads(json.dumps(cumulative_by_level[1], ensure_ascii=False))
                normalized = extract_cumulative_evidence(retrieval)
                for analog_id, analog in normalized.items():
                    cards = {
                        card_id: card
                        for card_id, card in (analog.get("cards") or {}).items()
                        if int(card.get("_family_level") or 0) > 1
                    }
                    if not cards:
                        continue
                    if analog_id not in cumulative:
                        cumulative[analog_id] = {**analog, "cards": cards}
                    else:
                        cumulative[analog_id]["cards"].update(cards)
                cumulative_by_level[level] = cumulative
                retrieval_audits[level] = {
                    **dict(retrieval.get("coverage") or {}),
                    "candidate_source": "conditioned_gold_l1_plus_normalized_v7",
                    "n_cumulative_visible_molecules": len(cumulative),
                    "n_cumulative_visible_cards": sum(
                        len(row.get("cards") or {}) for row in cumulative.values()
                    ),
                }
    else:
        if index is None:
            raise ValueError("normalized-v7 retrieval requires an index")
        level_ids = [int(row["level"]) for row in levels]
        retrievals = retrieve_family_molecule_prefixes(
            query_smiles,
            index,
            levels=level_ids,
            min_similarity=0.3,
            neighbor_identity_policy=IDENTITY_POLICY,
        )
        for level_row in levels:
            level = int(level_row["level"])
            retrieval = retrievals[level]
            if retrieval.get("status") != "ok":
                raise RuntimeError(f"{task} query {query_index} level {level} retrieval failed")
            cumulative = extract_cumulative_evidence(retrieval)
            cumulative_by_level[level] = cumulative
            retrieval_audits[level] = {
                **dict(retrieval.get("coverage") or {}),
                "n_cumulative_visible_molecules": len(cumulative),
                "n_cumulative_visible_cards": sum(
                    len(row.get("cards") or {}) for row in cumulative.values()
                ),
            }

    snapshots: dict[int, dict[str, dict[str, Any]]] = {}
    selection_audits: dict[int, dict[str, Any]] = {}
    active: dict[str, dict[str, Any]] = {}
    previous_cumulative: dict[str, dict[str, Any]] = {}
    for level_row in levels:
        level = int(level_row["level"])
        cumulative = cumulative_by_level[level]
        if level == 1:
            active, selection_audit = select_initial_evidence(cumulative, level=level)
        else:
            new, augmentations, selection_audit = select_progressive_delta(
                previous_cumulative,
                cumulative,
                active,
                level=level,
            )
            active = append_evidence(active, new, augmentations)
        snapshots[level] = json.loads(json.dumps(active, ensure_ascii=False))
        selection_audits[level] = selection_audit
        previous_cumulative = cumulative

    if prefetch_tools and active:
        summaries, tool_failures = _prefetch_analog_tools(
            query_smiles=query_smiles,
            analogs=active,
            tool_service_url=tool_service_url,
            timeout_s=timeout_s,
        )
        for snapshot in snapshots.values():
            attach_analog_tool_summaries(snapshot, summaries)
    else:
        tool_failures = []

    condition_sentence = external_condition_sentence(dict(record))
    previous_ids: set[str] = set()
    for level_row in levels:
        level = int(level_row["level"])
        snapshot = snapshots[level]
        current_ids = card_ids(snapshot)
        new_ids = current_ids - previous_ids
        prepared = {
            "protocol": PROGRESSIVE_PROTOCOL_VERSION,
            "query_prior_mode": query_prior_mode,
            "l1_source": l1_source,
            "l1_ranking": l1_ranking,
            "task": task,
            "query_index": query_index,
            "benchmark_row_id": record.get("benchmark_row_id"),
            "molecule_identity_key": record.get("molecule_identity_key"),
            "condition_group": record.get("condition_group"),
            "reused_single_source_index": single_source_index,
            "query_smiles": query_smiles,
            "condition_sentence": condition_sentence,
            "query_prior": query_prior,
            "query_tool_summary": query_tool_summary,
            "reused_none_final": none_final,
            "level": level,
            "level_definition": level_row,
            "retrieval_audit": retrieval_audits[level],
            "selection_audit": selection_audits[level],
            "active_evidence": snapshot,
            "new_card_ids": sorted(new_ids),
            "n_active_molecules": len(snapshot),
            "n_active_cards": len(current_ids),
            "should_call_model": bool(new_ids),
            "tool_prefetch_complete": prefetch_tools,
            "tool_prefetch_failures": [
                row
                for row in tool_failures
                if str(row.get("analog_id")) in snapshot
            ],
        }
        level_dir = query_dir / "levels" / f"level_{level}"
        write_json_atomic(level_dir / "prepared.json", prepared)
        previous_ids = current_ids

    write_json_atomic(
        complete_path,
        {
            "status": "ok",
            "protocol": PROGRESSIVE_PROTOCOL_VERSION,
            "query_prior_mode": query_prior_mode,
            "l1_source": l1_source,
            "l1_ranking": l1_ranking,
            "task": task,
            "query_index": query_index,
            "n_levels": len(levels),
            "tool_prefetch_complete": prefetch_tools,
            "n_tool_prefetch_failures": len(tool_failures),
            "prepared_at": _now(),
        },
    )
    return PreparedQuery(task, query_index, query_dir)


def _prepare_context_record_query(
    *,
    task: str,
    query_index: int,
    record: Mapping[str, Any],
    contexts: list[Mapping[str, Any]],
    levels: list[dict[str, Any]],
    output_root: Path,
    single_root: Path,
    query_prior_mode: str,
    record_limit: int,
    l2_record_limit: int,
    indirect_record_limit: int,
    context_limit: int = context_records.CONTEXT_LIMIT,
    indirect_record_limits: Mapping[str, int] | None = None,
    indirect_records: Mapping[str, Mapping[str, Any]] | None = None,
    l2_reuse_root: Path | None = None,
    prompt_version: str = "v1",
    context_ranking: str = "assay_transfer",
    ranking_tie_seed: int = 0,
    cache_profile: str = "v19_1",
    record_levels: tuple[str, ...] | None = None,
    v21_selection_mode: str = "record_only",
) -> PreparedQuery:
    """Write context snapshots and optional cache-backed later-level bundles."""
    query_dir = _query_dir(output_root, task, query_index)
    complete_path = query_dir / "prepared_manifest.json"
    protocol = (
        context_records.INDIRECT_PROTOCOL_VERSION
        if indirect_records is not None
        else context_records.PROTOCOL_VERSION
    )
    if complete_path.is_file():
        manifest = _read_json(complete_path)
        if (
            manifest.get("status") == "ok"
            and manifest.get("protocol") == protocol
            and manifest.get("query_prior_mode") == query_prior_mode
            and manifest.get("profile") == "context_records"
            and manifest.get("prompt_version") == prompt_version
            and manifest.get("context_ranking") == context_ranking
            and manifest.get("assay_transfer_cache_profile", "v19_1")
            == cache_profile
            and manifest.get("v21_selection_mode", "record_only")
            == v21_selection_mode
            and manifest.get("ranking_tie_seed") == ranking_tie_seed
            and manifest.get("context_limit", context_records.CONTEXT_LIMIT)
            == context_limit
            and manifest.get("record_limit_per_context_level") == record_limit
            and manifest.get("l2_record_limit_per_context") == l2_record_limit
            and manifest.get("indirect_record_limit_per_level")
            == (
                indirect_record_limit
                if indirect_records is not None
                else None
            )
            and manifest.get("indirect_record_limits_by_level")
            == (
                dict(indirect_record_limits)
                if indirect_record_limits is not None
                and len(set(indirect_record_limits.values())) > 1
                else None
            )
            and manifest.get("l2_reuse_root")
            == (str(l2_reuse_root) if l2_reuse_root is not None else None)
            and (
                l2_reuse_root is None
                or all(
                    (
                        query_dir
                        / "levels"
                        / f"level_{level}"
                        / "output.json"
                    ).is_file()
                    for level in (1, 2)
                )
            )
        ):
            return PreparedQuery(task, query_index, query_dir)

    if query_prior_mode == "fresh":
        query_prior, query_tool_summary, none_final, single_source_index = (
            _load_reused_query_prior(task, record, single_root)
        )
    else:
        query_prior, query_tool_summary, none_final, single_source_index = {}, {}, {}, None

    query_smiles = str(record.get("drug") or "")
    condition_sentence = external_condition_sentence(dict(record))
    snapshots = context_records.snapshots(
        contexts,
        task=task,
        indirect_records=indirect_records,
        indirect_record_limit=indirect_record_limits or indirect_record_limit,
        prompt_version=prompt_version,
        record_levels=record_levels,
        transfer_model="V21" if cache_profile == V21_CACHE_PROFILE else "V19.1",
    )
    previous_ids: set[str] = set()
    for level_row in levels:
        level = int(level_row["level"])
        level_record_limit = (
            indirect_record_limits.get(f"L{level}")
            if indirect_record_limits is not None
            else indirect_record_limit
        )
        snapshot = snapshots[level]
        current_ids = card_ids(snapshot)
        new_ids = current_ids - previous_ids
        write_json_atomic(
            query_dir / "levels" / f"level_{level}" / "prepared.json",
            {
                "protocol": protocol,
                "profile": "context_records",
                "query_prior_mode": query_prior_mode,
                "l1_source": (
                    "v21_ranked_stage3_record_molecules"
                    if cache_profile == V21_CACHE_PROFILE
                    and v21_selection_mode == "molecule_cards"
                    else "v21_independently_ranked_stage3_records"
                    if cache_profile == V21_CACHE_PROFILE
                    else "current_conditioned_gold_raw_records"
                ),
                "l1_ranking": context_ranking,
                "task": task,
                "query_index": query_index,
                "benchmark_row_id": record.get("benchmark_row_id"),
                "molecule_identity_key": record.get("molecule_identity_key"),
                "condition_group": record.get("condition_group"),
                "reused_single_source_index": single_source_index,
                "query_smiles": query_smiles,
                "condition_sentence": condition_sentence,
                "query_prior": query_prior,
                "query_tool_summary": query_tool_summary,
                "reused_none_final": none_final,
                "level": level,
                "level_definition": level_row,
                "retrieval_audit": {
                    "candidate_source": (
                        f"{cache_profile}_stage3_top{context_limit}_molecules_with_v21_ranked_records"
                        if cache_profile == V21_CACHE_PROFILE
                        and v21_selection_mode == "molecule_cards"
                        and level <= 2
                        else f"{cache_profile}_stage3_top{level_record_limit}_records"
                        if cache_profile == V21_CACHE_PROFILE
                        else
                        "luna_relevance_top_quartile_then_stage3_"
                        f"top{level_record_limit}_records"
                        if level >= 3
                        and cache_profile == LUNA_RELEVANCE_CACHE_PROFILE
                        else
                        f"v19_1_stage3_morgan_top{level_record_limit}_records"
                        if level >= 3 and context_ranking == "morgan"
                        else f"v19_1_stage3_top{level_record_limit}_records"
                        if level >= 3
                        else f"v9_candidate_universe_morgan_top{context_limit}_contexts"
                        if context_ranking == "morgan"
                        else f"v9_top{context_limit}_current_conditioned_gold_contexts"
                    ),
                    "n_contexts": sum(
                        row.get("_card_kind") != "level_record_bundle"
                        for row in snapshot.values()
                    ),
                    "n_level_record_bundles": sum(
                        row.get("_card_kind") == "level_record_bundle"
                        for row in snapshot.values()
                    ),
                    "n_visible_records": len(current_ids),
                },
                "selection_audit": {
                    "context_limit": context_limit,
                    "record_limit_per_context_level": record_limit,
                    "l2_record_limit_per_context": l2_record_limit,
                    "indirect_record_limit_per_level": (
                        indirect_record_limit
                        if indirect_records is not None
                        else None
                    ),
                    "indirect_record_limit_for_current_level": (
                        level_record_limit if indirect_records is not None else None
                    ),
                    "deterministic_sampling": True,
                    "ranking": context_ranking,
                    "ranking_tie_seed": (
                        ranking_tie_seed if context_ranking == "morgan" else None
                    ),
                },
                "active_evidence": snapshot,
                "new_card_ids": sorted(new_ids),
                "n_active_molecules": len(snapshot),
                "n_active_cards": len(current_ids),
                "should_call_model": bool(new_ids),
                # The profile intentionally disables per-analog tools. The reused
                # query property prior is complete, so inference may proceed.
                "tool_prefetch_complete": True,
                "analog_tool_policy": "disabled_by_context_records_profile",
                "tool_prefetch_failures": [],
            },
        )
        previous_ids = current_ids

    if l2_reuse_root is not None:
        _reuse_context_l1_l2_outputs(
            task=task,
            query_index=query_index,
            query_dir=query_dir,
            source_root=l2_reuse_root,
            prompt_version=prompt_version,
            record_limit=record_limit,
            l2_record_limit=l2_record_limit,
        )

    write_json_atomic(
        complete_path,
        {
            "status": "ok",
            "protocol": protocol,
            "profile": "context_records",
            "prompt_version": prompt_version,
            "context_ranking": context_ranking,
            "assay_transfer_cache_profile": cache_profile,
            "v21_selection_mode": (
                v21_selection_mode
                if cache_profile == V21_CACHE_PROFILE
                else None
            ),
            "ranking_tie_seed": ranking_tie_seed,
            "query_prior_mode": query_prior_mode,
            "context_limit": context_limit,
            "record_limit_per_context_level": record_limit,
            "l2_record_limit_per_context": l2_record_limit,
            "indirect_record_limit_per_level": (
                indirect_record_limit
                if indirect_records is not None
                else None
            ),
            "indirect_record_limits_by_level": (
                dict(indirect_record_limits)
                if indirect_record_limits is not None
                and len(set(indirect_record_limits.values())) > 1
                else None
            ),
            "l2_reuse_root": (
                str(l2_reuse_root) if l2_reuse_root is not None else None
            ),
            "task": task,
            "query_index": query_index,
            "n_levels": len(levels),
            "tool_prefetch_complete": True,
            "analog_tool_policy": "disabled_by_context_records_profile",
            "prepared_at": _now(),
        },
    )
    return PreparedQuery(task, query_index, query_dir)


def _none_state(
    *,
    contract: ProgressiveTaskContract,
    level: int,
    none_final: Mapping[str, Any],
) -> dict[str, Any]:
    content = (none_final.get("llm") or {}).get("content") or {}
    return {
        "level": level,
        contract.prediction_field: content.get(contract.prediction_field),
        "confidence": content.get("confidence", "low"),
        "revision_action": "initial",
        "supportive_card_ids": [],
        "contradictory_card_ids": [],
        "not_used_card_ids": [],
        "prediction_basis_card_ids": [],
        "claims": [],
        "new_evidence_assessment": [],
        "evidence_gaps": content.get("evidence_gaps") or [],
        "decision_summary": content.get("final_summary") or "Reused visible none decision.",
    }


def _resolve_provider_pool_config(args: argparse.Namespace) -> ProviderPoolConfig:
    config_path = str(getattr(args, "provider_pool_config", "") or "").strip()
    if config_path:
        config = load_provider_pool_config(config_path)
    else:
        config = ProviderPoolConfig(
            providers=(
                ProviderSpec(
                    name="single",
                    base_url=args.base_url.rstrip("/"),
                    model=args.model,
                    api_key_env=args.api_key_env,
                    max_inflight=args.parallelism,
                ),
            ),
            max_failovers=0,
        )
    expected_identity = _model_identity(args.model)
    for spec in config.providers:
        if _model_identity(spec.model) != expected_identity:
            raise ValueError(
                f"provider {spec.name!r} model {spec.model!r} does not match "
                f"run model identity {expected_identity!r}"
            )
        if config_path and spec.api_key_env and not os.getenv(spec.api_key_env):
            raise ValueError(
                f"provider {spec.name!r} is missing API key env {spec.api_key_env!r}"
            )
    return config


def _make_client(
    args: argparse.Namespace,
    provider_config: ProviderPoolConfig,
) -> OpenAIProviderPool:
    def client_factory(spec: ProviderSpec) -> OpenAICompatibleClient:
        return OpenAICompatibleClient(
            api_key=os.getenv(spec.api_key_env) or "local-no-auth",
            base_url=spec.base_url,
            model=spec.model,
            timeout_s=spec.timeout_s or args.timeout_s,
            max_tokens=args.max_tokens,
            temperature=0.0,
            tool_service_url=args.tool_service_url,
            enable_group_tools=False,
            max_tool_rounds=0,
            reasoning_effort="",
            enable_thinking=False,
            transport_max_retries=args.transport_max_retries,
            request_extra_body=spec.request_extra_body,
        )

    return OpenAIProviderPool(provider_config, client_factory=client_factory)


def _run_query(
    args: argparse.Namespace,
    prepared_query: PreparedQuery,
    client: OpenAIProviderPool,
) -> dict[str, Any]:
    task = prepared_query.task
    contract = _task_contract(task)
    levels = _run_levels(args, task)
    prior_state: dict[str, Any] | None = None
    n_calls = 0
    for level_row in levels:
        level = int(level_row["level"])
        level_dir = prepared_query.query_dir / "levels" / f"level_{level}"
        output_path = level_dir / "output.json"
        if output_path.is_file():
            existing = _read_json(output_path)
            if existing.get("status") in COMPLETE_LEVEL_STATUSES:
                _write_level_trace(args, prepared_query, level, output_path, existing)
                prior_state = dict(existing["state"])
                n_calls += int(existing.get("model_called") is True)
                continue
        prepared = _read_json(level_dir / "prepared.json")
        if not prepared.get("tool_prefetch_complete"):
            raise ValueError(
                f"formal inference requires visible tool prefetch: {level_dir}"
            )
        if not prepared.get("should_call_model"):
            if prior_state is None:
                if args.query_prior == "none":
                    raise ValueError(f"standalone L1 has no visible evidence: {level_dir}")
                state = _none_state(
                    contract=contract,
                    level=level,
                    none_final=prepared["reused_none_final"],
                )
                status = "reused_none"
            else:
                state = {**prior_state, "level": level, "revision_action": "keep"}
                status = "carried_forward"
            write_json_atomic(
                output_path,
                {
                    "status": status,
                    "model_called": False,
                    "state": state,
                    "created_at": _now(),
                },
            )
            prior_state = state
            continue

        active = prepared["active_evidence"]
        card_id_to_alias, alias_to_card_id = card_alias_maps(active)
        if args.profile == "context_records":
            record_limits = _indirect_record_limits(args, task)
            prompt_record_limits: int | Mapping[str, int] = (
                next(iter(record_limits.values()))
                if len(set(record_limits.values())) == 1
                else record_limits
            )
            messages = context_records.build_messages(
                contract=contract,
                current_level=level,
                query_smiles=prepared["query_smiles"],
                condition_sentence=prepared["condition_sentence"],
                query_prior=prepared["query_prior"] or None,
                query_tool_summary=prepared.get("query_tool_summary") or {},
                active=active,
                prior_state=prior_state,
                prompt_version=_context_prompt_version(args, task),
                record_limit=args.record_limit_per_context_level,
                l2_record_limit=args.l2_record_limit_per_context,
                indirect_record_limit=prompt_record_limits,
                include_indirect=_record_cache_enabled(args),
            )
        else:
            messages = build_progressive_messages(
                contract=contract,
                levels=levels,
                current_level=level,
                query_smiles=prepared["query_smiles"],
                condition_sentence=prepared["condition_sentence"],
                query_prior=prepared["query_prior"] or None,
                query_tool_summary=prepared.get("query_tool_summary") or {},
                active=active,
                prior_state=prior_state,
            )
        write_json_atomic(
            level_dir / "request.json",
            {
                "messages": messages,
                "prompt_characters": sum(len(str(row.get("content") or "")) for row in messages),
                "prompt_utf8_bytes": sum(len(str(row.get("content") or "").encode("utf-8")) for row in messages),
                "card_alias_map": alias_to_card_id,
            },
        )
        visible_aliases = set(alias_to_card_id)
        new_aliases = {
            card_id_to_alias[card_id]
            for card_id in map(str, prepared.get("new_card_ids") or [])
        }
        execution_provider_attempts: list[dict[str, Any]] = []

        def routed_chat_json(call_messages: list[dict[str, Any]]) -> dict[str, Any]:
            routed_response = client.chat_json(call_messages)
            execution_provider_attempts.extend(
                routed_response.get("execution_provider_attempts") or []
            )
            return routed_response

        response = call_with_json_validation(
            routed_chat_json,
            messages,
            required_fields=(
                contract.prediction_field,
                "confidence",
                "revision_action",
                "supportive_card_ids",
                "contradictory_card_ids",
                "prediction_basis_card_ids",
                "claims",
                "new_evidence_assessment",
                "evidence_gaps",
                "decision_summary",
            ),
            allowed_values={
                contract.prediction_field: contract.prediction_values,
                "confidence": {"high", "moderate", "low"},
                "revision_action": (
                    {"initial"}
                    if prior_state is None
                    else {"keep", "strengthen", "weaken", "flip"}
                ),
            },
            content_validator=lambda content: progressive_state_errors(
                content,
                contract=contract,
                visible_card_ids=visible_aliases,
                new_card_ids=new_aliases,
                prior_state=prior_state,
            ),
            branch_name=f"{task} progressive level {level}",
            max_attempts=4,
        )
        response["execution_provider_attempts"] = execution_provider_attempts
        n_calls += int((response.get("structured_output_validation") or {}).get("attempt_count") or 1)
        if not structured_response_is_valid(response):
            output = {
                "status": "error",
                "model_called": True,
                "llm": response,
                "card_alias_map": alias_to_card_id,
                "created_at": _now(),
            }
            write_json_atomic(
                output_path,
                output,
            )
            _write_level_trace(args, prepared_query, level, output_path, output)
            return {"task": task, "index": prepared_query.index, "status": "error", "level": level}
        restored_content = restore_card_ids(
            response["content"],
            alias_to_card_id=alias_to_card_id,
        )
        state = state_from_content(
            restored_content,
            contract=contract,
            level=level,
            visible_card_ids=set(alias_to_card_id.values()),
        )
        output = {
            "status": "ok",
            "model_called": True,
            "state": state,
            "llm": response,
            "card_alias_map": alias_to_card_id,
            "created_at": _now(),
        }
        write_json_atomic(output_path, output)
        _write_level_trace(args, prepared_query, level, output_path, output)
        prior_state = state
    return {
        "task": task,
        "index": prepared_query.index,
        "status": "ok",
        "n_model_attempts": n_calls,
    }


def _run_query_safe(
    args: argparse.Namespace,
    prepared_query: PreparedQuery,
    client: OpenAIProviderPool,
) -> dict[str, Any]:
    """Keep one transport/provider failure from terminating unrelated queries."""
    try:
        result = _run_query(args, prepared_query, client)
        error_path = prepared_query.query_dir / "run_error.json"
        if result.get("status") == "ok" and error_path.is_file():
            write_json_atomic(
                error_path,
                {"status": "resolved", "resolved_at": _now()},
            )
        return result
    except Exception as exc:  # noqa: BLE001 - persisted for resumable batch audit
        error = {
            "task": prepared_query.task,
            "index": prepared_query.index,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_at": _now(),
        }
        if isinstance(exc, ProviderPoolExhausted):
            error["execution_provider_attempts"] = exc.attempts
        write_json_atomic(prepared_query.query_dir / "run_error.json", error)
        return error


def _prediction_to_label(contract: ProgressiveTaskContract, prediction: Any) -> int | None:
    value = str(prediction or "")
    if value == contract.positive_prediction:
        return 1
    if value == contract.negative_prediction:
        return 0
    return None


def _class_metrics(rows: list[dict[str, Any]], label: int) -> dict[str, Any]:
    tp = sum(row["label"] == label and row["pred_label"] == label for row in rows)
    fp = sum(row["label"] != label and row["pred_label"] == label for row in rows)
    fn = sum(row["label"] == label and row["pred_label"] != label for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _write_prediction_summary(
    *,
    task: str,
    predictions: list[dict[str, Any]],
    summary_dir: Path,
    metric_fields: Mapping[str, Any],
) -> None:
    successful = [row for row in predictions if row.get("pred_label") in (0, 1)]
    per_class = {str(label): _class_metrics(successful, label) for label in (0, 1)}
    metrics = {
        "task": task,
        **dict(metric_fields),
        "n_total": len(predictions),
        "n_successful": len(successful),
        "n_failed_runs": len(predictions) - len(successful),
        "accuracy": (
            sum(bool(row["correct"]) for row in successful) / len(successful)
            if successful
            else 0.0
        ),
        "macro_f1": (per_class["0"]["f1"] + per_class["1"]["f1"]) / 2,
        "per_class": per_class,
        "n_model_called": sum(row.get("model_called") is True for row in predictions),
        "finished_at": _now(),
    }
    write_jsonl_atomic(summary_dir / "predictions.jsonl", predictions)
    write_json_atomic(summary_dir / "metrics.json", metrics)


def _summarize_task(
    *,
    task: str,
    records: list[dict[str, Any]],
    indices: list[int],
    output_root: Path,
    max_level: int,
    query_prior_mode: str,
    profile: str = "standard",
    context_record_l3_l5: bool = False,
) -> None:
    contract = _task_contract(task)
    if query_prior_mode == "fresh":
        none_predictions = []
        for query_index in indices:
            prepared_path = (
                _query_dir(output_root, task, query_index)
                / "levels"
                / "level_1"
                / "prepared.json"
            )
            prepared = _read_json(prepared_path)
            none_final = prepared.get("reused_none_final") or {}
            content = ((none_final.get("llm") or {}).get("content") or {})
            prediction = content.get(contract.prediction_field)
            pred_label = _prediction_to_label(contract, prediction)
            label = records[query_index].get("Y")
            none_predictions.append(
                {
                    "index": query_index,
                    "label": label,
                    contract.prediction_field: prediction,
                    "pred_label": pred_label,
                    "correct": pred_label == label if pred_label is not None else None,
                    "confidence": content.get("confidence"),
                    "status": none_final.get("status"),
                    "model_called": False,
                    "source": str(prepared_path),
                }
            )
        _write_prediction_summary(
            task=task,
            predictions=none_predictions,
            summary_dir=output_root / task / "none",
            metric_fields={"level": 0, "family": "none", "reuse": "stable parent-condition identity"},
        )

    level_rows = (
        context_records.levels(
            max_level,
            task=task,
            include_indirect=context_record_l3_l5,
        )
        if profile == "context_records"
        else _levels(task, max_level)
    )
    for level_row in level_rows:
        level = int(level_row["level"])
        predictions = []
        for query_index in indices:
            output_path = _query_dir(output_root, task, query_index) / "levels" / f"level_{level}" / "output.json"
            if not output_path.is_file():
                predictions.append({"index": query_index, "label": records[query_index].get("Y"), "status": "missing"})
                continue
            output = _read_json(output_path)
            state = output.get("state") or {}
            prediction = state.get(contract.prediction_field)
            pred_label = _prediction_to_label(contract, prediction)
            label = records[query_index].get("Y")
            predictions.append(
                {
                    "index": query_index,
                    "label": label,
                    contract.prediction_field: prediction,
                    "pred_label": pred_label,
                    "correct": pred_label == label if pred_label is not None else None,
                    "confidence": state.get("confidence"),
                    "revision_action": state.get("revision_action"),
                    "status": output.get("status"),
                    "model_called": output.get("model_called"),
                    "output": str(output_path),
                }
            )
        level_summary = output_root / task / "levels" / f"level_{level}"
        _write_prediction_summary(
            task=task,
            predictions=predictions,
            summary_dir=level_summary,
            metric_fields={"level": level, "family": level_row["endpoint_group"]},
        )


def _selected_indices(args: argparse.Namespace, n_records: int) -> list[int]:
    if args.indices:
        indices = sorted({int(value) for value in args.indices})
    else:
        indices = list(range(n_records))
        if args.limit > 0:
            indices = indices[: args.limit]
    if not indices or indices[0] < 0 or indices[-1] >= n_records:
        raise ValueError(f"invalid query indices for {n_records} records: {indices[:3]}")
    return indices


def _validate_inputs(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    records_by_task = {}
    for task in args.tasks:
        spec = PROGRESSIVE_TASKS[task]
        if not spec.input_jsonl.is_file():
            raise FileNotFoundError(spec.input_jsonl)
        if args.profile == "standard":
            for path in (spec.index, spec.family_manifest):
                if not path.exists():
                    raise FileNotFoundError(path)
            manifest = _read_json(
                spec.index / "manifest.json"
                if spec.index.is_dir()
                else spec.index.with_name("manifest.json")
            )
            if spec.index.is_dir():
                expected = {
                    "task_id": task,
                    "heldout_filter_mode": "direct_source_only",
                }
                swap = manifest.get("gold_label_swap_for_direct_labels") or {}
                if swap.get("version") != "gold_label_swap_for_direct_labels.v1":
                    raise ValueError(f"{task} compact index lacks the direct gold swap")
            else:
                expected = {
                    "reference_pool": REFERENCE_POOL,
                    "neighbor_identity_policy_default": IDENTITY_POLICY,
                    "n_direct_heldout_records_after_filter": 0,
                }
                if task in {"bioavailability_ma", "skin_reaction"}:
                    expected.update(
                        {
                            "filter_source_id": "",
                            "filter_scope_field": "group_id",
                            "filter_scope_value": (
                                "Observed.direct_oral_bioavailability"
                                if task == "bioavailability_ma"
                                else "Direct.skin_reaction"
                            ),
                        }
                    )
            for field, value in expected.items():
                if manifest.get(field) != value:
                    raise ValueError(
                        f"{task} index has wrong {field}: {manifest.get(field)!r}"
                    )
        if args.query_prior == "fresh":
            batch_dir = _single_batch_dir(task, Path(args.single_source_root))
            if not (batch_dir / "runs").is_dir():
                raise FileNotFoundError(batch_dir / "runs")
            single_manifest_path = batch_dir / "manifest.json"
            single_manifest = _read_json(single_manifest_path)
            reusable_model = str(single_manifest.get("model") or "")
            if _model_identity(reusable_model) != _model_identity(args.model):
                raise ValueError(
                    f"{task} reusable single batch has wrong model: {reusable_model!r}"
                )
            expected_single = {
                "visibility_mode": "deployment_visible_prefetched",
                "identity_blind": False,
                "harness_prefetch_tools": True,
            }
            for field, value in expected_single.items():
                if single_manifest.get(field) != value:
                    raise ValueError(
                        f"{task} reusable single batch has wrong {field}: "
                        f"{single_manifest.get(field)!r}"
                    )
            current_keys = {_stable_query_key(row) for row in read_jsonl(spec.input_jsonl)}
            reusable_keys = set(_single_source_index(task, str(Path(args.single_source_root))))
            missing_keys = current_keys - reusable_keys
            if missing_keys:
                raise ValueError(
                    f"{task} has {len(missing_keys)} rows without stable-identity single reuse"
                )
        records_by_task[task] = read_jsonl(spec.input_jsonl)
    return records_by_task


def run(args: argparse.Namespace) -> int:
    provider_config = _resolve_provider_pool_config(args)
    if sum(spec.max_inflight for spec in provider_config.providers) > args.endpoint_concurrency_budget:
        raise ValueError("provider pool capacity exceeds the configured endpoint budget")
    records_by_task = _validate_inputs(args)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    indices_by_task = {
        task: _selected_indices(args, len(records_by_task[task]))
        for task in args.tasks
    }
    l2_reuse_audits = {
        task: _validate_context_l2_reuse_source(
            args,
            task,
            PROGRESSIVE_TASKS[task].input_jsonl,
            indices_by_task[task],
        )
        for task in args.tasks
        if args.reuse_context_l2
    }
    gold_candidates_by_task: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    context_candidates_by_task: dict[str, dict[str, list[dict[str, Any]]]] = {}
    indirect_candidates_by_task: dict[
        str, dict[str, dict[str, dict[str, Any]]]
    ] = {}
    ranking_audits: dict[str, dict[str, Any]] = {}
    indirect_ranking_audits: dict[str, dict[str, Any]] = {}
    if args.profile == "context_records":
        for task in args.tasks:
            selected_queries = {
                str(records_by_task[task][index]["benchmark_row_id"]): str(
                    records_by_task[task][index]["drug"]
                )
                for index in indices_by_task[task]
            }
            if (
                args.assay_transfer_cache_profile == V21_CACHE_PROFILE
                and args.v21_selection_mode == "record_only"
            ):
                context_candidates_by_task[task] = {
                    str(row["benchmark_row_id"]): [] for row in records_by_task[task]
                }
            elif args.assay_transfer_cache_profile == V21_CACHE_PROFILE:
                ranked_molecules, audit = load_v21_molecule_card_records(
                    selected_queries,
                    molecule_limit=args.context_limit,
                    l1_record_limit=args.record_limit_per_context_level,
                    l2_record_limit=args.l2_record_limit_per_context,
                    workers=args.preparation_workers,
                )
                context_candidates_by_task[task] = {
                    query_id: context_records.v21_molecule_contexts(
                        query_id, rows
                    )
                    for query_id, rows in ranked_molecules.items()
                }
                ranking_audits[task] = audit
            else:
                candidates, audit = context_records.load_candidates(
                    task=task,
                    valid_records=records_by_task[task],
                    ranking_root=Path(args.v9_ranking_root),
                    benchmark_root=Path(args.benchmark_data_root),
                    v7_root=Path(args.v7_root),
                    record_limit=args.record_limit_per_context_level,
                    l2_record_limit=args.l2_record_limit_per_context,
                    context_limit=args.context_limit,
                    ranking=args.context_ranking,
                    tie_seed=args.ranking_tie_seed,
                )
                context_candidates_by_task[task] = candidates
                ranking_audits[task] = audit
            if _record_cache_enabled(args):
                indirect, indirect_audit = load_top_ranked_records(
                    task,
                    selected_queries,
                    cache_profile=args.assay_transfer_cache_profile,
                    levels=_record_level_names(args, task),
                    limit=_indirect_record_limits(args, task),
                    workers=args.preparation_workers,
                    ranking=args.context_ranking,
                    tie_seed=args.ranking_tie_seed,
                )
                indirect_candidates_by_task[task] = indirect
                indirect_ranking_audits[task] = indirect_audit
                if (
                    args.assay_transfer_cache_profile == V21_CACHE_PROFILE
                    and args.v21_selection_mode == "record_only"
                ):
                    ranking_audits[task] = indirect_audit
    elif args.l1_source == "gold_train":
        for task in args.tasks:
            candidates, audit = _gold_l1_candidates(
                task=task,
                records=records_by_task[task],
                ranking_root=Path(args.v9_ranking_root),
                ranking=args.l1_ranking,
                benchmark_root=Path(args.benchmark_data_root),
                min_similarity=args.gold_l1_min_similarity,
            )
            gold_candidates_by_task[task] = candidates
            ranking_audits[task] = audit
    inputs = {}
    for task in args.tasks:
        spec = PROGRESSIVE_TASKS[task]
        task_inputs = {
            "input_jsonl": str(spec.input_jsonl),
            "input_sha256": sha256_file(spec.input_jsonl),
            "single_source_manifest_sha256": (
                sha256_file(
                    _single_batch_dir(task, Path(args.single_source_root))
                    / "manifest.json"
                )
                if args.query_prior == "fresh"
                else None
            ),
            "levels": _run_levels(args, task),
            "l1_ranking_cache": ranking_audits.get(task),
        }
        if _record_cache_enabled(args):
            task_inputs.update(
                {
                    "indirect_ranking_cache": indirect_ranking_audits[task],
                    "context_record_prompt_version": _context_prompt_version(
                        args, task
                    ),
                }
            )
        if args.profile == "standard":
            task_inputs.update(
                {
                    "index": str(spec.index),
                    "index_sha256": sha256_file(
                        spec.index / "manifest.json" if spec.index.is_dir() else spec.index
                    ),
                    "family_manifest": str(spec.family_manifest),
                    "family_manifest_sha256": sha256_file(spec.family_manifest),
                }
            )
        inputs[task] = task_inputs
    if args.profile == "context_records":
        if args.assay_transfer_prompt_version == "task_best":
            context_card_contracts = {}
            for task in args.tasks:
                prompt_version = _context_prompt_version(args, task)
                contract_payload = context_records.card_contract(prompt_version)
                context_card_contracts[task] = {
                    "prompt_version": prompt_version,
                    "path": str(context_records.prompt_profile(prompt_version)["card"]),
                    "schema_version": contract_payload["schema_version"],
                    "semantic_sha256": hashlib.sha256(
                        json.dumps(
                            contract_payload,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                }
            card_contract_manifest: dict[str, Any] = {
                "by_task": context_card_contracts
            }
        else:
            prompt_version = args.assay_transfer_prompt_version
            contract_payload = context_records.card_contract(prompt_version)
            card_contract_manifest = {
                "path": str(context_records.prompt_profile(prompt_version)["card"]),
                "schema_version": contract_payload["schema_version"],
                "semantic_sha256": hashlib.sha256(
                    json.dumps(
                        contract_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
        if _record_cache_enabled(args):
            card_contract_manifest["level_record_bundle_by_task"] = {
                task: {
                    "path": str(context_records.LEVEL_RECORD_BUNDLE_PATH),
                    "schema_version": context_records.level_record_bundle_contract(
                        _context_prompt_version(args, task)
                    )["schema_version"],
                    "semantic_sha256": hashlib.sha256(
                        json.dumps(
                            context_records.level_record_bundle_contract(
                                _context_prompt_version(args, task)
                            ),
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                }
                for task in args.tasks
            }
    else:
        standard_card_contract = molecule_card_contract()
        card_contract_manifest = {
            "path": str(MOLECULE_CARD_CONTRACT_PATH),
            "schema_version": standard_card_contract["schema_version"],
            "semantic_sha256": hashlib.sha256(
                json.dumps(
                    standard_card_contract,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
    manifest = {
        "experiment": _run_protocol(args),
        "profile": args.profile,
        "evaluation_subset": "valid",
        "tasks": args.tasks,
        "visibility_mode": "deployment_visible_prefetched",
        "reference_pool": (
            "conditioned_gold_train_plus_normalized_v7"
            if args.l1_source == "gold_train" and args.max_level > 1
            else "conditioned_gold_train"
            if args.l1_source == "gold_train"
            else REFERENCE_POOL
        ),
        "neighbor_identity_policy": IDENTITY_POLICY,
        "min_similarity": (
            {"gold_l1": args.gold_l1_min_similarity, "normalized_later_levels": 0.3}
            if args.l1_source == "gold_train" and args.max_level > 1
            else args.gold_l1_min_similarity
            if args.l1_source == "gold_train"
            else 0.3
        ),
        "candidate_generation": {
            "unit": "molecule",
            "scope": (
                "V9 top-75 conditioned gold L1 plus normalized-v7 later families"
                if args.l1_source == "gold_train" and args.max_level > 1
                else "V9 top-75 conditioned gold-training pool"
                if args.l1_source == "gold_train"
                else "cumulative record-family pool"
            ),
            "ranking": args.l1_ranking,
            "per_assay_neighbor_cap": None,
            "assay_role": "card provenance and diversity only",
        },
        "selection": {
            "level_1": f"top 10 molecules by {args.l1_ranking}; at most 4 cards per molecule",
            "later_new_pool": "top 3 previously unseen molecules; at most 2 newly unlocked cards each",
            "later_augmentation_pool": "top 3 already-active molecules; at most 2 newly unlocked cards each",
            "slot_borrowing": False,
            "append_only": True,
            "cumulative_per_molecule_card_cap": None,
        },
        "prompt_profile": (
            "progressive_compact_tools_short_aliases.gold_l1.v1"
            if args.l1_source == "gold_train"
            else "progressive_compact_tools_short_aliases.v2"
        ),
        "molecule_card_contract": card_contract_manifest,
        "condition_policy": "natural-language sentence for non-null group; omit null group",
        "query_prior_mode": args.query_prior,
        "single_reuse_root": (
            str(Path(args.single_source_root)) if args.query_prior == "fresh" else None
        ),
        "l1_source": args.l1_source,
        "l1_ranking": args.l1_ranking,
        "max_level": args.max_level,
        "model": args.model,
        "model_identity": _model_identity(args.model),
        "base_url": args.base_url,
        "parallelism": args.parallelism,
        "endpoint_concurrency_budget": args.endpoint_concurrency_budget,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "thinking": "provider_default",
        "reasoning_effort": "omitted",
        "transport_max_retries": args.transport_max_retries,
        "tool_prefetch_complete": (
            True if args.profile == "context_records" else not args.skip_tool_prefetch
        ),
        "evaluation_indices_by_task": indices_by_task,
        "inputs": inputs,
        "l2_reuse": l2_reuse_audits or None,
        "started_at": _now(),
        "provider_routing": provider_config.public_dict(),
    }
    if args.profile == "context_records":
        prompt_manifests = {}
        for task in args.tasks:
            prompt_version = _context_prompt_version(args, task)
            prompt = context_records.prompt_profile(prompt_version)
            prompt_manifests[task] = {
                "version": prompt_version,
                "name": prompt["name"],
                "path": str(prompt["template"]),
                "sha256": sha256_file(prompt["template"]),
            }
        prompt_profile_manifest: Any
        prompt_template_manifest: Any
        if args.assay_transfer_prompt_version == "task_best":
            prompt_profile_manifest = prompt_manifests
            prompt_template_manifest = prompt_manifests
        else:
            only_prompt = next(iter(prompt_manifests.values()))
            prompt_profile_manifest = only_prompt["name"]
            prompt_template_manifest = {
                "path": only_prompt["path"],
                "sha256": only_prompt["sha256"],
            }
        context_selection = {
            "contexts": (
                f"top {args.context_limit} by Morgan similarity from the exact V9 top-100 candidate universe"
                if args.context_ranking == "morgan"
                else f"exact V9 top {args.context_limit}"
            ),
            "context_limit": args.context_limit,
            "record_limit_per_context_level": args.record_limit_per_context_level,
            "level_1": (
                f"up to {args.record_limit_per_context_level} deterministic "
                "current-gold constituent records per context"
            ),
            "level_2": (
                f"append up to {args.l2_record_limit_per_context} deterministic "
                "associated nonvoting records per context"
            ),
            "append_only": True,
        }
        if args.assay_transfer_cache_profile == V21_CACHE_PROFILE:
            if args.v21_selection_mode == "record_only":
                context_selection = {
                    "contexts": "none; independently ranked raw Stage 3 records",
                    "record_limits_by_level": _indirect_record_limits(
                        args, "bbb_martins"
                    ),
                    "levels": list(_record_level_names(args, "bbb_martins")),
                    "append_only": True,
                }
            else:
                context_selection = {
                    "contexts": (
                        f"top {args.context_limit} V21-ranked scaffold-disjoint "
                        "reference molecules"
                    ),
                    "context_limit": args.context_limit,
                    "record_limit_per_context_level": (
                        args.record_limit_per_context_level
                    ),
                    "l2_record_limit_per_context": (
                        args.l2_record_limit_per_context
                    ),
                    "within_context_record_ranking": "V21 transfer likelihood",
                    "later_record_limits_by_level": _indirect_record_limits(
                        args, "bbb_martins"
                    ),
                    "append_only": True,
                }
        if args.l2_record_limit_per_context != args.record_limit_per_context_level:
            context_selection["l2_record_limit_per_context"] = (
                args.l2_record_limit_per_context
            )
        if (
            _record_cache_enabled(args)
            and args.indirect_record_limit_per_level
            != context_records.INDIRECT_RECORD_LIMIT
        ):
            context_selection["indirect_record_limit_per_level"] = (
                args.indirect_record_limit_per_level
            )
        if _record_cache_enabled(args):
            context_selection["later_levels_by_task"] = {
                task: {
                    level: (
                        "append one bundle containing the top "
                        f"{_indirect_record_limits(args, task)[level]} "
                        + (
                            "Morgan-ranked records from Luna's top-quartile "
                            "relevance buckets"
                            if args.context_ranking == "morgan"
                            and args.assay_transfer_cache_profile
                            == LUNA_RELEVANCE_CACHE_PROFILE
                            else "assay-transfer-ranked records from Luna's "
                            "top-quartile relevance buckets"
                            if args.assay_transfer_cache_profile
                            == LUNA_RELEVANCE_CACHE_PROFILE
                            else "Morgan-ranked Stage 3 records"
                            if args.context_ranking == "morgan"
                            else "independently ranked Stage 3 records"
                        )
                    )
                    for level in _record_level_names(args, task)
                }
                for task in args.tasks
            }
        manifest.update(
            {
                "visibility_mode": "identity_blind_context_records",
                "reference_pool": (
                    "normalized_v7_stage3_l1_l5_records"
                    if args.assay_transfer_cache_profile == V21_CACHE_PROFILE
                    else "current_conditioned_gold_contexts_plus_luna_"
                    "top_quartile_non_direct_v7_records"
                    if args.assay_transfer_cache_profile
                    == LUNA_RELEVANCE_CACHE_PROFILE
                    else "current_conditioned_gold_contexts_plus_v7_gold_source_domain_rows"
                ),
                "neighbor_identity_policy": (
                    "scaffold_disjoint"
                    if args.assay_transfer_cache_profile == V21_CACHE_PROFILE
                    else "gold_split_parent_disjoint"
                ),
                "min_similarity": None,
                "candidate_generation": {
                    "unit": (
                        "individual_stage3_record"
                        if args.assay_transfer_cache_profile == V21_CACHE_PROFILE
                        and args.v21_selection_mode == "record_only"
                        else "v21_ranked_molecule_then_individual_stage3_record"
                        if args.assay_transfer_cache_profile == V21_CACHE_PROFILE
                        else "parent_condition_context_then_individual_stage3_record"
                        if args.context_record_l3_l5
                        else "parent_condition_context"
                    ),
                    "scope": (
                        "V21-ranked Stage 3 Morgan-top-75 molecules with V21-ranked records within L1/L2 cards, then independent L3-L5 records"
                        if args.assay_transfer_cache_profile == V21_CACHE_PROFILE
                        and args.v21_selection_mode == "molecule_cards"
                        else "independent Stage 3 Morgan-top-75 V21 rankings for L1-L5"
                        if args.assay_transfer_cache_profile == V21_CACHE_PROFILE
                        else "exact V9 L1/L2 contexts plus source-local Luna "
                        "top-quartile relevance filtering before Stage 3 "
                        "Morgan-top-75 candidate generation"
                        if args.assay_transfer_cache_profile
                        == LUNA_RELEVANCE_CACHE_PROFILE
                        else
                        "exact V9 Morgan-top-100 assignments for L1/L2; task-configured "
                        "Stage 3 Morgan-top-75 scaffold-disjoint assignments for later levels"
                        if args.context_ranking == "morgan"
                        else f"V9 ranks 0-{args.context_limit - 1} for L1/L2; task-configured independent Stage 3 "
                        "Morgan-top-75 V19.1 rankings for later levels"
                        if args.context_record_l3_l5
                        else "exact V9 Morgan-top-100 assignments for L1/L2"
                        if args.context_ranking == "morgan"
                        else "V9 ranks 0-9 from the full Morgan top-100 candidate pool"
                    ),
                    "ranking": (
                        {
                            "L1_L2": args.context_ranking,
                            "later_levels": args.context_ranking,
                        }
                        if _record_cache_enabled(args)
                        else args.context_ranking
                    ),
                    "structure_score_visible": args.context_ranking == "morgan",
                    "assay_transfer_scores_used": args.context_ranking == "assay_transfer",
                    "ranking_tie_seed": (
                        args.ranking_tie_seed
                        if args.context_ranking == "morgan"
                        else None
                    ),
                    "morgan_fingerprint": (
                        {"radius": 2, "bits": 2048, "similarity": "Tanimoto"}
                        if args.context_ranking == "morgan"
                        else None
                    ),
                },
                "selection": context_selection,
                "assay_transfer_cache_profile": args.assay_transfer_cache_profile,
                "v21_selection_mode": (
                    args.v21_selection_mode
                    if args.assay_transfer_cache_profile == V21_CACHE_PROFILE
                    else None
                ),
                "prompt_profile": prompt_profile_manifest,
                "prompt_template": prompt_template_manifest,
                "l1_source": (
                    "v21_ranked_stage3_direct_record_molecules"
                    if args.assay_transfer_cache_profile == V21_CACHE_PROFILE
                    and args.v21_selection_mode == "molecule_cards"
                    else "v21_independently_ranked_stage3_direct_records"
                    if args.assay_transfer_cache_profile == V21_CACHE_PROFILE
                    else "current_conditioned_gold_raw_records"
                ),
                "l1_ranking": args.context_ranking,
                "max_level": args.max_level,
                "analog_tool_policy": "disabled_by_context_records_profile",
            }
        )
    manifest["execution_providers"] = [
        _execution_provider_from_spec(
            spec,
            transport_max_retries=args.transport_max_retries,
        )
        for spec in provider_config.providers
    ]
    manifest_path = output_root / "experiment_manifest.json"
    if manifest_path.is_file():
        previous = _read_json(manifest_path)
        manifest = _merge_resume_manifest(previous, manifest)
    write_json_atomic(manifest_path, manifest)

    prepared_queries: list[PreparedQuery] = []
    for task in args.tasks:
        levels = _run_levels(args, task)
        index = None
        if args.profile == "standard" and (
            args.l1_source == "normalized_v7" or (
            args.l1_source == "gold_train" and len(levels) > 1
            )
        ):
            index = build_family_molecule_prefix_view(
                _load_progressive_index(task),
                levels=[int(row["level"]) for row in levels],
            )
        records = records_by_task[task]
        indices = indices_by_task[task]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.preparation_workers, len(indices))) as pool:
            if args.profile == "context_records":
                futures = {
                    pool.submit(
                        _prepare_context_record_query,
                        task=task,
                        query_index=query_index,
                        record=records[query_index],
                        contexts=context_candidates_by_task[task][
                            str(records[query_index]["benchmark_row_id"])
                        ],
                        levels=levels,
                        output_root=output_root,
                        single_root=Path(args.single_source_root),
                        query_prior_mode=args.query_prior,
                        record_limit=args.record_limit_per_context_level,
                        l2_record_limit=args.l2_record_limit_per_context,
                        indirect_record_limit=args.indirect_record_limit_per_level,
                        context_limit=args.context_limit,
                        indirect_record_limits=_indirect_record_limits(args, task),
                        indirect_records=(
                            indirect_candidates_by_task[task][
                                str(records[query_index]["benchmark_row_id"])
                            ]
                            if _record_cache_enabled(args)
                            else None
                        ),
                        l2_reuse_root=(
                            CONTEXT_L2_REUSE_ROOTS[task]
                            if args.reuse_context_l2
                            else None
                        ),
                        prompt_version=_context_prompt_version(args, task),
                        context_ranking=args.context_ranking,
                        ranking_tie_seed=args.ranking_tie_seed,
                        cache_profile=args.assay_transfer_cache_profile,
                        record_levels=(
                            _record_level_names(args, task)
                            if args.assay_transfer_cache_profile == V21_CACHE_PROFILE
                            else None
                        ),
                        v21_selection_mode=args.v21_selection_mode,
                    ): query_index
                    for query_index in indices
                }
            else:
                futures = {
                    pool.submit(
                        _prepare_query,
                        task=task,
                        query_index=query_index,
                        record=records[query_index],
                        index=index,
                        levels=levels,
                        gold_l1_candidates=(
                            gold_candidates_by_task[task][str(records[query_index]["benchmark_row_id"])]
                            if args.l1_source == "gold_train"
                            else None
                        ),
                        output_root=output_root,
                        single_root=Path(args.single_source_root),
                        query_prior_mode=args.query_prior,
                        l1_source=args.l1_source,
                        l1_ranking=args.l1_ranking,
                        tool_service_url=args.tool_service_url,
                        timeout_s=args.timeout_s,
                        prefetch_tools=not args.skip_tool_prefetch,
                    ): query_index
                    for query_index in indices
                }
            for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                prepared_queries.append(future.result())
                if completed % 10 == 0 or completed == len(futures):
                    print(f"[{task}] prepared {completed}/{len(futures)}", flush=True)

    prepared_queries.sort(key=lambda row: row.index)
    manifest["query_scheduling"] = "interleaved_by_query_index_across_tasks"
    manifest["prepared_at"] = _now()
    write_json_atomic(manifest_path, manifest)
    if args.prepare_only:
        return 0

    client = _make_client(args, provider_config)
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.parallelism, len(prepared_queries))) as pool:
        futures = [
            pool.submit(_run_query_safe, args, prepared, client)
            for prepared in prepared_queries
        ]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            failed += int(result.get("status") != "ok")
            if completed % 10 == 0 or completed == len(futures):
                print(f"[inference] completed {completed}/{len(futures)} failed={failed}", flush=True)

    for task in args.tasks:
        records = records_by_task[task]
        _summarize_task(
            task=task,
            records=records,
            indices=indices_by_task[task],
            output_root=output_root,
            max_level=args.max_level,
            query_prior_mode=args.query_prior,
            profile=args.profile,
            context_record_l3_l5=_record_cache_enabled(args),
        )
    manifest["finished_at"] = _now()
    manifest["n_failed_queries"] = failed
    manifest["provider_pool_final_snapshot"] = client.snapshot()
    write_json_atomic(manifest_path, manifest)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("standard", "context_records"),
        default="standard",
        help="Select the established molecule-card profile or the raw context-record L1/L2 profile.",
    )
    parser.add_argument(
        "--assay-transfer-prompt-version",
        choices=("task_best", *context_records.PROMPT_PROFILES),
        default="v1",
        help=(
            "Prompt/card contract for --profile context_records. V1 reproduces the "
            "first run; task_best selects BBB V4, Bioavailability V3, and Skin V6."
        ),
    )
    parser.add_argument(
        "--context-ranking",
        choices=("assay_transfer", "morgan"),
        default="assay_transfer",
        help="Rank context-record candidates with the frozen assay-transfer scores or Morgan similarity.",
    )
    parser.add_argument(
        "--assay-transfer-cache-profile",
        choices=("v19_1", V21_CACHE_PROFILE, LUNA_RELEVANCE_CACHE_PROFILE),
        default="v19_1",
        help=(
            "Use the established V19.1 cache, the opt-in V21 BBB raw-record "
            "cache, or the Luna relevance-top-quartile V19.1 cache."
        ),
    )
    parser.add_argument(
        "--v21-selection-mode",
        choices=("record_only", "molecule_cards"),
        default="record_only",
        help=(
            "For the BBB V21 cache, rank independent records at every level or "
            "group V21-ranked L1/L2 records into molecule cards before appending "
            "record-level L3-L5 evidence."
        ),
    )
    parser.add_argument(
        "--ranking-tie-seed",
        type=int,
        default=0,
        help="Seed for reproducible random ordering within equal Morgan similarities.",
    )
    parser.add_argument(
        "--context-record-l3-l5",
        action="store_true",
        help=(
            "After context-record L1/L2, append V19.1 Stage 3 record bundles "
            "at each task-configured later level."
        ),
    )
    parser.add_argument(
        "--reuse-context-l2",
        action="store_true",
        help=(
            "Reuse verified frozen task-best L1/L2 outputs, "
            "then begin new inference at L3."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASK_NAMES,
        default=["bioavailability_ma"],
        help=(
            "Tasks to run. The default is the task matching the current v10 output root; "
            "other tasks should use an explicit task and output root."
        ),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--trace-root", default=str(DEFAULT_TRACE_ROOT))
    parser.add_argument(
        "--benchmark-data-root",
        default="data/gold_labels",
        help="Canonical root containing versioned <Task> gold-label releases.",
    )
    parser.add_argument(
        "--evidence-root",
        default="",
        help=(
            "Optional paper evidence directory containing bbb_starling_v7 and "
            "bioavailability_starling_v7 and skin_reaction_starling_v7 compact indices."
        ),
    )
    parser.add_argument(
        "--l1-source",
        choices=("normalized_v7", "gold_train"),
        default="normalized_v7",
        help="Use the full normalized-v7 direct family or the matched conditioned gold-training L1 pool.",
    )
    parser.add_argument(
        "--l1-ranking",
        choices=("morgan", "v9"),
        default="morgan",
        help="Rank the matched gold L1 candidates by Morgan parent rank or frozen V9 transfer score.",
    )
    parser.add_argument("--v9-ranking-root", default=str(DEFAULT_V9_RANKING_ROOT))
    parser.add_argument(
        "--v7-root",
        default=str(context_records.V7_ROOT),
        help="Root containing <task>/v7/03_pair_buckets for the context-record profile.",
    )
    parser.add_argument(
        "--gold-l1-min-similarity",
        type=float,
        default=0.0,
        help="Minimum Morgan similarity for matched gold-training L1 candidates.",
    )
    parser.add_argument(
        "--context-limit",
        type=int,
        default=context_records.CONTEXT_LIMIT,
        help="Number of V9 parent-condition contexts retained at L1 and L2.",
    )
    parser.add_argument(
        "--record-limit-per-context-level",
        type=int,
        default=context_records.RECORD_LIMIT,
        help="Maximum raw L1 records shown per context.",
    )
    parser.add_argument(
        "--l2-record-limit-per-context",
        type=int,
        default=0,
        help="New L2 records per context; 0 uses the L1 record limit.",
    )
    parser.add_argument(
        "--indirect-record-limit-per-level",
        type=int,
        default=context_records.INDIRECT_RECORD_LIMIT,
        help="Maximum independently ranked Stage 3 records appended at each level from L3 onward.",
    )
    parser.add_argument(
        "--indirect-final-level-record-limit",
        type=int,
        default=0,
        help="Optional record cap for only the task's final indirect level; 0 uses the common cap.",
    )
    parser.add_argument(
        "--query-prior",
        choices=("fresh", "none"),
        default="fresh",
        help="Reuse a freshly generated same-checkpoint main-scaffold prior, or run standalone L1.",
    )
    parser.add_argument(
        "--max-level",
        type=int,
        default=0,
        help="Stop after this progressive level; 0 runs every configured level.",
    )
    parser.add_argument("--single-source-root", default=str(DEFAULT_SINGLE_CACHE_ROOT))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument(
        "--provider-pool-config",
        default="",
        help=(
            "JSON provider-pool config. Each provider supplies its own base URL, "
            "model alias, API-key env name, and max in-flight budget."
        ),
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional KEY=VALUE file used to resolve provider API-key env names.",
    )
    parser.add_argument("--tool-service-url", default="http://127.0.0.1:8765")
    parser.add_argument("--parallelism", type=int, default=128)
    parser.add_argument(
        "--endpoint-concurrency-budget",
        type=int,
        default=512,
        help="Explicit endpoint-wide in-flight ceiling shared by the whole run.",
    )
    parser.add_argument("--preparation-workers", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=20_480)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument(
        "--transport-max-retries",
        type=int,
        default=0,
        help=(
            "HTTP transport retries inside one level call. Default 0 avoids duplicate long "
            "generations; resume retries the missing checkpoint explicitly."
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--indices", nargs="*", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--skip-tool-prefetch",
        action="store_true",
        help="Diagnostic preparation only; formal inference requires visible prefetched tools.",
    )
    args = parser.parse_args(argv)
    _configure_v7_paths(args)
    load_env_file(args.env_file)
    if not 1 <= args.endpoint_concurrency_budget <= MAX_ENDPOINT_CONCURRENCY_BUDGET:
        parser.error(f"--endpoint-concurrency-budget must be between 1 and {MAX_ENDPOINT_CONCURRENCY_BUDGET}")
    if args.parallelism < 1 or args.parallelism > args.endpoint_concurrency_budget:
        parser.error("--parallelism must be positive and not exceed --endpoint-concurrency-budget")
    if args.preparation_workers < 1:
        parser.error("--preparation-workers must be positive")
    if args.transport_max_retries < 0:
        parser.error("--transport-max-retries must be non-negative")
    if args.max_level < 0:
        parser.error("--max-level must be non-negative")
    if not 1 <= args.context_limit <= 100:
        parser.error("--context-limit must be between 1 and 100")
    if args.record_limit_per_context_level < 1:
        parser.error("--record-limit-per-context-level must be positive")
    if args.l2_record_limit_per_context < 0:
        parser.error("--l2-record-limit-per-context must be non-negative")
    if args.l2_record_limit_per_context == 0:
        args.l2_record_limit_per_context = args.record_limit_per_context_level
    if args.indirect_record_limit_per_level < 1:
        parser.error("--indirect-record-limit-per-level must be positive")
    if args.indirect_final_level_record_limit < 0:
        parser.error("--indirect-final-level-record-limit must be non-negative")
    if not 0 <= args.gold_l1_min_similarity <= 1:
        parser.error("--gold-l1-min-similarity must be between 0 and 1")
    if args.profile == "context_records":
        if args.assay_transfer_cache_profile == V21_CACHE_PROFILE:
            if args.tasks != ["bbb_martins"]:
                parser.error("the V21 L1-L5 cache profile only supports --tasks bbb_martins")
            if args.context_ranking != "assay_transfer":
                parser.error(
                    "the V21 L1-L5 cache profile requires "
                    "--context-ranking assay_transfer"
                )
            if args.context_record_l3_l5:
                parser.error(
                    "the V21 cache profile supplies its configured record levels "
                    "without --context-record-l3-l5"
                )
            if args.reuse_context_l2:
                parser.error("the V21 L1-L5 cache profile cannot reuse V9 L1/L2 outputs")
        if args.assay_transfer_cache_profile == LUNA_RELEVANCE_CACHE_PROFILE:
            unsupported = set(args.tasks) - {"bbb_martins", "skin_reaction"}
            if unsupported:
                parser.error(
                    "the Luna relevance cache only supports BBB and Skin"
                )
            if not args.context_record_l3_l5:
                parser.error(
                    "the Luna relevance cache requires --context-record-l3-l5"
                )
        if args.assay_transfer_prompt_version != "task_best":
            prompt_ranking = str(
                context_records.prompt_profile(
                    args.assay_transfer_prompt_version
                ).get("ranking")
                or "assay_transfer"
            )
            if prompt_ranking != args.context_ranking:
                parser.error(
                    "--assay-transfer-prompt-version does not match --context-ranking"
                )
        if args.reuse_context_l2 and not args.context_record_l3_l5:
            parser.error("--reuse-context-l2 requires --context-record-l3-l5")
        if args.reuse_context_l2 and (
            args.context_limit != context_records.CONTEXT_LIMIT
            or
            args.record_limit_per_context_level != 10
            or args.l2_record_limit_per_context != 10
        ):
            parser.error("--reuse-context-l2 requires the frozen L1/L2 record caps of 10")
        if args.indirect_final_level_record_limit and not _record_cache_enabled(args):
            parser.error(
                "--indirect-final-level-record-limit requires --context-record-l3-l5"
            )
        if _record_cache_enabled(args):
            supported = set(
                context_records.level_record_bundle_contract()["task_levels"]
            )
            unsupported = set(args.tasks) - supported
            if unsupported:
                parser.error(
                    "cache-backed later levels are unsupported for "
                    + ", ".join(sorted(unsupported))
                )
            too_high = {
                task: _record_level_names(args, task)[-1]
                for task in args.tasks
                if args.max_level
                > int(_record_level_names(args, task)[-1][1:])
            }
            if too_high:
                parser.error(
                    "--max-level exceeds configured cache levels: "
                    + ", ".join(
                        f"{task} ends at {level}"
                        for task, level in too_high.items()
                    )
                )
        elif args.max_level > 2:
            parser.error(
                "--profile context_records has only L1/L2 unless "
                "--context-record-l3-l5 is set"
            )
        if args.assay_transfer_prompt_version == "task_best":
            unsupported = set(args.tasks) - set(
                context_records.TASK_BEST_PROMPT_VERSIONS
            )
            if unsupported:
                parser.error(
                    "--assay-transfer-prompt-version task_best is unsupported for "
                    + ", ".join(sorted(unsupported))
                )
        if args.query_prior != "fresh":
            parser.error("--profile context_records currently requires --query-prior fresh")
        if args.skip_tool_prefetch:
            parser.error("context_records disables analog tools internally; do not use --skip-tool-prefetch")
    else:
        if args.assay_transfer_cache_profile != "v19_1":
            parser.error("--assay-transfer-cache-profile requires --profile context_records")
        if args.context_ranking != "assay_transfer" or args.ranking_tie_seed != 0:
            parser.error("--context-ranking and --ranking-tie-seed require --profile context_records")
        if args.reuse_context_l2:
            parser.error("--reuse-context-l2 requires --profile context_records")
        if args.context_record_l3_l5:
            parser.error("--context-record-l3-l5 requires --profile context_records")
        if args.assay_transfer_prompt_version != "v1":
            parser.error("--assay-transfer-prompt-version applies only to --profile context_records")
        if args.l1_source == "gold_train":
            if not args.evidence_root:
                parser.error("--l1-source gold_train requires --evidence-root")
            if args.max_level < 1:
                parser.error("--l1-source gold_train requires an explicit positive --max-level")
        elif args.l1_ranking != "morgan":
            parser.error("--l1-ranking v9 requires --l1-source gold_train")
        elif args.gold_l1_min_similarity:
            parser.error("--gold-l1-min-similarity requires --l1-source gold_train")
    if args.skip_tool_prefetch and not args.prepare_only:
        parser.error("--skip-tool-prefetch is allowed only with --prepare-only")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
