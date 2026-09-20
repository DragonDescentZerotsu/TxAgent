"""Run cache-matched Reranked Progressive v2 through one resumable LLM loop.

The default shared prompt uses --reranking assay-transfer, joint, or morgan.
L1 selects molecule cards; later stages independently select records before
merging into append-only molecule cards. Cache configuration supplies scores,
not stage policy. Preparation assembles evidence/tools; the shared model pool
executes dependent levels and checkpoints each query. No cache builder runs here.

Historical library-level helpers remain for archived artifact inspection, not CLI dispatch.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
from functools import lru_cache
from typing import Any, Mapping

import pyarrow.parquet as pq

from predict.harnesses.progressive.retrieval import (
    ASSAY_MECHANISM_TAGGED_PROMPT_PROFILE,
    build_family_molecule_prefix_view,
    retrieve_family_molecule_prefixes,
)
from predict.harnesses.progressive import _records as context_records
from predict.retrieval.retrieve import load_index
from predict.retrieval.assay_reranking.runtime import cache_profile_root
from predict.retrieval.assay_reranking.runtime import ARCHIVE_CACHE_ROOT
from predict.retrieval.assay_reranking.progressive_levels import (
    LUNA_RELEVANCE_CACHE_PROFILE,
    V21_CACHE_PROFILE,
    load_top_ranked_records,
    load_v21_molecule_card_records,
)
from predict.retrieval.assay_reranking.v24_1_levels import (
    PROFILE as V24_1_LEVEL_CACHE_PROFILE,
    load_top_ranked_records as load_v24_1_level_records,
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
from predict.api_client.pool import (
    DEFAULT_PROVIDER_POOL_CONFIG,
    OpenAIProviderPool,
    ProviderPoolConfig,
    ProviderSpec,
    build_provider_pool,
    load_provider_pool_config,
    preflight_sglang_tokenized_completion,
    select_healthy_providers,
)
from predict.harnesses.progressive.state import (
    MOLECULE_CARD_CONTRACT_PATH,
    PROGRESSIVE_PROTOCOL_VERSION,
    ProgressiveTaskContract,
    append_evidence,
    attach_analog_tool_summaries,
    card_alias_maps,
    card_ids,
    extract_cumulative_evidence,
    molecule_card_contract,
    select_initial_evidence,
    select_progressive_delta,
    stable_analog_id,
)
from predict.tools.client import ToolServiceClient
from predict.tools.prefetch import invoke_with_retry
from predict.traces.io import DEFAULT_TRACE_ROOT, write_trace
from predict.llm_io.query import external_condition_sentence
from predict.llm_io.response import structured_response_is_valid
from data.processing.gold_labels.conditioned_benchmark import (
    TASK_DIRECTORIES,
    split_path,
    tdc_split_path,
)
from data.processing.llm_api import DEFAULT_ENV_FILE
from predict.harnesses.progressive.tasks import bbb_martins as bbb_config
from predict.harnesses.progressive.tasks import bioavailability_ma as bio_config
from predict.harnesses.progressive.tasks import skin_reaction as skin_config


MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
BASE_URL = "https://litellm.parcc.upenn.edu/v1"
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/paper/starling_conditioned_assay_progressive_visible_v10_bio_legacy_gold_vote_pure_v1/"
    "scaffold_valid_deepseek_v4_flash_0731"
)
DEFAULT_SINGLE_CACHE_ROOT = Path(
    "outputs/paper/legacy/starling_conditioned_gold_l1_deepseek_v4_flash_nvfp4_query_prior/"
    "runs_deployment_visible_parent_disjoint"
)
DEFAULT_V9_RANKING_ROOT = cache_profile_root(RANKING_PROFILE_NAME)
TDC_MIXED_L1_HARNESS = "tdc-mixed-progressive-v1"
TDC_MIXED_L1_PROFILE = "tdc_mixed_l1_v1"
TDC_MIXED_L1_PROMPT = "tdc_mixed_progressive_v1"
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
MOLECULE_DESCRIPTION_PATH = Path(
    "/vast/projects/myatskar/design-documents/canonical_smiles_quotient.parquet"
)
MOLECULE_DESCRIPTION_SHA256 = (
    "abd6f6d31ee74d854fd42330e816516c39a5ca63c67df56965cc8ec44468f183"
)
MOLECULE_DESCRIPTION_ARTIFACTS = {
    "v1": {
        "path": MOLECULE_DESCRIPTION_PATH,
        "sha256": MOLECULE_DESCRIPTION_SHA256,
    },
    "v2": {
        "path": Path(
            "/vast/projects/myatskar/design-documents/"
            "canonical_smiles_quotient_v2.parquet"
        ),
        "sha256": (
            "3e1ca4ece0f137b492c4d6c60713cf9771fbab7871a11b4c6c126df674b0abb7"
        ),
    },
}
MOLECULE_DESCRIPTION_COLUMNS = {
    "raw": "description_raw",
    "motif": "description_motif",
    "coarse": "description_coarse",
}
MOLECULE_DESCRIPTION_PROMPTS = {
    "reranked-progressive-l1-context-v1": (
        "reranked_progressive_l1_context_v3",
        "reranked_progressive_l1_context_v4",
    ),
    "reranked-progressive-l1-context-l2-v1": (
        "reranked_progressive_l1_context_l2_semantic_molecule_metadata_v1",
    ),
    "reranked-progressive-l1-context-l2-weighted-v1": (
        "reranked_progressive_l1_context_l2_weighted_molecule_metadata_v1",
    ),
}
SQLITE_SELECTION_CONTRACTS = frozenset({
    "cache_matched_retrieval.v2",
    "cache_matched_retrieval.v3",
    "ranked_evidence_retrieval.v1",
    "ranked_level_retrieval.v2",
    "ranked_uid_retrieval.v1",
    "semantic_bucket_reranking.v1",
    "l1_context_retrieval.v1",
    "l1_context_retrieval.v2",
    "l1_context_semantic_l2.v1",
    "l1_context_semantic_weighted_l2.v1",
    "l1_context_morgan_semantic_l2.v1",
    "l1_context_morgan_semantic_l2.v2",
    "l1_context_morgan_semantic_l2.v3",
    "l1_context_morgan_semantic_l2.v4",
    "indirect_morgan_semantic_l2_l4.v1",
    "indirect_morgan_semantic_l2_l4.v2",
    "indirect_morgan_semantic_l2_l4.v3",
})
INDIRECT_ONLY_HARNESS = "reranked-progressive-indirect-only-v1"
INDIRECT_FILTER_HARNESS = "reranked-progressive-indirect-only-v2"
INDIRECT_HARNESSES = frozenset({INDIRECT_ONLY_HARNESS, INDIRECT_FILTER_HARNESS})
FULL_FLAT_PROGRESSIVE_HARNESS = "full-flat-progressive-v1"
FULL_FLAT_PROGRESSIVE_HARNESSES = {
    FULL_FLAT_PROGRESSIVE_HARNESS: "full_flat_progressive_v1",
    "full-flat-progressive-v2": "full_flat_progressive_v2",
    "full-flat-progressive-v3": "full_flat_progressive_v3",
}
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
    "evaluation_subset",
    "tasks",
    "visibility_mode",
    "reference_pool",
    "neighbor_identity_policy",
    "selection",
    "candidate_generation",
    "prompt_profile",
    "prompt_template",
    "molecule_card_contract",
    "molecule_description",
    "max_tokens",
    "timeout_s",
    "temperature",
    "thinking",
    "reasoning_effort",
    "tool_prefetch_complete",
    "evaluation_indices_by_task",
    "inputs",
    "l2_reuse",
    "reasoning_phase",
    "l1_prior_run",
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
    if previous.get('prompt_assets') is not None and previous['prompt_assets'] != current.get('prompt_assets'):
        raise ValueError('cannot resume progressive run with changed prompt_assets')
    if previous.get('prompt_assets') is None:
        for assets in (current.get('prompt_assets') or {}).values():
            provenance = _read_json(Path(assets['directory']) / 'provenance.json')
            if assets['files_sha256'] != provenance.get('migration_files_sha256'):
                raise ValueError('cannot resume a legacy run with prompt assets changed after migration')
            if assets['assembly_files_sha256'] != provenance.get('migration_assembly_files_sha256'):
                raise ValueError('cannot resume a legacy run with prompt assembly changed after migration')
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


def _load_molecule_descriptions(
    mode: str,
    *,
    path: Path = MOLECULE_DESCRIPTION_PATH,
    expected_sha256: str = MOLECULE_DESCRIPTION_SHA256,
) -> dict[str, Any] | None:
    """Load one exact Quotient text surface; selected-row coverage is checked later."""
    if mode == "none":
        return None
    try:
        column = MOLECULE_DESCRIPTION_COLUMNS[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported molecule description mode: {mode}") from exc
    path = path.resolve()
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"molecule description Parquet hash mismatch: {path}")
    schema = pq.ParquetFile(path).schema_arrow
    expected_columns = [
        "canonical_smiles",
        "description_raw",
        "description_motif",
        "description_coarse",
        "error",
    ]
    if schema.names != expected_columns or any(
        str(field.type) not in {"string", "large_string"} for field in schema
    ):
        raise ValueError(f"molecule description Parquet has an incompatible schema: {path}")
    entries: dict[str, dict[str, str | None]] = {}
    for row in pq.read_table(path, columns=["canonical_smiles", column, "error"]).to_pylist():
        smiles = row["canonical_smiles"]
        if not isinstance(smiles, str) or not smiles or smiles in entries:
            raise ValueError("molecule description Parquet has blank or duplicate SMILES")
        entries[smiles] = {
            "description": row[column],
            "error": row["error"],
        }
    return {
        "mode": mode,
        "column": column,
        "path": str(path),
        "sha256": actual_sha256,
        "row_count": len(entries),
        "entries": entries,
    }


def _validate_molecule_description_coverage(
    cache: Mapping[str, Any],
    *,
    records_by_task: Mapping[str, list[Mapping[str, Any]]],
    indices_by_task: Mapping[str, list[int]],
    contexts_by_task: Mapping[str, Mapping[str, list[Mapping[str, Any]]]],
    later_by_task: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    receipt_path: Path,
    missing_policy: str = "error",
) -> dict[str, str]:
    """Audit selected-row coverage, failing only for strict prompt contracts."""
    if missing_policy not in {"error", "omit"}:
        raise ValueError(f"unsupported molecule description missing policy: {missing_policy}")
    references: dict[str, list[dict[str, Any]]] = {}
    counts_by_task: dict[str, dict[str, int]] = {}
    for task, indices in indices_by_task.items():
        task_queries: set[str] = set()
        task_evidence: set[str] = set()
        for index in indices:
            record = records_by_task[task][index]
            query_id = str(record["benchmark_row_id"])
            query_smiles = str(record.get("drug") or "")
            references.setdefault(query_smiles, []).append(
                {"task": task, "query_index": index, "benchmark_row_id": query_id,
                 "role": "query"}
            )
            task_queries.add(query_smiles)
            for context in contexts_by_task[task][query_id]:
                smiles = str(context.get("canonical_smiles") or "")
                references.setdefault(smiles, []).append(
                    {"task": task, "query_index": index, "benchmark_row_id": query_id,
                     "role": "L1"}
                )
                task_evidence.add(smiles)
            for level_name, selection in later_by_task[task][query_id].items():
                for row in selection.get("records") or []:
                    payload = row.get("payload") or {}
                    smiles = str(
                        payload.get("canonical_smiles")
                        or row.get("reference_parent_smiles")
                        or ""
                    )
                    references.setdefault(smiles, []).append(
                        {"task": task, "query_index": index, "benchmark_row_id": query_id,
                         "role": level_name}
                    )
                    task_evidence.add(smiles)
        counts_by_task[task] = {
            "queries": len(indices),
            "unique_query_molecules": len(task_queries),
            "unique_evidence_molecules": len(task_evidence),
        }

    entries = cache["entries"]
    invalid = []
    values: dict[str, str] = {}
    for smiles, refs in references.items():
        entry = entries.get(smiles)
        reason = None
        if entry is None:
            reason = "missing"
        elif isinstance(entry.get("error"), str) and entry["error"].strip():
            reason = "error"
        elif not isinstance(entry.get("description"), str) or not entry["description"].strip():
            reason = "blank_description"
        else:
            values[smiles] = entry["description"]
        if reason:
            invalid.append({
                "canonical_smiles": smiles,
                "reason": reason,
                "error": entry.get("error") if entry else None,
                "references": refs,
            })
    status = "failed" if invalid and missing_policy == "error" else (
        "ok_with_omissions" if invalid else "ok"
    )
    receipt = {
        "schema_version": "quotient_molecule_metadata_preflight.v1",
        "status": status,
        "missing_policy": missing_policy,
        "mode": cache["mode"],
        **(
            {"cache_version": cache["cache_version"]}
            if cache.get("cache_version") else {}
        ),
        "column": cache["column"],
        "path": cache["path"],
        "sha256": cache["sha256"],
        "parquet_rows": cache["row_count"],
        "selected_unique_molecules": len(references),
        "valid_unique_molecules": len(values),
        "counts_by_task": counts_by_task,
        "invalid": invalid,
        "checked_at": _now(),
    }
    write_json_atomic(receipt_path, receipt)
    if invalid and missing_policy == "error":
        raise ValueError(
            f"molecule description preflight failed for {len(invalid)} selected molecules; "
            f"see {receipt_path}"
        )
    return values


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _levels(task: str, max_level: int = 0) -> list[dict[str, Any]]:
    from predict.harnesses.progressive.prompt import prompt_assets
    descriptions = prompt_assets('standard_v1')['level_descriptions'][task]
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
        endpoint_descriptions = descriptions['by_endpoint']
        row["description"] = (
            descriptions['v7_index'].get(level)
            if task == "bbb_martins" and PROGRESSIVE_TASKS[task].index.is_dir()
            else endpoint_descriptions.get(endpoint)
            or descriptions['by_level'][level]
        )
    if [int(row["level"]) for row in levels] != list(range(1, len(levels) + 1)):
        raise ValueError(f"{task} has a non-contiguous family-level catalog")
    return levels[:max_level] if max_level else levels


def _run_levels(args: argparse.Namespace, task: str) -> list[dict[str, Any]]:
    """Dispatch only the level catalog; each profile owns its level meaning."""
    if args.harness_version == TDC_MIXED_L1_HARNESS:
        return [{
            "level": 1,
            "endpoint_group": "tdc_mixed_training_labels",
            "description": (
                "Morgan-ranked source-specific labels from the union of "
                "Gold-v1 and TDC training cards."
            ),
        }]
    if args.harness_version in FULL_FLAT_PROGRESSIVE_HARNESSES:
        levels = context_records.tianang_aligned_levels(
            task, prompt_version=_context_prompt_version(args, task)
        )
        return levels[:1] if args.reasoning_phase == "l1" else [levels[0], levels[-1]]
    if args.profile == "context_records":
        if getattr(args, 'retrieval_policy', None):
            levels = context_records.tianang_aligned_levels(task, args.max_level,
                prompt_version=args.assay_transfer_prompt_version)
            selected = set(args.retrieval_policies[task]['stages'])
            return [row for row in levels if f"L{row['level']}" in selected]
        if args.assay_transfer_cache_profile == V24_1_LEVEL_CACHE_PROFILE:
            return context_records.tianang_aligned_levels(
                task,
                args.max_level or 4,
                prompt_version=_context_prompt_version(args, task),
            )
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
        or args.assay_transfer_cache_profile == V24_1_LEVEL_CACHE_PROFILE
    )


def _record_level_names(args: argparse.Namespace, task: str) -> tuple[str, ...]:
    if getattr(args, 'retrieval_policy', None):
        return tuple(k for k in args.retrieval_policies[task]['stages'] if k != 'L1')
    if args.assay_transfer_cache_profile == V24_1_LEVEL_CACHE_PROFILE:
        return ("L2", "L3", "L4")
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
    for level, value in getattr(args, "level_record_limits", {}).items():
        if level in limits:
            limits[level] = value
    return limits


def _level_record_limit(value: str) -> tuple[str, int]:
    level, separator, raw_limit = value.partition("=")
    if not separator or level not in {"L2", "L3", "L4", "L5", "L6"}:
        raise argparse.ArgumentTypeError(
            "level record limits must use LEVEL=K for L2 through L6"
        )
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("level record limits must be integers") from exc
    if limit < 1:
        raise argparse.ArgumentTypeError("level record limits must be positive")
    return level, limit


def _run_protocol(args: argparse.Namespace) -> str:
    if (
        args.profile == "context_records"
        and args.assay_transfer_prompt_version != "task_best"
        and context_records.is_tianang_aligned(args.assay_transfer_prompt_version)
    ):
        return context_records.TIANANG_ALIGNED_PROTOCOL_VERSION
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


def _is_tianang_aligned(args: argparse.Namespace, task: str) -> bool:
    return context_records.is_tianang_aligned(_context_prompt_version(args, task))


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
    contract = _task_contract(task, prompt_version)
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


def _reuse_full_flat_l1_output(
    *, query_dir: Path, source_query_dir: Path,
) -> None:
    """Reuse one stable-identity L1 result after its visible inputs match."""
    level_dir = query_dir / "levels" / "level_1"
    source_level_dir = source_query_dir / "levels" / "level_1"
    prepared = _read_json(level_dir / "prepared.json")
    source_prepared = _read_json(source_level_dir / "prepared.json")
    fields = (
        "task", "benchmark_row_id", "molecule_identity_key", "condition_group",
        "query_smiles", "condition_sentence", "query_prior", "active_evidence",
    )
    for field in fields:
        if prepared.get(field) != source_prepared.get(field):
            raise ValueError(
                f"full-flat L1 reuse differs for {prepared.get('benchmark_row_id')}: {field}"
            )
    source_output_path = source_level_dir / "output.json"
    source_request_path = source_level_dir / "request.json"
    source_output = _read_json(source_output_path)
    if (
        source_output.get("status") not in COMPLETE_LEVEL_STATUSES
        or not isinstance(source_output.get("state"), Mapping)
        or not source_request_path.is_file()
    ):
        raise ValueError(f"full-flat L1 source is incomplete: {source_level_dir}")
    reused = dict(source_output)
    reused.update({
        "status": "reused",
        "model_called": False,
        "source_model_called": source_output.get("model_called") is True,
        "reused_from": str(source_output_path),
        "reused_output_sha256": sha256_file(source_output_path),
        "reused_at": _now(),
    })
    write_json_atomic(level_dir / "output.json", reused)
    shutil.copy2(source_request_path, level_dir / "request.json")


def _validate_full_flat_l1_source(
    args: argparse.Namespace,
    records_by_task: Mapping[str, list[Mapping[str, Any]]],
    indices_by_task: Mapping[str, list[int]],
) -> tuple[dict[str, dict[str, Path]], dict[str, Any]]:
    root = Path(args.l1_prior_run).resolve()
    manifest_path = root / "experiment_manifest.json"
    manifest = _read_json(manifest_path)
    source_status = manifest.get("status")
    if source_status is None and (root / "run.json").is_file():
        source_status = _read_json(root / "run.json").get("status")
    source_harness = manifest.get("harness_version")
    source_prompt = manifest.get("prompt_profile")
    same_prompt_contract = (
        source_harness == args.harness_version
        and source_prompt == args.prompt_version
    )
    metadata_only_successor = (
        source_harness == "full-flat-progressive-v2"
        and source_prompt == "full_flat_progressive_v2"
        and args.harness_version == "full-flat-progressive-v3"
        and args.prompt_version == "full_flat_progressive_v3"
    )
    if (
        not (same_prompt_contract or metadata_only_successor)
        or manifest.get("reasoning_phase") != "l1"
        or _model_identity(str(manifest.get("model") or "")) != _model_identity(args.model)
    ):
        raise ValueError(f"incompatible full-flat L1 prior run: {root}")
    from predict.harnesses.progressive.prompt import prompt_asset_manifest

    current_prompt = prompt_asset_manifest(args.prompt_version)
    source_prompts = manifest.get("prompt_assets") or {}
    def compatible_prompt(source: Mapping[str, Any]) -> bool:
        if same_prompt_contract:
            return source.get("sha256") == current_prompt["sha256"]
        source_files = dict(source.get("files_sha256") or {})
        current_files = dict(current_prompt.get("files_sha256") or {})
        source_files.pop("provenance.json", None)
        current_files.pop("provenance.json", None)
        return source_files == current_files

    if any(not compatible_prompt(source_prompts.get(task) or {}) for task in args.tasks):
        raise ValueError(f"full-flat L1 prompt assets differ from the current bundle: {root}")
    sources: dict[str, dict[str, Path]] = {}
    counts: dict[str, int] = {}
    missing_counts: dict[str, int] = {}
    incomplete_counts: dict[str, int] = {}
    for task in args.tasks:
        expected_input_sha256 = sha256_file(PROGRESSIVE_TASKS[task].input_jsonl)
        if (manifest.get("inputs", {}).get(task) or {}).get(
            "input_sha256"
        ) != expected_input_sha256:
            raise ValueError(f"{task} L1 prior input hash differs: {root}")
        expected = {
            str(records_by_task[task][index]["benchmark_row_id"])
            for index in indices_by_task[task]
        }
        discovered: dict[str, Path] = {}
        for prepared_path in sorted(
            (root / task / "queries").glob("query_idx*/levels/level_1/prepared.json")
        ):
            prepared = _read_json(prepared_path)
            query_id = str(prepared.get("benchmark_row_id") or "")
            if not query_id or query_id in discovered:
                raise ValueError(f"duplicate or blank L1 prior identity in {root}: {query_id!r}")
            discovered[query_id] = prepared_path.parents[2]
        extra = set(discovered) - expected
        if extra:
            raise ValueError(
                f"{task} L1 prior contains {len(extra)} unexpected query identities"
            )
        task_sources: dict[str, Path] = {}
        incomplete = 0
        for query_id, source_query_dir in discovered.items():
            level_dir = source_query_dir / "levels/level_1"
            output_path = level_dir / "output.json"
            request_path = level_dir / "request.json"
            if not output_path.is_file() or not request_path.is_file():
                incomplete += 1
                continue
            output = _read_json(output_path)
            if (
                output.get("status") not in COMPLETE_LEVEL_STATUSES
                or not isinstance(output.get("state"), Mapping)
            ):
                incomplete += 1
                continue
            task_sources[query_id] = source_query_dir
        missing = len(expected - set(discovered))
        if args.require_complete_l1_prior and (
            source_status != "complete" or missing or incomplete
        ):
            raise ValueError(
                f"{task} L1 prior is not complete: status={source_status!r}, "
                f"missing={missing}, incomplete={incomplete}"
            )
        sources[task] = task_sources
        counts[task] = len(task_sources)
        missing_counts[task] = missing
        incomplete_counts[task] = incomplete
    return sources, {
        "path": str(root),
        "experiment_manifest_sha256": sha256_file(manifest_path),
        "source_status": source_status,
        "prompt_assets_sha256": current_prompt["sha256"],
        "reused_counts_by_task": counts,
        "missing_counts_by_task": missing_counts,
        "incomplete_counts_by_task": incomplete_counts,
        "require_complete": args.require_complete_l1_prior,
    }


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


def _configure_evaluation_subset(args: argparse.Namespace) -> None:
    for task in args.tasks:
        spec = PROGRESSIVE_TASKS[task]
        if args.harness_version == TDC_MIXED_L1_HARNESS:
            input_jsonl = tdc_split_path(task, args.evaluation_subset)
        elif args.gold_label_version == "current":
            input_jsonl = split_path(task, args.evaluation_subset)
        else:
            input_jsonl = (
                Path(args.benchmark_data_root)
                / TASK_DIRECTORIES[task]
                / args.gold_label_version
                / "scaffold"
                / f"{args.evaluation_subset}.jsonl"
            )
        if not input_jsonl.is_file():
            raise FileNotFoundError(input_jsonl)
        PROGRESSIVE_TASKS[task] = ProgressiveTaskSpec(
            input_jsonl, spec.index, spec.family_manifest
        )


def _task_contract(task: str, prompt_version: str = 'standard_v1') -> ProgressiveTaskContract:
    try:
        from predict.harnesses.progressive.prompt import prompt_assets
        values = dict(prompt_assets(prompt_version)['tasks'][task])
        values['task_instructions'] = tuple(values['task_instructions'])
        return ProgressiveTaskContract(**values)
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
        run_id=(getattr(args, "live_run_ids", {}) or {}).get(prepared_query.task, ""),
        method=getattr(args, "live_method", "progressive"),
        run_dir=(getattr(args, "live_run_dirs", {}) or {}).get(prepared_query.task),
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


@dataclass(frozen=True)
class _QueryPriorSource:
    run_dir: str
    source_index: int


@lru_cache(maxsize=None)
def _single_source_index(
    task: str, single_root_text: str
) -> dict[tuple[str, str], _QueryPriorSource]:
    single_root = Path(single_root_text)
    batch_dir = _single_batch_dir(task, single_root)
    manifest_path = batch_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") == "progressive_query_prior_overlay.v1":
        base_root = Path(str(manifest["base_root"]))
        base_manifest_path = _single_batch_dir(task, base_root) / "manifest.json"
        if sha256_file(base_manifest_path) != manifest.get("base_manifest_sha256"):
            raise ValueError(f"{task} query-prior base manifest hash mismatch")
        input_path = Path(str(manifest["base_input_jsonl"]))
        if sha256_file(input_path) != manifest.get("base_input_sha256"):
            raise ValueError(f"{task} query-prior base input hash mismatch")
        records = read_jsonl(input_path)
        mapping = {
            _stable_query_key(record): _QueryPriorSource(
                str(_source_run_dir(task, index, base_root)), index
            )
            for index, record in enumerate(records)
        }
        if len(mapping) != len(records):
            raise ValueError(f"duplicate reusable single identity in {input_path}")
        for override in manifest.get("overrides") or []:
            key = (str(override["molecule_identity_key"]), str(override["condition_group"]))
            if key in mapping:
                raise ValueError(f"duplicate query-prior overlay identity: {key}")
            run_dir = Path(str(override["run_dir"]))
            for name, expected in (override.get("files_sha256") or {}).items():
                if sha256_file(run_dir / name) != expected:
                    raise ValueError(f"{task} query-prior override hash mismatch: {name}")
            mapping[key] = _QueryPriorSource(str(run_dir), int(override["source_index"]))
        return mapping
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
    mapping: dict[tuple[str, str], _QueryPriorSource] = {}
    for index, record in enumerate(records):
        key = _stable_query_key(record)
        if key in mapping:
            raise ValueError(f"duplicate reusable single identity: {key}")
        mapping[key] = _QueryPriorSource(
            str(_source_run_dir(task, index, single_root)), index
        )
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


def _tdc_mixed_l1_candidates(
    *,
    task: str,
    records: list[dict[str, Any]],
    cache_root: Path,
    subset: str,
    prompt_version: str,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    """Load an immutable Morgan top-10 union of Gold-v1 and TDC label cards."""

    repository_root = Path(__file__).resolve().parents[3]
    manifest_path = cache_root / task / "manifest.json"
    manifest = _read_json(manifest_path)
    expected = {
        "schema_version": "tdc_mixed_l1_morgan.v1",
        "status": "complete",
        "profile": TDC_MIXED_L1_PROFILE,
        "task": task,
        "ranking": "morgan",
        "selection_unit": "source_specific_label_card",
        "top_k": 10,
        "neighbor_identity_policy": "parent_and_nonempty_scaffold_disjoint",
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"{task} TDC mixed-L1 manifest has invalid {field}: "
                f"{manifest.get(field)!r}"
            )
    for source in (manifest.get("candidate_sources") or {}).values():
        path = repository_root / str(source.get("path") or "")
        if not path.is_file() or sha256_file(path) != source.get("sha256"):
            raise ValueError(f"{task} TDC mixed-L1 source hash mismatch: {path}")

    subset_manifest = (manifest.get("subsets") or {}).get(subset) or {}
    cache_path = manifest_path.parent / str(subset_manifest.get("cache") or "")
    query_path = repository_root / str(subset_manifest.get("query_input") or "")
    if not cache_path.is_file() or sha256_file(cache_path) != subset_manifest.get(
        "cache_sha256"
    ):
        raise ValueError(f"{task} TDC mixed-L1 cache hash mismatch: {cache_path}")
    if not query_path.is_file() or sha256_file(query_path) != subset_manifest.get(
        "query_input_sha256"
    ):
        raise ValueError(f"{task} TDC mixed-L1 query hash mismatch: {query_path}")

    cache_rows = read_jsonl(cache_path)
    current_by_id = {str(row["benchmark_row_id"]): row for row in records}
    cached_by_id = {str(row["benchmark_row_id"]): row for row in cache_rows}
    if set(cached_by_id) != set(current_by_id):
        raise ValueError(f"{task} TDC mixed-L1 cache query set differs from {subset}")

    family = "tdc_mixed_training_labels"
    endpoint = _task_contract(task, prompt_version).endpoint_name
    candidates: dict[str, dict[str, dict[str, Any]]] = {}
    source_counts = {"gold_v1": 0, "tdc_v1": 0}
    for query_id, current in current_by_id.items():
        cached = cached_by_id[query_id]
        if (
            str(cached.get("query_drug")) != str(current.get("drug"))
            or str(cached.get("query_molecule_identity_key"))
            != str(current.get("molecule_identity_key"))
        ):
            raise ValueError(f"{task} TDC mixed-L1 query identity mismatch: {query_id}")
        cards = list(cached.get("cards") or [])
        if len(cards) != 10:
            raise ValueError(f"{task} TDC mixed-L1 query {query_id} does not have 10 cards")
        query_candidates: dict[str, dict[str, Any]] = {}
        for rank, row in enumerate(cards, start=1):
            source_kind = str(row["source_kind"])
            if source_kind not in source_counts:
                raise ValueError(f"unsupported TDC mixed-L1 source: {source_kind}")
            source_counts[source_kind] += 1
            analog_id = stable_analog_id({
                "standard_inchi_key": row["molecule_identity_key"],
                "canonical_smiles": row["drug"],
            })
            analog = query_candidates.setdefault(
                analog_id,
                {
                    "analog_id": analog_id,
                    "canonical_smiles": str(row["drug"]),
                    "similarity": float(row["morgan_similarity"]),
                    "molecule_relation": "structural_analog",
                    "_selection_rank": rank,
                    "cards": {},
                },
            )
            analog["_selection_rank"] = min(int(analog["_selection_rank"]), rank)
            record_id = str(row["benchmark_row_id"])
            card_id = "card_" + hashlib.sha256(
                f"{task}:{source_kind}:{record_id}".encode("utf-8")
            ).hexdigest()[:16]
            label_counts = row.get("label_counts") or {str(row["Y"]): 1}
            total = sum(int(value) for value in label_counts.values())
            positive_fraction = int(label_counts.get("1", 0)) / total
            source_name = (
                "Gold-v1 conditioned benchmark"
                if source_kind == "gold_v1"
                else "TDC external dataset"
            )
            reported_value = f"Frozen label={int(row['Y'])}"
            if source_kind == "gold_v1":
                reported_value += (
                    f"; positive-vote fraction={100.0 * positive_fraction:.1f}%"
                )
            analog["cards"][card_id] = {
                "card_id": card_id,
                "evidence_family": family,
                "assay_context": f"{source_name} training label",
                "endpoint": endpoint,
                "reported_value": reported_value,
                "reported_unit": "",
                "qualifying_conditions": (
                    ""
                    if row.get("condition_group")
                    == "no_reported_external_condition"
                    else str(row.get("condition_group") or "")
                ),
                "experimental_details": {
                    "label_source": source_kind,
                    "benchmark_row_id": record_id,
                },
                "support_text": (
                    "Frozen conditioned-benchmark training outcome."
                    if source_kind == "gold_v1"
                    else "External TDC training label; not a direct assay record."
                ),
                "_assay_key": f"{source_kind}:{record_id}",
                "_selection_rank": rank,
            }
        candidates[query_id] = query_candidates

    return candidates, {
        "schema_version": manifest["schema_version"],
        "selection_policy": manifest["schema_version"],
        "ranking": "morgan",
        "top_k": 10,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "cache": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "query_input": str(query_path),
        "query_input_sha256": sha256_file(query_path),
        "n_current_queries": len(current_by_id),
        "visible_source_card_counts": source_counts,
    }


def _load_reused_query_prior(
    task: str,
    record: Mapping[str, Any],
    single_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    key = _stable_query_key(record)
    try:
        source = _single_source_index(task, str(single_root))[key]
    except KeyError as exc:
        raise ValueError(f"no reusable single branch for stable query identity: {key}") from exc
    run_dir = Path(source.run_dir)
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
        source.source_index,
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
    if query_prior_mode in {"fresh", "cached"}:
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
            "candidate_source": (
                "gold_v1_plus_tdc_training_labels"
                if l1_source == "tdc_mixed_train"
                else "conditioned_gold_training_labels"
            ),
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
    prefetch_tools: bool = True,
    tool_service_url: str = "http://127.0.0.1:8765",
    timeout_s: int = 120,
    retrieval_policy: Mapping[str, Any] | None = None,
    molecule_description: Mapping[str, Any] | None = None,
    candidate_output: bool = False,
    l1_prior_source: Path | None = None,
) -> PreparedQuery:
    """Write context snapshots and optional cache-backed later-level bundles."""
    if retrieval_policy:
        cache_profile = 'per_stage_policy'
    query_dir = _query_dir(output_root, task, query_index)
    complete_path = query_dir / "prepared_manifest.json"
    aligned = context_records.is_tianang_aligned(prompt_version)
    mapped_selection = prompt_version.startswith('reranked_progressive_') or prompt_version.startswith(("tianang_aligned_relevance_", "tianang_aligned_mapped_", "tianang_aligned_ranked_"))
    protocol = (
        context_records.TIANANG_ALIGNED_PROTOCOL_VERSION
        if aligned
        else context_records.INDIRECT_PROTOCOL_VERSION
        if indirect_records is not None
        else context_records.PROTOCOL_VERSION
    )
    molecule_description_identity = (
        {
            key: molecule_description[key]
            for key in ("mode", "column", "path", "sha256", "missing_policy")
            if key in molecule_description
        }
        if molecule_description is not None
        else {"mode": "none"}
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
            and manifest.get('retrieval_policy') == retrieval_policy
            and manifest.get("molecule_description") == molecule_description_identity
            and bool(manifest.get("candidate_output")) is candidate_output
            and bool(manifest.get("tool_prefetch_complete")) is prefetch_tools
            and manifest.get("assay_transfer_cache_profile", "v19_1")
            == cache_profile
            and manifest.get("v21_selection_mode", "record_only")
            == (v21_selection_mode if cache_profile == V21_CACHE_PROFILE else None)
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
            and manifest.get("l1_prior_source")
            == (str(l1_prior_source) if l1_prior_source is not None else None)
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

    if query_prior_mode in {"fresh", "cached"}:
        query_prior, query_tool_summary, none_final, single_source_index = (
            _load_reused_query_prior(task, record, single_root)
        )
    else:
        query_prior, query_tool_summary, none_final, single_source_index = {}, {}, {}, None

    query_smiles = str(record.get("drug") or "")
    molecule_descriptions = (
        molecule_description["values"] if molecule_description is not None else None
    )
    query_molecule_description = (
        molecule_descriptions.get(query_smiles)
        if molecule_descriptions is not None
        else None
    )
    condition_sentence = external_condition_sentence(dict(record))
    if aligned:
        snapshots = context_records.stage_ranked_snapshots(
            contexts, task=task, records_by_level=indirect_records or {}, prompt_version=prompt_version,
            molecule_descriptions=molecule_descriptions,
        ) if retrieval_policy else context_records.tianang_aligned_snapshots(
            contexts,
            task=task,
            indirect_records=indirect_records or {},
            indirect_record_limit=indirect_record_limits or indirect_record_limit,
            prompt_version=prompt_version,
            record_levels=() if indirect_records is None else record_levels,
        )
        analog_tools, tool_failures = _prefetch_analog_tools(
            query_smiles=query_smiles,
            analogs=snapshots[max(snapshots)],
            tool_service_url=tool_service_url,
            timeout_s=timeout_s,
        ) if prefetch_tools else ({}, [])
        for snapshot in snapshots.values():
            attach_analog_tool_summaries(snapshot, analog_tools)
    else:
        snapshots = context_records.snapshots(
            contexts,
            task=task,
            indirect_records=indirect_records,
            indirect_record_limit=indirect_record_limits or indirect_record_limit,
            prompt_version=prompt_version,
            record_levels=record_levels,
            transfer_model="V21" if cache_profile == V21_CACHE_PROFILE else "V19.1",
        )
        tool_failures = []
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
            query_dir / "levels" / f"level_{level}" / (
                "candidate_prepared.json" if candidate_output else "prepared.json"
            ),
            {
                "protocol": protocol,
                "profile": "context_records",
                "query_prior_mode": query_prior_mode,
                "l1_source": (
                    "none_indirect_only"
                    if retrieval_policy and 'L1' not in retrieval_policy['stages'] else
                    "frozen_v9_gold_train_parents_with_mapped_v10_l1_records"
                    if prompt_version.startswith('reranked_progressive_') else
                    "v9_gold_train_ranked_contexts_without_associated_l2"
                    if cache_profile == V24_1_LEVEL_CACHE_PROFILE else
                    "configured_level_membership_after_heldout_parent_exclusion"
                    if mapped_selection else
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
                "query_molecule_description": query_molecule_description,
                "molecule_description": molecule_description_identity,
                "condition_sentence": condition_sentence,
                "query_prior": query_prior,
                "query_tool_summary": query_tool_summary,
                "reused_none_final": none_final,
                "level": level,
                "level_definition": level_row,
                **({'retrieval_policy': retrieval_policy} if retrieval_policy else {}),
                "retrieval_audit": {
                    "candidate_source": (
                        ('exact_finalized_indirect_only_cache_assignments'
                         if retrieval_policy and 'L1' not in retrieval_policy['stages'] else
                         'frozen_v9_gold_train_top100_parents_with_v10_l1_records' if level == 1 else
                         'full_mapped_v10_morgan_pool' if level == 5 else
                         'exact_finalized_cache_assignments')
                        if prompt_version.startswith('reranked_progressive_') else
                        (
                            "v9_gold_train_morgan_top100_contexts"
                            if level == 1
                            else f"v10_uid_mapped_L{level}_morgan_top75_"
                            "then_level_specific_v24_1_assay_transfer_ranking"
                        )
                        if cache_profile == V24_1_LEVEL_CACHE_PROFILE else
                        "level_specific_pool_then_configured_stage_ranking"
                        if retrieval_policy else
                        "mapping_specific_morgan_pool_then_relevance_bucket_selection"
                        if mapped_selection else
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
                    "l2_record_limit_per_context": None if retrieval_policy else l2_record_limit,
                    "indirect_record_limit_per_level": (
                        indirect_record_limit
                        if indirect_records is not None
                        else None
                    ),
                    "indirect_record_limit_for_current_level": (
                        level_record_limit if indirect_records is not None else None
                    ),
                    "deterministic_sampling": True,
                    "ranking": retrieval_policy['stages'][f'L{level}'] if retrieval_policy else context_ranking,
                    "ranking_tie_seed": (
                        ranking_tie_seed if retrieval_policy or context_ranking == "morgan" else None
                    ),
                    "matched_control_record_ids": (
                        (indirect_records or {}).get(f"L{level}", {}).get(
                            "matched_control_record_ids", []
                        )
                    ),
                    "original_semantic_record_ids": (
                        (indirect_records or {}).get(f"L{level}", {}).get(
                            "original_semantic_record_ids", []
                        )
                    ),
                },
                "active_evidence": snapshot,
                "new_card_ids": sorted(new_ids),
                "n_active_molecules": len(snapshot),
                "n_active_cards": len(current_ids),
                "should_call_model": bool(new_ids),
                "tool_prefetch_complete": prefetch_tools,
                "analog_tool_policy": (
                    "tianang_mmp_structure_compare_plus_properties_compare"
                    if aligned
                    else "disabled_by_context_records_profile"
                ),
                "tool_prefetch_failures": tool_failures,
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
    if l1_prior_source is not None:
        _reuse_full_flat_l1_output(
            query_dir=query_dir, source_query_dir=l1_prior_source
        )
    write_json_atomic(
        complete_path,
        {
            "status": "ok",
            "protocol": protocol,
            "profile": "context_records",
            "prompt_version": prompt_version,
            "molecule_description": molecule_description_identity,
            "candidate_output": candidate_output,
            "context_ranking": context_ranking,
            **({'retrieval_policy': retrieval_policy} if retrieval_policy else {}),
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
            "l1_prior_source": (
                str(l1_prior_source) if l1_prior_source is not None else None
            ),
            "task": task,
            "query_index": query_index,
            "n_levels": len(levels),
            "tool_prefetch_complete": prefetch_tools,
            "analog_tool_policy": (
                "tianang_mmp_structure_compare_plus_properties_compare"
                if aligned
                else "disabled_by_context_records_profile"
            ),
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
        providers = [
            ProviderSpec(
                name="primary",
                base_url=args.base_url.rstrip("/"),
                model=args.model,
                api_key_env=args.api_key_env,
                max_inflight=args.parallelism,
            )
        ]
        config = ProviderPoolConfig(
            providers=tuple(providers),
            max_failovers=int(len(providers) > 1),
        )
    expected_identity = _model_identity(args.model)
    for spec in config.providers:
        if _model_identity(spec.model) != expected_identity:
            raise ValueError(
                f"provider {spec.name!r} model {spec.model!r} does not match "
                f"run model identity {expected_identity!r}"
            )
    return config


def _make_client(
    args: argparse.Namespace,
    provider_config: ProviderPoolConfig,
) -> OpenAIProviderPool:
    return build_provider_pool(
        provider_config,
        env_file=args.env_file,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
        temperature=0.0,
        tool_service_url=args.tool_service_url,
        enable_group_tools=False,
        max_tool_rounds=0,
        reasoning_effort="",
        enable_thinking=False,
        transport_max_retries=args.transport_max_retries,
    )


def query_steps(args, prepared_query, client):
    """Compatibility wrapper; inference.py owns progressive model execution."""
    if (args.harness_version == INDIRECT_FILTER_HARNESS
            and args.reranking == "morgan-parent-llm-semantic"):
        from predict.harnesses.progressive.record_filter import query_steps as filter_steps

        return (yield from filter_steps(args, prepared_query, client))
    from predict.harnesses.progressive.inference import query_steps as engine_steps

    return (yield from engine_steps(args, prepared_query, client))


def _run_query(args, prepared_query, client):
    """Compatibility wrapper for historical experiment launchers."""
    from predict.harnesses.progressive.inference import run_query

    return run_query(args, prepared_query, client)


def _run_query_safe(args, prepared_query, client):
    """Compatibility wrapper for historical experiment launchers."""
    from predict.harnesses.progressive.inference import run_query_safe

    return run_query_safe(args, prepared_query, client)


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
    prompt_version: str | None = None,
    levels_override: list[dict[str, Any]] | None = None,
) -> None:
    contract = _task_contract(task, prompt_version or 'standard_v1')
    if query_prior_mode in {"fresh", "cached"}:
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

    level_rows = levels_override or (
        context_records.tianang_aligned_levels(task, max_level, prompt_version=prompt_version)
        if prompt_version and context_records.prompt_profile(prompt_version).get('stage_ranked') else
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
        if (
            args.profile == "standard"
            and args.harness_version != TDC_MIXED_L1_HARNESS
        ):
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
        if args.query_prior in {"fresh", "cached"}:
            batch_dir = _single_batch_dir(task, Path(args.single_source_root))
            single_manifest_path = batch_dir / "manifest.json"
            single_manifest = _read_json(single_manifest_path)
            if (
                single_manifest.get("schema_version")
                != "progressive_query_prior_overlay.v1"
                and not (batch_dir / "runs").is_dir()
            ):
                raise FileNotFoundError(batch_dir / "runs")
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
            if (
                args.evaluation_subset == "test"
                and single_manifest.get("input_jsonl_sha256")
                != sha256_file(spec.input_jsonl)
            ):
                raise ValueError(
                    f"{task} test query-prior input hash differs from the selected gold split"
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


def run(args: argparse.Namespace, *, prepared_callback=None,
        prepared_query_callback=None, candidate_loader=None) -> int:
    provider_config = _resolve_provider_pool_config(args)
    reasoning_efforts = {
        str((provider.request_extra_body or {}).get("chat_template_kwargs", {}).get(
            "reasoning_effort"
        ) or "")
        for provider in provider_config.providers
    }
    if not args.prepare_only and reasoning_efforts != {"high"}:
        raise ValueError("progressive inference requires reasoning_effort=high on every provider")
    if not args.prepare_only and any(
        provider.max_inflight > args.endpoint_concurrency_budget
        for provider in provider_config.providers
    ):
        raise ValueError("provider max_inflight exceeds the per-endpoint concurrency budget")
    endpoint_selection = None
    if args.prepare_only:
        args.requested_parallelism = args.parallelism
        args.parallelism = args.parallelism or 1
    else:
        if args.parallelism is None:
            raise ValueError("full-batch inference requires explicit --parallelism")
        args.requested_parallelism = args.parallelism
        endpoint_selection = select_healthy_providers(
            provider_config, args.requested_parallelism
        )
        provider_config = endpoint_selection.config
        args.parallelism = endpoint_selection.effective_parallelism
    endpoint_preflight = None
    if not args.prepare_only:
        from predict.harnesses.progressive.prompt import prompt_assets

        endpoint_preflight = {
            "models": endpoint_selection.public_dict(),
            "tokenized_reasoning": (
                preflight_sglang_tokenized_completion(provider_config)
                if any(
                    prompt_assets(_context_prompt_version(args, task))["settings"].get(
                        "reasoning_transport"
                    )
                    for task in args.tasks
                )
                else None
            ),
        }
    records_by_task = _validate_inputs(args)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    indices_by_task = {
        task: _selected_indices(args, len(records_by_task[task]))
        for task in args.tasks
    }
    args.full_flat_l1_sources = {}
    full_flat_l1_audit = None
    if (
        args.harness_version in FULL_FLAT_PROGRESSIVE_HARNESSES
        and args.reasoning_phase == "indirect-update"
        and args.l1_prior_run is not None
    ):
        args.full_flat_l1_sources, full_flat_l1_audit = _validate_full_flat_l1_source(
            args, records_by_task, indices_by_task
        )
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
            if getattr(args, 'harness_version', None) in {
                    'reranked-progressive-v2', 'reranked-progressive-v3',
                    'reranked-progressive-v4', 'reranked-progressive-l1-context-v1',
                    'reranked-progressive-l1-context-l2-v1',
                    'reranked-progressive-l1-context-l2-weighted-v1',
                    'reranked-progressive-l1-context-l2-morgan-bucket-v1',
                    *INDIRECT_HARNESSES, *FULL_FLAT_PROGRESSIVE_HARNESSES}:
                from predict.harnesses.progressive.retrieval_cache import load_candidates
                molecules, later, audit = (candidate_loader or load_candidates)(selected_queries, task=task,
                    subset=args.evaluation_subset, library=args.evidence_libraries[task],
                    mapper=args.level_mapper, policy=args.retrieval_policies[task],
                    gold_context_mapping=args.gold_context_mapping if task == 'bbb_martins' else None,
                    allow_frozen_l1_vote_scores=args.allow_frozen_l1_vote_scores,
                    cache_pool=args.cache_pool,
                    molecule_limit=args.context_limit, l1_limit=args.record_limit_per_context_level,
                    later_limit=(
                        args.l2_records_per_molecule
                        if args.harness_version == 'reranked-progressive-l1-context-l2-morgan-bucket-v1'
                        else _indirect_record_limits(args, task)
                        if args.retrieval_policies[task].get('selection_contract')
                        in {'ranked_level_retrieval.v2', 'ranked_uid_retrieval.v1'}
                        else args.indirect_record_limit_per_level
                    ),
                    tie_seed=args.ranking_tie_seed,
                    joint_panel_sizes=args.joint_panel_sizes,
                    **(
                        {
                            'min_contrast': args.l1_min_contrast,
                            'morgan_primary_parent_width': args.morgan_primary_parent_width,
                        }
                        if args.harness_version in {
                            'reranked-progressive-l1-context-v1',
                            *FULL_FLAT_PROGRESSIVE_HARNESSES,
                        }
                        else {}
                    ))
                context_candidates_by_task[task] = molecules
                indirect_candidates_by_task[task] = later
                ranking_audits[task] = indirect_ranking_audits[task] = audit
                continue
            if args.level_mapping != "local":
                from predict.harnesses.progressive.level_selection import load_mapped_candidates, IMPORTED
                ranked_molecules, indirect, audit = load_mapped_candidates(
                    selected_queries,
                    mapping=IMPORTED if args.level_mapping == "tianang" else Path(args.level_mapping),
                    ranking_root=Path(args.relevance_ranking_root),
                    v7_root=Path(args.v7_root), molecule_limit=args.context_limit,
                    l1_limit=args.record_limit_per_context_level,
                    l2_limit=args.l2_record_limit_per_context,
                    later_limit=args.indirect_record_limit_per_level,
                    tie_seed=args.ranking_tie_seed, ranking=args.context_ranking,
                    score_cache=args.level_score_cache,
                    task=task, sampler=args.record_sampler,
                    retrieval_policy=getattr(args, 'retrieval_policies', {}).get(task),
                )
                for molecules in ranked_molecules.values():
                    for molecule in molecules:
                        molecule["level_mapping"] = args.level_mapping
                context_candidates_by_task[task] = ranked_molecules if getattr(args, 'retrieval_policy', None) else {
                    query_id: context_records.v21_molecule_contexts(query_id, rows)
                    for query_id, rows in ranked_molecules.items()
                }
                indirect_candidates_by_task[task] = indirect
                ranking_audits[task] = audit
                indirect_ranking_audits[task] = audit
                continue
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
                    distinct_molecules=_is_tianang_aligned(args, task),
                    cache_dir=(
                        Path(args.v9_ranking_root)
                        / task
                        / "scaffold"
                        / args.evaluation_subset
                        if args.evaluation_subset == "test"
                        else None
                    ),
                    reference_path=(
                        _benchmark_scaffold_root(
                            Path(args.benchmark_data_root), task
                        )
                        / "train_molecule_condition_labels.jsonl"
                        if args.evaluation_subset == "test"
                        else None
                    ),
                )
                if args.assay_transfer_cache_profile == V24_1_LEVEL_CACHE_PROFILE:
                    for contexts in candidates.values():
                        for context in contexts:
                            context["l2_cards"] = []
                            context["available_l2"] = 0
                context_candidates_by_task[task] = candidates
                ranking_audits[task] = audit
            if _record_cache_enabled(args):
                exclusions = (
                    {
                        query_id: context_records.selected_physical_record_ids(contexts)
                        for query_id, contexts in context_candidates_by_task[task].items()
                    }
                    if _is_tianang_aligned(args, task)
                    else None
                )
                if args.assay_transfer_cache_profile == V24_1_LEVEL_CACHE_PROFILE:
                    indirect, indirect_audit = load_v24_1_level_records(
                        selected_queries,
                        subset=args.evaluation_subset,
                        levels=_record_level_names(args, task),
                        limit=_indirect_record_limits(args, task),
                        workers=args.preparation_workers,
                        exclude_record_ids_by_query=exclusions,
                    )
                else:
                    indirect, indirect_audit = load_top_ranked_records(
                        task,
                        selected_queries,
                        cache_profile=args.assay_transfer_cache_profile,
                        levels=_record_level_names(args, task),
                        limit=_indirect_record_limits(args, task),
                        workers=args.preparation_workers,
                        ranking=args.context_ranking,
                        tie_seed=args.ranking_tie_seed,
                        exclude_record_ids_by_query=exclusions,
                    )
                indirect_candidates_by_task[task] = indirect
                indirect_ranking_audits[task] = indirect_audit
                if (
                    args.assay_transfer_cache_profile == V21_CACHE_PROFILE
                    and args.v21_selection_mode == "record_only"
                ):
                    ranking_audits[task] = indirect_audit
    elif args.l1_source in {"gold_train", "tdc_mixed_train"}:
        for task in args.tasks:
            if args.l1_source == "tdc_mixed_train":
                candidates, audit = _tdc_mixed_l1_candidates(
                    task=task,
                    records=records_by_task[task],
                    cache_root=Path(args.tdc_mixed_l1_cache),
                    subset=args.evaluation_subset,
                    prompt_version=args.prompt_version,
                )
            else:
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
    description_artifact = MOLECULE_DESCRIPTION_ARTIFACTS[
        args.molecule_description_cache_version
    ]
    molecule_description = _load_molecule_descriptions(
        args.molecule_description_mode,
        path=description_artifact["path"],
        expected_sha256=description_artifact["sha256"],
    )
    molecule_description_identity: dict[str, Any] = {"mode": "none"}
    if molecule_description is not None:
        molecule_description = {
            **molecule_description,
            "cache_version": args.molecule_description_cache_version,
        }
        values = _validate_molecule_description_coverage(
            molecule_description,
            records_by_task=records_by_task,
            indices_by_task=indices_by_task,
            contexts_by_task=context_candidates_by_task,
            later_by_task=indirect_candidates_by_task,
            receipt_path=output_root / "molecule_description_preflight.json",
            missing_policy=args.molecule_description_missing_policy,
        )
        molecule_description = {
            **molecule_description,
            "missing_policy": args.molecule_description_missing_policy,
            "values": values,
        }
        molecule_description_identity = {
            key: molecule_description[key]
            for key in (
                "mode", "cache_version", "column", "path", "sha256",
                "missing_policy",
            )
        }
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
                if args.query_prior in {"fresh", "cached"}
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
        if (
            args.profile == "standard"
            and args.harness_version != TDC_MIXED_L1_HARNESS
        ):
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
            contract_payload = (
                context_records.tianang_aligned_card_contract(prompt_version)
                if context_records.is_tianang_aligned(prompt_version)
                else context_records.card_contract(prompt_version)
            )
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
        if _record_cache_enabled(args) and not context_records.is_tianang_aligned(
            prompt_version
        ):
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
        from predict.harnesses.progressive.prompt import prompt_asset_path

        card_contract_path = prompt_asset_path(args.prompt_version, "card.yaml")
        standard_card_contract = molecule_card_contract(card_contract_path)
        card_contract_manifest = {
            "path": str(card_contract_path),
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
        "evaluation_subset": args.evaluation_subset,
        "tasks": args.tasks,
        "visibility_mode": "deployment_visible_prefetched",
        "reference_pool": (
            "gold_v1_plus_tdc_train_label_cards"
            if args.l1_source == "tdc_mixed_train"
            else "conditioned_gold_train_plus_normalized_v7"
            if args.l1_source == "gold_train" and args.max_level > 1
            else "conditioned_gold_train"
            if args.l1_source == "gold_train"
            else REFERENCE_POOL
        ),
        "neighbor_identity_policy": IDENTITY_POLICY,
        "min_similarity": (
            None
            if args.l1_source == "tdc_mixed_train"
            else {"gold_l1": args.gold_l1_min_similarity, "normalized_later_levels": 0.3}
            if args.l1_source == "gold_train" and args.max_level > 1
            else args.gold_l1_min_similarity
            if args.l1_source == "gold_train"
            else 0.3
        ),
        "candidate_generation": {
            "unit": "molecule",
            "scope": (
                "Morgan top-10 from the Gold-v1 plus TDC training-card union"
                if args.l1_source == "tdc_mixed_train"
                else "V9 top-75 conditioned gold L1 plus normalized-v7 later families"
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
            TDC_MIXED_L1_PROMPT
            if args.l1_source == "tdc_mixed_train"
            else "progressive_compact_tools_short_aliases.gold_l1.v1"
            if args.l1_source == "gold_train"
            else "progressive_compact_tools_short_aliases.v2"
        ),
        "molecule_card_contract": card_contract_manifest,
        "condition_policy": "natural-language sentence for non-null group; omit null group",
        "query_prior_mode": args.query_prior,
        "single_reuse_root": (
            str(Path(args.single_source_root)) if args.query_prior in {"fresh", "cached"} else None
        ),
        "l1_source": args.l1_source,
        "l1_ranking": args.l1_ranking,
        "max_level": args.max_level,
        "model": args.model,
        "model_identity": _model_identity(args.model),
        "base_url": args.base_url,
        "parallelism": args.parallelism,
        "requested_parallelism": args.requested_parallelism,
        "endpoint_concurrency_budget": args.endpoint_concurrency_budget,
        "max_tokens": args.max_tokens,
        "timeout_s": args.timeout_s,
        "temperature": 0.0,
        "thinking": "provider_default",
        "reasoning_effort": (
            "high" if reasoning_efforts == {"high"} else "not_executed"
        ),
        "transport_max_retries": args.transport_max_retries,
        "tool_prefetch_complete": (
            not args.skip_tool_prefetch
        ),
        "evaluation_indices_by_task": indices_by_task,
        "inputs": inputs,
        "l2_reuse": l2_reuse_audits or None,
        "started_at": _now(),
        "provider_routing": provider_config.public_dict(),
        "provider_candidate_inventory": (
            {
                "path": str(Path(args.provider_pool_config).resolve()),
                "sha256": sha256_file(Path(args.provider_pool_config)),
            }
            if args.provider_pool_config
            else {"path": None, "source": "single_endpoint_cli"}
        ),
        "endpoint_preflight": endpoint_preflight,
        "molecule_description": molecule_description_identity,
    }
    from predict.harnesses.progressive.prompt import prompt_asset_manifest
    manifest['prompt_assets'] = {
        task: prompt_asset_manifest(
            _context_prompt_version(args, task)
            if args.profile == 'context_records'
            else args.prompt_version
        )
        for task in args.tasks
    }
    if args.harness_version in FULL_FLAT_PROGRESSIVE_HARNESSES:
        manifest.update(
            reasoning_phase=args.reasoning_phase,
            l1_prior_run=full_flat_l1_audit,
            reasoning_calls_per_query=(
                {"reused_l1": 1, "fresh_l1": 2}
                if args.reasoning_phase == "indirect-update"
                else 1
            ),
            consumed_scientific_levels={
                task: (
                    [int(name[1:]) for name in args.retrieval_policies[task]["stages"]]
                    if args.reasoning_phase == "indirect-update"
                    else [1]
                )
                for task in args.tasks
            },
        )
    if args.harness_version == INDIRECT_FILTER_HARNESS:
        from predict.harnesses.progressive.record_filter import asset_manifest

        manifest['record_filter_assets'] = asset_manifest()
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
        if all(_is_tianang_aligned(args, task) for task in args.tasks):
            context_selection = {
                "molecules": (
                    f"top {args.context_limit} distinct molecules by {args.context_ranking}"
                ),
                "molecule_limit": args.context_limit,
                "level_1": (
                    f"up to {args.record_limit_per_context_level} current-gold records per molecule"
                ),
                "level_2": (
                    f"append up to {args.l2_record_limit_per_context} associated records per molecule"
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
        if args.assay_transfer_cache_profile == V24_1_LEVEL_CACHE_PROFILE:
            context_selection = {
                "L1": (
                    f"top {args.context_limit} V9 gold-train contexts; retain only "
                    f"their first {args.record_limit_per_context_level} direct records"
                ),
                "L2_L4": (
                    "independent V10 source_row_uid level membership, Morgan top-75 "
                    "scaffold-disjoint parent pools, and level-specific V24.1 ranking"
                ),
                "record_limits_by_level": _indirect_record_limits(
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
            if all(_is_tianang_aligned(args, task) for task in args.tasks):
                context_selection.update(
                    later_levels="select globally ranked records before grouping them under molecule cards",
                    later_record_limits_by_task={
                        task: _indirect_record_limits(args, task)
                        for task in args.tasks
                    },
                    molecule_quota_after_l2=None,
                )
            context_selection["later_levels_by_task"] = {
                task: {
                    level: (
                        (
                            "append the top unseen records under molecule cards: "
                            if all(_is_tianang_aligned(args, item) for item in args.tasks)
                            else "append one bundle containing the top "
                        )
                        +
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
        if args.assay_transfer_cache_profile == V24_1_LEVEL_CACHE_PROFILE:
            context_selection["later_levels_by_task"] = {
                "bbb_martins": {
                    level: (
                        "append the top unseen V10 records assigned by source_row_uid "
                        f"to {level}, after Morgan top-75 prefiltering and the pinned "
                        f"{level} V24.1 reranker"
                    )
                    for level in _record_level_names(args, "bbb_martins")
                }
            }
        manifest.update(
            {
                "visibility_mode": (
                    "identity_blind_tianang_aligned_molecule_cards"
                    if all(_is_tianang_aligned(args, task) for task in args.tasks)
                    else "identity_blind_context_records"
                ),
                "reference_pool": (
                    "v9_gold_train_l1_plus_v10_uid_mapped_l2_l4_records"
                    if args.assay_transfer_cache_profile == V24_1_LEVEL_CACHE_PROFILE
                    else "normalized_v7_stage3_l1_l5_records"
                    if args.assay_transfer_cache_profile == V21_CACHE_PROFILE
                    else "current_conditioned_gold_contexts_plus_luna_"
                    "top_quartile_non_direct_v7_records"
                    if args.assay_transfer_cache_profile
                    == LUNA_RELEVANCE_CACHE_PROFILE
                    else "current_conditioned_gold_contexts_plus_v7_gold_source_domain_rows"
                ),
                "neighbor_identity_policy": (
                    "scaffold_disjoint_and_parent_disjoint"
                    if args.assay_transfer_cache_profile == V24_1_LEVEL_CACHE_PROFILE
                    else "scaffold_disjoint"
                    if args.assay_transfer_cache_profile == V21_CACHE_PROFILE
                    else "scaffold_disjoint_and_parent_disjoint"
                    if all(_is_tianang_aligned(args, task) for task in args.tasks)
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
                        else "distinct_molecule_then_individual_stage3_record"
                        if all(_is_tianang_aligned(args, task) for task in args.tasks)
                        and _record_cache_enabled(args)
                        else "distinct_molecule"
                        if all(_is_tianang_aligned(args, task) for task in args.tasks)
                        else "parent_condition_context_then_individual_stage3_record"
                        if args.context_record_l3_l5
                        else "parent_condition_context"
                    ),
                    "scope": (
                        "V9 gold-train L1 contexts without associated L2; independent "
                        "V10 UID-mapped L2-L4 Morgan-top-75 pools ranked by each level's "
                        "pinned V24.1 checkpoint"
                        if args.assay_transfer_cache_profile == V24_1_LEVEL_CACHE_PROFILE
                        else "V21-ranked Stage 3 Morgan-top-75 molecules with V21-ranked records within L1/L2 cards, then independent L3-L5 records"
                        if args.assay_transfer_cache_profile == V21_CACHE_PROFILE
                        and args.v21_selection_mode == "molecule_cards"
                        else "independent Stage 3 Morgan-top-75 V21 rankings for L1-L5"
                        if args.assay_transfer_cache_profile == V21_CACHE_PROFILE
                        else "exact V9 L1/L2 contexts plus source-local Luna "
                        "top-quartile relevance filtering before Stage 3 "
                        "Morgan-top-75 candidate generation"
                        if args.assay_transfer_cache_profile
                        == LUNA_RELEVANCE_CACHE_PROFILE
                        else f"top {args.context_limit} distinct molecules by "
                        f"V9 {args.context_ranking} rank from the exact Morgan-top-100 "
                        "candidate pool; task-configured Stage 3 Morgan-top-75 "
                        "scaffold-disjoint assignments for later levels"
                        if all(_is_tianang_aligned(args, task) for task in args.tasks)
                        and _record_cache_enabled(args)
                        else "exact V9 Morgan-top-100 assignments for L1/L2; task-configured "
                        "Stage 3 Morgan-top-75 scaffold-disjoint assignments for later levels"
                        if args.context_ranking == "morgan"
                        else f"V9 ranks 0-{args.context_limit - 1} for L1/L2; task-configured independent Stage 3 "
                        "Morgan-top-75 V19.1 rankings for later levels"
                        if args.context_record_l3_l5
                        else "exact V9 Morgan-top-100 assignments for L1/L2"
                        if args.context_ranking == "morgan"
                        else f"top {args.context_limit} distinct molecules by V9 "
                        "assay-transfer rank from the full Morgan top-100 candidate pool"
                        if all(_is_tianang_aligned(args, task) for task in args.tasks)
                        else "V9 ranks 0-9 from the full Morgan top-100 candidate pool"
                    ),
                    "ranking": (
                        {
                            "L1": "V9 assay-transfer",
                            "L2": "V24.1 L2 assay-transfer",
                            "L3": "V24.1 L3 assay-transfer",
                            "L4": "V24.1 L4 assay-transfer",
                        }
                        if args.assay_transfer_cache_profile == V24_1_LEVEL_CACHE_PROFILE
                        else {
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
                        or args.assay_transfer_cache_profile
                        == V24_1_LEVEL_CACHE_PROFILE
                        else None
                    ),
                    "similarity_floor": None,
                    "tie_break": (
                        "seeded_immutable_row_identity"
                        if args.context_ranking == "morgan"
                        else "external_record_id"
                    ),
                    "selection_before_molecule_grouping": all(
                        _is_tianang_aligned(args, task) for task in args.tasks
                    ),
                    "physical_record_deduplication": (
                        "exclude already-visible physical record IDs, then take the next ranked records"
                        if all(_is_tianang_aligned(args, task) for task in args.tasks)
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
                    "v9_gold_train_ranked_contexts_without_associated_l2"
                    if args.assay_transfer_cache_profile == V24_1_LEVEL_CACHE_PROFILE
                    else "v21_ranked_stage3_direct_record_molecules"
                    if args.assay_transfer_cache_profile == V21_CACHE_PROFILE
                    and args.v21_selection_mode == "molecule_cards"
                    else "v21_independently_ranked_stage3_direct_records"
                    if args.assay_transfer_cache_profile == V21_CACHE_PROFILE
                    else "current_conditioned_gold_raw_records"
                ),
                "l1_ranking": args.context_ranking,
                "max_level": args.max_level,
                "analog_tool_policy": (
                    "tianang_mmp_structure_compare_plus_properties_compare"
                    if all(_is_tianang_aligned(args, task) for task in args.tasks)
                    else "disabled_by_context_records_profile"
                ),
            }
        )
    manifest["execution_providers"] = [
        _execution_provider_from_spec(
            spec,
            transport_max_retries=args.transport_max_retries,
        )
        for spec in provider_config.providers
    ]
    if args.level_mapping != "local":
        manifest.update(
            level_mapping=args.level_mapping,
            record_sampler=args.record_sampler,
            l1_source="configured_level_membership_after_heldout_parent_exclusion",
            reference_pool="normalized_v7_with_explicit_level_membership",
            min_similarity=0.0,
            candidate_generation={"unit": "molecule", "pool_size_per_level": 75,
                                  "ranking": "morgan", "neighbor_identity_policy": "scaffold_disjoint"},
            selection={"L1": "mapped L1 molecules and records",
                       "L2": ("two records per relevance bucket per pass within selected molecules"
                              if args.record_sampler == "relevance"
                              else "Morgan-ranked records from percentile-20 "
                              "relevance-eligible buckets within selected molecules"
                              if args.record_sampler == "relevance_filter"
                              else "deterministic record sample within selected molecules"),
                       "later_levels": ("five records per ranked relevance bucket, continuing until the record cap or exhaustion"
                                        if args.record_sampler == "relevance"
                                        else "top Morgan-ranked records from percentile-20 "
                                        "relevance-eligible buckets in the level-specific top-75 molecule pool"
                                        if args.record_sampler == "relevance_filter"
                                        else "top Morgan-ranked records from the level-specific top-75 molecule pool"),
                       "record_ranking": args.context_ranking, "append_only": True},
        )
    if getattr(args, 'retrieval_policy', None):
        manifest.update(retrieval_policy=args.retrieval_policies, min_similarity=None,
            assay_transfer_cache_profile=None,
            candidate_generation={
                'unit': 'L1_molecules_then_independent_L2_plus_records',
                'pool_size_by_level': {'L1': 100, 'cached_later_levels': 75, 'L5': None},
                'ranking_by_task': {task: policy['stages'] for task, policy in args.retrieval_policies.items()},
                'neighbor_identity_policy': 'scaffold_disjoint_and_parent_disjoint',
                'similarity_floor': None,
                'morgan_fingerprint': {'radius': 2, 'bits': 2048, 'similarity': 'Tanimoto'},
                'structure_score_visible': True,
                'assay_transfer_scores_used': any(
                    method in {
                        'assay_transfer', 'assay_transfer_within_morgan',
                        'assay_transfer_contrastive', 'joint'
                    }
                    for policy in args.retrieval_policies.values() for method in policy['stages'].values()),
                'ranking_tie_seed': args.ranking_tie_seed,
                'tie_break': 'seeded_immutable_row_identity',
                'physical_record_deduplication': 'exclude already-visible physical record IDs before applying caps',
            },
            selection={'L1': 'configured molecule ranking; deterministic records within molecule',
                       'molecule_limit': args.context_limit,
                       'l1_record_limit_per_molecule': args.record_limit_per_context_level,
                       'joint_panels': ({'order': ['assay_transfer', 'morgan'],
                                         'sizes': {'assay_transfer': 3, 'morgan': 7},
                                         'rank_fields': ['assay_transfer_panel_rank',
                                                         'morgan_panel_rank']}
                                        if args.joint_panel_sizes else
                                        {'order': ['morgan', 'assay_transfer'],
                                         'sizes': {'morgan': 5, 'assay_transfer': 5},
                                         'rank_fields': ['morgan_top5_rank',
                                                         'assay_transfer_top5_rank']}),
                       'joint_overlap_refill': False,
                       'L2_and_later': 'independent top records per level, then merge by parent with per-record conditions',
                       'later_record_limits_by_task': {task: _indirect_record_limits(args, task) for task in args.tasks},
                       'record_ranking': 'per_level_yaml', 'append_only': True})
        if getattr(args, 'harness_version', None):
            manifest.update(harness_version=args.harness_version, reranking=args.reranking,
                            prompt_mode=('morgan' if args.reranking in {
                                'semantic-lap', 'semantic-weighted',
                                'morgan-parent-control', 'morgan-parent-semantic',
                                'morgan-parent-llm-semantic'
                            }
                                         else args.reranking))
            manifest['selection']['record_ranking'] = 'versioned_reranking_mode'
            if args.harness_version == 'reranked-progressive-l1-context-l2-v1':
                manifest['selection'].update(
                    L2_and_later=(
                        'global top-12 Morgan records grouped by parent molecule'
                        if args.reranking == 'morgan'
                        else 'semantic-rank laps with three Morgan-ranked records per bucket'
                    ),
                    semantic_metadata_visible=False,
                )
            if args.harness_version == 'reranked-progressive-l1-context-l2-weighted-v1':
                manifest['selection'].update(
                    L2_and_later=(
                        'global top-12 Morgan records grouped by parent molecule'
                        if args.reranking == 'morgan'
                        else 'global top-10 semantic buckets ranked by expert weight times Morgan similarity'
                    ),
                    semantic_metadata_visible=False,
                )
            if args.harness_version == 'reranked-progressive-l1-context-l2-morgan-bucket-v1':
                semantic = args.reranking == 'morgan-parent-semantic'
                manifest['selection'].update(
                    L2_and_later=(
                        'same independent Morgan top-10 parents; up to 10 eligible records '
                        'per parent ordered by semantic bucket rank'
                        if semantic else
                        'same independent Morgan top-10 parents; up to 10 eligible records '
                        'per parent in deterministic control order'
                    ),
                    l2_molecule_limit=args.l2_molecules,
                    l2_record_limit_per_molecule=args.l2_records_per_molecule,
                    semantic_metadata_visible=False,
                    zero_weight_buckets_visible=False,
                )
            if args.harness_version in INDIRECT_HARNESSES:
                filtered = (
                    args.harness_version == INDIRECT_FILTER_HARNESS
                    and args.reranking == 'morgan-parent-llm-semantic'
                )
                manifest['selection'].update(
                    L1='none',
                    L2_and_later='one independent indirect level only',
                    selected_level=f'L{args.indirect_level}',
                    total_record_limit=("llm_selected_10_to_25" if filtered else 25),
                    semantic_candidate_record_limit=(100 if filtered else None),
                    parent_limit='adaptive_minimal_prefix_up_to_100',
                    record_limit_per_parent=10,
                    semantic_record_limit_per_parent_bucket=None,
                    semantic_metadata_visible=False,
                    query_prior_visible=False,
                )
    if getattr(args, 'harness_version', None) in {
            'reranked-progressive-v2', 'reranked-progressive-v3',
            'reranked-progressive-v4', 'reranked-progressive-l1-context-v1',
            'reranked-progressive-l1-context-l2-v1',
            'reranked-progressive-l1-context-l2-weighted-v1',
            'reranked-progressive-l1-context-l2-morgan-bucket-v1',
            *INDIRECT_HARNESSES, *FULL_FLAT_PROGRESSIVE_HARNESSES}:
        cache_contracts = {
            audit['selection_policy'] for audit in ranking_audits.values()
        }
        reference_pool_version = (
            'semantic_bucket_v1'
            if cache_contracts == {'semantic_bucket_reranking.v1'}
            else 'context_semantic_l2_v1'
            if cache_contracts == {'l1_context_semantic_l2.v1'}
            else 'context_semantic_weighted_l2_v1'
            if cache_contracts == {'l1_context_semantic_weighted_l2.v1'}
            else 'context_morgan_semantic_l2_v1'
            if cache_contracts == {'l1_context_morgan_semantic_l2.v1'}
            else 'context_morgan_semantic_l2_v2'
            if cache_contracts == {'l1_context_morgan_semantic_l2.v2'}
            else 'context_morgan_semantic_l2_v3'
            if cache_contracts == {'l1_context_morgan_semantic_l2.v3'}
            else 'context_morgan_semantic_l2_v4'
            if cache_contracts == {'l1_context_morgan_semantic_l2.v4'}
            else 'indirect_morgan_semantic_l2_l4_v1'
            if cache_contracts == {'indirect_morgan_semantic_l2_l4.v1'}
            else 'indirect_morgan_semantic_l2_l4_v2'
            if cache_contracts == {'indirect_morgan_semantic_l2_l4.v2'}
            else 'indirect_morgan_semantic_l2_l4_v3'
            if cache_contracts == {'indirect_morgan_semantic_l2_l4.v3'}
            else 'ranked_level' if cache_contracts <= {
                'ranked_level_retrieval.v2', 'ranked_uid_retrieval.v1'
            }
            else 'ranked_v1' if cache_contracts == {'ranked_evidence_retrieval.v1'}
            else 'v3' if cache_contracts == {'cache_matched_retrieval.v3'} else 'v2'
        )
        manifest.update(reference_pool=f'cache_matched_v10_with_gold_train_l1.{reference_pool_version}',
            l1_source='frozen_v9_gold_train_parents_with_mapped_v10_l1_records',
            evidence_libraries={task: str(path.resolve()) for task, path in args.evidence_libraries.items()},
            level_mapper=str(args.level_mapper.resolve()), record_pool=args.record_pool,
            cache_pool=args.cache_pool)
        if args.harness_version in {
            'reranked-progressive-l1-context-v1',
            'reranked-progressive-l1-context-l2-v1',
            'reranked-progressive-l1-context-l2-weighted-v1',
            'reranked-progressive-l1-context-l2-morgan-bucket-v1',
            *FULL_FLAT_PROGRESSIVE_HARNESSES,
        }:
            primary_widths = {
                audit['contract']['morgan_primary_parent_width']
                for audit in ranking_audits.values()
            }
            fallback_widths = {
                audit['contract'].get('morgan_fallback_parent_width')
                for audit in ranking_audits.values()
            }
            if len(primary_widths) != 1 or len(fallback_widths) != 1:
                raise ValueError('L1 context tasks use different Morgan candidate widths')
            manifest.update(
                reference_pool='gold_train_condition_contexts_from_morgan100_v9',
                l1_source='exact_gold_voter_records_by_parent_condition_context',
            )
            manifest['candidate_generation'].update(
                unit='parent_condition_context',
                morgan_primary_parent_width=primary_widths.pop(),
                morgan_fallback_parent_width=(
                    fallback_widths.pop()
                    if fallback_widths != {None}
                    else None
                ),
                query_target_stored=False,
            )
            manifest['selection'].update(
                context_limit=args.context_limit,
                min_examples_per_binary_label=(
                    args.l1_min_contrast
                    if args.reranking in {
                        'morgan-contrastive', 'assay-transfer-contrastive'
                    } else None
                ),
                balance_unit='parent_condition_context',
                records='exact context voters, deterministic cap',
            )
        if args.harness_version in INDIRECT_HARNESSES:
            manifest.update(
                reference_pool='all_records_morgan100_indirect_only',
                l1_source='none',
            )
            manifest['candidate_generation'].update(
                unit='indirect_parent_molecule_records',
                query_target_stored=False,
            )
        manifest['candidate_generation'].pop('pool_size_by_level', None)
        if all(audit['selection_policy'] in SQLITE_SELECTION_CONTRACTS
               for audit in ranking_audits.values()):
            manifest['candidate_generation'].update(
                cache_capacities_by_task={
                    task: ranking_audits[task]['cache_capacities'] for task in args.tasks
                },
            )
            content_ids = {
                task: ranking_audits[task].get('cache_content_ids')
                for task in args.tasks
            }
            if all(content_ids.values()):
                manifest['candidate_generation']['cache_content_ids_by_task'] = content_ids
            else:
                manifest['candidate_generation']['cache_content_id_by_task'] = {
                    task: ranking_audits[task]['cache_content_id'] for task in args.tasks
                }
        else:
            manifest['candidate_generation']['pool_size_by_task'] = {
                task: {level: (ranking_audits[task]['l1_pool_size'] if level == 'L1' else None if level == 'L5'
                              else ranking_audits[task]['cache_versions'][policy['cache_manifests'][level]]['morgan_pool_size'])
                       for level in policy['stages']}
                for task, policy in args.retrieval_policies.items()}
    manifest["versions"] = {
        "harness": getattr(args, "harness_version", _run_protocol(args)),
        "prompt_bundle": getattr(args, "assay_transfer_prompt_version", "standard_v1"),
        "retrieval_selection_schema_by_task": {
            task: ranking_audits[task].get("selection_policy")
            for task in args.tasks
            if task in ranking_audits
        },
        "retrieval_bundle_manifest": str(getattr(args, "assay_transfer_cache", "")),
        "evidence_release_by_task": {
            task: Path(path).name for task, path in getattr(args, "evidence_libraries", {}).items()
        },
    }
    manifest_path = output_root / "experiment_manifest.json"
    if manifest_path.is_file():
        previous = _read_json(manifest_path)
        manifest = _merge_resume_manifest(previous, manifest)
    write_json_atomic(manifest_path, manifest)

    pilot_only = (
        not args.prepare_only
        and args.execution_mode == "live"
        and bool(getattr(args, "live_run_dirs", {}))
        and not getattr(args, "continue_after_pilot", False)
    )
    pilot_indices_by_task = {
        task: indices_by_task[task][:3] for task in args.tasks
    }
    levels_by_task: dict[str, list[dict[str, Any]]] = {}
    indexes_by_task: dict[str, Any] = {}
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
        levels_by_task[task] = levels
        indexes_by_task[task] = index

    def prepare_one(task: str, query_index: int) -> PreparedQuery:
        records = records_by_task[task]
        record = records[query_index]
        if args.profile == "context_records":
            prepared = _prepare_context_record_query(
                task=task,
                query_index=query_index,
                record=record,
                contexts=context_candidates_by_task[task][str(record["benchmark_row_id"])],
                levels=levels_by_task[task],
                output_root=output_root,
                single_root=Path(args.single_source_root),
                query_prior_mode=args.query_prior,
                record_limit=args.record_limit_per_context_level,
                l2_record_limit=args.l2_record_limit_per_context,
                indirect_record_limit=args.indirect_record_limit_per_level,
                context_limit=args.context_limit,
                indirect_record_limits=_indirect_record_limits(args, task),
                indirect_records=(
                    indirect_candidates_by_task[task][str(record["benchmark_row_id"])]
                    if _record_cache_enabled(args)
                    else None
                ),
                l2_reuse_root=(
                    CONTEXT_L2_REUSE_ROOTS[task] if args.reuse_context_l2 else None
                ),
                prompt_version=_context_prompt_version(args, task),
                context_ranking=args.context_ranking,
                ranking_tie_seed=args.ranking_tie_seed,
                cache_profile=args.assay_transfer_cache_profile,
                record_levels=(
                    _record_level_names(args, task)
                    if args.assay_transfer_cache_profile
                    in {V21_CACHE_PROFILE, V24_1_LEVEL_CACHE_PROFILE}
                    else None
                ),
                v21_selection_mode=args.v21_selection_mode,
                prefetch_tools=not args.skip_tool_prefetch,
                tool_service_url=args.tool_service_url,
                timeout_s=args.timeout_s,
                retrieval_policy=(
                    dict(
                        args.retrieval_policies[task],
                        selection_inputs_sha256=hashlib.sha256(
                            json.dumps(
                                ranking_audits[task]["contract"], sort_keys=True
                            ).encode()
                        ).hexdigest(),
                    )
                    if getattr(args, "retrieval_policy", None)
                    else None
                ),
                molecule_description=molecule_description,
                candidate_output=(
                    args.harness_version == INDIRECT_FILTER_HARNESS
                    and args.reranking == 'morgan-parent-llm-semantic'
                ),
                l1_prior_source=(
                    args.full_flat_l1_sources.get(task, {}).get(
                        str(record["benchmark_row_id"])
                    )
                    if args.full_flat_l1_sources else None
                ),
            )
            return prepared
        return _prepare_query(
            task=task,
            query_index=query_index,
            record=record,
            index=indexes_by_task[task],
            levels=levels_by_task[task],
            gold_l1_candidates=(
                gold_candidates_by_task[task][str(record["benchmark_row_id"])]
                if args.l1_source in {"gold_train", "tdc_mixed_train"}
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
        )

    def submit_preparations(pool, selected_by_task):
        return {
            pool.submit(prepare_one, task, query_index): (task, query_index)
            for task, selected in selected_by_task.items()
            for query_index in selected
        }

    def collect_preparations(
        futures, *, phase: str, on_prepared=None
    ) -> list[PreparedQuery]:
        totals = {
            task: sum(queued_task == task for queued_task, _ in futures.values())
            for task in args.tasks
        }
        completed_by_task = {task: 0 for task in args.tasks}
        prepared = []
        for future in concurrent.futures.as_completed(futures):
            task, _ = futures[future]
            query = future.result()
            prepared.append(query)
            if on_prepared is not None:
                on_prepared(query)
            completed_by_task[task] += 1
            completed = completed_by_task[task]
            if completed % 10 == 0 or completed == totals[task]:
                print(
                    f"[{task}] prepared {completed}/{totals[task]} ({phase})",
                    flush=True,
                )
        return prepared

    def run_inference(inference_queries, client) -> int:
        failed = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.parallelism, len(inference_queries))
        ) as pool:
            futures = [
                pool.submit(_run_query_safe, args, prepared, client)
                for prepared in inference_queries
            ]
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                result = future.result()
                failed += int(result.get("status") != "ok")
                if completed % 10 == 0 or completed == len(futures):
                    print(
                        f"[inference] completed {completed}/{len(futures)} "
                        f"failed={failed}",
                        flush=True,
                    )
        return failed

    prepared_queries: list[PreparedQuery] = []
    failed = 0
    client = None
    inference_pool = None
    inference_futures = []
    preparation_total = sum(len(indices) for indices in indices_by_task.values())
    initial_indices = pilot_indices_by_task if pilot_only else indices_by_task
    remaining_indices = {
        task: indices_by_task[task][len(initial_indices[task]):]
        for task in args.tasks
    }
    if not args.prepare_only and not pilot_only:
        client = _make_client(args, provider_config)
        inference_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=args.parallelism
        )

    def on_prepared(query):
        if prepared_query_callback is not None:
            prepared_query_callback(query)
        if inference_pool is not None:
            inference_futures.append(
                inference_pool.submit(_run_query_safe, args, query, client)
            )

    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.preparation_workers, preparation_total)
        ) as preparation_pool:
            initial_futures = submit_preparations(preparation_pool, initial_indices)
            prepared_queries.extend(
                collect_preparations(
                    initial_futures,
                    phase="pilot" if pilot_only else "full",
                    on_prepared=on_prepared,
                )
            )
            remaining_futures = (
                submit_preparations(preparation_pool, remaining_indices)
                if pilot_only
                else {}
            )
            if pilot_only:
                from predict.live import update_run

                for task, run_dir in args.live_run_dirs.items():
                    update_run(
                        run_dir,
                        status="pilot_running_preparing_remaining",
                        prepared_pilot_at=_now(),
                        preparation_completed=len(initial_indices[task]),
                        preparation_total=len(indices_by_task[task]),
                    )
                client = _make_client(args, provider_config)
                pilot_queries = sorted(prepared_queries, key=lambda row: row.index)
                failed = run_inference(pilot_queries, client)
                interim_status = "pilot_failed" if failed else "pilot_complete_preparing_remaining"
                for task, run_dir in args.live_run_dirs.items():
                    update_run(
                        run_dir,
                        status=interim_status,
                        pilot_completed_at=_now(),
                        pilot_indices=pilot_indices_by_task[task],
                    )
                prepared_queries.extend(
                    collect_preparations(remaining_futures, phase="remaining")
                )
        if inference_pool is not None:
            for completed, future in enumerate(
                concurrent.futures.as_completed(inference_futures), start=1
            ):
                failed += int(future.result().get("status") != "ok")
                if completed % 10 == 0 or completed == len(inference_futures):
                    print(
                        f"[inference] completed {completed}/{len(inference_futures)} "
                        f"failed={failed}",
                        flush=True,
                    )
    finally:
        if inference_pool is not None:
            inference_pool.shutdown(wait=True)

    prepared_queries.sort(key=lambda row: row.index)
    manifest["query_scheduling"] = (
        "live_pilot_first_with_background_preparation"
        if pilot_only
        else "interleaved_by_query_index_across_tasks"
    )
    manifest["prepared_at"] = _now()
    write_json_atomic(manifest_path, manifest)
    if prepared_callback is not None:
        prepared_callback(prepared_queries, records_by_task, indices_by_task)
    if args.prepare_only:
        for run_dir in (getattr(args, "live_run_dirs", {}) or {}).values():
            from predict.live import update_run

            update_run(run_dir, status="prepared")
        return 0

    if not pilot_only and inference_pool is None:
        client = _make_client(args, provider_config)
        failed = run_inference(prepared_queries, client)

    pilot_query_count = sum(len(indices) for indices in pilot_indices_by_task.values())
    if pilot_only and pilot_query_count < len(prepared_queries):
        from predict.live import update_run

        status = "pilot_failed" if failed else "awaiting_review"
        for task, run_dir in args.live_run_dirs.items():
            update_run(
                run_dir,
                status=status,
                pilot_completed_at=_now(),
                pilot_indices=pilot_indices_by_task[task],
                preparation_completed=len(indices_by_task[task]),
                preparation_total=len(indices_by_task[task]),
                preparation_completed_at=_now(),
            )
        manifest["status"] = status
        manifest["pilot_indices_by_task"] = pilot_indices_by_task
        manifest["provider_pool_final_snapshot"] = client.snapshot()
        write_json_atomic(manifest_path, manifest)
        return 1 if failed else 0

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
            prompt_version=(
                args.assay_transfer_prompt_version
                if getattr(args, 'retrieval_policy', None)
                or args.harness_version == TDC_MIXED_L1_HARNESS
                else None
            ),
            levels_override=(
                _run_levels(args, task)
                if args.harness_version in {
                    *FULL_FLAT_PROGRESSIVE_HARNESSES,
                    TDC_MIXED_L1_HARNESS,
                }
                else None
            ),
        )
    diagnostic_error = None
    if not failed:
        try:
            from predict.harnesses.progressive.diagnostics import build_run_diagnostics

            build_run_diagnostics(output_root)
        except Exception as exc:
            diagnostic_error = str(exc)
    manifest["finished_at"] = _now()
    manifest["n_failed_queries"] = failed
    manifest["diagnostic_error"] = diagnostic_error
    manifest["status"] = "complete" if not failed and diagnostic_error is None else "incomplete"
    manifest["provider_pool_final_snapshot"] = client.snapshot()
    write_json_atomic(manifest_path, manifest)
    if getattr(args, "live_run_dirs", {}):
        from predict.live import update_run

        for run_dir in args.live_run_dirs.values():
            update_run(
                run_dir,
                status="failed" if failed or diagnostic_error else "complete",
                finished_at=_now(),
            )
    return 1 if failed or diagnostic_error else 0


def _configure_indirect_only_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from predict.harnesses.progressive.retrieval_cache import (
        DEFAULT_CACHE_BUNDLE,
        INDIRECT_MORGAN_SEMANTIC_CACHE_BUNDLE,
        INDIRECT_MORGAN_SEMANTIC_V2_CACHE_BUNDLE,
        INDIRECT_MORGAN_SEMANTIC_V3_CACHE_BUNDLE,
    )

    if args.reranking not in {
            'morgan-parent-control', 'morgan-parent-semantic',
            'morgan-parent-llm-semantic'}:
        parser.error('The indirect-only harness requires a matched Morgan-parent arm')
    if args.indirect_level not in {2, 3, 4}:
        parser.error('The indirect-only harness requires --indirect-level 2, 3, or 4')
    if args.max_level not in {0, args.indirect_level}:
        parser.error('--max-level must be omitted or match --indirect-level')
    if args.l1_contexts is not None:
        parser.error('The indirect-only harness cannot accept L1 contexts')
    if args.molecule_description_mode != 'none':
        parser.error('The indirect-only harness does not expose molecule descriptions')
    expected_prompt = 'reranked_progressive_l1_context_v4_no_query_prior'
    if args.prompt_version == 'reranked_progressive_v8':
        args.prompt_version = expected_prompt
    elif args.prompt_version != expected_prompt:
        parser.error(f'The indirect-only harness requires {expected_prompt}')
    if args.assay_transfer_cache == DEFAULT_CACHE_BUNDLE:
        args.assay_transfer_cache = (
            INDIRECT_MORGAN_SEMANTIC_V3_CACHE_BUNDLE
            if args.harness_version == INDIRECT_FILTER_HARNESS
            else INDIRECT_MORGAN_SEMANTIC_CACHE_BUNDLE
        )
    args.max_level = args.indirect_level
    args.context_limit = 100
    args.l1_min_contrast = 0
    args.indirect_record_limit_per_level = (
        100 if args.harness_version == INDIRECT_FILTER_HARNESS
        and args.reranking == "morgan-parent-llm-semantic" else 25
    )
    args.query_prior = 'none'
    args.record_pool = args.cache_pool = 'all'


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from predict.harnesses.progressive.retrieval_cache import (
        DEFAULT_CACHE_BUNDLE,
        DEFAULT_GOLD_CONTEXT_MAPPING,
        L1_CONTEXT_CACHE_BUNDLE,
        SEMANTIC_BUCKET_CACHE_BUNDLE,
        load_cache_policy,
    )
    from predict.harnesses.progressive.level_selection import IMPORTED, TASKS
    from predict.harnesses.progressive.prompt import split_prompt_version

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    explicit_prompt_version = any(
        value == "--prompt-version" or value.startswith("--prompt-version=")
        for value in raw_argv
    )
    explicit_reranking = any(
        value == "--reranking" or value.startswith("--reranking=")
        for value in raw_argv
    )
    explicit_gold_label_version = any(
        value == "--gold-label-version" or value.startswith("--gold-label-version=")
        for value in raw_argv
    )
    parser = argparse.ArgumentParser(description="Cache-matched Reranked Progressive", allow_abbrev=False)
    parser.add_argument(
        '--harness-version',
        choices=('reranked-progressive-v2', 'reranked-progressive-v3',
                 'reranked-progressive-v4', 'reranked-progressive-l1-context-v1',
                 'reranked-progressive-l1-context-l2-v1',
                 'reranked-progressive-l1-context-l2-weighted-v1',
                 'reranked-progressive-l1-context-l2-morgan-bucket-v1',
                 TDC_MIXED_L1_HARNESS,
                 INDIRECT_ONLY_HARNESS, INDIRECT_FILTER_HARNESS,
                 *FULL_FLAT_PROGRESSIVE_HARNESSES),
        default='reranked-progressive-v2',
    )
    parser.add_argument('--reranking', choices=(
        'joint', 'morgan', 'morgan-contrastive', 'assay-transfer',
        'assay-transfer-within-morgan', 'assay-transfer-contrastive',
        'semantic-lap',
        'semantic-weighted',
        'morgan-parent-control',
        'morgan-parent-semantic',
        'morgan-parent-llm-semantic',
    ), default='assay-transfer',
                        help='Joint changes L1 only; assay-transfer always uses Morgan at L5.')
    parser.add_argument('--tasks', nargs='+', choices=tuple(TASKS), default=['bioavailability_ma'])
    parser.add_argument('--evidence-library', action='append', default=[], metavar='TASK=PATH',
                        help='Repeatable task library directory; defaults to data/evidence_libraries/<task>/v10.')
    parser.add_argument('--level-mapper', type=Path, default=IMPORTED, help='Source-UID level mapping manifest.')
    parser.add_argument('--gold-context-mapping', type=Path, default=DEFAULT_GOLD_CONTEXT_MAPPING,
                        help='Verified BBB training-context membership manifest; other tasks are unchanged.')
    parser.add_argument('--allow-frozen-l1-vote-scores', action='store_true',
                        help='Accept frozen BBB V9 scores when only mapped vote percentages changed; audit the approximation.')
    parser.add_argument('--assay-transfer-cache', type=Path, default=DEFAULT_CACHE_BUNDLE,
                        help='Task/split/level cache-manifest YAML; required for matched Morgan pools too.')
    record_pools = {'assay-transfer-trained': 'tool-accepted',
                    'all_transfer_eligible': 'tool-compatible', 'all': 'all'}
    parser.add_argument('--record_pool', '-record_pool', choices=tuple(record_pools),
                        default='all',
                        help='Cached later-level record view: training/calibration buckets, finite-scalar records, or all records. L1 and Morgan L5 are unchanged; parent count comes from the cache manifest.')
    parser.add_argument('--l1-molecules', dest='context_limit', type=int, default=10)
    parser.add_argument('--l1-contexts', type=int, default=None,
                        help='Number of parent-condition L1 cards for the L1-context harness.')
    parser.add_argument('--l1-min-contrast', type=int, default=3,
                        help='Minimum examples of each binary label in contrastive L1.')
    parser.add_argument('--morgan-primary-parent-width', type=int, default=100)
    parser.add_argument('--l1-records-per-molecule', dest='record_limit_per_context_level', type=int, default=10)
    parser.add_argument('--records-per-level', dest='indirect_record_limit_per_level', type=int, default=50)
    parser.add_argument(
        '--level-record-limit', action='append', type=_level_record_limit, default=[],
        metavar='LEVEL=K',
        help='Override --records-per-level for one later level; repeat as needed.',
    )
    parser.add_argument('--l2-molecules', type=int, default=10)
    parser.add_argument('--l2-records-per-molecule', type=int, default=10)
    parser.add_argument('--ranking-tie-seed', type=int, default=0)
    parser.add_argument('--query-prior', choices=('cached', 'none'), default='cached')
    parser.add_argument('--prior-root', dest='single_source_root', default=str(DEFAULT_SINGLE_CACHE_ROOT))
    parser.add_argument('--evaluation-subset', choices=('valid', 'test'), default='valid')
    parser.add_argument(
        '--allow-test-inference', action='store_true',
        help='Explicitly authorize non-prepare-only inference on the formal test split.',
    )
    parser.add_argument(
        '--gold-label-version', choices=('current', 'v1'), default='current',
        help='Use CURRENT gold labels unless an immutable historical V1 cache is selected.',
    )
    parser.add_argument('--max-level', type=int, default=0, help='0 runs all task levels.')
    parser.add_argument(
        '--reasoning-phase', choices=('l1', 'indirect-update'), default=None,
        help='Two-call full-flat progressive phase.',
    )
    parser.add_argument('--l1-prior-run', type=Path)
    parser.add_argument(
        '--require-complete-l1-prior', action='store_true',
        help=(
            'Require a completed, full-coverage L1 source. By default an indirect '
            'run reuses finished L1 queries and immediately reruns missing ones.'
        ),
    )
    parser.add_argument('--indirect-level', type=int, choices=(2, 3, 4))
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--indices', nargs='*', type=int)
    parser.add_argument('--output-root')
    parser.add_argument('--trace-root', default=str(DEFAULT_TRACE_ROOT))
    parser.add_argument('--execution-mode', choices=('live', 'throughput'), default='throughput',
                        help='Live publishes three samples then pauses; throughput runs all samples privately.')
    parser.add_argument('--prompt-version', default='reranked_progressive_v8',
                        help='Immutable prompt bundle directory name.')
    parser.add_argument(
        '--tdc-mixed-l1-cache',
        default=str(cache_profile_root(TDC_MIXED_L1_PROFILE)),
        help='Immutable Gold-v1 plus TDC Morgan top-10 cache root.',
    )
    parser.add_argument(
        '--molecule-description-mode',
        choices=('none', *MOLECULE_DESCRIPTION_COLUMNS),
        default='none',
        help='Add hash-pinned cached Quotient metadata to supported prompt levels.',
    )
    parser.add_argument(
        '--molecule-description-cache-version',
        choices=tuple(MOLECULE_DESCRIPTION_ARTIFACTS),
        default='v1',
        help='Select the immutable Quotient description artifact.',
    )
    parser.add_argument('--continue-after-pilot', action='store_true',
                        help='Resume past the three-sample review gate.')
    parser.add_argument('--legacy', action='store_true',
                        help='Allow an explicitly selected archived retrieval cache.')
    parser.add_argument('--live-run-id', default='', help=argparse.SUPPRESS)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--skip-tool-prefetch', action='store_true',
                        help='Skip tools for offline preparation or a prompt that declares tools disabled.')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--base-url', default=BASE_URL)
    parser.add_argument('--api-key-env', default='LITE_LLM_KEY')
    parser.add_argument('--provider-pool-config', default=str(DEFAULT_PROVIDER_POOL_CONFIG),
                        help='Mutable candidate endpoint JSON; healthy exact-model endpoints are selected at launch.')
    parser.add_argument('--env-file', default=str(DEFAULT_ENV_FILE))
    parser.add_argument('--tool-service-url', default='http://127.0.0.1:8765')
    parser.add_argument('--parallelism', type=int, default=None,
                        help='Required global outstanding-request budget for inference.')
    parser.add_argument('--endpoint-concurrency-budget', type=int, default=512)
    parser.add_argument('--preparation-workers', type=int, default=32)
    parser.add_argument('--max-tokens', type=int, default=262144)
    parser.add_argument('--timeout-s', type=int, default=900)
    parser.add_argument('--transport-max-retries', type=int, default=0)
    args = parser.parse_args(raw_argv)
    if len({level for level, _ in args.level_record_limit}) != len(args.level_record_limit):
        parser.error('--level-record-limit may specify each level only once')
    args.level_record_limits = dict(args.level_record_limit)
    args.cache_pool = record_pools[args.record_pool]
    args.joint_panel_sizes = (3, 7) if args.harness_version == 'reranked-progressive-v4' else None
    if args.harness_version == TDC_MIXED_L1_HARNESS:
        if set(args.tasks) - {"bbb_martins", "bioavailability_ma"}:
            parser.error('tdc-mixed-progressive-v1 supports BBB and Bioavailability only')
        if any((args.reasoning_phase, args.l1_prior_run, args.require_complete_l1_prior)):
            parser.error('tdc-mixed-progressive-v1 does not accept full-flat phase options')
        if explicit_gold_label_version:
            parser.error('tdc-mixed-progressive-v1 selects its TDC v1 queries directly')
        if explicit_reranking and args.reranking != 'morgan':
            parser.error('tdc-mixed-progressive-v1 supports Morgan ranking only')
        args.reranking = 'morgan'
        if explicit_prompt_version and args.prompt_version != TDC_MIXED_L1_PROMPT:
            parser.error(
                f'tdc-mixed-progressive-v1 requires --prompt-version {TDC_MIXED_L1_PROMPT}'
            )
        args.prompt_version = TDC_MIXED_L1_PROMPT
        if args.max_level not in {0, 1}:
            parser.error('tdc-mixed-progressive-v1 is an L1-only harness')
        args.max_level = 1
        args.context_limit = 10
        args.query_prior = 'none'
        args.record_pool = args.cache_pool = 'all'
        args.skip_tool_prefetch = True
    elif args.harness_version in FULL_FLAT_PROGRESSIVE_HARNESSES:
        if args.reasoning_phase is None:
            parser.error('full-flat-progressive requires --reasoning-phase')
        if args.reasoning_phase == 'l1' and args.l1_prior_run is not None:
            parser.error('L1 does not accept --l1-prior-run')
        if args.reasoning_phase == 'l1' and args.require_complete_l1_prior:
            parser.error('L1 does not accept --require-complete-l1-prior')
        if args.require_complete_l1_prior and args.l1_prior_run is None:
            parser.error('--require-complete-l1-prior requires --l1-prior-run')
        if args.reranking != 'assay-transfer-contrastive':
            parser.error('full-flat-progressive fixes assay-transfer-contrastive retrieval')
        if args.molecule_description_mode != 'none':
            parser.error('full-flat-progressive does not expose molecule descriptions')
        args.prompt_version = FULL_FLAT_PROGRESSIVE_HARNESSES[args.harness_version]
        args.context_limit = 10
        if args.harness_version == FULL_FLAT_PROGRESSIVE_HARNESS:
            args.l1_min_contrast = 1
        args.morgan_primary_parent_width = 25
        args.record_limit_per_context_level = 10
        args.indirect_record_limit_per_level = 10
        args.query_prior = 'cached'
        args.record_pool = args.cache_pool = 'all'
        args.max_level = 1 if args.reasoning_phase == 'l1' else 0
        args.skip_tool_prefetch = True
        if args.l1_prior_run is not None:
            args.l1_prior_run = args.l1_prior_run.resolve()
    elif (
        args.reasoning_phase is not None
        or args.l1_prior_run is not None
        or args.require_complete_l1_prior
    ):
        parser.error(
            '--reasoning-phase, --l1-prior-run, and --require-complete-l1-prior '
            'require full-flat-progressive'
        )
    elif args.harness_version == 'reranked-progressive-l1-context-v1':
        from predict.harnesses.progressive.prompt import split_prompt_version

        if args.reranking not in {
            'morgan', 'morgan-contrastive', 'assay-transfer',
            'assay-transfer-within-morgan',
            'assay-transfer-contrastive'
        }:
            parser.error('The L1-context harness supports Morgan and assay-transfer modes only')
        if args.l1_contexts is not None:
            if args.context_limit != 10:
                parser.error('Use --l1-contexts, not --l1-molecules, for the L1-context harness')
            args.context_limit = args.l1_contexts
        if args.max_level not in {0, 1}:
            parser.error('The L1-context harness requires --max-level 1')
        args.max_level = 1
        l1_context_prompts = {
            'reranked_progressive_l1_context_v1',
            'reranked_progressive_l1_context_v2',
            'reranked_progressive_l1_context_v3',
            'reranked_progressive_l1_context_v4',
            'reranked_progressive_l1_context_order_only_v1',
            'reranked_progressive_l1_context_v2_references_v1',
            'reranked_progressive_l1_context_order_only_v1_references_v1',
        }
        if args.prompt_version == 'reranked_progressive_v8':
            args.prompt_version = 'reranked_progressive_l1_context_v2'
        elif split_prompt_version(args.prompt_version)[0] not in l1_context_prompts:
            parser.error('The L1-context harness requires an L1-context prompt bundle')
        if args.assay_transfer_cache == DEFAULT_CACHE_BUNDLE:
            args.assay_transfer_cache = L1_CONTEXT_CACHE_BUNDLE
    elif args.harness_version == 'reranked-progressive-l1-context-l2-v1':
        from predict.harnesses.progressive.prompt import split_prompt_version
        from predict.harnesses.progressive.retrieval_cache import CONTEXT_L2_CACHE_BUNDLE

        if args.reranking not in {'morgan', 'semantic-lap'}:
            parser.error('The context-L2 harness supports Morgan and semantic-lap only')
        if args.l1_contexts is not None:
            if args.context_limit != 10:
                parser.error('Use --l1-contexts, not --l1-molecules, for the context-L2 harness')
            args.context_limit = args.l1_contexts
        if args.context_limit != 10 or args.record_limit_per_context_level != 10:
            parser.error('The context-L2 harness fixes L1 at 10 contexts and 10 records per context')
        if args.max_level not in {0, 2}:
            parser.error('The context-L2 harness requires --max-level 2')
        args.max_level = 2
        args.indirect_record_limit_per_level = 12
        args.record_pool = args.cache_pool = 'all'
        prompts = {
            'reranked_progressive_l1_context_l2_semantic_v1',
            'reranked_progressive_l1_context_l2_semantic_v1_references_v1',
            *MOLECULE_DESCRIPTION_PROMPTS[args.harness_version],
        }
        if args.prompt_version == 'reranked_progressive_v8':
            args.prompt_version = 'reranked_progressive_l1_context_l2_semantic_v1'
        elif split_prompt_version(args.prompt_version)[0] not in prompts:
            parser.error('The context-L2 harness requires its versioned L1/L2 prompt')
        if args.assay_transfer_cache == DEFAULT_CACHE_BUNDLE:
            args.assay_transfer_cache = CONTEXT_L2_CACHE_BUNDLE
    elif args.harness_version == 'reranked-progressive-l1-context-l2-weighted-v1':
        from predict.harnesses.progressive.prompt import split_prompt_version
        from predict.harnesses.progressive.retrieval_cache import (
            CONTEXT_L2_WEIGHTED_CACHE_BUNDLE,
        )

        if args.reranking not in {'morgan', 'semantic-weighted'}:
            parser.error('The weighted context-L2 harness supports Morgan and semantic-weighted only')
        if args.l1_contexts is not None:
            if args.context_limit != 10:
                parser.error('Use --l1-contexts, not --l1-molecules, for the weighted context-L2 harness')
            args.context_limit = args.l1_contexts
        if args.context_limit != 10 or args.record_limit_per_context_level != 10:
            parser.error('The weighted context-L2 harness fixes L1 at 10 contexts and 10 records per context')
        if args.max_level not in {0, 2}:
            parser.error('The weighted context-L2 harness requires --max-level 2')
        args.max_level = 2
        args.indirect_record_limit_per_level = 12
        args.record_pool = args.cache_pool = 'all'
        prompts = {
            'reranked_progressive_l1_context_l2_weighted_v1',
            'reranked_progressive_l1_context_l2_weighted_v1_references_v1',
            *MOLECULE_DESCRIPTION_PROMPTS[args.harness_version],
        }
        if args.prompt_version == 'reranked_progressive_v8':
            args.prompt_version = 'reranked_progressive_l1_context_l2_weighted_v1'
        elif split_prompt_version(args.prompt_version)[0] not in prompts:
            parser.error('The weighted context-L2 harness requires its versioned nested L1/L2 prompt')
        if args.assay_transfer_cache == DEFAULT_CACHE_BUNDLE:
            args.assay_transfer_cache = CONTEXT_L2_WEIGHTED_CACHE_BUNDLE
    elif args.harness_version == 'reranked-progressive-l1-context-l2-morgan-bucket-v1':
        from predict.harnesses.progressive.prompt import split_prompt_version
        from predict.harnesses.progressive.retrieval_cache import (
            CONTEXT_L2_MORGAN_SEMANTIC_CACHE_BUNDLE,
        )

        if args.reranking not in {'morgan-parent-control', 'morgan-parent-semantic'}:
            parser.error('The molecule-first L2 harness requires a matched parent mode')
        if (args.context_limit, args.record_limit_per_context_level,
                args.l2_molecules, args.l2_records_per_molecule) != (10, 10, 10, 10):
            parser.error('The molecule-first L2 harness fixes L1 and L2 shapes at 10 by 10')
        if args.l1_contexts is not None or args.max_level not in {0, 2}:
            parser.error('The molecule-first L2 harness requires --max-level 2')
        args.max_level = 2
        args.indirect_record_limit_per_level = 100
        args.record_pool = args.cache_pool = 'all'
        prompt = 'reranked_progressive_l1_context_l2_morgan_bucket_v1'
        if args.prompt_version == 'reranked_progressive_v8':
            args.prompt_version = prompt
        elif split_prompt_version(args.prompt_version)[0] != prompt:
            parser.error('The molecule-first L2 harness requires its versioned prompt')
        if args.assay_transfer_cache == DEFAULT_CACHE_BUNDLE:
            args.assay_transfer_cache = CONTEXT_L2_MORGAN_SEMANTIC_CACHE_BUNDLE
    elif args.harness_version in INDIRECT_HARNESSES:
        _configure_indirect_only_args(args, parser)
    elif args.reranking in {
        'morgan-contrastive', 'assay-transfer-within-morgan',
        'assay-transfer-contrastive'
    }:
        parser.error(f'{args.reranking} requires reranked-progressive-l1-context-v1')
    if args.harness_version == 'reranked-progressive-v3':
        if args.reranking == 'joint':
            parser.error('Reranked Progressive v3 supports morgan or assay-transfer only')
        if args.record_pool != 'all':
            parser.error('Reranked Progressive v3 requires --record_pool all')
        if args.assay_transfer_cache == DEFAULT_CACHE_BUNDLE:
            args.assay_transfer_cache = SEMANTIC_BUCKET_CACHE_BUNDLE
    if args.harness_version == 'reranked-progressive-v4' and args.reranking != 'joint':
        parser.error('Reranked Progressive v4 is the fixed assay-transfer 3 + Morgan 7 joint L1')
    if (
        args.evaluation_subset == 'test'
        and not args.prepare_only
        and not args.allow_test_inference
    ):
        parser.error(
            'Formal-test inference requires --allow-test-inference; '
            'use --prepare-only to render it without inference.'
        )
    if not 1 <= args.endpoint_concurrency_budget <= MAX_ENDPOINT_CONCURRENCY_BUDGET:
        parser.error('--endpoint-concurrency-budget must be between 1 and 512')
    if (args.parallelism is not None
            and not 1 <= args.parallelism <= args.endpoint_concurrency_budget):
        parser.error('--parallelism must be positive and within the endpoint budget')
    if min(args.preparation_workers, args.max_tokens, args.timeout_s,
           args.record_limit_per_context_level, args.indirect_record_limit_per_level) < 1:
        parser.error('Worker, token, timeout, and record limits must be positive')
    if min(args.transport_max_retries, args.limit, args.max_level) < 0:
        parser.error('Retries, limit, and max-level must be non-negative')
    if args.context_limit < 1:
        parser.error('--l1-molecules must be positive and fit the selected cache pool')
    if args.l1_min_contrast < 0:
        parser.error('--l1-min-contrast must be non-negative')
    if (args.reranking in {'morgan-contrastive', 'assay-transfer-contrastive'}
            and args.l1_min_contrast
            and args.context_limit < 2 * args.l1_min_contrast):
        parser.error('Contrastive L1 requires --l1-contexts K >= 2*M')
    if args.reranking == 'joint' and args.context_limit != 10:
        parser.error('Joint L1 requires --l1-molecules 10: five per method, no refill')
    if args.molecule_description_mode != 'none':
        allowed_prompts = MOLECULE_DESCRIPTION_PROMPTS.get(args.harness_version)
        if allowed_prompts is None:
            parser.error(
                '--molecule-description-mode is unsupported by this harness'
            )
        if (
            not explicit_prompt_version
            or split_prompt_version(args.prompt_version)[0] not in allowed_prompts
        ):
            parser.error(
                '--molecule-description-mode requires an explicit supported prompt version'
            )
    args.tasks = list(dict.fromkeys(args.tasks))
    args.evidence_libraries = {task: Path('data/evidence_libraries') / task / 'v10' for task in args.tasks}
    overrides = set()
    for value in args.evidence_library:
        task, separator, path = value.partition('=')
        if not separator or task not in args.tasks or not path.strip() or task in overrides:
            parser.error('--evidence-library requires one TASK=PATH per selected task')
        args.evidence_libraries[task] = Path(path)
        overrides.add(task)
    try:
        args.retrieval_policies = {task: load_cache_policy(args.assay_transfer_cache, task,
            args.evaluation_subset, args.reranking, args.max_level) for task in args.tasks}
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    uses_ranked_level_cache = any(
        policy.get('selection_contract') in {
            'ranked_level_retrieval.v2', 'ranked_uid_retrieval.v1'
        }
        for policy in args.retrieval_policies.values()
    )
    if uses_ranked_level_cache:
        if args.record_pool != 'all':
            parser.error('ranked_level_retrieval.v2 requires --record_pool all')
        if (
            args.reranking not in {'morgan', 'assay-transfer'}
            and args.harness_version not in FULL_FLAT_PROGRESSIVE_HARNESSES
        ):
            parser.error('ranked_level_retrieval.v2 supports morgan or assay-transfer only')
        uses_v2 = any(
            policy.get('selection_contract') == 'ranked_level_retrieval.v2'
            for policy in args.retrieval_policies.values()
        )
        oversized = {
            level: limit for level, limit in args.level_record_limits.items()
            if limit > 100
        }
        if uses_v2 and (args.indirect_record_limit_per_level > 100 or oversized):
            parser.error('ranked_level_retrieval.v2 record requests cannot exceed 100')
    if args.harness_version == 'reranked-progressive-v4' and any(
            policy.get('selection_contract') not in {
                'cache_matched_retrieval.v3', 'ranked_evidence_retrieval.v1'
            } for policy in args.retrieval_policies.values()):
        parser.error('Reranked Progressive v4 requires a pooled L5 indexed cache')
    if args.harness_version == 'reranked-progressive-l1-context-v1' and any(
            policy.get('selection_contract') not in {
                'l1_context_retrieval.v1', 'l1_context_retrieval.v2'
            }
            for policy in args.retrieval_policies.values()):
        parser.error('The L1-context harness requires an L1 context cache')
    if args.harness_version in FULL_FLAT_PROGRESSIVE_HARNESSES and any(
            policy.get('selection_contract') != 'ranked_uid_retrieval.v1'
            for policy in args.retrieval_policies.values()):
        parser.error('full-flat-progressive requires ranked_uid_retrieval.v1')
    if args.harness_version == 'reranked-progressive-l1-context-l2-v1' and any(
            policy.get('selection_contract') != 'l1_context_semantic_l2.v1'
            for policy in args.retrieval_policies.values()):
        parser.error('The context-L2 harness requires its combined cache')
    if args.harness_version == 'reranked-progressive-l1-context-l2-weighted-v1' and any(
            policy.get('selection_contract') != 'l1_context_semantic_weighted_l2.v1'
            for policy in args.retrieval_policies.values()):
        parser.error('The weighted context-L2 harness requires its combined cache')
    if args.harness_version == 'reranked-progressive-l1-context-l2-morgan-bucket-v1' and any(
            policy.get('selection_contract') != 'l1_context_morgan_semantic_l2.v1'
            for policy in args.retrieval_policies.values()):
        parser.error('The molecule-first L2 harness requires its combined cache')
    if args.harness_version in INDIRECT_HARNESSES and any(
            policy.get('selection_contract') != (
                'indirect_morgan_semantic_l2_l4.v3'
                if args.harness_version == INDIRECT_FILTER_HARNESS
                else 'indirect_morgan_semantic_l2_l4.v1'
            )
            for policy in args.retrieval_policies.values()):
        parser.error('The indirect-only harness requires its L2-L4 cache')
    archived_root = ARCHIVE_CACHE_ROOT.resolve()
    archived_manifests = []
    for policy in args.retrieval_policies.values():
        paths = [policy.get('cache_manifest'), policy.get('cache_index')]
        paths.extend((policy.get('cache_manifests') or {}).values())
        archived_manifests.extend(Path(path).resolve() for path in paths if path)
    uses_archive = any(path.is_relative_to(archived_root) for path in archived_manifests)
    l1_context_harness = args.harness_version in {
        'reranked-progressive-l1-context-v1',
        'reranked-progressive-l1-context-l2-v1',
        'reranked-progressive-l1-context-l2-weighted-v1',
        'reranked-progressive-l1-context-l2-morgan-bucket-v1',
        *FULL_FLAT_PROGRESSIVE_HARNESSES,
        TDC_MIXED_L1_HARNESS,
    }
    approved_archive_harness = l1_context_harness or args.harness_version in INDIRECT_HARNESSES
    if uses_archive and not args.legacy and not approved_archive_harness:
        parser.error('archived retrieval caches require --legacy')
    if args.legacy and (not uses_archive or approved_archive_harness):
        parser.error('--legacy requires an archived retrieval cache')
    output_version = (
        'semantic_bucket_reranking_v1'
        if args.harness_version == 'reranked-progressive-v3'
        else args.prompt_version
    )
    output_name = f'{output_version}_{args.reranking.replace("-", "_")}'
    if args.molecule_description_mode != 'none':
        output_name += f'_molecule_description_{args.molecule_description_mode}'
        if args.molecule_description_cache_version != 'v1':
            output_name += f'_cache_{args.molecule_description_cache_version}'
    if l1_context_harness:
        output_name += f'_k{args.context_limit}'
        if args.reranking in {'morgan-contrastive', 'assay-transfer-contrastive'}:
            output_name += f'_m{args.l1_min_contrast}'
        output_name += (
            f'_{TDC_MIXED_L1_PROFILE}'
            if args.harness_version == TDC_MIXED_L1_HARNESS
            else f'_{Path(args.assay_transfer_cache).stem}'
        )
    if args.harness_version in FULL_FLAT_PROGRESSIVE_HARNESSES:
        output_name += f'_{args.reasoning_phase}'
    args.output_root = args.output_root or str(
        Path('outputs/paper/assay_transfer_harness/joseph') / output_name
    )
    # These are internal adapter settings, not historical CLI aliases.
    args.profile = (
        'standard'
        if args.harness_version == TDC_MIXED_L1_HARNESS
        else 'context_records'
    )
    try:
        from predict.harnesses.progressive.prompt import prompt_assets

        prompt_settings = prompt_assets(args.prompt_version)['settings']
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    metadata_contract = prompt_settings.get('molecule_metadata_contract')
    if args.molecule_description_mode != 'none' and (
        not metadata_contract
        or args.molecule_description_mode not in metadata_contract.get('modes', ())
    ):
        parser.error('The selected prompt does not support this molecule description mode')
    args.molecule_description_missing_policy = (
        metadata_contract['missing_policy']
        if args.molecule_description_mode != 'none'
        else None
    )
    if args.skip_tool_prefetch and not (
        args.prepare_only or prompt_settings.get('tools') is False
    ):
        parser.error('--skip-tool-prefetch requires --prepare-only or a no-tool prompt')
    if prompt_settings.get('tools') is False and not args.skip_tool_prefetch:
        parser.error('This prompt requires --skip-tool-prefetch')
    if prompt_settings.get('query_prior') is False and args.query_prior != 'none':
        parser.error('This prompt requires --query-prior none')
    if prompt_settings.get('query_prior') is True and args.query_prior != 'cached':
        parser.error('This prompt requires --query-prior cached')
    if (prompt_settings.get('max_level') and args.max_level != prompt_settings['max_level']
            and args.harness_version not in INDIRECT_HARNESSES):
        parser.error(f"This prompt requires --max-level {prompt_settings['max_level']}")
    args.assay_transfer_prompt_version = args.prompt_version
    args.retrieval_policy = (
        None
        if args.harness_version == TDC_MIXED_L1_HARNESS
        else args.assay_transfer_cache
    )
    args.level_mapping = str(args.level_mapper)
    stages = args.retrieval_policies[args.tasks[0]]['stages']
    args.context_ranking = stages.get('L1', next(iter(stages.values())))
    args.record_sampler = 'plain'
    args.context_record_l3_l5 = args.harness_version != TDC_MIXED_L1_HARNESS
    args.assay_transfer_cache_profile = 'cache_matched_v2'
    args.l2_record_limit_per_context = args.record_limit_per_context_level
    args.indirect_final_level_record_limit = 0
    args.reuse_context_l2 = False
    args.v21_selection_mode = 'molecule_cards'
    args.l1_source = (
        'tdc_mixed_train'
        if args.harness_version == TDC_MIXED_L1_HARNESS
        else 'gold_train'
    )
    args.l1_ranking = (
        'morgan'
        if args.harness_version == TDC_MIXED_L1_HARNESS
        else 'v9'
    )
    args.gold_l1_min_similarity = 0.0
    args.benchmark_data_root = 'data/gold_labels'
    args.v9_ranking_root = str(DEFAULT_V9_RANKING_ROOT)
    args.v7_root = str(context_records.V7_ROOT)
    _configure_evaluation_subset(args)
    return args


def main(argv: list[str] | None = None) -> int:
    from predict.live import PILOT_SIZE, create_run, update_run

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    if not args.prepare_only and args.parallelism is None:
        raise SystemExit("full-batch inference requires explicit --parallelism")
    args.live_method = (
        f"progressive_{args.reranking}_{args.record_pool}_records"
        f"{args.indirect_record_limit_per_level}"
    )
    if args.harness_version in {
        'reranked-progressive-l1-context-v1',
        'reranked-progressive-l1-context-l2-v1',
        'reranked-progressive-l1-context-l2-weighted-v1',
        'reranked-progressive-l1-context-l2-morgan-bucket-v1',
        *FULL_FLAT_PROGRESSIVE_HARNESSES,
        TDC_MIXED_L1_HARNESS,
    }:
        args.live_method += f'_k{args.context_limit}'
        if args.reranking in {'morgan-contrastive', 'assay-transfer-contrastive'}:
            args.live_method += f'_m{args.l1_min_contrast}'
    if args.harness_version in FULL_FLAT_PROGRESSIVE_HARNESSES:
        args.live_method += f'_{args.reasoning_phase}'
    if args.molecule_description_mode != 'none':
        args.live_method += f'_quotient_{args.molecule_description_mode}'
        if args.molecule_description_cache_version != 'v1':
            args.live_method += f'_cache_{args.molecule_description_cache_version}'
    command = [sys.executable, "-m", "predict.harnesses.progressive", *raw_argv]
    command.extend(["--output-root", args.output_root, "--prompt-version", args.prompt_version])
    args.live_run_dirs = {}
    args.live_run_ids = {}
    for task in args.tasks:
        description_artifact = MOLECULE_DESCRIPTION_ARTIFACTS[
            args.molecule_description_cache_version
        ]
        run_dir = create_run(
            root=args.trace_root,
            dataset=task,
            method=args.live_method,
            command=command,
            requested_id=args.live_run_id,
            execution_mode=args.execution_mode,
            metadata={
                "harness": "progressive",
                "harness_version": args.harness_version,
                "prompt_version": args.prompt_version,
                "molecule_description_mode": args.molecule_description_mode,
                "molecule_description": (
                    {
                        "mode": args.molecule_description_mode,
                        "cache_version": args.molecule_description_cache_version,
                        "column": MOLECULE_DESCRIPTION_COLUMNS[args.molecule_description_mode],
                        "path": str(description_artifact["path"].resolve()),
                        "sha256": description_artifact["sha256"],
                        "missing_policy": args.molecule_description_missing_policy,
                    }
                    if args.molecule_description_mode != "none"
                    else {"mode": "none"}
                ),
                "retrieval_cache_bundle": str(args.assay_transfer_cache),
                "evaluation_subset": args.evaluation_subset,
                "pilot_size": PILOT_SIZE,
                "output_root": args.output_root,
            },
        )
        args.live_run_dirs[task] = run_dir
        args.live_run_ids[task] = run_dir.name
        update_run(
            run_dir,
            resume_command=[*command, "--live-run-id", run_dir.name],
        )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
