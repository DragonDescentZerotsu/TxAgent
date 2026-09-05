"""Run visible append-only progressive reasoning over conditioned assay families.

The historical cumulative-family and geometric assay-prefix launchers remain
unchanged.  This launcher consumes the same frozen mechanism-tagged indices,
reuses complete visible none/single branches, and checkpoints one progressive
state per query and family level.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import time
from collections import deque
from functools import lru_cache
from typing import Any, Mapping

from tools.chembl_tool.common.assay_retrieval import (
    build_family_molecule_prefix_view,
    retrieve_family_molecule_prefixes,
)
from tools.chembl_tool.common.json_utils import (
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.common.openai_reasoning_client import (
    OpenAICompatibleClient,
    ToolServiceClient,
)
from tools.chembl_tool.common.openai_provider_pool import (
    OpenAIProviderPool,
    ProviderPoolConfig,
    ProviderPoolExhausted,
    ProviderSpec,
    load_env_file,
    load_provider_pool_config,
)
from tools.chembl_tool.common.progressive_assay_reasoning import (
    AUGMENTED_MOLECULE_LIMIT,
    INITIAL_MOLECULE_LIMIT,
    NEW_MOLECULE_LIMIT,
    PROGRESSIVE_PROTOCOL_VERSION,
    ProgressiveTaskContract,
    append_evidence,
    attach_analog_tool_summaries,
    build_progressive_messages,
    card_alias_maps,
    card_ids,
    extract_cumulative_evidence,
    progressive_state_errors,
    restore_card_ids,
    select_initial_evidence,
    select_progressive_delta,
    state_from_content,
)
from tools.chembl_tool.common.reasoning_payload import external_condition_sentence
from tools.chembl_tool.common.reasoning_validation import (
    response_validation_errors,
    call_with_json_validation,
    structured_response_is_valid,
)
from tools.chembl_tool.common.reasoning_race import ParallelRetryClient
from tools.chembl_tool.common.retrieval_features import (
    STRUCTURAL_ELIGIBILITY_VERSION,
    monatomic_query_element,
    structural_eligibility_metadata,
)
from tools.chembl_tool.common.starling.conditioned_benchmark import (
    SPLIT_SCHEMES,
    split_path,
    task_root,
)
from tools.chembl_tool.tasks.bbb_martins import experiment_config as bbb_config
from tools.chembl_tool.tasks.bioavailability_ma import experiment_config as bio_config
from tools.chembl_tool.tasks.skin_reaction import experiment_config as skin_config


MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
BASE_URL = "http://127.0.0.1:50001/v1"
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/paper/starling_conditioned_assay_progressive_visible_v10_bio_legacy_gold_vote_pure_v1/"
    "scaffold_valid_deepseek_v4_flash_0731"
)
ARCHIVED_SINGLE_CACHE_ROOT = Path(
    "outputs/archive/conditioned_assay_superseded_20260827/family_curve/"
    "scaffold_valid_top20_control_matrix_deepseek_v4_flash_0731_v1/"
    "visible_standard"
)
TASK_NAMES = ("bbb_martins", "bioavailability_ma", "skin_reaction")
REFERENCE_POOL = "direct_only_heldout_filtered"
IDENTITY_POLICY_BY_SPLIT = {
    "scaffold": "scaffold_disjoint",
    "random": "parent_disjoint",
}

_MODEL_IDENTITY_ALIASES = {
    "deepseek-ai/deepseek-v4-flash-0731": "deepseek-v4-flash-0731",
    "deepseek/deepseek-v4-flash-0731": "deepseek-v4-flash-0731",
    "deepseek/deepseek-v4-flash": "deepseek-v4-flash-0731",
    "deepseek-v4-flash": "deepseek-v4-flash-0731",
}


def _model_identity(model: str) -> str:
    normalized = str(model or "").strip().lower()
    return _MODEL_IDENTITY_ALIASES.get(normalized, normalized)


def _neighbor_identity_policy(split_scheme: str) -> str:
    try:
        return IDENTITY_POLICY_BY_SPLIT[split_scheme]
    except KeyError as exc:
        raise ValueError(f"unsupported split scheme: {split_scheme}") from exc


_RESUME_INVARIANT_FIELDS = (
    "experiment",
    "split_scheme",
    "evaluation_subset",
    "tasks",
    "visibility_mode",
    "reference_pool",
    "neighbor_identity_policy",
    "selection",
    "candidate_generation",
    "prompt_profile",
    "max_tokens",
    "temperature",
    "thinking",
    "reasoning_effort",
    "tool_prefetch_complete",
    "evaluation_indices_by_task",
    "inputs",
    "query_prior_source_root",
    "progressive_reuse_source_root",
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
RANDOM_INDEX_ROOT = SOURCE_PURITY_ROOT / "indices_random_valid_test"


@dataclass(frozen=True)
class ProgressiveTaskSpec:
    input_jsonl: Path
    index: Path
    family_manifest: Path


def _progressive_task_specs(
    split_scheme: str, evaluation_subset: str = "valid"
) -> dict[str, ProgressiveTaskSpec]:
    index_root = (
        SOURCE_PURITY_ROOT / "indices"
        if split_scheme == "scaffold"
        else RANDOM_INDEX_ROOT
    )
    return {
        "bbb_martins": ProgressiveTaskSpec(
            split_path("bbb_martins", evaluation_subset, split_scheme),
            index_root
            / "bbb_martins/mechanism_tagged_v4_source_purity_v6/assay_neighbor_index.pkl",
            SOURCE_PURITY_ROOT
            / "family_catalogs_mechanism_tagged_v1/"
            "bbb_martins_source_purity_v6/manifest.json",
        ),
        "bioavailability_ma": ProgressiveTaskSpec(
            split_path("bioavailability_ma", evaluation_subset, split_scheme),
            index_root
            / "bioavailability_ma/"
            "mechanism_tagged_v4_legacy_record_supported_v2_vote_pure_v1/"
            "assay_neighbor_index.pkl",
            SOURCE_PURITY_ROOT
            / "family_catalogs/"
            "bioavailability_ma_legacy_record_supported_v2_vote_pure_v1/manifest.json",
        ),
        "skin_reaction": ProgressiveTaskSpec(
            split_path("skin_reaction", evaluation_subset, split_scheme),
            index_root
            / "skin_reaction/mechanism_tagged_v4_source_purity_v5/assay_neighbor_index.pkl",
            SOURCE_PURITY_ROOT
            / "family_catalogs_mechanism_tagged_v1/"
            "skin_reaction_source_purity_v5/manifest.json",
        ),
    }


PROGRESSIVE_TASKS = _progressive_task_specs("scaffold")


@dataclass(frozen=True)
class PreparedQuery:
    task: str
    index: int
    query_dir: Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _levels(task: str) -> list[dict[str, Any]]:
    manifest = _read_json(PROGRESSIVE_TASKS[task].family_manifest)
    levels = [dict(row) for row in manifest.get("levels") or []]
    for row in levels:
        level = int(row["level"])
        endpoint = str(row.get("endpoint_group") or "")
        endpoint_descriptions = getattr(
            TASK_CONFIGS[task], "PROGRESSIVE_ASSAY_ENDPOINT_DESCRIPTIONS", {}
        )
        row["description"] = endpoint_descriptions.get(endpoint) or TASK_CONFIGS[
            task
        ].PROGRESSIVE_ASSAY_LEVEL_DESCRIPTIONS[level]
    if [int(row["level"]) for row in levels] != list(range(1, len(levels) + 1)):
        raise ValueError(f"{task} has a non-contiguous family-level catalog")
    return levels


def _heldout_filter_validation(
    *,
    task: str,
    split_scheme: str,
    index_manifest: Mapping[str, Any],
    heldout_path: Path,
) -> dict[str, Any]:
    current_hash = sha256_file(heldout_path)
    recorded_hash = str(index_manifest.get("heldout_molecules_jsonl_sha256") or "")
    if current_hash == recorded_hash:
        return {
            "status": "ok",
            "method": "exact_heldout_file_sha256",
            "current_sha256": current_hash,
            "recorded_sha256": recorded_hash,
        }

    receipt_path = Path("data/conditioned_benchmark/migration_receipt.json")
    receipt = _read_json(receipt_path)
    task_receipt = (receipt.get("tasks") or {}).get(task) or {}
    source_root = Path(str(task_receipt.get("legacy_source_root") or "")).resolve()
    canonical_root = Path(str(task_receipt.get("canonical_root") or "")).resolve()
    recorded_heldout = Path(
        str(index_manifest.get("heldout_molecules") or "")
    ).resolve()
    if split_scheme != "scaffold":
        raise ValueError(f"{task} heldout hash mismatch without a scaffold migration receipt")
    if not task_receipt.get("input_rows_byte_identical_expected"):
        raise ValueError(f"{task} heldout hash mismatch without byte-identical migration")
    if (
        recorded_heldout.parent != source_root
        or heldout_path.resolve().parent != canonical_root
    ):
        raise ValueError(f"{task} heldout hash mismatch has incompatible migration roots")
    split_hashes: dict[str, str] = {}
    for subset in ("valid", "test"):
        subset_receipt = (task_receipt.get("splits") or {}).get(subset) or {}
        subset_path = split_path(task, subset, split_scheme)
        subset_hash = sha256_file(subset_path)
        if (
            subset_receipt.get("byte_identical") is not True
            or subset_receipt.get("canonical_sha256") != subset_hash
            or subset_receipt.get("source_sha256") != subset_hash
        ):
            raise ValueError(
                f"{task} heldout hash mismatch is not covered by the migration receipt"
            )
        split_hashes[subset] = subset_hash
    rows = read_jsonl(heldout_path)
    n_parents = len({_stable_query_key(row)[0] for row in rows})
    if len(rows) != int(index_manifest.get("n_heldout_rows") or -1):
        raise ValueError(f"{task} migrated heldout row count mismatch")
    if n_parents != int(index_manifest.get("n_heldout_parent_identities") or -1):
        raise ValueError(f"{task} migrated heldout parent count mismatch")
    return {
        "status": "ok",
        "method": "migration_receipt_byte_identical_valid_test",
        "current_sha256": current_hash,
        "recorded_sha256": recorded_hash,
        "migration_receipt": str(receipt_path),
        "valid_sha256": split_hashes["valid"],
        "test_sha256": split_hashes["test"],
        "n_heldout_rows": len(rows),
        "n_heldout_parent_identities": n_parents,
    }


def _task_contract(task: str) -> ProgressiveTaskContract:
    try:
        return TASK_CONFIGS[task].get_progressive_task_contract()
    except KeyError as exc:
        raise ValueError(f"unsupported progressive task: {task}") from exc


def _query_dir(output_root: Path, task: str, query_index: int) -> Path:
    return output_root / task / "queries" / f"query_idx{query_index:05d}"


def _source_run_dir(task: str, query_index: int, single_root: Path) -> Path:
    batch = single_root / task / "none"
    return batch / "runs" / f"none_idx{query_index:05d}"


def _stable_query_key(record: Mapping[str, Any]) -> tuple[str, str]:
    parent = str(record.get("molecule_identity_key") or "").strip()
    condition = str(record.get("condition_group") or "").strip()
    if not parent:
        raise ValueError("conditioned benchmark row lacks molecule_identity_key")
    return parent, condition


@lru_cache(maxsize=None)
def _single_source_index(task: str, single_root_text: str) -> dict[tuple[str, str], int]:
    single_root = Path(single_root_text)
    manifest = _read_json(single_root / task / "none" / "manifest.json")
    input_path = Path(str(manifest.get("input_jsonl") or ""))
    if not input_path.is_file():
        raise FileNotFoundError(f"reusable single input is unavailable: {input_path}")
    records = read_jsonl(input_path)
    if len(records) != int(manifest.get("n_items") or -1):
        raise ValueError(f"reusable single input count disagrees with manifest: {input_path}")
    mapping: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        key = _stable_query_key(record)
        if key in mapping:
            raise ValueError(f"duplicate reusable single identity: {key}")
        mapping[key] = index
    return mapping


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


def _load_progressive_query_prior(
    task: str,
    query_index: int,
    record: Mapping[str, Any],
    source_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    """Load the frozen none/single prior copied into a completed progressive run."""
    prepared_path = (
        _query_dir(source_root, task, query_index)
        / "levels"
        / "level_1"
        / "prepared.json"
    )
    prepared = _read_json(prepared_path)
    if prepared.get("molecule_identity_key"):
        identity_matches = _stable_query_key(prepared) == _stable_query_key(record)
    else:
        identity_matches = (
            int(prepared.get("query_index") or 0) == query_index
            and str(prepared.get("query_smiles") or "") == str(record.get("drug") or "")
        )
    if not identity_matches:
        raise ValueError(
            f"query-prior source identity mismatch for {task} index {query_index}: "
            f"{prepared_path}"
        )
    if not prepared.get("tool_prefetch_complete"):
        raise ValueError(f"query-prior source lacks visible tool prefetch: {prepared_path}")
    return (
        dict(prepared.get("query_prior") or {}),
        dict(prepared.get("query_tool_summary") or {}),
        dict(prepared.get("reused_none_final") or {}),
        int(prepared.get("reused_single_source_index", query_index)),
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
    results = client.invoke_many(calls)
    if len(results) != len(calls):
        raise ValueError(f"tool prefetch result mismatch: {len(results)} != {len(calls)}")
    # A transient service failure gets one cheap retry. Molecule-specific tool
    # failures remain visible as unavailable evidence instead of discarding the
    # entire query; the full error receipt is retained outside the model prompt.
    for position, result in enumerate(results):
        if result.get("status") != "ok":
            tool_name, arguments = calls[position]
            results[position] = client.invoke(tool_name, arguments)
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
    index: Mapping[str, Any],
    output_root: Path,
    single_root: Path,
    query_prior_source_root: Path | None,
    tool_service_url: str,
    timeout_s: int,
    prefetch_tools: bool,
    neighbor_identity_policy: str,
    initial_card_limit: int,
    delta_card_limit: int,
) -> PreparedQuery:
    query_dir = _query_dir(output_root, task, query_index)
    complete_path = query_dir / "prepared_manifest.json"
    query_smiles = str(record.get("drug") or "")
    query_is_monatomic = monatomic_query_element(query_smiles) is not None
    selection_budget = {
        "initial_card_limit_per_molecule": initial_card_limit,
        "delta_card_limit_per_molecule": delta_card_limit,
    }
    if complete_path.is_file():
        manifest = _read_json(complete_path)
        if (
            manifest.get("status") == "ok"
            and manifest.get("protocol") == PROGRESSIVE_PROTOCOL_VERSION
            and bool(manifest.get("tool_prefetch_complete")) is prefetch_tools
            and manifest.get("neighbor_identity_policy")
            == neighbor_identity_policy
            and manifest.get("selection_budget") == selection_budget
            and (
                not query_is_monatomic
                or manifest.get("structural_eligibility_version")
                == STRUCTURAL_ELIGIBILITY_VERSION
            )
        ):
            return PreparedQuery(task, query_index, query_dir)

    levels = _levels(task)
    if query_prior_source_root is not None:
        query_prior, query_tool_summary, none_final, single_source_index = (
            _load_progressive_query_prior(
                task,
                query_index,
                record,
                query_prior_source_root,
            )
        )
    else:
        query_prior, query_tool_summary, none_final, single_source_index = (
            _load_reused_query_prior(task, record, single_root)
        )
    level_ids = [int(row["level"]) for row in levels]
    retrievals = retrieve_family_molecule_prefixes(
        query_smiles,
        index,
        levels=level_ids,
        min_similarity=0.3,
        neighbor_identity_policy=neighbor_identity_policy,
    )
    cumulative_by_level: dict[int, dict[str, dict[str, Any]]] = {}
    retrieval_audits: dict[int, dict[str, Any]] = {}
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
            active, selection_audit = select_initial_evidence(
                cumulative,
                level=level,
                card_limit=initial_card_limit,
            )
        else:
            new, augmentations, selection_audit = select_progressive_delta(
                previous_cumulative,
                cumulative,
                active,
                level=level,
                card_limit=delta_card_limit,
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
            "selection_budget": selection_budget,
            "active_evidence": snapshot,
            "new_card_ids": sorted(new_ids),
            "n_active_molecules": len(snapshot),
            "n_active_cards": len(current_ids),
            "should_call_model": bool(new_ids),
            "tool_prefetch_complete": prefetch_tools,
            "neighbor_identity_policy": neighbor_identity_policy,
            "structural_eligibility": structural_eligibility_metadata(),
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
            "task": task,
            "query_index": query_index,
            "n_levels": len(levels),
            "tool_prefetch_complete": prefetch_tools,
            "n_tool_prefetch_failures": len(tool_failures),
            "neighbor_identity_policy": neighbor_identity_policy,
            "selection_budget": selection_budget,
            "structural_eligibility_version": STRUCTURAL_ELIGIBILITY_VERSION,
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

    pool = OpenAIProviderPool(provider_config, client_factory=client_factory)
    if getattr(args, "retry_race_width", 1) > 1:
        return ParallelRetryClient(
            pool, parallelism=args.parallelism,
            task_limits={task: args.parallelism_per_task or args.parallelism for task in args.tasks},
        )
    return pool


def _run_query(
    args: argparse.Namespace,
    prepared_query: PreparedQuery,
    client: OpenAIProviderPool,
) -> dict[str, Any]:
    task = prepared_query.task
    contract = _task_contract(task)
    levels = _levels(task)
    prior_state: dict[str, Any] | None = None
    n_calls = 0
    retry_pending = bool(getattr(args, "retry_round", 0))
    error_path = prepared_query.query_dir / "run_error.json"
    if error_path.is_file() and _read_json(error_path).get("status") == "error":
        retry_pending = True
    for level_row in levels:
        independent = getattr(args, "independent_levels", False)
        if independent:
            prior_state = None
        level = int(level_row["level"])
        level_dir = prepared_query.query_dir / "levels" / f"level_{level}"
        output_path = level_dir / "output.json"
        if output_path.is_file():
            existing = _read_json(output_path)
            if existing.get("status") in {"ok", "carried_forward", "reused_none"}:
                prior_state = dict(existing["state"])
                n_calls += int(existing.get("model_called") is True)
                continue
            retry_pending = True
            if not getattr(args, "prepare_only", False):
                write_json_atomic(level_dir / "failed_attempts" / f"{time.time_ns()}.json", existing)
        prepared = _read_json(level_dir / "prepared.json")
        if not prepared.get("tool_prefetch_complete"):
            raise ValueError(
                f"formal inference requires visible tool prefetch: {level_dir}"
            )
        if not (bool(prepared["active_evidence"]) if independent else prepared.get("should_call_model")):
            if getattr(args, "prepare_only", False):
                continue
            if prior_state is None:
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
        messages = build_progressive_messages(
            contract=contract,
            levels=levels,
            current_level=level,
            query_smiles=prepared["query_smiles"],
            condition_sentence=prepared["condition_sentence"],
            query_prior=prepared["query_prior"],
            query_tool_summary=prepared.get("query_tool_summary") or {},
            active=active,
            prior_state=prior_state,
            independent=independent,
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
        if getattr(args, "prepare_only", False):
            continue
        new_aliases = {
            card_id_to_alias[card_id]
            for card_id in map(str, prepared.get("new_card_ids") or [])
        }
        if independent:
            new_aliases = visible_aliases
        execution_provider_attempts: list[dict[str, Any]] = []

        validation_kwargs = dict(
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
        )

        def routed_chat_json(call_messages: list[dict[str, Any]], *, retry=False) -> dict[str, Any]:
            if isinstance(client, ParallelRetryClient):
                routed_response = client.chat_validated(
                    call_messages, task=task,
                    width=args.retry_race_width if retry or retry_pending else 1,
                    validate=lambda result: response_validation_errors(result, **validation_kwargs),
                    receipt_path=level_dir / "retry_races" / f"{time.time_ns()}.json",
                )
            else:
                routed_response = client.chat_json(call_messages)
            execution_provider_attempts.extend(routed_response.get("execution_provider_attempts") or [])
            return routed_response

        response = call_with_json_validation(
            routed_chat_json, messages, **validation_kwargs,
            retry_call=lambda call_messages: routed_chat_json(call_messages, retry=True),
            branch_name=f"{task} {'independent full-flat' if independent else 'progressive'} level {level}",
            max_attempts=4,
        )
        response["execution_provider_attempts"] = execution_provider_attempts
        n_calls += int((response.get("structured_output_validation") or {}).get("attempt_count") or 1)
        if not structured_response_is_valid(response):
            write_json_atomic(
                output_path,
                {
                    "status": "error",
                    "model_called": True,
                    "llm": response,
                    "card_alias_map": alias_to_card_id,
                    "created_at": _now(),
                },
            )
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
        write_json_atomic(
            output_path,
            {
                "status": "ok",
                "model_called": True,
                "state": state,
                "llm": response,
                "card_alias_map": alias_to_card_id,
                "created_at": _now(),
            },
        )
        prior_state = state
        retry_pending = False
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
    *, run_query=None,
) -> dict[str, Any]:
    """Keep one transport/provider failure from terminating unrelated queries."""
    error_path = prepared_query.query_dir / "run_error.json"
    try:
        if error_path.is_file() and not getattr(args, "prepare_only", False):
            previous_error = _read_json(error_path)
            if previous_error.get("status") == "error":
                write_json_atomic(prepared_query.query_dir / "failed_attempts" / f"{time.time_ns()}.json", previous_error)
        result = (run_query or _run_query)(args, prepared_query, client)
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
) -> None:
    contract = _task_contract(task)
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

    for level_row in _levels(task):
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


_PREPARED_MODEL_INPUT_FIELDS = (
    "protocol",
    "task",
    "query_smiles",
    "condition_sentence",
    "query_prior",
    "query_tool_summary",
    "reused_none_final",
    "level",
    "level_definition",
    "active_evidence",
    "new_card_ids",
    "should_call_model",
    "tool_prefetch_complete",
)


def _prepared_model_input(prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Return exactly the prepared fields that can affect a level decision.

    Retrieval and selection audits are deliberately excluded: they are retained
    for provenance but never rendered into the model prompt.  Prior state is
    supplied by the preceding output, so reuse is restricted to an unchanged
    prefix and stops permanently at the first changed level.
    """

    signature = {
        field: prepared.get(field) for field in _PREPARED_MODEL_INPUT_FIELDS
    }
    signature["level_definition"] = _level_prompt_definition(
        prepared.get("level_definition") or {}
    )
    return signature


def _level_prompt_definition(level: Mapping[str, Any]) -> dict[str, Any]:
    """Project a catalog level to the fields rendered in the LLM level plan."""

    return {
        "level": level.get("level"),
        "family": level.get("endpoint_group") or level.get("family_id"),
        "description": level.get("description") or level.get("label") or "",
    }


def _validate_progressive_reuse_source(
    *,
    source_root: Path,
    current_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    source_manifest_path = source_root / "experiment_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    source_manifest = _read_json(source_manifest_path)
    invariant_fields = (
        "experiment",
        "split_scheme",
        "visibility_mode",
        "reference_pool",
        "neighbor_identity_policy",
        "min_similarity",
        "candidate_generation",
        "selection",
        "prompt_profile",
        "condition_policy",
        "max_tokens",
        "temperature",
        "thinking",
        "reasoning_effort",
        "tool_prefetch_complete",
    )
    for field in invariant_fields:
        if source_manifest.get(field) != current_manifest.get(field):
            raise ValueError(
                f"progressive reuse source uses different {field}: "
                f"{source_manifest.get(field)!r}"
            )
    if _model_identity(str(source_manifest.get("model") or "")) != _model_identity(
        str(current_manifest.get("model") or "")
    ):
        raise ValueError("progressive reuse source uses a different model identity")
    source_inputs = source_manifest.get("inputs") or {}
    for task in current_manifest.get("tasks") or []:
        source_task = source_inputs.get(task) or {}
        current_task = (current_manifest.get("inputs") or {}).get(task) or {}
        # A changed family/index lineage is precisely when prefix reuse is useful:
        # every prepared model input is compared below and reuse stops at the
        # first changed level.  The benchmark and frozen single prior must still
        # match exactly.
        for field in ("input_sha256", "single_source_manifest_sha256"):
            if source_task.get(field) != current_task.get(field):
                raise ValueError(
                    f"progressive reuse source has different {task} {field}"
                )
        source_level_plan = [
            _level_prompt_definition(row) for row in source_task.get("levels") or []
        ]
        current_level_plan = [
            _level_prompt_definition(row) for row in current_task.get("levels") or []
        ]
        if source_level_plan != current_level_plan:
            raise ValueError(
                f"progressive reuse source has different {task} model-visible level plan"
            )
    return source_manifest


def _reuse_unchanged_progressive_prefixes(
    *,
    prepared_queries: list[PreparedQuery],
    output_root: Path,
    source_root: Path,
) -> dict[str, Any]:
    if source_root.resolve() == output_root.resolve():
        raise ValueError("progressive reuse source must differ from output root")
    by_task_level: dict[str, dict[str, int]] = {}
    n_reused_levels = 0
    n_reused_model_calls = 0
    n_queries_with_reuse = 0
    for prepared_query in prepared_queries:
        reused_for_query = 0
        for level_row in _levels(prepared_query.task):
            level = int(level_row["level"])
            relative = (
                Path(prepared_query.task)
                / "queries"
                / f"query_idx{prepared_query.index:05d}"
                / "levels"
                / f"level_{level}"
            )
            source_level = source_root / relative
            target_level = output_root / relative
            source_prepared_path = source_level / "prepared.json"
            source_output_path = source_level / "output.json"
            target_prepared_path = target_level / "prepared.json"
            if not (
                source_prepared_path.is_file()
                and source_output_path.is_file()
                and target_prepared_path.is_file()
            ):
                break
            source_prepared = _read_json(source_prepared_path)
            target_prepared = _read_json(target_prepared_path)
            if _prepared_model_input(source_prepared) != _prepared_model_input(
                target_prepared
            ):
                break
            source_output = _read_json(source_output_path)
            if source_output.get("status") not in {
                "ok",
                "carried_forward",
                "reused_none",
            } or not isinstance(source_output.get("state"), Mapping):
                break
            reused_output = dict(source_output)
            reused_output["checkpoint_reuse"] = {
                "source": str(source_output_path),
                "reason": "exact model-visible input match in unchanged prefix",
                "reused_at": _now(),
            }
            write_json_atomic(target_level / "output.json", reused_output)
            source_request_path = source_level / "request.json"
            if source_request_path.is_file():
                write_json_atomic(
                    target_level / "request.json", _read_json(source_request_path)
                )
            level_counts = by_task_level.setdefault(prepared_query.task, {})
            key = f"level_{level}"
            level_counts[key] = level_counts.get(key, 0) + 1
            n_reused_levels += 1
            n_reused_model_calls += int(source_output.get("model_called") is True)
            reused_for_query += 1
        n_queries_with_reuse += int(reused_for_query > 0)
    receipt = {
        "status": "ok",
        "source_root": str(source_root),
        "reuse_contract": "exact model-visible input match; unchanged prefix only",
        "n_queries_with_reuse": n_queries_with_reuse,
        "n_reused_levels": n_reused_levels,
        "n_reused_model_calls": n_reused_model_calls,
        "by_task_level": by_task_level,
        "created_at": _now(),
    }
    write_json_atomic(output_root / "progressive_reuse_receipt.json", receipt)
    return receipt


def _validate_inputs(
    args: argparse.Namespace,
    task_specs: Mapping[str, ProgressiveTaskSpec],
) -> dict[str, list[dict[str, Any]]]:
    records_by_task = {}
    run_identity_policy = _neighbor_identity_policy(args.split_scheme)
    for task in args.tasks:
        spec = task_specs[task]
        for path in (spec.input_jsonl, spec.index, spec.family_manifest):
            if not path.is_file():
                raise FileNotFoundError(path)
        manifest = _read_json(spec.index.with_name("manifest.json"))
        heldout_path = (
            task_root(task, args.split_scheme)
            / "heldout_molecule_condition_labels.jsonl"
        )
        expected = {
            "reference_pool": REFERENCE_POOL,
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
                raise ValueError(f"{task} index has wrong {field}: {manifest.get(field)!r}")
        _heldout_filter_validation(
            task=task,
            split_scheme=args.split_scheme,
            index_manifest=manifest,
            heldout_path=heldout_path,
        )
        index_default_policy = manifest.get("neighbor_identity_policy_default")
        if index_default_policy not in set(IDENTITY_POLICY_BY_SPLIT.values()):
            raise ValueError(
                f"{task} index has unsupported default identity policy: "
                f"{index_default_policy!r}"
            )
        if args.split_scheme == "scaffold" and index_default_policy != run_identity_policy:
            raise ValueError(
                f"{task} scaffold index has wrong default identity policy: "
                f"{index_default_policy!r}"
            )
        source_run = _source_run_dir(task, 0, Path(args.single_source_root))
        if not source_run.parent.is_dir():
            raise FileNotFoundError(source_run.parent)
        single_manifest_path = Path(args.single_source_root) / task / "none" / "manifest.json"
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
        query_prior_source_text = str(args.query_prior_source_root or "").strip()
        if query_prior_source_text:
            query_prior_source = Path(query_prior_source_text)
            prior_manifest_path = query_prior_source / "experiment_manifest.json"
            prior_manifest = _read_json(prior_manifest_path)
            if task not in set(prior_manifest.get("tasks") or []):
                raise ValueError(
                    f"{task} is absent from query-prior source: {prior_manifest_path}"
                )
            if int(prior_manifest.get("n_failed_queries") or 0) != 0:
                raise ValueError(
                    f"{task} query-prior source is not complete: {prior_manifest_path}"
                )
            if _model_identity(str(prior_manifest.get("model") or "")) != _model_identity(
                args.model
            ):
                raise ValueError(
                    f"{task} query-prior source has a different model: {prior_manifest_path}"
                )
            if prior_manifest.get("visibility_mode") != "deployment_visible_prefetched":
                raise ValueError(
                    f"{task} query-prior source has the wrong visibility contract: "
                    f"{prior_manifest_path}"
                )
            if prior_manifest.get("tool_prefetch_complete") is not True:
                raise ValueError(
                    f"{task} query-prior source lacks tool prefetch: {prior_manifest_path}"
                )
            prior_input = (prior_manifest.get("inputs") or {}).get(task) or {}
            if prior_input.get("input_sha256") != sha256_file(spec.input_jsonl):
                raise ValueError(
                    f"{task} query-prior source input hash mismatch: {prior_manifest_path}"
                )
            if prior_input.get("single_source_manifest_sha256") != sha256_file(
                single_manifest_path
            ):
                raise ValueError(
                    f"{task} query-prior source single-cache hash mismatch: "
                    f"{prior_manifest_path}"
                )
            records_by_task[task] = read_jsonl(spec.input_jsonl)
            continue
        source_input = Path(str(single_manifest.get("input_jsonl") or ""))
        if not source_input.is_file():
            raise FileNotFoundError(source_input)
        if len(read_jsonl(source_input)) != int(single_manifest.get("n_items") or -1):
            raise ValueError(f"{task} reusable single source input count mismatch")
        current_keys = {_stable_query_key(row) for row in read_jsonl(spec.input_jsonl)}
        reusable_keys = set(_single_source_index(task, str(Path(args.single_source_root))))
        missing_keys = current_keys - reusable_keys
        if missing_keys:
            raise ValueError(
                f"{task} has {len(missing_keys)} rows without stable-identity single reuse"
            )
        records_by_task[task] = read_jsonl(spec.input_jsonl)
    return records_by_task


def _query_results(args, prepared_queries, client, *, run_query=None):
    """Submit ready queries without occupying workers while a task is capped."""
    pending = deque(sorted(prepared_queries, key=lambda query: query.index))
    if not pending:
        return
    active = dict.fromkeys(args.tasks, 0)
    task_cap = args.parallelism_per_task or args.parallelism
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.parallelism, len(pending))
    ) as pool:
        futures = {}
        while pending or futures:
            for _ in range(len(pending)):
                if len(futures) >= args.parallelism:
                    break
                query = pending.popleft()
                if active[query.task] >= task_cap:
                    pending.append(query)
                    continue
                kwargs = {"run_query": run_query} if run_query is not None else {}
                futures[pool.submit(_run_query_safe, args, query, client, **kwargs)] = query.task
                active[query.task] += 1
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                active[futures.pop(future)] -= 1
                yield future.result()


def _run_query_rounds(args, prepared_queries, client, output_root: Path, *, run_query=None) -> int:
    """Retry failed queries after cooldown; successful level checkpoints are skipped."""
    by_key = {(query.task, query.index): query for query in prepared_queries}
    pending, succeeded, rounds = list(prepared_queries), set(), []
    retries = 0 if args.prepare_only else args.max_stage_requeues
    status = {"mode": "prepare_only" if args.prepare_only else "inference",
              "n_total_queries": len(pending), "max_requeues": retries,
              "retry_delay_s": args.retry_delay_s, "retry_backoff_cap_s": 900, "rounds": rounds,
              "retry_race_width": getattr(args, "retry_race_width", 1)}

    def publish(phase, **details):
        status.update(phase=phase, updated_at=_now(), n_succeeded_queries=len(succeeded), **details)
        write_json_atomic(output_root / "execution_status.json", status)

    for attempt in range(retries + 1):
        args.retry_round = attempt
        failed = []
        publish("running", round=attempt + 1, round_size=len(pending), round_completed=0,
                failed_queries=[], next_retry_at=None)
        kwargs = {"run_query": run_query} if run_query is not None else {}
        for completed, result in enumerate(_query_results(args, pending, client, **kwargs), 1):
            key = (result["task"], result["index"])
            if result["status"] == "ok":
                succeeded.add(key)
            else:
                failed.append(by_key[key])
            publish("running", round_completed=completed,
                    failed_queries=[{"task": query.task, "index": query.index} for query in failed])
        rounds.append({"round": attempt + 1, "n_queries": len(pending),
                       "n_failed": len(failed), "finished_at": _now()})
        pending = failed
        if not pending or attempt == retries:
            break
        delay = min(args.retry_delay_s * 2 ** min(attempt, 20), 900)
        publish("retry_wait", next_retry_at=datetime.fromtimestamp(time.time() + delay, timezone.utc).isoformat())
        print(f"[retry] {len(pending)} failed queries; retry in {delay}s", flush=True)
        time.sleep(delay)
    publish("needs_attention" if pending else ("prepared" if args.prepare_only else "complete"))
    return len(pending)


def run(args: argparse.Namespace) -> int:
    task_specs = _progressive_task_specs(args.split_scheme, args.evaluation_subset)
    provider_config = _resolve_provider_pool_config(args)
    if sum(spec.max_inflight for spec in provider_config.providers) > args.endpoint_concurrency_budget:
        raise ValueError("provider pool capacity exceeds the explicit endpoint budget")
    records_by_task = _validate_inputs(args, task_specs)
    run_identity_policy = _neighbor_identity_policy(args.split_scheme)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    indices_by_task = {
        task: _selected_indices(args, len(records_by_task[task]))
        for task in args.tasks
    }
    inputs = {
        task: {
            "input_jsonl": str(task_specs[task].input_jsonl),
            "input_sha256": sha256_file(task_specs[task].input_jsonl),
            "index": str(task_specs[task].index),
            "index_sha256": sha256_file(task_specs[task].index),
            "family_manifest": str(task_specs[task].family_manifest),
            "family_manifest_sha256": sha256_file(
                task_specs[task].family_manifest
            ),
            "single_source_manifest_sha256": sha256_file(
                Path(args.single_source_root) / task / "none" / "manifest.json"
            ),
            "query_prior_source_manifest_sha256": (
                sha256_file(Path(args.query_prior_source_root) / "experiment_manifest.json")
                if args.query_prior_source_root
                else ""
            ),
            "index_neighbor_identity_policy_default": _read_json(
                task_specs[task].index.with_name("manifest.json")
            ).get("neighbor_identity_policy_default"),
            "heldout_filter_validation": _heldout_filter_validation(
                task=task,
                split_scheme=args.split_scheme,
                index_manifest=_read_json(
                    task_specs[task].index.with_name("manifest.json")
                ),
                heldout_path=(
                    task_root(task, args.split_scheme)
                    / "heldout_molecule_condition_labels.jsonl"
                ),
            ),
            "levels": _levels(task),
        }
        for task in args.tasks
    }
    manifest = {
        "experiment": PROGRESSIVE_PROTOCOL_VERSION,
        "split_scheme": args.split_scheme,
        "evaluation_subset": args.evaluation_subset,
        "tasks": args.tasks,
        "visibility_mode": "deployment_visible_prefetched",
        "reference_pool": REFERENCE_POOL,
        "neighbor_identity_policy": run_identity_policy,
        "min_similarity": 0.3,
        "candidate_generation": {
            "unit": "molecule",
            "scope": "cumulative record-family pool",
            "ranking": "global Morgan similarity",
            "per_assay_neighbor_cap": None,
            "assay_role": "card provenance and diversity only",
        },
        "selection": {
            "level_1": {
                "molecule_limit": INITIAL_MOLECULE_LIMIT,
                "card_limit_per_molecule": args.initial_card_limit,
            },
            "later_new_pool": {
                "molecule_limit": NEW_MOLECULE_LIMIT,
                "delta_card_limit_per_molecule": args.delta_card_limit,
            },
            "later_augmentation_pool": {
                "molecule_limit": AUGMENTED_MOLECULE_LIMIT,
                "delta_card_limit_per_molecule": args.delta_card_limit,
            },
            "slot_borrowing": False,
            "append_only": True,
            "cumulative_per_molecule_card_cap": None,
        },
        "prompt_profile": "progressive_compact_tools_short_aliases.v2",
        "condition_policy": "natural-language sentence for non-null group; omit null group",
        "single_reuse_root": str(Path(args.single_source_root)),
        "query_prior_source_root": (
            str(Path(args.query_prior_source_root))
            if args.query_prior_source_root
            else ""
        ),
        "progressive_reuse_source_root": str(
            Path(args.progressive_reuse_source_root)
        ) if args.progressive_reuse_source_root else "",
        "model": args.model,
        "model_identity": _model_identity(args.model),
        "base_url": args.base_url,
        "parallelism": args.parallelism,
        "parallelism_per_task": args.parallelism_per_task,
        "endpoint_concurrency_budget": args.endpoint_concurrency_budget,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "thinking": "provider_default",
        "reasoning_effort": "omitted",
        "transport_max_retries": args.transport_max_retries,
        "retry_race_width": args.retry_race_width,
        "tool_prefetch_complete": not args.skip_tool_prefetch,
        "evaluation_indices_by_task": indices_by_task,
        "inputs": inputs,
        "started_at": _now(),
        "provider_routing": provider_config.public_dict(),
    }
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
        spec = task_specs[task]
        with spec.index.open("rb") as handle:
            index = pickle.load(handle)
        index = build_family_molecule_prefix_view(
            index,
            levels=[int(row["level"]) for row in _levels(task)],
        )
        records = records_by_task[task]
        indices = indices_by_task[task]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.preparation_workers, len(indices))
        ) as pool:
            futures = {
                pool.submit(
                    _prepare_query,
                    task=task,
                    query_index=query_index,
                    record=records[query_index],
                    index=index,
                    output_root=output_root,
                    single_root=Path(args.single_source_root),
                    query_prior_source_root=(
                        Path(args.query_prior_source_root)
                        if args.query_prior_source_root
                        else None
                    ),
                    tool_service_url=args.tool_service_url,
                    timeout_s=args.timeout_s,
                    prefetch_tools=not args.skip_tool_prefetch,
                    neighbor_identity_policy=run_identity_policy,
                    initial_card_limit=args.initial_card_limit,
                    delta_card_limit=args.delta_card_limit,
                ): query_index
                for query_index in indices
            }
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                prepared_queries.append(future.result())
                if completed % 10 == 0 or completed == len(futures):
                    print(f"[{task}] prepared {completed}/{len(futures)}", flush=True)

    if args.progressive_reuse_source_root:
        source_root = Path(args.progressive_reuse_source_root)
        source_manifest = _validate_progressive_reuse_source(
            source_root=source_root,
            current_manifest=manifest,
        )
        receipt = _reuse_unchanged_progressive_prefixes(
            prepared_queries=prepared_queries,
            output_root=output_root,
            source_root=source_root,
        )
        receipt["family_manifest_lineage"] = {
            task: {
                "source_sha256": (
                    (source_manifest.get("inputs") or {}).get(task) or {}
                ).get("family_manifest_sha256"),
                "current_sha256": ((manifest.get("inputs") or {}).get(task) or {}).get(
                    "family_manifest_sha256"
                ),
            }
            for task in manifest.get("tasks") or []
        }
        write_json_atomic(output_root / "progressive_reuse_receipt.json", receipt)
        manifest["progressive_reuse"] = receipt
        print(
            "[reuse] "
            f"queries={receipt['n_queries_with_reuse']} "
            f"levels={receipt['n_reused_levels']} "
            f"model_calls_avoided={receipt['n_reused_model_calls']}",
            flush=True,
        )

    manifest["prepared_at"] = _now()
    write_json_atomic(manifest_path, manifest)
    if args.prepare_only:
        return 0

    client = _make_client(args, provider_config)
    try:
        failed = _run_query_rounds(args, prepared_queries, client, output_root)
    finally:
        if isinstance(client, ParallelRetryClient):
            client.close()

    for task in args.tasks:
        records = records_by_task[task]
        _summarize_task(
            task=task,
            records=records,
            indices=indices_by_task[task],
            output_root=output_root,
        )
    manifest["finished_at"] = _now()
    manifest["n_failed_queries"] = failed
    manifest["provider_pool_final_snapshot"] = client.snapshot()
    write_json_atomic(manifest_path, manifest)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--split-scheme",
        choices=SPLIT_SCHEMES,
        default="scaffold",
        help=(
            "Conditioned benchmark split scheme; random uses its own "
            "valid+test heldout-filtered indices."
        ),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--evaluation-subset", choices=("valid", "test"), default="valid")
    parser.add_argument("--single-source-root", default=str(ARCHIVED_SINGLE_CACHE_ROOT))
    parser.add_argument(
        "--query-prior-source-root",
        default="",
        help=(
            "Optional completed progressive root whose hash-matched prepared rows "
            "supply the frozen none/single prior when the archived single input is unavailable."
        ),
    )
    parser.add_argument(
        "--progressive-reuse-source-root",
        default="",
        help=(
            "Optional completed progressive run. Reuse only the exact model-visible "
            "unchanged prefix for each query; rerun from the first changed level."
        ),
    )
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
    parser.add_argument("--max-stage-requeues", type=int, default=3,
                        help="Additional failed-query rounds; successful levels are skipped.")
    parser.add_argument("--retry-delay-s", type=int, default=60,
                        help="Initial failed-query cooldown; doubles up to 900 seconds.")
    parser.add_argument("--parallelism-per-task", type=int, default=0,
                        help="Per-task query cap within the shared pool; 0 uses the global cap.")
    parser.add_argument("--endpoint-concurrency-budget", type=int, default=512,
                        help="Explicit endpoint-wide cap; increase only for an authorized run.")
    parser.add_argument("--preparation-workers", type=int, default=32)
    parser.add_argument("--initial-card-limit", type=int, default=4)
    parser.add_argument("--delta-card-limit", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=20_480)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--retry-race-width", type=int, default=1,
                        help="Concurrent first-valid attempts for each failed level or JSON repair.")
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
    load_env_file(args.env_file)
    if args.endpoint_concurrency_budget < 1 or not 1 <= args.parallelism <= args.endpoint_concurrency_budget:
        parser.error("--parallelism must be between 1 and --endpoint-concurrency-budget")
    if not 0 <= args.parallelism_per_task <= args.parallelism:
        parser.error("--parallelism-per-task must be between 0 and --parallelism")
    if args.preparation_workers < 1:
        parser.error("--preparation-workers must be positive")
    if args.initial_card_limit < 1:
        parser.error("--initial-card-limit must be positive")
    if args.delta_card_limit < 1:
        parser.error("--delta-card-limit must be positive")
    if args.transport_max_retries < 0:
        parser.error("--transport-max-retries must be non-negative")
    if not 1 <= args.retry_race_width <= (args.parallelism_per_task or args.parallelism):
        parser.error("retry race width must fit the task request budget")
    if args.retry_race_width > 1 and args.transport_max_retries:
        parser.error("parallel racing requires --transport-max-retries 0")
    if args.max_stage_requeues < 0 or args.retry_delay_s < 0:
        parser.error("retry count and delay must be non-negative")
    if (
        (args.split_scheme == "random" or args.evaluation_subset == "test")
        and Path(args.output_root) == DEFAULT_OUTPUT_ROOT
    ):
        parser.error("random/test evaluation requires an explicit --output-root")
    if args.skip_tool_prefetch and not args.prepare_only:
        parser.error("--skip-tool-prefetch is allowed only with --prepare-only")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
