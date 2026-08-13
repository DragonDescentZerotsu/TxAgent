"""Run BBB Martins analog reasoning with group-level parallel LLM calls."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.evidence_contract import evidence_for_llm
from tools.chembl_tool.common.coverage_reasoning import (
    NEIGHBOR_CONTEXT_PROFILES,
    STANDARD_NEIGHBOR_CONTEXT,
)
from tools.chembl_tool.common.experiment_retrieval import EXPERIMENT_MODES, retrieve_experiment_view
from tools.chembl_tool.common.export import ensure_dir
from tools.chembl_tool.common.final_evidence_surface import (
    SUMMARY_ONLY,
    add_final_evidence_surface_argument,
    build_final_evidence_fields,
    compact_group_reasoning_outputs,
    final_evidence_instructions,
    prepare_resumed_final_inputs,
)
from tools.chembl_tool.common.final_decision_prior import (
    STANDARD_FINAL_DECISION,
    TrainRatioPrior,
    add_final_decision_profile_argument,
)
from tools.chembl_tool.common.identity_blind import (
    prepare_reasoning_retrieval,
    sanitize_identity_blind_branch_outputs,
)
from tools.chembl_tool.common.json_utils import parse_json_content
from tools.chembl_tool.common.neighbor_selection import (
    NEIGHBOR_SELECTORS,
    SIMILARITY_SELECTOR,
)
from tools.chembl_tool.common.retrieval_policy import NEIGHBOR_IDENTITY_POLICIES
from tools.chembl_tool.common.openai_reasoning_client import OpenAICompatibleClient
from tools.chembl_tool.common.prompt_profile import (
    prompt_profile_from_manifest,
    require_matching_prompt_profiles,
)
from tools.chembl_tool.common.reasoning_payload import (
    clean_exact_match as _clean_exact_match,
    clean_shared_assay_context as _clean_shared_assay_context,
    llm_query_payload as _llm_query_payload,
    load_env_file as _load_env,
    read_jsonl_record as _read_jsonl_record,
    write_trace_jsonl,
)
from tools.chembl_tool.common.reasoning_calls import (
    bound_group_prompt_payload,
    call_group_branch,
    call_single_molecule_branch,
    load_frozen_single_analysis,
)
from tools.chembl_tool.common.reasoning_validation import (
    call_with_json_validation,
    structured_response_is_valid,
    validated_branch_content,
)
from tools.chembl_tool.common.retrieval_replay import load_retrieval_replay
from tools.chembl_tool.common.retrieval_ablation import load_reusable_group_outputs
from tools.chembl_tool.common.task_workflows.evidence_library import standardize_smiles_and_fp
from tools.chembl_tool.tasks.bbb_martins.chembl_exact_context import (
    DEFAULT_CHEMBL_SQLITE,
    enrich_retrieval_with_chembl_context,
)
from tools.chembl_tool.tasks.bbb_martins.experiment_config import get_source_config
from tools.chembl_tool.tasks.bbb_martins.prompt_profiles import (
    BBB_PROMPT_PROFILES,
    DEFAULT_BBB_PROMPT_PROFILE,
    HISTORICAL_BBB_PROMPT_PROFILE,
    SINGLE_SCHEMA,
    final_profile_validation_errors,
    get_bbb_prompt_profile,
)
from tools.chembl_tool.tasks.bbb_martins.retrieve_neighbors import load_index, retrieve_neighbors
from tools.chembl_tool.tasks.bbb_martins.final_decision_profiles import (
    BBB_FINAL_DECISION_PROFILES,
    bbb_final_decision_allowed_values,
    bbb_final_decision_validation_errors,
    build_bbb_final_decision_prompt,
)


DEFAULT_INPUT = "data/processed/BBB_Martins/B3DB_cleaned/test/test_efflux.jsonl"
DEFAULT_INDEX = "outputs/chembl_tool/tasks/bbb_martins/evidence_library/bbb_neighbor_index.pkl"
DEFAULT_TIER1_REPLACEMENT_GROUPS = ["Tier 1.starling_direct_bbb_evidence"]
DEFAULT_OUT_ROOT = "outputs/chembl_tool/tasks/bbb_martins/reasoning/single_runs"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TOOL_SERVICE_URL = "http://127.0.0.1:8765"
_write_trace_jsonl = partial(write_trace_jsonl, prediction_field="bbb_prediction")
TRAIN_RATIO_PRIOR = TrainRatioPrior(
    dataset_lineage="experimental_meaningful_cns_access_v2",
    split="scaffold/train",
    positive_count=2162,
    negative_count=773,
    positive_label="pass",
    negative_label="fail",
)


GROUP_REASONING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mmp_structure_compare",
            "description": (
                "Compare the query molecule to one neighbor using Morgan Tanimoto, MCS coverage, "
                "and mmpdb matched-pair transformation. Use this to judge structural transferability."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_smiles": {"type": "string", "description": "The query molecule SMILES."},
                    "reference_smiles": {"type": "string", "description": "The neighbor/reference molecule SMILES."},
                    "max_mmp_alternatives": {
                        "type": "integer",
                        "description": "Maximum matched-pair alternatives to return.",
                        "default": 5,
                    },
                    "mcs_timeout_s": {
                        "type": "integer",
                        "description": "MCS search timeout in seconds.",
                        "default": 5,
                    },
                },
                "required": ["query_smiles", "reference_smiles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "properties_compare",
            "description": (
                "Compare query and neighbor molecule properties, including RDKit descriptors and MolGpKa/logD "
                "features. Use this to assess whether property changes affect BBB evidence transferability."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_smiles": {"type": "string", "description": "The query molecule SMILES."},
                    "reference_smiles": {"type": "string", "description": "The neighbor/reference molecule SMILES."},
                    "logd_ph": {
                        "type": "number",
                        "description": "pH for logD comparison.",
                        "default": 7.4,
                    },
                },
                "required": ["query_smiles", "reference_smiles"],
            },
        },
    },
]

SINGLE_MOLECULE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "molecule_properties",
            "description": (
                "Compute the query molecule's RDKit descriptors, MolGpKa pKa/logD features, "
                "and functional-group profile. This is the only tool allowed for single-molecule analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_smiles": {"type": "string", "description": "The query molecule SMILES."},
                    "logd_ph": {
                        "type": "number",
                        "description": "pH for logD estimation.",
                        "default": 7.4,
                    },
                },
                "required": ["query_smiles"],
            },
        },
    }
]
SINGLE_MOLECULE_TOOL_CHOICE = {
    "type": "function",
    "function": {"name": "molecule_properties"},
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_prompt_profile_reuse(args)
    _load_env(Path(args.env_file))
    api_key = os.getenv(args.api_key_env)
    if not api_key and not args.prepare_only:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    if args.resume_final_from_run_dir:
        client = OpenAICompatibleClient(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            timeout_s=args.timeout_s,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            tool_service_url=args.tool_service_url,
            enable_group_tools=not args.disable_group_tools,
            max_tool_rounds=args.max_tool_rounds,
            reasoning_effort=args.reasoning_effort,
            enable_thinking=args.enable_thinking,
        )
        return _resume_final_from_run_dir(Path(args.resume_final_from_run_dir), client)

    run_id = args.run_id or time.strftime("bbb_reasoning_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    out_dir = ensure_dir(Path(args.out_root) / run_id)
    _log(f"run_id={run_id}")

    query_record = _read_jsonl_record(Path(args.input_jsonl), args.query_index)
    query_smiles = str(query_record.get(args.smiles_field) or "")
    if not query_smiles:
        raise SystemExit(f"Input record has no `{args.smiles_field}` value.")

    retrieval = load_retrieval_replay(args.retrieval_replay_run_dir, query_smiles)
    index = None
    if retrieval is None and args.experiment_mode != "none":
        _log("loading neighbor index")
        index = load_index(Path(args.index))
    if retrieval is not None:
        _log(f"replaying frozen retrieval from {args.retrieval_replay_run_dir}")
    elif args.experiment_mode == "native":
        base_groups = _base_retrieval_groups(
            index,
            requested_groups=args.groups,
            tier1_replacement_enabled=bool(args.tier1_replacement_index),
        )
        if args.tier1_replacement_index and base_groups == []:
            retrieval = _empty_retrieval(query_smiles, index, args.top_k_per_group, args.min_similarity)
        else:
            retrieval = retrieve_neighbors(
                query_smiles,
                index,
                top_k_per_group=args.top_k_per_group,
                min_similarity=args.min_similarity,
                groups=base_groups,
                neighbor_selector=args.neighbor_selector,
            )
    else:
        _log(f"building retrieval view mode={args.experiment_mode} source={args.retrieval_source}")
        retrieval = retrieve_experiment_view(
            query_smiles,
            index,
            mode=args.experiment_mode,
            config=get_source_config(args.retrieval_source) if args.experiment_mode != "none" else None,
            top_k_per_group=args.top_k_per_group,
            min_similarity=args.min_similarity,
            native_groups=args.groups,
            neighbor_identity_policy=args.neighbor_identity_policy,
            neighbor_selector=args.neighbor_selector,
        )
    if retrieval.get("status") != "ok":
        raise SystemExit(json.dumps(retrieval.get("errors", []), ensure_ascii=False))
    if retrieval is None:
        raise RuntimeError("Retrieval was not built or replayed.")
    if args.enable_chembl_exact_context and index is not None:
        _log("enriching retrieval with exact ChEMBL context")
        retrieval = enrich_retrieval_with_chembl_context(
            retrieval,
            index,
            chembl_sqlite=args.chembl_sqlite,
        )
    if (
        not args.retrieval_replay_run_dir
        and args.tier1_replacement_index
        and args.experiment_mode == "native"
    ):
        _log(f"retrieving Tier 1 replacement neighbors: {args.tier1_replacement_index}")
        replacement_index = load_index(Path(args.tier1_replacement_index))
        replacement_retrieval = retrieve_neighbors(
            query_smiles,
            replacement_index,
            top_k_per_group=args.top_k_per_group,
            min_similarity=args.min_similarity,
            groups=_replacement_retrieval_groups(args.groups, args.tier1_replacement_groups),
            neighbor_selector=args.neighbor_selector,
        )
        if replacement_retrieval.get("status") != "ok":
            raise SystemExit(json.dumps(replacement_retrieval.get("errors", []), ensure_ascii=False))
        retrieval = _merge_tier1_replacement_retrieval(retrieval, replacement_retrieval)

    retrieval_path = out_dir / "retrieval.json"
    _write_json(retrieval_path, retrieval)
    _log(f"wrote {retrieval_path}")

    groups = [group for group in retrieval["groups"] if group.get("neighbors")]
    if args.max_groups:
        groups = groups[: args.max_groups]
    _log(f"group reasoning calls={len(groups)}")
    if args.prepare_only:
        _log("prepare-only complete; reasoning stages deferred to the global pool")
        return 0

    client = OpenAICompatibleClient(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tool_service_url=args.tool_service_url,
        enable_group_tools=not args.disable_group_tools,
        max_tool_rounds=args.max_tool_rounds,
        reasoning_effort=args.reasoning_effort,
        enable_thinking=args.enable_thinking,
    )
    reasoning_retrieval = prepare_reasoning_retrieval(
        retrieval,
        client.tool_service,
        identity_blind=args.identity_blind,
        harness_prefetch_tools=args.harness_prefetch_tools,
        prefetched_tool_replay_run_dir=args.prefetched_tool_replay_run_dir,
        neighbor_context_profile=args.neighbor_context_profile,
    )
    reasoning_groups = [group for group in reasoning_retrieval["groups"] if group.get("neighbors")]
    if args.max_groups:
        reasoning_groups = reasoning_groups[: args.max_groups]
    frozen_single = load_frozen_single_analysis(args.single_analysis_source_run_dir)
    frozen_groups = load_reusable_group_outputs(
        args.group_analysis_source_run_dir,
        retrieval,
        target_neighbor_context_profile=args.neighbor_context_profile,
    )

    single_output, group_outputs = _run_parallel_reasoning(
        client,
        reasoning_retrieval,
        reasoning_groups,
        max_workers=args.max_workers,
        single_output=frozen_single,
        group_outputs=frozen_groups,
        prompt_profile=args.bbb_prompt_profile,
    )
    single_path = out_dir / "single_molecule_reasoning_output.json"
    _write_json(single_path, single_output)
    _log(f"wrote {single_path}")

    raw_group_path = out_dir / "group_reasoning_outputs_raw.jsonl"
    if args.identity_blind:
        _write_jsonl(raw_group_path, group_outputs)
        group_outputs = sanitize_identity_blind_branch_outputs(group_outputs, retrieval)
        _log(f"wrote {raw_group_path}")
    group_path = out_dir / "group_reasoning_outputs.jsonl"
    _write_jsonl(group_path, group_outputs)
    _log(f"wrote {group_path}")

    final_output = _run_final_reasoning(
        client,
        reasoning_retrieval,
        single_output,
        group_outputs,
        final_evidence_surface=args.final_evidence_surface,
        final_decision_profile=args.final_decision_profile,
        prompt_profile=args.bbb_prompt_profile,
    )
    final_path = out_dir / "final_reasoning_output.json"
    _write_json(final_path, final_output)
    _log(f"wrote {final_path}")

    trace_path = out_dir / "trace_messages.jsonl"
    _write_trace_jsonl(
        trace_path,
        query_record=query_record,
        query_index=args.query_index,
        smiles="[identity_blind]" if args.identity_blind else query_smiles,
        single_output=single_output,
        group_outputs=group_outputs,
        final_output=final_output,
    )
    _log(f"wrote {trace_path}")

    manifest = {
        "run_id": run_id,
        "input_jsonl": args.input_jsonl,
        "query_index": args.query_index,
        "smiles_field": args.smiles_field,
        "experiment_mode": args.experiment_mode,
        "retrieval_source": args.retrieval_source,
        "neighbor_identity_policy": args.neighbor_identity_policy,
        "neighbor_selector": args.neighbor_selector,
        "neighbor_context_profile": args.neighbor_context_profile,
        "final_evidence_surface": args.final_evidence_surface,
        "final_decision_profile": args.final_decision_profile,
        "task_prompt_profile": args.bbb_prompt_profile,
        "label_scope": get_bbb_prompt_profile(args.bbb_prompt_profile).label_scope,
        "retrieval_replay_source_run_dir": args.retrieval_replay_run_dir,
        "prefetched_tool_replay_source_run_dir": args.prefetched_tool_replay_run_dir,
        "identity_blind": args.identity_blind,
        "harness_prefetch_tools": args.identity_blind or args.harness_prefetch_tools,
        "model": args.model,
        "base_url": args.base_url,
        "tool_service_url": args.tool_service_url,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "thinking": {"type": "enabled"} if args.enable_thinking else {"type": "disabled"},
        "neighbor_index": args.index if args.experiment_mode != "none" else "",
        "tier1_replacement_index": args.tier1_replacement_index,
        "tier1_replacement_groups": args.tier1_replacement_groups or DEFAULT_TIER1_REPLACEMENT_GROUPS
        if args.tier1_replacement_index
        else [],
        "groups": args.groups or [],
        "retrieval_evidence_source": retrieval.get("evidence_source", {}),
        "group_tools_enabled": not args.disable_group_tools,
        "tool_execution_mode": (
            "harness_prefetch" if args.identity_blind or args.harness_prefetch_tools else "llm_function_call"
        ),
        "single_analysis_source_run_dir": args.single_analysis_source_run_dir,
        "group_analysis_source_run_dir": args.group_analysis_source_run_dir,
        "chembl_exact_context_enabled": args.enable_chembl_exact_context,
        "chembl_sqlite": args.chembl_sqlite,
        "group_tool_names": [tool["function"]["name"] for tool in GROUP_REASONING_TOOLS]
        if not args.disable_group_tools
        else [],
        "max_tool_rounds": args.max_tool_rounds,
        "top_k_per_group": args.top_k_per_group,
        "min_similarity": args.min_similarity,
        "n_groups_with_neighbors": len(groups),
        "paths": {
            "retrieval": str(retrieval_path),
            "single_molecule_reasoning_output": str(single_path),
            "group_reasoning_outputs": str(group_path),
            "group_reasoning_outputs_raw": str(raw_group_path) if args.identity_blind else "",
            "final_reasoning_output": str(final_path),
            "trace_messages": str(trace_path),
        },
        "query_label_for_eval_only": query_record.get("Y"),
    }
    _write_json(out_dir / "manifest.json", manifest)
    _print_summary(final_output, manifest)
    return 0


def _run_parallel_reasoning(
    client: OpenAICompatibleClient,
    retrieval: dict[str, Any],
    groups: list[dict[str, Any]],
    *,
    max_workers: int,
    single_output: dict[str, Any] | None = None,
    group_outputs: list[dict[str, Any]] | None = None,
    prompt_profile: str = DEFAULT_BBB_PROMPT_PROFILE,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    outputs = list(group_outputs or [])
    reused_group_ids = {str(output.get("group_id") or "") for output in outputs}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _reason_one_group,
                client,
                _llm_query_payload(retrieval["query"]),
                group,
                prompt_profile=prompt_profile,
            ): group["group_id"]
            for group in groups
            if str(group.get("group_id") or "") not in reused_group_ids
        }
        if single_output is None:
            futures[
                executor.submit(
                    _reason_single_molecule,
                    client,
                    _llm_query_payload(retrieval["query"]),
                    _clean_query_chembl_context(retrieval.get("query_chembl_context") or {}),
                    prompt_profile=prompt_profile,
                )
            ] = "single_molecule"
        else:
            _log("reusing frozen single molecule analysis")
        for future in concurrent.futures.as_completed(futures):
            item_id = futures[future]
            try:
                result = future.result()
                if item_id == "single_molecule":
                    single_output = result
                    _log("single molecule analysis done")
                else:
                    outputs.append(result)
                    _log(f"group done: {item_id}")
            except Exception as exc:
                if item_id == "single_molecule":
                    single_output = {"analysis_id": "single_molecule", "status": "error", "error": str(exc)}
                    _log(f"single molecule analysis error: {exc}")
                else:
                    outputs.append({"group_id": item_id, "status": "error", "error": str(exc)})
                    _log(f"group error: {item_id}: {exc}")
    return single_output or {"analysis_id": "single_molecule", "status": "error", "error": "missing result"}, sorted(
        outputs,
        key=lambda item: str(item.get("group_id", "")),
    )


def _base_retrieval_groups(
    index: dict[str, Any],
    *,
    requested_groups: list[str] | None = None,
    tier1_replacement_enabled: bool,
) -> list[str] | None:
    if requested_groups:
        base_requested = [group_id for group_id in requested_groups if not str(group_id).startswith("Tier 1.")]
        return base_requested
    if not tier1_replacement_enabled:
        return None
    return [
        group_id
        for group_id in sorted((index.get("group_to_molecule_indices") or {}).keys())
        if not str(group_id).startswith("Tier 1.")
    ]


def _replacement_retrieval_groups(
    requested_groups: list[str] | None,
    replacement_groups: list[str] | None,
) -> list[str]:
    if replacement_groups:
        return replacement_groups
    if requested_groups:
        requested_tier1 = [group_id for group_id in requested_groups if str(group_id).startswith("Tier 1.")]
        return requested_tier1 or DEFAULT_TIER1_REPLACEMENT_GROUPS
    return DEFAULT_TIER1_REPLACEMENT_GROUPS


def _empty_retrieval(
    query_smiles: str,
    index: dict[str, Any],
    top_k_per_group: int,
    min_similarity: float,
) -> dict[str, Any]:
    canonical_smiles, inchi_key, _ = standardize_smiles_and_fp(query_smiles)
    return {
        "status": "ok",
        "evidence_source": index.get("source", {}),
        "query": {
            "input_smiles": query_smiles,
            "canonical_smiles": canonical_smiles,
            "standard_inchi_key": inchi_key,
            "fingerprint": index.get("fingerprint", {}),
        },
        "groups": [],
        "coverage": {
            "n_groups": 0,
            "n_groups_with_neighbors": 0,
            "n_neighbors_total": 0,
            "min_similarity": min_similarity,
            "top_k_per_group": top_k_per_group,
        },
    }


def _merge_tier1_replacement_retrieval(
    base_retrieval: dict[str, Any],
    replacement_retrieval: dict[str, Any],
) -> dict[str, Any]:
    replacement_groups = [
        group for group in replacement_retrieval.get("groups") or [] if str(group.get("group_id") or "").startswith("Tier 1.")
    ]
    base_groups = [
        group for group in base_retrieval.get("groups") or [] if not str(group.get("group_id") or "").startswith("Tier 1.")
    ]
    groups = [*replacement_groups, *base_groups]
    result = dict(base_retrieval)
    result["groups"] = groups
    result["evidence_source"] = {
        "type": "tier1_replaced_source_retrievals",
        "base_source": base_retrieval.get("evidence_source") or {},
        "tier1_replacement_source": replacement_retrieval.get("evidence_source") or {},
        "replacement_group_ids": [group.get("group_id") for group in replacement_groups],
    }
    result["coverage"] = _merged_coverage(
        groups,
        base_retrieval.get("coverage") or {},
        replacement_retrieval.get("coverage") or {},
    )
    result["source_retrieval_coverage"] = [
        {
            "role": "tier1_replacement",
            "evidence_source": replacement_retrieval.get("evidence_source") or {},
            "coverage": replacement_retrieval.get("coverage") or {},
            "group_ids": [group.get("group_id") for group in replacement_retrieval.get("groups") or []],
        },
        {
            "role": "base_without_tier1",
            "evidence_source": base_retrieval.get("evidence_source") or {},
            "coverage": base_retrieval.get("coverage") or {},
            "group_ids": [group.get("group_id") for group in base_retrieval.get("groups") or []],
        },
    ]
    return result


def _merged_coverage(
    groups: list[dict[str, Any]],
    base_coverage: dict[str, Any],
    replacement_coverage: dict[str, Any],
) -> dict[str, Any]:
    return {
        "n_groups": len(groups),
        "n_groups_with_neighbors": sum(1 for group in groups if group.get("neighbors")),
        "n_neighbors_total": sum(len(group.get("neighbors") or []) for group in groups),
        "min_similarity": base_coverage.get("min_similarity", replacement_coverage.get("min_similarity")),
        "top_k_per_group": base_coverage.get("top_k_per_group", replacement_coverage.get("top_k_per_group")),
        "tier1_replacement_enabled": True,
        "base_n_groups_without_tier1": base_coverage.get("n_groups"),
        "replacement_n_groups": replacement_coverage.get("n_groups"),
    }


def _reason_single_molecule(
    client: OpenAICompatibleClient,
    query: dict[str, Any],
    chembl_context: dict[str, Any] | None = None,
    *,
    prompt_profile: str = DEFAULT_BBB_PROMPT_PROFILE,
) -> dict[str, Any]:
    profile = get_bbb_prompt_profile(prompt_profile)
    instructions = [
        "Call molecule_properties for the query molecule before analysis.",
        *profile.single_instructions,
    ]
    if query.get("prefetched_molecule_properties"):
        instructions[0] = "Use the harness-prefetched molecule_properties result."
        if query.get("identity_hidden"):
            instructions[0] += " Do not identify or name the query."
    payload: dict[str, Any] = {
        "task": "Single-molecule BBB plausibility analysis.",
        "query": query,
        "instructions": instructions,
        "required_json_schema": SINGLE_SCHEMA,
    }
    if chembl_context:
        instructions.append(
            "If exact_query_chembl_context is found, distinguish direct same-molecule ChEMBL BBB evidence from the physicochemical prior."
        )
        payload["exact_query_chembl_context"] = chembl_context
    messages = [
        {
            "role": "system",
            "content": (
                "You are a medicinal chemistry BBB single-molecule analyst. "
                "Only analyze the query molecule itself, without analog evidence. "
                + (
                    "The harness already supplied molecule_properties; do not call tools. "
                    + ("Do not infer query identity. " if query.get("identity_hidden") else "")
                    if query.get("prefetched_molecule_properties")
                    else "You may call exactly one tool: molecule_properties. "
                )
                + "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]
    response = call_single_molecule_branch(
        client,
        messages,
        query=query,
        tools=SINGLE_MOLECULE_TOOLS,
        first_tool_choice=SINGLE_MOLECULE_TOOL_CHOICE,
    )
    return {
        "analysis_id": "single_molecule",
        "status": "ok" if structured_response_is_valid(response) else "error",
        "llm": response,
    }


def _reason_one_group(
    client: OpenAICompatibleClient,
    query: dict[str, Any],
    group: dict[str, Any],
    *,
    prompt_profile: str = DEFAULT_BBB_PROMPT_PROFILE,
) -> dict[str, Any]:
    profile = get_bbb_prompt_profile(prompt_profile)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a medicinal chemistry BBB analog evidence analyst. "
                "Reason about whether analog evidence in one endpoint group is transferable to the query molecule. "
                + (
                    "Use the harness-prefetched comparison results; do not call tools. "
                    + ("Do not infer query identity. " if group.get("identity_blind") else "")
                    if group.get("tools_prefetched") or group.get("identity_blind")
                    else "You may call the provided molecule comparison tools when structural or property differences matter. "
                )
                + "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                _group_prompt_payload(query, group, prompt_profile=prompt_profile),
                ensure_ascii=False,
            ),
        },
    ]
    response = call_group_branch(
        client,
        messages,
        group=group,
        tools=GROUP_REASONING_TOOLS,
        required_fields=profile.group_required_fields,
    )
    return {
        "group_id": group["group_id"],
        "status": "ok" if structured_response_is_valid(response) else "error",
        "tier": group["tier"],
        "endpoint_group": group["endpoint_group"],
        "n_neighbors": len(group["neighbors"]),
        "llm": response,
    }


def _run_final_reasoning(
    client: OpenAICompatibleClient,
    retrieval: dict[str, Any],
    single_output: dict[str, Any],
    group_outputs: list[dict[str, Any]],
    *,
    final_evidence_surface: str = SUMMARY_ONLY,
    final_decision_profile: str = STANDARD_FINAL_DECISION,
    prompt_profile: str = DEFAULT_BBB_PROMPT_PROFILE,
) -> dict[str, Any]:
    profile = get_bbb_prompt_profile(prompt_profile)
    evidence_fields, surface_audit = build_final_evidence_fields(
        retrieval,
        compact_group_reasoning_outputs(group_outputs),
        surface=final_evidence_surface,
    )
    decision_prompt = build_bbb_final_decision_prompt(
        final_decision_profile,
        TRAIN_RATIO_PRIOR,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior BBB reasoning model. Integrate group-level analog evidence into one final BBB assessment. "
                "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "Final BBB prediction from analog evidence.",
                        "query": _llm_query_payload(retrieval["query"]),
                    "retrieval_coverage": retrieval["coverage"],
                    "single_molecule_analysis": {
                        "status": single_output.get("status"),
                        "content": validated_branch_content(single_output),
                    },
                    **evidence_fields,
                    **decision_prompt.fields,
                    "instructions": list(profile.final_instructions)
                    + list(decision_prompt.instructions)
                    + final_evidence_instructions(final_evidence_surface),
                    "required_json_schema": {
                        **profile.final_schema,
                        **decision_prompt.schema,
                    },
                },
                ensure_ascii=False,
            ),
        },
    ]
    response = call_with_json_validation(
        client.chat_json,
        messages,
        required_fields=(*profile.final_required_fields, *decision_prompt.required_fields),
        allowed_values={
            **profile.final_allowed_values,
            **bbb_final_decision_allowed_values(final_decision_profile),
        },
        content_validator=lambda content: [
            *final_profile_validation_errors(content, profile=prompt_profile),
            *(
                bbb_final_decision_validation_errors(
                    content,
                    profile=final_decision_profile,
                    prior=TRAIN_RATIO_PRIOR,
                    prediction_field="bbb_prediction",
                )
                if final_decision_profile != STANDARD_FINAL_DECISION
                else []
            ),
        ],
        branch_name="final",
    )
    output = {"status": "ok" if structured_response_is_valid(response) else "error", "llm": response}
    if surface_audit is not None:
        output["final_evidence_surface"] = surface_audit
    return output


def build_group_prompt_payload(
    query: dict[str, Any],
    group: dict[str, Any],
    *,
    prompt_profile: str = DEFAULT_BBB_PROMPT_PROFILE,
) -> dict[str, Any]:
    profile = get_bbb_prompt_profile(prompt_profile)
    return bound_group_prompt_payload({
        "task": "Group-level BBB analog transferability analysis.",
        "query": query,
        "group": {
            "group_id": group["group_id"],
            "tier": group["tier"],
            "endpoint_group": group["endpoint_group"],
        },
        "neighbors": [
            {
                "rank": neighbor["rank"],
                "molecule_chembl_id": neighbor["molecule_chembl_id"],
                "canonical_smiles": neighbor["canonical_smiles"],
                "similarity": neighbor["similarity"],
                "similarity_bucket": neighbor["similarity_bucket"],
                "prefetched_comparisons": neighbor.get("prefetched_comparisons") or [],
                "evidence_rows": [_clean_evidence_row(row) for row in neighbor["evidence_rows"]],
                "shared_assay_context": _clean_shared_assay_context(neighbor.get("shared_assay_context") or {}),
            }
            for neighbor in group["neighbors"]
        ],
        "instructions": list(profile.group_instructions),
        "required_json_schema": profile.group_schema,
    })


# Historical internal callers keep working while external materializers use
# the explicit public adapter above.
_group_prompt_payload = build_group_prompt_payload


def _clean_evidence_row(row: dict[str, Any]) -> dict[str, Any]:
    return evidence_for_llm(row)


def _clean_query_chembl_context(context: dict[str, Any]) -> dict[str, Any]:
    status = context.get("status") or "not_available"
    if status != "found":
        return {key: context.get(key, "") for key in ["status", "reason", "standard_inchi_key"] if context.get(key)}
    return {
        "status": "found",
        "selected_molecule_chembl_id": context.get("selected_molecule_chembl_id", ""),
        "exact_matches": [_clean_exact_match(match) for match in context.get("exact_matches", [])],
        "bbb_relevant_evidence_rows": [
            _clean_evidence_row(row) for row in context.get("bbb_relevant_evidence_rows", [])
        ],
    }


def _parse_json_content(content: str) -> Any:
    return parse_json_content(content)


def _validate_prompt_profile_reuse(args: argparse.Namespace) -> None:
    """Reject branch reuse across BBB prompt contracts."""
    require_matching_prompt_profiles(
        target_profile=str(args.bbb_prompt_profile),
        source_dirs=(
            args.single_analysis_source_run_dir,
            args.group_analysis_source_run_dir,
        ),
        historical_profile=HISTORICAL_BBB_PROMPT_PROFILE,
    )


def _manifest_prompt_profile(manifest: dict[str, Any]) -> str:
    return prompt_profile_from_manifest(
        manifest,
        historical_profile=HISTORICAL_BBB_PROMPT_PROFILE,
    )


def _resume_final_from_run_dir(run_dir: Path, client: OpenAICompatibleClient) -> int:
    retrieval = json.loads((run_dir / "retrieval.json").read_text(encoding="utf-8"))
    single_output = json.loads((run_dir / "single_molecule_reasoning_output.json").read_text(encoding="utf-8"))
    group_outputs = [
        json.loads(line)
        for line in (run_dir / "group_reasoning_outputs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    prompt_profile = _manifest_prompt_profile(manifest)
    retrieval, group_outputs, final_surface = prepare_resumed_final_inputs(
        retrieval,
        single_output,
        group_outputs,
        manifest,
        tool_service=client.tool_service,
    )
    final_output = _run_final_reasoning(
        client,
        retrieval,
        single_output,
        group_outputs,
        final_evidence_surface=final_surface,
        final_decision_profile=str(
            manifest.get("final_decision_profile") or STANDARD_FINAL_DECISION
        ),
        prompt_profile=prompt_profile,
    )
    final_path = run_dir / "final_reasoning_output.json"
    _write_json(final_path, final_output)
    trace_path = run_dir / "trace_messages.jsonl"
    query_record = {"Y": manifest.get("query_label_for_eval_only")}
    query_index = int(manifest.get("query_index") or 0)
    smiles = (
        "[identity_blind]"
        if manifest.get("identity_blind")
        else str((retrieval.get("query") or {}).get("input_smiles") or "")
    )
    _write_trace_jsonl(
        trace_path,
        query_record=query_record,
        query_index=query_index,
        smiles=smiles,
        single_output=single_output,
        group_outputs=group_outputs,
        final_output=final_output,
    )
    manifest["final_rerun_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["task_prompt_profile"] = prompt_profile
    manifest["label_scope"] = get_bbb_prompt_profile(prompt_profile).label_scope
    manifest.setdefault("paths", {})["final_reasoning_output"] = str(final_path)
    manifest.setdefault("paths", {})["trace_messages"] = str(trace_path)
    _write_json(manifest_path, manifest)
    _print_summary(final_output, manifest)
    return 0


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _print_summary(final_output: dict[str, Any], manifest: dict[str, Any]) -> None:
    content = (final_output.get("llm") or {}).get("content") or {}
    print(json.dumps({"manifest": manifest, "final": content}, ensure_ascii=False, indent=2), flush=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--smiles-field", default="drug")
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--experiment-mode", choices=sorted(EXPERIMENT_MODES), default="native")
    parser.add_argument("--retrieval-source", default="chembl")
    parser.add_argument(
        "--neighbor-identity-policy",
        choices=NEIGHBOR_IDENTITY_POLICIES,
        default="operational",
    )
    parser.add_argument("--identity-blind", action="store_true")
    parser.add_argument("--harness-prefetch-tools", action="store_true")
    parser.add_argument("--single-analysis-source-run-dir", default="")
    parser.add_argument("--group-analysis-source-run-dir", default="")
    parser.add_argument("--retrieval-replay-run-dir", default="")
    parser.add_argument("--prefetched-tool-replay-run-dir", default="")
    parser.add_argument(
        "--tier1-replacement-index",
        default="",
        help="Optional neighbor index used to replace all base Tier 1 retrieval groups.",
    )
    parser.add_argument(
        "--tier1-replacement-groups",
        nargs="*",
        default=None,
        help="Optional group ids to retrieve from --tier1-replacement-index.",
    )
    parser.add_argument("--chembl-sqlite", default=DEFAULT_CHEMBL_SQLITE)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--resume-final-from-run-dir", default="")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write retrieval.json and stop before any LLM request.",
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--tool-service-url", default=DEFAULT_TOOL_SERVICE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--max-groups", type=int, default=0, help="Debug limit; 0 means all groups with neighbors.")
    parser.add_argument("--timeout-s", type=int, default=180)
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
    parser.add_argument("--disable-group-tools", action="store_true")
    parser.add_argument("--enable-chembl-exact-context", action="store_true")
    parser.add_argument("--top-k-per-group", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.3)
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
    add_final_evidence_surface_argument(parser)
    add_final_decision_profile_argument(
        parser,
        choices=BBB_FINAL_DECISION_PROFILES,
    )
    parser.add_argument(
        "--bbb-prompt-profile",
        choices=BBB_PROMPT_PROFILES,
        default=DEFAULT_BBB_PROMPT_PROFILE,
    )
    parser.add_argument("--groups", nargs="*", default=None, help="Optional exact Tier.endpoint_group ids to reason over.")
    args = parser.parse_args(argv)
    args.groups = _normalize_group_args(args.groups)
    args.tier1_replacement_groups = _normalize_group_args(args.tier1_replacement_groups)
    return args


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


def _log(message: str) -> None:
    print(f"[bbb_reasoning_pipeline] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
