"""Run Skin_Reaction analog reasoning with group-level parallel LLM calls."""

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

from tools.chembl_tool.common.assay_reranking.v11 import (
    TEMPLATE_PROFILE as V11_TEMPLATE_PROFILE,
    V11CachedAssayReranker,
    default_cache_paths,
    model_profile,
)
from tools.chembl_tool.common.assay_transfer_prompt_policy import (
    SCORED_NEIGHBORS_POLICY_NAME,
    prepare_assay_transfer_selected_neighbors,
    public_assay_transfer_records,
    scored_neighbors_prompt_enabled,
    validate_scored_neighbors_configuration,
)
from tools.chembl_tool.common.assay_transfer_selection import (
    ASSAY_TRANSFER_DIVERSITY_MODES,
    ASSAY_TRANSFER_DIVERSITY_NONE,
    ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
    ASSAY_TRANSFER_RECORDS_PER_MOLECULE_MAX,
    ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    ASSAY_TRANSFER_SELECTION_UNITS,
    validate_assay_transfer_diversity,
    assay_transfer_selection_policy,
    validate_assay_transfer_records_per_molecule,
)
from tools.chembl_tool.common.cli.retrieval_args import add_retrieval_strategy_args
from tools.chembl_tool.common.evidence_contract import evidence_for_group_llm, evidence_for_llm
from tools.chembl_tool.common.coverage_reasoning import (
    NEIGHBOR_CONTEXT_PROFILES,
    STANDARD_NEIGHBOR_CONTEXT,
)
from tools.chembl_tool.common.experiment_retrieval import (
    ASSAY_TRANSFER_TOOL_STRATEGY,
    EXPERIMENT_MODES,
    retrieve_experiment_view,
)
from tools.chembl_tool.common.export import ensure_dir
from tools.chembl_tool.common.final_evidence_surface import (
    SUMMARY_ONLY,
    add_final_evidence_surface_argument,
    build_final_evidence_fields,
    compact_group_reasoning_outputs,
    final_evidence_instructions,
    prepare_resumed_final_inputs,
)
from tools.chembl_tool.common.identity_blind import (
    expose_neighbor_smiles_only,
    prepare_reasoning_retrieval,
    query_without_prefetched_tools,
    sanitize_identity_blind_branch_outputs,
)
from tools.chembl_tool.common.task_workflows.analogous_flat_prompt import (
    PROMPT_IDENTITY_VIEW,
    prompt_provenance as analogous_flat_prompt_provenance,
    reason_final as reason_analogous_flat_final,
    reason_group as reason_analogous_flat_group,
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
    attach_external_condition,
    clean_exact_match as _clean_exact_match,
    clean_shared_assay_context as _clean_shared_assay_context,
    llm_evidence_query_payload as _llm_evidence_query_payload,
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
from tools.chembl_tool.tasks.skin_reaction.chembl_exact_context import (
    DEFAULT_CHEMBL_SQLITE,
    enrich_retrieval_with_chembl_context,
)
from tools.chembl_tool.tasks.skin_reaction.experiment_config import get_source_config
from tools.chembl_tool.tasks.skin_reaction.prompt_profiles import (
    DEFAULT_SKIN_PROMPT_PROFILE,
    HISTORICAL_SKIN_PROMPT_PROFILE,
    SKIN_PROMPT_PROFILES,
    get_skin_prompt_profile,
)
from tools.chembl_tool.tasks.skin_reaction.retrieve_neighbors import load_index


DEFAULT_INPUT = "data/processed/Skin_Reaction/test.jsonl"
DEFAULT_INDEX = "outputs/chembl_tool/tasks/skin_reaction/evidence_library/skin_reaction_neighbor_index.pkl"
V11_DEFAULT_INDEX = (
    "outputs/chembl_tool/tasks/skin_reaction/evidence_library/"
    "starling_normalized_v7/08_neighbor_index/scaffold"
)
DEFAULT_OUT_ROOT = "outputs/chembl_tool/tasks/skin_reaction/reasoning/single_runs"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TOOL_SERVICE_URL = "http://127.0.0.1:8765"
V11_DEFAULT_PATHS = default_cache_paths("skin_reaction")
V11_MODEL_PROFILE = model_profile("skin_reaction")
_write_trace_jsonl = partial(
    write_trace_jsonl,
    prediction_field="skin_reaction_prediction",
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
                "features. Use this to assess whether property changes affect Skin_Reaction evidence transferability."
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
    if args.analogous_reasoning_only:
        if not args.identity_blind:
            raise SystemExit("Skin analogous-flat reasoning requires --identity-blind")
        if args.retrieval_strategy not in {
            ASSAY_TRANSFER_TOOL_STRATEGY,
            "morgan_fingerprint",
        }:
            raise SystemExit(
                "Skin analogous-flat reasoning requires assay_transfer_tool or "
                "morgan_fingerprint retrieval"
            )
        if (
            args.retrieval_strategy == ASSAY_TRANSFER_TOOL_STRATEGY
            and not args.enable_assay_transfer_scores
        ):
            raise SystemExit(
                "Skin analogous-flat assay-transfer reasoning requires "
                "--enable-assay-transfer-scores"
            )
    if args.retrieval_strategy == ASSAY_TRANSFER_TOOL_STRATEGY:
        if args.group_prompt_format != "assay_transfer_tool":
            raise SystemExit(
                "--retrieval-strategy assay_transfer_tool requires "
                "--group-prompt-format assay_transfer_tool"
            )
        if args.morgan_neighbor_selector != SIMILARITY_SELECTOR:
            raise SystemExit(
                "--morgan-neighbor-selector applies only to morgan_fingerprint retrieval"
            )
        if args.retrieval_source != "starling":
            raise SystemExit("Skin assay-transfer reranking requires --retrieval-source starling")
        if args.experiment_mode not in {"direct", "full_flat", "full_mechanism"}:
            raise SystemExit("Skin assay-transfer reranking requires a paper-facing experiment mode")
        if args.rerank_cache_mode != "read_only":
            raise SystemExit("Reasoning runs require --rerank-cache-mode read_only")
    elif args.group_prompt_format == "assay_transfer_tool":
        raise SystemExit(
            "--group-prompt-format assay_transfer_tool requires assay_transfer_tool retrieval"
        )
    if args.assay_transfer_min_score is not None and not 0.0 <= args.assay_transfer_min_score <= 1.0:
        raise SystemExit("--assay-transfer-min-score must be between 0 and 1")
    try:
        validate_assay_transfer_diversity(
            mode=args.assay_transfer_diversity_mode,
            score_slack=args.assay_transfer_diversity_score_slack,
        )
        validate_assay_transfer_records_per_molecule(
            args.assay_transfer_records_per_molecule,
            selection_unit=args.assay_transfer_selection_unit,
        )
        validate_scored_neighbors_configuration(
            enabled=args.enable_assay_transfer_scores,
            experiment_mode=args.experiment_mode,
            retrieval_source=args.retrieval_source,
            retrieval_reranker=(
                "assay_transfer"
                if args.retrieval_strategy == ASSAY_TRANSFER_TOOL_STRATEGY
                else "none"
            ),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if (
        args.assay_transfer_records_per_molecule > 1
        and args.retrieval_strategy != ASSAY_TRANSFER_TOOL_STRATEGY
    ):
        raise SystemExit(
            "--assay-transfer-records-per-molecule greater than 1 requires "
            "--retrieval-strategy assay_transfer_tool"
        )
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
            enable_group_tools=not args.disable_group_tools and not args.disable_flat_tools,
            max_tool_rounds=args.max_tool_rounds,
            reasoning_effort=args.reasoning_effort,
            enable_thinking=args.enable_thinking,
        )
        return _resume_final_from_run_dir(Path(args.resume_final_from_run_dir), client)

    run_id = args.run_id or time.strftime("skin_reaction_reasoning_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    out_dir = ensure_dir(Path(args.out_root) / run_id)
    _log(f"run_id={run_id}")

    query_record = _read_jsonl_record(Path(args.input_jsonl), args.query_index)
    query_smiles = str(query_record.get(args.smiles_field) or "")
    if not query_smiles:
        raise SystemExit(f"Input record has no `{args.smiles_field}` value.")

    reranker = None
    if args.retrieval_strategy == ASSAY_TRANSFER_TOOL_STRATEGY:
        reranker = V11CachedAssayReranker(
            task_id="skin_reaction",
            catalog_path=args.rerank_catalog,
            candidate_manifest_path=args.rerank_candidate_manifest,
            cache_path=args.rerank_cache,
            cache_mode="read_only",
            model=args.assay_transfer_model,
            model_revision=args.assay_transfer_model_revision,
        )
    retrieval = load_retrieval_replay(
        args.retrieval_replay_run_dir,
        query_smiles,
        expected_neighbor_selector=args.morgan_neighbor_selector,
        expected_reranker_provenance=(
            reranker.provenance() if reranker is not None else {"name": "none"}
        ),
        expected_assay_transfer_selection_policy=(
            {
                "min_score": args.assay_transfer_min_score,
                "diversity": assay_transfer_selection_policy(
                    mode=args.assay_transfer_diversity_mode,
                    score_slack=args.assay_transfer_diversity_score_slack,
                    selection_unit=args.assay_transfer_selection_unit,
                    records_per_molecule=(
                        args.assay_transfer_records_per_molecule
                    ),
                ),
            }
            if reranker is not None
            else None
        ),
    )
    index = None
    if retrieval is None and args.experiment_mode != "none":
        _log("loading neighbor index")
        index = load_index(Path(args.index))
    if retrieval is None:
        _log(f"building retrieval view mode={args.experiment_mode} source={args.retrieval_source}")
        retrieval = retrieve_experiment_view(
            query_smiles,
            index,
            mode=args.experiment_mode,
            config=get_source_config(args.retrieval_source) if args.experiment_mode not in {"none", "native"} else None,
            top_k_per_group=args.top_k_per_group,
            min_similarity=args.min_similarity,
            native_groups=args.groups,
            neighbor_identity_policy=args.neighbor_identity_policy,
            neighbor_selector=args.morgan_neighbor_selector,
            reranker=reranker,
            assay_transfer_initial_morgan_filter=(
                args.assay_transfer_initial_morgan_filter
            ),
            assay_transfer_min_score=args.assay_transfer_min_score,
            assay_transfer_diversity_mode=args.assay_transfer_diversity_mode,
            assay_transfer_diversity_score_slack=(
                args.assay_transfer_diversity_score_slack
            ),
            assay_transfer_selection_unit=args.assay_transfer_selection_unit,
            assay_transfer_records_per_molecule=(
                args.assay_transfer_records_per_molecule
            ),
        )
    else:
        _log(f"replaying frozen retrieval from {args.retrieval_replay_run_dir}")
    if retrieval.get("status") != "ok":
        raise SystemExit(json.dumps(retrieval.get("errors", []), ensure_ascii=False))
    if reranker is not None:
        reranker.cache.close()
        try:
            prepare_assay_transfer_selected_neighbors(
                retrieval, expose_scores=args.enable_assay_transfer_scores
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    attach_external_condition(retrieval, query_record)
    if args.enable_chembl_exact_context and index is not None:
        _log("enriching retrieval with exact ChEMBL context")
        retrieval = enrich_retrieval_with_chembl_context(
            retrieval,
            index,
            chembl_sqlite=args.chembl_sqlite,
        )

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
        enable_group_tools=(
            not args.disable_group_tools
            and not args.disable_flat_tools
            and not args.analogous_reasoning_only
        ),
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
        include_query_tools=not args.analogous_reasoning_only,
        include_neighbor_tools=(
            not args.disable_flat_tools and not args.analogous_reasoning_only
        ),
    )
    if args.analogous_reasoning_only:
        reasoning_retrieval = expose_neighbor_smiles_only(
            reasoning_retrieval, retrieval
        )
    reasoning_groups = [group for group in reasoning_retrieval["groups"] if group.get("neighbors")]
    if args.max_groups:
        reasoning_groups = reasoning_groups[: args.max_groups]
    if args.analogous_reasoning_only:
        single_output = {
            "analysis_id": "single_molecule",
            "status": "omitted",
            "reason": "analogous_reasoning_only",
        }
        group_outputs = [
            reason_analogous_flat_group(client, group, task_id="skin_reaction")
            for group in reasoning_groups
        ]
    else:
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
            prompt_profile=args.skin_prompt_profile,
            disable_flat_tools=args.disable_flat_tools,
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

    final_output = (
        reason_analogous_flat_final(client, group_outputs, task_id="skin_reaction")
        if args.analogous_reasoning_only
        else _run_final_reasoning(
            client,
            reasoning_retrieval,
            single_output,
            group_outputs,
            final_evidence_surface=args.final_evidence_surface,
            prompt_profile=args.skin_prompt_profile,
        )
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
        "retrieval_strategy": args.retrieval_strategy,
        "assay_transfer_profile": args.assay_transfer_profile,
        "enable_assay_transfer_scores": args.enable_assay_transfer_scores,
        "assay_transfer_selection_unit": args.assay_transfer_selection_unit,
        "assay_transfer_records_per_molecule": (
            args.assay_transfer_records_per_molecule
        ),
        "assay_transfer_initial_morgan_filter": (
            args.assay_transfer_initial_morgan_filter
        ),
        "rerank_catalog": args.rerank_catalog if reranker is not None else "",
        "rerank_cache": args.rerank_cache if reranker is not None else "",
        "rerank_candidate_manifest": (
            args.rerank_candidate_manifest if reranker is not None else ""
        ),
        "neighbor_identity_policy": args.neighbor_identity_policy,
        "morgan_neighbor_selector": args.morgan_neighbor_selector,
        "neighbor_selector": args.morgan_neighbor_selector,
        "neighbor_context_profile": args.neighbor_context_profile,
        "final_evidence_surface": args.final_evidence_surface,
        "task_prompt_profile": args.skin_prompt_profile,
        "label_scope": get_skin_prompt_profile(args.skin_prompt_profile).label_scope,
        "retrieval_replay_source_run_dir": args.retrieval_replay_run_dir,
        "prefetched_tool_replay_source_run_dir": args.prefetched_tool_replay_run_dir,
        "identity_blind": args.identity_blind,
        "analogous_reasoning_only": args.analogous_reasoning_only,
        "prompt_identity_view": (
            PROMPT_IDENTITY_VIEW if args.analogous_reasoning_only else "identity_blind"
            if args.identity_blind else "deployment_visible"
        ),
        "analogous_flat_prompt_provenance": (
            analogous_flat_prompt_provenance("skin_reaction")
            if args.analogous_reasoning_only else {}
        ),
        "disable_flat_tools": args.disable_flat_tools,
        "harness_prefetch_tools": (
            False
            if args.analogous_reasoning_only
            else args.identity_blind or args.harness_prefetch_tools
        ),
        "neighbor_index": args.index if args.experiment_mode != "none" else "",
        "retrieval_evidence_source": retrieval.get("evidence_source", {}),
        "model": args.model,
        "base_url": args.base_url,
        "tool_service_url": args.tool_service_url,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "thinking": {"type": "enabled"} if args.enable_thinking else {"type": "disabled"},
        "group_tools_enabled": (
            not args.disable_group_tools
            and not args.disable_flat_tools
            and not args.analogous_reasoning_only
        ),
        "tool_execution_mode": (
            "omitted"
            if args.analogous_reasoning_only
            else "harness_prefetch_query_only"
            if args.disable_flat_tools
            else "harness_prefetch"
            if args.identity_blind or args.harness_prefetch_tools
            else "llm_function_call"
        ),
        "single_analysis_source_run_dir": args.single_analysis_source_run_dir,
        "group_analysis_source_run_dir": args.group_analysis_source_run_dir,
        "chembl_exact_context_enabled": args.enable_chembl_exact_context,
        "chembl_sqlite": args.chembl_sqlite,
        "group_tool_names": [tool["function"]["name"] for tool in GROUP_REASONING_TOOLS]
        if not args.disable_group_tools
        and not args.disable_flat_tools
        and not args.analogous_reasoning_only
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
    prompt_profile: str = DEFAULT_SKIN_PROMPT_PROFILE,
    disable_flat_tools: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    outputs = list(group_outputs or [])
    reused_group_ids = {str(output.get("group_id") or "") for output in outputs}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _reason_one_group,
                client,
                _llm_evidence_query_payload(
                    query_without_prefetched_tools(retrieval["query"])
                    if disable_flat_tools
                    else retrieval["query"]
                ),
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


def _reason_single_molecule(
    client: OpenAICompatibleClient,
    query: dict[str, Any],
    chembl_context: dict[str, Any] | None = None,
    *,
    prompt_profile: str = DEFAULT_SKIN_PROMPT_PROFILE,
) -> dict[str, Any]:
    profile = get_skin_prompt_profile(prompt_profile)
    instructions = [
        "Call molecule_properties for the query molecule before analysis.",
        *profile.single_instructions,
    ]
    if query.get("prefetched_molecule_properties"):
        instructions[0] = "Use the harness-prefetched molecule_properties result."
        if query.get("identity_hidden"):
            instructions[0] += " Do not identify or name the query."
    payload: dict[str, Any] = {
        "task": profile.single_task,
        "query": query,
        "instructions": instructions,
        "required_json_schema": profile.single_schema,
    }
    if chembl_context:
        instructions.append(
            "If exact_query_chembl_context is found, distinguish direct same-molecule ChEMBL Skin_Reaction evidence from the structural/physicochemical prior."
        )
        payload["exact_query_chembl_context"] = chembl_context
    messages = [
        {
            "role": "system",
            "content": (
                profile.single_system_role
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
    prompt_profile: str = DEFAULT_SKIN_PROMPT_PROFILE,
) -> dict[str, Any]:
    profile = get_skin_prompt_profile(prompt_profile)
    messages = [
        {
            "role": "system",
            "content": (
                profile.group_system_role
                + (
                    "Use the harness-prefetched comparison results; do not call tools. "
                    + ("Do not infer query identity. " if group.get("identity_blind") else "")
                    if group.get("tools_prefetched")
                    else (
                        "No tools are available for this branch. "
                        + ("Do not infer query identity. " if group.get("identity_blind") else "")
                        if not client.enable_group_tools
                        else "You may call the provided molecule comparison tools when structural or property differences matter. "
                    )
                )
                + "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                _group_prompt_payload(
                    query,
                    group,
                    prompt_profile=prompt_profile,
                    include_query_tool_guidance=(
                        bool(group.get("tools_prefetched"))
                        or client.enable_group_tools
                    ),
                ),
                ensure_ascii=False,
            ),
        },
    ]
    response = call_group_branch(
        client,
        messages,
        group=group,
        tools=GROUP_REASONING_TOOLS,
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
    prompt_profile: str = DEFAULT_SKIN_PROMPT_PROFILE,
) -> dict[str, Any]:
    profile = get_skin_prompt_profile(prompt_profile)
    evidence_fields, surface_audit = build_final_evidence_fields(
        retrieval,
        compact_group_reasoning_outputs(group_outputs),
        surface=final_evidence_surface,
    )
    messages = [
        {
            "role": "system",
            "content": (
                profile.final_system_role
                + "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": profile.final_task,
                    "query": _llm_evidence_query_payload(retrieval["query"]),
                    "retrieval_coverage": retrieval["coverage"],
                    "single_molecule_analysis": {
                        "status": single_output.get("status"),
                        "content": validated_branch_content(single_output),
                    },
                    **evidence_fields,
                    "instructions": list(profile.final_instructions)
                    + final_evidence_instructions(final_evidence_surface),
                    "required_json_schema": profile.final_schema,
                },
                ensure_ascii=False,
            ),
        },
    ]
    response = call_with_json_validation(
        client.chat_json,
        messages,
        required_fields=profile.final_required_fields,
        allowed_values=profile.final_allowed_values,
    )
    output = {"status": "ok" if structured_response_is_valid(response) else "error", "llm": response}
    if surface_audit is not None:
        output["final_evidence_surface"] = surface_audit
    return output


def _selected_assay_record(
    neighbor: dict[str, Any],
    group: dict[str, Any],
    selected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = selected or neighbor.get("transfer_winning_record") or {}
    if not selected:
        return {}
    example = {
        "source_contract": selected.get("source_contract"),
        "source_fields": selected.get("source_fields"),
        **(
            {"resolved_measurement_display": selected["resolved_measurement_display"]}
            if selected.get("resolved_measurement_display")
            else {}
        ),
    }
    return evidence_for_llm(
        {
            "evidence_source": "Starling normalized skin reaction",
            "molecule_chembl_id": neighbor.get("molecule_chembl_id"),
            "canonical_smiles": neighbor.get("canonical_smiles"),
            "group_id": group.get("group_id"),
            "tier": group.get("tier"),
            "endpoint_group": group.get("endpoint_group"),
            "standard_type": "selected source assay record",
            "evidence_text": "Source-contracted Stage 07 retrieval record selected by the assay-transfer reranker.",
            "source_record_examples": [example],
        }
    )


def _assay_transfer_record_cards(
    neighbor: dict[str, Any], group: dict[str, Any]
) -> dict[str, Any]:
    records = public_assay_transfer_records(neighbor)
    if len(records) == 1:
        return {
            "assay_transfer_score": records[0]["assay_transfer_score"],
            "assay_transfer_record": _selected_assay_record(
                neighbor, group, records[0]["record"]
            ),
        }
    return {
        "assay_transfer_records": [
            {
                "record_rank": record["record_rank"],
                "assay_transfer_score": record["assay_transfer_score"],
                "assay_transfer_record": _selected_assay_record(
                    neighbor, group, record["record"]
                ),
            }
            for record in records
        ]
    }


def build_group_prompt_payload(
    query: dict[str, Any],
    group: dict[str, Any],
    *,
    prompt_profile: str = DEFAULT_SKIN_PROMPT_PROFILE,
    include_query_tool_guidance: bool = True,
) -> dict[str, Any]:
    profile = get_skin_prompt_profile(prompt_profile)
    include_assay_transfer_score = bool(
        (group.get("transfer_neighbor_selection") or {}).get(
            "selection_score_is_llm_visible"
        )
    )
    multiple_assay_records = include_assay_transfer_score and any(
        len(neighbor.get("transfer_selected_records") or []) > 1
        for neighbor in group.get("neighbors") or []
    )
    return bound_group_prompt_payload({
        "task": profile.group_task,
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
                **(
                    _assay_transfer_record_cards(neighbor, group)
                    if include_assay_transfer_score
                    else {}
                ),
                **(
                    {
                        "prefetched_comparisons": (
                            neighbor.get("prefetched_comparisons") or []
                        )
                    }
                    if include_query_tool_guidance
                    else {}
                ),
                "evidence_rows": [
                    evidence_for_group_llm(row, group)
                    for row in neighbor["evidence_rows"]
                ],
                "shared_assay_context": _clean_shared_assay_context(neighbor.get("shared_assay_context") or {}),
            }
            for neighbor in group["neighbors"]
        ],
        "instructions": [
            *[
                line
                for line in profile.group_instructions
                if include_query_tool_guidance
                or not any(
                    tool_name in line
                    for tool_name in (
                        "mmp_structure_compare",
                        "properties_compare",
                        "molecule_properties",
                    )
                )
            ],
            *(
                [
                    (
                        "Each assay_transfer_records item is an endpoint-distinct Stage 07 source "
                        "record for the same selected molecule, ordered by its assay_transfer_score. "
                        "Each score is a 0-1 cached model estimate that the exact record transfers "
                        "to the query under the copied assay context. Scores are uncalibrated, may "
                        "be extrapolative for qualitative records, and are not skin-reaction "
                        "probabilities, label votes, or deterministic overrides."
                        if multiple_assay_records
                        else "assay_transfer_score is a 0-1 cached model estimate that the exact selected Stage 07 source record transfers to the query under the copied assay context. It is uncalibrated, may be extrapolative for qualitative records, and is not a skin-reaction probability, label vote, or deterministic override."
                    )
                ]
                if include_assay_transfer_score
                else []
            ),
        ],
        "required_json_schema": profile.group_schema,
    }, evidence_prompt_profile=str(group.get("evidence_prompt_profile") or ""))


# Historical internal callers keep working while external materializers use
# the explicit public adapter above.
_group_prompt_payload = build_group_prompt_payload


def _clean_query_chembl_context(context: dict[str, Any]) -> dict[str, Any]:
    status = context.get("status") or "not_available"
    if status != "found":
        return {key: context.get(key, "") for key in ["status", "reason", "standard_inchi_key"] if context.get(key)}
    return {
        "status": "found",
        "selected_molecule_chembl_id": context.get("selected_molecule_chembl_id", ""),
        "exact_matches": [_clean_exact_match(match) for match in context.get("exact_matches", [])],
        "skin_reaction_relevant_evidence_rows": [
            evidence_for_llm(row)
            for row in context.get("skin_reaction_relevant_evidence_rows", [])
        ],
    }


def _parse_json_content(content: str) -> Any:
    return parse_json_content(content)


def _validate_prompt_profile_reuse(args: argparse.Namespace) -> None:
    """Reject branch reuse across Skin prompt contracts."""
    require_matching_prompt_profiles(
        target_profile=str(args.skin_prompt_profile),
        source_dirs=(
            args.single_analysis_source_run_dir,
            args.group_analysis_source_run_dir,
        ),
        historical_profile=HISTORICAL_SKIN_PROMPT_PROFILE,
    )


def _manifest_prompt_profile(manifest: dict[str, Any]) -> str:
    return prompt_profile_from_manifest(
        manifest,
        historical_profile=HISTORICAL_SKIN_PROMPT_PROFILE,
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
    manifest["label_scope"] = get_skin_prompt_profile(prompt_profile).label_scope
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
    parser.add_argument("--groups", nargs="*", default=None, help="Optional native index group ids.")
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
    parser.add_argument("--disable-flat-tools", action="store_true")
    parser.add_argument(
        "--analogous-reasoning-only",
        "--analogous_reasoning_only",
        dest="analogous_reasoning_only",
        action="store_true",
    )
    parser.add_argument("--enable-chembl-exact-context", action="store_true")
    parser.add_argument("--top-k-per-group", type=int, default=3)
    parser.add_argument("--enable-assay-transfer-scores", action="store_true")
    parser.add_argument("--min-similarity", type=float, default=0.3)
    add_retrieval_strategy_args(parser)
    parser.add_argument(
        "--assay-transfer-profile",
        choices=["v11_with_categorical"],
        default="v11_with_categorical",
    )
    parser.add_argument("--assay-transfer-min-score", type=float, default=None)
    parser.add_argument(
        "--assay-transfer-diversity-mode",
        choices=ASSAY_TRANSFER_DIVERSITY_MODES,
        default=ASSAY_TRANSFER_DIVERSITY_NONE,
    )
    parser.add_argument("--assay-transfer-diversity-score-slack", type=float, default=0.0)
    parser.add_argument(
        "--assay-transfer-selection-unit",
        choices=ASSAY_TRANSFER_SELECTION_UNITS,
        default=ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    )
    parser.add_argument(
        "--assay-transfer-records-per-molecule",
        type=int,
        choices=range(
            ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
            ASSAY_TRANSFER_RECORDS_PER_MOLECULE_MAX + 1,
        ),
        default=ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
        metavar="N",
    )
    parser.add_argument("--rerank-catalog", default=V11_DEFAULT_PATHS["catalog"])
    parser.add_argument("--rerank-cache", default=V11_DEFAULT_PATHS["cache"])
    parser.add_argument(
        "--rerank-candidate-manifest",
        default=V11_DEFAULT_PATHS["candidate_manifest"],
    )
    parser.add_argument("--rerank-cache-mode", choices=["read_only"], default="read_only")
    parser.add_argument("--assay-transfer-model", default=V11_MODEL_PROFILE["model"])
    parser.add_argument(
        "--assay-transfer-model-revision", default=V11_MODEL_PROFILE["revision"]
    )
    parser.add_argument(
        "--assay-transfer-template-profile",
        choices=[V11_TEMPLATE_PROFILE],
        default=V11_TEMPLATE_PROFILE,
    )
    parser.add_argument(
        "--group-prompt-format",
        choices=["legacy", "assay_transfer_tool"],
        default="legacy",
    )
    parser.add_argument(
        "--neighbor-context-profile",
        choices=NEIGHBOR_CONTEXT_PROFILES,
        default=STANDARD_NEIGHBOR_CONTEXT,
    )
    add_final_evidence_surface_argument(parser)
    parser.add_argument(
        "--skin-prompt-profile",
        choices=SKIN_PROMPT_PROFILES,
        default=DEFAULT_SKIN_PROMPT_PROFILE,
    )
    args = parser.parse_args(argv)
    if args.disable_flat_tools and args.experiment_mode != "full_flat":
        parser.error("--disable-flat-tools requires --experiment-mode full_flat")
    if args.analogous_reasoning_only and args.experiment_mode != "full_flat":
        parser.error(
            "Skin --analogous-reasoning-only requires --experiment-mode full_flat"
        )
    if (
        args.retrieval_strategy == ASSAY_TRANSFER_TOOL_STRATEGY
        and args.assay_transfer_profile == "v11_with_categorical"
        and args.index == DEFAULT_INDEX
    ):
        args.index = V11_DEFAULT_INDEX
    if args.assay_transfer_initial_morgan_filter == 100:
        args.assay_transfer_initial_morgan_filter = 50
    if args.min_similarity == 0.3:
        args.min_similarity = 0.0
    return args


def _log(message: str) -> None:
    print(f"[skin_reaction_reasoning_pipeline] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
