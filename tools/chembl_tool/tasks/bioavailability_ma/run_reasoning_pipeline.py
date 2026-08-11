"""Run Bioavailability_Ma analog reasoning with group-level parallel LLM calls."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
import uuid
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
from tools.chembl_tool.common.identity_blind import (
    prepare_reasoning_retrieval,
    sanitize_identity_blind_branch_outputs,
)
from tools.chembl_tool.common.json_utils import parse_json_content
from tools.chembl_tool.common.neighbor_selection import (
    NEIGHBOR_SELECTORS,
    SIMILARITY_SELECTOR,
)
from tools.chembl_tool.common.openai_reasoning_client import OpenAICompatibleClient
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
from tools.chembl_tool.tasks.bioavailability_ma.chembl_exact_context import (
    DEFAULT_CHEMBL_SQLITE,
    enrich_retrieval_with_chembl_context,
)
from tools.chembl_tool.tasks.bioavailability_ma.constants import BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT
from tools.chembl_tool.tasks.bioavailability_ma.experiment_config import get_source_config
from tools.chembl_tool.tasks.bioavailability_ma.retrieve_neighbors import load_index


DEFAULT_INPUT = "data/processed/Bioavailability_Ma/test.jsonl"
DEFAULT_INDEX = "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/bioavailability_neighbor_index.pkl"
DEFAULT_OUT_ROOT = "outputs/chembl_tool/tasks/bioavailability_ma/reasoning/single_runs"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TOOL_SERVICE_URL = "http://127.0.0.1:8765"


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
                "features. Use this to assess whether property changes affect oral bioavailability evidence transferability."
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

SINGLE_MOLECULE_TOOL_CHOICE = {"type": "function", "function": {"name": "molecule_properties"}}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
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

    run_id = args.run_id or time.strftime("bioavailability_reasoning_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
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
            neighbor_selector=args.neighbor_selector,
        )
    else:
        _log(f"replaying frozen retrieval from {args.retrieval_replay_run_dir}")
    if retrieval.get("status") != "ok":
        raise SystemExit(json.dumps(retrieval.get("errors", []), ensure_ascii=False))
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
        "retrieval_replay_source_run_dir": args.retrieval_replay_run_dir,
        "prefetched_tool_replay_source_run_dir": args.prefetched_tool_replay_run_dir,
        "identity_blind": args.identity_blind,
        "harness_prefetch_tools": args.identity_blind or args.harness_prefetch_tools,
        "neighbor_index": args.index if args.experiment_mode != "none" else "",
        "retrieval_evidence_source": retrieval.get("evidence_source", {}),
        "model": args.model,
        "base_url": args.base_url,
        "tool_service_url": args.tool_service_url,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "thinking": {"type": "enabled"} if args.enable_thinking else {"type": "disabled"},
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
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    outputs = list(group_outputs or [])
    reused_group_ids = {str(output.get("group_id") or "") for output in outputs}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_reason_one_group, client, _llm_query_payload(retrieval["query"]), group): group["group_id"]
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
) -> dict[str, Any]:
    instructions = [
        "Call molecule_properties for the query molecule before analysis.",
        "Assess oral bioavailability prior from molecular weight, logP/logD, TPSA, HBD/HBA, ionization/pKa, charge, rotatable bonds, and functional groups.",
        "Return JSON with oral_bioavailability_prior, absorption_prior, solubility_or_dissolution_prior, metabolism_or_clearance_prior, confidence, reasoning_summary, property_drivers, caveats.",
    ]
    if query.get("prefetched_molecule_properties"):
        instructions[0] = "Use the harness-prefetched molecule_properties result."
        if query.get("identity_hidden"):
            instructions[0] += " Do not identify or name the query."
    payload: dict[str, Any] = {
        "task": "Single-molecule oral bioavailability plausibility analysis.",
        "query": query,
        "instructions": instructions,
        "required_json_schema": {
            "oral_bioavailability_prior": "high | low | mixed_or_unclear",
            "absorption_prior": "favorable | unfavorable | mixed_or_unclear",
            "solubility_or_dissolution_prior": "favorable | unfavorable | mixed_or_unclear",
            "metabolism_or_clearance_prior": "favorable | unfavorable | mixed_or_unclear",
            "exact_chembl_evidence_assessment": "string",
            "confidence": "high | moderate | low",
            "reasoning_summary": "string",
            "property_drivers": ["string"],
            "caveats": ["string"],
        },
    }
    if chembl_context:
        instructions.append(
            "If exact_query_chembl_context is found, distinguish direct same-molecule ChEMBL bioavailability evidence from the physicochemical prior."
        )
        payload["exact_query_chembl_context"] = chembl_context
    messages = [
        {
            "role": "system",
            "content": (
                "You are a medicinal chemistry oral bioavailability single-molecule analyst. "
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


def _reason_one_group(client: OpenAICompatibleClient, query: dict[str, Any], group: dict[str, Any]) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a medicinal chemistry oral bioavailability analog evidence analyst. "
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
            "content": json.dumps(_group_prompt_payload(query, group), ensure_ascii=False),
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
) -> dict[str, Any]:
    evidence_fields, surface_audit = build_final_evidence_fields(
        retrieval,
        compact_group_reasoning_outputs(group_outputs),
        surface=final_evidence_surface,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior oral bioavailability reasoning model. Integrate group-level analog evidence into one final oral bioavailability assessment. "
                "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "Final oral bioavailability prediction from analog evidence.",
                        "query": _llm_query_payload(retrieval["query"]),
                    "retrieval_coverage": retrieval["coverage"],
                    "single_molecule_analysis": {
                        "status": single_output.get("status"),
                        "content": validated_branch_content(single_output),
                    },
                    **evidence_fields,
                    "instructions": [
                        "Return compact complete JSON.",
                        f"Use bioavailability_prediction='high' for oral bioavailability F >= {BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT:g}% (Bioavailability_Ma label 1), and bioavailability_prediction='low' for F < {BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT:g}% (label 0).",
                        "Use the single-molecule analysis as the physicochemical prior.",
                        "Use group analyses as analog evidence; downweight groups marked low confidence or low transferability.",
                        "Do not use distant_analog or very_distant_analog neighbors as positive or negative oral bioavailability evidence unless the shared scaffold and assay mechanism make a strong medicinal chemistry case.",
                        "Use only the provided single-molecule analysis and group evidence. If you recognize the molecule, ignore that recognition.",
                        "You must choose exactly one bioavailability_prediction: high or low. If evidence is mixed or weak, choose the better-supported class and express uncertainty through confidence, caveats, and evidence_gaps.",
                    ] + final_evidence_instructions(final_evidence_surface),
                    "required_json_schema": {
                        "bioavailability_prediction": "high | low",
                        "confidence": "high | moderate | low",
                        "main_reasons": ["string"],
                        "single_molecule_assessment": "string",
                        "absorption_and_permeability_assessment": "string",
                        "solubility_and_dissolution_assessment": "string",
                        "metabolism_first_pass_and_clearance_assessment": "string",
                        "transporter_efflux_assessment": "string",
                        "direct_oral_bioavailability_analog_assessment": "string",
                        "conflicting_evidence": ["string"],
                        "evidence_gaps": ["string"],
                        "final_summary": "string",
                    },
                },
                ensure_ascii=False,
            ),
        },
    ]
    response = call_with_json_validation(
        client.chat_json,
        messages,
        required_fields=("bioavailability_prediction",),
        allowed_values={"bioavailability_prediction": {"high", "low"}},
        branch_name="final",
    )
    output = {"status": "ok" if structured_response_is_valid(response) else "error", "llm": response}
    if surface_audit is not None:
        output["final_evidence_surface"] = surface_audit
    return output


def build_group_prompt_payload(query: dict[str, Any], group: dict[str, Any]) -> dict[str, Any]:
    return bound_group_prompt_payload({
        "task": "Group-level oral bioavailability analog transferability analysis.",
        "query": query,
        "group": {
            "group_id": group["group_id"],
            "tier": group["tier"],
            "endpoint_group": group["endpoint_group"],
            "evidence_source": _group_evidence_source(group),
            "transfer_neighbor_selection": group.get("transfer_neighbor_selection", {}),
        },
        "neighbors": [
            {
                "rank": neighbor["rank"],
                "structural_rank": neighbor.get("structural_rank", neighbor["rank"]),
                "transfer_selection_rank": neighbor.get("transfer_selection_rank"),
                "transfer_selection_score": neighbor.get("transfer_selection_score"),
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
        "instructions": [
            "Use only this group's evidence.",
            "Each evidence_rows item follows minimal_evidence.v1; read endpoint/measurement, text, annotations, quality, provenance, and examples without assuming a source-specific schema.",
            "Assess structural transferability from neighbors to the query.",
            "Low-similarity analogs are intentionally included. You must explicitly judge whether they are transferable.",
            "Do not use distant_analog or very_distant_analog neighbors as positive or negative oral bioavailability evidence unless the shared scaffold and assay mechanism make a strong medicinal chemistry case.",
            "Use mmp_structure_compare to inspect scaffold/MCS/matched-pair differences when similarity bucket alone is not enough.",
            "Use properties_compare when property differences such as pKa, logD, TPSA, charge, HBD/HBA, logP, molecular size, or polarity could affect oral bioavailability transferability.",
            "Tool outputs are authoritative only for the pair they compare; cite which neighbor each tool result supports.",
            "Use same_endpoint_activity as direct query-vs-neighbor assay comparison when present.",
            "Use same_assay_different_endpoint_activity only as same-assay context; do not directly compare numeric values across different endpoints.",
            "Distinguish direct oral bioavailability, in vivo oral exposure/absorption, in vitro permeability, solubility/dissolution, metabolism/clearance, formulation/food-effect context, and weak inhibition/binding evidence.",
            "Do not convert CYP IC50/inhibition into metabolic instability, and do not convert transporter IC50/inhibition directly into substrate/transport unless assay context supports it.",
            "Return key_evidence as structured evidence cards, not a plain list of molecule ids.",
            "For aggregated evidence, examples preserve endpoint, value, condition, and support-text pairings. Do not treat qualitative examples or surrogate_proxy roles as direct F% measurements.",
            "For each key_evidence item, derive assay_signal and activity_values from the provided evidence_rows, derive tool_summary from tool outputs, and judge transferability/effect_on_bioavailability_reasoning yourself.",
            "Return JSON with useful_for_bioavailability_reasoning, transferability, evidence_direction, confidence, reasoning_summary, key_evidence, caveats.",
        ],
        "required_json_schema": {
            "useful_for_bioavailability_reasoning": "boolean",
            "transferability": "high | moderate | low | not_applicable",
            "evidence_direction": (
                "supports_high_bioavailability | argues_against_high_bioavailability | absorption_support | "
                "permeability_support | solubility_support | solubility_risk | metabolic_stability_support | "
                "first_pass_or_clearance_risk | transporter_efflux_risk | neutral_or_unclear"
            ),
            "confidence": "high | moderate | low",
            "reasoning_summary": "string",
            "key_evidence": [
                {
                    "molecule_chembl_id": "string",
                    "similarity": "number or null",
                    "similarity_bucket": "string",
                    "assay_signal": "string",
                    "activity_values": ["string"],
                    "tool_summary": "string",
                    "transferability": "high | moderate | low | not_applicable",
                    "effect_on_bioavailability_reasoning": "string",
                }
            ],
            "caveats": ["string"],
        },
    })


# Historical internal callers keep working while external materializers use
# the explicit public adapter above.
_group_prompt_payload = build_group_prompt_payload


def _llm_query_payload(query: dict[str, Any]) -> dict[str, Any]:
    if query.get("identity_hidden"):
        return {
            "molecule_id": "query",
            "identity_hidden": True,
            "prefetched_molecule_properties": query.get("prefetched_molecule_properties") or {},
        }
    payload = {
        "input_smiles": query.get("input_smiles", ""),
        "canonical_smiles": query.get("canonical_smiles", ""),
    }
    if query.get("prefetched_molecule_properties"):
        payload["tools_prefetched"] = True
        payload["prefetched_molecule_properties"] = query["prefetched_molecule_properties"]
    return payload


def _clean_evidence_row(row: dict[str, Any]) -> dict[str, Any]:
    return evidence_for_llm(row)


def _group_evidence_source(group: dict[str, Any]) -> str:
    sources: list[str] = []
    for neighbor in group.get("neighbors", []):
        for row in neighbor.get("evidence_rows", []):
            source = str(row.get("evidence_source") or "").strip()
            if source and source not in sources:
                sources.append(source)
    if len(sources) > 1:
        return " + ".join(sources)
    if sources:
        return sources[0]
    return "ChEMBL"


def _clean_query_chembl_context(context: dict[str, Any]) -> dict[str, Any]:
    status = context.get("status") or "not_available"
    if status != "found":
        return {key: context.get(key, "") for key in ["status", "reason", "standard_inchi_key"] if context.get(key)}
    return {
        "status": "found",
        "selected_molecule_chembl_id": context.get("selected_molecule_chembl_id", ""),
        "exact_matches": [_clean_exact_match(match) for match in context.get("exact_matches", [])],
        "bioavailability_relevant_evidence_rows": [
            _clean_evidence_row(row) for row in context.get("bioavailability_relevant_evidence_rows", [])
        ],
    }


def _clean_exact_match(match: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_inchi_key",
        "mw_freebase",
        "alogp",
        "hba",
        "hbd",
        "psa",
        "rtb",
        "num_ro5_violations",
        "full_mwt",
        "aromatic_rings",
        "heavy_atoms",
        "qed_weighted",
        "full_molformula",
        "np_likeness_score",
    ]
    return {field: match.get(field, "") for field in fields}


def _clean_shared_assay_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "same_endpoint_activity": [
            _clean_shared_activity_card(card) for card in context.get("same_endpoint_activity", [])
        ],
        "same_assay_different_endpoint_activity": [
            _clean_shared_activity_card(card)
            for card in context.get("same_assay_different_endpoint_activity", [])
        ],
        "n_same_endpoint_activity": context.get("n_same_endpoint_activity", 0),
        "n_same_assay_different_endpoint_activity": context.get("n_same_assay_different_endpoint_activity", 0),
    }


def _clean_shared_activity_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "assay_chembl_id": card.get("assay_chembl_id", ""),
        "standard_type_match": card.get("standard_type_match", False),
        "units_match": card.get("units_match", False),
        "query_activity": _clean_activity_value(card.get("query_activity") or {}),
        "neighbor_activity": _clean_activity_value(card.get("neighbor_activity") or {}),
    }


def _clean_activity_value(activity: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "assay_chembl_id",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "activity_comment",
        "data_validity_comment",
        "standard_text_value",
        "action_type",
    ]
    return {field: activity.get(field, "") for field in fields}


def _parse_json_content(content: str) -> Any:
    return parse_json_content(content)


def _write_trace_jsonl(
    path: Path,
    *,
    query_record: dict[str, Any],
    query_index: int,
    smiles: str,
    single_output: dict[str, Any],
    group_outputs: list[dict[str, Any]],
    final_output: dict[str, Any],
) -> None:
    records = []
    records.append(_trace_record("single_molecule", query_index, smiles, query_record, single_output))
    for group_output in group_outputs:
        records.append(
            _trace_record(
                str(group_output.get("group_id") or "unknown_group"),
                query_index,
                smiles,
                query_record,
                group_output,
            )
        )
    records.append(_trace_record("final_summary", query_index, smiles, query_record, final_output))
    _write_jsonl(path, records)


def _trace_record(
    task: str,
    query_index: int,
    smiles: str,
    query_record: dict[str, Any],
    output: dict[str, Any],
) -> dict[str, Any]:
    llm = output.get("llm") or {}
    content = llm.get("content")
    return {
        "task": task,
        "index": query_index,
        "sample_id": query_index,
        "molecule_key": f"index:{query_index}",
        "smiles": smiles,
        "label": query_record.get("Y"),
        "status": output.get("status"),
        "prediction": content.get("bioavailability_prediction") if isinstance(content, dict) else None,
        "response_text": json.dumps(content, ensure_ascii=False, indent=2) if content is not None else output.get("error"),
        "messages": llm.get("messages") or [],
        "tool_count": len(llm.get("tool_calls") or []),
        "usage": llm.get("usage") or {},
        "raw_output": {
            key: value
            for key, value in output.items()
            if key != "llm"
        },
    }


def _read_jsonl_record(path: Path, index: int) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i == index:
                return json.loads(line)
    raise SystemExit(f"No record at index {index}: {path}")


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
    manifest.setdefault("paths", {})["final_reasoning_output"] = str(final_path)
    manifest.setdefault("paths", {})["trace_messages"] = str(trace_path)
    _write_json(manifest_path, manifest)
    _print_summary(final_output, manifest)
    return 0


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # The explicit --env-file is the run configuration source of truth.
        os.environ[key] = value


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
        choices=["operational", "parent_disjoint"],
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
    parser.add_argument("--groups", nargs="*", default=None, help="Optional exact Tier.endpoint_group ids to reason over.")
    return parser.parse_args(argv)


def _log(message: str) -> None:
    print(f"[bioavailability_reasoning_pipeline] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
