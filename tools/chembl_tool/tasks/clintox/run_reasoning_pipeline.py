"""Run ClinTox analog reasoning with group-level parallel LLM calls."""

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

from tools.chembl_tool.common.evidence_contract import evidence_for_group_llm, evidence_for_llm
from tools.chembl_tool.common.coverage_reasoning import (
    NEIGHBOR_CONTEXT_PROFILES,
    STANDARD_NEIGHBOR_CONTEXT,
)
from tools.chembl_tool.common.experiment_retrieval import EXPERIMENT_MODES, retrieve_experiment_view
from tools.chembl_tool.common.export import ensure_dir
from tools.chembl_tool.common.identity_blind import (
    prepare_identity_blind_final_retrieval,
    prepare_prefetched_final_retrieval,
    prepare_reasoning_retrieval,
    sanitize_identity_blind_branch_outputs,
)
from tools.chembl_tool.common.neighbor_selection import (
    NEIGHBOR_SELECTORS,
    SIMILARITY_SELECTOR,
)
from tools.chembl_tool.common.retrieval_policy import NEIGHBOR_IDENTITY_POLICIES
from tools.chembl_tool.common.json_utils import parse_json_content
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
from tools.chembl_tool.tasks.clintox.chembl_exact_context import (
    DEFAULT_CHEMBL_SQLITE,
    enrich_retrieval_with_chembl_context,
)
from tools.chembl_tool.tasks.clintox.constants import (
    CLINTOX_NEGATIVE_PREDICTION,
    CLINTOX_POSITIVE_PREDICTION,
)
from tools.chembl_tool.tasks.clintox.experiment_config import get_source_config
from tools.chembl_tool.tasks.clintox.prompt_profiles import (
    CLINTOX_PROMPT_PROFILES,
    DEFAULT_CLINTOX_PROMPT_PROFILE,
    get_clintox_prompt_profile,
)
from tools.chembl_tool.tasks.clintox.retrieve_neighbors import load_index


DEFAULT_INPUT = (
    "data/processed_clintox_clinical_trial_failure_v1/ClinTox/scaffold/test.jsonl"
)
DEFAULT_INDEX = (
    "outputs/paper/molecular_evidence_agent_starling_scaffold_"
    "clinical_trial_failure_v1/evidence/clintox_starling_full/"
    "starling_clintox_neighbor_index.pkl"
)
DEFAULT_OUT_ROOT = (
    "outputs/chembl_tool/tasks/clintox/reasoning/clinical_trial_failure_v1/single_runs"
)
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEFAULT_BASE_URL = "http://127.0.0.1:50001/v1"
DEFAULT_TOOL_SERVICE_URL = "http://127.0.0.1:8765"
_write_trace_jsonl = partial(write_trace_jsonl, prediction_field="clintox_prediction")


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
                "features. Use this to assess whether property changes affect clinical toxicity evidence transferability."
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

    run_id = args.run_id or time.strftime("clintox_reasoning_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
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
    if args.groups and args.experiment_mode == "native":
        selected_groups = set(args.groups)
        groups = [group for group in groups if group.get("group_id") in selected_groups]
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
        prompt_profile=args.clintox_prompt_profile,
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
        prompt_profile=args.clintox_prompt_profile,
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
        "task_prompt_profile": args.clintox_prompt_profile,
        "label_scope": get_clintox_prompt_profile(args.clintox_prompt_profile).label_scope,
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
    prompt_profile: str = DEFAULT_CLINTOX_PROMPT_PROFILE,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    outputs = list(group_outputs or [])
    reused_group_ids = {str(output.get("group_id") or "") for output in outputs}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _reason_one_group,
                client,
                _llm_evidence_query_payload(retrieval["query"]),
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
    prompt_profile: str = DEFAULT_CLINTOX_PROMPT_PROFILE,
) -> dict[str, Any]:
    # Keep the shared stage shape fixed while the canonical profile supplies
    # the task ontology shown to the model.
    profile = get_clintox_prompt_profile(prompt_profile)
    instructions = [
        "Call molecule_properties for the query molecule before analysis.",
        "Assess only a physicochemical toxicity-risk prior from molecular weight, logP/logD, TPSA, HBD/HBA, ionization/pKa, charge, rotatable bonds, electrophilic/reactive functional groups, cationic amphiphilicity, and structural alerts.",
        "Molecular properties cannot establish that a clinical trial or development program failed because of toxicity.",
        "Return JSON with clinical_toxicity_prior, physicochemical_risk_prior, reactive_or_structural_alert_prior, exposure_accumulation_prior, confidence, reasoning_summary, property_drivers, caveats.",
    ]
    if query.get("prefetched_molecule_properties"):
        instructions[0] = "Use the harness-prefetched molecule_properties result."
        if query.get("identity_hidden"):
            instructions[0] += " Do not identify or name the query."
    payload: dict[str, Any] = {
        "task": f"Single-molecule prior for predicting {profile.prediction_target}.",
        "query": query,
        "instructions": instructions,
        "required_json_schema": {
            "clinical_toxicity_prior": "high_risk | low_risk | mixed_or_unclear",
            "physicochemical_risk_prior": "favorable | concerning | mixed_or_unclear",
            "reactive_or_structural_alert_prior": "concerning | not_apparent | mixed_or_unclear",
            "exposure_accumulation_prior": "concerning | not_apparent | mixed_or_unclear",
            "exact_chembl_evidence_assessment": "string",
            "confidence": "high | moderate | low",
            "reasoning_summary": "string",
            "property_drivers": ["string"],
            "caveats": ["string"],
        },
    }
    if chembl_context:
        instructions.append(
            "If exact_query_chembl_context is found, distinguish direct same-molecule ChEMBL clintox evidence from the physicochemical prior."
        )
        payload["exact_query_chembl_context"] = chembl_context
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a medicinal chemistry analyst estimating a property-only prior for {profile.prediction_target}. "
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
    prompt_profile: str = DEFAULT_CLINTOX_PROMPT_PROFILE,
) -> dict[str, Any]:
    profile = get_clintox_prompt_profile(prompt_profile)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a medicinal chemistry ClinTox analog evidence analyst. "
                f"Reason about whether one evidence family's analog findings transfer to the query molecule and whether they bear on {profile.prediction_target}. "
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
                build_group_prompt_payload(query, group, prompt_profile=prompt_profile),
                ensure_ascii=False,
            ),
        },
    ]
    response = call_group_branch(
        client,
        messages,
        group=group,
        tools=GROUP_REASONING_TOOLS,
        required_fields=(
            "transferability",
            "confidence",
            "reasoning_summary",
            "evidence_direction",
            "direct_evidence_status",
        ),
        content_validator=lambda content: _group_provenance_validation_errors(
            content, group
        ),
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
    prompt_profile: str = DEFAULT_CLINTOX_PROMPT_PROFILE,
) -> dict[str, Any]:
    profile = get_clintox_prompt_profile(prompt_profile)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior drug-development safety reasoning model. "
                + profile.final_system_instruction
                + " "
                "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": profile.final_task_instruction,
                    "query": _llm_evidence_query_payload(retrieval["query"]),
                    "retrieval_coverage": retrieval["coverage"],
                    "single_molecule_analysis": {
                        "status": single_output.get("status"),
                        "content": validated_branch_content(single_output),
                    },
                    "group_reasoning_outputs": [
                        {
                            "group_id": item.get("group_id"),
                            "status": item.get("status"),
                            "content": validated_branch_content(item),
                        }
                        for item in group_outputs
                    ],
                    "instructions": [
                        "Return compact complete JSON.",
                        profile.class_definition_instruction,
                        "Do not reinterpret the target as whether the molecule has any toxicity, adverse event, organ injury, safety liability, narrow therapeutic index, or monitoring requirement.",
                        "Use the single-molecule analysis only as a physicochemical plausibility prior; it cannot by itself determine clintox_prediction.",
                        "Use group analyses as analog evidence; downweight groups marked low confidence or low transferability.",
                        "Patient-level dose reduction, treatment discontinuation, DLT, MTD, severe adverse events, organ injury, FDA market withdrawal, regulatory restriction, or approval status are proximal/contextual evidence but are not themselves the target event unless the provided Direct.clinical_trial_failure evidence explicitly states the required trial/development failure.",
                        "Animal toxicity, organ toxicity, genotoxicity/carcinogenicity, cellular stress, general cytotoxicity, hERG/ion-channel, CYP, transporter, DDI, exposure, and physicochemical signals are predictive risk evidence, not observations of the target event.",
                        "FDA approval and ordinary tolerability are contextual evidence, not proof that a toxicity-failed trial never occurred; approval and a failed trial can coexist.",
                        "Absence of toxicity in one assay, study, or paper is local negative evidence, not proof of the global negative class.",
                        *profile.final_policy_instructions,
                        "Do not use distant_analog or very_distant_analog neighbors as positive or negative trial-failure evidence unless a strong shared scaffold and development-limiting mechanism justify transfer.",
                        "Use only the provided single-molecule analysis and group evidence. If you recognize the molecule or therapeutic class, ignore that recognition.",
                        f"You must choose exactly one clintox_prediction: {CLINTOX_POSITIVE_PREDICTION} or {CLINTOX_NEGATIVE_PREDICTION}. If evidence is mixed or weak, choose the better-supported class and express uncertainty through confidence, caveats, and evidence_gaps.",
                    ],
                    "required_json_schema": {
                        "clintox_prediction": f"{CLINTOX_POSITIVE_PREDICTION} | {CLINTOX_NEGATIVE_PREDICTION}",
                        "confidence": "high | moderate | low",
                        "main_reasons": ["string"],
                        "single_molecule_assessment": "string",
                        "clinical_or_human_safety_assessment": "string",
                        "direct_trial_failure_anchor_assessment": "string",
                        "direct_evidence_status": (
                            "direct_analog_retrieved_transferable | "
                            "direct_analog_retrieved_not_transferable | "
                            "no_direct_analog_retrieved"
                        ),
                        "in_vivo_toxicology_assessment": "string",
                        "organ_safety_pharmacology_assessment": "string",
                        "genotoxicity_or_carcinogenicity_assessment": "string",
                        "cell_stress_and_cytotoxicity_assessment": "string",
                        "offtarget_ddi_or_exposure_assessment": "string",
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
        required_fields=("clintox_prediction", "direct_evidence_status"),
        allowed_values={
            "clintox_prediction": {
                CLINTOX_POSITIVE_PREDICTION,
                CLINTOX_NEGATIVE_PREDICTION,
            },
            "direct_evidence_status": {
                "direct_analog_retrieved_transferable",
                "direct_analog_retrieved_not_transferable",
                "no_direct_analog_retrieved",
            },
        },
        content_validator=lambda content: _final_provenance_validation_errors(
            content,
            retrieval,
            group_outputs,
        ),
        branch_name="final",
    )
    return {"status": "ok" if structured_response_is_valid(response) else "error", "llm": response}


def build_group_prompt_payload(
    query: dict[str, Any],
    group: dict[str, Any],
    *,
    prompt_profile: str = DEFAULT_CLINTOX_PROMPT_PROFILE,
) -> dict[str, Any]:
    profile = get_clintox_prompt_profile(prompt_profile)
    direction_values = (
        "supports_higher_trial_failure_risk | supports_lower_trial_failure_risk | "
    )
    return bound_group_prompt_payload({
        "task": f"Group-level analog transferability analysis for predicting {profile.prediction_target}.",
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
                "evidence_rows": [
                    evidence_for_group_llm(row, group)
                    for row in neighbor["evidence_rows"]
                ],
                "shared_assay_context": _clean_shared_assay_context(neighbor.get("shared_assay_context") or {}),
            }
            for neighbor in group["neighbors"]
        ],
        "instructions": [
            "Use only this group's evidence.",
            "Each evidence_rows item follows minimal_evidence.v1; read endpoint/measurement, text, annotations, quality, provenance, and examples without assuming a source-specific schema.",
            "Assess structural transferability from neighbors to the query.",
            "Low-similarity analogs are intentionally included. You must explicitly judge whether they are transferable.",
            "Do not use distant_analog or very_distant_analog neighbors as positive or negative trial-failure evidence unless the shared scaffold and development-limiting mechanism make a strong medicinal chemistry case.",
            "Use mmp_structure_compare to inspect scaffold/MCS/matched-pair differences when similarity bucket alone is not enough.",
            "Use properties_compare when property differences such as pKa, logD, TPSA, charge, HBD/HBA, logP, molecular size, or polarity could affect clinical toxicity transferability.",
            "Tool outputs are authoritative only for the pair they compare; cite which neighbor each tool result supports.",
            "Use same_endpoint_activity as direct query-vs-neighbor assay comparison when present.",
            "Use same_assay_different_endpoint_activity only as same-assay context; do not directly compare numeric values across different endpoints.",
            "Distinguish literal trial/development failure, proximal clinical safety, in vivo animal toxicology, organ toxicity, genotoxicity/carcinogenicity, cellular stress, general cytotoxicity, and safety-relevant off-target/DDI evidence.",
            "Only evidence explicitly stating that a trial or development program stopped, terminated, suspended, or was withdrawn because of toxicity is direct evidence for the target event.",
            *profile.group_policy_instructions,
            "Patient treatment discontinuation, dose reduction, DLT/MTD, serious adverse events, approval status, and ordinary tolerability are proximal or contextual evidence, not direct trial-failure outcomes.",
            "FDA market withdrawal, postmarketing restriction, or regulatory suspension is not a clinical-trial/development failure unless the provided evidence also explicitly states that the clinical trial or development program stopped because of toxicity.",
            "Do not convert generic inhibition, activity, growth, viability, or ratio endpoints into clinical toxicity without the assay context, target, endpoint, cell type, species, dose, route, and duration supporting that interpretation.",
            "For safety-relevant off-target, receptor-binding, ion-channel, CYP, transporter, or DDI/exposure groups, distinguish target engagement from clinical toxicity. Binding or inhibition alone is usually a liability signal, not supports_clinical_toxicity, unless this group contains direct severe clinical or in vivo toxicity evidence tied to that mechanism.",
            "For hERG/QT, 5-HT2B, AChE, GABA/NMDA, sodium/calcium-channel, CYP, and transporter evidence, describe potency and transferability, but avoid upgrading to clinical toxicity solely from pharmacology or monitoring liability.",
            "For animal LD50/TD50/MTD/NOAEL evidence, report species, route, dose, duration, and endpoint severity, but do not convert it into a clinical-trial-failure event.",
            "For generic cytotoxicity or cell-viability evidence, distinguish intended efficacy, nonspecific cell stress, and safety cytotoxicity; it can explain risk but cannot establish trial failure.",
            "Treat inactive/not toxic/no effect comments as evidence against that specific assay liability, not as proof that no toxicity-failed trial occurred.",
            "Do not convert CYP IC50/inhibition, hERG, transporter, receptor, DDI, or exposure evidence directly into trial failure.",
            "Return key_evidence as structured evidence cards, not a plain list of molecule ids.",
            "For each key_evidence item, derive assay_signal and activity_values from the provided evidence_rows, derive tool_summary from tool outputs, and judge transferability/effect_on_clintox_reasoning yourself.",
            "Return JSON with useful_for_clintox_reasoning, transferability, evidence_direction, confidence, reasoning_summary, key_evidence, caveats.",
        ],
        "required_json_schema": {
            "useful_for_clintox_reasoning": "boolean",
            "transferability": "high | moderate | low | not_applicable",
            "evidence_direction": (
                direction_values + "clinical_toxicity_risk | cardiotoxicity_risk | "
                "hepatotoxicity_risk | nephrotoxicity_risk | neurotoxicity_risk | "
                "genotoxicity_or_carcinogenicity_risk | mitochondrial_or_cell_stress_risk | "
                "general_cytotoxicity_risk | drug_interaction_or_exposure_risk | mechanistic_context | "
                "neutral_or_unclear"
            ),
            "direct_evidence_status": (
                "direct_rows_present_transferable | "
                "direct_rows_present_not_transferable | no_direct_rows"
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
                    "effect_on_clintox_reasoning": "string",
                    "source_evidence_group_id": "string",
                    "evidence_claim_type": (
                        "direct_observed_outcome | contextual_clinical_risk | mechanistic_risk"
                    ),
                }
            ],
            "caveats": ["string"],
        },
    }, evidence_prompt_profile=str(group.get("evidence_prompt_profile") or ""))


# Preserve the task-pipeline adapter contract used by shared materializers.
_group_prompt_payload = build_group_prompt_payload


def _group_has_direct_evidence(group: dict[str, Any]) -> bool:
    """Return true only when this concrete retrieval group contains a strict direct row."""
    for neighbor in group.get("neighbors") or []:
        for evidence in neighbor.get("evidence_rows") or []:
            minimal_group = (((evidence.get("minimal_evidence") or {}).get("group") or {}).get("id"))
            if minimal_group == "Direct.clinical_trial_failure" or evidence.get("group_id") == "Direct.clinical_trial_failure":
                return True
    return False


def _group_provenance_validation_errors(
    content: dict[str, Any],
    group: dict[str, Any],
) -> list[str]:
    """Validate direct-outcome claims without constraining the predicted direction."""
    status = str(content.get("direct_evidence_status") or "")
    has_direct = _group_has_direct_evidence(group)
    if not has_direct and status != "no_direct_rows":
        return [
            "direct_evidence_status must be no_direct_rows because this group contains no "
            "Direct.clinical_trial_failure evidence row"
        ]
    if has_direct and status == "no_direct_rows":
        return [
            "direct_evidence_status cannot be no_direct_rows because this group contains a "
            "Direct.clinical_trial_failure evidence row; report whether it is transferable"
        ]
    return []


def _final_provenance_validation_errors(
    content: dict[str, Any],
    retrieval: dict[str, Any],
    group_outputs: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Keep final direct-evidence provenance separate from label eligibility."""
    status = str(content.get("direct_evidence_status") or "")
    has_direct = any(
        _group_has_direct_evidence(group)
        for group in retrieval.get("groups") or []
    )
    if not has_direct and status != "no_direct_analog_retrieved":
        return [
            "direct_evidence_status must be no_direct_analog_retrieved because retrieval "
            "contains no Direct.clinical_trial_failure row"
        ]
    if has_direct and status == "no_direct_analog_retrieved":
        return [
            "direct_evidence_status cannot be no_direct_analog_retrieved because retrieval "
            "contains a Direct.clinical_trial_failure row; report its transferability"
        ]
    if not has_direct:
        return []

    group_statuses = {
        str(((output.get("llm") or {}).get("content") or {}).get("direct_evidence_status") or "")
        for output in group_outputs or []
    }
    group_statuses.discard("")
    if (
        status == "direct_analog_retrieved_transferable"
        and group_statuses
        and "direct_rows_present_transferable" not in group_statuses
    ):
        return [
            "final direct_evidence_status claims a transferable direct analog but no group "
            "reported direct_rows_present_transferable"
        ]
    if (
        status == "direct_analog_retrieved_not_transferable"
        and "direct_rows_present_transferable" in group_statuses
    ):
        return [
            "final direct_evidence_status says direct analogs are not transferable but a group "
            "reported direct_rows_present_transferable"
        ]
    return []


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
        "clintox_relevant_evidence_rows": [
            _clean_evidence_row(row) for row in context.get("clintox_relevant_evidence_rows", [])
        ],
    }


def _parse_json_content(content: str) -> Any:
    return parse_json_content(content)


def _validate_prompt_profile_reuse(args: argparse.Namespace) -> None:
    """Reject single/group branch reuse across ClinTox prompt contracts."""
    require_matching_prompt_profiles(
        target_profile=str(args.clintox_prompt_profile),
        source_dirs=(
            args.single_analysis_source_run_dir,
            args.group_analysis_source_run_dir,
        ),
        historical_profile="",
    )


def _manifest_prompt_profile(manifest: dict[str, Any]) -> str:
    return prompt_profile_from_manifest(
        manifest,
        historical_profile="",
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
    if manifest.get("identity_blind"):
        group_outputs = sanitize_identity_blind_branch_outputs(group_outputs, retrieval)
        retrieval = prepare_identity_blind_final_retrieval(retrieval, single_output)
    elif manifest.get("harness_prefetch_tools"):
        retrieval = prepare_prefetched_final_retrieval(retrieval, single_output, identity_blind=False)
    final_output = _run_final_reasoning(
        client,
        retrieval,
        single_output,
        group_outputs,
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
    manifest["label_scope"] = get_clintox_prompt_profile(prompt_profile).label_scope
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
    parser.add_argument("--api-key-env", default="CLINTOX_LOCAL_API_KEY")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--tool-service-url", default=DEFAULT_TOOL_SERVICE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--groups", nargs="*", default=None, help="Optional exact Tier.endpoint_group ids to reason over.")
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
    parser.add_argument(
        "--clintox-prompt-profile",
        choices=CLINTOX_PROMPT_PROFILES,
        default=DEFAULT_CLINTOX_PROMPT_PROFILE,
    )
    return parser.parse_args(argv)


def _log(message: str) -> None:
    print(f"[clintox_reasoning_pipeline] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
