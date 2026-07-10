"""Run BBB Martins analog reasoning with group-level parallel LLM calls."""

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
from tools.chembl_tool.common.export import ensure_dir
from tools.chembl_tool.common.json_utils import parse_json_content
from tools.chembl_tool.common.openai_reasoning_client import OpenAICompatibleClient
from tools.chembl_tool.common.reasoning_validation import (
    call_with_json_validation,
    structured_response_is_valid,
    validated_branch_content,
)
from tools.chembl_tool.common.task_workflows.evidence_library import standardize_smiles_and_fp
from tools.chembl_tool.tasks.bbb_martins.chembl_exact_context import (
    DEFAULT_CHEMBL_SQLITE,
    enrich_retrieval_with_chembl_context,
)
from tools.chembl_tool.tasks.bbb_martins.retrieve_neighbors import load_index, retrieve_neighbors


DEFAULT_INPUT = "data/processed/BBB_Martins/B3DB_cleaned/test/test_efflux.jsonl"
DEFAULT_INDEX = "outputs/chembl_tool/tasks/bbb_martins/evidence_library/bbb_neighbor_index.pkl"
DEFAULT_TIER1_REPLACEMENT_GROUPS = ["Tier 1.starling_direct_bbb_evidence"]
DEFAULT_OUT_ROOT = "outputs/chembl_tool/tasks/bbb_martins/reasoning/single_runs"
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _load_env(Path(args.env_file))
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    if args.resume_final_from_run_dir:
        client = OpenAICompatibleClient(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            timeout_s=args.timeout_s,
            max_tokens=args.max_tokens,
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

    _log("loading neighbor index")
    index = load_index(Path(args.index))
    _log("retrieving top neighbors by group")
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
        )
    if retrieval.get("status") != "ok":
        raise SystemExit(json.dumps(retrieval.get("errors", []), ensure_ascii=False))
    if args.enable_chembl_exact_context:
        _log("enriching retrieval with exact ChEMBL context")
        retrieval = enrich_retrieval_with_chembl_context(
            retrieval,
            index,
            chembl_sqlite=args.chembl_sqlite,
        )
    if args.tier1_replacement_index:
        _log(f"retrieving Tier 1 replacement neighbors: {args.tier1_replacement_index}")
        replacement_index = load_index(Path(args.tier1_replacement_index))
        replacement_retrieval = retrieve_neighbors(
            query_smiles,
            replacement_index,
            top_k_per_group=args.top_k_per_group,
            min_similarity=args.min_similarity,
            groups=_replacement_retrieval_groups(args.groups, args.tier1_replacement_groups),
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

    client = OpenAICompatibleClient(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
        tool_service_url=args.tool_service_url,
        enable_group_tools=not args.disable_group_tools,
        max_tool_rounds=args.max_tool_rounds,
        reasoning_effort=args.reasoning_effort,
        enable_thinking=args.enable_thinking,
    )

    single_output, group_outputs = _run_parallel_reasoning(
        client,
        retrieval,
        groups,
        max_workers=args.max_workers,
    )
    single_path = out_dir / "single_molecule_reasoning_output.json"
    _write_json(single_path, single_output)
    _log(f"wrote {single_path}")

    group_path = out_dir / "group_reasoning_outputs.jsonl"
    _write_jsonl(group_path, group_outputs)
    _log(f"wrote {group_path}")

    final_output = _run_final_reasoning(client, retrieval, single_output, group_outputs)
    final_path = out_dir / "final_reasoning_output.json"
    _write_json(final_path, final_output)
    _log(f"wrote {final_path}")

    trace_path = out_dir / "trace_messages.jsonl"
    _write_trace_jsonl(
        trace_path,
        query_record=query_record,
        query_index=args.query_index,
        smiles=query_smiles,
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
        "model": args.model,
        "base_url": args.base_url,
        "tool_service_url": args.tool_service_url,
        "reasoning_effort": args.reasoning_effort,
        "thinking": {"type": "enabled"} if args.enable_thinking else {"type": "disabled"},
        "neighbor_index": args.index,
        "tier1_replacement_index": args.tier1_replacement_index,
        "tier1_replacement_groups": args.tier1_replacement_groups or DEFAULT_TIER1_REPLACEMENT_GROUPS
        if args.tier1_replacement_index
        else [],
        "groups": args.groups or [],
        "retrieval_evidence_source": retrieval.get("evidence_source", {}),
        "group_tools_enabled": not args.disable_group_tools,
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
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    outputs: list[dict[str, Any]] = []
    single_output: dict[str, Any] | None = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_reason_one_group, client, _llm_query_payload(retrieval["query"]), group): group["group_id"]
            for group in groups
        }
        futures[
            executor.submit(
                _reason_single_molecule,
                client,
                _llm_query_payload(retrieval["query"]),
                _clean_query_chembl_context(retrieval.get("query_chembl_context") or {}),
            )
        ] = "single_molecule"
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
        "status": "ok" if structured_response_is_valid(response) else "error",
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
) -> dict[str, Any]:
    instructions = [
        "Call molecule_properties for the query molecule before analysis.",
        "Assess passive BBB plausibility from molecular weight, logP/logD, TPSA, HBD/HBA, ionization/pKa, charge, rotatable bonds, and functional groups.",
        "Return JSON with passive_bbb_plausibility, efflux_or_transporter_prior, confidence, reasoning_summary, property_drivers, caveats.",
    ]
    payload: dict[str, Any] = {
        "task": "Single-molecule BBB plausibility analysis.",
        "query": query,
        "instructions": instructions,
        "required_json_schema": {
            "passive_bbb_plausibility": "high | moderate | low | uncertain",
            "efflux_or_transporter_prior": "high | moderate | low | uncertain",
            "exact_chembl_evidence_assessment": "string",
            "confidence": "high | moderate | low",
            "reasoning_summary": "string",
            "property_drivers": ["string"],
            "caveats": ["string"],
        },
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
                "You may call exactly one tool: molecule_properties. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]
    response = call_with_json_validation(
        lambda retry_messages: client.chat_json_with_tools(
            retry_messages,
            tools=SINGLE_MOLECULE_TOOLS,
            allowed_tool_names={"molecule_properties"},
            first_tool_choice="auto",
        ),
        messages,
        required_fields=("confidence", "reasoning_summary"),
        branch_name="single-molecule",
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
                "You are a medicinal chemistry BBB analog evidence analyst. "
                "Reason about whether analog evidence in one endpoint group is transferable to the query molecule. "
                "You may call the provided molecule comparison tools when structural or property differences matter. "
                "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(_group_prompt_payload(query, group), ensure_ascii=False),
        },
    ]
    response = call_with_json_validation(
        lambda retry_messages: client.chat_json_with_optional_tools(
            retry_messages,
            tools=GROUP_REASONING_TOOLS,
            allowed_tool_names={"properties_compare", "mmp_structure_compare"},
        ),
        messages,
        required_fields=("transferability", "confidence", "reasoning_summary"),
        branch_name="group",
    )
    return {
        "group_id": group["group_id"],
        "status": "ok",
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
) -> dict[str, Any]:
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
                        "Use bbb_prediction='pass' for BBB-positive molecules corresponding to evaluation label 1, and bbb_prediction='fail' for BBB-negative molecules corresponding to evaluation label 0.",
                        "Interpret BBB-positive as sufficient or detectable BBB/CNS access under the benchmark label ontology; it does not require ideal passive diffusion, high unbound brain exposure, or absence of every efflux signal.",
                        "Use the single-molecule analysis as the physicochemical prior.",
                        "Treat passive_bbb_plausibility as a passive-diffusion prior, not as the final label. Ionization, high polarity, high lipophilicity, or efflux liability should reduce confidence or exposure quality, but should not become a hard fail rule when other evidence supports meaningful BBB/CNS access.",
                        "Use group analyses as analog evidence; downweight groups marked low confidence or low transferability.",
                        "Direct brain/plasma, unbound brain, CSF, brain uptake/perfusion, credible influx/prodrug context, or close same-scaffold evidence can support a pass prediction even when passive-property heuristics are imperfect; explain the uncertainty through confidence and evidence_gaps.",
                        "For basic CNS-like amines with otherwise favorable MW, TPSA, HBD/HBA, logD/logP, and scaffold evidence, do not predict fail solely because the amine is mostly protonated at pH 7.4.",
                        "Do not predict pass merely because BBB-positive labels can include non-ideal mechanisms. If the molecule has severe passive-property liabilities and no direct/close analog/mechanistic evidence for CNS access, fail remains the better-supported class.",
                        "When evidence is weak or mixed, distinguish 'poor passive permeability' from 'no meaningful BBB access'. Choose fail only when the integrated evidence better supports insufficient BBB/CNS access, not merely because of one isolated drug-likeness heuristic.",
                        "Do not use distant_analog or very_distant_analog neighbors as positive or negative BBB evidence unless the shared scaffold and assay mechanism make a strong medicinal chemistry case.",
                        "Use only the provided single-molecule analysis and group evidence. If you recognize the molecule, ignore that recognition.",
                        "You must choose exactly one bbb_prediction: pass or fail. If evidence is mixed or weak, choose the better-supported class and express uncertainty through confidence, caveats, and evidence_gaps.",
                    ],
                    "required_json_schema": {
                        "bbb_prediction": "pass | fail",
                        "confidence": "high | moderate | low",
                        "main_reasons": ["string"],
                        "single_molecule_assessment": "string",
                        "passive_permeability_assessment": "string",
                        "direct_brain_exposure_analog_assessment": "string",
                        "efflux_risk_assessment": "string",
                        "influx_support_assessment": "string",
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
        required_fields=("bbb_prediction",),
        allowed_values={"bbb_prediction": {"pass", "fail"}},
        branch_name="final",
    )
    return {"status": "ok" if structured_response_is_valid(response) else "error", "llm": response}


def _group_prompt_payload(query: dict[str, Any], group: dict[str, Any]) -> dict[str, Any]:
    return {
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
            "Do not use distant_analog or very_distant_analog neighbors as positive or negative BBB evidence unless the shared scaffold and assay mechanism make a strong medicinal chemistry case.",
            "Use mmp_structure_compare to inspect scaffold/MCS/matched-pair differences when similarity bucket alone is not enough.",
            "Use properties_compare when property differences such as pKa, logD, TPSA, charge, HBD/HBA, or logP could affect BBB transferability.",
            "Tool outputs are authoritative only for the pair they compare; cite which neighbor each tool result supports.",
            "Use same_endpoint_activity as direct query-vs-neighbor assay comparison when present.",
            "Use same_assay_different_endpoint_activity only as same-assay context; do not directly compare numeric values across different endpoints.",
            "Distinguish direct BBB exposure, passive permeability, efflux substrate risk, influx support, and weak inhibition/binding evidence.",
            "Do not convert transporter IC50/inhibition directly into substrate/transport unless assay context supports it.",
            "Return key_evidence as structured evidence cards, not a plain list of molecule ids.",
            "For each key_evidence item, derive assay_signal and activity_values from the provided evidence_rows, derive tool_summary from tool outputs, and judge transferability/effect_on_bbb_reasoning yourself.",
            "Return JSON with useful_for_bbb_reasoning, transferability, evidence_direction, confidence, reasoning_summary, key_evidence, caveats.",
        ],
        "required_json_schema": {
            "useful_for_bbb_reasoning": "boolean",
            "transferability": "high | moderate | low | not_applicable",
            "evidence_direction": (
                "supports_bbb_crossing | argues_against_bbb_crossing | efflux_risk | "
                "influx_support | neutral_or_unclear"
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
                    "effect_on_bbb_reasoning": "string",
                }
            ],
            "caveats": ["string"],
        },
    }


def _llm_query_payload(query: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_smiles": query.get("input_smiles", ""),
        "canonical_smiles": query.get("canonical_smiles", ""),
    }


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
        "prediction": content.get("bbb_prediction") if isinstance(content, dict) else None,
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
    final_output = _run_final_reasoning(client, retrieval, single_output, group_outputs)
    final_path = run_dir / "final_reasoning_output.json"
    _write_json(final_path, final_output)
    trace_path = run_dir / "trace_messages.jsonl"
    query_record = {"Y": manifest.get("query_label_for_eval_only")}
    query_index = int(manifest.get("query_index") or 0)
    smiles = str((retrieval.get("query") or {}).get("input_smiles") or "")
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
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--tool-service-url", default=DEFAULT_TOOL_SERVICE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--max-groups", type=int, default=0, help="Debug limit; 0 means all groups with neighbors.")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=4096)
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
