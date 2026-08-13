"""Run DILI analog reasoning with group-level parallel LLM calls."""

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
from tools.chembl_tool.common.export import ensure_dir
from tools.chembl_tool.common.json_utils import parse_json_content
from tools.chembl_tool.common.neighbor_selection import (
    NEIGHBOR_SELECTORS,
    SIMILARITY_SELECTOR,
)
from tools.chembl_tool.common.openai_reasoning_client import OpenAICompatibleClient
from tools.chembl_tool.common.reasoning_payload import (
    clean_exact_match as _clean_exact_match,
    clean_shared_assay_context as _clean_shared_assay_context,
    load_env_file as _load_env,
    read_jsonl_record as _read_jsonl_record,
    write_trace_jsonl,
)
from tools.chembl_tool.common.reasoning_validation import (
    call_with_json_validation,
    structured_response_is_valid,
    validated_branch_content,
)
from tools.chembl_tool.tasks.dili.chembl_exact_context import (
    DEFAULT_CHEMBL_SQLITE,
    enrich_retrieval_with_chembl_context,
)
from tools.chembl_tool.tasks.dili.constants import (
    DILI_NEGATIVE_PREDICTION,
    DILI_POSITIVE_PREDICTION,
)
from tools.chembl_tool.tasks.dili.retrieve_neighbors import load_index, retrieve_neighbors


DEFAULT_INPUT = "data/processed/DILI/test.jsonl"
DEFAULT_INDEX = "outputs/chembl_tool/tasks/dili/evidence_library/dili_neighbor_index.pkl"
DEFAULT_OUT_ROOT = "outputs/chembl_tool/tasks/dili/reasoning/single_runs"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TOOL_SERVICE_URL = "http://127.0.0.1:8765"
_write_trace_jsonl = partial(write_trace_jsonl, prediction_field="dili_prediction")


GROUP_REASONING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mmp_structure_compare",
            "description": (
                "Compare the query molecule to one neighbor using Morgan Tanimoto, MCS coverage, "
                "and mmpdb matched-pair transformation. Use this to judge DILI analog transferability."
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
                "features. Use this when solubility, ionization, lipophilicity, size, polarity, or charge could "
                "change DILI evidence transferability."
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

    run_id = args.run_id or time.strftime("dili_reasoning_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    out_dir = ensure_dir(Path(args.out_root) / run_id)
    _log(f"run_id={run_id}")

    query_record = _read_jsonl_record(Path(args.input_jsonl), args.query_index)
    query_smiles = str(query_record.get(args.smiles_field) or "")
    if not query_smiles:
        raise SystemExit(f"Input record has no `{args.smiles_field}` value.")

    _log("loading neighbor index")
    index = load_index(Path(args.index))
    _log("retrieving top neighbors by group")
    retrieval = retrieve_neighbors(
        query_smiles,
        index,
        top_k_per_group=args.top_k_per_group,
        min_similarity=args.min_similarity,
        groups=args.groups,
        neighbor_selector=args.morgan_neighbor_selector,
    )
    if retrieval.get("status") != "ok":
        raise SystemExit(json.dumps(retrieval.get("errors", []), ensure_ascii=False))
    if args.enable_chembl_exact_context:
        _log("enriching retrieval with exact ChEMBL context")
        retrieval = enrich_retrieval_with_chembl_context(retrieval, index, chembl_sqlite=args.chembl_sqlite)

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
        "group_tools_enabled": not args.disable_group_tools,
        "chembl_exact_context_enabled": args.enable_chembl_exact_context,
        "chembl_sqlite": args.chembl_sqlite,
        "group_tool_names": [tool["function"]["name"] for tool in GROUP_REASONING_TOOLS]
        if not args.disable_group_tools
        else [],
        "max_tool_rounds": args.max_tool_rounds,
        "top_k_per_group": args.top_k_per_group,
        "min_similarity": args.min_similarity,
        "morgan_neighbor_selector": args.morgan_neighbor_selector,
        "groups": args.groups,
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
    query = _llm_query_payload(retrieval["query"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_reason_one_group, client, query, group): group["group_id"]
            for group in groups
        }
        futures[
            executor.submit(
                _reason_single_molecule,
                client,
                query,
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


def _reason_single_molecule(
    client: OpenAICompatibleClient,
    query: dict[str, Any],
    chembl_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    instructions = [
        "Call molecule_properties for the query molecule before analysis.",
        "Assess only molecule-intrinsic plausibility for DILI. Do not use analog evidence here.",
        "Consider daily-dose-like exposure plausibility only qualitatively from molecular properties; you do not know the real dose.",
        "Consider lipophilicity/logD, ionization and cationic amphiphilicity, TPSA, HBD/HBA, molecular size, aromaticity, reactive/electrophilic motifs, acyl glucuronide-like acid motifs, quinone-imine/anilide/phenol redox motifs, mitochondrial accumulation potential, and poor-solubility/high-exposure liabilities.",
        "This single-molecule prior can support or weaken the final assessment, but it cannot by itself decide DILI risk.",
        "Return compact JSON with the requested fields.",
    ]
    payload: dict[str, Any] = {
        "task": "Single-molecule DILI plausibility analysis.",
        "query": query,
        "instructions": instructions,
        "required_json_schema": {
            "dili_intrinsic_prior": "high_risk | low_risk | mixed_or_unclear",
            "physicochemical_exposure_prior": "concerning | not_apparent | mixed_or_unclear",
            "reactive_metabolite_prior": "concerning | not_apparent | mixed_or_unclear",
            "mitochondrial_or_organelle_prior": "concerning | not_apparent | mixed_or_unclear",
            "cholestasis_property_prior": "concerning | not_apparent | mixed_or_unclear",
            "exact_chembl_evidence_assessment": "string",
            "confidence": "high | moderate | low",
            "reasoning_summary": "string",
            "property_drivers": ["string"],
            "caveats": ["string"],
        },
    }
    if chembl_context:
        instructions.append(
            "If exact_query_chembl_context is found, distinguish same-molecule ChEMBL DILI evidence from the intrinsic property prior."
        )
        payload["exact_query_chembl_context"] = chembl_context
    messages = [
        {
            "role": "system",
            "content": (
                "You are a medicinal chemistry DILI single-molecule analyst. "
                "Only analyze the query molecule itself, without analog evidence. "
                "You may call exactly one tool: molecule_properties. Return only valid JSON."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
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
                "You are a medicinal chemistry DILI analog evidence analyst. "
                "Reason about whether analog evidence in one DILI Tier.endpoint_group transfers to the query molecule. "
                "Use molecule comparison tools when structural or property differences matter. Return only valid JSON."
            ),
        },
        {"role": "user", "content": json.dumps(_group_prompt_payload(query, group), ensure_ascii=False)},
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
) -> dict[str, Any]:
    payload = {
        "task": "Final DILI prediction from intrinsic prior and analog evidence.",
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
        "instructions": _final_instructions(),
        "required_json_schema": {
            "dili_prediction": f"{DILI_POSITIVE_PREDICTION} | {DILI_NEGATIVE_PREDICTION}",
            "confidence": "high | moderate | low",
            "main_reasons": ["string"],
            "single_molecule_assessment": "string",
            "human_or_clinical_dili_assessment": "string",
            "in_vivo_liver_injury_assessment": "string",
            "cholestasis_transporter_assessment": "string",
            "mitochondrial_organelle_stress_assessment": "string",
            "reactive_metabolite_bioactivation_assessment": "string",
            "hepatic_cell_injury_assessment": "string",
            "conflicting_evidence": ["string"],
            "evidence_gaps": ["string"],
            "final_summary": "string",
        },
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior DILI reasoning model. Integrate intrinsic molecular prior and analog evidence "
                "into one final drug-induced liver injury risk prediction. Return only valid JSON."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    response = call_with_json_validation(
        client.chat_json,
        messages,
        required_fields=("dili_prediction",),
        allowed_values={"dili_prediction": {DILI_POSITIVE_PREDICTION, DILI_NEGATIVE_PREDICTION}},
        branch_name="final",
    )
    return {"status": "ok" if structured_response_is_valid(response) else "error", "llm": response}


def _final_instructions() -> list[str]:
    return [
        "Return compact complete JSON.",
        f"Use dili_prediction='{DILI_POSITIVE_PREDICTION}' for TDC DILI-positive molecules corresponding to evaluation label 1.",
        f"Use dili_prediction='{DILI_NEGATIVE_PREDICTION}' for TDC DILI-negative molecules corresponding to evaluation label 0.",
        "DILI is a human clinical liver-injury phenotype. Do not treat generic cytotoxicity, generic CYP inhibition, generic transporter inhibition, or weak structural alerts as sufficient by themselves.",
        "Tier 1 direct human/clinical DILI evidence is strongest. Severe liver outcome, Hy's-law-like signal, withdrawal/warning liver signal, or clear human DILI analog evidence can drive dili_risk when transferability is high.",
        "Tier 2 in vivo liver injury evidence can support dili_risk when liver-specific and transferable. Hepatic necrosis, ALT/AST/ALP/bilirubin/bile-acid changes, or repeated-dose liver pathology are stronger than liver-weight-only findings.",
        "Liver-weight-only evidence is moderate supporting evidence, not a decisive positive anchor unless it is close-transferable and corroborated by stronger liver injury or mechanistic evidence.",
        "Tier 3 cholestasis/hepatobiliary transporter evidence supports DILI risk when it involves BSEP, MRP2, MDR3, NTCP, OATP, bile-acid accumulation, or cholestasis with credible potency and transferability.",
        "Tier 4 mitochondrial, oxidative, ER, lysosomal, or lipid stress supports DILI risk when hepatic/organelle injury is clear. GSH content/elevation without depletion/ROS/oxidative-stress direction is weak contextual evidence.",
        "Tier 5 reactive metabolite, covalent binding, bioactivation, acyl glucuronide, GSH adduct, or immune/idiosyncratic evidence can strongly support DILI risk when liver metabolism context and transferability are credible.",
        "Tier 6 hepatic cell injury is supporting evidence. Primary hepatocyte, HepaRG, HepG2/C3A ADMET, LDH/apoptosis/caspase, or spheroid evidence is useful but should not dominate over direct clinical/in vivo evidence.",
        "Use the single-molecule analysis only as a plausibility prior; it cannot by itself determine dili_prediction.",
        "Downweight low-transferability, distant_analog, very_distant_analog, weak, neutral_or_unclear, and context_dependent groups.",
        "Do not use exact-query ChEMBL context unless it was explicitly provided in the payload. If exact context is disabled, ignore any outside knowledge of the molecule or therapeutic class.",
        "When evidence is mixed, weigh directness, severity, liver specificity, analog similarity, structural transferability, property transferability, assay direction, species, route, dose, and duration.",
        "If the strongest positive case is only weak Tier 6 cytotoxicity, liver-weight-only evidence, GSH elevation, or distant analogs, prefer no_dili_risk unless multiple independent mechanisms coherently support DILI risk.",
        f"You must choose exactly one dili_prediction: {DILI_POSITIVE_PREDICTION} or {DILI_NEGATIVE_PREDICTION}. Express uncertainty through confidence, caveats, and evidence_gaps.",
    ]


def _group_prompt_payload(query: dict[str, Any], group: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "Group-level DILI analog transferability analysis.",
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
            "Assess structural and property transferability from each neighbor to the query.",
            "Low-similarity analogs are intentionally included. Explicitly judge whether they are transferable.",
            "Do not use distant_analog or very_distant_analog neighbors as positive or negative DILI evidence unless the shared scaffold and assay mechanism make a strong medicinal chemistry case.",
            "Use mmp_structure_compare when scaffold/MCS/matched-pair changes could alter reactive metabolite formation, bile-acid transporter liability, mitochondrial accumulation, or hepatic exposure.",
            "Use properties_compare when pKa, logD, TPSA, charge, HBD/HBA, molecular size, polarity, or lipophilicity could affect transferability.",
            "Tool outputs are authoritative only for the pair they compare; cite which neighbor each tool result supports.",
            "Use same_endpoint_activity as direct query-vs-neighbor assay comparison when present.",
            "Use same_assay_different_endpoint_activity only as same-assay context; do not directly compare numeric values across different endpoints.",
            "Infer evidence direction and strength from the group tier/endpoint_group plus the minimal evidence records; make your own transferability judgment.",
            "For Tier 1, distinguish human/clinical DILI, severe liver outcome, regulatory liver signal, and human liver lab monitoring.",
            "For Tier 2, distinguish hepatic necrosis/pathology or liver clinical chemistry from liver-weight-only evidence.",
            "For Tier 3, distinguish hepatobiliary/cholestasis transporters from generic transporters.",
            "For Tier 4, distinguish oxidative stress or mitochondrial dysfunction from weak GSH content/elevation context.",
            "For Tier 5, distinguish liver metabolism/bioactivation/covalent binding from generic target covalent binding.",
            "For Tier 6, distinguish primary hepatocyte/HepaRG/HepG2-C3A ADMET injury from generic cancer-cell cytotoxicity.",
            "Treat inactive/no effect/no toxicity activity comments as evidence against that specific assay liability only, not proof of global liver safety.",
            "Return key_evidence as structured evidence cards, not a plain list of molecule ids.",
            "For each key_evidence item, derive assay_signal and activity_values from the provided evidence_rows, derive tool_summary from tool outputs, and judge transferability/effect_on_dili_reasoning yourself.",
            "Return JSON with useful_for_dili_reasoning, transferability, evidence_direction, confidence, reasoning_summary, key_evidence, caveats.",
        ],
        "required_json_schema": {
            "useful_for_dili_reasoning": "boolean",
            "transferability": "high | moderate | low | not_applicable",
            "evidence_direction": (
                "supports_dili_risk | argues_against_dili_risk | clinical_dili_signal | "
                "in_vivo_liver_injury_signal | cholestasis_or_bile_acid_transport_risk | "
                "mitochondrial_or_organelle_stress_risk | reactive_metabolite_or_bioactivation_risk | "
                "immune_or_idiosyncratic_context | hepatic_cell_injury_risk | exposure_or_property_context | "
                "neutral_or_unclear | context_dependent"
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
                    "effect_on_dili_reasoning": "string",
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
        "dili_relevant_evidence_rows": [
            _clean_evidence_row(row) for row in context.get("dili_relevant_evidence_rows", [])
        ],
    }


def _parse_json_content(content: str) -> Any:
    return parse_json_content(content)


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
    parser.add_argument("--groups", nargs="*", default=None, help="Optional exact Tier.endpoint_group ids to reason over.")
    parser.add_argument("--max-groups", type=int, default=0, help="Debug limit; 0 means all groups with neighbors.")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=4096)
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
        "--morgan-neighbor-selector",
        choices=NEIGHBOR_SELECTORS,
        default=SIMILARITY_SELECTOR,
    )
    return parser.parse_args(argv)


def _log(message: str) -> None:
    print(f"[dili_reasoning_pipeline] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
