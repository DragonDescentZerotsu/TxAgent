"""Native query priors and cached FINAL retrieval preparation."""

from __future__ import annotations

import json

from functools import partial

from typing import Any

from predict.llm_io.evidence import evidence_for_llm

from predict.api_client.client import OpenAICompatibleClient

from predict.harnesses.branches.prompt import clean_exact_match as _clean_exact_match, llm_query_payload as _llm_query_payload

from predict.harnesses.branches.artifacts import write_trace_jsonl

from predict.harnesses.branches.inference import call_single_molecule_branch

from predict.llm_io.response import structured_response_is_valid

from predict.prompts import native_prior_messages, task_profiles

DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE = task_profiles("bioavailability_ma")["default"]

_write_trace_jsonl = partial(
    write_trace_jsonl,
    prediction_field="bioavailability_prediction",
)

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

def single_molecule_messages(
    query: dict[str, Any],
    chembl_context: dict[str, Any] | None = None,
    *,
    prompt_profile: str = DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE,
) -> list[dict[str, str]]:
    return native_prior_messages("bioavailability_ma", query, chembl_context, prompt_profile=prompt_profile)

def _reason_single_molecule(
    client: OpenAICompatibleClient,
    query: dict[str, Any],
    chembl_context: dict[str, Any] | None = None,
    *,
    prompt_profile: str = DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE,
) -> dict[str, Any]:
    messages = single_molecule_messages(
        query,
        chembl_context,
        prompt_profile=prompt_profile,
    )
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

def _clean_query_chembl_context(context: dict[str, Any]) -> dict[str, Any]:
    status = context.get("status") or "not_available"
    if status != "found":
        return {key: context.get(key, "") for key in ["status", "reason", "standard_inchi_key"] if context.get(key)}
    return {
        "status": "found",
        "selected_molecule_chembl_id": context.get("selected_molecule_chembl_id", ""),
        "exact_matches": [_clean_exact_match(match) for match in context.get("exact_matches", [])],
        "bioavailability_relevant_evidence_rows": [
            evidence_for_llm(row)
            for row in context.get("bioavailability_relevant_evidence_rows", [])
        ],
    }

from predict.harnesses.branches.tasks.flat_only_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
