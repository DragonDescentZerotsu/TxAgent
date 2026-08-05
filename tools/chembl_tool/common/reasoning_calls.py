"""Shared validated call patterns for single-molecule and evidence-group branches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.coverage_reasoning import (
    augment_group_messages_with_neighbor_context,
)
from tools.chembl_tool.common.openai_reasoning_client import OpenAICompatibleClient
from tools.chembl_tool.common.reasoning_validation import call_with_json_validation


# Leave headroom for the frozen 20,480-token completion budget within GLM's
# 131,072-token context window.  The prior 750 KB gate allowed a 439 KB JSON
# payload to reach 110,593 input tokens and exceed the window by one token.
MAX_GROUP_PROMPT_BYTES = 400_000
MAX_OVERSIZE_NEIGHBOR_EVIDENCE_ROWS = 100


def bound_group_prompt_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Bound only oversized group prompts while recording deterministic truncation."""
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(serialized) <= MAX_GROUP_PROMPT_BYTES:
        return payload

    for neighbor in payload.get("neighbors") or []:
        rows = neighbor.get("evidence_rows") or []
        total = len(rows)
        if total <= MAX_OVERSIZE_NEIGHBOR_EVIDENCE_ROWS:
            continue
        neighbor["evidence_rows"] = _evenly_spaced_items(rows, MAX_OVERSIZE_NEIGHBOR_EVIDENCE_ROWS)
        neighbor["evidence_rows_total"] = total
        neighbor["evidence_rows_in_prompt"] = len(neighbor["evidence_rows"])
        neighbor["evidence_rows_truncated"] = True

    payload["prompt_transport"] = {
        "oversize_guard_applied": True,
        "original_bytes": len(serialized),
        "max_bytes": MAX_GROUP_PROMPT_BYTES,
        "sampling": "deterministic_even_spacing",
        "max_evidence_rows_per_oversize_neighbor": MAX_OVERSIZE_NEIGHBOR_EVIDENCE_ROWS,
    }
    return payload


def _evenly_spaced_items(items: list[Any], limit: int) -> list[Any]:
    if len(items) <= limit:
        return list(items)
    if limit <= 1:
        return [items[0]]
    return [items[round(index * (len(items) - 1) / (limit - 1))] for index in range(limit)]


def load_frozen_single_analysis(source_run_dir: str) -> dict[str, Any] | None:
    """Load a successful single-molecule branch for controlled ablations."""
    if not source_run_dir:
        return None
    path = Path(source_run_dir) / "single_molecule_reasoning_output.json"
    if not path.exists():
        raise FileNotFoundError(f"Frozen single-molecule analysis not found: {path}")
    output = json.loads(path.read_text(encoding="utf-8"))
    if output.get("status") != "ok":
        raise ValueError(f"Frozen single-molecule analysis is not successful: {path}")
    output["reused_from"] = str(path)
    return output


def call_single_molecule_branch(
    client: OpenAICompatibleClient,
    messages: list[dict[str, Any]],
    *,
    query: dict[str, Any],
    tools: list[dict[str, Any]],
    first_tool_choice: Any = "auto",
) -> dict[str, Any]:
    """Use a forced live property tool, or a harness-prefetched blind result."""
    prefetched = query.get("prefetched_molecule_properties")
    if prefetched:
        response = call_with_json_validation(
            client.chat_json,
            messages,
            required_fields=("confidence", "reasoning_summary"),
            branch_name="single-molecule",
        )
        response["tool_results"] = [prefetched]
        return response
    return call_with_json_validation(
        lambda retry_messages: client.chat_json_with_tools(
            retry_messages,
            tools=tools,
            allowed_tool_names={"molecule_properties"},
            first_tool_choice=first_tool_choice,
        ),
        messages,
        required_fields=("confidence", "reasoning_summary"),
        required_tool_names=("molecule_properties",),
        branch_name="single-molecule",
    )


def call_group_branch(
    client: OpenAICompatibleClient,
    messages: list[dict[str, Any]],
    *,
    group: dict[str, Any],
    tools: list[dict[str, Any]],
    required_fields: tuple[str, ...] = (
        "transferability",
        "confidence",
        "reasoning_summary",
    ),
    allowed_values: dict[str, set[str]] | None = None,
    forbidden_field_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Use live comparison tools, or harness-prefetched blind comparisons."""
    messages = augment_group_messages_with_neighbor_context(messages, group)
    if group.get("tools_prefetched") or group.get("identity_blind"):
        call = client.chat_json
    else:
        def call(retry_messages: list[dict[str, Any]]) -> dict[str, Any]:
            return client.chat_json_with_optional_tools(
                retry_messages,
                tools=tools,
                allowed_tool_names={"properties_compare", "mmp_structure_compare"},
            )
    return call_with_json_validation(
        call,
        messages,
        required_fields=required_fields,
        allowed_values=allowed_values,
        forbidden_field_names=forbidden_field_names,
        branch_name="group",
    )
