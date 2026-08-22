"""Shared validated call patterns for single-molecule and evidence-group branches."""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from tools.chembl_tool.common.coverage_reasoning import (
    augment_group_messages_with_neighbor_context,
)
from tools.chembl_tool.common.evidence_contract import ASSAY_RAW_CARD_PROMPT_PROFILE
from tools.chembl_tool.common.openai_reasoning_client import OpenAICompatibleClient
from tools.chembl_tool.common.reasoning_validation import (
    allowed_values_from_required_schema,
    call_with_json_validation,
)


# Leave headroom for the frozen 20,480-token completion budget within GLM's
# 131,072-token context window.  The prior 750 KB gate allowed a 439 KB JSON
# payload to reach 110,593 input tokens and exceed the window by one token.
MAX_GROUP_PROMPT_BYTES = 400_000
DEEPSEEK_CONTEXT_TOKENS = 1_048_576
DEEPSEEK_COMPLETION_TOKENS = 20_480
RAW_ASSAY_MAX_PAYLOAD_TOKENS = 900_000
RAW_ASSAY_FAST_ACCEPT_BYTES = RAW_ASSAY_MAX_PAYLOAD_TOKENS
DEEPSEEK_TOKENIZER_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEEPSEEK_TOKENIZER_PATH_ENV = "TXAGENT_DEEPSEEK_TOKENIZER_JSON"
MAX_OVERSIZE_NEIGHBOR_EVIDENCE_ROWS = 100
_GROUP_PROMPT_METADATA_HEADROOM_BYTES = 4_000
_RAW_ASSAY_METADATA_HEADROOM_TOKENS = 5_000


def bound_group_prompt_payload(
    payload: dict[str, Any],
    *,
    evidence_prompt_profile: str = "",
    token_counter: Callable[[Mapping[str, Any]], int] | None = None,
) -> dict[str, Any]:
    """Apply an isolated legacy-byte or raw-assay token transport guard."""
    if evidence_prompt_profile == ASSAY_RAW_CARD_PROMPT_PROFILE:
        if _payload_bytes(payload) <= RAW_ASSAY_FAST_ACCEPT_BYTES:
            return payload
        counter = token_counter or _deepseek_payload_tokens
        return _bound_group_prompt_payload(
            payload,
            size=counter,
            limit=RAW_ASSAY_MAX_PAYLOAD_TOKENS,
            target=RAW_ASSAY_MAX_PAYLOAD_TOKENS - _RAW_ASSAY_METADATA_HEADROOM_TOKENS,
            size_metadata={
                "limit_type": "deepseek_payload_tokens",
                "max_payload_tokens": RAW_ASSAY_MAX_PAYLOAD_TOKENS,
                "deepseek_context_tokens": DEEPSEEK_CONTEXT_TOKENS,
                "reserved_completion_tokens": DEEPSEEK_COMPLETION_TOKENS,
            },
            original_size_key="original_payload_tokens",
        )
    return _bound_group_prompt_payload(
        payload,
        size=_payload_bytes,
        limit=MAX_GROUP_PROMPT_BYTES,
        target=MAX_GROUP_PROMPT_BYTES - _GROUP_PROMPT_METADATA_HEADROOM_BYTES,
        size_metadata={
            "limit_type": "serialized_bytes",
            "max_bytes": MAX_GROUP_PROMPT_BYTES,
        },
        original_size_key="original_bytes",
    )


def _bound_group_prompt_payload(
    payload: dict[str, Any],
    *,
    size: Callable[[Mapping[str, Any]], int],
    limit: int,
    target: int,
    size_metadata: dict[str, Any],
    original_size_key: str,
) -> dict[str, Any]:
    original_size = size(payload)
    if original_size <= limit:
        return payload

    neighbors = payload.get("neighbors") or []
    original_neighbors = len(neighbors)
    original_rows = [len(neighbor.get("evidence_rows") or []) for neighbor in neighbors]
    for neighbor in neighbors:
        rows = neighbor.get("evidence_rows") or []
        if len(rows) > MAX_OVERSIZE_NEIGHBOR_EVIDENCE_ROWS:
            neighbor["evidence_rows"] = _evenly_spaced_items(
                rows, MAX_OVERSIZE_NEIGHBOR_EVIDENCE_ROWS
            )

    while size(payload) > target:
        reducible = [
            neighbor
            for neighbor in payload.get("neighbors") or []
            if len(neighbor.get("evidence_rows") or []) > 1
        ]
        if reducible:
            current_size = size(payload)
            ratio = max(0.1, min(0.9, target / current_size * 0.9))
            for neighbor in reducible:
                rows = neighbor.get("evidence_rows") or []
                row_limit = max(1, int(len(rows) * ratio))
                neighbor["evidence_rows"] = _evenly_spaced_items(rows, row_limit)
            continue
        current_neighbors = payload.get("neighbors") or []
        if len(current_neighbors) <= 1:
            raise ValueError(
                "group prompt fixed metadata plus one evidence row exceeds the transport bound"
            )
        ratio = max(0.1, min(0.9, target / size(payload) * 0.9))
        keep = max(1, int(len(current_neighbors) * ratio))
        payload["neighbors"] = _evenly_spaced_items(current_neighbors, keep)

    payload["prompt_transport"] = {
        "oversize_guard_applied": True,
        original_size_key: original_size,
        **size_metadata,
        "sampling": "deterministic_even_spacing",
        "original_neighbors": original_neighbors,
        "neighbors_in_prompt": len(payload.get("neighbors") or []),
        "original_evidence_rows": sum(original_rows),
        "evidence_rows_in_prompt": sum(
            len(neighbor.get("evidence_rows") or [])
            for neighbor in payload.get("neighbors") or []
        ),
    }
    if size(payload) > limit:
        raise AssertionError("group prompt transport bound was not enforced")
    return payload


def _payload_bytes(payload: Mapping[str, Any]) -> int:
    return len(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


@lru_cache(maxsize=1)
def _deepseek_tokenizer() -> Any:
    from tokenizers import Tokenizer

    tokenizer_path = os.getenv(DEEPSEEK_TOKENIZER_PATH_ENV, "").strip()
    if tokenizer_path:
        return Tokenizer.from_file(tokenizer_path)
    return Tokenizer.from_pretrained(DEEPSEEK_TOKENIZER_ID)


def _deepseek_payload_tokens(payload: Mapping[str, Any]) -> int:
    serialized = json.dumps(payload, ensure_ascii=False)
    return len(_deepseek_tokenizer().encode(serialized).ids)


def _evenly_spaced_items(items: list[Any], limit: int) -> list[Any]:
    if len(items) <= limit:
        return list(items)
    if limit <= 1:
        return [items[0]]
    return [
        items[round(index * (len(items) - 1) / (limit - 1))] for index in range(limit)
    ]


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
            allowed_values=allowed_values_from_required_schema(messages),
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
        allowed_values=allowed_values_from_required_schema(messages),
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
    content_validator: Callable[[Mapping[str, Any]], list[str]] | None = None,
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
        allowed_values=(
            allowed_values
            if allowed_values is not None
            else allowed_values_from_required_schema(messages)
        ),
        forbidden_field_names=forbidden_field_names,
        content_validator=content_validator,
        branch_name="group",
    )
