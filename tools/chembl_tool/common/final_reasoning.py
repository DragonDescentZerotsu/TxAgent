"""Shared assembly helpers for the single canonical final-reasoning prompt."""

from __future__ import annotations

from typing import Any, Mapping

from tools.chembl_tool.common.identity_blind import (
    prepare_identity_blind_final_retrieval,
    prepare_prefetched_final_retrieval,
    sanitize_identity_blind_branch_outputs,
)
from tools.chembl_tool.common.reasoning_validation import validated_branch_content


def compact_group_reasoning_outputs(
    group_outputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project persisted group artifacts onto the canonical final prompt."""
    return [
        {
            "group_id": item.get("group_id"),
            "status": item.get("status"),
            "content": validated_branch_content(item),
        }
        for item in group_outputs
    ]


def prepare_resumed_final_inputs(
    retrieval: dict[str, Any],
    single_output: dict[str, Any],
    group_outputs: list[dict[str, Any]],
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Restore the canonical final-stage view for a resumed standard run."""
    identity_blind = bool(manifest.get("identity_blind"))
    if identity_blind:
        group_outputs = sanitize_identity_blind_branch_outputs(
            group_outputs,
            retrieval,
        )
        retrieval = prepare_identity_blind_final_retrieval(retrieval, single_output)
    elif manifest.get("harness_prefetch_tools"):
        retrieval = prepare_prefetched_final_retrieval(
            retrieval,
            single_output,
            identity_blind=False,
        )
    return retrieval, group_outputs
