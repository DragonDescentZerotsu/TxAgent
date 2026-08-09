"""Versioned evidence surfaces for final-stage molecular reasoning.

The historical ``summary_only`` surface is a strict no-op: callers receive
exactly the group-summary field they already serialized.  Experimental card
surfaces deterministically expose a bounded, identity-safe subset of the
already prepared reasoning retrieval.  They never re-retrieve evidence,
change neighbor selection, or derive task labels.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any, Mapping

from tools.chembl_tool.common.evidence_contract import evidence_for_llm
from tools.chembl_tool.common.identity_blind import (
    prepare_identity_blind_final_retrieval,
    prepare_prefetched_final_retrieval,
    prepare_reasoning_retrieval,
    sanitize_identity_blind_branch_outputs,
)
from tools.chembl_tool.common.json_utils import canonical_json_bytes
from tools.chembl_tool.common.reasoning_validation import validated_branch_content


SUMMARY_ONLY = "summary_only"
SUMMARY_PLUS_CARDS = "summary_plus_cards"
CARDS_ONLY = "cards_only"
FINAL_EVIDENCE_SURFACES = (
    SUMMARY_ONLY,
    SUMMARY_PLUS_CARDS,
    CARDS_ONLY,
)
FINAL_EVIDENCE_CARD_CONTRACT_VERSION = "final_evidence_cards.v1"

MAX_CARDS = 12
MAX_EVIDENCE_ROWS_PER_CARD = 3
MAX_EXAMPLES_PER_EVIDENCE_ROW = 2
MAX_EVIDENCE_TEXT_CHARS = 700
MAX_EXAMPLE_TEXT_CHARS = 350
MAX_COMPARISON_TEXT_CHARS = 900


def add_final_evidence_surface_argument(parser: Any) -> None:
    """Register the shared, versioned final-surface CLI option."""
    parser.add_argument(
        "--final-evidence-surface",
        choices=FINAL_EVIDENCE_SURFACES,
        default=SUMMARY_ONLY,
        help=(
            "Final-stage evidence surface. summary_only preserves the historical "
            "prompt; card surfaces are explicit final-only ablations."
        ),
    )


def build_final_evidence_fields(
    retrieval: dict[str, Any],
    group_reasoning_outputs: list[dict[str, Any]],
    *,
    surface: str = SUMMARY_ONLY,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Build final-prompt evidence fields and an optional card audit.

    ``summary_only`` intentionally returns the historical field without
    copying or normalizing it.  This preserves the previous prompt contract.
    """
    _validate_surface(surface)
    if surface == SUMMARY_ONLY:
        return {"group_reasoning_outputs": group_reasoning_outputs}, None

    card_payload = build_final_evidence_cards(retrieval, surface=surface)
    fields: dict[str, Any] = {"final_evidence_cards": card_payload}
    if surface == SUMMARY_PLUS_CARDS:
        fields = {
            "group_reasoning_outputs": group_reasoning_outputs,
            **fields,
        }
    return fields, deepcopy(card_payload["audit"])


def compact_group_reasoning_outputs(
    group_outputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project persisted branch artifacts onto the final-prompt contract."""
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
    *,
    tool_service: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Restore the exact final-stage view for direct resume entrypoints."""
    surface = str(manifest.get("final_evidence_surface") or SUMMARY_ONLY)
    _validate_surface(surface)
    identity_blind = bool(manifest.get("identity_blind"))
    if identity_blind:
        group_outputs = sanitize_identity_blind_branch_outputs(
            group_outputs,
            retrieval,
        )
    if surface != SUMMARY_ONLY:
        retrieval = prepare_reasoning_retrieval(
            retrieval,
            tool_service,
            identity_blind=identity_blind,
            harness_prefetch_tools=bool(manifest.get("harness_prefetch_tools")),
            prefetched_tool_replay_run_dir=str(
                manifest.get("prefetched_tool_replay_source_run_dir") or ""
            ),
            neighbor_context_profile=str(
                manifest.get("neighbor_context_profile") or "standard"
            ),
        )
    elif identity_blind:
        retrieval = prepare_identity_blind_final_retrieval(retrieval, single_output)
    elif manifest.get("harness_prefetch_tools"):
        retrieval = prepare_prefetched_final_retrieval(
            retrieval,
            single_output,
            identity_blind=False,
        )
    return retrieval, group_outputs, surface


def final_evidence_instructions(surface: str) -> list[str]:
    """Return only the additional instructions required by card surfaces."""
    _validate_surface(surface)
    if surface == SUMMARY_ONLY:
        return []
    if surface == SUMMARY_PLUS_CARDS:
        return [
            "The final_evidence_cards are deterministic source-backed excerpts from the same retrieved analogs used by the group analyses.",
            "Use the cards to verify or recover endpoint, measurement, context, uncertainty, and molecular-comparison details that may have been compressed in the group summaries.",
            "Do not count a card and its corresponding group summary as independent evidence.",
        ]
    return [
        "Reason directly from final_evidence_cards; group reasoning summaries are intentionally withheld in this ablation.",
        "Each card is a deterministic source-backed excerpt containing endpoint, measurement, context, uncertainty, and molecular-comparison details.",
        "Do not infer missing molecule identities or treat repeated records as independent studies.",
    ]


def build_final_evidence_cards(
    retrieval: dict[str, Any],
    *,
    surface: str,
) -> dict[str, Any]:
    """Create bounded cards from an already prepared LLM-facing retrieval."""
    _validate_surface(surface)
    if surface == SUMMARY_ONLY:
        raise ValueError("summary_only does not materialize final evidence cards")

    candidates: list[tuple[str, dict[str, Any]]] = []
    for group in retrieval.get("groups") or []:
        group_id = str(group.get("group_id") or "")
        for neighbor in group.get("neighbors") or []:
            candidates.append((group_id, neighbor))

    selected = _evenly_spaced_items(candidates, MAX_CARDS)
    cards: list[dict[str, Any]] = []
    evidence_rows_total = 0
    evidence_rows_included = 0
    for card_index, (group_id, neighbor) in enumerate(selected, start=1):
        rows = list(neighbor.get("evidence_rows") or [])
        evidence_rows_total += len(rows)
        selected_rows = _evenly_spaced_items(rows, MAX_EVIDENCE_ROWS_PER_CARD)
        evidence_rows_included += len(selected_rows)
        cards.append(
            {
                "card_id": f"card_{card_index:02d}",
                "group_id": group_id,
                "neighbor_rank": neighbor.get("rank"),
                "similarity": neighbor.get("similarity"),
                "similarity_bucket": neighbor.get("similarity_bucket"),
                "similarity_metric": neighbor.get("similarity_metric"),
                "molecule_relation": neighbor.get("molecule_relation"),
                "source_group_ids": list(neighbor.get("source_group_ids") or []),
                "evidence_rows": [
                    _compact_evidence_record(row) for row in selected_rows
                ],
                "molecular_comparisons": [
                    _compact_comparison(row)
                    for row in neighbor.get("prefetched_comparisons") or []
                ],
                "truncation": {
                    "evidence_rows_total": len(rows),
                    "evidence_rows_included": len(selected_rows),
                    "sampling": "deterministic_even_spacing",
                },
            }
        )

    serialized_cards = canonical_json_bytes(cards)
    audit = {
        "contract_version": FINAL_EVIDENCE_CARD_CONTRACT_VERSION,
        "surface": surface,
        "n_candidate_cards": len(candidates),
        "n_cards": len(cards),
        "max_cards": MAX_CARDS,
        "evidence_rows_total": evidence_rows_total,
        "evidence_rows_included": evidence_rows_included,
        "max_evidence_rows_per_card": MAX_EVIDENCE_ROWS_PER_CARD,
        "card_sampling": "deterministic_even_spacing",
        "identity_source": "prepared_reasoning_retrieval",
        "cards_sha256": hashlib.sha256(serialized_cards).hexdigest(),
        "cards_bytes": len(serialized_cards),
    }
    return {
        "contract_version": FINAL_EVIDENCE_CARD_CONTRACT_VERSION,
        "surface": surface,
        "cards": cards,
        "audit": audit,
    }


def _compact_evidence_record(row: dict[str, Any]) -> dict[str, Any]:
    record = evidence_for_llm(row)
    text = record.get("text") or {}
    examples = list(record.get("examples") or [])
    return {
        "source": _without_empty(record.get("source") or {}),
        "group": _without_empty(record.get("group") or {}),
        "endpoint": _without_empty(record.get("endpoint") or {}),
        "text": _without_empty(
            {
                "evidence": _truncate_text(text.get("evidence"), MAX_EVIDENCE_TEXT_CHARS),
                "context": _truncate_text(text.get("context"), MAX_EVIDENCE_TEXT_CHARS),
            }
        ),
        "annotations": _without_empty(record.get("annotations") or {}),
        "quality": _without_empty(record.get("quality") or {}),
        "provenance": _without_empty(record.get("provenance") or {}),
        "examples": [
            _compact_example(example)
            for example in examples[:MAX_EXAMPLES_PER_EVIDENCE_ROW]
            if isinstance(example, dict)
        ],
    }


def _compact_example(example: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in sorted(example):
        if key in {
            "canonical_smiles",
            "global_identifier",
            "molecule_name",
            "smiles",
            "source_index",
            "source_molecule_id",
            "source_record_id",
        }:
            continue
        value = example[key]
        if isinstance(value, str):
            value = _truncate_text(value, MAX_EXAMPLE_TEXT_CHARS)
        if value not in (None, "", [], {}):
            output[str(key)] = value
    return output


def _compact_comparison(result: dict[str, Any]) -> dict[str, Any]:
    return _without_empty(
        {
            "tool_name": result.get("tool_name"),
            "status": result.get("status"),
            "content": _truncate_text(
                result.get("content"),
                MAX_COMPARISON_TEXT_CHARS,
            ),
            "warnings": result.get("warnings") or [],
            "errors": result.get("errors") or [],
        }
    )


def _truncate_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 16)].rstrip() + " …[truncated]"


def _without_empty(value: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): item
        for key, item in value.items()
        if item not in (None, "", [], {})
    }


def _evenly_spaced_items(items: list[Any], limit: int) -> list[Any]:
    if len(items) <= limit:
        return list(items)
    if limit <= 1:
        return [items[0]]
    return [
        items[round(index * (len(items) - 1) / (limit - 1))]
        for index in range(limit)
    ]


def _validate_surface(surface: str) -> None:
    if surface not in FINAL_EVIDENCE_SURFACES:
        raise ValueError(f"Unknown final evidence surface: {surface}")
