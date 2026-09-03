"""Shared contracts for mapping task-owned direct-label source rows.

Task modules decide whether a source row counted as a direct vote.  This module
only standardizes the condition-key and audit fields consumed by the canonical
record-collapse stage.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NO_REPORTED_CONDITION = "no_reported_external_condition"
CONDITION_KEY_STATUSES = frozenset(
    {
        "selected_reviewed",
        "accepted_unselected",
        "proposed_rejected",
        "none_reported",
        "unresolved",
    }
)


@dataclass(frozen=True)
class ConditionKey:
    group: str
    atoms: tuple[str, ...]
    scope: str
    status: str


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def condition_key(
    *,
    canonical_record_id: str,
    condition_text: Any,
    reviewed: Mapping[str, Any] | None,
    proposal: Mapping[str, Any] | None,
    selected_groups: Sequence[str],
) -> ConditionKey:
    """Resolve one exact condition key without inventing missing context."""
    text = _text(condition_text)
    selected = set(selected_groups)
    if reviewed is not None:
        review_status = _text(reviewed.get("review_status"))
        accepted_group = _text(reviewed.get("condition_group"))
        proposed_group = _text(reviewed.get("proposed_condition_group"))
        if review_status == "accepted" and accepted_group:
            status = (
                "selected_reviewed"
                if accepted_group in selected
                else "accepted_unselected"
            )
            return ConditionKey(
                accepted_group,
                _atoms(reviewed.get("condition_atoms"), accepted_group),
                "external",
                status,
            )
        if proposed_group:
            return ConditionKey(
                proposed_group,
                _atoms(reviewed.get("proposed_condition_atoms"), proposed_group),
                "external",
                "proposed_rejected",
            )
    if proposal is not None:
        proposed_group = _text(proposal.get("proposed_condition_group"))
        if proposed_group:
            return ConditionKey(
                proposed_group,
                _atoms(proposal.get("proposed_condition_atoms"), proposed_group),
                "external",
                "proposed_rejected",
            )
    if not text:
        return ConditionKey(NO_REPORTED_CONDITION, (), "none_reported", "none_reported")
    # A singleton key prevents unrelated unresolved conditions from collapsing.
    return ConditionKey(
        f"unresolved:{canonical_record_id}",
        (),
        "unresolved",
        "unresolved",
    )


def direct_mapping_row(
    record: Mapping[str, Any],
    *,
    condition: ConditionKey,
    counted: bool,
    reason: str,
    label: int | None,
    group_id: str | None = None,
) -> dict[str, Any]:
    if condition.status not in CONDITION_KEY_STATUSES:
        raise ValueError(f"unsupported condition-key status: {condition.status}")
    if counted and label not in {0, 1}:
        raise ValueError("a counted direct record requires a binary label")
    return {
        "canonical_record_id": str(record.get("canonical_record_id") or ""),
        "source_id": str(record.get("source_id") or ""),
        "source_record_id": str(record.get("source_record_id") or ""),
        "direct_vote_status": "counted" if counted else "ignored",
        "direct_vote_reason": str(reason),
        "direct_vote_label": int(label) if label in {0, 1} else None,
        "direct_vote_unit_id": (
            f"{record.get('source_id')}:row:{int(record.get('source_row_number') or 0)}"
        ),
        "condition_group": condition.group,
        "condition_atoms": list(condition.atoms),
        "condition_scope": condition.scope,
        "condition_key_status": condition.status,
        "retrieval_source_id": "direct_vote" if counted else "direct_residual",
        "direct_group_id": group_id,
        "dedup_status": "pending",
    }


def _atoms(value: Any, group: str) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        atoms = tuple(sorted({_text(item) for item in value if _text(item)}))
        if atoms:
            return atoms
    return tuple(sorted(set(group.split("+")))) if group else ()


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null"} else text


__all__ = [
    "CONDITION_KEY_STATUSES",
    "ConditionKey",
    "NO_REPORTED_CONDITION",
    "condition_key",
    "direct_mapping_row",
    "read_jsonl",
]
