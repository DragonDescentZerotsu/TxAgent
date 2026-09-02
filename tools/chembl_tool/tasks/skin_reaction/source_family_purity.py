"""Exact voter-membership rules for the Skin retrieval source."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.source_family_purity import FamilyMove
from tools.chembl_tool.common.starling.conditioned_benchmark import task_root
from tools.chembl_tool.tasks.skin_reaction.canonical_starling_source import (
    direct_outcome_reason,
)
from tools.chembl_tool.tasks.skin_reaction.starling_benchmark import (
    SOURCE_PATH,
    load_label_decisions,
)


DIRECT_GROUP = "Direct.skin_reaction"
NEAR_DIRECT_GROUP = "Observed.nonvoter_skin_outcome"
AOP_GROUP = "Mechanism.sensitization_aop"
PURITY_VERSION = "skin_source_family_purity.vote_pure.v1"
DEFAULT_CONDITION_REVIEW = task_root("skin_reaction") / "source_condition_review.jsonl"


def load_voter_source_indices(
    *,
    source_path: Path = SOURCE_PATH,
    condition_review: Path = DEFAULT_CONDITION_REVIEW,
) -> set[int]:
    """Load records that emitted a base vote or passed condition review."""

    decisions, _ = load_label_decisions(source_path=source_path)
    voters = {
        index
        for index, decision in enumerate(decisions)
        if decision.record is not None
    }
    if condition_review.exists():
        with condition_review.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("review_status") or "").lower() == "accepted":
                    voters.add(int(row["source_index"]))
    return voters


def upstream_source_index(record: Mapping[str, Any]) -> int | None:
    """Map a normalized direct-source row to its zero-based source index."""

    if str(record.get("source_id") or "") != "direct_skin_reaction":
        return None
    try:
        return int(record.get("source_row_number")) - 1
    except (TypeError, ValueError):
        return None


def vote_pure_family_move(
    record: Mapping[str, Any],
    voter_source_indices: set[int] | frozenset[int],
) -> FamilyMove:
    """Put actual voters in L1 and direct-like nonvoters in near-direct L2."""

    original = str(record.get("group_id") or "")
    index = upstream_source_index(record)
    if index is not None and index in voter_source_indices:
        if original == DIRECT_GROUP:
            return FamilyMove("", "")
        return FamilyMove(DIRECT_GROUP, "accepted_gold_vote_promoted_to_l1")

    direct_like_reason = (
        direct_outcome_reason(record) if original == AOP_GROUP else ""
    )
    if original == DIRECT_GROUP or direct_like_reason:
        reason = (
            "nonvoter_removed_from_l1"
            if original == DIRECT_GROUP
            else f"direct_like_nonvoter_to_l2:{direct_like_reason}"
        )
        return FamilyMove(NEAR_DIRECT_GROUP, reason)
    return FamilyMove("", "")
