"""Map normalized Skin direct-source rows to canonical vote/condition semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.direct_record_mapping import (
    condition_key,
    direct_mapping_row,
    read_jsonl,
)
from tools.chembl_tool.tasks.skin_reaction.canonical_starling_source import (
    DIRECT_PARTITION,
    load_partition_audit,
    normalized_direct_label,
    partition_record_id,
)
from tools.chembl_tool.tasks.skin_reaction.starling_benchmark import label_record


SELECTED_EXTERNAL_CONDITION_GROUPS = ("disease=atopic_dermatitis",)


SELECTED_REVIEW = Path(
    "data/processed_starling_context_conditioned_selected_v1/"
    "Skin_Reaction/scaffold/source_condition_review.jsonl"
)
PROPOSAL_AUDIT = Path(
    "data/starling_data/skin_reaction/context_conditioned_review_v1/"
    "proposal_audit.jsonl"
)
PARTITION_AUDIT = Path(
    "data/starling_data/skin_reaction/canonical_sensitization_v3/"
    "partition_audit.parquet"
)
DIRECT_MAPPING_INPUTS = (SELECTED_REVIEW, PROPOSAL_AUDIT, PARTITION_AUDIT)
DIRECT_GROUP_ID = "Direct.skin_reaction"


def build_direct_record_mapping(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reviewed = {
        str(row["source_record_id"]): row for row in read_jsonl(SELECTED_REVIEW)
    }
    proposals = {
        str(row["source_record_id"]): row for row in read_jsonl(PROPOSAL_AUDIT)
    }
    partition = load_partition_audit()
    output: list[dict[str, Any]] = []
    for record in records:
        external_id = partition_record_id(record)
        decision = partition.get(external_id)
        if external_id and decision is None:
            raise ValueError(f"Skin row lacks canonical partition: {external_id}")
        if not decision or decision.partition != DIRECT_PARTITION:
            continue
        review = reviewed.get(external_id)
        proposal = proposals.get(external_id)
        condition = condition_key(
            canonical_record_id=external_id,
            condition_text=record.get("qualifying_conditions"),
            reviewed=review,
            proposal=proposal,
            selected_groups=SELECTED_EXTERNAL_CONDITION_GROUPS,
        )
        if condition.status == "selected_reviewed":
            label = int(review["Y"])
            counted = True
            reason = str(review.get("label_method") or "selected_condition_review")
        elif condition.status == "none_reported":
            if str(record.get("source_id") or "") == "sensitization_aop":
                normalized = normalized_direct_label(record.get("result_label"))
                label = 1 if normalized == "positive" else 0 if normalized == "negative" else None
                reason = (
                    "canonical_aop_direct_outcome_label"
                    if label is not None
                    else "canonical_aop_direct_outcome_without_binary_label"
                )
            else:
                label, reason = label_record(
                    {**record, "reaction_type": record.get("endpoint_name")}
                )
            counted = label is not None
        else:
            label = int(review["Y"]) if review and review.get("Y") in {0, 1} else None
            counted = False
            reason = str(
                (review or {}).get("review_reason")
                or (proposal or {}).get("proposal_reason")
                or decision.reason
            )
        output.append(
            direct_mapping_row(
                record,
                condition=condition,
                counted=counted,
                reason=reason,
                label=label,
                group_id=DIRECT_GROUP_ID,
            )
        )
    return output


__all__ = ["DIRECT_MAPPING_INPUTS", "build_direct_record_mapping"]
