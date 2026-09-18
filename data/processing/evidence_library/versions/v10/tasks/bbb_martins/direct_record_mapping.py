"""Map normalized BBB direct-source rows to main-branch vote/condition semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from data.processing.evidence_library.shared.v2.direct_record_mapping import (
    condition_key,
    direct_mapping_row,
    read_jsonl,
)
from data.processing.evidence_library.versions.v10.tasks.bbb_martins.experimental_meaningful_cns_access_benchmark import (
    label_record,
)


SELECTED_EXTERNAL_CONDITION_GROUPS = (
    "barrier_state=disrupted",
    "co_treatment=cyclosporine",
    "disease=meningitis_unspecified",
    "disease=bacterial_meningitis",
    "disease=pneumococcal_meningitis",
    "disease=tuberculous_meningitis",
    "disease=brain_tumor_or_glioma",
    "disease=cerebral_ischemia",
)


def redistribute_training_context_membership(contexts, records_by_source_id, levels, approved_source_ids):
    """Project frozen training references through UID lineage; never infer a level or label."""
    targets = {}
    for context in contexts:
        key = (context["molecule_identity_key"], context["condition_group"])
        if key in targets:
            raise ValueError("Duplicate training parent/condition context")
        targets[key] = context
    output, applied, transfers = [], set(), []
    for context in contexts:
        for source_id in context["source_record_ids"]:
            record = records_by_source_id[source_id]
            uid = record["source_row_uid"]
            level = levels.get(uid)
            destination = None
            if level is None:
                status = "excluded_unmapped"
            elif not record.get("canonical_record_id"):
                status = "excluded_from_library"
            elif level != 1:
                status = "mapped_other_level"
            elif record["parent_identity_key"] == context["molecule_identity_key"]:
                destination, status = context, "unchanged"
            elif source_id not in approved_source_ids:
                status = "unresolved_parent_change"
            else:
                destination = targets.get((record["parent_identity_key"], context["condition_group"]))
                unanimous = context["label_counts"].get(str(context["Y"]), 0) == context["source_record_count"]
                vote = record.get("direct_vote_label", context["Y"] if unanimous else None)
                if destination is None or vote not in (0, 1) or destination["Y"] != vote:
                    raise ValueError(f"Approved context transfer changes labels or lacks a destination: {source_id}")
                if not unanimous:
                    votes = Counter(str(records_by_source_id[s].get("direct_vote_label")) for s in context["source_record_ids"])
                    if votes != Counter(context["label_counts"]):
                        raise ValueError(f"Source record votes do not replay frozen context: {source_id}")
                transfers.append((context, destination, str(vote)))
                applied.add(source_id)
                status = "redistributed"
            output.append({
                "source_context_id": context["benchmark_row_id"],
                "destination_context_id": destination["benchmark_row_id"] if destination else None,
                "source_record_id": source_id, "source_row_uid": uid,
                "canonical_record_id": record.get("canonical_record_id"),
                "source_parent": context["molecule_identity_key"],
                "destination_parent": record.get("parent_identity_key"),
                "level": level, "status": status,
            })
    if applied != set(approved_source_ids):
        raise ValueError("Approved context transfers were not applied exactly")
    affected = {c["benchmark_row_id"]: c for source, destination, _ in transfers for c in (source, destination)}
    counts = {cid: Counter(c["label_counts"]) for cid, c in affected.items()}
    for source, destination, vote in transfers:
        counts[source["benchmark_row_id"]][vote] -= 1
        counts[destination["benchmark_row_id"]][vote] += 1
    for cid, c in affected.items():
        votes = counts[cid]
        total = sum(votes.values())
        if min(votes.values()) < 0 or (total and (
                votes[str(c["Y"])] <= votes[str(1 - c["Y"])]
                or votes[str(c["Y"])] / total < c.get("agreement_threshold", 0.7))):
            raise ValueError(f"Joint context transfer changes labels or eligibility: {cid}")
    return output


SELECTED_REVIEW = Path(
    "data/gold_labels/BBB_Martins/v1/scaffold/source_condition_review.jsonl"
)
PROPOSAL_AUDIT = Path(
    "data/artifacts/starling/bbb_martins/source_reviews/context_conditioned_review_v1/"
    "proposal_audit.jsonl"
)
DIRECT_MAPPING_INPUTS = (SELECTED_REVIEW, PROPOSAL_AUDIT)


def build_direct_record_mapping(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reviewed, proposals = _mapping_inputs()
    output: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("source_id") or "") != "direct_bbb":
            continue
        source_index = int(
            record.get("source_index")
            if record.get("source_index") is not None
            else record.get("source_record_id")
        )
        review = reviewed.get(source_index)
        condition = condition_key(
            canonical_record_id=f"direct_bbb:row:{source_index}",
            condition_text=record.get("qualifying_conditions"),
            reviewed=review,
            proposal=proposals.get(source_index),
            selected_groups=SELECTED_EXTERNAL_CONDITION_GROUPS,
        )
        if condition.status == "selected_reviewed":
            label = int(review["Y"])
            counted = True
            reason = str(review.get("label_method") or "selected_condition_review")
        elif condition.status == "none_reported":
            label_input = {
                **record,
                "quant_metric": record.get("endpoint_name"),
                "quant_value": (
                    record.get("pre_resolution_measurement_text")
                    or record.get("measurement_text")
                ),
                "quant_units": (
                    record.get("pre_resolution_unit_text")
                    or record.get("unit_text")
                ),
            }
            label, reason = label_record(label_input, source_index=source_index)
            counted = label is not None
        else:
            label = int(review["Y"]) if review and review.get("Y") in {0, 1} else None
            counted = False
            reason = str(
                (review or {}).get("review_reason")
                or (proposals.get(source_index) or {}).get("proposal_reason")
                or "condition_not_selected_for_direct_voting"
            )
        output.append(
            direct_mapping_row(
                record,
                condition=condition,
                counted=counted,
                reason=reason,
                label=label,
            )
        )
    return output


@lru_cache(maxsize=1)
def _mapping_inputs() -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    return (
        {int(row["source_index"]): row for row in read_jsonl(SELECTED_REVIEW)},
        {int(row["source_index"]): row for row in read_jsonl(PROPOSAL_AUDIT)},
    )


__all__ = ["DIRECT_MAPPING_INPUTS", "build_direct_record_mapping"]
