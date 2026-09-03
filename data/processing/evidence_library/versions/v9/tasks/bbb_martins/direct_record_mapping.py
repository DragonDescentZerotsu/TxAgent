"""Map normalized BBB direct-source rows to main-branch vote/condition semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from data.processing.evidence_library.shared.v2.direct_record_mapping import (
    condition_key,
    direct_mapping_row,
    read_jsonl,
)
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.experimental_meaningful_cns_access_benchmark import (
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
