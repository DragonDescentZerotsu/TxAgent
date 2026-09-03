"""Expand main Bioavailability direct-claim decisions back to v7 source rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from data.processing.evidence_library.shared.v1.direct_record_mapping import (
    condition_key,
    direct_mapping_row,
    read_jsonl,
)
from data.processing.evidence_library.versions.v8.tasks.bioavailability_ma.canonical_source import (
    DIRECT_CLAIMS_PATH,
    DIRECT_SOURCE_ROWS_PATH,
)
from data.processing.evidence_library.versions.v8.tasks.bioavailability_ma.starling_benchmark import (
    is_human_context,
    label_bioavailability_value,
)


CORE_EXTERNAL_CONDITION_GROUPS = (
    "prandial_state=fasted",
    "prandial_state=fed_unspecified",
    "prandial_state=fed_high_fat",
    "co_treatment=rifampin",
    "disease=cirrhosis",
    "disease=cystic_fibrosis",
    "release_profile=modified_release",
)


SELECTED_REVIEW = Path(
    "data/gold_labels/Bioavailability_Ma/v1/scaffold/source_condition_review.jsonl"
)
PROPOSAL_AUDIT = Path(
    "data/artifacts/starling/bioavailability_ma/source_reviews/context_conditioned_review_v2/"
    "proposal_audit.jsonl"
)
DIRECT_MAPPING_INPUTS = (
    DIRECT_SOURCE_ROWS_PATH,
    DIRECT_CLAIMS_PATH,
    SELECTED_REVIEW,
    PROPOSAL_AUDIT,
)
DIRECT_GROUP_ID = "Observed.direct_oral_bioavailability"


def build_direct_record_mapping(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    direct_source_ids, claim_by_source, reviewed, proposals = _mapping_inputs()
    output: list[dict[str, Any]] = []
    for record in records:
        source_id = str(record.get("source_id") or "")
        canonical_source_id = _canonical_source_record_id(record)
        is_direct_source = source_id == "hf_bioavailability" or (
            source_id == "oral_exposure" and canonical_source_id in direct_source_ids
        )
        if not is_direct_source:
            continue
        claim = claim_by_source.get(canonical_source_id)
        internal_claim_id = str((claim or {}).get("canonical_claim_id") or "")
        review = reviewed.get(internal_claim_id)
        proposal = proposals.get(internal_claim_id)
        condition = condition_key(
            canonical_record_id=canonical_source_id,
            condition_text=(claim or record).get("qualifying_conditions"),
            reviewed=review,
            proposal=proposal,
            selected_groups=CORE_EXTERNAL_CONDITION_GROUPS,
        )
        if condition.status == "selected_reviewed":
            label = int(review["Y"])
            counted = True
            reason = str(review.get("label_method") or "selected_condition_review")
        elif condition.status == "none_reported" and claim is not None:
            if not is_human_context(claim.get("species_or_population")):
                label, reason = None, "nonhuman_or_unresolved_population"
            else:
                label, reason = label_bioavailability_value(
                    claim.get("oral_bioavailability_value")
                )
            counted = label is not None
        else:
            label = int(review["Y"]) if review and review.get("Y") in {0, 1} else None
            counted = False
            reason = str(
                (review or {}).get("review_reason")
                or (proposal or {}).get("proposal_reason")
                or (claim or {}).get("classification_reason")
                or "not_a_canonical_direct_claim"
            )
        mapping = direct_mapping_row(
            record,
            condition=condition,
            counted=counted,
            reason=reason,
            label=label,
            group_id=DIRECT_GROUP_ID,
        )
        if (
            mapping["retrieval_source_id"] == "direct_residual"
            and str(record.get("canonical_endpoint_name") or "")
            in {"bioavailability", "oral_bioavailability"}
        ):
            mapping["direct_residual_endpoint_name"] = "oral_bioavailability"
        output.append(mapping)
    return output


@lru_cache(maxsize=1)
def _mapping_inputs() -> tuple[
    set[str],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    direct_source_ids = set(
        pd.read_parquet(DIRECT_SOURCE_ROWS_PATH, columns=["source_record_id"])[
            "source_record_id"
        ].astype(str)
    )
    frame = pd.read_parquet(
        DIRECT_CLAIMS_PATH,
        columns=[
            "canonical_claim_id",
            "source_record_ids",
            "qualifying_conditions",
            "species_or_population",
            "oral_bioavailability_value",
            "classification_reason",
        ],
    ).astype(object)
    claims = frame.where(pd.notna(frame), None).to_dict(orient="records")
    claim_by_source: dict[str, dict[str, Any]] = {}
    for claim in claims:
        source_record_ids = claim.get("source_record_ids")
        for source_record_id in (() if source_record_ids is None else source_record_ids):
            key = str(source_record_id)
            if key in claim_by_source:
                raise ValueError(f"direct source row maps to multiple claims: {key}")
            claim_by_source[key] = claim
    return (
        direct_source_ids,
        claim_by_source,
        {str(row["source_record_id"]): row for row in read_jsonl(SELECTED_REVIEW)},
        {str(row["source_record_id"]): row for row in read_jsonl(PROPOSAL_AUDIT)},
    )


def _canonical_source_record_id(record: Mapping[str, Any]) -> str:
    source_id = str(record.get("source_id") or "")
    source_record_id = str(record.get("source_record_id") or "")
    if source_id == "hf_bioavailability":
        return f"hf:{source_record_id}"
    if source_id == "oral_exposure":
        return f"local:{int(record.get('source_row_number') or 0) - 1}:{source_record_id}"
    return ""


__all__ = ["DIRECT_MAPPING_INPUTS", "build_direct_record_mapping"]
