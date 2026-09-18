"""Expand main Bioavailability direct-claim decisions back to v7 source rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from data.processing.evidence_library.shared.v2.direct_record_mapping import (
    ConditionKey,
    condition_key,
    direct_mapping_row,
)
from data.processing.evidence_library.versions.v10.tasks.bioavailability_ma.canonical_source import (
    DIRECT_SOURCE_ROWS_PATH,
)
VOTER_MEMBERSHIP = Path(
    "data/gold_labels/Bioavailability_Ma/v1/scaffold/voter_membership.parquet"
)
DIRECT_MAPPING_INPUTS = (
    DIRECT_SOURCE_ROWS_PATH,
    VOTER_MEMBERSHIP,
)
DIRECT_GROUP_ID = "Observed.direct_oral_bioavailability"


def build_direct_record_mapping(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    direct_source_ids, voters = _mapping_inputs()
    output: list[dict[str, Any]] = []
    for record in records:
        source_id = str(record.get("source_id") or "")
        canonical_source_id = _canonical_source_record_id(record)
        voter = voters.get(str(record.get("source_row_uid") or ""))
        is_direct_source = voter is not None or source_id == "hf_bioavailability" or (
            source_id == "oral_exposure" and canonical_source_id in direct_source_ids
        )
        if not is_direct_source:
            continue
        if voter is not None:
            group = str(voter["condition_group"])
            external = group != "no_reported_external_condition"
            condition = ConditionKey(
                group=group,
                atoms=tuple(sorted(group.split("+"))) if external else (),
                scope="external" if external else "none_reported",
                status="selected_reviewed" if external else "none_reported",
            )
            label = int(voter["vote_label"])
            counted = True
            reason = "gold_v1_physical_voter"
        else:
            condition = condition_key(
                canonical_record_id=str(record.get("canonical_record_id") or ""),
                condition_text=record.get("qualifying_conditions"),
                reviewed=None,
                proposal=None,
                selected_groups=(),
            )
            label = None
            counted = False
            reason = "not_gold_v1_physical_voter"
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
def _mapping_inputs() -> tuple[set[str], dict[str, dict[str, Any]]]:
    direct_source_ids = set(
        pd.read_parquet(DIRECT_SOURCE_ROWS_PATH, columns=["source_record_id"])[
            "source_record_id"
        ].astype(str)
    )
    membership = pd.read_parquet(
        VOTER_MEMBERSHIP,
        columns=["source_row_uid", "vote_label", "condition_group"],
    )
    if membership["source_row_uid"].duplicated().any():
        raise ValueError("Gold-v1 voter membership must be unique by physical UID")
    return direct_source_ids, {
        str(row["source_row_uid"]): row for row in membership.to_dict(orient="records")
    }


def _canonical_source_record_id(record: Mapping[str, Any]) -> str:
    source_id = str(record.get("source_id") or "")
    source_record_id = str(record.get("source_record_id") or "")
    if source_id == "hf_bioavailability":
        return f"hf:{source_record_id}"
    if source_id == "oral_exposure":
        return f"local:{int(record.get('source_row_number') or 0) - 1}:{source_record_id}"
    return ""


__all__ = [
    "DIRECT_MAPPING_INPUTS",
    "build_direct_record_mapping",
]
