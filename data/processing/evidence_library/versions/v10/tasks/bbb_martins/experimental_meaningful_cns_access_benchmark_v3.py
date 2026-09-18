"""Corrected BBB gold contract excluding source-native prediction rows.

This adapter is intentionally separate from the frozen v2 implementation.  It
adds the audited prediction-source boundary discovered while rebuilding the
progressive retrieval families, then reuses every other v2 scope and label
rule unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Any

from data.processing.gold_labels.benchmark_dataset import LabelDecision

from .experimental_meaningful_cns_access_benchmark import (
    SOURCE_REVISION,
    label_record as label_record_v2,
    load_label_decisions as load_label_decisions_v2,
)


CONTRACT_VERSION = "bbb_experimental_meaningful_cns_access_gold.v3"
_SOURCE_NATIVE_PREDICTION_PATTERN = re.compile(
    r"\badme/?t analysis\b|\btcmsp database\b",
    re.IGNORECASE,
)


def label_record(
    record: Mapping[str, Any],
    *,
    source_index: int | None = None,
    allow_conditioned_context: bool = False,
) -> tuple[int | None, str]:
    """Apply v2 unchanged after excluding audited computational sources."""

    searchable = " | ".join(
        str(record.get(field) or "")
        for field in (
            "assay_model",
            "support_text",
            "extra_details",
            "quant_metric",
            "quant_value",
        )
    )
    if _SOURCE_NATIVE_PREDICTION_PATTERN.search(searchable):
        return None, "computational_or_predicted_result"
    return label_record_v2(
        record,
        source_index=source_index,
        allow_conditioned_context=allow_conditioned_context,
    )


def load_label_decisions(
    *,
    revision: str = SOURCE_REVISION,
    max_rows: int = 0,
) -> tuple[Iterable[LabelDecision], dict[str, Any]]:
    """Load the frozen Starling revision and apply the corrected v3 contract."""

    decisions, metadata = load_label_decisions_v2(
        revision=revision,
        max_rows=max_rows,
        label_decider=label_record,
        contract_version=CONTRACT_VERSION,
    )
    metadata["gold_contract"]["v3_change_from_v2"] = (
        "exclude source-native ADME/T or TCMSP computational BBB predictions"
    )
    return decisions, metadata
