"""TDC-compatible binary-label adapter for the Starling BBB direct dataset."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from data.processing.gold_labels.benchmark_dataset import (
    LabelDecision,
    LabeledSourceRecord,
    accepted,
    classify_interval,
    has_reported_text,
    parse_numeric_interval,
    rejected,
)


SOURCE_DATASET = "starling-labs/BBB"
SOURCE_REVISION = "f50c638621fcc2dedfc9e00f7074f867640efc1b"
TDC_LOGBB_THRESHOLD = -1.0

POSITIVE_LABELS = {
    "permeable",
    "good_penetration",
    "increased_permeability",
    "high_permeability",
}
NEGATIVE_LABELS = {
    "poor_penetration",
    "impermeable",
    "low_permeability",
    "restricted",
}


def load_label_decisions(
    *,
    revision: str = SOURCE_REVISION,
    max_rows: int = 0,
) -> tuple[Iterable[LabelDecision], dict[str, Any]]:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    dataset = load_dataset(SOURCE_DATASET, split="train", revision=revision)
    if max_rows:
        dataset = dataset.select(range(min(max_rows, len(dataset))))
    resolved_revision = HfApi().dataset_info(SOURCE_DATASET, revision=revision).sha
    metadata = {
        "dataset": SOURCE_DATASET,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "split": "train",
        "tdc_compatibility": {
            "positive_label": "BBB permeable/pass",
            "negative_label": "BBB non-permeable/fail",
            "qualitative_rule": "only bbb_permeability_label is a gold outcome; transport labels are context only",
            "numeric_rule": "logBB >= -1 is positive; only explicit logBB is converted",
        },
    }
    return (_label_record(index, row) for index, row in enumerate(dataset)), metadata


def label_record(record: Mapping[str, Any]) -> tuple[int | None, str]:
    """Return a label only when all TDC-compatible signals in the row agree."""
    labels: set[int] = set()
    methods: list[str] = []

    value = _normalized_label(record.get("bbb_permeability_label"))
    if value in POSITIVE_LABELS:
        labels.add(1)
        methods.append("bbb_permeability_label:explicit_positive")
    elif value in NEGATIVE_LABELS:
        labels.add(0)
        methods.append("bbb_permeability_label:explicit_negative")

    metric = _normalized_metric(record.get("quant_metric"))
    if metric == "logbb":
        interval = parse_numeric_interval(record.get("quant_value"))
        if interval is not None:
            numeric_label = classify_interval(interval, threshold=TDC_LOGBB_THRESHOLD)
            if numeric_label is not None:
                labels.add(numeric_label)
                methods.append(f"logBB_threshold:{interval.method}")

    if not labels:
        return None, "no_tdc_compatible_qualitative_or_logbb_label"
    if len(labels) > 1:
        return None, "within_record_label_conflict"
    return next(iter(labels)), "+".join(methods)


def _label_record(index: int, row: Mapping[str, Any]) -> LabelDecision:
    if has_reported_text(row.get("qualifying_conditions")):
        return rejected(
            "interpretation_altering_qualifying_conditions",
            source_id=SOURCE_DATASET,
            source_index=index,
            qualifying_conditions=row.get("qualifying_conditions"),
        )
    label, method = label_record(row)
    if label is None:
        return rejected(
            method,
            source_id=SOURCE_DATASET,
            source_index=index,
            permeability_label=row.get("bbb_permeability_label"),
            transport_label=row.get("bbb_transport_label"),
            quant_metric=row.get("quant_metric"),
            quant_value=row.get("quant_value"),
        )
    smiles = str(row.get("smiles") or "").strip()
    if not smiles:
        return rejected("missing_smiles", source_id=SOURCE_DATASET, source_index=index)
    return accepted(
        LabeledSourceRecord(
            smiles=smiles,
            label=label,
            source_id=SOURCE_DATASET,
            source_record_id=f"row:{index}",
            pmid=str(row.get("pmid") or ""),
            label_method=method,
            raw_value=" | ".join(
                str(value)
                for value in (
                    row.get("bbb_permeability_label"),
                    row.get("bbb_transport_label"),
                    row.get("quant_metric"),
                    row.get("quant_value"),
                )
                if value not in (None, "")
            ),
            context=str(row.get("qualifying_conditions") or ""),
        )
    )


def _normalized_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _normalized_metric(value: Any) -> str:
    return "".join(character for character in str(value or "").lower() if character.isalnum())
