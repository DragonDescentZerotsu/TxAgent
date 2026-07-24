"""TDC-compatible binary-label adapter for Starling oral bioavailability data."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from pathlib import Path
import re
from typing import Any

from tools.chembl_tool.common.starling.benchmark_dataset import (
    LabelDecision,
    LabeledSourceRecord,
    accepted,
    classify_interval,
    has_reported_text,
    normalize_numeric_text,
    parse_numeric_interval,
    rejected,
    sha256_file,
)


HF_SOURCE_DATASET = "starling-labs/Oral_Bioavailability"
HF_SOURCE_REVISION = "01bbe3ee9cdd3dc081c39973529c9da0c814d465"
LOCAL_SOURCE_PATH = Path(
    "data/starling_data/bioavailability_ma/Oral_AUC-Cmax_Exposure/extractions.parquet"
)
TDC_BIOAVAILABILITY_THRESHOLD_PERCENT = 20.0
DIRECT_REPORT_TYPES = {"absolute", "systemic_availability", "extent_f", "unspecified"}

POSITIVE_QUALITATIVE_PATTERNS = (
    r"\bhigh\b",
    r"\bgood\b",
    r"\bexcellent\b",
    r"\bcomplete(?:ly)?\b",
    r"\bnear(?:ly)? complete\b",
    r"\balmost complete\b",
)
NEGATIVE_QUALITATIVE_PATTERNS = (
    r"\bvery low\b",
    r"\blow\b",
    r"\bpoor\b",
    r"\bnegligible\b",
    r"\bminimal\b",
)
AMBIGUOUS_QUALITATIVE_PATTERNS = (
    r"\bmoderate\b",
    r"\bvariable\b",
    r"\bunpredictable\b",
    r"\bintermediate\b",
    r"\borally bioavailable\b",
    r"\borally available\b",
)
NON_HUMAN_PATTERN = re.compile(
    r"\b(?:rat|rats|mouse|mice|dog|dogs|canine|beagle|pig|pigs|swine|monkey|monkeys|"
    r"macaque|primate|rabbit|rabbits|guinea pig|hamster|sheep|goat|horse|horses|"
    r"bovine|cow|cattle|chicken|fish|zebrafish|rodent|animal|non-human|nonhuman)\b",
    flags=re.IGNORECASE,
)
HUMAN_PATTERN = re.compile(
    r"\b(?:human|humans|subject|subjects|participant|participants|volunteer|volunteers|"
    r"patient|patients|men|women|male adults|female adults|healthy adults|elderly|"
    r"pediatric|paediatric|children|adolescents|infants)\b",
    flags=re.IGNORECASE,
)


def load_label_decisions(
    *,
    revision: str = HF_SOURCE_REVISION,
    local_source_path: str | Path = LOCAL_SOURCE_PATH,
    max_rows: int = 0,
) -> tuple[Iterable[LabelDecision], dict[str, Any]]:
    import pandas as pd
    from datasets import load_dataset
    from huggingface_hub import HfApi

    hf_dataset = load_dataset(HF_SOURCE_DATASET, split="train", revision=revision)
    if max_rows:
        hf_dataset = hf_dataset.select(range(min(max_rows, len(hf_dataset))))

    local_path = Path(local_source_path)
    local_frame = pd.read_parquet(local_path)
    if max_rows:
        local_frame = local_frame.head(max_rows)
    local_records = local_frame.to_dict(orient="records")
    resolved_revision = HfApi().dataset_info(HF_SOURCE_DATASET, revision=revision).sha

    def decisions() -> Iterable[LabelDecision]:
        for index, row in enumerate(hf_dataset):
            yield _label_hf_record(index, row)
        for index, row in enumerate(local_records):
            yield _label_local_record(index, row, source_path=local_path)

    metadata = {
        "sources": [
            {
                "dataset": HF_SOURCE_DATASET,
                "requested_revision": revision,
                "resolved_revision": resolved_revision,
                "split": "train",
            },
            {
                "path": str(local_path),
                "sha256": sha256_file(local_path),
            },
        ],
        "tdc_compatibility": {
            "population_scope": "human only",
            "positive_label": "human oral bioavailability F >= 20%",
            "negative_label": "human oral bioavailability F < 20%",
            "numeric_units": "percent and unitless fraction normalized to percent",
            "qualitative_policy": "only explicit high/good/complete or low/poor/negligible descriptors",
        },
    }
    return decisions(), metadata


def label_bioavailability_value(value: Any) -> tuple[int | None, str]:
    """Map a direct oral-F value to the TDC threshold without hiding ambiguity."""
    text = normalize_numeric_text(value)
    if not text:
        return None, "missing_bioavailability_value"
    lowered = text.lower()
    if re.search(
        r"\b(?:fold|times|relative)\b|\b(?:increase|increased|decrease|decreased)\s+by\b|"
        r"\b(?:higher|lower)\s+than\b",
        lowered,
    ):
        return None, "relative_not_absolute_bioavailability"

    if any(re.search(pattern, lowered) for pattern in AMBIGUOUS_QUALITATIVE_PATTERNS):
        if not re.search(r"\d", text):
            return None, "qualitative_value_not_threshold_anchored"
    if not re.search(r"\d", text):
        positive = any(re.search(pattern, lowered) for pattern in POSITIVE_QUALITATIVE_PATTERNS)
        negative = any(re.search(pattern, lowered) for pattern in NEGATIVE_QUALITATIVE_PATTERNS)
        if positive and not negative:
            return 1, "explicit_qualitative_high"
        if negative and not positive:
            return 0, "explicit_qualitative_low"
        return None, "unmapped_or_ambiguous_qualitative_value"

    interval = parse_numeric_interval(text, fraction_to_percent=True)
    if interval is None:
        return None, "unparseable_numeric_value"
    if interval.lower != -math.inf and interval.lower < 0:
        return None, "numeric_value_out_of_percent_range"
    if interval.upper != math.inf and interval.upper > 100:
        return None, "numeric_value_out_of_percent_range"
    label = classify_interval(interval, threshold=TDC_BIOAVAILABILITY_THRESHOLD_PERCENT)
    if label is None:
        return None, "numeric_interval_crosses_20_percent_threshold"
    return label, f"numeric_20_percent_threshold:{interval.method}"


def is_human_context(value: Any) -> bool:
    text = str(value or "").strip()
    if not text or NON_HUMAN_PATTERN.search(text):
        return False
    return HUMAN_PATTERN.search(text) is not None


def _label_hf_record(index: int, row: Mapping[str, Any]) -> LabelDecision:
    report_type = str(row.get("bioavailability_report_type") or "").strip().lower()
    if report_type not in DIRECT_REPORT_TYPES:
        return rejected(
            "non_direct_oral_bioavailability_report_type",
            source_id=HF_SOURCE_DATASET,
            source_index=index,
            report_type=report_type,
        )
    population = row.get("species_or_population")
    if not is_human_context(population):
        return rejected(
            "nonhuman_or_unresolved_population",
            source_id=HF_SOURCE_DATASET,
            source_index=index,
            population=population,
        )
    if has_reported_text(row.get("qualifying_conditions")):
        return rejected(
            "interpretation_altering_qualifying_conditions",
            source_id=HF_SOURCE_DATASET,
            source_index=index,
            qualifying_conditions=row.get("qualifying_conditions"),
        )
    label, method = label_bioavailability_value(row.get("oral_bioavailability_value"))
    if label is None:
        return rejected(
            method,
            source_id=HF_SOURCE_DATASET,
            source_index=index,
            value=row.get("oral_bioavailability_value"),
        )
    smiles = str(row.get("smiles") or "").strip()
    if not smiles:
        return rejected("missing_smiles", source_id=HF_SOURCE_DATASET, source_index=index)
    return accepted(
        LabeledSourceRecord(
            smiles=smiles,
            label=label,
            source_id=HF_SOURCE_DATASET,
            source_record_id=f"row:{index}",
            pmid=str(row.get("pmid") or ""),
            label_method=f"hf:{method}",
            raw_value=str(row.get("oral_bioavailability_value") or ""),
            context=str(population or ""),
        )
    )


def _label_local_record(
    index: int,
    row: Mapping[str, Any],
    *,
    source_path: Path = LOCAL_SOURCE_PATH,
) -> LabelDecision:
    source_id = f"local:{source_path}"
    if str(row.get("exposure_measure") or "").strip().lower() != "bioavailability":
        return rejected(
            "non_bioavailability_exposure_measure",
            source_id=source_id,
            source_index=index,
            exposure_measure=row.get("exposure_measure"),
        )
    context = row.get("study_context")
    if not is_human_context(context):
        return rejected(
            "nonhuman_or_unresolved_population",
            source_id=source_id,
            source_index=index,
            population=context,
        )
    if has_reported_text(row.get("qualifying_conditions")):
        return rejected(
            "interpretation_altering_qualifying_conditions",
            source_id=source_id,
            source_index=index,
            qualifying_conditions=row.get("qualifying_conditions"),
        )

    value = _float_or_none(row.get("parameter_value"))
    units = str(row.get("parameter_units") or "").strip().lower()
    if value is None:
        return rejected("missing_bioavailability_value", source_id=source_id, source_index=index)
    if units in {"fold", "times", "ratio", "% increase", "% lower"}:
        return rejected("relative_not_absolute_bioavailability", source_id=source_id, source_index=index, units=units)
    if units in {"unitless", "fraction"} or (not units and 0 <= value <= 1.5):
        value_percent = value * 100.0
        method = "local_fraction_to_percent"
    elif units in {"%", "percent", "per cent", ""}:
        value_percent = value
        method = "local_percent"
    else:
        return rejected(
            "unsupported_bioavailability_unit",
            source_id=source_id,
            source_index=index,
            units=units,
        )
    if not 0 <= value_percent <= 100:
        return rejected(
            "numeric_value_out_of_percent_range",
            source_id=source_id,
            source_index=index,
            value=value,
            units=units,
        )
    label = int(value_percent >= TDC_BIOAVAILABILITY_THRESHOLD_PERCENT)
    smiles = str(row.get("smiles") or row.get("SMILES") or "").strip()
    if not smiles:
        return rejected("missing_smiles", source_id=source_id, source_index=index)
    return accepted(
        LabeledSourceRecord(
            smiles=smiles,
            label=label,
            source_id=source_id,
            source_record_id=str(row.get("extraction_id") or f"row:{index}"),
            pmid=str(row.get("pmid") or ""),
            label_method=f"{method}:20_percent_threshold",
            raw_value=f"{value} {units}".strip(),
            context=str(context or ""),
        )
    )


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
