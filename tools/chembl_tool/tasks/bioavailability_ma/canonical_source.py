"""Canonical Bioavailability Starling source paths and row classification.

The original Hugging Face dataset and local oral-exposure parquet are immutable
upstream inputs.  This module defines the versioned derived artifacts shared by
gold-label construction and inference-time evidence ingestion.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import math
import re
from typing import Any


HF_SOURCE_DATASET = "starling-labs/Oral_Bioavailability"
HF_SOURCE_REVISION = "01bbe3ee9cdd3dc081c39973529c9da0c814d465"
RAW_LOCAL_SOURCE_PATH = Path(
    "data/starling_data/bioavailability_ma/Oral_AUC-Cmax_Exposure/extractions.parquet"
)

CANONICAL_VERSION = "bioavailability_canonical_direct.v2"
NONDIRECT_MEASUREMENT_EXTRACTION_VERSION = (
    "bioavailability_nondirect_measurement_extraction.v1"
)
CANONICAL_SOURCE_DIR = Path(
    "data/starling_data/bioavailability_ma/canonical_direct_v2"
)
HF_SNAPSHOT_PATH = CANONICAL_SOURCE_DIR / "hf_oral_bioavailability_snapshot.parquet"
HF_NONDIRECT_RECORDS_PATH = CANONICAL_SOURCE_DIR / "hf_nondirect_records.parquet"
DIRECT_SOURCE_ROWS_PATH = CANONICAL_SOURCE_DIR / "direct_source_rows.parquet"
DIRECT_CLAIMS_PATH = CANONICAL_SOURCE_DIR / "direct_claims.parquet"
DIRECT_REJECTED_ROWS_PATH = CANONICAL_SOURCE_DIR / "direct_rejected_rows.parquet"
DEDUP_AUDIT_PATH = CANONICAL_SOURCE_DIR / "cross_source_dedup_audit.parquet"
LOCAL_PARTITION_AUDIT_PATH = CANONICAL_SOURCE_DIR / "local_partition_audit.parquet"
MANIFEST_PATH = CANONICAL_SOURCE_DIR / "merge_manifest.json"

RESIDUAL_SOURCE_DIR = Path(
    "data/starling_data/bioavailability_ma/oral_exposure_residual_v2"
)
RESIDUAL_RECORDS_PATH = RESIDUAL_SOURCE_DIR / "exposure_records.parquet"
RESIDUAL_MANIFEST_PATH = RESIDUAL_SOURCE_DIR / "partition_manifest.json"

DIRECT_REPORT_TYPES = {
    "absolute",
    "systemic_availability",
    "extent_f",
    "unspecified",
}

LOCAL_PARTITION_DIRECT = "canonical_absolute_bioavailability"
LOCAL_PARTITION_NON_BIOAVAILABILITY = "residual_non_bioavailability_exposure"
LOCAL_PARTITION_RELATIVE = "residual_relative_bioavailability"
LOCAL_PARTITION_AMBIGUOUS = "residual_ambiguous_bioavailability"

_EXPLICIT_ABSOLUTE_PATTERN = re.compile(
    r"\babsolute\s+(?:oral\s+)?bioavailability\b|"
    r"\babsolute\s+systemic\s+availability\b",
    flags=re.IGNORECASE,
)
_IV_PATTERN = re.compile(
    r"\bintravenous\b|\bi\.?v\.?\b|\boral\s*(?:/|[- ]?to[- ]?|vs\.?|versus)\s*i\.?v\.?\b",
    flags=re.IGNORECASE,
)
_RELATIVE_PATTERN = re.compile(
    r"\brelative\s+(?:oral\s+|systemic\s+)?bioavailability\b|"
    r"\brelative\s+availability\b|\bbioequivalen\w*\b|"
    r"\btest\s+(?:and|vs\.?|versus)\s+reference\b|"
    r"\breference\s+(?:formulation|product|tablet|capsule)\b|"
    r"\b(?:fed|food)\s+(?:vs\.?|versus|compared\s+with)\s+(?:fasted|fasting)\b",
    flags=re.IGNORECASE,
)
_RELATIVE_UNITS = {"fold", "times", "ratio", "% increase", "% lower"}
_NONDIRECT_PERCENT_UNIT = re.compile(r"%|\bper\s*cent\b|\bpercent(?:age)?\b", re.IGNORECASE)
_NONDIRECT_FOLD_UNIT = re.compile(
    r"\b(?:fold|times?)\b|(?<=\d)\s*[x×](?=\s|$)", re.IGNORECASE
)
_NONDIRECT_RATIO_UNIT = re.compile(r"\bratio\b", re.IGNORECASE)
_NONDIRECT_DIRECTION_WORDS = re.compile(
    r"\b(?:about|approximately|approximate|reported|relative|apparent|oral|"
    r"bioavailability|was|is|of|by|to|than|compared|versus|vs|"
    r"increase(?:d|s)?|improve(?:d|s)?|enhance(?:d|s)?|higher|greater|"
    r"decrease(?:d|s)?|reduce(?:d|s)?|lower|less)\b",
    re.IGNORECASE,
)
_NONDIRECT_POINT = re.compile(
    r"^[<>≤≥~≈]?\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:\s*(?:±|\+/-)\s*(?:\d+(?:\.\d*)?|\.\d+))?$"
)


def classify_local_record(row: Mapping[str, Any]) -> tuple[str, str]:
    """Partition a local oral-exposure row without treating ambiguous F as absolute."""
    endpoint = _text(row.get("exposure_measure")).lower()
    if endpoint != "bioavailability":
        return LOCAL_PARTITION_NON_BIOAVAILABILITY, "endpoint_is_not_bioavailability"

    combined = " | ".join(
        _text(row.get(field))
        for field in (
            "support_text",
            "comparator_exposure",
            "extra_details",
            "study_context",
        )
    )
    units = _text(row.get("parameter_units")).lower()
    absolute_signal = bool(_EXPLICIT_ABSOLUTE_PATTERN.search(combined) or _IV_PATTERN.search(combined))
    relative_signal = bool(_RELATIVE_PATTERN.search(combined) or units in _RELATIVE_UNITS)
    if absolute_signal and relative_signal:
        return LOCAL_PARTITION_AMBIGUOUS, "conflicting_absolute_and_relative_signals"
    if relative_signal:
        return LOCAL_PARTITION_RELATIVE, "explicit_relative_or_comparator_signal"
    if absolute_signal:
        return LOCAL_PARTITION_DIRECT, "explicit_absolute_or_oral_iv_signal"
    return LOCAL_PARTITION_AMBIGUOUS, "bioavailability_without_absolute_anchor"


def local_value_percent(row: Mapping[str, Any]) -> float | None:
    """Normalize a local numeric F value to percent without inventing a value."""
    value = _float_or_none(row.get("parameter_value"))
    if value is None:
        return None
    units = _text(row.get("parameter_units")).lower()
    if units in {"unitless", "fraction"} or (not units and 0 <= value <= 1.5):
        value *= 100.0
    elif units not in {"%", "percent", "per cent", ""}:
        return None
    if not 0 <= value <= 100:
        return None
    return float(value)


def local_classification_signals(row: Mapping[str, Any]) -> dict[str, bool]:
    """Expose stable audit flags used by the local partition classifier."""
    combined = " | ".join(
        _text(row.get(field))
        for field in (
            "support_text",
            "comparator_exposure",
            "extra_details",
            "study_context",
        )
    )
    units = _text(row.get("parameter_units")).lower()
    return {
        "absolute_signal": bool(_EXPLICIT_ABSOLUTE_PATTERN.search(combined) or _IV_PATTERN.search(combined)),
        "relative_signal": bool(_RELATIVE_PATTERN.search(combined) or units in _RELATIVE_UNITS),
    }


def nondirect_measurement_fields(value: Any) -> dict[str, Any]:
    """Extract only explicit scalar/unit pairs from a non-direct HF value."""
    raw = _text(value)
    if not raw:
        return {
            "measurement_text": "",
            "numeric_value": None,
            "value_units": "",
            "measurement_unit_extraction_status": "missing_value",
        }
    unit = ""
    unit_pattern: re.Pattern[str] | None = None
    if _NONDIRECT_PERCENT_UNIT.search(raw):
        unit, unit_pattern = "%", _NONDIRECT_PERCENT_UNIT
    elif _NONDIRECT_FOLD_UNIT.search(raw):
        unit, unit_pattern = "fold", _NONDIRECT_FOLD_UNIT
    elif _NONDIRECT_RATIO_UNIT.search(raw):
        unit, unit_pattern = "ratio", _NONDIRECT_RATIO_UNIT
    if unit_pattern is None:
        return {
            "measurement_text": raw,
            "numeric_value": None,
            "value_units": "",
            "measurement_unit_extraction_status": "no_explicit_unit",
        }
    measurement = unit_pattern.sub("", raw)
    measurement = re.sub(r"\s+", " ", measurement.replace(",", "")).strip(" -:()")
    atomic_measurement = _NONDIRECT_DIRECTION_WORDS.sub("", measurement)
    atomic_measurement = re.sub(r"\s+", " ", atomic_measurement).strip(" -:()")
    if not _NONDIRECT_POINT.fullmatch(atomic_measurement):
        return {
            "measurement_text": raw,
            "numeric_value": None,
            "value_units": "",
            "measurement_unit_extraction_status": "non_atomic_or_qualitative",
        }
    numeric_match = re.search(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", atomic_measurement
    )
    return {
        "measurement_text": atomic_measurement,
        "numeric_value": float(numeric_match.group(0)) if numeric_match else None,
        "value_units": unit,
        "measurement_unit_extraction_status": "explicit_atomic_scalar_unit",
    }


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
