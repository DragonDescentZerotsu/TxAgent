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

from data.processing.paths import ARTIFACTS_ROOT, raw_starling_task_root

HF_SOURCE_DATASET = "starling-labs/Oral_Bioavailability"
HF_SOURCE_REVISION = "01bbe3ee9cdd3dc081c39973529c9da0c814d465"
RAW_LOCAL_SOURCE_PATH = Path(
    raw_starling_task_root("bioavailability_ma")
    / "Oral_AUC-Cmax_Exposure/extractions.parquet"
)

CANONICAL_VERSION = "bioavailability_canonical_direct.v2"
NONDIRECT_MEASUREMENT_EXTRACTION_VERSION = (
    "bioavailability_nondirect_measurement_extraction.v2"
)
DIRECT_MEASUREMENT_EXTRACTION_VERSION = (
    "bioavailability_direct_measurement_extraction.v1"
)
CANONICAL_SOURCE_DIR = (
    ARTIFACTS_ROOT
    / "starling/bioavailability_ma/canonical_sources/canonical_direct_v2"
)
HF_SNAPSHOT_PATH = CANONICAL_SOURCE_DIR / "hf_oral_bioavailability_snapshot.parquet"
HF_NONDIRECT_RECORDS_PATH = CANONICAL_SOURCE_DIR / "hf_nondirect_records.parquet"
DIRECT_SOURCE_ROWS_PATH = CANONICAL_SOURCE_DIR / "direct_source_rows.parquet"
DIRECT_CLAIMS_PATH = CANONICAL_SOURCE_DIR / "direct_claims.parquet"
DIRECT_REJECTED_ROWS_PATH = CANONICAL_SOURCE_DIR / "direct_rejected_rows.parquet"
DEDUP_AUDIT_PATH = CANONICAL_SOURCE_DIR / "cross_source_dedup_audit.parquet"
LOCAL_PARTITION_AUDIT_PATH = CANONICAL_SOURCE_DIR / "local_partition_audit.parquet"
MANIFEST_PATH = CANONICAL_SOURCE_DIR / "merge_manifest.json"

RESIDUAL_SOURCE_DIR = (
    ARTIFACTS_ROOT
    / "starling/bioavailability_ma/canonical_sources/oral_exposure_residual_v2"
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
_DIRECTION_OR_COMPARISON = re.compile(
    r"\b(?:relative|compared|versus|vs\.?|by|than|"
    r"increase(?:d|s|ing)?|improve(?:d|s|ing)?|enhance(?:d|s|ing)?|"
    r"higher|greater|decrease(?:d|s|ing)?|reduc(?:e|ed|es|ing|tion)|"
    r"lower|less)\b",
    re.IGNORECASE,
)
_SIGNED_POINT = re.compile(r"(?:^|[\s:(])[-+]\s*(?=\d)")
_ATOMIC_NUMBER = (
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?(?:[eE][-+]?\d+)?"
    r"|\.\d+(?:[eE][-+]?\d+)?"
)
_ATOMIC_APPROX = r"(?P<prefix>≈|~|about\s+|approximately\s+|approximate\s+)?"
_ATOMIC_POINT = (
    rf"{_ATOMIC_APPROX}(?P<value>{_ATOMIC_NUMBER})"
    rf"(?:\s*(?:±|\+/-)\s*(?P<variation>{_ATOMIC_NUMBER}))?"
)
_ATOMIC_PERCENT = re.compile(
    rf"^\s*{_ATOMIC_POINT}\s*(?:%|per\s*cent|percent(?:age)?)\s*$",
    re.IGNORECASE,
)
_ATOMIC_PERCENT_REPEATED = re.compile(
    rf"^\s*{_ATOMIC_APPROX}(?P<value>{_ATOMIC_NUMBER})\s*"
    rf"(?:%|per\s*cent|percent(?:age)?)\s*(?:±|\+/-)\s*"
    rf"(?P<variation>{_ATOMIC_NUMBER})\s*"
    rf"(?:%|per\s*cent|percent(?:age)?)\s*$",
    re.IGNORECASE,
)
_ATOMIC_FOLD = re.compile(
    rf"^\s*{_ATOMIC_POINT}\s*(?:-?\s*fold|times?|[x×])\s*$",
    re.IGNORECASE,
)
_ATOMIC_RATIO_SUFFIX = re.compile(
    rf"^\s*{_ATOMIC_POINT}\s*ratio\s*$", re.IGNORECASE
)
_ATOMIC_RATIO_PREFIX = re.compile(
    rf"^\s*ratio\s*(?:of|=|:)?\s*{_ATOMIC_POINT}\s*$", re.IGNORECASE
)
_ATOMIC_FRACTION = re.compile(
    rf"^\s*{_ATOMIC_POINT}\s*(?:unitless\s+)?fraction\s*$",
    re.IGNORECASE,
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
    """Extract one complete, unsigned scalar/unit pair from nondirect HF text."""
    raw = _text(value)
    if not raw:
        return {
            "measurement_text": "",
            "numeric_value": None,
            "value_units": "",
            "measurement_unit_extraction_status": "missing_value",
        }
    if _SIGNED_POINT.search(raw):
        return _unresolved_measurement(raw, "signed_value_not_atomic")
    if _DIRECTION_OR_COMPARISON.search(raw):
        return _unresolved_measurement(raw, "directional_or_comparative")
    unit_present = bool(
        _NONDIRECT_PERCENT_UNIT.search(raw)
        or _NONDIRECT_FOLD_UNIT.search(raw)
        or _NONDIRECT_RATIO_UNIT.search(raw)
    )
    if not unit_present:
        return {
            "measurement_text": raw,
            "numeric_value": None,
            "value_units": "",
            "measurement_unit_extraction_status": "no_explicit_unit",
        }
    for unit, pattern in (
        ("%", _ATOMIC_PERCENT_REPEATED),
        ("%", _ATOMIC_PERCENT),
        ("fold", _ATOMIC_FOLD),
        ("ratio", _ATOMIC_RATIO_SUFFIX),
        ("ratio", _ATOMIC_RATIO_PREFIX),
    ):
        match = pattern.fullmatch(raw)
        if match is not None:
            return _resolved_measurement(match, unit)
    return _unresolved_measurement(raw, "non_atomic_or_qualitative")


def direct_measurement_fields(value: Any) -> dict[str, Any]:
    """Require an explicit percent or fraction unit inside a direct HF value."""
    raw = _text(value)
    if not raw:
        return _unresolved_measurement(raw, "missing_value")
    if _SIGNED_POINT.search(raw):
        return _unresolved_measurement(raw, "signed_value_not_atomic")
    if _DIRECTION_OR_COMPARISON.search(raw):
        return _unresolved_measurement(raw, "directional_or_comparative")
    for unit, pattern in (
        ("%", _ATOMIC_PERCENT_REPEATED),
        ("%", _ATOMIC_PERCENT),
        ("fraction", _ATOMIC_FRACTION),
    ):
        match = pattern.fullmatch(raw)
        if match is not None:
            return _resolved_measurement(match, unit)
    has_explicit_unit = bool(
        _NONDIRECT_PERCENT_UNIT.search(raw) or re.search(r"\bfraction\b", raw, re.I)
    )
    return _unresolved_measurement(
        raw,
        "non_atomic_or_qualitative" if has_explicit_unit else "no_explicit_unit",
    )


def _resolved_measurement(match: re.Match[str], unit: str) -> dict[str, Any]:
    prefix = (match.groupdict().get("prefix") or "").strip()
    value = match.group("value")
    variation = match.groupdict().get("variation")
    measurement = f"{prefix}{value}" if prefix in {"≈", "~"} else (
        f"≈{value}" if prefix else value
    )
    if variation is not None:
        measurement += f" ± {variation}"
    return {
        "measurement_text": measurement,
        "numeric_value": float(value.replace(",", "")),
        "value_units": unit,
        "measurement_unit_extraction_status": "explicit_atomic_scalar_unit",
    }


def _unresolved_measurement(raw: str, status: str) -> dict[str, Any]:
    return {
        "measurement_text": raw,
        "numeric_value": None,
        "value_units": "",
        "measurement_unit_extraction_status": status,
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
