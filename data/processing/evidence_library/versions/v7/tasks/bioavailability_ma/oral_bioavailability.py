"""Shared cleaning for the pinned Starling oral-bioavailability snapshot."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

from rdkit import Chem

ORAL_BIOAVAILABILITY_DATASET = "starling-labs/Oral_Bioavailability"
ORAL_BIOAVAILABILITY_REVISION = "01bbe3ee9cdd3dc081c39973529c9da0c814d465"
ORAL_BIOAVAILABILITY_SPLIT = "train"
ALLOWED_REPORT_TYPES = frozenset({"absolute", "unspecified", "systemic_availability"})
NULL_STRINGS = {"", "none", "null", "nan", "n/a", "na", "not specified", "unknown"}
QUALITATIVE_REJECT_WORDS = {
    "high", "low", "moderate", "excellent", "complete", "negligible", "similar", "reduced",
    "increased", "increase", "decrease", "unchanged", "variable", "unity",
}
NUMBER_PATTERN = r"(?:\d+(?:\.\d+)?|\.\d+)"
CONDITION_COLUMNS = (
    "species_or_population", "dose", "oral_exposure_mode", "qualifying_conditions", "comparator", "extra_details",
)


@dataclass(frozen=True)
class CleanOralBioavailabilityRow:
    source_index: int
    molecule_name: str
    canonical_smiles: str
    value_percent: float
    condition_text: str
    parse_method: str
    parse_modifier: str
    raw_row: dict[str, Any]


def load_pinned_oral_bioavailability_dataset(
    *, dataset: str = ORAL_BIOAVAILABILITY_DATASET, revision: str = ORAL_BIOAVAILABILITY_REVISION,
    split: str = ORAL_BIOAVAILABILITY_SPLIT,
) -> Any:
    """Load the immutable upstream snapshot; import datasets only when requested."""
    from datasets import load_dataset

    return load_dataset(dataset, revision=revision, split=split)


def clean_oral_bioavailability_rows(
    dataset: Iterable[dict[str, Any]], *, allowed_report_types: set[str] | frozenset[str] = ALLOWED_REPORT_TYPES,
    min_value_percent: float = 0.0, max_value_percent: float = 100.0,
) -> tuple[list[CleanOralBioavailabilityRow], list[dict[str, Any]]]:
    rows: list[CleanOralBioavailabilityRow] = []
    dropped: list[dict[str, Any]] = []
    allowed = {str(item).strip() for item in allowed_report_types}
    for source_index, raw_value in enumerate(dataset):
        raw = dict(raw_value)
        if clean_text(raw.get("bioavailability_report_type")) not in allowed:
            dropped.append(drop_record(source_index, raw, "report_type_not_allowed"))
            continue
        canonical_smiles = canonicalize_smiles(clean_text(raw.get("smiles")))
        if not canonical_smiles:
            dropped.append(drop_record(source_index, raw, "invalid_smiles"))
            continue
        parsed = parse_bioavailability_value(raw.get("oral_bioavailability_value"))
        if parsed is None:
            dropped.append(drop_record(source_index, raw, "unparseable_or_non_numeric_value"))
            continue
        value_percent, parse_method, parse_modifier = parsed
        if not math.isfinite(value_percent):
            dropped.append(drop_record(source_index, raw, "nonfinite_value"))
            continue
        if value_percent < min_value_percent or value_percent > max_value_percent:
            dropped.append(drop_record(source_index, raw, "value_out_of_range"))
            continue
        rows.append(CleanOralBioavailabilityRow(
            source_index=source_index, molecule_name=clean_text(raw.get("molecule_name")),
            canonical_smiles=canonical_smiles, value_percent=value_percent,
            condition_text=build_condition_text(raw), parse_method=parse_method,
            parse_modifier=parse_modifier, raw_row=raw,
        ))
    return rows, dropped


def parse_bioavailability_value(value: Any) -> tuple[float, str, str] | None:
    text = normalize_value_text(value)
    if not text or has_only_qualitative_signal(text):
        return None
    lowered = text.lower()
    modifier = "lower_bound" if re.search(r"(?:>=|≥|at least|greater than|more than|above)", lowered) else ""
    if not modifier and re.search(r"(?:<=|≤|less than|lower than|below|up to)", lowered):
        modifier = "upper_bound"
    mean = re.search(rf"\b(?:mean|average|averaged)\s*[=:]?\s*([<>≤≥~≈]?\s*{NUMBER_PATTERN})", lowered)
    if mean:
        return convert_to_percent(parse_float_token(mean.group(1)), text), "explicit_mean", modifier
    median = re.search(rf"\bmedian\b[^0-9<>≤≥~≈-]*([<>≤≥~≈]?\s*{NUMBER_PATTERN})", lowered)
    if median:
        return convert_to_percent(parse_float_token(median.group(1)), text), "explicit_median", modifier
    plus_minus = re.search(rf"([<>≤≥~≈]?\s*{NUMBER_PATTERN})\s*(?:±|\+/-|\+-|plus/minus)\s*{NUMBER_PATTERN}", text)
    if plus_minus:
        return convert_to_percent(parse_float_token(plus_minus.group(1)), text), "mean_plus_minus", modifier
    ranges = extract_ranges(text)
    if ranges:
        lo, hi = ranges[0]
        return convert_to_percent((lo + hi) / 2.0, text), "range_midpoint", modifier
    numbers = extract_numbers(text)
    return (convert_to_percent(numbers[0], text), "first_numeric_value", modifier) if numbers else None


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in NULL_STRINGS else text


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None else ""


def build_condition_text(row: dict[str, Any]) -> str:
    return "\n".join(f"{field}: {clean_text(row.get(field)) or 'not specified'}" for field in CONDITION_COLUMNS)


def drop_record(source_index: int, raw_row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"source_index": source_index, "drop_reason": reason, "raw_row": raw_row}


def normalize_value_text(value: Any) -> str:
    text = clean_text(value)
    for old, new in {"−": "-", "–": "-", "—": "-", "‐": "-", "‑": "-", "·": ".", "％": "%", "﹪": "%", "per cent": "%", "percent": "%", "approximately": "about", "approx.": "about", "approx": "about"}.items():
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def has_only_qualitative_signal(text: str) -> bool:
    lowered = text.lower()
    if "%" not in lowered and re.search(r"\b(?:auc|fold|higher|lower|comparable|similar|increased|reduced|increase|decrease|unchanged)\b", lowered):
        return True
    return False if re.search(r"\d", lowered) else any(word in lowered for word in QUALITATIVE_REJECT_WORDS)


def parse_float_token(value: str) -> float:
    return float(re.sub(r"[<>≤≥~≈\s]", "", value))


def extract_numbers(text: str) -> list[float]:
    return [float(match.group(0)) for match in re.finditer(rf"(?<![A-Za-z]){NUMBER_PATTERN}", text)]


def extract_ranges(text: str) -> list[tuple[float, float]]:
    result = []
    for pattern in (rf"({NUMBER_PATTERN})\s*(?:-|to|and)\s*({NUMBER_PATTERN})\s*%?", rf"between\s+({NUMBER_PATTERN})\s+and\s+({NUMBER_PATTERN})", rf"range(?:d|s)?(?:\s+between|\s+of|\s*[:=])?\s*({NUMBER_PATTERN})\s*(?:-|to|and)\s*({NUMBER_PATTERN})"):
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            a, b = float(match.group(1)), float(match.group(2))
            result.append((min(a, b), max(a, b)))
    return result


def convert_to_percent(number: float, source_text: str) -> float:
    return number if re.search(r"%|\bper\s*cent\b|percent", source_text, flags=re.IGNORECASE) or number > 1.5 else number * 100.0
