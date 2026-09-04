"""Skin v9 deterministic routing and one-value model resolution."""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from data.processing.paths import evidence_library_root
from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v9.measurement_routing import (
    RouteDecision,
    SourceRoutingRules,
    has_digit,
    is_pure_number,
    measurement_source_text,
)
from data.processing.evidence_library.versions.v9.prompts import PROMPT_ROOT
from data.processing.evidence_library.versions.v9.unit_vocabulary import (
    DEFAULT_VOCABULARY_PATH,
    VOCABULARY_VERSION,
    clean_unit,
    load_unit_vocabulary,
)


TASK_ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = PROMPT_ROOT / "measurement_resolution/skin_v8.jinja"

PROMPT_VERSION = "skin_reaction_measurement_resolution_prompt.v8"
MAPPING_VERSION = "skin_reaction_measurement_resolution.v4"
MAX_MEASUREMENTS_PER_ROW = 1
BATCH_SIZE = 20
ALLOW_REBATCH_UNATTEMPTED = True
STRATIFY_BATCHES = True
REASONING_EFFORT = "low"
RETRY_TERMINAL_FAILURES_ON_UNMETERED = True
MAX_UNMETERED_ATTEMPTS_PER_ROW = 5
_ROUTED_SOURCE_IDS = (
    "direct_skin_reaction",
    "sensitization_aop",
    "skin_exposure",
    "phototoxicity_irritation_local_damage",
)
SOURCE_IDS = _ROUTED_SOURCE_IDS
_EMBEDDED_UNIT_SOURCES = frozenset(
    {"direct_skin_reaction", "phototoxicity_irritation_local_damage"}
)

DEFAULT_CLEANED_RECORDS = (
    evidence_library_root("skin_reaction", "v9") / "01_cleaned/records.parquet"
)
DEFAULT_CANONICAL_RECORDS = DEFAULT_CLEANED_RECORDS
DEFAULT_MAPPING_PATH = (
    TASK_ROOT / "data_processing/measurement_resolution_v4/measurement_resolution.parquet"
)
DEFAULT_BASE_MAPPING_PATH = None
DEFAULT_PROFILE_PATH = DEFAULT_CLEANED_RECORDS.parent / "endpoint_unit_profile.json"
EXACT_UNIT_MAPPING_PATH = (
    TASK_ROOT / "data_processing/canonicalization_v7/exact_measurement_unit_map.v2.json"
)
RULE_POLICY_VERSION = "skin_reaction_measurement_resolution_rules.v1"
EXPECTED_EXTRACTION_ROWS = 97_812
EXPECTED_INFERENCE = {
    "gpt-5.4-mini": "https://api.openai.com/v1",
    "openai/gpt-5.6-luna": "https://openrouter.ai/api/v1",
}

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_UNCERTAINTY = r"(?:±|\+/-)"
_FACTOR = r"(?:[x×*]\s*)10\s*(?:(?:\^|\*\*)\s*)?[+-]?\d+"
_PERCENT = r"(?:%|％|percent|per\s+cent)"
_PERCENT_UNIT = re.compile(rf"^{_PERCENT}$", re.IGNORECASE)
_NUMBER_PERCENT = re.compile(
    rf"^\s*(?P<value>{_NUMBER})\s*{_PERCENT}\s*$", re.IGNORECASE
)
_POINT_SPREAD = re.compile(
    rf"^\s*(?P<value>{_NUMBER})\s*{_UNCERTAINTY}\s*{_NUMBER}\s*$"
)
_PERCENT_SPREAD = re.compile(
    rf"^\s*(?P<value>{_NUMBER})\s*{_PERCENT}\s*{_UNCERTAINTY}"
    rf"\s*{_NUMBER}\s*{_PERCENT}\s*$",
    re.IGNORECASE,
)
_RELATIVE_SPREAD = re.compile(
    rf"^\s*(?P<value>{_NUMBER})\s*{_UNCERTAINTY}\s*{_NUMBER}"
    rf"\s*{_PERCENT}\s*$",
    re.IGNORECASE,
)
_SCALED_NUMBER = re.compile(
    rf"^\s*(?P<value>{_NUMBER})(?:\s*{_UNCERTAINTY}\s*{_NUMBER})?"
    rf"\s*(?P<factor>{_FACTOR})\s*$",
    re.IGNORECASE,
)
_VALUE_WITH_UNIT = re.compile(
    rf"^\s*(?P<value>{_NUMBER})"
    rf"(?:\s*{_UNCERTAINTY}\s*{_NUMBER})?"
    rf"(?:\s*(?P<factor>{_FACTOR}))?\s*"
    rf"(?P<unit>\D\S*(?:\s+.*\S)?)\s*$",
    re.IGNORECASE,
)
_HARD_BOUND = re.compile(
    rf"^\s*(?:<|>|≤|≥|at\s+(?:least|most)|up\s+to|"
    rf"no\s+(?:less|more)\s+than)\s*{_NUMBER}\s*(?P<unit>.*\S)?\s*$",
    re.IGNORECASE,
)
_RANGE = re.compile(
    rf"^\s*{_NUMBER}\s*(?:-|–|—|to)\s*{_NUMBER}"
    rf"\s*(?P<unit>.*\S)?\s*$",
    re.IGNORECASE,
)
_STANDALONE_NUMBER = re.compile(
    rf"(?<![A-Za-z0-9.]){_NUMBER}(?![A-Za-z0-9.])"
)
_POWER_OF_TEN = re.compile(_FACTOR, re.IGNORECASE)
_FACTOR_EXPONENT = re.compile(
    r"10\s*(?:(?:\^|\*\*)\s*)?(?P<exponent>[+-]?\d+)$", re.IGNORECASE
)
_SPREAD = re.compile(rf"{_UNCERTAINTY}\s*{_NUMBER}\s*%?")
_IDENTIFIER = re.compile(
    r"\b(?:[A-Za-z]+\d[A-Za-z0-9]*|[A-Za-z][A-Za-z0-9]*(?:-\d[A-Za-z0-9]*)+)\b"
)
_DIRECT_PERCENT = re.compile(
    rf"^\s*(?P<value>{_NUMBER})\s*%\s*(?:positive(?:\s+reactions?)?|"
    rf"positivity\s+rate|prevalence|contact[-‐ ]allergy\s+frequency|"
    rf"of\s+(?:the\s+)?animals\s+sensitized|sensitized)\s*$",
    re.IGNORECASE,
)
_FOLD = re.compile(
    rf"^\s*(?P<value>{_NUMBER})\s*(?:-?fold(?:\s+(?:higher|increase|greater))?"
    rf"|times(?:\s+(?:greater|higher|more(?:\s+(?:permeable|penetrating|"
    rf"absorbed))?))?)\s*$",
    re.IGNORECASE,
)
_SCORE = re.compile(
    rf"^\s*(?:(?:all|mean|average|median|overall|total|cumulative|contact|skin|"
    rf"reaction|response|erythema|edema|irritation|irritancy|phototoxicity|"
    rf"photoallergy|I\.S\.|CS)\s+)*scores?\s*(?:=|:)?\s*(?P<value>{_NUMBER})"
    rf"(?:\s*{_UNCERTAINTY}\s*{_NUMBER})?[^0-9]*$",
    re.IGNORECASE,
)
_DIRECT_PERCENT_ENDPOINTS = frozenset(
    {
        "allergic_contact_dermatitis_contact_allergy",
        "photocontact_photoallergy",
        "sensitization",
    }
)
_STIMULATION_INDEX_ENDPOINTS = frozenset(
    {
        "cd86 stimulation index",
        "il-18 stimulation index",
        "il-2 stimulation index",
        "stimulation index",
        "lymph node cell number stimulation index",
        "lymph node cell proliferation stimulation index",
        "lymph node weight stimulation index",
        "lymph node cellularity stimulation index",
        "t-cell proliferation stimulation index",
        "lymphocyte proliferation stimulation index",
        "splenocyte stimulation index",
    }
)
_POTENCY_IDENTIFIER = re.compile(r"\b(?:EC|IC|ED)\d+(?:\.\d+)?\b", re.IGNORECASE)
_COUNT_NOUN_UNIT = re.compile(
    r"^(?:animal|case|count|individual|mouse|mice|participant|patient|people|"
    r"person|reaction|subject|volunteer)s?$",
    re.IGNORECASE,
)


def _load_safe_units() -> frozenset[str]:
    vocabulary = json.loads(DEFAULT_VOCABULARY_PATH.read_text(encoding="utf-8"))
    observed = {
        clean_unit(row["unit"])
        for row in vocabulary["units"]
        if any(
            str(source).startswith("skin_reaction:")
            for source in row.get("sources", [])
        )
    }
    mapping = json.loads(EXACT_UNIT_MAPPING_PATH.read_text(encoding="utf-8"))
    mapped = {clean_unit(row["input_unit"]) for row in mapping["entries"]}
    return frozenset(observed & mapped)


_SAFE_UNITS = _load_safe_units()


def _text(record: Mapping[str, Any], field: str) -> str:
    return str(record.get(field) or "").strip()


def _safe_unit(record: Mapping[str, Any]) -> str:
    unit = clean_unit(record.get("unit_text"))
    return unit if unit in _SAFE_UNITS else ""


def _accept(
    rule_id: str, value: str, unit: str, *, canonical: bool = False
) -> RouteDecision:
    return RouteDecision("accept", rule_id, value, unit, canonical)


def _percent_unit_or_blank(record: Mapping[str, Any], unit: str) -> bool:
    return not _text(record, "unit_text") or (
        bool(unit) and _PERCENT_UNIT.fullmatch(unit) is not None
    )


def _is_count_noun_unit(unit: str) -> bool:
    return _COUNT_NOUN_UNIT.fullmatch(unit) is not None


def _bound_or_range_has_safe_unit(
    match: re.Match[str], record: Mapping[str, Any]
) -> bool:
    embedded = clean_unit(match.group("unit"))
    separate = _safe_unit(record)
    if not embedded:
        return not _text(record, "unit_text") or bool(separate)
    return embedded in _SAFE_UNITS and (not separate or embedded == separate)


def _candidate_coefficients(text: str) -> list[str]:
    text = _IDENTIFIER.sub("", text)
    text = _SPREAD.sub("", text)
    text = _POWER_OF_TEN.sub("", text)
    return [match.group() for match in _STANDALONE_NUMBER.finditer(text)]


def _reviewed_scaled_unit(factor: str, unit: str) -> str:
    exponent = _FACTOR_EXPONENT.search(factor)
    candidates = []
    if exponent is not None:
        candidates.append(f"10^{int(exponent.group('exponent'))} {unit}")
    candidates.append(clean_unit(f"{factor} {unit}"))
    return next((candidate for candidate in candidates if candidate in _SAFE_UNITS), "")

def source_routing_rules() -> dict[str, SourceRoutingRules]:
    return {
        source_id: SourceRoutingRules(
            source_id=source_id,
            unit_field="" if source_id in _EMBEDDED_UNIT_SOURCES else "unit_text",
        )
        for source_id in _ROUTED_SOURCE_IDS
    }


def source_exact_route(record: Mapping[str, Any]) -> RouteDecision | None:
    """Return an exact incidence fraction from explicit source count columns."""
    if str(record.get("source_id") or "") != "direct_skin_reaction":
        return None
    try:
        positive = Decimal(str(record.get("positive_count")).strip())
        total = Decimal(str(record.get("total_tested")).strip())
    except (InvalidOperation, ValueError):
        return None
    if (
        not positive.is_finite()
        or not total.is_finite()
        or positive != positive.to_integral_value()
        or total != total.to_integral_value()
        or total <= 0
        or positive < 0
        or positive > total
    ):
        return None
    fraction = format((positive / total).normalize(), "f")
    return RouteDecision(
        "accept",
        "structured_positive_count_fraction.v1",
        fraction,
        "fraction",
        True,
    )


def route_measurement(record: Mapping[str, Any]) -> RouteDecision:
    """Apply only reviewed, high-precision Skin scalar rules.

    A rule copies one source coefficient; all semantic conversion remains in the
    exact Stage-02 unit map. Numeric prose that is not settled here is deliberately
    left for the deferred extraction stage.
    """
    structured = source_exact_route(record)
    if structured is not None:
        return structured

    measurement = _text(record, "measurement_text")
    unit = _safe_unit(record)
    source_id = _text(record, "source_id")
    endpoint = _text(record, "canonical_endpoint_name")

    match = _DIRECT_PERCENT.fullmatch(measurement)
    if (
        match is not None
        and source_id == "direct_skin_reaction"
        and endpoint in _DIRECT_PERCENT_ENDPOINTS
    ):
        return _accept(
            "direct_outcome_percent.v1", match.group("value"), "%", canonical=True
        )

    if source_id == "skin_exposure" and endpoint == "relative_penetration":
        match = _FOLD.fullmatch(measurement)
        if match is not None:
            return _accept("relative_penetration_fold.v1", match.group("value"), "fold")
        if is_pure_number(record.get("measurement_text")) and unit.lower() in {
            "fold",
            "times",
            "x",
            "×",
        }:
            return _accept(
                "relative_penetration_separate_fold_unit.v1",
                measurement_source_text(record.get("measurement_text")) or "",
                unit,
            )

    match = _SCORE.fullmatch(measurement)
    if match is not None and source_id in {
        "direct_skin_reaction",
        "phototoxicity_irritation_local_damage",
    }:
        return _accept("explicit_score.v1", match.group("value"), "score")

    if (
        source_id == "sensitization_aop"
        and endpoint in _STIMULATION_INDEX_ENDPOINTS
        and not _text(record, "unit_text")
        and is_pure_number(record.get("measurement_text"))
    ):
        value = measurement_source_text(record.get("measurement_text")) or ""
        support_pattern = re.compile(
            rf"(?:stimulation\s+index|\bSI\b)(?:\s*\([^)]*\))?"
            rf"\s*(?:value\s*)?(?:of|=|was|is|:)?\s*"
            rf"(?<![\d.]){re.escape(value)}(?!\d)",
            re.IGNORECASE,
        )
        if support_pattern.search(_text(record, "support_text")):
            return _accept("support_confirmed_stimulation_index.v1", value, "SI")

    match = _NUMBER_PERCENT.fullmatch(measurement)
    if match is not None and _percent_unit_or_blank(record, unit):
        return _accept(
            "number_percent_literal.v1", match.group("value"), "%", canonical=True
        )

    match = _PERCENT_SPREAD.fullmatch(measurement)
    if match is not None and _percent_unit_or_blank(record, unit):
        return _accept(
            "percent_with_uncertainty.v1", match.group("value"), "%", canonical=True
        )

    match = _RELATIVE_SPREAD.fullmatch(measurement)
    if match is not None:
        if _percent_unit_or_blank(record, unit):
            return _accept(
                "percent_with_relative_uncertainty.v1",
                match.group("value"),
                "%",
                canonical=True,
            )
        if unit and not _is_count_noun_unit(unit):
            return _accept(
                "point_with_relative_uncertainty_and_unit.v1",
                match.group("value"),
                unit,
            )

    match = _HARD_BOUND.fullmatch(measurement)
    if match is not None and _bound_or_range_has_safe_unit(match, record):
        return RouteDecision("reject", "hard_bound_no_point.v1")
    match = _RANGE.fullmatch(measurement)
    if match is not None and _bound_or_range_has_safe_unit(match, record):
        return RouteDecision("reject", "explicit_range_no_point.v1")

    if (
        unit
        and not _is_count_noun_unit(unit)
        and is_pure_number(record.get("measurement_text"))
    ):
        return _accept(
            "finite_number_with_separate_unit.v1",
            measurement_source_text(record.get("measurement_text")) or "",
            unit,
        )

    match = _POINT_SPREAD.fullmatch(measurement)
    if match is not None and unit and not _is_count_noun_unit(unit):
        return _accept(
            "point_with_uncertainty_separate_unit.v1", match.group("value"), unit
        )

    match = _SCALED_NUMBER.fullmatch(measurement)
    if match is not None and unit:
        scaled_unit = _reviewed_scaled_unit(match.group("factor"), unit)
        if scaled_unit:
            return _accept(
                "scaled_number_with_separate_unit.v1",
                match.group("value"),
                scaled_unit,
            )

    match = _VALUE_WITH_UNIT.fullmatch(measurement)
    if match is not None:
        embedded_unit = clean_unit(match.group("unit"))
        combined_unit = (
            _reviewed_scaled_unit(match.group("factor"), embedded_unit)
            if match.group("factor")
            else embedded_unit
        )
        if (
            combined_unit in _SAFE_UNITS
            and not _is_count_noun_unit(combined_unit)
            and (not _text(record, "unit_text") or embedded_unit == unit)
        ):
            return _accept(
                "embedded_value_and_reviewed_unit.v1",
                match.group("value"),
                combined_unit,
            )

    if not has_digit(record.get("measurement_text")):
        return RouteDecision("reject", "no_digit_in_measurement_column.v1")
    if _POTENCY_IDENTIFIER.search(measurement):
        return RouteDecision("extract")
    coefficients = _candidate_coefficients(measurement)
    if not coefficients:
        return RouteDecision("reject", "no_standalone_numeric_measurement.v1")
    if len(coefficients) > 1:
        return RouteDecision("reject", "multiple_numeric_candidates.v1")
    return RouteDecision("extract")


def prompt_row_fields(source_id: str) -> tuple[str, ...]:
    if source_id not in SOURCE_IDS:
        raise ValueError(f"unknown source_id={source_id!r}")
    if source_id in _EMBEDDED_UNIT_SOURCES:
        return ("endpoint_name", "measurement_text", "support_text")
    return ("endpoint_name", "measurement_text", "unit_text", "support_text")


def canonical_endpoint_record(record: Mapping[str, Any]) -> str:
    """Read the final endpoint materialized by the authoritative clean stage."""
    endpoint = record.get("canonical_endpoint_name")
    if not endpoint:
        raise ValueError(
            "Skin measurement resolution requires Stage-01 canonical_endpoint_name"
        )
    return str(endpoint)


def canonical_endpoint_name(source_id: str, endpoint_name: object) -> str:
    """Reject the legacy partial-row fallback for Skin endpoint identity."""
    del source_id, endpoint_name
    raise ValueError("Skin endpoint identity must come from Stage-01 canonical_endpoint_name")


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_PATH.parent)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def _source_kind(source_id: str) -> str:
    return {
        "direct_skin_reaction": "direct skin sensitization or contact-allergy outcome",
        "sensitization_aop": "sensitization adverse-outcome-pathway readout",
        "phototoxicity_irritation_local_damage": "phototoxicity, irritation, or local-damage outcome",
        "skin_exposure": "skin exposure, absorption, permeation, or retention readout",
    }[source_id]


def _base_render(source_id: str) -> str:
    return _environment().get_template(TEMPLATE_PATH.name).render(
        batch_size=BATCH_SIZE,
        row_fields=list(prompt_row_fields(source_id)),
        has_unit_column=source_id not in _EMBEDDED_UNIT_SOURCES,
        source_kind=_source_kind(source_id),
    )


def render_prompt(
    source_id: str,
    *,
    batch_size: int = BATCH_SIZE,
    endpoint_profiles: tuple[str, ...] = (),
) -> str:
    del endpoint_profiles
    if source_id not in SOURCE_IDS:
        raise ValueError(f"unknown source_id={source_id!r}")
    if batch_size != BATCH_SIZE:
        raise ValueError(f"Skin measurement batches are frozen at {BATCH_SIZE}")
    return _base_render(source_id)


def prompt_manifest(*, batch_size: int = BATCH_SIZE) -> dict[str, object]:
    return {
        "prompt_version": PROMPT_VERSION,
        "template_path": str(TEMPLATE_PATH),
        "template_sha256": hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest(),
        "batch_size": batch_size,
        "maximum_measurements_per_row": MAX_MEASUREMENTS_PER_ROW,
        "routing_rule_policy_version": RULE_POLICY_VERSION,
        "stratified_batch_order": STRATIFY_BATCHES,
        "unit_vocabulary": {
            "version": VOCABULARY_VERSION,
            "path": str(DEFAULT_VOCABULARY_PATH),
            "sha256": file_sha256(DEFAULT_VOCABULARY_PATH),
            "units": len(load_unit_vocabulary()),
        },
        "source_measurement_fields": {
            source_id: {
                "measurement_field": "measurement_text",
                "unit_field": (
                    None if source_id in _EMBEDDED_UNIT_SOURCES else "unit_text"
                ),
            }
            for source_id in SOURCE_IDS
        },
        "rendered_sha256": {
            source_id: hashlib.sha256(render_prompt(source_id).encode()).hexdigest()
            for source_id in SOURCE_IDS
        },
    }


def validate_mapping_provenance(mapping_path: str | Path) -> None:
    """Reject a stale, partial, or incorrectly attributed Skin extraction."""
    path = Path(mapping_path)
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError(f"Skin v9 extraction or manifest not found: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "task_id": "skin_reaction",
        "mapping_version": MAPPING_VERSION,
        "model": "mixed",
        "api_base_url": "mixed",
        "mapping_rows": EXPECTED_EXTRACTION_ROWS,
        "cleaned_records_sha256": file_sha256(DEFAULT_CLEANED_RECORDS),
        "mapping_sha256": file_sha256(path),
    }
    mismatches = {
        key: {"expected": value, "found": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    prompt = manifest.get("prompt") or {}
    if prompt.get("prompt_version") != PROMPT_VERSION:
        mismatches["prompt_version"] = {
            "expected": PROMPT_VERSION,
            "found": prompt.get("prompt_version"),
        }
    inference = manifest.get("inference") or {}
    if inference.get("max_completion_tokens") != 8_192:
        mismatches["max_completion_tokens"] = {
            "expected": 8_192,
            "found": inference.get("max_completion_tokens"),
        }
    if manifest.get("base_mapping") is not None:
        mismatches["base_mapping"] = {
            "expected": None,
            "found": manifest.get("base_mapping"),
        }
    model_counts = manifest.get("inference_model_counts") or {}
    base_url_counts = manifest.get("inference_base_url_counts") or {}
    if (
        set(model_counts) != set(EXPECTED_INFERENCE)
        or any(int(count) <= 0 for count in model_counts.values())
        or sum(int(count) for count in model_counts.values())
        != EXPECTED_EXTRACTION_ROWS
    ):
        mismatches["inference_model_counts"] = {
            "expected_models": sorted(EXPECTED_INFERENCE),
            "expected_rows": EXPECTED_EXTRACTION_ROWS,
            "found": model_counts,
        }
    if set(base_url_counts) != set(EXPECTED_INFERENCE.values()):
        mismatches["inference_base_url_counts"] = {
            "expected": sorted(EXPECTED_INFERENCE.values()),
            "found": base_url_counts,
        }
    validations = manifest.get("validations") or {}
    if validations.get("maximum_measurements_per_row") is not True:
        mismatches["maximum_measurements_per_row"] = {
            "expected": True,
            "found": validations.get("maximum_measurements_per_row"),
        }
    if mismatches:
        raise ValueError(f"Skin v9 extraction provenance mismatch: {mismatches}")


__all__ = [
    "ALLOW_REBATCH_UNATTEMPTED",
    "BATCH_SIZE",
    "DEFAULT_CANONICAL_RECORDS",
    "DEFAULT_CLEANED_RECORDS",
    "DEFAULT_MAPPING_PATH",
    "DEFAULT_PROFILE_PATH",
    "EXACT_UNIT_MAPPING_PATH",
    "EXPECTED_EXTRACTION_ROWS",
    "EXPECTED_INFERENCE",
    "MAPPING_VERSION",
    "MAX_MEASUREMENTS_PER_ROW",
    "MAX_UNMETERED_ATTEMPTS_PER_ROW",
    "PROMPT_VERSION",
    "REASONING_EFFORT",
    "RETRY_TERMINAL_FAILURES_ON_UNMETERED",
    "RULE_POLICY_VERSION",
    "SOURCE_IDS",
    "STRATIFY_BATCHES",
    "canonical_endpoint_name",
    "canonical_endpoint_record",
    "prompt_manifest",
    "prompt_row_fields",
    "render_prompt",
    "route_measurement",
    "source_exact_route",
    "source_routing_rules",
    "validate_mapping_provenance",
]
