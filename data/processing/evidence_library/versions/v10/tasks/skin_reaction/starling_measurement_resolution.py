"""Skin v9 deterministic routing and one-value model resolution."""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from data.processing.paths import REPO_ROOT, evidence_library_root
from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v10.measurement_routing import (
    RouteDecision,
    SourceRoutingRules,
    has_digit,
    source_role_contract,
)
from data.processing.evidence_library.versions.v10.numeric_syntax import (
    NUMBER,
    number_text,
)
from data.processing.evidence_library.versions.v10.prompts import PROMPT_ROOT
from data.processing.evidence_library.versions.v10.unit_vocabulary import (
    DEFAULT_VOCABULARY_PATH,
    VOCABULARY_VERSION,
    clean_unit,
    load_unit_vocabulary,
)


TASK_ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = PROMPT_ROOT / "measurement_resolution/skin_v12.jinja"

PROMPT_VERSION = "skin_reaction_measurement_resolution_prompt.v12"
MAPPING_VERSION = "skin_reaction_measurement_resolution.v8"
EXACT_UNIT_MAPPING_VERSION = "starling_exact_measurement_units.v3"
MAX_MEASUREMENTS_PER_ROW = 1
BATCH_SIZE = 20
ALLOW_REBATCH_UNATTEMPTED = True
AUTO_VALIDATE_GENERATED_MAPPING = True
ENFORCE_EXACT_UNITS_DURING_EXTRACTION = False
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
    evidence_library_root("skin_reaction", "v10") / "01_cleaned/records.parquet"
)
DEFAULT_CANONICAL_RECORDS = DEFAULT_CLEANED_RECORDS
DEFAULT_MAPPING_PATH = (
    REPO_ROOT
    / "data/caches/evidence_library/skin_reaction/v10_main_universe_v2"
    / "measurement_resolution/measurement_resolution.parquet"
)
DEFAULT_BASE_MAPPING_PATH = None
DEFAULT_PROFILE_PATH = DEFAULT_CLEANED_RECORDS.parent / "endpoint_unit_profile.json"
DEFAULT_GOLD_FIXTURE = (
    REPO_ROOT
    / "tests/chembl_tool/common/measurement_resolution_quality/gold/skin_reaction.v10.jsonl"
)
ROUTING_EXACT_UNIT_MAPPING_PATH = (
    TASK_ROOT / "data_processing/canonicalization_v7/exact_measurement_unit_map.v3.json"
)
EXACT_UNIT_MAPPING_PATH = (
    REPO_ROOT
    / "data/caches/evidence_library/skin_reaction/v10_main_universe_v2"
    / "unit_reconciliation/exact_measurement_unit_map.json"
)
RULE_POLICY_VERSION = "skin_reaction_measurement_resolution_rules.v1"
PERCENT_UNIT_POSTPROCESS_VERSION = "skin_percent_unit_postprocess.v1"

_NUMBER = NUMBER
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
    # Routing must remain byte-compatible with the extraction cohort.  The
    # reviewed successor map is applied after extraction during Stage 1.
    mapping = json.loads(
        ROUTING_EXACT_UNIT_MAPPING_PATH.read_text(encoding="utf-8")
    )
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
    return RouteDecision("accept", rule_id, number_text(value) or value, unit, canonical)


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
        if number_text(record.get("measurement_text")) and unit.lower() in {
            "fold",
            "times",
            "x",
            "×",
        }:
            return _accept(
                "relative_penetration_separate_fold_unit.v1",
                number_text(record.get("measurement_text")) or "",
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
        and number_text(record.get("measurement_text"))
    ):
        value = number_text(record.get("measurement_text")) or ""
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


def postprocess_extracted_assignment(
    record: Mapping[str, Any], assignment: dict[str, Any]
) -> dict[str, Any]:
    """Collapse percentage descriptors after extraction, before unit mapping."""
    del record
    if assignment.get("status") != "ok":
        return assignment
    measurements = json.loads(str(assignment.get("measurements_json") or "[]"))
    changed = False
    for measurement in measurements:
        unit = str(measurement.get("unit") or "")
        if "%" in unit or "％" in unit:
            measurement["unit"] = "%"
            changed = True
    if not changed:
        return assignment
    assignment["measurements_json"] = json.dumps(measurements, ensure_ascii=False)
    assignment["unit_postprocess_rule_id"] = PERCENT_UNIT_POSTPROCESS_VERSION
    return assignment


def post_extraction_processing_manifest() -> dict[str, object]:
    return {
        "version": PERCENT_UNIT_POSTPROCESS_VERSION,
        "rule": "Any extracted unit containing % or ％ is canonicalized to %.",
        "raw_response_preserved": True,
        "unit_reconciliation": "deferred_to_stage_1c",
    }


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


def source_role_manifest() -> dict[str, object]:
    from data.processing.evidence_library.versions.v10.tasks.skin_reaction.starling_schema import (
        RECORD_CONTRACT,
        UNIT_EXCEPTIONS,
    )

    return source_role_contract(RECORD_CONTRACT.sources, UNIT_EXCEPTIONS)


def validate_mapping_provenance(
    mapping_path: str | Path,
    *,
    expected_record_ids: set[str] | None = None,
) -> None:
    path = Path(mapping_path)
    manifest_path = path.with_suffix(".manifest.json")
    if path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("generation_version") == "stage1_measurement_resolution_replay.v1":
            rows = pq.read_table(path, columns=["cleaned_record_id"]).to_pylist()
            record_ids = [str(row["cleaned_record_id"]) for row in rows]
            digest = hashlib.sha256(
                "\n".join(sorted(record_ids)).encode("utf-8")
            ).hexdigest()
            mismatches: dict[str, object] = {}
            if manifest.get("task_id") != "skin_reaction":
                mismatches["task_id"] = manifest.get("task_id")
            if manifest.get("mapping_version") != MAPPING_VERSION:
                mismatches["mapping_version"] = manifest.get("mapping_version")
            if manifest.get("mapping_sha256") != file_sha256(path):
                mismatches["mapping_sha256"] = manifest.get("mapping_sha256")
            if manifest.get("mapping_rows") != len(record_ids):
                mismatches["mapping_rows"] = manifest.get("mapping_rows")
            if manifest.get("record_ids_sha256") != digest:
                mismatches["record_ids_sha256"] = manifest.get("record_ids_sha256")
            if len(record_ids) != len(set(record_ids)):
                mismatches["unique_record_ids"] = False
            if expected_record_ids is not None and set(record_ids) != expected_record_ids:
                mismatches["expected_record_ids"] = "coverage differs"
            for name, source in (manifest.get("sources") or {}).items():
                source_path = Path(str(source.get("path") or ""))
                if not source_path.is_file() or source.get("sha256") != file_sha256(
                    source_path
                ):
                    mismatches[name] = "missing or hash-mismatched"
            if not all((manifest.get("validations") or {}).values()):
                mismatches["validations"] = manifest.get("validations")
            if mismatches:
                raise ValueError(
                    f"Skin measurement replay provenance mismatch: {mismatches}"
                )
            return
    from data.processing.evidence_library.versions.v10.build_measurement_resolution_mapping import (
        validate_full_mapping_provenance,
    )

    validate_full_mapping_provenance(
        mapping_path,
        task="skin_reaction",
        expected_record_ids=expected_record_ids,
    )


__all__ = [
    "ALLOW_REBATCH_UNATTEMPTED",
    "AUTO_VALIDATE_GENERATED_MAPPING",
    "BATCH_SIZE",
    "DEFAULT_CANONICAL_RECORDS",
    "DEFAULT_CLEANED_RECORDS",
    "DEFAULT_MAPPING_PATH",
    "DEFAULT_PROFILE_PATH",
    "ENFORCE_EXACT_UNITS_DURING_EXTRACTION",
    "EXACT_UNIT_MAPPING_PATH",
    "MAPPING_VERSION",
    "MAX_MEASUREMENTS_PER_ROW",
    "MAX_UNMETERED_ATTEMPTS_PER_ROW",
    "PERCENT_UNIT_POSTPROCESS_VERSION",
    "PROMPT_VERSION",
    "REASONING_EFFORT",
    "RETRY_TERMINAL_FAILURES_ON_UNMETERED",
    "ROUTING_EXACT_UNIT_MAPPING_PATH",
    "RULE_POLICY_VERSION",
    "SOURCE_IDS",
    "STRATIFY_BATCHES",
    "canonical_endpoint_name",
    "canonical_endpoint_record",
    "prompt_manifest",
    "prompt_row_fields",
    "post_extraction_processing_manifest",
    "postprocess_extracted_assignment",
    "render_prompt",
    "route_measurement",
    "source_exact_route",
    "source_routing_rules",
    "source_role_manifest",
    "validate_mapping_provenance",
]
