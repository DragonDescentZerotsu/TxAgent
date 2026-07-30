"""Layer 3: atomic measurement/unit normalization and scalar metadata."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from tools.chembl_tool.common.units import (
    canonicalize_measurement,
    canonicalize_unit,
    clean_unit,
)

from .cleaning import clean_measurement_text, clean_text, finite_float, stable_id
from .contracts import (
    NORMALIZATION_STAGE_VERSION,
    NORMALIZED_RECORD_VERSION,
    FamilyAssignment,
    MeasurementPair,
    ParsedPoint,
)


_NUMBER = (
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?(?:[eE][-+]?\d+)?"
    r"|[-+]?\.\d+(?:[eE][-+]?\d+)?"
)
_POINT = re.compile(
    rf"^\s*(?P<prefix>≈|~|about\s+|approx(?:imately)?\.?\s+|ca\.?\s+|estimated\s+)?"
    rf"(?P<value>{_NUMBER})\s*$",
    re.IGNORECASE,
)
_PLUS_MINUS = re.compile(
    rf"^\s*(?P<prefix>≈|~|about\s+|approx(?:imately)?\.?\s+)?"
    rf"(?P<value>{_NUMBER})\s*(?:±|\+/-)\s*(?P<variation>{_NUMBER})\s*$",
    re.IGNORECASE,
)
_PLUS_MINUS_LEADING = re.compile(
    rf"^\s*(?P<prefix>≈|~|about\s+|approx(?:imately)?\.?\s+)?"
    rf"(?P<value>{_NUMBER})\s*(?:±|\+/-)\s*(?P<variation>{_NUMBER})(?P<tail>.*)$",
    re.IGNORECASE,
)
_LEADING_POINT = re.compile(
    rf"^\s*(?P<prefix>≈|~|about\s+|approx(?:imately)?\.?\s+|ca\.?\s+|estimated\s+)?"
    rf"(?P<value>{_NUMBER})(?P<tail>.*)$",
    re.IGNORECASE,
)
_BOUND = re.compile(
    r"^\s*(?:[<>]=?|≥|≤|at least\b|at most\b|no (?:less|more) than\b)",
    re.IGNORECASE,
)
_RANGE = re.compile(
    rf"^\s*(?:{_NUMBER})\s*(?:-|–|—|\bto\b)\s*(?:{_NUMBER})",
    re.IGNORECASE,
)
_INTERVAL = re.compile(
    rf"(?:\b(?:90|95|99)\s*%?\s*)?(?:ci|confidence interval|range)"
    rf"\s*[:=]?\s*[\[(]?\s*(?:{_NUMBER})\s*(?:-|–|—|,|\bto\b)\s*(?:{_NUMBER})\s*[\])]?",
    re.IGNORECASE,
)
_PAREN_INTERVAL = re.compile(
    rf"[\[(]\s*(?:{_NUMBER})\s*(?:-|–|—|,|\bto\b)\s*(?:{_NUMBER})\s*[\])]",
    re.IGNORECASE,
)
_APPROX_PREFIX = (
    r"(?P<prefix>≈|~|about\s+|approx(?:imately)?\.?\s+|ca\.?\s+|estimated\s+)?"
)
_SCIENTIFIC_FACTOR_TEXT = (
    r"(?:×|x)\s*10(?:\s*\^\s*[+-]?\d+|\s*[+-]\d+)"
)
_SCIENTIFIC_FACTOR = re.compile(
    r"^(?:×|x)\s*10(?:\s*\^\s*(?P<caret>[+-]?\d+)|"
    r"\s*(?P<signed>[+-]\d+))$",
    re.IGNORECASE,
)
_SCIENTIFIC_POINT = re.compile(
    rf"^\s*{_APPROX_PREFIX}(?P<value>{_NUMBER})\s*"
    rf"(?P<factor>{_SCIENTIFIC_FACTOR_TEXT})\s*$",
    re.IGNORECASE,
)
_SCIENTIFIC_PLUS_MINUS_SHARED = re.compile(
    rf"^\s*{_APPROX_PREFIX}(?P<value>{_NUMBER})\s*(?:±|\+/-)\s*"
    rf"(?P<variation>{_NUMBER})\s*(?P<factor>{_SCIENTIFIC_FACTOR_TEXT})\s*$",
    re.IGNORECASE,
)
_SCIENTIFIC_PLUS_MINUS_PARENTHESIZED = re.compile(
    rf"^\s*{_APPROX_PREFIX}\(\s*(?P<value>{_NUMBER})\s*(?:±|\+/-)\s*"
    rf"(?P<variation>{_NUMBER})\s*\)\s*"
    rf"(?P<factor>{_SCIENTIFIC_FACTOR_TEXT})\s*$",
    re.IGNORECASE,
)
_SCIENTIFIC_PLUS_MINUS_EACH = re.compile(
    rf"^\s*{_APPROX_PREFIX}(?P<value>{_NUMBER})\s*"
    rf"(?P<value_factor>{_SCIENTIFIC_FACTOR_TEXT})\s*(?:±|\+/-)\s*"
    rf"(?P<variation>{_NUMBER})\s*"
    rf"(?P<variation_factor>{_SCIENTIFIC_FACTOR_TEXT})\s*$",
    re.IGNORECASE,
)
_AMBIGUOUS_SCIENTIFIC_MARKER = re.compile(
    r"(?:×|x)\s*10(?:[4-9]\b|\d{2,}|[?⁇])", re.IGNORECASE
)

@dataclass(frozen=True)
class EndpointOrthography:
    """One explicit, reviewable spacing/spelling decision."""

    endpoint_name: str
    spacing_and_spelling_endpoint: str
    status: str
    reason: str
    spacing_and_spelling_version: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


EndpointNormalizer = Callable[[str, str], EndpointOrthography | str]
EndpointStandardizer = Callable[[str, MeasurementPair], MeasurementPair]
FamilyResolver = Callable[[str, str], FamilyAssignment | None]
RecordEnricher = Callable[[Mapping[str, Any]], Mapping[str, Any]]

_ENDPOINT_SEPARATORS = re.compile(r"[\s_\-\u2010-\u2015\u2212]+")


def canonicalize_endpoint(value: Any) -> str:
    """Mechanically canonicalize endpoint separators without semantic aliasing."""
    text = clean_text(value) or "missing_endpoint"
    text = unicodedata.normalize("NFKC", text).casefold()
    return _ENDPOINT_SEPARATORS.sub("_", text).strip("_")


def parse_point_measurement(value: Any) -> ParsedPoint:
    text = clean_text(value)
    if text is None:
        return ParsedPoint(None, None, False, "missing")
    if _BOUND.match(text):
        return ParsedPoint(None, None, False, "bound")
    if _RANGE.match(text):
        return ParsedPoint(None, None, False, "range")
    plus_minus = _PLUS_MINUS.match(text)
    if plus_minus:
        return ParsedPoint(
            finite_float(plus_minus.group("value")),
            finite_float(plus_minus.group("variation")),
            True,
            "mean_with_variation",
        )
    plus_minus_leading = _PLUS_MINUS_LEADING.match(text)
    if plus_minus_leading:
        tail = _INTERVAL.sub("", plus_minus_leading.group("tail"))
        tail = _PAREN_INTERVAL.sub("", tail)
        if not re.search(_NUMBER, tail):
            return ParsedPoint(
                finite_float(plus_minus_leading.group("value")),
                finite_float(plus_minus_leading.group("variation")),
                True,
                "mean_with_context",
            )
    point = _POINT.match(text)
    if point:
        return ParsedPoint(
            finite_float(point.group("value")),
            None,
            bool(point.group("prefix")),
            "approximate_point" if point.group("prefix") else "point",
        )
    leading = _LEADING_POINT.match(text)
    if leading:
        stripped = _INTERVAL.sub("", leading.group("tail"))
        stripped = _PAREN_INTERVAL.sub("", stripped)
        stripped = re.sub(
            r"[%\s,;()[\]]|\b(?:gmean|geometric mean|mean|median|sd|se|sem)\b",
            "",
            stripped,
            flags=re.IGNORECASE,
        )
        if not re.search(_NUMBER, stripped) and not stripped:
            return ParsedPoint(
                finite_float(leading.group("value")),
                None,
                bool(leading.group("prefix")),
                "point_with_interval",
            )
    return ParsedPoint(None, None, False, "qualitative_or_compound")


def _scientific_factor(value: str) -> float:
    match = _SCIENTIFIC_FACTOR.fullmatch(value.strip())
    if not match:
        raise ValueError(f"invalid explicit scientific factor: {value!r}")
    return 10.0 ** int(match.group("caret") or match.group("signed"))


def _parse_atomic_scientific_measurement(
    value: str,
) -> tuple[ParsedPoint | None, str, float | None]:
    """Parse only complete point/variation expressions with explicit factors."""
    for pattern in (
        _SCIENTIFIC_PLUS_MINUS_SHARED,
        _SCIENTIFIC_PLUS_MINUS_PARENTHESIZED,
    ):
        match = pattern.fullmatch(value)
        if match:
            factor = _scientific_factor(match.group("factor"))
            return (
                ParsedPoint(
                    finite_float(match.group("value")) * factor,
                    finite_float(match.group("variation")) * factor,
                    bool(match.group("prefix")),
                    "mean_with_variation",
                ),
                "unambiguous_scientific_notation",
                factor,
            )
    each = _SCIENTIFIC_PLUS_MINUS_EACH.fullmatch(value)
    if each:
        value_factor = _scientific_factor(each.group("value_factor"))
        variation_factor = _scientific_factor(each.group("variation_factor"))
        if value_factor != variation_factor:
            return None, "ambiguous_scientific_notation", None
        return (
            ParsedPoint(
                finite_float(each.group("value")) * value_factor,
                finite_float(each.group("variation")) * variation_factor,
                bool(each.group("prefix")),
                "mean_with_variation",
            ),
            "unambiguous_scientific_notation",
            value_factor,
        )
    point = _SCIENTIFIC_POINT.fullmatch(value)
    if point:
        factor = _scientific_factor(point.group("factor"))
        return (
            ParsedPoint(
                finite_float(point.group("value")) * factor,
                None,
                bool(point.group("prefix")),
                "approximate_point" if point.group("prefix") else "point",
            ),
            "unambiguous_scientific_notation",
            factor,
        )
    if _AMBIGUOUS_SCIENTIFIC_MARKER.search(value):
        return None, "ambiguous_scientific_notation", None
    return None, "none", None


def separate_measurement_unit(measurement_text: str, unit: Any) -> str:
    cleaned = clean_measurement_text(measurement_text) or ""
    unit_text = clean_measurement_text(unit)
    normalized_unit = clean_unit(unit)
    if normalized_unit == "%":
        if re.search(r"(?:±|\+/-)", cleaned):
            return re.sub(r"\s*%", "", cleaned)
        return re.sub(
            rf"^(\s*(?:≈|~|about\s+|approx(?:imately)?\.?\s+|ca\.?\s+|estimated\s+)?"
            rf"(?:{_NUMBER}))\s*%",
            r"\1",
            cleaned,
            count=1,
            flags=re.IGNORECASE,
        )
    if unit_text and cleaned.casefold().endswith(unit_text.casefold()):
        return cleaned[: -len(unit_text)].rstrip()
    return cleaned


def format_number(value: float) -> str:
    number = float(value)
    if number and 1e-12 <= abs(number) < 1e-4:
        return f"{number:.15f}".rstrip("0").rstrip(".")
    return f"{number:.15g}"


def render_point_measurement(source_text: str, value: float, variation: float | None) -> str:
    prefix_match = re.match(
        r"^\s*(≈|~|about\s+|approx(?:imately)?\.?\s+|ca\.?\s+|estimated\s+)?",
        source_text,
        flags=re.IGNORECASE,
    )
    prefix = (prefix_match.group(1) if prefix_match else "") or ""
    rendered = f"{prefix}{format_number(value)}"
    if variation is not None:
        rendered += f" ± {format_number(variation)}"
    return rendered.strip()


def normalize_measurement_and_unit(measurement: Any, unit: Any) -> MeasurementPair:
    """Clean and fold a scalar/unit pair without ever guessing unit notation."""
    measurement_text = clean_measurement_text(measurement)
    cleaned_unit = clean_unit(unit)
    unit_result = canonicalize_unit(unit)
    notation_status = unit_result.notation_status
    notation_factor = unit_result.notation_factor
    if measurement_text is None:
        return MeasurementPair(
            None, None, "missing_measurement", notation_status, notation_factor
        )
    display_measurement = separate_measurement_unit(measurement_text, unit)
    scientific_parsed, measurement_notation_status, measurement_notation_factor = (
        _parse_atomic_scientific_measurement(display_measurement)
    )
    parsed = scientific_parsed or parse_point_measurement(display_measurement)
    recognized_unit = bool(unit_result.cleaned) and not unit_result.unknown_tokens
    if (
        notation_status == "ambiguous_scientific_notation"
        or measurement_notation_status == "ambiguous_scientific_notation"
        or (
            measurement_notation_status == "unambiguous_scientific_notation"
            and notation_status == "unambiguous_scientific_notation"
        )
    ):
        return MeasurementPair(
            display_measurement,
            cleaned_unit,
            "ambiguous_scientific_notation",
            "ambiguous_scientific_notation",
            None,
        )
    if measurement_notation_status == "unambiguous_scientific_notation":
        notation_status = measurement_notation_status
        notation_factor = measurement_notation_factor
    if parsed.value is None or not recognized_unit:
        status = (
            "cleaned_only_non_scalar"
            if parsed.value is None
            else "cleaned_only_unrecognized_unit"
        )
        return MeasurementPair(
            display_measurement, cleaned_unit, status, notation_status, notation_factor
        )
    if parsed.kind in {"point_with_interval", "mean_with_context"}:
        return MeasurementPair(
            display_measurement,
            cleaned_unit,
            "cleaned_pair_with_context",
            notation_status,
            notation_factor,
        )
    value, canonical_unit = canonicalize_measurement(parsed.value, unit)
    if value is None or canonical_unit is None:
        return MeasurementPair(
            measurement_text,
            cleaned_unit,
            "cleaned_only_pair_failed",
            notation_status,
            notation_factor,
        )
    variation = None
    if parsed.variation is not None:
        variation, variation_unit = canonicalize_measurement(parsed.variation, unit)
        if variation is None or variation_unit != canonical_unit:
            return MeasurementPair(
                display_measurement,
                cleaned_unit,
                "cleaned_only_pair_failed",
                notation_status,
                notation_factor,
            )
    rendered = render_point_measurement(display_measurement, value, variation)
    status = (
        "folded_pair"
        if canonical_unit != cleaned_unit or rendered != display_measurement
        else "cleaned_pair"
    )
    return MeasurementPair(
        rendered, canonical_unit, status, notation_status, notation_factor
    )


def standardize_measurement_pair(
    pair: MeasurementPair,
    *,
    measurement_class: str | None = None,
    targets: Sequence[str] | None = None,
) -> MeasurementPair:
    if pair.canonical_measurement is None or pair.canonical_unit is None:
        return pair
    if pair.unit_notation_status == "ambiguous_scientific_notation":
        return pair
    parsed = parse_point_measurement(pair.canonical_measurement)
    if parsed.value is None or parsed.kind in {"point_with_interval", "mean_with_context"}:
        return pair
    value, unit = canonicalize_measurement(
        parsed.value,
        pair.canonical_unit,
        measurement_class=measurement_class,
        targets=targets,
    )
    if value is None or unit is None:
        return pair
    variation = None
    if parsed.variation is not None:
        variation, variation_unit = canonicalize_measurement(
            parsed.variation,
            pair.canonical_unit,
            measurement_class=measurement_class,
            targets=targets,
        )
        if variation is None or variation_unit != unit:
            return pair
    return MeasurementPair(
        render_point_measurement(pair.canonical_measurement, value, variation),
        unit,
        "endpoint_standardized",
        pair.unit_notation_status,
        pair.unit_notation_factor,
    )


def _recognized_unit(value: Any) -> bool:
    result = canonicalize_unit(value)
    return bool(result.cleaned) and not result.unknown_tokens


def normalize_cleaned_records(
    cleaned_records: Sequence[Mapping[str, Any]],
    *,
    endpoint_normalizer: EndpointNormalizer,
    endpoint_standardizer: EndpointStandardizer | None = None,
    family_resolver: FamilyResolver,
    record_enricher: RecordEnricher | None = None,
) -> list[dict[str, Any]]:
    """Normalize each cleaned source record exactly once."""
    records: list[dict[str, Any]] = []
    for cleaned in cleaned_records:
        source_id = str(cleaned.get("source_id") or "")
        endpoint_name = str(cleaned.get("endpoint_name") or "")
        decision = endpoint_normalizer(source_id, endpoint_name)
        if isinstance(decision, str):
            decision = EndpointOrthography(
                endpoint_name=endpoint_name,
                spacing_and_spelling_endpoint=decision,
                status="unchanged",
                reason="no_reviewed_correction",
                spacing_and_spelling_version="unspecified",
            )
        spacing_and_spelling_endpoint = decision.spacing_and_spelling_endpoint
        canonical_endpoint = canonicalize_endpoint(spacing_and_spelling_endpoint)
        baseline = normalize_measurement_and_unit(
            cleaned.get("measurement_text"), cleaned.get("unit_text")
        )
        pair = (
            endpoint_standardizer(canonical_endpoint, baseline)
            if endpoint_standardizer is not None
            else baseline
        )
        parsed = parse_point_measurement(pair.canonical_measurement)
        recognized_unit = _recognized_unit(pair.canonical_unit)
        finite_scalar = (
            parsed.value
            if parsed.value is not None
            and recognized_unit
            and pair.status != "ambiguous_scientific_notation"
            else None
        )
        absolute_continuous = finite_scalar is not None
        family = family_resolver(source_id, endpoint_name)
        unit_result = canonicalize_unit(pair.canonical_unit)
        payload = json.loads(str(cleaned.get("source_payload_json") or "{}"))
        record = dict(cleaned)
        record.update(
            {
                "normalization_version": NORMALIZED_RECORD_VERSION,
                "measurement_normalization_version": NORMALIZATION_STAGE_VERSION,
                "normalized_record_id": stable_id(
                    "normalized", cleaned.get("cleaned_record_id")
                ),
                "spacing_and_spelling_endpoint": spacing_and_spelling_endpoint,
                "spacing_and_spelling_status": decision.status,
                "spacing_and_spelling_reason": decision.reason,
                "spacing_and_spelling_version": decision.spacing_and_spelling_version,
                "canonical_endpoint": canonical_endpoint,
                "canonical_measurement": pair.canonical_measurement,
                "canonical_unit": pair.canonical_unit,
                "measurement_parse_kind": parsed.kind,
                "measurement_unit_status": pair.status,
                "unit_notation_status": pair.unit_notation_status,
                "unit_notation_factor": pair.unit_notation_factor,
                "unit_dimension_json": json.dumps(
                    unit_result.dimension, ensure_ascii=False
                ),
                "finite_scalar_value": finite_scalar,
                "is_absolute_and_continuous": bool(absolute_continuous),
                "absolute_and_continuous_value": (
                    finite_scalar if absolute_continuous else None
                ),
                "variation_value": parsed.variation,
                "group_id": family.group_id if family else None,
                "assay_tier": family.assay_tier if family else None,
                "endpoint_group": family.endpoint_group if family else None,
                "evidence_role": family.evidence_role if family else None,
                "target_pref_name": family.target_pref_name if family else None,
                "source_payload_json": json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, default=str
                ),
            }
        )
        if record_enricher is not None:
            record.update(dict(record_enricher(record)))
        records.append(record)
    return records


__all__ = [
    "EndpointNormalizer",
    "EndpointOrthography",
    "EndpointStandardizer",
    "FamilyResolver",
    "RecordEnricher",
    "canonicalize_endpoint",
    "format_number",
    "normalize_cleaned_records",
    "normalize_measurement_and_unit",
    "parse_point_measurement",
    "render_point_measurement",
    "separate_measurement_unit",
    "standardize_measurement_pair",
]
