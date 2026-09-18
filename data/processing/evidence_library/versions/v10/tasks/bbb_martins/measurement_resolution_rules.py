"""Reviewable BBB V8 rules that decide whether scalar extraction needs an LLM.

Each function implements one narrow rule and returns ``None`` when it does not
apply.  ``route_measurement`` is the only precedence list.  An ``accept`` copies
a source value; a ``reject`` skips scalar extraction but retains the source row.
Categorical encoding is deliberately not part of this module.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from data.processing.evidence_library.versions.v10.measurement_routing import (
    RouteDecision,
    has_digit,
    is_pure_number,
    measurement_source_text,
)
from data.processing.evidence_library.versions.v10.unit_vocabulary import (
    clean_unit,
    load_unit_vocabulary,
)


RULE_POLICY_VERSION = "bbb_measurement_resolution_rules.v1"
OBSERVED_UNITS = load_unit_vocabulary()

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_UNCERTAINTY_OPERATOR = r"(?:±|\+/-)"
_PERCENT_UNIT = re.compile(r"^(?:%|％|percent|per\s+cent)$", re.IGNORECASE)
_NUMBER_PERCENT = re.compile(
    rf"^\s*(?P<value>{_NUMBER})\s*(?:%|％|percent|per\s+cent)\s*$",
    re.IGNORECASE,
)
_POINT_WITH_UNCERTAINTY = re.compile(
    rf"^\s*(?P<value>{_NUMBER})\s*{_UNCERTAINTY_OPERATOR}\s*{_NUMBER}\s*$"
)
_PERCENT_WITH_UNCERTAINTY = re.compile(
    rf"^\s*(?P<value>{_NUMBER})\s*{_UNCERTAINTY_OPERATOR}\s*{_NUMBER}"
    rf"\s*(?:%|％|percent|per\s+cent)\s*$",
    re.IGNORECASE,
)
_SCIENTIFIC_FACTOR = r"(?:[x×*]\s*)10\s*(?:(?:\^|\*\*)\s*)?[+-]?\d+"
_SCALED_NUMBER = re.compile(
    rf"^\s*(?P<value>{_NUMBER})\s*(?P<factor>{_SCIENTIFIC_FACTOR})\s*$",
    re.IGNORECASE,
)
_HARD_BOUND = re.compile(
    rf"^\s*(?:<|>|≤|≥|at\s+(?:least|most)|up\s+to|"
    rf"no\s+(?:less|more)\s+than)\s*{_NUMBER}"
    rf"\s*(?:%|％|percent|per\s+cent)?\s*$",
    re.IGNORECASE,
)
_EXPLICIT_RANGE = re.compile(
    rf"^\s*{_NUMBER}\s*(?:-|–|—|to)\s*{_NUMBER}"
    rf"\s*(?:%|％|percent|per\s+cent)?\s*$",
    re.IGNORECASE,
)
_STANDALONE_NUMBER = re.compile(
    rf"(?<![A-Za-z0-9.]){_NUMBER}(?![A-Za-z0-9.])"
)
_POWER_OF_TEN = re.compile(
    rf"[x×*]\s*10\s*(?:(?:\^|\*\*)\s*)?[+-]?\d+",
    re.IGNORECASE,
)
_UNCERTAINTY = re.compile(rf"{_UNCERTAINTY_OPERATOR}\s*{_NUMBER}")
_IDENTIFIER_TOKEN = re.compile(
    r"\b(?:[A-Za-z]+\d[A-Za-z0-9]*|[A-Za-z][A-Za-z0-9]*(?:-\d[A-Za-z0-9]*)+)\b"
)
_EMBEDDED_VALUE_UNIT = re.compile(
    rf"^\s*(?P<value>{_NUMBER})"
    rf"(?:\s*{_UNCERTAINTY_OPERATOR}\s*{_NUMBER})?"
    rf"(?:\s*(?P<factor>{_SCIENTIFIC_FACTOR}))?\s+"
    rf"(?P<unit>\S(?:.*\S)?)\s*$",
    re.IGNORECASE,
)


def _text(record: Mapping[str, Any], field: str) -> str:
    return str(record.get(field) or "").strip()


def _observed_unit(record: Mapping[str, Any]) -> str:
    unit = clean_unit(record.get("unit_text"))
    return unit if unit in OBSERVED_UNITS else ""


def exact_finite_number_with_separate_unit(
    record: Mapping[str, Any],
) -> RouteDecision | None:
    """Copy a whole-field finite number and nonempty source unit, including zero.

    Sign and scientific compatibility belong to later endpoint/unit validation;
    asking a model to recopy the same two source fields cannot resolve them.
    """
    measurement = record.get("measurement_text")
    unit = _observed_unit(record)
    if not unit or not is_pure_number(measurement):
        return None
    return RouteDecision(
        "accept",
        "finite_number_with_separate_unit.v2",
        measurement_source_text(measurement),
        unit,
    )


def exact_direct_logbb(record: Mapping[str, Any]) -> RouteDecision | None:
    """Assign a bare direct logBB number to the explicit base-10 ratio axis."""
    if (
        record.get("source_id") != "direct_bbb"
        or _text(record, "unit_text")
        or _text(record, "canonical_endpoint_name") != "logbb"
        or not is_pure_number(record.get("measurement_text"))
    ):
        return None
    return RouteDecision(
        "accept",
        "direct_logbb_log10_ratio.v4",
        measurement_source_text(record.get("measurement_text")),
        "log10_ratio",
        True,
    )


def exact_number_percent(record: Mapping[str, Any]) -> RouteDecision | None:
    """Copy ``87%`` only when a separate unit is absent or also means percent.

    Units such as ``of control`` or ``ID/g`` carry additional semantics and are
    left for extraction rather than being flattened to a bare percentage.
    """
    match = _NUMBER_PERCENT.fullmatch(_text(record, "measurement_text"))
    unit = _observed_unit(record)
    if match is None or (unit and _PERCENT_UNIT.fullmatch(unit) is None):
        return None
    return RouteDecision(
        "accept", "number_percent_literal.v1", match.group("value"), "%", True
    )


def exact_point_with_uncertainty_and_unit(
    record: Mapping[str, Any],
) -> RouteDecision | None:
    """Copy the point from ``8 ± 1.5`` when the source unit is separate."""
    match = _POINT_WITH_UNCERTAINTY.fullmatch(_text(record, "measurement_text"))
    unit = _observed_unit(record)
    if match is None or not unit:
        return None
    return RouteDecision(
        "accept", "point_with_uncertainty_separate_unit.v2", match.group("value"), unit
    )


def exact_percent_with_uncertainty(
    record: Mapping[str, Any],
) -> RouteDecision | None:
    """Copy the point from ``45 ± 5%`` when no conflicting unit is declared."""
    match = _PERCENT_WITH_UNCERTAINTY.fullmatch(
        _text(record, "measurement_text")
    )
    unit = _observed_unit(record)
    if match is None or (unit and _PERCENT_UNIT.fullmatch(unit) is None):
        return None
    return RouteDecision(
        "accept", "percent_with_uncertainty.v2", match.group("value"), "%", True
    )


def exact_scaled_number_with_separate_unit(
    record: Mapping[str, Any],
) -> RouteDecision | None:
    """Move a printed ``x 10^n`` beside the separate unit without rescaling."""
    match = _SCALED_NUMBER.fullmatch(_text(record, "measurement_text"))
    unit = _observed_unit(record)
    if match is None or not unit or _POWER_OF_TEN.search(unit):
        return None
    return RouteDecision(
        "accept",
        "scaled_number_with_separate_unit.v2",
        match.group("value"),
        f"{match.group('factor')} {unit}",
    )


def reject_hard_bound(record: Mapping[str, Any]) -> RouteDecision | None:
    """Reject only a whole-field bound such as ``<5`` or ``at most 10%``."""
    if _HARD_BOUND.fullmatch(_text(record, "measurement_text")):
        return RouteDecision("reject", "hard_bound_no_point.v2")
    return None


def reject_explicit_range(record: Mapping[str, Any]) -> RouteDecision | None:
    """Reject only a whole-field interval such as ``7-12%``."""
    if _EXPLICIT_RANGE.fullmatch(_text(record, "measurement_text")):
        return RouteDecision("reject", "explicit_range_no_point.v2")
    return None


def exact_embedded_number_and_unit(
    record: Mapping[str, Any],
) -> RouteDecision | None:
    """Copy one whole-field inline pair only when its unit was source-observed."""
    if _text(record, "unit_text"):
        return None
    match = _EMBEDDED_VALUE_UNIT.fullmatch(_text(record, "measurement_text"))
    if match is None:
        return None
    unit = clean_unit(
        " ".join(filter(None, (match.group("factor"), match.group("unit"))))
    )
    if unit not in OBSERVED_UNITS:
        return None
    return RouteDecision(
        "accept",
        "observed_embedded_value_and_unit.v1",
        match.group("value"),
        unit,
    )


def reject_no_digit(record: Mapping[str, Any]) -> RouteDecision | None:
    """Digit-free measurement text has no scalar coefficient to extract."""
    if not has_digit(record.get("measurement_text")):
        return RouteDecision("reject", "no_digit_in_measurement_column.v2")
    return None


def _candidate_coefficients(text: str) -> list[str]:
    without_identifiers = _IDENTIFIER_TOKEN.sub("", text)
    without_spread = _UNCERTAINTY.sub("", without_identifiers)
    without_scale = _POWER_OF_TEN.sub("", without_spread)
    return [match.group() for match in _STANDALONE_NUMBER.finditer(without_scale)]


def reject_no_standalone_number(record: Mapping[str, Any]) -> RouteDecision | None:
    """Digits used only in identifiers or notation are not outcome coefficients."""
    if not _candidate_coefficients(_text(record, "measurement_text")):
        return RouteDecision("reject", "no_standalone_numeric_measurement.v3")
    return None


def reject_multiple_independent_numbers(
    record: Mapping[str, Any],
) -> RouteDecision | None:
    """Do not choose among multiple coefficients after spread/scale removal."""
    if len(_candidate_coefficients(_text(record, "measurement_text"))) > 1:
        return RouteDecision("reject", "multiple_numeric_candidates.v3")
    return None


Rule = Callable[[Mapping[str, Any]], RouteDecision | None]

# This tuple is the complete V8 review surface and the exact precedence used by
# the router. Broad condition and Km/Kt rejection are intentionally absent.
ROUTING_RULES: tuple[Rule, ...] = (
    exact_finite_number_with_separate_unit,
    exact_direct_logbb,
    exact_number_percent,
    exact_point_with_uncertainty_and_unit,
    exact_percent_with_uncertainty,
    exact_scaled_number_with_separate_unit,
    reject_hard_bound,
    reject_explicit_range,
    exact_embedded_number_and_unit,
    reject_no_digit,
    reject_no_standalone_number,
    reject_multiple_independent_numbers,
)


def route_measurement(record: Mapping[str, Any]) -> RouteDecision:
    """Apply the reviewed rules in order; unresolved numeric prose needs the LLM."""
    for rule in ROUTING_RULES:
        decision = rule(record)
        if decision is not None:
            from data.processing.evidence_library.versions.v10.percentage_delta import review_percentage
            return review_percentage(record, decision)
    return RouteDecision("extract")


__all__ = [
    "ROUTING_RULES",
    "RULE_POLICY_VERSION",
    "exact_direct_logbb",
    "exact_embedded_number_and_unit",
    "exact_finite_number_with_separate_unit",
    "exact_number_percent",
    "exact_percent_with_uncertainty",
    "exact_point_with_uncertainty_and_unit",
    "exact_scaled_number_with_separate_unit",
    "reject_explicit_range",
    "reject_hard_bound",
    "reject_multiple_independent_numbers",
    "reject_no_digit",
    "reject_no_standalone_number",
    "route_measurement",
]
