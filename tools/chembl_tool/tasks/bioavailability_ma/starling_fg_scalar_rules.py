"""Strict, audit-first scalar proposals for Starling Fg measurements.

This module does not participate in the normalized-v5 build.  It proposes one
canonical scalar pair only when the complete source ``measured_value`` encodes
one atomic outcome.  Source-facing text is never rewritten.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from tools.chembl_tool.common.starling.normalization.measurements import (
    SOURCE_SPECIFIC_ATOMIC_SCALAR_STATUS,
    normalize_measurement_and_unit,
    parse_point_measurement,
    render_point_measurement,
)
from tools.chembl_tool.common.starling.normalization.contracts import MeasurementPair
from tools.chembl_tool.tasks.bioavailability_ma.starling_normalization_policy import (
    endpoint_specific_standardization_of_unit,
)


FG_SCALAR_RULE_VERSION = "bioavailability_fg_single_outcome_scalar_rules.v2"

_NUMBER = (
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?(?:[eE][-+]?\d+)?"
    r"|[-+]?\.\d+(?:[eE][-+]?\d+)?"
)
_APPROX = r"(?:≈|~|about\s+|approx(?:imately)?\.?\s+|ca\.?\s+|estimated\s+)?"
_POINT = rf"(?P<prefix>{_APPROX})(?P<value>{_NUMBER})"
_VARIATION = rf"(?:\s*(?:±|\+/-)\s*(?P<variation>{_NUMBER}))?"
_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212"

_BOUND = re.compile(
    r"(?:^|[\s:(])(?:[<>]=?|≥|≤)\s*[-+]?\d|"
    r"\b(?:up to|at least|at most|more than|less than|"
    r"no (?:less|more) than)\b",
    re.IGNORECASE,
)
_RANGE = re.compile(
    rf"(?:{_NUMBER})\s*(?:-|[{_DASHES}]|\bto\b)\s*(?:{_NUMBER})",
    re.IGNORECASE,
)
_COMPARISON = re.compile(r"\b(?:vs\.?|versus)\b", re.IGNORECASE)
_MULTI_METRIC = re.compile(
    r"\b(?:p\s*[_ ]?\s*app|p\s*[_ ]?\s*eff|k\s*[_ ]?\s*a|"
    r"(?:unidirectional|net)\s+flux(?:es)?|flux(?:es)?|"
    r"efflux ratio|uptake ratio|auc(?:\s*0)?|cmax|"
    r"peak plasma concentration|jmax|vmax|km|"
    r"clearance|clint|fg|fa|fh|bioavailability|exposure|absorption|"
    r"excretion|half[- ]life)\b",
    re.IGNORECASE,
)
_TWO_DIRECTIONS = re.compile(
    r"(?:a|ap|b|bl)\s*(?:-|→|to)\s*(?:a|ap|b|bl)",
    re.IGNORECASE,
)
_COORDINATED_OUTCOMES = re.compile(
    r"\b(?:both\s+(?:apical\s+and\s+basolateral|directions?)|"
    r"unidirectional\s+and\s+net\s+flux(?:es)?|"
    r"and\s+(?:unchanged|no\s+(?:change|effect))\b|"
    r"single\W+and\W+double\W*strength|"
    r"(?:ttt|cgc/cgt)\s+and\s+(?:ttt|cgc/cgt)|"
    r"(?:jejunum|ileum)\s+and\s+(?:lower\s+)?(?:jejunum|ileum))\b",
    re.IGNORECASE,
)
_QUALITATIVE_NEGATION = re.compile(
    r"^\s*(?:no|not)\s+(?:significant\s+)?"
    r"(?:change|difference|effect|transport|absorption|uptake|secretion)\b",
    re.IGNORECASE,
)
_STANDALONE_NUMBER = re.compile(
    rf"(?<![A-Za-z0-9])(?:{_NUMBER})(?![A-Za-z0-9])"
)

_RATIO_LABELS = (
    r"efflux\s+ratio|uptake\s+ratio|relative\s+efflux\s+ratio|"
    r"concentration\s+ratio|improvement\s+ratio|auc\s+ratio|"
    r"permeability\s+ratio|transport\s+ratio|"
    r"er|rer|re"
)
_RATIO = re.compile(
    rf"^\s*(?P<label>{_RATIO_LABELS})\s*(?:=|:|of\s+|is\s+|was\s+)?\s*"
    rf"{_POINT}{_VARIATION}\s*(?P<tail>\([^)]*\))?\s*$",
    re.IGNORECASE,
)
_FOLD = re.compile(
    rf"(?P<prefix>{_APPROX})(?P<value>{_NUMBER})"
    rf"(?:\s*(?:±|\+/-)\s*(?P<variation>{_NUMBER}))?"
    rf"\s*[- ]?\s*fold\b",
    re.IGNORECASE,
)
_PAREN_FOLD = re.compile(
    rf"(?P<prefix>{_APPROX})(?P<value>{_NUMBER})"
    rf"(?:\s*(?:±|\+/-)\s*(?P<variation>{_NUMBER}))?"
    rf"\s*\(\s*fold\b[^)]*\)",
    re.IGNORECASE,
)
_WORD_FOLD = re.compile(
    rf"(?P<prefix>{_APPROX})"
    rf"(?P<word>one|two|three|four|five|six|seven|eight|nine|ten)"
    rf"\s*[- ]?\s*fold\b",
    re.IGNORECASE,
)
_WORD_VALUES = {
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
}

_FG_LABEL = re.compile(
    r"(?P<label>"
    r"fabs\s*[·×x*]\s*f\s*[_ ]?\s*g|"
    r"f\s*[_ ]?\s*a\s*[·×x*]\s*f\s*[_ ]?\s*g|"
    r"fa\s*fg|"
    r"f\s*[_ ]?\s*g(?:\s*,\s*[a-z0-9]+)?|"
    r"fg|fgut"
    r")",
    re.IGNORECASE,
)
_FG_PERCENT_AFTER_LABEL = re.compile(
    rf"(?P<label>"
    rf"fabs\s*[·×x*]\s*f\s*[_ ]?\s*g|"
    rf"f\s*[_ ]?\s*a\s*[·×x*]\s*f\s*[_ ]?\s*g|"
    rf"fa\s*fg|"
    rf"f\s*[_ ]?\s*g(?:\s*,\s*[a-z0-9]+)?|fg|fgut|"
    rf"fg"
    rf")"
    rf"[^0-9<>≥≤;]*?{_POINT}\s*%"
    rf"(?:\s*(?:±|\+/-)\s*(?P<variation>{_NUMBER})\s*%)?",
    re.IGNORECASE,
)
_FG_FRACTION_AFTER_LABEL = re.compile(
    rf"(?P<label>"
    rf"fabs\s*[·×x*]\s*f\s*[_ ]?\s*g|"
    rf"f\s*[_ ]?\s*a\s*[·×x*]\s*f\s*[_ ]?\s*g|"
    rf"fa\s*fg|"
    rf"f\s*[_ ]?\s*g(?:\s*,\s*[a-z0-9]+)?|fg|fgut|"
    rf"fg"
    rf")"
    rf"[^0-9%<>≥≤;]*?{_POINT}{_VARIATION}",
    re.IGNORECASE,
)

_PERCENT = re.compile(
    rf"(?=[^;]*%){_POINT}\s*%?"
    rf"(?:\s*(?:±|\+/-)\s*(?P<variation>{_NUMBER})\s*%)?",
    re.IGNORECASE,
)
_PERCENT_OUTCOME = re.compile(
    r"\b(?:inhibition|increase|increased|decrease|decreased|reduction|"
    r"reduce|reduced|uptake|absorption|"
    r"extraction|elimination|availability|bioavailability|metaboli[sz]ed|"
    r"transport(?:ed)?|permeation|secretion|recovery|remaining|"
    r"of (?:control|dose)|first[- ]pass)\b",
    re.IGNORECASE,
)
_RELATIVE_CHANGE = re.compile(
    r"\b(?:increase|increased|decrease|decreased|reduction|"
    r"reduce|reduced|change|higher|lower)\b",
    re.IGNORECASE,
)
_PERCENT_CONDITION_SUFFIX = re.compile(
    r"^\s*(?:ethanol|supplementation|solution|medium|media|formulation|"
    r"w/v|v/v|m/v)\b",
    re.IGNORECASE,
)
_PERCENT_CONDITION_PREFIX = re.compile(
    r"\b(?:at|with|containing|in the presence of)\s*$",
    re.IGNORECASE,
)

_PHYSICAL_LABEL = re.compile(
    r"^(?P<label>"
    r"papp(?:\s*(?:a|ap|b|bl)\s*(?:-|→|to)\s*(?:a|ap|b|bl))?|"
    r"peff|clint|clearance(?:\s+k[_ ]?b)?|k[_ ]?b|"
    r"jmax|vmax|km|pc|pm|penetration\s+depth"
    r")\s*(?:=|:)?\s*",
    re.IGNORECASE,
)
_PHYSICAL_POINT = re.compile(
    rf"^\s*{_POINT}{_VARIATION}\s*"
    rf"(?P<unit>(?:(?:×|x)\s*10(?:\s*\^\s*[+-]?\d+|\s*[+-]\d+)\s*)?"
    rf"[A-Za-zµμ%][A-Za-zµμ0-9%·.*^/() _-]*)\s*$",
    re.IGNORECASE,
)
_UNSCALED_NUMERIC_DENOMINATOR = re.compile(
    r"(?:/|\bper\s+)100\b", re.IGNORECASE
)
_AMBIGUOUS_PROTEIN_NORMALIZATION = re.compile(
    r"\bmg\s*(?:\^?\s*[-−]\s*1)\s+protein\b|"
    r"\bmg\s+protein\s*(?:\^?\s*[-−]\s*1)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FgScalarDecision:
    """One auditable proposal or rejection for an Fg source row."""

    accepted: bool
    reason: str
    rule_id: str | None = None
    semantic_label: str | None = None
    canonical_measurement: str | None = None
    canonical_unit: str | None = None
    finite_scalar_value: float | None = None
    variation_value: float | None = None
    approximate: bool = False
    unit_notation_status: str = "none"
    unit_notation_factor: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"rule_version": FG_SCALAR_RULE_VERSION, **asdict(self)}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except (TypeError, ValueError):
        pass
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    return re.sub(r"\s+", " ", text).strip()


def _float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value.replace(",", ""))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _point_from_match(match: re.Match[str]) -> tuple[float, float | None, bool] | None:
    value = _float(match.groupdict().get("value"))
    variation = _float(match.groupdict().get("variation"))
    approximate = bool((match.groupdict().get("prefix") or "").strip())
    if value is None or (variation is not None and variation < 0):
        return None
    return value, variation, approximate


def _render(value: float, variation: float | None, approximate: bool) -> str:
    prefix = "≈" if approximate else ""
    source = f"{prefix}{value}"
    return render_point_measurement(source, value, variation)


def _semantic_fg_label(label: str) -> str:
    compact = re.sub(r"[\s_*]", "", label.casefold())
    if "fabs" in compact:
        return "fabs_times_fg_fraction"
    if compact.startswith("fafg") or (
        "fa" in compact and ("·" in label or "×" in label or "x" in compact)
    ):
        return "fa_times_fg_fraction"
    return "fg_fraction"


def _outside_numbers(text: str, span: tuple[int, int]) -> list[str]:
    remainder = f"{text[: span[0]]} {text[span[1] :]}"
    return _STANDALONE_NUMBER.findall(remainder)


def _metric_count(text: str) -> int:
    without_products = re.sub(
        r"\bf(?:abs|\s*[_ ]?\s*a)\s*[·×x*]\s*f\s*[_ ]?\s*g\b",
        " fg ",
        text,
        flags=re.IGNORECASE,
    )
    return len(_MULTI_METRIC.findall(without_products))


def _reject_reason(text: str) -> str:
    if not text:
        return "missing_measurement"
    if _BOUND.search(text):
        return "bounded_measurement"
    if _RANGE.search(text):
        return "range_measurement"
    if (
        ";" in text
        or (_COMPARISON.search(text) and len(_STANDALONE_NUMBER.findall(text)) > 1)
        or _metric_count(text) > 1
    ):
        return "compound_or_multi_outcome"
    if _STANDALONE_NUMBER.search(text):
        return "numeric_but_unsupported_atomic_form"
    return "qualitative_or_missing_scalar"


def _accepted(
    *,
    value: float,
    variation: float | None,
    approximate: bool,
    unit: str,
    rule_id: str,
    semantic_label: str,
    unit_notation_status: str = "none",
    unit_notation_factor: float | None = None,
) -> FgScalarDecision:
    if not math.isfinite(value):
        return FgScalarDecision(False, "nonfinite_value")
    if variation is not None and (not math.isfinite(variation) or variation < 0):
        return FgScalarDecision(False, "invalid_variation")
    if unit in {"ratio", "fold"} and value <= 0:
        return FgScalarDecision(False, "nonpositive_dimensionless_ratio")
    if unit == "fraction" and not 0 <= value <= 1:
        return FgScalarDecision(False, "outside_bounded_fraction_domain")
    if unit == "%" and not 0 <= value <= 100:
        return FgScalarDecision(False, "outside_bounded_percentage_domain")
    if unit not in {"ratio", "fold", "fraction", "%"} and value <= 0:
        if not unit.startswith(("log(", "log10(")):
            return FgScalarDecision(False, "nonpositive_physical_measurement")
    return FgScalarDecision(
        True,
        "accepted_single_outcome",
        rule_id=rule_id,
        semantic_label=semantic_label,
        canonical_measurement=_render(value, variation, approximate),
        canonical_unit=unit,
        finite_scalar_value=value,
        variation_value=variation,
        approximate=approximate,
        unit_notation_status=unit_notation_status,
        unit_notation_factor=unit_notation_factor,
    )


def _try_ratio(text: str) -> FgScalarDecision | None:
    match = _RATIO.fullmatch(text)
    if not match:
        return None
    tail = match.group("tail") or ""
    if _STANDALONE_NUMBER.search(tail):
        return FgScalarDecision(False, "numeric_context_inside_ratio_tail")
    point = _point_from_match(match)
    if point is None:
        return FgScalarDecision(False, "invalid_ratio_value")
    value, variation, approximate = point
    label = re.sub(r"\s+", "_", match.group("label").casefold())
    semantic = (
        "directional_efflux_ratio"
        if label in {"efflux_ratio", "er", "rer", "re", "relative_efflux_ratio"}
        else label
    )
    return _accepted(
        value=value,
        variation=variation,
        approximate=approximate,
        unit="ratio",
        rule_id="explicit_ratio_label",
        semantic_label=semantic,
    )


def _try_fold(text: str) -> FgScalarDecision | None:
    matches = (
        list(_FOLD.finditer(text))
        or list(_PAREN_FOLD.finditer(text))
        or list(_WORD_FOLD.finditer(text))
    )
    if len(matches) != 1:
        return None
    match = matches[0]
    if _outside_numbers(text, match.span()):
        return FgScalarDecision(False, "additional_numeric_context_in_fold_measurement")
    if match.groupdict().get("word"):
        value = _WORD_VALUES[match.group("word").casefold()]
        variation = None
        approximate = bool((match.groupdict().get("prefix") or "").strip())
    else:
        point = _point_from_match(match)
        if point is None:
            return FgScalarDecision(False, "invalid_fold_value")
        value, variation, approximate = point
    return _accepted(
        value=value,
        variation=variation,
        approximate=approximate,
        unit="fold",
        rule_id="explicit_fold_label",
        semantic_label="fold_change",
    )


def _try_fg_fraction(text: str) -> FgScalarDecision | None:
    if not _FG_LABEL.search(text):
        return None
    if _RELATIVE_CHANGE.search(text):
        return None
    percent_matches = list(_FG_PERCENT_AFTER_LABEL.finditer(text))
    fraction_matches = list(_FG_FRACTION_AFTER_LABEL.finditer(text))
    matches = percent_matches or fraction_matches
    if len(matches) != 1:
        return None
    match = matches[0]
    if _outside_numbers(text, match.span()):
        return FgScalarDecision(False, "additional_numeric_context_in_fg_measurement")
    point = _point_from_match(match)
    if point is None:
        return FgScalarDecision(False, "invalid_fg_fraction_value")
    value, variation, approximate = point
    unit = "%" if percent_matches else "fraction"
    return _accepted(
        value=value,
        variation=variation,
        approximate=approximate,
        unit=unit,
        rule_id="explicit_fg_fraction_label",
        semantic_label=_semantic_fg_label(match.group("label")),
    )


def _try_percentage(text: str) -> FgScalarDecision | None:
    if not _PERCENT_OUTCOME.search(text):
        return None
    matches = list(_PERCENT.finditer(text))
    if len(matches) != 1:
        return None
    match = matches[0]
    if _PERCENT_CONDITION_PREFIX.search(text[: match.start()]) or (
        _PERCENT_CONDITION_SUFFIX.search(text[match.end() :])
    ):
        return FgScalarDecision(False, "percentage_is_assay_condition")
    if _outside_numbers(text, match.span()):
        return FgScalarDecision(False, "additional_numeric_context_in_percentage")
    point = _point_from_match(match)
    if point is None:
        return FgScalarDecision(False, "invalid_percentage_value")
    value, variation, approximate = point
    return _accepted(
        value=value,
        variation=variation,
        approximate=approximate,
        unit="%",
        rule_id="explicit_single_percentage",
        semantic_label="percentage_outcome",
    )


def _try_physical(text: str, canonical_endpoint: str) -> FgScalarDecision | None:
    label_match = _PHYSICAL_LABEL.match(text)
    body = text[label_match.end() :] if label_match else text
    match = _PHYSICAL_POINT.fullmatch(body)
    if not match:
        return None
    point = _point_from_match(match)
    if point is None:
        return FgScalarDecision(False, "invalid_physical_value")
    value, variation, approximate = point
    raw_unit = match.group("unit")
    if _UNSCALED_NUMERIC_DENOMINATOR.search(raw_unit):
        return FgScalarDecision(False, "unsupported_numeric_unit_denominator")
    if _AMBIGUOUS_PROTEIN_NORMALIZATION.search(raw_unit):
        return FgScalarDecision(False, "ambiguous_protein_normalization_notation")
    if re.search(r"\s+\([A-Za-z ]+\)\s*$", raw_unit):
        return FgScalarDecision(False, "context_suffix_inside_physical_unit")
    measurement = _render(value, variation, approximate)
    pair = normalize_measurement_and_unit(measurement, raw_unit)
    pair = endpoint_specific_standardization_of_unit(canonical_endpoint, pair)
    parsed = parse_point_measurement(pair.canonical_measurement)
    if (
        pair.canonical_unit is None
        or parsed.value is None
        or pair.status
        in {
            "ambiguous_scientific_notation",
            "cleaned_only_pair_failed",
            "cleaned_only_unrecognized_unit",
            "incompatible_endpoint_unit",
        }
    ):
        return FgScalarDecision(False, "unrecognized_or_incompatible_physical_unit")
    if (
        pair.canonical_unit == "cm/s"
        and not 0 < parsed.value <= 1
    ):
        return FgScalarDecision(False, "outside_permeability_domain")
    label = (
        re.sub(r"\s+", "_", label_match.group("label").casefold())
        if label_match
        else "atomic_physical_measurement"
    )
    return _accepted(
        value=parsed.value,
        variation=parsed.variation,
        approximate=approximate,
        unit=pair.canonical_unit,
        rule_id="explicit_embedded_physical_unit",
        semantic_label=label,
        unit_notation_status=pair.unit_notation_status,
        unit_notation_factor=pair.unit_notation_factor,
    )


def propose_fg_scalar(
    measurement_text: Any,
    *,
    canonical_endpoint: str = "",
) -> FgScalarDecision:
    """Propose one canonical scalar without consulting support/context fields."""
    text = _clean(measurement_text)
    if not text:
        return FgScalarDecision(False, "missing_measurement")
    if _BOUND.search(text):
        return FgScalarDecision(False, "bounded_measurement")
    if _RANGE.search(text):
        return FgScalarDecision(False, "range_measurement")
    if ";" in text:
        return FgScalarDecision(False, "compound_or_multi_outcome")
    if (
        _metric_count(text) > 1
        or len(_TWO_DIRECTIONS.findall(text)) > 1
        or _COORDINATED_OUTCOMES.search(text)
    ):
        return FgScalarDecision(False, "compound_or_multi_outcome")
    if _QUALITATIVE_NEGATION.search(text) and _STANDALONE_NUMBER.search(text):
        return FgScalarDecision(False, "qualitative_outcome_with_numeric_condition")

    for resolver in (_try_ratio, _try_fold, _try_fg_fraction, _try_percentage):
        decision = resolver(text)
        if decision is not None:
            return decision
    physical = _try_physical(text, canonical_endpoint)
    if physical is not None:
        return physical
    return FgScalarDecision(False, _reject_reason(text))


def resolve_fg_measurement_pair(
    record: Mapping[str, Any],
    canonical_endpoint: str,
    baseline_pair: MeasurementPair,
) -> MeasurementPair:
    """Apply the audited FG rule only to FG rows with one accepted outcome."""
    source_id = str(record.get("source_id") or "")
    measurement_text = record.get("measurement_text")
    if source_id != "fg":
        return baseline_pair
    decision = propose_fg_scalar(
        measurement_text,
        canonical_endpoint=canonical_endpoint,
    )
    if not decision.accepted:
        return baseline_pair
    return MeasurementPair(
        decision.canonical_measurement,
        decision.canonical_unit,
        SOURCE_SPECIFIC_ATOMIC_SCALAR_STATUS,
        decision.unit_notation_status,
        decision.unit_notation_factor,
    )


def fg_scalar_rule_provenance(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Return deterministic row-level rule provenance for normalized records."""
    empty = {
        "source_scalar_rule_version": None,
        "source_scalar_rule_id": None,
        "source_scalar_rule_reason": None,
        "scalar_semantic_label": None,
    }
    source_id = str(record.get("source_id") or "")
    if source_id != "fg":
        return empty
    decision = propose_fg_scalar(
        record.get("measurement_text"),
        canonical_endpoint=str(record.get("canonical_endpoint") or ""),
    )
    return {
        "source_scalar_rule_version": FG_SCALAR_RULE_VERSION,
        "source_scalar_rule_id": decision.rule_id,
        "source_scalar_rule_reason": decision.reason,
        "scalar_semantic_label": decision.semantic_label,
    }


__all__ = [
    "FG_SCALAR_RULE_VERSION",
    "FgScalarDecision",
    "fg_scalar_rule_provenance",
    "propose_fg_scalar",
    "resolve_fg_measurement_pair",
]
