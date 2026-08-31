"""Deterministic routing of cleaned rows into measurement-resolution buckets.

The scalar router settles a cleaned row into exactly one of three routes:

``reject``
    The source's measurement column carries no digit, so there is no quantity to
    extract. Stage 01 then checks the controlled categorical encoder, so a persisted
    rejection has neither an explicit controlled category nor a scalar extraction
    candidate. The source row is still retained.

``accept``
    The measurement column is a finite positive numeric literal accepted by
    Python ``float(...)`` and has a nonempty unit. Value and unit are taken
    exactly as written, with no model call. Stage 02's exact JSON map is the sole
    authority for converting or preserving the pair.

``extract``
    Everything else, resolved by a frozen offline extraction.

Stage 01 first preserves any task-declared structured source-exact quantity,
then routes the source measurement column. A controlled categorical encoder is
only the fallback when neither path finds a numeric candidate. The source row
itself is not modified.

This module is imported by both the offline generator and the runtime resolver.
Identical routing in both places is what makes the frozen artifact's candidate set
reproducible, so it must stay a pure function of one cleaned row plus its declared
rules -- no I/O, no clock, no task-specific special cases.
"""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Literal

from tools.chembl_tool.common.units import canonicalize_unit


MEASUREMENT_ROUTING_VERSION = "starling_measurement_routing.v5"

_DIGIT_RE = re.compile(r"\d")

#: Canonical forms whose denominator is not stated, so a bare number carrying one
#: cannot be shown absolute: ``45%`` may be a percent of applied dose, of a
#: control, or of a cohort.  Matched on the canonical form so every spelling
#: collapses here -- ``percent`` and ``per cent`` both canonicalize to ``%``.
BARE_PROPORTION_CANONICALS = frozenset(
    {"%", "fold", "ratio", "ppm", "ppb", "x", "×", "times", "fold·change"}
)

#: No reviewed ADME unit exceeds a cubic power in any base dimension.  OCR damage
#: produces forms like ``10-6 cm s-6``, which resolves cleanly to ``cm/s^6``.
MAX_PLAUSIBLE_DIMENSION_EXPONENT = 3

NO_DIGIT_RULE_ID = "no_digit_in_measurement_column.v1"
PURE_NUMBER_RULE_ID = "positive_decimal_exact_unit.v1"

ROUTE_BUCKETS = ("reject", "accept", "extract")
STAGE1_ROUTE_BUCKETS = ("categorical", *ROUTE_BUCKETS)


@dataclass(frozen=True)
class DeclarativeNonScalarRule:
    """One source column whose declared values mean 'no absolute quantity'.

    Retained for columns that describe the *measurement* -- bioavailability's
    ``bioavailability_report_type == 'relative_comparison'`` states outright that
    the reported value is referenced to a comparator.  BBB declares none: four
    candidates were audited and all covered rows carrying a hard measurement,
    because those columns record the conclusion a study drew rather than whether a
    quantity was measured.
    """

    rule_id: str
    field: str
    values: frozenset[str]

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("declarative rule_id must be nonempty")
        if not self.field:
            raise ValueError(f"declarative rule {self.rule_id!r} has no field")
        if not self.values:
            raise ValueError(f"declarative rule {self.rule_id!r} declares no values")
        if any(not value for value in self.values):
            raise ValueError(
                f"declarative rule {self.rule_id!r} declares an empty value; a "
                "missing column is not a declaration"
            )

    def fires(self, record: Mapping[str, Any]) -> bool:
        value = record.get(self.field)
        if value is None:
            return False
        return str(value).strip() in self.values

    def manifest(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "field": self.field,
            "values": sorted(self.values),
        }


@dataclass(frozen=True)
class SourceRoutingRules:
    """The complete reviewed rule set for one source."""

    source_id: str
    measurement_field: str = "measurement_text"
    unit_field: str = "unit_text"
    declarative_non_scalar: tuple[DeclarativeNonScalarRule, ...] = ()
    #: Refuse a non-positive value on the accept path.  An obvious unit names a
    #: strictly positive physical quantity, so a negative value means the *unit
    #: label* is wrong: 238 of 240 negative BBB candidates are log values whose
    #: endpoint reads ``log Pe`` or ``Log Kp`` while the unit column still says
    #: ``cm/s``.  Accepting -6.36 as cm/s would place a logarithm on the same axis
    #: as linear permeabilities.
    require_positive_value: bool = True

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must be nonempty")
        if not self.measurement_field:
            raise ValueError(f"{self.source_id!r} has no measurement field")
        rule_ids = [rule.rule_id for rule in self.declarative_non_scalar]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError(f"{self.source_id!r} repeats a declarative rule ID")
        if NO_DIGIT_RULE_ID in rule_ids or PURE_NUMBER_RULE_ID in rule_ids:
            raise ValueError(
                f"{self.source_id!r} reuses a reserved rule ID for a declarative rule"
            )

    def manifest(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "routing_version": MEASUREMENT_ROUTING_VERSION,
            "measurement_field": self.measurement_field,
            "unit_field": self.unit_field or None,
            "require_positive_value": self.require_positive_value,
            "declarative_non_scalar": [
                rule.manifest() for rule in self.declarative_non_scalar
            ],
        }


@dataclass(frozen=True)
class RouteDecision:
    """Which bucket a row lands in, and the value pair when a rule settles it."""

    bucket: Literal["reject", "accept", "extract"]
    rule_id: str | None = None
    measurement_text: str | None = None
    unit_text: str | None = None
    unit_is_canonical: bool = False

    def __post_init__(self) -> None:
        if self.bucket not in ROUTE_BUCKETS:
            raise ValueError(f"unsupported route bucket: {self.bucket!r}")
        if self.bucket == "accept":
            if self.measurement_text is None:
                raise ValueError("accept must carry a measurement")
            if self.unit_text is None:
                raise ValueError("accept must carry a unit")
            if self.rule_id is None:
                raise ValueError("accept must name the rule that fired")
        if self.bucket == "reject" and self.rule_id is None:
            raise ValueError("reject must name the rule that fired")
        if self.bucket == "extract" and (
            self.rule_id is not None
            or self.measurement_text is not None
            or self.unit_text is not None
            or self.unit_is_canonical
        ):
            raise ValueError("extract rows are unresolved and carry no rule output")
        if self.bucket != "accept" and self.unit_is_canonical:
            raise ValueError("only an accept route can declare a canonical unit")


def _measurement_source_text(value: Any) -> str | None:
    """Render a cleaned measurement value for routing.

    ``clean_scalar`` preserves genuine numerics, so a source column typed as a
    float arrives here as a float rather than a string.  An integral float renders
    without the trailing zero: formatting it as ``4.0`` would introduce a spelling
    the source never had.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        # A boolean is not a measurement, and ``isinstance(True, int)`` is True.
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return repr(int(value)) if value.is_integer() else repr(value)
    text = str(value).strip()
    return text or None


def has_digit(value: Any) -> bool:
    text = _measurement_source_text(value)
    return bool(text) and bool(_DIGIT_RE.search(text))


def is_pure_number(value: Any) -> bool:
    text = _measurement_source_text(value)
    if not text:
        return False
    try:
        number = float(text)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number)


def is_obvious_unit(unit_text: Any, *, task: str | None = None) -> bool:
    """Historical diagnostic; active routing delegates every unit to the exact map.

    Three tests, each closing a way the pair could be wrong:

    * every token resolves, so an unreviewed word cannot ride along inside a unit;
    * the denominator is stated.  Note this is *not* ``is_dimensionless``:
      ``mg/kg`` and ``ng/g tissue`` are dimensionally dimensionless because
      mass/mass cancels, yet they are absolute concentrations, while ``%`` and
      ``fold`` name no denominator at all;
    * no embedded power-of-ten scale factor.  ``cm s^-1`` and ``ml g^-1 min^-1``
      qualify -- their digits are dimension exponents, which are unambiguous --
      but ``10^-6 cm/s`` does not, because a scale factor inside a unit label is a
      presentation convention whose readings disagree.  A header ``Papp x 10^6
      (cm/s)`` means the printed number *is* Papp x 10^6, so the true value is
      smaller; the literal reading multiplies instead, turning carbamazepine's
      PAMPA permeability of 1.15e-05 cm/s into 1.15e+07.
    """
    if unit_text is None:
        return False
    text = str(unit_text).strip()
    if not text:
        return False
    resolved = canonicalize_unit(text, task=task)
    if resolved.unknown_tokens or not resolved.canonical:
        return False
    if resolved.canonical in BARE_PROPORTION_CANONICALS:
        return False
    if resolved.notation_factor is not None:
        return False
    if resolved.transform:
        return False
    return all(
        abs(exponent) <= MAX_PLAUSIBLE_DIMENSION_EXPONENT
        for _, exponent in resolved.dimension
    )


def route(
    record: Mapping[str, Any],
    rules: SourceRoutingRules,
    *,
    task: str | None = None,
) -> RouteDecision:
    """Assign one cleaned row to exactly one resolution route."""
    measurement = record.get(rules.measurement_field)
    if not has_digit(measurement):
        return RouteDecision("reject", NO_DIGIT_RULE_ID)
    for rule in rules.declarative_non_scalar:
        if rule.fires(record):
            return RouteDecision("reject", rule.rule_id)
    if rules.unit_field and is_pure_number(measurement):
        text = _measurement_source_text(measurement)
        positive = text is not None and float(text) > 0
        if positive or not rules.require_positive_value:
            unit = record.get(rules.unit_field)
            unit_text = str(unit or "").strip()
            if unit_text:
                return RouteDecision(
                    "accept", PURE_NUMBER_RULE_ID, text, unit_text
                )
    return RouteDecision("extract")


def attach_stage1_routes(
    records: list[dict[str, Any]],
    *,
    task: str,
    endpoint_resolver: Callable[[Mapping[str, Any]], str] | None = None,
) -> list[dict[str, Any]]:
    """Annotate cleaned rows with the endpoint and route consumed by Stage 02."""
    config = importlib.import_module(
        f"tools.chembl_tool.tasks.{task}.starling_measurement_resolution"
    )
    rules_by_source = config.source_routing_rules()
    categorical_module = (
        f"tools.chembl_tool.tasks.{task}.starling_categorical_response"
    )
    try:
        module = importlib.import_module(categorical_module)
    except ModuleNotFoundError as error:
        if error.name != categorical_module:
            raise
        categorical = None
    else:
        categorical = getattr(module, "POLICY", None)
    source_exact_route = getattr(config, "source_exact_route", None)
    record_resolver = getattr(config, "canonical_endpoint_record", None)
    output: list[dict[str, Any]] = []
    for record in records:
        source_id = str(record.get("source_id") or "")
        rules = rules_by_source.get(source_id)
        if rules is None:
            raise ValueError(f"no measurement-routing rules for {source_id!r}")
        if endpoint_resolver is not None:
            endpoint = endpoint_resolver(record)
        elif record_resolver is not None:
            endpoint = record_resolver(record)
        else:
            endpoint = config.canonical_endpoint_name(
                source_id, record.get("endpoint_name")
            )
        decision = (
            source_exact_route(record) if source_exact_route is not None else None
        )
        if decision is not None and decision.bucket != "accept":
            raise ValueError("source_exact_route must return an accept decision")
        decision = decision or route(record, rules, task=task)
        encoded = (
            categorical.encode(record)
            if decision.bucket == "reject" and categorical is not None
            else None
        )
        # Stage-01 builder rows are mutable dictionaries. Reuse them so routing a
        # full corpus does not briefly duplicate the complete object graph.
        enriched = record if isinstance(record, dict) else dict(record)
        enriched.update(
            {
                "canonical_endpoint_name": str(endpoint or "missing_endpoint"),
                "measurement_resolution_route": (
                    "categorical" if encoded is not None else decision.bucket
                ),
                "measurement_resolution_rule_id": (
                    encoded.encoder_id if encoded is not None else decision.rule_id
                ),
                "measurement_resolution_exact_measurement": (
                    decision.measurement_text
                    if encoded is None and decision.bucket == "accept"
                    else None
                ),
                "measurement_resolution_exact_unit": (
                    decision.unit_text
                    if encoded is None and decision.bucket == "accept"
                    else None
                ),
                "measurement_resolution_exact_unit_is_canonical": (
                    decision.unit_is_canonical
                    if encoded is None and decision.bucket == "accept"
                    else False
                ),
                "measurement_routing_version": MEASUREMENT_ROUTING_VERSION,
            }
        )
        output.append(enriched)
    return output


__all__ = [
    "BARE_PROPORTION_CANONICALS",
    "DeclarativeNonScalarRule",
    "MAX_PLAUSIBLE_DIMENSION_EXPONENT",
    "MEASUREMENT_ROUTING_VERSION",
    "NO_DIGIT_RULE_ID",
    "PURE_NUMBER_RULE_ID",
    "ROUTE_BUCKETS",
    "STAGE1_ROUTE_BUCKETS",
    "RouteDecision",
    "SourceRoutingRules",
    "attach_stage1_routes",
    "has_digit",
    "is_obvious_unit",
    "is_pure_number",
    "route",
]
