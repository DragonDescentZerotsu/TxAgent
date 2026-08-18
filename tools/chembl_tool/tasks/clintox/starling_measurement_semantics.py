"""Reviewed ClinTox measurement and unit semantics for Starling v7.

The common unit parser answers what a unit string means mechanically.  This
module freezes the narrower scientific decision: whether a scalar belongs to a
reviewed ClinTox endpoint/metric stratum, which canonical unit it uses for
assay transfer, and which reference denominator the value describes.  Unknown
or ambiguous strata remain source-visible evidence and fail closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from tools.chembl_tool.common.starling.normalization.contracts import MeasurementPair
from tools.chembl_tool.common.starling.normalization.measurements import (
    normalize_measurement_and_unit,
    parse_point_measurement,
    standardize_measurement_pair,
)
from tools.chembl_tool.common.units import canonicalize_unit


TASK_ID = "clintox"
SEMANTICS_SCHEMA_VERSION = "clintox_measurement_semantics.schema.v1"
MEASUREMENT_SEMANTICS_VERSION = "clintox_measurement_semantics.v2"
DEFAULT_SEMANTICS_PATH = (
    Path(__file__).resolve().parent
    / "data_processing/canonicalization_v7/measurement_semantics.json"
)
_CENSORED_SUPPORT_RULES = {
    "clintox.cytotoxicity.cell_death_percent.v1": (
        r"(?:apoptosis|cell(?:s)?\s+(?:death|killed)|cytotoxicity|haemolysis|"
        r"hemolysis|killed)"
    ),
    "clintox.cytotoxicity.membrane_control_percent.v1": (
        r"(?:cytotoxicity|ldh\s+release|membrane\s+integrity)"
    ),
    "clintox.cytotoxicity.potency_mass.v1": (
        r"(?:(?:cc|ec|ed|gi|ic|lc|mic|tc)\s*\d+)"
    ),
    "clintox.cytotoxicity.potency_molar.v1": (
        r"(?:(?:cc|ec|ed|gi|ic|lc|mic|tc)\s*\d+)"
    ),
    "clintox.cytotoxicity.viability_control_percent.v1": (
        r"(?:cell\s+)?(?:growth|inhibition|survival|viability)"
    ),
}
_CENSOR_QUALIFIER = (
    r"(?:[<>≤≥]=?|at\s+(?:least|most)|no\s+(?:more|less)\s+than|"
    r"(?:equal\s+or\s+)?(?:(?:much|far|significantly)\s+)?"
    r"(?:greater|less)\s+than(?:\s+or\s+equal\s+to)?|more\s+than|"
    r"in\s+excess\s+of)"
)


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _key(value: Any) -> str:
    return re.sub(r"\s+", " ", _text(value).casefold())


@dataclass(frozen=True)
class MeasurementSemantics:
    rule_id: str
    policy_version: str
    status: str
    quantity_kind: str
    numeric_domain: str
    semantic_unit_basis: str | None = None
    target_unit: str | None = None
    default_unit: str | None = None
    reference_scope: str = "unknown"
    reference_basis: str = "unknown"

    def fields(self) -> dict[str, Any]:
        return {
            "canonical_semantics_status": self.status,
            "canonical_quantity_kind": self.quantity_kind,
            "canonical_numeric_domain": self.numeric_domain,
            "canonical_semantics_rule_id": self.rule_id,
            "canonical_semantics_policy_version": self.policy_version,
            "measurement_semantic_unit_basis": self.semantic_unit_basis,
            "canonical_reference_scope_hint": self.reference_scope,
            "canonical_reference_basis_hint": self.reference_basis,
        }


@dataclass(frozen=True)
class _Selector:
    source_ids: tuple[str, ...]
    field_exact: Mapping[str, tuple[str, ...]]
    field_regex: Mapping[str, str]
    unit_kinds: tuple[str, ...]

    def matches(self, record: Mapping[str, Any], unit_kind: str) -> bool:
        source_id = _text(record.get("source_id"))
        if self.source_ids and source_id not in self.source_ids:
            return False
        if self.unit_kinds and unit_kind not in self.unit_kinds:
            return False
        for field, values in self.field_exact.items():
            if _key(record.get(field)) not in values:
                return False
        for field, pattern in self.field_regex.items():
            if re.fullmatch(pattern, _key(record.get(field))) is None:
                return False
        return True


@dataclass(frozen=True)
class _Rule:
    rule_id: str
    priority: int
    selector: _Selector
    semantics: MeasurementSemantics


@dataclass(frozen=True)
class _UnitAlias:
    rule_id: str
    source_ids: tuple[str, ...]
    raw_units: tuple[str, ...]
    replacement_unit: str


@dataclass(frozen=True)
class MeasurementSemanticsPolicy:
    schema_version: str
    policy_version: str
    rules: tuple[_Rule, ...]
    unit_aliases: tuple[_UnitAlias, ...]
    source_sha256: Mapping[str, str]
    path: str
    sha256: str

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "path": self.path,
            "sha256": self.sha256,
            "source_sha256": dict(sorted(self.source_sha256.items())),
            "unknown_policy": "evidence_only",
            "n_rules": len(self.rules),
            "n_unit_aliases": len(self.unit_aliases),
        }


def _semantics(rule_id: str, policy_version: str, raw: Mapping[str, Any]) -> MeasurementSemantics:
    status = str(raw.get("status") or "")
    if status not in {"approved", "evidence_only"}:
        raise ValueError(f"invalid ClinTox semantic status for {rule_id!r}")
    return MeasurementSemantics(
        rule_id=rule_id,
        policy_version=policy_version,
        status=status,
        quantity_kind=str(raw.get("quantity_kind") or "unreviewed"),
        numeric_domain=str(raw.get("numeric_domain") or "unreviewed"),
        semantic_unit_basis=(
            str(raw["semantic_unit_basis"])
            if raw.get("semantic_unit_basis") is not None
            else None
        ),
        target_unit=(str(raw["target_unit"]) if raw.get("target_unit") else None),
        default_unit=(str(raw["default_unit"]) if raw.get("default_unit") else None),
        reference_scope=str(raw.get("reference_scope") or "unknown"),
        reference_basis=str(raw.get("reference_basis") or "unknown"),
    )


@lru_cache(maxsize=4)
def load_measurement_semantics_policy(
    path: str | Path = DEFAULT_SEMANTICS_PATH,
) -> MeasurementSemanticsPolicy:
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SEMANTICS_SCHEMA_VERSION:
        raise ValueError("unexpected ClinTox measurement-semantics schema version")
    policy_version = str(payload.get("policy_version") or "")
    if policy_version != MEASUREMENT_SEMANTICS_VERSION:
        raise ValueError("unexpected ClinTox measurement-semantics policy version")
    from tools.chembl_tool.tasks.clintox.starling_normalization_sources import (
        EXPECTED_SOURCE_SHA256,
    )

    source_sha256 = {
        str(key): str(value)
        for key, value in (payload.get("source_sha256") or {}).items()
    }
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise ValueError("ClinTox measurement-semantics source digests do not match")

    rule_ids: set[str] = set()
    rules: list[_Rule] = []
    for raw in payload.get("rules", []):
        rule_id = str(raw.get("rule_id") or "")
        if not rule_id or rule_id in rule_ids:
            raise ValueError(f"duplicate or empty ClinTox rule ID: {rule_id!r}")
        rule_ids.add(rule_id)
        selector_raw = raw.get("selector") or {}
        field_exact = {
            str(field): tuple(_key(value) for value in values)
            for field, values in (selector_raw.get("field_exact") or {}).items()
        }
        field_regex = {
            str(field): str(pattern)
            for field, pattern in (selector_raw.get("field_regex") or {}).items()
        }
        for pattern in field_regex.values():
            re.compile(pattern)
        semantics = _semantics(rule_id, policy_version, raw.get("action") or {})
        for unit in (semantics.target_unit, semantics.default_unit):
            if unit:
                parsed = canonicalize_unit(unit, task=TASK_ID)
                if not parsed.cleaned or parsed.unknown_tokens:
                    raise ValueError(f"ClinTox rule {rule_id!r} has invalid unit {unit!r}")
        rules.append(
            _Rule(
                rule_id=rule_id,
                priority=int(raw.get("priority", 0)),
                selector=_Selector(
                    source_ids=tuple(str(value) for value in selector_raw.get("source_ids", [])),
                    field_exact=field_exact,
                    field_regex=field_regex,
                    unit_kinds=tuple(str(value) for value in selector_raw.get("unit_kinds", [])),
                ),
                semantics=semantics,
            )
        )

    aliases: list[_UnitAlias] = []
    alias_keys: set[tuple[str, str]] = set()
    for raw in payload.get("unit_aliases", []):
        alias = _UnitAlias(
            rule_id=str(raw.get("rule_id") or ""),
            source_ids=tuple(str(value) for value in raw.get("source_ids", [])),
            raw_units=tuple(_key(value) for value in raw.get("raw_units", [])),
            replacement_unit=str(raw.get("replacement_unit") or ""),
        )
        parsed = canonicalize_unit(alias.replacement_unit, task=TASK_ID)
        if not alias.rule_id or not alias.source_ids or not alias.raw_units or parsed.unknown_tokens:
            raise ValueError(f"invalid ClinTox unit alias {alias.rule_id!r}")
        for source_id in alias.source_ids:
            for raw_unit in alias.raw_units:
                key = (source_id, raw_unit)
                if key in alias_keys:
                    raise ValueError(f"overlapping ClinTox unit alias for {key!r}")
                alias_keys.add(key)
        aliases.append(alias)

    return MeasurementSemanticsPolicy(
        schema_version=SEMANTICS_SCHEMA_VERSION,
        policy_version=policy_version,
        rules=tuple(sorted(rules, key=lambda item: (-item.priority, item.rule_id))),
        unit_aliases=tuple(aliases),
        source_sha256=source_sha256,
        path=str(target),
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    )


def _effective_unit(record: Mapping[str, Any], canonical_unit: Any) -> str:
    source_id = _text(record.get("source_id"))
    raw_unit = _key(record.get("unit_text"))
    for alias in load_measurement_semantics_policy().unit_aliases:
        if source_id in alias.source_ids and raw_unit in alias.raw_units:
            return alias.replacement_unit
    if _text(record.get("unit_text")):
        return _text(record.get("unit_text"))
    return _text(canonical_unit)


def _unit_kind(value: Any) -> str:
    parsed = canonicalize_unit(value, task=TASK_ID)
    if not parsed.cleaned:
        return "missing"
    if parsed.unknown_tokens:
        return "unrecognized"
    canonical = parsed.canonical
    dimensions = dict(parsed.dimension)
    if "%" in canonical:
        return "percent"
    if canonical in {"fraction", "fold", "ratio"}:
        return canonical
    if canonical in {"ppm", "ppb"}:
        return canonical
    if dimensions == {"amount": 1, "volume": -1}:
        return "molar_concentration"
    if dimensions == {"mass": 1, "volume": -1}:
        return "mass_concentration"
    if dimensions == {"amount": 1, "time": 1, "volume": -1}:
        return "molar_auc"
    if dimensions == {"mass": 1, "time": 1, "volume": -1}:
        return "mass_auc"
    if dimensions == {"time": 1}:
        return "duration"
    if (
        dimensions == {"time": -1}
        and re.search(r"(?:^|[·/])kg(?:$|[·/])", canonical)
        and re.search(r"(?:^|[·/])[pnumµk]?g(?:$|[·/])", canonical)
    ):
        return "mass_dose_rate"
    if not dimensions and re.search(r"(?:[pnumµk]?g)/(?:[pnumµk]?g|kg)", canonical):
        return "mass_dose"
    if dimensions == {"volume": 1, "time": -1}:
        return "clearance"
    if dimensions == {"mass": -1, "time": -1, "volume": 1}:
        return "mass_normalized_clearance"
    if dimensions == {"time": -1}:
        return "rate_constant"
    if dimensions == {"mass": 1}:
        return "mass"
    return "recognized"


def resolve_measurement_semantics(
    record: Mapping[str, Any],
    canonical_endpoint: str | None = None,
    canonical_unit: Any = None,
) -> MeasurementSemantics:
    policy = load_measurement_semantics_policy()
    endpoint = _text(canonical_endpoint) or _text(record.get("canonical_endpoint"))
    probe = {**record, "canonical_endpoint": endpoint}
    unit = _effective_unit(record, canonical_unit if canonical_unit is not None else record.get("canonical_unit"))
    kind = _unit_kind(unit)
    matches = [rule for rule in policy.rules if rule.selector.matches(probe, kind)]
    if matches:
        highest = matches[0].priority
        winners = [rule for rule in matches if rule.priority == highest]
        if len(winners) != 1:
            raise ValueError(
                f"ambiguous ClinTox measurement semantics for {record.get('source_id')}/"
                f"{endpoint}: {[rule.rule_id for rule in winners]}"
            )
        return winners[0].semantics
    return MeasurementSemantics(
        rule_id="clintox.unreviewed_measurement.v1",
        policy_version=policy.policy_version,
        status="evidence_only",
        quantity_kind="unreviewed",
        numeric_domain="unreviewed",
    )


def resolve_measurement_pair(
    record: Mapping[str, Any],
    canonical_endpoint: str,
    pair: MeasurementPair,
) -> MeasurementPair:
    """Apply reviewed exact aliases/defaults and one atomic unit conversion."""
    effective = _effective_unit(record, pair.canonical_unit)
    working = pair
    if effective and effective != pair.canonical_unit and record.get("measurement_text") not in (None, ""):
        resolved = normalize_measurement_and_unit(
            record.get("measurement_text"), effective, task=TASK_ID
        )
        working = MeasurementPair(
            resolved.canonical_measurement,
            resolved.canonical_unit,
            "reviewed_unit_alias",
            resolved.unit_notation_status,
            resolved.unit_notation_factor,
        )
    semantics = resolve_measurement_semantics(
        record, canonical_endpoint, working.canonical_unit
    )
    if semantics.status != "approved":
        return working
    if (
        not working.canonical_unit
        and semantics.default_unit
        and parse_point_measurement(working.canonical_measurement).value is not None
    ):
        resolved = normalize_measurement_and_unit(
            working.canonical_measurement, semantics.default_unit, task=TASK_ID
        )
        working = MeasurementPair(
            resolved.canonical_measurement,
            resolved.canonical_unit,
            "reviewed_metric_default_unit",
            resolved.unit_notation_status,
            resolved.unit_notation_factor,
        )
    if semantics.target_unit and working.canonical_unit:
        standardized = standardize_measurement_pair(
            working,
            targets=(semantics.target_unit,),
            task=TASK_ID,
        )
        if standardized.status == "endpoint_standardized":
            return MeasurementPair(
                standardized.canonical_measurement,
                standardized.canonical_unit,
                "reviewed_semantic_standardization",
                standardized.unit_notation_status,
                standardized.unit_notation_factor,
            )
    return working


def censored_support_error(
    record: Mapping[str, Any], semantics: MeasurementSemantics
) -> str | None:
    """Reject a point whose source support explicitly reports the same value as a bound.

    Starling occasionally drops ``<``/``>`` from ``measurement_text`` while
    retaining it in ``support_text``.  This check is deliberately limited to
    reviewed potency and cell-response rules, requires the same numeric point,
    and requires the matching endpoint concept in the same sentence fragment.
    Source-visible evidence remains unchanged; only assay transfer fails closed.
    """
    anchor = _CENSORED_SUPPORT_RULES.get(semantics.rule_id)
    if not anchor:
        return None
    raw = _text(record.get("measurement_text")).replace(",", "")
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", raw) is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    forms = {raw, format(value, ".15g")}
    if value.is_integer():
        forms.add(str(int(value)))
    number = "(?:" + "|".join(
        re.escape(item) for item in sorted(forms, key=len, reverse=True)
    ) + ")"
    if "potency" in semantics.rule_id:
        unit_suffix = r"\s*(?:[fpnumµ]?M|[fpnumµk]?g\s*(?:/|·)\s*(?:mL|L))"
    else:
        unit_suffix = r"\s*%"
    support = _text(record.get("support_text")).replace(",", "")
    bounded_value = rf"{number}(?![\d.]){unit_suffix}"
    before = rf"{anchor}[^.;\n]{{0,100}}?{_CENSOR_QUALIFIER}\s*{bounded_value}"
    after = rf"{_CENSOR_QUALIFIER}\s*{bounded_value}[^.;\n]{{0,60}}?{anchor}"
    if re.search(before, support, flags=re.IGNORECASE) or re.search(
        after, support, flags=re.IGNORECASE
    ):
        return "support_reports_censored_value"
    return None


def numeric_domain_error(semantics: MeasurementSemantics, value: Any) -> str | None:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return "non_scalar_measurement"
    if not math.isfinite(scalar):
        return "non_scalar_measurement"
    domain = semantics.numeric_domain
    if domain in {"finite_signed", "declared_categorical"}:
        return None
    if domain == "positive":
        return None if scalar > 0 else "outside_positive_domain"
    if domain == "nonnegative":
        return None if scalar >= 0 else "outside_nonnegative_domain"
    if domain == "bounded_0_100":
        return None if 0 <= scalar <= 100 else "outside_bounded_0_100_domain"
    if domain == "bounded_0_1":
        return None if 0 <= scalar <= 1 else "outside_bounded_0_1_domain"
    return "unreviewed_measurement_semantics"


def measurement_semantics_audit(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    policy = load_measurement_semantics_policy()
    status_counts: dict[str, int] = {}
    rule_counts: dict[str, int] = {}
    validity_counts: dict[str, int] = {}
    for record in records:
        status = _text(record.get("canonical_semantics_status"))
        rule_id = _text(record.get("canonical_semantics_rule_id"))
        if not status or not rule_id:
            semantics = resolve_measurement_semantics(record)
            status = semantics.status
            rule_id = semantics.rule_id
        status_counts[status] = status_counts.get(status, 0) + 1
        rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1
        validity = _text(record.get("normalization_validity_status")) or "missing"
        validity_counts[validity] = validity_counts.get(validity, 0) + 1
    return {
        **policy.manifest(),
        "status_counts": dict(sorted(status_counts.items())),
        "rule_counts": dict(sorted(rule_counts.items())),
        "normalization_validity_counts": dict(sorted(validity_counts.items())),
    }


__all__ = [
    "DEFAULT_SEMANTICS_PATH",
    "MEASUREMENT_SEMANTICS_VERSION",
    "MeasurementSemantics",
    "load_measurement_semantics_policy",
    "measurement_semantics_audit",
    "numeric_domain_error",
    "censored_support_error",
    "resolve_measurement_pair",
    "resolve_measurement_semantics",
]
