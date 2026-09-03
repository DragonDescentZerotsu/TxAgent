"""Frozen endpoint, numeric-domain, and qualified-unit semantics for BBB v7.

The common parser answers what a unit string mechanically means.  This module
answers the task-specific question of whether that unit and numeric domain are
valid for a BBB endpoint.  Unknown endpoints fail closed: their rows remain
available as evidence but cannot enter pair buckets or distance calibration.
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

from tools.chembl_tool.common.units import canonicalize_unit, units_compatible


SEMANTICS_SCHEMA_VERSION = "bbb_martins_measurement_semantics.schema.v1"
DEFAULT_SEMANTICS_PATH = (
    Path(__file__).resolve().parent
    / "data_processing/canonicalization_v7/measurement_semantics.v1.json"
)

# Qualifier vocabulary for every unit parsed by this task.
_TASK_VOCAB = "bbb_martins"



def _key(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def _text(value: Any) -> str:
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)


@dataclass(frozen=True)
class MeasurementSemantics:
    rule_id: str
    policy_version: str
    status: str
    quantity_kind: str
    numeric_domain: str
    compatible_units: tuple[str, ...] = ()
    compatible_unit_kinds: tuple[str, ...] = ()
    default_unit: str | None = None

    def fields(self) -> dict[str, Any]:
        return {
            "canonical_semantics_status": self.status,
            "canonical_quantity_kind": self.quantity_kind,
            "canonical_numeric_domain": self.numeric_domain,
            "canonical_semantics_rule_id": self.rule_id,
            "canonical_semantics_policy_version": self.policy_version,
        }


@dataclass(frozen=True)
class _Rule:
    rule_id: str
    priority: int
    source_ids: tuple[str, ...]
    endpoint_exact: tuple[str, ...]
    endpoint_regex: tuple[str, ...]
    raw_endpoint_exact: tuple[str, ...]
    semantics: MeasurementSemantics

    def matches(self, record: Mapping[str, Any], endpoint: str) -> bool:
        source_id = _text(record.get("source_id"))
        if self.source_ids and source_id not in self.source_ids:
            return False
        raw_endpoint = _key(record.get("endpoint_name"))
        if self.raw_endpoint_exact and raw_endpoint not in self.raw_endpoint_exact:
            return False
        if self.endpoint_exact and endpoint not in self.endpoint_exact:
            return False
        if self.endpoint_regex and not any(
            re.fullmatch(pattern, endpoint) for pattern in self.endpoint_regex
        ):
            return False
        return bool(
            self.raw_endpoint_exact or self.endpoint_exact or self.endpoint_regex
        )


@dataclass(frozen=True)
class _UnitAlias:
    rule_id: str
    source_ids: tuple[str, ...]
    raw_units: tuple[str, ...]
    canonical_unit: str


@dataclass(frozen=True)
class MeasurementSemanticsPolicy:
    schema_version: str
    policy_version: str
    rules: tuple[_Rule, ...]
    unit_aliases: tuple[_UnitAlias, ...]
    path: str
    sha256: str

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "path": self.path,
            "sha256": self.sha256,
            "unknown_endpoint_policy": "evidence_only",
            "n_rules": len(self.rules),
            "n_unit_aliases": len(self.unit_aliases),
        }


@lru_cache(maxsize=4)
def load_measurement_semantics_policy(
    path: str | Path = DEFAULT_SEMANTICS_PATH,
) -> MeasurementSemanticsPolicy:
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SEMANTICS_SCHEMA_VERSION:
        raise ValueError("unexpected BBB measurement-semantics schema version")
    policy_version = str(payload.get("policy_version") or "")
    if not policy_version:
        raise ValueError("BBB measurement-semantics policy has no version")
    rules: list[_Rule] = []
    rule_ids: set[str] = set()
    for raw in payload.get("rules", []):
        rule_id = str(raw.get("rule_id") or "")
        if not rule_id or rule_id in rule_ids:
            raise ValueError(f"duplicate or empty BBB semantics rule ID: {rule_id!r}")
        rule_ids.add(rule_id)
        status = str(raw.get("status") or "")
        if status not in {"approved", "evidence_only"}:
            raise ValueError(f"invalid BBB semantics status for {rule_id!r}")
        compatible_units = tuple(str(value) for value in raw.get("compatible_units", []))
        for unit in compatible_units:
            parsed = canonicalize_unit(unit, task=_TASK_VOCAB)
            if not parsed.cleaned or parsed.unknown_tokens:
                raise ValueError(
                    f"BBB semantics rule {rule_id!r} has unknown unit {unit!r}"
                )
        default_unit = raw.get("default_unit")
        if default_unit is not None and str(default_unit) not in compatible_units:
            raise ValueError(
                f"BBB semantics rule {rule_id!r} default is not compatible"
            )
        semantics = MeasurementSemantics(
            rule_id=rule_id,
            policy_version=policy_version,
            status=status,
            quantity_kind=str(raw.get("quantity_kind") or "unreviewed"),
            numeric_domain=str(raw.get("numeric_domain") or "unreviewed"),
            compatible_units=compatible_units,
            compatible_unit_kinds=tuple(
                str(value) for value in raw.get("compatible_unit_kinds", [])
            ),
            default_unit=str(default_unit) if default_unit is not None else None,
        )
        rules.append(
            _Rule(
                rule_id=rule_id,
                priority=int(raw.get("priority", 0)),
                source_ids=tuple(str(value) for value in raw.get("source_ids", [])),
                endpoint_exact=tuple(
                    str(value).casefold() for value in raw.get("endpoint_exact", [])
                ),
                endpoint_regex=tuple(
                    str(value) for value in raw.get("endpoint_regex", [])
                ),
                raw_endpoint_exact=tuple(
                    _key(value) for value in raw.get("raw_endpoint_exact", [])
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
            raw_units=tuple(
                dict.fromkeys(_key(value) for value in raw.get("raw_units", []))
            ),
            canonical_unit=str(raw.get("canonical_unit") or ""),
        )
        parsed = canonicalize_unit(alias.canonical_unit, task=_TASK_VOCAB)
        if not alias.rule_id or not alias.raw_units or not parsed.cleaned or parsed.unknown_tokens:
            raise ValueError(f"invalid BBB qualified-unit alias {alias.rule_id!r}")
        for source_id in alias.source_ids:
            for raw_unit in alias.raw_units:
                key = (source_id, raw_unit)
                if key in alias_keys:
                    raise ValueError(f"overlapping BBB unit alias for {key!r}")
                alias_keys.add(key)
        aliases.append(alias)
    return MeasurementSemanticsPolicy(
        schema_version=SEMANTICS_SCHEMA_VERSION,
        policy_version=policy_version,
        rules=tuple(sorted(rules, key=lambda item: (-item.priority, item.rule_id))),
        unit_aliases=tuple(aliases),
        path=str(target),
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    )


def resolve_measurement_semantics(
    record: Mapping[str, Any],
    canonical_endpoint: str | None = None,
) -> MeasurementSemantics:
    scale_id = _text(record.get("categorical_encoder_id")) or _text(
        record.get("canonical_measurement_scale_id")
    )
    endpoint = (
        _text(canonical_endpoint)
        or _text(record.get("canonical_endpoint"))
        or _text(record.get("canonical_endpoint_name"))
    ).casefold()
    return _resolve_measurement_semantics_cached(
        _text(record.get("source_id")),
        _key(record.get("endpoint_name")),
        endpoint,
        scale_id,
    )


@lru_cache(maxsize=32_768)
def _resolve_measurement_semantics_cached(
    source_id: str,
    raw_endpoint: str,
    endpoint: str,
    scale_id: str,
) -> MeasurementSemantics:
    """Resolve one declared semantic stratum once across all record passes."""
    policy = load_measurement_semantics_policy()
    if scale_id:
        return MeasurementSemantics(
            rule_id=scale_id,
            policy_version=policy.policy_version,
            status="approved",
            quantity_kind="controlled_categorical_outcome",
            numeric_domain="declared_categorical",
            compatible_units=("binary_outcome_class",),
        )
    probe = {"source_id": source_id, "endpoint_name": raw_endpoint}
    matches = [rule for rule in policy.rules if rule.matches(probe, endpoint)]
    if matches:
        highest = matches[0].priority
        winners = [rule for rule in matches if rule.priority == highest]
        if len(winners) != 1:
            raise ValueError(
                f"ambiguous BBB measurement semantics for "
                f"{source_id}/{endpoint}: "
                f"{[rule.rule_id for rule in winners]}"
            )
        return winners[0].semantics
    return MeasurementSemantics(
        rule_id="bbb.unreviewed_endpoint.v1",
        policy_version=policy.policy_version,
        status="evidence_only",
        quantity_kind="unreviewed",
        numeric_domain="unreviewed",
    )


def semantics_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    return resolve_measurement_semantics(record).fields()


def resolve_qualified_unit_alias(
    record: Mapping[str, Any],
) -> tuple[str, str] | None:
    policy = load_measurement_semantics_policy()
    source_id = _text(record.get("source_id"))
    raw_unit = _key(record.get("unit_text"))
    if not raw_unit:
        return None
    matches = [
        alias
        for alias in policy.unit_aliases
        if source_id in alias.source_ids and raw_unit in alias.raw_units
    ]
    if len(matches) > 1:
        raise ValueError(f"multiple BBB qualified-unit aliases matched {raw_unit!r}")
    if not matches:
        return None
    return matches[0].canonical_unit, matches[0].rule_id


def unit_is_compatible(semantics: MeasurementSemantics, unit: Any) -> bool | None:
    parsed = canonicalize_unit(unit, task=_TASK_VOCAB)
    if not parsed.cleaned:
        return None
    if parsed.canonical in semantics.compatible_units:
        return True
    if parsed.unknown_tokens:
        return None
    for kind in semantics.compatible_unit_kinds:
        if kind == "recognized":
            return True
        if kind == "concentration":
            if units_compatible(unit, "concentration", task=_TASK_VOCAB) is True:
                return True
            if units_compatible(unit, "mass_concentration", task=_TASK_VOCAB) is True:
                return True
            # Tissue concentrations such as ng/g and nmol/g are dimensionally
            # cancelled by the simple engine but retain an explicit mass basis.
            if re.search(r"/(?:[pnumµk]?g)(?:$|[·/])", parsed.canonical):
                return True
            if re.search(r"/(?:brain|tissue)(?:$|[·/])", parsed.canonical) and any(
                dimension in dict(parsed.dimension) for dimension in {"mass", "amount"}
            ):
                return True
            continue
        if kind == "auc":
            if units_compatible(unit, "auc", task=_TASK_VOCAB) is True or parsed.dimension == (
                ("amount", 1),
                ("time", 1),
                ("volume", -1),
            ):
                return True
            # Tissue AUC is commonly time·mass/mass (for example h·ng/g),
            # whose mass dimensions cancel in the generic parser.
            if (
                parsed.dimension == (("time", 1),)
                and re.search(r"/(?:[pnumµk]?g)(?:$|[·/])", parsed.canonical)
            ):
                return True
            continue
        if kind == "rate_constant":
            if parsed.dimension == (("time", -1),):
                return True
            continue
        if kind == "influx_clearance_rate":
            if parsed.dimension in {
                (("time", -1),),
                (("time", -1), ("volume", 1)),
                (("mass", -1), ("time", -1), ("volume", 1)),
                (("length", 3), ("mass", -1), ("time", -1)),
            }:
                return True
            continue
        try:
            compatible = units_compatible(unit, kind, task=_TASK_VOCAB)
        except KeyError:
            raise ValueError(f"unknown BBB compatible unit kind: {kind!r}") from None
        if compatible is True:
            return True
    return False


def numeric_domain_status(
    semantics: MeasurementSemantics,
    value: Any,
) -> str | None:
    if isinstance(value, bool):
        return "non_scalar_measurement"
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return "non_scalar_measurement"
    if not math.isfinite(scalar):
        return "non_scalar_measurement"
    domain = semantics.numeric_domain
    if domain in {"finite_signed", "declared_categorical"}:
        return None
    if domain == "nonnegative":
        return None if scalar >= 0.0 else "outside_nonnegative_domain"
    if domain == "bounded_0_100":
        return None if 0.0 <= scalar <= 100.0 else "outside_bounded_0_100_domain"
    if domain == "bounded_0_1":
        return None if 0.0 <= scalar <= 1.0 else "outside_bounded_0_1_domain"
    return "unreviewed_endpoint_semantics"


def measurement_semantics_audit(
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    policy = load_measurement_semantics_policy()
    decisions: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    status_counts: dict[str, int] = {}
    for record in records:
        semantics = resolve_measurement_semantics(record)
        status_counts[semantics.status] = status_counts.get(semantics.status, 0) + 1
        key = (
            _text(record.get("source_id")),
            _text(record.get("endpoint_name")),
            _text(record.get("canonical_endpoint"))
            or _text(record.get("canonical_endpoint_name")),
            _text(record.get("canonical_unit"))
            or _text(record.get("canonical_unit_text")),
            _text(record.get("categorical_encoder_id"))
            or _text(record.get("canonical_measurement_scale_id")),
        )
        item = decisions.setdefault(
            key,
            {
                "source_id": key[0],
                "endpoint_name": key[1],
                "canonical_endpoint": key[2],
                "canonical_unit": key[3] or None,
                "canonical_measurement_scale_id": key[4] or None,
                **semantics.fields(),
                "record_count": 0,
                "valid_record_count": 0,
            },
        )
        item["record_count"] += 1
        item["valid_record_count"] += int(
            (
                record.get("normalization_validity_status")
                or record.get("canonicalization_status")
            )
            == "valid"
        )
    return {
        **policy.manifest(),
        "record_status_counts": dict(sorted(status_counts.items())),
        "inventory_count": len(decisions),
        "inventory": [decisions[key] for key in sorted(decisions)],
    }


__all__ = [
    "DEFAULT_SEMANTICS_PATH",
    "MeasurementSemantics",
    "load_measurement_semantics_policy",
    "measurement_semantics_audit",
    "numeric_domain_status",
    "resolve_measurement_semantics",
    "resolve_qualified_unit_alias",
    "semantics_fields",
    "unit_is_compatible",
]
