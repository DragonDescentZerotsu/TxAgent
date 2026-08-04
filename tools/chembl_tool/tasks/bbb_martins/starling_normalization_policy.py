"""Reviewed scalar and unit normalization rules for BBB Martins v6."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from tools.chembl_tool.common.starling.normalization.contracts import MeasurementPair
from tools.chembl_tool.common.starling.normalization.cleaning import clean_text
from tools.chembl_tool.common.starling.normalization.measurements import (
    normalize_measurement_and_unit,
    parse_point_measurement,
    render_point_measurement,
    standardize_measurement_pair,
)
from tools.chembl_tool.common.units import canonicalize_unit, units_compatible


ENDPOINT_POLICY_VERSION = "bbb_martins_endpoint_unit_policy.v2"
SOURCE_MEASUREMENT_RESOLVER_VERSION = "bbb_martins_source_measurement_resolver.v2"

_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?(?:[eE][-+]?\d+)?|[-+]?\.\d+(?:[eE][-+]?\d+)?"
_EMBEDDED_MEASUREMENT_UNIT = re.compile(
    rf"^\s*(?P<measurement>(?:≈|~|about\s+|approx(?:imately)?\.?\s+|ca\.?\s+)?"
    rf"{_NUMBER}(?:\s*(?:±|\+/-)\s*{_NUMBER})?)\s*(?P<unit>\S.+?)\s*$",
    re.IGNORECASE,
)
_PERMEABILITY_UNIT_IN_ENDPOINT = re.compile(
    r"(?P<unit>(?:(?:×|x)\s*)?10(?:\^\(?-?\d+\)?|-\d+)\s*cm\s*(?:/\s*(?:s|sec|second)|[·. ]\s*s\s*\^?-?1)|"
    r"cm\s*(?:/\s*(?:s|sec|second)|[·. ]\s*s\s*\^?-?1)(?:\s*(?:×|x)\s*10\s*\^?-?\d+)?|"
    r"nm\s*(?:/\s*(?:s|sec|second)|[·. ]\s*s\s*\^?-?1))",
    re.IGNORECASE,
)

_PERMEABILITY_ENDPOINTS = frozenset(
    {
        "apparent_permeability",
        "papp_a_to_b",
        "papp_b_to_a",
        "effective_permeability",
        "effective_permeability_a_to_b",
        "intrinsic_permeability",
        "passive_permeability",
    }
)
_PERCENT_ENDPOINTS = frozenset(
    {"percent_transport", "percent_transport_a_to_b", "percent_dose_permeated", "percent_inhibition"}
)
_RATIO_ENDPOINTS = frozenset(
    {
        "brain_to_blood_ratio",
        "brain_to_plasma_ratio",
        "efflux_ratio",
        "relative_permeability_ratio",
    }
)
_FOLD_ENDPOINTS = frozenset(
    {
        "brain_exposure_fold_change",
        "concentration_fold_change",
        "fold_change",
        "permeability_fold_change",
    }
)
_FRACTION_ENDPOINTS = frozenset({"probability"})
_DIMENSIONLESS_ENDPOINTS = frozenset(
    {
        "logbb",
        "intrinsic_partition_coefficient",
        "membrane_partitioning",
        "negative_log_partition_coefficient",
        "permeability_class",
    }
)
_LOG_PERMEABILITY_DEFAULTS = {
    "log_apparent_permeability": "log10(cm/s)",
    "log_effective_permeability": "log10(cm/s)",
    "log_passive_permeability": "log10(cm/s)",
    "negative_log_effective_permeability": "-log10(cm/s)",
}


def _token(value: Any) -> str:
    return re.sub(r"[^a-z0-9%]+", "_", str(value or "").casefold()).strip("_")


def endpoint_measurement_classes(canonical_endpoint: str) -> tuple[str, ...]:
    endpoint = canonical_endpoint.casefold()
    if endpoint in _PERMEABILITY_ENDPOINTS:
        return ("permeability",)
    if endpoint in _PERCENT_ENDPOINTS:
        return ("percent",)
    return ()


def endpoint_specific_standardization_of_unit(
    canonical_endpoint: str, pair: MeasurementPair
) -> MeasurementPair:
    classes = endpoint_measurement_classes(canonical_endpoint)
    if not classes or pair.canonical_measurement is None or pair.canonical_unit is None:
        return pair
    parsed = parse_point_measurement(pair.canonical_measurement)
    if parsed.value is None or parsed.kind in {"point_with_interval", "mean_with_context"}:
        return pair
    if classes == ("percent",):
        if pair.canonical_unit == "%":
            return pair
        if pair.canonical_unit == "fraction":
            value = parsed.value * 100.0
            variation = (
                parsed.variation * 100.0 if parsed.variation is not None else None
            )
            return MeasurementPair(
                render_point_measurement(pair.canonical_measurement, value, variation),
                "%",
                "endpoint_standardized",
                pair.unit_notation_status,
                pair.unit_notation_factor,
            )
        return pair
    for measurement_class in classes:
        standardized = standardize_measurement_pair(
            pair, measurement_class=measurement_class
        )
        if standardized.status == "endpoint_standardized":
            return standardized
    return pair


def source_measurement_resolver(
    record: Mapping[str, Any],
    canonical_endpoint: str,
    pair: MeasurementPair,
) -> MeasurementPair:
    """Resolve units with explicit > embedded > exact-default precedence."""
    source_id = str(record.get("source_id") or "")
    del source_id
    if record.get("unit_text") not in (None, ""):
        raw_measurement = clean_text(record.get("measurement_text"))
        if (
            canonical_endpoint in _LOG_PERMEABILITY_DEFAULTS
            and _is_permeability_basis_unit(record.get("unit_text"))
            and raw_measurement is not None
        ):
            # A scale annotation such as ``10^-6 cm/s`` describes the basis of
            # an already logarithmic value; it must never scale that value.
            return MeasurementPair(
                raw_measurement,
                _LOG_PERMEABILITY_DEFAULTS[canonical_endpoint],
                "explicit_log_basis_unit",
                "none",
                None,
            )
        return _validate_endpoint_compatibility(canonical_endpoint, pair)

    embedded = _embedded_measurement_pair(record, canonical_endpoint)
    if embedded is not None:
        return _validate_endpoint_compatibility(canonical_endpoint, embedded)

    endpoint_embedded = _endpoint_embedded_pair(record, canonical_endpoint)
    if endpoint_embedded is not None:
        return _validate_endpoint_compatibility(canonical_endpoint, endpoint_embedded)

    unit = _reviewed_endpoint_default(canonical_endpoint)
    if unit is None or record.get("measurement_text") in (None, ""):
        return pair
    resolved = normalize_measurement_and_unit(record.get("measurement_text"), unit)
    if resolved.canonical_unit is None:
        return pair
    resolved = MeasurementPair(
        resolved.canonical_measurement,
        resolved.canonical_unit,
        "reviewed_endpoint_default_unit",
        resolved.unit_notation_status,
        resolved.unit_notation_factor,
    )
    return _validate_endpoint_compatibility(canonical_endpoint, resolved)


def _embedded_measurement_pair(
    record: Mapping[str, Any], canonical_endpoint: str
) -> MeasurementPair | None:
    text = str(record.get("measurement_text") or "")
    match = _EMBEDDED_MEASUREMENT_UNIT.fullmatch(text)
    if not match:
        return None
    unit = match.group("unit").strip(" ,;:()[]")
    if canonical_endpoint in _LOG_PERMEABILITY_DEFAULTS and _is_permeability_basis_unit(unit):
        return MeasurementPair(
            match.group("measurement"),
            _LOG_PERMEABILITY_DEFAULTS[canonical_endpoint],
            "embedded_log_basis_unit",
            "none",
            None,
        )
    result = canonicalize_unit(unit)
    if not result.cleaned or result.unknown_tokens:
        return None
    resolved = normalize_measurement_and_unit(match.group("measurement"), unit)
    if resolved.canonical_unit is None:
        return None
    return MeasurementPair(
        resolved.canonical_measurement,
        resolved.canonical_unit,
        "embedded_measurement_unit",
        resolved.unit_notation_status,
        resolved.unit_notation_factor,
    )


def _endpoint_embedded_pair(
    record: Mapping[str, Any], canonical_endpoint: str
) -> MeasurementPair | None:
    endpoint_text = str(record.get("endpoint_name") or "")
    match = _PERMEABILITY_UNIT_IN_ENDPOINT.search(endpoint_text)
    if not match or record.get("measurement_text") in (None, ""):
        return None
    unit = match.group("unit")
    if canonical_endpoint in _LOG_PERMEABILITY_DEFAULTS:
        unit = _LOG_PERMEABILITY_DEFAULTS[canonical_endpoint]
    resolved = normalize_measurement_and_unit(record.get("measurement_text"), unit)
    if resolved.canonical_unit is None:
        return None
    return MeasurementPair(
        resolved.canonical_measurement,
        resolved.canonical_unit,
        "embedded_endpoint_unit",
        resolved.unit_notation_status,
        resolved.unit_notation_factor,
    )


def _is_permeability_basis_unit(value: Any) -> bool:
    """Accept physical or logarithmic cm/s spellings as a log endpoint basis."""
    text = str(value or "").strip().strip("[]")
    log_match = re.fullmatch(r"(?:-\s*)?log(?:10)?\s*\((.*)\)", text, re.IGNORECASE)
    if log_match:
        text = log_match.group(1).strip()
    result = canonicalize_unit(text)
    return (
        bool(result.cleaned)
        and not result.unknown_tokens
        and result.canonical == "cm/s"
        and units_compatible(text, "permeability") is True
    )


def _reviewed_endpoint_default(canonical_endpoint: str) -> str | None:
    endpoint = canonical_endpoint.casefold()
    # Physical permeability values without an explicit source or embedded unit
    # remain unresolved.  Real BBB tables mix base cm/s values with 10^-6-scale
    # table values, so an endpoint name alone cannot safely choose the scale.
    if endpoint in _PERMEABILITY_ENDPOINTS:
        return None
    if endpoint in _LOG_PERMEABILITY_DEFAULTS:
        return _LOG_PERMEABILITY_DEFAULTS[endpoint]
    if endpoint in _PERCENT_ENDPOINTS:
        return "%"
    if endpoint in _RATIO_ENDPOINTS:
        return "ratio"
    if endpoint in _FOLD_ENDPOINTS:
        return "fold"
    if endpoint in _FRACTION_ENDPOINTS:
        return "fraction"
    if endpoint in _DIMENSIONLESS_ENDPOINTS:
        return "dimensionless"
    return None


def _validate_endpoint_compatibility(
    canonical_endpoint: str, pair: MeasurementPair
) -> MeasurementPair:
    if not pair.canonical_unit:
        return pair
    classes = endpoint_measurement_classes(canonical_endpoint)
    for measurement_class in classes:
        compatibility_kind = {
            "percent": "fraction_percent",
            "permeability": "permeability",
        }[measurement_class]
        compatible = units_compatible(pair.canonical_unit, compatibility_kind)
        if compatible is False:
            return MeasurementPair(
                pair.canonical_measurement,
                pair.canonical_unit,
                "incompatible_endpoint_unit",
                pair.unit_notation_status,
                pair.unit_notation_factor,
            )
    standardized = endpoint_specific_standardization_of_unit(canonical_endpoint, pair)
    if standardized.status == "endpoint_standardized":
        return MeasurementPair(
            standardized.canonical_measurement,
            standardized.canonical_unit,
            f"{pair.status}_endpoint_standardized",
            standardized.unit_notation_status,
            standardized.unit_notation_factor,
        )
    return pair


def unit_provenance_fields(
    record: Mapping[str, Any], *, encoded: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    encoded = encoded or {}
    if encoded:
        source = "categorical_encoder"
        rule = str(encoded.get("categorical_encoder_id") or "")
        measurement_source = "categorical_encoder"
    else:
        status = str(record.get("measurement_unit_status") or "")
        if record.get("unit_text") not in (None, ""):
            source = "explicit_source_unit"
        elif status.startswith(("embedded_measurement_unit", "embedded_log_basis_unit")):
            source = "embedded_measurement_unit"
        elif status.startswith("embedded_endpoint_unit"):
            source = "embedded_endpoint_unit"
        elif status.startswith("reviewed_endpoint_default_unit"):
            source = "reviewed_endpoint_default"
        else:
            source = "unresolved"
        rule = status or "missing_measurement"
        measurement_source = "source_scalar" if record.get("canonical_measurement") is not None else "missing"
    return {
        "canonical_measurement_source": measurement_source,
        "canonical_unit_resolution_source": source,
        "canonical_unit_rule_id": rule,
        "canonical_unit_policy_version": SOURCE_MEASUREMENT_RESOLVER_VERSION,
    }


def policy_manifest() -> dict[str, Any]:
    return {
        "endpoint_policy_version": ENDPOINT_POLICY_VERSION,
        "source_measurement_resolver_version": SOURCE_MEASUREMENT_RESOLVER_VERSION,
        "no_unit_guessing": True,
        "resolution_precedence": [
            "explicit_source_unit",
            "embedded_measurement_unit",
            "embedded_endpoint_unit",
            "reviewed_exact_endpoint_default",
            "categorical_encoder",
            "unresolved",
        ],
        "reviewed_unitless_metrics": {
            "direct_bbb": ["logBB", "explicit ratio/fold/fraction/percent metric names"],
            "efflux_transport": [
                "efflux or brain/plasma ratio",
                "fold change",
                "probability",
                "percent inhibition",
            ],
        },
        "explicitly_not_inferred": ["IC50", "Km", "permeability", "other unclear metrics"],
    }


__all__ = [
    "ENDPOINT_POLICY_VERSION",
    "SOURCE_MEASUREMENT_RESOLVER_VERSION",
    "endpoint_measurement_classes",
    "endpoint_specific_standardization_of_unit",
    "policy_manifest",
    "source_measurement_resolver",
    "unit_provenance_fields",
]
