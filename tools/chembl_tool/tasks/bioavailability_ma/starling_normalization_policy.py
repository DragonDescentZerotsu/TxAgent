"""Endpoint-selected unit standardization for normalized Bioavailability evidence.

Endpoint spelling/canonicalization is handled elsewhere.  This module may convert
only the measurement/unit pair, and its dispatch key is only ``canonical_endpoint``.
"""

from __future__ import annotations

import re

from tools.chembl_tool.common.starling.normalization.contracts import MeasurementPair
from tools.chembl_tool.common.starling.normalization.measurements import (
    parse_point_measurement,
    render_point_measurement,
    standardize_measurement_pair,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_spacing_and_spelling import (
    family_assignment,
    validate_endpoint_inventory,
)


ENDPOINT_POLICY_VERSION = "bioavailability_endpoint_unit_policy.v3"


_PERCENT_ENDPOINTS = {
    "absolute_bioavailability",
    "absorption",
    "bioavailability",
    "corrected_bioavailability",
    "dissolution",
    "dissolution_efficiency",
    "fraction_absorbed",
    "fraction_dissolved",
    "gastric_absorption",
    "human_intestinal_absorption",
    "intestinal_absorption",
    "oral_bioavailability",
    "relative_bioavailability",
}
_DURATION_ENDPOINTS = {
    "metabolic_half_life",
    "tmax",
}
_RATE_ENDPOINTS = {
    "disappearance_rate",
    "elimination_rate",
}
_CLEARANCE_ENDPOINTS = {
    "biliary_clearance",
    "hepatic_clearance",
    "intrinsic_clearance",
    "metabolic_clearance",
    "oral_clearance",
}


def endpoint_measurement_classes(canonical_endpoint: str) -> tuple[str, ...]:
    """Return reviewed compatible target representations for one endpoint."""
    endpoint = canonical_endpoint.casefold()
    if endpoint.startswith("auc") and "/" not in endpoint:
        return ("auc_mass", "auc_molar")
    if endpoint in _PERCENT_ENDPOINTS:
        return ("percent",)
    if endpoint in _DURATION_ENDPOINTS or endpoint.endswith("_half_life"):
        return ("duration",)
    if endpoint in _RATE_ENDPOINTS:
        return ("rate_constant",)
    if endpoint == "intrinsic_clearance":
        return ("intrinsic_clearance",)
    if endpoint in _CLEARANCE_ENDPOINTS or endpoint.endswith("_clearance"):
        return ("weight_normalized_clearance", "clearance")
    if "permeability" in endpoint:
        return ("permeability",)
    if endpoint in {"solubility", "intrinsic_dissolution_rate"}:
        return ("molar_concentration", "mass_solubility")
    if (
        endpoint.startswith("cmax")
        or endpoint.startswith("cmin")
        or endpoint.startswith("cavg")
        or endpoint.startswith("c_tau")
        or endpoint
        in {
            "average_concentration",
            "c0",
            "c12",
            "c24",
            "c26",
            "c28",
            "c30",
            "c32",
            "cav",
            "clast",
            "css",
            "ctau",
            "ctrough",
        }
    ):
        return ("concentration", "molar_concentration")
    return ()


def endpoint_specific_standardization_of_unit(
    canonical_endpoint: str,
    pair: MeasurementPair,
) -> MeasurementPair:
    """Apply a reviewed conversion without changing endpoint identity."""
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

    if canonical_endpoint.casefold().startswith("auc") and _auc_amount_is_denominator(
        pair.canonical_unit
    ):
        return MeasurementPair(
            pair.canonical_measurement,
            pair.canonical_unit,
            "incompatible_endpoint_unit",
            pair.unit_notation_status,
            pair.unit_notation_factor,
        )

    for measurement_class in classes:
        standardized = standardize_measurement_pair(
            pair, measurement_class=measurement_class
        )
        if standardized.status == "endpoint_standardized":
            return standardized
    return pair


def _auc_amount_is_denominator(unit: str) -> bool:
    if "/" not in unit:
        return False
    numerator, denominator = unit.split("/", 1)
    amount_or_mass = (
        r"(?:^|·)(?:kg|g|mg|µg|ng|pg|mol|mmol|µmol|nmol|pmol)"
        r"(?:\^\d+)?(?:·|$)"
    )
    return not re.search(amount_or_mass, numerator) and bool(
        re.search(amount_or_mass, denominator)
    )


__all__ = [
    "ENDPOINT_POLICY_VERSION",
    "endpoint_measurement_classes",
    "endpoint_specific_standardization_of_unit",
    "family_assignment",
    "validate_endpoint_inventory",
]
