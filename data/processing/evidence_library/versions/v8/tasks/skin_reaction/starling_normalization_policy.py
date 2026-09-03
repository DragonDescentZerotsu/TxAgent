"""Endpoint-selected unit standardization for normalized Skin_Reaction evidence.

Endpoint spelling/canonicalization is handled elsewhere.  This module may convert
only the measurement/unit pair, and its dispatch key is only ``canonical_endpoint``.

Two of the four sources carry no scalar at all (``direct_skin_reaction`` reports
categorical reaction outcomes, ``phototoxicity_irritation_local_damage`` reports
prose), so the table below is built entirely from the ``skin_exposure``
``evidence_type`` controlled vocabulary and the recurring quantitative concepts in
the free-text ``sensitization_aop`` ``endpoint_or_target`` column.
"""

from __future__ import annotations

from data.processing.evidence_library.shared.v1.normalization.contracts import MeasurementPair
from data.processing.evidence_library.shared.v1.normalization.measurements import (
    parse_point_measurement,
    render_point_measurement,
    standardize_measurement_pair,
)
from data.processing.evidence_library.versions.v8.tasks.skin_reaction.starling_spacing_and_spelling import (
    family_assignment,
    validate_endpoint_inventory,
)


ENDPOINT_POLICY_VERSION = "skin_reaction_endpoint_unit_policy.v1"


# skin_exposure/evidence_type values whose measurements are steady-state permeation
# coefficients (cm/h, cm/s, cm/hr, log(cm/s)).
_PERMEABILITY_ENDPOINTS = {
    "permeability_coefficient",
}
# Amount per area per time.
_FLUX_ENDPOINTS = {
    "flux",
    "maximum_flux",
    "permeation_flux",
    "transdermal_flux",
}
# Amount per area: permeated, retained, or applied.
_AREIC_DOSE_ENDPOINTS = {
    "cumulative_permeated_amount",
    "dermally_applied_dose",
    "skin_retention",
    "tape_strip_result",
}
_DURATION_ENDPOINTS = {
    "exposure_time",
    "lag_time",
}
# Fractions of the applied dose, and control-relative penetration ratios that the
# source reports either as a percentage or as a bare fraction.
_PERCENT_ENDPOINTS = {
    "dermal_absorption",
    "relative_penetration",
    "skin_absorption",
}

# Qualifier vocabulary for every unit parsed by this task.
_TASK_VOCAB = "skin_reaction"



def endpoint_measurement_classes(canonical_endpoint: str) -> tuple[str, ...]:
    """Return reviewed compatible target representations for one endpoint."""
    endpoint = canonical_endpoint.casefold()
    if endpoint in _PERMEABILITY_ENDPOINTS or "permeability_coefficient" in endpoint:
        return ("permeability",)
    if endpoint in _FLUX_ENDPOINTS:
        return ("flux",)
    if endpoint in _AREIC_DOSE_ENDPOINTS:
        return ("areic_dose",)
    if endpoint in _DURATION_ENDPOINTS or endpoint.endswith("_lag_time"):
        return ("duration",)
    if endpoint in _PERCENT_ENDPOINTS:
        return ("percent",)
    # sensitization_aop is free prose; only the two quantitative concepts that
    # recur with a percentage unit are standardized, and only fraction -> %.
    if "ec3" in endpoint and not endpoint.startswith("p"):
        return ("percent",)
    if "depletion" in endpoint:
        return ("percent",)
    if "concentration" in endpoint:
        return ("molar_concentration", "mass_solubility")
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

    for measurement_class in classes:
        standardized = standardize_measurement_pair(
            pair, measurement_class=measurement_class, task=_TASK_VOCAB
        )
        if standardized.status == "endpoint_standardized":
            return standardized
    return pair


__all__ = [
    "ENDPOINT_POLICY_VERSION",
    "endpoint_measurement_classes",
    "endpoint_specific_standardization_of_unit",
    "family_assignment",
    "validate_endpoint_inventory",
]
