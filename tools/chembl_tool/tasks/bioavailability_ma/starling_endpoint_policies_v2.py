"""Endpoint-specific assay-transfer policy v2 for normalized Starling records.

This module is intentionally independent of ``starling_endpoint_policies``.
The latter is the frozen v1 compatibility policy; v2 classifies measurement
semantics explicitly and has no generic positive-scalar fallback.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from tools.chembl_tool.common.units import canonicalize_unit


ENDPOINT_POLICY_REGISTRY_VERSION = "bioavailability_endpoint_policies.v2"
DISTANCE_NORMALIZATION_VERSION = "endpoint_policy_distance_normalization.v1"
DISTANCE_NORMALIZATION_METHOD = (
    "linear_to_permissive_not_transfer_boundary_clipped_0_100"
)
DISTANCE_NORMALIZATION_OUTPUT_RANGE = {"minimum": 0.0, "maximum": 100.0}
DISTANCE_NORMALIZATION_DISPLAY_PRECISION = 2

_PERCENT_ENDPOINTS = {
    "absolute_bioavailability",
    "absorption",
    "bioavailability",
    "corrected_bioavailability",
    "fraction_absorbed",
    "fraction_dissolved",
    "gastric_absorption",
    "human_intestinal_absorption",
    "intestinal_absorption",
    "oral_bioavailability",
}
_TIMED_PERCENT_ENDPOINTS = {
    "dissolution",
    "dissolution_efficiency",
    "gi_stability",
    "hepatocyte_stability",
    "microsomal_stability",
    "parent_drug_metabolic_stability",
    "s9_stability",
    "substrate_depletion",
}
_HALF_LIFE_ENDPOINTS = {
    "metabolic_half_life",
    "half_life",
}
_HEPATIC_CLEARANCE_ENDPOINTS = {
    "biliary_clearance",
    "hepatic_clearance",
    "oral_clearance",
}
_INTRINSIC_CLEARANCE_ENDPOINTS = {
    "intrinsic_clearance",
    "metabolic_clearance",
}
_RATE_ENDPOINTS = {
    "disappearance_rate",
    "elimination_rate",
}
_CONCENTRATION_ENDPOINTS = {
    "average_concentration",
    "c0",
    "c12",
    "c24",
    "c26",
    "c28",
    "c30",
    "c32",
    "c_avg",
    "cav",
    "cavg",
    "clast",
    "cmax",
    "cmin",
    "css",
    "ctau",
    "ctrough",
}
_RATIO_UNITS = {"ratio", "fold", "dimensionless", "dimensionless_ratio"}
_FRACTION_UNITS = {"fraction"}
_PERCENT_UNITS = {"%"}

_TIME_CAPTURE = re.compile(
    r"\b(?:at|after|within|over|for|by|time(?:point| point)?(?:\s*(?:of|=|:))?)"
    r"\s*(?:approximately\s*)?(?:[<>~≈≤≥]?\s*)?"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>s|secs?|seconds?|mins?|minutes?|h|hrs?|hours?|d|days?)\b",
    re.IGNORECASE,
)
_DIRECTION_CONNECTOR = (
    r"(?:→|->|[-‐‑–—]|(?:[-‐‑–—]\s*)?to(?:\s*[-‐‑–—])?)"
)
_DIRECTION_AB = re.compile(
    rf"(?<![A-Za-z])(?:apical|ap|a)\s*{_DIRECTION_CONNECTOR}\s*"
    r"(?:basolateral|bl|b)(?![A-Za-z])",
    re.IGNORECASE,
)
_DIRECTION_BA = re.compile(
    rf"(?<![A-Za-z])(?:basolateral|bl|b)\s*{_DIRECTION_CONNECTOR}\s*"
    r"(?:apical|ap|a)(?![A-Za-z])",
    re.IGNORECASE,
)


def _thresholds(
    low: tuple[float, float],
    primary: tuple[float, float],
    high: tuple[float, float],
) -> dict[str, dict[str, float]]:
    return {
        "strict": {"transfer_max": low[0], "not_transfer_min": low[1]},
        "primary": {
            "transfer_max": primary[0],
            "not_transfer_min": primary[1],
        },
        "permissive": {"transfer_max": high[0], "not_transfer_min": high[1]},
    }


def _normalized_thresholds(
    thresholds: Mapping[str, Mapping[str, float]],
    raw_distance_anchor: float,
) -> dict[str, dict[str, float]]:
    return {
        variant: {
            boundary: 100.0 * float(raw_value) / raw_distance_anchor
            for boundary, raw_value in values.items()
        }
        for variant, values in thresholds.items()
    }


def _policy_family(
    *,
    transform: str,
    distance: str,
    thresholds: dict[str, dict[str, float]],
    raw_distance_anchor: float,
    input_domain: str,
    rationale: str,
) -> dict[str, Any]:
    return {
        "transform": transform,
        "distance": distance,
        "input_domain": input_domain,
        "thresholds": thresholds,
        "normalization": {
            "version": DISTANCE_NORMALIZATION_VERSION,
            "method": DISTANCE_NORMALIZATION_METHOD,
            "raw_distance_anchor": raw_distance_anchor,
            "anchor_semantics": "permissive_not_transfer_min",
            "output_range": dict(DISTANCE_NORMALIZATION_OUTPUT_RANGE),
            "display_precision_decimals": DISTANCE_NORMALIZATION_DISPLAY_PRECISION,
            "normalized_thresholds": _normalized_thresholds(
                thresholds, raw_distance_anchor
            ),
        },
        "rationale": rationale,
    }


POLICY_FAMILIES: dict[str, dict[str, Any]] = {
    "bounded_percentage": _policy_family(
        transform="identity",
        distance="absolute_percentage_points",
        thresholds=_thresholds((5.0, 20.0), (10.0, 30.0), (15.0, 40.0)),
        raw_distance_anchor=40.0,
        input_domain="nonnegative",
        rationale=(
            "Bounded percent outcomes are compared in absolute percentage points; "
            "the frozen cutoffs represent moderate analog agreement and separation."
        ),
    ),
    "bounded_fraction": _policy_family(
        transform="identity",
        distance="absolute_fraction",
        thresholds=_thresholds((0.05, 0.20), (0.10, 0.30), (0.15, 0.40)),
        raw_distance_anchor=0.40,
        input_domain="nonnegative",
        rationale=(
            "Bounded fractions are compared on their native zero-to-one scale."
        ),
    ),
    "ratio_or_tmax": _policy_family(
        transform="log10",
        distance="absolute_log10_ratio",
        thresholds=_thresholds(
            (math.log10(1.25), math.log10(2.0)),
            (math.log10(1.5), math.log10(3.0)),
            (math.log10(2.0), math.log10(5.0)),
        ),
        raw_distance_anchor=math.log10(5.0),
        input_domain="positive",
        rationale=(
            "Ratios and Tmax use multiplicative geometry; 1.5-fold/3-fold is the "
            "frozen primary analog-transfer boundary."
        ),
    ),
    "standard_log_metric": _policy_family(
        transform="log10",
        distance="absolute_log10_ratio",
        thresholds=_thresholds(
            (math.log10(1.5), math.log10(3.0)),
            (math.log10(2.0), math.log10(5.0)),
            (math.log10(3.0), math.log10(10.0)),
        ),
        raw_distance_anchor=math.log10(10.0),
        input_domain="positive",
        rationale=(
            "Positive exposure, half-life, permeability, hepatic-clearance and "
            "resolved kinetic values use regulatory-style multiplicative geometry; "
            "2-fold/5-fold is a practical analog boundary rather than a matched-"
            "formulation bioequivalence criterion."
        ),
    ),
    "wide_log_metric": _policy_family(
        transform="log10",
        distance="absolute_log10_ratio",
        thresholds=_thresholds(
            (math.log10(2.0), math.log10(5.0)),
            (math.log10(3.0), math.log10(10.0)),
            (math.log10(5.0), math.log10(20.0)),
        ),
        raw_distance_anchor=math.log10(20.0),
        input_domain="positive",
        rationale=(
            "Solubility and intrinsic/metabolic clearance use a wider frozen "
            "multiplicative tolerance because their assay systems are heterogeneous."
        ),
    ),
    "pretransformed_standard_log10_metric": _policy_family(
        transform="identity",
        distance="absolute_log10_coordinate_difference",
        thresholds=_thresholds(
            (math.log10(1.5), math.log10(3.0)),
            (math.log10(2.0), math.log10(5.0)),
            (math.log10(3.0), math.log10(10.0)),
        ),
        raw_distance_anchor=math.log10(10.0),
        input_domain="finite",
        rationale=(
            "Reviewed Bioavailability permeability readouts reported as bare log or "
            "log10 coordinates are already transformed. Their coordinate difference "
            "is the log10 fold distance, so values are not exponentiated again."
        ),
    ),
}


THRESHOLD_PROFILE_VERSION = "bioavailability_endpoint_threshold_profiles.v1"


def _threshold_profile(
    policy_family: str,
    *,
    rationale: str,
) -> dict[str, Any]:
    profile = deepcopy(POLICY_FAMILIES[policy_family])
    family_rationale = str(profile["rationale"])
    profile.update(
        {
            "profile_version": THRESHOLD_PROFILE_VERSION,
            "policy_family": policy_family,
            "family_rationale": family_rationale,
            "profile_rationale": rationale,
            "rationale": rationale,
        }
    )
    return profile


THRESHOLD_PROFILES: dict[str, dict[str, Any]] = {
    "bounded_percentage_outcome": _threshold_profile(
        "bounded_percentage",
        rationale=(
            "Absolute bioavailability, absorption, dissolution and stability "
            "percentages retain the reviewed percentage-point thresholds."
        ),
    ),
    "bounded_fraction_outcome": _threshold_profile(
        "bounded_fraction",
        rationale=(
            "Absolute bioavailability, absorption, dissolution and stability "
            "fractions retain the reviewed zero-to-one thresholds."
        ),
    ),
    "tmax_ratio": _threshold_profile(
        "ratio_or_tmax",
        rationale=(
            "Tmax retains the reviewed multiplicative timing thresholds."
        ),
    ),
    "dimensionless_ratio": _threshold_profile(
        "ratio_or_tmax",
        rationale=(
            "Dimensionless and directional efflux ratios retain the reviewed "
            "ratio thresholds."
        ),
    ),
    "auc_exposure": _threshold_profile(
        "standard_log_metric",
        rationale="AUC exposure uses the reviewed standard log-fold thresholds.",
    ),
    "cmax_exposure": _threshold_profile(
        "standard_log_metric",
        rationale="Cmax exposure uses the reviewed standard log-fold thresholds.",
    ),
    "other_concentration_exposure": _threshold_profile(
        "standard_log_metric",
        rationale=(
            "Other concentration exposure endpoints use the reviewed standard "
            "log-fold thresholds."
        ),
    ),
    "relative_or_dose_normalized_exposure": _threshold_profile(
        "standard_log_metric",
        rationale=(
            "Relative and dose-normalized exposure uses the reviewed standard "
            "log-fold thresholds."
        ),
    ),
    "half_life": _threshold_profile(
        "standard_log_metric",
        rationale="Half-life uses the reviewed standard log-fold thresholds.",
    ),
    "rate_constant": _threshold_profile(
        "standard_log_metric",
        rationale="Rate constants use the reviewed standard log-fold thresholds.",
    ),
    "permeability": _threshold_profile(
        "wide_log_metric",
        rationale=(
            "Papp and Peff use the reviewed wide log-fold thresholds because "
            "cross-study permeability systems are heterogeneous; direction and "
            "assay-system comparability remain enforced by subtype and pair bucket."
        ),
    ),
    "pretransformed_permeability": _threshold_profile(
        "pretransformed_standard_log10_metric",
        rationale=(
            "Papp and Peff values already reported as log10 coordinates use "
            "identity coordinate distance with the reviewed wide log-fold "
            "thresholds and anchor."
        ),
    ),
    "hepatic_clearance": _threshold_profile(
        "standard_log_metric",
        rationale=(
            "Hepatic clearance uses the reviewed standard log-fold thresholds."
        ),
    ),
    "intrinsic_or_metabolic_clearance": _threshold_profile(
        "wide_log_metric",
        rationale=(
            "Intrinsic and metabolic clearance use the reviewed wide log-fold "
            "thresholds after assay-system matching in the pair bucket."
        ),
    ),
    "equilibrium_solubility": _threshold_profile(
        "wide_log_metric",
        rationale=(
            "Equilibrium solubility uses the reviewed wide log-fold thresholds."
        ),
    ),
    "kinetic_solubility": _threshold_profile(
        "wide_log_metric",
        rationale=(
            "Kinetic solubility uses the reviewed wide log-fold thresholds."
        ),
    ),
    "michaelis_constant": _threshold_profile(
        "standard_log_metric",
        rationale=(
            "Resolved Michaelis constants use the reviewed standard log-fold "
            "thresholds."
        ),
    ),
    "maximum_velocity": _threshold_profile(
        "standard_log_metric",
        rationale=(
            "Resolved maximum velocities use the reviewed standard log-fold "
            "thresholds."
        ),
    ),
}

# Pretransformed permeability has the same identity coordinate geometry as the
# existing reviewed log10 policy, but uses the wide 2x/5x, 3x/10x, 5x/20x
# thresholds and log10(20) permissive far anchor.
THRESHOLD_PROFILES["pretransformed_permeability"].update(
    {
        "thresholds": deepcopy(POLICY_FAMILIES["wide_log_metric"]["thresholds"]),
        "normalization": deepcopy(
            POLICY_FAMILIES["wide_log_metric"]["normalization"]
        ),
    }
)

ENDPOINT_SUBTYPE_PROFILE_OVERRIDES = {
    ("cmax", "plasma_concentration_exposure"): "cmax_exposure",
    ("tmax", "tmax"): "tmax_ratio",
}

SUBTYPE_PROFILE_OVERRIDES = {
    "absolute_bioavailability_percentage": "bounded_percentage_outcome",
    "absolute_bioavailability_fraction": "bounded_fraction_outcome",
    "absorption_percentage": "bounded_percentage_outcome",
    "absorption_fraction": "bounded_fraction_outcome",
    "tmax": "tmax_ratio",
    "dimensionless_ratio": "dimensionless_ratio",
    "directional_efflux_ratio_a_over_b": "dimensionless_ratio",
    "directional_efflux_ratio_b_over_a": "dimensionless_ratio",
    "auc_exposure": "auc_exposure",
    "plasma_concentration_exposure": "other_concentration_exposure",
    "relative_exposure_magnitude": "relative_or_dose_normalized_exposure",
    "dose_normalized_or_relative_exposure": (
        "relative_or_dose_normalized_exposure"
    ),
    "explicit_half_life": "half_life",
    "explicit_rate_constant": "rate_constant",
    "hepatic_clearance": "hepatic_clearance",
    "intrinsic_or_metabolic_clearance": "intrinsic_or_metabolic_clearance",
    "equilibrium_solubility": "equilibrium_solubility",
    "kinetic_solubility": "kinetic_solubility",
    "michaelis_constant_km": "michaelis_constant",
    "maximum_velocity_vmax": "maximum_velocity",
}

FAMILY_DEFAULT_PROFILES = {
    "bounded_percentage": "bounded_percentage_outcome",
    "bounded_fraction": "bounded_fraction_outcome",
    "ratio_or_tmax": "dimensionless_ratio",
    "standard_log_metric": "other_concentration_exposure",
    "wide_log_metric": "intrinsic_or_metabolic_clearance",
    "pretransformed_standard_log10_metric": "pretransformed_permeability",
}


def resolve_threshold_profile(
    canonical_endpoint: str,
    measurement_subtype: str,
    classified_policy_family: str,
) -> tuple[str, dict[str, Any], str]:
    """Resolve exactly one reviewed profile with explicit deterministic precedence."""
    endpoint = str(canonical_endpoint)
    subtype = str(measurement_subtype)
    exact = ENDPOINT_SUBTYPE_PROFILE_OVERRIDES.get((endpoint, subtype))
    if exact:
        return exact, THRESHOLD_PROFILES[exact], "endpoint_subtype_override"
    if subtype.startswith("matched_time_dissolution_percentage__"):
        key = "bounded_percentage_outcome"
        return key, THRESHOLD_PROFILES[key], "subtype_prefix_override"
    if subtype.startswith("matched_time_percent_remaining__"):
        key = "bounded_percentage_outcome"
        return key, THRESHOLD_PROFILES[key], "subtype_prefix_override"
    if subtype.startswith("matched_time_dissolution_fraction__"):
        key = "bounded_fraction_outcome"
        return key, THRESHOLD_PROFILES[key], "subtype_prefix_override"
    if subtype.startswith("matched_time_fraction_remaining__"):
        key = "bounded_fraction_outcome"
        return key, THRESHOLD_PROFILES[key], "subtype_prefix_override"
    if subtype.startswith(("papp_", "peff_")):
        key = (
            "pretransformed_permeability"
            if subtype.endswith("__log10_coordinate")
            else "permeability"
        )
        return key, THRESHOLD_PROFILES[key], "subtype_prefix_override"
    direct = SUBTYPE_PROFILE_OVERRIDES.get(subtype)
    if direct:
        return direct, THRESHOLD_PROFILES[direct], "subtype_override"
    fallback = FAMILY_DEFAULT_PROFILES.get(str(classified_policy_family))
    if not fallback:
        raise ValueError(
            f"no threshold profile for endpoint={endpoint!r}, "
            f"subtype={subtype!r}, family={classified_policy_family!r}"
        )
    return fallback, THRESHOLD_PROFILES[fallback], "policy_family_default"


def _text(record: Mapping[str, Any]) -> str:
    fields = (
        "endpoint_name",
        "measurement_text",
        "unit_text",
        "canonical_measurement",
        "canonical_unit",
        "canonical_bioavailability_report_type",
        "bioavailability_report_type",
        "canonical_assay_system",
        "assay_system",
        "condition_medium",
        "extra_details",
        "support_text",
        "study_context",
        "comparator",
        "comparator_exposure",
        "formulation_or_solid_form",
    )
    return " | ".join(str(record.get(field) or "") for field in fields)


def _has_any(text: str, *patterns: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _permeability_direction(text: str) -> str:
    has_ab = bool(_DIRECTION_AB.search(text))
    has_ba = bool(_DIRECTION_BA.search(text))
    if has_ab and has_ba:
        return "mixed"
    if has_ab:
        return "apical_to_basolateral"
    if has_ba:
        return "basolateral_to_apical"
    return "unspecified"


def _time_context_key(text: str) -> str | None:
    matches: set[float] = set()
    factors = {
        "s": 1.0,
        "sec": 1.0,
        "secs": 1.0,
        "second": 1.0,
        "seconds": 1.0,
        "min": 60.0,
        "mins": 60.0,
        "minute": 60.0,
        "minutes": 60.0,
        "h": 3600.0,
        "hr": 3600.0,
        "hrs": 3600.0,
        "hour": 3600.0,
        "hours": 3600.0,
        "d": 86400.0,
        "day": 86400.0,
        "days": 86400.0,
    }
    for match in _TIME_CAPTURE.finditer(text):
        unit = match.group("unit").casefold()
        seconds = float(match.group("value")) * factors[unit]
        matches.add(seconds)
    if len(matches) == 1:
        seconds = next(iter(matches))
        rendered = str(int(seconds)) if seconds.is_integer() else f"{seconds:g}"
        return f"time_{rendered}s"
    if len(matches) > 1:
        return "mixed"
    if _has_any(text, r"\btime(?: course| profile| series)\b"):
        return "profile"
    return None


def _ratio_direction(text: str) -> str | None:
    if re.search(r"\bb\s*/\s*a\b", text, re.IGNORECASE):
        return "b_over_a"
    if re.search(r"\ba\s*/\s*b\b", text, re.IGNORECASE):
        return "a_over_b"
    if _has_any(text, r"\befflux ratio\b"):
        return "b_over_a"
    return None


def classify_measurement_subtype(
    record: Mapping[str, Any],
) -> tuple[str | None, str | None, str]:
    """Return ``(subtype, policy_family, status)`` for one normalized record."""
    endpoint = str(record.get("canonical_endpoint") or "").casefold()
    unit = str(record.get("canonical_unit") or "").casefold()
    report_type = str(
        record.get("canonical_bioavailability_report_type") or ""
    ).casefold()
    text = _text(record)

    if not endpoint or not unit:
        return None, None, "unresolved_measurement_subtype"

    unit_result = canonicalize_unit(unit)
    if unit_result.transform:
        if (
            "permeability" in endpoint
            and unit_result.transform in {"log", "log10"}
            and unit_result.canonical in {"log(cm/s)", "log10(cm/s)"}
            and str(record.get("unit_notation_status") or "none") == "none"
        ):
            direction = _permeability_direction(text)
            if direction == "mixed":
                return None, None, "mixed_measurement_semantics"
            base = (
                "peff"
                if "effective" in endpoint
                or _has_any(text, r"\bpeff\b", r"p[- ]?eff")
                else "papp"
            )
            return (
                f"{base}_{direction}__log10_coordinate",
                "pretransformed_standard_log10_metric",
                "assigned",
            )
        return None, None, "unresolved_measurement_subtype"

    has_km = _has_any(text, r"\bkm\b", r"michaelis[- ]menten constant")
    has_vmax = _has_any(text, r"\bvmax\b", r"maximum (?:reaction )?velocity")
    if (has_km and has_vmax) or _has_any(
        text,
        r"\bvmax\s*/\s*km\b",
        r"\bauc\s*/\s*cmax\b",
        r"\bcmax\s*/\s*auc\b",
    ):
        return None, None, "mixed_measurement_semantics"

    if endpoint == "tmax" and unit != "h":
        return None, None, "unresolved_measurement_subtype"
    if endpoint == "tmax":
        return "tmax", "ratio_or_tmax", "assigned"

    if endpoint in _HALF_LIFE_ENDPOINTS or endpoint.endswith("_half_life") or _has_any(
        text, r"\b(?:metabolic|elimination|terminal|apparent)\s+half[- ]life\b"
    ):
        if unit != "h":
            return None, None, "unresolved_measurement_subtype"
        return "explicit_half_life", "standard_log_metric", "assigned"

    if has_km:
        if not _has_any(unit, r"(?:^|[·/])(?:m|mm|µm|nm|pm|mol|mmol|µmol|nmol|mg|µg|ng)(?:$|[·/])"):
            return None, None, "missing_required_comparison_context"
        return "michaelis_constant_km", "standard_log_metric", "assigned"

    if has_vmax:
        if "/" not in unit and "min" not in unit and "h" not in unit:
            return None, None, "missing_required_comparison_context"
        return "maximum_velocity_vmax", "standard_log_metric", "assigned"

    if endpoint in _RATE_ENDPOINTS or _has_any(
        text, r"\brate constant\b", r"\b(?:elimination|disappearance) rate\b"
    ):
        return "explicit_rate_constant", "standard_log_metric", "assigned"

    # The 133 rows outside the reviewed v1 semantic boundary remain rejected.
    # In particular, a bare bioavailability scalar in fold/ratio units does not
    # identify the reference formulation or comparison direction.
    if "bioavailability" in endpoint and unit not in _PERCENT_UNITS | _FRACTION_UNITS:
        return None, None, "unresolved_measurement_subtype"

    is_relative = (
        report_type in {"relative_comparison", "apparent"}
        or endpoint.startswith("relative_")
        or endpoint in {"food_effect", "oral_iv_comparison"}
    )
    if is_relative:
        return "relative_exposure_magnitude", "standard_log_metric", "assigned"

    if endpoint.startswith("auc") or endpoint in {
        "aauc0_t",
        "aucall",
        "auclast",
        "auct",
    }:
        return "auc_exposure", "standard_log_metric", "assigned"
    if endpoint in _CONCENTRATION_ENDPOINTS or endpoint.startswith(
        ("cmax", "cmin", "cavg", "ctrough", "c_tau")
    ):
        return "plasma_concentration_exposure", "standard_log_metric", "assigned"
    if endpoint in {"dose_normalized_exposure", "other_oral_exposure"}:
        return "dose_normalized_or_relative_exposure", "standard_log_metric", "assigned"

    if endpoint in _PERCENT_ENDPOINTS and unit in _PERCENT_UNITS:
        return (
            "absolute_bioavailability_percentage"
            if "bioavailability" in endpoint
            else "absorption_percentage"
        ), "bounded_percentage", "assigned"
    if endpoint in _PERCENT_ENDPOINTS and unit in _FRACTION_UNITS:
        return (
            "absolute_bioavailability_fraction"
            if "bioavailability" in endpoint
            else "absorption_fraction"
        ), "bounded_fraction", "assigned"

    if endpoint in _TIMED_PERCENT_ENDPOINTS and unit in _PERCENT_UNITS:
        time_key = _time_context_key(text)
        if not time_key:
            return None, None, "missing_required_comparison_context"
        if time_key == "mixed":
            return None, None, "mixed_measurement_semantics"
        subtype = (
            "matched_time_dissolution_percentage"
            if "dissolution" in endpoint
            else "matched_time_percent_remaining"
        )
        return f"{subtype}__{time_key}", "bounded_percentage", "assigned"
    if endpoint in _TIMED_PERCENT_ENDPOINTS and unit in _FRACTION_UNITS:
        time_key = _time_context_key(text)
        if not time_key:
            return None, None, "missing_required_comparison_context"
        if time_key == "mixed":
            return None, None, "mixed_measurement_semantics"
        subtype = (
            "matched_time_dissolution_fraction"
            if "dissolution" in endpoint
            else "matched_time_fraction_remaining"
        )
        return f"{subtype}__{time_key}", "bounded_fraction", "assigned"

    if "permeability" in endpoint and unit in {"cm/s", "cm s^-1", "cm·s^-1"}:
        direction = _permeability_direction(text)
        if direction == "mixed":
            return None, None, "mixed_measurement_semantics"
        base = "peff" if "effective" in endpoint or _has_any(text, r"\bpeff\b", r"p[- ]?eff") else "papp"
        return f"{base}_{direction}", "standard_log_metric", "assigned"

    if endpoint in _HEPATIC_CLEARANCE_ENDPOINTS or (
        endpoint.endswith("_clearance") and "intrinsic" not in endpoint
    ):
        return "hepatic_clearance", "standard_log_metric", "assigned"
    if endpoint in _INTRINSIC_CLEARANCE_ENDPOINTS:
        return "intrinsic_or_metabolic_clearance", "wide_log_metric", "assigned"

    if endpoint in {"solubility", "intrinsic_dissolution_rate"}:
        if _has_any(
            text,
            r"\bkinetic solubility\b",
            r"\bsupersaturat",
            r"\bintrinsic dissolution rate\b",
        ):
            return "kinetic_solubility", "wide_log_metric", "assigned"
        if _has_any(
            text,
            r"\bequilibrium\b",
            r"\bthermodynamic solubility\b",
            r"\baqueous solubility\b",
            r"\bsaturation solubility\b",
            r"\bsolubility (?:assay|measurement|test)\b",
        ):
            return "equilibrium_solubility", "wide_log_metric", "assigned"
        return None, None, "missing_required_comparison_context"

    if unit in _RATIO_UNITS and _has_any(
        text,
        r"\befflux ratio\b",
        r"\b(?:b\s*/\s*a|a\s*/\s*b) ratio\b",
        r"\bdimensionless\b",
        r"\bratio\b",
    ):
        direction = _permeability_direction(text)
        if direction == "mixed":
            # A directional ratio legitimately contains both directions when its
            # numerator/denominator definition is explicit.
            if _has_any(text, r"\befflux ratio\b", r"\bb\s*/\s*a\b", r"\ba\s*/\s*b\b"):
                ratio_direction = _ratio_direction(text)
                return (
                    f"directional_efflux_ratio_{ratio_direction}",
                    "ratio_or_tmax",
                    "assigned",
                )
            return None, None, "mixed_measurement_semantics"
        ratio_direction = _ratio_direction(text)
        if ratio_direction:
            return (
                f"directional_efflux_ratio_{ratio_direction}",
                "ratio_or_tmax",
                "assigned",
            )
        return "dimensionless_ratio", "ratio_or_tmax", "assigned"

    return None, None, "unresolved_measurement_subtype"


def endpoint_policy_key(endpoint: str, measurement_subtype: str) -> str:
    return f"bioavailability_ma/{endpoint}/{measurement_subtype}/v2"


def policy_distance(policy: Mapping[str, Any], left: float, right: float) -> float:
    """Compute the policy distance, rejecting invalid log-domain values."""
    left_value = float(left)
    right_value = float(right)
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        raise ValueError("policy values must be finite")
    input_domain = str(policy.get("input_domain") or "nonnegative")
    if input_domain == "positive" and (left_value <= 0 or right_value <= 0):
        raise ValueError("policy values must be positive")
    if input_domain == "nonnegative" and (left_value < 0 or right_value < 0):
        raise ValueError("policy values must be nonnegative")
    if input_domain not in {"finite", "nonnegative", "positive"}:
        raise ValueError(f"unsupported policy input domain: {input_domain!r}")
    if policy.get("transform") == "log10":
        left_value = math.log10(left_value)
        right_value = math.log10(right_value)
    return abs(left_value - right_value)


def normalize_policy_distance(
    policy: Mapping[str, Any], raw_distance: float
) -> float:
    """Map a valid raw policy distance to the invariant clipped 0--100 scale."""
    distance = float(raw_distance)
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("raw policy distance must be finite and nonnegative")
    normalization = policy.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("policy normalization metadata is missing")
    anchor = float(normalization.get("raw_distance_anchor", math.nan))
    if not math.isfinite(anchor) or anchor <= 0:
        raise ValueError("policy normalization raw distance anchor must be finite and positive")
    return 100.0 * min(distance / anchor, 1.0)


def label_distance(
    policy: Mapping[str, Any],
    distance: float,
    *,
    variant: str = "primary",
) -> str | None:
    """Apply inclusive frozen boundaries; the intervening deadband is unlabeled."""
    raw_distance = float(distance)
    if not math.isfinite(raw_distance) or raw_distance < 0:
        raise ValueError("raw policy distance must be finite and nonnegative")
    thresholds = dict(policy.get("thresholds") or {}).get(variant)
    if not isinstance(thresholds, Mapping):
        raise KeyError(f"unknown policy threshold variant: {variant!r}")
    if raw_distance <= float(thresholds["transfer_max"]):
        return "transfer"
    if raw_distance >= float(thresholds["not_transfer_min"]):
        return "not_transfer"
    return None


def policy_distance_result(
    policy: Mapping[str, Any],
    left: float,
    right: float,
    *,
    variant: str = "primary",
) -> dict[str, Any]:
    """Return full-precision raw/normalized distance and the raw-distance label."""
    raw_distance = policy_distance(policy, left, right)
    normalized_distance = normalize_policy_distance(policy, raw_distance)
    normalization = policy.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("policy normalization metadata is missing")
    anchor = float(normalization["raw_distance_anchor"])
    return {
        "raw_distance": raw_distance,
        "normalized_distance_0_100": normalized_distance,
        "normalization_saturated": raw_distance > anchor,
        "threshold_variant": variant,
        "raw_distance_label": label_distance(policy, raw_distance, variant=variant),
        "normalization_version": normalization.get("version"),
        "normalization_provenance": {
            "method": normalization.get("method"),
            "raw_distance_anchor": anchor,
            "anchor_semantics": normalization.get("anchor_semantics"),
            "output_range": dict(normalization.get("output_range") or {}),
        },
    }


def format_normalized_policy_distance(
    normalized_distance: float, policy: Mapping[str, Any]
) -> str:
    """Format a normalized score for reports or LLM-visible text only."""
    value = float(normalized_distance)
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise ValueError("normalized policy distance must be finite and within [0, 100]")
    normalization = policy.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("policy normalization metadata is missing")
    precision = int(normalization.get("display_precision_decimals", -1))
    if precision < 0:
        raise ValueError("normalization display precision must be nonnegative")
    return f"{value:.{precision}f}"


def assign_endpoint_policies_v2(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Assign one v2 subtype/policy row for every normalized v5 record."""
    assignments: list[dict[str, Any]] = []
    policies: dict[str, dict[str, Any]] = {}
    statuses: Counter[str] = Counter()
    subtypes: Counter[str] = Counter()
    seen: set[str] = set()

    for record in records:
        record_id = str(record.get("normalized_record_id") or "")
        if not record_id or record_id in seen:
            raise ValueError(
                "endpoint-policy input normalized_record_id values must be nonempty "
                "and unique"
            )
        seen.add(record_id)
        validity = str(record.get("normalization_validity_status") or "")
        subtype: str | None = None
        policy_key: str | None = None
        if validity != "valid":
            status = f"normalization_invalid:{validity or 'missing_status'}"
        else:
            subtype, family_key, status = classify_measurement_subtype(record)
            if status == "assigned" and subtype and family_key:
                endpoint = str(record.get("canonical_endpoint") or "")
                policy_key = endpoint_policy_key(endpoint, subtype)
                profile_key, profile, resolution = resolve_threshold_profile(
                    endpoint, subtype, family_key
                )
                policies.setdefault(
                    policy_key,
                    {
                        "canonical_endpoint": endpoint,
                        "measurement_subtype": subtype,
                        "classified_policy_family": family_key,
                        "threshold_profile": profile_key,
                        "threshold_profile_version": THRESHOLD_PROFILE_VERSION,
                        "threshold_profile_resolution": resolution,
                        **profile,
                    },
                )
                subtypes[subtype] += 1
        statuses[status] += 1
        assignments.append(
            {
                "normalized_record_id": record_id,
                "measurement_subtype": subtype,
                "endpoint_policy_key": policy_key,
                "policy_assignment_status": status,
            }
        )

    registry = {
        "registry_version": ENDPOINT_POLICY_REGISTRY_VERSION,
        "policy_key_format": (
            "bioavailability_ma/<canonical_endpoint>/<measurement_subtype>/v2"
        ),
        "policy_families": POLICY_FAMILIES,
        "threshold_profile_version": THRESHOLD_PROFILE_VERSION,
        "threshold_profiles": THRESHOLD_PROFILES,
        "threshold_profile_resolution": {
            "precedence": [
                "endpoint_subtype_override",
                "subtype_prefix_override",
                "subtype_override",
                "policy_family_default",
            ],
            "endpoint_subtype_overrides": {
                f"{endpoint}::{subtype}": profile
                for (endpoint, subtype), profile in sorted(
                    ENDPOINT_SUBTYPE_PROFILE_OVERRIDES.items()
                )
            },
            "subtype_overrides": dict(sorted(SUBTYPE_PROFILE_OVERRIDES.items())),
            "policy_family_defaults": dict(sorted(FAMILY_DEFAULT_PROFILES.items())),
        },
        "endpoint_policies": dict(sorted(policies.items())),
        "boundary_semantics": {
            "inclusive": True,
            "deadband": (
                "Distances above transfer_max and below not_transfer_min are unlabeled."
            ),
            "labels_use": "raw_distance",
            "normalization_changes_labels": False,
        },
        "distance_normalization": {
            "version": DISTANCE_NORMALIZATION_VERSION,
            "method": DISTANCE_NORMALIZATION_METHOD,
            "invariant_across_threshold_variants": True,
            "anchor_semantics": "permissive_not_transfer_min",
            "output_range": dict(DISTANCE_NORMALIZATION_OUTPUT_RANGE),
            "display_precision_decimals": DISTANCE_NORMALIZATION_DISPLAY_PRECISION,
            "storage_precision": "full_floating_point",
        },
        "scientific_basis": {
            "exposure_geometry": (
                "Exposure comparisons use log-scale ratios, consistent with "
                "regulatory pharmacokinetic comparison practice."
            ),
            "threshold_intent": (
                "A priori named endpoint/subtype profiles resolved before model "
                "evaluation; thresholds are not fitted bioequivalence limits or "
                "benchmark-selected cutoffs."
            ),
            "references": [
                "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/statistical-approaches-establishing-bioequivalence-0",
                "https://www.fda.gov/media/161199/download",
                "https://www.oecd.org/content/dam/oecd/en/publications/reports/2021/02/guidance-document-on-the-characterisation-validation-and-reporting-of-physiologically-based-kinetic-pbk-models-for-regulatory-purposes_670da2f4/d0de241f-en.pdf",
            ],
        },
        "scope": {
            "pair_enumeration": False,
            "pair_labels": False,
            "threshold_calibration": False,
            "benchmark_outcome_selection": False,
        },
    }
    audit = {
        "registry_version": ENDPOINT_POLICY_REGISTRY_VERSION,
        "distance_normalization": dict(registry["distance_normalization"]),
        "stats": {
            "input_records": len(records),
            "assignment_records": len(assignments),
            "assigned_records": statuses["assigned"],
            "unassigned_records": len(assignments) - statuses["assigned"],
            "endpoint_policies": len(policies),
            "measurement_subtypes": len(subtypes),
        },
        "assignment_status_counts": dict(sorted(statuses.items())),
        "measurement_subtype_counts": dict(sorted(subtypes.items())),
        "validations": {
            "one_assignment_per_input_record": len(assignments) == len(records),
            "assigned_rows_have_subtype_and_policy": all(
                bool(row["measurement_subtype"] and row["endpoint_policy_key"])
                for row in assignments
                if row["policy_assignment_status"] == "assigned"
            ),
            "unassigned_rows_have_null_subtype_and_policy": all(
                not row["measurement_subtype"] and not row["endpoint_policy_key"]
                for row in assignments
                if row["policy_assignment_status"] != "assigned"
            ),
            "all_assigned_policies_are_registered": all(
                not row["endpoint_policy_key"]
                or row["endpoint_policy_key"] in policies
                for row in assignments
            ),
            "all_policy_keys_are_v2": all(
                key.startswith("bioavailability_ma/") and key.endswith("/v2")
                for key in policies
            ),
            "all_assigned_policies_have_resolved_threshold_profiles": all(
                policy.get("threshold_profile") in THRESHOLD_PROFILES
                and policy.get("threshold_profile_version")
                == THRESHOLD_PROFILE_VERSION
                for policy in policies.values()
            ),
        },
    }
    return assignments, registry, audit


__all__ = [
    "DISTANCE_NORMALIZATION_DISPLAY_PRECISION",
    "DISTANCE_NORMALIZATION_METHOD",
    "DISTANCE_NORMALIZATION_OUTPUT_RANGE",
    "DISTANCE_NORMALIZATION_VERSION",
    "ENDPOINT_POLICY_REGISTRY_VERSION",
    "ENDPOINT_SUBTYPE_PROFILE_OVERRIDES",
    "FAMILY_DEFAULT_PROFILES",
    "POLICY_FAMILIES",
    "SUBTYPE_PROFILE_OVERRIDES",
    "THRESHOLD_PROFILES",
    "THRESHOLD_PROFILE_VERSION",
    "assign_endpoint_policies_v2",
    "classify_measurement_subtype",
    "endpoint_policy_key",
    "format_normalized_policy_distance",
    "label_distance",
    "normalize_policy_distance",
    "policy_distance",
    "policy_distance_result",
    "resolve_threshold_profile",
]
