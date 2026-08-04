"""Task-independent distance geometry for endpoint assay-transfer policies.

This module is pure math.  It carries no task semantics: it only knows how to
read the ``transform`` / ``input_domain`` / ``thresholds`` / ``normalization``
contract that an endpoint-policy registry publishes for every policy it emits.

The normalization contract is frozen as
``endpoint_policy_distance_normalization.v1``:

* labels and deadbands are always decided on the **full-precision raw
  distance**, never on the normalized score;
* the normalized score is linear in the resolved profile's raw comparison
  space and clipped to ``[0, 100]``;
* ``100`` is anchored at the profile's **permissive** ``not_transfer_min``,
  so the score is invariant across the strict/primary/permissive variants.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


DISTANCE_NORMALIZATION_VERSION = "endpoint_policy_distance_normalization.v1"
DISTANCE_NORMALIZATION_METHOD = (
    "linear_to_permissive_not_transfer_boundary_clipped_0_100"
)
DISTANCE_NORMALIZATION_OUTPUT_RANGE = {"minimum": 0.0, "maximum": 100.0}
DISTANCE_NORMALIZATION_DISPLAY_PRECISION = 2


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


def normalized_thresholds(
    thresholds: Mapping[str, Mapping[str, float]],
    raw_distance_anchor: float,
) -> dict[str, dict[str, float]]:
    """Project every raw threshold boundary onto the normalized 0--100 scale."""
    return {
        variant: {
            boundary: 100.0 * float(raw_value) / raw_distance_anchor
            for boundary, raw_value in values.items()
        }
        for variant, values in thresholds.items()
    }


__all__ = [
    "DISTANCE_NORMALIZATION_DISPLAY_PRECISION",
    "DISTANCE_NORMALIZATION_METHOD",
    "DISTANCE_NORMALIZATION_OUTPUT_RANGE",
    "DISTANCE_NORMALIZATION_VERSION",
    "format_normalized_policy_distance",
    "label_distance",
    "normalize_policy_distance",
    "normalized_thresholds",
    "policy_distance",
    "policy_distance_result",
]
