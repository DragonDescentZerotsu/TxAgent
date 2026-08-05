"""Pair-bucket-level assay-transfer eligibility and distance contract.

The task's pair bucket is the only comparison stratum.  Raw source columns may
flag a heterogeneous bucket, but they never create child buckets or
record-level policy assignments.  Eligible buckets use their own observed
sample standard deviation and empirical within-bucket distance curve.

Everything here is task-agnostic.  A task supplies a
:class:`TransferPolicyProfile` naming its version string and the raw candidate
columns of each source; the version participates in the deterministic pair
sampling, so it must be threaded through rather than read from a constant.
"""

from __future__ import annotations

import bisect
import gzip
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools.chembl_tool.common.starling.normalization.cleaning import clean_scalar


MIN_ASSAY_TRANSFER_SAMPLES = 25
MINIMUM_LEVEL_RECORDS = 3
MINIMUM_COVERAGE = 0.50
MINIMUM_OMEGA_SQUARED = 0.20
MINIMUM_MEDIAN_RANGE_IQR = 1.00
MINIMUM_MEDIAN_RANGE_SD = 1.35
STANDARD_DEVIATION_DDOF = 1
TRANSFER_MAX_STANDARD_DEVIATIONS = 1.0
PERCENTILE_KNOTS = 101
MAX_REFERENCE_PAIRS = 10_000
NO_CANDIDATE = "__none__"
QUALIFYING_BASELINE = "__baseline__"

# Soft target.  The midpoint is the boolean decision boundary, so the two agree
# by construction: a pair exactly one SD apart scores 0.5 and flips the label.
# The temperature is derived, not tuned: half an SD either side of the boundary
# should read as 0.9 and 0.1, which fixes tau = 0.5 / ln(9) exactly.
SOFT_TRANSFER_MIDPOINT_SD = TRANSFER_MAX_STANDARD_DEVIATIONS
SOFT_TRANSFER_REFERENCE_OFFSET_SD = 0.5
SOFT_TRANSFER_REFERENCE_PROBABILITY = 0.9
SOFT_TRANSFER_TEMPERATURE = SOFT_TRANSFER_REFERENCE_OFFSET_SD / math.log(
    SOFT_TRANSFER_REFERENCE_PROBABILITY / (1.0 - SOFT_TRANSFER_REFERENCE_PROBABILITY)
)
SOFT_TRANSFER_CONTRACT_VERSION = "assay_transfer_soft_probability.v1"


@dataclass(frozen=True)
class TransferPolicyProfile:
    """Per-task identity for one pair-bucket transfer policy."""

    version: str
    source_candidate_fields: Mapping[str, tuple[str, ...]]
    # Buckets whose measurements take fewer than this many distinct values are
    # ineligible.  Anchor-encoded categorical buckets can otherwise clear the
    # record-count gate while carrying almost no spread.  1 disables the guard.
    minimum_distinct_levels: int = 1
    # Optional scale-aware overrides.  This avoids treating a complete binary
    # scale as deficient merely because an ordinal scale requires three levels.
    minimum_distinct_levels_by_scale: Mapping[str, int] | None = None
    # Only sources whose measurement is itself the benchmark gold outcome are
    # removed for held-out parents during calibration.
    heldout_sources: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        values = self.minimum_distinct_levels_by_scale or {}
        if self.minimum_distinct_levels < 1 or any(
            int(value) < 1 for value in values.values()
        ):
            raise ValueError("minimum distinct levels must be positive")

    def required_distinct_levels(self, scale_id: str | None) -> int:
        return int(
            (self.minimum_distinct_levels_by_scale or {}).get(
                str(scale_id or ""), self.minimum_distinct_levels
            )
        )

    def candidate_fields(self, source_id: str) -> tuple[str, ...]:
        try:
            return tuple(self.source_candidate_fields[source_id])
        except KeyError as error:
            raise ValueError(
                f"unknown source candidate contract: {source_id!r}"
            ) from error


def normalize_candidate_value(field_name: str, value: Any) -> str | None:
    """Apply the frozen raw-column missing-value and cleaning policy."""
    if value is None or bool(pd.isna(value)):
        return QUALIFYING_BASELINE if field_name == "qualifying_conditions" else None
    cleaned = clean_scalar(value)
    if cleaned in {None, ""}:
        return QUALIFYING_BASELINE if field_name == "qualifying_conditions" else None
    return str(cleaned)


def score_candidate(
    measurements: Sequence[Any] | pd.Series,
    values: Sequence[Any] | pd.Series,
    *,
    candidate_field: str,
) -> dict[str, Any] | None:
    """Score one raw categorical column with coverage-adjusted omega-squared."""
    y = pd.to_numeric(
        pd.Series(measurements, dtype="object").reset_index(drop=True),
        errors="coerce",
    )
    raw = pd.Series(values, dtype="object").reset_index(drop=True)
    if len(y) != len(raw):
        raise ValueError("candidate score inputs have different lengths")
    finite = np.isfinite(y.to_numpy(dtype=float, na_value=np.nan))
    y = y[finite].reset_index(drop=True)
    raw = raw[finite].reset_index(drop=True)
    if len(y) < MIN_ASSAY_TRANSFER_SAMPLES:
        return None

    categories = raw.map(
        lambda value: normalize_candidate_value(candidate_field, value)
    )
    present = categories.notna()
    level_counts = categories[present].value_counts(sort=False)
    supported_levels = sorted(
        str(level)
        for level, count in level_counts.items()
        if int(count) >= MINIMUM_LEVEL_RECORDS
    )
    if len(supported_levels) < 2:
        return None
    supported = present & categories.astype("string").isin(supported_levels)
    supported_count = int(supported.sum())
    coverage = supported_count / len(y)
    if coverage < MINIMUM_COVERAGE:
        return None

    supported_y = y[supported].astype(float)
    supported_categories = categories[supported].astype(str)
    n = len(supported_y)
    k = len(supported_levels)
    degrees_within = n - k
    if degrees_within <= 0:
        return None
    mean = float(supported_y.mean())
    ss_total = float(np.square(supported_y - mean).sum())
    if not math.isfinite(ss_total) or ss_total <= 0:
        return None
    grouped = supported_y.groupby(supported_categories, sort=True)
    ss_between = float(
        sum(
            len(group) * (float(group.mean()) - mean) ** 2
            for _, group in grouped
        )
    )
    ss_within = max(0.0, ss_total - ss_between)
    mean_square_within = ss_within / degrees_within
    omega_squared = max(
        0.0,
        (ss_between - (k - 1) * mean_square_within)
        / (ss_total + mean_square_within),
    )

    full_y = y.astype(float)
    bucket_iqr = float(full_y.quantile(0.75) - full_y.quantile(0.25))
    bucket_sd = float(full_y.std(ddof=STANDARD_DEVIATION_DDOF))
    denominator_iqr = bucket_iqr
    if not math.isfinite(denominator_iqr) or denominator_iqr <= 0:
        denominator_iqr = bucket_sd
    if not math.isfinite(denominator_iqr) or denominator_iqr <= 0:
        denominator_iqr = 1.0

    medians = grouped.median()
    median_range = float(medians.max() - medians.min())
    median_range_iqr = median_range / denominator_iqr
    median_range_sd = (
        median_range / bucket_sd
        if math.isfinite(bucket_sd) and bucket_sd > 0
        else None
    )
    gate_flagged = automatic_variance_gate(
        omega_squared=omega_squared,
        median_range_iqr=median_range_iqr,
        median_range_sd=median_range_sd,
    )
    level_summaries = sorted(
        (
            {
                "value": str(level),
                "record_count": int(len(group)),
                "median": float(group.median()),
                "q25": float(group.quantile(0.25)),
                "q75": float(group.quantile(0.75)),
            }
            for level, group in grouped
        ),
        key=lambda item: (-item["record_count"], item["value"]),
    )
    return {
        "candidate_column": candidate_field,
        "candidate_score": float(omega_squared * coverage),
        "omega_squared": float(omega_squared),
        "coverage": float(coverage),
        "median_range": median_range,
        "median_range_iqr": float(median_range_iqr),
        "median_range_sd": (
            float(median_range_sd) if median_range_sd is not None else None
        ),
        "supported_level_count": k,
        "supported_record_count": supported_count,
        "variance_gate_flagged": gate_flagged,
        "candidate_level_examples": level_summaries[:8],
    }


def select_variance_candidate(
    rows: pd.DataFrame,
    *,
    profile: TransferPolicyProfile,
    source_id: str,
) -> dict[str, Any]:
    """Select the highest-scoring raw source column for one pair bucket."""
    fields = profile.candidate_fields(source_id)
    required = {"finite_scalar_value", *fields}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"variance-gate rows lack columns: {sorted(missing)}")
    scores = [
        result
        for field_name in fields
        if (
            result := score_candidate(
                rows["finite_scalar_value"],
                rows[field_name],
                candidate_field=field_name,
            )
        )
        is not None
    ]
    scores.sort(
        key=lambda item: (
            -item["candidate_score"],
            -item["omega_squared"],
            -item["coverage"],
            item["candidate_column"],
        )
    )
    if not scores or scores[0]["candidate_score"] <= 0:
        return {
            "evaluated": True,
            "candidate_column": NO_CANDIDATE,
            "candidate_score": None,
            "omega_squared": None,
            "coverage": None,
            "median_range": None,
            "median_range_iqr": None,
            "median_range_sd": None,
            "supported_level_count": 0,
            "supported_record_count": 0,
            "variance_gate_flagged": False,
            "candidate_level_examples": [],
        }
    return {"evaluated": True, **scores[0]}


def automatic_variance_gate(
    *,
    omega_squared: Any,
    median_range_iqr: Any,
    median_range_sd: Any,
) -> bool:
    omega = _finite_or_none(omega_squared)
    iqr_ratio = _finite_or_none(median_range_iqr)
    sd_ratio = _finite_or_none(median_range_sd)
    return bool(
        omega is not None
        and omega >= MINIMUM_OMEGA_SQUARED
        and (
            (iqr_ratio is not None and iqr_ratio >= MINIMUM_MEDIAN_RANGE_IQR)
            or (sd_ratio is not None and sd_ratio >= MINIMUM_MEDIAN_RANGE_SD)
        )
    )


def build_distance_policy(
    measurements: Sequence[Any],
    record_ids: Sequence[str],
    *,
    pair_bucket_key: str,
    profile: TransferPolicyProfile,
) -> dict[str, Any]:
    """Build one SD scale and a deterministic local empirical distance curve."""
    if len(measurements) != len(record_ids):
        raise ValueError("distance-policy measurements and record IDs differ in length")
    ordered = sorted(
        ((str(record_id), float(value)) for record_id, value in zip(record_ids, measurements)),
        key=lambda item: item[0],
    )
    values = np.asarray([value for _, value in ordered], dtype=float)
    if len(values) < MIN_ASSAY_TRANSFER_SAMPLES or not np.isfinite(values).all():
        raise ValueError("distance policy requires at least 25 finite measurements")
    sample_sd = float(np.std(values, ddof=STANDARD_DEVIATION_DDOF))
    if not math.isfinite(sample_sd) or sample_sd <= 0:
        raise ValueError("distance policy requires a positive finite sample SD")

    total_pairs = len(values) * (len(values) - 1) // 2
    if total_pairs <= MAX_REFERENCE_PAIRS:
        absolute_differences = np.fromiter(
            (
                abs(float(values[left]) - float(values[right]))
                for left in range(len(values) - 1)
                for right in range(left + 1, len(values))
            ),
            dtype=float,
            count=total_pairs,
        )
        reference_method = "all_unordered_record_pairs"
    else:
        absolute_differences = np.empty(MAX_REFERENCE_PAIRS, dtype=float)
        for pair_index in range(MAX_REFERENCE_PAIRS):
            left, right = _hashed_distinct_indices(
                pair_bucket_key, pair_index, len(values), version=profile.version
            )
            absolute_differences[pair_index] = abs(
                float(values[left]) - float(values[right])
            )
        reference_method = "deterministic_hash_sample_with_replacement"
    standardized = absolute_differences / sample_sd
    percentiles = np.linspace(0.0, 1.0, PERCENTILE_KNOTS)
    knots = [
        float(value)
        for value in np.quantile(standardized, percentiles, method="linear")
    ]
    policy = {
        "sample_standard_deviation": sample_sd,
        "standard_deviation_ddof": STANDARD_DEVIATION_DDOF,
        "distance_definition": "abs(left - right) / sample_standard_deviation",
        "transfer_max_standard_deviations": TRANSFER_MAX_STANDARD_DEVIATIONS,
        "percentile_scope": "within_pair_bucket",
        "percentile_knot_count": PERCENTILE_KNOTS,
        "standardized_distance_percentile_knots": knots,
        "reference_method": reference_method,
        "reference_pair_count": int(len(standardized)),
        "total_possible_unordered_pairs": total_pairs,
    }
    policy["one_sd_percentile_0_100"] = percentile_score(1.0, policy)
    return policy


def percentile_score(
    standardized_distance: float,
    distance_policy: Mapping[str, Any],
) -> float:
    """Interpolate the local empirical percentile, with exact equality at zero."""
    distance = float(standardized_distance)
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("standardized distance must be finite and nonnegative")
    if distance == 0:
        return 0.0
    knots = [
        float(value)
        for value in distance_policy["standardized_distance_percentile_knots"]
    ]
    if len(knots) != PERCENTILE_KNOTS or any(
        not math.isfinite(value) for value in knots
    ):
        raise ValueError("distance policy has invalid percentile knots")
    if any(right < left for left, right in zip(knots, knots[1:])):
        raise ValueError("distance policy percentile knots are not monotone")
    if distance >= knots[-1]:
        return 100.0
    lower_index = max(0, bisect.bisect_right(knots, distance) - 1)
    upper_index = min(PERCENTILE_KNOTS - 1, lower_index + 1)
    lower_value = knots[lower_index]
    upper_value = knots[upper_index]
    if upper_value <= lower_value:
        return float(upper_index)
    fraction = (distance - lower_value) / (upper_value - lower_value)
    return min(100.0, max(0.0, float(lower_index) + fraction))


def soft_transfer_probability(
    standardized_distance: float,
    *,
    midpoint: float = SOFT_TRANSFER_MIDPOINT_SD,
    temperature: float = SOFT_TRANSFER_TEMPERATURE,
) -> float:
    """Squash an SD-standardized distance into a continuous transfer target.

    This is the logistic complement of the boolean label: it is exactly 0.5 at
    ``midpoint``, which is the same standardized distance at which
    ``assay_transfer_label`` flips, so the soft and hard targets never disagree
    about which side of the boundary a pair falls on.  As ``temperature``
    approaches zero the function approaches the boolean step.

    Categorical evidence encoded onto a few discrete levels has a discrete
    distance; this step is where its continuity comes from.
    """
    distance = float(standardized_distance)
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("standardized distance must be finite and nonnegative")
    if temperature <= 0:
        raise ValueError("soft transfer temperature must be positive")
    # Numerically stable logistic, matching the pattern used for the reranker's
    # two-logit softmax rather than calling exp on a large positive argument.
    exponent = (distance - float(midpoint)) / float(temperature)
    if exponent >= 0:
        decayed = math.exp(-exponent)
        return decayed / (1.0 + decayed)
    grown = math.exp(exponent)
    return 1.0 / (1.0 + grown)


def soft_transfer_contract() -> dict[str, Any]:
    """Self-describing metadata for the soft target, for the policy artifact."""
    return {
        "contract_version": SOFT_TRANSFER_CONTRACT_VERSION,
        "definition": "1 / (1 + exp((standardized_distance - midpoint) / temperature))",
        "midpoint_standard_deviations": SOFT_TRANSFER_MIDPOINT_SD,
        "temperature": SOFT_TRANSFER_TEMPERATURE,
        "agrees_with_boolean_at_midpoint": True,
        "value_at_midpoint": 0.5,
        "reference_points": {
            "0.5_sd": soft_transfer_probability(0.5),
            "1.0_sd": soft_transfer_probability(1.0),
            "1.5_sd": soft_transfer_probability(1.5),
        },
        "replaces_boolean_label": False,
        "is_a_calibrated_probability": False,
    }


def evaluate_pair_bucket_transfer(
    policy: Mapping[str, Any],
    pair_bucket_key: str,
    left: float,
    right: float,
) -> dict[str, Any]:
    """Evaluate one numeric pair without turning ineligibility into a negative label."""
    entries = policy.get("buckets")
    if not isinstance(entries, Mapping):
        raise ValueError("pair-bucket transfer policy has no bucket mapping")
    entry = entries.get(str(pair_bucket_key))
    if not isinstance(entry, Mapping):
        return _unscored_result("unknown_pair_bucket")
    if not bool(entry.get("assay_transfer_eligible")):
        return _unscored_result(str(entry.get("eligibility_reason") or "ineligible_bucket"))
    distance_policy = entry.get("distance_policy")
    if not isinstance(distance_policy, Mapping):
        raise ValueError("eligible pair bucket lacks a distance policy")
    left_value = float(left)
    right_value = float(right)
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        raise ValueError("assay-transfer measurements must be finite")
    sample_sd = float(distance_policy["sample_standard_deviation"])
    standardized = abs(left_value - right_value) / sample_sd
    return {
        "assay_transfer_eligible": True,
        "eligibility_reason": "eligible",
        "absolute_difference": abs(left_value - right_value),
        "standardized_difference_sd": standardized,
        "distance_percentile_0_100": percentile_score(
            standardized, distance_policy
        ),
        "assay_transfer_label": bool(
            standardized <= TRANSFER_MAX_STANDARD_DEVIATIONS
        ),
        "soft_transfer_probability": soft_transfer_probability(standardized),
        "transfer_max_standard_deviations": TRANSFER_MAX_STANDARD_DEVIATIONS,
    }


def load_pair_bucket_transfer_policy(
    path: str | Path, *, profile: TransferPolicyProfile
) -> dict[str, Any]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_pair_bucket_transfer_policy(payload, profile=profile)
    return payload


def validate_pair_bucket_transfer_policy(
    payload: Mapping[str, Any], *, profile: TransferPolicyProfile
) -> None:
    if payload.get("policy_version") != profile.version:
        raise ValueError("pair-bucket transfer policy version mismatch")
    buckets = payload.get("buckets")
    if not isinstance(buckets, Mapping) or not buckets:
        raise ValueError("pair-bucket transfer policy has no buckets")
    for bucket_key, entry in buckets.items():
        if not isinstance(bucket_key, str) or not isinstance(entry, Mapping):
            raise ValueError("invalid pair-bucket transfer policy entry")
        record_count = int(entry.get("record_count") or 0)
        support_met = record_count >= MIN_ASSAY_TRANSFER_SAMPLES
        if bool(entry.get("minimum_support_met")) != support_met:
            raise ValueError(f"minimum-support mismatch for {bucket_key}")
        gate = entry.get("variance_gate")
        if not isinstance(gate, Mapping):
            raise ValueError(f"variance-gate evidence is missing for {bucket_key}")
        expected_eligible = bool(
            support_met
            and not bool(gate.get("variance_gate_flagged"))
            and entry.get("distance_policy") is not None
        )
        if bool(entry.get("assay_transfer_eligible")) != expected_eligible:
            raise ValueError(f"eligibility is inconsistent for {bucket_key}")
        if expected_eligible:
            distance_policy = entry["distance_policy"]
            sample_sd = float(distance_policy["sample_standard_deviation"])
            if not math.isfinite(sample_sd) or sample_sd <= 0:
                raise ValueError(f"invalid sample SD for {bucket_key}")
            knots = distance_policy["standardized_distance_percentile_knots"]
            if len(knots) != PERCENTILE_KNOTS or any(
                float(right) < float(left)
                for left, right in zip(knots, knots[1:])
            ):
                raise ValueError(f"invalid percentile knots for {bucket_key}")


def _hashed_distinct_indices(
    pair_bucket_key: str, pair_index: int, size: int, *, version: str
) -> tuple[int, int]:
    digest = hashlib.sha256(
        f"{version}\0{pair_bucket_key}\0{pair_index}".encode("utf-8")
    ).digest()
    left = int.from_bytes(digest[:8], "big") % size
    right_without_left = int.from_bytes(digest[8:16], "big") % (size - 1)
    right = right_without_left if right_without_left < left else right_without_left + 1
    return left, right


def _unscored_result(reason: str) -> dict[str, Any]:
    return {
        "assay_transfer_eligible": False,
        "eligibility_reason": reason,
        "absolute_difference": None,
        "standardized_difference_sd": None,
        "distance_percentile_0_100": None,
        "assay_transfer_label": None,
        "soft_transfer_probability": None,
        "transfer_max_standard_deviations": TRANSFER_MAX_STANDARD_DEVIATIONS,
    }


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


__all__ = [
    "MAX_REFERENCE_PAIRS",
    "MIN_ASSAY_TRANSFER_SAMPLES",
    "MINIMUM_COVERAGE",
    "MINIMUM_LEVEL_RECORDS",
    "MINIMUM_MEDIAN_RANGE_IQR",
    "MINIMUM_MEDIAN_RANGE_SD",
    "MINIMUM_OMEGA_SQUARED",
    "NO_CANDIDATE",
    "PERCENTILE_KNOTS",
    "QUALIFYING_BASELINE",
    "SOFT_TRANSFER_CONTRACT_VERSION",
    "SOFT_TRANSFER_MIDPOINT_SD",
    "SOFT_TRANSFER_REFERENCE_OFFSET_SD",
    "SOFT_TRANSFER_REFERENCE_PROBABILITY",
    "SOFT_TRANSFER_TEMPERATURE",
    "STANDARD_DEVIATION_DDOF",
    "TRANSFER_MAX_STANDARD_DEVIATIONS",
    "TransferPolicyProfile",
    "automatic_variance_gate",
    "build_distance_policy",
    "evaluate_pair_bucket_transfer",
    "load_pair_bucket_transfer_policy",
    "normalize_candidate_value",
    "percentile_score",
    "score_candidate",
    "select_variance_candidate",
    "soft_transfer_contract",
    "soft_transfer_probability",
    "validate_pair_bucket_transfer_policy",
]
