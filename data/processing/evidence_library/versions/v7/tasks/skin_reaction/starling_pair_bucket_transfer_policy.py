"""Skin_Reaction binding for the shared pair-bucket transfer policy.

The algorithm lives in ``common/starling/pair_bucket_transfer_policy.py``.
This module pins this task's policy version, the raw candidate columns of each
source and the degenerate-scale guard, then
re-exports the wrappers under the names task callers use.

Candidate columns are raw factual source values only.  They are read for a
numerical heterogeneity check; they never create child buckets, never appear in
a pair-bucket key, and are never shown to an LLM.  The column that a source
feeds into ``global_context`` is deliberately absent, because that value is
already the bucket identity: ``study_design`` for ``skin_exposure`` and
``assay_type`` for ``sensitization_aop``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from data.processing.evidence_library.shared.v1.pair_bucket_transfer_policy import (
    MAX_REFERENCE_PAIRS,
    MIN_ASSAY_TRANSFER_SAMPLES,
    MINIMUM_COVERAGE,
    MINIMUM_LEVEL_RECORDS,
    MINIMUM_MEDIAN_RANGE_IQR,
    MINIMUM_MEDIAN_RANGE_SD,
    MINIMUM_OMEGA_SQUARED,
    NO_CANDIDATE,
    PERCENTILE_KNOTS,
    QUALIFYING_BASELINE,
    STANDARD_DEVIATION_DDOF,
    TRANSFER_MAX_STANDARD_DEVIATIONS,
    TransferPolicyProfile,
    automatic_variance_gate,
    evaluate_pair_bucket_transfer,
    normalize_candidate_value,
    percentile_score,
    score_candidate,
    soft_transfer_contract,
    soft_transfer_probability,
)
from data.processing.evidence_library.shared.v1.pair_bucket_transfer_policy import (
    build_distance_policy as _build_distance_policy,
)
from data.processing.evidence_library.shared.v1.pair_bucket_transfer_policy import (
    load_pair_bucket_transfer_policy as _load_pair_bucket_transfer_policy,
)
from data.processing.evidence_library.shared.v1.pair_bucket_transfer_policy import (
    select_variance_candidate as _select_variance_candidate,
)
from data.processing.evidence_library.shared.v1.pair_bucket_transfer_policy import (
    validate_pair_bucket_transfer_policy as _validate_pair_bucket_transfer_policy,
)


PAIR_BUCKET_TRANSFER_POLICY_VERSION = "skin_reaction_pair_bucket_transfer_policy.v3"

# Every column below is a persisted column of ``03_records/records.parquet`` and
# a declared source column of its source in ``starling_source_column_contracts``.
SOURCE_CANDIDATE_FIELDS: dict[str, tuple[str, ...]] = {
    "direct_skin_reaction": (
        "assay_or_test",
        "species_or_population",
        "dose_or_concentration",
        "extra_details",
    ),
    "sensitization_aop": (
        # `aop_event` is deliberately absent: it is a pair-bucket identity field,
        # so it is constant inside every bucket and can never score as a
        # heterogeneity candidate.  Same reason `assay_type` is absent -- it is
        # the raw input to this source's reconciled context and species fields.
        "experimental_conditions",
        "qualifying_conditions",
    ),
    "phototoxicity_irritation_local_damage": (
        "evidence_system",
        "assay_method",
        "light_conditions",
        "qualifying_conditions",
    ),
    "skin_exposure": (
        "skin_source",
        "formulation_vehicle",
        "exposure_time",
        "qualifying_conditions",
    ),
}

# Anchor-encoded categorical evidence can clear the record-count gate while
# taking almost no distinct values, which makes the within-bucket SD scale
# meaningless.  Three distinct measurements is the minimum that admits a spread.
MINIMUM_DISTINCT_MEASUREMENT_LEVELS = 3

TRANSFER_POLICY_PROFILE = TransferPolicyProfile(
    version=PAIR_BUCKET_TRANSFER_POLICY_VERSION,
    source_candidate_fields=SOURCE_CANDIDATE_FIELDS,
    minimum_distinct_levels=MINIMUM_DISTINCT_MEASUREMENT_LEVELS,
    minimum_distinct_levels_by_scale={"single_subject_fraction.v1": 2},
)


def candidate_fields(source_id: str) -> tuple[str, ...]:
    return TRANSFER_POLICY_PROFILE.candidate_fields(source_id)


def select_variance_candidate(
    rows: pd.DataFrame, *, source_id: str
) -> dict[str, Any]:
    return _select_variance_candidate(
        rows, profile=TRANSFER_POLICY_PROFILE, source_id=source_id
    )


def build_distance_policy(
    measurements: Sequence[Any],
    record_ids: Sequence[str],
    *,
    pair_bucket_key: str,
) -> dict[str, Any]:
    return _build_distance_policy(
        measurements,
        record_ids,
        pair_bucket_key=pair_bucket_key,
        profile=TRANSFER_POLICY_PROFILE,
    )


def load_pair_bucket_transfer_policy(path: str | Path) -> dict[str, Any]:
    return _load_pair_bucket_transfer_policy(path, profile=TRANSFER_POLICY_PROFILE)


def validate_pair_bucket_transfer_policy(payload: Mapping[str, Any]) -> None:
    _validate_pair_bucket_transfer_policy(payload, profile=TRANSFER_POLICY_PROFILE)


__all__ = [
    "MAX_REFERENCE_PAIRS",
    "MIN_ASSAY_TRANSFER_SAMPLES",
    "MINIMUM_COVERAGE",
    "MINIMUM_DISTINCT_MEASUREMENT_LEVELS",
    "MINIMUM_LEVEL_RECORDS",
    "MINIMUM_MEDIAN_RANGE_IQR",
    "MINIMUM_MEDIAN_RANGE_SD",
    "MINIMUM_OMEGA_SQUARED",
    "NO_CANDIDATE",
    "PAIR_BUCKET_TRANSFER_POLICY_VERSION",
    "PERCENTILE_KNOTS",
    "QUALIFYING_BASELINE",
    "SOURCE_CANDIDATE_FIELDS",
    "STANDARD_DEVIATION_DDOF",
    "TRANSFER_MAX_STANDARD_DEVIATIONS",
    "TRANSFER_POLICY_PROFILE",
    "automatic_variance_gate",
    "build_distance_policy",
    "candidate_fields",
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
