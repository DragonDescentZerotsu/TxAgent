"""Bioavailability_Ma binding for the shared pair-bucket transfer policy.

The algorithm lives in ``common/starling/pair_bucket_transfer_policy.py``.
This module pins this task's policy version and the raw candidate columns of
each source, and re-exports the wrappers under their historical names so
existing callers keep working.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from tools.chembl_tool.common.starling.pair_bucket_transfer_policy import (
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
from tools.chembl_tool.common.starling.pair_bucket_transfer_policy import (
    build_distance_policy as _build_distance_policy,
)
from tools.chembl_tool.common.starling.pair_bucket_transfer_policy import (
    load_pair_bucket_transfer_policy as _load_pair_bucket_transfer_policy,
)
from tools.chembl_tool.common.starling.pair_bucket_transfer_policy import (
    select_variance_candidate as _select_variance_candidate,
)
from tools.chembl_tool.common.starling.pair_bucket_transfer_policy import (
    validate_pair_bucket_transfer_policy as _validate_pair_bucket_transfer_policy,
)


PAIR_BUCKET_TRANSFER_POLICY_VERSION = (
    "bioavailability_ma_pair_bucket_transfer_policy.v1"
)

# Only factual source columns are candidates.  They are used for a numerical
# heterogeneity check, not as pair-bucket identity fields or LLM prompt inputs.
SOURCE_CANDIDATE_FIELDS: dict[str, tuple[str, ...]] = {
    "direct_hf": (
        "species_or_population",
        "dose",
        "oral_exposure_mode",
        "comparator",
        "qualifying_conditions",
    ),
    "oral_exposure": (
        "statistic_type",
        "oral_dose",
        "study_context",
        "comparator_exposure",
        "qualifying_conditions",
    ),
    "fa": (
        "condition_medium",
        "biological_context",
        "formulation_or_solid_form",
        "qualifying_conditions",
    ),
    "fg": (
        "transporter_or_enzyme",
        "substrate_status",
        "intestinal_site",
        "qualifying_conditions",
    ),
    "fh": (
        "molecular_form",
        "enzyme_or_pathway",
        "qualifying_conditions",
    ),
}

TRANSFER_POLICY_PROFILE = TransferPolicyProfile(
    version=PAIR_BUCKET_TRANSFER_POLICY_VERSION,
    source_candidate_fields=SOURCE_CANDIDATE_FIELDS,
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
