"""BBB Martins binding for the shared assay-transfer distance policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from tools.chembl_tool.common.starling.pair_bucket_transfer_policy import (
    TransferPolicyProfile,
    build_distance_policy as _build_distance_policy,
    load_pair_bucket_transfer_policy as _load_policy,
    select_variance_candidate as _select_variance_candidate,
    validate_pair_bucket_transfer_policy as _validate_policy,
)


PAIR_BUCKET_TRANSFER_POLICY_VERSION = "bbb_martins_pair_bucket_transfer_policy.v1"
MINIMUM_DISTINCT_MEASUREMENT_LEVELS = 2
HELDOUT_SOURCES = frozenset({"direct_bbb"})

SOURCE_CANDIDATE_FIELDS: dict[str, tuple[str, ...]] = {
    "direct_bbb": ("bbb_transport_label", "qualifying_conditions"),
    "passive_permeability": (
        "metric_uncertainty",
        "qualifying_conditions",
        "needs_more_context",
    ),
    "efflux_transport": (
        "perturbation",
        "qualifying_conditions",
        "needs_more_context",
    ),
    "influx_transport": (),
}

TRANSFER_POLICY_PROFILE = TransferPolicyProfile(
    version=PAIR_BUCKET_TRANSFER_POLICY_VERSION,
    source_candidate_fields=SOURCE_CANDIDATE_FIELDS,
    minimum_distinct_levels=MINIMUM_DISTINCT_MEASUREMENT_LEVELS,
    heldout_sources=HELDOUT_SOURCES,
)


def select_variance_candidate(rows: pd.DataFrame, *, source_id: str) -> dict[str, Any]:
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
    return _load_policy(path, profile=TRANSFER_POLICY_PROFILE)


def validate_pair_bucket_transfer_policy(payload: Mapping[str, Any]) -> None:
    _validate_policy(payload, profile=TRANSFER_POLICY_PROFILE)


__all__ = [
    "HELDOUT_SOURCES",
    "MINIMUM_DISTINCT_MEASUREMENT_LEVELS",
    "PAIR_BUCKET_TRANSFER_POLICY_VERSION",
    "SOURCE_CANDIDATE_FIELDS",
    "TRANSFER_POLICY_PROFILE",
    "build_distance_policy",
    "load_pair_bucket_transfer_policy",
    "select_variance_candidate",
    "validate_pair_bucket_transfer_policy",
]

