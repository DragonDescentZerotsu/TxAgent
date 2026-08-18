"""Nominate assay-transfer tail buckets and probable unit-scale defects.

This is an audit tool, not an automatic policy writer. Statistical evidence may
nominate a row, but only reviewed source/unit provenance may make it
assay-transfer-ineligible in a frozen task policy.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import skew


REVIEW_VERSION = "assay_transfer_measurement_review.v1"
_POSITIVE_FACTOR_RE = re.compile(
    r"(?:(?:×|x)\s*10\s*(?:\^\s*)?|10\s*\^\s*)"
    r"\+?(?P<exponent>[1-9]\d+|[2-9])",
    re.IGNORECASE,
)


def review_artifacts(
    records_path: str | Path,
    buckets_path: str | Path,
    *,
    minimum_records: int = 25,
) -> dict[str, Any]:
    records = pd.read_parquet(records_path)
    buckets = pd.read_parquet(buckets_path)
    joined = _join_records(records, buckets)
    eligible = joined[_eligibility_mask(joined)].copy()
    eligible["review_value"] = _review_values(eligible)
    eligible["review_bucket_key"] = _review_keys(eligible)
    candidates = _tail_candidates(eligible, minimum_records)
    defects = _probable_unit_defects(eligible, candidates)
    return {
        "review_version": REVIEW_VERSION,
        "scope": "assay_transfer_only",
        "screen": {
            "minimum_records": minimum_records,
            "candidate_if": "abs(raw_skew) >= 2 OR log10(Q95/Q50) >= 2",
            "probable_unit_defect_if": (
                "positive factor >=1e2; robust log z >=6; removing the factor "
                "or reversing its sign lands within 3 scaled MAD"
            ),
            "automatic_policy_mutation": False,
        },
        "summary": {
            "eligible_records": int(len(eligible)),
            "tail_candidate_buckets": len(candidates),
            "probable_unit_defect_rows": len(defects),
        },
        "tail_candidates": candidates,
        "probable_unit_defects": defects,
    }


def _join_records(records: pd.DataFrame, buckets: pd.DataFrame) -> pd.DataFrame:
    key = "canonical_record_id" if "canonical_record_id" in records else "normalized_record_id"
    if not records[key].is_unique or not buckets[key].is_unique:
        raise ValueError("record IDs must be unique in both artifacts")
    if len(records) != len(buckets):
        raise ValueError("Stage-03/04 row coverage differs")
    joined = buckets.merge(records, on=key, how="left", validate="one_to_one")
    if len(joined) != len(records):
        raise ValueError("Stage-03/04 join lost records")
    return joined


def _eligibility_mask(rows: pd.DataFrame) -> pd.Series:
    field = (
        "assay_transfer_eligible"
        if "assay_transfer_eligible" in rows
        else "bucket_eligible"
    )
    return rows[field].fillna(False).astype(bool) & rows["pair_bucket_key"].notna()


def _review_values(rows: pd.DataFrame) -> pd.Series:
    field = (
        "assay_transfer_pretransform_scalar_value"
        if "assay_transfer_pretransform_scalar_value" in rows
        else "finite_scalar_value"
    )
    return pd.to_numeric(rows[field], errors="coerce")


def _review_keys(rows: pd.DataFrame) -> pd.Series:
    if "assay_transfer_policy_key" in rows:
        return rows["assay_transfer_policy_key"].fillna(rows["pair_bucket_key"])
    return rows["pair_bucket_key"]


def _tail_candidates(rows: pd.DataFrame, minimum_records: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    # V7 libraries can contain millions of singleton context strata.  Count
    # first so the Python-level statistics loop never materializes groups that
    # cannot pass the already-frozen minimum-support gate.
    counts = rows.groupby("review_bucket_key", sort=False).size()
    supported_keys = counts[counts >= minimum_records].index
    supported = rows[rows["review_bucket_key"].isin(supported_keys)]
    for key, group in supported.groupby("review_bucket_key", sort=True):
        values = group["review_value"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < minimum_records or np.any(values <= 0):
            continue
        stats = _tail_statistics(values)
        reasons = {
            "absolute_raw_skew_at_least_2": abs(stats["raw_skew"]) >= 2,
            "log10_q95_over_q50_at_least_2": stats["log10_q95_over_q50"] >= 2,
        }
        if any(reasons.values()):
            output.append({"pair_bucket_key": str(key), **stats, "screen_reasons": reasons})
    return output


def _tail_statistics(values: np.ndarray) -> dict[str, Any]:
    q05, q50, q95 = np.quantile(values, [0.05, 0.5, 0.95])
    raw_skew = _safe_skew(values)
    log_skew = _safe_skew(np.log10(values))
    return {
        "record_count": int(len(values)),
        "minimum": float(np.min(values)),
        "q05": float(q05),
        "q50": float(q50),
        "q95": float(q95),
        "maximum": float(np.max(values)),
        "raw_skew": raw_skew,
        "log10_skew": log_skew,
        "log10_q95_over_q50": float(math.log10(q95 / q50)),
    }


def _safe_skew(values: np.ndarray) -> float:
    if np.ptp(values) <= np.finfo(float).eps * max(1.0, abs(float(values[0]))):
        return 0.0
    result = float(skew(values, bias=False))
    return result if math.isfinite(result) else 0.0


def _probable_unit_defects(
    rows: pd.DataFrame, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    candidate_keys = {item["pair_bucket_key"] for item in candidates}
    output: list[dict[str, Any]] = []
    candidate_rows = rows[
        rows["review_bucket_key"].astype(str).isin(candidate_keys)
    ]
    for key, group in candidate_rows.groupby("review_bucket_key", sort=True):
        center, spread = _robust_log_center(group["review_value"])
        if spread is None:
            continue
        for _, row in group.iterrows():
            item = _unit_defect_candidate(row, center, spread)
            if item is not None:
                output.append(item)
    return output


def _robust_log_center(values: pd.Series) -> tuple[float, float | None]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array) & (array > 0)]
    logs = np.log10(array)
    center = float(np.median(logs))
    spread = float(1.4826 * np.median(np.abs(logs - center)))
    return center, spread if math.isfinite(spread) and spread > 0 else None


def _unit_defect_candidate(
    row: pd.Series, center: float, spread: float
) -> dict[str, Any] | None:
    value = float(row["review_value"])
    factor = _positive_factor(row)
    if not math.isfinite(value) or value <= 0 or factor is None:
        return None
    log_value = math.log10(value)
    log_factor = math.log10(factor)
    robust_z = (log_value - center) / spread
    shifted = (log_value - log_factor - center) / spread
    inverted = (log_value - 2 * log_factor - center) / spread
    if robust_z < 6 or min(abs(shifted), abs(inverted)) > 3:
        return None
    record_id = row.get("canonical_record_id") or row.get("normalized_record_id")
    return {
        "record_id": str(record_id),
        "pair_bucket_key": str(row["review_bucket_key"]),
        "value": value,
        "positive_factor": factor,
        "robust_log_z": robust_z,
        "remove_factor_robust_z": shifted,
        "reverse_sign_robust_z": inverted,
        "measurement_text": row.get("measurement_text"),
        "unit_text": row.get("unit_text"),
        "support_text": row.get("support_text"),
        "action": "manual_review_only",
    }


def _positive_factor(row: pd.Series) -> float | None:
    persisted = pd.to_numeric(row.get("unit_notation_factor"), errors="coerce")
    if pd.notna(persisted) and float(persisted) >= 100:
        return float(persisted)
    text = " ".join(str(row.get(field) or "") for field in ("unit_text", "support_text"))
    match = _POSITIVE_FACTOR_RE.search(text)
    if match is None:
        return None
    exponent = int(match["exponent"])
    # The reviewer is a nomination layer, not a permissive parser.  Oversized
    # exponents are not representable as finite float scale factors and must
    # fail closed instead of crashing the full-corpus audit.
    return 10.0**exponent if exponent <= 300 else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--pair-buckets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-records", type=int, default=25)
    args = parser.parse_args(argv)
    payload = review_artifacts(
        args.records, args.pair_buckets, minimum_records=args.minimum_records
    )
    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
