"""Build v7 pair-bucket distance geometry without transfer labels or cutoffs.

Stage 04 is the sole authority for bucket membership and measurement kind.
This module only validates whether the observed bucket can support a distance
scale, audits residual heterogeneity in untouched source fields, and stores an
empirical within-bucket percentile curve.  It never emits a pair label, a
transfer threshold, or a probability.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.starling.build_pair_bucket_transfer_policy import (
    TransferPolicyBuildSpec,
    _validate_global_context_contract,
    write_deterministic_gzip,
)
from tools.chembl_tool.common.starling.canonicalization_v7 import (
    StarlingRecordContract,
)
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.starling.pair_bucket_transfer_policy import (
    MAX_REFERENCE_PAIRS,
    MINIMUM_COVERAGE,
    MINIMUM_LEVEL_RECORDS,
    MINIMUM_OMEGA_SQUARED,
    PERCENTILE_KNOTS,
    STANDARD_DEVIATION_DDOF,
    normalize_candidate_value,
    select_variance_candidate,
)


CALIBRATION_FILENAME = "pair_bucket_distance_calibration.json.gz"
CALIBRATION_VERSION = "pair_bucket_distance_calibration.v1"
MINIMUM_BUCKET_RECORDS = 25
MINIMUM_CATEGORICAL_CRAMERS_V_SQUARED = 0.20


def build_pair_bucket_distance_calibration(
    *,
    spec: TransferPolicyBuildSpec,
    record_contract: StarlingRecordContract,
    records_path: str | Path,
    pair_bucket_records_path: str | Path,
    pair_bucket_metadata_path: str | Path,
    auxiliary_manifest_path: str | Path,
    out_dir: str | Path,
    minimum_samples: int = MINIMUM_BUCKET_RECORDS,
    workers: int = 1,
) -> dict[str, Any]:
    """Materialize one observed distance calibration per Stage-04 bucket."""
    if minimum_samples != MINIMUM_BUCKET_RECORDS:
        raise ValueError("the v7 calibration contract requires exactly 25 records")
    records_path = Path(records_path)
    bucket_path = Path(pair_bucket_records_path)
    metadata_path = Path(pair_bucket_metadata_path)
    auxiliary_path = Path(auxiliary_manifest_path)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    bucket_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    auxiliary_metadata = json.loads(auxiliary_path.read_text(encoding="utf-8"))
    _validate_global_context_contract(spec, bucket_metadata, auxiliary_metadata)

    candidate_columns = sorted(
        {
            field
            for bucket in record_contract.pair_buckets.values()
            for field in bucket.variance_candidates
        }
    )
    schema = set(pq.read_schema(records_path).names)
    required = {
        "canonical_record_id",
        "finite_scalar_value",
        "measurement_kind",
        "canonical_measurement_scale_id",
        "canonical_category_id",
        "canonical_category_rank",
        *candidate_columns,
    }
    if spec.profile.heldout_sources:
        required.add(spec.heldout_identity_column)
    missing = required - schema
    if missing:
        raise ValueError(f"v7 calibration records lack columns: {sorted(missing)}")
    records = pd.read_parquet(records_path, columns=sorted(required))
    buckets = pd.read_parquet(bucket_path)
    if not records["canonical_record_id"].is_unique:
        raise ValueError("finalized canonical_record_id values must be unique")
    if not buckets["canonical_record_id"].is_unique:
        raise ValueError("pair-bucket canonical_record_id values must be unique")
    if len(records) != len(buckets):
        raise ValueError("record/pair-bucket coverage mismatch")
    joined = buckets.merge(
        records,
        on="canonical_record_id",
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("_bucket", ""),
    )
    if not (joined["_merge"] == "both").all():
        raise ValueError("one or more pair-bucket rows lack a finalized record")
    rows = joined[
        joined["bucket_eligible"].astype(bool) & joined["pair_bucket_key"].notna()
    ].copy()
    heldout_audit = _remove_heldout(spec, rows)
    if heldout_audit is not None:
        rows = rows.loc[~heldout_audit.pop("_drop_mask")].copy()

    entries = _build_calibration_entries(
        rows,
        spec=spec,
        record_contract=record_contract,
        minimum_samples=minimum_samples,
        workers=workers,
    )

    valid_entries = [entry for entry in entries.values() if entry["calibration_valid"]]
    flagged_entries = [
        entry
        for entry in entries.values()
        if entry["residual_heterogeneity_gate"]["variance_gate_flagged"]
    ]
    reasons = Counter(str(entry["calibration_reason"]) for entry in entries.values())
    payload: dict[str, Any] = {
        "calibration_version": CALIBRATION_VERSION,
        "record_contract_version": record_contract.version,
        "pair_bucket_version": spec.pair_bucket_version,
        "semantics": {
            "pair_bucket_membership_authority": "04_pair_buckets",
            "calibration_changes_membership": False,
            "pair_labels_emitted": False,
            "transfer_cutoff_emitted": False,
            "transfer_probability_emitted": False,
            "continuous_geometry": "finite_scalar_value",
            "binary_and_ordinal_geometry": "canonical_category_rank",
            "distance": "abs(left - right) / observed_sample_standard_deviation",
            "percentile_scope": "within_pair_bucket",
        },
        "measurement_scales": {
            key: value.manifest()
            for key, value in sorted(record_contract.measurement_scales.items())
        },
        "validation_contract": {
            "minimum_bucket_records": minimum_samples,
            "binary": "all declared levels observed; every observed level has >=3 records",
            "ordinal": "at least three declared levels observed; every observed level has >=3 records",
            "categorical_residual_heterogeneity": {
                "statistic": "bias_corrected_cramers_v_squared",
                "minimum_records_per_candidate_level": MINIMUM_LEVEL_RECORDS,
                "minimum_coverage": MINIMUM_COVERAGE,
                "flag_at_or_above": MINIMUM_CATEGORICAL_CRAMERS_V_SQUARED,
            },
            "continuous_residual_heterogeneity": {
                "statistic": "omega_squared",
                "flag_at_or_above": MINIMUM_OMEGA_SQUARED,
            },
        },
        "inputs": {
            "records": {"path": str(records_path), "sha256": file_sha256(records_path)},
            "pair_bucket_records": {"path": str(bucket_path), "sha256": file_sha256(bucket_path)},
            "pair_bucket_metadata": {"path": str(metadata_path), "sha256": file_sha256(metadata_path)},
            "auxiliary_mapping_manifest": {"path": str(auxiliary_path), "sha256": file_sha256(auxiliary_path)},
        },
        "summary": {
            "pair_buckets": len(entries),
            "records_in_pair_buckets": sum(item["record_count"] for item in entries.values()),
            "minimum_support_buckets": sum(item["minimum_support_met"] for item in entries.values()),
            "residual_heterogeneity_flagged_buckets": len(flagged_entries),
            "calibration_valid_buckets": len(valid_entries),
            "calibration_valid_records": sum(item["record_count"] for item in valid_entries),
            "calibration_reason_counts": dict(sorted(reasons.items())),
        },
        "buckets": entries,
    }
    if heldout_audit is not None:
        payload["heldout_exclusion"] = heldout_audit
    validate_pair_bucket_distance_calibration(payload, record_contract=record_contract)
    write_deterministic_gzip(target / CALIBRATION_FILENAME, payload)
    return payload


_CALIBRATION_CONTEXT: tuple[
    pd.DataFrame,
    Mapping[str, Any],
    TransferPolicyBuildSpec,
    StarlingRecordContract,
    int,
] | None = None


def _build_calibration_entries(
    rows: pd.DataFrame,
    *,
    spec: TransferPolicyBuildSpec,
    record_contract: StarlingRecordContract,
    minimum_samples: int,
    workers: int,
) -> dict[str, dict[str, Any]]:
    """Build independent bucket entries in fork workers, preserving key order."""
    grouped_indices = {
        str(key): indices
        for key, indices in rows.groupby("pair_bucket_key", sort=True).indices.items()
    }
    keys = sorted(grouped_indices)
    workers = min(max(1, int(workers or 1)), len(keys) or 1)
    global _CALIBRATION_CONTEXT
    _CALIBRATION_CONTEXT = (
        rows,
        grouped_indices,
        spec,
        record_contract,
        minimum_samples,
    )
    try:
        if workers == 1 or len(keys) < 100 or "fork" not in mp.get_all_start_methods():
            parts = [_build_calibration_partition(keys)]
        else:
            partitions = [keys[index::workers] for index in range(workers)]
            with mp.get_context("fork").Pool(processes=workers) as pool:
                parts = pool.map(_build_calibration_partition, partitions, chunksize=1)
        return {
            key: entry
            for key, entry in sorted(
                (item for part in parts for item in part), key=lambda item: item[0]
            )
        }
    finally:
        _CALIBRATION_CONTEXT = None


def _build_calibration_partition(
    keys: Sequence[str],
) -> list[tuple[str, dict[str, Any]]]:
    if _CALIBRATION_CONTEXT is None:
        raise RuntimeError("distance-calibration worker has no build context")
    rows, grouped_indices, spec, record_contract, minimum_samples = (
        _CALIBRATION_CONTEXT
    )
    return [
        (
            key,
            _build_calibration_entry(
                key,
                rows.iloc[grouped_indices[key]],
                spec=spec,
                record_contract=record_contract,
                minimum_samples=minimum_samples,
            ),
        )
        for key in keys
    ]


def _build_calibration_entry(
    key: str,
    group: pd.DataFrame,
    *,
    spec: TransferPolicyBuildSpec,
    record_contract: StarlingRecordContract,
    minimum_samples: int,
) -> dict[str, Any]:
    source_id = _one(group["source_id"], key, "source_id")
    kind = _one(group["measurement_kind"], key, "measurement_kind")
    scale_values = _unique_text(group["canonical_measurement_scale_id"])
    if len(scale_values) > 1:
        raise ValueError(f"pair bucket {key!r} spans measurement scales")
    scale_id = scale_values[0] if scale_values else None
    record_count = len(group)
    support_met = record_count >= minimum_samples
    category_gate = _category_gate(
        group,
        record_contract=record_contract,
        source_id=source_id,
        kind=kind,
        scale_id=scale_id,
    )
    geometry_values = _geometry_values(group, kind=kind)
    finite = pd.to_numeric(geometry_values, errors="coerce")
    finite_complete = bool(len(finite) and finite.notna().all())
    sample_sd = (
        float(finite.std(ddof=STANDARD_DEVIATION_DDOF))
        if finite_complete and len(finite) > 1
        else None
    )
    positive_sd = bool(
        sample_sd is not None and math.isfinite(sample_sd) and sample_sd > 0
    )
    if support_met:
        variance_gate = (
            _select_categorical_variance_candidate(
                group,
                record_contract=record_contract,
                source_id=source_id,
                scale_id=scale_id,
            )
            if kind in {"binary", "ordinal"}
            else select_variance_candidate(
                group,
                profile=spec.profile,
                source_id=source_id,
            )
        )
    else:
        variance_gate = _empty_variance_gate(evaluated=False)
    variance_flagged = bool(variance_gate["variance_gate_flagged"])

    calibration = None
    reason = "valid"
    if not support_met:
        reason = f"fewer_than_{minimum_samples}_records"
    elif kind not in {"continuous", "binary", "ordinal"}:
        reason = "unsupported_measurement_kind"
    elif not category_gate["valid"]:
        reason = str(category_gate["reason"])
    elif variance_flagged:
        reason = "automatic_residual_heterogeneity_gate"
    elif not finite_complete:
        reason = "nonfinite_geometry_value"
    elif not positive_sd:
        reason = "nonpositive_or_nonfinite_sample_sd"
    else:
        calibration = _distance_calibration(
            finite.tolist(),
            group["canonical_record_id"].astype(str).tolist(),
            pair_bucket_key=key,
        )
    return {
        "source_id": source_id,
        "measurement_kind": kind,
        "canonical_measurement_scale_id": scale_id,
        "record_count": record_count,
        "minimum_record_count": minimum_samples,
        "minimum_support_met": support_met,
        "category_domain_gate": category_gate,
        "residual_heterogeneity_gate": variance_gate,
        "observed_sample_standard_deviation": sample_sd if positive_sd else None,
        "calibration_valid": calibration is not None,
        "calibration_reason": reason,
        "distance_calibration": calibration,
    }


def _category_gate(
    group: pd.DataFrame,
    *,
    record_contract: StarlingRecordContract,
    source_id: str,
    kind: str,
    scale_id: str | None,
) -> dict[str, Any]:
    if kind == "continuous":
        if scale_id:
            spec = record_contract.measurement_scales.get(scale_id)
            if spec is None or spec.source_id != source_id or spec.kind != "continuous":
                return {"valid": False, "reason": "invalid_continuous_scale_declaration"}
        return {"valid": True, "reason": "not_categorical"}
    if kind not in {"binary", "ordinal"} or not scale_id:
        return {"valid": False, "reason": "missing_controlled_categorical_scale"}
    spec = record_contract.measurement_scales.get(scale_id)
    if spec is None or spec.source_id != source_id or spec.kind != kind:
        return {"valid": False, "reason": "categorical_scale_contract_mismatch"}
    declared = {item.category_id: item.rank for item in spec.categories}
    if (
        group["canonical_category_id"].isna().any()
        or group["canonical_category_rank"].isna().any()
    ):
        return {"valid": False, "reason": "missing_canonical_category"}
    observed_rows = [
        (str(category), int(rank))
        for category, rank in group[
            ["canonical_category_id", "canonical_category_rank"]
        ].itertuples(index=False)
    ]
    if any(declared.get(category) != rank for category, rank in observed_rows):
        return {"valid": False, "reason": "categorical_values_outside_declared_domain"}
    observed_pairs = {
        category: rank for category, rank in observed_rows
    }
    counts = group["canonical_category_id"].dropna().astype(str).value_counts()
    if set(observed_pairs) != set(counts.index):
        return {"valid": False, "reason": "categorical_values_outside_declared_domain"}
    if any(int(count) < MINIMUM_LEVEL_RECORDS for count in counts.values):
        return {"valid": False, "reason": "categorical_level_below_minimum_support"}
    if kind == "binary" and set(counts.index) != set(declared):
        return {"valid": False, "reason": "incomplete_binary_domain"}
    if kind == "ordinal" and len(counts) < 3:
        return {"valid": False, "reason": "fewer_than_three_ordinal_levels"}
    return {
        "valid": True,
        "reason": "complete_binary_domain" if kind == "binary" else "sufficient_ordinal_domain",
        "declared_category_ids": list(declared),
        "observed_category_counts": {
            str(key): int(value) for key, value in sorted(counts.items())
        },
    }


def _geometry_values(group: pd.DataFrame, *, kind: str) -> pd.Series:
    return (
        group["canonical_category_rank"]
        if kind in {"binary", "ordinal"}
        else group["finite_scalar_value"]
    )


def _select_categorical_variance_candidate(
    group: pd.DataFrame,
    *,
    record_contract: StarlingRecordContract,
    source_id: str,
    scale_id: str | None = None,
) -> dict[str, Any]:
    scale = record_contract.measurement_scales.get(str(scale_id or ""))
    controlled_inputs = set(scale.input_fields) if scale is not None else set()
    fields = tuple(
        field
        for field in record_contract.pair_buckets[source_id].variance_candidates
        if field not in controlled_inputs
    )
    scores = [
        score
        for field in fields
        if (score := _score_categorical_candidate(group, field)) is not None
    ]
    scores.sort(
        key=lambda item: (
            -item["candidate_score"],
            -item["cramers_v_squared"],
            -item["coverage"],
            item["candidate_column"],
        )
    )
    result = (
        {"evaluated": True, **scores[0]}
        if scores
        else _empty_variance_gate()
    )
    result["excluded_controlled_input_fields"] = sorted(controlled_inputs)
    return result


def _score_categorical_candidate(
    group: pd.DataFrame, candidate_field: str
) -> dict[str, Any] | None:
    outcomes = group["canonical_category_id"].astype("string")
    candidate = group[candidate_field].map(
        lambda value: normalize_candidate_value(candidate_field, value)
    )
    counts = candidate.dropna().value_counts(sort=False)
    supported_levels = {
        str(level) for level, count in counts.items() if int(count) >= MINIMUM_LEVEL_RECORDS
    }
    if len(supported_levels) < 2:
        return None
    supported = candidate.astype("string").isin(supported_levels) & outcomes.notna()
    supported_count = int(supported.sum())
    coverage = supported_count / len(group)
    if coverage < MINIMUM_COVERAGE:
        return None
    table = pd.crosstab(outcomes[supported], candidate[supported])
    if min(table.shape) < 2:
        return None
    v2 = _bias_corrected_cramers_v_squared(table.to_numpy(dtype=float))
    if v2 is None:
        return None
    return {
        "candidate_column": candidate_field,
        "candidate_score": float(v2 * coverage),
        "cramers_v_squared": float(v2),
        "coverage": float(coverage),
        "supported_level_count": len(supported_levels),
        "supported_record_count": supported_count,
        "variance_gate_flagged": v2 >= MINIMUM_CATEGORICAL_CRAMERS_V_SQUARED,
        "candidate_level_examples": [
            {"value": str(level), "record_count": int(counts[level])}
            for level in sorted(supported_levels)
        ][:8],
    }


def _bias_corrected_cramers_v_squared(table: np.ndarray) -> float | None:
    n = float(table.sum())
    rows, columns = table.shape
    if n <= 1 or rows < 2 or columns < 2:
        return None
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    if np.any(expected <= 0):
        return None
    chi_squared = float(np.sum(np.square(table - expected) / expected))
    phi_squared = chi_squared / n
    corrected_phi = max(
        0.0, phi_squared - ((columns - 1) * (rows - 1)) / (n - 1)
    )
    corrected_rows = rows - ((rows - 1) ** 2) / (n - 1)
    corrected_columns = columns - ((columns - 1) ** 2) / (n - 1)
    denominator = min(corrected_rows - 1, corrected_columns - 1)
    return corrected_phi / denominator if denominator > 0 else None


def _empty_variance_gate(*, evaluated: bool = True) -> dict[str, Any]:
    return {
        "evaluated": evaluated,
        "candidate_column": "__none__",
        "candidate_score": None,
        "cramers_v_squared": None,
        "omega_squared": None,
        "coverage": None,
        "supported_level_count": 0,
        "supported_record_count": 0,
        "variance_gate_flagged": False,
        "candidate_level_examples": [],
    }


def _distance_calibration(
    measurements: Sequence[Any],
    record_ids: Sequence[str],
    *,
    pair_bucket_key: str,
) -> dict[str, Any]:
    ordered = sorted(
        ((str(record_id), float(value)) for record_id, value in zip(record_ids, measurements)),
        key=lambda item: item[0],
    )
    values = np.asarray([value for _, value in ordered], dtype=float)
    sample_sd = float(np.std(values, ddof=STANDARD_DEVIATION_DDOF))
    total_pairs = len(values) * (len(values) - 1) // 2
    if total_pairs <= MAX_REFERENCE_PAIRS:
        differences = np.fromiter(
            (
                abs(float(values[left]) - float(values[right]))
                for left in range(len(values) - 1)
                for right in range(left + 1, len(values))
            ),
            dtype=float,
            count=total_pairs,
        )
        method = "all_unordered_record_pairs"
    else:
        differences = np.empty(MAX_REFERENCE_PAIRS, dtype=float)
        for pair_index in range(MAX_REFERENCE_PAIRS):
            left, right = _hashed_distinct_indices(pair_bucket_key, pair_index, len(values))
            differences[pair_index] = abs(float(values[left]) - float(values[right]))
        method = "deterministic_hash_sample_with_replacement"
    standardized = differences / sample_sd
    knots = np.quantile(
        standardized, np.linspace(0.0, 1.0, PERCENTILE_KNOTS), method="linear"
    )
    return {
        "sample_standard_deviation": sample_sd,
        "standard_deviation_ddof": STANDARD_DEVIATION_DDOF,
        "distance_definition": "abs(left - right) / sample_standard_deviation",
        "percentile_knot_count": PERCENTILE_KNOTS,
        "standardized_distance_percentile_knots": [float(value) for value in knots],
        "reference_method": method,
        "reference_pair_count": int(len(standardized)),
        "total_possible_unordered_pairs": total_pairs,
    }


def _hashed_distinct_indices(key: str, index: int, size: int) -> tuple[int, int]:
    digest = hashlib.sha256(
        f"{CALIBRATION_VERSION}\0{key}\0{index}".encode("utf-8")
    ).digest()
    left = int.from_bytes(digest[:8], "big") % size
    right_without_left = int.from_bytes(digest[8:16], "big") % (size - 1)
    right = right_without_left if right_without_left < left else right_without_left + 1
    return left, right


def _remove_heldout(
    spec: TransferPolicyBuildSpec, rows: pd.DataFrame
) -> dict[str, Any] | None:
    if not spec.profile.heldout_sources:
        return None
    if spec.heldout_key_loader is None:
        raise ValueError("heldout sources require a heldout key loader")
    heldout_keys = set(spec.heldout_key_loader())
    if not heldout_keys:
        raise ValueError("heldout key loader returned no identities")
    scoped = rows["source_id"].astype(str).isin(spec.profile.heldout_sources)

    def parent(value: Any) -> str:
        identity = normalize_molecule_identity(str(value or ""))
        return identity.parent_inchi_key or identity.parent_smiles

    keys = rows[spec.heldout_identity_column].map(parent)
    drop = scoped & keys.isin(heldout_keys)
    by_source = rows.loc[drop, "source_id"].astype(str).value_counts().to_dict()
    normalize_molecule_identity.cache_clear()
    return {
        "_drop_mask": drop,
        "sources": sorted(spec.profile.heldout_sources),
        "identity_column": spec.heldout_identity_column,
        "heldout_key_count": len(heldout_keys),
        "dropped_records_by_source": {
            str(key): int(value) for key, value in sorted(by_source.items())
        },
    }


def validate_pair_bucket_distance_calibration(
    payload: Mapping[str, Any], *, record_contract: StarlingRecordContract
) -> None:
    if payload.get("calibration_version") != CALIBRATION_VERSION:
        raise ValueError("distance calibration version mismatch")
    if payload.get("record_contract_version") != record_contract.version:
        raise ValueError("distance calibration record contract mismatch")
    buckets = payload.get("buckets")
    if not isinstance(buckets, Mapping):
        raise ValueError("distance calibration has no bucket mapping")
    forbidden = {
        "assay_transfer_eligible",
        "assay_transfer_label",
        "soft_transfer_probability",
        "transfer_max_standard_deviations",
    }
    encoded = json.dumps(payload, sort_keys=True)
    if any(f'"{field}"' in encoded for field in forbidden):
        raise ValueError("v7 calibration contains a forbidden transfer-policy field")
    for key, entry in buckets.items():
        if not isinstance(key, str) or not isinstance(entry, Mapping):
            raise ValueError("invalid distance-calibration entry")
        valid = bool(entry.get("calibration_valid"))
        calibration = entry.get("distance_calibration")
        if valid != isinstance(calibration, Mapping):
            raise ValueError(f"calibration validity mismatch for {key}")
        if valid:
            knots = calibration.get("standardized_distance_percentile_knots")
            if not isinstance(knots, list) or len(knots) != PERCENTILE_KNOTS:
                raise ValueError(f"invalid percentile knots for {key}")
            if any(float(right) < float(left) for left, right in zip(knots, knots[1:])):
                raise ValueError(f"nonmonotone percentile knots for {key}")


def _one(values: pd.Series, key: str, field: str) -> str:
    unique = _unique_text(values)
    if len(unique) != 1:
        raise ValueError(f"pair bucket {key!r} spans {field}: {unique}")
    return unique[0]


def _unique_text(values: pd.Series) -> list[str]:
    return sorted({str(value) for value in values if pd.notna(value) and str(value)})


__all__ = [
    "CALIBRATION_FILENAME",
    "CALIBRATION_VERSION",
    "MINIMUM_BUCKET_RECORDS",
    "build_pair_bucket_distance_calibration",
    "validate_pair_bucket_distance_calibration",
]
