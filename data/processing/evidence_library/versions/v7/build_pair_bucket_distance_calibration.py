"""Build v7 pair-bucket SD and empirical-CDF geometry without transfer targets.

Stage 3 defines source-specific bucket membership and measurement kind, then
deduplicates records and refines direct-residual membership. This module only
validates whether the resulting bucket can support a distance scale and stores a
first-class sample SD, an exact value CDF for valid continuous buckets, and an
exact category-rank CDF for valid ordinal buckets. It never emits a pair label,
transfer threshold, soft target, or raw-distance CDF.
"""

from __future__ import annotations

import bisect
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
from data.processing.evidence_library.shared.v1.build_pair_bucket_transfer_policy import (
    TransferPolicyBuildSpec,
    _validate_global_context_contract,
    write_deterministic_gzip,
)
from data.processing.evidence_library.shared.v1.canonicalization_v7 import (
    StarlingRecordContract,
)
from data.processing.evidence_library.shared.v1.categorical_response import (
    ControlledMeasurementSpec,
)
from data.processing.evidence_library.shared.v1.normalization.cleaning import file_sha256
from data.processing.evidence_library.shared.v1.pair_bucket_transfer_policy import (
    MINIMUM_LEVEL_RECORDS,
    STANDARD_DEVIATION_DDOF,
)


CALIBRATION_FILENAME = "pair_bucket_distance_calibration.json.gz"
CALIBRATION_VERSION = "pair_bucket_distance_calibration.v6"
V5_CALIBRATION_VERSION = "pair_bucket_distance_calibration.v5"
PREVIOUS_CALIBRATION_VERSION = "pair_bucket_distance_calibration.v4"
V3_CALIBRATION_VERSION = "pair_bucket_distance_calibration.v3"
V2_CALIBRATION_VERSION = "pair_bucket_distance_calibration.v2"
LEGACY_CALIBRATION_VERSION = "pair_bucket_distance_calibration.v1"
VALUE_CDF_VERSION = "empirical_value_cdf.v1"
CATEGORY_CDF_VERSION = "empirical_category_rank_cdf.v1"
LEGACY_DISTANCE_PERCENTILE_KNOTS = 101
MINIMUM_BUCKET_RECORDS = 20
MINIMUM_BUCKET_MOLECULES = 16
LEGACY_MINIMUM_BUCKET_RECORDS = 25


def build_pair_bucket_distance_calibration(
    *,
    spec: TransferPolicyBuildSpec,
    record_contract: StarlingRecordContract,
    records_path: str | Path,
    pair_bucket_records_path: str | Path,
    pair_bucket_metadata_path: str | Path,
    auxiliary_manifest_path: str | Path,
    out_dir: str | Path,
    minimum_samples: int | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Materialize SD and empirical CDF metadata per deduplicated Stage-3 bucket."""
    records_path = Path(records_path)
    bucket_path = Path(pair_bucket_records_path)
    metadata_path = Path(pair_bucket_metadata_path)
    auxiliary_path = Path(auxiliary_manifest_path)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    bucket_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    auxiliary_metadata = json.loads(auxiliary_path.read_text(encoding="utf-8"))
    _validate_global_context_contract(spec, bucket_metadata, auxiliary_metadata)

    schema = set(pq.read_schema(records_path).names)
    required = {
        "canonical_record_id",
        "canonical_smiles",
        "finite_scalar_value",
        "measurement_kind",
        "canonical_measurement_scale_id",
        "canonical_category_id",
        "canonical_category_rank",
    }
    if spec.profile.heldout_sources:
        required.add(spec.heldout_identity_column)
    expected_minimum = MINIMUM_BUCKET_RECORDS
    if minimum_samples is None:
        minimum_samples = expected_minimum
    if minimum_samples != expected_minimum:
        raise ValueError(
            f"the v7 calibration contract requires exactly {expected_minimum} "
            "deduplicated records"
        )
    finalized = {
        "source_id",
        "pair_bucket_key",
        "assay_transfer_eligible",
    }.issubset(schema)
    if finalized:
        required.update(
            {
                "retrieval_source_id",
                "pair_bucket_key",
                "assay_transfer_eligible",
                "source_id",
            }
        )
    missing = required - schema
    if missing:
        raise ValueError(f"v7 calibration records lack columns: {sorted(missing)}")
    records = pd.read_parquet(records_path, columns=sorted(required))
    if not records["canonical_record_id"].is_unique:
        raise ValueError("finalized canonical_record_id values must be unique")
    if finalized:
        rows = records[
            records["assay_transfer_eligible"].astype(bool)
            & records["pair_bucket_key"].notna()
        ].copy()
    else:
        buckets = pd.read_parquet(bucket_path)
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
        eligibility_field = (
            "assay_transfer_eligible"
            if "assay_transfer_eligible" in joined.columns
            else "bucket_eligible"
        )
        rows = joined[
            joined[eligibility_field].astype(bool) & joined["pair_bucket_key"].notna()
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
    value_cdf_entries = [entry for entry in entries.values() if entry["value_cdf_valid"]]
    category_cdf_entries = [
        entry for entry in entries.values() if entry["category_cdf_valid"]
    ]
    reasons = Counter(str(entry["calibration_reason"]) for entry in entries.values())
    value_cdf_reasons = Counter(
        str(entry["value_cdf_reason"]) for entry in entries.values()
    )
    category_cdf_reasons = Counter(
        str(entry["category_cdf_reason"]) for entry in entries.values()
    )
    payload: dict[str, Any] = {
        "calibration_version": CALIBRATION_VERSION,
        "record_contract_version": record_contract.version,
        "pair_bucket_version": spec.pair_bucket_version,
        "semantics": {
            "pair_bucket_membership_authority": (
                "stage3_source_bucket_with_direct_condition_refinement"
            ),
            "calibration_changes_membership": False,
            "pair_labels_emitted": False,
            "transfer_cutoff_emitted": False,
            "transfer_probability_emitted": False,
            "standard_deviation": "first_class_per_pair_bucket_sample_sd",
            "standard_deviation_ddof": STANDARD_DEVIATION_DDOF,
            "continuous_standard_deviation_value_field": "finite_scalar_value",
            "binary_and_ordinal_standard_deviation_value_field": (
                "canonical_category_rank"
            ),
            "canonical_measurement_scope": "deduplicated_source_records",
            "assay_transfer_bucket_eligibility_scope": (
                "global_stage3_before_paper_views"
            ),
            "calibration_record_unit": "deduplicated_record",
            "raw_distance_cdf_emitted": False,
            "continuous_value_cdf": (
                "exact empirical midrank CDF over finite_scalar_value"
            ),
            "value_cdf_scope": "within_pair_bucket",
            "value_cdf_fit_scope": "same_records_as_standard_deviation",
            "value_cdf_measurement_kinds": ["continuous"],
            "value_cdf_tie_convention": "midrank",
            "value_cdf_unseen_value_rule": "records_strictly_less_than_value / total_records",
            "value_cdf_outside_support": "saturate_to_zero_or_one",
            "ordinal_category_cdf": (
                "exact empirical midrank CDF over canonical_category_rank"
            ),
            "category_cdf_scope": "within_pair_bucket",
            "category_cdf_measurement_kinds": ["ordinal"],
            "category_cdf_tie_convention": "midrank",
            "binary_category_cdf_emitted": False,
        },
        "measurement_scales": {
            key: value.manifest()
            for key, value in sorted(record_contract.measurement_scales.items())
        },
        "validation_contract": {
            "minimum_bucket_records": minimum_samples,
            "minimum_bucket_records_unit": "deduplicated_records",
            "minimum_distinct_molecules": MINIMUM_BUCKET_MOLECULES,
            "binary": "all declared levels observed; every observed level has >=3 records",
            "ordinal": (
                "at least three declared levels observed; every observed level "
                "has >=3 records"
            ),
        },
        "inputs": {
            "records": {"path": str(records_path), "sha256": file_sha256(records_path)},
            "pair_bucket_records": {"path": str(bucket_path), "sha256": file_sha256(bucket_path)},
            "pair_bucket_metadata": {
                "path": str(metadata_path),
                "sha256": file_sha256(metadata_path),
            },
            "auxiliary_mapping_manifest": {
                "path": str(auxiliary_path),
                "sha256": file_sha256(auxiliary_path),
            },
        },
        "summary": {
            "pair_buckets": len(entries),
            "records_in_pair_buckets": sum(item["record_count"] for item in entries.values()),
            "minimum_support_buckets": sum(
                item["minimum_support_met"] for item in entries.values()
            ),
            "calibration_valid_buckets": len(valid_entries),
            "calibration_valid_records": sum(item["record_count"] for item in valid_entries),
            "calibration_reason_counts": dict(sorted(reasons.items())),
            "value_cdf_valid_buckets": len(value_cdf_entries),
            "value_cdf_valid_records": sum(
                item["record_count"] for item in value_cdf_entries
            ),
            "value_cdf_distinct_support_points": sum(
                item["value_cdf"]["distinct_value_count"]
                for item in value_cdf_entries
            ),
            "value_cdf_reason_counts": dict(sorted(value_cdf_reasons.items())),
            "category_cdf_valid_buckets": len(category_cdf_entries),
            "category_cdf_valid_records": sum(
                item["record_count"] for item in category_cdf_entries
            ),
            "category_cdf_reason_counts": dict(
                sorted(category_cdf_reasons.items())
            ),
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
    source_id = _calibration_source_id(
        group["source_id"], key=key, record_contract=record_contract
    )
    kind = _one(group["measurement_kind"], key, "measurement_kind")
    scale_values = _unique_text(group["canonical_measurement_scale_id"])
    if len(scale_values) > 1:
        raise ValueError(f"pair bucket {key!r} spans measurement scales")
    scale_id = scale_values[0] if scale_values else None
    record_count = len(group)
    molecule_count = int(group["canonical_smiles"].dropna().astype(str).nunique())
    record_support_met = record_count >= minimum_samples
    molecule_support_met = molecule_count >= MINIMUM_BUCKET_MOLECULES
    support_met = record_support_met and molecule_support_met
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
    reason = "valid"
    if not record_support_met:
        reason = f"fewer_than_{minimum_samples}_records"
    elif not molecule_support_met:
        reason = f"fewer_than_{MINIMUM_BUCKET_MOLECULES}_distinct_molecules"
    elif kind not in {"continuous", "binary", "ordinal"}:
        reason = "unsupported_measurement_kind"
    elif not category_gate["valid"]:
        reason = str(category_gate["reason"])
    elif not finite_complete:
        reason = "nonfinite_geometry_value"
    elif not positive_sd:
        reason = "nonpositive_or_nonfinite_sample_sd"
    calibration_valid = reason == "valid"
    value_cdf = None
    if kind != "continuous":
        value_cdf_reason = "non_continuous_measurement"
    elif not calibration_valid:
        value_cdf_reason = reason
    else:
        value_cdf = _value_cdf(finite.tolist())
        value_cdf_reason = "valid"
    category_cdf = None
    if kind != "ordinal":
        category_cdf_reason = "non_ordinal_measurement"
    elif not calibration_valid:
        category_cdf_reason = reason
    else:
        scale = record_contract.measurement_scales.get(str(scale_id or ""))
        if scale is None:
            raise ValueError(f"valid ordinal bucket {key!r} has no declared scale")
        category_cdf = _category_cdf(group, scale=scale)
        category_cdf_reason = "valid"
    return {
        "source_id": source_id,
        "measurement_kind": kind,
        "canonical_measurement_scale_id": scale_id,
        "record_count": record_count,
        "distinct_molecule_count": molecule_count,
        "minimum_record_count": minimum_samples,
        "minimum_distinct_molecule_count": MINIMUM_BUCKET_MOLECULES,
        "minimum_record_support_met": record_support_met,
        "minimum_molecule_support_met": molecule_support_met,
        "minimum_support_met": support_met,
        "category_domain_gate": category_gate,
        "observed_sample_standard_deviation": (
            sample_sd if calibration_valid else None
        ),
        "standard_deviation_ddof": STANDARD_DEVIATION_DDOF,
        "standard_deviation_value_field": (
            "canonical_category_rank"
            if kind in {"binary", "ordinal"}
            else "finite_scalar_value"
        ),
        "standard_deviation_valid": calibration_valid,
        "standard_deviation_reason": reason,
        "assay_transfer_bucket_eligible": calibration_valid,
        "automatic_transfer_eligible": calibration_valid,
        "pruning_status": "unreviewed",
        "assay_transfer_bucket_ineligibility_reason": (
            None if calibration_valid else reason
        ),
        "calibration_valid": calibration_valid,
        "calibration_reason": reason,
        "value_cdf_valid": value_cdf is not None,
        "value_cdf_reason": value_cdf_reason,
        "value_cdf": value_cdf,
        "category_cdf_valid": category_cdf is not None,
        "category_cdf_reason": category_cdf_reason,
        "category_cdf": category_cdf,
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


def _value_cdf(measurements: Sequence[Any]) -> dict[str, Any]:
    """Build an exact tie-aware empirical CDF for one continuous bucket."""
    values = np.asarray([float(value) for value in measurements], dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("value CDF requires finite measurements")
    support, counts = np.unique(values, return_counts=True)
    cumulative_before = np.cumsum(counts, dtype=np.int64) - counts
    midranks = (cumulative_before + 0.5 * counts) / len(values)
    return {
        "contract_version": VALUE_CDF_VERSION,
        "measurement_field": "finite_scalar_value",
        "definition": "(records_less_than_value + 0.5 * records_equal_to_value) / total_records",
        "tie_convention": "midrank",
        "unseen_value_rule": "records_strictly_less_than_value / total_records",
        "tie_equality": "exact_canonical_float_equality",
        "outside_support": "saturate_to_zero_or_one",
        "total_record_count": int(len(values)),
        "distinct_value_count": int(len(support)),
        "support_values": [float(value) for value in support],
        "support_counts": [int(value) for value in counts],
        "support_midranks_0_1": [float(value) for value in midranks],
    }


def _category_cdf(
    group: pd.DataFrame, *, scale: ControlledMeasurementSpec
) -> dict[str, Any]:
    """Build an exact empirical CDF over one declared ordinal domain."""
    if scale.kind != "ordinal":
        raise ValueError("category CDF requires an ordinal measurement scale")
    observed = Counter(group["canonical_category_id"].dropna().astype(str))
    declared = sorted(scale.categories, key=lambda item: item.rank)
    declared_ids = {item.category_id for item in declared}
    if set(observed) - declared_ids:
        raise ValueError("category CDF contains values outside its declared domain")
    total = len(group)
    if total <= 0 or sum(observed.values()) != total:
        raise ValueError("category CDF requires one declared category per record")
    cumulative = 0
    categories: list[dict[str, Any]] = []
    for category in declared:
        count = int(observed[category.category_id])
        midrank = (cumulative + 0.5 * count) / total
        categories.append(
            {
                "category_id": category.category_id,
                "rank": category.rank,
                "observed_count": count,
                "midrank_0_1": float(midrank),
            }
        )
        cumulative += count
    return {
        "contract_version": CATEGORY_CDF_VERSION,
        "measurement_field": "canonical_category_rank",
        "category_identity_field": "canonical_category_id",
        "canonical_measurement_scale_id": scale.scale_id,
        "definition": (
            "(records_with_lower_rank + 0.5 * records_with_equal_rank) "
            "/ total_records"
        ),
        "tie_convention": "midrank",
        "unobserved_declared_category_rule": "records_with_lower_rank / total_records",
        "total_record_count": total,
        "declared_category_count": len(declared),
        "observed_category_count": sum(count > 0 for count in observed.values()),
        "categories": categories,
    }


def value_cdf_percentile(value: Any, value_cdf: Mapping[str, Any]) -> float:
    """Evaluate a stored exact-midrank CDF with deterministic step lookup."""
    measurement = float(value)
    if not math.isfinite(measurement):
        raise ValueError("value-CDF measurement must be finite")
    if value_cdf.get("contract_version") != VALUE_CDF_VERSION:
        raise ValueError("unsupported value-CDF contract")
    if value_cdf.get("measurement_field") != "finite_scalar_value":
        raise ValueError("value CDF has an unsupported measurement field")
    support = [float(item) for item in value_cdf.get("support_values") or []]
    counts = [int(item) for item in value_cdf.get("support_counts") or []]
    midranks = [
        float(item) for item in value_cdf.get("support_midranks_0_1") or []
    ]
    total = int(value_cdf.get("total_record_count") or 0)
    if not support or len(support) != len(counts) or len(support) != len(midranks):
        raise ValueError("value CDF has invalid support arrays")
    if total <= 0 or total != sum(counts) or any(count <= 0 for count in counts):
        raise ValueError("value CDF has invalid support counts")
    if int(value_cdf.get("distinct_value_count") or 0) != len(support):
        raise ValueError("value CDF has an invalid distinct-value count")
    if any(not math.isfinite(item) for item in support) or any(
        right <= left for left, right in zip(support, support[1:])
    ):
        raise ValueError("value CDF has invalid support values")
    cumulative = 0
    for count, observed in zip(counts, midranks):
        expected = (cumulative + 0.5 * count) / total
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("value CDF has invalid support midranks")
        cumulative += count
    index = bisect.bisect_left(support, measurement)
    if index < len(support) and support[index] == measurement:
        return midranks[index]
    if index == 0:
        return 0.0
    if index == len(support):
        return 1.0
    return midranks[index - 1] + counts[index - 1] / (2.0 * total)


def category_cdf_percentile(
    category_id: str, category_cdf: Mapping[str, Any]
) -> float:
    """Return the stored empirical percentile for one declared category ID."""
    if category_cdf.get("contract_version") != CATEGORY_CDF_VERSION:
        raise ValueError("unsupported category-CDF contract")
    if category_cdf.get("measurement_field") != "canonical_category_rank":
        raise ValueError("category CDF has an unsupported measurement field")
    categories = category_cdf.get("categories")
    if not isinstance(categories, list):
        raise ValueError("category CDF has no declared categories")
    matches = [
        item
        for item in categories
        if isinstance(item, Mapping) and item.get("category_id") == str(category_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown category_id: {category_id}")
    percentile = float(matches[0].get("midrank_0_1"))
    if not math.isfinite(percentile) or not 0.0 <= percentile <= 1.0:
        raise ValueError("category CDF has an invalid midrank")
    return percentile


def calibration_standard_deviation(entry: Mapping[str, Any]) -> float:
    """Read one valid v2 bucket's first-class sample SD."""
    if not bool(entry.get("standard_deviation_valid")):
        reason = str(entry.get("standard_deviation_reason") or "invalid")
        raise ValueError(f"pair bucket has no valid standard deviation: {reason}")
    if int(entry.get("standard_deviation_ddof") or -1) != STANDARD_DEVIATION_DDOF:
        raise ValueError("pair bucket has an unsupported standard-deviation ddof")
    try:
        value = float(entry.get("observed_sample_standard_deviation"))
    except (TypeError, ValueError) as error:
        raise ValueError("pair bucket has an invalid standard deviation") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError("pair bucket has an invalid standard deviation")
    return value


def value_cdf_separation(
    calibration: Mapping[str, Any],
    *,
    left_pair_bucket_key: str,
    left_value: Any,
    right_pair_bucket_key: str,
    right_value: Any,
) -> dict[str, float]:
    """Evaluate location-sensitive separation for two same-bucket values."""
    left_key = str(left_pair_bucket_key)
    right_key = str(right_pair_bucket_key)
    if left_key != right_key:
        raise ValueError("value-CDF comparison requires the same pair_bucket_key")
    buckets = calibration.get("buckets")
    entry = buckets.get(left_key) if isinstance(buckets, Mapping) else None
    if not isinstance(entry, Mapping):
        raise ValueError(f"unknown pair_bucket_key: {left_key}")
    if not bool(entry.get("value_cdf_valid")) or not isinstance(
        entry.get("value_cdf"), Mapping
    ):
        reason = str(entry.get("value_cdf_reason") or "invalid")
        raise ValueError(f"pair bucket has no valid value CDF: {reason}")
    left = value_cdf_percentile(left_value, entry["value_cdf"])
    right = value_cdf_percentile(right_value, entry["value_cdf"])
    return {
        "left_percentile_0_1": left,
        "right_percentile_0_1": right,
        "percentile_separation_0_1": abs(left - right),
    }


def category_cdf_separation(
    calibration: Mapping[str, Any],
    *,
    left_pair_bucket_key: str,
    left_category_id: str,
    right_pair_bucket_key: str,
    right_category_id: str,
) -> dict[str, float]:
    """Evaluate empirical rank separation for two same-bucket categories."""
    left_key = str(left_pair_bucket_key)
    right_key = str(right_pair_bucket_key)
    if left_key != right_key:
        raise ValueError("category-CDF comparison requires the same pair_bucket_key")
    buckets = calibration.get("buckets")
    entry = buckets.get(left_key) if isinstance(buckets, Mapping) else None
    if not isinstance(entry, Mapping):
        raise ValueError(f"unknown pair_bucket_key: {left_key}")
    if not bool(entry.get("category_cdf_valid")) or not isinstance(
        entry.get("category_cdf"), Mapping
    ):
        reason = str(entry.get("category_cdf_reason") or "invalid")
        raise ValueError(f"pair bucket has no valid category CDF: {reason}")
    left = category_cdf_percentile(left_category_id, entry["category_cdf"])
    right = category_cdf_percentile(right_category_id, entry["category_cdf"])
    return {
        "left_percentile_0_1": left,
        "right_percentile_0_1": right,
        "percentile_separation_0_1": abs(left - right),
    }


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
    version = payload.get("calibration_version")
    if version not in {
        LEGACY_CALIBRATION_VERSION,
        V2_CALIBRATION_VERSION,
        V3_CALIBRATION_VERSION,
        PREVIOUS_CALIBRATION_VERSION,
        V5_CALIBRATION_VERSION,
        CALIBRATION_VERSION,
    }:
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
        if version == LEGACY_CALIBRATION_VERSION:
            calibration = entry.get("distance_calibration")
            if valid != isinstance(calibration, Mapping):
                raise ValueError(f"calibration validity mismatch for {key}")
            if valid:
                knots = calibration.get("standardized_distance_percentile_knots")
                if (
                    not isinstance(knots, list)
                    or len(knots) != LEGACY_DISTANCE_PERCENTILE_KNOTS
                ):
                    raise ValueError(f"invalid percentile knots for {key}")
                if any(
                    float(right) < float(left)
                    for left, right in zip(knots, knots[1:])
                ):
                    raise ValueError(f"nonmonotone percentile knots for {key}")
        else:
            _validate_standard_deviation_entry(
                str(key),
                entry,
                require_bucket_eligibility=version
                in {
                    V3_CALIBRATION_VERSION,
                    PREVIOUS_CALIBRATION_VERSION,
                    V5_CALIBRATION_VERSION,
                    CALIBRATION_VERSION,
                },
            )
            _validate_value_cdf_entry(str(key), entry)
            _validate_category_cdf_entry(
                str(key), entry, record_contract=record_contract
            )

    if version == CALIBRATION_VERSION:
        raw_distance_fields = {
            "distance_calibration",
            "standardized_distance_percentile_knots",
            "percentile_knot_count",
            "reference_method",
            "reference_pair_count",
            "total_possible_unordered_pairs",
        }
        if any(f'"{field}"' in encoded for field in raw_distance_fields):
            raise ValueError("v2 calibration contains forbidden raw-distance CDF data")


def _validate_standard_deviation_entry(
    key: str,
    entry: Mapping[str, Any],
    *,
    require_bucket_eligibility: bool = False,
) -> None:
    valid = bool(entry.get("calibration_valid"))
    if valid != (entry.get("calibration_reason") == "valid"):
        raise ValueError(f"calibration validity/reason mismatch for {key}")
    if bool(entry.get("standard_deviation_valid")) != valid:
        raise ValueError(f"standard-deviation validity mismatch for {key}")
    if require_bucket_eligibility:
        if bool(entry.get("assay_transfer_bucket_eligible")) != valid:
            raise ValueError(f"assay-transfer bucket eligibility mismatch for {key}")
        expected = None if valid else entry.get("calibration_reason")
        if entry.get("assay_transfer_bucket_ineligibility_reason") != expected:
            raise ValueError(f"assay-transfer bucket reason mismatch for {key}")
    if entry.get("standard_deviation_reason") != entry.get("calibration_reason"):
        raise ValueError(f"standard-deviation reason mismatch for {key}")
    if int(entry.get("standard_deviation_ddof") or -1) != STANDARD_DEVIATION_DDOF:
        raise ValueError(f"standard-deviation ddof mismatch for {key}")
    kind = str(entry.get("measurement_kind") or "")
    expected_field = (
        "canonical_category_rank"
        if kind in {"binary", "ordinal"}
        else "finite_scalar_value"
    )
    if entry.get("standard_deviation_value_field") != expected_field:
        raise ValueError(f"standard-deviation value field mismatch for {key}")
    raw_sd = entry.get("observed_sample_standard_deviation")
    if valid:
        calibration_standard_deviation(entry)
    elif raw_sd is not None:
        raise ValueError(f"invalid bucket exposes a standard deviation for {key}")


def _validate_value_cdf_entry(key: str, entry: Mapping[str, Any]) -> None:
    kind = str(entry.get("measurement_kind") or "")
    calibration_valid = bool(entry.get("calibration_valid"))
    value_cdf_valid = bool(entry.get("value_cdf_valid"))
    value_cdf = entry.get("value_cdf")
    expected_valid = kind == "continuous" and calibration_valid
    if value_cdf_valid != expected_valid or value_cdf_valid != isinstance(
        value_cdf, Mapping
    ):
        raise ValueError(f"value-CDF validity mismatch for {key}")
    expected_reason = (
        "valid"
        if expected_valid
        else (
            "non_continuous_measurement"
            if kind != "continuous"
            else str(entry.get("calibration_reason") or "")
        )
    )
    if entry.get("value_cdf_reason") != expected_reason:
        raise ValueError(f"value-CDF reason mismatch for {key}")
    if not expected_valid:
        return
    if value_cdf.get("contract_version") != VALUE_CDF_VERSION:
        raise ValueError(f"value-CDF contract mismatch for {key}")
    expected_contract = {
        "measurement_field": "finite_scalar_value",
        "tie_convention": "midrank",
        "unseen_value_rule": (
            "records_strictly_less_than_value / total_records"
        ),
        "tie_equality": "exact_canonical_float_equality",
        "outside_support": "saturate_to_zero_or_one",
    }
    for field, expected in expected_contract.items():
        if value_cdf.get(field) != expected:
            raise ValueError(f"value-CDF {field} mismatch for {key}")
    support = [float(item) for item in value_cdf.get("support_values") or []]
    counts = [int(item) for item in value_cdf.get("support_counts") or []]
    midranks = [
        float(item) for item in value_cdf.get("support_midranks_0_1") or []
    ]
    if not support or not (len(support) == len(counts) == len(midranks)):
        raise ValueError(f"invalid value-CDF support arrays for {key}")
    if any(not math.isfinite(value) for value in support) or any(
        right <= left for left, right in zip(support, support[1:])
    ):
        raise ValueError(f"non-increasing value-CDF support for {key}")
    if any(count <= 0 for count in counts):
        raise ValueError(f"nonpositive value-CDF count for {key}")
    total = int(value_cdf.get("total_record_count") or 0)
    if total != sum(counts) or total != int(entry.get("record_count") or 0):
        raise ValueError(f"value-CDF record count mismatch for {key}")
    if int(value_cdf.get("distinct_value_count") or 0) != len(support):
        raise ValueError(f"value-CDF distinct count mismatch for {key}")
    cumulative = 0
    for count, observed in zip(counts, midranks):
        expected = (cumulative + 0.5 * count) / total
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"value-CDF midrank mismatch for {key}")
        cumulative += count


def _validate_category_cdf_entry(
    key: str,
    entry: Mapping[str, Any],
    *,
    record_contract: StarlingRecordContract,
) -> None:
    kind = str(entry.get("measurement_kind") or "")
    calibration_valid = bool(entry.get("calibration_valid"))
    category_cdf_valid = bool(entry.get("category_cdf_valid"))
    category_cdf = entry.get("category_cdf")
    expected_valid = kind == "ordinal" and calibration_valid
    if category_cdf_valid != expected_valid or category_cdf_valid != isinstance(
        category_cdf, Mapping
    ):
        raise ValueError(f"category-CDF validity mismatch for {key}")
    expected_reason = (
        "valid"
        if expected_valid
        else (
            "non_ordinal_measurement"
            if kind != "ordinal"
            else str(entry.get("calibration_reason") or "")
        )
    )
    if entry.get("category_cdf_reason") != expected_reason:
        raise ValueError(f"category-CDF reason mismatch for {key}")
    if not expected_valid:
        return
    if category_cdf.get("contract_version") != CATEGORY_CDF_VERSION:
        raise ValueError(f"category-CDF contract mismatch for {key}")
    expected_contract = {
        "measurement_field": "canonical_category_rank",
        "category_identity_field": "canonical_category_id",
        "tie_convention": "midrank",
        "unobserved_declared_category_rule": (
            "records_with_lower_rank / total_records"
        ),
    }
    for field, expected in expected_contract.items():
        if category_cdf.get(field) != expected:
            raise ValueError(f"category-CDF {field} mismatch for {key}")
    scale_id = str(entry.get("canonical_measurement_scale_id") or "")
    if category_cdf.get("canonical_measurement_scale_id") != scale_id:
        raise ValueError(f"category-CDF scale mismatch for {key}")
    scale = record_contract.measurement_scales.get(scale_id)
    if scale is None or scale.kind != "ordinal":
        raise ValueError(f"category-CDF undeclared ordinal scale for {key}")
    categories = category_cdf.get("categories")
    if not isinstance(categories, list):
        raise ValueError(f"category-CDF categories missing for {key}")
    declared = sorted(scale.categories, key=lambda item: item.rank)
    if len(categories) != len(declared):
        raise ValueError(f"category-CDF declared category count mismatch for {key}")
    if int(category_cdf.get("declared_category_count") or 0) != len(declared):
        raise ValueError(f"category-CDF declared category count mismatch for {key}")
    total = int(category_cdf.get("total_record_count") or 0)
    if total != int(entry.get("record_count") or 0):
        raise ValueError(f"category-CDF record count mismatch for {key}")
    cumulative = 0
    observed_categories = 0
    for stored, expected in zip(categories, declared):
        if not isinstance(stored, Mapping):
            raise ValueError(f"invalid category-CDF category for {key}")
        if stored.get("category_id") != expected.category_id or int(
            stored.get("rank", -1)
        ) != expected.rank:
            raise ValueError(f"category-CDF declared domain mismatch for {key}")
        count = int(stored.get("observed_count", -1))
        if count < 0:
            raise ValueError(f"category-CDF negative count for {key}")
        observed = float(stored.get("midrank_0_1"))
        expected_midrank = (cumulative + 0.5 * count) / total
        if not math.isclose(
            observed, expected_midrank, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError(f"category-CDF midrank mismatch for {key}")
        cumulative += count
        observed_categories += int(count > 0)
    if cumulative != total:
        raise ValueError(f"category-CDF record count mismatch for {key}")
    if int(category_cdf.get("observed_category_count") or 0) != observed_categories:
        raise ValueError(f"category-CDF observed category count mismatch for {key}")


def _one(values: pd.Series, key: str, field: str) -> str:
    unique = _unique_text(values)
    if len(unique) != 1:
        raise ValueError(f"pair bucket {key!r} spans {field}: {unique}")
    return unique[0]


def _calibration_source_id(
    values: pd.Series,
    *,
    key: str,
    record_contract: StarlingRecordContract,
) -> str:
    """Resolve the scientific source while preserving multi-source lineage."""
    observed = _unique_text(values)
    if len(observed) == 1 and observed[0] != "multi_source":
        return observed[0]
    try:
        bucket_source = str(json.loads(key)[0])
    except (IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            f"pair bucket {key!r} spans source_id: {observed}"
        ) from error
    if bucket_source not in record_contract.sources or any(
        source not in {bucket_source, "multi_source"} for source in observed
    ):
        raise ValueError(f"pair bucket {key!r} spans source_id: {observed}")
    return bucket_source


def _unique_text(values: pd.Series) -> list[str]:
    return sorted({str(value) for value in values if pd.notna(value) and str(value)})


__all__ = [
    "CATEGORY_CDF_VERSION",
    "CALIBRATION_FILENAME",
    "CALIBRATION_VERSION",
    "LEGACY_CALIBRATION_VERSION",
    "LEGACY_DISTANCE_PERCENTILE_KNOTS",
    "LEGACY_MINIMUM_BUCKET_RECORDS",
    "MINIMUM_BUCKET_MOLECULES",
    "MINIMUM_BUCKET_RECORDS",
    "VALUE_CDF_VERSION",
    "build_pair_bucket_distance_calibration",
    "calibration_standard_deviation",
    "category_cdf_percentile",
    "category_cdf_separation",
    "value_cdf_percentile",
    "value_cdf_separation",
    "validate_pair_bucket_distance_calibration",
]
