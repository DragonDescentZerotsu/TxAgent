"""Build one task's pair-bucket assay-transfer policy.

This downstream builder reads finalized normalized records and immutable
pair-bucket membership.  It does not rewrite records, create refined buckets,
or call an LLM.

Everything task-specific arrives through a :class:`TransferPolicyBuildSpec`.
Features added after a task froze its artifact (the soft transfer target and
distinct-level guard) are opt-in, so enabling them for one task never perturbs
another task's frozen policy.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v1.normalization.cleaning import file_sha256
from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
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
    STANDARD_DEVIATION_DDOF,
    TRANSFER_MAX_STANDARD_DEVIATIONS,
    TransferPolicyProfile,
    build_distance_policy,
    select_variance_candidate,
    soft_transfer_contract,
    validate_pair_bucket_transfer_policy,
)


POLICY_FILENAME = "pair_bucket_transfer_policy.json.gz"


@dataclass(frozen=True)
class TransferPolicyBuildSpec:
    """Everything the shared builder needs that is specific to one task."""

    profile: TransferPolicyProfile
    pair_bucket_version: str
    source_pair_fields: Mapping[str, tuple[str, ...]]
    auxiliary_mapping_version: str
    auxiliary_attachment_version: str
    # Auxiliary fields this task's buckets actually depend on.  Checked as a
    # subset so a task may reconcile more than it strata-fies on.
    required_auxiliary_output_fields: tuple[str, ...] = (
        "global_context",
        "global_species_context",
    )
    endpoint_field_by_source: Mapping[str, str] | None = None
    # Opt-in additions.  Leaving these at their defaults reproduces the payload
    # exactly as it was before they existed.
    include_soft_transfer_contract: bool = False
    heldout_key_loader: Callable[[], set[str]] | None = None
    heldout_identity_column: str = "canonical_smiles"


def build_pair_bucket_transfer_policy(
    *,
    spec: TransferPolicyBuildSpec,
    records_path: str | Path,
    pair_bucket_records_path: str | Path,
    pair_bucket_metadata_path: str | Path,
    auxiliary_manifest_path: str | Path,
    out_dir: str | Path,
    minimum_samples: int = MIN_ASSAY_TRANSFER_SAMPLES,
) -> dict[str, Any]:
    """Materialize one eligibility and distance entry per existing pair bucket."""
    if minimum_samples != MIN_ASSAY_TRANSFER_SAMPLES:
        raise ValueError(
            "the frozen transfer-policy contract requires exactly 25 records"
        )
    profile = spec.profile
    records_path = Path(records_path)
    bucket_path = Path(pair_bucket_records_path)
    bucket_metadata_path = Path(pair_bucket_metadata_path)
    auxiliary_path = Path(auxiliary_manifest_path)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    bucket_metadata = json.loads(bucket_metadata_path.read_text(encoding="utf-8"))
    auxiliary_metadata = json.loads(auxiliary_path.read_text(encoding="utf-8"))
    _validate_global_context_contract(spec, bucket_metadata, auxiliary_metadata)

    candidate_columns = sorted(
        {
            column
            for fields in profile.source_candidate_fields.values()
            for column in fields
        }
    )
    record_schema = pq.read_schema(records_path).names
    record_id_field = (
        "canonical_record_id"
        if "canonical_record_id" in record_schema
        else "normalized_record_id"
    )
    record_columns = [
        record_id_field,
        "finite_scalar_value",
        *candidate_columns,
    ]
    if profile.heldout_sources:
        record_columns.append(spec.heldout_identity_column)
    record_columns = list(dict.fromkeys(record_columns))
    records = pd.read_parquet(records_path, columns=record_columns)
    buckets = pd.read_parquet(bucket_path)
    if not records[record_id_field].is_unique:
        raise ValueError(f"finalized {record_id_field} values must be unique")
    if not buckets[record_id_field].is_unique:
        raise ValueError(f"pair-bucket {record_id_field} values must be unique")
    if len(records) != len(buckets):
        raise ValueError(
            f"record/pair-bucket coverage mismatch: {len(records)} != {len(buckets)}"
        )

    joined = buckets.merge(
        records,
        on=record_id_field,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not (joined["_merge"] == "both").all():
        raise ValueError("one or more pair-bucket rows lack a finalized record")
    eligibility_field = (
        "assay_transfer_eligible"
        if "assay_transfer_eligible" in joined.columns
        else "bucket_eligible"
    )
    bucket_rows = joined[
        joined[eligibility_field].astype(bool) & joined["pair_bucket_key"].notna()
    ].copy()
    heldout_audit: dict[str, Any] | None = None
    if profile.heldout_sources:
        if spec.heldout_key_loader is None:
            raise ValueError("heldout sources require a heldout key loader")
        heldout_keys = set(spec.heldout_key_loader())
        if not heldout_keys:
            raise ValueError("heldout key loader returned no parent identities")
        source_scope = bucket_rows["source_id"].astype(str).isin(profile.heldout_sources)

        def parent_key(value: Any) -> str:
            identity = normalize_molecule_identity(str(value or ""))
            return identity.parent_inchi_key or identity.parent_smiles

        parent_keys = bucket_rows[spec.heldout_identity_column].map(parent_key)
        dropped = source_scope & parent_keys.isin(heldout_keys)
        dropped_rows = bucket_rows[dropped]
        dropped_by_source = (
            dropped_rows["source_id"].astype(str).value_counts().to_dict()
        )
        bucket_rows = bucket_rows[~dropped].copy()
        normalize_molecule_identity.cache_clear()
        heldout_audit = {
            "scope": (
                "only sources whose measurement is the benchmark gold outcome; "
                "mechanistic sources retain heldout-parent records"
            ),
            "sources": sorted(profile.heldout_sources),
            "identity_column": spec.heldout_identity_column,
            "heldout_key_count": len(heldout_keys),
            "dropped_records_by_source": {
                str(key): int(value) for key, value in sorted(dropped_by_source.items())
            },
            "retained_records": len(bucket_rows),
        }
    entries: dict[str, dict[str, Any]] = {}
    for pair_bucket_key, group in bucket_rows.groupby(
        "pair_bucket_key", sort=True, dropna=False
    ):
        key = str(pair_bucket_key)
        source_id = _one(group["source_id"], key, "source_id")
        canonical_endpoint_field = (
            "canonical_endpoint_name"
            if "canonical_endpoint_name" in group.columns
            else "canonical_endpoint"
        )
        canonical_unit_field = (
            "canonical_unit_text"
            if "canonical_unit_text" in group.columns
            else "canonical_unit"
        )
        endpoint_field = (spec.endpoint_field_by_source or {}).get(
            source_id, canonical_endpoint_field
        )
        if endpoint_field == canonical_endpoint_field:
            canonical_endpoint = _one(
                group[canonical_endpoint_field], key, canonical_endpoint_field
            )
        else:
            canonical_endpoint = _bucket_endpoint(key, source_id=source_id)
        canonical_unit = _one(group[canonical_unit_field], key, canonical_unit_field)
        record_count = len(group)
        support_met = record_count >= minimum_samples
        if support_met:
            variance_gate = select_variance_candidate(
                group, profile=profile, source_id=source_id
            )
        else:
            variance_gate = {
                "evaluated": False,
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

        finite_values = pd.to_numeric(group["finite_scalar_value"], errors="coerce")
        sample_sd = (
            float(finite_values.std(ddof=STANDARD_DEVIATION_DDOF))
            if len(finite_values) > 1
            else None
        )
        positive_finite_sd = bool(
            sample_sd is not None and math.isfinite(sample_sd) and sample_sd > 0
        )
        variance_flagged = bool(variance_gate["variance_gate_flagged"])
        distinct_levels = int(finite_values.dropna().nunique())
        scale_column = (
            "canonical_measurement_scale_id"
            if "canonical_measurement_scale_id" in group.columns
            else "categorical_encoder_id"
        )
        scale_values = (
            sorted(
                {
                    str(value)
                    for value in group[scale_column].dropna().tolist()
                    if str(value)
                }
            )
            if scale_column in group.columns
            else []
        )
        if len(scale_values) > 1:
            raise ValueError(f"pair bucket {key!r} spans measurement scales")
        measurement_scale_id = scale_values[0] if scale_values else None
        required_distinct_levels = profile.required_distinct_levels(
            measurement_scale_id
        )
        enough_levels = distinct_levels >= required_distinct_levels
        distance_policy = None
        if support_met and not variance_flagged and positive_finite_sd and enough_levels:
            distance_policy = build_distance_policy(
                finite_values.tolist(),
                group[record_id_field].astype(str).tolist(),
                pair_bucket_key=key,
                profile=profile,
            )

        if not support_met:
            reason = f"fewer_than_{minimum_samples}_records"
        elif variance_flagged:
            reason = "automatic_variance_gate"
        elif not positive_finite_sd:
            reason = "nonpositive_or_nonfinite_sample_sd"
        elif not enough_levels:
            reason = "fewer_than_minimum_distinct_levels"
        else:
            reason = "eligible"
        entry = {
            "source_id": source_id,
            "canonical_endpoint": canonical_endpoint,
            "canonical_unit": canonical_unit,
            "record_count": record_count,
            "sample_count": record_count,
            "sample_unit": "normalized_record",
            "minimum_sample_count": minimum_samples,
            "minimum_support_met": support_met,
            "variance_gate": variance_gate,
            "observed_sample_standard_deviation": (
                sample_sd if positive_finite_sd else None
            ),
            "assay_transfer_eligible": distance_policy is not None,
            "eligibility_reason": reason,
            "distance_policy": distance_policy,
        }
        if required_distinct_levels > 1:
            entry["distinct_measurement_levels"] = distinct_levels
            entry["minimum_distinct_measurement_levels"] = required_distinct_levels
            entry["measurement_scale_id"] = measurement_scale_id
        entries[key] = entry

    reason_counts = Counter(entry["eligibility_reason"] for entry in entries.values())
    supported_entries = [
        entry for entry in entries.values() if entry["minimum_support_met"]
    ]
    flagged_entries = [
        entry
        for entry in supported_entries
        if entry["variance_gate"]["variance_gate_flagged"]
    ]
    eligible_entries = [
        entry for entry in entries.values() if entry["assay_transfer_eligible"]
    ]
    variance_gate_contract: dict[str, Any] = {
        "candidate_fields_by_source": {
            source: list(fields)
            for source, fields in profile.source_candidate_fields.items()
        },
        "minimum_records_per_level": MINIMUM_LEVEL_RECORDS,
        "minimum_coverage": MINIMUM_COVERAGE,
        "ranking_score": "omega_squared * coverage",
        "minimum_omega_squared": MINIMUM_OMEGA_SQUARED,
        "minimum_median_range_iqr": MINIMUM_MEDIAN_RANGE_IQR,
        "minimum_median_range_sd": MINIMUM_MEDIAN_RANGE_SD,
        "gate": (
            "omega_squared >= 0.20 AND "
            "(median_range/IQR >= 1.00 OR median_range/SD >= 1.35)"
        ),
        "missing_value_policy": {
            "qualifying_conditions": "__baseline__",
            "all_other_fields": "excluded",
        },
    }
    if profile.minimum_distinct_levels > 1:
        variance_gate_contract["minimum_distinct_measurement_levels"] = (
            profile.minimum_distinct_levels
        )
    if profile.minimum_distinct_levels_by_scale:
        variance_gate_contract["minimum_distinct_measurement_levels_by_scale"] = {
            key: int(value)
            for key, value in sorted(
                profile.minimum_distinct_levels_by_scale.items()
            )
        }
    distance_contract: dict[str, Any] = {
        "standard_deviation_ddof": STANDARD_DEVIATION_DDOF,
        "transfer_max_standard_deviations": TRANSFER_MAX_STANDARD_DEVIATIONS,
        "percentile_knots": PERCENTILE_KNOTS,
        "maximum_reference_pairs": MAX_REFERENCE_PAIRS,
        "small_bucket_reference": "all unordered record pairs",
        "large_bucket_reference": "deterministic hash sample with replacement",
        "exact_equality_percentile": 0.0,
    }
    if spec.include_soft_transfer_contract:
        distance_contract["soft_transfer_target"] = soft_transfer_contract()

    payload: dict[str, Any] = {
        "policy_version": profile.version,
        "pair_bucket_version": spec.pair_bucket_version,
        "semantics": {
            "comparison_stratum": "pair_bucket_key_only",
            "sample_unit": "normalized_record",
            "repeated_records_for_one_molecule_count_independently": True,
            "minimum_sample_count": minimum_samples,
            "raw_candidate_columns_create_child_buckets": False,
            "raw_candidate_columns_receive_llm_labels": False,
            "record_level_policy_subdivision_used": False,
            "endpoint_specific_thresholds_used": False,
            "distance": "abs(left - right) / pair_bucket_sample_standard_deviation",
            "transfer_condition": "standardized_difference_sd <= 1.0",
            "ineligible_bucket_produces_negative_label": False,
            "percentile_scope": "within_pair_bucket",
            "percentile_scores_are_globally_physical": False,
        },
        "variance_gate_contract": variance_gate_contract,
        "distance_contract": distance_contract,
        "global_context_contract": {
            "mapping_version": spec.auxiliary_mapping_version,
            "attachment_version": spec.auxiliary_attachment_version,
            "source_pair_fields": {
                source: list(fields)
                for source, fields in spec.source_pair_fields.items()
            },
        },
        "inputs": {
            "records": {
                "path": str(records_path),
                "sha256": file_sha256(records_path),
                "rows": len(records),
            },
            "pair_bucket_records": {
                "path": str(bucket_path),
                "sha256": file_sha256(bucket_path),
                "rows": len(buckets),
            },
            "pair_bucket_metadata": {
                "path": str(bucket_metadata_path),
                "sha256": file_sha256(bucket_metadata_path),
            },
            "auxiliary_mapping_manifest": {
                "path": str(auxiliary_path),
                "sha256": file_sha256(auxiliary_path),
                "mapping_sha256": auxiliary_metadata["mapping_sha256"],
            },
        },
        "summary": {
            "pair_buckets": len(entries),
            "records_in_pair_buckets": int(
                sum(entry["record_count"] for entry in entries.values())
            ),
            "minimum_support_buckets": len(supported_entries),
            "minimum_support_records": int(
                sum(entry["record_count"] for entry in supported_entries)
            ),
            "variance_gate_flagged_buckets": len(flagged_entries),
            "variance_gate_flagged_records": int(
                sum(entry["record_count"] for entry in flagged_entries)
            ),
            "assay_transfer_eligible_buckets": len(eligible_entries),
            "assay_transfer_eligible_records": int(
                sum(entry["record_count"] for entry in eligible_entries)
            ),
            "eligibility_reason_counts": dict(sorted(reason_counts.items())),
        },
        "buckets": entries,
    }
    if spec.endpoint_field_by_source:
        payload["global_context_contract"]["bucket_endpoint_field_by_source"] = {
            source: spec.endpoint_field_by_source.get(source, "canonical_endpoint")
            for source in sorted(spec.source_pair_fields)
        }
        payload["semantics"]["policy_entry_canonical_endpoint_role"] = (
            "the endpoint value occupying the pair-bucket endpoint slot; raw "
            "record canonical_endpoint remains in the pair-bucket sidecar"
        )
    if heldout_audit is not None:
        payload["heldout_exclusion"] = heldout_audit
    validate_pair_bucket_transfer_policy(payload, profile=profile)
    write_deterministic_gzip(target / POLICY_FILENAME, payload)
    return payload


def _validate_global_context_contract(
    spec: TransferPolicyBuildSpec,
    pair_metadata: dict[str, Any],
    auxiliary_metadata: dict[str, Any],
) -> None:
    if pair_metadata.get("contract_version") != spec.pair_bucket_version:
        raise ValueError("pair-bucket version does not match the frozen contract")
    actual = pair_metadata.get("source_required_fields") or {}
    for source_id, expected_fields in spec.source_pair_fields.items():
        if actual.get(source_id) != list(expected_fields):
            raise ValueError(
                f"{source_id} pair-bucket fields differ from the frozen contract"
            )
    actual_endpoint_fields = pair_metadata.get("bucket_endpoint_field_by_source")
    default_endpoint_field = (
        "canonical_endpoint_name"
        if str(spec.pair_bucket_version).endswith(".v7")
        else "canonical_endpoint"
    )
    expected_endpoint_fields = {
        source: (spec.endpoint_field_by_source or {}).get(
            source, default_endpoint_field
        )
        for source in spec.source_pair_fields
    }
    if actual_endpoint_fields is None and not spec.endpoint_field_by_source:
        actual_endpoint_fields = expected_endpoint_fields
    if actual_endpoint_fields != expected_endpoint_fields:
        raise ValueError("pair-bucket endpoint-field contract differs from the build spec")
    if auxiliary_metadata.get("mapping_version") != spec.auxiliary_mapping_version:
        raise ValueError("globally reconciled mapping version mismatch")
    if (
        auxiliary_metadata.get("attachment_version")
        != spec.auxiliary_attachment_version
    ):
        raise ValueError("global auxiliary attachment version mismatch")
    declared = set(auxiliary_metadata.get("output_fields") or ())
    missing = set(spec.required_auxiliary_output_fields) - declared
    if missing:
        raise ValueError(
            f"global auxiliary output fields are incomplete: missing {sorted(missing)}"
        )


def _bucket_endpoint(bucket_key: str, *, source_id: str) -> str:
    try:
        values = json.loads(bucket_key)
    except json.JSONDecodeError as exc:
        raise ValueError(f"pair bucket key is not JSON: {bucket_key}") from exc
    if not isinstance(values, list) or len(values) < 3:
        raise ValueError(f"pair bucket key has no endpoint slot: {bucket_key}")
    if str(values[0]) != source_id:
        raise ValueError(f"pair bucket source slot mismatch: {bucket_key}")
    endpoint = str(values[1] or "")
    if not endpoint:
        raise ValueError(f"pair bucket endpoint slot is empty: {bucket_key}")
    return endpoint


def _one(values: pd.Series, bucket_key: str, field_name: str) -> str:
    unique = sorted({str(value) for value in values if pd.notna(value)})
    if len(unique) != 1:
        raise ValueError(
            f"pair bucket has inconsistent {field_name}: {bucket_key}: {unique}"
        )
    return unique[0]


def write_deterministic_gzip(path: Path, payload: Any) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(gzip.compress(encoded, compresslevel=9, mtime=0))


__all__ = [
    "POLICY_FILENAME",
    "TransferPolicyBuildSpec",
    "build_pair_bucket_transfer_policy",
    "write_deterministic_gzip",
]
