"""Build the single Bioavailability pair-bucket assay-transfer policy.

This downstream builder reads finalized normalized records and immutable v8
pair-bucket membership.  It does not rewrite records, create refined buckets,
or call an LLM.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.tasks.bioavailability_ma.build_starling_pair_bucket_sidecar import (
    DEFAULT_NORMALIZED_DIR,
    PAIR_BUCKET_METADATA_FILENAME,
    PAIR_BUCKET_RECORDS_FILENAME,
)
from tools.chembl_tool.tasks.bioavailability_ma.data_processing.auxiliary_mapping_helpers.reconciliation import (
    MAPPING_VERSION,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_auxiliary_metadata import (
    AUXILIARY_ATTACHMENT_VERSION,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_pair_bucket_transfer_policy import (
    MAX_REFERENCE_PAIRS,
    MIN_ASSAY_TRANSFER_SAMPLES,
    MINIMUM_COVERAGE,
    MINIMUM_LEVEL_RECORDS,
    MINIMUM_MEDIAN_RANGE_IQR,
    MINIMUM_MEDIAN_RANGE_SD,
    MINIMUM_OMEGA_SQUARED,
    NO_CANDIDATE,
    PAIR_BUCKET_TRANSFER_POLICY_VERSION,
    PERCENTILE_KNOTS,
    SOURCE_CANDIDATE_FIELDS,
    STANDARD_DEVIATION_DDOF,
    TRANSFER_MAX_STANDARD_DEVIATIONS,
    build_distance_policy,
    select_variance_candidate,
    validate_pair_bucket_transfer_policy,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_pair_buckets import (
    BIOAVAILABILITY_PAIR_BUCKET_VERSION,
    SOURCE_PAIR_FIELDS,
)


DEFAULT_PAIR_BUCKET_DIR = DEFAULT_NORMALIZED_DIR / "06_pair_buckets"
DEFAULT_OUTPUT_DIR = DEFAULT_NORMALIZED_DIR / "07_assay_transfer_policy"
DEFAULT_AUXILIARY_MANIFEST = (
    DEFAULT_NORMALIZED_DIR / "02_normalized/auxiliary_mapping_manifest.json"
)
POLICY_FILENAME = "pair_bucket_transfer_policy.json.gz"


def build_pair_bucket_transfer_policy(
    *,
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
    records_path = Path(records_path)
    bucket_path = Path(pair_bucket_records_path)
    bucket_metadata_path = Path(pair_bucket_metadata_path)
    auxiliary_path = Path(auxiliary_manifest_path)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    bucket_metadata = json.loads(bucket_metadata_path.read_text(encoding="utf-8"))
    auxiliary_metadata = json.loads(auxiliary_path.read_text(encoding="utf-8"))
    _validate_global_context_contract(bucket_metadata, auxiliary_metadata)

    candidate_columns = sorted(
        {field for fields in SOURCE_CANDIDATE_FIELDS.values() for field in fields}
    )
    record_columns = [
        "normalized_record_id",
        "finite_scalar_value",
        *candidate_columns,
    ]
    records = pd.read_parquet(records_path, columns=record_columns)
    buckets = pd.read_parquet(bucket_path)
    if not records["normalized_record_id"].is_unique:
        raise ValueError("finalized normalized_record_id values must be unique")
    if not buckets["normalized_record_id"].is_unique:
        raise ValueError("pair-bucket normalized_record_id values must be unique")
    if len(records) != len(buckets):
        raise ValueError(
            f"record/pair-bucket coverage mismatch: {len(records)} != {len(buckets)}"
        )

    joined = buckets.merge(
        records,
        on="normalized_record_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not (joined["_merge"] == "both").all():
        raise ValueError("one or more pair-bucket rows lack a finalized record")
    bucket_rows = joined[
        joined["bucket_eligible"].astype(bool) & joined["pair_bucket_key"].notna()
    ].copy()

    entries: dict[str, dict[str, Any]] = {}
    for pair_bucket_key, group in bucket_rows.groupby(
        "pair_bucket_key", sort=True, dropna=False
    ):
        key = str(pair_bucket_key)
        source_id = _one(group["source_id"], key, "source_id")
        canonical_endpoint = _one(
            group["canonical_endpoint"], key, "canonical_endpoint"
        )
        canonical_unit = _one(group["canonical_unit"], key, "canonical_unit")
        record_count = len(group)
        support_met = record_count >= minimum_samples
        if support_met:
            variance_gate = select_variance_candidate(group, source_id=source_id)
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

        finite_values = pd.to_numeric(
            group["finite_scalar_value"], errors="coerce"
        )
        sample_sd = (
            float(finite_values.std(ddof=STANDARD_DEVIATION_DDOF))
            if len(finite_values) > 1
            else None
        )
        positive_finite_sd = bool(
            sample_sd is not None and math.isfinite(sample_sd) and sample_sd > 0
        )
        variance_flagged = bool(variance_gate["variance_gate_flagged"])
        distance_policy = None
        if support_met and not variance_flagged and positive_finite_sd:
            distance_policy = build_distance_policy(
                finite_values.tolist(),
                group["normalized_record_id"].astype(str).tolist(),
                pair_bucket_key=key,
            )

        if not support_met:
            reason = "fewer_than_25_records"
        elif variance_flagged:
            reason = "automatic_variance_gate"
        elif not positive_finite_sd:
            reason = "nonpositive_or_nonfinite_sample_sd"
        else:
            reason = "eligible"
        entries[key] = {
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

    reason_counts = Counter(
        entry["eligibility_reason"] for entry in entries.values()
    )
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
    payload: dict[str, Any] = {
        "policy_version": PAIR_BUCKET_TRANSFER_POLICY_VERSION,
        "pair_bucket_version": BIOAVAILABILITY_PAIR_BUCKET_VERSION,
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
        "variance_gate_contract": {
            "candidate_fields_by_source": {
                source: list(fields)
                for source, fields in SOURCE_CANDIDATE_FIELDS.items()
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
        },
        "distance_contract": {
            "standard_deviation_ddof": STANDARD_DEVIATION_DDOF,
            "transfer_max_standard_deviations": TRANSFER_MAX_STANDARD_DEVIATIONS,
            "percentile_knots": PERCENTILE_KNOTS,
            "maximum_reference_pairs": MAX_REFERENCE_PAIRS,
            "small_bucket_reference": "all unordered record pairs",
            "large_bucket_reference": (
                "deterministic hash sample with replacement"
            ),
            "exact_equality_percentile": 0.0,
        },
        "global_context_contract": {
            "mapping_version": MAPPING_VERSION,
            "attachment_version": AUXILIARY_ATTACHMENT_VERSION,
            "source_pair_fields": {
                source: list(fields) for source, fields in SOURCE_PAIR_FIELDS.items()
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
    validate_pair_bucket_transfer_policy(payload)
    _write_deterministic_gzip(target / POLICY_FILENAME, payload)
    return payload


def _validate_global_context_contract(
    pair_metadata: dict[str, Any], auxiliary_metadata: dict[str, Any]
) -> None:
    if pair_metadata.get("contract_version") != BIOAVAILABILITY_PAIR_BUCKET_VERSION:
        raise ValueError("pair-bucket version does not match the frozen v8 contract")
    actual = pair_metadata.get("source_required_fields") or {}
    for source_id, expected_fields in SOURCE_PAIR_FIELDS.items():
        if actual.get(source_id) != list(expected_fields):
            raise ValueError(
                f"{source_id} pair-bucket fields differ from the frozen contract"
            )
    if auxiliary_metadata.get("mapping_version") != MAPPING_VERSION:
        raise ValueError("globally reconciled mapping version mismatch")
    if auxiliary_metadata.get("attachment_version") != AUXILIARY_ATTACHMENT_VERSION:
        raise ValueError("global auxiliary attachment version mismatch")
    if tuple(auxiliary_metadata.get("output_fields") or ()) != (
        "global_context",
        "global_species_context",
    ):
        raise ValueError("global auxiliary output fields are incomplete")


def _one(values: pd.Series, bucket_key: str, field: str) -> str:
    unique = sorted({str(value) for value in values if pd.notna(value)})
    if len(unique) != 1:
        raise ValueError(
            f"pair bucket has inconsistent {field}: {bucket_key}: {unique}"
        )
    return unique[0]


def _write_deterministic_gzip(path: Path, payload: Any) -> None:
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        default=str(DEFAULT_NORMALIZED_DIR / "03_records/records.parquet"),
    )
    parser.add_argument(
        "--pair-bucket-records",
        default=str(DEFAULT_PAIR_BUCKET_DIR / PAIR_BUCKET_RECORDS_FILENAME),
    )
    parser.add_argument(
        "--pair-bucket-metadata",
        default=str(DEFAULT_PAIR_BUCKET_DIR / PAIR_BUCKET_METADATA_FILENAME),
    )
    parser.add_argument(
        "--auxiliary-manifest", default=str(DEFAULT_AUXILIARY_MANIFEST)
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = build_pair_bucket_transfer_policy(
        records_path=args.records,
        pair_bucket_records_path=args.pair_bucket_records,
        pair_bucket_metadata_path=args.pair_bucket_metadata,
        auxiliary_manifest_path=args.auxiliary_manifest,
        out_dir=args.out_dir,
    )
    summary = payload["summary"]
    print(
        "[build_starling_pair_bucket_transfer_policy] "
        f"buckets={summary['pair_buckets']:,} "
        f"supported={summary['minimum_support_buckets']:,} "
        f"variance_flagged={summary['variance_gate_flagged_buckets']:,} "
        f"eligible={summary['assay_transfer_eligible_buckets']:,} "
        f"out={args.out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
