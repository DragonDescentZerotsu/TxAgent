"""Audit values after joining v5 buckets to endpoint-policy assignments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.tasks.bioavailability_ma.build_normalized_starling_evidence_library import (
    DEFAULT_OUT_DIR,
    RECORDS_FILENAME,
)
from tools.chembl_tool.tasks.bioavailability_ma.build_starling_endpoint_policy_assignments import (
    ASSIGNMENTS_FILENAME,
    DEFAULT_OUT_DIR as DEFAULT_POLICY_DIR,
    REGISTRY_FILENAME,
)
from tools.chembl_tool.tasks.bioavailability_ma.build_starling_pair_bucket_sidecar import (
    PAIR_BUCKET_RECORDS_FILENAME,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_endpoint_policies_v2 import (
    normalize_policy_distance,
)


DEFAULT_V5_DIR = Path(DEFAULT_OUT_DIR)
DEFAULT_SIDECAR = DEFAULT_V5_DIR / "pair_buckets" / PAIR_BUCKET_RECORDS_FILENAME
DEFAULT_ASSIGNMENTS = DEFAULT_POLICY_DIR / ASSIGNMENTS_FILENAME
DEFAULT_REGISTRY = DEFAULT_POLICY_DIR / REGISTRY_FILENAME
DEFAULT_ANALYSIS_DIR = DEFAULT_V5_DIR / "analysis" / "pair_bucket_distributions"
AUDIT_FILENAME = "pair_bucket_distribution_audit.parquet"
SUMMARY_FILENAME = "pair_bucket_distribution_summary.json"
SUPPORTED_POLICY_VERSIONS = ("v1", "v2")


def audit_distributions(
    *,
    records_path: str | Path,
    sidecar_path: str | Path,
    assignments_path: str | Path,
    registry_path: str | Path,
    out_dir: str | Path,
    policy_version: str | None = None,
) -> dict[str, Any]:
    records_path = Path(records_path)
    sidecar_path = Path(sidecar_path)
    assignments_path = Path(assignments_path)
    registry_path = Path(registry_path)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    records = pd.read_parquet(
        records_path,
        columns=[
            "normalized_record_id",
            "canonical_smiles",
            "finite_scalar_value",
        ],
    )
    sidecar = pd.read_parquet(sidecar_path)
    assignments = pd.read_parquet(assignments_path)
    if len(records) != len(sidecar) or len(records) != len(assignments):
        raise ValueError("records/sidecar/assignment row-count mismatch")
    merged = sidecar.merge(
        records,
        on="normalized_record_id",
        how="left",
        validate="one_to_one",
    ).merge(
        assignments,
        on="normalized_record_id",
        how="left",
        validate="one_to_one",
    )
    if merged["canonical_smiles"].isna().all() and len(merged):
        raise ValueError("records/sidecar identity join failed")

    policy_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_version = str(policy_payload.get("registry_version") or "")
    inferred_policy_version = registry_version.rsplit(".", 1)[-1]
    selected_policy_version = policy_version or inferred_policy_version
    if selected_policy_version not in SUPPORTED_POLICY_VERSIONS:
        raise ValueError(f"unsupported policy version: {selected_policy_version!r}")
    if inferred_policy_version != selected_policy_version:
        raise ValueError(
            f"registry {registry_version!r} does not match requested "
            f"policy version {selected_policy_version!r}"
        )
    policies = dict(policy_payload.get("endpoint_policies") or {})
    eligible = merged[
        merged["bucket_eligible"].fillna(False)
        & merged["policy_assignment_status"].eq("assigned")
    ].copy()
    eligible["finite_scalar_value"] = pd.to_numeric(
        eligible["finite_scalar_value"], errors="coerce"
    )
    eligible = eligible[np.isfinite(eligible["finite_scalar_value"])]
    rows: list[dict[str, Any]] = []
    group_columns = ["pair_bucket_key", "endpoint_policy_key"]
    for (bucket_key, policy_key), group in eligible.groupby(
        group_columns, sort=False, dropna=False
    ):
        policy_key = str(policy_key)
        policy = policies.get(policy_key)
        if not isinstance(policy, dict):
            raise ValueError(f"missing policy definition for {policy_key!r}")
        values = group["finite_scalar_value"].to_numpy(dtype=float)
        transformed = (
            np.log10(values)
            if policy.get("distance") == "absolute_log10_ratio"
            else values
        )
        quantiles = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
        transformed_quantiles = np.quantile(transformed, [0.05, 0.95])
        robust_span = float(transformed_quantiles[1] - transformed_quantiles[0])
        threshold_variants = _threshold_variants(policy)
        primary_thresholds = threshold_variants["primary"]
        far_threshold = float(primary_thresholds["not_transfer_min"])
        comparison_bucket_key = json.dumps(
            [bucket_key, policy_key],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        variant_columns: dict[str, Any] = {}
        normalization = policy.get("normalization")
        has_normalization = isinstance(normalization, dict)
        for variant, thresholds in threshold_variants.items():
            variant_columns[f"{variant}_transfer_max"] = float(
                thresholds["transfer_max"]
            )
            variant_columns[f"{variant}_not_transfer_min"] = float(
                thresholds["not_transfer_min"]
            )
            variant_columns[f"{variant}_wide_distribution_flag"] = (
                robust_span >= float(thresholds["not_transfer_min"])
            )
            if has_normalization:
                normalized_thresholds = normalization["normalized_thresholds"][
                    variant
                ]
                variant_columns[
                    f"{variant}_normalized_transfer_max_0_100"
                ] = float(normalized_thresholds["transfer_max"])
                variant_columns[
                    f"{variant}_normalized_not_transfer_min_0_100"
                ] = float(normalized_thresholds["not_transfer_min"])
        normalized_columns: dict[str, Any] = {}
        if has_normalization:
            anchor = float(normalization["raw_distance_anchor"])
            median_value = float(np.median(transformed))
            normalized_deviations = np.asarray(
                [
                    normalize_policy_distance(policy, abs(float(value) - median_value))
                    for value in transformed
                ],
                dtype=float,
            )
            deviation_quantiles = np.quantile(
                normalized_deviations, [0.05, 0.25, 0.50, 0.75, 0.95]
            )
            normalized_columns = {
                "normalized_comparison_space_robust_span_0_100": (
                    normalize_policy_distance(policy, robust_span)
                ),
                "normalization_saturated": robust_span > anchor,
                "normalization_raw_distance_anchor": anchor,
                "normalized_absolute_deviation_from_median_p05_0_100": float(
                    deviation_quantiles[0]
                ),
                "normalized_absolute_deviation_from_median_p25_0_100": float(
                    deviation_quantiles[1]
                ),
                "normalized_absolute_deviation_from_median_p50_0_100": float(
                    deviation_quantiles[2]
                ),
                "normalized_absolute_deviation_from_median_p75_0_100": float(
                    deviation_quantiles[3]
                ),
                "normalized_absolute_deviation_from_median_p95_0_100": float(
                    deviation_quantiles[4]
                ),
                "normalization_saturated_record_count": int(
                    np.sum(np.abs(transformed - median_value) > anchor)
                ),
                "normalization_saturated_record_rate": float(
                    np.mean(np.abs(transformed - median_value) > anchor)
                ),
            }
        rows.append(
            {
                "comparison_bucket_key": comparison_bucket_key,
                "pair_bucket_key": bucket_key,
                "source_id": _one(group["source_id"], "source"),
                "canonical_endpoint": _one(
                    group["canonical_endpoint"], "canonical endpoint"
                ),
                "canonical_unit": _one(group["canonical_unit"], "canonical unit"),
                "endpoint_policy_key": policy_key,
                "threshold_profile": policy.get("threshold_profile"),
                "threshold_profile_version": policy.get(
                    "threshold_profile_version"
                ),
                "n_records": len(group),
                "n_molecules": group["canonical_smiles"].nunique(),
                "minimum": float(np.min(values)),
                "p05": float(quantiles[0]),
                "p25": float(quantiles[1]),
                "median": float(quantiles[2]),
                "p75": float(quantiles[3]),
                "p95": float(quantiles[4]),
                "maximum": float(np.max(values)),
                "comparison_space_robust_span": robust_span,
                "not_transfer_min": far_threshold,
                "robust_span_over_far_threshold": (
                    robust_span / far_threshold if far_threshold > 0 else None
                ),
                "wide_distribution_flag": robust_span >= far_threshold,
                **normalized_columns,
                **variant_columns,
            }
        )

    audit = pd.DataFrame(rows)
    if not audit.empty:
        audit = audit.sort_values(
            ["robust_span_over_far_threshold", "n_records"],
            ascending=[False, False],
            kind="stable",
        ).reset_index(drop=True)
    audit_path = target / AUDIT_FILENAME
    audit.to_parquet(audit_path, index=False)
    widest_columns = [
        "pair_bucket_key",
        "source_id",
        "canonical_endpoint",
        "canonical_unit",
        "endpoint_policy_key",
        "n_records",
        "n_molecules",
        "p05",
        "median",
        "p95",
        "robust_span_over_far_threshold",
    ]
    widest = (
        audit.loc[audit["n_molecules"] >= 2, widest_columns].head(100)
        if not audit.empty
        else audit
    )
    summary = {
        "policy_version": selected_policy_version,
        "registry_version": registry_version,
        "input": {
            "records_path": str(records_path),
            "records_sha256": file_sha256(records_path),
            "sidecar_path": str(sidecar_path),
            "sidecar_sha256": file_sha256(sidecar_path),
            "assignments_path": str(assignments_path),
            "assignments_sha256": file_sha256(assignments_path),
            "registry_path": str(registry_path),
            "registry_sha256": file_sha256(registry_path),
        },
        "stats": {
            "eligible_records_with_finite_values": len(eligible),
            "buckets": len(audit),
            "pairable_buckets": int((audit["n_molecules"] >= 2).sum())
            if not audit.empty
            else 0,
            "wide_buckets": int(audit["wide_distribution_flag"].sum())
            if not audit.empty
            else 0,
        },
        "wide_definition": (
            "comparison-space p95-p05 is at least the policy not-transfer threshold"
        ),
        "threshold_variant_summaries": {
            variant: {
                "distinct_threshold_pairs": thresholds,
                "wide_buckets": int(
                    audit[f"{variant}_wide_distribution_flag"].sum()
                )
                if not audit.empty
                else 0,
            }
            for variant, thresholds in _registry_threshold_variant_ranges(
                policies
            ).items()
        },
        "widest_pairable_buckets": widest.to_dict(orient="records"),
        "output": {
            "audit_path": str(audit_path),
            "audit_sha256": file_sha256(audit_path),
        },
    }
    if selected_policy_version == "v2":
        saturated_bucket_count = int(audit["normalization_saturated"].sum())
        saturated_record_count = int(
            audit["normalization_saturated_record_count"].sum()
        )
        normalization_record_count = int(audit["n_records"].sum())
        summary["distance_normalization"] = {
            **dict(policy_payload.get("distance_normalization") or {}),
            "threshold_profile_version": policy_payload.get(
                "threshold_profile_version"
            ),
            "threshold_profile_summaries": {
                profile_key: {
                    "policy_family": profile["policy_family"],
                    "raw_distance_anchor": float(
                        profile["normalization"]["raw_distance_anchor"]
                    ),
                    "normalized_thresholds_display_0_100": {
                        variant: {
                            boundary: f"{float(value):.2f}"
                            for boundary, value in thresholds.items()
                        }
                        for variant, thresholds in profile["normalization"][
                            "normalized_thresholds"
                        ].items()
                    },
                }
                for profile_key, profile in sorted(
                    dict(policy_payload.get("threshold_profiles") or {}).items()
                )
            },
            "saturated_bucket_count": saturated_bucket_count,
            "saturated_bucket_rate": (
                saturated_bucket_count / len(audit) if len(audit) else 0.0
            ),
            "saturated_record_distance_count": saturated_record_count,
            "saturated_record_distance_rate": (
                saturated_record_count / normalization_record_count
                if normalization_record_count
                else 0.0
            ),
            "normalized_quantile_reference": (
                "absolute comparison-space distance from the bucket median"
            ),
        }
    (target / SUMMARY_FILENAME).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    return summary


def _one(series: pd.Series, label: str) -> str:
    values = {str(value) for value in series.dropna().tolist()}
    if len(values) != 1:
        raise ValueError(f"bucket spans {label} values: {sorted(values)}")
    return next(iter(values))


def _threshold_variants(policy: dict[str, Any]) -> dict[str, dict[str, float]]:
    nested = policy.get("thresholds")
    if isinstance(nested, dict) and isinstance(nested.get("primary"), dict):
        return {
            str(name): {
                "transfer_max": float(values["transfer_max"]),
                "not_transfer_min": float(values["not_transfer_min"]),
            }
            for name, values in nested.items()
        }
    return {
        "primary": {
            "transfer_max": float(policy["transfer_max"]),
            "not_transfer_min": float(policy["not_transfer_min"]),
        }
    }


def _registry_threshold_variant_ranges(
    policies: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, float]]]:
    """List distinct boundaries without mixing endpoint-family geometries."""
    collected: dict[str, set[tuple[float, float]]] = {}
    for policy in policies.values():
        for variant, thresholds in _threshold_variants(policy).items():
            collected.setdefault(variant, set()).add(
                (
                    float(thresholds["transfer_max"]),
                    float(thresholds["not_transfer_min"]),
                )
            )
    return {
        variant: [
            {"transfer_max": transfer, "not_transfer_min": not_transfer}
            for transfer, not_transfer in sorted(values)
        ]
        for variant, values in collected.items()
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-version",
        choices=SUPPORTED_POLICY_VERSIONS,
        required=True,
        help="Select the policy artifact explicitly; v1 and v2 are never mixed.",
    )
    parser.add_argument("--records", default=str(DEFAULT_V5_DIR / RECORDS_FILENAME))
    parser.add_argument("--sidecar", default=str(DEFAULT_SIDECAR))
    parser.add_argument("--assignments")
    parser.add_argument("--registry")
    parser.add_argument("--out-dir")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    policy_dir = DEFAULT_V5_DIR / "endpoint_policies" / args.policy_version
    assignments = args.assignments or str(policy_dir / ASSIGNMENTS_FILENAME)
    registry = args.registry or str(policy_dir / REGISTRY_FILENAME)
    out_dir = args.out_dir or str(
        DEFAULT_V5_DIR
        / "analysis"
        / "pair_bucket_distributions"
        / args.policy_version
    )
    summary = audit_distributions(
        records_path=args.records,
        sidecar_path=args.sidecar,
        assignments_path=assignments,
        registry_path=registry,
        out_dir=out_dir,
        policy_version=args.policy_version,
    )
    print(
        "[audit_starling_pair_bucket_distributions] "
        f"buckets={summary['stats']['buckets']:,} "
        f"pairable={summary['stats']['pairable_buckets']:,} "
        f"wide={summary['stats']['wide_buckets']:,} out={out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
