"""Train-only diagnostic stress test for Bioavailability endpoint policy v2.

The audit excludes the union of random/scaffold benchmark test parents and
emits aggregate diagnostics only.  It does not enumerate a training-pair
dataset and cannot tune policy thresholds from benchmark outcomes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.starling.heldout_index import load_heldout_identity_keys
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.tasks.bioavailability_ma.build_normalized_starling_evidence_library import (
    DEFAULT_OUT_DIR,
    RECORDS_FILENAME,
)
from tools.chembl_tool.tasks.bioavailability_ma.build_starling_endpoint_policy_assignments_v2 import (
    ASSIGNMENTS_FILENAME,
    DEFAULT_OUT_DIR as DEFAULT_POLICY_DIR,
    REGISTRY_FILENAME,
)
from tools.chembl_tool.tasks.bioavailability_ma.build_starling_pair_bucket_sidecar import (
    PAIR_BUCKET_RECORDS_FILENAME,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_endpoint_policies_v2 import (
    label_distance,
    normalize_policy_distance,
)


DEFAULT_V5_DIR = Path(DEFAULT_OUT_DIR)
DEFAULT_SIDECAR = DEFAULT_V5_DIR / "pair_buckets" / PAIR_BUCKET_RECORDS_FILENAME
DEFAULT_ASSIGNMENTS = DEFAULT_POLICY_DIR / ASSIGNMENTS_FILENAME
DEFAULT_REGISTRY = DEFAULT_POLICY_DIR / REGISTRY_FILENAME
DEFAULT_RANDOM_HELDOUT = Path(
    "data/processed_starling/Bioavailability_Ma/random/test_molecule_labels.jsonl"
)
DEFAULT_SCAFFOLD_HELDOUT = Path(
    "data/processed_starling/Bioavailability_Ma/scaffold/test_molecule_labels.jsonl"
)
DEFAULT_ANALYSIS_DIR = (
    DEFAULT_V5_DIR / "analysis" / "endpoint_policy_calibration" / "v2"
)

PAIR_YIELDS_FILENAME = "pair_label_yields.parquet"
SAME_PARENT_FILENAME = "same_parent_observational_dispersion.parquet"
CROSS_PARENT_FILENAME = "cross_parent_distance_distributions.parquet"
COVERAGE_FILENAME = "endpoint_policy_coverage.parquet"
STABILITY_FILENAME = "threshold_label_stability.parquet"
SUMMARY_FILENAME = "endpoint_policy_calibration_summary.json"


def _comparison_values(values: np.ndarray, policy: dict[str, Any]) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if policy.get("transform") == "log10":
        if np.any(result <= 0):
            raise ValueError("assigned log-policy values must be positive")
        result = np.log10(result)
    return result


def _count_pairs_leq(values: np.ndarray, threshold: float) -> int:
    ordered = np.sort(np.asarray(values, dtype=float))
    count = 0
    left = 0
    for right in range(len(ordered)):
        while ordered[right] - ordered[left] > threshold:
            left += 1
        count += right - left
    return int(count)


def _count_pairs_lt(values: np.ndarray, threshold: float) -> int:
    ordered = np.sort(np.asarray(values, dtype=float))
    count = 0
    left = 0
    for right in range(len(ordered)):
        while left < right and ordered[right] - ordered[left] >= threshold:
            left += 1
        count += right - left
    return int(count)


def aggregate_pair_yields(
    values: np.ndarray,
    parent_keys: np.ndarray,
    *,
    transfer_max: float,
    not_transfer_min: float,
) -> dict[str, int]:
    """Count cross-parent near/deadband/far pairs without enumerating them."""
    values = np.asarray(values, dtype=float)
    parent_keys = np.asarray(parent_keys, dtype=object)
    total = len(values) * (len(values) - 1) // 2
    near = _count_pairs_leq(values, transfer_max)
    below_far = _count_pairs_lt(values, not_transfer_min)
    for parent in np.unique(parent_keys):
        same = values[parent_keys == parent]
        total -= len(same) * (len(same) - 1) // 2
        near -= _count_pairs_leq(same, transfer_max)
        below_far -= _count_pairs_lt(same, not_transfer_min)
    far = total - below_far
    deadband = total - near - far
    if min(total, near, deadband, far) < 0:
        raise AssertionError("invalid aggregate pair counts")
    return {
        "cross_parent_pairs": int(total),
        "transfer_pairs": int(near),
        "deadband_pairs": int(deadband),
        "not_transfer_pairs": int(far),
    }


def _sample_cross_parent_distances(
    values: np.ndarray,
    parent_keys: np.ndarray,
    *,
    key: str,
    limit: int = 10_000,
) -> tuple[np.ndarray, bool]:
    n = len(values)
    total = n * (n - 1) // 2
    same_total = sum(
        count * (count - 1) // 2 for count in Counter(parent_keys.tolist()).values()
    )
    cross_total = total - same_total
    if cross_total <= limit and n <= 2_000:
        distances = [
            abs(float(values[j]) - float(values[i]))
            for i in range(n)
            for j in range(i + 1, n)
            if parent_keys[i] != parent_keys[j]
        ]
        return np.asarray(distances, dtype=float), False
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    distances: list[float] = []
    attempts = 0
    maximum_attempts = max(limit * 30, 1_000)
    while len(distances) < min(limit, cross_total) and attempts < maximum_attempts:
        left = int(rng.integers(0, n))
        right = int(rng.integers(0, n - 1))
        if right >= left:
            right += 1
        attempts += 1
        if parent_keys[left] == parent_keys[right]:
            continue
        distances.append(abs(float(values[left]) - float(values[right])))
    return np.asarray(distances, dtype=float), True


def _parent_key(smiles: str) -> str:
    identity = normalize_molecule_identity(str(smiles or ""))
    return identity.parent_inchi_key or identity.parent_smiles


def audit_policy_v2_calibration(
    *,
    records_path: str | Path,
    sidecar_path: str | Path,
    assignments_path: str | Path,
    registry_path: str | Path,
    heldout_paths: list[str | Path],
    out_dir: str | Path,
) -> dict[str, Any]:
    records_path = Path(records_path)
    sidecar_path = Path(sidecar_path)
    assignments_path = Path(assignments_path)
    registry_path = Path(registry_path)
    heldout_paths = [Path(path) for path in heldout_paths]
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    records = pd.read_parquet(
        records_path,
        columns=[
            "normalized_record_id",
            "source_id",
            "canonical_smiles",
            "canonical_endpoint",
            "canonical_unit",
            "finite_scalar_value",
        ],
    )
    sidecar = pd.read_parquet(sidecar_path)
    assignments = pd.read_parquet(assignments_path)
    if not (len(records) == len(sidecar) == len(assignments)):
        raise ValueError("records/sidecar/assignment row-count mismatch")
    merged = (
        records.merge(
            sidecar[
                [
                    "normalized_record_id",
                    "pair_bucket_key",
                    "bucket_eligible",
                    "bucket_exclusion_reason",
                ]
            ],
            on="normalized_record_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            assignments,
            on="normalized_record_id",
            how="left",
            validate="one_to_one",
        )
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("registry_version") != "bioavailability_endpoint_policies.v2":
        raise ValueError("calibration audit requires the v2 registry")
    policies = dict(registry.get("endpoint_policies") or {})

    heldout_keys: set[str] = set()
    for path in heldout_paths:
        heldout_keys.update(load_heldout_identity_keys(path))
    unique_smiles = sorted(
        {str(value) for value in merged["canonical_smiles"].dropna().tolist()}
    )
    parent_by_smiles = {smiles: _parent_key(smiles) for smiles in unique_smiles}
    merged["molecular_parent_key"] = merged["canonical_smiles"].map(parent_by_smiles)
    unresolved_parent = merged["molecular_parent_key"].fillna("").eq("")
    heldout_mask = merged["molecular_parent_key"].isin(heldout_keys)
    train = merged[~unresolved_parent & ~heldout_mask].copy()
    train["finite_scalar_value"] = pd.to_numeric(
        train["finite_scalar_value"], errors="coerce"
    )
    eligible = train[
        train["bucket_eligible"].fillna(False)
        & train["policy_assignment_status"].eq("assigned")
        & np.isfinite(train["finite_scalar_value"])
    ].copy()
    eligible["threshold_profile"] = eligible["endpoint_policy_key"].map(
        lambda key: (policies.get(str(key)) or {}).get("threshold_profile")
    )
    eligible["comparison_bucket_key"] = eligible.apply(
        lambda row: json.dumps(
            [row["pair_bucket_key"], row["endpoint_policy_key"]],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        axis=1,
    )

    pair_yields: list[dict[str, Any]] = []
    same_parent_rows: list[dict[str, Any]] = []
    cross_parent_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    grouped = eligible.groupby("comparison_bucket_key", sort=False)
    for comparison_key, group in grouped:
        policy_key = str(group["endpoint_policy_key"].iloc[0])
        policy = policies.get(policy_key)
        if not isinstance(policy, dict):
            raise ValueError(f"missing v2 policy: {policy_key}")
        values = _comparison_values(
            group["finite_scalar_value"].to_numpy(dtype=float), policy
        )
        parents = group["molecular_parent_key"].to_numpy(dtype=object)
        thresholds = dict(policy["thresholds"])
        common = {
            "comparison_bucket_key": comparison_key,
            "pair_bucket_key": str(group["pair_bucket_key"].iloc[0]),
            "endpoint_policy_key": policy_key,
            "canonical_endpoint": str(group["canonical_endpoint"].iloc[0]),
            "measurement_subtype": str(group["measurement_subtype"].iloc[0]),
            "threshold_profile": str(policy["threshold_profile"]),
            "threshold_profile_version": str(
                policy["threshold_profile_version"]
            ),
            "n_records": len(group),
            "n_molecular_parents": int(group["molecular_parent_key"].nunique()),
        }
        for variant, boundary in thresholds.items():
            normalized_boundary = dict(policy["normalization"])[
                "normalized_thresholds"
            ][variant]
            record_counts = aggregate_pair_yields(
                values,
                parents,
                transfer_max=float(boundary["transfer_max"]),
                not_transfer_min=float(boundary["not_transfer_min"]),
            )
            parent_frame = pd.DataFrame({"parent": parents, "value": values})
            parent_values = (
                parent_frame.groupby("parent", sort=False)["value"]
                .median()
                .to_numpy(dtype=float)
            )
            unique_parents = np.arange(len(parent_values), dtype=object)
            molecule_counts = aggregate_pair_yields(
                parent_values,
                unique_parents,
                transfer_max=float(boundary["transfer_max"]),
                not_transfer_min=float(boundary["not_transfer_min"]),
            )
            pair_yields.extend(
                [
                    {
                        **common,
                        "threshold_variant": variant,
                        "comparison_level": "record_comparison",
                        "normalized_transfer_max_0_100": float(
                            normalized_boundary["transfer_max"]
                        ),
                        "normalized_not_transfer_min_0_100": float(
                            normalized_boundary["not_transfer_min"]
                        ),
                        **record_counts,
                    },
                    {
                        **common,
                        "threshold_variant": variant,
                        "comparison_level": "unique_molecule_pair",
                        "normalized_transfer_max_0_100": float(
                            normalized_boundary["transfer_max"]
                        ),
                        "normalized_not_transfer_min_0_100": float(
                            normalized_boundary["not_transfer_min"]
                        ),
                        **molecule_counts,
                    },
                ]
            )

        for parent, parent_group in pd.DataFrame(
            {"parent": parents, "value": values}
        ).groupby("parent", sort=False):
            if len(parent_group) < 2:
                continue
            parent_values = parent_group["value"].to_numpy(dtype=float)
            raw_span = float(np.max(parent_values) - np.min(parent_values))
            normalized_span = normalize_policy_distance(policy, raw_span)
            anchor = float(policy["normalization"]["raw_distance_anchor"])
            same_parent_rows.append(
                {
                    **common,
                    "molecular_parent_key": parent,
                    "n_records_for_parent": len(parent_values),
                    "observed_minimum": float(np.min(parent_values)),
                    "observed_maximum": float(np.max(parent_values)),
                    "observed_span": raw_span,
                    "normalized_observed_span_0_100": normalized_span,
                    "normalization_saturated": raw_span > anchor,
                    "interpretation": (
                        "observational within-parent dispersion; not replicate noise"
                    ),
                }
            )

        sampled, sampled_flag = _sample_cross_parent_distances(
            values, parents, key=comparison_key
        )
        if len(sampled):
            normalized_sampled = np.asarray(
                [
                    normalize_policy_distance(policy, float(distance))
                    for distance in sampled
                ],
                dtype=float,
            )
            quantiles = np.quantile(sampled, [0.05, 0.25, 0.5, 0.75, 0.95])
            normalized_quantiles = np.quantile(
                normalized_sampled, [0.05, 0.25, 0.5, 0.75, 0.95]
            )
            anchor = float(policy["normalization"]["raw_distance_anchor"])
            saturation_count = int(np.sum(sampled > anchor))
            cross_parent_rows.append(
                {
                    **common,
                    "distance_sample_size": len(sampled),
                    "deterministic_sample": sampled_flag,
                    "distance_p05": float(quantiles[0]),
                    "distance_p25": float(quantiles[1]),
                    "distance_median": float(quantiles[2]),
                    "distance_p75": float(quantiles[3]),
                    "distance_p95": float(quantiles[4]),
                    "distance_minimum": float(np.min(sampled)),
                    "distance_maximum": float(np.max(sampled)),
                    "distance_span": float(np.max(sampled) - np.min(sampled)),
                    "normalized_distance_p05_0_100": float(normalized_quantiles[0]),
                    "normalized_distance_p25_0_100": float(normalized_quantiles[1]),
                    "normalized_distance_median_0_100": float(
                        normalized_quantiles[2]
                    ),
                    "normalized_distance_p75_0_100": float(normalized_quantiles[3]),
                    "normalized_distance_p95_0_100": float(normalized_quantiles[4]),
                    "normalized_distance_minimum_0_100": float(
                        np.min(normalized_sampled)
                    ),
                    "normalized_distance_maximum_0_100": float(
                        np.max(normalized_sampled)
                    ),
                    "normalized_distance_span_0_100": float(
                        np.max(normalized_sampled) - np.min(normalized_sampled)
                    ),
                    "normalization_saturation_count": saturation_count,
                    "normalization_saturation_rate": saturation_count / len(sampled),
                }
            )
            primary_labels = [
                label_distance(policy, float(distance), variant="primary")
                for distance in sampled
            ]
            for variant in ("strict", "permissive"):
                sensitivity_labels = [
                    label_distance(policy, float(distance), variant=variant)
                    for distance in sampled
                ]
                agreements = sum(
                    left == right
                    for left, right in zip(primary_labels, sensitivity_labels)
                )
                opposing = sum(
                    {left, right} == {"transfer", "not_transfer"}
                    for left, right in zip(primary_labels, sensitivity_labels)
                )
                stability_rows.append(
                    {
                        **common,
                        "sensitivity_variant": variant,
                        "distance_sample_size": len(sampled),
                        "exact_label_agreements": agreements,
                        "exact_label_agreement_rate": agreements / len(sampled),
                        "opposing_label_count": opposing,
                        "primary_labeled_count": sum(
                            value is not None for value in primary_labels
                        ),
                        "sensitivity_labeled_count": sum(
                            value is not None for value in sensitivity_labels
                        ),
                    }
                )

    coverage = (
        eligible.groupby(
            [
                "source_id",
                "canonical_endpoint",
                "measurement_subtype",
                "endpoint_policy_key",
                "threshold_profile",
            ],
            dropna=False,
        )
        .agg(
            records=("normalized_record_id", "size"),
            molecular_parents=("molecular_parent_key", "nunique"),
            comparison_buckets=("comparison_bucket_key", "nunique"),
        )
        .reset_index()
    )
    outputs = {
        PAIR_YIELDS_FILENAME: pd.DataFrame(pair_yields),
        SAME_PARENT_FILENAME: pd.DataFrame(same_parent_rows),
        CROSS_PARENT_FILENAME: pd.DataFrame(cross_parent_rows),
        COVERAGE_FILENAME: coverage,
        STABILITY_FILENAME: pd.DataFrame(stability_rows),
    }
    output_meta: dict[str, Any] = {}
    for filename, frame in outputs.items():
        path = target / filename
        frame.to_parquet(path, index=False)
        output_meta[filename] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "rows": len(frame),
        }

    status_counts = Counter(str(value) for value in train["policy_assignment_status"])
    normalized_threshold_summaries = {
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
            dict(registry.get("threshold_profiles") or {}).items()
        )
    }
    total_sampled_distances = sum(
        int(row["distance_sample_size"]) for row in cross_parent_rows
    )
    total_saturated_distances = sum(
        int(row["normalization_saturation_count"]) for row in cross_parent_rows
    )
    summary = {
        "audit_type": "preregistered_train_only_policy_stress_test",
        "policy_version": "v2",
        "threshold_selection_from_benchmark_performance": False,
        "heldout_exclusion": {
            "paths": [
                {"path": str(path), "sha256": file_sha256(path)}
                for path in heldout_paths
            ],
            "union_test_parent_count": len(heldout_keys),
            "excluded_records": int(heldout_mask.sum()),
            "unresolved_parent_records_excluded": int(unresolved_parent.sum()),
            "residual_test_parent_overlap": int(
                train["molecular_parent_key"].isin(heldout_keys).sum()
            ),
            "zero_test_parent_overlap": not bool(
                train["molecular_parent_key"].isin(heldout_keys).any()
            ),
        },
        "assignment_and_exclusion_counts": {
            "input_records": len(merged),
            "train_only_records": len(train),
            "eligible_assay_transfer_records": len(eligible),
            "policy_assignment_status_counts": dict(sorted(status_counts.items())),
        },
        "distance_normalization": {
            **dict(registry.get("distance_normalization") or {}),
            "threshold_profile_version": registry.get(
                "threshold_profile_version"
            ),
            "threshold_profile_summaries": normalized_threshold_summaries,
            "sampled_distance_count": total_sampled_distances,
            "saturation_count": total_saturated_distances,
            "saturation_rate": (
                total_saturated_distances / total_sampled_distances
                if total_sampled_distances
                else 0.0
            ),
        },
        "diagnostic_interpretation": {
            "same_parent_dispersion": (
                "Observational within-bucket dispersion for the same standardized "
                "molecular parent; it is not interpreted as replicate noise."
            ),
            "cross_parent_distances": (
                "Exact pair-label yields plus deterministic capped distance samples "
                "for quantiles and stability."
            ),
            "threshold_use": (
                "Sensitivity analysis only. No benchmark outcome is read and no "
                "threshold is fitted or selected."
            ),
            "normalized_distance": (
                "Derived from each full-precision raw distance using the invariant "
                "permissive-boundary anchor. Pair labels and stability remain based "
                "only on raw distance."
            ),
        },
        "scope": {
            "training_pairs_materialized": False,
            "modeling_datasets_modified": False,
            "heldout_identity_manifests_read": True,
            "benchmark_label_values_used": False,
            "aggregate_pair_counts_only": True,
        },
        "inputs": {
            "records": {
                "path": str(records_path),
                "sha256": file_sha256(records_path),
            },
            "sidecar": {
                "path": str(sidecar_path),
                "sha256": file_sha256(sidecar_path),
            },
            "assignments": {
                "path": str(assignments_path),
                "sha256": file_sha256(assignments_path),
            },
            "registry": {
                "path": str(registry_path),
                "sha256": file_sha256(registry_path),
            },
        },
        "outputs": output_meta,
    }
    summary_path = target / SUMMARY_FILENAME
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default=str(DEFAULT_V5_DIR / RECORDS_FILENAME))
    parser.add_argument("--sidecar", default=str(DEFAULT_SIDECAR))
    parser.add_argument("--assignments", default=str(DEFAULT_ASSIGNMENTS))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument(
        "--heldout",
        nargs="+",
        default=[str(DEFAULT_RANDOM_HELDOUT), str(DEFAULT_SCAFFOLD_HELDOUT)],
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_ANALYSIS_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = audit_policy_v2_calibration(
        records_path=args.records,
        sidecar_path=args.sidecar,
        assignments_path=args.assignments,
        registry_path=args.registry,
        heldout_paths=args.heldout,
        out_dir=args.out_dir,
    )
    counts = summary["assignment_and_exclusion_counts"]
    print(
        "[audit_starling_endpoint_policy_v2_calibration] "
        f"train_records={counts['train_only_records']:,} "
        f"eligible={counts['eligible_assay_transfer_records']:,} "
        f"out={args.out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "aggregate_pair_yields",
    "audit_policy_v2_calibration",
]
