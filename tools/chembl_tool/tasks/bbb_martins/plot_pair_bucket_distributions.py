"""Plot every numeric BBB pair-bucket distribution on one normalized axis."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from tools.chembl_tool.common.starling.final_endpoint_pruning import supported_gap
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


DEFAULT_ROOT = Path(
    "outputs/chembl_tool/tasks/bbb_martins/evidence_library/starling_normalized_v7"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path, default=DEFAULT_ROOT / "03_pair_buckets/records.parquet"
    )
    parser.add_argument(
        "--pair-buckets",
        type=Path,
        default=DEFAULT_ROOT / "03_pair_buckets/pair_bucket_records.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/chembl_tool/tasks/bbb_martins/analysis/pair_bucket_distributions"),
    )
    parser.add_argument("--bins", type=int, default=160)
    args = parser.parse_args(argv)
    if args.bins < 20:
        parser.error("--bins must be at least 20")

    buckets = pd.read_parquet(
        args.pair_buckets,
        columns=[
            "canonical_record_id",
            "pair_bucket_key",
            "canonical_unit_text",
            "canonical_endpoint_name",
            "source_id",
            "measurement_kind",
        ],
    )
    records = pd.read_parquet(
        args.records,
        columns=[
            "canonical_record_id",
            "finite_scalar_value",
            "measurement_resolution_input_unit",
            "measurement_unit_mapping_scale",
        ],
    )
    rows = buckets.merge(records, on="canonical_record_id", validate="one_to_one")
    keyed = rows[rows["pair_bucket_key"].notna()].copy()
    keyed["finite_scalar_value"] = pd.to_numeric(
        keyed["finite_scalar_value"], errors="coerce"
    )
    numeric = keyed[
        (keyed["measurement_kind"] == "continuous")
        & np.isfinite(keyed["finite_scalar_value"])
    ].copy()

    diagnostics: list[dict[str, object]] = []
    histograms: list[np.ndarray] = []
    for key, group in numeric.groupby("pair_bucket_key", sort=True):
        values = group["finite_scalar_value"].to_numpy(dtype=float)
        transformed = values.copy()
        order = np.argsort(transformed)
        transformed = transformed[order]
        span = float(transformed[-1] - transformed[0]) if len(values) > 1 else 0.0
        positions = (
            (transformed - transformed[0]) / span
            if span > 0
            else np.full(len(values), 0.5)
        )
        histogram = np.bincount(
            np.minimum((positions * args.bins).astype(int), args.bins - 1),
            minlength=args.bins,
        ).astype(float)
        histogram = np.log1p(histogram)
        histogram /= histogram.max()
        histograms.append(histogram)

        unit_text = str(group["canonical_unit_text"].iloc[0] or "")
        gap = supported_gap(values, unit_text=unit_text)
        gap_lower = gap["gap_lower_value"] if gap else None
        gap_upper = gap["gap_upper_value"] if gap else None
        gap_decades = gap["gap_decades"] if gap else None
        gap_fraction = (
            float(gap_decades / span)
            if gap and span > 0 and np.all(values > 0)
            else None
        )

        input_units = Counter(
            str(value)
            for value in group["measurement_resolution_input_unit"].dropna()
        )
        unit_medians = (
            group.dropna(subset=["measurement_resolution_input_unit"])
            .groupby("measurement_resolution_input_unit")["finite_scalar_value"]
            .median()
        )
        unit_median_span_decades = None
        if len(unit_medians) > 1 and np.all(unit_medians > 0):
            unit_median_span_decades = float(
                math.log10(unit_medians.max() / unit_medians.min())
            )
        mapping_scales = Counter(
            str(value)
            for value in group["measurement_unit_mapping_scale"].dropna()
        )
        first = group.iloc[0]
        diagnostics.append(
            {
                "pair_bucket_key": key,
                "source_id": first["source_id"],
                "canonical_endpoint_name": first["canonical_endpoint_name"],
                "canonical_unit": first["canonical_unit_text"],
                "record_count": len(values),
                "distinct_value_count": len(np.unique(values)),
                "normalization": "persisted_assay_transfer_geometry_then_minmax",
                "minimum_value": float(np.min(values)),
                "maximum_value": float(np.max(values)),
                "largest_supported_gap_fraction": gap_fraction,
                "gap_lower_value": gap_lower,
                "gap_upper_value": gap_upper,
                "gap_decades": gap_decades,
                "input_unit_count": len(input_units),
                "input_unit_counts_json": json.dumps(
                    input_units, ensure_ascii=False, sort_keys=True
                ),
                "input_unit_medians_json": json.dumps(
                    {str(key): float(value) for key, value in unit_medians.items()},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "input_unit_median_span_decades": unit_median_span_decades,
                "mapping_scale_counts_json": json.dumps(mapping_scales, sort_keys=True),
            }
        )

    table = pd.DataFrame(diagnostics)
    image = np.vstack(histograms)
    table["_sort_gap"] = table["gap_decades"].fillna(
        table["largest_supported_gap_fraction"].fillna(-1)
    )
    order = table.sort_values(
        ["_sort_gap", "record_count", "pair_bucket_key"],
        ascending=[False, False, True],
    ).index.to_numpy()
    table = table.loc[order].drop(columns="_sort_gap").reset_index(drop=True)
    table.insert(0, "plot_row", np.arange(1, len(table) + 1))
    image = image[order]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "bbb_pair_bucket_distribution_rows.csv"
    plot_path = args.output_dir / "bbb_pair_bucket_distributions.png"
    summary_path = args.output_dir / "summary.json"
    table.to_csv(csv_path, index=False)

    gap_panel = table["largest_supported_gap_fraction"].fillna(0).to_numpy(
        dtype=float
    )[:, None]
    unit_span_panel = table["input_unit_median_span_decades"].fillna(0).to_numpy(
        dtype=float
    )[:, None]
    unit_panel = np.log10(1 + table["input_unit_count"].to_numpy())[:, None]
    size_panel = np.log10(table["record_count"].to_numpy())[:, None]
    figure, axes = plt.subplots(
        1, 5, figsize=(26, 46), gridspec_kw={"width_ratios": [24, 1, 1, 1, 1]}
    )
    extent = (0, 1, len(table), 0)
    axes[0].imshow(
        image,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0,
        vmax=1,
        extent=extent,
    )
    axes[0].set_xlabel(
        "Within-bucket position: persisted assay-transfer geometry, then min-max to [0, 1]"
    )
    axes[0].set_ylabel("Bucket row (sorted by largest supported gap; see CSV)")
    for axis, panel, title, cmap in (
        (axes[1], gap_panel, "gap\nfraction", "inferno"),
        (axes[2], unit_span_panel, "input-unit median\nspan (decades)", "inferno"),
        (axes[3], unit_panel, "log10(1 +\ninput units)", "viridis"),
        (axes[4], size_panel, "log10(n)", "viridis"),
    ):
        panel = np.asarray(panel, dtype=float)
        axis.imshow(
            panel,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            extent=(0, 1, len(table), 0),
        )
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    kind_counts = keyed.groupby("measurement_kind")["pair_bucket_key"].nunique().to_dict()
    figure.suptitle(
        "BBB active numeric pair-bucket distributions\n"
        f"All {len(table):,} continuous buckets with finite values are shown; "
        "singletons/constants appear at x=0.5",
        fontsize=18,
    )
    figure.text(
        0.5,
        0.004,
        f"Not magnitude-plotted: {kind_counts.get('binary', 0):,} binary and "
        f"{kind_counts.get('non_scalar', 0):,} non-scalar buckets. "
        "A supported gap leaves at least max(2, 5%) observations on each side.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.012, 1, 0.985))
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)

    summary = {
        "records_path": str(args.records),
        "records_sha256": file_sha256(args.records),
        "pair_buckets_path": str(args.pair_buckets),
        "pair_buckets_sha256": file_sha256(args.pair_buckets),
        "keyed_bucket_counts_by_measurement_kind": {
            str(key): int(value) for key, value in sorted(kind_counts.items())
        },
        "plotted_continuous_buckets": len(table),
        "plotted_records": int(table["record_count"].sum()),
        "buckets_with_supported_gap": int(table["largest_supported_gap_fraction"].notna().sum()),
        "buckets_with_multiple_input_units": int((table["input_unit_count"] > 1).sum()),
        "buckets_with_input_unit_medians_at_least_one_decade_apart": int(
            (table["input_unit_median_span_decades"] >= 1).sum()
        ),
        "plot": str(plot_path),
        "plot_sha256": file_sha256(plot_path),
        "row_table": str(csv_path),
        "row_table_sha256": file_sha256(csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
