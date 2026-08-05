"""Plot candidate v6 bucket sizes using source + canonical_endpoint only."""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.chembl_tool.tasks.bioavailability_ma.build_normalized_starling_evidence_library import (
    DEFAULT_OUT_DIR,
)
from tools.chembl_tool.tasks.bioavailability_ma.build_starling_pair_bucket_sidecar import (
    PAIR_BUCKET_RECORDS_FILENAME,
)


SOURCE_ORDER = ("oral_exposure", "fa", "fg", "fh", "hf_bioavailability")
SOURCE_LABELS = {
    "oral_exposure": "Oral exposure",
    "fa": "Fa",
    "fg": "FG",
    "fh": "FH",
    "hf_bioavailability": "HF bioavailability",
}
SOURCE_COLORS = {
    "oral_exposure": "#3B6FB6",
    "fa": "#2A9D8F",
    "fg": "#E07A5F",
    "fh": "#7A5195",
    "hf_bioavailability": "#D4A72C",
}

DEFAULT_ROOT = Path(DEFAULT_OUT_DIR)
DEFAULT_SIDECAR = DEFAULT_ROOT / "04_pair_buckets" / PAIR_BUCKET_RECORDS_FILENAME
DEFAULT_OUT_DIR = DEFAULT_ROOT / "analysis" / "canonical_endpoint_bucket_distributions"


def canonical_endpoint_bucket_sizes(sidecar: pd.DataFrame) -> pd.DataFrame:
    """Return one candidate-v6 bucket per source and canonical endpoint."""
    required = {"source_id", "canonical_endpoint", "bucket_eligible"}
    missing = required - set(sidecar.columns)
    if missing:
        raise ValueError(f"pair-bucket sidecar is missing columns: {sorted(missing)}")
    eligible = sidecar[sidecar["bucket_eligible"].fillna(False)].copy()
    if eligible["canonical_endpoint"].isna().any():
        raise ValueError("eligible record has no canonical_endpoint")
    sizes = (
        eligible.groupby(["source_id", "canonical_endpoint"], sort=True, observed=True)
        .size()
        .rename("n_records")
        .reset_index()
    )
    sizes["candidate_bucket_key"] = sizes.apply(
        lambda row: json.dumps(
            [row["source_id"], row["canonical_endpoint"]],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        axis=1,
    )
    return sizes


def canonical_endpoint_unit_counts(sidecar: pd.DataFrame) -> pd.DataFrame:
    """Return every eligible source/endpoint/unit combination and its record count."""
    required = {"source_id", "canonical_endpoint", "canonical_unit", "bucket_eligible"}
    missing = required - set(sidecar.columns)
    if missing:
        raise ValueError(f"pair-bucket sidecar is missing columns: {sorted(missing)}")
    eligible = sidecar[sidecar["bucket_eligible"].fillna(False)].copy()
    return (
        eligible.groupby(
            ["source_id", "canonical_endpoint", "canonical_unit"],
            sort=True,
            observed=True,
            dropna=False,
        )
        .size()
        .rename("n_records")
        .reset_index()
        .sort_values(
            ["source_id", "canonical_endpoint", "n_records", "canonical_unit"],
            ascending=[True, True, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def distribution_summary(sizes: pd.DataFrame) -> dict[str, Any]:
    bands = (
        (1, 1, "1"),
        (2, 4, "2-4"),
        (5, 9, "5-9"),
        (10, 24, "10-24"),
        (25, 49, "25-49"),
        (50, 99, "50-99"),
        (100, 249, "100-249"),
        (250, 999, "250-999"),
        (1000, np.inf, "1000+"),
    )
    sources: dict[str, Any] = {}
    for source_id in SOURCE_ORDER:
        values = sizes.loc[sizes["source_id"].eq(source_id), "n_records"]
        if values.empty:
            continue
        sources[source_id] = {
            "n_records": int(values.sum()),
            "n_canonical_endpoint_buckets": int(len(values)),
            "minimum": int(values.min()),
            "p25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "mean": float(values.mean()),
            "p75": float(values.quantile(0.75)),
            "p90": float(values.quantile(0.90)),
            "p95": float(values.quantile(0.95)),
            "p99": float(values.quantile(0.99)),
            "maximum": int(values.max()),
            "bucket_size_band_counts": {
                label: int(((values >= lower) & (values <= upper)).sum())
                for lower, upper, label in bands
            },
        }
    return {
        "candidate_bucket_definition": "source_id + canonical_endpoint",
        "eligibility": "existing pair-bucket bucket_eligible records only",
        "n_records": int(sizes["n_records"].sum()),
        "n_canonical_endpoint_buckets": int(len(sizes)),
        "sources": sources,
    }


def plot_distribution(sizes: pd.DataFrame, output_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    positions = np.arange(len(SOURCE_ORDER))
    data = [
        sizes.loc[sizes["source_id"].eq(source), "n_records"].to_numpy()
        for source in SOURCE_ORDER
    ]
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.5},
        whiskerprops={"color": "#666666"},
        capprops={"color": "#666666"},
    )
    for patch, source in zip(box["boxes"], SOURCE_ORDER):
        patch.set_facecolor(SOURCE_COLORS[source])
        patch.set_alpha(0.28)
        patch.set_edgecolor(SOURCE_COLORS[source])
    rng = np.random.default_rng(20260731)
    for position, source, values in zip(positions, SOURCE_ORDER, data):
        jitter = rng.uniform(-0.16, 0.16, len(values))
        ax.scatter(
            position + jitter,
            values,
            s=22,
            color=SOURCE_COLORS[source],
            alpha=0.75,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
    ax.set_yscale("log")
    ax.set_xticks(positions, [SOURCE_LABELS[source] for source in SOURCE_ORDER])
    ax.set_ylabel("Records per canonical-endpoint bucket (log scale)")
    ax.set_title("Candidate normalized-v6 bucket sizes by source")
    ax.grid(axis="y", which="both", color="#D8D8D8", linewidth=0.6, alpha=0.7)
    _caption(fig, "Bucket key = source_id + canonical_endpoint; current pair-eligible scalar records only.")
    _save(fig, output_base)


def plot_rank_size(sizes: pd.DataFrame, output_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for source in SOURCE_ORDER:
        values = np.sort(
            sizes.loc[sizes["source_id"].eq(source), "n_records"].to_numpy()
        )[::-1]
        ranks = np.arange(1, len(values) + 1)
        ax.plot(
            ranks,
            values,
            marker="o",
            markersize=3.8,
            linewidth=1.6,
            color=SOURCE_COLORS[source],
            label=f"{SOURCE_LABELS[source]} (n={len(values)})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Canonical-endpoint bucket rank within source (log scale)")
    ax.set_ylabel("Records in bucket (log scale)")
    ax.set_title("Rank–size distribution of candidate normalized-v6 buckets")
    ax.grid(which="both", color="#D8D8D8", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False, fontsize=9)
    _caption(fig, "Endpoints are ranked independently within each source from largest to smallest.")
    _save(fig, output_base)


def plot_top_endpoints(sizes: pd.DataFrame, output_base: Path, *, top_n: int = 15) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 13.0))
    for ax, source in zip(axes.flat, SOURCE_ORDER):
        subset = (
            sizes[sizes["source_id"].eq(source)]
            .nlargest(top_n, "n_records")
            .sort_values("n_records")
        )
        labels = [
            "\n".join(textwrap.wrap(str(value).replace("_", " "), width=27))
            for value in subset["canonical_endpoint"]
        ]
        ax.barh(
            np.arange(len(subset)),
            subset["n_records"],
            color=SOURCE_COLORS[source],
            alpha=0.88,
        )
        ax.set_yticks(np.arange(len(subset)), labels, fontsize=8.5)
        ax.set_xscale("log")
        ax.set_xlabel("Records (log scale)")
        ax.set_title(
            f"{SOURCE_LABELS[source]}: top {min(top_n, len(subset))} of {len(sizes[sizes['source_id'].eq(source)])} endpoints"
        )
        ax.grid(axis="x", which="both", color="#D8D8D8", linewidth=0.6, alpha=0.7)
    axes.flat[-1].axis("off")
    fig.suptitle("Largest canonical-endpoint buckets within each source", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0.025, 1, 0.98))
    fig.text(
        0.5,
        0.008,
        "Bucket key = source_id + canonical_endpoint; current pair-eligible scalar records only.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    _save(fig, output_base, tight=False)


def _caption(fig: plt.Figure, value: str) -> None:
    fig.text(0.5, 0.012, value, ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.035, 1, 1))


def _save(fig: plt.Figure, output_base: Path, *, tight: bool = True) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(output_base.with_suffix(".svg"), **kwargs)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, **kwargs)
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", default=str(DEFAULT_SIDECAR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args(argv)

    sidecar_path = Path(args.sidecar)
    out_dir = Path(args.out_dir)
    sidecar = pd.read_parquet(
        sidecar_path,
        columns=[
            "source_id", "canonical_endpoint", "canonical_unit", "bucket_eligible"
        ],
    )
    sizes = canonical_endpoint_bucket_sizes(sidecar)
    unit_counts = canonical_endpoint_unit_counts(sidecar)
    summary = distribution_summary(sizes)
    summary["input"] = {
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": _sha256(sidecar_path),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes.to_parquet(out_dir / "canonical_endpoint_bucket_sizes.parquet", index=False)
    unit_counts.to_parquet(
        out_dir / "canonical_endpoint_unit_counts.parquet", index=False
    )
    unit_counts.to_csv(out_dir / "canonical_endpoint_unit_counts.csv", index=False)
    for source_id in SOURCE_ORDER:
        unit_counts[unit_counts["source_id"].eq(source_id)].to_csv(
            out_dir / f"{source_id}_canonical_endpoint_unit_counts.csv",
            index=False,
        )
    (out_dir / "canonical_endpoint_bucket_distribution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_distribution(sizes, out_dir / "canonical_endpoint_bucket_size_distribution")
    plot_rank_size(sizes, out_dir / "canonical_endpoint_bucket_rank_size")
    plot_top_endpoints(
        sizes, out_dir / "canonical_endpoint_bucket_top_endpoints", top_n=args.top_n
    )
    print(
        "[plot_starling_canonical_endpoint_bucket_distributions] "
        f"records={summary['n_records']:,} "
        f"buckets={summary['n_canonical_endpoint_buckets']:,} out={out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
