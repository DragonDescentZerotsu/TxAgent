import pandas as pd

from tools.chembl_tool.tasks.bioavailability_ma.plot_starling_canonical_endpoint_bucket_distributions import (
    canonical_endpoint_bucket_sizes,
    canonical_endpoint_unit_counts,
    distribution_summary,
)


def test_canonical_endpoint_buckets_ignore_unit_and_context_fragmentation():
    sidecar = pd.DataFrame(
        [
            {"source_id": "fg", "canonical_endpoint": "efflux", "canonical_unit": "ratio", "bucket_eligible": True},
            {"source_id": "fg", "canonical_endpoint": "efflux", "canonical_unit": "%", "bucket_eligible": True},
            {"source_id": "fg", "canonical_endpoint": "metabolism", "canonical_unit": "/min", "bucket_eligible": True},
            {"source_id": "fg", "canonical_endpoint": "metabolism", "canonical_unit": "/h", "bucket_eligible": False},
        ]
    )

    sizes = canonical_endpoint_bucket_sizes(sidecar)
    summary = distribution_summary(sizes)

    assert sizes.set_index("canonical_endpoint")["n_records"].to_dict() == {
        "efflux": 2,
        "metabolism": 1,
    }
    assert summary["n_records"] == 3
    assert summary["n_canonical_endpoint_buckets"] == 2
    assert summary["sources"]["fg"]["bucket_size_band_counts"]["1"] == 1
    units = canonical_endpoint_unit_counts(sidecar)
    assert units[["canonical_endpoint", "canonical_unit", "n_records"]].to_dict(
        orient="records"
    ) == [
        {"canonical_endpoint": "efflux", "canonical_unit": "%", "n_records": 1},
        {"canonical_endpoint": "efflux", "canonical_unit": "ratio", "n_records": 1},
        {"canonical_endpoint": "metabolism", "canonical_unit": "/min", "n_records": 1},
    ]
