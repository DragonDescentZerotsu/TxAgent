from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.build_starling_pair_bucket_transfer_policy import (
    POLICY_FILENAME,
    build_pair_bucket_transfer_policy,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.data_processing.auxiliary_mapping_helpers.reconciliation import (
    MAPPING_VERSION,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.starling_auxiliary_metadata import (
    AUXILIARY_ATTACHMENT_VERSION,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.starling_pair_bucket_transfer_policy import (
    MIN_ASSAY_TRANSFER_SAMPLES,
    SOURCE_CANDIDATE_FIELDS,
    PAIR_BUCKET_TRANSFER_POLICY_VERSION,
    automatic_variance_gate,
    evaluate_pair_bucket_transfer,
    load_pair_bucket_transfer_policy,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.starling_pair_buckets import (
    BIOAVAILABILITY_PAIR_BUCKET_VERSION,
    SOURCE_PAIR_FIELDS,
)


def test_automatic_gate_is_omega_intersection_with_iqr_sd_union():
    assert automatic_variance_gate(
        omega_squared=0.20,
        median_range_iqr=1.00,
        median_range_sd=0.0,
    )
    assert automatic_variance_gate(
        omega_squared=0.20,
        median_range_iqr=0.0,
        median_range_sd=1.35,
    )
    assert not automatic_variance_gate(
        omega_squared=0.19,
        median_range_iqr=10.0,
        median_range_sd=10.0,
    )
    assert not automatic_variance_gate(
        omega_squared=0.20,
        median_range_iqr=0.99,
        median_range_sd=1.34,
    )


def test_combined_policy_uses_25_records_gate_and_one_sd_boundary(tmp_path):
    candidate_columns = sorted(
        {field for fields in SOURCE_CANDIDATE_FIELDS.values() for field in fields}
    )
    records: list[dict] = []
    buckets: list[dict] = []

    def add_bucket(key: str, values: list[float], species: list[str | None]) -> None:
        for index, (value, species_value) in enumerate(zip(values, species)):
            record_id = f"{key}-{index:03d}"
            row = {
                "normalized_record_id": record_id,
                "finite_scalar_value": value,
                **{column: None for column in candidate_columns},
            }
            row["species_or_population"] = species_value
            records.append(row)
            buckets.append(
                {
                    "normalized_record_id": record_id,
                    "source_id": "hf_bioavailability",
                    "canonical_endpoint": "absolute_bioavailability",
                    "canonical_unit": "%",
                    "pair_bucket_key": key,
                    "bucket_eligible": True,
                }
            )

    add_bucket("below", list(range(24)), [None] * 24)
    add_bucket("clean", list(range(25)), [None] * 25)
    add_bucket("zero-sd", [5.0] * 25, [None] * 25)
    add_bucket(
        "variance",
        [0.0] * 13 + [100.0] * 12,
        ["human"] * 13 + ["rat"] * 12,
    )

    records_path = tmp_path / "records.parquet"
    buckets_path = tmp_path / "pair_bucket_records.parquet"
    pair_metadata_path = tmp_path / "pair_bucket_metadata.json"
    auxiliary_path = tmp_path / "auxiliary_mapping_manifest.json"
    pd.DataFrame(records).to_parquet(records_path, index=False)
    pd.DataFrame(buckets).to_parquet(buckets_path, index=False)
    pair_metadata_path.write_text(
        json.dumps(
            {
                "contract_version": BIOAVAILABILITY_PAIR_BUCKET_VERSION,
                "source_required_fields": {
                    source: list(fields)
                    for source, fields in SOURCE_PAIR_FIELDS.items()
                },
            }
        ),
        encoding="utf-8",
    )
    auxiliary_path.write_text(
        json.dumps(
            {
                "mapping_version": MAPPING_VERSION,
                "mapping_sha256": "fixture",
                "attachment_version": AUXILIARY_ATTACHMENT_VERSION,
                "output_fields": ["global_context", "global_species_context"],
            }
        ),
        encoding="utf-8",
    )

    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    payload = build_pair_bucket_transfer_policy(
        records_path=records_path,
        pair_bucket_records_path=buckets_path,
        pair_bucket_metadata_path=pair_metadata_path,
        auxiliary_manifest_path=auxiliary_path,
        out_dir=first_output,
    )
    build_pair_bucket_transfer_policy(
        records_path=records_path,
        pair_bucket_records_path=buckets_path,
        pair_bucket_metadata_path=pair_metadata_path,
        auxiliary_manifest_path=auxiliary_path,
        out_dir=second_output,
    )
    assert (first_output / POLICY_FILENAME).read_bytes() == (
        second_output / POLICY_FILENAME
    ).read_bytes()
    loaded = load_pair_bucket_transfer_policy(first_output / POLICY_FILENAME)
    assert loaded == payload

    assert MIN_ASSAY_TRANSFER_SAMPLES == 25
    assert payload["buckets"]["below"]["eligibility_reason"] == (
        "fewer_than_25_records"
    )
    assert payload["buckets"]["zero-sd"]["eligibility_reason"] == (
        "nonpositive_or_nonfinite_sample_sd"
    )
    assert payload["buckets"]["variance"]["eligibility_reason"] == (
        "automatic_variance_gate"
    )
    assert payload["buckets"]["variance"]["variance_gate"][
        "candidate_column"
    ] == "species_or_population"
    assert payload["buckets"]["clean"]["assay_transfer_eligible"] is True

    sd = payload["buckets"]["clean"]["distance_policy"][
        "sample_standard_deviation"
    ]
    identical = evaluate_pair_bucket_transfer(payload, "clean", 10.0, 10.0)
    at_boundary = evaluate_pair_bucket_transfer(payload, "clean", 0.0, sd)
    beyond = evaluate_pair_bucket_transfer(payload, "clean", 0.0, 1.001 * sd)
    assert identical["distance_percentile_0_100"] == 0.0
    assert at_boundary["standardized_difference_sd"] == pytest.approx(1.0)
    assert at_boundary["assay_transfer_label"] is True
    assert beyond["assay_transfer_label"] is False
    assert 0.0 <= at_boundary["distance_percentile_0_100"] <= 100.0
    assert payload["buckets"]["clean"]["distance_policy"][
        "one_sd_percentile_0_100"
    ] == pytest.approx(at_boundary["distance_percentile_0_100"])

    unscored = evaluate_pair_bucket_transfer(payload, "variance", 0.0, 100.0)
    assert unscored["assay_transfer_eligible"] is False
    assert unscored["assay_transfer_label"] is None
    assert "measurement_subtype" not in json.dumps(payload, sort_keys=True)


def test_frozen_full_policy_counts_when_local_artifact_is_available():
    path = Path(
        "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
        "starling_normalized_v6/05_assay_transfer_policy/"
        "pair_bucket_transfer_policy.json.gz"
    )
    if not path.exists():
        pytest.skip("restored full normalized-v6 policy artifact is unavailable")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        stored = json.load(handle)
    if stored.get("policy_version") != PAIR_BUCKET_TRANSFER_POLICY_VERSION:
        pytest.skip(
            "local normalized-v6 policy predates the unified HF source schema"
        )
    payload = load_pair_bucket_transfer_policy(path)
    assert payload["summary"] == {
        "assay_transfer_eligible_buckets": 289,
        "assay_transfer_eligible_records": 199595,
        "eligibility_reason_counts": {
            "automatic_variance_gate": 76,
            "eligible": 289,
            "fewer_than_25_records": 4856,
        },
        "minimum_support_buckets": 365,
        "minimum_support_records": 226913,
        "pair_buckets": 5221,
        "records_in_pair_buckets": 246385,
        "variance_gate_flagged_buckets": 76,
        "variance_gate_flagged_records": 27318,
    }
    assert all(
        math.isfinite(entry["distance_policy"]["sample_standard_deviation"])
        and entry["distance_policy"]["sample_standard_deviation"] > 0
        for entry in payload["buckets"].values()
        if entry["assay_transfer_eligible"]
    )
