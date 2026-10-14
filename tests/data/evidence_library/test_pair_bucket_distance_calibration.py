from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from data.processing.evidence_library.versions.v7.build_pair_bucket_distance_calibration import (
    CALIBRATION_VERSION,
    CATEGORY_CDF_VERSION,
    LEGACY_CALIBRATION_VERSION,
    _build_calibration_entries,
    _category_gate,
    _value_cdf,
    build_pair_bucket_distance_calibration,
    calibration_standard_deviation,
    category_cdf_percentile,
    category_cdf_separation,
    validate_pair_bucket_distance_calibration,
    value_cdf_percentile,
    value_cdf_separation,
)
from data.processing.evidence_library.versions.v7.tasks.bbb_martins.build_starling_pair_bucket_transfer_policy import (
    BUILD_SPEC,
)
from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_schema import (
    RECORD_CONTRACT as BBB_RECORD_CONTRACT,
)
from data.processing.evidence_library.versions.v7.tasks.skin_reaction.build_starling_pair_bucket_transfer_policy import (
    BUILD_SPEC as SKIN_BUILD_SPEC,
)
from data.processing.evidence_library.versions.v7.tasks.skin_reaction.starling_schema import RECORD_CONTRACT


def _binary_group(*, include_response: bool = True) -> pd.DataFrame:
    categories = ["no_response"] * 25
    ranks = [0] * 25
    if include_response:
        categories = ["no_response"] * 12 + ["response"] * 13
        ranks = [0] * 12 + [1] * 13
    return pd.DataFrame(
        {
            "canonical_category_id": categories,
            "canonical_category_rank": ranks,
        }
    )


def test_binary_gate_requires_both_declared_levels() -> None:
    complete = _category_gate(
        _binary_group(),
        record_contract=RECORD_CONTRACT,
        source_id="direct_skin_reaction",
        kind="binary",
        scale_id="single_subject_fraction.v1",
    )
    incomplete = _category_gate(
        _binary_group(include_response=False),
        record_contract=RECORD_CONTRACT,
        source_id="direct_skin_reaction",
        kind="binary",
        scale_id="single_subject_fraction.v1",
    )
    assert complete["valid"]
    assert incomplete == {"valid": False, "reason": "incomplete_binary_domain"}


def test_distance_calibration_worker_count_preserves_exact_entries() -> None:
    rows = []
    for bucket_index in range(120):
        count = 25 if bucket_index == 0 else 2
        for record_index in range(count):
            rows.append(
                {
                    "pair_bucket_key": f"bucket-{bucket_index:03d}",
                    "source_id": "direct_bbb",
                    "measurement_kind": "continuous",
                    "canonical_measurement_scale_id": None,
                    "canonical_category_id": None,
                    "canonical_category_rank": None,
                    "finite_scalar_value": float(record_index + 1),
                    "canonical_record_id": (
                        f"record-{bucket_index:03d}-{record_index:03d}"
                    ),
                    "canonical_smiles": f"C{bucket_index}N{record_index}",
                    "bbb_transport_label": None,
                    "qualifying_conditions": None,
                }
            )
    frame = pd.DataFrame(rows)
    serial = _build_calibration_entries(
        frame,
        spec=BUILD_SPEC,
        record_contract=BBB_RECORD_CONTRACT,
        minimum_samples=25,
        workers=1,
    )
    parallel = _build_calibration_entries(
        frame,
        spec=BUILD_SPEC,
        record_contract=BBB_RECORD_CONTRACT,
        minimum_samples=25,
        workers=2,
    )
    assert parallel == serial


def test_calibration_uses_twenty_records_and_sixteen_molecules() -> None:
    frame = pd.DataFrame(
        {
            "pair_bucket_key": "collapsed-bucket",
            "source_id": "direct_bbb",
            "measurement_kind": "continuous",
            "canonical_measurement_scale_id": None,
            "canonical_category_id": None,
            "canonical_category_rank": None,
            "finite_scalar_value": [float(index + 1) for index in range(20)],
            "canonical_record_id": [f"record-{index:03d}" for index in range(20)],
            "canonical_smiles": [f"C{index}" for index in range(20)],
        }
    )

    entry = _build_calibration_entries(
        frame,
        spec=BUILD_SPEC,
        record_contract=BBB_RECORD_CONTRACT,
        minimum_samples=20,
        workers=1,
    )["collapsed-bucket"]

    assert entry["minimum_support_met"] is True
    assert entry["calibration_valid"] is True
    assert entry["distinct_molecule_count"] == 20
    assert "residual_heterogeneity_gate" not in entry


def test_postcollapse_calibration_includes_eligible_direct_residuals(tmp_path) -> None:
    bucket = json.dumps(
        [
            "direct_bbb",
            "brain_uptake",
            "mg/L",
            "encoder",
            "context",
            "species",
            "no_reported_external_condition",
        ]
    )
    records = pd.DataFrame(
        {
            "canonical_record_id": [f"record-{index:03d}" for index in range(20)],
            "collapsed_record_id": [f"collapsed-{index:03d}" for index in range(20)],
            "canonical_smiles": ["C" * (index + 1) for index in range(20)],
            "retrieval_source_id": "direct_residual",
            "pair_bucket_key": bucket,
            "assay_transfer_eligible": True,
            "source_id": "direct_bbb",
            "finite_scalar_value": [float(index + 1) for index in range(20)],
            "measurement_kind": "continuous",
            "canonical_measurement_scale_id": None,
            "canonical_category_id": None,
            "canonical_category_rank": None,
        }
    )
    records_path = tmp_path / "records.parquet"
    sidecar_path = tmp_path / "pair_bucket_records.parquet"
    metadata_path = tmp_path / "pair_bucket_metadata.json"
    auxiliary_path = tmp_path / "auxiliary_manifest.json"
    records.to_parquet(records_path, index=False)
    records[["canonical_record_id", "pair_bucket_key"]].to_parquet(
        sidecar_path, index=False
    )
    metadata_path.write_text(
        json.dumps(
            {
                "contract_version": BUILD_SPEC.pair_bucket_version,
                "source_required_fields": {
                    source: list(fields)
                    for source, fields in BUILD_SPEC.source_pair_fields.items()
                },
            }
        ),
        encoding="utf-8",
    )
    auxiliary_path.write_text(
        json.dumps(
            {
                "mapping_version": BUILD_SPEC.auxiliary_mapping_version,
                "attachment_version": BUILD_SPEC.auxiliary_attachment_version,
                "output_fields": list(BUILD_SPEC.required_auxiliary_output_fields),
            }
        ),
        encoding="utf-8",
    )

    payload = build_pair_bucket_distance_calibration(
        spec=BUILD_SPEC,
        record_contract=BBB_RECORD_CONTRACT,
        records_path=records_path,
        pair_bucket_records_path=sidecar_path,
        pair_bucket_metadata_path=metadata_path,
        auxiliary_manifest_path=auxiliary_path,
        out_dir=tmp_path / "calibration",
        workers=1,
    )

    assert list(payload["buckets"]) == [bucket]
    assert payload["buckets"][bucket]["calibration_valid"] is True


def _continuous_entry(values: list[float], *, key: str = "continuous") -> dict:
    frame = pd.DataFrame(
        {
            "pair_bucket_key": key,
            "source_id": "direct_bbb",
            "measurement_kind": "continuous",
            "canonical_measurement_scale_id": None,
            "canonical_category_id": None,
            "canonical_category_rank": None,
            "finite_scalar_value": values,
            "canonical_record_id": [f"record-{index:03d}" for index in range(len(values))],
            "canonical_smiles": [f"C{index}" for index in range(len(values))],
            "bbb_transport_label": None,
            "qualifying_conditions": None,
        }
    )
    return _build_calibration_entries(
        frame,
        spec=BUILD_SPEC,
        record_contract=BBB_RECORD_CONTRACT,
        minimum_samples=25,
        workers=1,
    )[key]


def test_multi_source_lineage_uses_pair_bucket_scientific_source():
    key = json.dumps(
        [
            "direct_bbb",
            "permeability",
            "cm/s",
            "encoder",
            "context",
            "species",
            "no_reported_external_condition",
        ],
        separators=(",", ":"),
    )
    frame = pd.DataFrame(
        {
            "pair_bucket_key": key,
            "source_id": ["direct_bbb"] * 24 + ["multi_source"],
            "measurement_kind": "continuous",
            "canonical_measurement_scale_id": None,
            "canonical_category_id": None,
            "canonical_category_rank": None,
            "finite_scalar_value": [float(index + 1) for index in range(25)],
            "canonical_record_id": [f"record-{index:03d}" for index in range(25)],
            "canonical_smiles": [f"C{index}" for index in range(25)],
            "bbb_transport_label": None,
            "qualifying_conditions": None,
        }
    )

    entry = _build_calibration_entries(
        frame,
        spec=BUILD_SPEC,
        record_contract=BBB_RECORD_CONTRACT,
        minimum_samples=25,
        workers=1,
    )[key]

    assert entry["source_id"] == "direct_bbb"
    assert entry["calibration_valid"] is True


def _ordinal_entry(*, key: str = "ordinal") -> dict:
    category_ids = ["grade_0"] * 5 + ["grade_1"] * 10 + ["grade_2"] * 10
    ranks = [0] * 5 + [1] * 10 + [2] * 10
    frame = pd.DataFrame(
        {
            "pair_bucket_key": key,
            "source_id": "direct_skin_reaction",
            "measurement_kind": "ordinal",
            "canonical_measurement_scale_id": "ordinal_severity_grade",
            "canonical_category_id": category_ids,
            "canonical_category_rank": ranks,
            "finite_scalar_value": None,
            "canonical_record_id": [
                f"ordinal-{index:03d}" for index in range(len(ranks))
            ],
            "canonical_smiles": [f"C{index}" for index in range(len(ranks))],
            "dose_or_concentration": None,
            "extra_details": None,
        }
    )
    return _build_calibration_entries(
        frame,
        spec=SKIN_BUILD_SPEC,
        record_contract=RECORD_CONTRACT,
        minimum_samples=25,
        workers=1,
    )[key]


def _v2_payload(entry: dict, *, key: str = "continuous") -> dict:
    return {
        "calibration_version": CALIBRATION_VERSION,
        "record_contract_version": BBB_RECORD_CONTRACT.version,
        "buckets": {key: entry},
    }


def test_v2_persists_first_class_sd_and_exact_value_cdf_only() -> None:
    entry = _continuous_entry([10.0] * 8 + [15.0] * 9 + [20.0] * 8)

    assert entry["standard_deviation_valid"] is True
    assert entry["standard_deviation_ddof"] == 1
    assert entry["standard_deviation_value_field"] == "finite_scalar_value"
    assert calibration_standard_deviation(entry) == pytest.approx(
        entry["observed_sample_standard_deviation"]
    )
    assert entry["value_cdf_valid"] is True
    assert entry["value_cdf"]["support_values"] == [10.0, 15.0, 20.0]
    assert entry["value_cdf"]["support_counts"] == [8, 9, 8]
    assert entry["value_cdf"]["support_midranks_0_1"] == pytest.approx(
        [0.16, 0.5, 0.84]
    )
    assert entry["category_cdf_valid"] is False
    assert entry["category_cdf_reason"] == "non_ordinal_measurement"
    assert entry["category_cdf"] is None
    assert "distance_calibration" not in entry
    assert "percentile_knot_count" not in json.dumps(entry, sort_keys=True)
    validate_pair_bucket_distance_calibration(
        _v2_payload(entry), record_contract=BBB_RECORD_CONTRACT
    )


def test_context_heterogeneity_is_not_computed_or_used() -> None:
    values = [1.0] * 13 + [100.0] * 12
    frame = pd.DataFrame(
        {
            "pair_bucket_key": "heterogeneous",
            "source_id": "direct_bbb",
            "measurement_kind": "continuous",
            "canonical_measurement_scale_id": None,
            "canonical_category_id": None,
            "canonical_category_rank": None,
            "finite_scalar_value": values,
            "canonical_record_id": [f"record-{index:03d}" for index in range(25)],
            "canonical_smiles": [f"C{index}" for index in range(25)],
            "bbb_transport_label": None,
            "qualifying_conditions": ["low"] * 13 + ["high"] * 12,
        }
    )
    entry = _build_calibration_entries(
        frame,
        spec=BUILD_SPEC,
        record_contract=BBB_RECORD_CONTRACT,
        minimum_samples=25,
        workers=1,
    )["heterogeneous"]

    assert "residual_heterogeneity_gate" not in entry
    assert entry["calibration_valid"] is True
    assert entry["calibration_reason"] == "valid"
    assert entry["assay_transfer_bucket_eligible"] is True
    assert entry["assay_transfer_bucket_ineligibility_reason"] is None
    assert entry["standard_deviation_valid"] is True


def test_value_cdf_lookup_uses_midranks_and_empirical_unseen_values() -> None:
    cdf = _value_cdf([10.0, 10.0, 15.0, 20.0])

    assert value_cdf_percentile(9.0, cdf) == 0.0
    assert value_cdf_percentile(10.0, cdf) == 0.25
    assert value_cdf_percentile(12.0, cdf) == 0.5
    assert value_cdf_percentile(15.0, cdf) == 0.625
    assert value_cdf_percentile(18.0, cdf) == 0.75
    assert value_cdf_percentile(20.0, cdf) == 0.875
    assert value_cdf_percentile(21.0, cdf) == 1.0


def test_value_cdf_is_invariant_to_strictly_monotone_rescaling() -> None:
    original = _value_cdf([10.0, 10.0, 15.0, 20.0])
    transformed = _value_cdf([101.0, 101.0, 226.0, 401.0])

    for left, right in zip((10.0, 15.0, 20.0), (101.0, 226.0, 401.0)):
        assert value_cdf_percentile(left, original) == value_cdf_percentile(
            right, transformed
        )


def test_equal_raw_gaps_have_location_sensitive_cdf_separations() -> None:
    cdf = _value_cdf(
        [10.0] * 4 + [12.0] * 6 + [15.0] * 2 + [20.0] * 4 + [25.0] * 4
    )

    assert value_cdf_percentile(10.0, cdf) == pytest.approx(0.10)
    assert value_cdf_percentile(15.0, cdf) == pytest.approx(0.55)
    assert value_cdf_percentile(20.0, cdf) == pytest.approx(0.70)
    assert abs(
        value_cdf_percentile(10.0, cdf) - value_cdf_percentile(15.0, cdf)
    ) == pytest.approx(0.45)
    assert abs(
        value_cdf_percentile(15.0, cdf) - value_cdf_percentile(20.0, cdf)
    ) == pytest.approx(0.15)


def test_value_cdf_separation_requires_the_same_bucket() -> None:
    entry = _continuous_entry([10.0] * 8 + [15.0] * 9 + [20.0] * 8)
    payload = _v2_payload(entry)

    result = value_cdf_separation(
        payload,
        left_pair_bucket_key="continuous",
        left_value=10.0,
        right_pair_bucket_key="continuous",
        right_value=20.0,
    )
    assert result == pytest.approx(
        {
            "left_percentile_0_1": 0.16,
            "right_percentile_0_1": 0.84,
            "percentile_separation_0_1": 0.68,
        }
    )
    with pytest.raises(ValueError, match="same pair_bucket_key"):
        value_cdf_separation(
            payload,
            left_pair_bucket_key="continuous",
            left_value=10.0,
            right_pair_bucket_key="other",
            right_value=20.0,
        )


def test_ordinal_category_cdf_persists_declared_domain_and_midranks() -> None:
    entry = _ordinal_entry()
    cdf = entry["category_cdf"]

    assert entry["calibration_valid"] is True
    assert entry["value_cdf_valid"] is False
    assert entry["category_cdf_valid"] is True
    assert cdf["contract_version"] == CATEGORY_CDF_VERSION
    assert cdf["canonical_measurement_scale_id"] == "ordinal_severity_grade"
    assert [item["category_id"] for item in cdf["categories"]] == [
        "grade_0",
        "grade_1",
        "grade_2",
        "grade_3",
        "grade_4",
    ]
    assert [item["observed_count"] for item in cdf["categories"]] == [
        5,
        10,
        10,
        0,
        0,
    ]
    assert [item["midrank_0_1"] for item in cdf["categories"]] == pytest.approx(
        [0.1, 0.4, 0.8, 1.0, 1.0]
    )
    validate_pair_bucket_distance_calibration(
        {
            "calibration_version": CALIBRATION_VERSION,
            "record_contract_version": RECORD_CONTRACT.version,
            "buckets": {"ordinal": entry},
        },
        record_contract=RECORD_CONTRACT,
    )


def test_category_cdf_separation_uses_category_ids_and_same_bucket() -> None:
    entry = _ordinal_entry()
    payload = {
        "calibration_version": CALIBRATION_VERSION,
        "record_contract_version": RECORD_CONTRACT.version,
        "buckets": {"ordinal": entry},
    }

    assert category_cdf_percentile("grade_3", entry["category_cdf"]) == 1.0
    assert category_cdf_separation(
        payload,
        left_pair_bucket_key="ordinal",
        left_category_id="grade_0",
        right_pair_bucket_key="ordinal",
        right_category_id="grade_2",
    ) == pytest.approx(
        {
            "left_percentile_0_1": 0.1,
            "right_percentile_0_1": 0.8,
            "percentile_separation_0_1": 0.7,
        }
    )
    assert category_cdf_separation(
        payload,
        left_pair_bucket_key="ordinal",
        left_category_id="grade_1",
        right_pair_bucket_key="ordinal",
        right_category_id="grade_1",
    )["percentile_separation_0_1"] == 0.0
    with pytest.raises(ValueError, match="same pair_bucket_key"):
        category_cdf_separation(
            payload,
            left_pair_bucket_key="ordinal",
            left_category_id="grade_0",
            right_pair_bucket_key="other",
            right_category_id="grade_1",
        )
    with pytest.raises(ValueError, match="unknown category_id"):
        category_cdf_percentile("grade_5", entry["category_cdf"])


def test_invalid_or_categorical_buckets_do_not_publish_a_value_cdf() -> None:
    invalid = _continuous_entry([1.0] * 25)
    assert invalid["calibration_valid"] is False
    assert invalid["assay_transfer_bucket_eligible"] is False
    assert invalid["assay_transfer_bucket_ineligibility_reason"] == (
        "nonpositive_or_nonfinite_sample_sd"
    )
    assert invalid["standard_deviation_valid"] is False
    assert invalid["standard_deviation_reason"] == "nonpositive_or_nonfinite_sample_sd"
    assert invalid["value_cdf_valid"] is False
    assert invalid["value_cdf"] is None

    undersupported = _continuous_entry([float(index) for index in range(10)])
    assert undersupported["standard_deviation_reason"] == "fewer_than_25_records"
    assert undersupported["observed_sample_standard_deviation"] is None

    categorical = _binary_group()
    categorical["pair_bucket_key"] = "binary"
    categorical["source_id"] = "direct_skin_reaction"
    categorical["measurement_kind"] = "binary"
    categorical["canonical_measurement_scale_id"] = "single_subject_fraction.v1"
    categorical["finite_scalar_value"] = None
    categorical["canonical_record_id"] = [
        f"binary-{index:03d}" for index in range(len(categorical))
    ]
    categorical["canonical_smiles"] = [
        f"C{index}" for index in range(len(categorical))
    ]
    categorical["dose_or_concentration"] = None
    categorical["extra_details"] = None
    entries = _build_calibration_entries(
        categorical,
        spec=SKIN_BUILD_SPEC,
        record_contract=RECORD_CONTRACT,
        minimum_samples=25,
        workers=1,
    )
    entry = entries["binary"]
    assert calibration_standard_deviation(entry) > 0
    assert entry["standard_deviation_value_field"] == "canonical_category_rank"
    assert entry["value_cdf_valid"] is False
    assert entry["value_cdf_reason"] == "non_continuous_measurement"
    assert entry["value_cdf"] is None
    assert entry["category_cdf_valid"] is False
    assert entry["category_cdf_reason"] == "non_ordinal_measurement"
    assert entry["category_cdf"] is None
    validate_pair_bucket_distance_calibration(
        {
            "calibration_version": CALIBRATION_VERSION,
            "record_contract_version": RECORD_CONTRACT.version,
            "buckets": {"binary": entry},
        },
        record_contract=RECORD_CONTRACT,
    )


def test_v2_rejects_raw_distance_cdf_fields_and_malformed_cdf() -> None:
    entry = _continuous_entry([float(index) for index in range(25)])
    raw_distance = copy.deepcopy(_v2_payload(entry))
    raw_distance["buckets"]["continuous"]["distance_calibration"] = {
        "standardized_distance_percentile_knots": [0.0, 1.0]
    }
    with pytest.raises(ValueError, match="raw-distance CDF"):
        validate_pair_bucket_distance_calibration(
            raw_distance, record_contract=BBB_RECORD_CONTRACT
        )

    malformed = copy.deepcopy(_v2_payload(entry))
    malformed["buckets"]["continuous"]["value_cdf"]["support_counts"][0] += 1
    with pytest.raises(ValueError, match="record count mismatch"):
        validate_pair_bucket_distance_calibration(
            malformed, record_contract=BBB_RECORD_CONTRACT
        )

    ordinal = _ordinal_entry()
    malformed_category = {
        "calibration_version": CALIBRATION_VERSION,
        "record_contract_version": RECORD_CONTRACT.version,
        "buckets": {"ordinal": copy.deepcopy(ordinal)},
    }
    malformed_category["buckets"]["ordinal"]["category_cdf"]["categories"][0][
        "observed_count"
    ] += 1
    with pytest.raises(ValueError, match="midrank mismatch|record count mismatch"):
        validate_pair_bucket_distance_calibration(
            malformed_category, record_contract=RECORD_CONTRACT
        )


def test_validator_accepts_archived_v1_shape() -> None:
    payload = {
        "calibration_version": LEGACY_CALIBRATION_VERSION,
        "record_contract_version": BBB_RECORD_CONTRACT.version,
        "buckets": {
            "legacy": {
                "calibration_valid": True,
                "distance_calibration": {
                    "standardized_distance_percentile_knots": [
                        float(index) for index in range(101)
                    ]
                },
            }
        },
    }

    validate_pair_bucket_distance_calibration(
        payload, record_contract=BBB_RECORD_CONTRACT
    )
