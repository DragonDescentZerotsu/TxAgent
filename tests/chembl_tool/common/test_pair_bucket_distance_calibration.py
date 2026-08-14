from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from tools.chembl_tool.common.starling.build_pair_bucket_distance_calibration import (
    CALIBRATION_VERSION,
    CATEGORY_CDF_VERSION,
    LEGACY_CALIBRATION_VERSION,
    _build_calibration_entries,
    _bias_corrected_cramers_v_squared,
    _category_gate,
    _score_categorical_candidate,
    _select_categorical_variance_candidate,
    _value_cdf,
    calibration_standard_deviation,
    category_cdf_percentile,
    category_cdf_separation,
    validate_pair_bucket_distance_calibration,
    value_cdf_percentile,
    value_cdf_separation,
)
from tools.chembl_tool.tasks.bbb_martins.build_starling_pair_bucket_transfer_policy import (
    BUILD_SPEC,
)
from tools.chembl_tool.tasks.bbb_martins.starling_schema import (
    RECORD_CONTRACT as BBB_RECORD_CONTRACT,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_schema import (
    RECORD_CONTRACT as BIO_RECORD_CONTRACT,
)
from tools.chembl_tool.tasks.skin_reaction.build_starling_pair_bucket_transfer_policy import (
    BUILD_SPEC as SKIN_BUILD_SPEC,
)
from tools.chembl_tool.tasks.skin_reaction.starling_schema import RECORD_CONTRACT


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
        scale_id="single_subject_logit",
    )
    incomplete = _category_gate(
        _binary_group(include_response=False),
        record_contract=RECORD_CONTRACT,
        source_id="direct_skin_reaction",
        kind="binary",
        scale_id="single_subject_logit",
    )
    assert complete["valid"]
    assert incomplete == {"valid": False, "reason": "incomplete_binary_domain"}


def test_categorical_residual_gate_uses_bias_corrected_cramers_v_squared() -> None:
    group = _binary_group()
    group["assay_batch"] = ["batch_0"] * 12 + ["batch_1"] * 13
    score = _score_categorical_candidate(group, "assay_batch")
    assert score is not None
    assert score["cramers_v_squared"] >= 0.20
    assert score["variance_gate_flagged"]

    independent = pd.DataFrame([[10.0, 10.0], [10.0, 10.0]])
    assert _bias_corrected_cramers_v_squared(independent.to_numpy()) == 0.0


def test_categorical_residual_gate_excludes_the_selected_scale_inputs() -> None:
    group = _binary_group()
    group["substrate_status"] = ["not_substrate"] * 12 + ["substrate"] * 13
    group["transporter_or_enzyme"] = ["ABCB1"] * 25
    group["intestinal_site"] = None
    group["qualifying_conditions"] = None
    result = _select_categorical_variance_candidate(
        group,
        record_contract=BIO_RECORD_CONTRACT,
        source_id="fg",
        scale_id="fg_substrate_status_binary.v1",
    )
    assert result["excluded_controlled_input_fields"] == [
        "substrate_status",
        "transporter_or_enzyme",
    ]
    assert result["candidate_column"] == "__none__"


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
    categorical["canonical_measurement_scale_id"] = "single_subject_logit"
    categorical["finite_scalar_value"] = None
    categorical["canonical_record_id"] = [
        f"binary-{index:03d}" for index in range(len(categorical))
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
