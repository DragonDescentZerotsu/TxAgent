from __future__ import annotations

import pytest

from tools.chembl_tool.common.starling import build_context_bioavailability_continuous as subject


def test_calibration_uses_only_named_train_nodes() -> None:
    records = [
        {"benchmark_row_id": "train", "record_vote": 0,
         "raw_observed_continuous_value": 5.0},
        {"benchmark_row_id": "train", "record_vote": 1,
         "raw_observed_continuous_value": 60.0},
        {"benchmark_row_id": "test", "record_vote": 0,
         "raw_observed_continuous_value": 99.0},
    ]
    result = subject._calibration(records, {"train"})
    assert result["class_means"] == {"0": 5.0, "1": 60.0}
    assert result["observed_record_counts"] == {"0": 1, "1": 1}


@pytest.mark.parametrize(
    ("method", "expected"),
    [("lower_bound", "v7_train_class_mean_one_sided_bound"),
     ("train_class_mean_categorical", "v7_train_class_mean_categorical")],
)
def test_missing_values_use_vote_class_mean(method: str, expected: str) -> None:
    row = {"record_vote": 1, "raw_observed_continuous_value": None,
           "raw_contribution_method": method}
    result = subject._contribution(row, {"0": 5.0, "1": 60.0})
    assert result["continuous_value_contribution"] == 60.0
    assert result["continuous_value_contribution_method"] == expected


def test_node_mean_weights_each_record_once() -> None:
    records = [
        {"voting_record_key": "a", "observed_continuous_value": 10.0,
         "continuous_value_contribution": 10.0},
        {"voting_record_key": "b", "observed_continuous_value": None,
         "continuous_value_contribution": 60.0},
    ]
    result = subject._enrich({"benchmark_row_id": "node"}, records)
    assert result["continuous_value_mean"] == 35.0
    assert result["voting_record_type"] == "mixed"
    assert result["observed_record_count"] == result["imputed_record_count"] == 1
