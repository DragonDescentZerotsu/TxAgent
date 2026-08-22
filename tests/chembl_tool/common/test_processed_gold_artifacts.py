import math
from pathlib import Path

import pytest

from tools.chembl_tool.common.starling.build_processed_gold_artifacts import (
    TaskSpec,
    VotingRecord,
    _enrich_molecule,
    _json_safe,
    _prepare_contributions,
    _validate_votes,
    scalarize_bioavailability_vote,
)
from tools.chembl_tool.common.starling.build_processed_gold_retrieval_view import (
    _record as retrieval_record,
)


def _spec(kind: str = "bioavailability") -> TaskSpec:
    return TaskSpec(
        key="test",
        task="Test",
        lineage="test_v1",
        source_path=Path("source.parquet"),
        gold_root=Path("gold"),
        output_root=Path("output"),
        source_kind=kind,
        expected_source_sha256="0" * 64,
    )


def _vote(
    index: int,
    vote: int,
    molecule: str = "molecule",
    method: str = "canonical:explicit_qualitative_high",
    raw_value: str = "high",
) -> VotingRecord:
    return VotingRecord(
        source_index=index,
        source_id="source",
        source_record_id=f"record:{index}",
        pmid="1",
        label_method=method,
        raw_value=raw_value,
        vote=vote,
        molecule_key=molecule,
    )


@pytest.mark.parametrize(
    ("method", "raw_value", "expected"),
    [
        ("reported_point", "25%", 25.0),
        ("reported_mean_plus_minus", "40 +/- 10%", 40.0),
        ("reported_range", "40-60%", 50.0),
        ("lower_bound", ">=40%", None),
        ("upper_bound", "<10%", None),
    ],
)
def test_bioavailability_scalarization(method, raw_value, expected):
    record = _vote(
        0,
        int(expected is not None and expected >= 20),
        method=f"canonical:numeric_20_percent_threshold:{method}",
        raw_value=raw_value,
    )
    assert scalarize_bioavailability_vote(record) == expected


def test_bioavailability_conversion_uses_train_only_class_means():
    votes = [
        _vote(0, 0, "train_0", "canonical:numeric_20_percent_threshold:reported_point", "10%"),
        _vote(1, 1, "train_1", "canonical:numeric_20_percent_threshold:reported_range", "40-80%"),
        _vote(2, 1, "train_1", raw_value="high"),
        _vote(3, 0, "valid_0", "canonical:numeric_20_percent_threshold:reported_point", "1%"),
        _vote(4, 1, "valid_0", "canonical:numeric_20_percent_threshold:lower_bound", ">30%"),
    ]
    gold = {
        "train_0": {"split": "train"},
        "train_1": {"split": "train"},
        "valid_0": {"split": "valid"},
    }
    policy = _prepare_contributions(_spec(), votes, gold)
    assert policy["class_means"] == {"0": 10.0, "1": 60.0}
    assert policy["calibration_counts"] == {"0": 1, "1": 1}
    assert votes[2].contribution == 60.0
    assert votes[2].contribution_method == "train_class_mean_categorical"
    assert votes[4].contribution == 60.0
    assert votes[4].contribution_method == "train_class_mean_one_sided_bound"


def test_binary_vote_mean_preserves_minority_votes_and_uncapped_mapping():
    votes = [_vote(index, 0 if index == 59 else 1) for index in range(60)]
    gold = {
        "drug": "CC",
        "molecule_identity_key": "molecule",
        "split": "train",
        "Y": 1,
        "source_record_count": 60,
        "label_counts": {"0": 1, "1": 59},
    }
    _prepare_contributions(_spec("bbb"), votes, {"molecule": gold})
    enriched = _enrich_molecule(_spec("bbb"), gold, votes)
    assert enriched["voting_record_type"] == "categorical_only"
    assert enriched["voting_record_count"] == 60
    assert enriched["continuous_value_mean"] == pytest.approx(59 / 60)
    assert enriched["continuous_value_unit"] == "positive_vote_fraction"
    assert len(enriched["voting_record_keys"]) == 60
    assert len(set(enriched["voting_record_keys"])) == 60


def test_vote_reconstruction_requires_all_majority_and_minority_records():
    gold = {
        "molecule": {
            "source_record_count": 4,
            "label_counts": {"0": 1, "1": 3},
        }
    }
    votes = [_vote(index, 0 if index == 3 else 1) for index in range(4)]
    grouped = _validate_votes(gold, votes)
    assert len(grouped["molecule"]) == 4
    with pytest.raises(ValueError, match="reconstruction failed"):
        _validate_votes(gold, votes[:-1])


def test_json_safe_preserves_nested_values_and_replaces_nan():
    payload = {
        "array": (1, 2),
        "nested": {"missing": float("nan")},
        "finite": 3.5,
    }
    converted = _json_safe(payload)
    assert converted == {
        "array": [1, 2],
        "nested": {"missing": None},
        "finite": 3.5,
    }
    assert not math.isnan(converted["finite"])


def test_processed_voter_becomes_one_cache_record_with_frozen_labels():
    voter = {
        "voting_record_key": "BBB_Martins:source_row:9",
        "source_lineage": "experimental_meaningful_cns_access_v2",
        "split": "train",
        "drug": "CCO",
        "record_vote": 0,
        "molecule_Y": 1,
        "voting_value_type": "categorical",
        "continuous_value_contribution": 0.0,
        "source_id": "starling-labs/BBB",
        "source_record_id": "row:9",
        "source_row_index": 9,
        "source_record": {
            "quant_metric": "brain concentration",
            "quant_value": "2",
            "quant_units": "ng/mL",
            "bbb_permeability_label": "poor_penetration",
            "smiles": "CCO",
        },
    }
    record = retrieval_record("bbb_martins", voter)
    assert record["canonical_record_id"] == voter["voting_record_key"]
    assert record["record_vote"] == 0
    assert record["molecule_Y"] == 1
    assert record["endpoint_name"] == "brain concentration"
    assert record["measurement_text"] == "2"
    assert record["unit_text"] == "ng/mL"
