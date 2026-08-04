"""Categorical evidence must become a continuous measurement, honestly."""

import math

import pytest

from tools.chembl_tool.common.starling.categorical_response import (
    ENCODED_UNITS,
    LOGIT_RESPONSE_UNIT,
    ORDINAL_SEVERITY_UNIT,
    SIGNED_DIRECTION_UNIT,
    count_logit,
    encoded_unit_validity_status,
    logit,
    shrunk_rate,
    sigmoid,
)
from tools.chembl_tool.tasks.skin_reaction.starling_categorical_response import (
    POLICY,
    SIGNED_DIRECTION_ANCHORS,
    encoding_policy_manifest,
    parse_severity_grade,
)
from tools.chembl_tool.tasks.skin_reaction.starling_record_canonicalization import (
    normalization_validity_status,
)


def _direct(**overrides):
    return {
        "source_id": "direct_skin_reaction",
        "structure_status": "resolved",
        "canonical_smiles": "CCO",
        "canonical_endpoint": "sensitization",
        **overrides,
    }


def _photo(**overrides):
    return {
        "source_id": "phototoxicity_irritation_local_damage",
        "structure_status": "resolved",
        "canonical_smiles": "CCO",
        "canonical_endpoint": "skin_irritation",
        **overrides,
    }


# --- shrinkage -------------------------------------------------------------


@pytest.mark.parametrize("k,n", [(0, 1), (1, 1), (0, 10), (10, 10), (0, 500), (500, 500)])
def test_shrinkage_keeps_every_rate_strictly_inside_the_unit_interval(k, n):
    rate = shrunk_rate(k, n)
    assert 0.0 < rate < 1.0
    assert math.isfinite(count_logit(k, n))


def test_degenerate_rates_are_symmetric_not_infinite():
    assert count_logit(0, 1) == pytest.approx(-count_logit(1, 1))
    assert count_logit(0, 10) == pytest.approx(-count_logit(10, 10))


def test_sample_size_moves_the_encoded_value_away_from_the_midpoint():
    """A 1/1 report must not claim as much as a 100/100 report."""
    assert abs(count_logit(1, 1)) < abs(count_logit(10, 10)) < abs(count_logit(100, 100))


def test_a_half_rate_encodes_to_zero_regardless_of_n():
    assert count_logit(5, 10) == pytest.approx(0.0)
    assert count_logit(50, 100) == pytest.approx(0.0)


def test_logit_refuses_an_unshrunk_boundary():
    for bad in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError):
            logit(bad)


def test_sigmoid_inverts_logit_and_never_overflows():
    for value in (-800.0, -3.0, 0.0, 3.0, 800.0):
        probability = sigmoid(value)
        assert 0.0 <= probability <= 1.0
    assert sigmoid(logit(0.3)) == pytest.approx(0.3)
    assert sigmoid(0.0) == pytest.approx(0.5)


def test_counts_outside_their_denominator_are_rejected():
    with pytest.raises(ValueError):
        shrunk_rate(5, 2)
    with pytest.raises(ValueError):
        shrunk_rate(1, 0)


# --- encoder selection -----------------------------------------------------


def test_real_incidence_beats_a_severity_grade():
    encoding = POLICY.encode(
        _direct(positive_count=3, total_tested=40, effect_metric="++")
    )
    assert encoding.encoder_id == "count_logit"
    assert encoding.sample_size == 40


def test_single_subject_counts_are_quarantined_from_real_incidence():
    """A 1/1 report must not share a bucket with a 45/50 one."""
    single = POLICY.encode(_direct(positive_count=1, total_tested=1))
    many = POLICY.encode(_direct(positive_count=45, total_tested=50))
    assert single.encoder_id == "single_subject_logit"
    assert many.encoder_id == "count_logit"
    assert single.encoder_id != many.encoder_id


def test_a_severity_grade_is_not_an_incidence_rate():
    """The + ladder grades one subject's reaction, so it gets its own scale."""
    encoding = POLICY.encode(_direct(effect_metric="+++"))
    assert encoding.encoder_id == "ordinal_severity_grade"
    assert encoding.unit == ORDINAL_SEVERITY_UNIT
    assert encoding.unit != LOGIT_RESPONSE_UNIT
    assert encoding.value == 3.0


def test_reconciled_explicit_severity_is_a_fallback_for_free_text():
    encoding = POLICY.encode(
        _direct(effect_metric="mild erythematous response", global_severity_grade="1")
    )
    assert encoding.encoder_id == "ordinal_severity_grade"
    assert encoding.value == 1.0
    assert encoding.inputs["grade_source"] == "global_severity_grade"


def test_local_explicit_grade_takes_precedence_over_reconciled_fallback():
    encoding = POLICY.encode(_direct(effect_metric="+++", global_severity_grade="1"))
    assert encoding.value == 3.0
    assert encoding.inputs["grade_source"] == "effect_metric"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("-", 0), ("negative", 0), ("no reaction", 0), ("0", 0),
        ("+", 1), ("+ reaction", 1), ("1+", 1),
        ("++", 2), ("++ (strong positive)", 2), ("2+ reaction", 2),
        ("+++", 3), ("++++", 4),
    ],
)
def test_severity_ladder_parses_reviewed_forms(text, expected):
    assert parse_severity_grade(text) == expected


@pytest.mark.parametrize("text", ["", "positive reaction", "weak sensitizer", "3.5", "n/a"])
def test_severity_ladder_refuses_free_text(text):
    assert parse_severity_grade(text) is None


def test_a_reported_percentage_encodes_as_a_rate():
    encoding = POLICY.encode(_direct(effect_metric="8% positive"))
    assert encoding.encoder_id == "percent_positive_logit"
    assert encoding.value == pytest.approx(logit(0.08))


def test_a_reported_zero_percent_is_finite():
    encoding = POLICY.encode(_direct(effect_metric="0% positive"))
    assert math.isfinite(encoding.value)
    assert encoding.value < 0
    assert encoding.inputs["nominal_sample_size_used"] is True


def test_an_embedded_fraction_is_treated_as_real_counts():
    encoding = POLICY.encode(_direct(effect_metric="19/85 positive"))
    assert encoding.encoder_id == "count_logit"
    assert encoding.sample_size == 85


def test_total_tested_stored_as_a_string_still_parses():
    """The source column is str-typed; a naive numeric read would drop it."""
    encoding = POLICY.encode(_direct(positive_count=2.0, total_tested="20"))
    assert encoding is not None
    assert encoding.sample_size == 20


# --- signed direction ------------------------------------------------------


def test_protective_is_the_opposite_of_positive_not_a_weaker_positive():
    protective = POLICY.encode(_photo(result_label="protective")).value
    negative = POLICY.encode(_photo(result_label="negative")).value
    positive = POLICY.encode(_photo(result_label="positive")).value
    assert protective < negative < positive
    assert protective == -positive


def test_signed_direction_uses_its_own_scale():
    encoding = POLICY.encode(_photo(result_label="positive"))
    assert encoding.unit == SIGNED_DIRECTION_UNIT
    assert encoding.unit not in {LOGIT_RESPONSE_UNIT, ORDINAL_SEVERITY_UNIT}


@pytest.mark.parametrize(
    "label", ["not_classified", "mixed_or_inconclusive", "positivetext>", "", "..."]
)
def test_uninformative_and_junk_labels_get_no_value(label):
    """Absence of information must never be encoded as evidence of no effect."""
    assert POLICY.encode(_photo(result_label=label)) is None


def test_inconclusive_direct_outcomes_get_no_value():
    assert POLICY.encode(_direct(outcome_label="inconclusive")) is None


# --- integration with the record contract ----------------------------------


def test_an_encoding_never_overwrites_a_real_measurement():
    record = _direct(positive_count=3, total_tested=40, finite_scalar_value=1.5)
    assert POLICY.apply(record) == {}


def test_applied_fields_make_the_record_bucket_eligible():
    record = _direct(positive_count=3, total_tested=40)
    record.update(POLICY.apply(record))
    assert record["normalization_validity_status"] == "valid"
    assert record["canonical_unit"] in ENCODED_UNITS
    assert record["is_absolute_and_continuous"] is True
    assert record["categorical_encoder_id"] == "count_logit"


def test_a_negative_log_odds_is_valid_not_a_domain_violation():
    """Half of all encoded values are negative; the physical domains must not apply."""
    record = _direct(positive_count=1, total_tested=40)
    record.update(POLICY.apply(record))
    assert record["finite_scalar_value"] < 0
    assert normalization_validity_status(record) == "valid"


def test_a_negative_severity_grade_is_rejected():
    record = _direct(
        canonical_unit=ORDINAL_SEVERITY_UNIT,
        categorical_encoder_id="ordinal_severity_grade",
        finite_scalar_value=-1.0,
    )
    assert normalization_validity_status(record) == "nonpositive_positive_scalar"


def test_an_encoded_unit_without_an_encoder_is_rejected():
    record = _direct(canonical_unit=LOGIT_RESPONSE_UNIT, finite_scalar_value=1.0)
    assert encoded_unit_validity_status(record) == "encoded_unit_without_encoder"


def test_encoders_are_confined_to_their_own_source():
    assert POLICY.encode(_photo(positive_count=3, total_tested=40)) is None
    assert POLICY.encode(_direct(result_label="protective")) is None


def test_manifest_declares_every_encoder_and_its_scale():
    manifest = encoding_policy_manifest()
    ids = {entry["encoder_id"] for entry in manifest["encoders"]}
    assert ids == {
        "count_logit",
        "single_subject_logit",
        "ordinal_severity_grade",
        "percent_positive_logit",
        "signed_direction",
    }
    assert set(manifest["uninformative_labels_receive_no_value"]) >= {
        "inconclusive",
        "not_classified",
        "mixed_or_inconclusive",
    }
    assert manifest["encoder_id_is_part_of_the_pair_bucket_key"] is True
    assert SIGNED_DIRECTION_ANCHORS["protective"] < SIGNED_DIRECTION_ANCHORS["positive"]
