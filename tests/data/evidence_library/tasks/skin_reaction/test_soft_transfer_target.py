"""The soft transfer target must never disagree with the boolean it softens."""

import math

import pytest

from data.processing.evidence_library.shared.v1.pair_bucket_transfer_policy import (
    SOFT_TRANSFER_MIDPOINT_SD,
    SOFT_TRANSFER_TEMPERATURE,
    TRANSFER_MAX_STANDARD_DEVIATIONS,
    soft_transfer_contract,
    soft_transfer_probability,
)


def test_the_midpoint_is_the_boolean_decision_boundary():
    assert SOFT_TRANSFER_MIDPOINT_SD == TRANSFER_MAX_STANDARD_DEVIATIONS
    assert soft_transfer_probability(TRANSFER_MAX_STANDARD_DEVIATIONS) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "distance", [0.0, 0.1, 0.5, 0.9, 0.999, 1.0, 1.001, 1.5, 3.0, 50.0]
)
def test_soft_and_hard_targets_agree_on_which_side_of_the_boundary(distance):
    soft = soft_transfer_probability(distance)
    hard = distance <= TRANSFER_MAX_STANDARD_DEVIATIONS
    assert 0.0 <= soft <= 1.0
    if distance == TRANSFER_MAX_STANDARD_DEVIATIONS:
        assert soft == pytest.approx(0.5)
    else:
        assert (soft > 0.5) == hard


def test_the_target_decreases_monotonically_with_distance():
    values = [soft_transfer_probability(d) for d in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)]
    assert all(left > right for left, right in zip(values, values[1:]))


def test_the_frozen_temperature_places_the_reviewed_reference_points():
    assert soft_transfer_probability(0.5) == pytest.approx(0.9, abs=1e-6)
    assert soft_transfer_probability(1.5) == pytest.approx(0.1, abs=1e-6)


def test_a_colder_temperature_approaches_the_boolean_step():
    warm = soft_transfer_probability(1.2, temperature=SOFT_TRANSFER_TEMPERATURE)
    cold = soft_transfer_probability(1.2, temperature=0.01)
    assert cold < warm
    assert cold == pytest.approx(0.0, abs=1e-6)
    assert soft_transfer_probability(0.8, temperature=0.01) == pytest.approx(1.0, abs=1e-6)


def test_extreme_distances_do_not_overflow():
    assert soft_transfer_probability(1e6) == pytest.approx(0.0)
    assert math.isfinite(soft_transfer_probability(1e6))
    assert soft_transfer_probability(0.0) < 1.0


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_an_invalid_distance_is_rejected(bad):
    with pytest.raises(ValueError):
        soft_transfer_probability(bad)


def test_a_nonpositive_temperature_is_rejected():
    with pytest.raises(ValueError):
        soft_transfer_probability(1.0, temperature=0.0)


def test_the_contract_is_self_describing_and_honest_about_calibration():
    contract = soft_transfer_contract()
    assert contract["value_at_midpoint"] == 0.5
    assert contract["agrees_with_boolean_at_midpoint"] is True
    assert contract["replaces_boolean_label"] is False
    # The repo is explicit elsewhere that these scores are not calibrated
    # probabilities; the soft target must not quietly claim otherwise.
    assert contract["is_a_calibrated_probability"] is False
    assert contract["reference_points"]["1.0_sd"] == pytest.approx(0.5)
