from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from data.processing.evidence_library.versions.v10.measurement_routing import (
    attach_stage1_routes,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_categorical_response import (
    POLICY as CATEGORICAL_POLICY,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_policy import (
    EXPECTED_SOURCE_ROWS,
    POLICY,
    endpoint_inventory,
    source_profiles,
    validate_arguments,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_schema import (
    ROLE_FIELDS,
    SOURCE_COLUMNS,
)


def test_four_source_contract_is_pinned_and_stage1_only():
    assert sum(EXPECTED_SOURCE_ROWS.values()) == 1_510_447
    assert set(POLICY.record_contract.sources) == set(EXPECTED_SOURCE_ROWS)
    assert set(SOURCE_COLUMNS) == set(EXPECTED_SOURCE_ROWS)
    assert POLICY.stage1_measurement_routing_enabled
    assert not POLICY.measurement_resolution_enabled
    assert set(POLICY.record_contract.measurement_scales) == {
        "ames_mutagenicity_ordinal.v1",
        "ames_fixed_mutation_ordinal.v1",
        "ames_damage_direction.v1",
        "ames_mechanism_detection.v1",
        "ames_mechanism_direction.v1",
    }


def test_stage2_build_is_explicitly_blocked():
    parser = argparse.ArgumentParser()
    try:
        validate_arguments(parser, argparse.Namespace(through_stage="normalize"))
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("Stage 2 unexpectedly accepted")


def test_source_roles_preserve_each_layer_schema():
    profiles = {profile.source_id: profile for profile in source_profiles(Path("root"))}
    for source_id, (endpoint, measurement, unit) in ROLE_FIELDS.items():
        profile = profiles[source_id]
        assert profile.endpoint_field == endpoint
        assert profile.measurement_field == measurement
        assert profile.unit_field == unit
        assert profile.smiles_field == "SMILES"
        assert profile.structure_mode == "direct"


def test_stage1_scalar_routing_covers_reject_accept_and_extract():
    rows = attach_stage1_routes(
        [
            {
                "source_id": "mutagenicity_outcomes",
                "endpoint_name": "bacterial_reverse_mutation",
                "measurement_text": "positive",
                "unit_text": None,
            },
            {
                "source_id": "fixed_mutation",
                "endpoint_name": "micronucleus_assay",
                "measurement_text": "12",
                "unit_text": "%",
                "result_call": "positive",
            },
            {
                "source_id": "premutagenic_damage",
                "endpoint_name": "comet_assay",
                "measurement_text": "3 to 5",
                "unit_text": "%",
                "result_status": "quantitative_result_without_directional_call",
            },
            {
                "source_id": "fixed_mutation",
                "endpoint_name": "micronucleus_assay",
                "measurement_text": "equivocal",
                "unit_text": None,
                "result_call": "equivocal",
            },
        ],
        task="ames",
    )
    assert [row["measurement_resolution_route"] for row in rows] == [
        "reject",
        "accept",
        "extract",
        "reject",
    ]
    assert rows[0]["measurement_resolution_rule_id"] == (
        "no_digit_in_measurement_column.v1"
    )
    assert CATEGORICAL_POLICY.apply(rows[0])["categorical_encoder_id"] == (
        "ames_mutagenicity_ordinal.v1"
    )
    assert rows[1]["measurement_resolution_exact_measurement"] == "12"
    assert rows[1]["measurement_resolution_exact_unit"] == "%"


def test_ambiguous_controlled_calls_abstain():
    assert not CATEGORICAL_POLICY.apply(
        {
            "source_id": "mutagenicity_outcomes",
            "measurement_text": "conflicting",
            "finite_scalar_value": None,
        }
    )
    assert not CATEGORICAL_POLICY.apply(
        {
            "source_id": "mutagenicity_mechanism",
            "result_direction": "mixed_or_condition_dependent",
            "finite_scalar_value": None,
        }
    )


def test_endpoint_inventory_fails_closed_on_unknown_endpoint():
    with pytest.raises(ValueError, match="unmapped Ames endpoint"):
        endpoint_inventory(
            "fixed_mutation",
            ["micronucleus_assay", "new_assay_family"],
            strict=True,
        )
