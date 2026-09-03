"""Behavior checks for the BBB V8 measurement rules."""

import json

from data.processing.evidence_library.versions.v8.measurement_routing import (
    attach_stage1_routes,
)
from data.processing.evidence_library.versions.v8.tasks.bbb_martins.measurement_resolution_rules import (
    route_measurement,
)
from data.processing.evidence_library.versions.v8.tasks.bbb_martins.starling_categorical_response import (
    POLICY as CATEGORICAL_POLICY,
)
from data.processing.evidence_library.versions.v8.unit_vocabulary import (
    DEFAULT_VOCABULARY_PATH,
    build_unit_vocabulary,
    clean_unit,
    load_unit_vocabulary,
)


def _route(measurement, unit=None, **fields):
    return route_measurement(
        {"measurement_text": measurement, "unit_text": unit, **fields}
    )


def test_simple_source_exact_rules() -> None:
    assert _route("0", "cm/s").measurement_text == "0"
    assert _route("-6.36", "cm/s").measurement_text == "-6.36"
    assert _route("87 %", "%").measurement_text == "87"
    assert _route("8 +/- 1.5", "ng/mL").measurement_text == "8"
    assert _route("45 ± 5%", "%").measurement_text == "45"
    scaled = _route("3.1 × 10^-6", "cm/s")
    assert scaled.measurement_text == "3.1"
    assert scaled.unit_text == "× 10^-6 cm/s"
    logbb = _route(
        "-0.42",
        source_id="direct_bbb",
        canonical_endpoint_name="logbb",
    )
    assert logbb.unit_text == "log10_ratio"


def test_only_whole_field_bounds_and_ranges_are_rejected() -> None:
    assert _route("<5", "ng/mL").rule_id == "hard_bound_no_point.v2"
    assert _route("7-12", "%").rule_id == "explicit_range_no_point.v2"
    assert _route("<5 after treatment", "ng/mL").bucket == "extract"
    assert _route("range: 7-12", "%").rule_id == "multiple_numeric_candidates.v3"
    assert _route("dose = 10 mg/kg").bucket == "extract"


def test_km_is_a_measurement_and_ordinary_109_is_not_a_power_of_ten() -> None:
    km = _route("Km = 2.2 µM", source_id="influx_transport")
    assert km.bucket == "extract"
    assert _route("109", source_id="efflux_transport").bucket == "extract"


def test_uncertainty_and_inline_value_rules_reject_extra_prose_or_numbers() -> None:
    assert _route("8 ± 1.5", "ng/mL").bucket == "accept"
    assert _route("mean 8 ± 1.5", "ng/mL").bucket == "extract"
    assert _route("8 ± 1.5 SD", "ng/mL").bucket == "extract"
    assert _route("8 + / - 1.5", "ng/mL").bucket != "accept"
    assert _route("(8 ± 1.5)", "ng/mL").bucket != "accept"
    assert _route("8 ± 1.5 mg", "ng/mL").bucket == "extract"
    assert _route("15 ng/mL / 570 ng/mL").bucket == "reject"
    assert _route("4.7 ng/mL").bucket == "accept"
    assert _route("4.7 invented_unit").bucket == "extract"
    assert _route("Observed influx = 5 nmol/g/min").bucket == "extract"


def test_observed_unit_vocabulary_matches_all_declared_source_columns() -> None:
    payload = json.loads(DEFAULT_VOCABULARY_PATH.read_text(encoding="utf-8"))
    assert payload == build_unit_vocabulary()
    assert {
        (source["task"], source["source_id"], source["unit_column"])
        for source in payload["sources"]
    } == {
        ("bbb_martins", "direct_bbb", "quant_units"),
        ("bbb_martins", "passive_permeability", "metric_units"),
        ("bioavailability_ma", "oral_exposure", "parameter_units"),
        ("bioavailability_ma", "fa", "reported_units"),
        ("bioavailability_ma", "fh", "reported_units"),
        ("skin_reaction", "sensitization_aop", "result_unit"),
        ("skin_reaction", "skin_exposure", "result_unit"),
    }
    units = load_unit_vocabulary()
    assert {"%", "cm/s", "ng/mL", "ratio", "µM"} <= units
    assert clean_unit("  ng/\t mL  ") == "ng/ mL"


def test_multiple_outputs_are_rejected_but_spread_is_not_a_second_output() -> None:
    assert _route("plasma 54 ng/mL; CSF 94 ng/mL").rule_id == (
        "multiple_numeric_candidates.v3"
    )
    assert _route("8 ± 1.5", "ng/mL").bucket == "accept"


def test_stage1_does_not_use_the_categorical_encoder_for_routing() -> None:
    [record] = attach_stage1_routes(
        [
            {
                "source_id": "efflux_transport",
                "endpoint_name": "Efflux transport",
                "measurement_text": None,
                "interaction_conclusion": "substrate",
            }
        ],
        task="bbb_martins",
    )
    assert record["measurement_resolution_route"] == "reject"
    encoded = CATEGORICAL_POLICY.apply(record)
    assert encoded["categorical_encoder_id"] == "efflux_substrate_binary.v1"
