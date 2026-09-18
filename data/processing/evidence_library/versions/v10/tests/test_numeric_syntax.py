"""Numeric grouping agrees across bare values, units, and uncertainty terms."""

import unittest

from data.processing.evidence_library.versions.v10.numeric_syntax import (
    finite_point_text,
    number_text,
)
from data.processing.evidence_library.versions.v10.tasks.bioavailability_ma.canonical_source import direct_measurement_fields
from data.processing.evidence_library.versions.v10.tasks.bioavailability_ma.starling_fg_scalar_rules import propose_fg_scalar
from data.processing.evidence_library.versions.v10.tasks.bioavailability_ma.starling_measurement_resolution import route_measurement


CASES = [
    ("1,234", "1234"), ("1 , 234", "1234"),
    ("1, 234, 567.89", "1234567.89"), ("1,\u00a0234", "1234"),
    ("0", "0"), ("-0.25", "-0.25"), ("1.2e-6", "1.2e-6"),
    ("1,23", None), ("1,2345", None), ("1, 234, 56", None),
    ("1, 2 34", None), ("1_000", None), ("NaN", None),
    ("12,34.5", None), (True, None),
]


def check_unit_and_uncertainty_grouping():
    row = {
        "source_id": "fa",
        "measurement_text": "1, 234 ± 1, 000",
        "unit_text": "mg/mL",
    }
    decision = route_measurement(row)
    assert decision.measurement_text == "1234"
    assert decision.rule_id == "finite_point_with_separate_unit.v1"
    assert row["measurement_text"] == "1, 234 ± 1, 000"
    row["measurement_text"] = "1, 23"
    assert route_measurement(row).bucket == "extract"
    assert direct_measurement_fields("1, 234 ± 1, 000%")['numeric_value'] == 1234
    assert direct_measurement_fields("1, 234 ± 1, 00%")['numeric_value'] is None
    assert direct_measurement_fields("1, 23 ± 1, 000%")['numeric_value'] is None
    assert propose_fg_scalar("efflux ratio = 1, 234 ± 1, 000").accepted
    assert not propose_fg_scalar("efflux ratio = 1, 234 ± 1, 00").accepted


class NumericSyntaxTests(unittest.TestCase):
    def test_whole_number_grouping(self):
        for text, expected in CASES:
            with self.subTest(text=text):
                self.assertEqual(number_text(text), expected)

    def test_unit_and_uncertainty_grouping(self):
        check_unit_and_uncertainty_grouping()

    def test_complete_finite_point(self):
        cases = (
            ("1, 234", ("1234", None)),
            ("-1, 234.5 +/- +2.0", ("-1234.5", "+2.0")),
            (".5 ± 1e-2", (".5", "1e-2")),
            ("1 ±", None),
            ("1 +/-", None),
            ("1 ± 1,23", None),
            ("1 ± inf", None),
            ("1 ± 2 mg", None),
        )
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(finite_point_text(value), expected)

    def test_separate_unit_acceptance_does_not_depend_on_sign_or_endpoint(self):
        for source in ("oral_exposure", "fa", "fh"):
            for value, expected in (("0", "0"), ("-1, 234.5", "-1234.5"), ("+2", "+2")):
                with self.subTest(source=source, value=value):
                    row = {"source_id": source, "measurement_text": value,
                           "unit_text": "unmapped source unit",
                           "canonical_endpoint_name": "unmapped endpoint"}
                    decision = route_measurement(row)
                    self.assertEqual(decision.bucket, "accept")
                    self.assertEqual(decision.measurement_text, expected)
                    self.assertEqual(decision.unit_text, "unmapped source unit")
                    self.assertEqual(decision.rule_id, "finite_point_with_separate_unit.v1")
        for value, unit in (
            ("1,23", "mg"),
            ("2", " "),
            ("1e999", "mg"),
            ("inf", "mg"),
            ("2 ± 1,23", "mg"),
        ):
            with self.subTest(value=value, unit=unit):
                decision = route_measurement({"source_id": "fa", "measurement_text": value, "unit_text": unit})
                self.assertNotEqual(decision.bucket, "accept")
