"""Hand-validated regression corpus for normalization and parsing defects.

`fixtures/normalization_regressions.jsonl` holds one entry per real record that exposed a
class of normalization bug, together with the output a human confirmed to be correct and
the reasoning behind it.

Unlike `units_golden.json`, this corpus is **hand-written and never machine-regenerated**.
There is deliberately no `--write` flag. A failure here means either a real regression or a
decision that must be re-validated by hand -- never an expectation to be overwritten to get
back to green. See the "Normalization regression corpus" section of
`tools/chembl_tool/tasks/AGENTS.md`.

To add a case: reproduce the defect from a built `01_cleaned` stage, take the real
`measurement_text` / `unit_text` verbatim (never an invented string), confirm the corrected
output by hand, and record why it is correct in `validation.note`.
"""

import json
from pathlib import Path

import pytest

from tools.chembl_tool.common.starling.normalization.measurements import (
    normalize_measurement_and_unit,
)
from tools.chembl_tool.common.units import canonicalize_unit

CORPUS_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "normalization_regressions.jsonl"
)
# Every class discovered so far. A case may be retired only by deleting its class here
# too, which makes the removal explicit in review rather than a silently shrinking file.
EXPECTED_BUG_CLASSES = {
    "per_divisor_grouping",
    "alphanumeric_designator",
    "mismatched_exponent_variation",
    "silently_dropped_exponent",
    "qualifier_vocabulary_leak",
    "shared_qualifier_exponent",
    "compound_basis_exponent",
    "percent_suffix_scalar",
    "numeric_basis_denominator",
    "missing_time_unit",
    "mole_singular_spelling",
    "percent_spelling",
    "fail_closed_deliberate",
    "numeric_word_basis",
    "areic_spelling",
    "organ_basis",
    "radiolabel_equivalents",
    "fold_spelling",
    "dalton_mass_unit",
    "ambiguous_source_range",
}


def _load_corpus() -> list[dict]:
    lines = CORPUS_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


CORPUS = _load_corpus()


def test_corpus_is_present_and_well_formed():
    assert CORPUS, "regression corpus must not be empty"
    case_ids = [case["case_id"] for case in CORPUS]
    assert len(set(case_ids)) == len(case_ids), "duplicate case_id in corpus"
    for case in CORPUS:
        assert set(case) == {
            "case_id",
            "bug_class",
            "stage",
            "discovered",
            "task",
            "source_id",
            "record",
            "expected",
            "validation",
        }, f"unexpected entry shape for {case['case_id']!r}"
        assert case["expected"], f"{case['case_id']!r} asserts nothing"
        note = case["validation"].get("note", "")
        assert len(note) > 40, (
            f"{case['case_id']!r} must record why its expected output is correct"
        )


def test_every_bug_class_is_represented():
    """A discovered class must never silently lose its coverage."""
    assert {case["bug_class"] for case in CORPUS} == EXPECTED_BUG_CLASSES


@pytest.mark.parametrize("case", CORPUS, ids=[c["case_id"] for c in CORPUS])
def test_hand_validated_case(case):
    record, expected, task = case["record"], case["expected"], case["task"]
    pair = normalize_measurement_and_unit(
        record["measurement_text"], record["unit_text"], task=task
    )
    unit = canonicalize_unit(pair.canonical_unit, task=task)
    actual = {
        "canonical_measurement": pair.canonical_measurement,
        "canonical_unit": pair.canonical_unit,
        "status": pair.status,
        "unit_canonical": unit.canonical,
        "unit_dimension": [list(item) for item in unit.dimension],
        "unit_unknown_tokens": list(unit.unknown_tokens),
    }
    # Report every mismatch at once: a parser change usually moves several fields, and
    # seeing only the first makes the blast radius look smaller than it is.
    mismatches = {
        field: (value, actual[field])
        for field, value in expected.items()
        if actual[field] != value
    }
    assert not mismatches, (
        f"{case['case_id']}: " + "; ".join(
            f"{field} expected {want!r} got {got!r}"
            for field, (want, got) in mismatches.items()
        ) + f"\nhand-validation: {case['validation']['note']}"
    )
