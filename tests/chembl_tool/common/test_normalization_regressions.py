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

from data.processing.evidence_library.shared.v1.normalization import measurements
from data.processing.evidence_library.shared.v1.normalization.measurements import (
    normalize_cleaned_records,
    normalize_measurement_and_unit,
)
from tools.chembl_tool.common.units import canonicalize_unit
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.canonical_source import (
    direct_measurement_fields,
    nondirect_measurement_fields,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.starling_categorical_response import (
    classify_direct_qualitative_text,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.starling_contextual_unit_reconciliation import (
    contextual_canonical_record_fields,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.starling_fg_scalar_rules import (
    propose_fg_scalar,
)

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
    "evidence_qualitative_substring",
    "source_measurement_policy_overreach",
    "unencoded_directional_context",
}


def _load_corpus() -> list[dict]:
    lines = CORPUS_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


CORPUS = _load_corpus()


def _actual_for_case(case: dict) -> dict:
    record, task, stage = case["record"], case["task"], case["stage"]
    if stage == "normalized_scalar_gate":
        normalized = normalize_cleaned_records(
            [record],
            endpoint_normalizer=lambda _source_id, endpoint: endpoint,
            family_resolver=lambda _source_id, _endpoint, _record: None,
            task=task,
        )[0]
        return {
            "canonical_measurement": normalized["canonical_measurement"],
            "canonical_unit": normalized["canonical_unit"],
            "measurement_parse_kind": normalized["measurement_parse_kind"],
            "measurement_unit_status": normalized["measurement_unit_status"],
            "finite_scalar_value": normalized["finite_scalar_value"],
            "is_absolute_and_continuous": normalized[
                "is_absolute_and_continuous"
            ],
        }
    if stage == "bioavailability_direct_source_measurement":
        return direct_measurement_fields(record["measurement_text"])
    if stage == "bioavailability_nondirect_source_measurement":
        return nondirect_measurement_fields(record["measurement_text"])
    if stage == "bioavailability_fg_source_measurement":
        return propose_fg_scalar(
            record["measurement_text"],
            canonical_endpoint=record["endpoint_name"],
        ).to_dict()
    if stage == "bioavailability_evidence_qualitative":
        category, reason = classify_direct_qualitative_text(
            record["measurement_text"]
        )
        return {"category": category, "reason": reason}

    pair = normalize_measurement_and_unit(
        record["measurement_text"], record["unit_text"], task=task
    )
    unit = canonicalize_unit(pair.canonical_unit, task=task)
    return {
        "canonical_measurement": pair.canonical_measurement,
        "canonical_unit": pair.canonical_unit,
        "status": pair.status,
        "unit_canonical": unit.canonical,
        "unit_dimension": [list(item) for item in unit.dimension],
        "unit_unknown_tokens": list(unit.unknown_tokens),
    }


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
    expected = case["expected"]
    actual = _actual_for_case(case)
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


def test_directional_scalar_gate_cannot_be_undone_by_task_enrichment():
    """The common boundary stays authoritative over later task-local reparsing."""
    case = next(
        case
        for case in CORPUS
        if case["case_id"] == "scalar_gate.unencoded_directional_fa.v1"
    )
    normalized = normalize_cleaned_records(
        [case["record"]],
        endpoint_normalizer=lambda _source_id, endpoint: endpoint,
        family_resolver=lambda _source_id, _endpoint, _record: None,
        record_enricher=lambda _record: {
            "finite_scalar_value": 62.0,
            "is_absolute_and_continuous": True,
            "absolute_and_continuous_value": 62.0,
            "normalization_validity_status": "valid",
        },
        task=case["task"],
    )[0]
    assert normalized["finite_scalar_value"] is None
    assert normalized["is_absolute_and_continuous"] is False
    assert normalized["absolute_and_continuous_value"] is None
    assert normalized["measurement_unit_status"] == (
        "non_atomic_directional_context"
    )
    assert normalized["normalization_validity_status"] == (
        "non_scalar_measurement"
    )


def test_exact_resolution_bypasses_the_legacy_measurement_parser(monkeypatch):
    def legacy_parser_called(*_args, **_kwargs):
        raise AssertionError("the legacy parser handled an exact extraction")

    monkeypatch.setattr(
        measurements, "normalize_measurement_and_unit", legacy_parser_called
    )
    base = {
        "source_id": "source",
        "endpoint_name": "raw endpoint",
        "canonical_endpoint_name": "permeability",
        "measurement_resolution_status": "ok",
        "measurement_resolution_parent_cleaned_record_id": "parent",
        "measurement_text": "source prose",
        "unit_text": "source unit",
    }
    mapped = {
        **base,
        "cleaned_record_id": "mapped",
        "measurement_unit_mapping_status": "mapped",
        "resolved_measurement_text": "0.0000056",
        "resolved_unit_text": "cm/s",
        "resolved_scalar_value": 0.0000056,
    }
    excluded = {
        **base,
        "cleaned_record_id": "excluded",
        "measurement_unit_mapping_status": "excluded",
    }
    normalized = normalize_cleaned_records(
        [mapped, excluded],
        endpoint_normalizer=lambda _source_id, endpoint: endpoint,
        family_resolver=lambda _source_id, _endpoint, _record: None,
        task="test_task",
    )
    assert normalized[0]["canonical_measurement"] == "0.0000056"
    assert normalized[0]["canonical_unit"] == "cm/s"
    assert normalized[0]["finite_scalar_value"] == 0.0000056
    assert normalized[0]["variation_value"] is None
    assert normalized[1]["canonical_measurement"] is None
    assert normalized[1]["canonical_unit"] is None
    assert normalized[1]["finite_scalar_value"] is None
    assert normalized[1]["measurement_unit_status"] == "exact_unit_excluded"


def test_exact_resolution_preserves_enriched_endpoint_fallback():
    normalized = normalize_cleaned_records(
        [
            {
                "source_id": "source",
                "endpoint_name": "",
                "canonical_endpoint_name": "missing_endpoint",
                "cleaned_record_id": "row",
                "measurement_resolution_status": "ok",
                "measurement_unit_mapping_status": "mapped",
                "resolved_measurement_text": "1.5",
                "resolved_unit_text": "dimensionless",
                "resolved_scalar_value": 1.5,
            }
        ],
        endpoint_normalizer=lambda _source_id, endpoint: endpoint,
        family_resolver=lambda _source_id, _endpoint, _record: None,
        record_enricher=lambda _record: {
            "canonical_endpoint": "efflux_substrate_outcome"
        },
        task="test_task",
    )[0]
    assert normalized["canonical_endpoint"] == "efflux_substrate_outcome"
    assert normalized["finite_scalar_value"] == 1.5


def test_bioavailability_contextual_reconciliation_preserves_directional_gate():
    fields = contextual_canonical_record_fields(
        {
            "source_id": "fa",
            "canonical_endpoint": "gi_stability",
            "canonical_measurement": "60 ± 0.47 reduction",
            "canonical_unit": "%",
            "measurement_unit_status": "cleaned_pair",
            "unit_notation_status": "none",
        }
    )

    assert fields["finite_scalar_value"] is None
    assert fields["is_absolute_and_continuous"] is False
    assert fields["measurement_unit_status"] == "non_atomic_directional_context"
