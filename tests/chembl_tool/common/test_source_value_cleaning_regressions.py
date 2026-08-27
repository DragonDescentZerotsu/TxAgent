"""Hand-validated regressions for authoritative Stage-01 source values.

The JSONL corpus contains verbatim real records and independent expected
outputs.  It is intentionally hand-written and has no regeneration path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.chembl_tool.common.starling.normalization.measurements import (
    normalize_measurement_and_unit,
)
from tools.chembl_tool.common.starling.normalization.source_value_cleaning import (
    clean_source_values,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_policy import (
    DEFAULT_SOURCE_VALUE_REPAIRS,
    _resolve_source_measurement_pair,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_schema import (
    RECORD_CONTRACT as BIO_CONTRACT,
)
from tools.chembl_tool.tasks.bbb_martins.starling_policy import (
    DEFAULT_SOURCE_VALUE_REPAIRS as BBB_SOURCE_VALUE_REPAIRS,
)


CORPUS_PATH = (
    Path(__file__).resolve().parent
    / "fixtures/source_value_cleaning_regressions.jsonl"
)
CORPUS = [
    json.loads(line)
    for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
EXPECTED_BUG_CLASSES = {
    "decimal_comma",
    "encoded_space",
    "reviewed_missing_separator",
    "thousands_control",
    "chemical_locant_control",
    "ambiguous_mixed_range_control",
    "malformed_percent_control",
}


def _source_record(case: dict) -> dict:
    source = case["source"]
    return {
        "cleaned_record_id": f"fixture:{case['case_id']}",
        "source_id": source["source_id"],
        "source_sha256": source.get("source_sha256", "a" * 64),
        "source_row_number": source["source_row_number"],
        "source_record_id": source["source_record_id"],
        **case["record"],
    }


def test_corpus_is_hand_validated_and_complete() -> None:
    assert CORPUS
    assert len({case["case_id"] for case in CORPUS}) == len(CORPUS)
    assert {case["bug_class"] for case in CORPUS} == EXPECTED_BUG_CLASSES
    assert {case["task"] for case in CORPUS} == {
        "bbb_martins",
        "bioavailability_ma",
        "skin_reaction",
    }
    for case in CORPUS:
        assert set(case) == {
            "case_id",
            "bug_class",
            "task",
            "source",
            "record",
            "expected",
            "validation",
        }
        assert len(case["validation"]["note"]) > 40


@pytest.mark.parametrize("case", CORPUS, ids=[case["case_id"] for case in CORPUS])
def test_source_value_regression(case: dict) -> None:
    reviewed = case["bug_class"] == "reviewed_missing_separator"
    result = clean_source_values(
        [_source_record(case)],
        task_id=case["task"],
        reviewed_repairs_path=DEFAULT_SOURCE_VALUE_REPAIRS if reviewed else None,
    )
    record = result.records[0]
    expected = case["expected"]
    assert record["measurement_text"] == expected["measurement_text"]
    assert record["support_text"] == case["record"]["support_text"]
    assert record["support_text"] == expected["support_text"]

    pair = normalize_measurement_and_unit(
        record["measurement_text"], record.get("unit_text"), task=case["task"]
    )
    assert pair.canonical_measurement == expected["canonical_measurement"]
    assert pair.canonical_unit == expected["canonical_unit"]
    assert pair.status == expected["status"]


@pytest.mark.parametrize(
    "measurement",
    ["0,285", "35%50%", "50.6%0020 ± 10.0%"],
)
def test_parser_fails_closed_when_stage_01_is_bypassed(measurement: str) -> None:
    pair = normalize_measurement_and_unit(
        measurement, "%", task="bioavailability_ma"
    )
    assert pair.status == "cleaned_only_non_scalar"


@pytest.mark.parametrize(
    ("measurement", "expected", "expected_unit"),
    [
        ("0", "0", None),
        ("0.61", "0.61", None),
        ("1.5", "1.5", None),
        ("1.51", "1.51", None),
        ("50.6%", "50.6", "%"),
        ("0.61 ± 0.05", "0.61 ± 0.05", None),
        ("0.61 fraction", "61", "%"),
    ],
)
def test_hf_direct_scalar_requires_explicit_unit(
    measurement: str, expected: str, expected_unit: str | None
) -> None:
    baseline = normalize_measurement_and_unit(
        measurement, "%", task="bioavailability_ma"
    )
    resolved = _resolve_source_measurement_pair(
        {
            "source_id": "hf_bioavailability",
            "measurement_text": measurement,
            "bioavailability_report_type": "absolute",
        },
        "oral_bioavailability",
        baseline,
    )
    assert resolved.canonical_measurement == expected
    assert resolved.canonical_unit == expected_unit

    other_source = _resolve_source_measurement_pair(
        {"source_id": "fa", "measurement_text": measurement},
        "oral_bioavailability",
        baseline,
    )
    assert other_source == baseline


def test_hf_nondirect_row_does_not_use_direct_fraction_rule() -> None:
    baseline = normalize_measurement_and_unit(
        "0.61", "%", task="bioavailability_ma"
    )
    resolved = _resolve_source_measurement_pair(
        {
            "source_id": "hf_bioavailability",
            "measurement_text": "0.61",
            "bioavailability_report_type": "relative_comparison",
        },
        "oral_bioavailability",
        baseline,
    )
    assert resolved.canonical_measurement == "0.61"
    assert resolved.canonical_unit is None


def test_reviewed_repair_is_visible_but_canonical_range_is_not_substituted() -> None:
    case = next(
        case for case in CORPUS if case["bug_class"] == "reviewed_missing_separator"
    )
    record = clean_source_values(
        [_source_record(case)],
        task_id="bioavailability_ma",
        reviewed_repairs_path=DEFAULT_SOURCE_VALUE_REPAIRS,
    ).records[0]
    profile = BIO_CONTRACT.source("hf_bioavailability")
    record.update({field: record.get(field) for field in profile.source_columns})
    record.update(
        {
            "source_id": "hf_bioavailability",
            "endpoint_name": "oral_bioavailability",
            "evidence_context_json": "{}",
            "smiles": "CCO",
            "canonical_measurement": "SHOULD NOT BE DISPLAYED",
        }
    )
    cleaned = BIO_CONTRACT.clean_projection(record)
    source = BIO_CONTRACT.source_projection(cleaned)["source_fields"]
    assert source["measurement_text"] == "35%–50%"
    assert source["support_text"] == (
        "The oral bioavailability of amiodarone is 35%50%, which is described "
        "as low and variable."
    )
    assert "canonical_measurement" not in source


def test_reviewed_registry_fails_on_source_or_measurement_drift() -> None:
    case = next(
        case for case in CORPUS if case["bug_class"] == "reviewed_missing_separator"
    )
    source_drift = _source_record(case)
    source_drift["source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="source drift"):
        clean_source_values(
            [source_drift],
            task_id="bioavailability_ma",
            reviewed_repairs_path=DEFAULT_SOURCE_VALUE_REPAIRS,
        )

    measurement_drift = _source_record(case)
    measurement_drift["measurement_text"] += " changed"
    with pytest.raises(ValueError, match="precondition drift"):
        clean_source_values(
            [measurement_drift],
            task_id="bioavailability_ma",
            reviewed_repairs_path=DEFAULT_SOURCE_VALUE_REPAIRS,
        )


def test_every_cleaned_value_change_is_audited_and_support_is_immutable() -> None:
    cases = [case for case in CORPUS if case["bug_class"] != "reviewed_missing_separator"]
    records = [_source_record(case) for case in cases]
    before_by_id = {
        row["cleaned_record_id"]: dict(row)
        for row in records
    }
    result = clean_source_values(records, task_id="mixed_fixture")
    cleaned_by_id = {row["cleaned_record_id"]: row for row in result.records}
    audited = {
        (row["cleaned_record_id"], row["field"], row["before"], row["after"])
        for row in result.audit_rows
    }
    for record_id, before in before_by_id.items():
        after = cleaned_by_id[record_id]
        assert after.get("support_text") == before.get("support_text")
        for field in ("measurement_text", "unit_text"):
            if before.get(field) != after.get(field):
                assert (
                    record_id,
                    field,
                    before.get(field),
                    after.get(field),
                ) in audited
    assert not any(row["field"] == "support_text" for row in result.audit_rows)


def test_reviewed_registry_rejects_support_text_repairs(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_SOURCE_VALUE_REPAIRS.read_text(encoding="utf-8"))
    payload["before"] = {"support_text": "raw support"}
    payload["after"] = {"support_text": "rewritten support"}
    repair_path = tmp_path / "forbidden_support_repair.jsonl"
    repair_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="repairs a forbidden field"):
        clean_source_values(
            [],
            task_id="bioavailability_ma",
            reviewed_repairs_path=repair_path,
            require_all_reviewed_repairs=False,
        )


def test_scientific_scale_conflict_requires_and_accepts_review() -> None:
    conflict = {
        "cleaned_record_id": "0a3ab857",
        "source_id": "passive_permeability",
        "source_sha256": "1c1b602fe640666c4fb2e006c9712673ecba7ae097bbd7f3e027094762e7c5a2",
        "source_row_number": 2468,
        "source_record_id": "ext_1",
        "measurement_text": "1e-05",
        "unit_text": "10 × 10^-6 cm/s",
        "support_text": "The PAMPA value was 10 × 10^-6 cm/s.",
    }
    with pytest.raises(ValueError, match="scientific factor"):
        clean_source_values(
            [dict(conflict)],
            task_id="bbb_martins",
            require_scientific_scale_review=True,
        )

    result = clean_source_values(
        [dict(conflict)],
        task_id="bbb_martins",
        reviewed_repairs_path=BBB_SOURCE_VALUE_REPAIRS,
        require_all_reviewed_repairs=False,
        require_scientific_scale_review=True,
    )
    assert result.records[0]["measurement_text"] == "10"
    assert result.manifest["scientific_scale_review"] == {
        "n_candidates": 1,
        "n_resolved_by_reviewed_repair": 1,
        "n_unresolved": 0,
        "all_candidates_resolved": True,
    }

    control = {**conflict, "measurement_text": "0.00039"}
    control["support_text"] = "The permeability was 0.00039 × 10^-6 cm/s."
    clean_source_values(
        [control],
        task_id="bbb_martins",
        require_scientific_scale_review=True,
    )
