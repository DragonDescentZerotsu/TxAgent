"""Stage-01-only golden audit over real BBB, oral-bio, and skin values.

The fixture is independent of Stage 02 canonicalization.  Each case contains
the source-facing input and a set of reviewed acceptable Stage-01 states, so a
future formatter may choose an equivalent punctuation form without weakening
the semantic regression gate. Support text has no acceptable alternatives: it
must remain exactly as received from the common ingestion layer.
"""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from tools.chembl_tool.common.starling.normalization.source_value_cleaning import (
    clean_source_values,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_policy import (
    DEFAULT_SOURCE_VALUE_REPAIRS,
)


CORPUS_PATH = (
    Path(__file__).resolve().parent
    / "fixtures/source_value_cleaning_audit_corpus.jsonl"
)
ROWS = [
    json.loads(line)
    for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
MANIFEST = ROWS[0]
CASES = ROWS[1:]
STATE_FIELDS = {"measurement_text", "unit_text", "support_text"}


def _state(record: dict) -> dict:
    return {field: record.get(field) for field in sorted(STATE_FIELDS)}


def test_audit_corpus_is_complete_and_cleaning_stage_scoped() -> None:
    assert MANIFEST["kind"] == "manifest"
    assert MANIFEST["corpus_version"] == "source_value_cleaning_audit_corpus.v4"
    assert MANIFEST["support_text_policy"] == "immutable_after_ingestion"
    assert MANIFEST["sampling_seed"] == 20260805
    assert len(CASES) == 5_073
    assert Counter(case["task"] for case in CASES) == {
        "bbb_martins": 2_000,
        "bioavailability_ma": 2_000,
        "skin_reaction": 1_073,
    }
    assert len({case["case_id"] for case in CASES}) == len(CASES)
    assert not any(
        case["input"]["source_id"]
        in {"direct_hf", "hf_nondirect_bioavailability"}
        for case in CASES
    )
    assert sum(
        len(case["acceptable_cleaned_states"]) > 1 for case in CASES
    ) >= 1_000

    for case in CASES:
        assert case["kind"] == "case"
        assert case["selection_class"] in {
            "changed_census",
            "seeded_random_control",
            "seeded_random_round2",
            "changed_census_current_v7",
        }
        assert set(case["acceptable_cleaned_states"][0]) == STATE_FIELDS
        assert all(
            set(state) == STATE_FIELDS
            for state in case["acceptable_cleaned_states"]
        )
        assert "support_text" not in case["expected_changed_fields"]
        assert {
            state["support_text"]
            for state in case["acceptable_cleaned_states"]
        } == {case["input"]["support_text"]}
        assert not any(
            field.startswith("canonical_")
            for state in case["acceptable_cleaned_states"]
            for field in state
        )


@pytest.mark.parametrize(
    "task", ["bbb_martins", "bioavailability_ma", "skin_reaction"]
)
def test_cleaner_stays_within_reviewed_acceptable_states(task: str) -> None:
    task_cases = [case for case in CASES if case["task"] == task]
    records = [deepcopy(case["input"]) for case in task_cases]
    result = clean_source_values(
        records,
        task_id=task,
        reviewed_repairs_path=(
            DEFAULT_SOURCE_VALUE_REPAIRS
            if task == "bioavailability_ma"
            else None
        ),
    )
    output_by_id = {
        str(record["cleaned_record_id"]): record for record in result.records
    }
    changed_fields: dict[str, set[str]] = {}
    for audit_row in result.audit_rows:
        assert audit_row["field"] != "support_text"
        changed_fields.setdefault(
            str(audit_row["cleaned_record_id"]), set()
        ).add(str(audit_row["field"]))

    mismatches = []
    audit_mismatches = []
    for case in task_cases:
        record_id = str(case["input"]["cleaned_record_id"])
        assert output_by_id[record_id]["support_text"] == case["input"]["support_text"]
        actual = _state(output_by_id[record_id])
        if actual not in case["acceptable_cleaned_states"]:
            mismatches.append(
                {
                    "case_id": case["case_id"],
                    "actual": actual,
                    "acceptable": case["acceptable_cleaned_states"],
                }
            )
        actual_fields = sorted(changed_fields.get(record_id, set()))
        if actual_fields != case["expected_changed_fields"]:
            audit_mismatches.append(
                {
                    "case_id": case["case_id"],
                    "actual": actual_fields,
                    "expected": case["expected_changed_fields"],
                }
            )

    assert not mismatches, f"first unacceptable state: {mismatches[:1]}"
    assert not audit_mismatches, (
        f"first cleaning-audit mismatch: {audit_mismatches[:1]}"
    )


def test_ambiguous_mixed_range_is_a_fail_closed_golden_case() -> None:
    case = next(
        case
        for case in CASES
        if case["task"] == "bioavailability_ma"
        and case["input"]["source_id"] == "fa"
        and case["input"]["source_row_number"] == 26845
    )
    assert case["input"]["measurement_text"] == "40-70,50"
    assert case["expected_changed_fields"] == []
    assert {
        state["measurement_text"]
        for state in case["acceptable_cleaned_states"]
    } == {"40-70,50", "40–70,50"}
