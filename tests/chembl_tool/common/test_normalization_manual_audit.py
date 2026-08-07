"""Executable golden audit of complete v7 canonicalization and pair bucketing.

The fixture contains 500 real Stage-01 records from each migrated task.  It has
no writer or expectation-update mode: a changed result must be reviewed against
the source record and rationale before this file is edited.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from tools.chembl_tool.common.starling.build_normalized_evidence_library import (
    parse_args,
)
from tools.chembl_tool.common.starling.build_runtime import (
    normalize_and_project_records_ordered,
)
from tools.chembl_tool.common.starling.pair_buckets import (
    materialize_pair_buckets,
)
from tools.chembl_tool.tasks.bbb_martins.starling_policy import (
    POLICY as BBB_POLICY,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_policy import (
    POLICY as BIO_POLICY,
)
from tools.chembl_tool.tasks.skin_reaction.starling_policy import (
    POLICY as SKIN_POLICY,
)


CORPUS_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "normalization_manual_audit.jsonl"
)
ROWS = [
    json.loads(line)
    for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
MANIFEST = ROWS[0]
CASES = ROWS[1:]
POLICIES = {
    "bbb_martins": BBB_POLICY,
    "bioavailability_ma": BIO_POLICY,
    "skin_reaction": SKIN_POLICY,
}


def _mismatches(expected: dict, actual: dict) -> dict:
    return {
        field: {"expected": value, "actual": actual.get(field)}
        for field, value in expected.items()
        if actual.get(field) != value
    }


def test_manual_audit_corpus_is_fixed_complete_and_source_grounded() -> None:
    assert MANIFEST["kind"] == "manifest"
    assert MANIFEST["corpus_version"] == "normalization_manual_audit.v1"
    assert MANIFEST["sampling_seed"] == "normalization_manual_audit.v1.20260805"
    assert MANIFEST["expectation_update_policy"] == "manual_review_only"
    assert len(CASES) == 1_500
    assert Counter(case["task"] for case in CASES) == {
        "bbb_martins": 500,
        "bioavailability_ma": 500,
        "skin_reaction": 500,
    }
    assert len({case["audit_case_id"] for case in CASES}) == len(CASES)
    assert sum(
        case["human_review"]["finding"]
        == "excluded_unencoded_directional_context"
        for case in CASES
    ) == 25
    for case in CASES:
        source = case["stage_01_record"]
        assert source["cleaned_record_id"]
        assert source["source_id"]
        assert source["source_row_number"] is not None
        assert len(case["human_review"]["rationale"]) > 40
        assert case["expected_canonical"]
        assert case["expected_pair_bucket"]


@pytest.mark.parametrize("task", sorted(POLICIES))
def test_reviewed_v7_outputs_replay_exactly(task: str, tmp_path: Path) -> None:
    policy = POLICIES[task]
    task_cases = [case for case in CASES if case["task"] == task]
    args = parse_args(
        policy,
        [
            "--out-dir",
            str(tmp_path / task),
            "--through-stage",
            "normalize",
            "--progress-every",
            "0",
            "--allow-missing-reference-semantics",
        ],
    )
    hooks = policy.build_hooks(args)
    inflated = []
    for case in task_cases:
        record = policy.record_contract.inflate_cleaned(case["stage_01_record"])
        record.update(case["resolved_structure"])
        inflated.append(record)

    _, persisted = normalize_and_project_records_ordered(
        inflated,
        hooks=hooks,
        policy=policy,
        workers=1,
    )
    canonical_by_id = {
        str(row["cleaned_record_id"]): row for row in persisted
    }
    canonical_errors = []
    for case in task_cases:
        record_id = str(case["stage_01_record"]["cleaned_record_id"])
        errors = _mismatches(
            case["expected_canonical"], canonical_by_id[record_id]
        )
        if errors:
            canonical_errors.append(
                {"audit_case_id": case["audit_case_id"], "fields": errors}
            )
    assert not canonical_errors, canonical_errors[:1]

    pair_fields = {
        # This hand-reviewed corpus predates the row-level reference classifier;
        # its exact pair JSON continues to guard the cleaning/parser audit while
        # reference eligibility is covered by test_starling_reference_semantics.
        source: tuple(
            field
            for field in spec.additional_dimensions
            if field
            not in {"canonical_reference_scope", "canonical_reference_basis"}
        )
        for source, spec in policy.record_contract.pair_buckets.items()
    }
    sidecar, _ = materialize_pair_buckets(
        persisted,
        source_required_fields=pair_fields,
        contract_version=policy.record_contract.version,
    )
    pair_by_id = {str(row["canonical_record_id"]): row for row in sidecar}
    pair_errors = []
    for case in task_cases:
        expected = case["expected_pair_bucket"]
        record_id = str(case["expected_canonical"]["canonical_record_id"])
        errors = _mismatches(expected, pair_by_id[record_id])
        if errors:
            pair_errors.append(
                {"audit_case_id": case["audit_case_id"], "fields": errors}
            )
    assert not pair_errors, pair_errors[:1]
