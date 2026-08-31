"""Hand-validated regressions for authoritative Stage-01 source values.

The JSONL corpus contains verbatim real records and independent expected
outputs.  It is intentionally hand-written and has no regeneration path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
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
        require_all_reviewed_repairs=not reviewed,
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
        require_all_reviewed_repairs=False,
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
    payload = json.loads(
        DEFAULT_SOURCE_VALUE_REPAIRS.read_text(encoding="utf-8").splitlines()[0]
    )
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


def test_reviewed_smiles_repair_requires_matching_audit(tmp_path: Path) -> None:
    record = {
        "cleaned_record_id": "fixture:smiles",
        "source_id": "direct_bbb",
        "source_sha256": "a" * 64,
        "source_row_number": 7,
        "source_record_id": "6",
        "measurement_text": None,
        "unit_text": None,
        "support_text": "The indexed molecule is pentazocine.",
        "source_smiles": "c1ccc2c(c1)Nc1ccccc1S2",
        "source_payload_json": json.dumps(
            {"smiles": "c1ccc2c(c1)Nc1ccccc1S2"}
        ),
    }
    audit = {
        "audit_id": "smiles_fixture_0001",
        "task_id": "bbb_martins",
        "source_id": "direct_bbb",
        "source_sha256": "a" * 64,
        "source_row_number": 7,
        "source_record_id": "6",
        "stored_smiles": record["source_smiles"],
        "classification": "confirmed_mismatch",
    }
    repair = {
        "repair_id": "smiles_fixture_0001",
        "task_id": "bbb_martins",
        "source_id": "direct_bbb",
        "source_sha256": "a" * 64,
        "source_row_number": 7,
        "source_record_id": "6",
        "before": {"smiles": record["source_smiles"]},
        "after": {"smiles": "CC(C)=CCN1CCC2(C)c3cc(O)ccc3CC1C2C"},
        "evidence": {
            "audit_id": audit["audit_id"],
            "note": "The support names pentazocine while the stored structure is phenothiazine.",
        },
    }
    repair_path = tmp_path / "repairs.jsonl"
    audit_path = tmp_path / "audit.jsonl"
    repair_path.write_text(json.dumps(repair) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(audit) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="lacks a matching audit entry"):
        clean_source_values(
            [dict(record)],
            task_id="bbb_martins",
            reviewed_repairs_path=repair_path,
        )

    result = clean_source_values(
        [dict(record)],
        task_id="bbb_martins",
        reviewed_repairs_path=repair_path,
        smiles_identity_audit_path=audit_path,
    )
    assert result.records[0]["source_smiles"] == record["source_smiles"]
    assert result.records[0]["canonical_smiles"] == repair["after"]["smiles"]
    assert json.loads(result.records[0]["source_payload_json"])["smiles"] == record[
        "source_smiles"
    ]
    assert result.audit_rows[-1]["field"] == "smiles"

    repair["after"]["smiles"] = "not a smiles"
    repair_path.write_text(json.dumps(repair) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid replacement SMILES"):
        clean_source_values(
            [dict(record)],
            task_id="bbb_martins",
            reviewed_repairs_path=repair_path,
            smiles_identity_audit_path=audit_path,
        )


def test_reviewed_source_row_drop_is_exact_and_audited(tmp_path: Path) -> None:
    dropped = {
        "cleaned_record_id": "fixture:drop",
        "source_id": "direct_bbb",
        "source_sha256": "a" * 64,
        "source_row_number": 7,
        "source_record_id": "6",
        "measurement_text": None,
        "unit_text": None,
        "support_text": "The indexed molecule has an unresolved identity.",
    }
    retained = {
        **dropped,
        "cleaned_record_id": "fixture:retained",
        "source_row_number": 8,
        "source_record_id": "7",
    }
    drop = {
        "drop_id": "drop_fixture_0001",
        "task_id": "bbb_martins",
        "source_id": "direct_bbb",
        "source_sha256": "a" * 64,
        "source_row_number": 7,
        "source_record_id": "6",
        "reason_code": "unresolved_molecule_identity",
        "evidence": {
            "note": "The paper establishes a mismatch but does not provide a recoverable structure."
        },
    }
    drop_path = tmp_path / "drops.jsonl"
    drop_path.write_text(json.dumps(drop) + "\n", encoding="utf-8")

    result = clean_source_values(
        [dict(dropped), dict(retained)],
        task_id="bbb_martins",
        reviewed_drops_path=drop_path,
    )
    assert [row["cleaned_record_id"] for row in result.records] == [
        "fixture:retained"
    ]
    assert result.audit_rows == [
        {
            "cleaning_version": "starling_source_value_cleaning.v6",
            "cleaned_record_id": "fixture:drop",
            "source_id": "direct_bbb",
            "source_sha256": "a" * 64,
            "source_row_number": 7,
            "source_record_id": "6",
            "field": "record",
            "before": "retained",
            "after": "dropped",
            "rule_id": "reviewed_drop:drop_fixture_0001",
            "review_status": "reviewed",
            "evidence_id": "",
            "rationale": drop["evidence"]["note"],
        }
    ]
    assert result.manifest["n_dropped_records"] == 1

    drop["source_sha256"] = "b" * 64
    drop_path.write_text(json.dumps(drop) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source drift"):
        clean_source_values(
            [dict(dropped)],
            task_id="bbb_martins",
            reviewed_drops_path=drop_path,
        )


def test_reviewed_name_smiles_override_is_applied_with_exact_precondition(tmp_path):
    review_path = tmp_path / "reviewed.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "candidate_id": "candidate-1",
                    "task_id": "bbb_martins",
                    "source_id": "direct_bbb",
                    "source_sha256": "a" * 64,
                    "source_row_number": 9,
                    "source_record_id": "8",
                    "canonical_smiles": "CCO",
                    "decision": "override",
                    "override_smiles": "CCN",
                    "confidence": "high",
                    "rationale": "The reviewed source names ethylamine and establishes the replacement structure.",
                },
                {
                    "candidate_id": "candidate-2",
                    "task_id": "bbb_martins",
                    "source_id": "direct_bbb",
                    "source_sha256": "a" * 64,
                    "source_row_number": 10,
                    "source_record_id": "9",
                    "canonical_smiles": "CCC",
                    "decision": "override",
                    "override_smiles": "CCCl",
                    "confidence": "medium",
                    "rationale": "The replacement remains quarantined until an independent review confirms it.",
                }
            ]
        ),
        review_path,
    )
    record = {
        "cleaned_record_id": "fixture:smiles",
        "source_id": "direct_bbb",
        "source_sha256": "a" * 64,
        "source_row_number": 9,
        "source_record_id": "8",
        "source_smiles": "CCO",
        "canonical_smiles": "OCC",
        "structure_status": "resolved",
        "measurement_text": None,
        "unit_text": None,
        "support_text": "The reviewed source names ethylamine.",
    }

    medium_record = {
        **record,
        "cleaned_record_id": "fixture:medium-smiles",
        "source_row_number": 10,
        "source_record_id": "9",
        "source_smiles": "CCC",
        "canonical_smiles": "CCC",
        "support_text": "The medium-confidence source names chloroethane.",
    }
    result = clean_source_values(
        [record, medium_record],
        task_id="bbb_martins",
        reviewed_smiles_conflicts_path=review_path,
    )

    assert result.records[0]["canonical_smiles"] == "CCN"
    assert result.records[0]["structure_status"] == "resolved"
    assert result.records[1]["canonical_smiles"] == "CCC"
    assert result.manifest["reviewed_name_smiles_conflicts"]["n_applied_overrides"] == 1
    assert result.manifest["reviewed_name_smiles_conflicts"]["n_declared_overrides"] == 2
    assert result.manifest["reviewed_name_smiles_conflicts"]["n_eligible_overrides"] == 1
    assert result.manifest["reviewed_name_smiles_conflicts"]["n_quarantined_overrides"] == 1
    assert result.audit_rows[0]["rule_id"] == (
        "reviewed_name_smiles_override:candidate-1"
    )


@pytest.mark.parametrize(
    ("override_smiles", "confidence", "error"),
    [
        ("not-a-smiles", "medium", "invalid override SMILES"),
        ("CCN", "hgh", "invalid confidence"),
    ],
)
def test_reviewed_name_smiles_ledger_fails_closed(
    tmp_path, override_smiles, confidence, error
):
    review_path = tmp_path / "reviewed.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "candidate_id": "candidate-1",
                    "task_id": "bbb_martins",
                    "source_id": "direct_bbb",
                    "source_sha256": "a" * 64,
                    "source_row_number": 9,
                    "source_record_id": "8",
                    "canonical_smiles": "CCO",
                    "decision": "override",
                    "override_smiles": override_smiles,
                    "confidence": confidence,
                    "rationale": "The reviewed source provides enough text to exercise ledger validation.",
                }
            ]
        ),
        review_path,
    )
    with pytest.raises(ValueError, match=error):
        clean_source_values(
            [],
            task_id="bbb_martins",
            reviewed_smiles_conflicts_path=review_path,
        )


def test_reviewed_name_smiles_ledger_rejects_duplicate_source_rows(tmp_path):
    rows = []
    for candidate_id, confidence in (("candidate-1", "medium"), ("candidate-2", "high")):
        rows.append(
            {
                "candidate_id": candidate_id,
                "task_id": "bbb_martins",
                "source_id": "direct_bbb",
                "source_sha256": "a" * 64,
                "source_row_number": 9,
                "source_record_id": "8",
                "canonical_smiles": "CCO",
                "decision": "override",
                "override_smiles": "CCN",
                "confidence": confidence,
                "rationale": "The duplicate target must fail before confidence filtering is applied.",
            }
        )
    review_path = tmp_path / "reviewed.parquet"
    pq.write_table(pa.Table.from_pylist(rows), review_path)
    with pytest.raises(ValueError, match="reviews one source row more than once"):
        clean_source_values(
            [],
            task_id="bbb_martins",
            reviewed_smiles_conflicts_path=review_path,
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
