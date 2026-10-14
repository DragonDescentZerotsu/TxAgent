import json

import pyarrow.parquet as pq
import pytest

from data.processing.gold_labels.benchmark_dataset import (
    LabelDecision,
    LabeledSourceRecord,
    accepted,
    build_benchmark_dataset,
)
from data.processing.gold_labels.build_repaired_v2 import (
    CoverageError,
    EXCLUDED_EXTERNAL_CONDITION,
    _preserved_split_assignments,
    validate_condition_ledger,
)
from data.processing.gold_labels.stage1_repaired_sources import (
    _label_oral_stage1_row,
    scientific_payload_sha256,
)


def _record(smiles, label, suffix):
    return accepted(
        LabeledSourceRecord(
            smiles=smiles,
            label=label,
            source_id="source",
            source_record_id=suffix,
            label_method="fixture",
            source_row_uid="sr_" + suffix * 32,
        )
    )


def test_base_builder_writes_uncapped_physical_membership(tmp_path):
    output = tmp_path / "gold"
    membership = output / "voter_membership.parquet"
    smiles = [
        "CCO",
        "C1CCCCC1",
        "c1ccccc1",
        "c1ccncc1",
        "C1CCCC1",
        "C1CCC1",
        "C1CC1",
        "c1ccoc1",
        "c1ccsc1",
        "C1CCNCC1",
        "C1COCCO1",
        "c1ncc[nH]1",
    ]
    decisions = [
        _record(value, index % 2, suffix)
        for index, (value, suffix) in enumerate(zip(smiles, "0123456789ab"))
    ]
    decisions.extend([_record("CCN", 0, "e"), _record("CCN", 1, "f")])
    build_benchmark_dataset(
        task_name="Fixture",
        decisions=decisions,
        source_metadata={},
        output_dir=output,
        max_eval_size=1,
        voter_membership_path=membership,
        voter_membership_stage1_sha256="d" * 64,
    )
    rows = pq.read_table(membership).to_pylist()
    assert len(rows) == len(decisions)
    assert {row["aggregate_status"] for row in rows} == {
        "published",
        "parent_label_tie",
    }
    manifest = json.loads(
        (output / "voter_membership.manifest.json").read_text()
    )
    assert manifest["stage1_sha256"] == "d" * 64


def test_oral_stage1_adapter_keeps_source_uid_and_source_native_vote():
    decision = _label_oral_stage1_row(
        {
            "source_id": "oral_exposure",
            "source_index": 4,
            "source_record_id": "ext_1",
            "source_row_uid": "sr_" + "e" * 32,
            "canonical_smiles": "CCO",
            "endpoint_name": "bioavailability",
            "measurement_text": "10%",
            "unit_text": "%",
            "support_text": "Absolute oral bioavailability was 10% in human volunteers.",
            "study_context": "human volunteers",
            "oral_dose": "10 mg oral versus IV",
            "qualifying_conditions": "",
            "pmid": "123",
        }
    )
    assert decision.record is not None
    assert decision.record.label == 0
    assert decision.record.source_record_id == "ext_1"
    assert decision.record.source_row_uid == "sr_" + "e" * 32


def test_oral_stage1_adapter_can_label_reviewed_conditioned_voter():
    row = {
        "source_id": "oral_exposure",
        "source_index": 4,
        "source_record_id": "ext_1",
        "source_row_uid": "sr_" + "e" * 32,
        "canonical_smiles": "CCO",
        "endpoint_name": "bioavailability",
        "measurement_text": "10%",
        "unit_text": "%",
        "support_text": "Absolute oral bioavailability was 10% in human volunteers.",
        "study_context": "human volunteers",
        "oral_dose": "10 mg oral versus IV",
        "qualifying_conditions": "fasted",
    }
    assert _label_oral_stage1_row(row).record is None
    assert (
        _label_oral_stage1_row(row, allow_conditioned_context=True).record.label == 0
    )


def _physical_oral_voter():
    source = {
        "source_id": "oral_exposure",
        "source_record_id": "local:1",
        "source_row_uid": "sr_" + "a" * 32,
        "canonical_smiles": "CCO",
        "endpoint_name": "bioavailability",
        "measurement_text": "10%",
        "unit_text": "%",
        "study_context": "human volunteers",
        "qualifying_conditions": "fasted",
        "support_text": "Measured after an overnight fast.",
    }
    record = LabeledSourceRecord(
        smiles="CCO",
        label=0,
        source_id="oral_exposure",
        source_record_id="local:1",
        label_method="stage1_physical:fixture",
        source_row_uid=source["source_row_uid"],
    )
    return source, LabelDecision(record=record)


def _condition_review(source, decision):
    return {
        "source_row_uid": source["source_row_uid"],
        "source_id": source["source_id"],
        "source_record_id": source["source_record_id"],
        "source_payload_sha256": scientific_payload_sha256(
            "bioavailability_ma", source
        ),
        "vote_label": decision.record.label,
        "label_method": decision.record.label_method,
        "decision": "active_external",
        "condition_group": "prandial_state=fasted",
        "condition_atoms": ["prandial_state=fasted"],
        "reviewer": "fixture",
        "rationale": "The measured arm is explicitly fasted.",
        "mapping_origin": "physical_review",
        "reviewed_source_id": "oral_exposure",
        "reviewed_source_payload_sha256": scientific_payload_sha256(
            "bioavailability_ma", source
        ),
    }


def test_scientific_payload_hash_allows_only_smiles_reparenting():
    source, _ = _physical_oral_voter()
    repaired = {**source, "canonical_smiles": "CC"}
    changed_support = {**source, "support_text": "Different measured arm."}
    assert scientific_payload_sha256(
        "bioavailability_ma", source
    ) == scientific_payload_sha256("bioavailability_ma", repaired)
    assert scientific_payload_sha256(
        "bioavailability_ma", source
    ) != scientific_payload_sha256("bioavailability_ma", changed_support)


def test_condition_ledger_requires_exact_payload_bound_voter_coverage(tmp_path):
    source, decision = _physical_oral_voter()
    ledger = tmp_path / "reviews.jsonl"
    ledger.write_text(json.dumps(_condition_review(source, decision)) + "\n")
    rows = validate_condition_ledger(
        task="bioavailability_ma",
        accepted_voters=[(source, decision)],
        ledger_path=ledger,
        allowed_condition_groups={"prandial_state=fasted"},
    )
    assert rows[0]["source_row_uid"] == source["source_row_uid"]

    ledger.write_text("")
    with pytest.raises(CoverageError) as error:
        validate_condition_ledger(
            task="bioavailability_ma",
            accepted_voters=[(source, decision)],
            ledger_path=ledger,
            allowed_condition_groups={"prandial_state=fasted"},
        )
    assert error.value.report["missing_voters"] == 1


def test_oral_local_voter_cannot_reuse_hf_condition_mapping(tmp_path):
    source, decision = _physical_oral_voter()
    review = {
        **_condition_review(source, decision),
        "mapping_origin": "unchanged_payload_v1_reuse",
        "reviewed_source_id": "hf_bioavailability",
    }
    ledger = tmp_path / "reviews.jsonl"
    ledger.write_text(json.dumps(review) + "\n")
    with pytest.raises(CoverageError) as error:
        validate_condition_ledger(
            task="bioavailability_ma",
            accepted_voters=[(source, decision)],
            ledger_path=ledger,
            allowed_condition_groups={"prandial_state=fasted"},
        )
    assert any(
        row["reason"] == "oral_local_inherited_hf_mapping"
        for row in error.value.report["problem_examples"]
    )


def test_condition_ledger_accepts_explicit_external_exclusion(tmp_path):
    source, decision = _physical_oral_voter()
    review = {
        **_condition_review(source, decision),
        "decision": EXCLUDED_EXTERNAL_CONDITION,
        "condition_group": "co_treatment=grapefruit_juice",
        "condition_atoms": ["co_treatment=grapefruit_juice"],
        "mapping_origin": "condition_allowlist_exclusion",
    }
    ledger = tmp_path / "reviews.jsonl"
    ledger.write_text(json.dumps(review) + "\n")
    rows = validate_condition_ledger(
        task="bioavailability_ma",
        accepted_voters=[(source, decision)],
        ledger_path=ledger,
        allowed_condition_groups={"prandial_state=fasted"},
    )
    assert rows[0]["decision"] == EXCLUDED_EXTERNAL_CONDITION


def test_deleted_or_nonvoter_ledger_row_blocks_publication(tmp_path):
    source, decision = _physical_oral_voter()
    extra = {
        **_condition_review(source, decision),
        "source_row_uid": "sr_" + "b" * 32,
    }
    ledger = tmp_path / "reviews.jsonl"
    ledger.write_text(
        json.dumps(_condition_review(source, decision)) + "\n" + json.dumps(extra) + "\n"
    )
    with pytest.raises(CoverageError) as error:
        validate_condition_ledger(
            task="bioavailability_ma",
            accepted_voters=[(source, decision)],
            ledger_path=ledger,
            allowed_condition_groups={"prandial_state=fasted"},
        )
    assert error.value.report["extra_nonvoters_or_deleted_duplicates"] == 1


def test_split_assignment_freezes_survivors_and_fills_retired_v1_slots():
    v1 = {
        ("parent-a", "condition-a"): {
            "split": "valid",
        }
    }
    rows = [
        {
            "molecule_identity_key": "parent-a",
            "condition_group": "condition-a",
            "bemis_murcko_scaffold": "repaired-scaffold-a",
        },
        {
            "molecule_identity_key": "parent-b",
            "condition_group": "condition-a",
            "bemis_murcko_scaffold": "scaffold-b",
        },
        {
            "molecule_identity_key": "parent-c",
            "condition_group": "condition-a",
            "bemis_murcko_scaffold": "scaffold-c",
        },
    ]

    assignments = _preserved_split_assignments(
        rows,
        v1_by_context=v1,
        parent_splits={"parent-a": "valid"},
        scaffold_splits={"scaffold-a": "valid"},
        external_scaffold_allocation={"repaired-scaffold-a": "train"},
        target_counts={"train": 1, "valid": 1, "test": 1},
    )

    assert assignments[("parent-a", "condition-a")] == "valid"
    assert assignments[("parent-b", "condition-a")] == "train"
    assert assignments[("parent-c", "condition-a")] == "test"


def test_frozen_scaffold_reallocates_repaired_parent():
    assignments = _preserved_split_assignments(
        [
            {
                "molecule_identity_key": "repaired-parent",
                "condition_group": "condition-a",
                "bemis_murcko_scaffold": "frozen-train-scaffold",
            }
        ],
        v1_by_context={
            ("repaired-parent", "condition-a"): {"split": "test"}
        },
        parent_splits={"repaired-parent": "test"},
        scaffold_splits={"frozen-train-scaffold": "train"},
        external_scaffold_allocation={},
        target_counts={"train": 0, "valid": 0, "test": 1},
    )
    assert assignments[("repaired-parent", "condition-a")] == "train"
