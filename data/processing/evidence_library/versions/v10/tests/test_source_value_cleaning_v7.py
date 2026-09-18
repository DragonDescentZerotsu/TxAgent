import json

from rdkit import Chem

from data.processing.evidence_library.shared.v2.normalization.source_value_cleaning import (
    SOURCE_VALUE_CLEANING_VERSION,
    clean_source_values,
)


def test_reviewed_repair_can_atomically_correct_pmid_and_smiles(tmp_path):
    wrong_smiles = "Cl.NCCCNCCCNC(CC1CCCCC1)CC1CCCCC1"
    correct_smiles = "CN(C)CCC1=CNC2=C1C=C(C=C2)O"
    record = {
        "cleaned_record_id": "fixture:bufotenine",
        "source_id": "direct_bbb",
        "source_sha256": "a" * 64,
        "source_row_number": 8,
        "source_record_id": "7",
        "measurement_text": None,
        "unit_text": None,
        "support_text": "The indexed molecule is bufotenine.",
        "pmid": "533889",
        "source_smiles": wrong_smiles,
        "canonical_smiles": wrong_smiles,
        "source_payload_json": json.dumps({"smiles": wrong_smiles}),
    }
    audit = {
        "audit_id": "bufotenine_fixture",
        "task_id": "bbb_martins",
        "source_id": "direct_bbb",
        "source_sha256": "a" * 64,
        "source_row_number": 8,
        "source_record_id": "7",
        "stored_smiles": wrong_smiles,
        "classification": "confirmed_mismatch",
    }
    repair = {
        "repair_id": "bufotenine_fixture",
        "task_id": "bbb_martins",
        "source_id": "direct_bbb",
        "source_sha256": "a" * 64,
        "source_row_number": 8,
        "source_record_id": "7",
        "before": {"smiles": wrong_smiles, "pmid": "533889"},
        "after": {"smiles": correct_smiles, "pmid": "533890"},
        "evidence": {
            "audit_id": "bufotenine_fixture",
            "note": "The indexed support names bufotenine; the article PMID and structure were both shifted to unrelated records.",
        },
    }
    repair_path = tmp_path / "repairs.jsonl"
    audit_path = tmp_path / "audit.jsonl"
    repair_path.write_text(json.dumps(repair) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(audit) + "\n", encoding="utf-8")

    result = clean_source_values(
        [record],
        task_id="bbb_martins",
        reviewed_repairs_path=repair_path,
        smiles_identity_audit_path=audit_path,
    )

    assert SOURCE_VALUE_CLEANING_VERSION == "starling_source_value_cleaning.v7"
    assert result.records[0]["pmid"] == "533890"
    assert result.records[0]["canonical_smiles"] == Chem.MolToSmiles(
        Chem.MolFromSmiles(correct_smiles), canonical=True, isomericSmiles=True
    )
    assert result.records[0]["support_text"] == record["support_text"]
    assert {row["field"] for row in result.audit_rows} == {"pmid", "smiles"}


def test_authoritative_source_values_supersede_older_field_repairs(
    tmp_path,
):
    repair = {
        "repair_id": "fixture",
        "task_id": "task",
        "source_id": "source",
        "source_sha256": "a" * 64,
        "source_row_number": 1,
        "source_record_id": "1",
        "before": {"smiles": "CCN", "pmid": "1"},
        "after": {"smiles": "CCC", "pmid": "2"},
        "evidence": {
            "audit_id": "fixture",
            "note": "Reviewed fixture with enough detail to establish the correction.",
        },
    }
    audit = {
        "audit_id": "fixture",
        "task_id": "task",
        "source_id": "source",
        "source_sha256": "a" * 64,
        "source_row_number": 1,
        "source_record_id": "1",
        "stored_smiles": "CCN",
        "classification": "confirmed_mismatch",
    }
    repair_path = tmp_path / "repairs.jsonl"
    audit_path = tmp_path / "audit.jsonl"
    repair_path.write_text(json.dumps(repair) + "\n")
    audit_path.write_text(json.dumps(audit) + "\n")
    record = {
        "cleaned_record_id": "fixture",
        "source_id": "source",
        "source_sha256": "b" * 64,
        "source_row_number": 1,
        "source_record_id": "1",
        "measurement_text": None,
        "unit_text": None,
        "support_text": "fixture",
        "pmid": "1",
        "source_smiles": "CCO",
        "canonical_smiles": "CCO",
        "source_payload_json": json.dumps({"smiles": "CCO"}),
    }

    result = clean_source_values(
        [record],
        task_id="task",
        reviewed_repairs_path=repair_path,
        smiles_identity_audit_path=audit_path,
        authoritative_source_values=True,
    )

    assert result.records[0]["canonical_smiles"] == "CCO"
    assert result.records[0]["pmid"] == "1"
    assert result.audit_rows == []


def test_protected_main_structure_skips_only_smiles_repair(tmp_path):
    uid = "sr_00000000000000000000000000000001"
    repair = {
        "repair_id": "fixture",
        "task_id": "task",
        "source_id": "source",
        "source_sha256": "a" * 64,
        "source_row_number": 1,
        "source_record_id": "1",
        "before": {"smiles": "CCN", "pmid": "1"},
        "after": {"smiles": "CCC", "pmid": "2"},
        "evidence": {
            "audit_id": "fixture",
            "note": "Reviewed fixture with enough detail to establish the correction.",
        },
    }
    audit = {
        "audit_id": "fixture",
        "task_id": "task",
        "source_id": "source",
        "source_sha256": "a" * 64,
        "source_row_number": 1,
        "source_record_id": "1",
        "stored_smiles": "CCN",
        "classification": "confirmed_mismatch",
    }
    repair_path = tmp_path / "repairs.jsonl"
    audit_path = tmp_path / "audit.jsonl"
    repair_path.write_text(json.dumps(repair) + "\n")
    audit_path.write_text(json.dumps(audit) + "\n")
    record = {
        "cleaned_record_id": "fixture",
        "source_row_uid": uid,
        "source_id": "source",
        "source_sha256": "b" * 64,
        "source_row_number": 1,
        "source_record_id": "1",
        "measurement_text": None,
        "unit_text": None,
        "support_text": "fixture",
        "pmid": None,
        "source_smiles": "CCO",
        "canonical_smiles": "CCO",
        "source_payload_json": json.dumps({"smiles": "CCO", "pmid": "1"}),
    }

    result = clean_source_values(
        [record],
        task_id="task",
        reviewed_repairs_path=repair_path,
        smiles_identity_audit_path=audit_path,
        allow_reviewed_source_hash_mismatch=True,
        protected_structure_uids={uid},
    )

    assert result.records[0]["canonical_smiles"] == "CCO"
    assert result.records[0]["pmid"] == "2"
    assert {row["field"] for row in result.audit_rows} == {"pmid"}
    assert result.manifest["reviewed_repairs"][
        "n_protected_structure_fields_skipped"
    ] == 1


def test_source_subset_requires_only_its_reviewed_repairs(tmp_path):
    def repair(source_id, before, after):
        return {
            "repair_id": f"repair_{source_id}",
            "task_id": "task",
            "source_id": source_id,
            "source_sha256": "a" * 64,
            "source_row_number": 1,
            "source_record_id": "0",
            "before": {"pmid": before},
            "after": {"pmid": after},
            "evidence": {
                "note": "This fixture verifies that a scoped canonical-source build does not require unrelated source repairs."
            },
        }

    path = tmp_path / "repairs.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (repair("selected", "1", "2"), repair("other", "3", "4"))
        )
        + "\n",
        encoding="utf-8",
    )
    record = {
        "cleaned_record_id": "selected:0",
        "source_id": "selected",
        "source_sha256": "a" * 64,
        "source_row_number": 1,
        "source_record_id": "0",
        "measurement_text": None,
        "unit_text": None,
        "support_text": "fixture",
        "pmid": "1",
    }

    result = clean_source_values(
        [record],
        task_id="task",
        reviewed_repairs_path=path,
        source_ids={"selected"},
    )

    assert result.records[0]["pmid"] == "2"
    assert result.manifest["reviewed_repairs"]["n_declared"] == 1


def test_row_subset_requires_only_repairs_inside_selected_universe(tmp_path):
    def repair(row_number, before, after):
        return {
            "repair_id": f"repair_{row_number}",
            "task_id": "task",
            "source_id": "source",
            "source_sha256": "a" * 64,
            "source_row_number": row_number,
            "source_record_id": str(row_number),
            "before": {"pmid": before},
            "after": {"pmid": after},
            "evidence": {
                "note": "This fixture verifies that repairs outside the selected UID universe remain registered but are not applied."
            },
        }

    path = tmp_path / "repairs.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (repair(1, "1", "2"), repair(2, "3", "4"))
        )
        + "\n",
        encoding="utf-8",
    )
    record = {
        "cleaned_record_id": "source:2",
        "source_id": "source",
        "source_sha256": "a" * 64,
        "source_row_number": 2,
        "source_record_id": "2",
        "measurement_text": None,
        "unit_text": None,
        "support_text": "fixture",
        "pmid": "3",
    }

    result = clean_source_values(
        [record],
        task_id="task",
        reviewed_repairs_path=path,
        reviewed_row_keys={("source", 2, "2")},
    )

    assert result.records[0]["pmid"] == "4"
    repair_manifest = result.manifest["reviewed_repairs"]
    assert repair_manifest["n_declared"] == 1
    assert repair_manifest["n_registry_declared"] == 2
    assert repair_manifest["n_out_of_scope"] == 1
