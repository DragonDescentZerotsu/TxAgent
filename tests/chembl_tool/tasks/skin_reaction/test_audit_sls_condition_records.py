from pathlib import Path

from tools.chembl_tool.tasks.skin_reaction.audit_sls_condition_records import audit


def test_real_sls_manual_audit_is_complete_and_fails_formal_group_gate(tmp_path: Path):
    summary = audit(
        audit_path=tmp_path / "audit.jsonl",
        summary_path=tmp_path / "summary.json",
    )

    assert summary["record_counts"] == {
        "all_condition_atom_records": 133,
        "pure_signature_records": 50,
        "composite_signature_records": 83,
        "strict_eligible_records": 4,
        "strict_excluded_records": 129,
    }
    assert summary["parent_label_counts"] == {
        "total": 4,
        "Y=0": 3,
        "Y=1": 1,
        "vote_rejected": 0,
    }
    assert summary["split_parent_counts"] == {
        "train": 4,
        "valid": 0,
        "test": 0,
        "unassigned": 0,
    }
    assert summary["distinct_scaffold_count"] == 1
    assert summary["promotion_gates"] == {
        "at_least_3_parents": True,
        "at_least_3_distinct_scaffolds": False,
        "train_valid_test_each_nonempty": False,
        "all_parents_assigned": True,
    }
    assert summary["formal_benchmark_decision"] == "reject_group_from_formal_benchmark"
