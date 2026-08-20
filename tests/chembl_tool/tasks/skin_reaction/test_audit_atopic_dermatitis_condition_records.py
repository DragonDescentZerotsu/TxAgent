from pathlib import Path

from tools.chembl_tool.tasks.skin_reaction.audit_atopic_dermatitis_condition_records import (
    audit,
)


def test_real_atopic_dermatitis_audit_passes_formal_group_gate(tmp_path: Path):
    summary = audit(
        audit_path=tmp_path / "audit.jsonl",
        summary_path=tmp_path / "summary.json",
    )

    assert summary["record_counts"] == {
        "all_condition_atom_records": 388,
        "pure_signature_records": 191,
        "composite_signature_records": 197,
        "strict_eligible_records": 57,
        "strict_excluded_records": 331,
    }
    assert summary["parent_label_counts"] == {
        "total": 35,
        "Y=0": 3,
        "Y=1": 32,
        "vote_rejected": 0,
    }
    assert summary["split_parent_counts"] == {
        "train": 31,
        "valid": 1,
        "test": 3,
        "unassigned": 0,
    }
    assert summary["distinct_scaffold_count"] == 18
    assert all(summary["promotion_gates"].values())
    assert summary["formal_benchmark_decision"] == "promote_group"
