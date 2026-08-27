import json

from tools.chembl_tool.common.starling.build_record_supported_benchmark import (
    allocate_scaffold_groups,
    build_task_preserving_split,
    target_eval_size,
)


def _row(scaffold: str, label: int, records: int, identity: str, prior=False):
    return {
        "bemis_murcko_scaffold": scaffold,
        "Y": label,
        "source_record_count": records,
        "molecule_identity_key": identity,
        "drug": identity,
        "_prior_valid": prior,
    }


def test_record_supported_targets_keep_bbb_cap_and_other_811_sizes():
    assert target_eval_size("BBB_Martins", 19_425) == 500
    assert target_eval_size("Bioavailability_Ma", 2_092) == 209
    assert target_eval_size("Skin_Reaction", 2_456) == 245


def test_scaffold_allocator_prioritizes_record_support_and_zero_group_overlap():
    rows = [
        _row("A", 0, 2, "A0", prior=True),
        _row("B", 1, 2, "B1", prior=True),
        _row("C", 0, 2, "C0"),
        _row("D", 1, 2, "D1"),
        _row("E", 0, 1, "E0"),
        _row("F", 1, 1, "F1"),
    ]

    assignment, audit = allocate_scaffold_groups(rows, target_size=2)

    assert {assignment["A"], assignment["B"]} == {"valid"}
    assert {assignment["C"], assignment["D"]} == {"test"}
    assert assignment["E"] == assignment["F"] == "train"
    assert audit["minimum_singletons_in_heldout"] == 0
    assert audit["maximum_prior_valid_reuse"] == 2


def test_scaffold_allocator_accepts_a_lineage_specific_seed():
    rows = [
        _row(chr(65 + index), index % 2, 2, str(index))
        for index in range(8)
    ]
    first, _ = allocate_scaffold_groups(rows, target_size=2, seed=17)
    replay, _ = allocate_scaffold_groups(rows, target_size=2, seed=17)
    assert first == replay


def test_scaffold_allocator_can_disable_record_support_and_keep_acyclic_train_only():
    rows = [
        _row("", 0, 2, "acyclic"),
        _row("A", 0, 1, "A0"),
        _row("B", 1, 1, "B1"),
        _row("C", 0, 1, "C0"),
        _row("D", 1, 1, "D1"),
        _row("E", 0, 1, "E0"),
        _row("F", 1, 1, "F1"),
    ]

    assignment, audit = allocate_scaffold_groups(
        rows,
        target_size=2,
        optimize_record_support=False,
        exclude_empty_scaffold_from_heldout=True,
    )

    assert assignment[""] == "train"
    assert audit["record_support_objective_enabled"] is False
    assert audit["empty_scaffold_heldout_eligible"] is False
    assert audit["minimum_singletons_in_heldout"] is None
    assert audit["n_empty_scaffold_parents"] == 1


def test_preserved_split_can_inherit_new_parent_scaffolds(tmp_path):
    reference = tmp_path / "reference"
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_task = source_root / "BBB_Martins"
    source_task.mkdir(parents=True)
    rows = {
        "train": [_row("train-scaffold", 0, 2, "train-old")],
        "valid": [_row("valid-scaffold", 1, 2, "valid-old")],
        "test": [_row("test-scaffold", 0, 2, "test-old")],
    }
    reference.mkdir(parents=True)
    for split, values in rows.items():
        (reference / f"{split}_molecule_labels.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in values)
        )
    source_rows = [
        *[row for values in rows.values() for row in values],
        _row("test-scaffold", 1, 1, "test-new"),
        _row("novel-scaffold", 0, 1, "train-new"),
    ]
    (source_task / "molecule_labels.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in source_rows)
    )
    (source_task / "summary.json").write_text(json.dumps({"paths": {}}))

    summary = build_task_preserving_split(
        "BBB_Martins",
        source_root=source_root,
        output_root=output_root,
        reference_split_root=reference,
        lineage="test-lineage",
        protocol_version="test.v1",
        allow_new_parents=True,
    )

    test_rows = {
        json.loads(line)["molecule_identity_key"]
        for line in (
            output_root / "BBB_Martins/scaffold/test_molecule_labels.jsonl"
        ).read_text().splitlines()
    }
    train_rows = {
        json.loads(line)["molecule_identity_key"]
        for line in (
            output_root / "BBB_Martins/scaffold/train_molecule_labels.jsonl"
        ).read_text().splitlines()
    }
    assert "test-new" in test_rows
    assert "train-new" in train_rows
    optimization = summary["splits"]["scaffold"]["optimization"]
    assert optimization["n_added_parents"] == 2
    assert optimization["added_parent_assignment_counts"] == {"test": 1, "train": 1}
