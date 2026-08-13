from tools.chembl_tool.common.starling.build_record_supported_benchmark import (
    allocate_scaffold_groups,
    target_eval_size,
)


def _row(scaffold: str, label: int, records: int, identity: str, prior=False):
    return {
        "bemis_murcko_scaffold": scaffold,
        "Y": label,
        "source_record_count": records,
        "molecule_identity_key": identity,
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
