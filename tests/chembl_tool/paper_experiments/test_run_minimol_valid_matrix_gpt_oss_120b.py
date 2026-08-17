from tools.chembl_tool.paper_experiments.run_minimol_valid_matrix_gpt_oss_120b import (
    PHASES,
    command_for_phase,
    parse_args,
    phases_for_profile,
)


def test_current_valid_matrix_freezes_requested_retrieval_contract():
    args = parse_args([])
    commands = [command_for_phase(phase, args) for phase in PHASES]

    assert [phase.task for phase in PHASES] == [
        "bbb_martins",
        "bioavailability_ma",
        "skin_reaction",
    ]
    assert sum(len(phase.experiments) for phase in PHASES) == 19
    for command in commands:
        assert command[command.index("--evaluation-subset") + 1] == "valid"
        assert command[command.index("--retrieval-feature") + 1] == "minimol"
        assert command[command.index("--top-k-per-group") + 1] == "5"
        assert command[command.index("--min-similarity") + 1] == "0.0"
        assert command[command.index("--visibility-mode") + 1] == "identity_blind"
        assert command[command.index("--neighbor-identity-policy") + 1] == "parent_disjoint"
        assert command[command.index("--model") + 1] == "gpt-oss-120b"


def test_current_valid_matrix_uses_current_mixed_lineages():
    args = parse_args([])
    by_task = {phase.task: command_for_phase(phase, args) for phase in PHASES}

    bbb = by_task["bbb_martins"]
    assert bbb[bbb.index("--benchmark-lineage") + 1] == (
        "experimental_meaningful_cns_access_v2"
    )
    for task in ("bioavailability_ma", "skin_reaction"):
        command = by_task[task]
        assert command[command.index("--benchmark-lineage") + 1] == "record_supported_v2"


def test_starling_core_profile_keeps_only_direct_full_and_flat():
    phases = phases_for_profile("starling_core")

    assert {phase.task: phase.experiments for phase in phases} == {
        "bbb_martins": (
            "bbb_martins__starling_direct",
            "bbb_martins__starling_full_flat",
        ),
        "bioavailability_ma": (
            "bioavailability_ma__starling_direct_full",
            "bioavailability_ma__starling_full_flat",
        ),
        "skin_reaction": (
            "skin_reaction__starling_direct",
            "skin_reaction__starling_full_flat",
        ),
    }
