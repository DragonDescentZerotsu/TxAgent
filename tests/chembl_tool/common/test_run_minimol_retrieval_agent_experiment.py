from argparse import Namespace

from tools.chembl_tool.paper_experiments import (
    run_minimol_retrieval_agent_experiment as runner,
)


def test_matrix_orchestration_covers_all_feature_conditions(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(
        runner,
        "_run_commands",
        lambda commands, *, log_root: captured.extend(commands),
    )
    args = Namespace(
        python_executable="python",
        parallelism=8,
        group_workers=4,
        output_dir=tmp_path,
    )

    runner._run_matrix("operational", args)

    assert len(captured) == 6
    selected = [
        value
        for _, command in captured
        for value in command[command.index("--experiments") + 1 : command.index("--parallelism")]
    ]
    assert len(selected) == 38
    assert all("__none" not in name for name in selected)
    assert all("--retrieval-feature" in command for _, command in captured)
    assert all(command[command.index("--neighbor-identity-policy") + 1] == "operational"
               for _, command in captured)
