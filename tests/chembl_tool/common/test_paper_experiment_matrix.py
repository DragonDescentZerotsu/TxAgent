import argparse

from tools.chembl_tool.paper_experiments.molecular_evidence_agent import EXPERIMENTS, _command


def test_frozen_matrix_has_unique_expected_conditions():
    names = [experiment.name for experiment in EXPERIMENTS]
    assert len(names) == 21
    assert len(names) == len(set(names))
    assert "bioavailability_ma__starling_full_mechanism" in names
    assert "bbb_martins__starling_direct" in names


def test_matrix_command_freezes_glm_and_identity_conditions():
    args = argparse.Namespace(
        python_executable="python",
        api_key_env="GLM_API_KEY",
        parallelism=2,
        group_workers=3,
    )
    command = _command(EXPERIMENTS[0], args)

    assert "--identity-blind" in command
    assert command[command.index("--temperature") + 1] == "0"
    assert command[command.index("--max-tokens") + 1] == "20480"
    assert command[command.index("--api-key-env") + 1] == "GLM_API_KEY"
    assert "--single-analysis-source-batch" not in command
    assert not any(part.startswith("sk-") for part in command)

    direct_command = _command(EXPERIMENTS[1], args)
    assert "--single-analysis-source-batch" in direct_command
    assert direct_command[direct_command.index("--single-analysis-source-batch") + 1].endswith(
        "bbb_martins__none"
    )
