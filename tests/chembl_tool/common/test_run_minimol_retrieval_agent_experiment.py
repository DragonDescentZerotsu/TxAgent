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
        output_dir=tmp_path,
        api_key_env="LITELLM_API_KEY",
        base_url="https://litellm.example.test/v1",
        model="nvidia/GLM-5.2-NVFP4",
        reasoning_effort="",
    )

    runner._run_matrix("operational", args)

    assert len(captured) == 2
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
    assert all(command[command.index("--base-url") + 1] == "https://litellm.example.test/v1"
               for _, command in captured)
    assert all(command[command.index("--api-key-env") + 1] == "LITELLM_API_KEY"
               for _, command in captured)
    assert all(command[command.index("--model") + 1] == "nvidia/GLM-5.2-NVFP4"
               for _, command in captured)
    assert all(command[command.index("--reasoning-effort") + 1] == ""
               for _, command in captured)


def test_minimol_launcher_defaults_to_local_litellm_config():
    args = runner._parse_args([])

    assert args.base_url is None
    assert args.api_key_env == "LITELLM_API_KEY"
    assert args.model == "nvidia/GLM-5.2-NVFP4"
