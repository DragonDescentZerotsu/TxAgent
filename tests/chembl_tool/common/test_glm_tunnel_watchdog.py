from argparse import Namespace
from pathlib import Path

from tools.chembl_tool.paper_experiments.watch_glm_tunnel_and_matrix import (
    _run_root_name,
    count_final_results,
    matrix_command,
)


def test_run_root_name_tracks_visibility_and_parent_policy() -> None:
    assert (
        _run_root_name("deployment_visible", "parent_disjoint")
        == "runs_deployment_visible_parent_disjoint"
    )
    assert (
        _run_root_name("identity_blind", "parent_disjoint")
        == "runs_identity_blind_parent_disjoint"
    )
    assert _run_root_name("deployment_visible", "operational") == "runs_deployment_visible"
    assert (
        _run_root_name("identity_blind", "scaffold_disjoint")
        == "runs_identity_blind_scaffold_disjoint"
    )


def test_count_final_results_uses_only_selected_run_root(tmp_path: Path) -> None:
    selected = (
        tmp_path
        / "runs_deployment_visible_parent_disjoint"
        / "bbb_martins"
        / "condition"
    )
    other = (
        tmp_path
        / "runs_identity_blind_parent_disjoint"
        / "bbb_martins"
        / "condition"
    )
    for index in range(3):
        run_dir = selected / f"run-{index}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "final_reasoning_output.json").write_text(
            '{"status":"ok","llm":{"content":{"bbb_prediction":"pass"}}}\n',
            encoding="utf-8",
        )
        (run_dir / "single_molecule_reasoning_output.json").write_text(
            '{"status":"ok"}\n', encoding="utf-8"
        )
        (run_dir / "manifest.json").write_text(
            '{"n_groups_with_neighbors":0}\n', encoding="utf-8"
        )
    run_dir = other / "run-0"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "final_reasoning_output.json").write_text(
        '{"status":"ok","llm":{"content":{"bbb_prediction":"pass"}}}\n',
        encoding="utf-8",
    )
    (run_dir / "single_molecule_reasoning_output.json").write_text(
        '{"status":"ok"}\n', encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        '{"n_groups_with_neighbors":0}\n', encoding="utf-8"
    )

    assert count_final_results(tmp_path, "deployment_visible", "parent_disjoint") == 3


def test_count_final_results_rejects_failed_or_missing_group_outputs(
    tmp_path: Path,
) -> None:
    run_dir = (
        tmp_path
        / "runs_deployment_visible_parent_disjoint"
        / "bbb_martins"
        / "condition"
        / "run-0"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "final_reasoning_output.json").write_text(
        '{"status":"ok","llm":{"content":{"bbb_prediction":"pass"}}}\n',
        encoding="utf-8",
    )
    (run_dir / "single_molecule_reasoning_output.json").write_text(
        '{"status":"ok"}\n', encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        '{"n_groups_with_neighbors":1}\n', encoding="utf-8"
    )
    (run_dir / "group_reasoning_outputs.jsonl").write_text(
        '{"status":"error","error":"Connection error."}\n', encoding="utf-8"
    )

    assert count_final_results(tmp_path, "deployment_visible", "parent_disjoint") == 0

def test_count_final_results_rejects_duplicate_or_substituted_groups(
    tmp_path: Path,
) -> None:
    run_dir = (
        tmp_path
        / "runs_deployment_visible_parent_disjoint"
        / "bbb_martins"
        / "condition"
        / "run-0"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "final_reasoning_output.json").write_text(
        '{"status":"ok","llm":{"content":{"bbb_prediction":"pass"}}}\n',
        encoding="utf-8",
    )
    (run_dir / "single_molecule_reasoning_output.json").write_text(
        '{"status":"ok"}\n', encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        '{"n_groups_with_neighbors":2}\n', encoding="utf-8"
    )
    (run_dir / "retrieval.json").write_text(
        '{"groups":['
        '{"group_id":"Mechanism.a","neighbors":[{"rank":1}]},'
        '{"group_id":"Mechanism.b","neighbors":[{"rank":1}]}'
        ']}\n',
        encoding="utf-8",
    )
    (run_dir / "group_reasoning_outputs.jsonl").write_text(
        '{"group_id":"Mechanism.a","status":"ok"}\n'
        '{"group_id":"Mechanism.a","status":"ok"}\n',
        encoding="utf-8",
    )

    assert count_final_results(tmp_path, "deployment_visible", "parent_disjoint") == 0

    (run_dir / "group_reasoning_outputs.jsonl").write_text(
        '{"group_id":"Mechanism.a","status":"ok"}\n'
        '{"group_id":"Mechanism.b","status":"ok"}\n',
        encoding="utf-8",
    )
    assert count_final_results(tmp_path, "deployment_visible", "parent_disjoint") == 1


def test_count_final_results_rejects_missing_prediction(tmp_path: Path) -> None:
    run_dir = (
        tmp_path
        / "runs_deployment_visible_parent_disjoint"
        / "bbb_martins"
        / "condition"
        / "run-0"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "final_reasoning_output.json").write_text(
        '{"status":"ok","llm":{"content":{}}}\n', encoding="utf-8"
    )
    (run_dir / "single_molecule_reasoning_output.json").write_text(
        '{"status":"ok"}\n', encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        '{"n_groups_with_neighbors":0}\n', encoding="utf-8"
    )

    assert count_final_results(tmp_path, "deployment_visible", "parent_disjoint") == 0


def test_matrix_command_preserves_frozen_runtime_contract() -> None:
    args = Namespace(
        python_executable="/env/bin/python",
        benchmark_split="scaffold",
        evaluation_subset="valid",
        visibility_mode="deployment_visible",
        neighbor_identity_policy="parent_disjoint",
        api_key_env="GLM_LOCAL_API_KEY",
        base_url="http://127.0.0.1:50000/v1",
        model="nvidia/GLM-5.2-NVFP4",
        output_root="/tmp/results",
        parallelism=128,
        max_stage_requeues=2,
        timeout_s=600,
    )
    command = matrix_command(args)

    assert command[:4] == [
        "/env/bin/python",
        "-u",
        "-m",
        "tools.chembl_tool.paper_experiments.starling_benchmark_matrix",
    ]
    assert command[command.index("--reasoning-effort") + 1] == ""
    assert command[command.index("--parallelism") + 1] == "128"
    assert "--condition-workers" not in command
    assert "--scheduler" not in command
    assert command[command.index("--max-stage-requeues") + 1] == "2"
    assert command[command.index("--timeout-s") + 1] == "600"
