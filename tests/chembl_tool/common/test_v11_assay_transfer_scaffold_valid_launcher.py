from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from tools.chembl_tool.common.task_workflows.reasoning_batch import (
    _parse_args as parse_batch_args,
)
from tools.chembl_tool.paper_experiments import (
    run_v11_assay_transfer_scaffold_valid as launcher,
)


def _args(tmp_path: Path):
    return launcher._parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--base-url",
            "http://127.0.0.1:50000/v1",
            "--parallelism",
            "128",
        ]
    )


def _command_map(commands):
    return {command.experiment_name: command.command for command in commands}


def _value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_launcher_builds_exact_v11_k3_commands(tmp_path):
    commands = _command_map(launcher.build_batch_commands(_args(tmp_path)))
    assert set(commands) == {
        "bbb_martins__starling_full_flat__assay_transfer_v11_scored_k3_morgan50",
        "bioavailability_ma__starling_full_flat__assay_transfer_v11_scored_assay_schema_k3_morgan50",
        "skin_reaction__starling_full_flat__assay_transfer_v11_scored_k3_morgan50",
    }

    for command in commands.values():
        assert _value(command, "--experiment-mode") == "full_flat"
        assert _value(command, "--retrieval-source") == "starling"
        assert _value(command, "--neighbor-identity-policy") == "parent_disjoint"
        assert _value(command, "--retrieval-strategy") == "assay_transfer_tool"
        assert _value(command, "--assay-transfer-profile") == "v11_with_categorical"
        assert _value(command, "--assay-transfer-initial-morgan-filter") == "50"
        assert _value(command, "--assay-transfer-records-per-molecule") == "1"
        assert _value(command, "--top-k-per-group") == "3"
        assert _value(command, "--min-similarity") == "0.0"
        assert _value(command, "--rerank-cache-mode") == "read_only"
        assert _value(command, "--neighbor-context-profile") == "standard"
        assert _value(command, "--model") == "nvidia/GLM-5.2-NVFP4"
        assert _value(command, "--reasoning-effort") == ""
        assert _value(command, "--tool-service-url") == "http://127.0.0.1:8766"
        assert "--identity-blind" in command
        assert "--enable-assay-transfer-scores" in command
        assert "--disable-thinking" in command
        assert "--skip-existing" in command
        assert "--assay-transfer-min-score" not in command
        assert "--single-analysis-source-batch" not in command
        assert "--exclude-nondirect-bioavailability-records" not in command

    bbb = next(value for key, value in commands.items() if key.startswith("bbb_martins"))
    bio = next(value for key, value in commands.items() if key.startswith("bioavailability"))
    assert _value(bio, "--group-output-schema") == "assay-transfer"
    assert _value(bio, "--group-prompt-version") == "bioavailability_text_v1"
    assert _value(bio, "--rerank-expected-score-count") == "870332"
    assert _value(bbb, "--rerank-expected-score-count") == "633694"
    assert "jiosephlee/assay-transfer-tool-soft-v11-bbb-martins-with-categorical" in bbb
    assert "jiosephlee/assay-transfer-tool-soft-v11-bioavailability-ma-with-categorical" in bio
    assert "record_supported_v2/evidence/bioavailability_starling_v7/08_neighbor_index" in _value(bio, "--index")
    assert "experimental_meaningful_cns_access_v2/evidence/bbb_starling_v7/08_neighbor_index" in _value(bbb, "--index")


def test_commands_retain_v11_paths_after_real_task_parser_normalization(tmp_path):
    for spec in launcher.build_batch_commands(_args(tmp_path)):
        config = importlib.import_module(spec.command[2]).CONFIG
        parsed = parse_batch_args(config, spec.command[3:])
        assert parsed.rerank_catalog == ""
        assert parsed.rerank_cache.endswith("/scores.sqlite3")
        assert parsed.rerank_candidate_manifest == ""
        assert parsed.rerank_cache_version_manifest.endswith("/VERSION.json")
        assert parsed.assay_transfer_model.startswith("jiosephlee/")
        assert len(parsed.assay_transfer_model_revision) == 40


def test_limit_is_applied_to_both_tasks(tmp_path):
    args = launcher._parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--base-url",
            "http://127.0.0.1:50000/v1",
            "--limit",
            "1",
        ]
    )
    for command in launcher.build_batch_commands(args):
        assert _value(command.command, "--limit") == "1"


def test_task_scope_includes_ready_skin_cache_and_supports_explicit_subset():
    default_args = launcher._parse_args([])
    assert default_args.tasks == ["bbb_martins", "bioavailability_ma", "skin_reaction"]
    skin_args = launcher._parse_args(["--tasks", "skin_reaction"])
    assert [
        command.experiment_name.split("__", 1)[0]
        for command in launcher.build_batch_commands(skin_args)
    ] == ["skin_reaction"]


def test_disable_flat_tools_is_isolated_and_forwarded_to_all_tasks():
    args = launcher._parse_args(["--disable-flat-tools"])
    assert Path(args.output_root).name.endswith("_no_flat_tools")
    for spec in launcher.build_batch_commands(args):
        assert "--disable-flat-tools" in spec.command
        parsed = parse_batch_args(
            importlib.import_module(spec.command[2]).CONFIG,
            spec.command[3:],
        )
        assert parsed.disable_flat_tools is True


def test_disable_flat_tools_rejects_mechanism_mode():
    with pytest.raises(SystemExit):
        launcher._parse_args(
            ["--experiment-mode", "full_mechanism", "--disable-flat-tools"]
        )


def test_analogous_flat_v1_is_isolated_and_forwarded_to_all_tasks():
    args = launcher._parse_args(["--analogous-reasoning-only"])
    assert Path(args.output_root).name.endswith("_analogous_flat_v1")
    for spec in launcher.build_batch_commands(args):
        assert "--analogous-reasoning-only" in spec.command
        assert "--group-output-schema" not in spec.command
        assert "--group-prompt-version" not in spec.command
        parsed = parse_batch_args(
            importlib.import_module(spec.command[2]).CONFIG,
            spec.command[3:],
        )
        assert parsed.analogous_reasoning_only is True
        assert parsed.experiment_mode == "full_flat"
        assert parsed.identity_blind is True
        assert parsed.enable_assay_transfer_scores is True


def test_retrieval_comparison_uses_one_nine_batch_prompt_pool(tmp_path):
    args = launcher._parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--analogous-reasoning-only",
            "--retrieval-condition",
            "morgan",
            "--retrieval-condition",
            "assay_transfer_record",
            "--retrieval-condition",
            "assay_transfer_molecule",
        ]
    )
    commands = launcher.build_batch_commands(args)

    assert len(commands) == 9
    assert len({command.experiment_name for command in commands}) == 9
    for spec in commands:
        command = spec.command
        assert _value(command, "--top-k-per-group") == "3"
        assert _value(command, "--min-similarity") == "0.0"
        assert _value(command, "--neighbor-identity-policy") == "parent_disjoint"
        assert "--identity-blind" in command
        assert "--analogous-reasoning-only" in command
        parsed = parse_batch_args(
            importlib.import_module(command[2]).CONFIG,
            command[3:],
        )
        assert parsed.experiment_mode == "full_flat"
        if "__morgan__" in spec.experiment_name:
            assert parsed.retrieval_strategy == "morgan_fingerprint"
            assert parsed.enable_assay_transfer_scores is False
            assert "--rerank-cache" not in command
        elif "__assay_transfer_record__" in spec.experiment_name:
            assert parsed.retrieval_strategy == "assay_transfer_tool"
            assert parsed.assay_transfer_selection_unit == "scored_record"
            assert parsed.assay_transfer_records_per_molecule == 1
        else:
            assert "__assay_transfer_molecule__" in spec.experiment_name
            assert parsed.retrieval_strategy == "assay_transfer_tool"
            assert parsed.assay_transfer_selection_unit == "mean_score_molecule"
            assert parsed.assay_transfer_records_per_molecule == 6


def test_retrieval_conditions_require_analogous_prompt():
    with pytest.raises(SystemExit):
        launcher._parse_args(["--retrieval-condition", "morgan"])


def test_analogous_reasoning_rejects_mechanism_launcher_mode():
    with pytest.raises(SystemExit):
        launcher._parse_args(
            ["--experiment-mode", "full_mechanism", "--analogous-reasoning-only"]
        )


def test_unique_molecule_variant_is_isolated_and_reuses_exact_singles(tmp_path):
    args = launcher._parse_args(
        [
            "--base-url",
            "http://127.0.0.1:50000/v1",
            "--assay-transfer-selection-unit",
            "unique_molecule",
        ]
    )
    assert Path(args.output_root).name == "full_flat_k3_unique_molecules_scored_assay_schema"
    for spec in launcher.build_batch_commands(args):
        assert "unique_molecule_k3_morgan50" in spec.experiment_name
        assert _value(spec.command, "--assay-transfer-selection-unit") == "unique_molecule"
        assert "--single-analysis-source-batch" not in spec.command


def test_k7_variants_have_isolated_roots_ids_and_reuse_exact_singles():
    for selection_unit, root_suffix, batch_marker in (
        ("scored_record", "k7_scored_assay_schema", "_k7_morgan50"),
        (
            "unique_molecule",
            "k7_unique_molecules_scored_assay_schema",
            "_unique_molecule_k7_morgan50",
        ),
    ):
        args = launcher._parse_args(
            [
                "--base-url",
                "http://127.0.0.1:50000/v1",
                "--assay-transfer-selection-unit",
                selection_unit,
                "--top-k-per-group",
                "7",
            ]
        )
        assert Path(args.output_root).name.endswith(root_suffix)
        for spec in launcher.build_batch_commands(args):
            assert batch_marker in spec.experiment_name
            assert _value(spec.command, "--top-k-per-group") == "7"
            assert "--single-analysis-source-batch" not in spec.command


def test_multiple_records_per_molecule_are_isolated_and_forwarded():
    args = launcher._parse_args(
        [
            "--base-url",
            "http://127.0.0.1:50000/v1",
            "--assay-transfer-selection-unit",
            "unique_molecule",
            "--assay-transfer-records-per-molecule",
            "4",
        ]
    )

    assert Path(args.output_root).name.endswith("_r4")
    for spec in launcher.build_batch_commands(args):
        assert "_unique_molecule_r4_k3_morgan50" in spec.experiment_name
        assert _value(spec.command, "--assay-transfer-records-per-molecule") == "4"
        parsed = parse_batch_args(importlib.import_module(spec.command[2]).CONFIG, spec.command[3:])
        assert parsed.assay_transfer_records_per_molecule == 4


def test_multiple_records_per_molecule_require_unique_molecule_selection():
    with pytest.raises(SystemExit) as exc_info:
        launcher._parse_args(
            [
                "--base-url",
                "http://127.0.0.1:50000/v1",
                "--assay-transfer-records-per-molecule",
                "2",
            ]
        )
    assert exc_info.value.code == 2


@pytest.mark.parametrize("value", ["0", "11"])
def test_records_per_molecule_is_bounded_to_ten(value):
    with pytest.raises(SystemExit) as exc_info:
        launcher._parse_args(
            [
                "--base-url",
                "http://127.0.0.1:50000/v1",
                "--assay-transfer-selection-unit",
                "unique_molecule",
                "--assay-transfer-records-per-molecule",
                value,
            ]
        )
    assert exc_info.value.code == 2


def test_structural_diversity_uses_task_slacks_and_isolated_artifacts():
    args = launcher._parse_args(
        [
            "--base-url",
            "http://127.0.0.1:50000/v1",
            "--assay-transfer-selection-unit",
            "unique_molecule",
            "--assay-transfer-diversity-mode",
            "structural",
            "--bioavailability-diversity-score-slack",
            "0.009",
            "--skin-reaction-diversity-score-slack",
            "0.025",
        ]
    )
    assert Path(args.output_root).name == (
        "full_flat_k3_unique_molecules_"
        "structural_diversity_scored_assay_schema"
    )
    commands = _command_map(launcher.build_batch_commands(args))
    for batch_id, command in commands.items():
        assert "_unique_molecule_structural_diversity_k3_morgan50" in batch_id
        assert _value(command, "--assay-transfer-selection-unit") == "unique_molecule"
        assert _value(command, "--assay-transfer-diversity-mode") == "structural"
        expected_slack = {
            "bbb_martins": "0.0",
            "bioavailability_ma": "0.009",
            "skin_reaction": "0.025",
        }[batch_id.split("__", 1)[0]]
        assert _value(command, "--assay-transfer-diversity-score-slack") == expected_slack
        assert "--single-analysis-source-batch" not in command


def test_diversity_requires_unique_molecule_selection():
    try:
        launcher._parse_args(
            [
                "--base-url",
                "http://127.0.0.1:50000/v1",
                "--assay-transfer-diversity-mode",
                "structural",
                "--bioavailability-diversity-score-slack",
                "0.009",
                "--skin-reaction-diversity-score-slack",
                "0.025",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected diversity/selection-unit validation to fail")


def test_top_k_cannot_exceed_initial_morgan_pool():
    try:
        launcher._parse_args(
            [
                "--base-url",
                "http://127.0.0.1:50000/v1",
                "--top-k-per-group",
                "51",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected top-k validation to fail")


def test_manifest_contract_never_contains_api_key_value(tmp_path, monkeypatch):
    secret = "must-not-be-persisted"
    monkeypatch.setenv("LITELLM_API_KEY", secret)
    monkeypatch.setattr(
        launcher,
        "_validate_inputs",
        lambda _tasks=None: {"bbb_martins": {}, "bioavailability_ma": {}},
    )
    assert launcher.main(
        [
            "--output-root",
            str(tmp_path),
            "--base-url",
            "http://127.0.0.1:50000/v1",
            "--manifest-only",
        ]
    ) == 0
    raw = (tmp_path / "launch_manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(raw)
    assert secret not in raw
    assert manifest["api_key_env"] == "LITELLM_API_KEY"
    assert manifest["logit_extraction_dtype"] == "float32"
    assert manifest["assay_transfer_min_score"] is None
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    assert tasks["bbb_martins"]["group_output_schema"] == ""
    assert tasks["bioavailability_ma"]["group_output_schema"] == "assay-transfer"
    assert set(tasks) == {"bbb_martins", "bioavailability_ma", "skin_reaction"}
