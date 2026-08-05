import json
import importlib
from pathlib import Path

import pytest

from tools.chembl_tool.common.task_workflows.reasoning_batch import (
    BatchConfig,
    _filter_final_only_run_artifacts,
    _parse_args,
    _result_is_complete,
    _single_run_command,
)
from tools.chembl_tool.tasks.bbb_martins.run_reasoning_batch import (
    CONFIG as BBB_CONFIG,
)
from tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch import (
    CONFIG as BIOAVAILABILITY_CONFIG,
)
from tools.chembl_tool.tasks.clintox.run_reasoning_batch import (
    CONFIG as CLINTOX_CONFIG,
)
from tools.chembl_tool.tasks.dili.run_reasoning_batch import CONFIG as DILI_CONFIG
from tools.chembl_tool.tasks.skin_reaction.run_reasoning_batch import (
    CONFIG as SKIN_CONFIG,
)


def test_result_completion_requires_all_reasoning_stages():
    complete = {
        "final_status": "ok",
        "pred_label": 1,
        "single_status": "ok",
        "n_groups_with_neighbors": 2,
        "n_group_outputs": 2,
        "n_failed_group_outputs": 0,
    }

    assert _result_is_complete(complete)
    assert not _result_is_complete({**complete, "single_status": "error"})
    assert not _result_is_complete({**complete, "n_group_outputs": 1})
    assert not _result_is_complete({**complete, "n_failed_group_outputs": 1})
    assert not _result_is_complete({**complete, "pred_label": None})


def test_result_completion_accepts_a_retrieval_free_run():
    assert _result_is_complete(
        {
            "final_status": "ok",
            "pred_label": 0,
            "single_status": "ok",
            "n_groups_with_neighbors": 0,
            "n_group_outputs": 0,
            "n_failed_group_outputs": 0,
        }
    )


def test_result_completion_requires_the_exact_expected_group_ids():
    complete = {
        "final_status": "ok",
        "pred_label": 1,
        "single_status": "ok",
        "n_groups_with_neighbors": 2,
        "n_group_outputs": 2,
        "n_failed_group_outputs": 0,
        "expected_group_ids": ["Mechanism.a", "Mechanism.b"],
        "group_ids": ["Mechanism.a", "Mechanism.b"],
    }

    assert _result_is_complete(complete)
    assert not _result_is_complete(
        {**complete, "group_ids": ["Mechanism.a", "Mechanism.a"]}
    )
    assert not _result_is_complete(
        {**complete, "group_ids": ["Mechanism.a", "Mechanism.c"]}
    )


@pytest.mark.parametrize(
    "selector_flag",
    ("--neighbor-selector", "--morgan-neighbor-selector"),
)
def test_batch_forwards_neighbor_selector_to_single_run(selector_flag):
    config = BatchConfig(
        description="test",
        default_input="input.jsonl",
        default_batch_root="batch-root",
        default_index="index.pkl",
        default_model="model",
        batch_id_prefix="batch",
        pipeline_module="test.pipeline",
        log_prefix="test",
        report_title="test",
        prediction_field="prediction",
        canonical_positive="positive",
        canonical_negative="negative",
        positive_predictions=frozenset({"positive"}),
        negative_predictions=frozenset({"negative"}),
    )
    args = _parse_args(
        config,
        [selector_flag, "query_feature_coverage"],
    )

    command = _single_run_command(config, args, 7, "run-7", Path("run-root"))

    selector_position = command.index("--morgan-neighbor-selector")
    assert command[selector_position + 1] == "query_feature_coverage"


def test_batch_forwards_neighbor_context_profile_to_single_run():
    config = BatchConfig(
        description="test",
        default_input="input.jsonl",
        default_batch_root="batch-root",
        default_index="index.pkl",
        default_model="model",
        batch_id_prefix="batch",
        pipeline_module="test.pipeline",
        log_prefix="test",
        report_title="test",
        prediction_field="prediction",
        canonical_positive="positive",
        canonical_negative="negative",
        positive_predictions=frozenset({"positive"}),
        negative_predictions=frozenset({"negative"}),
    )
    args = _parse_args(
        config,
        ["--neighbor-context-profile", "coverage_aware"],
    )

    command = _single_run_command(config, args, 7, "run-7", Path("run-root"))

    profile_position = command.index("--neighbor-context-profile")
    assert command[profile_position + 1] == "coverage_aware"


@pytest.mark.parametrize(
    "config",
    (BBB_CONFIG, BIOAVAILABILITY_CONFIG, CLINTOX_CONFIG, DILI_CONFIG, SKIN_CONFIG),
)
def test_prepare_command_is_accepted_by_every_task_pipeline(config):
    args = _parse_args(config, [])
    command = _single_run_command(
        config,
        args,
        7,
        "run-7",
        Path("run-root"),
    )
    pipeline = importlib.import_module(config.pipeline_module)

    parsed = pipeline._parse_args([*command[3:], "--prepare-only"])

    assert parsed.prepare_only is True


def test_legacy_task_rejects_unimplemented_shared_retrieval_contract():
    with pytest.raises(SystemExit):
        _parse_args(DILI_CONFIG, ["--neighbor-identity-policy", "parent_disjoint"])


def test_final_only_group_filter_restricts_retrieval_and_group_outputs(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    retrieval = {
        "coverage": {
            "n_groups": 4,
            "n_groups_with_neighbors": 3,
            "n_neighbors_total": 6,
            "min_similarity": 0.3,
            "top_k_per_group": 3,
        },
        "experiment": {"mode": "full_mechanism"},
        "groups": [
            {"group_id": "Mechanism.tier_1", "neighbors": [{"rank": 1}]},
            {"group_id": "Mechanism.tier_2", "neighbors": []},
            {"group_id": "Mechanism.tier_3", "neighbors": [{"rank": 1}, {"rank": 2}]},
            {"group_id": "Mechanism.tier_4", "neighbors": [{"rank": 1}]},
        ],
    }
    (run_dir / "retrieval.json").write_text(
        json.dumps(retrieval),
        encoding="utf-8",
    )
    (run_dir / "group_reasoning_outputs.jsonl").write_text(
        "\n".join(
            json.dumps({"group_id": group_id, "status": "ok"})
            for group_id in ("Mechanism.tier_1", "Mechanism.tier_3", "Mechanism.tier_4")
        )
        + "\n",
        encoding="utf-8",
    )

    audit = _filter_final_only_run_artifacts(
        run_dir,
        ["Mechanism.tier_1", "Mechanism.tier_2"],
    )

    filtered_retrieval = json.loads((run_dir / "retrieval.json").read_text(encoding="utf-8"))
    filtered_outputs = [
        json.loads(line)
        for line in (run_dir / "group_reasoning_outputs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [group["group_id"] for group in filtered_retrieval["groups"]] == [
        "Mechanism.tier_1",
        "Mechanism.tier_2",
    ]
    assert filtered_retrieval["coverage"] == {
        "n_groups": 2,
        "n_groups_with_neighbors": 1,
        "n_neighbors_total": 1,
        "min_similarity": 0.3,
        "top_k_per_group": 3,
    }
    assert [output["group_id"] for output in filtered_outputs] == ["Mechanism.tier_1"]
    assert audit["n_source_group_outputs"] == 3
    assert audit["n_retained_group_outputs"] == 1
