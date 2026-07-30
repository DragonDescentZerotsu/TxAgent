from pathlib import Path

from tools.chembl_tool.common.task_workflows.reasoning_batch import (
    BatchConfig,
    _parse_args,
    _result_is_complete,
    _single_run_command,
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


def test_batch_forwards_strict_morgan_neighbor_selector_to_single_run():
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
        ["--morgan-neighbor-selector", "query_feature_coverage"],
    )

    command = _single_run_command(config, args, 7, "run-7", Path("run-root"))

    selector_position = command.index("--morgan-neighbor-selector")
    assert command[selector_position + 1] == "query_feature_coverage"
    assert "--neighbor-selector" not in command
