from tools.chembl_tool.common.task_workflows.reasoning_batch import _result_is_complete


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
