import json

from tools.chembl_tool.paper_experiments.summarize_parent_disjoint_results import (
    summarize_condition,
)


def _write_batch(path, predictions, *, macro_f1, accuracy, confusion_matrix):
    path.mkdir()
    (path / "runs").mkdir()
    (path / "manifest.json").write_text(json.dumps({"min_similarity": 0.3}))
    (path / "metrics.json").write_text(
        json.dumps(
            {
                "macro_f1": macro_f1,
                "accuracy": accuracy,
                "confusion_matrix": confusion_matrix,
                "n_failed_runs": 0,
            }
        )
    )
    (path / "predictions.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in predictions)
    )


def test_summarize_condition_reports_corrected_and_broken_flips(tmp_path):
    operational = tmp_path / "operational"
    parent = tmp_path / "parent"
    _write_batch(
        operational,
        [
            {"query_index": 0, "label": 1, "pred_label": 0, "smiles": "CCO"},
            {"query_index": 1, "label": 0, "pred_label": 0, "smiles": "CCN"},
        ],
        macro_f1=0.5,
        accuracy=0.5,
        confusion_matrix={"tn": 1, "fp": 0, "fn": 1, "tp": 0},
    )
    _write_batch(
        parent,
        [
            {"query_index": 0, "label": 1, "pred_label": 1, "smiles": "CCO"},
            {"query_index": 1, "label": 0, "pred_label": 1, "smiles": "CCN"},
        ],
        macro_f1=0.33,
        accuracy=0.5,
        confusion_matrix={"tn": 0, "fp": 1, "fn": 0, "tp": 1},
    )
    experiment = {
        "experiment": "task__chembl_direct",
        "task": "task",
        "source": "chembl",
        "mode": "direct",
        "n_total": 2,
        "n_changed": 2,
        "n_reused": 0,
        "changed_indices": [0, 1],
    }

    condition, flips = summarize_condition(
        experiment, operational_batch=operational, parent_batch=parent
    )

    assert condition["n_corrected"] == 1
    assert condition["n_broken"] == 1
    assert condition["macro_f1_delta"] == -0.17
    assert [row["flip_effect"] for row in flips] == ["corrected", "broken"]


def test_summarize_condition_rejects_flip_for_reused_prompt(tmp_path):
    operational = tmp_path / "operational"
    parent = tmp_path / "parent"
    metrics = {
        "macro_f1": 0.5,
        "accuracy": 0.5,
        "confusion_matrix": {"tn": 0, "fp": 0, "fn": 1, "tp": 0},
    }
    _write_batch(
        operational,
        [{"query_index": 0, "label": 1, "pred_label": 0}],
        **metrics,
    )
    _write_batch(
        parent,
        [{"query_index": 0, "label": 1, "pred_label": 1}],
        **metrics,
    )
    experiment = {
        "experiment": "task__chembl_direct",
        "task": "task",
        "source": "chembl",
        "mode": "direct",
        "n_total": 1,
        "n_changed": 0,
        "n_reused": 1,
        "changed_indices": [],
    }

    try:
        summarize_condition(experiment, operational_batch=operational, parent_batch=parent)
    except ValueError as error:
        assert "identical retrieval input" in str(error)
    else:
        raise AssertionError("Expected unchanged-input flip to be rejected")
