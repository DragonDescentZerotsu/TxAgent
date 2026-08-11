import csv
import json
from argparse import Namespace
from types import SimpleNamespace

from tools.chembl_tool.paper_experiments import summarize_starling_benchmark


def test_compose_task_metrics_selects_each_task_from_its_declared_summary(tmp_path):
    specifications = []
    for index, task in enumerate(summarize_starling_benchmark.TASK_DATA_NAMES):
        path = tmp_path / f"{task}.tsv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "benchmark_split",
                    "evaluation_subset",
                    "task",
                    "method",
                    "macro_f1",
                ),
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerow(
                {
                    "benchmark_split": "scaffold",
                    "evaluation_subset": "valid",
                    "task": task,
                    "method": "none",
                    "macro_f1": 0.5 + index / 10,
                }
            )
        specifications.append(f"{task}={path}")

    rows = summarize_starling_benchmark._compose_task_metrics(
        tuple(specifications),
        splits=("scaffold",),
        evaluation_subset="valid",
    )

    assert [row["task"] for row in rows] == list(
        summarize_starling_benchmark.TASK_DATA_NAMES
    )
    assert [float(row["macro_f1"]) for row in rows] == [0.5, 0.6, 0.7]


def test_valid_pipeline_summary_does_not_implicitly_mix_historical_baselines():
    roots = summarize_starling_benchmark._baseline_roots(
        Namespace(
            evaluation_subset="valid",
            pipeline_root="outputs/paper/v4-valid",
            minimol_root="",
            structure_knn_root="",
            minimol_embedding_knn_root="",
        )
    )

    assert roots == {
        "minimol": None,
        "structure_knn": None,
        "minimol_embedding_knn": None,
    }


def test_condition_metric_override_is_explicit_and_versionable(tmp_path):
    metrics = tmp_path / "metrics.json"
    overrides = summarize_starling_benchmark._condition_metric_overrides(
        (f"bioavailability_ma__none={metrics}",)
    )

    assert overrides == {"bioavailability_ma__none": metrics}


def test_composed_summary_adds_explicit_condition_metrics(monkeypatch, tmp_path):
    experiment = SimpleNamespace(
        task="bioavailability_ma",
        name="bioavailability_ma__chembl_direct",
        mode="direct",
        source="chembl",
    )
    monkeypatch.setattr(
        summarize_starling_benchmark,
        "experiments_for_starling_benchmark",
        lambda split: [experiment],
    )
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "n_total": 2,
                "n_successful": 2,
                "n_failed_runs": 0,
                "accuracy": 0.5,
                "macro_f1": 0.4,
                "positive_class_precision": 0.5,
                "positive_class_recall": 0.5,
                "positive_class_f1": 0.5,
                "confusion_matrix": {"tn": 1, "fp": 0, "fn": 1, "tp": 0},
            }
        ),
        encoding="utf-8",
    )

    rows = summarize_starling_benchmark._condition_metric_rows(
        (f"{experiment.name}={metrics}",),
        splits=("scaffold",),
        evaluation_subset="valid",
        model_label="GPT-OSS-120B",
        visibility_mode="identity_blind",
    )

    assert len(rows) == 1
    assert rows[0]["method"] == "chembl_direct"
    assert rows[0]["macro_f1"] == 0.4
    assert rows[0]["metrics_path"] == str(metrics)


def test_minimol_embedding_knn_rows_are_included(monkeypatch, tmp_path):
    root = tmp_path / "minimol_embedding_knn"
    metrics_path = root / "BBB_Martins" / "random" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        json.dumps(
            {
                "method": "minimol_embedding_cosine_knn_k3",
                "n_train": 100,
                "n_test": 20,
                "n_evaluated": 20,
                "accuracy": 0.8,
                "macro_f1": 0.75,
                "auroc": 0.82,
                "positive_precision": 0.8,
                "positive_recall": 0.9,
                "positive_f1": 0.847,
                "confusion_matrix": {"tn": 5, "fp": 2, "fn": 2, "tp": 11},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        summarize_starling_benchmark,
        "MINIMOL_EMBEDDING_KNN_ROOT",
        root,
    )

    rows = summarize_starling_benchmark._minimol_embedding_knn_rows("random")

    assert len(rows) == 1
    assert rows[0]["task"] == "bbb_martins"
    assert rows[0]["method"] == "minimol_embedding_cosine_knn_k3"
    assert rows[0]["method_family"] == "minimol_embedding_knn"
    assert rows[0]["macro_f1"] == 0.75


def test_failed_pipeline_prediction_is_counted_as_incorrect():
    confusion = summarize_starling_benchmark._failure_inclusive_confusion(
        [
            {"label": 1, "pred_label": 1, "status": "ok"},
            {"label": 1, "pred_label": 1, "status": "error"},
            {"label": 0, "pred_label": 0, "status": "ok"},
        ]
    )
    metrics = summarize_starling_benchmark._metrics_from_confusion(confusion)

    assert confusion == {"tn": 1, "fp": 0, "fn": 1, "tp": 1}
    assert metrics["accuracy"] == 2 / 3


def test_failed_pipeline_prediction_does_not_require_pred_label():
    confusion = summarize_starling_benchmark._failure_inclusive_confusion(
        [{"label": 0, "pred_label": None, "status": "error"}]
    )

    assert confusion == {"tn": 0, "fp": 1, "fn": 0, "tp": 0}


def test_pipeline_rows_can_read_fresh_deployment_visible_parent_disjoint(
    monkeypatch, tmp_path
):
    experiment = SimpleNamespace(
        task="bbb_martins",
        name="bbb_martins__chembl_direct",
        mode="direct",
        source="chembl",
    )
    monkeypatch.setattr(
        summarize_starling_benchmark,
        "experiments_for_starling_benchmark",
        lambda split: [experiment],
    )
    metrics_path = (
        tmp_path
        / "runs_deployment_visible_parent_disjoint"
        / experiment.task
        / experiment.name
        / "metrics.json"
    )
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        json.dumps(
            {
                "n_total": 2,
                "n_successful": 2,
                "n_failed_runs": 0,
                "accuracy": 0.5,
                "macro_f1": 0.5,
                "positive_class_precision": 0.5,
                "positive_class_recall": 0.5,
                "positive_class_f1": 0.5,
                "confusion_matrix": {"tn": 1, "fp": 0, "fn": 1, "tp": 0},
            }
        ),
        encoding="utf-8",
    )

    rows = summarize_starling_benchmark._pipeline_rows(
        "scaffold",
        pipeline_root=tmp_path,
        evaluation_subset="valid",
        model_label="GPT-OSS-20B visible",
        visibility_mode="deployment_visible",
    )

    assert len(rows) == 1
    assert rows[0]["visibility_mode"] == "deployment_visible"
    assert rows[0]["neighbor_identity_policy"] == "parent_disjoint"
