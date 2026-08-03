import json

from tools.chembl_tool.paper_experiments import summarize_starling_benchmark


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
