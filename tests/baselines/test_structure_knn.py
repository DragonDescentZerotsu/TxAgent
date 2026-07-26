import json

import pytest

from baselines.structure_knn.run import _metrics, _roc_auc, main


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_roc_auc_handles_tied_scores():
    assert _roc_auc([(0, 0.0), (1, 1.0)]) == 1.0
    assert _roc_auc([(0, 0.5), (1, 0.5)]) == 0.5


def test_metrics_match_expected_confusion():
    rows = [
        {"Y": 0, "prediction": 0, "score": 0.0},
        {"Y": 0, "prediction": 1, "score": 2 / 3},
        {"Y": 1, "prediction": 1, "score": 1.0},
        {"Y": 1, "prediction": 0, "score": 1 / 3},
    ]
    metrics = _metrics(rows)
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_f1"] == 0.5
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_runner_uses_three_train_neighbors(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    _write_jsonl(
        data_dir / "train.jsonl",
        [
            {"drug": "CCO", "Y": 1},
            {"drug": "CCCO", "Y": 1},
            {"drug": "c1ccccc1", "Y": 0},
            {"drug": "ClCCl", "Y": 0},
        ],
    )
    _write_jsonl(
        data_dir / "test.jsonl",
        [
            {"drug": "CCCCO", "Y": 1},
            {"drug": "c1ccccc1Cl", "Y": 0},
        ],
    )

    assert main(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--k",
            "3",
        ]
    ) == 0
    metrics = json.loads((output_dir / "metrics.json").read_text())
    predictions = [
        json.loads(line)
        for line in (output_dir / "test_predictions.jsonl").read_text().splitlines()
    ]
    assert metrics["k"] == 3
    assert metrics["n_train"] == 4
    assert metrics["neighbor_selector"]["name"] == "similarity"
    assert metrics["n_evaluated"] == 2
    assert all(len(row["neighbors"]) == 3 for row in predictions)


def test_runner_supports_query_feature_coverage_selection(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    _write_jsonl(
        data_dir / "train.jsonl",
        [
            {"drug": "CCO", "Y": 1},
            {"drug": "CCCO", "Y": 1},
            {"drug": "COC", "Y": 0},
            {"drug": "c1ccccc1", "Y": 0},
        ],
    )
    _write_jsonl(data_dir / "test.jsonl", [{"drug": "CCCCO", "Y": 1}])

    assert main(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--k",
            "3",
            "--neighbor-selector",
            "query_feature_coverage",
        ]
    ) == 0

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["neighbor_selector"]["name"] == "query_feature_coverage"
    assert metrics["retrieval_diagnostics"]["query_feature_coverage"] > 0


def test_minimum_similarity_reports_unsupported_test_rows(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    _write_jsonl(
        data_dir / "train.jsonl",
        [
            {"drug": "CCO", "Y": 1},
            {"drug": "CCCO", "Y": 1},
            {"drug": "CCCCO", "Y": 1},
            {"drug": "c1ccccc1", "Y": 0},
            {"drug": "c1ccccc1", "Y": 0},
            {"drug": "c1ccccc1", "Y": 0},
        ],
    )
    _write_jsonl(
        data_dir / "test.jsonl",
        [
            {"drug": "CCCO", "Y": 1},
            {"drug": "c1ccccc1", "Y": 0},
            {"drug": "[U]", "Y": 0},
        ],
    )

    assert main(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--k",
            "3",
            "--min-similarity",
            "0.3",
        ]
    ) == 0

    metrics = json.loads((output_dir / "metrics.json").read_text())
    predictions = [
        json.loads(line)
        for line in (output_dir / "test_predictions.jsonl").read_text().splitlines()
    ]
    assert metrics["n_evaluated"] == 2
    assert metrics["n_skipped_insufficient_neighbors"] == 1
    assert predictions[2]["status"] == "insufficient_neighbors"
    assert predictions[2]["prediction"] is None


def test_runner_rejects_k_larger_than_train(tmp_path):
    data_dir = tmp_path / "data"
    _write_jsonl(data_dir / "train.jsonl", [{"drug": "CCO", "Y": 1}])
    _write_jsonl(data_dir / "test.jsonl", [{"drug": "CCC", "Y": 0}])
    with pytest.raises(ValueError, match="fewer than k"):
        main(
            [
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(tmp_path / "output"),
                "--k",
                "3",
            ]
        )
