import json

import pytest

from baselines.conditioned_knn import select_conditioned_ranked_indices
from baselines.structure_knn.run import (
    _metrics,
    _roc_auc,
    main,
)


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


def test_conditioned_selection_prefers_same_group_then_distinct_null_molecules():
    reference = [
        {"drug": "A", "Y": 0, "molecule_identity_key": "A", "condition_group": "no_reported_external_condition"},
        {"drug": "A", "Y": 1, "molecule_identity_key": "A", "condition_group": "disease=x"},
        {"drug": "B", "Y": 1, "molecule_identity_key": "B", "condition_group": "disease=x"},
        {"drug": "C", "Y": 0, "molecule_identity_key": "C", "condition_group": "no_reported_external_condition"},
        {"drug": "D", "Y": 1, "molecule_identity_key": "D", "condition_group": "disease=y"},
    ]
    selected = select_conditioned_ranked_indices(
        [1, 0, 2, 4, 3],
        reference,
        {"condition_group": "disease=x"},
        k=3,
        policy="same_condition_then_null",
    )
    assert selected == [(1, "same_condition"), (2, "same_condition"), (3, "null_fallback")]


def test_condition_agnostic_selection_deduplicates_and_prefers_null_label():
    reference = [
        {"drug": "A", "Y": 0, "molecule_identity_key": "A", "condition_group": "no_reported_external_condition"},
        {"drug": "A", "Y": 1, "molecule_identity_key": "A", "condition_group": "disease=x"},
        {"drug": "B", "Y": 1, "molecule_identity_key": "B", "condition_group": "disease=x"},
        {"drug": "C", "Y": 0, "molecule_identity_key": "C", "condition_group": "no_reported_external_condition"},
    ]
    selected = select_conditioned_ranked_indices(
        [1, 0, 2, 3],
        reference,
        {"condition_group": "disease=x"},
        k=3,
        policy="all_train_unique_molecules",
    )
    assert selected == [
        (0, "all_train_unique_molecules"),
        (2, "all_train_unique_molecules"),
        (3, "all_train_unique_molecules"),
    ]


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


def test_runner_can_evaluate_valid_split(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    _write_jsonl(
        data_dir / "train.jsonl",
        [
            {"drug": "CCO", "Y": 1},
            {"drug": "CCCO", "Y": 1},
            {"drug": "c1ccccc1", "Y": 0},
        ],
    )
    _write_jsonl(data_dir / "valid.jsonl", [{"drug": "CCCCO", "Y": 1}])

    assert main(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--evaluation-split",
            "valid",
        ]
    ) == 0

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["evaluation_split"] == "valid"
    assert metrics["n_evaluation"] == 1
    assert (output_dir / "valid_predictions.jsonl").exists()
    assert not (output_dir / "test_predictions.jsonl").exists()


def test_runner_can_use_train_and_valid_as_test_reference_pool(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    train = [
        {"drug": "CCO", "Y": 0},
        {"drug": "c1ccccc1", "Y": 0},
        {"drug": "ClCCl", "Y": 0},
    ]
    valid = [{"drug": "CCCCO", "Y": 1}]
    _write_jsonl(data_dir / "train.jsonl", train)
    _write_jsonl(data_dir / "valid.jsonl", valid)
    _write_jsonl(data_dir / "test.jsonl", [{"drug": "CCCO", "Y": 1}])

    assert main(
        [
            "--data-dir", str(data_dir),
            "--output-dir", str(output_dir),
            "--reference-splits", "train", "valid",
            "--k", "1",
        ]
    ) == 0
    metrics = json.loads((output_dir / "metrics.json").read_text())
    prediction = json.loads((output_dir / "test_predictions.jsonl").read_text())
    assert metrics["reference_splits"] == ["train", "valid"]
    assert metrics["n_reference_by_split"] == {"train": 3, "valid": 1}
    assert metrics["n_train"] == 3
    assert metrics["n_reference"] == 4
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["train_path"] == str(data_dir / "train.jsonl")
    assert prediction["neighbors"][0]["reference_split"] == "valid"
    assert prediction["neighbors"][0]["reference_pool_index"] == (
        prediction["neighbors"][0]["train_index"]
    )


def test_runner_rejects_train_valid_reference_for_valid_evaluation(tmp_path):
    with pytest.raises(ValueError, match="only for test"):
        main(
            [
                "--data-dir", str(tmp_path),
                "--output-dir", str(tmp_path / "output"),
                "--evaluation-split", "valid",
                "--reference-splits", "train", "valid",
            ]
        )


@pytest.mark.parametrize(
    "reference_splits",
    [["valid"], ["valid", "train"], ["train", "train"]],
)
def test_runner_rejects_noncanonical_reference_pools(tmp_path, reference_splits):
    with pytest.raises(ValueError, match="exactly 'train' or 'train valid'"):
        main(
            [
                "--data-dir", str(tmp_path),
                "--output-dir", str(tmp_path / "output"),
                "--reference-splits", *reference_splits,
            ]
        )


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
