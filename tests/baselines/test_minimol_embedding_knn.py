import json

import pytest
import torch

from baselines.minimol.run_embedding_knn import main


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_cache(path, rows, embeddings):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "smiles": [row["drug"] for row in rows],
            "labels": [row["Y"] for row in rows],
            "embeddings": torch.tensor(embeddings, dtype=torch.float32),
        },
        path,
    )


def test_runner_uses_cosine_similarity_and_three_neighbor_vote(tmp_path):
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "embeddings"
    output_dir = tmp_path / "output"
    train = [
        {"drug": "train-a", "Y": 1},
        {"drug": "train-b", "Y": 1},
        {"drug": "train-c", "Y": 0},
        {"drug": "train-d", "Y": 0},
    ]
    test = [
        {"drug": "test-positive", "Y": 1},
        {"drug": "test-negative", "Y": 0},
    ]
    _write_jsonl(data_dir / "train.jsonl", train)
    _write_jsonl(data_dir / "test.jsonl", test)
    _write_cache(
        cache_dir / "train.pt",
        train,
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0]],
    )
    _write_cache(cache_dir / "test.pt", test, [[1.0, 0.0], [0.0, 1.0]])

    assert main(
        [
            "--data-dir",
            str(data_dir),
            "--embedding-cache-dir",
            str(cache_dir),
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
    assert metrics["method"] == "minimol_embedding_cosine_knn_k3"
    assert metrics["embedding"]["similarity"] == "cosine"
    assert metrics["n_train"] == 4
    assert metrics["n_test"] == 2
    assert [row["prediction"] for row in predictions] == [1, 0]
    assert all(len(row["neighbors"]) == 3 for row in predictions)
    assert predictions[0]["neighbors"][0]["train_index"] == 0
    assert predictions[1]["neighbors"][0]["train_index"] == 3


def test_runner_can_evaluate_valid_split(tmp_path):
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "embeddings"
    output_dir = tmp_path / "output"
    train = [
        {"drug": "train-a", "Y": 1},
        {"drug": "train-b", "Y": 1},
        {"drug": "train-c", "Y": 0},
    ]
    valid = [{"drug": "valid-a", "Y": 1}]
    _write_jsonl(data_dir / "train.jsonl", train)
    _write_jsonl(data_dir / "valid.jsonl", valid)
    _write_cache(
        cache_dir / "train.pt",
        train,
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
    )
    _write_cache(cache_dir / "valid.pt", valid, [[1.0, 0.0]])

    assert main(
        [
            "--data-dir",
            str(data_dir),
            "--embedding-cache-dir",
            str(cache_dir),
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


def test_runner_rejects_cache_that_does_not_match_split(tmp_path):
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "embeddings"
    train = [
        {"drug": "train-a", "Y": 1},
        {"drug": "train-b", "Y": 0},
        {"drug": "train-c", "Y": 0},
    ]
    test = [{"drug": "test-a", "Y": 1}]
    _write_jsonl(data_dir / "train.jsonl", train)
    _write_jsonl(data_dir / "test.jsonl", test)
    _write_cache(
        cache_dir / "train.pt",
        [train[1], train[0], train[2]],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )
    _write_cache(cache_dir / "test.pt", test, [[1.0, 0.0]])

    with pytest.raises(ValueError, match="SMILES do not exactly match"):
        main(
            [
                "--data-dir",
                str(data_dir),
                "--embedding-cache-dir",
                str(cache_dir),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
