import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from predict.baselines.minimol.run_bioavailability_ma import (
    EmbeddingDataset,
    _json_safe,
    _load_reusable_embeddings,
    make_model,
    train_one_epoch,
    SplitData,
    featurize_splits,
)
from predict.baselines.minimol.run_train_cv import (
    calibrate_oof_threshold,
    make_scaffold_folds,
    select_epoch,
)


def test_reusable_embeddings_are_indexed_by_exact_smiles(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    torch.save(
        {
            "smiles": ["CCO", "CCC"],
            "labels": [0, 1],
            "embeddings": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        },
        cache_dir / "train.pt",
    )

    registry, files = _load_reusable_embeddings([cache_dir])

    assert files == [cache_dir / "train.pt"]
    assert set(registry) == {"CCO", "CCC"}
    assert torch.equal(registry["CCO"], torch.tensor([1.0, 0.0]))


@pytest.mark.parametrize("conflict", [[0.0, 1.0], [1.0]])
def test_reusable_embeddings_reject_conflicting_vectors(tmp_path, conflict):
    for name, vector in (("train", [1.0, 0.0]), ("test", conflict)):
        torch.save({"smiles": ["CC"], "embeddings": torch.tensor([vector])}, tmp_path / f"{name}.pt")
    with pytest.raises(ValueError, match="Conflicting cached MiniMol embedding"):
        _load_reusable_embeddings([tmp_path])


@pytest.mark.parametrize("embeddings", [torch.ones(2), torch.tensor([[float('nan')], [1.0]])])
def test_split_cache_rejects_invalid_tensor_before_featurizing(tmp_path, embeddings):
    cache = tmp_path / "cache"
    cache.mkdir()
    torch.save({"smiles": ["CC", "CCC"], "labels": [0, 1], "embeddings": embeddings}, cache / "train.pt")
    args = SimpleNamespace(embedding_cache_dir=cache, output_dir=tmp_path, force_embed=False)
    with pytest.raises(ValueError, match="Invalid MiniMol embedding cache"):
        featurize_splits({"train": SplitData(["CC", "CCC"], [0, 1])}, args)


def test_cache_reuse_refreshes_labels_without_featurizing(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    tensor = torch.eye(2)
    torch.save({"smiles": ["CC", "CCC"], "labels": [0, 0], "embeddings": tensor}, cache / "train.pt")
    args = SimpleNamespace(embedding_cache_dir=cache, output_dir=tmp_path, force_embed=False,
                           reuse_embedding_cache_dir=[cache])
    import predict.baselines.minimol.run_bioavailability_ma as runner
    monkeypatch.setattr(runner, "create_featurizer", lambda **kwargs: pytest.fail("Cached molecules must be reused"))
    embeddings = featurize_splits({"train": SplitData(["CC", "CCC"], [0, 1])}, args)
    assert torch.equal(embeddings["train"], tensor)
    assert torch.load(cache / "train.pt", weights_only=False)["labels"] == [0, 1]


@pytest.mark.parametrize("embeddings_only", [False, True])
def test_unseen_condition_fails_before_head_embedding_but_allows_knn_cache_preparation(tmp_path, monkeypatch, embeddings_only):
    import predict.baselines.minimol.run_bioavailability_ma as runner
    for name, condition in (("train", "fed"), ("test", "unseen")):
        (tmp_path / f"{name}.jsonl").write_text(json.dumps({"drug": "CC", "Y": 0, "condition_group": condition}) + "\n")
    argv = ["runner", "--data-dir", str(tmp_path), "--output-dir", str(tmp_path / "output"),
            "--train-all", "--device", "cpu", "--condition-field", "condition_group"]
    monkeypatch.setattr(sys, "argv", argv + (["--embeddings-only"] if embeddings_only else []))
    def embeddings(splits, args):
        assert embeddings_only, "Head categories must be checked before embedding"
        return {name: torch.ones(len(split.smiles), 2) for name, split in splits.items()}
    monkeypatch.setattr(runner, "featurize_splits", embeddings)
    if embeddings_only:
        runner.main()
        assert (tmp_path / "output/embeddings_only_manifest.json").exists()
    else:
        with pytest.raises(ValueError, match="absent from train"):
            runner.main()


def test_conditioned_recipe_runs_shared_methods_and_reloads_fixed_epoch_head(tmp_path):
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    data.mkdir()
    cache.mkdir()
    null = "no_reported_external_condition"
    train = [
        {"drug": drug, "Y": index % 2, "condition_group": null if index % 2 else "fed",
         "condition_scope": "none_reported" if index % 2 else "external",
         "molecule_identity_key": f"p{index}", "benchmark_row_id": f"t{index}"}
        for index, drug in enumerate(["CC", "CCC", "CCO", "CCN"])
    ]
    evaluation = [
        {"drug": drug, "Y": index, "condition_group": group, "condition_scope": scope,
         "molecule_identity_key": f"q{index}", "benchmark_row_id": f"q{index}"}
        for index, (drug, group, scope) in enumerate([("CCCC", "fed", "external"), ("CO", null, "none_reported")])
    ]
    generator = torch.Generator().manual_seed(11)
    tensors = {"train": torch.randn(4, 8, generator=generator), "test": torch.randn(2, 8, generator=generator)}
    for name, rows in (("train", train), ("test", evaluation)):
        (data / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        torch.save({"smiles": [row["drug"] for row in rows], "labels": [row["Y"] for row in rows], "embeddings": tensors[name]}, cache / f"{name}.pt")
    env = dict(os.environ, DATA_DIR=str(data), OUTPUT_ROOT=str(output), PYTHON_BIN=sys.executable,
               REUSE_EMBEDDING_CACHE_DIR=str(cache), DEVICE="cpu", K="2", CUDA_VISIBLE_DEVICES="",
               METHODS="morgan_knn minimol_knn minimol_head", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = ["bash", "experiments/baselines/conditioned.sh", "--epochs", "2", "--ensemble-size", "2",
               "--hidden-dim", "4", "--dropout", "0", "--train-batch-size", "2", "--seed", "7"]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    for method in ("morgan_knn", "minimol_knn", "minimol_head"):
        predictions = [json.loads(line) for line in (output / method / "test_predictions.jsonl").read_text().splitlines()]
        assert [row["benchmark_row_id"] for row in predictions] == ["q0", "q1"]
        assert [row["condition_scope"] for row in predictions] == ["external", "none_reported"]
        metrics = json.loads((output / method / "metrics.json").read_text())
        assert metrics["condition_metrics"]["overall"]["n_total"] == 2
        assert metrics["condition_metrics"]["overall"]["evaluation_coverage"] == 1.0
    metrics = json.loads((output / "minimol_head/metrics.json").read_text())
    assert metrics["valid_metrics"] is None
    assert [member["best_epoch"] for member in metrics["members"]] == [2, 2]
    checkpoint = torch.load(output / "minimol_head/best_ensemble.pt", weights_only=False)
    from predict.baselines.minimol.condition_features import append_condition_one_hot
    from predict.baselines.minimol.head_runtime import TaskHead, predict_scores
    features = append_condition_one_hot(tensors["test"], [row["condition_group"] for row in evaluation], checkpoint["condition_features"]["vocabulary"])
    loader = DataLoader(EmbeddingDataset(features, [row["Y"] for row in evaluation]), batch_size=2)
    scores = []
    for state in checkpoint["model_state_dicts"]:
        model = TaskHead(input_dim=10, hidden_dim=4, depth=3, dropout=0)
        model.load_state_dict(state)
        scores.append(predict_scores(model, loader, torch.device("cpu")))
    observed = [json.loads(line)["score"] for line in (output / "minimol_head/test_predictions.jsonl").read_text().splitlines()]
    assert observed == pytest.approx(sum(scores) / len(scores), abs=1e-7)
    second = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    assert second.returncode != 0
    assert "fresh run directory" in second.stderr


def test_json_safe_serializes_repeated_cache_paths():
    assert _json_safe([Path("first"), Path("second")]) == ["first", "second"]


def test_scaffold_folds_are_disjoint_deterministic_and_cover_all_rows():
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]

    first = make_scaffold_folds(labels, groups, n_folds=2, seed=3)
    second = make_scaffold_folds(labels, groups, n_folds=2, seed=3)

    assert [valid.tolist() for _, valid in first] == [valid.tolist() for _, valid in second]
    assert sorted(index for _, valid in first for index in valid.tolist()) == list(range(8))
    for train, valid in first:
        assert not ({groups[index] for index in train} & {groups[index] for index in valid})


def test_select_epoch_uses_valid_auroc_then_f1_then_earlier_epoch():
    rows = []
    for fold in range(2):
        for epoch, auroc, macro_f1 in ((1, 0.7, 0.6), (2, 0.8, 0.5), (3, 0.8, 0.5)):
            metrics = {"bce": 0.4, "auroc": auroc, "macro_f1": macro_f1, "accuracy": 0.7}
            rows.append({"fold": fold, "epoch": epoch, "train": metrics, "valid": metrics})

    selected, summary = select_epoch(rows)

    assert selected == 2
    assert len(summary) == 3


def test_oof_threshold_calibration_uses_all_train_rows_and_improves_macro_f1():
    labels = [0, 0, 1, 1]
    scores = torch.tensor([0.60, 0.70, 0.80, 0.90]).numpy()

    calibration = calibrate_oof_threshold(labels, scores)

    assert calibration["strategy"] == "pooled_outer_train_scaffold_oof_macro_f1"
    assert calibration["threshold"] > 0.5
    assert (
        calibration["calibrated_metrics"]["macro_f1"]
        > calibration["fixed_0.5_metrics"]["macro_f1"]
    )


def test_learning_rate_warms_up_and_steps_after_optimizer_update():
    args = SimpleNamespace(
        hidden_dim=4,
        depth=3,
        dropout=0.0,
        lr=1e-3,
        weight_decay=0.0,
        warmup=5,
        epochs=10,
    )
    device = torch.device("cpu")
    model, optimizer, scheduler, loss_fn = make_model(args, device)
    loader = DataLoader(
        EmbeddingDataset(torch.ones(4, 512), [0, 1, 0, 1]),
        batch_size=4,
    )

    assert optimizer.param_groups[0]["lr"] == 0.0002
    train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device)
    assert optimizer.param_groups[0]["lr"] == 0.0004


def test_zero_warmup_starts_at_configured_learning_rate():
    args = SimpleNamespace(
        hidden_dim=4,
        depth=3,
        dropout=0.0,
        lr=1e-3,
        weight_decay=0.0,
        warmup=0,
        epochs=10,
    )

    _, optimizer, _, _ = make_model(args, torch.device("cpu"))

    assert optimizer.param_groups[0]["lr"] == 0.001
