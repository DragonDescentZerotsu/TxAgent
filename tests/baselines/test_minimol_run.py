import torch
import json
import pytest

from pathlib import Path
from types import SimpleNamespace

from torch.utils.data import DataLoader

from baselines.minimol.run_bioavailability_ma import (
    EmbeddingDataset,
    _json_safe,
    _load_reusable_embeddings,
    make_model,
    train_one_epoch,
)
from baselines.minimol.run_train_cv import (
    calibrate_oof_threshold,
    make_scaffold_folds,
    select_epoch,
    scaffold_groups,
    training_scaffold_groups,
)


def test_frozen_training_scaffold_groups_preserve_dataset_and_empty_groups(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [
        {"drug": "CCO", "Y": 0, "bemis_murcko_scaffold": ""},
        {"drug": "CCC", "Y": 1, "bemis_murcko_scaffold": ""},
        {"drug": "c1ccccc1", "Y": 0, "bemis_murcko_scaffold": "conservative_group"},
        {"drug": "C1CCCCC1", "Y": 1, "bemis_murcko_scaffold": "conservative_group"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    smiles = [row["drug"] for row in rows]
    labels = [row["Y"] for row in rows]
    groups = training_scaffold_groups(path, smiles, labels, field="bemis_murcko_scaffold")
    assert groups == ["", "", "conservative_group", "conservative_group"]
    for train, valid in make_scaffold_folds(labels, groups, n_folds=2, seed=1):
        assert not {groups[i] for i in train} & {groups[i] for i in valid}
    assert training_scaffold_groups(path, smiles, labels) == scaffold_groups(smiles)
    with pytest.raises(ValueError, match="Missing or invalid"):
        training_scaffold_groups(path, smiles, labels, field="absent")
    with pytest.raises(ValueError, match="identity mismatch"):
        training_scaffold_groups(path, list(reversed(smiles)), labels, field="bemis_murcko_scaffold")


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


def test_validation_selection_saves_chosen_checkpoint_and_valid_predictions(tmp_path, monkeypatch):
    import sys
    import numpy as np
    from baselines.minimol import run_bioavailability_ma as runner

    data, out = tmp_path / 'data', tmp_path / 'out'
    data.mkdir()
    for split, labels in [('train', [0, 1, 0, 1]), ('valid', [0, 0, 1, 1]), ('test', [1, 1, 0, 0])]:
        (data / f'{split}.jsonl').write_text(''.join(
            json.dumps({'drug': f'{split}_{i}', 'Y': y}) + '\n' for i, y in enumerate(labels)))
    monkeypatch.setattr(sys, 'argv', ['run', '--data-dir', str(data), '--output-dir', str(out),
        '--device', 'cpu', '--epochs', '3', '--ensemble-size', '2', '--hidden-dim', '4',
        '--threshold-strategy', 'valid_macro_f1'])
    monkeypatch.setattr(runner, 'featurize_splits', lambda splits, args: {
        name: torch.ones(len(split.labels), 512) for name, split in splits.items()})

    def train(model, *args):
        model.test_epoch = getattr(model, 'test_epoch', 0) + 1
    monkeypatch.setattr(runner, 'train_one_epoch', train)
    monkeypatch.setattr(runner, 'evaluate_loss', lambda model, *args: [0.6, 0.2, 0.4][model.test_epoch - 1])

    def predict(model, *args):
        assert model.test_epoch == 2  # The saved best checkpoint, not the last epoch.
        return np.array([0.6, 0.7, 0.8, 0.9])
    monkeypatch.setattr(runner, 'predict_proba', predict)
    runner.main()
    metrics = json.loads((out / 'metrics.json').read_text())
    assert [m['best_epoch'] for m in metrics['members']] == [2, 2]
    assert len(metrics['validation_history']) == 6
    assert metrics['valid_metrics']['macro_f1'] == 1.0
    assert metrics['test_metrics']['macro_f1'] == 0.0
    assert metrics['model_selection']['decision_threshold_source'] == 'valid_macro_f1'
    for split in ('valid', 'test'):
        rows = [json.loads(line) for line in (out / f'{split}_predictions.jsonl').read_text().splitlines()]
        assert [r['prediction'] for r in rows] == [0, 0, 1, 1]
        assert [r['drug'] for r in rows] == [f'{split}_{i}' for i in range(4)]
