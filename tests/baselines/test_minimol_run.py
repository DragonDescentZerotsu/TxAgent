import torch

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
