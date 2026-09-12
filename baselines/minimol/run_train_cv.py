"""Select MiniMol head training length with train-only scaffold cross-validation.

The upstream MiniMol encoder remains frozen.  This runner operates only on the
cached embeddings and labels from ``train.jsonl``; outer ``valid.jsonl`` and
``test.jsonl`` are deliberately never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from baselines.minimol.condition_features import (
    append_condition_one_hot,
    condition_feature_contract,
    condition_vocabulary,
)
from baselines.minimol.run_bioavailability_ma import (
    EmbeddingDataset,
    choose_threshold,
    load_split,
    make_model,
    predict_proba,
    set_seed,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3, choices=(3, 4))
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--scaffold-group-field",
        default=None,
        help="Optional frozen grouping field in outer train.jsonl; otherwise recompute Murcko scaffolds.",
    )
    parser.add_argument(
        "--condition-field",
        default=None,
        help=(
            "Optional categorical JSONL field appended to MiniMol embeddings as a "
            "train-derived one-hot feature."
        ),
    )
    return parser.parse_args()


def scaffold_groups(smiles: list[str]) -> list[str]:
    groups = []
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid SMILES at train row {index}: {value!r}")
        groups.append(
            MurckoScaffold.MurckoScaffoldSmiles(
                mol=molecule,
                includeChirality=False,
            )
        )
    return groups


def training_scaffold_groups(
    path: Path, smiles: list[str], labels: list[int], *, field: str | None = None
) -> list[str]:
    """Use explicit dataset groups without reading outer validation or test."""
    if field is None:
        return scaffold_groups(smiles)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != len(smiles) or len(labels) != len(smiles):
        raise ValueError("Frozen scaffold groups do not align with training rows")
    groups = []
    for index, (row, value, label) in enumerate(zip(rows, smiles, labels, strict=True)):
        if row.get("drug") != value or row.get("Y") != label:
            raise ValueError(f"Frozen scaffold group identity mismatch at train row {index}")
        if field not in row or not isinstance(row[field], str):
            raise ValueError(f"Missing or invalid scaffold group field {field!r} at train row {index}")
        # Empty scaffolds are a valid shared acyclic group, not missing data.
        groups.append(row[field])
    return groups


def make_scaffold_folds(
    labels: list[int],
    groups: list[str],
    *,
    n_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_folds < 2:
        raise ValueError("--folds must be at least 2")
    labels_array = np.asarray(labels, dtype=np.int64)
    groups_array = np.asarray(groups, dtype=object)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(labels_array)), labels_array, groups_array))
    validation_indices = np.concatenate([valid for _, valid in folds])
    if sorted(validation_indices.tolist()) != list(range(len(labels))):
        raise AssertionError("Cross-validation folds do not cover each training row exactly once")
    for train_indices, valid_indices in folds:
        if set(groups_array[train_indices]) & set(groups_array[valid_indices]):
            raise AssertionError("Scaffold leakage detected inside a cross-validation fold")
        if len(set(labels_array[valid_indices].tolist())) != 2:
            raise ValueError("Every inner validation fold must contain both binary labels")
    return folds


def load_embedding_cache(path: Path, smiles: list[str], labels: list[int]) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if list(map(str, payload.get("smiles", []))) != smiles:
        raise ValueError(f"Embedding cache SMILES do not exactly match train.jsonl: {path}")
    if [int(value) for value in payload.get("labels", [])] != labels:
        raise ValueError(f"Embedding cache labels do not exactly match train.jsonl: {path}")
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError(f"Embedding cache must contain a rank-2 tensor: {path}")
    if embeddings.shape[0] != len(labels) or not torch.isfinite(embeddings).all():
        raise ValueError(f"Embedding cache has invalid rows: {path}")
    return embeddings.detach().cpu().float()


def binary_metrics(
    labels: list[int], scores: np.ndarray, *, threshold: float = 0.5
) -> dict[str, float]:
    label_array = np.asarray(labels, dtype=np.int64)
    predictions = (scores >= threshold).astype(np.int64)
    score_tensor = torch.from_numpy(scores.astype(np.float64))
    label_tensor = torch.from_numpy(label_array.astype(np.float64))
    loss = F.binary_cross_entropy(score_tensor.clamp(1e-12, 1 - 1e-12), label_tensor)
    return {
        "bce": float(loss.item()),
        "auroc": float(roc_auc_score(label_array, scores)),
        "macro_f1": float(f1_score(label_array, predictions, average="macro")),
        "accuracy": float(accuracy_score(label_array, predictions)),
    }


def calibrate_oof_threshold(labels: list[int], scores: np.ndarray) -> dict[str, Any]:
    """Choose a macro-F1 threshold from complete outer-train OOF predictions."""
    if scores.shape != (len(labels),) or not np.isfinite(scores).all():
        raise ValueError("OOF scores must contain one finite value per training row")
    threshold, _ = choose_threshold(labels, scores, "valid_macro_f1")
    return {
        "strategy": "pooled_outer_train_scaffold_oof_macro_f1",
        "threshold": threshold,
        "fixed_0.5_metrics": binary_metrics(labels, scores),
        "calibrated_metrics": binary_metrics(labels, scores, threshold=threshold),
    }


def select_epoch(rows: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    epochs = sorted({int(row["epoch"]) for row in rows})
    summary = []
    for epoch in epochs:
        current = [row for row in rows if int(row["epoch"]) == epoch]
        entry: dict[str, Any] = {"epoch": epoch, "n_folds": len(current)}
        for split in ("train", "valid"):
            for metric in ("bce", "auroc", "macro_f1", "accuracy"):
                values = np.asarray([row[split][metric] for row in current], dtype=float)
                entry[f"mean_{split}_{metric}"] = float(values.mean())
                entry[f"std_{split}_{metric}"] = float(values.std(ddof=0))
        summary.append(entry)
    best = max(
        summary,
        key=lambda row: (
            row["mean_valid_auroc"],
            row["mean_valid_macro_f1"],
            -row["epoch"],
        ),
    )
    return int(best["epoch"]), summary


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.train_batch_size < 2:
        raise ValueError("--train-batch-size must be at least 2 for BatchNorm training")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.data_dir / "train.jsonl"
    cache_path = args.embedding_cache_dir / "train.pt"
    train = load_split(train_path, condition_field=args.condition_field)
    molecule_embeddings = load_embedding_cache(cache_path, train.smiles, train.labels)
    vocabulary = (
        condition_vocabulary(train.conditions)
        if train.conditions is not None
        else None
    )
    embeddings = append_condition_one_hot(
        molecule_embeddings, train.conditions, vocabulary
    )
    feature_contract = condition_feature_contract(
        field=args.condition_field,
        vocabulary=vocabulary,
        molecule_embedding_dim=int(molecule_embeddings.shape[1]),
    )
    groups = training_scaffold_groups(
        train_path, train.smiles, train.labels, field=args.scaffold_group_field
    )
    folds = make_scaffold_folds(train.labels, groups, n_folds=args.folds, seed=args.seed)
    device = torch.device(args.device)

    fold_manifest = []
    history = []
    labels_array = np.asarray(train.labels, dtype=np.int64)
    groups_array = np.asarray(groups, dtype=object)
    oof_scores_by_epoch = {
        epoch: np.full(len(train.labels), np.nan, dtype=np.float64)
        for epoch in range(1, args.epochs + 1)
    }
    oof_fold = np.full(len(train.labels), -1, dtype=np.int64)
    for fold_index, (train_indices, valid_indices) in enumerate(folds):
        fold_seed = args.seed + fold_index
        set_seed(fold_seed)
        generator = torch.Generator().manual_seed(fold_seed)
        inner_train_embeddings = embeddings[train_indices]
        inner_valid_embeddings = embeddings[valid_indices]
        inner_train_labels = labels_array[train_indices].tolist()
        inner_valid_labels = labels_array[valid_indices].tolist()
        if len(inner_train_labels) < 2:
            raise ValueError(
                f"fold {fold_index} has fewer than two training rows for BatchNorm"
            )
        # The frozen MiniMol head contains BatchNorm layers.  A shuffled final
        # batch of exactly one row is not a valid training input, so drop only
        # that otherwise-unavoidable singleton.  All other fold shapes retain
        # the historical complete-row loader contract.
        drop_singleton_batch = (
            len(inner_train_labels) % args.train_batch_size == 1
        )
        train_loader = DataLoader(
            EmbeddingDataset(inner_train_embeddings, inner_train_labels),
            batch_size=args.train_batch_size,
            shuffle=True,
            generator=generator,
            drop_last=drop_singleton_batch,
        )
        train_eval_loader = DataLoader(
            EmbeddingDataset(inner_train_embeddings, inner_train_labels),
            batch_size=args.eval_batch_size,
            shuffle=False,
        )
        valid_loader = DataLoader(
            EmbeddingDataset(inner_valid_embeddings, inner_valid_labels),
            batch_size=args.eval_batch_size,
            shuffle=False,
        )
        model, optimizer, scheduler, loss_fn = make_model(
            args, device, input_dim=embeddings.shape[1]
        )
        fold_manifest.append(
            {
                "fold": fold_index,
                "seed": fold_seed,
                "n_train": len(train_indices),
                "n_valid": len(valid_indices),
                "train_labels": dict(sorted(Counter(inner_train_labels).items())),
                "valid_labels": dict(sorted(Counter(inner_valid_labels).items())),
                "n_train_scaffolds": len(set(groups_array[train_indices])),
                "n_valid_scaffolds": len(set(groups_array[valid_indices])),
                "scaffold_overlap": 0,
                "drop_singleton_train_batch": drop_singleton_batch,
            }
        )
        for epoch_index in range(args.epochs):
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                loss_fn,
                device,
            )
            train_scores = predict_proba(model, train_eval_loader, device)
            valid_scores = predict_proba(model, valid_loader, device)
            row = {
                "fold": fold_index,
                "seed": fold_seed,
                "epoch": epoch_index + 1,
                "train": binary_metrics(inner_train_labels, train_scores),
                "valid": binary_metrics(inner_valid_labels, valid_scores),
            }
            history.append(row)
            oof_scores_by_epoch[epoch_index + 1][valid_indices] = valid_scores
            oof_fold[valid_indices] = fold_index
            print(
                f"[minimol-cv] fold={fold_index + 1}/{args.folds} "
                f"epoch={epoch_index + 1}/{args.epochs} "
                f"train_auc={row['train']['auroc']:.4f} "
                f"valid_auc={row['valid']['auroc']:.4f}",
                flush=True,
            )

    selected_epoch, epoch_summary = select_epoch(history)
    selected_oof_scores = oof_scores_by_epoch[selected_epoch]
    threshold_calibration = calibrate_oof_threshold(
        train.labels, selected_oof_scores
    )
    history_path = args.output_dir / "fold_epoch_metrics.jsonl"
    with history_path.open("w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row) + "\n")
    _write_json(args.output_dir / "epoch_summary.json", epoch_summary)
    oof_path = args.output_dir / "selected_epoch_oof_predictions.jsonl"
    with oof_path.open("w", encoding="utf-8") as handle:
        for index, (label, score, fold_index) in enumerate(
            zip(train.labels, selected_oof_scores, oof_fold, strict=True)
        ):
            row: dict[str, Any] = {
                "train_row_index": index,
                "fold": int(fold_index),
                "Y": label,
                "score": float(score),
                "prediction": int(score >= threshold_calibration["threshold"]),
            }
            if args.condition_field is not None and train.conditions is not None:
                row[args.condition_field] = train.conditions[index]
            handle.write(json.dumps(row) + "\n")
    result = {
        "method": "minimol_head_train_only_scaffold_cv.v2",
        "selection_metric": "mean inner-valid AUROC",
        "selected_epoch": selected_epoch,
        "decision_threshold": threshold_calibration,
        "outer_valid_used_for_selection": False,
        "outer_test_used_for_selection": False,
        "data_dir": str(args.data_dir),
        "train_path": str(train_path),
        "embedding_cache": str(cache_path),
        "n_train": len(train.labels),
        "n_folds": args.folds,
        "cv_members_per_fold": 1,
        "selected_epoch_oof_predictions": str(oof_path),
        "n_scaffolds": len(set(groups)),
        "scaffold_grouping": {
            "source": "outer_train_field" if args.scaffold_group_field else "rdkit_murcko_from_smiles",
            "field": args.scaffold_group_field,
            "ordered_groups_sha256": hashlib.sha256(json.dumps(groups).encode()).hexdigest(),
            "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        },
        "condition_features": feature_contract,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "folds": fold_manifest,
        "selected_epoch_metrics": next(
            row for row in epoch_summary if row["epoch"] == selected_epoch
        ),
    }
    _write_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
