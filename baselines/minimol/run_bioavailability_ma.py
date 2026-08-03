"""Train and evaluate a MiniMol baseline on Bioavailability_Ma.

This follows the downstream head used by MiniMol's TDC leaderboard script, but
uses the repo's fixed train/valid/test JSONL splits instead of TDC folds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from baselines.minimol.embedding_runtime import (
    DEFAULT_MINIMOL_SOURCE,
    create_featurizer,
    embed_smiles,
)

DEFAULT_DATA_DIR = Path("data/processed/Bioavailability_Ma")
DEFAULT_OUTPUT_DIR = Path("outputs/baselines/minimol/bioavailability_ma")


class TaskHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        input_dim: int = 512,
        dropout: float = 0.1,
        depth: int = 3,
        combine: bool = True,
    ) -> None:
        super().__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, hidden_dim)
        self.final_dense = nn.Linear(input_dim + hidden_dim, 1) if combine else nn.Linear(hidden_dim, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.combine = combine
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_x = x

        x = self.dense1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.dense2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        if self.depth == 4:
            x = self.dense3(x)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.dropout(x)

        x = torch.cat((x, original_x), dim=1) if self.combine else x
        return self.final_dense(x)


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, labels: Iterable[int]) -> None:
        self.embeddings = embeddings.float()
        self.labels = torch.tensor(list(labels), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


@dataclass
class SplitData:
    smiles: list[str]
    labels: list[int]


@dataclass
class RunResult:
    seed: int
    best_epoch: int
    best_valid_loss: float | None
    valid_threshold: float | None
    valid_macro_f1: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-cache-dir", type=Path, default=None)
    parser.add_argument("--minimol-source", type=Path, default=DEFAULT_MINIMOL_SOURCE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3, choices=[3, 4])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold-strategy", choices=["fixed_0.5", "valid_macro_f1"], default="fixed_0.5")
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Train for all configured epochs on train.jsonl, without requiring or selecting on valid.jsonl.",
    )
    parser.add_argument("--force-embed", action="store_true", help="Ignore cached MiniMol embeddings.")
    return parser.parse_args()


def load_split(path: Path) -> SplitData:
    smiles: list[str] = []
    labels: list[int] = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            row = json.loads(line)
            try:
                smiles.append(str(row["drug"]))
                labels.append(int(row["Y"]))
            except KeyError as exc:
                raise ValueError(f"{path}:{line_number} is missing required key {exc!s}") from exc
    return SplitData(smiles=smiles, labels=labels)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cache_path(output_dir: Path, split_name: str) -> Path:
    return output_dir / "embeddings" / f"{split_name}.pt"


def embedding_cache_path(args: argparse.Namespace, split_name: str) -> Path:
    if args.embedding_cache_dir is not None:
        return args.embedding_cache_dir / f"{split_name}.pt"
    return cache_path(args.output_dir, split_name)


def featurize_split(split_name: str, split: SplitData, args: argparse.Namespace) -> torch.Tensor:
    path = embedding_cache_path(args, split_name)
    if path.exists() and not args.force_embed:
        payload = torch.load(path, map_location="cpu")
        if payload["smiles"] == split.smiles:
            return payload["embeddings"].float()
        print(f"[minimol] cache mismatch for {split_name}; recomputing embeddings")

    print(f"[minimol] featurizing {split_name}: {len(split.smiles)} molecules")
    featurizer = create_featurizer(
        batch_size=args.embedding_batch_size,
        minimol_source=args.minimol_source,
    )
    tensor = embed_smiles(featurizer, split.smiles)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"smiles": split.smiles, "labels": split.labels, "embeddings": tensor}, path)
    return tensor


def make_model(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, optim.Optimizer, LambdaLR, nn.Module]:
    model = TaskHead(hidden_dim=args.hidden_dim, depth=args.depth, dropout=args.dropout, combine=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCELoss()

    def lr_fn(epoch: int) -> float:
        if args.warmup > 0 and epoch < args.warmup:
            return epoch / args.warmup
        denom = max(1, args.epochs - args.warmup)
        return (1 + math.cos(math.pi * (epoch - args.warmup) / denom)) / 2

    scheduler = LambdaLR(optimizer, lr_lambda=lr_fn)
    return model, optimizer, scheduler, loss_fn


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: LambdaLR,
    loss_fn: nn.Module,
    epoch: int,
    device: torch.device,
) -> None:
    model.train()
    scheduler.step(epoch)
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs).squeeze(-1)
        loss = loss_fn(torch.sigmoid(logits), targets)
        loss.backward()
        optimizer.step()


def evaluate_loss(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs).squeeze(-1)
            total_loss += float(loss_fn(torch.sigmoid(logits), targets).item())
    return total_loss / max(1, len(loader))


def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, _ in loader:
            logits = model(inputs.to(device)).squeeze(-1)
            predictions.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(predictions)


def choose_threshold(y_true: list[int], y_score: np.ndarray, strategy: str) -> tuple[float, float]:
    if strategy == "fixed_0.5":
        threshold = 0.5
        return threshold, float(f1_score(y_true, y_score >= threshold, average="macro"))

    candidates = sorted(set(float(x) for x in y_score))
    thresholds = [0.5]
    thresholds.extend(candidates)
    thresholds.extend((a + b) / 2 for a, b in zip(candidates, candidates[1:]))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        score = float(f1_score(y_true, y_score >= threshold, average="macro"))
        if score > best_score or (score == best_score and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_threshold = float(threshold)
            best_score = score
    return best_threshold, best_score


def evaluate_metrics(y_true: list[int], y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "threshold": float(threshold),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_cache_dir = args.output_dir / ".matplotlib"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
    device = torch.device(args.device)

    train = load_split(args.data_dir / "train.jsonl")
    valid = None if args.train_all else load_split(args.data_dir / "valid.jsonl")
    test = load_split(args.data_dir / "test.jsonl")

    print(
        "[minimol] loaded splits: "
        f"train={len(train.labels)} "
        f"valid={len(valid.labels) if valid is not None else 'not_used'} "
        f"test={len(test.labels)} device={device}"
    )

    train_embeddings = featurize_split("train", train, args)
    valid_embeddings = featurize_split("valid", valid, args) if valid is not None else None
    test_embeddings = featurize_split("test", test, args)

    valid_loader = (
        DataLoader(
            EmbeddingDataset(valid_embeddings, valid.labels),
            batch_size=args.eval_batch_size,
            shuffle=False,
        )
        if valid is not None and valid_embeddings is not None
        else None
    )
    test_loader = DataLoader(EmbeddingDataset(test_embeddings, test.labels), batch_size=args.eval_batch_size, shuffle=False)

    best_models: list[nn.Module] = []
    run_results: list[RunResult] = []

    for member_idx in range(args.ensemble_size):
        seed = args.seed + member_idx
        set_seed(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        train_loader = DataLoader(
            EmbeddingDataset(train_embeddings, train.labels),
            batch_size=args.train_batch_size,
            shuffle=True,
            generator=generator,
        )
        model, optimizer, scheduler, loss_fn = make_model(args, device)
        best_epoch = -1
        best_valid_loss = float("inf")
        best_model = None

        for epoch in range(args.epochs):
            train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, epoch, device)
            if valid_loader is None:
                print(
                    f"[minimol] member={member_idx + 1}/{args.ensemble_size} "
                    f"epoch={epoch + 1}/{args.epochs} train_all=true"
                )
            else:
                valid_loss = evaluate_loss(model, valid_loader, loss_fn, device)
                if valid_loss < best_valid_loss:
                    best_epoch = epoch + 1
                    best_valid_loss = valid_loss
                    best_model = deepcopy(model).cpu()
                print(
                    f"[minimol] member={member_idx + 1}/{args.ensemble_size} "
                    f"epoch={epoch + 1}/{args.epochs} valid_loss={valid_loss:.6f} "
                    f"best_epoch={best_epoch}"
                )

        if valid_loader is None:
            best_epoch = args.epochs
            best_model = deepcopy(model).cpu()

        if best_model is None:
            raise RuntimeError("No model was trained")

        if valid_loader is not None and valid is not None:
            best_model.to(device)
            valid_score = predict_proba(best_model, valid_loader, device)
            threshold, valid_macro_f1 = choose_threshold(valid.labels, valid_score, args.threshold_strategy)
            best_model.cpu()
        else:
            threshold = 0.5
            valid_macro_f1 = None
        best_models.append(best_model)
        run_results.append(
            RunResult(
                seed=seed,
                best_epoch=best_epoch,
                best_valid_loss=float(best_valid_loss) if valid_loader is not None else None,
                valid_threshold=threshold,
                valid_macro_f1=valid_macro_f1,
            )
        )

    valid_member_scores = []
    test_member_scores = []
    for model in best_models:
        model.to(device)
        if valid_loader is not None:
            valid_member_scores.append(predict_proba(model, valid_loader, device))
        test_member_scores.append(predict_proba(model, test_loader, device))
        model.cpu()

    test_scores = np.mean(np.stack(test_member_scores), axis=0)
    if valid is not None:
        valid_scores = np.mean(np.stack(valid_member_scores), axis=0)
        threshold, valid_macro_f1 = choose_threshold(valid.labels, valid_scores, args.threshold_strategy)
        valid_metrics = evaluate_metrics(valid.labels, valid_scores, threshold)
        valid_metrics_fixed = evaluate_metrics(valid.labels, valid_scores, 0.5)
        valid_tuned_threshold, _ = choose_threshold(valid.labels, valid_scores, "valid_macro_f1")
        valid_metrics_tuned = evaluate_metrics(valid.labels, valid_scores, valid_tuned_threshold)
        test_metrics_tuned = evaluate_metrics(test.labels, test_scores, valid_tuned_threshold)
    else:
        threshold = 0.5
        valid_macro_f1 = None
        valid_metrics = None
        valid_metrics_fixed = None
        valid_metrics_tuned = None
        test_metrics_tuned = None
    test_metrics = evaluate_metrics(test.labels, test_scores, threshold)
    test_metrics_fixed = evaluate_metrics(test.labels, test_scores, 0.5)

    output = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "splits": {
            "train": len(train.labels),
            "valid": len(valid.labels) if valid is not None else 0,
            "test": len(test.labels),
        },
        "model_selection": {
            "criterion": (
                "fixed configured epochs using all training molecules"
                if args.train_all
                else "lowest validation BCE loss per ensemble member"
            ),
            "threshold_strategy": "fixed_0.5" if args.train_all else args.threshold_strategy,
            "ensemble_valid_macro_f1_at_threshold": (
                float(valid_macro_f1) if valid_macro_f1 is not None else None
            ),
        },
        "members": [asdict(result) for result in run_results],
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "valid_metrics_fixed_0.5": valid_metrics_fixed,
        "test_metrics_fixed_0.5": test_metrics_fixed,
        "valid_metrics_valid_macro_f1_threshold": valid_metrics_tuned,
        "test_metrics_valid_macro_f1_threshold": test_metrics_tuned,
    }

    metrics_path = args.output_dir / "metrics.json"
    predictions_path = args.output_dir / "test_predictions.jsonl"
    checkpoint_path = args.output_dir / "best_ensemble.pt"
    with metrics_path.open("w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    with predictions_path.open("w") as f:
        for smiles, label, score in zip(test.smiles, test.labels, test_scores):
            row = {"drug": smiles, "Y": label, "score": float(score), "prediction": int(score >= threshold)}
            f.write(json.dumps(row) + "\n")
    torch.save(
        {
            "model_state_dicts": [model.state_dict() for model in best_models],
            "run_results": [asdict(result) for result in run_results],
            "threshold": threshold,
            "hparams": output["args"],
        },
        checkpoint_path,
    )

    print("[minimol] valid metrics:", json.dumps(valid_metrics, sort_keys=True))
    print("[minimol] test metrics:", json.dumps(test_metrics, sort_keys=True))
    print(f"[minimol] wrote {metrics_path}")
    print(f"[minimol] wrote {predictions_path}")
    print(f"[minimol] wrote {checkpoint_path}")


if __name__ == "__main__":
    main()
