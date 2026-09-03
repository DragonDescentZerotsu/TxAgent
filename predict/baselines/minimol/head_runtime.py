"""Shared frozen-embedding head runtime for MiniMol baselines."""

from __future__ import annotations

import math
from typing import Iterable, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

TaskType = Literal["classification", "regression"]


class TaskHead(nn.Module):
    """Leaderboard-style MiniMol task head over frozen 512-D embeddings."""

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
        output_dim = input_dim + hidden_dim if combine else hidden_dim
        self.final_dense = nn.Linear(output_dim, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.combine = combine
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_x = x

        x = self.dropout(F.relu(self.bn1(self.dense1(x))))
        x = self.dropout(F.relu(self.bn2(self.dense2(x))))
        if self.depth == 4:
            x = self.dropout(F.relu(self.bn3(self.dense3(x))))

        x = torch.cat((x, original_x), dim=1) if self.combine else x
        return self.final_dense(x)


class EmbeddingDataset(Dataset):
    """Pair frozen MiniMol embeddings with targets and optional sample weights."""

    def __init__(
        self,
        embeddings: torch.Tensor,
        labels: Iterable[float],
        sample_weights: Iterable[float] | None = None,
    ) -> None:
        self.embeddings = embeddings.float()
        self.labels = torch.tensor(list(labels), dtype=torch.float32)
        self.sample_weights = (
            torch.tensor(list(sample_weights), dtype=torch.float32)
            if sample_weights is not None
            else None
        )
        if self.sample_weights is not None and len(self.sample_weights) != len(self.labels):
            raise ValueError("sample_weights must have one value per label")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        row = (self.embeddings[idx], self.labels[idx])
        return row if self.sample_weights is None else (*row, self.sample_weights[idx])


def make_model(
    args,
    device: torch.device,
    *,
    task_type: TaskType = "classification",
    input_dim: int = 512,
) -> tuple[nn.Module, optim.Optimizer, LambdaLR, nn.Module]:
    """Build the current TxAgent MiniMol head, optimizer, schedule, and loss."""
    model = TaskHead(
        hidden_dim=args.hidden_dim,
        input_dim=input_dim,
        depth=args.depth,
        dropout=args.dropout,
        combine=True,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn: nn.Module = (
        nn.BCEWithLogitsLoss(reduction="none")
        if task_type == "classification"
        else nn.MSELoss(reduction="none")
    )

    def lr_fn(epoch: int) -> float:
        schedule_epoch = epoch + 1
        if args.warmup > 0 and schedule_epoch <= args.warmup:
            return schedule_epoch / args.warmup
        denom = max(1, args.epochs - args.warmup)
        decay_epoch = schedule_epoch - args.warmup if args.warmup > 0 else epoch
        decay_epoch = min(denom, max(0, decay_epoch))
        return (1 + math.cos(math.pi * decay_epoch / denom)) / 2

    scheduler = LambdaLR(optimizer, lr_lambda=lr_fn)
    return model, optimizer, scheduler, loss_fn


def _batch_loss(
    loss_fn: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor | None,
) -> torch.Tensor:
    losses = loss_fn(logits, targets)
    if sample_weights is None:
        return losses.mean()
    if torch.any(sample_weights <= 0):
        raise ValueError("sample weights must be positive")
    return torch.sum(losses * sample_weights) / torch.sum(sample_weights)


def _unpack_batch(batch, device: torch.device):
    inputs, targets, *optional_weights = batch
    weights = optional_weights[0].to(device) if optional_weights else None
    return inputs.to(device), targets.to(device), weights


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: LambdaLR,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Train one epoch and advance the corrected epoch-level schedule."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        inputs, targets, sample_weights = _unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs).squeeze(-1)
        loss = _batch_loss(loss_fn, logits, targets, sample_weights)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    scheduler.step()
    return total_loss / max(1, len(loader))


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    loss_numerator = 0.0
    loss_denominator = 0.0
    with torch.no_grad():
        for batch in loader:
            inputs, targets, sample_weights = _unpack_batch(batch, device)
            logits = model(inputs).squeeze(-1)
            losses = loss_fn(logits, targets)
            if sample_weights is None:
                loss_numerator += float(losses.sum().item())
                loss_denominator += float(losses.numel())
            else:
                loss_numerator += float(torch.sum(losses * sample_weights).item())
                loss_denominator += float(torch.sum(sample_weights).item())
    return loss_numerator / max(1.0, loss_denominator)


def predict_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    task_type: TaskType = "classification",
) -> np.ndarray:
    """Return probabilities for classification and raw values for regression."""
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0]
            logits = model(inputs.to(device)).squeeze(-1)
            scores = torch.sigmoid(logits) if task_type == "classification" else logits
            predictions.append(scores.detach().cpu().numpy())
    return np.concatenate(predictions)
