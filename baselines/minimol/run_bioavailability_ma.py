"""Train and evaluate a MiniMol baseline on Bioavailability_Ma.

This follows the downstream head used by MiniMol's TDC leaderboard script, but
uses the repo's fixed train/valid/test JSONL splits instead of TDC folds.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
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


DEFAULT_DATA_DIR = Path("data/processed/Bioavailability_Ma")
DEFAULT_OUTPUT_DIR = Path("outputs/baselines/minimol/bioavailability_ma")
DEFAULT_MINIMOL_SOURCE = Path("/data1/tianang/Projects/minimol")


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
    best_valid_loss: float
    valid_threshold: float
    valid_macro_f1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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


def ensure_minimol_import(minimol_source: Path) -> None:
    if importlib.util.find_spec("minimol") is not None:
        return
    if minimol_source.exists():
        sys.path.insert(0, str(minimol_source))


def patch_graphium_float32_featurization() -> None:
    """Avoid SciPy sparse float16 failures in Graphium's CPU featurization path."""
    from scipy.sparse import coo_matrix

    import graphium.data.datamodule as graphium_datamodule
    import graphium.features as graphium_features
    import graphium.features.featurizer as graphium_featurizer
    import graphium.features.nmp as graphium_nmp

    if getattr(graphium_featurizer.mol_to_pyggraph, "_txagent_float32_patch", False):
        return

    original_mol_to_pyggraph = graphium_featurizer.mol_to_pyggraph

    def mol_to_adjacency_matrix_float32(
        mol,
        use_bonds_weights: bool = False,
        add_self_loop: bool = False,
        dtype=np.float32,
    ):
        adj_idx = []
        adj_val = []
        for bond in mol.GetBonds():
            adj_idx.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            adj_idx.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
            val = graphium_nmp.BOND_TYPES[bond.GetBondType()] if use_bonds_weights else 1.0
            adj_val.extend([val, val])

        if adj_val:
            data = np.asarray(adj_val, dtype=np.float32)
            coords = np.asarray(adj_idx, dtype=np.int64).T.reshape(2, -1)
            adj = coo_matrix((data, coords), shape=(mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=np.float32)
        else:
            adj = coo_matrix(([], np.array([[], []])), shape=(mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=np.float32)

        if add_self_loop:
            arange = np.arange(adj.shape[0], dtype=int)
            adj[arange, arange] = 1
        return adj

    def mol_to_pyggraph_float32(*args, **kwargs):
        kwargs["dtype"] = np.float32
        return original_mol_to_pyggraph(*args, **kwargs)

    mol_to_pyggraph_float32._txagent_float32_patch = True
    graphium_featurizer.mol_to_adjacency_matrix = mol_to_adjacency_matrix_float32
    graphium_featurizer.mol_to_pyggraph = mol_to_pyggraph_float32
    graphium_features.mol_to_pyggraph = mol_to_pyggraph_float32
    graphium_datamodule.mol_to_pyggraph = mol_to_pyggraph_float32


def cache_path(output_dir: Path, split_name: str) -> Path:
    return output_dir / "embeddings" / f"{split_name}.pt"


def featurize_split(split_name: str, split: SplitData, args: argparse.Namespace) -> torch.Tensor:
    path = cache_path(args.output_dir, split_name)
    if path.exists() and not args.force_embed:
        payload = torch.load(path, map_location="cpu")
        if payload["smiles"] == split.smiles:
            return payload["embeddings"].float()
        print(f"[minimol] cache mismatch for {split_name}; recomputing embeddings")

    ensure_minimol_import(args.minimol_source)
    patch_graphium_float32_featurization()
    from hydra.core.global_hydra import GlobalHydra
    from minimol import Minimol

    print(f"[minimol] featurizing {split_name}: {len(split.smiles)} molecules")
    original_torch_load = torch.load

    def torch_load_weights_compatible(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    try:
        # MiniMol's bundled checkpoint predates PyTorch's weights_only=True default.
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        torch.load = torch_load_weights_compatible
        featurizer = Minimol(batch_size=args.embedding_batch_size)
    finally:
        torch.load = original_torch_load

    featurizer.datamodule.featurization_n_jobs = 1
    with torch.no_grad():
        embeddings = featurizer(split.smiles)

    if len(embeddings) != len(split.smiles):
        raise RuntimeError(f"MiniMol returned {len(embeddings)} embeddings for {len(split.smiles)} {split_name} molecules")

    tensor = torch.stack([embedding.detach().cpu().float() for embedding in embeddings])
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
    valid = load_split(args.data_dir / "valid.jsonl")
    test = load_split(args.data_dir / "test.jsonl")

    print(
        "[minimol] loaded splits: "
        f"train={len(train.labels)} valid={len(valid.labels)} test={len(test.labels)} device={device}"
    )

    train_embeddings = featurize_split("train", train, args)
    valid_embeddings = featurize_split("valid", valid, args)
    test_embeddings = featurize_split("test", test, args)

    valid_loader = DataLoader(EmbeddingDataset(valid_embeddings, valid.labels), batch_size=args.eval_batch_size, shuffle=False)
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

        if best_model is None:
            raise RuntimeError("No model was trained")

        best_model.to(device)
        valid_score = predict_proba(best_model, valid_loader, device)
        threshold, valid_macro_f1 = choose_threshold(valid.labels, valid_score, args.threshold_strategy)
        best_model.cpu()
        best_models.append(best_model)
        run_results.append(
            RunResult(
                seed=seed,
                best_epoch=best_epoch,
                best_valid_loss=float(best_valid_loss),
                valid_threshold=threshold,
                valid_macro_f1=valid_macro_f1,
            )
        )

    valid_member_scores = []
    test_member_scores = []
    for model in best_models:
        model.to(device)
        valid_member_scores.append(predict_proba(model, valid_loader, device))
        test_member_scores.append(predict_proba(model, test_loader, device))
        model.cpu()

    valid_scores = np.mean(np.stack(valid_member_scores), axis=0)
    test_scores = np.mean(np.stack(test_member_scores), axis=0)
    threshold, valid_macro_f1 = choose_threshold(valid.labels, valid_scores, args.threshold_strategy)
    valid_metrics = evaluate_metrics(valid.labels, valid_scores, threshold)
    test_metrics = evaluate_metrics(test.labels, test_scores, threshold)
    valid_metrics_fixed = evaluate_metrics(valid.labels, valid_scores, 0.5)
    test_metrics_fixed = evaluate_metrics(test.labels, test_scores, 0.5)
    valid_tuned_threshold, _ = choose_threshold(valid.labels, valid_scores, "valid_macro_f1")
    valid_metrics_tuned = evaluate_metrics(valid.labels, valid_scores, valid_tuned_threshold)
    test_metrics_tuned = evaluate_metrics(test.labels, test_scores, valid_tuned_threshold)

    output = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "splits": {"train": len(train.labels), "valid": len(valid.labels), "test": len(test.labels)},
        "model_selection": {
            "criterion": "lowest validation BCE loss per ensemble member",
            "threshold_strategy": args.threshold_strategy,
            "ensemble_valid_macro_f1_at_threshold": float(valid_macro_f1),
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
