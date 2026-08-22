"""Run the three condition-aware Bioavailability benchmark baselines.

The retrieval baselines rank train molecules from the query's exact canonical
condition first.  If that pool contains fewer than ``k`` unique parents, the
remaining slots come from the frozen no-reported-condition pool.  Null-condition
queries never retrieve condition-specific rows.

The trained MiniMol head concatenates a frozen 512-D molecule embedding with an
explicit one-hot encoding of the canonical condition group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from baselines.minimol.condition_features import (
    condition_one_hot as encode_condition_one_hot,
    condition_vocabulary as build_condition_vocabulary,
)
from baselines.minimol.embedding_runtime import (
    DEFAULT_MINIMOL_SOURCE,
    checkpoint_provenance,
    create_featurizer,
    embed_smiles,
)
from baselines.minimol.head_runtime import (
    EmbeddingDataset,
    TaskHead,
    predict_scores,
    train_one_epoch,
)
from tools.chembl_tool.tasks.bioavailability_ma.condition_ontology import (
    NO_REPORTED_CONDITION,
)


DEFAULT_DATA_DIR = Path(
    "data/processed_starling_context_conditioned_selected_v1/Bioavailability_Ma/scaffold"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/baselines/bioavailability_context_conditioned_selected_v1/scaffold_test"
)
DEFAULT_OLD_CACHE = Path(
    "outputs/baselines/minimol_embedding_cache_starling_record_supported_v2/"
    "Bioavailability_Ma/scaffold"
)
FP_RADIUS = 2
FP_BITS = 2048


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {
                "drug",
                "Y",
                "condition_group",
                "condition_scope",
                "molecule_identity_key",
            } - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields {sorted(missing)}")
            row["drug"] = str(row["drug"])
            row["Y"] = int(row["Y"])
            if row["Y"] not in (0, 1):
                raise ValueError(f"{path}:{line_number} has non-binary Y={row['Y']}")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def condition_vocabulary(train_rows: list[dict[str, Any]]) -> list[str]:
    groups = build_condition_vocabulary(
        [str(row["condition_group"]) for row in train_rows]
    )
    if NO_REPORTED_CONDITION not in groups:
        raise ValueError(f"Training rows are missing {NO_REPORTED_CONDITION!r}")
    return groups


def condition_one_hot(rows: list[dict[str, Any]], vocabulary: list[str]) -> torch.Tensor:
    return encode_condition_one_hot(
        [str(row["condition_group"]) for row in rows], vocabulary
    )


def _metric_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    y_true = np.asarray([int(row["Y"]) for row in rows], dtype=int)
    y_score = np.asarray([float(row["score"]) for row in rows], dtype=float)
    y_pred = np.asarray([int(row["prediction"]) for row in rows], dtype=int)
    labels = set(y_true.tolist())
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return {
        "n": len(rows),
        "n_y0": int(np.sum(y_true == 0)),
        "n_y1": int(np.sum(y_true == 1)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_score)) if labels == {0, 1} else None,
        "positive_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "positive_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "positive_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _metric_block(rows),
        "no_reported_external_condition": _metric_block(
            [row for row in rows if row["condition_group"] == NO_REPORTED_CONDITION]
        ),
        "external_conditions": _metric_block(
            [row for row in rows if row["condition_group"] != NO_REPORTED_CONDITION]
        ),
        "per_group": {
            group: _metric_block([row for row in rows if row["condition_group"] == group])
            for group in sorted({str(row["condition_group"]) for row in rows})
        },
    }


def select_context_neighbors(
    query: dict[str, Any],
    train_rows: list[dict[str, Any]],
    similarities: np.ndarray,
    *,
    k: int,
) -> list[tuple[int, str]]:
    """Select exact-condition neighbors, then null fallback, with unique parents."""
    if k <= 0:
        raise ValueError("k must be positive")
    if len(similarities) != len(train_rows):
        raise ValueError("similarities and train rows must align")
    ranked = np.lexsort((np.arange(len(train_rows)), -np.asarray(similarities)))
    query_group = str(query["condition_group"])
    selected: list[tuple[int, str]] = []
    used_parents: set[str] = set()

    def take(group: str, provenance: str) -> None:
        for index in ranked:
            if len(selected) >= k:
                return
            row = train_rows[int(index)]
            parent = str(row["molecule_identity_key"])
            if row["condition_group"] != group or parent in used_parents:
                continue
            selected.append((int(index), provenance))
            used_parents.add(parent)

    take(query_group, "exact_condition")
    if query_group != NO_REPORTED_CONDITION and len(selected) < k:
        take(NO_REPORTED_CONDITION, "fallback_no_reported_external_condition")
    if len(selected) < k:
        raise ValueError(
            f"Only {len(selected)} eligible unique-parent neighbors for {query_group!r}; k={k}"
        )
    return selected


def _knn_predictions(
    train_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    similarity_matrix: np.ndarray,
    *,
    k: int,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for query_index, query in enumerate(evaluation_rows):
        similarities = similarity_matrix[query_index]
        selected = select_context_neighbors(query, train_rows, similarities, k=k)
        neighbors = [
            {
                "train_index": index,
                "drug": train_rows[index]["drug"],
                "Y": train_rows[index]["Y"],
                "condition_group": train_rows[index]["condition_group"],
                "molecule_identity_key": train_rows[index]["molecule_identity_key"],
                "context_match": provenance,
                "similarity": float(similarities[index]),
            }
            for index, provenance in selected
        ]
        score = float(sum(row["Y"] for row in neighbors) / k)
        predictions.append(
            {
                "query_index": query_index,
                "benchmark_row_id": query.get("benchmark_row_id"),
                "drug": query["drug"],
                "Y": query["Y"],
                "condition_group": query["condition_group"],
                "condition_scope": query["condition_scope"],
                "score": score,
                "prediction": int(score >= 0.5),
                "correct": int(score >= 0.5) == query["Y"],
                "n_exact_condition_neighbors": sum(
                    row["context_match"] == "exact_condition" for row in neighbors
                ),
                "n_null_fallback_neighbors": sum(
                    row["context_match"] == "fallback_no_reported_external_condition"
                    for row in neighbors
                ),
                "neighbors": neighbors,
            }
        )
    return predictions


def _morgan_similarities(
    train_rows: list[dict[str, Any]], evaluation_rows: list[dict[str, Any]]
) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FP_RADIUS,
        fpSize=FP_BITS,
        includeChirality=False,
        useBondTypes=True,
    )

    def fp(row: dict[str, Any]):
        molecule = Chem.MolFromSmiles(row["drug"])
        if molecule is None:
            raise ValueError(f"Invalid SMILES: {row['drug']!r}")
        return generator.GetFingerprint(molecule)

    train_fps = [fp(row) for row in train_rows]
    result = np.empty((len(evaluation_rows), len(train_rows)), dtype=np.float32)
    for query_index, row in enumerate(evaluation_rows):
        result[query_index] = DataStructs.BulkTanimotoSimilarity(fp(row), train_fps)
    return result


def _load_reusable_embeddings(cache_dirs: list[Path]) -> tuple[dict[str, torch.Tensor], list[Path]]:
    registry: dict[str, torch.Tensor] = {}
    files = sorted(
        path for cache_dir in cache_dirs for path in cache_dir.glob("*.pt") if path.is_file()
    )
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        smiles = list(map(str, payload["smiles"]))
        embeddings = payload["embeddings"].detach().cpu().float()
        if len(smiles) != len(embeddings):
            raise ValueError(f"Embedding cache row mismatch: {path}")
        for molecule, embedding in zip(smiles, embeddings, strict=True):
            previous = registry.get(molecule)
            if previous is not None and not torch.allclose(
                previous, embedding, rtol=1e-5, atol=1e-6
            ):
                raise ValueError(f"Conflicting cached MiniMol embedding for {molecule!r}")
            registry[molecule] = embedding
    return registry, files


def materialize_embeddings(
    splits: dict[str, list[dict[str, Any]]],
    *,
    cache_dir: Path,
    reuse_cache_dirs: list[Path],
    minimol_source: Path,
    device: str,
    batch_size: int,
    force: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, torch.Tensor] = {}
    pending: dict[str, list[dict[str, Any]]] = {}
    for split, rows in splits.items():
        path = cache_dir / f"{split}.pt"
        if path.is_file() and not force:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            expected_smiles = [row["drug"] for row in rows]
            expected_labels = [row["Y"] for row in rows]
            if list(payload.get("smiles", [])) == expected_smiles and [
                int(value) for value in payload.get("labels", [])
            ] == expected_labels:
                loaded[split] = payload["embeddings"].detach().cpu().float()
                continue
        pending[split] = rows

    reusable, source_files = _load_reusable_embeddings(reuse_cache_dirs)
    source_count = len(reusable)
    missing = list(
        dict.fromkeys(
            row["drug"]
            for rows in pending.values()
            for row in rows
            if row["drug"] not in reusable
        )
    )
    if missing:
        featurizer = create_featurizer(
            batch_size=batch_size,
            minimol_source=minimol_source,
            device=device,
        )
        new_embeddings = embed_smiles(featurizer, missing)
        reusable.update(zip(missing, new_embeddings, strict=True))

    split_audit: dict[str, Any] = {}
    missing_set = set(missing)
    for split, rows in pending.items():
        tensor = torch.stack([reusable[row["drug"]] for row in rows]).float()
        path = cache_dir / f"{split}.pt"
        torch.save(
            {
                "smiles": [row["drug"] for row in rows],
                "labels": [row["Y"] for row in rows],
                "condition_groups": [row["condition_group"] for row in rows],
                "embeddings": tensor,
            },
            path,
        )
        loaded[split] = tensor
        split_audit[split] = {
            "n_rows": len(rows),
            "n_reused": sum(row["drug"] not in missing_set for row in rows),
            "n_newly_embedded": sum(row["drug"] in missing_set for row in rows),
        }
    for split, rows in splits.items():
        tensor = loaded[split]
        if tensor.ndim != 2 or tensor.shape[0] != len(rows) or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid MiniMol embeddings for {split}: {tuple(tensor.shape)}")
    return loaded, {
        "match_key": "exact_smiles",
        "source_cache_files": [str(path) for path in source_files],
        "n_source_unique_smiles": source_count,
        "n_unique_newly_embedded": len(missing),
        "splits_materialized": split_audit,
    }


def _cosine_similarities(train: torch.Tensor, evaluation: torch.Tensor) -> np.ndarray:
    train = torch.nn.functional.normalize(train.float(), p=2, dim=1)
    evaluation = torch.nn.functional.normalize(evaluation.float(), p=2, dim=1)
    return (evaluation @ train.T).clamp(-1.0, 1.0).cpu().numpy()


def _head_runtime(
    input_dim: int,
    *,
    device: torch.device,
    lr: float,
    weight_decay: float,
    epochs: int,
    warmup: int,
    hidden_dim: int,
    depth: int,
    dropout: float,
) -> tuple[nn.Module, optim.Optimizer, LambdaLR, nn.Module]:
    model = TaskHead(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
        combine=True,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn: nn.Module = nn.BCEWithLogitsLoss(reduction="none")

    def lr_fn(epoch: int) -> float:
        schedule_epoch = epoch + 1
        if warmup > 0 and schedule_epoch <= warmup:
            return schedule_epoch / warmup
        denom = max(1, epochs - warmup)
        decay_epoch = schedule_epoch - warmup if warmup > 0 else epoch
        decay_epoch = min(denom, max(0, decay_epoch))
        return (1 + math.cos(math.pi * decay_epoch / denom)) / 2

    return model, optimizer, LambdaLR(optimizer, lr_lambda=lr_fn), loss_fn


def _head_predictions(
    train_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    train_embeddings: torch.Tensor,
    evaluation_embeddings: torch.Tensor,
    vocabulary: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[nn.Module]]:
    train_features = torch.cat(
        [train_embeddings, condition_one_hot(train_rows, vocabulary)], dim=1
    )
    evaluation_features = torch.cat(
        [evaluation_embeddings, condition_one_hot(evaluation_rows, vocabulary)], dim=1
    )
    evaluation_loader = DataLoader(
        EmbeddingDataset(evaluation_features, [row["Y"] for row in evaluation_rows]),
        batch_size=args.eval_batch_size,
        shuffle=False,
    )
    models: list[nn.Module] = []
    scores: list[np.ndarray] = []
    device = torch.device(args.device)
    for member_index in range(args.ensemble_size):
        seed = args.seed + member_index
        _set_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            EmbeddingDataset(train_features, [row["Y"] for row in train_rows]),
            batch_size=args.train_batch_size,
            shuffle=True,
            generator=generator,
        )
        model, optimizer, scheduler, loss_fn = _head_runtime(
            train_features.shape[1],
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            warmup=args.warmup,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            dropout=args.dropout,
        )
        for epoch in range(args.epochs):
            loss = train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device)
            print(
                f"[condition-head] member={member_index + 1}/{args.ensemble_size} "
                f"epoch={epoch + 1}/{args.epochs} loss={loss:.6f}",
                flush=True,
            )
        models.append(deepcopy(model).cpu())
        scores.append(predict_scores(model, evaluation_loader, device))
        model.cpu()
    ensemble_scores = np.mean(np.stack(scores), axis=0)
    predictions = [
        {
            "query_index": index,
            "benchmark_row_id": row.get("benchmark_row_id"),
            "drug": row["drug"],
            "Y": row["Y"],
            "condition_group": row["condition_group"],
            "condition_scope": row["condition_scope"],
            "score": float(ensemble_scores[index]),
            "prediction": int(ensemble_scores[index] >= 0.5),
            "correct": int(ensemble_scores[index] >= 0.5) == row["Y"],
        }
        for index, row in enumerate(evaluation_rows)
    ]
    return predictions, models


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_method(
    output_dir: Path,
    method: str,
    predictions: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    evaluation_split: str,
) -> dict[str, Any]:
    method_dir = output_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "method": method,
        "evaluation_split": evaluation_split,
        **metadata,
        "metrics": summarize_predictions(predictions),
    }
    _write_jsonl(method_dir / f"{evaluation_split}_predictions.jsonl", predictions)
    (method_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-cache-dir", type=Path, default=None)
    parser.add_argument("--reuse-embedding-cache-dir", type=Path, action="append", default=[])
    parser.add_argument("--minimol-source", type=Path, default=DEFAULT_MINIMOL_SOURCE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3, choices=(3, 4))
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--evaluation-split",
        choices=("valid", "test"),
        default="test",
    )
    parser.add_argument("--force-embed", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("morgan_knn", "minimol_knn", "minimol_head"),
        default=["morgan_knn", "minimol_knn", "minimol_head"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.data_dir / "train.jsonl"
    evaluation_path = args.data_dir / f"{args.evaluation_split}.jsonl"
    train_rows = _read_rows(train_path)
    evaluation_rows = _read_rows(evaluation_path)
    vocabulary = condition_vocabulary(train_rows)
    if ({row["condition_group"] for row in evaluation_rows} - set(vocabulary)):
        raise ValueError("Every evaluation condition group must occur in train")
    results: dict[str, Any] = {}

    if "morgan_knn" in args.methods:
        similarities = _morgan_similarities(train_rows, evaluation_rows)
        predictions = _knn_predictions(train_rows, evaluation_rows, similarities, k=args.k)
        results["morgan_knn"] = _write_method(
            args.output_dir,
            "morgan_knn",
            predictions,
            {
                "k": args.k,
                "reference_splits": ["train"],
                "retrieval_policy": "exact_condition_then_null_fallback",
                "fingerprint": {"type": "RDKit-Morgan", "radius": FP_RADIUS, "n_bits": FP_BITS},
            },
            evaluation_split=args.evaluation_split,
        )

    needs_minimol = bool({"minimol_knn", "minimol_head"} & set(args.methods))
    embeddings: dict[str, torch.Tensor] = {}
    embedding_audit: dict[str, Any] | None = None
    if needs_minimol:
        cache_dir = args.embedding_cache_dir or args.output_dir / "embeddings"
        reuse_dirs = args.reuse_embedding_cache_dir or [DEFAULT_OLD_CACHE]
        embeddings, embedding_audit = materialize_embeddings(
            {"train": train_rows, args.evaluation_split: evaluation_rows},
            cache_dir=cache_dir,
            reuse_cache_dirs=reuse_dirs,
            minimol_source=args.minimol_source,
            device=args.device,
            batch_size=args.embedding_batch_size,
            force=args.force_embed,
        )
        (args.output_dir / "embedding_manifest.json").write_text(
            json.dumps(embedding_audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if "minimol_knn" in args.methods:
        similarities = _cosine_similarities(
            embeddings["train"], embeddings[args.evaluation_split]
        )
        predictions = _knn_predictions(
            train_rows, evaluation_rows, similarities, k=args.k
        )
        results["minimol_knn"] = _write_method(
            args.output_dir,
            "minimol_knn",
            predictions,
            {
                "k": args.k,
                "reference_splits": ["train"],
                "retrieval_policy": "exact_condition_then_null_fallback",
                "embedding": {"model": "MiniMol", "dimension": int(embeddings["train"].shape[1])},
            },
            evaluation_split=args.evaluation_split,
        )

    if "minimol_head" in args.methods:
        predictions, models = _head_predictions(
            train_rows,
            evaluation_rows,
            embeddings["train"],
            embeddings[args.evaluation_split],
            vocabulary,
            args,
        )
        results["minimol_head"] = _write_method(
            args.output_dir,
            "minimol_head",
            predictions,
            {
                "molecule_embedding_dim": int(embeddings["train"].shape[1]),
                "condition_one_hot_dim": len(vocabulary),
                "input_dim": int(embeddings["train"].shape[1]) + len(vocabulary),
                "condition_vocabulary": vocabulary,
                "threshold": 0.5,
                "training": {
                    "reference_splits": ["train"],
                    "epochs": args.epochs,
                    "ensemble_size": args.ensemble_size,
                    "seed_start": args.seed,
                    "hidden_dim": args.hidden_dim,
                    "depth": args.depth,
                    "dropout": args.dropout,
                    "lr": args.lr,
                    "warmup": args.warmup,
                    "weight_decay": args.weight_decay,
                },
            },
            evaluation_split=args.evaluation_split,
        )
        torch.save(
            {
                "model_state_dicts": [model.state_dict() for model in models],
                "condition_vocabulary": vocabulary,
                "molecule_embedding_dim": int(embeddings["train"].shape[1]),
                "threshold": 0.5,
            },
            args.output_dir / "minimol_head" / "ensemble.pt",
        )

    manifest = {
        "lineage": "bioavailability_context_conditioned_selected_v1",
        "evaluation_split": args.evaluation_split,
        "data_dir": str(args.data_dir),
        "input_sha256": {
            "train": _sha256(train_path),
            args.evaluation_split: _sha256(evaluation_path),
        },
        "n_train": len(train_rows),
        "n_evaluation": len(evaluation_rows),
        "n_condition_groups": len(vocabulary),
        "condition_vocabulary": vocabulary,
        "methods": args.methods,
        "embedding_audit": embedding_audit,
        "minimol_checkpoint": checkpoint_provenance(args.minimol_source) if needs_minimol else None,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({method: result["metrics"] for method, result in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
