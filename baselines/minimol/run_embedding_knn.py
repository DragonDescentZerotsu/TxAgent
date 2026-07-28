"""Evaluate label retrieval with cosine similarity over cached MiniMol embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from baselines.structure_knn.run import _metrics, _read_split, _write_jsonl


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.k <= 0:
        raise ValueError("--k must be positive")

    train_path = args.data_dir / "train.jsonl"
    test_path = args.data_dir / "test.jsonl"
    train = _read_split(train_path)
    test = _read_split(test_path)
    if len(train) < args.k:
        raise ValueError(f"Training set has {len(train)} rows, fewer than k={args.k}")

    train_cache_path = args.embedding_cache_dir / "train.pt"
    test_cache_path = args.embedding_cache_dir / "test.pt"
    train_embeddings = _load_embeddings(train_cache_path, train)
    test_embeddings = _load_embeddings(test_cache_path, test)
    if train_embeddings.shape[1] != test_embeddings.shape[1]:
        raise ValueError(
            "Train/test embedding dimensions differ: "
            f"{train_embeddings.shape[1]} != {test_embeddings.shape[1]}"
        )

    train_embeddings = F.normalize(train_embeddings.float(), p=2, dim=1)
    test_embeddings = F.normalize(test_embeddings.float(), p=2, dim=1)
    similarities = (test_embeddings @ train_embeddings.T).clamp(-1.0, 1.0)

    predictions = []
    train_indices = np.arange(len(train))
    for query_index, row in enumerate(test):
        query_similarities = similarities[query_index].numpy()
        # Primary key is descending cosine similarity; train index is a stable tie-break.
        ranked_indices = np.lexsort((train_indices, -query_similarities))
        selected_indices = ranked_indices[: args.k]
        neighbors = [
            {
                "train_index": int(train_index),
                "drug": train[int(train_index)]["drug"],
                "Y": train[int(train_index)]["Y"],
                "similarity": float(query_similarities[int(train_index)]),
            }
            for train_index in selected_indices
        ]
        score = sum(neighbor["Y"] for neighbor in neighbors) / args.k
        prediction = int(score >= 0.5)
        predictions.append(
            {
                "query_index": query_index,
                "drug": row["drug"],
                "Y": row["Y"],
                "prediction": prediction,
                "score": score,
                "correct": prediction == row["Y"],
                "status": "ok",
                "neighbors": neighbors,
            }
        )

    metrics = _metrics(predictions)
    top_k_similarities = [
        neighbor["similarity"]
        for prediction in predictions
        for neighbor in prediction["neighbors"]
    ]
    metrics.update(
        {
            "method": f"minimol_embedding_cosine_knn_k{args.k}",
            "k": args.k,
            "vote": "unweighted_majority",
            "score": "positive_neighbor_fraction",
            "n_train": len(train),
            "n_test": len(test),
            "n_evaluated": len(test),
            "evaluation_coverage": 1.0,
            "retrieval_diagnostics": {
                "mean_top_k_cosine_similarity": (
                    sum(top_k_similarities) / len(top_k_similarities)
                ),
                "minimum_top_k_cosine_similarity": min(top_k_similarities),
                "maximum_top_k_cosine_similarity": max(top_k_similarities),
            },
            "embedding": {
                "model": "MiniMol",
                "source": "cached formal baseline embeddings",
                "dimension": int(train_embeddings.shape[1]),
                "normalization": "L2",
                "similarity": "cosine",
            },
        }
    )

    manifest = {
        "data_dir": str(args.data_dir),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "embedding_cache_dir": str(args.embedding_cache_dir),
        "train_embedding_cache": str(train_cache_path),
        "test_embedding_cache": str(test_cache_path),
        "output_dir": str(args.output_dir),
        "input_sha256": {
            "train_jsonl": _sha256(train_path),
            "test_jsonl": _sha256(test_path),
            "train_embedding_cache": _sha256(train_cache_path),
            "test_embedding_cache": _sha256(test_cache_path),
        },
        **{
            key: metrics[key]
            for key in ("method", "k", "vote", "score", "embedding")
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "test_predictions.jsonl", predictions)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


def _load_embeddings(
    path: Path,
    split: list[dict[str, Any]],
) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Embedding cache does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Embedding cache must contain a dictionary: {path}")
    missing = {"smiles", "labels", "embeddings"} - payload.keys()
    if missing:
        raise ValueError(f"Embedding cache {path} is missing keys: {sorted(missing)}")

    expected_smiles = [row["drug"] for row in split]
    expected_labels = [row["Y"] for row in split]
    if list(payload["smiles"]) != expected_smiles:
        raise ValueError(f"Embedding cache SMILES do not exactly match the benchmark split: {path}")
    if [int(label) for label in payload["labels"]] != expected_labels:
        raise ValueError(f"Embedding cache labels do not exactly match the benchmark split: {path}")

    embeddings = payload["embeddings"]
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError(f"Embedding cache must contain a rank-2 tensor: {path}")
    if embeddings.shape[0] != len(split):
        raise ValueError(
            f"Embedding cache row count does not match split: {embeddings.shape[0]} != {len(split)}"
        )
    if not torch.isfinite(embeddings).all():
        raise ValueError(f"Embedding cache contains non-finite values: {path}")
    if torch.any(torch.linalg.vector_norm(embeddings.float(), dim=1) == 0):
        raise ValueError(f"Embedding cache contains zero-norm rows: {path}")
    return embeddings.detach().cpu()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=3)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
