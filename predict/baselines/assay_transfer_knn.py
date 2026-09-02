"""Evaluate top-5 label voting after V9 reranking of Morgan parent pools.

The frozen cache contains every condition-specific training row from each
selected Morgan parent. As in the V9 training metric, reranking and voting are
record-level: repeated parents under different conditions remain distinct.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, f1_score

from predict.retrieval.assay_reranking.runtime import CACHE_ROOT, file_sha256
from predict.retrieval.assay_reranking.v9 import (
    RANKING_PROFILE_NAME,
    RANKING_SCHEMA_VERSION,
    model_profile,
    verify_vendored_assets,
)
from predict.utils.json import atomic_output_path, write_json_atomic, write_jsonl_atomic


TASKS = ("bbb_martins", "bioavailability_ma", "skin_reaction")
DEFAULT_WIDTHS = (25, 50, 75, 100)


def select_neighbors(
    rows: list[dict[str, Any]], *, morgan_width: int, k: int
) -> list[dict[str, Any]]:
    """Restrict the Morgan pool, then take the k highest V9 transfer scores."""
    eligible = [
        row for row in rows if int(row["retrieval_parent_rank"]) < morgan_width
    ]
    eligible.sort(
        key=lambda row: (
            -float(row["model_score"]),
            int(row["retrieval_parent_rank"]),
            int(row["retrieval_parent_context_index"]),
            str(row["retrieval_record_id"]),
        )
    )
    if len(eligible) < k:
        raise ValueError(
            f"Morgan-{morgan_width} produced {len(eligible)} records, fewer than k={k}"
        )
    return eligible[:k]


def evaluate_task(
    task: str, *, cache_root: Path, widths: Sequence[int], k: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cache_dir = cache_root / task / "scaffold" / "valid"
    version_path = cache_dir / "VERSION.json"
    rankings_path = cache_dir / "rankings.parquet"
    version = json.loads(version_path.read_text(encoding="utf-8"))
    if version.get("schema_version") != RANKING_SCHEMA_VERSION:
        raise ValueError(f"{task} uses an incompatible ranking-cache schema")
    if version.get("status") != "complete":
        raise ValueError(f"{task} ranking cache is incomplete")
    if version.get("model") != model_profile(task):
        raise ValueError(f"{task} ranking cache uses another model")
    if version.get("prompt_assets") != verify_vendored_assets():
        raise ValueError(f"{task} ranking cache uses different prompt assets")
    if version.get("rankings_sha256") != file_sha256(rankings_path):
        raise ValueError(f"{task} rankings hash disagrees with VERSION.json")
    for split in ("train", "valid"):
        source = Path(version["inputs"][split])
        if file_sha256(source) != version["inputs"][f"{split}_sha256"]:
            raise ValueError(f"{task} current {split} gold differs from the cache")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in pq.read_table(rankings_path).to_pylist():
        grouped.setdefault(str(row["query_record_id"]), []).append(row)
    if set(grouped) != set(version["query_record_ids"]):
        raise ValueError(f"{task} query IDs disagree with VERSION.json")

    predictions, metrics = [], []
    for width in widths:
        task_predictions = []
        for query_id, rows in grouped.items():
            neighbors = select_neighbors(rows, morgan_width=width, k=k)
            target = int(rows[0]["query_gold_Y"])
            score = sum(int(row["retrieval_gold_Y"]) for row in neighbors) / k
            prediction = int(score >= 0.5)
            task_predictions.append({
                "task": task,
                "query_record_id": query_id,
                "morgan_width": width,
                "k": k,
                "target": target,
                "prediction": prediction,
                "score": score,
                "neighbors": [
                    {
                        key: row[key]
                        for key in (
                            "retrieval_record_id",
                            "retrieval_molecule_identity_key",
                            "retrieval_condition_group",
                            "retrieval_gold_Y",
                            "retrieval_parent_rank",
                            "morgan_tanimoto_similarity",
                            "model_score",
                            "prob_transfer",
                        )
                    }
                    for row in neighbors
                ],
            })
        targets = [row["target"] for row in task_predictions]
        predicted = [row["prediction"] for row in task_predictions]
        metrics.append({
            "task": task,
            "morgan_width": width,
            "k": k,
            "n": len(task_predictions),
            "accuracy": float(accuracy_score(targets, predicted)),
            "macro_f1": float(
                f1_score(
                    targets,
                    predicted,
                    labels=[0, 1],
                    average="macro",
                    zero_division=0.0,
                )
            ),
        })
        predictions.extend(task_predictions)
    provenance = {
        "version": str(version_path),
        "version_sha256": file_sha256(version_path),
        "rankings": str(rankings_path),
        "rankings_sha256": file_sha256(rankings_path),
        "model": version["model"],
        "prompt_assets": version["prompt_assets"],
    }
    return metrics, predictions, provenance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--morgan-widths", nargs="+", type=int, default=DEFAULT_WIDTHS)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=CACHE_ROOT / RANKING_PROFILE_NAME,
    )
    args = parser.parse_args(argv)
    widths = sorted(set(args.morgan_widths))
    if args.k <= 0 or not widths or widths[0] <= 0:
        raise ValueError("Require k > 0 and positive Morgan widths")

    metrics, predictions, provenance = [], [], {}
    for task in args.tasks:
        task_metrics, task_predictions, task_provenance = evaluate_task(
            task, cache_root=args.cache_root, widths=widths, k=args.k
        )
        metrics.extend(task_metrics)
        predictions.extend(task_predictions)
        provenance[task] = task_provenance

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.output_dir / "predictions.jsonl", predictions)
    write_json_atomic(
        args.output_dir / "manifest.json",
        {
            "method": "v9_transfer_reranked_record_knn",
            "evaluation_split": "valid",
            "candidate_policy": "Morgan top-N parents, expanded condition rows",
            "ranking": "descending V9 model_score",
            "vote": "unweighted top-k record-label majority",
            "tasks": args.tasks,
            "morgan_widths": widths,
            "k": args.k,
            "cache_root": str(args.cache_root),
            "inputs": provenance,
            "metrics": metrics,
        },
    )
    with atomic_output_path(args.output_dir / "macro_f1.tsv") as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
