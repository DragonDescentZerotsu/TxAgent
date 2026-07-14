"""Generic scalar-evidence molecular KNN baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import statistics
from typing import Any

from tools.chembl_tool.common.experiment_retrieval import SourceExperimentConfig, retrieve_experiment_view


@dataclass(frozen=True)
class ScalarKnnConfig:
    description: str
    default_input: str
    default_index: str
    default_output_dir: str
    threshold: float
    positive_prediction: str
    negative_prediction: str
    source_config: SourceExperimentConfig


def main(config: ScalarKnnConfig, argv: list[str] | None = None) -> int:
    args = _parse_args(config, argv)
    records = _read_jsonl(Path(args.input_jsonl))
    with Path(args.index).open("rb") as handle:
        index = pickle.load(handle)

    predictions = []
    for query_index, record in enumerate(records):
        retrieval = retrieve_experiment_view(
            str(record.get(args.smiles_field) or ""),
            index,
            mode="direct",
            config=config.source_config,
            top_k_per_group=args.k,
            min_similarity=args.min_similarity,
        )
        neighbors = [
            neighbor
            for group in retrieval.get("groups") or []
            for neighbor in group.get("neighbors") or []
        ]
        votes = [_neighbor_vote(neighbor, threshold=config.threshold) for neighbor in neighbors]
        votes = [vote for vote in votes if vote is not None]
        predicted_label = _majority_vote(votes, fallback_label=args.fallback_label)
        predictions.append(
            {
                "query_index": query_index,
                "label": int(record.get(args.label_field)),
                "predicted_label": predicted_label,
                "prediction": config.positive_prediction if predicted_label == 1 else config.negative_prediction,
                "correct": predicted_label == int(record.get(args.label_field)),
                "used_fallback": not votes,
                "n_neighbors": len(neighbors),
                "n_numeric_votes": len(votes),
                "neighbors": votes,
            }
        )

    metrics = _metrics(predictions)
    metrics.update(
        {
            "k": args.k,
            "min_similarity": args.min_similarity,
            "threshold": config.threshold,
            "fallback_label": args.fallback_label,
            "n_fallback": sum(row["used_fallback"] for row in predictions),
        }
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "predictions.jsonl", predictions)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "input_jsonl": args.input_jsonl,
                "index": args.index,
                "source": config.source_config.source_name,
                "mode": "direct_numeric_scalar_knn",
                "k": args.k,
                "min_similarity": args.min_similarity,
                "threshold": config.threshold,
                "fallback_label": args.fallback_label,
                "paths": {"predictions": str(out_dir / "predictions.jsonl"), "metrics": str(out_dir / "metrics.json")},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


def _neighbor_vote(neighbor: dict[str, Any], *, threshold: float) -> dict[str, Any] | None:
    values = []
    for row in neighbor.get("evidence_rows") or []:
        try:
            values.append(float(row.get("standard_value")))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    value = float(statistics.median(values))
    return {
        "molecule_id": neighbor.get("molecule_chembl_id", ""),
        "similarity": neighbor.get("similarity"),
        "value": value,
        "label": int(value >= threshold),
    }


def _majority_vote(votes: list[dict[str, Any]], *, fallback_label: int) -> int:
    if not votes:
        return fallback_label
    positive = sum(vote["label"] == 1 for vote in votes)
    negative = len(votes) - positive
    if positive == negative:
        return int(
            sum(float(vote.get("similarity") or 0.0) for vote in votes if vote["label"] == 1)
            >= sum(float(vote.get("similarity") or 0.0) for vote in votes if vote["label"] == 0)
        )
    return int(positive > negative)


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tn = sum(row["label"] == 0 and row["predicted_label"] == 0 for row in rows)
    fp = sum(row["label"] == 0 and row["predicted_label"] == 1 for row in rows)
    fn = sum(row["label"] == 1 and row["predicted_label"] == 0 for row in rows)
    tp = sum(row["label"] == 1 and row["predicted_label"] == 1 for row in rows)
    f1_0 = _safe_div(2 * tn, 2 * tn + fp + fn)
    f1_1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    return {
        "n_total": len(rows),
        "accuracy": _safe_div(tn + tp, len(rows)),
        "macro_f1": (f1_0 + f1_1) / 2,
        "per_class_f1": {"0": f1_0, "1": f1_1},
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _parse_args(config: ScalarKnnConfig, argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=config.description)
    parser.add_argument("--input-jsonl", default=config.default_input)
    parser.add_argument("--index", default=config.default_index)
    parser.add_argument("--output-dir", default=config.default_output_dir)
    parser.add_argument("--smiles-field", default="drug")
    parser.add_argument("--label-field", default="Y")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.3)
    parser.add_argument("--fallback-label", type=int, choices=[0, 1], default=1)
    return parser.parse_args(argv)
