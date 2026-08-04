"""Evaluate a train-label Morgan-fingerprint KNN classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from tools.chembl_tool.common.neighbor_selection import (
    NEIGHBOR_SELECTORS,
    SIMILARITY_SELECTOR,
    NeighborCandidate,
    select_neighbor_candidates,
    selector_metadata,
)


FP_RADIUS = 2
FP_BITS = 2048


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    train_path = args.data_dir / "train.jsonl"
    test_path = args.data_dir / f"{args.evaluation_split}.jsonl"
    train = _read_split(train_path)
    test = _read_split(test_path)
    if args.k <= 0:
        raise ValueError("--k must be positive")
    if len(train) < args.k:
        raise ValueError(f"Training set has {len(train)} rows, fewer than k={args.k}")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FP_RADIUS,
        fpSize=FP_BITS,
        includeChirality=False,
        useBondTypes=True,
    )
    train_fps = [
        _fingerprint(row["drug"], generator, source=train_path, index=index)
        for index, row in enumerate(train)
    ]

    predictions = []
    for query_index, row in enumerate(test):
        query_fp = _fingerprint(row["drug"], generator, source=test_path, index=query_index)
        similarities = DataStructs.BulkTanimotoSimilarity(query_fp, train_fps)
        candidates = [
            NeighborCandidate(
                molecule_index=train_index,
                molecule_id=f"{train_index:012d}",
                similarity=float(similarity),
            )
            for train_index, similarity in enumerate(similarities)
            if args.min_similarity is None or similarity >= args.min_similarity
        ]
        if len(candidates) < args.k:
            predictions.append(
                {
                    "query_index": query_index,
                    "drug": row["drug"],
                    "Y": row["Y"],
                    "prediction": None,
                    "score": None,
                    "correct": None,
                    "status": "insufficient_neighbors",
                    "n_eligible_neighbors": len(candidates),
                    "neighbors": [],
                }
            )
            continue
        selected = select_neighbor_candidates(
            candidates,
            query_fingerprint=query_fp,
            candidate_fingerprints=train_fps,
            top_k=args.k,
            selector=args.neighbor_selector,
        )
        neighbors = [
            {
                "train_index": candidate.molecule_index,
                "drug": train[candidate.molecule_index]["drug"],
                "Y": train[candidate.molecule_index]["Y"],
                "similarity": candidate.similarity,
            }
            for candidate in selected
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
                "n_eligible_neighbors": len(candidates),
                "retrieval_diagnostics": _retrieval_diagnostics(
                    query_fp,
                    [train_fps[candidate.molecule_index] for candidate in selected],
                    selected,
                ),
                "neighbors": neighbors,
            }
        )

    evaluated_predictions = [row for row in predictions if row["status"] == "ok"]
    if not evaluated_predictions:
        raise ValueError("No test rows have enough eligible neighbors for evaluation")
    metrics = _metrics(evaluated_predictions)
    method = f"morgan_knn_k{args.k}"
    if args.neighbor_selector != SIMILARITY_SELECTOR:
        method += f"_{args.neighbor_selector}"
    if args.min_similarity is not None:
        threshold = str(args.min_similarity).replace(".", "p")
        method += f"_minsim{threshold}_supported"
    metrics.update(
        {
            "method": method,
            "evaluation_split": args.evaluation_split,
            "k": args.k,
            "vote": "unweighted_majority",
            "score": "positive_neighbor_fraction",
            "n_train": len(train),
            "n_test": len(test),
            "n_evaluation": len(test),
            "n_evaluated": len(evaluated_predictions),
            "n_skipped_insufficient_neighbors": len(test) - len(evaluated_predictions),
            "evaluation_coverage": len(evaluated_predictions) / len(test),
            "minimum_similarity": args.min_similarity,
            "neighbor_selector": selector_metadata(args.neighbor_selector),
            "retrieval_diagnostics": _mean_retrieval_diagnostics(evaluated_predictions),
            "fingerprint": {
                "type": "RDKit-Morgan",
                "radius": FP_RADIUS,
                "n_bits": FP_BITS,
                "use_features": False,
                "use_chirality": False,
                "use_bond_types": True,
            },
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        args.output_dir / f"{args.evaluation_split}_predictions.jsonl",
        predictions,
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "data_dir": str(args.data_dir),
                "train_path": str(train_path),
                "test_path": str(test_path),
                "evaluation_path": str(test_path),
                "evaluation_split": args.evaluation_split,
                "output_dir": str(args.output_dir),
                **{
                    key: metrics[key]
                    for key in (
                        "method",
                        "k",
                        "vote",
                        "score",
                        "minimum_similarity",
                        "neighbor_selector",
                        "fingerprint",
                    )
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


def _read_split(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                label = int(row["Y"])
                smiles = str(row["drug"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number} must contain drug and binary Y") from exc
            if label not in (0, 1):
                raise ValueError(f"{path}:{line_number} has non-binary Y={label}")
            rows.append({"drug": smiles, "Y": label})
    return rows


def _fingerprint(
    smiles: str,
    generator: rdFingerprintGenerator.FingerprintGenerator64,
    *,
    source: Path,
    index: int,
) -> DataStructs.ExplicitBitVect:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES at {source} row index {index}: {smiles!r}")
    return generator.GetFingerprint(molecule)


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tn = sum(row["Y"] == 0 and row["prediction"] == 0 for row in rows)
    fp = sum(row["Y"] == 0 and row["prediction"] == 1 for row in rows)
    fn = sum(row["Y"] == 1 and row["prediction"] == 0 for row in rows)
    tp = sum(row["Y"] == 1 and row["prediction"] == 1 for row in rows)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    positive_f1 = _safe_div(2 * precision * recall, precision + recall)
    negative_f1 = _safe_div(2 * tn, 2 * tn + fp + fn)
    labels = {int(row["Y"]) for row in rows}
    return {
        "accuracy": _safe_div(tn + tp, len(rows)),
        "macro_f1": (negative_f1 + positive_f1) / 2,
        "auroc": (
            _roc_auc([(int(row["Y"]), float(row["score"])) for row in rows])
            if labels == {0, 1}
            else None
        ),
        "positive_precision": precision,
        "positive_recall": recall,
        "positive_f1": positive_f1,
        "per_class_f1": {"0": negative_f1, "1": positive_f1},
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _retrieval_diagnostics(
    query_fingerprint: DataStructs.ExplicitBitVect,
    neighbor_fingerprints: list[DataStructs.ExplicitBitVect],
    selected: list[NeighborCandidate],
) -> dict[str, float]:
    query_bits = set(query_fingerprint.GetOnBits())
    covered_query_bits: set[int] = set()
    for fingerprint in neighbor_fingerprints:
        covered_query_bits.update(query_bits.intersection(fingerprint.GetOnBits()))
    pairwise_similarities = [
        float(DataStructs.TanimotoSimilarity(neighbor_fingerprints[left], neighbor_fingerprints[right]))
        for left in range(len(neighbor_fingerprints))
        for right in range(left + 1, len(neighbor_fingerprints))
    ]
    return {
        "query_feature_coverage": _safe_div(len(covered_query_bits), len(query_bits)),
        "mean_query_similarity": sum(candidate.similarity for candidate in selected) / len(selected),
        "mean_neighbor_pairwise_similarity": (
            sum(pairwise_similarities) / len(pairwise_similarities)
            if pairwise_similarities
            else 0.0
        ),
    }


def _mean_retrieval_diagnostics(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "query_feature_coverage",
        "mean_query_similarity",
        "mean_neighbor_pairwise_similarity",
    )
    return {
        key: sum(row["retrieval_diagnostics"][key] for row in rows) / len(rows)
        for key in keys
    }


def _roc_auc(pairs: list[tuple[int, float]]) -> float:
    ordered = sorted(pairs, key=lambda item: item[1])
    rank_sum_positive = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        rank_sum_positive += average_rank * sum(
            label == 1 for label, _ in ordered[index:end]
        )
        index = end
    n_positive = sum(label == 1 for label, _ in ordered)
    n_negative = len(ordered) - n_positive
    if not n_positive or not n_negative:
        raise ValueError("AUROC requires both binary classes")
    return (
        rank_sum_positive - n_positive * (n_positive + 1) / 2
    ) / (n_positive * n_negative)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation-split",
        choices=("valid", "test"),
        default="test",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--neighbor-selector",
        choices=NEIGHBOR_SELECTORS,
        default=SIMILARITY_SELECTOR,
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        help=(
            "Optional eligibility threshold. Test rows with fewer than k eligible train "
            "neighbors are retained in predictions with status=insufficient_neighbors "
            "and excluded from supported-cohort metrics."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
