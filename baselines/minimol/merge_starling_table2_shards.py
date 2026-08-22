"""Merge disjoint repetition shards from the Starling Table 2 runner."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from baselines.minimol.starling_table2_data import PAPER_TABLE2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_metrics", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _stable_contract(result: dict[str, object]) -> dict[str, object]:
    args = result["args"]
    return {
        "type": result["type"],
        "task": result["task"],
        "data": result["data"],
        "minimol": result["minimol"],
        "train_label_policy": args["train_label_policy"],
        "extraction_weight": args.get("extraction_weight", "none"),
        "evaluation_extraction_weight": args.get("evaluation_extraction_weight", "none"),
        "epochs": args["epochs"],
        "repetitions": args["repetitions"],
        "ensemble_size": args["ensemble_size"],
        "dropout": args["dropout"],
        "weight_decay": args["weight_decay"],
        "warmup": args["warmup"],
        "train_batch_size": args["train_batch_size"],
        "eval_batch_size": args["eval_batch_size"],
    }


def _merge_predictions(
    loaded: list[tuple[Path, dict[str, object]]],
    output_dir: Path,
    condition: str,
    evaluation: str,
) -> None:
    template: list[dict[str, object]] | None = None
    weighted_scores: np.ndarray | None = None
    total_weight = 0
    for metrics_path, result in loaded:
        if condition not in result["conditions"]:
            continue
        rows = [
            json.loads(line)
            for line in (metrics_path.parent / f"{condition}_{evaluation}_predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        identity = [
            (row["canonical_smiles"], row["target"], row["raw_label"])
            for row in rows
        ]
        if template is None:
            template = rows
            expected_identity = identity
        elif identity != expected_identity:
            raise ValueError(f"Prediction rows differ across shards for {condition}/{evaluation}")
        weight = len(result["conditions"][condition]["repetitions"])
        scores = np.asarray([row["score"] for row in rows], dtype=float)
        weighted_scores = scores * weight if weighted_scores is None else weighted_scores + scores * weight
        total_weight += weight

    if template is None or weighted_scores is None or total_weight == 0:
        raise ValueError(f"Missing predictions for {condition}/{evaluation}")
    with (output_dir / f"{condition}_{evaluation}_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row, score in zip(template, weighted_scores / total_weight, strict=True):
            merged = {**row, "score": float(score)}
            handle.write(json.dumps(merged) + "\n")


def merge_shards(metrics_paths: list[Path], output_dir: Path) -> dict[str, object]:
    loaded = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in metrics_paths]
    if not loaded:
        raise ValueError("At least one shard is required")
    contract = _stable_contract(loaded[0][1])
    for path, result in loaded[1:]:
        if _stable_contract(result) != contract:
            raise ValueError(f"Shard contract mismatch: {path}")

    expected_repetitions = set(range(1, int(contract["repetitions"]) + 1))
    condition_names = {name for _, result in loaded for name in result["conditions"]}
    if condition_names != {"tdc_only", "augmented"}:
        raise ValueError(f"Incomplete condition set: {sorted(condition_names)}")

    conditions: dict[str, object] = {}
    for condition in sorted(condition_names):
        repetitions = [
            repetition
            for _, result in loaded
            if condition in result["conditions"]
            for repetition in result["conditions"][condition]["repetitions"]
        ]
        repetition_ids = [int(row["repetition"]) for row in repetitions]
        if len(repetition_ids) != len(set(repetition_ids)) or set(repetition_ids) != expected_repetitions:
            raise ValueError(
                f"{condition} repetitions must cover {sorted(expected_repetitions)} exactly; "
                f"got {sorted(repetition_ids)}"
            )
        repetitions.sort(key=lambda row: int(row["repetition"]))
        summary = {}
        for evaluation in ("tdc_test", "literature_test"):
            values = [float(row["metrics"][evaluation]) for row in repetitions]
            summary[evaluation] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }
        conditions[condition] = {
            "summary": summary,
            "repetitions": repetitions,
            "sample_weight": loaded[0][1]["conditions"][condition].get("sample_weight"),
        }

    task_slug = loaded[0][1]["task"]["slug"]
    observed = {
        "tdc_test": conditions["tdc_only"]["summary"]["tdc_test"]["mean"],
        "tdc_test_augmented": conditions["augmented"]["summary"]["tdc_test"]["mean"],
        "literature_test": conditions["tdc_only"]["summary"]["literature_test"]["mean"],
        "literature_test_augmented": conditions["augmented"]["summary"]["literature_test"]["mean"],
    }
    comparison = {
        key: {
            "paper": paper_value,
            "observed": float(observed[key]),
            "observed_minus_paper": float(observed[key] - paper_value),
        }
        for key, paper_value in PAPER_TABLE2[task_slug].items()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for condition in condition_names:
        for evaluation in ("tdc_test", "literature_test"):
            _merge_predictions(loaded, output_dir, condition, evaluation)

    merged = deepcopy(loaded[0][1])
    merged["type"] = "starling_table2_minimol_reproduction.merged.v1"
    merged["repetition_indices"] = sorted(expected_repetitions)
    merged["merged_from"] = [str(path) for path, _ in loaded]
    merged["conditions"] = conditions
    merged["paper_comparison"] = comparison
    (output_dir / "metrics.json").write_text(
        json.dumps(merged, indent=2) + "\n",
        encoding="utf-8",
    )
    return merged


def main() -> None:
    args = parse_args()
    merged = merge_shards(args.shard_metrics, args.output_dir)
    print(json.dumps(merged["paper_comparison"], indent=2))
    print(f"[table2] wrote {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
