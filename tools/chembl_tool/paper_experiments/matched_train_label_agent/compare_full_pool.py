"""Compare train-only label direct against the earlier full-Starling direct runs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
from typing import Any

from tools.chembl_tool.common.json_utils import write_json_atomic
from tools.chembl_tool.paper_experiments.paired_binary_predictions import paired_binary_summary

from .contract import DEFAULT_OUTPUT_ROOT, SCHEMA_VERSION, agent_batch, selected_specs


def compare_task(task: str, *, output_root: str | Path) -> dict[str, Any]:
    spec = selected_specs([task])[0]
    if spec.full_pool_direct_batch is None:
        raise ValueError(f"{task} has no frozen full-pool direct batch")
    train_batch = agent_batch(output_root, spec)
    full_batch = spec.full_pool_direct_batch
    train_rows = _index_rows(_read_jsonl(train_batch / "predictions.jsonl"))
    full_rows = _index_rows(_read_jsonl(full_batch / "predictions.jsonl"))
    if set(train_rows) != set(full_rows):
        raise ValueError(f"{task} paired coverage mismatch")

    labels: list[int] = []
    full_predictions: list[int] = []
    train_predictions: list[int] = []
    retrievals = {"full_starling_pool": [], "train_only_labels": []}
    for query_index in sorted(train_rows):
        train = train_rows[query_index]
        full = full_rows[query_index]
        if str(train["smiles"]) != str(full["smiles"]):
            raise ValueError(f"{task} molecule mismatch at query {query_index}")
        if int(train["label"]) != int(full["label"]):
            raise ValueError(f"{task} gold mismatch at query {query_index}")
        if train.get("status") != "ok" or full.get("status") != "ok":
            raise ValueError(f"{task} non-ok prediction at query {query_index}")
        labels.append(int(train["label"]))
        full_predictions.append(int(full["pred_label"]))
        train_predictions.append(int(train["pred_label"]))
        retrievals["full_starling_pool"].append(_retrieval(full_batch, full))
        retrievals["train_only_labels"].append(_retrieval(train_batch, train))

    paired = paired_binary_summary(labels, full_predictions, train_predictions)
    return {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "evaluation_subset": "valid",
        "n": len(labels),
        "full_starling_pool_direct": {
            "accuracy": paired["left_accuracy"],
            "macro_f1": paired["left_macro_f1"],
            "batch": str(full_batch),
            "retrieval": _retrieval_summary(retrievals["full_starling_pool"]),
        },
        "train_only_label_direct": {
            "accuracy": paired["right_accuracy"],
            "macro_f1": paired["right_macro_f1"],
            "batch": str(train_batch),
            "retrieval": _retrieval_summary(retrievals["train_only_labels"]),
        },
        "train_only_minus_full_pool": {
            "accuracy": paired["right_accuracy"] - paired["left_accuracy"],
            "macro_f1": paired["delta_macro_f1"],
            "macro_f1_bootstrap_95ci": paired["delta_macro_f1_bootstrap_95ci"],
            "prediction_flips": paired["prediction_flips"],
            "train_only_correct_full_pool_wrong": paired["right_only_correct"],
            "full_pool_correct_train_only_wrong": paired["left_only_correct"],
            "mcnemar_exact_p": paired["mcnemar_exact_p"],
        },
    }


def compare_all(*, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(output_root)
    results = [compare_task(spec.task, output_root=root) for spec in selected_specs()]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "comparison_scope": (
            "same valid molecules and model family; retrieval pool, evidence representation, "
            "and neighbor threshold differ"
        ),
        "tasks": results,
    }
    output = root / "analysis" / "full_pool_comparison"
    write_json_atomic(output / "summary.json", summary)
    _write_tsv(output / "metrics.tsv", results)
    (output / "report.md").write_text(_report(results), encoding="utf-8")
    return summary


def _retrieval_summary(retrievals: list[dict[str, Any]]) -> dict[str, Any]:
    counts: list[int] = []
    top_similarities: list[float] = []
    all_similarities: list[float] = []
    for retrieval in retrievals:
        similarities = [
            float(neighbor["similarity"])
            for group in retrieval.get("groups") or []
            for neighbor in group.get("neighbors") or []
        ]
        counts.append(len(similarities))
        if similarities:
            top_similarities.append(max(similarities))
            all_similarities.extend(similarities)
    return {
        "queries": len(retrievals),
        "neighbor_count_distribution": {
            str(key): value for key, value in sorted(Counter(counts).items())
        },
        "coverage": sum(count > 0 for count in counts) / len(counts),
        "mean_neighbors": statistics.mean(counts),
        "median_top1_similarity": statistics.median(top_similarities),
        "median_all_neighbor_similarity": statistics.median(all_similarities),
    }


def _report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Full-Starling pool direct vs train-only label direct",
        "",
        "两侧使用同一 current scaffold-valid cohort 和 GPT-OSS-120B identity-blind pipeline。",
        "但该比较同时改变 retrieval pool、neighbor similarity threshold 和 evidence representation，",
        "因此只能估计 full-pool 方案的整体净价值，不能把差值纯归因于数据数量。",
        "",
        "| task | full pool acc / macro-F1 | train-only acc / macro-F1 | train-full macro-F1 (95% CI) | train-only / full-only correct |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        full = result["full_starling_pool_direct"]
        train = result["train_only_label_direct"]
        delta = result["train_only_minus_full_pool"]
        low, high = delta["macro_f1_bootstrap_95ci"]
        lines.append(
            f"| {result['task']} | {full['accuracy']:.4f} / {full['macro_f1']:.4f} | "
            f"{train['accuracy']:.4f} / {train['macro_f1']:.4f} | "
            f"{delta['macro_f1']:+.4f} [{low:+.4f},{high:+.4f}] | "
            f"{delta['train_only_correct_full_pool_wrong']} / "
            f"{delta['full_pool_correct_train_only_wrong']} |"
        )
    lines.extend(
        [
            "",
            "| task | full-pool coverage / median top-1 sim | train-only coverage / median top-1 sim |",
            "|---|---:|---:|",
        ]
    )
    for result in results:
        full = result["full_starling_pool_direct"]["retrieval"]
        train = result["train_only_label_direct"]["retrieval"]
        lines.append(
            f"| {result['task']} | {full['coverage']:.3f} / "
            f"{full['median_top1_similarity']:.3f} | {train['coverage']:.3f} / "
            f"{train['median_top1_similarity']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _write_tsv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task", "n", "full_accuracy", "full_macro_f1", "train_accuracy",
        "train_macro_f1", "train_minus_full_accuracy", "train_minus_full_macro_f1",
        "ci_low", "ci_high", "train_only_correct", "full_pool_only_correct",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for result in results:
            full = result["full_starling_pool_direct"]
            train = result["train_only_label_direct"]
            delta = result["train_only_minus_full_pool"]
            writer.writerow({
                "task": result["task"], "n": result["n"],
                "full_accuracy": full["accuracy"], "full_macro_f1": full["macro_f1"],
                "train_accuracy": train["accuracy"], "train_macro_f1": train["macro_f1"],
                "train_minus_full_accuracy": delta["accuracy"],
                "train_minus_full_macro_f1": delta["macro_f1"],
                "ci_low": delta["macro_f1_bootstrap_95ci"][0],
                "ci_high": delta["macro_f1_bootstrap_95ci"][1],
                "train_only_correct": delta["train_only_correct_full_pool_wrong"],
                "full_pool_only_correct": delta["full_pool_correct_train_only_wrong"],
            })


def _retrieval(batch: Path, prediction: dict[str, Any]) -> dict[str, Any]:
    return json.loads(
        (batch / "runs" / str(prediction["run_id"]) / "retrieval.json").read_text(
            encoding="utf-8"
        )
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _index_rows(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result = {int(row["query_index"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate query_index")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(compare_all(output_root=args.output_root), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
