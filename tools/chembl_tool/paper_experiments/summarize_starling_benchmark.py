"""Summarize random/scaffold Starling pipeline and supervised baseline metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .build_starling_benchmark_indices import BENCHMARK_SPLITS, paper_root_for_benchmark_split
from .starling_benchmark_matrix import experiments_for_starling_benchmark


DEFAULT_OUTPUT_DIR = Path("outputs/paper/starling_benchmark_results")
MINIMOL_ROOT = Path("outputs/baselines/minimol_starling")
STRUCTURE_KNN_ROOT = Path("outputs/baselines/structure_knn_starling")
MINIMOL_EMBEDDING_KNN_ROOT = Path(
    "outputs/baselines/minimol_embedding_knn_starling"
)
TASK_DATA_NAMES = {
    "bbb_martins": "BBB_Martins",
    "bioavailability_ma": "Bioavailability_Ma",
    "skin_reaction": "Skin_Reaction",
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows: list[dict[str, Any]] = []
    for split in args.splits or BENCHMARK_SPLITS:
        rows.extend(_pipeline_rows(split))
        rows.extend(_minimol_rows(split))
        rows.extend(_structure_knn_rows(split))
        rows.extend(_minimol_embedding_knn_rows(split))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_tsv(output_dir / "metrics.tsv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_report(rows), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "n_rows": len(rows)}, indent=2))
    return 0


def _pipeline_rows(split: str) -> list[dict[str, Any]]:
    paper_root = paper_root_for_benchmark_split(split)
    operational_root = paper_root / "runs_deployment_visible"
    parent_root = paper_root / "runs_deployment_visible_parent_disjoint"
    rows: list[dict[str, Any]] = []
    for experiment in experiments_for_starling_benchmark(split):
        policy = "operational" if experiment.mode == "none" else "parent_disjoint"
        root = operational_root if policy == "operational" else parent_root
        metrics_path = root / experiment.task / experiment.name / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        confusion = metrics.get("confusion_matrix") or {}
        rows.append(
            {
                "benchmark_split": split,
                "task": experiment.task,
                "method": experiment.name.split("__", 1)[1],
                "method_family": "molecular_evidence_agent",
                "source": experiment.source if experiment.mode != "none" else "none",
                "reasoning_mode": experiment.mode,
                "neighbor_identity_policy": policy,
                "n_test": metrics.get("n_evaluable"),
                "n_successful": metrics.get("n_successful"),
                "n_failed": metrics.get("n_failed_runs"),
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "auroc": "",
                "positive_precision": metrics.get("positive_class_precision"),
                "positive_recall": metrics.get("positive_class_recall"),
                "positive_f1": metrics.get("positive_class_f1"),
                "tn": confusion.get("tn"),
                "fp": confusion.get("fp"),
                "fn": confusion.get("fn"),
                "tp": confusion.get("tp"),
                "metrics_path": str(metrics_path),
            }
        )
    return rows


def _minimol_rows(split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, data_name in TASK_DATA_NAMES.items():
        metrics_path = MINIMOL_ROOT / data_name / split / "metrics.json"
        predictions_path = MINIMOL_ROOT / data_name / split / "test_predictions.jsonl"
        if not metrics_path.exists() or not predictions_path.exists():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = payload["test_metrics_fixed_0.5"]
        predictions = _read_jsonl(predictions_path)
        confusion = _confusion(predictions)
        positive = _positive_metrics(confusion)
        rows.append(
            {
                "benchmark_split": split,
                "task": task,
                "method": "minimol_train_all",
                "method_family": "minimol",
                "source": "train_jsonl",
                "reasoning_mode": "supervised_baseline",
                "neighbor_identity_policy": "",
                "n_test": len(predictions),
                "n_successful": len(predictions),
                "n_failed": 0,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "auroc": metrics["auroc"],
                "positive_precision": positive["precision"],
                "positive_recall": positive["recall"],
                "positive_f1": positive["f1"],
                **confusion,
                "metrics_path": str(metrics_path),
            }
        )
    return rows


def _structure_knn_rows(split: str) -> list[dict[str, Any]]:
    return _knn_rows(
        split,
        root=STRUCTURE_KNN_ROOT,
        expected_method="morgan_knn_k3",
        method_family="structure_knn",
    )


def _minimol_embedding_knn_rows(split: str) -> list[dict[str, Any]]:
    return _knn_rows(
        split,
        root=MINIMOL_EMBEDDING_KNN_ROOT,
        expected_method="minimol_embedding_cosine_knn_k3",
        method_family="minimol_embedding_knn",
    )


def _knn_rows(
    split: str,
    *,
    root: Path,
    expected_method: str,
    method_family: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, data_name in TASK_DATA_NAMES.items():
        metrics_path = root / data_name / split / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("method") != expected_method:
            raise ValueError(
                f"Unexpected KNN method in {metrics_path}: "
                f"{metrics.get('method')!r} != {expected_method!r}"
            )
        confusion = metrics["confusion_matrix"]
        rows.append(
            {
                "benchmark_split": split,
                "task": task,
                "method": expected_method,
                "method_family": method_family,
                "source": "train_jsonl",
                "reasoning_mode": "supervised_knn_baseline",
                "neighbor_identity_policy": "",
                "n_test": metrics["n_test"],
                "n_successful": metrics.get("n_evaluated", metrics["n_test"]),
                "n_failed": 0,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "auroc": metrics["auroc"],
                "positive_precision": metrics["positive_precision"],
                "positive_recall": metrics["positive_recall"],
                "positive_f1": metrics["positive_f1"],
                **confusion,
                "metrics_path": str(metrics_path),
            }
        )
    return rows


def _confusion(rows: list[dict[str, Any]]) -> dict[str, int]:
    output = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for row in rows:
        label = int(row["Y"])
        prediction = int(row["prediction"])
        key = ("t" if label == prediction else "f") + ("p" if prediction else "n")
        output[key] += 1
    return output


def _positive_metrics(confusion: dict[str, int]) -> dict[str, float]:
    precision = _safe_div(confusion["tp"], confusion["tp"] + confusion["fp"])
    recall = _safe_div(confusion["tp"], confusion["tp"] + confusion["fn"])
    return {
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Starling random/scaffold benchmark 结果",
        "",
        "Pipeline 指标使用 deployment-visible 主制度：none 采用 operational；所有 retrieval conditions "
        "采用 parent-disjoint。MiniMol 使用全部 train、固定 epoch 和 threshold=0.5；Morgan KNN "
        "与 MiniMol embedding cosine KNN 都只检索同 split 的 train labels，固定 k=3 并使用"
        "未加权多数票。",
        "",
        "| split | task | method | n | failed | accuracy | macro-F1 | AUROC | positive P/R/F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        auroc = "-" if row["auroc"] == "" else f"{float(row['auroc']):.4f}"
        lines.append(
            f"| {row['benchmark_split']} | {row['task']} | {row['method']} | "
            f"{row['n_test']} | {row['n_failed']} | {float(row['accuracy']):.4f} | "
            f"{float(row['macro_f1']):.4f} | {auroc} | "
            f"{float(row['positive_precision']):.4f}/"
            f"{float(row['positive_recall']):.4f}/"
            f"{float(row['positive_f1']):.4f} |"
        )
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="*", choices=BENCHMARK_SPLITS, default=[])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
