"""Audit and summarize paired Morgan-KNN versus matched train-label agent predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic
from tools.chembl_tool.paper_experiments.paired_binary_predictions import paired_binary_summary

from .audit import (
    binary_diagnostics,
    evidence_sources,
    retrieval_signature,
    run_has_prompt_identity_leak,
    stratum_diagnostics,
)
from .contract import DEFAULT_OUTPUT_ROOT, SCHEMA_VERSION, TaskSpec, agent_batch, replay_batch, selected_specs


def summarize_task(spec: TaskSpec, *, output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    knn_rows = _index_rows(_read_jsonl(spec.knn_predictions), "query_index")
    batch = agent_batch(root, spec)
    agent_rows = _index_rows(_read_jsonl(batch / "predictions.jsonl"), "query_index")
    if set(knn_rows) != set(agent_rows):
        raise ValueError(f"{spec.task} paired coverage mismatch")

    labels = []
    knn_predictions = []
    agent_predictions = []
    paired_rows = []
    vote_patterns = []
    retrieval_mismatches = 0
    prompt_identity_leak_runs = 0
    for query_index in sorted(knn_rows):
        knn = knn_rows[query_index]
        agent = agent_rows[query_index]
        if agent.get("status") != "ok" or agent.get("final_status") != "ok":
            raise ValueError(f"{spec.task} agent failure at query {query_index}")
        if str(agent.get("smiles") or "") != str(knn.get("drug") or ""):
            raise ValueError(f"{spec.task} molecule mismatch at query {query_index}")
        if int(agent["label"]) != int(knn["Y"]):
            raise ValueError(f"{spec.task} gold mismatch at query {query_index}")
        replay = _load_retrieval(root, spec, query_index, replay=True)
        retrieval = _load_retrieval(root, spec, query_index, replay=False)
        if retrieval_signature(retrieval) != retrieval_signature(replay):
            retrieval_mismatches += 1
        replay_neighbors = retrieval["groups"][0]["neighbors"]
        observed = [
            (
                int(row["train_index"]),
                int(row["train_label"]),
                float(neighbor["similarity"]),
            )
            for neighbor in replay_neighbors
            for row in neighbor["evidence_rows"]
        ]
        expected = [
            (int(row["train_index"]), int(row["Y"]), round(float(row["similarity"]), 6))
            for row in knn["neighbors"]
        ]
        if observed != expected:
            retrieval_mismatches += 1
        if evidence_sources(retrieval) != ["frozen_benchmark_train_label"] * 3:
            raise ValueError(f"{spec.task} unexpected evidence source at query {query_index}")
        prompt_identity_leak_runs += int(
            run_has_prompt_identity_leak(batch, agent, retrieval, read_jsonl=_read_jsonl)
        )

        label = int(knn["Y"])
        knn_prediction = int(knn["prediction"])
        agent_prediction = int(agent["pred_label"])
        neighbor_labels = [int(row["Y"]) for row in knn["neighbors"]]
        vote_pattern = "unanimous" if len(set(neighbor_labels)) == 1 else "split_2_to_1"
        labels.append(label)
        knn_predictions.append(knn_prediction)
        agent_predictions.append(agent_prediction)
        vote_patterns.append(vote_pattern)
        paired_rows.append(
            {
                "task": spec.task,
                "query_index": query_index,
                "Y": label,
                "knn_prediction": knn_prediction,
                "agent_prediction": agent_prediction,
                "knn_correct": knn_prediction == label,
                "agent_correct": agent_prediction == label,
                "prediction_flip": knn_prediction != agent_prediction,
                "vote_pattern": vote_pattern,
                "neighbors": knn["neighbors"],
            }
        )
    if retrieval_mismatches:
        raise ValueError(f"{spec.task} has {retrieval_mismatches} retrieval mismatches")
    if prompt_identity_leak_runs:
        raise ValueError(
            f"{spec.task} has {prompt_identity_leak_runs} prompt identity leak runs"
        )

    paired = paired_binary_summary(labels, knn_predictions, agent_predictions)
    metrics = json.loads((batch / "metrics.json").read_text(encoding="utf-8"))
    result = {
        "schema_version": SCHEMA_VERSION,
        "task": spec.task,
        "evaluation_subset": "valid",
        "n": len(labels),
        "retrieval_parity": {
            "mismatches": 0,
            "neighbors_per_query": 3,
            "same_neighbor_identity_order_similarity_and_train_label": True,
            "external_starling_records_visible": False,
            "evidence_source": "frozen_benchmark_train_label",
            "prompt_identity_audited_runs": len(labels),
            "prompt_identity_leak_runs": 0,
        },
        "knn": {
            "accuracy": paired["left_accuracy"],
            "macro_f1": paired["left_macro_f1"],
            **binary_diagnostics(labels, knn_predictions),
            "predictions": str(spec.knn_predictions),
        },
        "matched_train_label_agent": {
            "accuracy": paired["right_accuracy"],
            "macro_f1": paired["right_macro_f1"],
            **binary_diagnostics(labels, agent_predictions),
            "batch_confusion_matrix": metrics.get("confusion_matrix"),
            "predictions": str(batch / "predictions.jsonl"),
        },
        "agent_minus_knn": {
            "accuracy": paired["right_accuracy"] - paired["left_accuracy"],
            "macro_f1": paired["delta_macro_f1"],
            "macro_f1_bootstrap_95ci": paired["delta_macro_f1_bootstrap_95ci"],
            "prediction_flips": paired["prediction_flips"],
            "agent_only_correct": paired["right_only_correct"],
            "knn_only_correct": paired["left_only_correct"],
            "mcnemar_exact_p": paired["mcnemar_exact_p"],
        },
        "vote_pattern_analysis": {
            pattern: stratum_diagnostics(
                labels,
                knn_predictions,
                agent_predictions,
                vote_patterns,
                pattern,
            )
            for pattern in ("unanimous", "split_2_to_1")
        },
    }
    task_out = root / "analysis" / spec.task
    write_json_atomic(task_out / "result.json", result)
    write_jsonl_atomic(task_out / "paired_predictions.jsonl", paired_rows)
    return result


def summarize_all(
    *,
    output_root: str | Path,
    tasks: list[str] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    results = [summarize_task(spec, output_root=root) for spec in selected_specs(tasks)]
    summary = {"schema_version": SCHEMA_VERSION, "tasks": results}
    analysis = root / "analysis"
    write_json_atomic(analysis / "summary.json", summary)
    _write_metrics_tsv(analysis / "metrics.tsv", results)
    (analysis / "report.md").write_text(_report(results), encoding="utf-8")
    return summary


def _write_metrics_tsv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "task",
        "n",
        "knn_accuracy",
        "knn_macro_f1",
        "agent_accuracy",
        "agent_macro_f1",
        "delta_accuracy",
        "delta_macro_f1",
        "ci_low",
        "ci_high",
        "flips",
        "agent_only_correct",
        "knn_only_correct",
        "mcnemar_exact_p",
        "knn_recall_y0",
        "knn_recall_y1",
        "agent_recall_y0",
        "agent_recall_y1",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for result in results:
            delta = result["agent_minus_knn"]
            writer.writerow(
                {
                    "task": result["task"],
                    "n": result["n"],
                    "knn_accuracy": result["knn"]["accuracy"],
                    "knn_macro_f1": result["knn"]["macro_f1"],
                    "agent_accuracy": result["matched_train_label_agent"]["accuracy"],
                    "agent_macro_f1": result["matched_train_label_agent"]["macro_f1"],
                    "delta_accuracy": delta["accuracy"],
                    "delta_macro_f1": delta["macro_f1"],
                    "ci_low": delta["macro_f1_bootstrap_95ci"][0],
                    "ci_high": delta["macro_f1_bootstrap_95ci"][1],
                    "flips": delta["prediction_flips"],
                    "agent_only_correct": delta["agent_only_correct"],
                    "knn_only_correct": delta["knn_only_correct"],
                    "mcnemar_exact_p": delta["mcnemar_exact_p"],
                    "knn_recall_y0": result["knn"]["recall_y0"],
                    "knn_recall_y1": result["knn"]["recall_y1"],
                    "agent_recall_y0": result["matched_train_label_agent"]["recall_y0"],
                    "agent_recall_y1": result["matched_train_label_agent"]["recall_y1"],
                }
            )


def _report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Matched train-label direct agent vs Morgan KNN",
        "",
        "两种方法共享每个 valid query 的同一组 scaffold-train Morgan top-3 neighbors、顺序、相似度和训练标签。",
        "KNN 使用未加权多数票；agent 仅额外使用现有 identity-blind 分子属性/比较工具与 reasoning，",
        "不读取任何外部 Starling evidence record。Formal test 未运行。",
        "",
        "| task | n | KNN acc / macro-F1 | matched agent acc / macro-F1 | agent-KNN macro-F1 (95% CI) | agent-only / KNN-only |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        delta = result["agent_minus_knn"]
        low, high = delta["macro_f1_bootstrap_95ci"]
        lines.append(
            f"| {result['task']} | {result['n']} | "
            f"{result['knn']['accuracy']:.4f} / {result['knn']['macro_f1']:.4f} | "
            f"{result['matched_train_label_agent']['accuracy']:.4f} / "
            f"{result['matched_train_label_agent']['macro_f1']:.4f} | "
            f"{delta['macro_f1']:+.4f} [{low:+.4f},{high:+.4f}] | "
            f"{delta['agent_only_correct']} / {delta['knn_only_correct']} |"
        )
    lines.extend(["", "## 类别与投票强度诊断", ""])
    for result in results:
        unanimous = result["vote_pattern_analysis"]["unanimous"]
        split = result["vote_pattern_analysis"]["split_2_to_1"]
        lines.extend(
            [
                f"- `{result['task']}` 类别召回：KNN Y=0/Y=1 "
                f"{result['knn']['recall_y0']:.4f}/{result['knn']['recall_y1']:.4f}；agent "
                f"{result['matched_train_label_agent']['recall_y0']:.4f}/"
                f"{result['matched_train_label_agent']['recall_y1']:.4f}。",
                f"- 邻居全票一致时 agent 改票 {unanimous['prediction_flips']}/{unanimous['n']}，"
                f"其中 1→0/0→1 为 {unanimous['knn_y1_to_agent_y0']}/"
                f"{unanimous['knn_y0_to_agent_y1']}，"
                f"救回/损坏 {unanimous['agent_only_correct']}/{unanimous['knn_only_correct']}；"
                f"2:1 分裂时改票 {split['prediction_flips']}/{split['n']}，"
                f"其中 1→0/0→1 为 {split['knn_y1_to_agent_y0']}/"
                f"{split['knn_y0_to_agent_y1']}，"
                f"救回/损坏 {split['agent_only_correct']}/{split['knn_only_correct']}。",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _load_retrieval(
    root: Path,
    spec: TaskSpec,
    query_index: int,
    *,
    replay: bool,
) -> dict[str, Any]:
    batch = replay_batch(root, spec) if replay else agent_batch(root, spec)
    run = batch / "runs" / f"{batch.name}_idx{query_index:05d}" / "retrieval.json"
    return json.loads(run.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _index_rows(rows: list[dict[str, Any]], key: str) -> dict[int, dict[str, Any]]:
    result = {int(row[key]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tasks", nargs="*", default=None)
    args = parser.parse_args(argv)
    result = summarize_all(output_root=args.output_root, tasks=args.tasks)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
