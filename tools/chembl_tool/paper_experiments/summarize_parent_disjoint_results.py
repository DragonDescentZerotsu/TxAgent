"""Summarize paired operational and parent-disjoint retrieval results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.molecule_identity import identity_from_record, normalize_molecule_identity
from tools.chembl_tool.common.retrieval_policy import MoleculeRelation, classify_molecule_relation
from tools.chembl_tool.paper_experiments.parent_disjoint_ablation import DEFAULT_OUTPUT
from tools.chembl_tool.paper_experiments.summarize_results import (
    mcnemar_exact_p,
    paired_bootstrap_delta_ci,
)


DEFAULT_OPERATIONAL_ROOT = Path("outputs/paper/molecular_evidence_agent/runs_deployment_visible")
DEFAULT_PARENT_DISJOINT_ROOT = Path(
    "outputs/paper/molecular_evidence_agent/runs_deployment_visible_parent_disjoint"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir)
    plan = _read_json(output_dir / "summary.json")
    operational_root = Path(args.operational_root)
    parent_root = Path(args.parent_disjoint_root)

    condition_rows: list[dict[str, Any]] = []
    flip_rows: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for experiment in plan.get("experiments") or []:
        task = str(experiment["task"])
        name = str(experiment["experiment"])
        operational_batch = operational_root / task / name
        parent_batch = parent_root / task / name
        if not (operational_batch / "metrics.json").exists() or not (parent_batch / "metrics.json").exists():
            incomplete.append(name)
            continue
        condition, flips = summarize_condition(
            experiment,
            operational_batch=operational_batch,
            parent_batch=parent_batch,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        condition_rows.append(condition)
        flip_rows.extend(flips)

    _add_holm_adjusted_p(condition_rows)

    _write_tsv(output_dir / "condition_results.tsv", condition_rows)
    _write_tsv(output_dir / "prediction_flips.tsv", flip_rows)
    payload = {
        "operational_root": str(operational_root),
        "parent_disjoint_root": str(parent_root),
        "n_planned_conditions": len(plan.get("experiments") or []),
        "n_complete_conditions": len(condition_rows),
        "incomplete_conditions": incomplete,
        "n_sample_conditions": sum(row["n_total"] for row in condition_rows),
        "n_retrieval_changed": sum(row["n_retrieval_changed"] for row in condition_rows),
        "n_sample_reused": sum(row["n_sample_reused"] for row in condition_rows),
        "n_queries_with_same_parent": plan.get("n_queries_with_same_parent"),
        "n_groups_with_same_parent": plan.get("n_groups_with_same_parent"),
        "n_retrieved_neighbor_slots": plan.get("n_retrieved_neighbor_slots"),
        "n_same_parent_neighbor_slots": plan.get("n_same_parent_neighbor_slots"),
        "same_parent_neighbor_slot_fraction": plan.get("same_parent_neighbor_slot_fraction"),
        "n_same_parent_unique_neighbors_summed_per_query": plan.get(
            "n_same_parent_unique_neighbors_summed_per_query"
        ),
        "n_same_parent_rank1_slots": plan.get("n_same_parent_rank1_slots"),
        "n_prediction_flips": len(flip_rows),
        "n_corrected": sum(row["flip_effect"] == "corrected" for row in flip_rows),
        "n_broken": sum(row["flip_effect"] == "broken" for row in flip_rows),
        "n_parent_policy_conflicts": sum(row["n_parent_policy_conflicts"] for row in condition_rows),
        "n_below_similarity_threshold": sum(
            row["n_below_similarity_threshold"] for row in condition_rows
        ),
        "n_full_runs_reused": sum(row["n_full_runs_reused"] for row in condition_rows),
        "n_changed_run_group_branches": sum(
            row["n_changed_run_group_branches"] for row in condition_rows
        ),
        "n_changed_run_group_branches_reused": sum(
            row["n_changed_run_group_branches_reused"] for row in condition_rows
        ),
        "operational_retained_neighbors": sum(
            row["operational_retained_neighbors"] for row in condition_rows
        ),
        "parent_disjoint_retained_neighbors": sum(
            row["parent_disjoint_retained_neighbors"] for row in condition_rows
        ),
        "conditions": condition_rows,
    }
    (output_dir / "result_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "result_report.md").write_text(_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 1 if incomplete else 0


def summarize_condition(
    experiment: dict[str, Any],
    *,
    operational_batch: Path,
    parent_batch: Path,
    bootstrap_replicates: int = 10_000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    operational_metrics = _read_json(operational_batch / "metrics.json")
    parent_metrics = _read_json(parent_batch / "metrics.json")
    none_metrics_path = operational_batch.parent / f"{experiment['task']}__none" / "metrics.json"
    none_metrics = _read_json(none_metrics_path) if none_metrics_path.exists() else {}
    operational_predictions = _predictions_by_index(operational_batch / "predictions.jsonl")
    parent_predictions = _predictions_by_index(parent_batch / "predictions.jsonl")
    if operational_predictions.keys() != parent_predictions.keys():
        raise ValueError(f"Prediction index mismatch for {experiment['experiment']}")

    changed_indices = {int(index) for index in experiment.get("changed_indices") or []}
    flips = []
    unchanged_input_flips = []
    for query_index in sorted(operational_predictions):
        operational = operational_predictions[query_index]
        parent = parent_predictions[query_index]
        old_label = operational.get("pred_label")
        new_label = parent.get("pred_label")
        if old_label == new_label:
            continue
        true_label = parent.get("label")
        old_correct = old_label == true_label
        new_correct = new_label == true_label
        if new_correct and not old_correct:
            effect = "corrected"
        elif old_correct and not new_correct:
            effect = "broken"
        else:
            effect = "changed_without_correctness_change"
        row = {
            "experiment": experiment["experiment"],
            "task": experiment["task"],
            "source": experiment["source"],
            "mode": experiment["mode"],
            "query_index": query_index,
            "retrieval_changed": query_index in changed_indices,
            "true_label": true_label,
            "operational_pred_label": old_label,
            "parent_disjoint_pred_label": new_label,
            "operational_correct": old_correct,
            "parent_disjoint_correct": new_correct,
            "flip_effect": effect,
            "smiles": parent.get("smiles") or operational.get("smiles") or "",
        }
        flips.append(row)
        if query_index not in changed_indices:
            unchanged_input_flips.append(query_index)
    if unchanged_input_flips:
        raise ValueError(
            f"Predictions changed despite identical retrieval input in {experiment['experiment']}: "
            f"{unchanged_input_flips[:10]}"
        )

    common = sorted(operational_predictions)
    labels = [int(operational_predictions[index]["label"]) for index in common]
    operational_pred = [int(operational_predictions[index]["pred_label"]) for index in common]
    parent_pred = [int(parent_predictions[index]["pred_label"]) for index in common]
    delta_ci_low, delta_ci_high = paired_bootstrap_delta_ci(
        labels, operational_pred, parent_pred, bootstrap_replicates
    )

    operational_cm = operational_metrics.get("confusion_matrix") or {}
    parent_cm = parent_metrics.get("confusion_matrix") or {}
    operational_coverage = _retrieval_coverage(operational_batch)
    identity_audit = _audit_parent_disjoint_batch(parent_batch)
    reuse_audit = _audit_reuse(parent_batch, changed_indices)
    condition = {
        "experiment": experiment["experiment"],
        "task": experiment["task"],
        "source": experiment["source"],
        "mode": experiment["mode"],
        "n_total": experiment["n_total"],
        "n_retrieval_changed": experiment["n_changed"],
        "n_sample_reused": experiment["n_reused"],
        "n_queries_with_same_parent": experiment.get("n_queries_with_same_parent"),
        "n_groups_with_same_parent": experiment.get("n_groups_with_same_parent"),
        "n_retrieved_neighbor_slots": experiment.get("n_retrieved_neighbor_slots"),
        "n_same_parent_neighbor_slots": experiment.get("n_same_parent_neighbor_slots"),
        "same_parent_neighbor_slot_fraction": experiment.get("same_parent_neighbor_slot_fraction"),
        "n_same_parent_unique_neighbors_summed_per_query": experiment.get(
            "n_same_parent_unique_neighbors_summed_per_query"
        ),
        "n_same_parent_rank1_slots": experiment.get("n_same_parent_rank1_slots"),
        "n_prediction_flips": len(flips),
        "n_corrected": sum(row["flip_effect"] == "corrected" for row in flips),
        "n_broken": sum(row["flip_effect"] == "broken" for row in flips),
        "operational_macro_f1": operational_metrics["macro_f1"],
        "parent_disjoint_macro_f1": parent_metrics["macro_f1"],
        "macro_f1_delta": round(parent_metrics["macro_f1"] - operational_metrics["macro_f1"], 6),
        "macro_f1_delta_ci_low": delta_ci_low,
        "macro_f1_delta_ci_high": delta_ci_high,
        "mcnemar_exact_p": mcnemar_exact_p(
            sum(row["flip_effect"] == "broken" for row in flips),
            sum(row["flip_effect"] == "corrected" for row in flips),
        ),
        "none_macro_f1": none_metrics.get("macro_f1"),
        "parent_disjoint_minus_none_macro_f1": (
            round(parent_metrics["macro_f1"] - none_metrics["macro_f1"], 6)
            if none_metrics.get("macro_f1") is not None
            else None
        ),
        "operational_accuracy": operational_metrics["accuracy"],
        "parent_disjoint_accuracy": parent_metrics["accuracy"],
        "accuracy_delta": round(parent_metrics["accuracy"] - operational_metrics["accuracy"], 6),
        "operational_tn": operational_cm.get("tn"),
        "operational_fp": operational_cm.get("fp"),
        "operational_fn": operational_cm.get("fn"),
        "operational_tp": operational_cm.get("tp"),
        "parent_disjoint_tn": parent_cm.get("tn"),
        "parent_disjoint_fp": parent_cm.get("fp"),
        "parent_disjoint_fn": parent_cm.get("fn"),
        "parent_disjoint_tp": parent_cm.get("tp"),
        "operational_retained_neighbors": operational_coverage["n_retained_neighbors"],
        "operational_samples_with_neighbors": operational_coverage["n_samples_with_neighbors"],
        **identity_audit,
        **reuse_audit,
        "operational_failed": operational_metrics.get("n_failed_runs"),
        "parent_disjoint_failed": parent_metrics.get("n_failed_runs"),
    }
    return condition, flips


def _audit_parent_disjoint_batch(batch: Path) -> dict[str, int]:
    forbidden = {
        MoleculeRelation.EXACT_RECORD,
        MoleculeRelation.SAME_CONNECTIVITY_VARIANT,
        MoleculeRelation.SAME_PARENT,
    }
    n_neighbors = 0
    n_samples_with_neighbors = 0
    n_identity_conflicts = 0
    n_below_similarity_threshold = 0
    for retrieval_path in sorted((batch / "runs").glob("*/retrieval.json")):
        run_manifest = _read_json(retrieval_path.parent / "manifest.json")
        min_similarity = float(run_manifest.get("min_similarity") or 0.0)
        retrieval = _read_json(retrieval_path)
        query_record = retrieval.get("query") or {}
        query = normalize_molecule_identity(
            str(query_record.get("input_smiles") or query_record.get("canonical_smiles") or "")
        )
        sample_has_neighbors = False
        for group in retrieval.get("groups") or []:
            for neighbor in group.get("neighbors") or []:
                sample_has_neighbors = True
                n_neighbors += 1
                relation = classify_molecule_relation(query, identity_from_record(neighbor))
                n_identity_conflicts += relation in forbidden
                similarity = neighbor.get("similarity")
                if similarity is not None and float(similarity) + 1e-12 < min_similarity:
                    n_below_similarity_threshold += 1
        n_samples_with_neighbors += sample_has_neighbors
    return {
        "parent_disjoint_retained_neighbors": n_neighbors,
        "parent_disjoint_samples_with_neighbors": n_samples_with_neighbors,
        "n_parent_policy_conflicts": n_identity_conflicts,
        "n_below_similarity_threshold": n_below_similarity_threshold,
    }


def _retrieval_coverage(batch: Path) -> dict[str, int]:
    n_neighbors = 0
    n_samples_with_neighbors = 0
    for retrieval_path in sorted((batch / "runs").glob("*/retrieval.json")):
        retrieval = _read_json(retrieval_path)
        sample_neighbors = sum(
            len(group.get("neighbors") or []) for group in retrieval.get("groups") or []
        )
        n_neighbors += sample_neighbors
        n_samples_with_neighbors += sample_neighbors > 0
    return {
        "n_retained_neighbors": n_neighbors,
        "n_samples_with_neighbors": n_samples_with_neighbors,
    }


def _audit_reuse(batch: Path, changed_indices: set[int]) -> dict[str, int]:
    n_full_runs_reused = 0
    n_group_branches = 0
    n_group_branches_reused = 0
    for run_dir in sorted((batch / "runs").glob("*_idx*")):
        query_index = int(run_dir.name.rsplit("_idx", 1)[1])
        if query_index not in changed_indices:
            n_full_runs_reused += (run_dir / "reuse.json").exists()
            continue
        group_path = run_dir / "group_reasoning_outputs.jsonl"
        if not group_path.exists():
            continue
        with group_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                group = json.loads(line)
                n_group_branches += 1
                n_group_branches_reused += bool(group.get("reused_from"))
    return {
        "n_full_runs_reused": n_full_runs_reused,
        "n_changed_run_group_branches": n_group_branches,
        "n_changed_run_group_branches_reused": n_group_branches_reused,
    }


def _add_holm_adjusted_p(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: float(item[1]["mcnemar_exact_p"]))
    running_max = 0.0
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, float(row["mcnemar_exact_p"]) * (len(rows) - rank))
        running_max = max(running_max, adjusted)
        rows[index]["mcnemar_holm_p"] = running_max


def _predictions_by_index(path: Path) -> dict[int, dict[str, Any]]:
    predictions = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            predictions[int(record["query_index"])] = record
    return predictions


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["experiment"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Parent-disjoint 配对实验结果",
        "",
        f"- 计划条件：{summary['n_planned_conditions']}",
        f"- 已完成条件：{summary['n_complete_conditions']}",
        f"- sample-condition 总数：{summary['n_sample_conditions']}",
        f"- retrieval 输入变化 / 整条 run 复用：{summary['n_retrieval_changed']} / "
        f"{summary['n_sample_reused']}",
        f"- same-parent query-condition / group：{summary['n_queries_with_same_parent']} / "
        f"{summary['n_groups_with_same_parent']}",
        f"- same-parent neighbor slots / 全部 slots：{summary['n_same_parent_neighbor_slots']} / "
        f"{summary['n_retrieved_neighbor_slots']} "
        f"({_format_percent(summary['same_parent_neighbor_slot_fraction'])})",
        f"- rank-1 same-parent slots：{summary['n_same_parent_rank1_slots']}",
        f"- prediction flips：{summary['n_prediction_flips']}",
        f"- 修正 / 破坏：{summary['n_corrected']} / {summary['n_broken']}",
        f"- retained neighbor identity 冲突：{summary['n_parent_policy_conflicts']}",
        f"- 低于 similarity threshold 的补位：{summary['n_below_similarity_threshold']}",
        f"- 整个 sample run 直接复用：{summary['n_full_runs_reused']}",
        f"- 变化样本的 group branch 复用：{summary['n_changed_run_group_branches_reused']} / "
        f"{summary['n_changed_run_group_branches']}",
        f"- retained neighbor 总数（operational / parent-disjoint）："
        f"{summary['operational_retained_neighbors']} / {summary['parent_disjoint_retained_neighbors']}",
    ]
    if summary["incomplete_conditions"]:
        lines.append(f"- 尚未完成：{', '.join(summary['incomplete_conditions'])}")
    lines.extend(
        [
            "",
            "| 实验 | retrieval 变化 | 预测翻转 | 修正 | 破坏 | operational F1 | parent-disjoint F1 | parent-operational | parent-none |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["conditions"]:
        lines.append(
            f"| {row['experiment']} | {row['n_retrieval_changed']} | {row['n_prediction_flips']} | "
            f"{row['n_corrected']} | {row['n_broken']} | {row['operational_macro_f1']:.4f} | "
            f"{row['parent_disjoint_macro_f1']:.4f} | {row['macro_f1_delta']:+.4f} | "
            f"{_format_delta(row['parent_disjoint_minus_none_macro_f1'])} |"
        )
    lines.extend(
        [
            "",
            "## 配对不确定性",
            "",
            "| 实验 | F1 差值 | paired bootstrap 95% CI | McNemar exact p | Holm p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["conditions"]:
        lines.append(
            f"| {row['experiment']} | {row['macro_f1_delta']:+.4f} | "
            f"[{row['macro_f1_delta_ci_low']:+.4f}, {row['macro_f1_delta_ci_high']:+.4f}] | "
            f"{row['mcnemar_exact_p']:.4f} | {row['mcnemar_holm_p']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _format_delta(value: float | None) -> str:
    return f"{value:+.4f}" if value is not None else "不适用"


def _format_percent(value: float | None) -> str:
    return f"{100.0 * value:.2f}%" if value is not None else "不适用"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operational-root", default=str(DEFAULT_OPERATIONAL_ROOT))
    parser.add_argument("--parent-disjoint-root", default=str(DEFAULT_PARENT_DISJOINT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
