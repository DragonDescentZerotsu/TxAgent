"""Build a task-balanced LLM eval set from task-scoped transfer pairs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.chembl_tool.activity_transfer_benchmark.build_llm_eval_set import (
    LABELS,
    TANIMOTO_BUCKETS,
    compute_prediction_metrics,
    write_jsonl,
    write_tsv,
)


DEFAULT_BENCHMARK_RUN = "outputs/chembl_tool/activity_transfer_benchmark/task_assay_runs/task_assay_transfer_v2"
DEFAULT_OUT_ROOT = "outputs/chembl_tool/activity_transfer_benchmark/llm_eval_sets"
TASKS = ["bbb_martins", "bioavailability_ma", "clintox", "skin_reaction"]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)
    benchmark_run = Path(args.benchmark_run)
    pairs_path = Path(args.pairs) if args.pairs else benchmark_run / args.label_mode / "pairs.tsv.gz"
    endpoint_summary_path = (
        Path(args.endpoint_summary)
        if args.endpoint_summary
        else benchmark_run / args.label_mode / "endpoint_summary.tsv"
    )
    out_dir = Path(args.out_root) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoint_meta = read_endpoint_meta(endpoint_summary_path)
    selected = stratified_select(
        pairs_path,
        endpoint_meta=endpoint_meta,
        n_total=args.n_total,
        tasks=parse_tasks(args.tasks),
        max_per_endpoint=args.max_per_endpoint,
        oversample_factor=args.oversample_factor,
        seed=args.seed,
    )
    rng.shuffle(selected)
    for sample_index, row in enumerate(selected):
        row["sample_index"] = sample_index

    jsonl_path = out_dir / "eval_pairs.jsonl"
    tsv_path = out_dir / "eval_pairs.tsv"
    write_jsonl(jsonl_path, selected)
    write_tsv(tsv_path, selected)
    summary = summarize(selected, args, pairs_path, endpoint_summary_path)
    summary["files"] = {"jsonl": str(jsonl_path), "tsv": str(tsv_path)}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(out_dir / "report_zh.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-run", default=DEFAULT_BENCHMARK_RUN)
    parser.add_argument("--label-mode", default="raw_robust_z", choices=["raw_robust_z", "log_raw_robust_z", "pchembl_delta"])
    parser.add_argument("--pairs", default="")
    parser.add_argument("--endpoint-summary", default="")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default="task_assay_raw_robust_z_llm_3k")
    parser.add_argument("--n-total", type=int, default=3000)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--max-per-endpoint", type=int, default=6)
    parser.add_argument("--oversample-factor", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args(argv)


def parse_tasks(raw: str) -> list[str]:
    tasks = [item.strip() for item in raw.split(",") if item.strip()]
    if not tasks:
        raise ValueError("At least one task is required.")
    return tasks


def read_endpoint_meta(path: Path) -> dict[tuple[str, str, str, str, str], dict[str, str]]:
    meta = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (
                row.get("task_name", ""),
                row.get("assay_chembl_id", ""),
                row.get("standard_type", ""),
                row.get("raw_units", ""),
                row.get("label_mode", ""),
            )
            meta[key] = row
    return meta


def stratified_select(
    pairs_path: Path,
    *,
    endpoint_meta: dict[tuple[str, str, str, str, str], dict[str, str]],
    n_total: int,
    tasks: list[str],
    max_per_endpoint: int,
    oversample_factor: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    quota = build_quota(n_total, tasks)
    reservoir_targets = {key: max(target * oversample_factor, target + 10) for key, target in quota.items()}
    reservoirs: dict[tuple[str, str, str], list[dict[str, Any]]] = {key: [] for key in quota}
    seen: Counter = Counter()

    with gzip.open(pairs_path, "rt", encoding="utf-8", newline="") as handle:
        for pair_index, row in enumerate(csv.DictReader(handle, delimiter="\t")):
            label = row.get("label", "")
            task_name = row.get("task_name", "")
            bucket = row.get("similarity_bucket", "")
            key = (task_name, label, bucket)
            if key not in reservoirs:
                continue
            seen[key] += 1
            candidate = build_sample_record(pair_index, row, endpoint_meta)
            reservoir = reservoirs[key]
            target = reservoir_targets[key]
            if len(reservoir) < target:
                reservoir.append(candidate)
            else:
                replace_index = rng.randrange(seen[key])
                if replace_index < target:
                    reservoir[replace_index] = candidate

    selected: list[dict[str, Any]] = []
    selected_pair_indices: set[int] = set()
    endpoint_counts: Counter = Counter()
    candidates_by_task_label: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    candidates_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    global_candidates: list[dict[str, Any]] = []
    target_by_task_label: Counter = Counter()
    target_by_task: Counter = Counter()

    for key, target in quota.items():
        task_name, label, _bucket = key
        target_by_task_label[(task_name, label)] += target
        target_by_task[task_name] += target
        rng.shuffle(reservoirs[key])
        for row in reservoirs[key]:
            candidates_by_task_label[(task_name, label)].append(row)
            candidates_by_task[task_name].append(row)
            global_candidates.append(row)
        selected.extend(
            pick_with_endpoint_cap(
                reservoirs[key],
                target,
                endpoint_counts,
                max_per_endpoint,
                selected_pair_indices,
            )
        )

    # Keep the LLM set analytically clean: first repair within each task+label,
    # then within task, and only cross task as a last resort if a stratum is exhausted.
    for key, target in target_by_task_label.items():
        current = sum(1 for row in selected if (row["task_name"], row["label"]) == key)
        if current >= target:
            continue
        rng.shuffle(candidates_by_task_label[key])
        selected.extend(
            pick_with_endpoint_cap(
                candidates_by_task_label[key],
                target - current,
                endpoint_counts,
                max_per_endpoint,
                selected_pair_indices,
            )
        )

    for task_name, target in target_by_task.items():
        current = sum(1 for row in selected if row["task_name"] == task_name)
        if current >= target:
            continue
        rng.shuffle(candidates_by_task[task_name])
        selected.extend(
            pick_with_endpoint_cap(
                candidates_by_task[task_name],
                target - current,
                endpoint_counts,
                max_per_endpoint,
                selected_pair_indices,
            )
        )

    target_n = sum(quota.values())
    if len(selected) < target_n:
        rng.shuffle(global_candidates)
        selected.extend(
            pick_with_endpoint_cap(
                global_candidates,
                target_n - len(selected),
                endpoint_counts,
                max_per_endpoint,
                selected_pair_indices,
            )
        )
    if len(selected) < target_n:
        rng.shuffle(global_candidates)
        for row in global_candidates:
            pair_index = row.get("pair_index")
            if pair_index in selected_pair_indices:
                continue
            selected.append(row)
            selected_pair_indices.add(pair_index)
            if len(selected) >= target_n:
                break
    if len(selected) < target_n:
        raise RuntimeError(f"Only selected {len(selected)} rows, expected {target_n}.")
    return selected[:target_n]


def build_quota(n_total: int, tasks: list[str]) -> dict[tuple[str, str, str], int]:
    quota = {}
    task_base = n_total // len(tasks)
    task_remainder = n_total - task_base * len(tasks)
    for task_index, task in enumerate(tasks):
        task_target = task_base + (1 if task_index < task_remainder else 0)
        label_base = task_target // len(LABELS)
        label_remainder = task_target - label_base * len(LABELS)
        for label_index, label in enumerate(LABELS):
            label_target = label_base + (1 if label_index < label_remainder else 0)
            bucket_base = label_target // len(TANIMOTO_BUCKETS)
            bucket_remainder = label_target - bucket_base * len(TANIMOTO_BUCKETS)
            for bucket_index, bucket in enumerate(TANIMOTO_BUCKETS):
                quota[(task, label, bucket)] = bucket_base + (1 if bucket_index < bucket_remainder else 0)
    return quota


def pick_with_endpoint_cap(
    rows: list[dict[str, Any]],
    target: int,
    endpoint_counts: Counter,
    max_per_endpoint: int,
    selected_pair_indices: set[int],
) -> list[dict[str, Any]]:
    picked = []
    for row in rows:
        pair_index = row.get("pair_index")
        if pair_index in selected_pair_indices:
            continue
        endpoint_key = endpoint_key_for_row(row)
        if endpoint_counts[endpoint_key] >= max_per_endpoint:
            continue
        picked.append(row)
        endpoint_counts[endpoint_key] += 1
        selected_pair_indices.add(pair_index)
        if len(picked) >= target:
            break
    return picked


def endpoint_key_for_row(row: dict[str, Any]) -> str:
    return "::".join(
        str(row.get(key, ""))
        for key in ("task_name", "assay_chembl_id", "standard_type", "raw_units", "label_mode")
    )


def build_sample_record(
    pair_index: int,
    pair: dict[str, str],
    endpoint_meta: dict[tuple[str, str, str, str, str], dict[str, str]],
) -> dict[str, Any]:
    meta_key = (
        pair.get("task_name", ""),
        pair.get("assay_chembl_id", ""),
        pair.get("standard_type", ""),
        pair.get("raw_units", ""),
        pair.get("label_mode", ""),
    )
    meta = endpoint_meta.get(meta_key, {})
    tanimoto = parse_float(pair.get("tanimoto"))
    label_mode = pair.get("label_mode", "")
    activity_scale = "pchembl" if label_mode == "pchembl_delta" else ("log10_raw" if label_mode == "log_raw_robust_z" else "raw")
    return {
        "pair_index": pair_index,
        "label": pair["label"],
        "task_name": pair.get("task_name", ""),
        "assay_chembl_id": pair["assay_chembl_id"],
        "assay_id": parse_int(pair.get("assay_id")),
        "standard_type": pair["standard_type"],
        "raw_units": pair.get("raw_units", ""),
        "label_mode": label_mode,
        "activity_scale": activity_scale,
        "target_chembl_id": pair.get("target_chembl_id", ""),
        "target_pref_name": pair.get("target_pref_name", ""),
        "target_type": "",
        "target_organism": meta.get("organism", ""),
        "assay_type": "",
        "assay_test_type": "",
        "assay_category": "",
        "assay_tier": pair.get("assay_tier", ""),
        "confidence_score": meta.get("confidence_score", ""),
        "relationship_type": meta.get("relationship_type", ""),
        "assay_description": pair.get("assay_description", ""),
        "query_molecule_chembl_id": pair["molecule_b_chembl_id"],
        "reference_molecule_chembl_id": pair["molecule_a_chembl_id"],
        "query_smiles": pair["molecule_b_smiles"],
        "reference_smiles": pair["molecule_a_smiles"],
        "reference_activity_value": parse_float(pair.get("activity_a")),
        "hidden_query_activity_value": parse_float(pair.get("activity_b")),
        "reference_raw_activity_value": parse_float(pair.get("raw_activity_a")),
        "hidden_query_raw_activity_value": parse_float(pair.get("raw_activity_b")),
        "reference_log_raw_activity_value": parse_float(pair.get("log_raw_activity_a")),
        "hidden_query_log_raw_activity_value": parse_float(pair.get("log_raw_activity_b")),
        "reference_pchembl_value": parse_float(pair.get("pchembl_a")),
        "hidden_query_pchembl_value": parse_float(pair.get("pchembl_b")),
        "abs_activity_delta": parse_float(pair.get("abs_activity_delta")),
        "normalized_delta": parse_float(pair.get("normalized_delta")),
        "robust_sigma": parse_float(pair.get("robust_sigma")),
        "tanimoto": tanimoto,
        "similarity_bucket": pair["similarity_bucket"],
        "mean_mcs_coverage": None,
        "query_mcs_coverage": None,
        "reference_mcs_coverage": None,
        "mcs_timed_out": None,
        "baseline_tanimoto_0_50_prediction": "similar" if tanimoto >= 0.50 else "different",
        "baseline_tanimoto_0_48_prediction": "similar" if tanimoto >= 0.48 else "different",
        "baseline_mcs_0_70_prediction": "",
    }


def parse_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value: Any) -> int | str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return int(text)
    except ValueError:
        return text


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, pairs_path: Path, endpoint_summary_path: Path) -> dict[str, Any]:
    label_counts = Counter(row["label"] for row in rows)
    task_counts = Counter(row["task_name"] for row in rows)
    task_label_counts = Counter((row["task_name"], row["label"]) for row in rows)
    task_bucket_counts = Counter((row["task_name"], row["similarity_bucket"]) for row in rows)
    endpoint_counts = Counter(endpoint_key_for_row(row) for row in rows)
    per_task = {}
    for task_name in sorted(task_counts):
        task_rows = [row for row in rows if row["task_name"] == task_name]
        per_task[task_name] = {
            "n": len(task_rows),
            "label_counts": dict(Counter(row["label"] for row in task_rows)),
            "n_endpoint_groups": len({endpoint_key_for_row(row) for row in task_rows}),
            "baseline_metrics": {
                "tanimoto_0_50": compute_prediction_metrics(task_rows, "baseline_tanimoto_0_50_prediction"),
                "tanimoto_0_48": compute_prediction_metrics(task_rows, "baseline_tanimoto_0_48_prediction"),
            },
        }
    return {
        "run_id": args.run_id,
        "n": len(rows),
        "pairs_path": str(pairs_path),
        "endpoint_summary_path": str(endpoint_summary_path),
        "parameters": vars(args),
        "label_counts": dict(label_counts),
        "task_counts": dict(task_counts),
        "task_label_counts": {f"{task}|{label}": count for (task, label), count in task_label_counts.items()},
        "task_bucket_counts": {f"{task}|{bucket}": count for (task, bucket), count in task_bucket_counts.items()},
        "n_endpoint_groups": len(endpoint_counts),
        "max_pairs_per_endpoint_observed": max(endpoint_counts.values()) if endpoint_counts else 0,
        "baseline_metrics": {
            "tanimoto_0_50": compute_prediction_metrics(rows, "baseline_tanimoto_0_50_prediction"),
            "tanimoto_0_48": compute_prediction_metrics(rows, "baseline_tanimoto_0_48_prediction"),
        },
        "per_task": per_task,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Task-scoped activity-transfer LLM eval set",
        "",
        f"- samples: {summary['n']:,}",
        f"- endpoint groups: {summary['n_endpoint_groups']:,}",
        f"- max pairs per endpoint group: {summary['max_pairs_per_endpoint_observed']:,}",
        f"- label counts: {summary['label_counts']}",
        f"- task counts: {summary['task_counts']}",
        "",
        "## Overall Baselines",
        "",
        "| baseline | accuracy | balanced accuracy | macro-F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, metrics in summary["baseline_metrics"].items():
        lines.append(metric_line(name, metrics))
    lines.extend(["", "## Per Task", "", "| task | n | endpoint groups | tanimoto_0_50 macro-F1 | tanimoto_0_48 macro-F1 |", "| --- | ---: | ---: | ---: | ---: |"])
    for task_name, task_summary in summary["per_task"].items():
        lines.append(
            f"| {task_name} | {task_summary['n']:,} | {task_summary['n_endpoint_groups']:,} | "
            f"{task_summary['baseline_metrics']['tanimoto_0_50']['macro_f1']:.4f} | "
            f"{task_summary['baseline_metrics']['tanimoto_0_48']['macro_f1']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_line(name: str, metrics: dict[str, Any]) -> str:
    return f"| {name} | {metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} | {metrics['macro_f1']:.4f} |"


if __name__ == "__main__":
    raise SystemExit(main())
