"""Summarize strict matched Morgan-versus-coverage LLM results across Starling."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT / "outputs/paper/coverage_selector_llm/analysis"
DEFAULT_AGENT_BATCH_SUBDIR = "runs_identity_blind_parent_disjoint"

CONDITIONS = (
    ("random", "bbb_martins", "BBB_Martins", 500),
    ("scaffold", "bbb_martins", "BBB_Martins", 500),
    ("random", "bioavailability_ma", "Bioavailability_Ma", 372),
    ("scaffold", "bioavailability_ma", "Bioavailability_Ma", 372),
    ("random", "skin_reaction", "Skin_Reaction", 380),
    ("scaffold", "skin_reaction", "Skin_Reaction", 380),
)

METRIC_FIELDS = (
    "accuracy",
    "macro_f1",
    "positive_precision",
    "positive_recall",
    "positive_f1",
)


def read_predictions(path: Path, *, expected_n: int | None) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            index = row.get("query_index")
            if not isinstance(index, int):
                raise ValueError(f"{path}:{line_number}: missing integer query_index")
            if index in rows:
                raise ValueError(f"{path}: duplicate query_index={index}")
            rows[index] = row

    expected_n = len(rows) if expected_n is None else expected_n
    expected = set(range(expected_n))
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        extra = sorted(set(rows) - expected)
        raise ValueError(
            f"{path}: expected indices 0..{expected_n - 1}; "
            f"missing={missing}, extra={extra}"
        )
    invalid = [
        index
        for index, row in sorted(rows.items())
        if row.get("status") != "ok"
        or row.get("final_status") != "ok"
        or row.get("pred_label") not in {0, 1}
        or row.get("label") not in {0, 1}
    ]
    if invalid:
        raise ValueError(f"{path}: invalid or failed rows at indices {invalid}")
    return rows


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("labels and predictions must be non-empty and aligned")
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions, strict=True))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions, strict=True))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions, strict=True))
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions, strict=True))

    precision_0 = safe_ratio(tn, tn + fn)
    recall_0 = safe_ratio(tn, tn + fp)
    f1_0 = safe_ratio(2 * precision_0 * recall_0, precision_0 + recall_0)
    precision_1 = safe_ratio(tp, tp + fp)
    recall_1 = safe_ratio(tp, tp + fn)
    f1_1 = safe_ratio(2 * precision_1 * recall_1, precision_1 + recall_1)
    return {
        "n": len(labels),
        "accuracy": safe_ratio(tn + tp, len(labels)),
        "macro_f1": (f1_0 + f1_1) / 2,
        "positive_precision": precision_1,
        "positive_recall": recall_1,
        "positive_f1": f1_1,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def exact_mcnemar_p(control_only_correct: int, coverage_only_correct: int) -> float:
    discordant = control_only_correct + coverage_only_correct
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(control_only_correct, coverage_only_correct) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_macro_f1_delta(
    labels: list[int],
    control_predictions: list[int],
    coverage_predictions: list[int],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not (
        len(labels) == len(control_predictions) == len(coverage_predictions)
    ) or not labels:
        raise ValueError("labels and predictions must be non-empty and aligned")
    rng = random.Random(seed)
    n = len(labels)
    deltas: list[float] = []
    for _ in range(iterations):
        sampled = [rng.randrange(n) for _ in range(n)]
        sampled_labels = [labels[index] for index in sampled]
        control = binary_metrics(
            sampled_labels, [control_predictions[index] for index in sampled]
        )
        coverage = binary_metrics(
            sampled_labels, [coverage_predictions[index] for index in sampled]
        )
        deltas.append(coverage["macro_f1"] - control["macro_f1"])
    return {
        "iterations": iterations,
        "seed": seed,
        "ci95_low": percentile(deltas, 0.025),
        "ci95_high": percentile(deltas, 0.975),
        "probability_delta_gt_zero": sum(value > 0 for value in deltas) / iterations,
    }


def run_paths(split: str, task: str) -> tuple[Path, Path, str, str]:
    paper_root = ROOT / f"outputs/paper/molecular_evidence_agent_starling_{split}"
    control_prefix = f"{task}__starling_full_mechanism"
    control_root = (
        paper_root
        / "runs_deployment_visible_parent_disjoint"
        / task
        / control_prefix
        / "runs"
    )
    if split == "scaffold" and task == "bbb_martins":
        coverage_prefix = "bbb_scaffold_starling_full_mechanism_pd_coverage_changed50_v1"
        coverage_root = (
            paper_root
            / "coverage_selector_pilot/bbb_martins/batches/"
            "bbb_scaffold_starling_full_mechanism_pd_coverage_changed50_v1/runs"
        )
    else:
        coverage_prefix = f"{task}__starling_full_mechanism__coverage"
        coverage_root = (
            paper_root
            / "coverage_selector_llm"
            / task
            / coverage_prefix
            / "runs"
        )
    return control_root, coverage_root, control_prefix, coverage_prefix


def prediction_paths(split: str, task: str) -> tuple[Path, Path]:
    control_root, coverage_root, _, _ = run_paths(split, task)
    return (
        control_root.parent / "predictions.jsonl",
        coverage_root.parent / "predictions.jsonl",
    )


def summarize_condition(
    *,
    split: str,
    task: str,
    task_label: str,
    expected_n: int,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    control_path, coverage_path = prediction_paths(split, task)
    return summarize_prediction_pair(
        control_path=control_path,
        candidate_path=coverage_path,
        split=split,
        task=task,
        task_label=task_label,
        expected_n=expected_n,
        control_key="morgan_similarity",
        candidate_key="query_feature_coverage",
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )


def summarize_prediction_pair(
    *,
    control_path: Path,
    candidate_path: Path,
    split: str,
    task: str,
    task_label: str,
    expected_n: int | None,
    control_key: str,
    candidate_key: str,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Summarize any two complete, index-aligned binary agent batches."""
    control = read_predictions(control_path, expected_n=expected_n)
    coverage = read_predictions(candidate_path, expected_n=expected_n)
    if set(control) != set(coverage):
        raise ValueError(
            f"{task}/{split}: prediction index sets differ between matched batches"
        )
    indices = sorted(control)
    label_mismatches = [
        index for index in indices if control[index]["label"] != coverage[index]["label"]
    ]
    if label_mismatches:
        raise ValueError(
            f"{task}/{split}: gold-label mismatches at indices {label_mismatches}"
        )

    labels = [int(control[index]["label"]) for index in indices]
    control_predictions = [int(control[index]["pred_label"]) for index in indices]
    coverage_predictions = [int(coverage[index]["pred_label"]) for index in indices]
    metrics = {
        control_key: binary_metrics(labels, control_predictions),
        candidate_key: binary_metrics(labels, coverage_predictions),
    }
    control_only_correct = sum(
        control[index]["correct"] is True and coverage[index]["correct"] is False
        for index in indices
    )
    candidate_only_correct = sum(
        control[index]["correct"] is False and coverage[index]["correct"] is True
        for index in indices
    )
    flip_indices = [
        index
        for index in indices
        if control[index]["pred_label"] != coverage[index]["pred_label"]
    ]
    deltas = {
        field: metrics[candidate_key][field] - metrics[control_key][field]
        for field in METRIC_FIELDS
    }
    result = {
        "split": split,
        "task": task,
        "task_label": task_label,
        "n_matched": len(indices),
        "metrics": metrics,
        "deltas_candidate_minus_control": deltas,
        "paired_outcomes": {
            "prediction_flips": len(flip_indices),
            "prediction_flip_indices": flip_indices,
            "control_only_correct": control_only_correct,
            "candidate_only_correct": candidate_only_correct,
            "both_correct": sum(
                control[index]["correct"] is True and coverage[index]["correct"] is True
                for index in indices
            ),
            "both_wrong": sum(
                control[index]["correct"] is False and coverage[index]["correct"] is False
                for index in indices
            ),
            "mcnemar_exact_p": exact_mcnemar_p(
                control_only_correct, candidate_only_correct
            ),
        },
        "macro_f1_delta_bootstrap": bootstrap_macro_f1_delta(
            labels,
            control_predictions,
            coverage_predictions,
            iterations=bootstrap_iterations,
            seed=seed,
        ),
        "sources": {
            control_key: str(control_path),
            candidate_key: str(candidate_path),
        },
    }
    if (control_key, candidate_key) == (
        "morgan_similarity",
        "query_feature_coverage",
    ):
        result["deltas_coverage_minus_morgan"] = deltas
        result["paired_outcomes"]["morgan_only_correct"] = control_only_correct
        result["paired_outcomes"]["coverage_only_correct"] = candidate_only_correct
    return result


def _explicit_prediction_path(
    root: Path,
    task: str,
    experiment: str,
    *,
    batch_subdir: str = DEFAULT_AGENT_BATCH_SUBDIR,
) -> Path:
    return (
        root
        / batch_subdir
        / task
        / experiment
        / "predictions.jsonl"
    )


def _expected_size(split: str, evaluation_subset: str, data_name: str) -> int:
    path = (
        ROOT
        / "data/gold_labels/legacy/processed_starling"
        / data_name
        / split
        / f"{evaluation_subset}.jsonl"
    )
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--control-root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--batch-subdir", default=DEFAULT_AGENT_BATCH_SUBDIR)
    parser.add_argument("--benchmark-split", choices=("random", "scaffold"), default="scaffold")
    parser.add_argument("--evaluation-subset", choices=("valid", "test"), default="valid")
    parser.add_argument(
        "--tasks",
        nargs="*",
        choices=("bbb_martins", "bioavailability_ma", "skin_reaction"),
        default=[],
    )
    parser.add_argument("--control-key", default="standard_context")
    parser.add_argument("--candidate-key", default="coverage_aware_context")
    parser.add_argument("--control-label", default="Coverage retrieve + standard context")
    parser.add_argument("--candidate-label", default="Coverage retrieve + coverage-aware context")
    parser.add_argument("--only-changed-factor", default="neighbor_context_profile")
    parser.add_argument(
        "--visibility-mode",
        choices=("identity_blind", "deployment_visible", "deployment_visible_prefetched"),
        default="identity_blind",
    )
    parser.add_argument("--report-title", default="Coverage-aware context matched LLM 对比")
    args = parser.parse_args()

    if bool(args.control_root) != bool(args.candidate_root):
        parser.error("--control-root and --candidate-root must be provided together")

    if args.control_root:
        task_specs = {
            "bbb_martins": ("BBB_Martins", "BBB"),
            "bioavailability_ma": ("Bioavailability_Ma", "Bioavailability"),
            "skin_reaction": ("Skin_Reaction", "Skin reaction"),
        }
        selected_tasks = args.tasks or list(task_specs)
        conditions = []
        for index, task in enumerate(selected_tasks):
            data_name, task_label = task_specs[task]
            experiment = f"{task}__starling_full_mechanism"
            conditions.append(
                summarize_prediction_pair(
                    control_path=_explicit_prediction_path(
                        args.control_root,
                        task,
                        experiment,
                        batch_subdir=args.batch_subdir,
                    ),
                    candidate_path=_explicit_prediction_path(
                        args.candidate_root,
                        task,
                        experiment,
                        batch_subdir=args.batch_subdir,
                    ),
                    split=args.benchmark_split,
                    task=task,
                    task_label=task_label,
                    expected_n=_expected_size(
                        args.benchmark_split,
                        args.evaluation_subset,
                        data_name,
                    ),
                    control_key=args.control_key,
                    candidate_key=args.candidate_key,
                    bootstrap_iterations=args.bootstrap_iterations,
                    seed=args.seed + index,
                )
            )
        method_specs = (
            (args.control_key, args.control_label),
            (args.candidate_key, args.candidate_label),
        )
        report_title = args.report_title
        contract = {
            "experiment_mode": "full_mechanism",
            "retrieval_source": "starling",
            "visibility_mode": args.visibility_mode,
            "neighbor_identity_policy": "parent_disjoint",
            "neighbor_selector": "query_feature_coverage",
            "top_k_per_group": 3,
            "min_similarity": 0.3,
            "only_changed_factor": args.only_changed_factor,
        }
    else:
        conditions = [
            summarize_condition(
                split=split,
                task=task,
                task_label=task_label,
                expected_n=expected_n,
                bootstrap_iterations=args.bootstrap_iterations,
                seed=args.seed + index,
            )
            for index, (split, task, task_label, expected_n) in enumerate(CONDITIONS)
        ]
        method_specs = (
            ("morgan_similarity", "Morgan similarity retrieve"),
            ("query_feature_coverage", "Coverage retrieve"),
        )
        report_title = "Coverage selector Starling matched LLM 对比"
        contract = {
            "experiment_mode": "full_mechanism",
            "retrieval_source": "starling",
            "neighbor_identity_policy": "parent_disjoint",
            "top_k_per_group": 3,
            "min_similarity": 0.3,
            "only_changed_factor": "neighbor_selector",
        }
    payload = {
        "benchmark": "Starling binary benchmark",
        "retrieval_contract": contract,
        "methods": [
            {"key": method, "label": label} for method, label in method_specs
        ],
        "conditions": conditions,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.tsv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "benchmark_split",
            "task",
            "task_label",
            "method",
            "method_label",
            "n",
            *METRIC_FIELDS,
            "tn",
            "fp",
            "fn",
            "tp",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for condition in conditions:
            for method, method_label in method_specs:
                metrics = condition["metrics"][method]
                writer.writerow(
                    {
                        "benchmark_split": condition["split"],
                        "task": condition["task"],
                        "task_label": condition["task_label"],
                        "method": method,
                        "method_label": method_label,
                        "n": metrics["n"],
                        **{
                            field: f"{metrics[field]:.6f}"
                            for field in METRIC_FIELDS
                        },
                        **metrics["confusion_matrix"],
                    }
                )

    comparison_path = args.output_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        f"# {report_title}",
        "",
        "所有行均为同一 task/split 上的严格全样本配对；除 retrieval_contract 中声明的"
        " only_changed_factor 外，其余模型、evidence、visibility、identity policy 和 decoding 均相同。",
        "",
        f"| Task | Split | n | {method_specs[0][1]} accuracy | {method_specs[1][1]} accuracy | Δ accuracy | "
        f"{method_specs[0][1]} macro-F1 | {method_specs[1][1]} macro-F1 | Δ macro-F1 | flips | McNemar p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        morgan = condition["metrics"][method_specs[0][0]]
        coverage = condition["metrics"][method_specs[1][0]]
        delta = condition["deltas_candidate_minus_control"]
        paired = condition["paired_outcomes"]
        report.append(
            f"| {condition['task_label']} | {condition['split']} | "
            f"{condition['n_matched']} | {morgan['accuracy']:.4f} | "
            f"{coverage['accuracy']:.4f} | {delta['accuracy']:+.4f} | "
            f"{morgan['macro_f1']:.4f} | {coverage['macro_f1']:.4f} | "
            f"{delta['macro_f1']:+.4f} | {paired['prediction_flips']} | "
            f"{paired['mcnemar_exact_p']:.4g} |"
        )
    report.append("")
    (args.output_dir / "report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(metrics_path)
    print(comparison_path)
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
