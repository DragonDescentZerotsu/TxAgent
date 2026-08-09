"""Summarize the matched final-evidence-surface ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any

from tools.chembl_tool.common.final_evidence_surface import (
    CARDS_ONLY,
    SUMMARY_ONLY,
    SUMMARY_PLUS_CARDS,
)
from tools.chembl_tool.common.json_utils import write_json_atomic
from tools.chembl_tool.paper_experiments.summarize_results import (
    macro_f1,
    mcnemar_exact_p,
    paired_bootstrap_delta_ci,
)

from .run_final_evidence_surface_experiment import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_RUN_ROOT,
    EXPECTED_TASK_ROWS,
    TASKS,
)


SURFACE_LABELS = {
    SUMMARY_ONLY: "Group summaries only",
    SUMMARY_PLUS_CARDS: "Summaries + evidence cards",
    CARDS_ONLY: "Evidence cards only",
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    surface_diagnostics: list[dict[str, Any]] = []
    for task in args.tasks:
        control_batch = (
            Path(args.source_run_root)
            / task
            / f"{task}__starling_full_flat"
        )
        control = _load_condition(control_batch, expected=EXPECTED_TASK_ROWS[task])
        rows.append(_metric_row(task, SUMMARY_ONLY, control, comparison_role="anchor"))
        for surface in (SUMMARY_PLUS_CARDS, CARDS_ONLY):
            candidate_batch = (
                Path(args.output_root)
                / "runs_identity_blind_parent_disjoint"
                / surface
                / task
                / f"{task}__starling_full_flat__{surface}"
            )
            if not candidate_batch.exists():
                if args.allow_incomplete:
                    continue
                raise FileNotFoundError(candidate_batch)
            candidate = _load_condition(
                candidate_batch,
                expected=EXPECTED_TASK_ROWS[task],
                allow_incomplete=args.allow_incomplete,
            )
            rows.append(_metric_row(task, surface, candidate, comparison_role="candidate"))
            comparisons.append(
                _paired_comparison(
                    task,
                    control,
                    candidate,
                    surface=surface,
                    bootstrap_replicates=args.bootstrap_replicates,
                )
            )
            surface_diagnostics.extend(
                _surface_diagnostics(
                    task,
                    surface,
                    control,
                    candidate,
                    candidate_batch,
                    control_batch,
                )
            )
    _add_holm(comparisons)
    output_dir = Path(args.output_root) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "starling.final_evidence_surface_summary.v1",
        "dataset_lineage": "record_supported_v2",
        "benchmark_split": "scaffold",
        "evaluation_subset": "valid",
        "visibility_mode": "identity_blind",
        "neighbor_identity_policy": "parent_disjoint",
        "bootstrap_replicates": args.bootstrap_replicates,
        "metrics": rows,
        "paired_comparisons": comparisons,
        "surface_diagnostics": surface_diagnostics,
    }
    write_json_atomic(output_dir / "summary.json", payload)
    _write_tsv(output_dir / "metrics.tsv", rows)
    _write_tsv(output_dir / "paired_comparisons.tsv", comparisons)
    _write_tsv(output_dir / "surface_diagnostics.tsv", surface_diagnostics)
    _write_report(output_dir / "report.md", rows, comparisons, surface_diagnostics)
    print(json.dumps({"analysis_dir": str(output_dir)}, indent=2))
    return 0


def _load_condition(
    batch: Path,
    *,
    expected: int,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    metrics_path = batch / "metrics.json"
    predictions_path = batch / "predictions.jsonl"
    if not metrics_path.exists() or not predictions_path.exists():
        raise FileNotFoundError(batch)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    prediction_map: dict[int, dict[str, Any]] = {}
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("pred_label") is not None:
                index = int(row["query_index"])
                if index in prediction_map:
                    raise ValueError(f"Duplicate query index {index}: {predictions_path}")
                prediction_map[index] = row
    if int(metrics.get("n_failed_runs") or 0) and not allow_incomplete:
        raise ValueError(f"Condition has failed runs: {batch}")
    if len(prediction_map) != expected and not allow_incomplete:
        raise ValueError(
            f"Condition is incomplete: {batch} rows={len(prediction_map)} expected={expected}"
        )
    return {"batch": str(batch), "metrics": metrics, "predictions": prediction_map}


def _metric_row(
    task: str,
    surface: str,
    condition: dict[str, Any],
    *,
    comparison_role: str,
) -> dict[str, Any]:
    metrics = condition["metrics"]
    return {
        "benchmark_split": "scaffold",
        "evaluation_subset": "valid",
        "task": task,
        "method": f"starling_full_flat__{surface}",
        "method_label": SURFACE_LABELS[surface],
        "plot_label": SURFACE_LABELS[surface],
        "comparison_role": comparison_role,
        "base_method": "starling_full_flat",
        "surface": surface,
        "n": metrics.get("n_successful"),
        "n_failed_runs": metrics.get("n_failed_runs"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "batch": condition["batch"],
    }


def _paired_comparison(
    task: str,
    control: dict[str, Any],
    candidate: dict[str, Any],
    *,
    surface: str,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    left = control["predictions"]
    right = candidate["predictions"]
    common = sorted(set(left) & set(right))
    if not common:
        raise ValueError(f"No paired rows for {task}:{surface}")
    labels = [int(left[index]["label"]) for index in common]
    if labels != [int(right[index]["label"]) for index in common]:
        raise ValueError(f"Gold-label mismatch for {task}:{surface}")
    left_predictions = [int(left[index]["pred_label"]) for index in common]
    right_predictions = [int(right[index]["pred_label"]) for index in common]
    left_f1 = macro_f1(labels, left_predictions)
    right_f1 = macro_f1(labels, right_predictions)
    ci_low, ci_high = paired_bootstrap_delta_ci(
        labels,
        left_predictions,
        right_predictions,
        bootstrap_replicates,
        seed=20260807,
    )
    left_only = sum(
        left_prediction == label and right_prediction != label
        for label, left_prediction, right_prediction in zip(
            labels,
            left_predictions,
            right_predictions,
        )
    )
    right_only = sum(
        left_prediction != label and right_prediction == label
        for label, left_prediction, right_prediction in zip(
            labels,
            left_predictions,
            right_predictions,
        )
    )
    return {
        "task": task,
        "left_surface": SUMMARY_ONLY,
        "right_surface": surface,
        "n_paired": len(common),
        "left_macro_f1": left_f1,
        "right_macro_f1": right_f1,
        "delta_macro_f1": right_f1 - left_f1,
        "delta_ci_low": ci_low,
        "delta_ci_high": ci_high,
        "n_prediction_flips": sum(
            left_prediction != right_prediction
            for left_prediction, right_prediction in zip(
                left_predictions,
                right_predictions,
            )
        ),
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "mcnemar_exact_p": mcnemar_exact_p(left_only, right_only),
    }


def _add_holm(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(
        enumerate(rows),
        key=lambda item: float(item[1]["mcnemar_exact_p"]),
    )
    running_max = 0.0
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, float(row["mcnemar_exact_p"]) * (len(rows) - rank))
        running_max = max(running_max, adjusted)
        rows[index]["mcnemar_holm_p"] = running_max


def _surface_diagnostics(
    task: str,
    surface: str,
    control: dict[str, Any],
    candidate: dict[str, Any],
    candidate_batch: Path,
    control_batch: Path,
) -> list[dict[str, Any]]:
    left = control["predictions"]
    right = candidate["predictions"]
    target_by_index = _run_dirs_by_index(candidate_batch / "runs")
    source_by_index = _run_dirs_by_index(control_batch / "runs")
    samples: list[dict[str, Any]] = []
    for index in sorted(set(left) & set(right)):
        target = json.loads(
            (target_by_index[index] / "final_reasoning_output.json").read_text(
                encoding="utf-8"
            )
        )
        source = json.loads(
            (source_by_index[index] / "final_reasoning_output.json").read_text(
                encoding="utf-8"
            )
        )
        audit = target.get("final_evidence_surface") or {}
        samples.append(
            {
                "n_cards": int(audit.get("n_cards") or 0),
                "control_correct": bool(left[index]["correct"]),
                "candidate_correct": bool(right[index]["correct"]),
                "prediction_flip": left[index]["pred_label"] != right[index]["pred_label"],
                "control_prompt_tokens": _prompt_tokens(source),
                "candidate_prompt_tokens": _prompt_tokens(target),
            }
        )
    output: list[dict[str, Any]] = []
    for stratum, selected in (
        ("all", samples),
        ("no_cards", [sample for sample in samples if sample["n_cards"] == 0]),
        ("with_cards", [sample for sample in samples if sample["n_cards"] > 0]),
    ):
        if not selected:
            continue
        n = len(selected)
        control_tokens = statistics.mean(
            sample["control_prompt_tokens"] for sample in selected
        )
        candidate_tokens = statistics.mean(
            sample["candidate_prompt_tokens"] for sample in selected
        )
        output.append(
            {
                "task": task,
                "surface": surface,
                "stratum": stratum,
                "n": n,
                "mean_cards": statistics.mean(sample["n_cards"] for sample in selected),
                "n_prediction_flips": sum(sample["prediction_flip"] for sample in selected),
                "control_only_correct": sum(
                    sample["control_correct"] and not sample["candidate_correct"]
                    for sample in selected
                ),
                "candidate_only_correct": sum(
                    not sample["control_correct"] and sample["candidate_correct"]
                    for sample in selected
                ),
                "control_accuracy": sum(sample["control_correct"] for sample in selected) / n,
                "candidate_accuracy": sum(sample["candidate_correct"] for sample in selected) / n,
                "control_mean_prompt_tokens": control_tokens,
                "candidate_mean_prompt_tokens": candidate_tokens,
                "delta_mean_prompt_tokens": candidate_tokens - control_tokens,
                "prompt_token_ratio": (
                    candidate_tokens / control_tokens if control_tokens else None
                ),
            }
        )
    return output


def _prompt_tokens(final_output: dict[str, Any]) -> int:
    return int(
        (((final_output.get("llm") or {}).get("usage") or {}).get("prompt_tokens"))
        or 0
    )


def _run_dirs_by_index(root: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for run_dir in root.glob("*_idx*"):
        index = int(run_dir.name.rsplit("_idx", 1)[1])
        if index in result:
            raise ValueError(f"Duplicate query index {index}: {root}")
        result[index] = run_dir
    return result


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    surface_diagnostics: list[dict[str, Any]],
) -> None:
    lines = [
        "# Final evidence surface valid 诊断",
        "",
        "该实验固定 retrieval、single 和 group artifacts，只重新运行 final。结果属于 scaffold-valid 诊断，不是 formal test。",
        "",
        "| task | surface | n | failed | accuracy | macro-F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['surface']} | {row['n']} | "
            f"{row['n_failed_runs']} | {float(row['accuracy']):.4f} | "
            f"{float(row['macro_f1']):.4f} |"
        )
    lines.extend(
        [
            "",
            "| task | candidate | Δ macro-F1 | 95% CI | flips | control-only / candidate-only correct | McNemar p / Holm p |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['task']} | {row['right_surface']} | "
            f"{row['delta_macro_f1']:+.4f} | "
            f"[{row['delta_ci_low']:+.4f}, {row['delta_ci_high']:+.4f}] | "
            f"{row['n_prediction_flips']} | "
            f"{row['left_only_correct']} / {row['right_only_correct']} | "
            f"{row['mcnemar_exact_p']:.4g} / {row['mcnemar_holm_p']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Card coverage 与 final prompt 成本",
            "",
            "| task | surface | mean cards | zero-card n | mean prompt tokens | vs summary-only | flips | control-only / candidate-only correct |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in surface_diagnostics:
        if row["stratum"] != "all":
            continue
        zero_row = next(
            (
                candidate
                for candidate in surface_diagnostics
                if candidate["task"] == row["task"]
                and candidate["surface"] == row["surface"]
                and candidate["stratum"] == "no_cards"
            ),
            None,
        )
        lines.append(
            f"| {row['task']} | {row['surface']} | {float(row['mean_cards']):.2f} | "
            f"{int(zero_row['n']) if zero_row else 0} | "
            f"{float(row['candidate_mean_prompt_tokens']):.1f} | "
            f"{float(row['prompt_token_ratio']):.2f}x | {row['n_prediction_flips']} | "
            f"{row['control_only_correct']} / {row['candidate_only_correct']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--source-run-root", default=str(DEFAULT_SOURCE_RUN_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
