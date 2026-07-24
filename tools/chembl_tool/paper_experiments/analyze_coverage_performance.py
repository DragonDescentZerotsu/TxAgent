"""Generate the focused coverage-versus-performance paper analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .molecular_evidence_agent import (
    DEPLOYMENT_VISIBLE,
    experiments_for_split,
    experiment_result_name,
    experiment_run_root,
    paper_root_for_split,
)
from .summarize_results import (
    _read_jsonl,
    _render_coverage_performance_section,
    _write_tsv,
    build_coverage_performance_rows,
    summarize_coverage_association,
)


def paired_bootstrap_delta_ci_fast(
    labels: list[int],
    left: list[int],
    right: list[int],
    replicates: int,
    *,
    seed: int = 23,
    batch_size: int = 2_000,
) -> tuple[float | None, float | None]:
    """Vectorized equivalent of the paired nonparametric bootstrap."""
    if not labels or replicates <= 0:
        return None, None
    y = np.asarray(labels, dtype=np.int8)
    left_pred = np.asarray(left, dtype=np.int8)
    right_pred = np.asarray(right, dtype=np.int8)
    rng = np.random.default_rng(seed)
    deltas: list[np.ndarray] = []
    for start in range(0, replicates, batch_size):
        count = min(batch_size, replicates - start)
        sampled = rng.integers(0, len(y), size=(count, len(y)))
        sampled_y = y[sampled]

        def sampled_macro(predictions: np.ndarray) -> np.ndarray:
            sampled_pred = predictions[sampled]
            class_scores = []
            for label in (0, 1):
                tp = np.sum((sampled_y == label) & (sampled_pred == label), axis=1)
                fp = np.sum((sampled_y != label) & (sampled_pred == label), axis=1)
                fn = np.sum((sampled_y == label) & (sampled_pred != label), axis=1)
                denominator = 2 * tp + fp + fn
                class_scores.append(np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0))
            return (class_scores[0] + class_scores[1]) / 2

        deltas.append(sampled_macro(right_pred) - sampled_macro(left_pred))
    low, high = np.percentile(np.concatenate(deltas), (2.5, 97.5))
    return float(low), float(high)


def generate(
    *,
    split: str,
    paper_root: Path,
    analysis_dir: Path,
    bootstrap_replicates: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    prediction_sets: dict[str, dict[int, dict[str, Any]]] = {}
    for experiment in experiments_for_split(split):
        batch_dir = experiment_run_root(DEPLOYMENT_VISIBLE, paper_root=paper_root) / experiment.task / experiment.name
        predictions_path = batch_dir / "predictions.jsonl"
        if not predictions_path.exists():
            continue
        result_name = experiment_result_name(experiment.name, DEPLOYMENT_VISIBLE)
        predictions = _read_jsonl(predictions_path)
        prediction_sets[result_name] = {
            int(row["query_index"]): row for row in predictions if row.get("pred_label") is not None
        }
        summaries.append(
            {
                "experiment": result_name,
                "task": experiment.task,
                "visibility_mode": DEPLOYMENT_VISIBLE,
            }
        )
    rows = build_coverage_performance_rows(
        summaries,
        prediction_sets,
        bootstrap_replicates,
        bootstrap_delta_fn=paired_bootstrap_delta_ci_fast,
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _write_tsv(analysis_dir / "coverage_performance.tsv", rows)
    association = summarize_coverage_association(rows)
    (analysis_dir / "coverage_performance_summary.json").write_text(
        json.dumps(
            {
                "data_split": split,
                "visibility_mode": DEPLOYMENT_VISIBLE,
                "association": association,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    section = "\n".join(_render_coverage_performance_section(rows))
    (analysis_dir / "coverage_performance_report.md").write_text(section + "\n", encoding="utf-8")
    report_path = analysis_dir / "report.md"
    if report_path.exists():
        report = report_path.read_text(encoding="utf-8")
        start_marker = "<!-- coverage-performance:start -->"
        end_marker = "<!-- coverage-performance:end -->"
        if start_marker in report and end_marker in report:
            before = report.split(start_marker, 1)[0].rstrip()
            after = report.split(end_marker, 1)[1].lstrip()
            report = f"{before}\n\n{section}\n\n{after}".rstrip() + "\n"
        else:
            report = report.rstrip() + "\n\n" + section + "\n"
        report_path.write_text(report, encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("test", "valid"), default="test")
    parser.add_argument("--paper-root", type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args()
    paper_root = args.paper_root or paper_root_for_split(args.split)
    analysis_dir = args.analysis_dir or paper_root / "analysis"
    rows = generate(
        split=args.split,
        paper_root=paper_root,
        analysis_dir=analysis_dir,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(f"Wrote {len(rows)} coverage-performance rows to {analysis_dir}")


if __name__ == "__main__":
    main()
