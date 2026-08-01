"""Compare Morgan- and MiniMol-retrieved Starling GLM agent conditions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.paper_experiments.build_starling_benchmark_indices import (
    BENCHMARK_SPLITS,
    paper_root_for_benchmark_split,
)
from tools.chembl_tool.paper_experiments.minimol_retrieval_contract import (
    paper_root_for_minimol_retrieval,
)
from tools.chembl_tool.paper_experiments.molecular_evidence_agent import (
    DEPLOYMENT_VISIBLE,
    PARENT_DISJOINT,
    experiment_run_root,
)
from tools.chembl_tool.paper_experiments.starling_benchmark_matrix import (
    experiments_for_starling_benchmark,
)
from tools.chembl_tool.paper_experiments.summarize_results import (
    macro_f1,
    mcnemar_exact_p,
    paired_bootstrap_delta_ci,
)


DEFAULT_OUTPUT = Path("outputs/paper/minimol_retrieval_agent_results")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for split in args.splits or list(BENCHMARK_SPLITS):
        split_rows, split_missing = summarize_split(
            split,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        rows.extend(split_rows)
        missing.extend(split_missing)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_tsv(args.output_dir / "condition_results.tsv", rows)
    payload = {
        "type": "minimol_vs_morgan_agent_retrieval.v1",
        "comparison": {
            "fixed": (
                "Starling split, evidence sources, experiment modes, top-k, "
                "minimum similarity, identity policy, GLM model and inference settings"
            ),
            "changed": "retrieval ranking feature: Morgan/Tanimoto -> MiniMol/L2 cosine",
        },
        "splits": args.splits or list(BENCHMARK_SPLITS),
        "n_expected_conditions": 19 * len(args.splits or list(BENCHMARK_SPLITS)),
        "n_complete_conditions": len(rows),
        "missing_conditions": missing,
        "n_failed_morgan": sum(int(row["morgan_n_failed"]) for row in rows),
        "n_failed_minimol": sum(int(row["minimol_n_failed"]) for row in rows),
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 1 if missing or payload["n_failed_morgan"] or payload["n_failed_minimol"] else 0


def summarize_split(
    split: str,
    *,
    bootstrap_replicates: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    experiments = [
        experiment
        for experiment in experiments_for_starling_benchmark(split)
        if experiment.mode != "none"
    ]
    morgan_root = experiment_run_root(
        DEPLOYMENT_VISIBLE,
        PARENT_DISJOINT,
        paper_root=paper_root_for_benchmark_split(split),
    )
    minimol_root = experiment_run_root(
        DEPLOYMENT_VISIBLE,
        PARENT_DISJOINT,
        paper_root=paper_root_for_minimol_retrieval(split),
    )
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for experiment in experiments:
        morgan_batch = morgan_root / experiment.task / experiment.name
        minimol_batch = minimol_root / experiment.task / experiment.name
        required = (
            morgan_batch / "metrics.json",
            morgan_batch / "predictions.jsonl",
            minimol_batch / "metrics.json",
            minimol_batch / "predictions.jsonl",
        )
        if not all(path.exists() for path in required):
            missing.append(f"{split}:{experiment.name}")
            continue
        rows.append(
            summarize_condition(
                split,
                experiment,
                morgan_batch=morgan_batch,
                minimol_batch=minimol_batch,
                bootstrap_replicates=bootstrap_replicates,
            )
        )
    return rows, missing


def summarize_condition(
    split: str,
    experiment: Any,
    *,
    morgan_batch: Path,
    minimol_batch: Path,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    morgan_metrics = _read_json(morgan_batch / "metrics.json")
    minimol_metrics = _read_json(minimol_batch / "metrics.json")
    morgan = _prediction_map(morgan_batch / "predictions.jsonl")
    minimol = _prediction_map(minimol_batch / "predictions.jsonl")
    if morgan.keys() != minimol.keys():
        raise ValueError(f"Prediction index mismatch for {split}:{experiment.name}")
    indices = sorted(morgan)
    labels = [_gold_label(morgan[index]) for index in indices]
    if labels != [_gold_label(minimol[index]) for index in indices]:
        raise ValueError(f"Gold label mismatch for {split}:{experiment.name}")
    morgan_predictions = [int(morgan[index]["pred_label"]) for index in indices]
    minimol_predictions = [int(minimol[index]["pred_label"]) for index in indices]
    morgan_f1 = macro_f1(labels, morgan_predictions)
    minimol_f1 = macro_f1(labels, minimol_predictions)
    ci_low, ci_high = paired_bootstrap_delta_ci(
        labels,
        morgan_predictions,
        minimol_predictions,
        replicates=bootstrap_replicates,
        seed=20260728,
    )
    morgan_only_correct = sum(
        morgan_prediction == label and minimol_prediction != label
        for label, morgan_prediction, minimol_prediction in zip(
            labels, morgan_predictions, minimol_predictions
        )
    )
    minimol_only_correct = sum(
        minimol_prediction == label and morgan_prediction != label
        for label, morgan_prediction, minimol_prediction in zip(
            labels, morgan_predictions, minimol_predictions
        )
    )
    return {
        "split": split,
        "task": experiment.task,
        "experiment": experiment.name,
        "condition": experiment.name.split("__", 1)[1],
        "source": experiment.source,
        "mode": experiment.mode,
        "n_total": len(indices),
        "morgan_macro_f1": morgan_f1,
        "minimol_macro_f1": minimol_f1,
        "delta_macro_f1": minimol_f1 - morgan_f1,
        "delta_macro_f1_ci_low": ci_low,
        "delta_macro_f1_ci_high": ci_high,
        "morgan_accuracy": float(morgan_metrics["accuracy"]),
        "minimol_accuracy": float(minimol_metrics["accuracy"]),
        "delta_accuracy": (
            float(minimol_metrics["accuracy"]) - float(morgan_metrics["accuracy"])
        ),
        "morgan_only_correct": morgan_only_correct,
        "minimol_only_correct": minimol_only_correct,
        "mcnemar_exact_p": mcnemar_exact_p(
            morgan_only_correct,
            minimol_only_correct,
        ),
        "n_prediction_flips": sum(
            left != right
            for left, right in zip(morgan_predictions, minimol_predictions)
        ),
        "morgan_n_failed": int(morgan_metrics.get("n_failed_runs") or 0),
        "minimol_n_failed": int(minimol_metrics.get("n_failed_runs") or 0),
        "morgan_batch": str(morgan_batch),
        "minimol_batch": str(minimol_batch),
    }


def _prediction_map(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("pred_label") is None:
                continue
            rows[int(row["query_index"])] = row
    return rows


def _gold_label(row: dict[str, Any]) -> int:
    """Accept both historical summary fixtures and current batch prediction rows."""
    values = [row[key] for key in ("true_label", "label") if row.get(key) is not None]
    if not values:
        raise KeyError("Prediction row has neither 'true_label' nor 'label'")
    labels = {int(value) for value in values}
    if len(labels) != 1:
        raise ValueError("Prediction row has conflicting 'true_label' and 'label' values")
    return labels.pop()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _report(payload: dict[str, Any]) -> str:
    lines = [
        "# MiniMol vs Morgan retrieval in the Starling GLM agent",
        "",
        "Only the neighbor-ranking feature changes: Morgan/Tanimoto is replaced by "
        "L2-normalized MiniMol embedding cosine. Reported agent conditions use the "
        "same parent-disjoint identity policy.",
        "",
        "| Split | Task | Condition | Morgan macro-F1 | MiniMol macro-F1 | Delta | "
        "95% paired CI | Morgan acc. | MiniMol acc. | Failures |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['split']} | {row['task']} | {row['condition']} | "
            f"{row['morgan_macro_f1']:.4f} | {row['minimol_macro_f1']:.4f} | "
            f"{row['delta_macro_f1']:+.4f} | "
            f"[{row['delta_macro_f1_ci_low']:+.4f}, "
            f"{row['delta_macro_f1_ci_high']:+.4f}] | "
            f"{row['morgan_accuracy']:.4f} | {row['minimol_accuracy']:.4f} | "
            f"{row['morgan_n_failed']} / {row['minimol_n_failed']} |"
        )
    if payload["missing_conditions"]:
        lines.extend(
            [
                "",
                "Incomplete conditions: " + ", ".join(payload["missing_conditions"]),
            ]
        )
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="*", choices=BENCHMARK_SPLITS, default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
