"""Pair each task's plotted best agent with every train-derived baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.chembl_tool.paper_experiments.plot_starling_benchmark_overview import (
    BASELINE_METHODS,
)


ROOT = Path(__file__).resolve().parents[3]
BASELINE_LABELS = {method.key: method.label for method in BASELINE_METHODS}
BASELINE_METHOD_KEYS = tuple(method.key for method in BASELINE_METHODS)
BASELINE_FAMILIES = frozenset(
    {"minimol", "structure_knn", "minimol_embedding_knn"}
)


def analyze(
    metrics_path: Path,
    output_prefix: Path,
    *,
    agent_metrics_path: Path | None = None,
    agent_model: str | None = None,
    permutation_replicates: int = 100_000,
    bootstrap_replicates: int = 10_000,
    permutation_seed: int = 29,
    bootstrap_seed: int = 23,
) -> list[dict[str, Any]]:
    metric_rows = _read_tsv(metrics_path)
    if (agent_metrics_path is None) != (agent_model is None):
        raise ValueError("agent_metrics_path and agent_model must be provided together")
    if agent_metrics_path is not None:
        metric_rows = _merge_experiment_agents(
            metric_rows,
            _read_tsv(agent_metrics_path),
            agent_model=agent_model,
        )
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in metric_rows:
        key = (
            row["benchmark_split"],
            row.get("evaluation_subset") or "test",
            row["task"],
        )
        groups.setdefault(key, []).append(row)

    output_rows: list[dict[str, Any]] = []
    for (split, subset, task), rows in sorted(groups.items()):
        agents = [
            row for row in rows if row["method_family"] == "molecular_evidence_agent"
        ]
        if not agents:
            raise ValueError(f"No agent rows for {split}/{subset}/{task}")
        best = max(agents, key=lambda row: (float(row["macro_f1"]), row["method"]))
        baselines = {row["method"]: row for row in rows if row["method"] in BASELINE_METHOD_KEYS}
        if set(baselines) != set(BASELINE_METHOD_KEYS):
            raise ValueError(
                f"Expected all current baselines for {split}/{subset}/{task}; "
                f"found {sorted(baselines)}"
            )
        agent_path = _prediction_path(best, subset=subset, agent=True)
        agent_predictions = _read_predictions(agent_path, agent=True)

        for baseline_method in BASELINE_METHOD_KEYS:
            baseline = baselines[baseline_method]
            if baseline["method_family"] not in BASELINE_FAMILIES:
                raise ValueError(
                    f"Unexpected baseline family for {baseline_method}: "
                    f"{baseline['method_family']}"
                )
            baseline_path = _prediction_path(baseline, subset=subset, agent=False)
            baseline_predictions = _read_predictions(baseline_path, agent=False)
            labels, agent_values, baseline_values = _align(
                agent_predictions,
                baseline_predictions,
                agent_path,
                baseline_path,
            )
            stats = paired_macro_f1_significance(
                labels,
                agent_values,
                baseline_values,
                permutation_replicates=permutation_replicates,
                bootstrap_replicates=bootstrap_replicates,
                permutation_seed=permutation_seed,
                bootstrap_seed=bootstrap_seed,
            )
            _validate_metric_value(best, stats["agent_macro_f1"], "agent")
            _validate_metric_value(baseline, stats["baseline_macro_f1"], "baseline")
            output_rows.append(
                {
                    "benchmark_split": split,
                    "evaluation_subset": subset,
                    "task": task,
                    "agent_model": best.get("model_label") or "Agent",
                    "agent_method": best["method"],
                    "baseline": BASELINE_LABELS[baseline_method],
                    "baseline_method": baseline_method,
                    **stats,
                    "alternative": "agent_greater_than_baseline",
                    "analysis_status": (
                        "exploratory_post_hoc_direction_and_valid_selected_agent"
                    ),
                    "agent_predictions": _portable_path(agent_path),
                    "baseline_predictions": _portable_path(baseline_path),
                    "molecule_and_label_alignment": True,
                }
            )

    adjusted = holm_adjust([float(row["p_value_one_sided"]) for row in output_rows])
    for row, value in zip(output_rows, adjusted, strict=True):
        row["p_value_one_sided_holm_all"] = value

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(output_prefix.with_suffix(".tsv"), output_rows)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema_version": "starling_best_agent_all_baselines.v1",
                "metrics": str(metrics_path),
                "agent_metrics": (
                    str(agent_metrics_path) if agent_metrics_path is not None else None
                ),
                "agent_model": agent_model,
                "hypothesis": "best_valid_agent_macro_f1_greater_than_baseline",
                "multiple_testing": "Holm adjustment across all plotted comparisons",
                "comparisons": output_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    output_prefix.with_suffix(".md").write_text(
        _markdown(output_rows), encoding="utf-8"
    )
    return output_rows


def _merge_experiment_agents(
    base_rows: list[dict[str, str]],
    experiment_rows: list[dict[str, str]],
    *,
    agent_model: str,
) -> list[dict[str, str]]:
    """Combine shared baselines with one plotted experiment-model family."""
    baselines = [row for row in base_rows if row["method"] in BASELINE_METHOD_KEYS]
    agents = [
        {
            **row,
            "method_family": "molecular_evidence_agent",
        }
        for row in experiment_rows
        if row.get("model_label") == agent_model
        and row.get("comparison_role", "").lower() not in {"anchor", "control"}
        and row["method"] != "morgan_standard"
    ]
    if not agents:
        raise ValueError(f"No experiment agent rows matched model {agent_model!r}")
    base_groups = {
        (row["benchmark_split"], row.get("evaluation_subset") or "test", row["task"])
        for row in baselines
    }
    agent_groups = {
        (row["benchmark_split"], row.get("evaluation_subset") or "test", row["task"])
        for row in agents
    }
    if agent_groups != base_groups:
        raise ValueError(
            "Experiment agent groups do not match baseline groups: "
            f"agent={sorted(agent_groups)}, baseline={sorted(base_groups)}"
        )
    return [*baselines, *agents]


def paired_macro_f1_significance(
    labels: np.ndarray,
    agent: np.ndarray,
    baseline: np.ndarray,
    *,
    permutation_replicates: int,
    bootstrap_replicates: int,
    permutation_seed: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if not permutation_replicates or not bootstrap_replicates:
        raise ValueError("Permutation and bootstrap replicate counts must be positive")
    agent_f1 = _macro_f1(labels, agent)
    baseline_f1 = _macro_f1(labels, baseline)
    observed = agent_f1 - baseline_f1

    rng = np.random.default_rng(permutation_seed)
    exceedances = 0
    remaining = permutation_replicates
    while remaining:
        size = min(2_000, remaining)
        swap = rng.integers(0, 2, size=(size, len(labels)), dtype=np.int8).astype(bool)
        randomized_agent = np.where(swap, baseline, agent)
        randomized_baseline = np.where(swap, agent, baseline)
        null_delta = _macro_f1_rows(labels[None, :], randomized_agent) - _macro_f1_rows(
            labels[None, :], randomized_baseline
        )
        exceedances += int(np.sum(null_delta >= observed - 1e-15))
        remaining -= size

    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_chunks: list[np.ndarray] = []
    remaining = bootstrap_replicates
    while remaining:
        size = min(1_000, remaining)
        indices = rng.integers(0, len(labels), size=(size, len(labels)))
        sampled_labels = labels[indices]
        bootstrap_chunks.append(
            _macro_f1_rows(sampled_labels, agent[indices])
            - _macro_f1_rows(sampled_labels, baseline[indices])
        )
        remaining -= size
    ci_low, ci_high = np.quantile(
        np.concatenate(bootstrap_chunks), (0.025, 0.975)
    )

    agent_correct = agent == labels
    baseline_correct = baseline == labels
    return {
        "n": len(labels),
        "agent_macro_f1": agent_f1,
        "baseline_macro_f1": baseline_f1,
        "delta_macro_f1_agent_minus_baseline": observed,
        "paired_ci95_low": float(ci_low),
        "paired_ci95_high": float(ci_high),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "agent_only_correct": int(np.sum(agent_correct & ~baseline_correct)),
        "baseline_only_correct": int(np.sum(~agent_correct & baseline_correct)),
        "both_correct": int(np.sum(agent_correct & baseline_correct)),
        "both_wrong": int(np.sum(~agent_correct & ~baseline_correct)),
        "ci_crosses_zero": bool(ci_low <= 0 <= ci_high),
        "p_value_one_sided": (exceedances + 1) / (permutation_replicates + 1),
        "permutation_replicates": permutation_replicates,
        "permutation_seed": permutation_seed,
    }


def holm_adjust(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, values[index] * (len(values) - rank)))
        adjusted[index] = running
    return adjusted


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"Metrics TSV is empty: {path}")
    return rows


def _prediction_path(row: dict[str, str], *, subset: str, agent: bool) -> Path:
    metrics_path = Path(row["metrics_path"])
    if not metrics_path.is_absolute():
        metrics_path = ROOT / metrics_path
    name = "predictions.jsonl" if agent else f"{subset}_predictions.jsonl"
    path = metrics_path.parent / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_predictions(path: Path, *, agent: bool) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        status = row.get("final_status", row.get("status")) if agent else row.get("status", "ok")
        if status != "ok":
            raise ValueError(f"Incomplete prediction in {path}: {row.get('query_index')}")
        molecule = str(row["smiles"] if agent else row["drug"])
        if molecule in result:
            raise ValueError(f"Duplicate molecule in {path}: {molecule}")
        result[molecule] = (
            int(row["label"] if agent else row["Y"]),
            int(row["pred_label"] if agent else row["prediction"]),
        )
    return result


def _align(
    agent: dict[str, tuple[int, int]],
    baseline: dict[str, tuple[int, int]],
    agent_path: Path,
    baseline_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if agent.keys() != baseline.keys():
        raise ValueError(f"Molecule set mismatch: {agent_path} vs {baseline_path}")
    molecules = sorted(agent)
    labels = np.asarray([agent[item][0] for item in molecules], dtype=np.int8)
    baseline_labels = np.asarray(
        [baseline[item][0] for item in molecules], dtype=np.int8
    )
    if not np.array_equal(labels, baseline_labels):
        raise ValueError(f"Gold-label mismatch: {agent_path} vs {baseline_path}")
    return (
        labels,
        np.asarray([agent[item][1] for item in molecules], dtype=np.int8),
        np.asarray([baseline[item][1] for item in molecules], dtype=np.int8),
    )


def _macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(_macro_f1_rows(labels[None, :], predictions[None, :])[0])


def _macro_f1_rows(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    tp = np.sum((labels == 1) & (predictions == 1), axis=1)
    tn = np.sum((labels == 0) & (predictions == 0), axis=1)
    fp = np.sum((labels == 0) & (predictions == 1), axis=1)
    fn = np.sum((labels == 1) & (predictions == 0), axis=1)
    f1_positive = np.divide(
        2 * tp,
        2 * tp + fp + fn,
        out=np.zeros_like(tp, dtype=float),
        where=(2 * tp + fp + fn) != 0,
    )
    f1_negative = np.divide(
        2 * tn,
        2 * tn + fp + fn,
        out=np.zeros_like(tn, dtype=float),
        where=(2 * tn + fp + fn) != 0,
    )
    return (f1_positive + f1_negative) / 2


def _validate_metric_value(row: dict[str, str], observed: float, role: str) -> None:
    if abs(float(row["macro_f1"]) - observed) > 5e-6:
        raise ValueError(
            f"{role} macro-F1 does not match metrics TSV: "
            f"{row['task']}/{row['method']}"
        )


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Best current agent versus all baselines",
        "",
        (
            "Exploratory one-sided paired permutation tests on identical samples; "
            "H1 is best valid-selected agent macro-F1 > baseline."
        ),
        "",
        "| Task | Best agent | Baseline | Delta macro-F1 | Raw p | Holm p |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['agent_method']} | {row['baseline']} | "
            f"{row['delta_macro_f1_agent_minus_baseline']:+.4f} | "
            f"{row['p_value_one_sided']:.4f} | "
            f"{row['p_value_one_sided_holm_all']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument(
        "--agent-metrics",
        type=Path,
        help="Optional plotted experiment-metrics TSV containing the target agent model.",
    )
    parser.add_argument(
        "--agent-model",
        help="Exact model_label selected from --agent-metrics.",
    )
    parser.add_argument("--permutation-replicates", type=int, default=100_000)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--permutation-seed", type=int, default=29)
    parser.add_argument("--bootstrap-seed", type=int, default=23)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = analyze(
        args.metrics,
        args.output_prefix,
        agent_metrics_path=args.agent_metrics,
        agent_model=args.agent_model,
        permutation_replicates=args.permutation_replicates,
        bootstrap_replicates=args.bootstrap_replicates,
        permutation_seed=args.permutation_seed,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps({"n_comparisons": len(rows), "output_prefix": str(args.output_prefix)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
