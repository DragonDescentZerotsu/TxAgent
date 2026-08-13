"""Paired valid-set inference for Starling direct agents versus train-label KNNs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from tools.chembl_tool.paper_experiments.molecular_evidence_agent import (
    PARENT_DISJOINT,
    experiment_run_root,
)


TASKS = (
    ("BBB_Martins", "bbb_martins", "bbb_martins__starling_direct"),
    (
        "Bioavailability_Ma",
        "bioavailability_ma",
        "bioavailability_ma__starling_direct_full",
    ),
    ("Skin_Reaction", "skin_reaction", "skin_reaction__starling_direct"),
)
BASELINES = (
    ("Morgan KNN", "structure_knn_starling_record_supported_v2"),
    ("MiniMol KNN", "minimol_embedding_knn_starling_record_supported_v2"),
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows: list[dict[str, Any]] = []
    integrity: dict[str, Any] = {}
    for agent_index, (agent_label, root_text, visibility) in enumerate(args.agent):
        root = Path(root_text)
        run_root = experiment_run_root(
            visibility,
            PARENT_DISJOINT,
            paper_root=root,
        )
        integrity[agent_label] = {}
        for task_index, (data_name, task, experiment) in enumerate(TASKS):
            agent_path = run_root / task / experiment / "predictions.jsonl"
            agent = _read_predictions(agent_path, agent=True)
            integrity[agent_label][data_name] = {
                "agent_predictions": str(agent_path),
                "n": len(agent),
                "visibility_mode": visibility,
                "baselines": {},
            }
            for baseline_index, (baseline_label, baseline_dir) in enumerate(BASELINES):
                baseline_path = (
                    args.baseline_root
                    / baseline_dir
                    / data_name
                    / "scaffold/valid_predictions.jsonl"
                )
                baseline = _read_predictions(baseline_path, agent=False)
                y, left, right = _align(agent, baseline, agent_path, baseline_path)
                seed_offset = agent_index * 100 + task_index * 10 + baseline_index
                row = _paired_comparison(
                    y,
                    left,
                    right,
                    permutation_replicates=args.permutation_replicates,
                    bootstrap_replicates=args.bootstrap_replicates,
                    permutation_seed=args.permutation_seed + seed_offset,
                    bootstrap_seed=args.bootstrap_seed + seed_offset,
                )
                rows.append(
                    {
                        "agent": agent_label,
                        "visibility_mode": visibility,
                        "task": data_name,
                        "comparison": f"{agent_label} Direct vs {baseline_label}",
                        "baseline": baseline_label,
                        **row,
                    }
                )
                integrity[agent_label][data_name]["baselines"][baseline_label] = {
                    "predictions": str(baseline_path),
                    "n": len(baseline),
                    "molecule_and_label_alignment": True,
                }

    for agent_label, _, _ in args.agent:
        family = [row for row in rows if row["agent"] == agent_label]
        _add_holm(
            family,
            "macro_f1_paired_permutation_p_raw",
            "macro_f1_permutation_p_holm",
        )
        _add_holm(
            family,
            "accuracy_mcnemar_exact_p_raw",
            "accuracy_mcnemar_exact_p_holm",
        )
    for row in rows:
        row["macro_f1_significant_holm_0_05"] = row["macro_f1_permutation_p_holm"] < 0.05
        row["accuracy_significant_holm_0_05"] = row["accuracy_mcnemar_exact_p_holm"] < 0.05

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "starling.record_supported_v2_direct_vs_knn_significance.v1",
        "split": "scaffold",
        "evaluation_subset": "valid",
        "methodology": {
            "primary_metric": "binary macro-F1",
            "macro_f1_test": "two-sided sample-level paired randomization",
            "macro_f1_ci": "sample-level paired bootstrap percentile 95% CI",
            "accuracy_test": "two-sided exact McNemar",
            "multiple_testing": (
                "Holm adjustment separately within each six-comparison agent series, "
                "and separately for macro-F1 and accuracy"
            ),
            "permutation_replicates": args.permutation_replicates,
            "bootstrap_replicates": args.bootstrap_replicates,
        },
        "integrity": integrity,
        "comparisons": rows,
    }
    (args.output_prefix.with_suffix(".json")).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_tsv(args.output_prefix.with_suffix(".tsv"), rows)
    (args.output_prefix.with_suffix(".md")).write_text(
        _markdown(rows, args), encoding="utf-8"
    )
    print(json.dumps({"output_prefix": str(args.output_prefix), "n_comparisons": len(rows)}, indent=2))
    return 0


def _read_predictions(path: Path, *, agent: bool) -> dict[str, tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(path)
    result: dict[str, tuple[int, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        status = row.get("status") if not agent else row.get("final_status", row.get("status"))
        if status != "ok":
            raise ValueError(f"Incomplete prediction in {path}: query_index={row.get('query_index')}")
        molecule = str(row.get("smiles") if agent else row.get("drug"))
        label = int(row.get("label") if agent else row.get("Y"))
        prediction = int(row.get("pred_label") if agent else row.get("prediction"))
        if molecule in result:
            raise ValueError(f"Duplicate molecule in {path}: {molecule}")
        result[molecule] = (label, prediction)
    return result


def _align(
    left: dict[str, tuple[int, int]],
    right: dict[str, tuple[int, int]],
    left_path: Path,
    right_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if left.keys() != right.keys():
        raise ValueError(f"Molecule set mismatch: {left_path} vs {right_path}")
    molecules = sorted(left)
    left_labels = np.asarray([left[item][0] for item in molecules], dtype=np.int8)
    right_labels = np.asarray([right[item][0] for item in molecules], dtype=np.int8)
    if not np.array_equal(left_labels, right_labels):
        raise ValueError(f"Gold-label mismatch: {left_path} vs {right_path}")
    return (
        left_labels,
        np.asarray([left[item][1] for item in molecules], dtype=np.int8),
        np.asarray([right[item][1] for item in molecules], dtype=np.int8),
    )


def _macro_f1(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(_macro_f1_rows(y[None, :], prediction[None, :])[0])


def _macro_f1_rows(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    tp = np.sum((y == 1) & (prediction == 1), axis=1)
    tn = np.sum((y == 0) & (prediction == 0), axis=1)
    fp = np.sum((y == 0) & (prediction == 1), axis=1)
    fn = np.sum((y == 1) & (prediction == 0), axis=1)
    f1_pos = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) != 0)
    f1_neg = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros_like(tn, dtype=float), where=(2 * tn + fp + fn) != 0)
    return (f1_pos + f1_neg) / 2


def _paired_comparison(
    y: np.ndarray,
    agent: np.ndarray,
    baseline: np.ndarray,
    *,
    permutation_replicates: int,
    bootstrap_replicates: int,
    permutation_seed: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    agent_f1 = _macro_f1(y, agent)
    baseline_f1 = _macro_f1(y, baseline)
    observed = agent_f1 - baseline_f1
    permutation_rng = np.random.default_rng(permutation_seed)
    exceedances = 0
    remaining = permutation_replicates
    while remaining:
        size = min(2000, remaining)
        swap = permutation_rng.integers(0, 2, size=(size, len(y)), dtype=np.int8).astype(bool)
        left = np.where(swap, baseline, agent)
        right = np.where(swap, agent, baseline)
        null = _macro_f1_rows(y[None, :], left) - _macro_f1_rows(y[None, :], right)
        exceedances += int(np.sum(np.abs(null) >= abs(observed) - 1e-15))
        remaining -= size
    permutation_p = (exceedances + 1) / (permutation_replicates + 1)

    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    bootstrap_values: list[np.ndarray] = []
    remaining = bootstrap_replicates
    while remaining:
        size = min(1000, remaining)
        indices = bootstrap_rng.integers(0, len(y), size=(size, len(y)))
        sampled_y = y[indices]
        bootstrap_values.append(
            _macro_f1_rows(sampled_y, agent[indices])
            - _macro_f1_rows(sampled_y, baseline[indices])
        )
        remaining -= size
    ci_low, ci_high = np.quantile(np.concatenate(bootstrap_values), (0.025, 0.975))

    agent_correct = agent == y
    baseline_correct = baseline == y
    agent_only = int(np.sum(agent_correct & ~baseline_correct))
    baseline_only = int(np.sum(~agent_correct & baseline_correct))
    mcnemar_p = (
        1.0
        if agent_only + baseline_only == 0
        else float(binomtest(agent_only, agent_only + baseline_only, 0.5).pvalue)
    )
    return {
        "n": len(y),
        "agent_macro_f1": agent_f1,
        "baseline_macro_f1": baseline_f1,
        "delta_macro_f1_agent_minus_baseline": observed,
        "delta_macro_f1_bootstrap_95ci_low": float(ci_low),
        "delta_macro_f1_bootstrap_95ci_high": float(ci_high),
        "macro_f1_paired_permutation_p_raw": permutation_p,
        "permutation_replicates": permutation_replicates,
        "permutation_seed": permutation_seed,
        "agent_accuracy": float(np.mean(agent_correct)),
        "baseline_accuracy": float(np.mean(baseline_correct)),
        "delta_accuracy_agent_minus_baseline": float(np.mean(agent_correct) - np.mean(baseline_correct)),
        "agent_only_correct": agent_only,
        "baseline_only_correct": baseline_only,
        "both_correct": int(np.sum(agent_correct & baseline_correct)),
        "both_wrong": int(np.sum(~agent_correct & ~baseline_correct)),
        "accuracy_mcnemar_exact_p_raw": mcnemar_p,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
    }


def _add_holm(rows: list[dict[str, Any]], source: str, target: str) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: float(item[1][source]))
    running = 0.0
    adjusted = [1.0] * len(rows)
    for rank, (index, row) in enumerate(ordered):
        running = max(running, min(1.0, float(row[source]) * (len(rows) - rank)))
        adjusted[index] = running
    for row, value in zip(rows, adjusted, strict=True):
        row[target] = value


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt_p(value: float) -> str:
    return f"{value:.3g}" if value < 0.001 else f"{value:.3f}"


def _markdown(rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    lines = [
        "# Direct agents vs KNN on record-supported v2 scaffold valid",
        "",
        (
            "Two-sided paired tests; delta is agent minus KNN. Holm correction is "
            "applied within each agent series, separately across its six macro-F1 and "
            "six accuracy comparisons."
        ),
        "",
        "| Agent | Task | Baseline | n | Delta macro-F1 (95% CI) | Permutation p / Holm p | Delta accuracy | Agent-only / KNN-only | McNemar p / Holm p |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['agent']} | {row['task']} | {row['baseline']} | {row['n']} | "
            f"{row['delta_macro_f1_agent_minus_baseline']:+.4f} "
            f"[{row['delta_macro_f1_bootstrap_95ci_low']:+.4f}, "
            f"{row['delta_macro_f1_bootstrap_95ci_high']:+.4f}] | "
            f"{_fmt_p(row['macro_f1_paired_permutation_p_raw'])} / "
            f"{_fmt_p(row['macro_f1_permutation_p_holm'])} | "
            f"{row['delta_accuracy_agent_minus_baseline']:+.4f} | "
            f"{row['agent_only_correct']} / {row['baseline_only_correct']} | "
            f"{_fmt_p(row['accuracy_mcnemar_exact_p_raw'])} / "
            f"{_fmt_p(row['accuracy_mcnemar_exact_p_holm'])} |"
        )
    lines.extend(
        [
            "",
            (
                f"Macro-F1 used {args.permutation_replicates:,} paired randomizations "
                f"and {args.bootstrap_replicates:,} paired bootstrap resamples."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent",
        nargs=3,
        action="append",
        metavar=("LABEL", "PAPER_ROOT", "VISIBILITY_MODE"),
        required=True,
    )
    parser.add_argument("--baseline-root", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--permutation-replicates", type=int, default=100_000)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--permutation-seed", type=int, default=29)
    parser.add_argument("--bootstrap-seed", type=int, default=23)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
