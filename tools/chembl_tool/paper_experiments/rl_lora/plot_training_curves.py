"""Audit and plot local RL reward, branch, loss, entropy, and KL curves."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from tools.chembl_tool.common.json_utils import write_json_atomic


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _raw_rollout_curves(log_path: Path) -> dict[str, list[float]]:
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(log_path.glob("iteration_*/train_rollout_summaries.jsonl")):
        for row in _read_jsonl(path):
            by_step[int(row["iteration"])].append(row)
    curves: dict[str, list[float]] = {
        "step": [],
        "reward_all_samples": [],
        "final_correct_all_samples": [],
        "single_correct_all_samples": [],
        "analog_correct_all_samples": [],
        "parsed_json_all_samples": [],
        "full_schema_all_samples": [],
    }
    for step, rows in sorted(by_step.items()):
        step_rows = [
            item
            for row in rows
            for item in (row.get("steps") or [])
            if isinstance(item, dict)
        ]
        if not step_rows:
            continue
        curves["step"].append(float(step))
        curves["reward_all_samples"].append(
            sum(float(item.get("reward") or 0.0) for item in step_rows) / len(step_rows)
        )
        for metric in (
            "final_correct",
            "single_correct",
            "analog_correct",
            "parsed_json",
            "full_schema",
        ):
            curves[f"{metric}_all_samples"].append(
                sum(
                    float((item.get("metrics") or {}).get(metric) or 0.0)
                    for item in step_rows
                )
                / len(step_rows)
            )
    return curves


def _series(rows: list[dict[str, Any]], *keys: str) -> tuple[list[float], list[float]]:
    steps = []
    values = []
    for row in rows:
        key = next((candidate for candidate in keys if candidate in row), None)
        if key is None:
            continue
        steps.append(float(row["step"]))
        values.append(float(row[key]))
    return steps, values


def plot(log_path: Path, output: Path, *, require_loss: bool) -> dict[str, Any]:
    metrics_path = log_path / "metrics.jsonl"
    if not metrics_path.exists():
        raise ValueError(f"metrics.jsonl is missing: {metrics_path}")
    metrics = _read_jsonl(metrics_path)
    if not metrics:
        raise ValueError(f"metrics.jsonl is empty: {metrics_path}")
    raw = _raw_rollout_curves(log_path)
    if not raw["step"]:
        raise ValueError(
            "rollout summaries are missing; all-sample reward cannot be audited"
        )
    loss_steps, loss_values = _series(metrics, "train/loss_sum")
    if require_loss and not loss_values:
        raise ValueError("train/loss_sum is missing from metrics.jsonl")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0][0]
    ax.plot(raw["step"], raw["reward_all_samples"], marker="o", label="all rollouts")
    opt_steps, opt_reward = _series(metrics, "env/all/reward/total")
    if opt_reward:
        ax.plot(opt_steps, opt_reward, marker="s", label="retained training rows")
    ax.set(title="Reward", xlabel="training step", ylabel="mean reward")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[0][1]
    for key, label in (
        ("final_correct_all_samples", "final"),
        ("single_correct_all_samples", "single"),
        ("analog_correct_all_samples", "analog"),
    ):
        ax.plot(raw["step"], raw[key], marker="o", label=label)
    ax.set(
        title="All-rollout branch accuracy",
        xlabel="training step",
        ylabel="fraction correct",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1][0]
    if loss_values:
        ax.plot(loss_steps, loss_values, marker="o", color="tab:red", label="loss:sum")
    ax.set(
        title="Training loss",
        xlabel="training step",
        ylabel="service-reported loss sum",
    )
    ax.grid(alpha=0.25)
    if loss_values:
        ax.legend()

    ax = axes[1][1]
    for keys, label in (
        (("optim/entropy", "sampling/entropy"), "entropy"),
        (("optim/kl_sample_train_v1", "kl/sample_train_v1"), "KL v1"),
        (("optim/kl_sample_train_v2", "kl/sample_train_v2"), "KL v2"),
    ):
        steps, values = _series(metrics, *keys)
        if values:
            ax.plot(steps, values, marker="o", label=label)
    ax.set(title="Policy diagnostics", xlabel="training step", ylabel="value")
    ax.grid(alpha=0.25)
    ax.legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    summary = {
        "contract": "txagent_rl_training_curves.v1",
        "log_path": str(log_path),
        "metrics_path": str(metrics_path),
        "figure": str(output),
        "n_metric_steps": len(metrics),
        "n_rollout_steps": len(raw["step"]),
        "loss_present": bool(loss_values),
        "curve_fields": sorted(
            key for key, values in raw.items() if key != "step" and values
        ),
        "last": {
            "step": metrics[-1].get("step"),
            "reward_all_samples": raw["reward_all_samples"][-1],
            "final_correct_all_samples": raw["final_correct_all_samples"][-1],
            "single_correct_all_samples": raw["single_correct_all_samples"][-1],
            "analog_correct_all_samples": raw["analog_correct_all_samples"][-1],
            "train_loss_sum": loss_values[-1] if loss_values else None,
        },
    }
    write_json_atomic(output.with_suffix(".json"), summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-loss", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            plot(args.log_path, args.output, require_loss=args.require_loss),
            indent=2,
        )
    )
