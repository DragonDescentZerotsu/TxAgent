"""Summarize one-pass predictions and compare them with aligned three-stage traces."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.json_utils import (
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.paper_experiments.paired_binary_predictions import (
    paired_binary_summary,
)
from tools.chembl_tool.paper_experiments.summarize_results import macro_f1

from .one_pass_contract import TASK_CONTRACTS


def _three_stage_predictions(root: Path, task: str) -> dict[int, int]:
    contract = TASK_CONTRACTS[task]
    mapping = {contract.negative_value: 0, contract.positive_value: 1}
    outputs = {}
    for path in sorted(root.rglob("trace_messages.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        final = [row for row in rows if row.get("task") == "final_summary"]
        if len(final) != 1:
            raise ValueError(f"{path}: expected one final_summary")
        prediction = mapping.get(str(final[0].get("prediction") or ""))
        if prediction is None:
            raise ValueError(f"{path}: invalid three-stage prediction")
        outputs[int(final[0]["index"])] = prediction
    return outputs


def summarize(
    *,
    results_root: Path,
    task: str,
    three_stage_root: Path | None,
    output: Path,
) -> dict[str, Any]:
    rows = []
    for path in sorted((results_root / "runs").glob("idx*/result.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise ValueError("no one-pass result rows")
    rows.sort(key=lambda row: int(row["source_index"]))
    indices = [int(row["source_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate one-pass result indices")
    source_tasks = {str(row.get("source_task") or "") for row in rows}
    if source_tasks != {task}:
        raise ValueError(f"result task mismatch: expected {task}, got {source_tasks}")
    source_subsets = {str(row.get("source_subset") or "") for row in rows}
    if len(source_subsets) != 1:
        raise ValueError(f"mixed one-pass result subsets: {source_subsets}")
    inference_hashes = {
        str(row.get("inference_contract_sha256") or "legacy_unbound") for row in rows
    }
    if len(inference_hashes) != 1:
        raise ValueError(f"mixed one-pass inference contracts: {inference_hashes}")
    successful = [row for row in rows if row.get("status") == "ok"]
    if not successful:
        raise ValueError("no successful one-pass result rows")

    def predictions(row: dict[str, Any]) -> dict[str, int | None]:
        direct = row.get("predictions")
        if isinstance(direct, dict):
            return direct
        reward = row.get("reward") or {}
        nested = reward.get("predictions")
        if isinstance(nested, dict):
            return nested
        raise ValueError(f"result index {row.get('source_index')} has no predictions")

    labels = [int(row["gold_label"]) for row in successful]
    final_predictions = [int(predictions(row)["final"]) for row in successful]
    single_predictions = [int(predictions(row)["single"]) for row in successful]
    analog_predictions = [int(predictions(row)["analog"]) for row in successful]

    def accuracy(predictions: list[int]) -> float:
        return sum(a == b for a, b in zip(labels, predictions, strict=True)) / len(
            labels
        )

    def class_recall(predictions: list[int]) -> dict[str, float | None]:
        output: dict[str, float | None] = {}
        for label in (0, 1):
            matching = [
                prediction
                for gold, prediction in zip(labels, predictions, strict=True)
                if gold == label
            ]
            output[str(label)] = (
                sum(prediction == label for prediction in matching) / len(matching)
                if matching
                else None
            )
        return output

    conflicts = [
        row
        for row in successful
        if row.get("branches_disagree")
        or (row.get("reward") or {}).get("branches_disagree")
    ]
    conflict_correct = sum(
        int(predictions(row)["final"]) == int(row["gold_label"]) for row in conflicts
    )
    result = {
        "task": task,
        "source_subset": next(iter(source_subsets)),
        "inference_contract_sha256": next(iter(inference_hashes)),
        "results_root": str(results_root),
        "n_total": len(rows),
        "n_successful": len(successful),
        "n_failed": len(rows) - len(successful),
        "schema_success_rate": len(successful) / len(rows),
        "one_pass": {
            "accuracy": accuracy(final_predictions),
            "macro_f1": macro_f1(labels, final_predictions),
            "class_recall": class_recall(final_predictions),
            "prediction_counts": dict(sorted(Counter(final_predictions).items())),
        },
        "single_branch": {
            "accuracy": accuracy(single_predictions),
            "macro_f1": macro_f1(labels, single_predictions),
            "class_recall": class_recall(single_predictions),
        },
        "analog_branch": {
            "accuracy": accuracy(analog_predictions),
            "macro_f1": macro_f1(labels, analog_predictions),
            "class_recall": class_recall(analog_predictions),
        },
        "branch_conflicts": {
            "n": len(conflicts),
            "final_correct": conflict_correct,
            "final_accuracy": conflict_correct / len(conflicts) if conflicts else None,
        },
        "attempt_counts": dict(
            sorted(Counter(int(row.get("attempt_count") or 0) for row in rows).items())
        ),
        "tokens": {
            "prompt": sum(
                int(attempt.get("prompt_tokens") or 0)
                for row in rows
                for attempt in row.get("attempts") or []
            ),
            "prompt_cache_hit": sum(
                int(attempt.get("prompt_cache_hit_tokens") or 0)
                for row in rows
                for attempt in row.get("attempts") or []
            ),
            "completion": sum(
                int(attempt.get("completion_tokens") or 0)
                for row in rows
                for attempt in row.get("attempts") or []
            ),
            "selected_response_usage": dict(
                sorted(
                    Counter(
                        {
                            key: sum(
                                int(
                                    (
                                        (row.get("response") or {}).get("usage") or {}
                                    ).get(key)
                                    or 0
                                )
                                for row in rows
                            )
                            for key in {
                                key
                                for row in rows
                                for key in (
                                    (row.get("response") or {}).get("usage") or {}
                                )
                            }
                        }
                    ).items()
                )
            ),
        },
    }
    manifest_path = results_root / "manifest.json"
    if manifest_path.exists():
        result["manifest"] = str(manifest_path)
        result["manifest_sha256"] = sha256_file(manifest_path)
    if three_stage_root is not None:
        old = _three_stage_predictions(three_stage_root, task)
        old_aligned = [old[int(row["source_index"])] for row in successful]
        result["paired_three_stage_to_one_pass"] = paired_binary_summary(
            labels,
            old_aligned,
            final_predictions,
        )
    write_json_atomic(output, result)
    write_jsonl_atomic(output.with_name("predictions.jsonl"), successful)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASK_CONTRACTS), required=True)
    parser.add_argument("--three-stage-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            summarize(
                results_root=args.results_root,
                task=args.task,
                three_stage_root=args.three_stage_root,
                output=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )
