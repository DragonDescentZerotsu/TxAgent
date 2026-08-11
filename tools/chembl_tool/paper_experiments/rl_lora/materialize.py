"""Materialize final-synthesis trace prompts without exposing gold labels to the LLM."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

from tools.chembl_tool.common.json_utils import (
    canonical_json_bytes,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


CONTRACT_VERSION = "starling_final_synthesis_grpo.v1"
TASK_CONTRACTS = {
    "bbb_martins": {
        "prediction_field": "bbb_prediction",
        "negative_value": "fail",
        "positive_value": "pass",
    },
    "bioavailability_ma": {
        "prediction_field": "bioavailability_prediction",
        "negative_value": "low",
        "positive_value": "high",
    },
    "skin_reaction": {
        "prediction_field": "skin_reaction_prediction",
        "negative_value": "no_risk",
        "positive_value": "risk",
    },
}


def _required_fields(messages: list[dict[str, Any]]) -> list[str]:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        try:
            payload = json.loads(str(message.get("content") or ""))
        except json.JSONDecodeError:
            continue
        schema = (
            payload.get("required_json_schema") if isinstance(payload, dict) else None
        )
        if isinstance(schema, dict) and schema:
            return [str(field) for field in schema]
    raise ValueError("final prompt does not contain required_json_schema")


def load_final_trace(path: Path, *, task: str, fold: int) -> dict[str, Any]:
    contract = TASK_CONTRACTS[task]
    final_rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("task") == "final_summary":
                final_rows.append(row)
    if len(final_rows) != 1:
        raise ValueError(
            f"{path}: expected exactly one final_summary row, got {len(final_rows)}"
        )
    row = final_rows[0]
    if row.get("status") != "ok":
        raise ValueError(f"{path}: final status is not ok")
    label = int(row["label"])
    if label not in {0, 1}:
        raise ValueError(f"{path}: non-binary label {label}")

    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError(f"{path}: malformed final messages")
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"{path}: final trace does not end with assistant output")
    prompt_messages = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages[:-1]
    ]
    if any(message["role"] == "assistant" for message in prompt_messages):
        raise ValueError(f"{path}: assistant content appears inside the prompt")
    required_fields = _required_fields(prompt_messages)
    if contract["prediction_field"] not in required_fields:
        raise ValueError(f"{path}: prediction field absent from required schema")

    return {
        "messages": prompt_messages,
        "gold_label": label,
        "prediction_field": contract["prediction_field"],
        "negative_value": contract["negative_value"],
        "positive_value": contract["positive_value"],
        "required_fields": required_fields,
        "source_task": task,
        "source_fold": fold,
        "source_index": int(row["index"]),
        "source_trace": str(path),
        "source_trace_sha256": sha256_file(path),
        "prompt_sha256": hashlib.sha256(
            canonical_json_bytes(prompt_messages)
        ).hexdigest(),
        "contract_version": CONTRACT_VERSION,
    }


def materialize(
    trace_roots: Iterable[Path],
    *,
    task: str,
    fold: int,
    output: Path,
    limit: int | None,
    seed: int,
    formal_eligible: bool,
) -> dict[str, Any]:
    paths = sorted(
        path for root in trace_roots for path in root.rglob("trace_messages.jsonl")
    )
    if not paths:
        raise ValueError("no trace_messages.jsonl files found")
    if limit is not None and limit < len(paths):
        paths = random.Random(seed).sample(paths, limit)
        paths.sort()
    rows = [load_final_trace(path, task=task, fold=fold) for path in paths]
    duplicate_prompts = len(rows) - len({row["prompt_sha256"] for row in rows})
    if duplicate_prompts:
        raise ValueError(f"duplicate prompts detected: {duplicate_prompts}")

    write_jsonl_atomic(output, rows)
    labels = Counter(int(row["gold_label"]) for row in rows)
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "task": task,
        "fold": fold,
        "formal_eligible": formal_eligible,
        "n_rows": len(rows),
        "label_counts": {str(label): count for label, count in sorted(labels.items())},
        "trace_roots": [str(path) for path in trace_roots],
        "output": str(output),
        "output_sha256": sha256_file(output),
        "selection_seed": seed,
        "limit": limit,
        "gold_location": "extra_env_info_only",
        "assistant_outputs_removed": True,
    }
    write_json_atomic(output.with_suffix(".manifest.json"), manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", action="append", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASK_CONTRACTS), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--formal-eligible", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = materialize(
        args.trace_root,
        task=args.task,
        fold=args.fold,
        output=args.output,
        limit=args.limit,
        seed=args.seed,
        formal_eligible=args.formal_eligible,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
