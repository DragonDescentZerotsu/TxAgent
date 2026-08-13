"""Historical trace-replay one-pass materializer.

This reconstructs aligned architecture-comparison prompts from completed
three-stage traces. New RL train/valid/test datasets must use
``materialize_split_one_pass.py`` so retrieval and tool inputs are built
directly without prior assistant generations.
"""

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

from .one_pass import build_one_pass_row_from_trace
from .one_pass_contract import CONTRACT_VERSION, TASK_CONTRACTS


MATERIALIZER_VERSION = "one_pass_trace_replay_materializer.v1"


def _group_template(paths: list[Path]) -> dict[str, Any]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        for row in rows:
            if row.get("task") != "Flat.all_evidence":
                continue
            user_messages = []
            for message in row.get("messages") or []:
                if message.get("role") != "user":
                    continue
                try:
                    payload = json.loads(str(message.get("content") or ""))
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and isinstance(
                    payload.get("required_json_schema"), dict
                ):
                    user_messages.append(payload)
            if len(user_messages) == 1:
                payload = user_messages[0]
                return {
                    "task": payload.get("task"),
                    "instructions": payload.get("instructions") or [],
                    "required_json_schema": payload.get("required_json_schema") or {},
                }
    raise ValueError(
        "no successful Flat.all_evidence prompt is available as a template"
    )


def materialize_one_pass(
    *,
    trace_root: Path,
    task: str,
    subset: str,
    output: Path,
) -> dict[str, Any]:
    paths = sorted(trace_root.rglob("trace_messages.jsonl"))
    if not paths:
        raise ValueError(f"no trace_messages.jsonl under {trace_root}")
    group_template = _group_template(paths)
    rows = [
        build_one_pass_row_from_trace(
            path,
            task=task,
            subset=subset,
            group_template=group_template,
        )
        for path in paths
    ]
    indices = [int(row["source_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate source indices")
    rows.sort(key=lambda row: int(row["source_index"]))
    write_jsonl_atomic(output, rows)
    labels = Counter(int(row["gold_label"]) for row in rows)
    manifest = {
        "materializer_version": MATERIALIZER_VERSION,
        "contract_version": CONTRACT_VERSION,
        "task": task,
        "subset": subset,
        "n_rows": len(rows),
        "label_counts": {str(key): value for key, value in sorted(labels.items())},
        "trace_root": str(trace_root),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "source_assistant_outputs_in_prompt": False,
        "formal_rl_training_eligible": False,
        "gold_location": "environment_private_metadata_only",
    }
    write_json_atomic(output.with_suffix(".manifest.json"), manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASK_CONTRACTS), required=True)
    parser.add_argument("--subset", choices=("train", "valid", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = materialize_one_pass(
        trace_root=args.trace_root,
        task=args.task,
        subset=args.subset,
        output=args.output,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
