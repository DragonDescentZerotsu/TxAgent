"""Select one frozen one-pass checkpoint from valid metrics only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.json_utils import sha256_file, write_json_atomic


def _parse_candidate(value: str) -> dict[str, Any]:
    parts = value.split(",", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "candidate must be STEP,CHECKPOINT,METRICS_JSON"
        )
    step_text, checkpoint, metrics_text = parts
    metrics_path = Path(metrics_text)
    if not metrics_path.exists():
        raise argparse.ArgumentTypeError(f"metrics do not exist: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    one_pass = metrics.get("one_pass") or {}
    return {
        "step": int(step_text),
        "checkpoint": checkpoint,
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "macro_f1": float(one_pass["macro_f1"]),
        "accuracy": float(one_pass["accuracy"]),
        "n": int(metrics["n_successful"]),
        "n_failed": int(metrics["n_failed"]),
    }


def select(candidates: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    sample_counts = {candidate["n"] for candidate in candidates}
    if len(sample_counts) != 1:
        raise ValueError(f"candidate valid sample counts differ: {sample_counts}")
    if any(candidate["n_failed"] for candidate in candidates):
        raise ValueError("checkpoint selection requires zero failed valid rows")
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -candidate["macro_f1"],
            -candidate["accuracy"],
            candidate["step"],
        ),
    )
    result = {
        "selection_contract": "valid_macro_f1_then_accuracy_then_earlier_step.v1",
        "selected": ordered[0],
        "candidates": sorted(candidates, key=lambda candidate: candidate["step"]),
        "test_read_or_used": False,
    }
    write_json_atomic(output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_candidate,
        required=True,
        help="STEP,CHECKPOINT,METRICS_JSON; repeat for each frozen checkpoint.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(select(args.candidate, args.output), indent=2, sort_keys=True))
