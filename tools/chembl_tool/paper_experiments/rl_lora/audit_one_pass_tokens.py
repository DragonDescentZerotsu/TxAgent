"""Token-length and context-window audit for one-pass JSONL prompts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from tools.chembl_tool.common.json_utils import sha256_file, write_json_atomic

from .experiment_contract import ONE_PASS_TRAINING_RECIPE


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def audit(
    data: Path,
    output: Path,
    *,
    model: str,
    renderer_name: str,
    max_tokens: int,
    context_window: int,
) -> dict[str, object]:
    tokenizer = get_tokenizer(model)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    rows = [
        json.loads(line)
        for line in data.read_text(encoding="utf-8").splitlines()
        if line
    ]
    lengths = [renderer.build_generation_prompt(row["messages"]).length for row in rows]
    if not lengths:
        raise ValueError(f"empty data: {data}")
    over = [
        index
        for index, length in enumerate(lengths)
        if length + max_tokens > context_window
    ]
    result = {
        "contract": "one_pass_token_audit.v1",
        "status": "pass" if not over else "fail",
        "data": str(data),
        "data_sha256": sha256_file(data),
        "model": model,
        "renderer": renderer_name,
        "context_window": context_window,
        "reserved_completion_tokens": max_tokens,
        "n_rows": len(lengths),
        "prompt_tokens": {
            "total": sum(lengths),
            "min": min(lengths),
            "p50": _percentile(lengths, 0.50),
            "p90": _percentile(lengths, 0.90),
            "p95": _percentile(lengths, 0.95),
            "p99": _percentile(lengths, 0.99),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
        "n_context_overflow": len(over),
        "context_overflow_source_indices": over,
    }
    write_json_atomic(output, result)
    if over:
        raise ValueError(f"{len(over)} rows exceed the reserved context window")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-oss-120b:peft:131072")
    parser.add_argument("--renderer", default="gpt_oss_medium_reasoning")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=ONE_PASS_TRAINING_RECIPE.max_completion_tokens,
    )
    parser.add_argument("--context-window", type=int, default=131072)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            audit(
                args.data,
                args.output,
                model=args.model,
                renderer_name=args.renderer,
                max_tokens=args.max_tokens,
                context_window=args.context_window,
            ),
            indent=2,
        )
    )
