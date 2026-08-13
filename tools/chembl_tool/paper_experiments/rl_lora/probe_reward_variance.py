"""Probe frozen training prompts for rollout reward variance before a GRPO smoke."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from tools.chembl_tool.paper_experiments.rl_lora.reward import score_response


def _probe_row(
    row: dict[str, Any],
    *,
    base_url: str,
    model: str,
    samples: int,
    max_tokens: int,
    temperature: float,
    api_key: str,
) -> dict[str, Any]:
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=row["messages"],
        n=samples,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    scores = [
        score_response(choice.message.content or "", row) for choice in response.choices
    ]
    return {
        "source_task": row["source_task"],
        "source_index": row["source_index"],
        "gold_label": row["gold_label"],
        "n": len(scores),
        "n_correct": sum(score.correct for score in scores),
        "n_parsed_json": sum(score.parsed_json for score in scores),
        "n_full_schema": sum(score.full_schema for score in scores),
        "rewards": [score.reward for score in scores],
        "completion_tokens": response.usage.completion_tokens
        if response.usage
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="append", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:9001/v1")
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--api-key-env", default="GPT_OSS_LOCAL_API_KEY")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise ValueError(f"missing API key environment variable: {args.api_key_env}")

    rows = [
        json.loads(line) for path in args.data for line in path.open(encoding="utf-8")
    ]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _probe_row,
                row,
                base_url=args.base_url,
                model=args.model,
                samples=args.samples,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                api_key=api_key,
            ): row
            for row in rows
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: (row["source_task"], row["source_index"]))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
