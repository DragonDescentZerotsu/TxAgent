"""Evaluate a Tinker sampler checkpoint on frozen one-pass rows."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from tools.chembl_tool.common.json_utils import sha256_file, write_json_atomic
from .experiment_contract import ONE_PASS_TRAINING_RECIPE
from .one_pass_runtime import (
    inference_contract_sha256,
    load_fresh_one_pass_audit,
    one_pass_result_path,
    one_pass_validation_errors,
    read_one_pass_rows,
)
from .training_contract import score_training_response
from .reward import parse_json_object


MODEL = "openai/gpt-oss-120b:peft:131072"
RENDERER = "gpt_oss_medium_reasoning"
EVALUATOR_VERSION = "one_pass_tinker_evaluator.v4"


def _valid_existing(
    path: Path,
    row: dict[str, Any],
    inference_sha256: str,
) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        result.get("status") == "ok"
        and result.get("evaluator_version") == EVALUATOR_VERSION
        and result.get("inference_contract_sha256") == inference_sha256
        and result.get("prompt_sha256") == row.get("prompt_sha256")
        and (result.get("reward") or {}).get("predictions", {}).get("final") in {0, 1}
    )


async def _evaluate_row(
    row: dict[str, Any],
    *,
    sampling_client: Any,
    renderer: renderers.Renderer,
    checkpoint: str,
    output_root: Path,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    max_attempts: int,
    temperature: float,
    seed: int,
    inference_sha256: str,
) -> dict[str, Any]:
    attempts = []
    messages = list(row["messages"])
    async with semaphore:
        for attempt_index in range(max_attempts):
            if attempt_index:
                messages = [
                    *row["messages"],
                    {
                        "role": "user",
                        "content": (
                            f"Retry the one_pass_full_flat response (attempt {attempt_index + 1} "
                            f"of {max_attempts}). Return one complete compact JSON object. "
                            f"Validation errors: {', '.join(attempts[-1]['errors'])}."
                        ),
                    },
                ]
            prompt = renderer.build_generation_prompt(messages)
            sampled = await sampling_client.sample_async(
                prompt=prompt,
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=renderer.get_stop_sequences(),
                    seed=seed + int(row["source_index"]) + attempt_index,
                ),
            )
            sequence = sampled.sequences[0]
            message, termination = renderer.parse_response(sequence.tokens)
            response_text = renderers.get_text_content(message)
            content = parse_json_object(response_text)
            errors = one_pass_validation_errors(content, row)
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "errors": errors,
                    "response_text": response_text,
                    "content": content,
                    "prompt_tokens": prompt.length,
                    "completion_tokens": len(sequence.tokens),
                    "prompt_cache_hit_tokens": sampled.prompt_cache_hit_tokens,
                    "stop_reason": str(sequence.stop_reason),
                    "clean_stop": bool(termination.is_clean),
                }
            )
            if not errors:
                break
    selected = attempts[-1]
    reward = score_training_response(selected["response_text"], row)
    result = {
        "evaluator_version": EVALUATOR_VERSION,
        "inference_contract_sha256": inference_sha256,
        "status": "ok" if not selected["errors"] else "error",
        "checkpoint": checkpoint,
        "source_task": row["source_task"],
        "source_subset": row["source_subset"],
        "source_index": row["source_index"],
        "gold_label": row["gold_label"],
        "prompt_sha256": row["prompt_sha256"],
        "attempt_count": len(attempts),
        "attempts": attempts,
        "reward": reward.as_dict(),
    }
    write_json_atomic(one_pass_result_path(output_root, row), result)
    return result


async def async_main(args: argparse.Namespace) -> None:
    rows = read_one_pass_rows(args.data)
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    data_audit = load_fresh_one_pass_audit(args.data)
    checkpoint = args.checkpoint or f"base_model:{args.model}"
    inference_contract = {
        "evaluator_version": EVALUATOR_VERSION,
        "provider": "tinker",
        "checkpoint": checkpoint,
        "model": args.model,
        "renderer": args.renderer,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_attempts": args.max_attempts,
        "seed": args.seed,
        "validation": "one_pass_nested_schema.v1",
        "tool_calls": "disabled_harness_prefetched_only",
    }
    inference_sha256 = inference_contract_sha256(inference_contract)
    manifest = {
        "evaluator_version": EVALUATOR_VERSION,
        "contract_version": rows[0]["contract_version"],
        "data": str(args.data),
        "data_sha256": sha256_file(args.data),
        "data_audit": data_audit["audit_path"],
        "data_audit_sha256": data_audit["audit_sha256"],
        "inference_contract": inference_contract,
        "inference_contract_sha256": inference_sha256,
        "inference_source": "checkpoint" if args.checkpoint else "frozen_base_model",
        "parallelism": args.parallelism,
        "n_rows": len(rows),
    }
    write_json_atomic(args.output_root / "manifest.json", manifest)
    pending = [
        row
        for row in rows
        if not _valid_existing(
            one_pass_result_path(args.output_root, row), row, inference_sha256
        )
    ]
    print(
        f"tinker eval total={len(rows)} existing={len(rows) - len(pending)} pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return
    service_client = tinker.ServiceClient(api_key=os.environ.get(args.api_key_env))
    sampling_client = (
        service_client.create_sampling_client(model_path=args.checkpoint)
        if args.checkpoint
        else service_client.create_sampling_client(base_model=args.model)
    )
    tokenizer = get_tokenizer(args.model)
    renderer = renderers.get_renderer(args.renderer, tokenizer=tokenizer)
    semaphore = asyncio.Semaphore(args.parallelism)
    completed = 0
    tasks = [
        asyncio.create_task(
            _evaluate_row(
                row,
                sampling_client=sampling_client,
                renderer=renderer,
                checkpoint=checkpoint,
                output_root=args.output_root,
                semaphore=semaphore,
                max_tokens=args.max_tokens,
                max_attempts=args.max_attempts,
                temperature=args.temperature,
                seed=args.seed,
                inference_sha256=inference_sha256,
            )
        )
        for row in pending
    ]
    for future in asyncio.as_completed(tasks):
        result = await future
        completed += 1
        if completed == 1 or completed % 10 == 0 or completed == len(pending):
            print(
                f"completed={completed}/{len(pending)} idx={result['source_index']} "
                f"status={result['status']}",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        help="Tinker sampler checkpoint; omit to evaluate the frozen base model.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--renderer", default=RENDERER)
    parser.add_argument("--api-key-env", default="TINKER_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=ONE_PASS_TRAINING_RECIPE.max_completion_tokens,
    )
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--parallelism", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
