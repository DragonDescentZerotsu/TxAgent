"""Run or resume frozen one-pass full-flat inference rows."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import threading
from typing import Any

from tools.chembl_tool.common.json_utils import sha256_file, write_json_atomic
from tools.chembl_tool.common.openai_reasoning_client import OpenAICompatibleClient
from tools.chembl_tool.common.reasoning_validation import call_with_json_validation

from .experiment_contract import ONE_PASS_TRAINING_RECIPE
from .one_pass_runtime import (
    inference_contract_sha256,
    load_fresh_one_pass_audit,
    one_pass_allowed_values,
    one_pass_analysis_schema_errors,
    one_pass_predictions,
    one_pass_result_path,
    read_one_pass_rows,
)


EVALUATOR_VERSION = "one_pass_openai_evaluator.v2"


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
        and result.get("predictions", {}).get("final") in {0, 1}
    )


def _run_row(
    client: OpenAICompatibleClient,
    row: dict[str, Any],
    output_root: Path,
    *,
    max_attempts: int,
    inference_sha256: str,
) -> dict[str, Any]:
    try:
        response = call_with_json_validation(
            client.chat_json,
            row["messages"],
            required_fields=row["required_fields"],
            allowed_values=one_pass_allowed_values(row),
            content_validator=lambda content: one_pass_analysis_schema_errors(
                content, row
            ),
            branch_name="one_pass_full_flat",
            max_attempts=max_attempts,
        )
        content = (
            response.get("content") if isinstance(response.get("content"), dict) else {}
        )
        predictions = one_pass_predictions(content, row)
        validation = response.get("structured_output_validation") or {}
        status = (
            "ok"
            if validation.get("valid") and predictions["final"] in {0, 1}
            else "error"
        )
        result = {
            "evaluator_version": EVALUATOR_VERSION,
            "inference_contract_sha256": inference_sha256,
            "status": status,
            "source_task": row["source_task"],
            "source_subset": row["source_subset"],
            "source_index": row["source_index"],
            "gold_label": row["gold_label"],
            "prompt_sha256": row["prompt_sha256"],
            "predictions": predictions,
            "branches_disagree": (
                predictions["single"] is not None
                and predictions["analog"] is not None
                and predictions["single"] != predictions["analog"]
            ),
            "attempt_count": int(validation.get("attempt_count") or 0),
            "response": response,
        }
    except Exception as exc:
        result = {
            "evaluator_version": EVALUATOR_VERSION,
            "inference_contract_sha256": inference_sha256,
            "status": "error",
            "source_task": row["source_task"],
            "source_subset": row["source_subset"],
            "source_index": row["source_index"],
            "gold_label": row["gold_label"],
            "prompt_sha256": row["prompt_sha256"],
            "predictions": {"single": None, "analog": None, "final": None},
            "error": f"{type(exc).__name__}: {exc}",
        }
    write_json_atomic(one_pass_result_path(output_root, row), result)
    return result


def run(args: argparse.Namespace) -> None:
    rows = read_one_pass_rows(args.data)
    if args.limit is not None:
        rows = rows[: args.limit]
    data_audit = load_fresh_one_pass_audit(args.data)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    inference_contract = {
        "evaluator_version": EVALUATOR_VERSION,
        "provider": "openai_compatible",
        "base_url": args.base_url,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_attempts": args.max_attempts,
        "validation": "one_pass_nested_schema.v1",
        "tool_calls": "disabled_harness_prefetched_only",
    }
    inference_sha256 = inference_contract_sha256(inference_contract)
    manifest = {
        "evaluator_version": EVALUATOR_VERSION,
        "contract_version": rows[0]["contract_version"] if rows else "",
        "data": str(args.data),
        "data_sha256": sha256_file(args.data),
        "data_audit": data_audit["audit_path"],
        "data_audit_sha256": data_audit["audit_sha256"],
        "n_rows": len(rows),
        "inference_contract": inference_contract,
        "inference_contract_sha256": inference_sha256,
        "parallelism": args.parallelism,
    }
    write_json_atomic(output_root / "manifest.json", manifest)
    pending = [
        row
        for row in rows
        if not _valid_existing(
            one_pass_result_path(output_root, row), row, inference_sha256
        )
    ]
    print(
        f"one-pass rows total={len(rows)} existing={len(rows) - len(pending)} pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return
    api_key = os.environ.get(args.api_key_env) or "not-required"
    client = OpenAICompatibleClient(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tool_service_url="http://127.0.0.1:8765",
        enable_group_tools=False,
        max_tool_rounds=0,
        reasoning_effort=args.reasoning_effort,
        enable_thinking=False,
    )
    counter = 0
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.parallelism
    ) as executor:
        futures = {
            executor.submit(
                _run_row,
                client,
                row,
                output_root,
                max_attempts=args.max_attempts,
                inference_sha256=inference_sha256,
            ): row
            for row in pending
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            with lock:
                counter += 1
                if counter == 1 or counter % 10 == 0 or counter == len(pending):
                    print(
                        f"completed={counter}/{len(pending)} latest_idx={result['source_index']} "
                        f"status={result['status']}",
                        flush=True,
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:9001/v1")
    parser.add_argument("--api-key-env", default="GPT_OSS_LOCAL_API_KEY")
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=ONE_PASS_TRAINING_RECIPE.max_completion_tokens,
    )
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--parallelism", type=int, default=64)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
