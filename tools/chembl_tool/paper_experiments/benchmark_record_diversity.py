"""Bounded real-completion benchmark for offline record annotations.

Uses the shared provider client/admission. Successful schema/quote validation is
reported separately from semantic correctness; no result is auto-published.
"""

from __future__ import annotations
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
import os
from pathlib import Path
import statistics
import time

from tools.chembl_tool.common.openai_provider_pool import (
    OpenAIProviderPool,
    load_provider_pool_config,
    load_env_file,
)
from tools.chembl_tool.common.openai_reasoning_client import OpenAICompatibleClient
from tools.chembl_tool.common.record_diversity import (
    VERSION,
    ANNOTATION_VERSION,
    BOUND_ANNOTATION_VERSION,
    QUALITY_ANNOTATION_VERSION,
    ENCODER_VERSION,
    cache_key,
    extraction_messages,
    extraction_request_body,
    validate_annotation,
    repair_annotation_quotes,
    annotation_quote_catalog,
    resolve_annotation_quotes,
    QUOTE_TRANSPORT_INSTRUCTION,
)




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--provider-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sample-file", type=Path)
    parser.add_argument("--parallelism", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--repair-quotes", action="store_true", help="Restore unique literal source citations before strict validation; retain an edit receipt")
    parser.add_argument("--quote-catalog", action="store_true", help="Offer source-span IDs and resolve them to literal citations before validation")
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument(
        "--annotation-version",
        choices=[ANNOTATION_VERSION, BOUND_ANNOTATION_VERSION, QUALITY_ANNOTATION_VERSION],
        default=ANNOTATION_VERSION,
    )
    args = parser.parse_args()
    if min(args.samples, args.parallelism, args.timeout, args.max_tokens, args.max_attempts) < 1:
        parser.error("Sample, concurrency, timeout, token and attempt bounds must be positive")
    load_env_file(".env")
    config = load_provider_pool_config(args.provider_config)
    # This is a source-extraction benchmark, not a change to reasoning experiments.
    specs = tuple(
        replace(
            s,
            request_extra_body=extraction_request_body(
                s.base_url, s.request_extra_body, thinking=args.enable_thinking,
            ),
        )
        for s in config.providers
    )
    config = replace(config, providers=specs)
    model_config = {
        "providers": [s.public_dict() for s in specs],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "reasoning_effort": args.reasoning_effort,
    }
    if args.repair_quotes:
        model_config["quote_repair"] = "source_quote_repair.v1"
    if args.quote_catalog:
        model_config["quote_transport"] = "source_quote_catalog.v1"
    buckets = {}
    for path in sorted(args.inventory.glob("*.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            continue
        for sample in data.get("samples", []):
            key = (
                sample["task"],
                sample["level"],
                sample["source_id"],
                sample["route"],
            )
            buckets.setdefault(key, []).append(sample)
    # Round-robin across task and source/level; do not benchmark only easy rows.
    task_buckets = {}
    for key, values in sorted(buckets.items()):
        task_buckets.setdefault(key[0], []).append(values)
    candidates = []
    while any(task_buckets.values()):
        for task in sorted(task_buckets):
            if task_buckets[task]:
                candidates.extend(task_buckets[task].pop(0))
    if args.sample_file:
        candidates = json.loads(args.sample_file.read_text())
    candidates = candidates[: args.samples]
    if not candidates:
        raise ValueError("No inventory samples")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "summary.json").exists():
        raise ValueError("Use a new benchmark output directory")
    (args.output_dir / "samples.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2) + "\n"
    )
    clients = []

    def client(spec):
        key = os.environ.get(spec.api_key_env, "") if spec.api_key_env else "EMPTY"
        if not key:
            raise ValueError(f"Empty credential alias: {spec.api_key_env}")
        c = OpenAICompatibleClient(
            api_key=key,
            base_url=spec.base_url,
            model=spec.model,
            timeout_s=args.timeout,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            tool_service_url="",
            enable_group_tools=False,
            max_tool_rounds=0,
            reasoning_effort=args.reasoning_effort,
            enable_thinking=False,
            transport_max_retries=0,
            request_extra_body=spec.request_extra_body,
        )
        clients.append(c)
        return c

    pool = OpenAIProviderPool(config, client_factory=client)

    def run(item):
        i, prepared = item
        started = time.monotonic()
        result = {
            "sample": prepared,
            "annotation_version": args.annotation_version,
            "cache_key": cache_key(prepared, model_config, args.annotation_version),
        }
        base_messages = extraction_messages(prepared, args.annotation_version)
        catalog = annotation_quote_catalog(prepared, args.annotation_version) if args.quote_catalog else {}
        if catalog:
            base_messages[0]["content"] += QUOTE_TRANSPORT_INSTRUCTION
            base_messages.append({"role":"user", "content":json.dumps({"source_quote_catalog":catalog},ensure_ascii=False)})
            result["source_quote_catalog"] = catalog
        result["effective_system_prompt_sha256"] = hashlib.sha256(
            base_messages[0]["content"].encode()
        ).hexdigest()
        messages = base_messages
        attempts = []
        for _ in range(args.max_attempts):
            attempt = {}
            try:
                response = pool.chat_json(messages)
                attempt["response"] = response
                content = response["content"]
                if catalog and isinstance(content, dict):
                    content = resolve_annotation_quotes(content, catalog)
                if args.repair_quotes and isinstance(content, dict):
                    content, attempt["quote_repairs"] = repair_annotation_quotes(content, prepared)
                annotation = validate_annotation(content, prepared, args.annotation_version)
                result.update(
                    response=response,
                    annotation=annotation,
                    status="schema_quote_valid",
                )
                attempts.append(attempt)
                break
            except Exception as exc:
                attempt["error"] = str(exc)
                if hasattr(exc, "attempts"):
                    attempt["provider_attempts"] = exc.attempts
                attempts.append(attempt)
                if "response" not in attempt:
                    break  # Shared pool owns transport failures; do not hide them.
                messages = [
                    *base_messages,
                    {
                        "role": "assistant",
                        "content": json.dumps(response["content"], ensure_ascii=False),
                    },
                    {
                        "role": "user",
                        "content": "Validation failed: "
                        + str(exc)
                        + ". Return corrected complete JSON using only the supplied source; do not guess.",
                    },
                ]
        result["attempts"] = attempts
        if "annotation" not in result:
            result.update(status="error", error=attempts[-1]["error"])
            if "response" in attempts[-1]:
                result["response"] = attempts[-1]["response"]
        result["elapsed_seconds"] = time.monotonic() - started
        path = args.output_dir / f"{i:04d}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result

    started = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=args.parallelism) as workers:
        futures = [workers.submit(run, item) for item in enumerate(candidates)]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 16 == 0:
                print(f"completed {len(results)}/{len(candidates)}", flush=True)
    elapsed = time.monotonic() - started
    for client in clients:
        client.close()
    valid = [r for r in results if r["status"] == "schema_quote_valid"]
    times = sorted(r["elapsed_seconds"] for r in results)
    usage = [
        a["response"].get("usage", {})
        for r in results
        for a in r["attempts"]
        if "response" in a
    ]
    summary = {
        "version": VERSION,
        "annotation_version": args.annotation_version,
        "encoder_version": ENCODER_VERSION,
        "system_prompt_sha256": hashlib.sha256(
            extraction_messages(candidates[0], args.annotation_version)[0][
                "content"
            ].encode()
        ).hexdigest(),
        "effective_system_prompt_sha256": sorted({r["effective_system_prompt_sha256"] for r in results}),
        "parallelism": args.parallelism,
        "attempts": len(results),
        "completion_attempts": sum(len(r["attempts"]) for r in results),
        "max_attempts_per_record": args.max_attempts,
        "thinking_requested": args.enable_thinking,
        "schema_quote_valid": len(valid),
        "wall_seconds": elapsed,
        "attempts_per_second": len(results) / elapsed,
        "valid_per_second": len(valid) / elapsed,
        "latency_p50": statistics.median(times),
        "latency_p95": times[min(len(times) - 1, int(0.95 * len(times)))],
        "mean_prompt_tokens": statistics.mean(u.get("prompt_tokens", 0) for u in usage)
        if usage
        else None,
        "mean_completion_tokens": statistics.mean(
            u.get("completion_tokens", 0) for u in usage
        )
        if usage
        else None,
        "reasoning_tokens": sum(u.get("reasoning_tokens", 0) or 0 for u in usage),
        "model_config": model_config,
        "semantic_accuracy_verified": False,
        "errors": [
            {"record_id": r["sample"]["record_id"], "error": r["error"]}
            for r in results
            if r["status"] == "error"
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
