"""Run an LLM assay-activity transfer benchmark on sampled ChEMBL pairs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI


DEFAULT_INPUT = "outputs/chembl_tool/activity_transfer_benchmark/llm_eval_sets/dynamic_v1_llm_3k/eval_pairs.jsonl"
DEFAULT_OUT_ROOT = "outputs/chembl_tool/activity_transfer_benchmark/llm_runs"
DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_TOOL_SERVICE_URL = "http://127.0.0.1:8765"
DEFAULT_MODEL = "gpt-oss-120b"
DEFAULT_MAX_TOKENS = 1024

TRANSFER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mmp_structure_compare",
            "description": (
                "Compare the query molecule to the reference molecule using Morgan Tanimoto, MCS coverage, "
                "and mmpdb matched-pair transformation. Use this to judge structural activity transferability."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_smiles": {"type": "string", "description": "The query molecule SMILES."},
                    "reference_smiles": {"type": "string", "description": "The reference molecule SMILES."},
                    "max_mmp_alternatives": {
                        "type": "integer",
                        "description": "Maximum matched-pair alternatives to return.",
                        "default": 5,
                    },
                    "mcs_timeout_s": {
                        "type": "integer",
                        "description": "MCS search timeout in seconds.",
                        "default": 5,
                    },
                },
                "required": ["query_smiles", "reference_smiles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "properties_compare",
            "description": (
                "Compare query and reference molecule properties, including RDKit descriptors and MolGpKa/logD "
                "features. Use this to assess whether property changes affect assay activity transferability."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_smiles": {"type": "string", "description": "The query molecule SMILES."},
                    "reference_smiles": {"type": "string", "description": "The reference molecule SMILES."},
                    "logd_ph": {
                        "type": "number",
                        "description": "pH for logD comparison.",
                        "default": 7.4,
                    },
                },
                "required": ["query_smiles", "reference_smiles"],
            },
        },
    },
]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.enable_thinking and args.disable_thinking:
        raise SystemExit("Use only one of --enable-thinking or --disable-thinking.")
    load_env(Path(args.env_file))
    api_key = resolve_api_key(args)
    records = prepare_input_records(read_jsonl(Path(args.input_jsonl)))
    selected_indices = select_indices(args, len(records))
    max_tokens = effective_max_tokens(args)
    response_format = should_use_response_format(args)

    run_id = args.run_id or time.strftime("gpt_oss_120b_llm_%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / run_id
    runs_dir = out_dir / "runs"
    logs_dir = out_dir / "logs"
    figures_dir = out_dir / "figures"
    for path in (runs_dir, logs_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "input_jsonl": args.input_jsonl,
        "model": args.model,
        "base_url": args.base_url,
        "base_urls": parse_base_urls(args),
        "tool_service_url": args.tool_service_url,
        "enable_tools": not args.disable_tools,
        "input_format": infer_input_format(records),
        "output_mode": args.output_mode,
        "max_tokens": max_tokens,
        "response_format": response_format,
        "enable_thinking": args.enable_thinking,
        "disable_thinking": args.disable_thinking,
        "n_selected": len(selected_indices),
        "selected_indices": selected_indices,
        "started_at": now(),
        "paths": {
            "out_dir": str(out_dir),
            "runs": str(runs_dir),
            "predictions": str(out_dir / "predictions.jsonl"),
            "metrics": str(out_dir / "metrics.json"),
            "report": str(out_dir / "report_zh.md"),
            "trace_messages": str(out_dir / "trace_messages.jsonl"),
            "figures": str(figures_dir),
        },
    }
    write_json(out_dir / "manifest.json", manifest)

    clients = [
        LlmClient(
            api_key=api_key,
            base_url=base_url,
            model=args.model,
            timeout_s=args.timeout_s,
            max_tokens=max_tokens,
            tool_service_url=args.tool_service_url,
            enable_tools=not args.disable_tools,
            max_tool_rounds=args.max_tool_rounds,
            response_format=response_format,
            reasoning_effort=args.reasoning_effort,
            enable_thinking=args.enable_thinking,
            disable_thinking=args.disable_thinking,
            output_mode=args.output_mode,
        )
        for base_url in parse_base_urls(args)
    ]

    lock = threading.Lock()
    completed = 0
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallelism) as executor:
        futures = {
            executor.submit(
                run_one,
                clients[position % len(clients)],
                records[index],
                index,
                runs_dir,
                args.skip_existing,
                args.output_mode,
            ): index
            for position, index in enumerate(selected_indices)
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - keep batch alive.
                result = error_result(records[index], index, str(exc))
            results.append(result)
            completed += 1
            if completed % max(1, args.progress_every) == 0 or completed == len(futures):
                metrics = compute_metrics(results)
                elapsed_s = time.monotonic() - started
                rate = completed / elapsed_s if elapsed_s > 0 else 0.0
                eta_s = (len(futures) - completed) / rate if rate > 0 else None
                with lock:
                    print(
                        "[activity_transfer_llm] "
                        f"progress {completed}/{len(futures)} "
                        f"elapsed={format_duration(elapsed_s)} "
                        f"eta={format_duration(eta_s)} "
                        f"rate={rate:.2f}/s "
                        f"macro_f1={metrics['llm'].get('macro_f1', 0):.4f} "
                        f"failed={metrics['n_failed']}",
                        file=sys.stderr,
                        flush=True,
                    )
    results.sort(key=lambda row: row["sample_index"])
    metrics = compute_metrics(results)
    metrics["finished_at"] = now()
    metrics["wall_s"] = time.monotonic() - started
    write_jsonl(out_dir / "predictions.jsonl", results)
    write_trace_jsonl(out_dir / "trace_messages.jsonl", results)
    write_json(out_dir / "metrics.json", metrics)
    write_report(out_dir / "report_zh.md", manifest, metrics)
    plot_metrics(figures_dir / "model_vs_baselines.svg", metrics)
    manifest["finished_at"] = metrics["finished_at"]
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps({"manifest": manifest, "metrics": metrics}, ensure_ascii=False, indent=2), flush=True)
    return 0 if metrics["n_failed"] == 0 else 1


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total_seconds = max(0, int(round(seconds)))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d{hours:02d}h{minutes:02d}m{secs:02d}s"
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--api-key", default="", help="Explicit API key. Overrides env vars when set.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-key-env-fallback", default="VLLM_API_KEY")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--base-urls",
        default="",
        help="Optional comma-separated OpenAI-compatible base URLs. Overrides --base-url and is used round-robin.",
    )
    parser.add_argument("--tool-service-url", default=DEFAULT_TOOL_SERVICE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--parallelism", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--indices", type=int, nargs="*", default=[])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-tool-rounds", type=int, default=3)
    parser.add_argument(
        "--output-mode",
        choices=("json", "choice"),
        default="json",
        help="json asks for predicted_transferability JSON; choice asks for a one-letter A/B answer.",
    )
    parser.add_argument("--disable-tools", action="store_true")
    parser.add_argument("--disable-response-format", action="store_true")
    parser.add_argument("--reasoning-effort", default="", help="Optional OpenAI-compatible reasoning_effort value.")
    parser.add_argument("--enable-thinking", action="store_true", help="Send extra_body thinking enabled.")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Send Qwen/vLLM chat_template_kwargs enable_thinking=false.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args(argv)


def effective_max_tokens(args: argparse.Namespace) -> int:
    if args.output_mode == "choice" and args.max_tokens == DEFAULT_MAX_TOKENS:
        return 1
    return args.max_tokens


def should_use_response_format(args: argparse.Namespace) -> bool:
    if args.output_mode != "json" or args.disable_response_format:
        return False
    # Qwen3 thinking is returned by vLLM as visible <think>...</think> text
    # unless the server is launched with a reasoning parser. JSON response
    # format suppresses those tags, so disable it when Qwen thinking is needed.
    if args.enable_thinking and "qwen" in str(args.model).lower():
        return False
    return True


def prepare_input_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return records
    if not all(is_hf_prompt_completion_record(record) for record in records):
        return records
    global_majority = majority_label(label for label in (hf_completion_to_label(row.get("completion")) for row in records) if label)
    bucket_majorities = hf_bucket_majorities(records, global_majority)
    prepared = []
    for index, record in enumerate(records):
        metadata = record.get("metadata") or {}
        label = hf_completion_to_label(record.get("completion"))
        tanimoto = parse_float(metadata.get("weighted_tanimoto"))
        bucket_key = str(metadata.get("similarity_bucket"))
        row = dict(record)
        row.update(
            {
                "input_format": "hf_prompt_completion",
                "pair_index": index,
                "label": label,
                "tanimoto": tanimoto,
                "similarity_bucket": metadata.get("similarity_bucket"),
                "baseline_tanimoto_0_50_prediction": "similar" if tanimoto >= 0.50 else "different",
                "baseline_tanimoto_0_48_prediction": "similar" if tanimoto >= 0.48 else "different",
                "baseline_mcs_0_70_prediction": "",
                "baseline_similarity_bucket_majority_prediction": bucket_majorities.get(bucket_key, global_majority),
                "baseline_assay_type_tanimoto_0_50_prediction": "similar" if tanimoto >= 0.50 else "different",
            }
        )
        prepared.append(row)
    return prepared


def is_hf_prompt_completion_record(record: dict[str, Any]) -> bool:
    return "prompt" in record and "completion" in record and isinstance(record.get("metadata"), dict)


def infer_input_format(records: list[dict[str, Any]]) -> str:
    if records and all(record.get("input_format") == "hf_prompt_completion" for record in records):
        return "hf_prompt_completion"
    return "activity_transfer_eval_pairs"


def hf_completion_to_label(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text == "A":
        return "similar"
    if text == "B":
        return "different"
    return normalize_prediction(value)


def hf_bucket_majorities(records: list[dict[str, Any]], default_label: str) -> dict[str, str]:
    counts: dict[str, Counter] = {}
    for record in records:
        metadata = record.get("metadata") or {}
        label = hf_completion_to_label(record.get("completion"))
        if not label:
            continue
        bucket = str(metadata.get("similarity_bucket"))
        counts.setdefault(bucket, Counter())[label] += 1
    return {bucket: majority_label(counter.elements(), default_label=default_label) for bucket, counter in counts.items()}


def majority_label(labels: Any, default_label: str = "similar") -> str:
    counter = Counter(labels)
    if counter["similar"] > counter["different"]:
        return "similar"
    if counter["different"] > counter["similar"]:
        return "different"
    return default_label


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class LlmClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_s: int,
        max_tokens: int,
        tool_service_url: str,
        enable_tools: bool,
        max_tool_rounds: int,
        response_format: bool,
        reasoning_effort: str,
        enable_thinking: bool,
        disable_thinking: bool,
        output_mode: str,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"), timeout=timeout_s)
        self.model = model
        self.max_tokens = max_tokens
        self.tool_service = ToolServiceClient(tool_service_url, timeout_s=timeout_s)
        self.enable_tools = enable_tools
        self.max_tool_rounds = max_tool_rounds
        self.response_format = response_format
        self.reasoning_effort = reasoning_effort
        self.enable_thinking = enable_thinking
        self.disable_thinking = disable_thinking
        self.output_mode = output_mode

    def chat_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.enable_tools:
            return self.chat_json_no_tools(messages)
        working_messages: list[Any] = list(messages)
        trace_messages: list[Any] = list(messages)
        tool_results = []
        responses = []
        for round_index in range(self.max_tool_rounds + 1):
            response = self.create_completion(
                working_messages,
                tools=TRANSFER_TOOLS,
                tool_choice="auto",
            )
            responses.append(response)
            message = response.choices[0].message
            working_messages.append(message)
            trace_messages.append(json_safe_message(message))
            tool_calls = message.tool_calls or []
            if not tool_calls:
                return response_to_result(response, trace_messages, tool_results, responses=responses)
            for tool_call in tool_calls:
                tool_result = self.tool_service.invoke_function_call(
                    tool_call,
                    allowed_tool_names={"mmp_structure_compare", "properties_compare"},
                )
                tool_results.append(tool_result)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result["content"],
                }
                working_messages.append(tool_message)
                trace_messages.append({**tool_message, "name": tool_result.get("tool_name"), "tool_result": tool_result})
        working_messages.append(
            {
                "role": "user",
                "content": final_answer_instruction(self.output_mode),
            }
        )
        response = self.create_completion(working_messages)
        responses.append(response)
        return response_to_result(response, trace_messages, tool_results, responses=responses)

    def chat_json_no_tools(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        response = self.create_completion(messages)
        return response_to_result(response, messages, [])

    def create_completion(
        self,
        messages: list[Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.response_format:
            kwargs["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        extra_body = self.extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception:
            if "response_format" in kwargs:
                kwargs.pop("response_format", None)
                return self.client.chat.completions.create(**kwargs)
            raise

    def extra_body(self) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if self.enable_thinking:
            if "qwen" in self.model.lower():
                extra["chat_template_kwargs"] = {"enable_thinking": True}
            else:
                extra["thinking"] = {"type": "enabled"}
        if self.disable_thinking:
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        return extra


class ToolServiceClient:
    def __init__(self, base_url: str, *, timeout_s: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def invoke_function_call(self, tool_call: Any, *, allowed_tool_names: set[str]) -> dict[str, Any]:
        tool_name = tool_call.function.name
        if tool_name not in allowed_tool_names:
            return {"tool_name": tool_name, "status": "error", "content": f"Tool `{tool_name}` is not allowed."}
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError as exc:
            return {"tool_name": tool_name, "status": "error", "content": f"Invalid tool JSON: {exc}"}
        response = requests.post(
            f"{self.base_url}/tools/{tool_name}/invoke",
            json={
                "tool_name": tool_name,
                "version": "v1",
                "input": arguments,
                "options": {"timeout_s": self.timeout_s, "return_debug": False},
            },
            timeout=self.timeout_s,
        )
        if response.status_code >= 400:
            return {
                "tool_name": tool_name,
                "status": "error",
                "arguments": arguments,
                "content": f"{tool_name} HTTP {response.status_code}: {response.text[:1000]}",
            }
        payload = response.json()
        output = payload.get("output") or {}
        errors = payload.get("errors") or []
        content = str(output.get("text") or "No LLM-readable tool output.")
        if payload.get("status") != "ok":
            content = "Tool returned error: " + json.dumps(errors, ensure_ascii=False)
        return {
            "tool_name": tool_name,
            "status": payload.get("status", "error"),
            "arguments": arguments,
            "content": f"[{tool_name}]\n{content}",
            "errors": errors,
            "warnings": payload.get("warnings") or [],
        }


def run_one(
    client: LlmClient,
    record: dict[str, Any],
    sample_index: int,
    runs_dir: Path,
    skip_existing: bool,
    output_mode: str,
) -> dict[str, Any]:
    run_path = runs_dir / f"sample_{sample_index:05d}.json"
    if skip_existing and run_path.exists():
        return json.loads(run_path.read_text(encoding="utf-8"))
    started = time.monotonic()
    messages = build_messages(record, output_mode=output_mode)
    response = client.chat_json(messages)
    content = response["content"]
    prediction = extract_prediction(content)
    if not prediction:
        retry_messages = messages + [
            {
                "role": "user",
                "content": retry_instruction(output_mode),
            }
        ]
        response = client.chat_json_no_tools(retry_messages)
        content = response["content"]
        prediction = extract_prediction(content)
    result = {
        "sample_index": sample_index,
        "pair_index": record["pair_index"],
        "status": "ok" if prediction else "error",
        "prediction": prediction or "",
        "label": record["label"],
        "correct": bool(prediction == record["label"]) if prediction else False,
        "confidence": content.get("confidence", "") if isinstance(content, dict) else "",
        "output_mode": output_mode,
        "raw_content": response["raw_content"],
        "parsed_content": content,
        "reasoning_content": response.get("reasoning_content", ""),
        "has_reasoning_content": bool(response.get("reasoning_content")),
        "final_message": response.get("final_message", {}),
        "visible_messages": response.get("messages", []),
        "response_id": response.get("id", ""),
        "usage": response["usage"],
        "model": response["model"],
        "tool_count": len(response["tool_results"]),
        "tool_results": response["tool_results"],
        "latency_s": round(time.monotonic() - started, 3),
        "input_record": compact_input_record(record),
        "baseline_tanimoto_0_50_prediction": record.get("baseline_tanimoto_0_50_prediction", ""),
        "baseline_tanimoto_0_48_prediction": record.get("baseline_tanimoto_0_48_prediction", ""),
        "baseline_mcs_0_70_prediction": record.get("baseline_mcs_0_70_prediction", ""),
        "baseline_similarity_bucket_majority_prediction": record.get(
            "baseline_similarity_bucket_majority_prediction", ""
        ),
        "baseline_assay_type_tanimoto_0_50_prediction": record.get(
            "baseline_assay_type_tanimoto_0_50_prediction", ""
        ),
    }
    if not prediction:
        result["error"] = "Missing or invalid predicted_transferability."
    write_json(run_path, result)
    return result


def retry_instruction(output_mode: str) -> str:
    if output_mode == "choice":
        return "Your previous answer was invalid. Return exactly one letter: A or B."
    return (
        "Your previous answer was invalid or missing `predicted_transferability`. "
        "Return exactly one compact JSON object with keys: "
        "predicted_transferability, confidence, should_transfer_activity, "
        "reasoning_summary, key_factors, rationale. "
        "Use predicted_transferability as either `similar` or `different`."
    )


def final_answer_instruction(output_mode: str) -> str:
    if output_mode == "choice":
        return "Return exactly one letter now: A or B. Do not call more tools."
    return "Return the required compact JSON now. Do not call more tools."


def build_messages(record: dict[str, Any], *, output_mode: str = "json") -> list[dict[str, str]]:
    if record.get("input_format") == "hf_prompt_completion":
        system = hf_system_prompt(output_mode)
        user = hf_user_prompt(record, output_mode)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    label_mode = str(record.get("label_mode") or "pchembl_delta")
    if label_mode in {"raw_robust_z", "log_raw_robust_z"}:
        activity_name = "log10(raw endpoint value)" if label_mode == "log_raw_robust_z" else "raw endpoint value"
        task_text = (
            "Predict whether the query molecule's endpoint activity is similar to the reference molecule's "
            "activity in this same ChEMBL assay endpoint. Use predicted_transferability='similar' when the "
            "expected absolute activity difference is within 0.5 assay-internal robust sigma, and 'different' "
            "when it is at least 1.0 robust sigma. The hidden query activity is not provided."
        )
        reference_activity = {
            "chembl_id": record["reference_molecule_chembl_id"],
            "smiles": record["reference_smiles"],
            "known_activity_value": record.get("reference_activity_value"),
            "activity_scale": activity_name,
            "raw_units": record.get("raw_units", ""),
        }
        query_activity = {
            "chembl_id": record["query_molecule_chembl_id"],
            "smiles": record["query_smiles"],
            "activity_value": "hidden_for_evaluation",
            "activity_scale": activity_name,
            "raw_units": record.get("raw_units", ""),
        }
        activity_context = {
            "label_mode": label_mode,
            "raw_units": record.get("raw_units", ""),
            "robust_sigma": record.get("robust_sigma"),
            "similar_threshold_sigma": 0.5,
            "different_threshold_sigma": 1.0,
        }
    else:
        task_text = (
            "Predict whether the query molecule's pChEMBL activity is similar to the reference molecule's "
            "activity in this same ChEMBL assay endpoint. Use predicted_transferability='similar' when "
            "|delta pChEMBL| is expected to be <= 0.5, and 'different' when it is expected to be >= 1.0."
        )
        reference_activity = {
            "chembl_id": record["reference_molecule_chembl_id"],
            "smiles": record["reference_smiles"],
            "known_pchembl_value": record.get("reference_pchembl_value", record.get("reference_activity_value")),
        }
        query_activity = {
            "chembl_id": record["query_molecule_chembl_id"],
            "smiles": record["query_smiles"],
            "pchembl_value": "hidden_for_evaluation",
        }
        activity_context = {"label_mode": label_mode}

    system = (
        "You are evaluating analog assay activity transferability for medicinal chemistry. "
        +
        ("Return only compact JSON. " if output_mode == "json" else "Return only one letter: A or B. ")
        + "The hidden query activity is not provided. "
        "Use assay context and structural/property similarity to decide whether the reference activity "
        "is likely transferable to the query molecule in the same assay endpoint."
    )
    user = {
        "task": task_text,
        "task_dataset": record.get("task_name", ""),
        "assay": {
            "assay_chembl_id": record["assay_chembl_id"],
            "standard_type": record["standard_type"],
            "raw_units": record.get("raw_units", ""),
            "assay_tier": record.get("assay_tier", ""),
            "target_chembl_id": record.get("target_chembl_id", ""),
            "target_pref_name": record.get("target_pref_name", ""),
            "target_type": record.get("target_type", ""),
            "target_organism": record.get("target_organism", ""),
            "assay_type": record.get("assay_type", ""),
            "assay_test_type": record.get("assay_test_type", ""),
            "assay_category": record.get("assay_category", ""),
            "confidence_score": record.get("confidence_score", ""),
            "relationship_type": record.get("relationship_type", ""),
            "description": truncate(str(record.get("assay_description", "")), 1200),
        },
        "activity_label_context": activity_context,
        "reference_molecule": reference_activity,
        "query_molecule": query_activity,
        "precomputed_similarity": {
            "morgan_tanimoto": record["tanimoto"],
            "similarity_bucket": record["similarity_bucket"],
            "mean_mcs_coverage": record.get("mean_mcs_coverage"),
            "query_mcs_coverage": record.get("query_mcs_coverage"),
            "reference_mcs_coverage": record.get("reference_mcs_coverage"),
            "mcs_timed_out": record.get("mcs_timed_out"),
        },
    }
    if output_mode == "json":
        user["output_schema"] = {
            "predicted_transferability": "similar or different",
            "confidence": "high, moderate, or low",
            "should_transfer_activity": "boolean",
            "reasoning_summary": [
                "2-5 concise visible reasoning steps based only on assay context, structure, properties, and tools"
            ],
            "key_factors": ["short strings"],
            "rationale": "one concise paragraph",
        }
    else:
        user["output_options"] = {
            "A": "transfer / predicted_transferability=similar",
            "B": "not transfer / predicted_transferability=different",
        }
        user["instruction"] = "Return exactly one letter: A or B."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def hf_system_prompt(output_mode: str) -> str:
    if output_mode == "choice":
        return (
            "You are evaluating analog assay activity transferability for medicinal chemistry. "
            "Return exactly one letter: A or B."
        )
    return (
        "You are evaluating analog assay activity transferability for medicinal chemistry. "
        "Return only compact JSON. Use predicted_transferability='similar' for answer A/transfer, "
        "and predicted_transferability='different' for answer B/not transfer."
    )


def hf_user_prompt(record: dict[str, Any], output_mode: str) -> str:
    prompt = str(record["prompt"]).rstrip()
    if output_mode == "choice":
        return prompt + "\n\nOutput exactly one letter: A or B."
    return (
        prompt
        + "\n\nReturn exactly one compact JSON object with keys: "
        "predicted_transferability, confidence, should_transfer_activity, "
        "reasoning_summary, key_factors, rationale. "
        "Use predicted_transferability as either `similar` or `different`."
    )


def response_to_result(
    response: Any,
    messages: list[Any],
    tool_results: list[dict[str, Any]],
    *,
    responses: list[Any] | None = None,
) -> dict[str, Any]:
    message = response.choices[0].message
    content = message.content or "{}"
    final_message = json_safe_message(message)
    trace_messages = [json_safe_message(item) for item in messages]
    if not trace_messages or trace_messages[-1] != final_message:
        trace_messages.append(final_message)
    reasoning_content = extract_reasoning_content(final_message)
    if not reasoning_content:
        reasoning_content = extract_think_content(content)
    content_for_parse = strip_think_blocks(content)
    return {
        "content": parse_json_content(content_for_parse),
        "raw_content": content,
        "reasoning_content": reasoning_content,
        "final_message": final_message,
        "messages": trace_messages,
        "usage": sum_usage(responses or [response]),
        "model": response.model or "",
        "id": response.id or "",
        "tool_results": tool_results,
    }


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok_results = [row for row in results if row.get("status") == "ok"]
    baseline_keys = {
        "tanimoto_0_50": "baseline_tanimoto_0_50_prediction",
        "tanimoto_0_48": "baseline_tanimoto_0_48_prediction",
        "mcs_0_70": "baseline_mcs_0_70_prediction",
        "similarity_bucket_majority": "baseline_similarity_bucket_majority_prediction",
        "assay_type_tanimoto_0_50": "baseline_assay_type_tanimoto_0_50_prediction",
    }
    metrics = {
        "n": len(results),
        "n_ok": len(ok_results),
        "n_failed": len(results) - len(ok_results),
        "llm": metrics_for_predictions(ok_results, "prediction"),
        "baselines": {
            name: metrics_for_predictions(ok_results, key)
            for name, key in baseline_keys.items()
            if any(row.get(key) for row in ok_results)
        },
        "subsets": {},
        "usage": sum_usage_dicts([row.get("usage") or {} for row in ok_results]),
        "tool_calls": sum(int(row.get("tool_count") or 0) for row in ok_results),
    }
    gray = [
        row
        for row in ok_results
        if 0.40 <= float((row.get("input_record") or {}).get("tanimoto", 0.0)) < 0.70
    ]
    baseline_wrong = [
        row for row in ok_results if row.get("baseline_tanimoto_0_50_prediction") != row["label"]
    ]
    metrics["subsets"]["tanimoto_0_40_to_0_70"] = {
        "n": len(gray),
        "llm": metrics_for_predictions(gray, "prediction"),
        "tanimoto_0_50": metrics_for_predictions(gray, "baseline_tanimoto_0_50_prediction"),
    }
    metrics["subsets"]["tanimoto_0_50_wrong"] = {
        "n": len(baseline_wrong),
        "llm_accuracy_on_baseline_errors": safe_div(
            sum(1 for row in baseline_wrong if row["prediction"] == row["label"]),
            len(baseline_wrong),
        ),
    }
    task_names = sorted(
        {
            str((row.get("input_record") or {}).get("task_name") or "")
            for row in ok_results
            if (row.get("input_record") or {}).get("task_name")
        }
    )
    if task_names:
        metrics["per_task"] = {}
        for task_name in task_names:
            task_rows = [
                row
                for row in ok_results
                if str((row.get("input_record") or {}).get("task_name") or "") == task_name
            ]
            metrics["per_task"][task_name] = {
                "n": len(task_rows),
                "llm": metrics_for_predictions(task_rows, "prediction"),
                "baselines": {
                    name: metrics_for_predictions(task_rows, key)
                    for name, key in baseline_keys.items()
                    if any(row.get(key) for row in task_rows)
                },
                "label_counts": dict(Counter(row.get("label") for row in task_rows)),
            }
    add_metadata_group_metrics(metrics, ok_results, baseline_keys)
    return metrics


def add_metadata_group_metrics(
    metrics: dict[str, Any],
    ok_results: list[dict[str, Any]],
    baseline_keys: dict[str, str],
) -> None:
    group_specs = {
        "per_assay_type": lambda row: ((row.get("input_record") or {}).get("hf_metadata") or {}).get("assay_type"),
        "per_similarity_bucket": lambda row: str(((row.get("input_record") or {}).get("hf_metadata") or {}).get("similarity_bucket")),
        "per_eval_subset": lambda row: ((row.get("input_record") or {}).get("hf_metadata") or {}).get("eval_subset"),
    }
    for output_key, key_fn in group_specs.items():
        groups = sorted({str(key_fn(row)) for row in ok_results if key_fn(row) not in (None, "", "None")})
        if not groups:
            continue
        metrics[output_key] = {}
        for group in groups:
            rows = [row for row in ok_results if str(key_fn(row)) == group]
            metrics[output_key][group] = {
                "n": len(rows),
                "llm": metrics_for_predictions(rows, "prediction"),
                "baselines": {
                    name: metrics_for_predictions(rows, key)
                    for name, key in baseline_keys.items()
                    if any(row.get(key) for row in rows)
                },
                "label_counts": dict(Counter(row.get("label") for row in rows)),
            }


def metrics_for_predictions(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        pred = normalize_prediction(row.get(prediction_key))
        label = row.get("label")
        if not pred or label not in {"similar", "different"}:
            continue
        pred_similar = pred == "similar"
        true_similar = label == "similar"
        if pred_similar and true_similar:
            tp += 1
        elif pred_similar and not true_similar:
            fp += 1
        elif not pred_similar and true_similar:
            fn += 1
        else:
            tn += 1
    precision_similar = safe_div(tp, tp + fp)
    recall_similar = safe_div(tp, tp + fn)
    precision_different = safe_div(tn, tn + fn)
    recall_different = safe_div(tn, tn + fp)
    f1_similar = safe_div(2 * precision_similar * recall_similar, precision_similar + recall_similar)
    f1_different = safe_div(2 * precision_different * recall_different, precision_different + recall_different)
    total = tp + fp + tn + fn
    return {
        "n": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": safe_div(tp + tn, total),
        "balanced_accuracy": (recall_similar + recall_different) / 2.0,
        "macro_f1": (f1_similar + f1_different) / 2.0,
        "precision_similar": precision_similar,
        "recall_similar": recall_similar,
        "precision_different": precision_different,
        "recall_different": recall_different,
    }


def write_report(path: Path, manifest: dict[str, Any], metrics: dict[str, Any]) -> None:
    lines = [
        "# Activity-transfer LLM benchmark",
        "",
        f"- run_id: {manifest['run_id']}",
        f"- model: {manifest['model']}",
        f"- n: {metrics['n']:,}",
        f"- ok / failed: {metrics['n_ok']:,} / {metrics['n_failed']:,}",
        f"- tool calls: {metrics['tool_calls']:,}",
        f"- total usage: {metrics['usage']}",
        "",
        "## Metrics",
        "",
        "| method | accuracy | balanced accuracy | macro-F1 |",
        "| --- | ---: | ---: | ---: |",
        metric_line("LLM", metrics["llm"]),
    ]
    for name, values in metrics["baselines"].items():
        if values.get("n", 0) == 0:
            continue
        lines.append(metric_line(name, values))
    gray = metrics["subsets"]["tanimoto_0_40_to_0_70"]
    lines.extend(
        [
            "",
            "## Key Subsets",
            "",
            f"- Tanimoto 0.40-0.70 subset n={gray['n']:,}: "
            f"LLM macro-F1={gray['llm']['macro_f1']:.4f}, "
            f"Tanimoto>=0.50 macro-F1={gray['tanimoto_0_50']['macro_f1']:.4f}",
            f"- On Tanimoto>=0.50 baseline errors, LLM accuracy="
            f"{metrics['subsets']['tanimoto_0_50_wrong']['llm_accuracy_on_baseline_errors']:.4f}",
        ]
    )
    if metrics.get("per_task"):
        lines.extend(
            [
                "",
                "## Per Task",
                "",
                "| task | n | LLM macro-F1 | Tanimoto 0.50 macro-F1 | Tanimoto 0.48 macro-F1 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for task_name, task_metrics in metrics["per_task"].items():
            lines.append(
                f"| {task_name} | {task_metrics['n']:,} | "
                f"{task_metrics['llm']['macro_f1']:.4f} | "
                f"{task_metrics['baselines'].get('tanimoto_0_50', {}).get('macro_f1', 0.0):.4f} | "
                f"{task_metrics['baselines'].get('tanimoto_0_48', {}).get('macro_f1', 0.0):.4f} |"
            )
    for group_key, title in (
        ("per_assay_type", "Per Assay Type"),
        ("per_similarity_bucket", "Per Similarity Bucket"),
        ("per_eval_subset", "Per Eval Subset"),
    ):
        if not metrics.get(group_key):
            continue
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| group | n | LLM macro-F1 | Tanimoto 0.50 macro-F1 | Bucket-majority macro-F1 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for group, group_metrics in metrics[group_key].items():
            lines.append(
                f"| {group} | {group_metrics['n']:,} | "
                f"{group_metrics['llm']['macro_f1']:.4f} | "
                f"{group_metrics['baselines'].get('tanimoto_0_50', {}).get('macro_f1', 0.0):.4f} | "
                f"{group_metrics['baselines'].get('similarity_bucket_majority', {}).get('macro_f1', 0.0):.4f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_line(name: str, values: dict[str, Any]) -> str:
    return f"| {name} | {values['accuracy']:.4f} | {values['balanced_accuracy']:.4f} | {values['macro_f1']:.4f} |"


def plot_metrics(path: Path, metrics: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["LLM"]
    values = [metrics["llm"]["macro_f1"]]
    colors = ["#4c78a8"]
    baseline_labels = {
        "tanimoto_0_50": ("Tanimoto 0.50", "#f58518"),
        "tanimoto_0_48": ("Tanimoto 0.48", "#e45756"),
        "mcs_0_70": ("MCS 0.70", "#72b7b2"),
        "similarity_bucket_majority": ("Bucket majority", "#54a24b"),
        "assay_type_tanimoto_0_50": ("Assay-type Tanimoto 0.50", "#b279a2"),
    }
    fallback_colors = ["#9d755d", "#bab0ac", "#4c78a8"]
    for index, (key, baseline) in enumerate(metrics["baselines"].items()):
        label, color = baseline_labels.get(key, (key, fallback_colors[index % len(fallback_colors)]))
        if baseline.get("n", 0) == 0:
            continue
        names.append(label)
        values.append(baseline["macro_f1"])
        colors.append(color)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(names, values, color=colors)
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    for index, value in enumerate(values):
        ax.text(index, value + 0.015, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def error_result(record: dict[str, Any], sample_index: int, error: str) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "pair_index": record.get("pair_index"),
        "status": "error",
        "prediction": "",
        "label": record.get("label"),
        "correct": False,
        "error": error,
        "input_record": compact_input_record(record),
        "baseline_tanimoto_0_50_prediction": record.get("baseline_tanimoto_0_50_prediction", ""),
        "baseline_tanimoto_0_48_prediction": record.get("baseline_tanimoto_0_48_prediction", ""),
        "baseline_mcs_0_70_prediction": record.get("baseline_mcs_0_70_prediction", ""),
        "baseline_similarity_bucket_majority_prediction": record.get(
            "baseline_similarity_bucket_majority_prediction", ""
        ),
        "baseline_assay_type_tanimoto_0_50_prediction": record.get(
            "baseline_assay_type_tanimoto_0_50_prediction", ""
        ),
    }


def write_trace_jsonl(path: Path, results: list[dict[str, Any]]) -> None:
    write_jsonl(path, [trace_record(row) for row in results])


def trace_record(result: dict[str, Any]) -> dict[str, Any]:
    content = result.get("parsed_content")
    input_record = result.get("input_record") or {}
    sample_index = result.get("sample_index")
    query_smiles = input_record.get("query_smiles") or ""
    messages = trace_messages(result)
    return {
        "task": "activity_transfer",
        "index": sample_index,
        "sample_id": sample_index,
        "molecule_key": f"activity_transfer:{sample_index}",
        "smiles": query_smiles,
        "label": result.get("label"),
        "prediction": result.get("prediction"),
        "status": result.get("status"),
        "response_text": json.dumps(content, ensure_ascii=False, indent=2)
        if content is not None
        else result.get("raw_content") or result.get("error", ""),
        "messages": messages,
        "tool_count": result.get("tool_count", 0),
        "usage": result.get("usage") or {},
        "raw_output": {
            "pair_index": result.get("pair_index"),
            "correct": result.get("correct"),
            "confidence": result.get("confidence"),
            "reasoning_content": result.get("reasoning_content", ""),
            "has_reasoning_content": result.get("has_reasoning_content", False),
            "input_record": input_record,
            "tool_results": result.get("tool_results") or [],
            "baseline_tanimoto_0_50_prediction": result.get("baseline_tanimoto_0_50_prediction"),
            "baseline_tanimoto_0_48_prediction": result.get("baseline_tanimoto_0_48_prediction"),
            "baseline_mcs_0_70_prediction": result.get("baseline_mcs_0_70_prediction"),
            "baseline_similarity_bucket_majority_prediction": result.get(
                "baseline_similarity_bucket_majority_prediction"
            ),
            "baseline_assay_type_tanimoto_0_50_prediction": result.get(
                "baseline_assay_type_tanimoto_0_50_prediction"
            ),
        },
    }


def trace_messages(result: dict[str, Any]) -> list[dict[str, Any]]:
    messages = [json.loads(json.dumps(message, ensure_ascii=False, default=str)) for message in result.get("visible_messages") or []]
    reasoning = result.get("reasoning_content")
    if reasoning:
        for message in reversed(messages):
            if message.get("role") == "assistant":
                message.setdefault("reasoning", reasoning)
                break
    return messages


def compact_input_record(record: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "task_name",
        "label_mode",
        "activity_scale",
        "assay_chembl_id",
        "standard_type",
        "raw_units",
        "assay_tier",
        "target_pref_name",
        "query_smiles",
        "reference_smiles",
        "query_molecule_chembl_id",
        "reference_molecule_chembl_id",
        "reference_activity_value",
        "hidden_query_activity_value",
        "reference_raw_activity_value",
        "hidden_query_raw_activity_value",
        "reference_log_raw_activity_value",
        "hidden_query_log_raw_activity_value",
        "reference_pchembl_value",
        "hidden_query_pchembl_value",
        "abs_activity_delta",
        "normalized_delta",
        "robust_sigma",
        "tanimoto",
        "similarity_bucket",
        "mean_mcs_coverage",
        "mcs_timed_out",
        "input_format",
        "hf_completion",
        "hf_sample_id",
        "hf_pair_id",
        "hf_metadata",
    ]
    compact = {key: record.get(key) for key in keys}
    if record.get("input_format") == "hf_prompt_completion":
        metadata = record.get("metadata") or {}
        compact.update(
            {
                "hf_completion": record.get("completion"),
                "hf_sample_id": metadata.get("sample_id"),
                "hf_pair_id": metadata.get("pair_id"),
                "hf_metadata": metadata,
            }
        )
    return compact


def normalize_prediction(value: Any) -> str:
    raw_text = str(value or "").strip()
    compact = raw_text.upper().strip(" .:;()[]{}")
    if compact == "A":
        return "similar"
    if compact == "B":
        return "different"
    match = re.search(r"(?:^|\b)(?:ANSWER\s*[:\-]?\s*)?([AB])(?:\b|$)", raw_text.upper())
    if match:
        return "similar" if match.group(1) == "A" else "different"
    text = raw_text.lower()
    if text in {"similar", "transferable", "yes", "true"}:
        return "similar"
    if text in {"different", "not_similar", "not transferable", "not_transferable", "no", "false"}:
        return "different"
    return ""


def extract_prediction(content: Any) -> str:
    if not isinstance(content, dict):
        return normalize_prediction(content)
    for key in (
        "predicted_transferability",
        "prediction",
        "transferability",
        "predicted_label",
        "label",
    ):
        prediction = normalize_prediction(content.get(key))
        if prediction:
            return prediction
    prediction = normalize_prediction(content.get("unparsed_text"))
    if prediction:
        return prediction
    return ""


def parse_json_content(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {"unparsed_text": content}


def extract_think_content(content: str) -> str:
    matches = re.findall(r"<think>\s*(.*?)\s*</think>", content or "", flags=re.IGNORECASE | re.DOTALL)
    return "\n\n".join(match.strip() for match in matches if match.strip())


def strip_think_blocks(content: str) -> str:
    return re.sub(r"<think>\s*.*?\s*</think>", "", content or "", flags=re.IGNORECASE | re.DOTALL).strip()


def json_safe_message(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return json.loads(json.dumps(message, ensure_ascii=False, default=str))
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json")
    return {"role": "unknown", "content": str(message)}


def extract_reasoning_content(message: Any) -> str:
    keys = {"reasoning_content", "reasoning", "thinking", "reasoning_details"}
    found = find_reasoning_value(message, keys)
    if found is None:
        return ""
    if isinstance(found, str):
        return found
    return json.dumps(found, ensure_ascii=False)


def find_reasoning_value(value: Any, keys: set[str]) -> Any:
    if isinstance(value, dict):
        for key in keys:
            item = value.get(key)
            if item:
                return item
        for item in value.values():
            found = find_reasoning_value(item, keys)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = find_reasoning_value(item, keys)
            if found:
                return found
    return None


def usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return usage.model_dump(mode="json") if hasattr(usage, "model_dump") else dict(usage)


def sum_usage(responses: list[Any]) -> dict[str, int]:
    return sum_usage_dicts([usage_dict(response) for response in responses])


def sum_usage_dicts(items: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for item in items:
        for key, value in item.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key
    key = os.getenv(args.api_key_env) or os.getenv(args.api_key_env_fallback)
    if key:
        return key
    raise SystemExit(
        f"Missing API key env var: {args.api_key_env} or {args.api_key_env_fallback}. "
        "For vLLM, set the server's expected bearer token."
    )


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        # Match ChEMBL reasoning pipelines: the explicit env file is the run
        # configuration source of truth and should override inherited shells.
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_indices(args: argparse.Namespace, n_records: int) -> list[int]:
    if args.indices:
        return [index for index in args.indices if 0 <= index < n_records]
    limit = args.limit or n_records
    return list(range(min(limit, n_records)))


def parse_base_urls(args: argparse.Namespace) -> list[str]:
    if args.base_urls:
        urls = [item.strip() for item in args.base_urls.split(",") if item.strip()]
        if urls:
            return urls
    return [args.base_url]


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


if __name__ == "__main__":
    raise SystemExit(main())
