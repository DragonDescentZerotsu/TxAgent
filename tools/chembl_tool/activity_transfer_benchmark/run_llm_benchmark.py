"""Run an LLM assay-activity transfer benchmark on sampled ChEMBL pairs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI


DEFAULT_INPUT = "outputs/chembl_tool/activity_transfer_benchmark/llm_eval_sets/dynamic_v1_llm_3k/eval_pairs.jsonl"
DEFAULT_OUT_ROOT = "outputs/chembl_tool/activity_transfer_benchmark/llm_runs"
DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_TOOL_SERVICE_URL = "http://127.0.0.1:8765"
DEFAULT_MODEL = "gpt-oss-120b"

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
    load_env(Path(args.env_file))
    api_key = resolve_api_key(args)
    records = read_jsonl(Path(args.input_jsonl))
    selected_indices = select_indices(args, len(records))

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
            max_tokens=args.max_tokens,
            tool_service_url=args.tool_service_url,
            enable_tools=not args.disable_tools,
            max_tool_rounds=args.max_tool_rounds,
            response_format=not args.disable_response_format,
            reasoning_effort=args.reasoning_effort,
            enable_thinking=args.enable_thinking,
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
                with lock:
                    print(
                        "[activity_transfer_llm] "
                        f"progress {completed}/{len(futures)} "
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
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-tool-rounds", type=int, default=2)
    parser.add_argument("--disable-tools", action="store_true")
    parser.add_argument("--disable-response-format", action="store_true")
    parser.add_argument("--reasoning-effort", default="", help="Optional OpenAI-compatible reasoning_effort value.")
    parser.add_argument("--enable-thinking", action="store_true", help="Send extra_body thinking enabled.")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args(argv)


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
                "content": "Return the required compact JSON now. Do not call more tools.",
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
        if self.enable_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
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
) -> dict[str, Any]:
    run_path = runs_dir / f"sample_{sample_index:05d}.json"
    if skip_existing and run_path.exists():
        return json.loads(run_path.read_text(encoding="utf-8"))
    started = time.monotonic()
    messages = build_messages(record)
    response = client.chat_json(messages)
    content = response["content"]
    prediction = extract_prediction(content)
    if not prediction:
        retry_messages = messages + [
            {
                "role": "user",
                "content": (
                    "Your previous answer was invalid or missing `predicted_transferability`. "
                    "Return exactly one compact JSON object with keys: "
                    "predicted_transferability, confidence, should_transfer_activity, "
                    "reasoning_summary, key_factors, rationale. "
                    "Use predicted_transferability as either `similar` or `different`."
                ),
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
        "baseline_tanimoto_0_50_prediction": record["baseline_tanimoto_0_50_prediction"],
        "baseline_tanimoto_0_48_prediction": record["baseline_tanimoto_0_48_prediction"],
        "baseline_mcs_0_70_prediction": record["baseline_mcs_0_70_prediction"],
    }
    if not prediction:
        result["error"] = "Missing or invalid predicted_transferability."
    write_json(run_path, result)
    return result


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are evaluating analog assay activity transferability for medicinal chemistry. "
        "Return only compact JSON. The hidden query activity is not provided. "
        "Use assay context and structural/property similarity to decide whether the reference activity "
        "is likely transferable to the query molecule in the same assay endpoint."
    )
    user = {
        "task": (
            "Predict whether the query molecule's pChEMBL activity is similar to the reference molecule's "
            "activity in this same ChEMBL assay endpoint. Use predicted_transferability='similar' when "
            "|delta pChEMBL| is expected to be <= 0.5, and 'different' when it is expected to be >= 1.0."
        ),
        "assay": {
            "assay_chembl_id": record["assay_chembl_id"],
            "standard_type": record["standard_type"],
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
        "reference_molecule": {
            "chembl_id": record["reference_molecule_chembl_id"],
            "smiles": record["reference_smiles"],
            "known_pchembl_value": record["reference_pchembl_value"],
        },
        "query_molecule": {
            "chembl_id": record["query_molecule_chembl_id"],
            "smiles": record["query_smiles"],
            "pchembl_value": "hidden_for_evaluation",
        },
        "precomputed_similarity": {
            "morgan_tanimoto": record["tanimoto"],
            "similarity_bucket": record["similarity_bucket"],
            "mean_mcs_coverage": record["mean_mcs_coverage"],
            "query_mcs_coverage": record["query_mcs_coverage"],
            "reference_mcs_coverage": record["reference_mcs_coverage"],
            "mcs_timed_out": record["mcs_timed_out"],
        },
        "output_schema": {
            "predicted_transferability": "similar or different",
            "confidence": "high, moderate, or low",
            "should_transfer_activity": "boolean",
            "reasoning_summary": [
                "2-5 concise visible reasoning steps based only on assay context, structure, properties, and tools"
            ],
            "key_factors": ["short strings"],
            "rationale": "one concise paragraph",
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


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
    return {
        "content": parse_json_content(content),
        "raw_content": content,
        "reasoning_content": extract_reasoning_content(final_message),
        "final_message": final_message,
        "messages": trace_messages,
        "usage": sum_usage(responses or [response]),
        "model": response.model or "",
        "id": response.id or "",
        "tool_results": tool_results,
    }


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok_results = [row for row in results if row.get("status") == "ok"]
    metrics = {
        "n": len(results),
        "n_ok": len(ok_results),
        "n_failed": len(results) - len(ok_results),
        "llm": metrics_for_predictions(ok_results, "prediction"),
        "baselines": {
            "tanimoto_0_50": metrics_for_predictions(ok_results, "baseline_tanimoto_0_50_prediction"),
            "tanimoto_0_48": metrics_for_predictions(ok_results, "baseline_tanimoto_0_48_prediction"),
            "mcs_0_70": metrics_for_predictions(ok_results, "baseline_mcs_0_70_prediction"),
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
        row for row in ok_results if row["baseline_tanimoto_0_50_prediction"] != row["label"]
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
    return metrics


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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_line(name: str, values: dict[str, Any]) -> str:
    return f"| {name} | {values['accuracy']:.4f} | {values['balanced_accuracy']:.4f} | {values['macro_f1']:.4f} |"


def plot_metrics(path: Path, metrics: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["LLM", "Tanimoto 0.50", "Tanimoto 0.48", "MCS 0.70"]
    values = [
        metrics["llm"]["macro_f1"],
        metrics["baselines"]["tanimoto_0_50"]["macro_f1"],
        metrics["baselines"]["tanimoto_0_48"]["macro_f1"],
        metrics["baselines"]["mcs_0_70"]["macro_f1"],
    ]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(names, values, color=["#4c78a8", "#f58518", "#e45756", "#72b7b2"])
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
        "assay_chembl_id",
        "standard_type",
        "target_pref_name",
        "query_smiles",
        "reference_smiles",
        "query_molecule_chembl_id",
        "reference_molecule_chembl_id",
        "reference_pchembl_value",
        "hidden_query_pchembl_value",
        "abs_activity_delta",
        "tanimoto",
        "similarity_bucket",
        "mean_mcs_coverage",
        "mcs_timed_out",
    ]
    return {key: record.get(key) for key in keys}


def normalize_prediction(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"similar", "transferable", "yes", "true"}:
        return "similar"
    if text in {"different", "not_similar", "not transferable", "not_transferable", "no", "false"}:
        return "different"
    return ""


def extract_prediction(content: Any) -> str:
    if not isinstance(content, dict):
        return ""
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
