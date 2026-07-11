"""Shared batch runner for molecule-level reasoning tasks."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BatchConfig:
    description: str
    default_input: str
    default_batch_root: str
    default_index: str
    default_model: str
    batch_id_prefix: str
    pipeline_module: str
    log_prefix: str
    report_title: str
    prediction_field: str
    canonical_positive: str
    canonical_negative: str
    positive_predictions: frozenset[str]
    negative_predictions: frozenset[str]


@dataclass(frozen=True)
class BatchItem:
    index: int
    record: dict[str, Any]


def main(config: BatchConfig, argv: list[str] | None = None) -> int:
    args = _parse_args(config, argv)
    records = _read_jsonl(Path(args.input_jsonl))
    indices = _select_indices(args, len(records))
    batch_id = args.batch_id or time.strftime(f"{config.batch_id_prefix}_%Y%m%d_%H%M%S")
    batch_dir = _ensure_dir(Path(args.batch_root) / batch_id)
    logs_dir = _ensure_dir(batch_dir / "logs")
    batch_run_root = _ensure_dir(batch_dir / "runs")

    items = [BatchItem(index=i, record=records[i]) for i in indices]
    manifest = {
        "batch_id": batch_id,
        "input_jsonl": args.input_jsonl,
        "smiles_field": args.smiles_field,
        "label_field": args.label_field,
        "n_items": len(items),
        "indices": indices,
        "parallelism": args.parallelism,
        "group_workers": args.group_workers,
        "save_trace": args.save_trace,
        "combine_traces": args.combine_traces,
        "stream_logs": args.stream_logs,
        "model": args.model,
        "experiment_mode": args.experiment_mode,
        "retrieval_source": args.retrieval_source,
        "identity_blind": args.identity_blind,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "transport_max_retries": 2,
        "single_analysis_source_batch": args.single_analysis_source_batch,
        "tier1_replacement_index": args.tier1_replacement_index,
        "tier1_replacement_groups": args.tier1_replacement_groups or [],
        "final_only_source_batch": args.final_only_source_batch,
        "started_at": _now(),
        "paths": {
            "batch_dir": str(batch_dir),
            "predictions": str(batch_dir / "predictions.jsonl"),
            "metrics": str(batch_dir / "metrics.json"),
            "report": str(batch_dir / "report.md"),
            "molecule_runs": str(batch_run_root),
            "combined_trace": str(batch_dir / "trace_messages.jsonl"),
            "logs": str(logs_dir),
        },
    }
    _write_json(batch_dir / "manifest.json", manifest)

    _log(config, f"batch_id={batch_id}")
    _log(config, f"items={len(items)} parallelism={args.parallelism}")

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallelism) as executor:
        future_to_item = {
            executor.submit(_run_one, config, args, item, batch_id, batch_run_root, logs_dir): item
            for item in items
        }
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - batch should keep going.
                result = _error_result(config, args, item, batch_id, str(exc))
            results.append(result)
            _log(
                config,
                f"progress {len(results)}/{len(items)} done; index={item.index} "
                f"status={result.get('status')} prediction={result.get(config.prediction_field)} "
                f"correct={result.get('correct')} latency_s={result.get('latency_s')}",
            )

    results.sort(key=lambda row: row["query_index"])
    if args.save_trace and args.combine_traces:
        _combine_traces(batch_dir / "trace_messages.jsonl", results)

    metrics = compute_metrics(config, results)
    metrics["finished_at"] = _now()
    _write_jsonl(batch_dir / "predictions.jsonl", results)
    _write_json(batch_dir / "metrics.json", metrics)
    _write_report(config, batch_dir / "report.md", manifest, metrics, results)

    manifest["finished_at"] = metrics["finished_at"]
    _write_json(batch_dir / "manifest.json", manifest)
    print(json.dumps({"manifest": manifest, "metrics": metrics}, ensure_ascii=False, indent=2), flush=True)
    return 0 if metrics["n_failed_runs"] == 0 else 1


def _run_one(
    config: BatchConfig,
    args: argparse.Namespace,
    item: BatchItem,
    batch_id: str,
    run_root: Path,
    logs_dir: Path,
) -> dict[str, Any]:
    run_id = f"{batch_id}_idx{item.index:05d}"
    run_dir = run_root / run_id
    stdout_path = logs_dir / f"{run_id}.stdout.log"
    stderr_path = logs_dir / f"{run_id}.stderr.log"
    started = time.monotonic()

    existing_final_path = run_dir / "final_reasoning_output.json"
    should_skip_existing = False
    if args.skip_existing and existing_final_path.exists():
        existing_result = _collect_result(config, args, item, run_id, run_dir)
        should_skip_existing = _result_is_complete(existing_result)
        if not should_skip_existing:
            _log(config, f"rerun invalid existing final index={item.index} run_id={run_id}")

    if should_skip_existing:
        returncode = 0
        if not stdout_path.exists():
            stdout_path.write_text("", encoding="utf-8")
        if not stderr_path.exists():
            stderr_path.write_text("skipped existing run\n", encoding="utf-8")
    else:
        if args.final_only_source_batch:
            _prepare_final_only_run_dir(args, item.index, run_id, run_dir)
            command = _final_only_command(config, args, run_dir)
        else:
            command = _single_run_command(config, args, item.index, run_id, run_root)
        _log(config, f"start index={item.index} run_id={run_id}")
        returncode = _run_subprocess_with_logs(
            command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            stream_logs=args.stream_logs,
            prefix=f"idx{item.index:05d}",
        )

    if not args.save_trace:
        trace_path = run_dir / "trace_messages.jsonl"
        if trace_path.exists():
            trace_path.unlink()

    result = _collect_result(config, args, item, run_id, run_dir)
    result.update(
        {
            "status": (
                "ok"
                if returncode == 0 and _result_is_complete(result)
                else "error"
            ),
            "returncode": returncode,
            "latency_s": round(time.monotonic() - started, 3),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }
    )
    if returncode != 0:
        result["error"] = _tail_text(stderr_path, stdout_path)
    return result


def _prepare_final_only_run_dir(args: argparse.Namespace, query_index: int, run_id: str, run_dir: Path) -> None:
    source_batch_dir = Path(args.final_only_source_batch)
    if not source_batch_dir.exists():
        raise FileNotFoundError(f"Final-only source batch does not exist: {source_batch_dir}")
    source_runs_dir = source_batch_dir / "runs"
    matches = sorted(source_runs_dir.glob(f"*_idx{query_index:05d}"))
    if not matches:
        raise FileNotFoundError(f"No source run for query index {query_index}: {source_runs_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous source runs for query index {query_index}: {matches}")
    source_run_dir = matches[0]
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("retrieval.json", "single_molecule_reasoning_output.json", "group_reasoning_outputs.jsonl"):
        source_path = source_run_dir / name
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source artifact for final-only rerun: {source_path}")
        shutil.copy2(source_path, run_dir / name)

    manifest_path = source_run_dir / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    manifest["run_id"] = run_id
    manifest["final_only_source_run_dir"] = str(source_run_dir)
    manifest["final_only_source_batch"] = str(source_batch_dir)
    manifest.setdefault("paths", {})
    manifest["paths"].update(
        {
            "retrieval": str(run_dir / "retrieval.json"),
            "single_molecule_reasoning_output": str(run_dir / "single_molecule_reasoning_output.json"),
            "group_reasoning_outputs": str(run_dir / "group_reasoning_outputs.jsonl"),
            "final_reasoning_output": str(run_dir / "final_reasoning_output.json"),
            "trace_messages": str(run_dir / "trace_messages.jsonl"),
        }
    )
    _write_json(run_dir / "manifest.json", manifest)


def _final_only_command(config: BatchConfig, args: argparse.Namespace, run_dir: Path) -> list[str]:
    command = [
        args.python_executable,
        "-m",
        config.pipeline_module,
        "--resume-final-from-run-dir",
        str(run_dir),
        "--env-file",
        args.env_file,
        "--api-key-env",
        args.api_key_env,
        "--base-url",
        args.base_url,
        "--tool-service-url",
        args.tool_service_url,
        "--model",
        args.model,
        "--timeout-s",
        str(args.timeout_s),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--max-tool-rounds",
        str(args.max_tool_rounds),
        "--reasoning-effort",
        args.reasoning_effort,
    ]
    if not args.enable_thinking:
        command.append("--disable-thinking")
    else:
        command.append("--enable-thinking")
    return command


def _single_run_command(
    config: BatchConfig,
    args: argparse.Namespace,
    query_index: int,
    run_id: str,
    run_root: Path,
) -> list[str]:
    command = [
        args.python_executable,
        "-m",
        config.pipeline_module,
        "--input-jsonl",
        args.input_jsonl,
        "--query-index",
        str(query_index),
        "--smiles-field",
        args.smiles_field,
        "--index",
        args.index,
        "--experiment-mode",
        args.experiment_mode,
        "--retrieval-source",
        args.retrieval_source,
        "--out-root",
        str(run_root),
        "--run-id",
        run_id,
        "--env-file",
        args.env_file,
        "--api-key-env",
        args.api_key_env,
        "--base-url",
        args.base_url,
        "--tool-service-url",
        args.tool_service_url,
        "--model",
        args.model,
        "--max-workers",
        str(args.group_workers),
        "--timeout-s",
        str(args.timeout_s),
        "--max-tokens",
        str(args.max_tokens),
        "--max-tool-rounds",
        str(args.max_tool_rounds),
        "--reasoning-effort",
        args.reasoning_effort,
        "--top-k-per-group",
        str(args.top_k_per_group),
        "--min-similarity",
        str(args.min_similarity),
    ]
    if not args.enable_thinking:
        command.append("--disable-thinking")
    else:
        command.append("--enable-thinking")
    if args.max_groups:
        command.extend(["--max-groups", str(args.max_groups)])
    if args.groups:
        command.append("--groups")
        command.extend(args.groups)
    if args.tier1_replacement_index:
        command.extend(["--tier1-replacement-index", args.tier1_replacement_index])
        if args.tier1_replacement_groups:
            command.append("--tier1-replacement-groups")
            command.extend(args.tier1_replacement_groups)
    if args.disable_group_tools:
        command.append("--disable-group-tools")
    if args.identity_blind:
        command.append("--identity-blind")
    if args.single_analysis_source_batch:
        source_batch = Path(args.single_analysis_source_batch)
        source_run_id = f"{source_batch.name}_idx{query_index:05d}"
        command.extend(
            [
                "--single-analysis-source-run-dir",
                str(source_batch / "runs" / source_run_id),
            ]
        )
    return command


def _run_subprocess_with_logs(
    command: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    stream_logs: bool,
    prefix: str,
) -> int:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        stdout_thread = threading.Thread(
            target=_copy_stream,
            args=(process.stdout, stdout_file, False, prefix, "stdout"),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_copy_stream,
            args=(process.stderr, stderr_file, stream_logs, prefix, "stderr"),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        returncode = process.wait()
        stdout_thread.join()
        stderr_thread.join()
    return returncode


def _copy_stream(stream: Any, log_file: Any, echo: bool, prefix: str, stream_name: str) -> None:
    if stream is None:
        return
    for line in stream:
        log_file.write(line)
        log_file.flush()
        if echo:
            print(f"[{prefix} {stream_name}] {line}", end="", file=sys.stderr, flush=True)


def _tail_text(*paths: Path, limit: int = 4000) -> str:
    chunks = []
    for path in paths:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            if text:
                chunks.append(f"==> {path}\n{text[-limit:]}")
    return "\n".join(chunks) or "subprocess failed"


def _collect_result(
    config: BatchConfig,
    args: argparse.Namespace,
    item: BatchItem,
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    final_path = run_dir / "final_reasoning_output.json"
    single_path = run_dir / "single_molecule_reasoning_output.json"
    group_path = run_dir / "group_reasoning_outputs.jsonl"
    manifest_path = run_dir / "manifest.json"
    final_output = _read_json(final_path) if final_path.exists() else {}
    single_output = _read_json(single_path) if single_path.exists() else {}
    group_outputs = _read_jsonl(group_path) if group_path.exists() else []
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    content = ((final_output.get("llm") or {}).get("content") or {}) if isinstance(final_output, dict) else {}
    prediction = _normalize_prediction(config, content.get(config.prediction_field))
    pred_label = prediction_to_label(config, prediction)
    true_label = _parse_label(item.record.get(args.label_field))
    correct = bool(pred_label == true_label) if pred_label is not None and true_label is not None else False
    return {
        "query_index": item.index,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "smiles": item.record.get(args.smiles_field, ""),
        "label": true_label,
        config.prediction_field: prediction,
        "pred_label": pred_label,
        "confidence": content.get("confidence"),
        "correct": correct,
        "final_status": (final_output.get("status") if isinstance(final_output, dict) else None),
        "single_status": (single_output.get("status") if isinstance(single_output, dict) else None),
        "n_group_outputs": len(group_outputs),
        "n_failed_group_outputs": sum(row.get("status") != "ok" for row in group_outputs),
        "n_groups_with_neighbors": manifest.get("n_groups_with_neighbors"),
        "trace_messages": str(run_dir / "trace_messages.jsonl") if (run_dir / "trace_messages.jsonl").exists() else "",
        "final_reasoning_output": str(final_path) if final_path.exists() else "",
        "final_summary": content.get("final_summary", ""),
    }


def _result_is_complete(result: dict[str, Any]) -> bool:
    if result.get("final_status") != "ok" or result.get("pred_label") is None:
        return False
    if result.get("single_status") != "ok":
        return False
    expected_groups = result.get("n_groups_with_neighbors")
    if expected_groups is not None and int(result.get("n_group_outputs") or 0) != int(expected_groups):
        return False
    return int(result.get("n_failed_group_outputs") or 0) == 0


def compute_metrics(config: BatchConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [row for row in rows if row.get("label") in (0, 1)]
    successful = [row for row in evaluable if row.get("status") == "ok"]
    correct = sum(1 for row in successful if row.get("correct"))
    per_class = {str(label): _class_metrics(successful, label) for label in (0, 1)}
    macro_f1 = round(sum(per_class[str(label)]["f1"] for label in (0, 1)) / 2, 6) if successful else 0.0
    prediction_distribution: dict[str, int] = {}
    for row in successful:
        prediction = str(row.get(config.prediction_field) or "missing")
        prediction_distribution[prediction] = prediction_distribution.get(prediction, 0) + 1
    confusion_matrix = {
        "tn": sum(1 for row in successful if row.get("label") == 0 and row.get("pred_label") == 0),
        "fp": sum(1 for row in successful if row.get("label") == 0 and row.get("pred_label") == 1),
        "fn": sum(1 for row in successful if row.get("label") == 1 and row.get("pred_label") == 0),
        "tp": sum(1 for row in successful if row.get("label") == 1 and row.get("pred_label") == 1),
    }
    positive_class = per_class["1"]
    return {
        "n_total": len(rows),
        "n_evaluable": len(evaluable),
        "n_successful": len(successful),
        "n_failed_runs": sum(1 for row in rows if row.get("status") != "ok"),
        "accuracy": _safe_div(correct, len(successful)),
        "accuracy_including_failed": _safe_div(correct, len(evaluable)),
        "macro_f1": macro_f1,
        "per_class": per_class,
        "positive_class_precision": positive_class["precision"],
        "positive_class_recall": positive_class["recall"],
        "positive_class_f1": positive_class["f1"],
        "confusion_matrix": confusion_matrix,
        "prediction_distribution": prediction_distribution,
    }


def _class_metrics(rows: list[dict[str, Any]], label: int) -> dict[str, Any]:
    tp = sum(1 for row in rows if row.get("label") == label and row.get("pred_label") == label)
    fp = sum(1 for row in rows if row.get("label") != label and row.get("pred_label") == label)
    fn = sum(1 for row in rows if row.get("label") == label and row.get("pred_label") != label)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def prediction_to_label(config: BatchConfig, prediction: str | None) -> int | None:
    normalized = str(prediction or "").strip().lower()
    if normalized in config.positive_predictions or normalized == config.canonical_positive:
        return 1
    if normalized in config.negative_predictions or normalized == config.canonical_negative:
        return 0
    return None


def _normalize_prediction(config: BatchConfig, value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in config.positive_predictions:
        return config.canonical_positive
    if text in config.negative_predictions:
        return config.canonical_negative
    return text or "missing"


def _parse_label(value: Any) -> int | None:
    try:
        label = int(value)
    except (TypeError, ValueError):
        return None
    return label if label in (0, 1) else None


def _safe_div(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _select_indices(args: argparse.Namespace, n_records: int) -> list[int]:
    if args.indices:
        indices = []
        for token in args.indices:
            indices.extend(_parse_index_token(token))
    else:
        end = n_records if args.limit <= 0 else min(n_records, args.start + args.limit)
        indices = list(range(args.start, end))
    bad = [index for index in indices if index < 0 or index >= n_records]
    if bad:
        raise SystemExit(f"Query indices out of range 0..{n_records - 1}: {bad}")
    return sorted(dict.fromkeys(indices))


def _parse_index_token(token: str) -> list[int]:
    if "-" not in token:
        return [int(token)]
    start, end = token.split("-", 1)
    return list(range(int(start), int(end) + 1))


def _error_result(
    config: BatchConfig,
    args: argparse.Namespace,
    item: BatchItem,
    batch_id: str,
    error: str,
) -> dict[str, Any]:
    run_id = f"{batch_id}_idx{item.index:05d}"
    return {
        "query_index": item.index,
        "run_id": run_id,
        "status": "error",
        "smiles": item.record.get(args.smiles_field, ""),
        "label": _parse_label(item.record.get(args.label_field)),
        config.prediction_field: "missing",
        "pred_label": None,
        "correct": False,
        "error": error,
    }


def _combine_traces(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            trace = row.get("trace_messages")
            if not trace:
                continue
            trace_path = Path(trace)
            if trace_path.exists():
                out.write(trace_path.read_text(encoding="utf-8"))


def _write_report(
    config: BatchConfig,
    path: Path,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        f"# {config.report_title}: {manifest['batch_id']}",
        "",
        f"- input_jsonl: `{manifest['input_jsonl']}`",
        f"- n_items: {metrics['n_total']}",
        f"- n_successful: {metrics['n_successful']}",
        f"- n_failed_runs: {metrics['n_failed_runs']}",
        f"- accuracy: {metrics['accuracy']}",
        f"- accuracy_including_failed: {metrics['accuracy_including_failed']}",
        f"- macro_f1: {metrics['macro_f1']}",
        f"- positive_class_precision: {metrics['positive_class_precision']}",
        f"- positive_class_recall: {metrics['positive_class_recall']}",
        f"- positive_class_f1: {metrics['positive_class_f1']}",
        f"- confusion_matrix: `{json.dumps(metrics['confusion_matrix'], ensure_ascii=False)}`",
        f"- prediction_distribution: `{json.dumps(metrics['prediction_distribution'], ensure_ascii=False)}`",
        "",
        "| index | label | prediction | pred_label | correct | confidence | run_id |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {query_index} | {label} | {prediction} | {pred_label} | {correct} | {confidence} | {run_id} |".format(
                **{
                    "prediction": row.get(config.prediction_field, ""),
                    **{key: row.get(key, "") for key in [
                        "query_index",
                        "label",
                        "pred_label",
                        "correct",
                        "confidence",
                        "run_id",
                    ]},
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _log(config: BatchConfig, message: str) -> None:
    print(f"[{config.log_prefix}] {message}", file=sys.stderr, flush=True)


def _parse_args(config: BatchConfig, argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=config.description)
    parser.add_argument("--input-jsonl", default=config.default_input)
    parser.add_argument("--smiles-field", default="drug")
    parser.add_argument("--label-field", default="Y")
    parser.add_argument("--index", default=config.default_index)
    parser.add_argument(
        "--experiment-mode",
        choices=["none", "direct", "full_flat", "full_mechanism", "native"],
        default="native",
    )
    parser.add_argument("--retrieval-source", default="chembl")
    parser.add_argument("--identity-blind", action="store_true")
    parser.add_argument(
        "--single-analysis-source-batch",
        default="",
        help="Reuse each query's frozen single-molecule branch from another batch.",
    )
    parser.add_argument("--batch-root", default=config.default_batch_root)
    parser.add_argument("--batch-id", default="")
    parser.add_argument(
        "--final-only-source-batch",
        default="",
        help="Existing batch directory whose retrieval/single/group artifacts should be reused for final-only reruns.",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--indices", nargs="*", default=None, help="Indices or inclusive ranges, e.g. 0 3 5-8.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all records from --start.")
    parser.add_argument("--parallelism", type=int, default=1, help="Number of molecules to run concurrently.")
    parser.add_argument("--group-workers", type=int, default=4, help="Per-molecule group-level LLM workers.")
    parser.add_argument("--save-trace", dest="save_trace", action="store_true", default=True)
    parser.add_argument("--no-save-trace", dest="save_trace", action="store_false")
    parser.add_argument("--combine-traces", dest="combine_traces", action="store_true", default=True)
    parser.add_argument("--no-combine-traces", dest="combine_traces", action="store_false")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stream-logs", dest="stream_logs", action="store_true", default=True)
    parser.add_argument("--no-stream-logs", dest="stream_logs", action="store_false")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--tool-service-url", default="http://127.0.0.1:8765")
    parser.add_argument("--model", default=config.default_model)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=20480)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tool-rounds", type=int, default=10)
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        help="OpenAI-compatible reasoning_effort value. Use an empty string to omit this parameter.",
    )
    parser.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", default=True)
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--top-k-per-group", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.3)
    parser.add_argument("--groups", nargs="*", default=None, help="Optional exact Tier.endpoint_group ids to reason over.")
    parser.add_argument(
        "--tier1-replacement-index",
        default="",
        help="Optional pipeline-specific neighbor index used to replace base Tier 1 retrieval groups.",
    )
    parser.add_argument(
        "--tier1-replacement-groups",
        nargs="*",
        default=None,
        help="Optional group ids to retrieve from --tier1-replacement-index.",
    )
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--disable-group-tools", action="store_true")
    args = parser.parse_args(argv)
    args.groups = _normalize_group_args(args.groups)
    args.tier1_replacement_groups = _normalize_group_args(args.tier1_replacement_groups)
    return args


def _normalize_group_args(groups: list[str] | None) -> list[str] | None:
    if not groups:
        return groups
    normalized: list[str] = []
    i = 0
    while i < len(groups):
        group = groups[i]
        if group in {"Tier", "Starling", "Combined"} and i + 1 < len(groups) and "." in groups[i + 1]:
            normalized.append(f"{group} {groups[i + 1]}")
            i += 2
        else:
            normalized.append(group)
            i += 1
    return normalized
