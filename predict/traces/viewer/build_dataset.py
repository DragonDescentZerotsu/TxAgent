"""Adapt centralized traces to the existing paper-viewer dataset contract."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from predict.traces.catalog import discover_traces


def build_dataset(trace_root: Path, output_root: Path) -> int:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for trace in discover_traces(trace_root):
        grouped[(trace["experiment_id"], trace["task"], trace["harness"])].append(trace)

    for (experiment, task, harness), traces in grouped.items():
        condition = f"{experiment}__{harness}"
        condition_root = output_root / "runs" / task / condition
        by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for trace in traces:
            by_sample[str(trace["sample_id"])].append(trace)
        predictions = []
        for sample_id, sample_traces in sorted(by_sample.items()):
            run_root = condition_root / "runs" / sample_id
            run_root.mkdir(parents=True, exist_ok=True)
            ordered_traces = sorted(sample_traces, key=_stage_key)
            records = [_viewer_record(trace) for trace in ordered_traces]
            trace_path = run_root / "trace_messages.jsonl"
            trace_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                encoding="utf-8",
            )
            final_content = ((ordered_traces[-1].get("llm") or {}).get("content") or {})
            prediction = next(
                (value for key, value in final_content.items() if key.endswith("_prediction")),
                None,
            )
            query_index = int(sample_id.removeprefix("query_idx")) if sample_id.startswith("query_idx") else sample_id
            predictions.append(
                {
                    "query_index": query_index,
                    "run_id": sample_id,
                    "status": "ok",
                    "pred_label": prediction,
                    "confidence": final_content.get("confidence"),
                    "trace_messages": f"runs/{task}/{condition}/runs/{sample_id}/trace_messages.jsonl",
                }
            )
        (condition_root / "predictions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
            encoding="utf-8",
        )
        (condition_root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "predict_trace_viewer_dataset.v1",
                    "experiment_id": experiment,
                    "task": task,
                    "harness": harness,
                    "n_samples": len(predictions),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return len(grouped)


def _stage_key(trace: dict[str, Any]) -> tuple[int, str]:
    stage = str(trace.get("stage") or "")
    if stage == "single":
        return (0, stage)
    if stage.startswith("level_") and stage.removeprefix("level_").isdigit():
        return (int(stage.removeprefix("level_")), stage)
    if stage == "final":
        return (10_000, stage)
    return (100, stage)


def _viewer_record(trace: dict[str, Any]) -> dict[str, Any]:
    llm = trace.get("llm") or {}
    content = llm.get("content") or {}
    return {
        "task": trace.get("stage"),
        "status": "ok",
        "prediction": next(
            (value for key, value in content.items() if key.endswith("_prediction")),
            None,
        ),
        "response_text": llm.get("raw_content") or json.dumps(content, ensure_ascii=False),
        "messages": llm.get("messages") or [],
        "usage": llm.get("usage") or {},
        "tool_count": len(llm.get("tool_calls") or []),
        "input_prompt": trace.get("input_prompt") or "",
        "execution_provider": llm.get("execution_provider") or {},
        "source_trace": trace.get("path") or "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(build_dataset(args.trace_root, args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
