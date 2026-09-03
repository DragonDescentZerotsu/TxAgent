"""Write one portable, viewer-facing copy of a completed model stage."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

from predict.utils.json import write_json_atomic


TRACE_SCHEMA_VERSION = "predict_trace.v1"
DEFAULT_TRACE_ROOT = Path("predict/traces/runs")


def input_prompt(messages: list[Mapping[str, Any]]) -> str:
    """Flatten every request-side message before the final assistant response."""
    request_messages = list(messages)
    if request_messages and request_messages[-1].get("role") == "assistant":
        request_messages.pop()
    sections = []
    for message in request_messages:
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, sort_keys=True)
        sections.append(f"[{str(message.get('role') or 'unknown').upper()}]\n{content}")
    return "\n\n".join(sections)


def write_trace(
    *,
    trace_root: str | Path,
    experiment_id: str,
    task: str,
    harness: str,
    sample_id: str,
    stage: str,
    checkpoint_path: str | Path,
    output: Mapping[str, Any],
) -> Path | None:
    llm = output.get("llm") if isinstance(output.get("llm"), Mapping) else output
    messages = llm.get("messages") if isinstance(llm, Mapping) else None
    if not isinstance(messages, list):
        return None
    path = (
        Path(trace_root)
        / _safe(experiment_id)
        / _safe(task)
        / _safe(harness)
        / _safe(sample_id)
        / f"{_safe(stage)}.json"
    )
    write_json_atomic(
        path,
        {
            "schema_version": TRACE_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "task": task,
            "harness": harness,
            "sample_id": sample_id,
            "stage": stage,
            "checkpoint_path": str(checkpoint_path),
            "input_prompt": input_prompt(messages),
            "llm": dict(llm),
        },
    )
    return path


def _safe(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "unnamed"

