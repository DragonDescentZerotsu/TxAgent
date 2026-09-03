"""Read branch inputs and write the historical branch trace artifact.

The three task pipelines use this module to stream one requested JSONL row and
to preserve their existing single/group/final trace format. It does not select
evidence, construct prompts, or call a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl_record(path: Path, index: int) -> dict[str, Any]:
    """Read one zero-based JSONL record without materializing the full file."""
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise SystemExit(f"No record at index {index}: {path}")


def write_trace_jsonl(
    path: Path,
    *,
    prediction_field: str,
    query_record: dict[str, Any],
    query_index: int,
    smiles: str,
    single_output: dict[str, Any],
    group_outputs: list[dict[str, Any]],
    final_output: dict[str, Any],
) -> None:
    """Write the frozen single/group/final trace contract for one query."""
    outputs = [
        ("single_molecule", single_output),
        *(
            (str(group.get("group_id") or "unknown_group"), group)
            for group in group_outputs
        ),
        ("final_summary", final_output),
    ]
    with path.open("w", encoding="utf-8") as handle:
        for task, output in outputs:
            handle.write(
                json.dumps(
                    _trace_record(
                        task,
                        output,
                        prediction_field=prediction_field,
                        query_record=query_record,
                        query_index=query_index,
                        smiles=smiles,
                    ),
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )


def _trace_record(
    task: str,
    output: dict[str, Any],
    *,
    prediction_field: str,
    query_record: dict[str, Any],
    query_index: int,
    smiles: str,
) -> dict[str, Any]:
    llm = output.get("llm") or {}
    content = llm.get("content")
    return {
        "task": task,
        "index": query_index,
        "sample_id": query_index,
        "molecule_key": f"index:{query_index}",
        "smiles": smiles,
        "label": query_record.get("Y"),
        "status": output.get("status"),
        "prediction": content.get(prediction_field)
        if isinstance(content, dict)
        else None,
        "response_text": (
            json.dumps(content, ensure_ascii=False, indent=2)
            if content is not None
            else output.get("error")
        ),
        "messages": llm.get("messages") or [],
        "tool_count": len(llm.get("tool_calls") or []),
        "usage": llm.get("usage") or {},
        "raw_output": {key: value for key, value in output.items() if key != "llm"},
    }
