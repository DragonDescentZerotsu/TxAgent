"""Small shared helpers for batched harness-side tool prefetch."""

from __future__ import annotations

from typing import Any, Iterable

from predict.tools.client import ToolServiceClient


def invoke_with_retry(
    client: ToolServiceClient,
    calls: Iterable[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Batch calls once, then retry only failed positions once."""
    ordered_calls = list(calls)
    results = client.invoke_many(ordered_calls)
    if len(results) != len(ordered_calls):
        raise ValueError(
            f"tool prefetch result mismatch: {len(results)} != {len(ordered_calls)}"
        )
    for position, result in enumerate(results):
        if result.get("status") != "ok":
            tool_name, arguments = ordered_calls[position]
            results[position] = client.invoke(tool_name, arguments)
    return results
