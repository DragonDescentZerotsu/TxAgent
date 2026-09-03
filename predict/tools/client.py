"""Client for the resident molecular tool service."""

from __future__ import annotations

import json
from typing import Any

import requests


class ToolServiceClient:
    """Invoke allowlisted functions through the persistent molecular tool service."""

    def __init__(self, base_url: str, *, timeout_s: int):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def invoke_function_call(self, tool_call: Any, *, allowed_tool_names: set[str]) -> dict[str, Any]:
        tool_name = tool_call.function.name
        if tool_name not in allowed_tool_names:
            return {
                "tool_name": tool_name,
                "status": "error",
                "content": f"Tool `{tool_name}` is not allowed in this workflow.",
            }
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError as exc:
            return {
                "tool_name": tool_name,
                "status": "error",
                "content": f"Invalid JSON tool arguments: {exc}",
            }
        return self.invoke(tool_name, arguments)

    def invoke(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
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
        except requests.RequestException as exc:
            return {
                "tool_name": tool_name,
                "status": "error",
                "arguments": arguments,
                "content": f"{tool_name} request failed: {exc}",
            }
        if response.status_code >= 400:
            content = f"{tool_name} HTTP error {response.status_code}: {response.text[:1000]}"
            return {"tool_name": tool_name, "status": "error", "arguments": arguments, "content": content}
        return self._format_result(tool_name, arguments, response.json())

    def invoke_many(self, calls: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        if not calls:
            return []
        try:
            response = requests.post(
                f"{self.base_url}/tools/batch",
                json={
                    "requests": [
                        {
                            "tool_name": tool_name,
                            "version": "v1",
                            "input": arguments,
                            "options": {"timeout_s": self.timeout_s, "return_debug": False},
                        }
                        for tool_name, arguments in calls
                    ]
                },
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            payloads = response.json().get("responses") or []
            if len(payloads) != len(calls):
                raise ValueError(f"batch response count mismatch: {len(payloads)} != {len(calls)}")
        except (requests.RequestException, ValueError, TypeError, json.JSONDecodeError):
            return [self.invoke(tool_name, arguments) for tool_name, arguments in calls]
        return [
            self._format_result(tool_name, arguments, payload)
            for (tool_name, arguments), payload in zip(calls, payloads)
        ]

    @staticmethod
    def _format_result(
        tool_name: str,
        arguments: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        output = payload.get("output") or {}
        warnings = payload.get("warnings") or []
        errors = payload.get("errors") or []
        content_parts = [f"[{tool_name}]"]
        if payload.get("status") == "ok":
            content_parts.append(str(output.get("text") or "No LLM-readable tool text returned."))
            if warnings:
                content_parts.append("Warnings: " + "; ".join(str(item) for item in warnings))
        else:
            content_parts.append("Tool returned error: " + json.dumps(errors, ensure_ascii=False))
        return {
            "tool_name": tool_name,
            "status": payload.get("status", "error"),
            "arguments": arguments,
            "content": "\n".join(content_parts),
            "warnings": warnings,
            "errors": errors,
            "latency_ms": (payload.get("metadata") or {}).get("latency_ms"),
            "cache_hit": bool((payload.get("metadata") or {}).get("cache_hit")),
        }

