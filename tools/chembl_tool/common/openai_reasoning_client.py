"""Provider-neutral OpenAI-compatible client for structured molecular reasoning."""

from __future__ import annotations

import json
from typing import Any, Mapping

import httpx
from openai import AsyncOpenAI, OpenAI
import requests

from tools.chembl_tool.common.json_utils import parse_json_content


TRANSPORT_MAX_RETRIES = 2


class OpenAICompatibleClient:
    """Run JSON-only completions and bounded tool loops against one endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_s: int,
        max_tokens: int,
        temperature: float | None,
        tool_service_url: str,
        enable_group_tools: bool,
        max_tool_rounds: int,
        reasoning_effort: str,
        enable_thinking: bool,
        transport_max_retries: int = TRANSPORT_MAX_RETRIES,
        request_extra_body: Mapping[str, Any] | None = None,
        async_max_connections: int | None = None,
    ):
        if transport_max_retries < 0:
            raise ValueError("transport_max_retries must be non-negative")
        if async_max_connections is not None and async_max_connections < 1:
            raise ValueError("async_max_connections must be positive")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            max_retries=transport_max_retries,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tool_service = ToolServiceClient(tool_service_url, timeout_s=timeout_s)
        self.enable_group_tools = enable_group_tools
        self.max_tool_rounds = max_tool_rounds
        self.reasoning_effort = reasoning_effort
        self.enable_thinking = enable_thinking
        self.request_extra_body = dict(request_extra_body or {})
        self._async_client = None
        self.async_max_connections = async_max_connections

    def chat_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        response = self._create_completion(messages)
        return self._json_response(response, messages)

    async def async_chat_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Cancellation closes the in-flight HTTP request; no worker thread is left behind."""
        if self._async_client is None:
            http_options = {}
            if self.async_max_connections is not None:
                http_options["http_client"] = httpx.AsyncClient(limits=httpx.Limits(
                    max_connections=self.async_max_connections,
                    max_keepalive_connections=self.async_max_connections,
                ))
            self._async_client = AsyncOpenAI(
                api_key=self.client.api_key, base_url=self.client.base_url,
                timeout=self.client.timeout, max_retries=self.client.max_retries,
                **http_options,
            )
        response = await self._async_client.chat.completions.create(
            **self._completion_kwargs(messages)
        )
        return self._json_response(response, messages)

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.close()
        self.client.close()

    def close(self) -> None:
        self.client.close()

    def _json_response(self, response: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
        message = response.choices[0].message
        content = message.content or "{}"
        trace_messages = [_json_safe_message(item) for item in messages]
        trace_messages.append(_assistant_message_to_trace(message))
        return {
            "content": parse_json_content(content),
            "raw_content": content,
            "reasoning_content": _reasoning_text(message),
            "tool_calls": [],
            "tool_results": [],
            "messages": trace_messages,
            "usage": _usage_dict(response),
            "model": response.model or self.model,
            "id": response.id or "",
        }

    def chat_json_with_optional_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        allowed_tool_names: set[str],
    ) -> dict[str, Any]:
        if not self.enable_group_tools:
            return self.chat_json(messages)
        return self.chat_json_with_tools(
            messages,
            tools=tools,
            allowed_tool_names=allowed_tool_names,
        )

    def chat_json_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        allowed_tool_names: set[str],
        first_tool_choice: Any = "auto",
    ) -> dict[str, Any]:
        working_messages: list[Any] = list(messages)
        trace_messages = [_json_safe_message(item) for item in messages]
        tool_results: list[dict[str, Any]] = []
        responses = []
        for round_index in range(max(0, self.max_tool_rounds) + 1):
            current_tool_choice = first_tool_choice if round_index == 0 else "auto"
            if self.enable_thinking and current_tool_choice not in (None, "auto"):
                current_tool_choice = "auto"
            response = self._create_completion(
                working_messages,
                tools=tools,
                tool_choice=current_tool_choice,
            )
            responses.append(response)
            message = response.choices[0].message
            working_messages.append(message)
            trace_messages.append(_assistant_message_to_trace(message))
            tool_calls = message.tool_calls or []
            if not tool_calls:
                content = message.content or "{}"
                return {
                    "content": parse_json_content(content),
                    "raw_content": content,
                    "reasoning_content": _reasoning_text(message),
                    "tool_calls": _tool_call_summaries(responses),
                    "tool_results": tool_results,
                    "messages": trace_messages,
                    "usage": _sum_usage(responses),
                    "model": response.model or self.model,
                    "id": response.id or "",
                }
            for tool_call in tool_calls:
                result = self.tool_service.invoke_function_call(
                    tool_call,
                    allowed_tool_names=allowed_tool_names,
                )
                tool_results.append(result)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result["content"],
                }
                working_messages.append(tool_message)
                trace_messages.append({**tool_message, "name": result.get("tool_name"), "tool_result": result})

        final_request = {
            "role": "user",
            "content": (
                "You have reached the maximum allowed tool-call rounds. "
                "Return the required JSON now using the available tool results."
            ),
        }
        working_messages.append(final_request)
        trace_messages.append(final_request)
        response = self._create_completion(working_messages)
        responses.append(response)
        message = response.choices[0].message
        content = message.content or "{}"
        trace_messages.append(_assistant_message_to_trace(message))
        return {
            "content": parse_json_content(content),
            "raw_content": content,
            "reasoning_content": _reasoning_text(message),
            "tool_calls": _tool_call_summaries(responses),
            "tool_results": tool_results,
            "messages": trace_messages,
            "usage": _sum_usage(responses),
            "model": response.model or self.model,
            "id": response.id or "",
        }

    def _completion_kwargs(
        self,
        messages: list[Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        extra_body = dict(getattr(self, "request_extra_body", {}) or {})
        if self.enable_thinking:
            extra_body["thinking"] = {"type": "enabled"}
        if extra_body:
            kwargs["extra_body"] = extra_body
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return kwargs

    def _create_completion(
        self, messages: list[Any], *, tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> Any:
        kwargs = self._completion_kwargs(messages, tools=tools, tool_choice=tool_choice)
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if self.enable_thinking and tool_choice is not None and _is_tool_choice_thinking_error(exc):
                fallback_kwargs = dict(kwargs)
                fallback_body = dict(fallback_kwargs.pop("extra_body", {}))
                fallback_body.pop("thinking", None)
                if fallback_body:
                    fallback_kwargs["extra_body"] = fallback_body
                fallback_kwargs.pop("reasoning_effort", None)
                return self.client.chat.completions.create(**fallback_kwargs)
            raise


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
        """Invoke a known tool directly for harness-side prefetching."""

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
        """Invoke a prefetched tool bundle in one bounded service request."""
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
            # Supports rolling upgrades while an older service is still bound.
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


def _is_tool_choice_thinking_error(exc: Exception) -> bool:
    return "Thinking mode does not support this tool_choice" in str(exc)


def _usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return usage.model_dump(mode="json") if hasattr(usage, "model_dump") else dict(usage)


def _sum_usage(responses: list[Any]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for response in responses:
        for key, value in _usage_dict(response).items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def _tool_call_summaries(responses: list[Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for response in responses:
        message = response.choices[0].message
        for tool_call in message.tool_calls or []:
            summaries.append(
                {
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                }
            )
    return summaries


def _assistant_message_to_trace(message: Any) -> dict[str, Any]:
    trace: dict[str, Any] = {"role": "assistant"}
    content = getattr(message, "content", None)
    if content:
        trace["content"] = content
    reasoning = _reasoning_text(message)
    if reasoning:
        trace["reasoning"] = reasoning
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        trace["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in tool_calls
        ]
    return trace


def _reasoning_text(message: Any) -> str:
    """Normalize reasoning fields used by OpenAI-compatible providers."""
    for field in ("reasoning_content", "reasoning"):
        value = getattr(message, field, None)
        if value:
            return str(value)
    model_extra = getattr(message, "model_extra", None) or {}
    if isinstance(model_extra, dict):
        value = model_extra.get("reasoning_content") or model_extra.get("reasoning")
        if value:
            return str(value)
    return ""


def _json_safe_message(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return json.loads(json.dumps(message, ensure_ascii=False, default=str))
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json")
    return {"role": "unknown", "content": str(message)}
