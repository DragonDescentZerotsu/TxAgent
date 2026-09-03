"""Provider-neutral OpenAI-compatible client for structured molecular reasoning."""

from __future__ import annotations

import json
from typing import Any, Mapping

from openai import OpenAI

from predict.tools.client import ToolServiceClient
from predict.utils.json import parse_json_content


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
        response_format: Mapping[str, Any] | None = None,
        openai_client: Any | None = None,
    ):
        if transport_max_retries < 0:
            raise ValueError("transport_max_retries must be non-negative")
        self.client = openai_client or OpenAI(
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
        self.response_format = dict(response_format or {"type": "json_object"})

    def chat_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        response = self._create_completion(messages)
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

    def _create_completion(
        self,
        messages: list[Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": getattr(
                self, "response_format", {"type": "json_object"}
            ),
        }
        provider_model = self.model.rsplit("/", 1)[-1]
        token_parameter = (
            "max_completion_tokens"
            if provider_model.startswith("gpt-5")
            else "max_tokens"
        )
        kwargs[token_parameter] = self.max_tokens
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
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if self.enable_thinking and tool_choice is not None and _is_tool_choice_thinking_error(exc):
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop("extra_body", None)
                fallback_kwargs.pop("reasoning_effort", None)
                return self.client.chat.completions.create(**fallback_kwargs)
            raise


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
