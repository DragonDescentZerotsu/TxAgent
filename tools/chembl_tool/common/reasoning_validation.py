"""Shared structured-output validation and retry for reasoning branches."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any


JsonCall = Callable[[list[dict[str, Any]]], dict[str, Any]]


def call_with_json_validation(
    call: JsonCall,
    messages: list[dict[str, Any]],
    *,
    required_fields: Iterable[str] = (),
    allowed_values: Mapping[str, set[str]] | None = None,
    required_tool_names: Iterable[str] = (),
    branch_name: str = "reasoning",
) -> dict[str, Any]:
    """Call once, then retry once when the returned JSON contract is invalid."""
    response = call(messages)
    errors = response_validation_errors(
        response,
        required_fields=required_fields,
        allowed_values=allowed_values,
        required_tool_names=required_tool_names,
    )
    if not errors:
        response["structured_output_validation"] = {
            "retried": False,
            "first_attempt_errors": [],
            "retry_errors": [],
            "valid": True,
        }
        return response

    retry_messages = [
        *messages,
        {
            "role": "user",
            "content": (
                f"Retry the {branch_name} response. Return one complete JSON object matching the required schema. "
                f"Validation errors: {', '.join(errors)}."
            ),
        },
    ]
    retry = call(retry_messages)
    retry_errors = response_validation_errors(
        retry,
        required_fields=required_fields,
        allowed_values=allowed_values,
        required_tool_names=required_tool_names,
    )
    retry["structured_output_validation"] = {
        "retried": True,
        "first_attempt_errors": errors,
        "retry_errors": retry_errors,
        "valid": not retry_errors,
    }
    return retry


def structured_response_is_valid(response: Mapping[str, Any]) -> bool:
    validation = response.get("structured_output_validation")
    return isinstance(validation, Mapping) and validation.get("valid") is True


def validated_branch_content(branch_output: Mapping[str, Any]) -> Any:
    """Return branch content only when both branch and response validation pass."""
    if branch_output.get("status") != "ok":
        return None
    response = branch_output.get("llm")
    if not isinstance(response, Mapping) or not structured_response_is_valid(response):
        return None
    return response.get("content")


def response_validation_errors(
    response: Mapping[str, Any],
    *,
    required_fields: Iterable[str] = (),
    allowed_values: Mapping[str, set[str]] | None = None,
    required_tool_names: Iterable[str] = (),
) -> list[str]:
    errors: list[str] = []
    content = response.get("content")
    if not isinstance(content, Mapping) or not content:
        return ["empty_or_non_object_json"]
    for field in required_fields:
        if field not in content or content.get(field) in (None, ""):
            errors.append(f"missing_field:{field}")
    for field, values in (allowed_values or {}).items():
        value = str(content.get(field) or "")
        if value not in values:
            errors.append(f"invalid_value:{field}")
    completed_tools = {
        str(result.get("tool_name") or "")
        for result in response.get("tool_results") or []
        if result.get("status") == "ok"
    }
    for tool_name in required_tool_names:
        if tool_name not in completed_tools:
            errors.append(f"missing_successful_tool:{tool_name}")
    return errors
