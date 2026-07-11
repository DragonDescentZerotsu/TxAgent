"""Shared structured-output validation and retry for reasoning branches."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import json
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
    max_attempts: int = 4,
) -> dict[str, Any]:
    """Call until the JSON contract is valid or the bounded attempts are used."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    attempt_errors: list[list[str]] = []
    response: dict[str, Any] = {}
    for attempt_index in range(max_attempts):
        if attempt_index == 0:
            attempt_messages = messages
        else:
            previous_errors = attempt_errors[-1]
            recovery_instruction = ""
            base_messages = messages
            if attempt_index >= max_attempts - 2:
                final_recovery = attempt_index == max_attempts - 1
                base_messages = _compact_json_user_messages(
                    messages,
                    reverse_keys=final_recovery,
                    envelope=final_recovery,
                )
                recovery_instruction = (
                    " Start the response immediately with `{`. Do not emit repeated punctuation or extended "
                    "internal reasoning; keep the complete JSON under about 1,200 words and use short sentences "
                    "for every field."
                )
            attempt_messages = [
                *base_messages,
                {
                    "role": "user",
                    "content": (
                        f"Retry the {branch_name} response (attempt {attempt_index + 1} of {max_attempts}). "
                        "Return one complete JSON object matching the required schema. "
                        f"Validation errors: {', '.join(previous_errors)}."
                        f"{recovery_instruction}"
                    ),
                },
            ]
        response = call(attempt_messages)
        errors = response_validation_errors(
            response,
            required_fields=required_fields,
            allowed_values=allowed_values,
            required_tool_names=required_tool_names,
        )
        attempt_errors.append(errors)
        if not errors:
            break

    response["structured_output_validation"] = {
        "retried": len(attempt_errors) > 1,
        "first_attempt_errors": attempt_errors[0],
        "retry_errors": attempt_errors[-1] if len(attempt_errors) > 1 else [],
        "attempt_errors": attempt_errors,
        "attempt_count": len(attempt_errors),
        "valid": not attempt_errors[-1],
    }
    return response


def _compact_json_user_messages(
    messages: list[dict[str, Any]],
    *,
    reverse_keys: bool = False,
    envelope: bool = False,
) -> list[dict[str, Any]]:
    """Semantically reserialize JSON payloads to escape deterministic degeneration."""
    compacted = []
    for message in messages:
        updated = dict(message)
        content = updated.get("content")
        if updated.get("role") == "user" and isinstance(content, str):
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(payload, dict):
                    payload = {
                        **payload,
                        "_recovery_output_control": (
                            "Return the complete required JSON in under 1200 words with no repeated punctuation."
                        ),
                    }
                    payload = dict(sorted(payload.items(), reverse=reverse_keys))
                serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                updated["content"] = f"Input JSON:\n{serialized}" if envelope else serialized
        compacted.append(updated)
    return compacted


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
