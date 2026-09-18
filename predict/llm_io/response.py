"""Validate model-response JSON and retry malformed structured outputs.

Harnesses provide messages plus required fields and semantic validators. This
module returns the model response with a bounded-attempt validation receipt; it
does not define task labels or decide a prediction.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import json
import re
from typing import Any


JsonCall = Callable[[list[dict[str, Any]]], dict[str, Any]]
ContentValidator = Callable[[Mapping[str, Any]], list[str]]
ENUM_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
SCHEMA_TYPE_TOKENS = frozenset(
    {"array", "boolean", "integer", "null", "number", "object", "string"}
)


def call_with_json_validation(
    call: JsonCall,
    messages: list[dict[str, Any]],
    *,
    required_fields: Iterable[str] = (),
    allowed_values: Mapping[str, set[str]] | None = None,
    forbidden_field_names: Iterable[str] = (),
    required_tool_names: Iterable[str] = (),
    content_validator: ContentValidator | None = None,
    branch_name: str = "reasoning",
    max_attempts: int = 4,
    private_reasoning_prefix: bool = False,
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
                    " Restart the required private-reasoning sequence from MOLECULE 1, complete GLOBAL "
                    "SYNTHESIS, and then emit one complete JSON object under about 1,200 words."
                    if private_reasoning_prefix
                    else " Start the response immediately with `{`. Do not emit repeated punctuation or extended "
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
            forbidden_field_names=forbidden_field_names,
            required_tool_names=required_tool_names,
            content_validator=content_validator,
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


def allowed_values_from_required_schema(
    messages: Iterable[Mapping[str, Any]],
) -> dict[str, set[str]]:
    """Extract top-level enum or fixed-literal contracts from a prompt schema."""
    for message in reversed(list(messages)):
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        schema = payload.get("required_json_schema")
        if not isinstance(schema, Mapping):
            continue
        allowed: dict[str, set[str]] = {}
        for field, description in schema.items():
            if not isinstance(description, str):
                continue
            values = {value.strip() for value in description.split("|")}
            if any(
                not value or not ENUM_TOKEN_PATTERN.fullmatch(value)
                for value in values
            ):
                continue
            if len(values) == 1 and next(iter(values)).lower() in SCHEMA_TYPE_TOKENS:
                continue
            allowed[str(field)] = values
        return allowed
    return {}


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
    forbidden_field_names: Iterable[str] = (),
    required_tool_names: Iterable[str] = (),
    content_validator: ContentValidator | None = None,
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
    forbidden = set(forbidden_field_names)
    for field in sorted(_nested_field_names(content) & forbidden):
        errors.append(f"forbidden_field:{field}")
    completed_tools = {
        str(result.get("tool_name") or "")
        for result in response.get("tool_results") or []
        if result.get("status") == "ok"
    }
    for tool_name in required_tool_names:
        if tool_name not in completed_tools:
            errors.append(f"missing_successful_tool:{tool_name}")
    if content_validator is not None:
        for error in content_validator(content):
            if error not in errors:
                errors.append(error)
    return errors


def _nested_field_names(value: Any) -> set[str]:
    """Collect mapping keys recursively so task contracts can forbid identity fields."""
    if isinstance(value, Mapping):
        names = {str(key) for key in value}
        for nested in value.values():
            names.update(_nested_field_names(nested))
        return names
    if isinstance(value, list):
        names: set[str] = set()
        for nested in value:
            names.update(_nested_field_names(nested))
        return names
    return set()
