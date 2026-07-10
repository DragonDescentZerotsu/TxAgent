from tools.chembl_tool.common.reasoning_validation import (
    call_with_json_validation,
    response_validation_errors,
    structured_response_is_valid,
    validated_branch_content,
)


def test_validation_retries_missing_field_and_accepts_second_response():
    responses = iter(
        [
            {"content": {}, "tool_results": []},
            {"content": {"prediction": "positive"}, "tool_results": []},
        ]
    )
    seen_messages = []

    def call(messages):
        seen_messages.append(messages)
        return next(responses)

    result = call_with_json_validation(
        call,
        [{"role": "user", "content": "classify"}],
        required_fields=("prediction",),
        allowed_values={"prediction": {"positive", "negative"}},
        branch_name="final",
    )

    assert result["content"]["prediction"] == "positive"
    assert result["structured_output_validation"]["valid"] is True
    assert len(seen_messages) == 2
    assert "Validation errors" in seen_messages[1][-1]["content"]
    assert structured_response_is_valid(result)


def test_valid_first_response_gets_explicit_validation_metadata():
    result = call_with_json_validation(
        lambda messages: {"content": {"prediction": "negative"}},
        [{"role": "user", "content": "classify"}],
        required_fields=("prediction",),
        allowed_values={"prediction": {"positive", "negative"}},
    )

    assert result["structured_output_validation"] == {
        "retried": False,
        "first_attempt_errors": [],
        "retry_errors": [],
        "valid": True,
    }
    assert structured_response_is_valid(result)


def test_validation_checks_required_successful_tool():
    response = {
        "content": {"confidence": "low"},
        "tool_results": [{"tool_name": "molecule_properties", "status": "error"}],
    }

    errors = response_validation_errors(
        response,
        required_fields=("confidence",),
        required_tool_names=("molecule_properties",),
    )

    assert errors == ["missing_successful_tool:molecule_properties"]


def test_invalid_branch_content_is_withheld_from_final_synthesis():
    branch = {
        "status": "error",
        "llm": {
            "content": {"unparsed_text": "partial"},
            "structured_output_validation": {"valid": False},
        },
    }

    assert validated_branch_content(branch) is None
