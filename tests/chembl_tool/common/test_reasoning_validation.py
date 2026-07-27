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
        "attempt_errors": [[]],
        "attempt_count": 1,
        "valid": True,
    }
    assert structured_response_is_valid(result)


def test_validation_can_recover_on_third_attempt():
    responses = iter(
        [
            {"content": {}},
            {"content": {"prediction": ""}},
            {"content": {"prediction": "positive"}},
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
        max_attempts=3,
    )

    validation = result["structured_output_validation"]
    assert validation["valid"] is True
    assert validation["attempt_count"] == 3
    assert validation["attempt_errors"] == [
        ["empty_or_non_object_json"],
        ["missing_field:prediction"],
        [],
    ]
    assert "Start the response immediately with `{`" in seen_messages[2][-1]["content"]


def test_final_recovery_compacts_json_user_payload_without_changing_data():
    responses = iter(
        [
            {"content": {}},
            {"content": {}},
            {"content": {}},
            {"content": {"prediction": "positive"}},
        ]
    )
    seen_messages = []

    def call(messages):
        seen_messages.append(messages)
        return next(responses)

    call_with_json_validation(
        call,
        [{"role": "user", "content": '{"z": 1, "a": [2, 3]}'}],
        required_fields=("prediction",),
    )

    assert seen_messages[2][0]["content"] == (
        '{"_recovery_output_control":"Return the complete required JSON in under 1200 words with no repeated '
        'punctuation.","a":[2,3],"z":1}'
    )
    assert seen_messages[3][0]["content"] == (
        'Input JSON:\n{"z":1,"a":[2,3],"_recovery_output_control":"Return the complete required JSON in under '
        '1200 words with no repeated punctuation."}'
    )


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


def test_validation_rejects_forbidden_field_name_at_any_depth():
    response = {
        "content": {
            "confidence": "high",
            "key_evidence": [{"molecule_chembl_id": "CHEMBL1"}],
        }
    }

    errors = response_validation_errors(
        response,
        required_fields=("confidence",),
        forbidden_field_names=("molecule_chembl_id",),
    )

    assert errors == ["forbidden_field:molecule_chembl_id"]


def test_invalid_branch_content_is_withheld_from_final_synthesis():
    branch = {
        "status": "error",
        "llm": {
            "content": {"unparsed_text": "partial"},
            "structured_output_validation": {"valid": False},
        },
    }

    assert validated_branch_content(branch) is None
