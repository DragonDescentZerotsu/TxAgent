import json
from copy import deepcopy

import pytest

from predict.utils.response import allowed_values_from_required_schema, call_with_json_validation, response_validation_errors, structured_response_is_valid, validated_branch_content, validate_query_prior_output


def test_allowed_values_are_extracted_from_prompt_schema_for_enums_and_literals():
    messages = [
        {
            "role": "user",
            "content": (
                '{"required_json_schema": {'
                '"confidence": "high | moderate | low", '
                '"reasoning_summary": "string", '
                '"label_scope": "skin_sensitization_contact_allergy.v2", '
                '"similarity": "number or null", '
                '"complex_description": "high | moderate confidence"}}'
            ),
        }
    ]

    assert allowed_values_from_required_schema(messages) == {
        "confidence": {"high", "moderate", "low"},
        "label_scope": {"skin_sensitization_contact_allergy.v2"},
    }


def test_schema_derived_allowed_values_trigger_retry():
    messages = [
        {
            "role": "user",
            "content": '{"required_json_schema":{"confidence":"high | moderate | low"}}',
        }
    ]
    responses = iter(
        [
            {"content": {"confidence": "medium"}},
            {"content": {"confidence": "moderate"}},
        ]
    )

    result = call_with_json_validation(
        lambda _: next(responses),
        messages,
        required_fields=("confidence",),
        allowed_values=allowed_values_from_required_schema(messages),
    )

    assert result["content"]["confidence"] == "moderate"
    assert result["structured_output_validation"]["attempt_errors"] == [
        ["invalid_value:confidence"],
        [],
    ]


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


def test_content_validator_can_trigger_a_structured_retry():
    responses = iter(
        [
            {"content": {"prediction": "positive", "prior_used": "true"}},
            {"content": {"prediction": "positive", "prior_used": True}},
        ]
    )

    result = call_with_json_validation(
        lambda _: next(responses),
        [{"role": "user", "content": "classify"}],
        required_fields=("prediction", "prior_used"),
        content_validator=lambda content: (
            []
            if isinstance(content.get("prior_used"), bool)
            else ["invalid_type:prior_used:expected_boolean"]
        ),
    )

    assert result["content"]["prior_used"] is True
    assert result["structured_output_validation"]["attempt_errors"] == [
        ["invalid_type:prior_used:expected_boolean"],
        [],
    ]


def test_invalid_branch_content_is_withheld_from_final_synthesis():
    branch = {
        "status": "error",
        "llm": {
            "content": {"unparsed_text": "partial"},
            "structured_output_validation": {"valid": False},
        },
    }

    assert validated_branch_content(branch) is None


@pytest.fixture
def historical_prior():
    schema = dict(endpoint_prior="positive | negative | mixed_or_unclear",
                  confidence="high | moderate | low", reasoning_summary="string",
                  property_drivers=["string"], caveats=["string"])
    return dict(status="ok", llm=dict(content=dict(
        endpoint_prior="positive", confidence="moderate", reasoning_summary="Structural prior",
        property_drivers=["Small molecule"], caveats=["No measured outcome"]),
        messages=[dict(role="user", content=json.dumps(dict(required_json_schema=schema)))],
        structured_output_validation=dict(valid=True)))


def test_saved_prior_receipt_is_reused_without_schema_validation(historical_prior, monkeypatch):
    from predict.utils import response

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved prior must not be revalidated")

    monkeypatch.setattr(response, "response_validation_errors", forbidden)
    monkeypatch.setattr(response, "allowed_values_from_required_schema", forbidden)
    before = deepcopy(historical_prior)
    assert validate_query_prior_output(historical_prior) == before
    assert historical_prior == before


@pytest.mark.parametrize("invalid", ("non_ok", "no_response", "no_receipt"))
def test_saved_prior_requires_a_successful_generation_receipt(historical_prior, invalid):
    if invalid == "non_ok":
        historical_prior["status"] = "error"
    elif invalid == "no_response":
        historical_prior.pop("llm")
    else:
        historical_prior["llm"].pop("structured_output_validation")
    with pytest.raises(ValueError, match="Query prior"):
        validate_query_prior_output(historical_prior)


@pytest.mark.parametrize("receipt", (None, {}, [], {"valid": False}, {"valid": "true"}))
def test_explicit_invalid_prior_receipt_is_not_repaired(historical_prior, receipt):
    historical_prior["llm"]["structured_output_validation"] = receipt
    with pytest.raises(ValueError, match="failed or malformed"):
        validate_query_prior_output(historical_prior)


def test_historical_loader_and_resume_comparison_preserve_saved_bytes(historical_prior, tmp_path):
    from predict.harnesses.branches.inference import load_frozen_single_analysis
    from predict.harnesses.branches.runtime import require_matching_query_prior

    saved = tmp_path / "single_molecule_reasoning_output.json"
    requested = tmp_path / "requested.json"
    saved.write_text(json.dumps(historical_prior))
    requested.write_text(json.dumps(validate_query_prior_output(historical_prior)))
    before = saved.read_bytes()
    loaded = load_frozen_single_analysis(str(tmp_path))
    assert validated_branch_content(loaded) == historical_prior["llm"]["content"]
    assert require_matching_query_prior(saved, requested, "query") == historical_prior["llm"]["content"]
    historical_prior["llm"]["content"]["reasoning_summary"] = "Changed"
    requested.write_text(json.dumps(historical_prior))
    with pytest.raises(ValueError, match="Query prior changed"):
        require_matching_query_prior(saved, requested, "query")
    assert saved.read_bytes() == before


@pytest.mark.parametrize("invalid", ("non_ok", "failed_receipt", "missing_receipt"))
def test_resume_never_replaces_an_invalid_saved_prior(historical_prior, tmp_path, invalid):
    from types import SimpleNamespace
    from predict.harnesses.branches.runtime import synchronize_configured_single_reuse

    source = tmp_path / "source" / "runs" / "source_idx00000"
    source.mkdir(parents=True)
    name = "single_molecule_reasoning_output.json"
    (source / name).write_text(json.dumps(historical_prior))
    saved = deepcopy(historical_prior)
    if invalid == "non_ok":
        saved["status"] = "error"
    elif invalid == "failed_receipt":
        saved["llm"]["structured_output_validation"] = {"valid": False}
    else:
        saved["llm"].pop("structured_output_validation")
    target = tmp_path / "target"
    target.mkdir()
    (target / name).write_text(json.dumps(saved))
    (target / "final_reasoning_output.json").write_text('{"status":"ok"}')
    before = {f: f.read_bytes() for f in target.iterdir()}
    prepared = SimpleNamespace(args=SimpleNamespace(single_analysis_source_batch=str(source.parent.parent)))
    with pytest.raises(ValueError, match="Cannot validate query prior"):
        synchronize_configured_single_reuse(prepared, SimpleNamespace(index=0, record={}), target)
    assert {f: f.read_bytes() for f in target.iterdir()} == before
