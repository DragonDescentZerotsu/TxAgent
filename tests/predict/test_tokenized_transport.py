"""Validate shared tokenized-completion transport behavior."""
import json
from types import SimpleNamespace
import pytest
from predict.api_client.client import OpenAICompatibleClient
from predict.api_client.pool import ProviderPoolConfig, ProviderSpec, preflight_sglang_tokenized_completion
from predict.utils.response import call_with_json_validation
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"

class _FakeOpenAI:
    def __init__(self, text, *, model=MODEL):
        self.posts = []
        self.completion_calls = []
        self.text = text
        self.response_model = model
        self.completions = SimpleNamespace(create=self.create)

    def post(self, path, *, cast_to, body):
        self.posts.append((path, cast_to, body))
        if "messages" in body:
            return {"tokens": [10, 11, 12], "count": 3, "max_model_len": 100}
        return {"tokens": [20, 21], "count": 2, "max_model_len": 100}

    def create(self, **kwargs):
        self.completion_calls.append(kwargs)
        choice = SimpleNamespace(
            text=self.text,
            finish_reason="stop",
            model_extra={"token_ids": [1, 2, 3, 4, 5]},
        )
        return SimpleNamespace(
            choices=[choice], model=self.response_model, id="request-1", usage=None
        )


def _client(transport):
    return OpenAICompatibleClient(
        api_key="unused",
        base_url="http://example/v1",
        model=MODEL,
        timeout_s=30,
        max_tokens=50,
        temperature=0.0,
        tool_service_url="http://example-tools",
        enable_group_tools=False,
        max_tool_rounds=0,
        reasoning_effort="",
        enable_thinking=False,
        request_extra_body={
            "chat_template_kwargs": {"thinking": True, "reasoning_effort": "max"}
        },
        openai_client=transport,
    )


def test_tokenized_completion_splits_private_reasoning_and_applies_grammar():
    transport = _FakeOpenAI('analysis text</think>\n{"prediction":"positive"}')
    response = _client(transport).chat_json(
        [{"role": "user", "content": "question"}],
        tokenized_completion=True,
        reasoning_grammar="root ::= .*",
    )
    assert response["content"] == {"prediction": "positive"}
    assert response["reasoning_content"] == "analysis text"
    assert response["structured_reasoning"] == {
        "grammar_applied": True,
        "terminator_found": True,
        "reasoning_characters": 13,
        "reasoning_tokens": 2,
        "completion_tokens_from_ids": 5,
        "finish_reason": "stop",
    }
    request = transport.completion_calls[0]
    assert request["prompt"] == [10, 11, 12]
    assert request["extra_body"] == {
        "skip_special_tokens": False,
        "return_token_ids": True,
        "ebnf": "root ::= .*",
    }


def test_tokenized_completion_fails_validation_when_reasoning_never_closes():
    response = _client(_FakeOpenAI("unfinished reasoning")).chat_json(
        [{"role": "user", "content": "question"}], tokenized_completion=True
    )
    assert response["structured_reasoning"]["terminator_found"] is False
    assert response["content"]["parse_error"]


def test_tokenized_completion_rejects_a_different_served_model():
    with pytest.raises(ValueError, match="served model"):
        _client(_FakeOpenAI("</think>\n{}", model="other-model")).chat_json(
            [{"role": "user", "content": "question"}], tokenized_completion=True
        )


def test_private_reasoning_retry_does_not_demand_json_as_the_first_token():
    calls = []

    def invalid(messages):
        calls.append(messages)
        return {"content": {}}

    call_with_json_validation(
        invalid,
        [{"role": "user", "content": "question"}],
        required_fields=("prediction",),
        max_attempts=2,
        private_reasoning_prefix=True,
    )
    recovery = calls[1][-1]["content"]
    assert "MOLECULE 1" in recovery
    assert "immediately with `{`" not in recovery


def test_sglang_preflight_checks_backend_and_thinking_suffix(monkeypatch):
    payloads = iter(
        [
            {
                "status": "ready",
                "served_model_name": MODEL,
                "grammar_backend": "xgrammar",
                "reasoning_parser": "deepseek-v4",
                "completion_template": None,
            },
            {"tokens": [1, 2, 3]},
            {"text": "prompt tail<think>"},
        ]
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self):
            return json.dumps(next(payloads)).encode()

    monkeypatch.setattr("predict.api_client.pool.urlopen", lambda *_, **__: Response())
    config = ProviderPoolConfig(
        providers=(
            ProviderSpec(
                name="local",
                base_url="http://local/v1",
                model=MODEL,
                api_key_env="",
                max_inflight=1,
                request_extra_body={"chat_template_kwargs": {"thinking": True}},
            ),
        )
    )
    receipt = preflight_sglang_tokenized_completion(config)
    assert receipt[0]["tokenized_prompt_suffix"] == "<think>"
