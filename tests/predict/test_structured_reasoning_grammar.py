"""Validate progressive private-reasoning grammar and transport behavior."""

import json
from types import SimpleNamespace

import pytest
import xgrammar as xgr

from predict.api_client.client import OpenAICompatibleClient
from predict.api_client.pool import (
    ProviderPoolConfig,
    ProviderSpec,
    preflight_sglang_tokenized_completion,
)
from predict.harnesses.progressive.grammar import render_reasoning_grammar
from predict.harnesses.progressive.prompt import (
    prompt_asset_manifest,
    render_progressive_messages,
)
from predict.llm_io.response import call_with_json_validation


MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
V14 = "reranked_progressive_l1_simple_v14"
V15 = "reranked_progressive_l1_simple_v15"
V22 = "reranked_progressive_l1_simple_v22"
V23 = "reranked_progressive_l1_simple_v23"
V25 = "reranked_progressive_l1_simple_v25"
V26 = "reranked_progressive_l1_simple_v26"
V27 = "reranked_progressive_l1_simple_v27"
V28 = "reranked_progressive_l1_simple_v28"
V29 = "reranked_progressive_l1_simple_v29"
V30 = "reranked_progressive_l1_simple_v30"
V31 = "reranked_progressive_l1_simple_v31"
V32 = "reranked_progressive_l1_simple_v32"
V33 = "reranked_progressive_l1_simple_v33"
V34 = "reranked_progressive_l1_simple_v34"
V35 = "reranked_progressive_l1_simple_v35"
V36 = "reranked_progressive_l1_simple_v36"
V37 = "reranked_progressive_l1_simple_v37"
V38 = "reranked_progressive_l1_simple_v38"
V39 = "reranked_progressive_l1_simple_v39"
V40 = "reranked_progressive_l1_simple_v40"
V41 = "reranked_progressive_l1_simple_v41"
V42 = "reranked_progressive_l1_simple_v42"
V43 = "reranked_progressive_l1_simple_v43"
V44 = "reranked_progressive_l1_simple_v44"
V45 = "reranked_progressive_l1_simple_v45"
V46 = "reranked_progressive_l1_simple_v46"


def _active_evidence():
    return {
        "later": {
            "_selection_rank": 2,
            "cards": {"b": {"card_id": "b", "first_seen_level": 1}},
        },
        "first": {
            "_selection_rank": 1,
            "cards": {
                "z": {"card_id": "z", "first_seen_level": 1},
                "a": {"card_id": "a", "first_seen_level": 1},
            },
        },
    }


def _message_payload():
    return {
        "protocol": {"prompt_mode": "morgan", "version": "test"},
        "task_definition": {
            "task": "bbb_martins",
            "endpoint": "endpoint",
            "label_scope": "scope",
            "prediction_values": {"positive": "label 1", "negative": "label 0"},
            "instructions": ["Use the evidence."],
        },
        "query": {"canonical_smiles": "CCN"},
        "active_evidence": [
            {
                "canonical_smiles": "CCO",
                "evidence_cards": [{"card_id": "C01", "endpoint": "measurement"}],
            }
        ],
        "required_json_schema": {"prediction": "positive | negative"},
    }


def test_v15_compiles_query_specific_molecule_grammar_without_changing_messages():
    aliases = {"a": "C01", "b": "C02", "z": "C03"}
    assert render_reasoning_grammar(V14, _active_evidence(), card_id_to_alias=aliases) is None
    grammar = render_reasoning_grammar(V15, _active_evidence(), card_id_to_alias=aliases)
    xgr.Grammar.from_ebnf(grammar)
    assert grammar.index("RECORDS C01,C03") < grammar.index("RECORDS C02")
    assert "reasoning.gbnf.jinja" in prompt_asset_manifest(V15)["files_sha256"]
    kwargs = {"system_role": "Reason carefully.", "payload": _message_payload()}
    assert render_progressive_messages(**kwargs, prompt_version=V14) == (
        render_progressive_messages(**kwargs, prompt_version=V15)
    )


def test_task_adaptive_reasoning_grammar_renders_valid_distinct_grammars():
    aliases = {"a": "C01", "b": "C02", "z": "C03"}
    grammars = [
        render_reasoning_grammar(
            V25,
            _active_evidence(),
            card_id_to_alias=aliases,
            task=task,
        )
        for task in ("bbb_martins", "bioavailability_ma")
    ]
    assert grammars[0] != grammars[1]
    for grammar in grammars:
        xgr.Grammar.from_ebnf(grammar)


def test_oral_calibration_grammar_does_not_change_bbb_branch():
    aliases = {"a": "C01", "b": "C02", "z": "C03"}
    rendered = {
        version: {
            task: render_reasoning_grammar(
                version,
                _active_evidence(),
                card_id_to_alias=aliases,
                task=task,
            )
            for task in ("bbb_martins", "bioavailability_ma")
        }
        for version in (V25, V26)
    }
    assert rendered[V26]["bbb_martins"] == rendered[V25]["bbb_martins"]
    assert rendered[V26]["bioavailability_ma"] != rendered[V25]["bioavailability_ma"]
    for grammar in rendered[V26].values():
        xgr.Grammar.from_ebnf(grammar)


def test_v27_joins_previously_rendered_task_branches():
    aliases = {"a": "C01", "b": "C02", "z": "C03"}

    def render(version, task):
        return render_reasoning_grammar(
            version,
            _active_evidence(),
            card_id_to_alias=aliases,
            task=task,
        )

    bbb = render(V27, "bbb_martins")
    oral = render(V27, "bioavailability_ma")
    assert bbb == render(V25, "bbb_martins")
    assert oral == render(V23, "bioavailability_ma")
    xgr.Grammar.from_ebnf(bbb)
    xgr.Grammar.from_ebnf(oral)


def test_v28_joins_v22_bbb_and_v23_oral_branches():
    aliases = {"a": "C01", "b": "C02", "z": "C03"}

    def render(version, task):
        return render_reasoning_grammar(
            version,
            _active_evidence(),
            card_id_to_alias=aliases,
            task=task,
        )

    bbb = render(V28, "bbb_martins")
    oral = render(V28, "bioavailability_ma")
    assert bbb == render(V22, "bbb_martins")
    assert oral == render(V23, "bioavailability_ma")
    xgr.Grammar.from_ebnf(bbb)
    xgr.Grammar.from_ebnf(oral)


def test_v29_changes_only_the_bbb_calibration_branch():
    aliases = {"a": "C01", "b": "C02", "z": "C03"}

    def render(version, task):
        return render_reasoning_grammar(
            version,
            _active_evidence(),
            card_id_to_alias=aliases,
            task=task,
        )

    bbb = render(V29, "bbb_martins")
    oral = render(V29, "bioavailability_ma")
    assert bbb != render(V28, "bbb_martins")
    assert oral == render(V28, "bioavailability_ma")
    xgr.Grammar.from_ebnf(bbb)
    xgr.Grammar.from_ebnf(oral)


def test_v30_changes_only_the_oral_calibration_branch():
    aliases = {"a": "C01", "b": "C02", "z": "C03"}

    def render(version, task):
        return render_reasoning_grammar(
            version,
            _active_evidence(),
            card_id_to_alias=aliases,
            task=task,
        )

    bbb = render(V30, "bbb_martins")
    oral = render(V30, "bioavailability_ma")
    assert bbb == render(V29, "bbb_martins")
    assert oral != render(V29, "bioavailability_ma")
    xgr.Grammar.from_ebnf(bbb)
    xgr.Grammar.from_ebnf(oral)


def test_v31_keeps_bbb_and_revises_oral_calibration():
    aliases = {"a": "C01", "b": "C02", "z": "C03"}

    def render(version, task):
        return render_reasoning_grammar(
            version,
            _active_evidence(),
            card_id_to_alias=aliases,
            task=task,
        )

    bbb = render(V31, "bbb_martins")
    oral = render(V31, "bioavailability_ma")
    assert bbb == render(V30, "bbb_martins")
    assert oral != render(V30, "bioavailability_ma")
    xgr.Grammar.from_ebnf(bbb)
    xgr.Grammar.from_ebnf(oral)


def test_v32_renders_task_specific_majority_grammars():
    aliases = {"a": "C01", "b": "C02", "z": "C03"}
    grammars = [
        render_reasoning_grammar(
            V32,
            _active_evidence(),
            card_id_to_alias=aliases,
            task=task,
        )
        for task in ("bbb_martins", "bioavailability_ma")
    ]
    assert grammars[0] != grammars[1]
    for grammar in grammars:
        xgr.Grammar.from_ebnf(grammar)


def test_v33_keeps_majority_bbb_and_restores_free_oral_branch():
    aliases = {"a": "C01", "b": "C02", "z": "C03"}

    def render(version, task):
        return render_reasoning_grammar(
            version,
            _active_evidence(),
            card_id_to_alias=aliases,
            task=task,
        )

    bbb = render(V33, "bbb_martins")
    oral = render(V33, "bioavailability_ma")
    assert bbb == render(V32, "bbb_martins")
    assert oral == render(V23, "bioavailability_ma")
    xgr.Grammar.from_ebnf(bbb)
    xgr.Grammar.from_ebnf(oral)


def test_v34_changes_only_the_bbb_reasoning_allowance():
    aliases = {"a": "C01", "b": "C02", "z": "C03"}

    def render(version, task):
        return render_reasoning_grammar(
            version,
            _active_evidence(),
            card_id_to_alias=aliases,
            task=task,
        )

    bbb = render(V34, "bbb_martins")
    oral = render(V34, "bioavailability_ma")
    assert bbb != render(V33, "bbb_martins")
    assert oral == render(V33, "bioavailability_ma")
    xgr.Grammar.from_ebnf(bbb)
    xgr.Grammar.from_ebnf(oral)


def test_v35_uses_established_bbb_and_new_oral_allowances():
    aliases = {"a": "C01", "b": "C02", "z": "C03"}

    def render(version, task):
        return render_reasoning_grammar(
            version,
            _active_evidence(),
            card_id_to_alias=aliases,
            task=task,
        )

    bbb = render(V35, "bbb_martins")
    oral = render(V35, "bioavailability_ma")
    assert bbb == render(V27, "bbb_martins")
    assert oral != render(V34, "bioavailability_ma")
    xgr.Grammar.from_ebnf(bbb)
    xgr.Grammar.from_ebnf(oral)


def test_v36_joins_majority_bbb_and_free_24k_oral_branches():
    aliases = {"a": "C01", "b": "C02", "z": "C03"}

    def render(version, task):
        return render_reasoning_grammar(
            version,
            _active_evidence(),
            card_id_to_alias=aliases,
            task=task,
        )

    bbb = render(V36, "bbb_martins")
    oral = render(V36, "bioavailability_ma")
    assert bbb == render(V33, "bbb_martins")
    assert oral != render(V35, "bioavailability_ma")
    xgr.Grammar.from_ebnf(bbb)
    xgr.Grammar.from_ebnf(oral)


def test_v37_joins_free_17k_bbb_and_free_24k_oral_branches():
    aliases = {"a": "C01", "b": "C02", "z": "C03"}

    def render(version, task):
        return render_reasoning_grammar(
            version,
            _active_evidence(),
            card_id_to_alias=aliases,
            task=task,
        )

    bbb = render(V37, "bbb_martins")
    oral = render(V37, "bioavailability_ma")
    assert bbb == render(V34, "bbb_martins")
    assert oral == render(V36, "bioavailability_ma")
    xgr.Grammar.from_ebnf(bbb)
    xgr.Grammar.from_ebnf(oral)


def test_v38_keeps_bbb_and_requires_separate_oral_audits():
    aliases = {"a": "C01", "b": "C02", "z": "C03"}

    def render(version, task):
        return render_reasoning_grammar(
            version,
            _active_evidence(),
            card_id_to_alias=aliases,
            task=task,
        )

    bbb = render(V38, "bbb_martins")
    oral = render(V38, "bioavailability_ma")
    assert bbb == render(V36, "bbb_martins")
    assert oral != render(V37, "bioavailability_ma")
    xgr.Grammar.from_ebnf(bbb)
    xgr.Grammar.from_ebnf(oral)


def test_v39_keeps_bbb_and_interpolates_oral_allowance():
    aliases = {"a": "C01", "b": "C02", "z": "C03"}

    def render(version, task):
        return render_reasoning_grammar(
            version,
            _active_evidence(),
            card_id_to_alias=aliases,
            task=task,
        )

    bbb = render(V39, "bbb_martins")
    oral = render(V39, "bioavailability_ma")
    assert bbb == render(V38, "bbb_martins")
    assert oral != render(V38, "bioavailability_ma")
    assert oral != render(V35, "bioavailability_ma")
    xgr.Grammar.from_ebnf(bbb)
    xgr.Grammar.from_ebnf(oral)


def test_v40_uses_free_16k_bbb_and_free_19k_oral():
    aliases = {"a": "C01", "b": "C02", "z": "C03"}

    def render(version, task):
        return render_reasoning_grammar(
            version,
            _active_evidence(),
            card_id_to_alias=aliases,
            task=task,
        )

    bbb = render(V40, "bbb_martins")
    oral = render(V40, "bioavailability_ma")
    assert bbb == render(V27, "bbb_martins")
    assert oral != render(V39, "bioavailability_ma")
    assert oral != render(V33, "bioavailability_ma")
    xgr.Grammar.from_ebnf(bbb)
    xgr.Grammar.from_ebnf(oral)


def test_v41_uses_free_15k_bbb_and_free_20k_oral():
    aliases = {"a": "C01", "b": "C02", "z": "C03"}

    def render(version, task):
        return render_reasoning_grammar(
            version,
            _active_evidence(),
            card_id_to_alias=aliases,
            task=task,
        )

    bbb = render(V41, "bbb_martins")
    oral = render(V41, "bioavailability_ma")
    assert bbb != render(V40, "bbb_martins")
    assert oral == render(V33, "bioavailability_ma")
    xgr.Grammar.from_ebnf(bbb)
    xgr.Grammar.from_ebnf(oral)


def test_v42_keeps_free_15k_bbb_and_uses_free_24k_oral():
    aliases = {"a": "C01", "b": "C02", "z": "C03"}

    def render(version, task):
        return render_reasoning_grammar(
            version,
            _active_evidence(),
            card_id_to_alias=aliases,
            task=task,
        )

    bbb = render(V42, "bbb_martins")
    oral = render(V42, "bioavailability_ma")
    assert bbb == render(V41, "bbb_martins")
    assert oral == render(V37, "bioavailability_ma")
    xgr.Grammar.from_ebnf(bbb)
    xgr.Grammar.from_ebnf(oral)


def test_v43_combines_majority_bbb_with_free_24k_oral():
    aliases = {"a": "C01", "b": "C02", "z": "C03"}

    def render(version, task):
        return render_reasoning_grammar(
            version,
            _active_evidence(),
            card_id_to_alias=aliases,
            task=task,
        )

    bbb = render(V43, "bbb_martins")
    oral = render(V43, "bioavailability_ma")
    assert bbb == render(V33, "bbb_martins")
    assert oral == render(V42, "bioavailability_ma")
    xgr.Grammar.from_ebnf(bbb)
    xgr.Grammar.from_ebnf(oral)


def test_current_prompt_versions_keep_grammar_assets_version_owned():
    aliases = {"a": "C01", "b": "C02", "z": "C03"}
    grammar = render_reasoning_grammar(
        V44,
        _active_evidence(),
        card_id_to_alias=aliases,
        task="bbb_martins",
    )
    xgr.Grammar.from_ebnf(grammar)
    assert grammar.index("RECORDS C01,C03") < grammar.index("RECORDS C02")
    assert "reasoning.gbnf.jinja" in prompt_asset_manifest(V44)["files_sha256"]
    assert (
        render_reasoning_grammar(
            V45,
            _active_evidence(),
            card_id_to_alias=aliases,
            task="bbb_martins",
        )
        is None
    )


def test_v46_enforces_one_non_nested_free_block_per_stage():
    aliases = {"a": "C01", "b": "C02", "z": "C03"}
    grammar = render_reasoning_grammar(
        V46,
        _active_evidence(),
        card_id_to_alias=aliases,
        task="bbb_martins",
    )
    xgr.Grammar.from_ebnf(grammar)
    assert grammar.index("RECORDS=C01,C03") < grammar.index("RECORDS=C02")
    assert "reasoning.gbnf.jinja" in prompt_asset_manifest(V46)["files_sha256"]

    free_reasoning = xgr.Grammar.from_ebnf(
        grammar, root_rule_name="free_reasoning"
    )
    tokenizer = xgr.TokenizerInfo(
        [""] + [chr(codepoint) for codepoint in range(10, 127)],
        stop_token_ids=[0],
    )
    compiled = xgr.GrammarCompiler(tokenizer).compile_grammar(free_reasoning)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_string("free molecular reasoning")
    assert matcher.is_completed()

    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert not matcher.accept_string("nested <stage> reasoning")


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
