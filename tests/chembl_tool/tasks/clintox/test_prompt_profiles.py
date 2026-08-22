import json
import inspect

from tools.chembl_tool.tasks.clintox.prompt_profiles import (
    DEFAULT_CLINTOX_PROMPT_PROFILE,
    TDC_SOURCE_ALIGNED_V3,
    get_clintox_prompt_profile,
)
from tools.chembl_tool.tasks.clintox.run_reasoning_batch import CONFIG
from tools.chembl_tool.tasks.clintox.run_reasoning_pipeline import (
    DEFAULT_BASE_URL,
    DEFAULT_INPUT,
    DEFAULT_MODEL,
    _final_provenance_validation_errors,
    _parse_args,
    _reason_one_group,
    _reason_single_molecule,
    _run_final_reasoning,
)


class _FakeClient:
    def __init__(self, content):
        self.content = content
        self.messages = []

    def chat_json(self, messages):
        self.messages.append(messages)
        return {
            "content": dict(self.content),
            "raw_content": json.dumps(self.content),
            "reasoning_content": "",
            "tool_calls": [],
            "tool_results": [],
            "messages": messages,
            "usage": {},
            "model": "fake",
            "id": "fake",
        }


def _retrieval(groups=None):
    return {
        "query": {"input_smiles": "CCO", "canonical_smiles": "CCO"},
        "coverage": {},
        "groups": groups or [],
    }


def _single_output():
    return {"status": "ok", "llm": {"content": {"confidence": "low"}}}


def test_source_aligned_profile_is_the_new_default_and_is_wired_through_batch_cli():
    assert DEFAULT_CLINTOX_PROMPT_PROFILE == TDC_SOURCE_ALIGNED_V3
    assert get_clintox_prompt_profile(DEFAULT_CLINTOX_PROMPT_PROFILE).name == (
        TDC_SOURCE_ALIGNED_V3
    )
    assert CONFIG.default_prompt_profile == DEFAULT_CLINTOX_PROMPT_PROFILE
    assert _parse_args([]).clintox_prompt_profile == DEFAULT_CLINTOX_PROMPT_PROFILE
    assert DEFAULT_INPUT == CONFIG.default_input
    assert DEFAULT_MODEL == CONFIG.default_model
    assert DEFAULT_BASE_URL == CONFIG.default_base_url
    assert CONFIG.default_api_key_env == "CLINTOX_LOCAL_API_KEY"


def test_all_shared_stage_adapters_accept_the_prompt_profile_contract():
    for function in (
        _reason_single_molecule,
        _reason_one_group,
        _run_final_reasoning,
    ):
        assert "prompt_profile" in inspect.signature(function).parameters


def test_predictive_final_accepts_positive_without_direct_analog():
    client = _FakeClient(
        {
            "clintox_prediction": "toxic",
            "direct_evidence_status": "no_direct_analog_retrieved",
        }
    )
    output = _run_final_reasoning(
        client,
        _retrieval(),
        _single_output(),
        [],
        prompt_profile=TDC_SOURCE_ALIGNED_V3,
    )
    assert output["status"] == "ok"
    payload = json.loads(client.messages[0][1]["content"])
    instructions = " ".join(payload["instructions"])
    assert "not required" in instructions
    assert "must choose clintox_prediction='non_toxic'" not in instructions


def test_source_aligned_final_states_the_frozen_class_contract_without_causal_overclaim():
    client = _FakeClient(
        {
            "clintox_prediction": "toxic",
            "direct_evidence_status": "no_direct_analog_retrieved",
        }
    )
    output = _run_final_reasoning(
        client,
        _retrieval(),
        _single_output(),
        [],
        prompt_profile=TDC_SOURCE_ALIGNED_V3,
    )
    assert output["status"] == "ok"
    system_text = client.messages[0][0]["content"]
    payload = json.loads(client.messages[0][1]["content"])
    instructions = " ".join(payload["instructions"])
    assert "source-defined ClinTox CT_TOX class" in system_text
    assert "not a clinical causality adjudication" in system_text
    assert "universally toxic or universally safe" in instructions
    assert "approval and an AACT toxicity-failure association can coexist" in instructions
    assert "generic hazard signal alone must not determine the class" in instructions


def test_predictive_final_validates_provenance_not_label_eligibility():
    content = {
        "clintox_prediction": "toxic",
        "direct_evidence_status": "no_direct_analog_retrieved",
    }
    assert _final_provenance_validation_errors(content, _retrieval()) == []

    false_direct_claim = {
        **content,
        "direct_evidence_status": "direct_analog_retrieved_transferable",
    }
    assert _final_provenance_validation_errors(false_direct_claim, _retrieval())
def test_removed_prompt_profiles_are_rejected():
    for removed in (
        "direct_anchor_required_v1",
        "clinical_trial_failure_evidence_predictive_v2",
    ):
        try:
            get_clintox_prompt_profile(removed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"removed prompt profile remains addressable: {removed}")
