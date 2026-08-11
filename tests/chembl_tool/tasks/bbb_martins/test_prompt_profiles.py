import json
from types import SimpleNamespace

import pytest

from tools.chembl_tool.tasks.bbb_martins.prompt_profiles import (
    DEFAULT_BBB_PROMPT_PROFILE,
    MEANINGFUL_CNS_ACCESS_V1,
    MEANINGFUL_CNS_ADJUDICATION_V2,
    MEANINGFUL_CNS_ADJUDICATION_V3,
    final_profile_validation_errors,
    get_bbb_prompt_profile,
)
from tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline import (
    _group_prompt_payload,
    _manifest_prompt_profile,
    _parse_args,
    _run_final_reasoning,
    _validate_prompt_profile_reuse,
)


class CapturingClient:
    def __init__(self):
        self.messages = None

    def chat_json(self, messages):
        self.messages = messages
        return {
            "content": {
                "bbb_prediction": "pass",
                "integrated_outcome_state": "transferable_positive",
                "negative_evidence_basis": "none",
            }
        }


def test_unpromoted_bbb_candidates_do_not_change_default_profile():
    profile = get_bbb_prompt_profile(DEFAULT_BBB_PROMPT_PROFILE)

    assert profile.name == MEANINGFUL_CNS_ACCESS_V1
    assert profile.label_scope == "meaningful_cns_access.v1"


def test_legacy_profile_preserves_previous_schema_and_instructions():
    legacy = get_bbb_prompt_profile(MEANINGFUL_CNS_ACCESS_V1)

    assert "observed_outcome_direction" not in legacy.group_schema
    assert "integrated_outcome_state" not in legacy.final_schema
    assert "negative_evidence_basis" not in legacy.final_schema
    assert "Do not use distant_analog" in " ".join(legacy.group_instructions)


def test_group_payload_uses_selected_observation_transfer_contract():
    payload = _group_prompt_payload(
        {"identity_hidden": True},
        {
            "group_id": "Direct.bbb",
            "tier": "Direct",
            "endpoint_group": "direct_bbb",
            "neighbors": [],
        },
        prompt_profile=MEANINGFUL_CNS_ADJUDICATION_V2,
    )
    instructions = " ".join(payload["instructions"])

    assert "observed_outcome_direction" in payload["required_json_schema"]
    assert "Low transferability lowers weight" in instructions
    assert "not observed evidence of restricted CNS access" in instructions


def test_final_payload_contains_affirmative_negative_evidence_gate():
    client = CapturingClient()
    output = _run_final_reasoning(
        client,
        {"query": {"identity_hidden": True}, "coverage": {}, "groups": []},
        {"status": "error"},
        [],
        prompt_profile=MEANINGFUL_CNS_ADJUDICATION_V2,
    )
    payload = json.loads(client.messages[-1]["content"])
    instructions = " ".join(payload["instructions"])

    assert output["status"] == "ok"
    assert "negative_evidence_basis" in payload["required_json_schema"]
    assert "Predict fail only when negative_evidence_basis" in instructions
    assert "cannot make that positive observation support fail" in instructions


def test_candidate_validation_rejects_fail_without_negative_basis():
    errors = final_profile_validation_errors(
        {
            "bbb_prediction": "fail",
            "integrated_outcome_state": "insufficient",
            "negative_evidence_basis": "none",
        },
        profile=MEANINGFUL_CNS_ADJUDICATION_V2,
    )

    assert errors == ["fail requires affirmative negative_evidence_basis"]
    assert not final_profile_validation_errors(
        {"bbb_prediction": "fail", "negative_evidence_basis": "none"},
        profile=MEANINGFUL_CNS_ACCESS_V1,
    )


def test_balanced_candidate_restores_narrow_intrinsic_negative_basis():
    profile = get_bbb_prompt_profile(MEANINGFUL_CNS_ADJUDICATION_V3)
    final = " ".join(profile.final_instructions)

    assert profile.label_scope == "meaningful_cns_access.adjudication_v3"
    assert (
        "convergent_intrinsic_barriers"
        in profile.final_schema["negative_evidence_basis"]
    )
    assert "at least two independent severe barriers" in final
    assert "do not default to pass" in final
    assert not final_profile_validation_errors(
        {
            "bbb_prediction": "fail",
            "integrated_outcome_state": "mixed",
            "negative_evidence_basis": "convergent_intrinsic_barriers",
        },
        profile=MEANINGFUL_CNS_ADJUDICATION_V3,
    )


def test_cli_and_old_manifest_keep_profiles_reproducible():
    assert _parse_args([]).bbb_prompt_profile == MEANINGFUL_CNS_ACCESS_V1
    assert (
        _parse_args(
            ["--bbb-prompt-profile", MEANINGFUL_CNS_ACCESS_V1]
        ).bbb_prompt_profile
        == MEANINGFUL_CNS_ACCESS_V1
    )
    assert _manifest_prompt_profile({}) == MEANINGFUL_CNS_ACCESS_V1


def test_branch_reuse_rejects_cross_profile_mix(tmp_path):
    source = tmp_path / "source-run"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps({"task_prompt_profile": MEANINGFUL_CNS_ACCESS_V1}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        bbb_prompt_profile=MEANINGFUL_CNS_ADJUDICATION_V2,
        single_analysis_source_run_dir=str(source),
        group_analysis_source_run_dir="",
    )

    with pytest.raises(ValueError, match="branch reuse is forbidden"):
        _validate_prompt_profile_reuse(args)
