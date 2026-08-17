import json
from types import SimpleNamespace

import pytest

from tools.chembl_tool.tasks.bioavailability_ma.prompt_profiles import (
    DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE,
    F20_EVIDENCE_CALIBRATED_V2,
    LEGACY_BIOAVAILABILITY_V1,
    get_bioavailability_prompt_profile,
)
from tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_pipeline import (
    _group_evidence_source,
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
        return {"content": {"bioavailability_prediction": "high"}}


def test_new_bio_runs_default_to_f20_calibrated_profile():
    profile = get_bioavailability_prompt_profile(DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE)
    single = " ".join(profile.single_instructions)
    group = " ".join(profile.group_instructions)
    final = " ".join(profile.final_instructions)

    assert profile.name == F20_EVIDENCE_CALIBRATED_V2
    assert profile.label_scope == "absolute_oral_bioavailability_f20.v2"
    assert "modest binary threshold" in single
    assert "not individually sufficient" in single
    assert "separately from transferability" in group
    assert "does not reverse the observed direction" in group
    assert "not evidence for F < 20%" in final
    assert "must not default" in final
    assert "Generic descriptor liabilities" in final


def test_legacy_profile_preserves_old_distant_analog_rule():
    legacy = get_bioavailability_prompt_profile(LEGACY_BIOAVAILABILITY_V1)
    calibrated = get_bioavailability_prompt_profile(F20_EVIDENCE_CALIBRATED_V2)

    assert "Do not use distant_analog" in " ".join(legacy.group_instructions)
    assert "Do not use distant_analog" not in " ".join(calibrated.group_instructions)


def test_group_payload_uses_selected_direction_transfer_contract():
    group = {
        "group_id": "Observed.direct_oral_bioavailability",
        "tier": "Observed",
        "endpoint_group": "direct_oral_bioavailability",
        "neighbors": [],
    }
    payload = _group_prompt_payload({"identity_hidden": True}, group)
    instructions = " ".join(payload["instructions"])

    assert (
        "consistently high analog outcomes remain supports_high_bioavailability"
        in instructions
    )
    assert (
        "Low or not-applicable transferability lowers evidence weight" in instructions
    )


def test_group_evidence_source_survives_identity_blind_wrapping():
    source_name = "starling-labs/bioavailability_ma/Fa"
    row = {"minimal_evidence": {"source": {"name": source_name}}}
    group = {"neighbors": [{"evidence_rows": [row]}]}

    assert _group_evidence_source(group) == source_name


def test_final_payload_contains_threshold_specific_low_evidence_gate():
    client = CapturingClient()
    output = _run_final_reasoning(
        client,
        {"query": {"identity_hidden": True}, "coverage": {}, "groups": []},
        {"status": "error"},
        [],
    )
    payload = json.loads(client.messages[-1]["content"])
    instructions = " ".join(payload["instructions"])

    assert output["status"] == "ok"
    assert (
        "Predict low only when affirmative threshold-relevant evidence" in instructions
    )
    assert "must not default to bioavailability_prediction='low'" in instructions
    assert (
        "A high prediction means only that F >= 20% is better supported" in instructions
    )


def test_cli_and_old_manifest_keep_profiles_reproducible():
    assert _parse_args([]).bioavailability_prompt_profile == F20_EVIDENCE_CALIBRATED_V2
    assert (
        _parse_args(
            ["--bioavailability-prompt-profile", LEGACY_BIOAVAILABILITY_V1]
        ).bioavailability_prompt_profile
        == LEGACY_BIOAVAILABILITY_V1
    )
    assert _manifest_prompt_profile({}) == LEGACY_BIOAVAILABILITY_V1


def test_branch_reuse_rejects_cross_profile_mix(tmp_path):
    source = tmp_path / "source-run"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps({"task_prompt_profile": LEGACY_BIOAVAILABILITY_V1}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        bioavailability_prompt_profile=F20_EVIDENCE_CALIBRATED_V2,
        single_analysis_source_run_dir=str(source),
        group_analysis_source_run_dir="",
    )

    with pytest.raises(ValueError, match="branch reuse is forbidden"):
        _validate_prompt_profile_reuse(args)
