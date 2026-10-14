import json
from types import SimpleNamespace

import pytest

from functools import partial
from predict.prompts import get_profile, task_profiles
get_bioavailability_prompt_profile = partial(get_profile, "bioavailability_ma")
DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE = task_profiles("bioavailability_ma")["default"]
F20_EVIDENCE_CALIBRATED_V2 = 'f20_evidence_calibrated_v2'
LEGACY_BIOAVAILABILITY_V1 = 'legacy_bioavailability_v1'


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










