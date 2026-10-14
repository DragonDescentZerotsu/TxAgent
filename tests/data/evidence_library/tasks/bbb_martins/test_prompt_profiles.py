import json
from types import SimpleNamespace

import pytest

from functools import partial
from predict.prompts import get_profile, task_profiles
get_bbb_prompt_profile = partial(get_profile, "bbb_martins")
DEFAULT_BBB_PROMPT_PROFILE = task_profiles("bbb_martins")["default"]
MEANINGFUL_CNS_ACCESS_V1 = 'meaningful_cns_access_v1'
MEANINGFUL_CNS_ADJUDICATION_V2 = 'meaningful_cns_adjudication_v2'
MEANINGFUL_CNS_ADJUDICATION_V3 = 'meaningful_cns_adjudication_v3'
from predict.tasks.bbb_martins.prediction_filter import final_profile_validation_errors


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




