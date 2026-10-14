import json
from types import SimpleNamespace

import pytest

from functools import partial
from predict.prompts import get_profile, task_profiles
get_skin_prompt_profile = partial(get_profile, "skin_reaction")
DEFAULT_SKIN_PROMPT_PROFILE = task_profiles("skin_reaction")["default"]
LEGACY_SKIN_REACTION_V1 = 'legacy_skin_reaction_v1'
SENSITIZATION_ALIGNED_V2 = 'sensitization_aligned_v2'
SENSITIZATION_NEGATIVE_TRANSFER_V3 = 'sensitization_negative_transfer_v3'


def test_new_skin_runs_default_to_sensitization_aligned_profile():
    profile = get_skin_prompt_profile(DEFAULT_SKIN_PROMPT_PROFILE)

    assert profile.name == SENSITIZATION_ALIGNED_V2
    assert profile.label_scope == "skin_sensitization_contact_allergy.v2"
    assert "label_scope" in profile.final_required_fields
    assert "phototoxicity" not in profile.final_schema["main_evidence_type"]
    assert "irritation_or_corrosion" not in profile.final_schema["main_evidence_type"]
    assert "exposure_context_only" not in profile.final_schema["main_evidence_type"]
    assert profile.final_allowed_values["main_evidence_type"] == {
        "direct_sensitization_anchor",
        "sensitization_aop",
        "structural_haptenation_prior",
        "weak_or_no_sensitization_evidence",
    }
    assert profile.final_allowed_values["confidence"] == {"high", "moderate", "low"}


def test_negative_transfer_profile_adds_only_the_asymmetric_negative_rule():
    aligned = get_skin_prompt_profile(SENSITIZATION_ALIGNED_V2)
    candidate = get_skin_prompt_profile(SENSITIZATION_NEGATIVE_TRANSFER_V3)

    assert candidate.single_instructions == aligned.single_instructions
    assert candidate.group_instructions[:-1] == aligned.group_instructions
    assert candidate.final_instructions[:-1] == aligned.final_instructions
    assert "transferability is high" in candidate.group_instructions[-1]
    assert "low- or moderate-transferability" in candidate.final_instructions[-1]








