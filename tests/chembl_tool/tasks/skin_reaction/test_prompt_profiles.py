import json
from types import SimpleNamespace

import pytest

from tools.chembl_tool.tasks.skin_reaction.prompt_profiles import (
    DEFAULT_SKIN_PROMPT_PROFILE,
    LEGACY_SKIN_REACTION_V1,
    SENSITIZATION_ALIGNED_V2,
    SENSITIZATION_NEGATIVE_TRANSFER_V3,
    get_skin_prompt_profile,
)
from tools.chembl_tool.tasks.skin_reaction.run_reasoning_pipeline import (
    _manifest_prompt_profile,
    _validate_prompt_profile_reuse,
)


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


def test_missing_historical_manifest_field_maps_to_legacy_profile():
    assert _manifest_prompt_profile({}) == LEGACY_SKIN_REACTION_V1


def test_direct_branch_reuse_rejects_cross_profile_mix(tmp_path):
    source = tmp_path / "source-run"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps({"task_prompt_profile": LEGACY_SKIN_REACTION_V1}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        skin_prompt_profile=SENSITIZATION_ALIGNED_V2,
        single_analysis_source_run_dir=str(source),
        group_analysis_source_run_dir="",
    )

    with pytest.raises(ValueError, match="branch reuse is forbidden"):
        _validate_prompt_profile_reuse(args)


def test_direct_branch_reuse_accepts_matching_profile(tmp_path):
    source = tmp_path / "source-run"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps({"task_prompt_profile": SENSITIZATION_ALIGNED_V2}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        skin_prompt_profile=SENSITIZATION_ALIGNED_V2,
        single_analysis_source_run_dir=str(source),
        group_analysis_source_run_dir="",
    )

    _validate_prompt_profile_reuse(args)


def test_global_pool_preparation_uses_batch_profile_before_run_manifest(tmp_path):
    batch = tmp_path / "source-batch"
    source = batch / "runs" / "source-run"
    source.mkdir(parents=True)
    (batch / "manifest.json").write_text(
        json.dumps({"task_prompt_profile": SENSITIZATION_ALIGNED_V2}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        skin_prompt_profile=SENSITIZATION_ALIGNED_V2,
        single_analysis_source_run_dir=str(source),
        group_analysis_source_run_dir="",
    )

    _validate_prompt_profile_reuse(args)
