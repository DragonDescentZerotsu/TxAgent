from tools.chembl_tool.paper_experiments.seed_starling_matrix_reuse import (
    _contract_mismatches,
    _merge_successful_group_outputs,
)


def _manifest(**updates):
    payload = {
        "model": "gpt-oss-120b",
        "reasoning_effort": "",
        "temperature": 0,
        "thinking": {"type": "disabled"},
        "identity_blind": True,
        "tool_execution_mode": "harness_prefetch",
        "experiment_mode": "direct",
        "retrieval_source": "starling",
        "max_tool_rounds": 3,
        "top_k_per_group": 3,
        "min_similarity": 0.3,
        "neighbor_selector": "similarity",
    }
    payload.update(updates)
    return payload


def test_reuse_contract_accepts_legacy_missing_standard_context_profile():
    source = _manifest()
    target = _manifest(neighbor_context_profile="standard")

    assert _contract_mismatches(source, target) == []


def test_reuse_contract_rejects_model_or_visibility_changes():
    source = _manifest()
    target = _manifest(model="other-model", identity_blind=False)

    assert _contract_mismatches(source, target) == ["model", "identity_blind"]


def test_reuse_merge_preserves_successful_target_group_output():
    reusable = [
        {"group_id": "a", "status": "ok", "value": "historical"},
        {"group_id": "b", "status": "ok", "value": "historical"},
    ]
    existing = [
        {"group_id": "b", "status": "ok", "value": "fresh"},
        {"group_id": "c", "status": "error", "value": "failed"},
    ]

    assert _merge_successful_group_outputs(existing, reusable) == [
        {"group_id": "a", "status": "ok", "value": "historical"},
        {"group_id": "b", "status": "ok", "value": "fresh"},
    ]
