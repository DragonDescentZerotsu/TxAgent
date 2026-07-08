from tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_pipeline_specific import (
    _apply_specific_decision_policy,
    _parse_json_content_specific,
)


def test_specific_json_parser_repairs_invalid_backslash_escape():
    parsed = _parse_json_content_specific('{"reasoning_summary": "Caco-2 P\\_app supports Fa"}')

    assert parsed["reasoning_summary"] == "Caco-2 P\\_app supports Fa"


def test_specific_json_parser_extracts_json_object_from_wrapped_text():
    parsed = _parse_json_content_specific('Here is JSON:\n{"bioavailability_prediction": "high"}\n')

    assert parsed["bioavailability_prediction"] == "high"


def test_policy_override_does_not_apply_soft_high_recommendation():
    response = {"content": {"bioavailability_prediction": "low", "confidence": "moderate", "main_reasons": [], "label_uncertainty_flags": []}}
    compiled = _compiled_with_policy_state(
        soft_context={
            "soft_clean_source_direction_counts": {"supports_high_bioavailability": 4},
            "net_soft_context_direction": "soft_context_leans_high",
        },
        guard={"factor_limiting_strength": {"Fa": [], "Fg": [], "Fh": []}, "single_molecule_prior": {}},
    )

    updated = _apply_specific_decision_policy(response, compiled)

    assert updated["content"]["bioavailability_prediction"] == "low"
    assert "policy_override" not in updated["content"]


def test_policy_override_applies_eligible_clean_consensus_high():
    response = {"content": {"bioavailability_prediction": "low", "confidence": "moderate", "main_reasons": [], "label_uncertainty_flags": []}}
    compiled = {
        "evidence_gate_summary": {
            "source_level_direct_f_summary": {
                "source_level_consensus": "eligible_clean_sources_consensus_high",
                "eligible_clean_source_direction_counts": {"supports_high_bioavailability": 2},
            },
            "soft_direct_f_context": {
                "soft_clean_source_direction_counts": {},
                "net_soft_context_direction": "no_soft_clean_direct_f_context",
            },
            "factor_product_guard": {
                "factor_limiting_strength": {"Fa": [], "Fg": [], "Fh": []},
                "single_molecule_prior": {},
            },
        }
    }

    updated = _apply_specific_decision_policy(response, compiled)

    assert updated["content"]["bioavailability_prediction"] == "high"
    assert updated["content"]["policy_override"]["applied"] is True
    assert updated["content"]["policy_override"]["new_prediction"] == "high"


def test_policy_override_does_not_flip_from_constraint_without_recommendation():
    response = {"content": {"bioavailability_prediction": "high", "confidence": "low", "main_reasons": [], "label_uncertainty_flags": []}}
    compiled = _compiled_with_policy_state(
        soft_context={
            "soft_clean_source_direction_counts": {"supports_high_bioavailability": 1, "mixed_threshold_straddling": 2},
            "net_soft_context_direction": "soft_context_mixed_or_weak",
        },
        guard={
            "class_constraints": {"high": "diagnostic high block"},
            "factor_limiting_strength": {"Fa": [], "Fg": [], "Fh": []},
            "single_molecule_prior": {},
        },
    )

    updated = _apply_specific_decision_policy(response, compiled)

    assert updated["content"]["bioavailability_prediction"] == "high"
    assert "policy_override" not in updated["content"]


def _compiled_with_policy_state(*, soft_context, guard):
    return {
        "evidence_gate_summary": {
            "source_level_direct_f_summary": {
                "source_level_consensus": "no_eligible_clean_source_level_direct_f_vote",
                "eligible_clean_source_direction_counts": {},
            },
            "soft_direct_f_context": soft_context,
            "factor_product_guard": guard,
        }
    }
