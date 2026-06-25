from tools.chembl_tool.tasks.bioavailability_ma.specific_fallback_policy import (
    candidate_rule_prediction,
    evaluate_candidate_rules,
    extract_feature_row,
    select_candidate_rule,
    summarize_feature_rows,
)


def test_extract_feature_row_captures_fallback_direct_f_and_factor_features():
    row = {
        "query_index": 7,
        "run_id": "batch_idx00007",
        "run_dir": "runs/batch_idx00007",
        "smiles": "CCO",
        "label": 1,
        "bioavailability_prediction": "low",
        "pred_label": 0,
        "confidence": "low",
        "correct": False,
        "status": "ok",
        "final_status": "ok",
    }
    final_output = {
        "status": "ok",
        "compiled_evidence_summary": {
            "context_flags": {"threshold_sensitive_15_25_percent": "near cutoff"},
            "source_level_direct_f_summary": {
                "source_level_consensus": "no_eligible_clean_source_level_direct_f_vote",
                "n_source_molecules_with_direct_f_values": 3,
                "n_eligible_clean_direct_vote_sources": 0,
                "n_clean_but_not_direct_vote_sources": 2,
                "n_review_required_context_only_sources": 1,
                "n_threshold_straddling_sources": 1,
                "eligible_clean_source_direction_counts": {},
                "clean_context_source_direction_counts": {
                    "supports_high_bioavailability": 1,
                    "mixed_threshold_straddling": 1,
                },
                "review_context_source_direction_counts": {
                    "argues_against_high_bioavailability": 1,
                },
                "source_summaries": [
                    {
                        "source_molecule_ids": ["STARLING_A"],
                        "evidence_sources": ["starling-labs/Oral_Bioavailability"],
                        "source_direction": "supports_high_bioavailability",
                    },
                    {
                        "source_molecule_ids": ["CHEMBL1"],
                        "evidence_sources": [],
                        "source_direction": "mixed_threshold_straddling",
                    },
                    {
                        "source_molecule_ids": ["STARLING_B"],
                        "evidence_sources": ["starling-labs/Oral_Bioavailability"],
                        "source_direction": "argues_against_high_bioavailability",
                        "parent_scope_reasons": ["formulation_or_salt_specific_scope"],
                    },
                ],
            },
            "source_transfer_summary": {
                "scope_class_counts": {
                    "clean_parent_or_close_analog_context": 2,
                    "same_active_moiety_salt_or_freebase_context": 1,
                },
                "scope_direction_counts": {
                    "same_active_moiety_salt_or_freebase_context": {
                        "argues_against_high_bioavailability": 1,
                    },
                },
                "anchor_candidate_sources": [
                    {
                        "source_direction": "argues_against_high_bioavailability",
                    }
                ],
            },
            "evidence_gate_summary": {
                "deterministic_decision_policy": {
                    "decision_state": "fallback_uncertain",
                    "forced_class": None,
                    "fallback_candidate_class": None,
                    "recommended_class": None,
                    "recommendation_strength": "none",
                    "reason_codes": ["soft_direct_f_context_is_context_only"],
                    "uncertainty_flags": ["no_eligible_clean_source_level_direct_f_vote"],
                    "soft_direct_f_context": {
                        "n_soft_clean_direct_f_sources": 2,
                        "soft_clean_source_direction_counts": {
                            "supports_high_bioavailability": 1,
                            "mixed_threshold_straddling": 1,
                        },
                        "net_soft_context_direction": "soft_context_leans_low_or_threshold_sensitive",
                        "min_value_percent": 12.0,
                        "max_value_percent": 85.0,
                    },
                },
                "single_molecule_prior": {
                    "oral_bioavailability_prior": "low",
                    "absorption_prior": "unfavorable",
                    "solubility_or_dissolution_prior": "unfavorable",
                    "metabolism_or_clearance_prior": "mixed_or_unclear",
                    "confidence": "moderate",
                },
                "factor_signal_counts": {
                    "Fa": {"risk_for_lower_F": 1},
                    "Fg": {"mixed_or_context": 1},
                    "Fh": {"supports_higher_F": 1},
                },
                "factor_product_guard": {
                    "factor_limiting_strength": {
                        "Fa": ["strong_limiting"],
                        "Fg": ["not_limiting"],
                        "Fh": ["not_limiting"],
                    },
                    "class_constraints": {"high": "blocked"},
                    "audit_warnings": {"no_direct_absorption_risk_without_fa_support": "warn"},
                },
            },
        },
    }

    feature = extract_feature_row(row, final_output)

    assert feature["decision_state"] == "fallback_uncertain"
    assert feature["n_clean_but_not_direct_vote_sources"] == 2
    assert feature["soft_high_sources"] == 1
    assert feature["soft_straddling_sources"] == 1
    assert feature["review_low_sources"] == 1
    assert feature["starling_source_count"] == 2
    assert feature["starling_high_sources"] == 1
    assert feature["starling_low_sources"] == 1
    assert feature["chembl_straddling_sources"] == 1
    assert feature["transfer_clean_parent_or_close_analog_sources"] == 2
    assert feature["transfer_same_active_moiety_salt_or_freebase_sources"] == 1
    assert feature["transfer_anchor_candidate_sources"] == 1
    assert feature["transfer_anchor_low_sources"] == 1
    assert feature["Fa_risk_for_lower_f_count"] == 1
    assert feature["Fh_supports_higher_f_count"] == 1
    assert feature["strong_limiting_factor_count"] == 1
    assert feature["strong_absorption_prior"] is True
    assert feature["low_property_prior"] is True
    assert feature["class_constraint_keys"] == ["high"]
    assert feature["has_threshold_sensitive_flag"] is True


def test_extract_feature_row_reconstructs_source_transfer_scope_for_older_outputs():
    row = {
        "query_index": 8,
        "run_id": "batch_idx00008",
        "run_dir": "runs/batch_idx00008",
        "smiles": "CCO",
        "label": 0,
        "bioavailability_prediction": "high",
        "pred_label": 1,
        "confidence": "low",
        "correct": False,
        "status": "ok",
        "final_status": "ok",
    }
    final_output = {
        "status": "ok",
        "compiled_evidence_summary": {
            "source_level_direct_f_summary": {
                "source_level_consensus": "no_eligible_clean_source_level_direct_f_vote",
                "source_summaries": [
                    {
                        "source_molecule_ids": ["CHEMBL_SALT"],
                        "evidence_sources": [],
                        "label_vote_bucket": "review_required_context_only",
                        "source_direction": "supports_high_bioavailability",
                        "parent_scope_reasons": ["formulation_or_salt_specific_scope"],
                    },
                    {
                        "source_molecule_ids": ["CHEMBL_PARENT"],
                        "evidence_sources": [],
                        "label_vote_bucket": "clean_but_not_direct_vote",
                        "source_direction": "argues_against_high_bioavailability",
                        "parent_scope_reasons": [],
                    },
                ],
            },
            "evidence_gate_summary": {
                "deterministic_decision_policy": {"decision_state": "fallback_uncertain"},
                "single_molecule_prior": {},
                "factor_signal_counts": {},
                "factor_product_guard": {},
            },
        },
    }

    feature = extract_feature_row(row, final_output)

    assert feature["transfer_same_active_moiety_salt_or_freebase_sources"] == 1
    assert feature["transfer_clean_parent_or_close_analog_sources"] == 1
    assert feature["transfer_anchor_candidate_sources"] == 1
    assert feature["transfer_anchor_high_sources"] == 1


def test_candidate_rules_respect_forced_state_before_fallback_logic():
    row = {
        "prediction": "low",
        "decision_state": "force_high",
        "forced_class": "high",
        "net_soft_context_direction": "soft_context_leans_low_or_threshold_sensitive",
        "strong_limiting_factor_count": 1,
        "strong_absorption_prior": True,
    }

    assert candidate_rule_prediction(row, "force_state_or_low") == "high"
    assert candidate_rule_prediction(row, "force_state_or_soft_net_visible") == "high"
    assert candidate_rule_prediction(row, "current_prediction") == "low"


def test_mechanism_low_blocker_candidate_uses_narrow_structural_alerts():
    row = {
        "prediction": "high",
        "decision_state": "fallback_uncertain",
        "forced_class": None,
        "source_level_consensus": "no_eligible_clean_source_level_direct_f_vote",
        "low_property_prior": True,
        "alert_permanent_quaternary_charge": True,
        "chembl_low_sources": 0,
        "starling_low_sources": 1,
        "starling_straddling_sources": 0,
        "supports_higher_f_factor_count": 0,
        "net_soft_context_direction": "soft_context_mixed_or_weak",
    }

    assert candidate_rule_prediction(row, "force_state_or_mechanism_low_blockers") == "low"

    row["alert_permanent_quaternary_charge"] = False
    assert candidate_rule_prediction(row, "force_state_or_mechanism_low_blockers") == "high"


def test_starling_soft_high_anchor_candidate_is_conservative():
    row = {
        "prediction": "low",
        "decision_state": "fallback_uncertain",
        "forced_class": None,
        "source_level_consensus": "no_eligible_clean_source_level_direct_f_vote",
        "soft_high_sources": 2,
        "soft_low_sources": 1,
        "soft_straddling_sources": 2,
        "starling_high_sources": 2,
    }

    assert candidate_rule_prediction(row, "force_state_or_starling_soft_high_anchor") == "high"

    row["starling_high_sources"] = 1
    assert candidate_rule_prediction(row, "force_state_or_starling_soft_high_anchor") == "low"
    row["starling_high_sources"] = 2
    row["soft_low_sources"] = 2
    assert candidate_rule_prediction(row, "force_state_or_starling_soft_high_anchor") == "low"
    row["soft_low_sources"] = 1
    row["prediction"] = "high"
    assert candidate_rule_prediction(row, "force_state_or_starling_soft_high_anchor") == "high"


def test_no_evidence_high_fallback_candidate_requires_empty_context_and_no_risk():
    row = {
        "prediction": "low",
        "decision_state": "fallback_uncertain",
        "forced_class": None,
        "source_level_consensus": "no_eligible_clean_source_level_direct_f_vote",
        "n_source_molecules_with_direct_f_values": 0,
        "n_clean_but_not_direct_vote_sources": 0,
        "n_review_required_context_only_sources": 0,
        "soft_high_sources": 0,
        "soft_low_sources": 0,
        "soft_straddling_sources": 0,
        "starling_source_count": 0,
        "chembl_source_count": 0,
        "transfer_anchor_candidate_sources": 0,
        "alert_permanent_quaternary_charge": False,
        "alert_beta_lactam_anionic": False,
        "alert_dihydropyridine_diester": False,
        "alert_ester_prodrug_or_active_moiety": False,
        "alert_high_ionization_low_permeability": False,
        "risk_for_lower_f_factor_count": 0,
        "strong_limiting_factor_count": 0,
        "moderate_analog_risk_factor_count": 0,
    }

    assert candidate_rule_prediction(row, "force_state_or_no_evidence_high_fallback") == "high"
    assert candidate_rule_prediction(row, "force_state_or_valid_selected_high_rescues") == "high"

    row["risk_for_lower_f_factor_count"] = 1
    assert candidate_rule_prediction(row, "force_state_or_no_evidence_high_fallback") == "low"
    row["risk_for_lower_f_factor_count"] = 0
    row["starling_source_count"] = 1
    assert candidate_rule_prediction(row, "force_state_or_no_evidence_high_fallback") == "low"


def test_evaluate_candidate_rules_reports_transition_audits():
    rows = [
        {
            "query_index": 1,
            "label": 1,
            "prediction": "low",
            "pred_label": 0,
            "correct": False,
            "status": "ok",
            "decision_state": "fallback_uncertain",
            "forced_class": None,
            "net_soft_context_direction": "no_soft_clean_direct_f_context",
            "strong_limiting_factor_count": 0,
            "strong_absorption_prior": False,
            "oral_bioavailability_prior": "mixed_or_unclear",
            "absorption_prior": "mixed_or_unclear",
            "supports_higher_f_factor_count": 0,
        },
        {
            "query_index": 2,
            "label": 0,
            "prediction": "high",
            "pred_label": 1,
            "correct": False,
            "status": "ok",
            "decision_state": "force_low",
            "forced_class": "low",
            "net_soft_context_direction": "soft_context_leans_high",
            "strong_limiting_factor_count": 0,
            "strong_absorption_prior": False,
        },
    ]

    rule_rows, metrics = evaluate_candidate_rules(rows)
    summary = summarize_feature_rows(rows)

    assert len(rule_rows) == 24
    conservative = metrics["force_state_or_conservative_high_fallback"]
    assert conservative["n_changed_vs_current"] == 2
    assert conservative["changed_wrong_to_correct_indices"] == [1, 2]
    assert conservative["metrics"]["accuracy"] == 1.0
    assert summary["fallback_uncertain_count"] == 1


def test_select_candidate_rule_uses_calibration_metric_and_guardrail():
    rows = [
        {
            "query_index": 1,
            "label": 1,
            "prediction": "low",
            "pred_label": 0,
            "correct": False,
            "status": "ok",
            "decision_state": "fallback_uncertain",
            "forced_class": None,
            "net_soft_context_direction": "no_soft_clean_direct_f_context",
            "strong_limiting_factor_count": 0,
            "strong_absorption_prior": False,
            "oral_bioavailability_prior": "mixed_or_unclear",
            "absorption_prior": "mixed_or_unclear",
            "low_property_prior": False,
            "risk_for_lower_f_factor_count": 0,
            "supports_higher_f_factor_count": 0,
            "soft_high_sources": 0,
            "soft_low_sources": 0,
            "soft_straddling_sources": 0,
            "starling_high_sources": 0,
            "starling_low_sources": 0,
            "starling_straddling_sources": 0,
            "review_high_sources": 0,
            "review_low_sources": 0,
            "review_straddling_sources": 0,
        },
        {
            "query_index": 2,
            "label": 0,
            "prediction": "high",
            "pred_label": 1,
            "correct": False,
            "status": "ok",
            "decision_state": "fallback_uncertain",
            "forced_class": None,
            "net_soft_context_direction": "soft_context_leans_low_or_threshold_sensitive",
            "strong_limiting_factor_count": 1,
            "strong_absorption_prior": False,
            "oral_bioavailability_prior": "low",
            "absorption_prior": "unfavorable",
            "low_property_prior": True,
            "risk_for_lower_f_factor_count": 1,
            "supports_higher_f_factor_count": 0,
            "soft_high_sources": 0,
            "soft_low_sources": 2,
            "soft_straddling_sources": 0,
            "starling_high_sources": 0,
            "starling_low_sources": 1,
            "starling_straddling_sources": 0,
            "review_high_sources": 0,
            "review_low_sources": 0,
            "review_straddling_sources": 0,
        },
    ]

    selected = select_candidate_rule(rows, selection_metric="macro_f1", min_negative_recall=1.0)

    assert selected["selected_rule"] in {
        "force_state_or_soft_net_visible",
        "force_state_or_conservative_high_fallback",
        "force_state_or_source_balanced",
        "force_state_or_high_unless_low_anchor",
        "force_state_or_starling_soft_high_anchor",
        "force_state_or_no_evidence_high_fallback",
        "force_state_or_valid_selected_high_rescues",
    }
    assert selected["selected_calibration_metrics"]["metrics"]["per_class"]["0"]["recall"] == 1.0
