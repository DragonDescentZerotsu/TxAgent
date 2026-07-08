from tools.chembl_tool.tasks.bioavailability_ma.specific_postprocess_policy import apply_postprocess_policy


def test_apply_postprocess_policy_updates_only_rule_hits():
    prediction_rows = [
        {
            "query_index": 1,
            "label": 1,
            "bioavailability_prediction": "low",
            "pred_label": 0,
            "correct": False,
            "final_summary": "old",
            "status": "ok",
        },
        {
            "query_index": 2,
            "label": 0,
            "bioavailability_prediction": "high",
            "pred_label": 1,
            "correct": False,
            "final_summary": "unchanged",
            "status": "ok",
        },
    ]
    feature_rows = [
        {
            "query_index": 1,
            "prediction": "low",
            "decision_state": "fallback_uncertain",
            "source_level_consensus": "no_eligible_clean_source_level_direct_f_vote",
            "soft_high_sources": 2,
            "soft_low_sources": 1,
            "soft_straddling_sources": 2,
            "starling_high_sources": 2,
        },
        {
            "query_index": 2,
            "prediction": "high",
            "decision_state": "fallback_uncertain",
            "source_level_consensus": "no_eligible_clean_source_level_direct_f_vote",
            "soft_high_sources": 2,
            "soft_low_sources": 1,
            "soft_straddling_sources": 2,
            "starling_high_sources": 2,
        },
    ]

    rows = apply_postprocess_policy(
        prediction_rows,
        feature_rows,
        "force_state_or_starling_soft_high_anchor",
    )

    assert rows[0]["bioavailability_prediction"] == "high"
    assert rows[0]["pred_label"] == 1
    assert rows[0]["correct"] is True
    assert rows[0]["postprocess_policy"]["changed"] is True
    assert rows[0]["pre_postprocess_prediction"] == "low"
    assert "force_state_or_starling_soft_high_anchor" in rows[0]["final_summary"]
    assert rows[1]["bioavailability_prediction"] == "high"
    assert rows[1]["correct"] is False
    assert rows[1]["postprocess_policy"]["changed"] is False
