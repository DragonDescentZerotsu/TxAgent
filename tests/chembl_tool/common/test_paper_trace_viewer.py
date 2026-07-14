from pathlib import Path


VIEWER = Path("tools/trace_viewer/viewer.html")
STARTER = Path("tools/trace_viewer/start_viewer.sh")


def test_trace_viewer_targets_only_final_paper_runs():
    html = VIEWER.read_text(encoding="utf-8")
    starter = STARTER.read_text(encoding="utf-8")

    assert 'id: "runs"' in html
    assert 'id: "runs_deployment_visible_prefetched"' in html
    assert 'id: "runs_deployment_visible"' in html
    assert "predictions.jsonl" in html
    assert "retrieval.json" in html
    assert "trace_messages.jsonl" in html
    assert 'TRACE_ROOT="outputs/paper/molecular_evidence_agent"' in starter


def test_trace_viewer_has_no_legacy_task_specific_rendering():
    html = VIEWER.read_text(encoding="utf-8")

    legacy_tokens = (
        "effect_on_bbb_reasoning",
        "effect_on_bioavailability_reasoning",
        "effect_on_clintox_reasoning",
        "effect_on_dili_reasoning",
        "direct_label_votes",
        "blocked_numeric_evidence_policy",
        "proxy_evidence_policy",
        "bioavailability-v2-final-prompt",
        "bbb_prediction",
        "bioavailability_prediction",
        "clintox_prediction",
        "skin_reaction_prediction",
    )
    for token in legacy_tokens:
        assert token not in html
