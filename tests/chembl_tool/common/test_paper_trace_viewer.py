from pathlib import Path


VIEWER = Path("tools/trace_viewer/viewer.html")
STARTER = Path("tools/trace_viewer/start_viewer.sh")


def test_trace_viewer_targets_only_current_progressive_runs():
    html = VIEWER.read_text(encoding="utf-8")
    starter = STARTER.read_text(encoding="utf-8")

    assert "paper-v3" in html
    assert "Progressive Agent Trace Viewer" in html
    assert "current_conditioned_results.json" in starter
    assert ".result_families.progressive_append_only.scaffold" in starter
    assert ".progressive_tasks[$task].status" in starter
    assert "experiment_manifest.json" in starter
    assert "task=progressive-run-root" in starter

    for legacy_token in (
        "molecular_evidence_agent_starling_random",
        "molecular_evidence_agent_starling_scaffold",
        "runs_deployment_visible",
        "trace_messages.jsonl",
        "retrieval.json",
        "paper-v2",
    ):
        assert legacy_token not in html
        assert legacy_token not in starter


def test_trace_viewer_builds_level_trajectory_and_transition_summary():
    html = VIEWER.read_text(encoding="utf-8")

    required_tokens = (
        "none/predictions.jsonl",
        "levels/${id}/predictions.jsonl",
        "classifyTrajectory",
        "always_correct",
        "always_wrong",
        "rescued",
        "harmed",
        "oscillating",
        "stageTransitions",
        "Level metrics",
        "Rescued",
        "Harmed",
        "Pred flips",
        "Called / carried",
        "trajectory-node",
    )
    for token in required_tokens:
        assert token in html


def test_trace_viewer_loads_progressive_query_artifacts_on_demand():
    html = VIEWER.read_text(encoding="utf-8")

    required_tokens = (
        "queries/query_idx${padIndex(sample.index)}",
        "prepared.json",
        "request.json",
        "output.json",
        "level_definition",
        "tracePart",
        "Prompt",
        "complete request messages",
        "parseJsonText",
        "renderUserPrompt",
        "renderAnalogEvidence",
        "renderEvidenceCard",
        "Task definition",
        "Level context",
        "Active evidence",
        "Required output schema",
        "Prior state",
        "Raw message JSON",
        "Reasoning",
        "complete model reasoning",
        "Output",
        "structured model response",
        "structured_output_validation",
        "no model call at this stage",
        'fieldName.endsWith("card_ids")',
        "card-id-list",
        "card-id-chip",
    )
    for token in required_tokens:
        assert token in html

    for redundant_section in (
        "prepared_manifest.json",
        "Decision state · LLM aliases",
        "Evidence added at this level · LLM aliases",
        "Active cumulative evidence · LLM aliases",
        "Raw prepared / request / output artifacts",
        "aliasDisplay",
    ):
        assert redundant_section not in html


def test_trace_viewer_keeps_sample_sidebar_and_stacks_detail_fields_vertically():
    html = VIEWER.read_text(encoding="utf-8")

    assert '<aside>' in html
    assert 'id="samples"' in html
    assert "renderSampleList" in html
    assert ".layout { display: grid; grid-template-columns: 410px minmax(0, 1fr)" in html
    assert ".json-row { min-width: 0; display: block;" in html
    assert ".json-value { display: block;" in html
    assert ".prompt-payload { display: grid;" in html
    assert ".evidence-card-list { display: grid;" in html
    assert ".evidence-card-meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(125px, 1fr))" in html
    assert ".evidence-support { margin-top:" in html
    assert ".analog-summary-tags { display: inline-flex;" in html
    assert ".analog-structure { padding:" in html
    assert "Canonical SMILES" in html
    assert "first L" in html
    assert 'prior: not used' in html
    assert "Support text" in html
    assert "Run metadata" in html
    assert "Level metrics" in html


def test_trace_viewer_computes_generic_metrics_from_level_predictions():
    html = VIEWER.read_text(encoding="utf-8")

    assert "computeMetrics" in html
    assert "row.label" in html
    assert "row.pred_label" in html
    assert "macroF1" in html
    assert "computed from per-level predictions" in html

    task_specific_prediction_fields = (
        "bbb_prediction",
        "bioavailability_prediction",
        "skin_reaction_prediction",
    )
    for token in task_specific_prediction_fields:
        assert token not in html
