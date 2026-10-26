import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from predict.harnesses.branches.runner import (
    BatchConfig,
    BatchItem,
    PreparedBatch,
    _collect_result,
)
from predict.harnesses.branches.tasks.bioavailability_ma.contract import (
    CONFIG as BIOAVAILABILITY_CONFIG,
)
from predict.harnesses.branches.runtime import (
    FINAL_STAGE,
    GROUP_STAGE,
    SINGLE_STAGE,
    StageState,
    _execute_final,
    _execute_flat_context_final,
    _hydrate_configured_branch_reuse,
    _invalidate_dependent_final,
    load_stage_state,
    prepare_stage_item,
    ready_stage_jobs,
    synchronize_configured_single_reuse,
)
from predict.harnesses.branches.flat import (
    CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
)


def _prepared(tmp_path):
    config = BatchConfig(
        description="test",
        default_input="input.jsonl",
        default_batch_root="batch-root",
        default_index="index.pkl",
        default_model="model",
        batch_id_prefix="batch",
        pipeline_module="test.pipeline",
        log_prefix="test",
        report_title="test",
        prediction_field="prediction",
        canonical_positive="positive",
        canonical_negative="negative",
        positive_predictions=frozenset({"positive"}),
        negative_predictions=frozenset({"negative"}),
    )
    item = BatchItem(0, {"drug": "CC", "Y": 1})
    return PreparedBatch(
        config=config,
        args=SimpleNamespace(
            final_only_source_batch="",
            label_field="Y",
            smiles_field="drug",
        ),
        batch_id="condition",
        batch_dir=tmp_path / "condition",
        logs_dir=tmp_path / "condition" / "logs",
        batch_run_root=tmp_path / "condition" / "runs",
        items=[item],
        manifest={},
    )






def test_flat_context_one_call_skips_single_and_group_stages(tmp_path):
    prepared = _prepared(tmp_path)
    prepared.args.experiment_mode = "full_flat"
    prepared.args.flat_prompt_version = "evidence_flat_context_v4_v1"
    prepared.args.flat_query_prior = "none"
    run_dir = prepared.batch_run_root / "condition_idx00000"
    run_dir.mkdir(parents=True)
    (run_dir / "retrieval.json").write_text(
        json.dumps({
            "status": "ok",
            "groups": [{
                "group_id": "Flat.all_evidence",
                "neighbors": [{"rank": 1}],
            }],
        }),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"n_groups_with_neighbors": 0}), encoding="utf-8"
    )
    (run_dir / "single_molecule_reasoning_output.json").write_text(
        json.dumps({"status": "omitted", "reason": "query_prior_disabled"}),
        encoding="utf-8",
    )
    (run_dir / "group_reasoning_outputs.jsonl").write_text("", encoding="utf-8")

    state = load_stage_state(prepared, prepared.items[0])

    assert state is not None
    assert state.expected_group_ids == ()
    assert [(job.stage, job.group_id) for job in ready_stage_jobs(state)] == [
        (FINAL_STAGE, "")
    ]


def test_flat_context_common_prediction_maps_to_task_metrics(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "final_reasoning_output.json").write_text(
        json.dumps({
            "status": "ok",
            "llm": {"content": {
                "final_prediction": "pass",
                "summary": "Threshold is more likely met.",
                "confidence": "moderate",
            }},
        }),
        encoding="utf-8",
    )
    (run_dir / "single_molecule_reasoning_output.json").write_text(
        json.dumps({"status": "omitted"}), encoding="utf-8"
    )
    (run_dir / "group_reasoning_outputs.jsonl").write_text("", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps({"flat_one_call": True, "flat_query_prior": "none"}),
        encoding="utf-8",
    )
    item = BatchItem(0, {"drug": "CC", "Y": 1})
    args = SimpleNamespace(label_field="Y", smiles_field="drug")

    result = _collect_result(
        BIOAVAILABILITY_CONFIG, args, item, "run", run_dir
    )

    assert result["bioavailability_prediction"] == "high"
    assert result["pred_label"] == 1
    assert result["final_summary"] == "Threshold is more likely met."


def test_frozen_single_dependency_wakes_per_sample_without_phase_barrier(tmp_path):
    prepared = _prepared(tmp_path)
    source_batch = tmp_path / "none_condition"
    prepared.args.single_analysis_source_batch = str(source_batch)
    run_dir = prepared.batch_run_root / "condition_idx00000"
    run_dir.mkdir(parents=True)
    (run_dir / "retrieval.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "groups": [
                    {"group_id": "Mechanism.a", "neighbors": [{"rank": 1}]}
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"n_groups_with_neighbors": 0}),
        encoding="utf-8",
    )
    state = load_stage_state(prepared, prepared.items[0])
    assert state is not None

    assert ready_stage_jobs(state) == []

    source_run = source_batch / "runs" / "none_condition_idx00000"
    source_run.mkdir(parents=True)
    (source_run / "single_molecule_reasoning_output.json").write_text(
        json.dumps({"status": "ok", "analysis_id": "single_molecule"}),
        encoding="utf-8",
    )
    assert [(job.stage, job.group_id) for job in ready_stage_jobs(state)] == [
        (SINGLE_STAGE, ""),
    ]


def test_repaired_prerequisite_invalidates_stale_final_and_trace(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "final_reasoning_output.json").write_text(
        json.dumps({"status": "ok"}),
        encoding="utf-8",
    )
    (run_dir / "trace_messages.jsonl").write_text("{}\n", encoding="utf-8")

    _invalidate_dependent_final(run_dir)

    assert not (run_dir / "final_reasoning_output.json").exists()
    assert not (run_dir / "trace_messages.jsonl").exists()


def test_hydrating_changed_frozen_single_raises_without_modifying_outputs(tmp_path):
    prepared = _prepared(tmp_path)
    source_batch = tmp_path / "source"
    prepared.args.single_analysis_source_batch = str(source_batch)
    prepared.args.group_analysis_source_batch = ""
    prepared.args.neighbor_context_profile = "standard"
    source_run = source_batch / "runs" / "source_idx00000"
    source_run.mkdir(parents=True)
    (source_run / "single_molecule_reasoning_output.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "analysis": "frozen",
                "llm": {
                    "content": {"reasoning_summary": "frozen"},
                    "structured_output_validation": {"valid": True},
                },
            }
        ),
        encoding="utf-8",
    )
    run_dir = prepared.batch_run_root / "condition_idx00000"
    run_dir.mkdir(parents=True)
    (run_dir / "single_molecule_reasoning_output.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "analysis": "stale",
                "llm": {
                    "content": {"reasoning_summary": "stale"},
                    "structured_output_validation": {"valid": True},
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "final_reasoning_output.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )
    (run_dir / "trace_messages.jsonl").write_text("{}\n", encoding="utf-8")

    before = {path: path.read_bytes() for path in run_dir.iterdir()}
    with pytest.raises(ValueError, match="Query prior changed"):
        _hydrate_configured_branch_reuse(prepared, prepared.items[0], {}, run_dir)
    assert {path: path.read_bytes() for path in run_dir.iterdir()} == before


def test_changed_frozen_single_raises_before_prepare_overwrites_it(tmp_path):
    prepared = _prepared(tmp_path)
    source_batch = tmp_path / "source"
    prepared.args.single_analysis_source_batch = str(source_batch)
    source_run = source_batch / "runs" / "source_idx00000"
    source_run.mkdir(parents=True)
    (source_run / "single_molecule_reasoning_output.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "llm": {
                    "content": {"reasoning_summary": "frozen"},
                    "structured_output_validation": {"valid": True},
                },
            }
        ),
        encoding="utf-8",
    )
    run_dir = prepared.batch_run_root / "condition_idx00000"
    run_dir.mkdir(parents=True)
    (run_dir / "single_molecule_reasoning_output.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "llm": {
                    "content": {"reasoning_summary": "stale"},
                    "structured_output_validation": {"valid": True},
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "final_reasoning_output.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )
    (run_dir / "trace_messages.jsonl").write_text("{}\n", encoding="utf-8")

    before = {path: path.read_bytes() for path in run_dir.iterdir()}
    with pytest.raises(ValueError, match="Query prior changed"):
        synchronize_configured_single_reuse(prepared, prepared.items[0], run_dir)
    assert {path: path.read_bytes() for path in run_dir.iterdir()} == before




@pytest.mark.parametrize("trace_fails", [False, True])
def test_six_task_flat_trace_uses_shared_serializer(tmp_path, monkeypatch, trace_fails):
    prepared = _prepared(tmp_path)
    prepared.config = replace(
        prepared.config,
        pipeline_module="predict.harnesses.branches.tasks.ames.pipeline",
    )
    prepared.args.flat_prompt_version = CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION
    prepared.args.flat_query_prior = "none"
    prepared.args.task_prompt_profile = "ames_v1"
    prepared.args.flat_layout = "level-grouped"
    prepared.args.flat_reranking = "assay-transfer-contrastive"
    prepared.args.save_trace = True
    run_dir = prepared.batch_run_root / "condition_idx00000"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "single_molecule_reasoning_output.json").write_text(
        json.dumps({"status": "omitted"}), encoding="utf-8"
    )
    state = StageState(
        prepared,
        prepared.items[0],
        "condition_idx00000",
        run_dir,
        {"status": "ok"},
        (),
    )
    monkeypatch.setattr(
        "predict.harnesses.branches.runtime.build_flat_context_request",
        lambda *args, **kwargs: (
            [{"role": "user", "content": "predict"}],
            {"reasoning_reference_index": []},
        ),
    )
    monkeypatch.setattr(
        "predict.harnesses.branches.runtime.flat_context_validation",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "predict.harnesses.branches.runtime.call_with_json_validation",
        lambda *args, **kwargs: {
            "content": {"final_prediction": "pass", "claims": []},
            "structured_output_validation": {"valid": True},
        },
    )
    monkeypatch.setattr(
        "predict.harnesses.branches.runtime.native_flat_prediction",
        lambda *args, **kwargs: ("positive", "mapping-hash"),
    )
    monkeypatch.setattr(
        "predict.harnesses.branches.runtime._write_stage_trace",
        lambda *args, **kwargs: None,
    )

    if trace_fails:
        def fail_trace(*args, **kwargs):
            raise OSError("trace write failed")

        monkeypatch.setattr("predict.harnesses.branches.runtime.write_trace_jsonl", fail_trace)
        with pytest.raises(OSError, match="trace write failed"):
            _execute_flat_context_final(state, SimpleNamespace(chat_json=lambda messages: messages))
        assert not (run_dir / "final_reasoning_output.json").exists()
        assert not (run_dir / "trace_messages.jsonl").exists()
        return

    output = _execute_flat_context_final(
        state, SimpleNamespace(chat_json=lambda messages: messages)
    )

    trace = [
        json.loads(line)
        for line in (run_dir / "trace_messages.jsonl").read_text().splitlines()
    ]
    assert output["status"] == "ok"
    assert trace[-1]["prediction"] == "positive"



@pytest.mark.parametrize("benchmark_field", ["benchmark", "benchmark_release"])
def test_named_trace_identity_separates_benchmark_and_split(tmp_path, benchmark_field):
    from predict.harnesses.branches.runtime import _write_stage_trace
    from predict.traces.viewer.build_dataset import build_dataset
    prepared = _prepared(tmp_path)
    prepared.args.run_name = "same-study"
    prepared.args.flat_prompt_version = "final_full_flat"
    prepared.args.experiment_mode = "full_flat"
    prepared.args.trace_root = tmp_path / "traces"
    checkpoint = tmp_path / "final_reasoning_output.json"
    output = {"status": "ok", "llm": {"messages": [{"role": "user", "content": "query"}],
                                         "content": {"final_prediction": "pass"}}}
    checkpoint.write_text(json.dumps(output))
    state = SimpleNamespace(prepared=prepared, item=prepared.items[0])
    for benchmark, subset in (("gold_v1", "valid"), ("gold_v1", "test"), ("tdc", "test")):
        prepared.manifest = {benchmark_field: benchmark, "evaluation_subset": subset}
        _write_stage_trace(state, "final", output, checkpoint)
    viewer = tmp_path / "viewer"
    assert build_dataset(prepared.args.trace_root, viewer) == 3
    manifests = list(viewer.glob("runs/*/*/manifest.json"))
    assert len(manifests) == 3
    assert all(json.loads(p.read_text())["n_samples"] == 1 for p in manifests)
