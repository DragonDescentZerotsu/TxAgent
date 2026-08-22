import json
from types import SimpleNamespace

import pytest

from tools.chembl_tool.common.task_workflows.reasoning_batch import (
    BatchConfig,
    BatchItem,
    PreparedBatch,
)
from tools.chembl_tool.common.task_workflows.reasoning_stage_runtime import (
    FINAL_STAGE,
    GROUP_STAGE,
    SINGLE_STAGE,
    _canonical_group_outputs,
    _execute_final,
    _hydrate_configured_branch_reuse,
    _invalidate_dependent_final,
    _merge_group_output,
    load_stage_state,
    prepare_stage_item,
    ready_stage_jobs,
    synchronize_configured_single_reuse,
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


def test_stage_dag_exposes_single_and_groups_before_final(tmp_path):
    prepared = _prepared(tmp_path)
    run_dir = prepared.batch_run_root / "condition_idx00000"
    run_dir.mkdir(parents=True)
    retrieval = {
        "status": "ok",
        "query": {"input_smiles": "CC"},
        "groups": [
            {"group_id": "Mechanism.a", "neighbors": [{"rank": 1}]},
            {"group_id": "Mechanism.b", "neighbors": [{"rank": 1}]},
        ],
    }
    (run_dir / "retrieval.json").write_text(json.dumps(retrieval), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps({"n_groups_with_neighbors": 2}),
        encoding="utf-8",
    )
    (run_dir / "single_molecule_reasoning_output.json").write_text(
        json.dumps({"status": "error"}),
        encoding="utf-8",
    )
    (run_dir / "group_reasoning_outputs.jsonl").write_text(
        json.dumps({"group_id": "Mechanism.a", "status": "ok"})
        + "\n"
        + json.dumps({"group_id": "Mechanism.b", "status": "error"})
        + "\n",
        encoding="utf-8",
    )

    state = load_stage_state(prepared, prepared.items[0])
    assert state is not None
    assert [(job.stage, job.group_id) for job in ready_stage_jobs(state)] == [
        (SINGLE_STAGE, ""),
        (GROUP_STAGE, "Mechanism.b"),
    ]

    (run_dir / "single_molecule_reasoning_output.json").write_text(
        json.dumps({"status": "ok"}),
        encoding="utf-8",
    )
    _merge_group_output(
        run_dir / "group_reasoning_outputs.jsonl",
        {"group_id": "Mechanism.b", "status": "ok"},
    )
    assert [(job.stage, job.group_id) for job in ready_stage_jobs(state)] == [
        (FINAL_STAGE, "")
    ]

    (run_dir / "final_reasoning_output.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "llm": {"content": {"prediction": "positive"}},
            }
        ),
        encoding="utf-8",
    )
    assert ready_stage_jobs(state) == []


def test_analog_stage_dag_omits_single_and_unlocks_final_after_groups(tmp_path):
    prepared = _prepared(tmp_path)
    prepared.args.analogous_reasoning_only = True
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
        json.dumps(
            {
                "n_groups_with_neighbors": 1,
                "analogous_reasoning_only": True,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "single_molecule_reasoning_output.json").write_text(
        json.dumps({"status": "omitted", "reason": "analogous_reasoning_only"}),
        encoding="utf-8",
    )

    state = load_stage_state(prepared, prepared.items[0])
    assert state is not None
    assert [(job.stage, job.group_id) for job in ready_stage_jobs(state)] == [
        (GROUP_STAGE, "Mechanism.a")
    ]

    (run_dir / "group_reasoning_outputs.jsonl").write_text(
        json.dumps({"group_id": "Mechanism.a", "status": "ok"}) + "\n",
        encoding="utf-8",
    )
    assert [(job.stage, job.group_id) for job in ready_stage_jobs(state)] == [
        (FINAL_STAGE, "")
    ]


def test_group_checkpoint_merge_replaces_only_matching_branch(tmp_path):
    path = tmp_path / "group_reasoning_outputs.jsonl"
    path.write_text(
        json.dumps({"group_id": "Mechanism.a", "status": "ok", "value": 1})
        + "\n"
        + json.dumps({"group_id": "Mechanism.b", "status": "error", "value": 2})
        + "\n",
        encoding="utf-8",
    )

    _merge_group_output(
        path,
        {"group_id": "Mechanism.b", "status": "ok", "value": 3},
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"group_id": "Mechanism.a", "status": "ok", "value": 1},
        {"group_id": "Mechanism.b", "status": "ok", "value": 3},
    ]


def test_final_group_inputs_keep_original_pipeline_sort_order(tmp_path):
    prepared = _prepared(tmp_path)
    run_dir = prepared.batch_run_root / "condition_idx00000"
    run_dir.mkdir(parents=True)
    (run_dir / "retrieval.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "groups": [
                    {"group_id": "Observed.z", "neighbors": [{"rank": 1}]},
                    {"group_id": "Mechanism.a", "neighbors": [{"rank": 1}]},
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"n_groups_with_neighbors": 2}),
        encoding="utf-8",
    )
    (run_dir / "group_reasoning_outputs.jsonl").write_text(
        json.dumps({"group_id": "Observed.z", "status": "ok"})
        + "\n"
        + json.dumps({"group_id": "Mechanism.a", "status": "ok"})
        + "\n",
        encoding="utf-8",
    )
    state = load_stage_state(prepared, prepared.items[0])

    assert state is not None
    assert [row["group_id"] for row in _canonical_group_outputs(state)] == [
        "Mechanism.a",
        "Observed.z",
    ]


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
        json.dumps({"n_groups_with_neighbors": 1}),
        encoding="utf-8",
    )
    state = load_stage_state(prepared, prepared.items[0])
    assert state is not None

    assert [(job.stage, job.group_id) for job in ready_stage_jobs(state)] == [
        (GROUP_STAGE, "Mechanism.a")
    ]

    source_run = source_batch / "runs" / "none_condition_idx00000"
    source_run.mkdir(parents=True)
    (source_run / "single_molecule_reasoning_output.json").write_text(
        json.dumps({"status": "ok", "analysis_id": "single_molecule"}),
        encoding="utf-8",
    )
    assert [(job.stage, job.group_id) for job in ready_stage_jobs(state)] == [
        (SINGLE_STAGE, ""),
        (GROUP_STAGE, "Mechanism.a"),
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


def test_hydrating_changed_frozen_single_invalidates_stale_final(tmp_path):
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

    _hydrate_configured_branch_reuse(prepared, prepared.items[0], {}, run_dir)

    assert json.loads(
        (run_dir / "single_molecule_reasoning_output.json").read_text(encoding="utf-8")
    )["analysis"] == "frozen"
    assert not (run_dir / "final_reasoning_output.json").exists()
    assert not (run_dir / "trace_messages.jsonl").exists()


def test_changed_frozen_single_invalidates_final_before_prepare_overwrites_it(tmp_path):
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

    synchronize_configured_single_reuse(
        prepared, prepared.items[0], run_dir
    )

    assert not (run_dir / "final_reasoning_output.json").exists()
    assert not (run_dir / "trace_messages.jsonl").exists()


def test_final_is_not_published_when_trace_serialization_fails(
    tmp_path,
    monkeypatch,
):
    prepared = _prepared(tmp_path)
    prepared.args.identity_blind = False
    prepared.args.save_trace = True
    run_dir = prepared.batch_run_root / "condition_idx00000"
    run_dir.mkdir(parents=True)
    retrieval = {"status": "ok", "query": {"input_smiles": "CC"}, "groups": []}
    (run_dir / "retrieval.json").write_text(json.dumps(retrieval), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps({"n_groups_with_neighbors": 0}),
        encoding="utf-8",
    )
    (run_dir / "single_molecule_reasoning_output.json").write_text(
        json.dumps({"status": "ok"}),
        encoding="utf-8",
    )
    (run_dir / "group_reasoning_outputs.jsonl").write_text("", encoding="utf-8")
    state = load_stage_state(prepared, prepared.items[0])
    assert state is not None

    def fail_trace(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("trace serialization failed")

    module = SimpleNamespace(
        _run_final_reasoning=lambda *args: {
            "status": "ok",
            "llm": {"content": {"prediction": "positive"}},
        },
        _write_trace_jsonl=fail_trace,
    )
    monkeypatch.setattr(
        "tools.chembl_tool.common.task_workflows.reasoning_stage_runtime._stage_context",
        lambda _state: {"module": module, "reasoning_retrieval": retrieval},
    )
    monkeypatch.setattr(
        "tools.chembl_tool.common.task_workflows.reasoning_stage_runtime._make_client",
        lambda _state: object(),
    )

    with pytest.raises(RuntimeError, match="trace serialization failed"):
        _execute_final(state)

    assert not (run_dir / "final_reasoning_output.json").exists()
    assert not (run_dir / "trace_messages.jsonl").exists()


def test_final_only_preparation_copies_checkpoints_and_enqueues_only_final(tmp_path):
    prepared = _prepared(tmp_path)
    source_batch = tmp_path / "source_condition"
    source_run = source_batch / "runs" / "source_condition_idx00000"
    source_run.mkdir(parents=True)
    retrieval = {
        "status": "ok",
        "query": {"input_smiles": "CC"},
        "groups": [
            {"group_id": "Mechanism.a", "neighbors": [{"rank": 1}]},
        ],
    }
    (source_run / "retrieval.json").write_text(json.dumps(retrieval), encoding="utf-8")
    (source_run / "single_molecule_reasoning_output.json").write_text(
        json.dumps({"status": "ok"}),
        encoding="utf-8",
    )
    (source_run / "group_reasoning_outputs.jsonl").write_text(
        json.dumps({"group_id": "Mechanism.a", "status": "ok"}) + "\n",
        encoding="utf-8",
    )
    (source_run / "manifest.json").write_text(
        json.dumps({"n_groups_with_neighbors": 1, "paths": {}}),
        encoding="utf-8",
    )
    prepared.logs_dir.mkdir(parents=True)
    prepared.args.skip_existing = False
    prepared.args.final_only_source_batch = str(source_batch)
    prepared.args.final_only_groups = None
    prepared.args.stream_logs = False

    result = prepare_stage_item(prepared, prepared.items[0])
    state = load_stage_state(prepared, prepared.items[0])

    assert result["status"] == "ok"
    assert state is not None
    assert [(job.stage, job.group_id) for job in ready_stage_jobs(state)] == [
        (FINAL_STAGE, "")
    ]
    target_manifest = json.loads(
        (state.run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert target_manifest["final_only_source_run_dir"] == str(source_run)
