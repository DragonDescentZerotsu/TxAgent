import argparse
import json
from types import SimpleNamespace

from tools.chembl_tool.common.task_workflows import global_prompt_pool as pool
from tools.chembl_tool.common.task_workflows import reasoning_stage_runtime as stages
from tools.chembl_tool.common.task_workflows.reasoning_batch import (
    BatchConfig,
    BatchItem,
    PreparedBatch,
)
from tools.chembl_tool.paper_experiments import starling_benchmark_matrix as matrix


def _config() -> BatchConfig:
    return BatchConfig(
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


def _command_args() -> argparse.Namespace:
    return argparse.Namespace(
        python_executable="python",
        input_jsonl="input.jsonl",
        smiles_field="drug",
        index="index.pkl",
        experiment_mode="full_mechanism",
        retrieval_source="starling",
        neighbor_identity_policy="parent_disjoint",
        neighbor_selector="similarity",
        neighbor_context_profile="standard",
        env_file=".env",
        api_key_env="TEST_KEY",
        base_url="http://localhost/v1",
        tool_service_url="http://localhost:8765",
        model="model",
        timeout_s=300,
        max_tokens=100,
        max_tool_rounds=3,
        reasoning_effort="",
        top_k_per_group=3,
        min_similarity=0.3,
        enable_thinking=False,
        max_groups=0,
        groups=None,
        tier1_replacement_index="",
        tier1_replacement_groups=None,
        disable_group_tools=False,
        identity_blind=False,
        harness_prefetch_tools=False,
        single_analysis_source_batch="",
        group_analysis_source_batch="",
        retrieval_replay_source_batch="",
        prefetched_tool_replay_source_batch="",
    )


def test_global_pool_requeues_only_failed_item(monkeypatch, tmp_path):
    prepared = PreparedBatch(
        config=_config(),
        args=SimpleNamespace(skip_existing=True),
        batch_id="condition",
        batch_dir=tmp_path / "condition",
        logs_dir=tmp_path / "condition" / "logs",
        batch_run_root=tmp_path / "condition" / "runs",
        items=[BatchItem(0, {"drug": "CC", "Y": 1})],
        manifest={},
    )
    calls = 0
    finalized = []
    state = pool.StageState(
        prepared=prepared,
        item=prepared.items[0],
        run_id="condition_idx00000",
        run_dir=prepared.batch_run_root / "condition_idx00000",
        retrieval={},
        expected_group_ids=(),
    )

    monkeypatch.setattr(pool, "_prepare_command", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(pool, "collect_completed_item", lambda *args: None)

    def fake_run(prepared_batch, item):
        nonlocal calls
        calls += 1
        return {"query_index": item.index, "status": "error" if calls == 1 else "ok"}

    monkeypatch.setattr(pool, "prepare_stage_item", fake_run)
    monkeypatch.setattr(pool, "_resolve_item_future", lambda prepared_batch, item, future: future.result())
    monkeypatch.setattr(
        pool,
        "load_stage_state",
        lambda *args: state if calls >= 2 else None,
    )
    monkeypatch.setattr(
        pool,
        "ready_stage_jobs",
        lambda current: [pool.StageJob(current, "final")],
    )
    monkeypatch.setattr(pool, "execute_stage", lambda job: {"status": "ok"})
    monkeypatch.setattr(
        pool,
        "collect_stage_result",
        lambda current: {"query_index": current.item.index, "status": "ok"},
    )

    def fake_finalize(prepared_batch, results):
        finalized.extend(results)
        return 0

    monkeypatch.setattr(pool, "finalize_batch", fake_finalize)

    failed = pool.run_global_prompt_pool(
        [pool.BatchCommand("condition", ["python", "-m", "test.batch"])],
        max_workers=4,
        max_stage_requeues=1,
    )

    assert failed == []
    assert calls == 2
    assert finalized == [{"query_index": 0, "status": "ok"}]


def test_matrix_global_pool_uses_one_full_budget_without_condition_barriers(monkeypatch):
    experiments = matrix.experiments_for_starling_benchmark("random")
    selected_names = {
        "bbb_martins__none",
        "skin_reaction__none",
        "bbb_martins__chembl_direct",
        "skin_reaction__chembl_direct",
    }
    selected = [item for item in experiments if item.name in selected_names]
    calls = []

    monkeypatch.setattr(
        matrix,
        "_command",
        lambda experiment, args: ["python", "-m", f"batch.{experiment.name}"],
    )

    def fake_pool(commands, *, max_workers, max_stage_requeues):
        calls.append(
            {
                "names": [item.experiment_name for item in commands],
                "max_workers": max_workers,
                "max_stage_requeues": max_stage_requeues,
            }
        )
        return []

    monkeypatch.setattr(matrix, "run_global_prompt_pool", fake_pool)
    args = SimpleNamespace(
        parallelism=128,
        max_stage_requeues=2,
    )

    assert matrix._run_selected_experiments(selected, args) == []
    assert calls == [
        {
            "names": [
                "bbb_martins__none",
                "bbb_martins__chembl_direct",
                "skin_reaction__none",
                "skin_reaction__chembl_direct",
            ],
            "max_workers": 128,
            "max_stage_requeues": 2,
        }
    ]


def test_global_pool_enqueues_final_when_group_dependency_completes(
    monkeypatch,
    tmp_path,
):
    prepared = PreparedBatch(
        config=_config(),
        args=SimpleNamespace(skip_existing=True, final_only_source_batch=""),
        batch_id="condition",
        batch_dir=tmp_path / "condition",
        logs_dir=tmp_path / "condition" / "logs",
        batch_run_root=tmp_path / "condition" / "runs",
        items=[BatchItem(0, {"drug": "CC", "Y": 1})],
        manifest={},
    )
    state = pool.StageState(
        prepared=prepared,
        item=prepared.items[0],
        run_id="condition_idx00000",
        run_dir=prepared.batch_run_root / "condition_idx00000",
        retrieval={},
        expected_group_ids=("Mechanism.a",),
    )
    state.step = 0
    executed = []
    finalized = []

    monkeypatch.setattr(pool, "_prepare_command", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(pool, "collect_completed_item", lambda *args: None)
    monkeypatch.setattr(pool, "load_stage_state", lambda *args: state)

    def fake_ready(current):
        if current.step == 0:
            return [pool.StageJob(current, "group", "Mechanism.a")]
        if current.step == 1:
            return [pool.StageJob(current, "final")]
        return []

    def fake_execute(job):
        executed.append((job.stage, job.group_id))
        job.state.step += 1
        return {"status": "ok"}

    monkeypatch.setattr(pool, "ready_stage_jobs", fake_ready)
    monkeypatch.setattr(pool, "execute_stage", fake_execute)
    monkeypatch.setattr(
        pool,
        "collect_stage_result",
        lambda current: {"query_index": current.item.index, "status": "ok"},
    )

    def fake_finalize(prepared_batch, results):
        finalized.extend(results)
        return 0

    monkeypatch.setattr(pool, "finalize_batch", fake_finalize)

    failed = pool.run_global_prompt_pool(
        [pool.BatchCommand("condition", ["python", "-m", "test.batch"])],
        max_workers=128,
    )

    assert failed == []
    assert executed == [("group", "Mechanism.a"), ("final", "")]
    assert finalized == [{"query_index": 0, "status": "ok"}]


def test_fresh_pool_item_runs_retrieval_only_before_prompt_stages(
    monkeypatch,
    tmp_path,
):
    prepared = PreparedBatch(
        config=_config(),
        args=SimpleNamespace(stream_logs=False),
        batch_id="condition",
        batch_dir=tmp_path / "condition",
        logs_dir=tmp_path / "condition" / "logs",
        batch_run_root=tmp_path / "condition" / "runs",
        items=[BatchItem(0, {"drug": "CC", "Y": 1})],
        manifest={},
    )
    prepared.logs_dir.mkdir(parents=True)
    command_seen = []
    initialized = []
    monkeypatch.setattr(
        stages,
        "_single_run_command",
        lambda *args, **kwargs: ["python", "-m", "test.pipeline"],
    )

    def fake_subprocess(command, **kwargs):
        command_seen.extend(command)
        return 0

    monkeypatch.setattr(stages, "_run_subprocess_with_logs", fake_subprocess)
    monkeypatch.setattr(
        stages,
        "_initialize_run_manifest",
        lambda *args: initialized.append(args[2]),
    )

    result = stages.prepare_stage_item(prepared, prepared.items[0])

    assert result["status"] == "ok"
    assert command_seen[-1] == "--prepare-only"
    assert initialized == ["condition_idx00000"]


def test_prepare_manifest_keeps_canonical_per_run_paths(monkeypatch, tmp_path):
    args = _command_args()
    args.input_jsonl = "input.jsonl"
    args.label_field = "Y"
    args.max_groups = 0
    args.temperature = 0.0
    args.save_trace = True
    args.stream_logs = False
    prepared = PreparedBatch(
        config=BatchConfig(
            **{
                **_config().__dict__,
                "pipeline_module": (
                    "tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline"
                ),
            }
        ),
        args=args,
        batch_id="condition",
        batch_dir=tmp_path / "condition",
        logs_dir=tmp_path / "condition" / "logs",
        batch_run_root=tmp_path / "condition" / "runs",
        items=[BatchItem(0, {"drug": "CC", "Y": 1})],
        manifest={},
    )
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
    monkeypatch.setattr(stages, "_hydrate_configured_branch_reuse", lambda *args: None)

    stages._initialize_run_manifest(
        prepared,
        prepared.items[0],
        "condition_idx00000",
        run_dir,
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_groups_with_neighbors"] == 1
    assert manifest["query_label_for_eval_only"] == 1
    assert manifest["paths"] == {
        "retrieval": str(run_dir / "retrieval.json"),
        "single_molecule_reasoning_output": str(
            run_dir / "single_molecule_reasoning_output.json"
        ),
        "group_reasoning_outputs": str(run_dir / "group_reasoning_outputs.jsonl"),
        "group_reasoning_outputs_raw": "",
        "final_reasoning_output": str(run_dir / "final_reasoning_output.json"),
        "trace_messages": str(run_dir / "trace_messages.jsonl"),
    }
    assert manifest["stage_pool"]["prepared_without_llm"] is True


def test_prepare_manifest_materializes_empty_group_artifact(monkeypatch, tmp_path):
    args = _command_args()
    args.input_jsonl = "input.jsonl"
    args.label_field = "Y"
    args.max_groups = 0
    args.temperature = 0.0
    args.save_trace = True
    args.stream_logs = False
    prepared = PreparedBatch(
        config=BatchConfig(
            **{
                **_config().__dict__,
                "pipeline_module": (
                    "tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline"
                ),
            }
        ),
        args=args,
        batch_id="condition",
        batch_dir=tmp_path / "condition",
        logs_dir=tmp_path / "condition" / "logs",
        batch_run_root=tmp_path / "condition" / "runs",
        items=[BatchItem(0, {"drug": "CC", "Y": 1})],
        manifest={},
    )
    run_dir = prepared.batch_run_root / "condition_idx00000"
    run_dir.mkdir(parents=True)
    (run_dir / "retrieval.json").write_text(
        json.dumps({"status": "ok", "groups": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(stages, "_hydrate_configured_branch_reuse", lambda *args: None)

    stages._initialize_run_manifest(
        prepared,
        prepared.items[0],
        "condition_idx00000",
        run_dir,
    )

    assert (run_dir / "group_reasoning_outputs.jsonl").exists()
    assert (run_dir / "group_reasoning_outputs.jsonl").read_text() == ""
