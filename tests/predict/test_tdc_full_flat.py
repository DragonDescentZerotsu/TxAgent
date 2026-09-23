from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from predict.harnesses.branches import flat, matrix, runtime
from predict.harnesses.branches.tdc_query_priors import _matching_source
from predict.harnesses.branches.six_task_query_priors import _inputs as query_prior_inputs
from predict.retrieval.assay_reranking import build_tdc_indirect_ranked_retrieval
from predict.retrieval.assay_reranking.ranked_uid_retrieval import _payload
from predict.retrieval.assay_reranking.runtime import cache_profile_root
from predict.tasks.prompt_profiles import require_matching_prompt_profiles
from predict.utils.json import sha256_file


def test_upstream_v2_safety_final_writes_shared_trace(tmp_path: Path, monkeypatch) -> None:
    args = SimpleNamespace(
        flat_prompt_version=flat.CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_PROMPT_VERSION,
        flat_query_prior="cached", task_prompt_profile="test", flat_layout="grouped",
        flat_reranking="assay-transfer", save_trace=True, smiles_field="drug",
        identity_blind=False,
    )
    prepared = SimpleNamespace(
        args=args,
        config=SimpleNamespace(pipeline_module="predict.harnesses.branches.tasks.ames.pipeline"),
    )
    state = SimpleNamespace(
        prepared=prepared, item=SimpleNamespace(record={"drug": "CC", "Y": 1}, index=0),
        run_dir=tmp_path, retrieval={},
    )
    (tmp_path / "single_molecule_reasoning_output.json").write_text(
        json.dumps({"status": "ok", "llm": {"content": {}}})
    )
    monkeypatch.setattr(runtime, "attach_external_condition", lambda value, row: value)
    monkeypatch.setattr(runtime, "validated_branch_content", lambda value: {})
    monkeypatch.setattr(runtime, "build_flat_context_request", lambda *a, **k: (
        [{"role": "user", "content": "query"}], {"reasoning_reference_index": {}}
    ))
    monkeypatch.setattr(runtime, "flat_context_validation", lambda *a, **k: {})
    monkeypatch.setattr(runtime, "call_with_json_validation", lambda *a, **k: {
        "content": {"final_prediction": "pass"}, "messages": [],
    })
    monkeypatch.setattr(runtime, "structured_response_is_valid", lambda value: True)
    monkeypatch.setattr(runtime, "derive_flat_claim_evidence", lambda value: {})
    monkeypatch.setattr(runtime, "_record_stage_event", lambda *a: None)
    monkeypatch.setattr(runtime, "_write_stage_trace", lambda *a: None)

    runtime._execute_flat_context_final(state, SimpleNamespace(chat_json=lambda messages: {}))
    output = json.loads((tmp_path / "final_reasoning_output.json").read_text())
    assert output["status"] == "ok"
    assert output["prediction_mapping"]["native_prediction"] == "positive"
    trace = [json.loads(line) for line in (tmp_path / "trace_messages.jsonl").read_text().splitlines()]
    assert trace[-1]["prediction"] == "positive"


def test_upstream_v2_saved_safety_labels_remain_scorable() -> None:
    from predict.harnesses.branches.runner import prediction_to_label
    from predict.harnesses.branches.tasks.ames.contract import CONFIG as ames
    from predict.harnesses.branches.tasks.dili.contract import CONFIG as dili
    from predict.harnesses.branches.tasks.carcinogens.contract import CONFIG as carcinogens

    for config in (ames, dili, carcinogens):
        assert prediction_to_label(config, "pass") == 1
        assert prediction_to_label(config, "fail") == 0


def test_tdc_prior_reuse_requires_exact_query_smiles(tmp_path: Path) -> None:
    old = tmp_path / "old"
    old.mkdir()
    (old / "retrieval.json").write_text(json.dumps({
        "query": {"input_smiles": "N=CNC"}
    }))
    row = {"drug": "NC=NC", "molecule_identity_key": "K", "condition_group": ""}
    assert _matching_source(row, {}, {("K", ""): old}) is None
    row["drug"] = "N=CNC"
    assert _matching_source(row, {}, {("K", ""): old}) == ("tdc_existing", old)


def test_tdc_matrix_uses_tdc_split() -> None:
    args = matrix._selection_args(
        "bbb_martins", Path("/tmp/tdc-full-flat"), limit=0,
        context_v5=True, all_levels=True, benchmark="tdc",
    )
    assert args.benchmark == "tdc"
    assert args.input_jsonl == Path(
        "data/gold_labels/TDC/BBB_Martins/v1/scaffold/valid_molecule_condition_labels.jsonl"
    ).resolve()
    assert args.prompt_version == flat.CONTEXT_V5_PROMPT_VERSION


def test_six_task_query_priors_accept_tdc_valid_splits() -> None:
    for task, size in (("ames", 720), ("dili", 47), ("carcinogens", 27)):
        path, rows = query_prior_inputs(task, "valid", "tdc")
        assert "/gold_labels/TDC/" in str(path)
        assert len(rows) == size


def test_mixed_context_score_cache_is_active() -> None:
    assert "data/caches/assay_reranking/active" in str(cache_profile_root(
        "ranked_level_retrieval_tdc_v1_gold_v1_mixed_l1_assay_v10_3_best_v2"
    ))


def test_tdc_indirect_wrapper_selects_tdc_queries(monkeypatch) -> None:
    from predict.retrieval.assay_reranking import build_ranked_uid_retrieval as builder

    monkeypatch.setattr(builder, "QUERY_BENCHMARK", "tdc")
    monkeypatch.setattr(builder, "LABEL_RELEASE", {"benchmark": "tdc", "version": "v1"})
    monkeypatch.setattr(builder, "TASK_LEVELS", build_tdc_indirect_ranked_retrieval.LEVELS)
    path, rows, queries = builder._queries("bioavailability_ma", "test")
    assert "/gold_labels/TDC/Bioavailability_Ma/v1/scaffold/" in str(path)
    assert len(rows) == len(queries) == 128
    assert "L1" not in builder.TASK_LEVELS["bioavailability_ma"]


def test_branch_query_prior_overlay_resolves_hash_pinned_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    single = source / "single_molecule_reasoning_output.json"
    single.write_text(json.dumps({"status": "ok"}))
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps({
        "drug": "C", "molecule_identity_key": "K", "condition_group": "none"
    }) + "\n")
    batch = tmp_path / "overlay"
    batch.mkdir()
    (batch / "manifest.json").write_text(json.dumps({
        "schema_version": "branch_query_prior_overlay.v1",
        "input_jsonl": str(input_path), "input_sha256": sha256_file(input_path),
        "n_items": 1,
        "sources": [{
            "target_index": 0, "molecule_identity_key": "K",
            "condition_group": "none", "run_dir": str(source),
            "files_sha256": {single.name: sha256_file(single)},
        }],
    }))
    runtime._query_prior_overlay.cache_clear()
    assert runtime._source_run_dir(batch, 0) == source


def test_tdc_projection_rows_receive_flat_prompt_source_contract() -> None:
    result = _payload(
        {
            "source_row_uid": "train:1",
            "external_record_id": "train:1",
            "payload": json.dumps({
                "canonical_smiles": "CC",
                "label_source": "tdc_v1",
                "progressive_level": "L1",
                "source_fields": {"label": 1, "source_record_id": "train:1"},
            }),
        },
        {
            "parent_smiles": "CC",
            "parent_id": "PARENT",
            "morgan_similarity": 0.9,
            "morgan_rank": 1,
            "assay_rank": 1,
            "assay_transfer_score": 0.8,
        },
        "assay_transfer",
    )
    payload = result["payload"]
    assert payload["family_key"] == "tdc_training_labels"
    assert payload["source_id"] == "tdc_v1"
    assert payload["record_id"] == payload["source_row_uid"] == "train:1"
    assert payload["source_contract"]["source_or_simply_cleaned"] == {
        "label": True,
        "source_record_id": True,
    }


def test_query_prior_overlay_checks_nested_prompt_profiles(tmp_path: Path) -> None:
    run = tmp_path / "source"
    run.mkdir()
    source_manifest = run / "manifest.json"
    source_manifest.write_text(json.dumps({"task_prompt_profile": "profile-v2"}))
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "manifest.json").write_text(json.dumps({
        "schema_version": "branch_query_prior_overlay.v1",
        "task_prompt_profile": "profile-v2",
        "sources": [{
            "run_dir": str(run),
            "task_prompt_profile": "profile-v2",
            "prompt_profile_manifest": str(source_manifest),
            "prompt_profile_manifest_sha256": sha256_file(source_manifest),
        }],
    }))
    require_matching_prompt_profiles(
        target_profile="profile-v2",
        source_dirs=[overlay],
        historical_profile="legacy",
    )


def test_profileless_run_manifest_uses_enclosing_batch_profile(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    run = batch / "runs" / "query-0"
    run.mkdir(parents=True)
    (batch / "manifest.json").write_text(
        json.dumps({"task_prompt_profile": "profile-v2"})
    )
    (run / "manifest.json").write_text(json.dumps({"schema_version": "old-run.v1"}))

    require_matching_prompt_profiles(
        target_profile="profile-v2",
        source_dirs=[run],
        historical_profile="legacy",
    )
