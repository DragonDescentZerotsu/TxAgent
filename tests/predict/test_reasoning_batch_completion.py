import json
import importlib
from pathlib import Path

import pytest

from predict.harnesses.branches.runner import BatchConfig, _parse_args, _result_is_complete, _single_run_command, compute_metrics
from predict.harnesses.branches.tasks.bbb_martins.contract import CONFIG as BBB_CONFIG
from predict.harnesses.branches.tasks.bioavailability_ma.contract import CONFIG as BIOAVAILABILITY_CONFIG
from predict.harnesses.branches.tasks.dili.contract import CONFIG as DILI_CONFIG
from predict.harnesses.branches.tasks.skin_reaction.contract import CONFIG as SKIN_CONFIG


def test_result_completion_requires_all_reasoning_stages():
    complete = {
        "final_status": "ok",
        "pred_label": 1,
        "single_status": "ok",
        "n_groups_with_neighbors": 2,
        "n_group_outputs": 2,
        "n_failed_group_outputs": 0,
    }

    assert _result_is_complete(complete)
    assert not _result_is_complete({**complete, "single_status": "error"})
    assert not _result_is_complete({**complete, "n_group_outputs": 1})
    assert not _result_is_complete({**complete, "n_failed_group_outputs": 1})
    assert not _result_is_complete({**complete, "pred_label": None})


def test_result_completion_accepts_a_retrieval_free_run():
    assert _result_is_complete(
        {
            "final_status": "ok",
            "pred_label": 0,
            "single_status": "ok",
            "n_groups_with_neighbors": 0,
            "n_group_outputs": 0,
            "n_failed_group_outputs": 0,
        }
    )




def test_result_completion_requires_the_exact_expected_group_ids():
    complete = {
        "final_status": "ok",
        "pred_label": 1,
        "single_status": "ok",
        "n_groups_with_neighbors": 2,
        "n_group_outputs": 2,
        "n_failed_group_outputs": 0,
        "expected_group_ids": ["Mechanism.a", "Mechanism.b"],
        "group_ids": ["Mechanism.a", "Mechanism.b"],
    }

    assert _result_is_complete(complete)
    assert not _result_is_complete(
        {**complete, "group_ids": ["Mechanism.a", "Mechanism.a"]}
    )
    assert not _result_is_complete(
        {**complete, "group_ids": ["Mechanism.a", "Mechanism.c"]}
    )


def test_full_batch_defaults_to_throughput_and_requires_a_parallelism_choice():
    args = _parse_args(BBB_CONFIG, [])

    assert args.execution_mode == "throughput"
    assert args.parallelism is None
    assert args.provider_pool_config.endswith("/providers/current_endpoints.json")


@pytest.mark.parametrize("config", (BBB_CONFIG, BIOAVAILABILITY_CONFIG, DILI_CONFIG, SKIN_CONFIG))
def test_prepare_command_materializes_cached_retrieval_without_a_model(config, tmp_path):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps({"drug": "CCO", "Y": 1}) + "\n")
    source = tmp_path / "source"
    run = source / "runs" / "source_idx00000"
    run.mkdir(parents=True)
    retrieval = {"status": "ok", "query": {"input_smiles": "CCO"}, "groups": []}
    (run / "retrieval.json").write_text(json.dumps(retrieval))
    args = _parse_args(config, ["--input-jsonl", str(input_path),
                                "--retrieval-replay-source-batch", str(source)])
    command = _single_run_command(config, args, 0, "query", tmp_path / "prepared")
    pipeline = importlib.import_module(config.pipeline_module)
    assert pipeline.main([*command[3:], "--prepare-only"]) == 0
    assert json.loads((tmp_path / "prepared/query/retrieval.json").read_text())["query"]["input_smiles"] == "CCO"


def test_native_query_prior_completes_without_a_final_prediction(tmp_path):
    from predict.harnesses.branches.tasks.flat_only_pipeline import main

    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps({"drug": "CCO"}) + "\n")
    assert main(["--input-jsonl", str(path), "--query-index", "0", "--run-id", "prior",
                 "--out-root", str(tmp_path), "--experiment-mode", "none", "--prepare-only"]) == 0
    retrieval = json.loads((tmp_path / "prior/retrieval.json").read_text())
    assert retrieval["groups"] == []
    assert _result_is_complete({"query_prior_only": True, "single_status": "ok"})
    assert not _result_is_complete({"query_prior_only": True, "single_status": "error"})

    metrics = compute_metrics(BBB_CONFIG, [
        {"query_prior_only": True, "label": 1, "status": "ok"},
        {"query_prior_only": True, "label": 0, "status": "error"},
    ])
    assert metrics["n_successful"] == 1
    assert metrics["n_failed_runs"] == 1
    assert metrics["n_evaluable"] == 0
    assert metrics["macro_f1"] is None


@pytest.mark.parametrize("task", ("bbb_martins", "bioavailability_ma", "skin_reaction", "ames", "dili", "carcinogens"))
def test_batch_preserves_final_prompt_provenance_and_rejects_changed_resume(tmp_path, monkeypatch, task):
    from predict.harnesses.branches import runner
    from predict.harnesses.branches.flat import prompt_assets
    config = importlib.import_module(f"predict.harnesses.branches.tasks.{task}.contract").CONFIG

    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps({"drug": "CCO", "Y": 1}) + "\n")
    args = _parse_args(config, [
        "--input-jsonl", str(path), "--batch-root", str(tmp_path), "--batch-id", "final",
        "--flat-prompt-version", "final_full_flat", "--flat-reranking", "morgan",
        "--flat-query-prior", "none", "--disable-flat-tools",
        "--retrieval-replay-source-batch", str(tmp_path / "source"),
        "--flat-selection-manifest", str(tmp_path / "selection.json"),
        config.prompt_profile_option, prompt_assets("final_full_flat")["tasks"][task]["prompt_profile"],
    ])
    args.requested_parallelism = args.parallelism
    args.endpoint_selection = None
    monkeypatch.setattr(runner, "_validated_flat_selection_manifest", lambda *a: {})
    prepared = runner.prepare_batch(config, args)
    provenance = prepared.manifest["group_prompt_provenance"]
    assert provenance["prompt_version"] == "final_full_flat"
    assert provenance["contract_sha256"]
    runner.prepare_batch(config, args)
    monkeypatch.setattr(runner, "flat_prompt_provenance", lambda *a, **k: {
        **provenance, "contract_sha256": "changed",
    })
    with pytest.raises(ValueError, match="provenance mismatch"):
        runner.prepare_batch(config, args)




