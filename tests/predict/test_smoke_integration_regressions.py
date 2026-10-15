"""Failures found while exercising the real cached prediction pipeline."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from predict.api_client.transport import DEFAULT_ENV_FILE
from predict.harnesses.branches import flat, runner, six_task_query_priors as priors
from predict.retrieval.assay_reranking.ranked_uid_retrieval import _payload


@pytest.mark.parametrize("task", tuple(flat.TASKS))
def test_final_launcher_defaults_to_shared_credentials_and_task_profile(task):
    config = importlib.import_module(f"predict.harnesses.branches.tasks.{task}.contract").CONFIG
    args = runner._parse_args(config, [
        "--experiment-mode", "full_flat", "--flat-prompt-version", "final_full_flat",
    ])
    assert Path(args.env_file) == DEFAULT_ENV_FILE
    assert args.task_prompt_profile == flat.prompt_assets("final_full_flat")["tasks"][task]["prompt_profile"]
    explicit = runner._parse_args(config, [
        "--experiment-mode", "full_flat", "--flat-prompt-version", "final_full_flat",
        config.prompt_profile_option, config.default_prompt_profile,
    ])
    assert explicit.task_prompt_profile == config.default_prompt_profile


def test_canonical_label_card_restores_only_its_known_source_contract():
    payload = dict(label_source="canonical_context_cards.v1", Y=1, source_fields={"label": 1})
    record = dict(source_row_uid="physical", external_record_id="external", payload=json.dumps(payload))
    rank = dict(morgan_similarity=0.5, morgan_rank=1, assay_rank=None, parent_smiles="CN", parent_id="parent")
    result = _payload(record, rank, "morgan")
    assert result["payload"]["source_contract"]["source_or_simply_cleaned"] == {"label": True}
    payload["source_fields"]["unapproved"] = "text"
    record["payload"] = json.dumps(payload)
    with pytest.raises(ValueError, match="invalid label contract"):
        _payload(record, rank, "morgan")


@pytest.mark.parametrize("task", tuple(flat.TASKS))
@pytest.mark.parametrize("label", (0, 1))
def test_final_collection_uses_prompt_label_contract(task, label, tmp_path, monkeypatch):
    config = importlib.import_module(f"predict.harnesses.branches.tasks.{task}.contract").CONFIG
    contract = flat.prompt_assets("final_full_flat")["tasks"][task]
    prediction = contract["positive_prediction" if label else "negative_prediction"]
    (tmp_path / "final_reasoning_output.json").write_text(json.dumps(dict(
        status="ok", llm=dict(content=dict(final_prediction=prediction)))))
    (tmp_path / "manifest.json").write_text("{}")
    monkeypatch.setattr(runner, "read_run", lambda path: dict(
        flat_one_call=True, group_prompt_version="final_full_flat"))
    result = runner._collect_result(config, SimpleNamespace(label_field="Y", smiles_field="drug"),
        runner.BatchItem(index=0, record=dict(Y=label, drug="CC")), "query", tmp_path)
    assert result["pred_label"] == label
    assert result["correct"]


@pytest.mark.parametrize("named", (False, True))
def test_saved_resume_uses_public_arguments_and_preserves_checkpoints(named, tmp_path, monkeypatch):
    from predict import live
    from predict.harnesses.branches import scheduler
    from predict.harnesses.branches.tasks.bbb_martins.contract import CONFIG

    captured = {}
    public = ["python", "-m", "predict.harnesses.branches", "--task", "bbb_martins"]
    if named:
        public += ["--run-name", "smoke"]
    prepared = SimpleNamespace(batch_id="smoke", batch_dir=tmp_path, items=[], manifest={})
    args = runner._parse_args(CONFIG, [])
    args.prepare_only = True
    monkeypatch.setattr(runner, "_parse_args", lambda *values: args)
    monkeypatch.setattr(runner, "prepare_batch", lambda *args: prepared)
    monkeypatch.setattr(live, "create_run", lambda **kwargs: tmp_path)
    monkeypatch.setattr(live, "update_run", lambda path, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(scheduler, "run_prepared_prompt_pool", lambda *args, **kwargs: 0)
    assert runner.main(CONFIG, ["--prepare-only"], launch_command=public) == 0
    command = captured["resume_command"]
    assert command[:len(public)] == public
    assert "--skip-existing" in command
    assert ("--batch-root" in command) != named
    assert "--organization" not in command


@pytest.mark.parametrize("task", tuple(flat.TASKS))
@pytest.mark.parametrize("outcome", ("valid", "empty", "missing_field", "invalid_label", "invalid_confidence"))
def test_query_prior_writer_emits_only_validated_reusable_records(tmp_path, monkeypatch, task, outcome):
    from unittest.mock import Mock

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / f"{task}.jsonl").write_text(json.dumps(dict(benchmark_row_id="q", drug="CC")) + "\n")
    profile = flat.DEFAULT_PROVIDER_CONFIG
    config = priors.load_provider_pool_config(profile)
    monkeypatch.setattr(priors, "select_healthy_providers", lambda *args: SimpleNamespace(
        config=config, effective_parallelism=1, public_dict=lambda: {},
    ))
    monkeypatch.setattr(priors, "ToolServiceClient", lambda *args, **kwargs: SimpleNamespace(
        invoke=lambda *args: {"status": "ok"},
    ))
    content = dict(endpoint_prior=priors._contract(task, flat.FINAL_FULL_FLAT_PROMPT_VERSION)["positive_prediction"],
                   confidence="moderate", reasoning_summary="Structural prior",
                   property_drivers=["Small molecule"], caveats=["No measured outcome"])
    if outcome == "empty":
        content = {}
    elif outcome == "missing_field":
        content.pop("caveats")
    elif outcome == "invalid_label":
        content["endpoint_prior"] = "invalid"
    elif outcome == "invalid_confidence":
        content["confidence"] = "certain"
    chat = Mock(return_value={"content": content})
    monkeypatch.setattr(priors, "provider_client", lambda *args, **kwargs: SimpleNamespace(
        chat_json=chat, snapshot=lambda: {},
    ))
    output = tmp_path / "priors"
    command = [
            "--output-root", str(output), "--run-name", "invalid-prior", "--subset", "fresh", "--input-root", str(inputs),
            "--tasks", task, "--provider-pool-config", str(profile),
            "--parallelism", "1", "--max-attempts", "1",
        ]
    if outcome == "valid":
        assert priors.main(command) == 0
        from predict.utils.response import validated_branch_content
        path = next(output.rglob("single_molecule_reasoning_output.json"))
        saved = json.loads(path.read_text())
        assert validated_branch_content(saved) == content
        assert priors._prior_source(path.parent) is not None
        from predict.utils import response
        def forbidden(*args, **kwargs):
            raise AssertionError("Saved priors must not be schema-validated again")
        monkeypatch.setattr(response, "response_validation_errors", forbidden)
        assert priors.main(command + ["--resume"]) == 0
        assert chat.call_count == 1
        saved["llm"].pop("structured_output_validation")
        path.write_text(json.dumps(saved))
        assert priors._prior_source(path.parent) is None
        saved["llm"]["structured_output_validation"] = {"valid": False}
        path.write_text(json.dumps(saved))
        assert priors._prior_source(path.parent) is None
        return
    with pytest.raises(RuntimeError, match="Query-prior recovery exhausted"):
        priors.main(command)
    execution = json.loads(next(output.rglob("execution.json")).read_text())
    assert execution["status"] == "failed"
    assert execution["prompt_version"] == flat.FINAL_FULL_FLAT_PROMPT_VERSION
    assert not list(output.rglob("single_molecule_reasoning_output.json"))
