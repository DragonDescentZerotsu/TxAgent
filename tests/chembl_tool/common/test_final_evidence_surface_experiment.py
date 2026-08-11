import os
from types import SimpleNamespace

from tools.chembl_tool.common.final_evidence_surface import SUMMARY_PLUS_CARDS
from tools.chembl_tool.paper_experiments.audit_final_evidence_surface_contract import (
    _audit_condition,
)
from tools.chembl_tool.paper_experiments.run_final_evidence_surface_experiment import (
    _apply_and_validate_runtime_contract,
    _batch_command,
    _ensure_endpoint_api_key,
)
from tools.chembl_tool.tasks.skin_reaction.prompt_profiles import (
    LEGACY_SKIN_REACTION_V1,
)
from tools.chembl_tool.tasks.bioavailability_ma.prompt_profiles import (
    LEGACY_BIOAVAILABILITY_V1,
)


def test_final_surface_command_is_final_only_and_isolated(tmp_path):
    experiment = SimpleNamespace(
        name="bbb_martins__starling_full_flat",
        batch_module="tools.chembl_tool.tasks.bbb_martins.run_reasoning_batch",
        input_jsonl="valid.jsonl",
        index="index.pkl",
        mode="full_flat",
        source="starling",
    )
    args = SimpleNamespace(
        source_run_root=tmp_path / "source",
        output_root=tmp_path / "output",
        python_executable="python",
        api_key_env="KEY",
        base_url="http://localhost:50000/v1",
        model="model",
        reasoning_effort="",
        timeout_s=300,
        parallelism=8,
        limit=1,
        indices=None,
    )
    command = _batch_command(
        experiment,
        "bbb_martins",
        SUMMARY_PLUS_CARDS,
        args,
    )
    joined = " ".join(str(value) for value in command)
    assert "--final-only-source-batch" in command
    assert "--final-evidence-surface summary_plus_cards" in joined
    assert "--identity-blind" in command
    assert "--neighbor-identity-policy parent_disjoint" in joined
    assert str(tmp_path / "output" / "runs_identity_blind_parent_disjoint") in joined


def test_runtime_contract_is_derived_from_frozen_source():
    args = SimpleNamespace(model=None, base_url=None, reasoning_effort=None)
    contract = {
        "model": "gpt-oss-120b",
        "base_url": "http://127.0.0.1:9001/v1",
        "reasoning_effort": "",
    }
    _apply_and_validate_runtime_contract(args, contract)
    assert args.model == contract["model"]
    assert args.base_url == contract["base_url"]
    assert args.reasoning_effort == contract["reasoning_effort"]


def test_skin_final_surface_replay_pins_legacy_prompt_profile(tmp_path):
    experiment = SimpleNamespace(
        name="skin_reaction__starling_full_flat",
        batch_module="tools.chembl_tool.tasks.skin_reaction.run_reasoning_batch",
        input_jsonl="valid.jsonl",
        index="index.pkl",
        mode="full_flat",
        source="starling",
    )
    args = SimpleNamespace(
        source_run_root=tmp_path / "source",
        output_root=tmp_path / "output",
        python_executable="python",
        api_key_env="KEY",
        base_url="http://localhost:50000/v1",
        model="model",
        reasoning_effort="",
        timeout_s=300,
        parallelism=8,
        limit=0,
        indices=None,
    )

    command = _batch_command(experiment, "skin_reaction", SUMMARY_PLUS_CARDS, args)

    position = command.index("--skin-prompt-profile")
    assert command[position + 1] == LEGACY_SKIN_REACTION_V1


def test_bio_final_surface_replay_pins_legacy_prompt_profile(tmp_path):
    experiment = SimpleNamespace(
        name="bioavailability_ma__starling_full_flat",
        batch_module="tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch",
        input_jsonl="valid.jsonl",
        index="index.pkl",
        mode="full_flat",
        source="starling",
    )
    args = SimpleNamespace(
        source_run_root=tmp_path / "source",
        output_root=tmp_path / "output",
        python_executable="python",
        api_key_env="KEY",
        base_url="http://localhost:50000/v1",
        model="model",
        reasoning_effort="",
        timeout_s=300,
        parallelism=8,
        limit=0,
        indices=None,
    )

    command = _batch_command(
        experiment,
        "bioavailability_ma",
        SUMMARY_PLUS_CARDS,
        args,
    )

    position = command.index("--bioavailability-prompt-profile")
    assert command[position + 1] == LEGACY_BIOAVAILABILITY_V1


def test_local_gpt_oss_key_uses_served_endpoint_placeholder(monkeypatch):
    monkeypatch.delenv("GPT_OSS_LOCAL_API_KEY", raising=False)
    _ensure_endpoint_api_key(
        "GPT_OSS_LOCAL_API_KEY",
        "http://127.0.0.1:9001/v1",
        "gpt-oss-120b",
    )
    assert os.environ["GPT_OSS_LOCAL_API_KEY"] == "EMPTY"


def test_contract_audit_rejects_an_empty_target_batch(tmp_path):
    batch = tmp_path / "target"
    batch.mkdir()
    rows, failures = _audit_condition(
        "bbb_martins",
        SUMMARY_PLUS_CARDS,
        batch,
        tmp_path / "source",
    )
    assert rows == []
    assert failures == [
        {
            "task": "bbb_martins",
            "surface": SUMMARY_PLUS_CARDS,
            "error": "incomplete_target_runs",
            "n_runs": 0,
            "expected_runs": 500,
        }
    ]
