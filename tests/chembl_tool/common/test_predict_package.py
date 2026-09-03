import json
from types import SimpleNamespace

from predict.llm_engine.endpoints import litellm_endpoint, openrouter_endpoint, vllm_endpoint
from predict.harnesses.branches import direct, flat, full
from predict.harnesses.branches import batch
from predict.harnesses.branches.retrieval import (
    retrieve_experiment_view as canonical_retrieve_experiment_view,
)
from predict.traces.io import input_prompt, write_trace
from predict.traces.viewer.build_dataset import build_dataset
from tools.chembl_tool.common.experiment_retrieval import (
    retrieve_experiment_view as compatible_retrieve_experiment_view,
)


def test_endpoint_profiles_share_transport_contract():
    local = vllm_endpoint(
        name="local", base_url="http://127.0.0.1:8000/v1", model="model", max_inflight=8
    )
    lite = litellm_endpoint(
        name="lite",
        base_url="https://litellm.example/v1",
        model="model",
        api_key_env="LITELLM_API_KEY",
        max_inflight=4,
    )
    router = openrouter_endpoint(
        name="router", model="org/model", api_key_env="OPENROUTER_API_KEY", max_inflight=2
    )
    assert local.api_key_env == ""
    assert lite.base_url == "https://litellm.example/v1"
    assert router.request_extra_body == {"reasoning": {"enabled": True}}


def test_historical_retrieval_import_is_the_canonical_implementation():
    assert compatible_retrieve_experiment_view is canonical_retrieve_experiment_view


def test_public_mode_harness_forces_existing_experiment_mode(monkeypatch):
    config = object()
    module = SimpleNamespace(CONFIG=config)
    captured = {}
    monkeypatch.setattr(batch.importlib, "import_module", lambda name: module)
    monkeypatch.setattr(
        batch,
        "main",
        lambda selected_config, argv: captured.update(config=selected_config, argv=argv) or 0,
    )
    assert batch.mode_main("full_flat", ["--task", "bbb_martins", "--limit", "1"]) == 0
    assert captured == {"config": config, "argv": ["--limit", "1", "--experiment-mode", "full_flat"]}


def test_public_harnesses_declare_their_evidence_plan(monkeypatch):
    captured = []

    def record(mode, argv):
        captured.append((mode, argv))
        return 0

    for module in (direct, flat, full):
        monkeypatch.setattr(module, "mode_main", record)
        assert module.main(["--task", "bbb_martins"]) == 0

    assert captured == [
        ("direct", ["--task", "bbb_martins"]),
        ("full_flat", ["--task", "bbb_martins"]),
        ("full_mechanism", ["--task", "bbb_martins"]),
    ]


def test_central_trace_and_viewer_adapter_cover_progressive(tmp_path):
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
        {"role": "assistant", "content": '{"bbb_prediction":"fail"}'},
    ]
    path = write_trace(
        trace_root=tmp_path / "traces",
        experiment_id="experiment",
        task="bbb_martins",
        harness="progressive",
        sample_id="query_idx00000",
        stage="level_01",
        checkpoint_path="outputs/example/output.json",
        output={
            "llm": {
                "messages": messages,
                "content": {"bbb_prediction": "fail", "confidence": "moderate"},
                "raw_content": '{"bbb_prediction":"fail"}',
                "usage": {"total_tokens": 3},
            }
        },
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["input_prompt"] == "[SYSTEM]\nsystem\n\n[USER]\nuser"
    assert input_prompt(messages) == payload["input_prompt"]

    output_root = tmp_path / "viewer"
    assert build_dataset(tmp_path / "traces", output_root) == 1
    predictions = (output_root / "runs/bbb_martins/experiment__progressive/predictions.jsonl")
    row = json.loads(predictions.read_text(encoding="utf-8"))
    assert row["pred_label"] == "fail"
    assert row["confidence"] == "moderate"
