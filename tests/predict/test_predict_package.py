import json
from types import SimpleNamespace

from predict.harnesses.branches import flat
from predict.harnesses.branches import runner
from predict.harnesses.branches import __main__ as branches_cli
from predict.harnesses.branches.retrieval import (
    retrieve_experiment_view as canonical_retrieve_experiment_view,
)
from predict.traces.io import input_prompt, write_trace
from predict.traces.viewer.build_dataset import build_dataset
from predict.harnesses.branches.retrieval import retrieve_experiment_view as compatible_retrieve_experiment_view


def test_public_mode_harness_forces_existing_experiment_mode(monkeypatch):
    config = object()
    module = SimpleNamespace(CONFIG=config)
    captured = {}
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: module)
    monkeypatch.setattr(
        runner,
        "main",
        lambda selected_config, argv, **kwargs: captured.update(config=selected_config, argv=argv) or 0,
    )
    assert runner.mode_main("full_flat", ["--task", "bbb_martins", "--limit", "1"]) == 0
    assert captured == {"config": config, "argv": ["--limit", "1", "--experiment-mode", "full_flat"]}


def test_public_harness_cli_selects_the_evidence_plan(monkeypatch):
    captured = []

    def record(mode, argv):
        captured.append((mode, argv))
        return 0

    monkeypatch.setattr(runner, "mode_main", record)
    monkeypatch.setattr(branches_cli, "run", lambda argv: record("full_flat", argv))
    assert branches_cli.main(["--task", "bbb_martins"]) == 0

    assert captured == [
        ("full_flat", ["--task", "bbb_martins"]),
    ]


def test_central_trace_and_viewer_adapter_cover_progressive(tmp_path, monkeypatch):
    monkeypatch.setattr("predict.live.record_trace", lambda *args: None)
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
        run_id="test-run",
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


def test_viewer_reads_named_run_traces_without_reading_query_snapshots(tmp_path):
    from predict.traces.catalog import discover_traces
    root = tmp_path/'flat/final_full_flat/semantic-top50'
    samples = root/'traces/ames/flat/run/samples/query_idx00000'
    samples.mkdir(parents=True)
    trace = {'schema_version':'predict_trace.v1','experiment_id':'semantic-top50',
             'task':'ames','harness':'flat','sample_id':'query_idx00000','stage':'final',
             'llm':{'content':{'final_prediction':'positive'},'messages':[]}}
    (samples/'final.json').write_text(json.dumps(trace))
    batch = root/'gold_v1/ames/test/default'
    (batch/'runs/query').mkdir(parents=True)
    (batch/'manifest.json').write_text('{}')
    (batch/'runs/query/retrieval.json').write_text('not a trace')
    assert len(discover_traces(tmp_path)) == 1
    assert build_dataset([tmp_path],tmp_path/'viewer') == 1
