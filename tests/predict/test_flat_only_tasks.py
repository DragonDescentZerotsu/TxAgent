import json

import pytest

from predict.harnesses.branches.tasks.flat_only_pipeline import main


@pytest.mark.parametrize("task,profile", [
    ("ames", "ames_gold_v1"),
    ("dili", "dili_gold_v1"),
    ("carcinogens", "carcinogens_gold_v1"),
])
def test_flat_only_task_contracts(task: str, profile: str) -> None:
    module = __import__(
        f"predict.harnesses.branches.tasks.{task}.contract",
        fromlist=["CONFIG"],
    )
    assert module.CONFIG.default_prompt_profile == profile
    assert module.CONFIG.pipeline_module.endswith(f".{task}.pipeline")


def test_flat_only_pipeline_prepares_replayed_retrieval(tmp_path) -> None:
    input_path = tmp_path / "valid.jsonl"
    input_path.write_text(
        json.dumps({
            "benchmark_row_id": "q1", "drug": "CCO",
            "condition_group": "species=human",
        }) + "\n",
        encoding="utf-8",
    )
    replay = tmp_path / "replay"
    replay.mkdir()
    (replay / "retrieval.json").write_text(json.dumps({
        "status": "ok", "query": {"input_smiles": "CCO"}, "groups": [],
    }), encoding="utf-8")

    assert main([
        "--input-jsonl", str(input_path), "--query-index", "0",
        "--out-root", str(tmp_path / "out"), "--run-id", "r1",
        "--retrieval-replay-run-dir", str(replay), "--prepare-only",
    ]) == 0

    prepared = json.loads((tmp_path / "out/r1/retrieval.json").read_text())
    assert prepared["status"] == "ok"
    assert prepared["query"]["external_condition"] == (
        "This prediction concerns the query molecule under species: human."
    )
