from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from predict.live import (
    cancel_run,
    clone_prompt,
    create_run,
    find_run,
    launch_saved,
    promote_run,
    publish_snapshot,
    sanitize,
    update_run,
)
from predict.traces.catalog import discover_traces
from predict.traces.io import write_trace


def test_live_trace_hierarchy_and_public_projection(tmp_path, monkeypatch):
    monkeypatch.setenv("TXAGENT_LIVE_PUBLISH", "0")
    run_dir = create_run(
        root=tmp_path,
        dataset="BBB_Martins",
        method="progressive assay-transfer",
        command=["python", "-m", "example"],
        metadata={"pilot_size": 3},
        requested_id="2026-09-09_12-34",
    )
    trace_path = write_trace(
        trace_root=tmp_path,
        experiment_id="experiment",
        task="BBB_Martins",
        harness="progressive",
        method="progressive assay-transfer",
        run_id=run_dir.name,
        sample_id="query_idx00000",
        stage="level_01",
        checkpoint_path="/private/checkpoint.json",
        output={
            "llm": {
                "messages": [{"role": "user", "content": "full prompt"}],
                "reasoning": "full reasoning",
                "content": {"bbb_prediction": "pass"},
                "base_url": "http://127.0.0.1:50000/v1",
            }
        },
    )

    assert trace_path == run_dir / "samples/query_idx00000/level_01.json"
    public = json.loads(
        (run_dir / "public/samples/query_idx00000/level_01.json").read_text()
    )
    assert public["input_prompt"] == "[USER]\nfull prompt"
    assert public["llm"]["reasoning"] == "full reasoning"
    assert "base_url" not in public["llm"]
    assert "checkpoint_path" not in public
    assert sanitize("TASK=/vast/private/data http://dgx027:50001/v1") == (
        "TASK=[local path omitted] [private endpoint omitted]"
    )
    run = json.loads((run_dir / "public/run.json").read_text())
    assert run["samples"] == {"query_idx00000": ["level_01"]}
    assert find_run("BBB_Martins/progressive_assay-transfer/2026-09-09_12-34", tmp_path) == run_dir
    assert len(discover_traces(tmp_path)) == 1

    publish_requests = []
    monkeypatch.setattr("predict.live.request_publish", publish_requests.append)
    for index in range(1, 4):
        write_trace(
            trace_root=tmp_path,
            experiment_id="experiment",
            task="BBB_Martins",
            harness="progressive",
            method="progressive assay-transfer",
            run_id=run_dir.name,
            sample_id=f"query_idx{index:05d}",
            stage="level_01",
            checkpoint_path="checkpoint.json",
            output={"llm": {"messages": [{"role": "user", "content": "prompt"}]}},
        )
    assert len(list((run_dir / "samples").glob("*"))) == 4
    assert sorted(path.name for path in (run_dir / "public/samples").glob("*")) == [
        "query_idx00000",
        "query_idx00001",
        "query_idx00002",
    ]
    assert len(publish_requests) == 2


def test_live_study_hierarchy_is_recursive(tmp_path, monkeypatch):
    monkeypatch.setenv("TXAGENT_LIVE_PUBLISH", "0")
    run_dir = create_run(
        root=tmp_path,
        dataset="bbb_martins",
        method="morgan",
        study="contrastive",
        run_group="k10_m3_20260914_1200",
        condition="cached",
        query_prior="cached",
        command=[],
        metadata={},
    )
    assert run_dir.relative_to(tmp_path).as_posix() == (
        "contrastive/morgan/with_query_prior/k10_m3_20260914_1200/"
        "bbb_martins/cached"
    )
    write_trace(
        trace_root=tmp_path,
        experiment_id="experiment",
        task="bbb_martins",
        harness="progressive",
        method="morgan",
        run_id=run_dir.name,
        run_dir=run_dir,
        sample_id="query_idx00000",
        stage="level_01",
        checkpoint_path="checkpoint.json",
        output={"llm": {"messages": [{"role": "user", "content": "prompt"}]}},
    )
    assert find_run(run_dir.relative_to(tmp_path).as_posix(), tmp_path) == run_dir
    assert len(discover_traces(tmp_path)) == 1
    catalog = json.loads((tmp_path / "catalog.json").read_text())
    assert catalog["runs"][0]["path"] == run_dir.relative_to(tmp_path).as_posix()
    assert catalog["runs"][0]["query_prior"] == "cached"


def test_live_review_publishes_explicit_sanitized_samples(tmp_path, monkeypatch):
    monkeypatch.setenv("TXAGENT_LIVE_PUBLISH", "0")
    run_dir = create_run(
        root=tmp_path,
        dataset="dataset",
        method="method",
        command=[],
        metadata={},
        requested_id="2026-09-09_12-34",
    )
    for index in range(4):
        write_trace(
            trace_root=tmp_path,
            experiment_id="experiment",
            task="dataset",
            harness="progressive",
            method="method",
            run_id=run_dir.name,
            sample_id=f"query_idx{index:05d}",
            stage="level_01",
            checkpoint_path="checkpoint.json",
            output={"llm": {"messages": [{"role": "user", "content": "prompt"}]}},
        )
    (run_dir / "review.json").write_text(
        json.dumps(
            {
                "schema_version": "predict_live_review.v1",
                "title": "Correct flips",
                "expected_sample_count": 2,
                "group_by": "primary_driver",
                "filter_fields": ["flip_direction"],
                "samples": {
                    "query_idx00001": {
                        "primary_driver": "scaffold_family",
                        "flip_direction": "1->0",
                        "private_note": "/vast/private/review.tsv",
                    },
                    "query_idx00003": {
                        "primary_driver": "query_memory",
                        "flip_direction": "0->1",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    update_run(run_dir, status="complete")

    public = json.loads((run_dir / "public/run.json").read_text())
    assert sorted(public["samples"]) == ["query_idx00001", "query_idx00003"]
    assert public["review"]["samples"]["query_idx00001"]["private_note"] == (
        "[local path omitted]"
    )
    assert sorted(path.parent.name for path in (run_dir / "public/samples").glob("*/*.json")) == [
        "query_idx00001",
        "query_idx00003",
    ]


def test_live_review_fails_closed_on_invalid_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("TXAGENT_LIVE_PUBLISH", "0")
    run_dir = create_run(
        root=tmp_path,
        dataset="dataset",
        method="method",
        command=[],
        metadata={},
        requested_id="2026-09-09_12-34",
    )
    review_path = run_dir / "review.json"
    review_path.write_text(
        json.dumps({"schema_version": "unknown", "samples": {"query_idx00000": {}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported live review schema"):
        update_run(run_dir, status="complete")

    review_path.write_text(
        json.dumps(
            {
                "schema_version": "predict_live_review.v1",
                "expected_sample_count": 2,
                "samples": {"query_idx00000": {}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sample count mismatch"):
        update_run(run_dir, status="complete")

    review_path.write_text(
        json.dumps(
            {
                "schema_version": "predict_live_review.v1",
                "expected_sample_count": 1,
                "samples": {"query_idx00000": {}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sample is unavailable"):
        update_run(run_dir, status="complete")


def test_live_review_fails_closed_on_missing_sample(tmp_path, monkeypatch):
    monkeypatch.setenv("TXAGENT_LIVE_PUBLISH", "0")
    run_dir = create_run(
        root=tmp_path,
        dataset="dataset",
        method="method",
        command=[],
        metadata={},
        requested_id="2026-09-09_12-34",
    )
    (run_dir / "review.json").write_text(
        json.dumps(
            {
                "schema_version": "predict_live_review.v1",
                "expected_sample_count": 1,
                "samples": {"query_idx99999": {"primary_driver": "query_memory"}},
            }
        ),
        encoding="utf-8",
    )

    try:
        update_run(run_dir, status="complete")
    except ValueError as exc:
        assert "sample is unavailable" in str(exc)
    else:
        raise AssertionError("missing review sample should fail closed")


def test_cancel_stops_the_recorded_process_group(tmp_path, monkeypatch):
    monkeypatch.setenv("TXAGENT_LIVE_PUBLISH", "0")
    run_dir = create_run(
        root=tmp_path,
        dataset="dataset",
        method="method",
        command=["unused"],
        metadata={},
        requested_id="2026-09-09_12-35",
    )
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    started = open(f"/proc/{process.pid}/stat", encoding="utf-8").read().split()[21]
    update_run(
        run_dir,
        pid=process.pid,
        process_group=process.pid,
        process_started_at=started,
    )
    cancel_run(run_dir)
    process.wait(timeout=2)
    status = json.loads((run_dir / "run.json").read_text())["status"]
    assert status == "cancelled"


def test_run_minute_collision_gets_readable_suffix(tmp_path, monkeypatch):
    monkeypatch.setenv("TXAGENT_LIVE_PUBLISH", "0")
    monkeypatch.setattr("predict.live.minute_id", lambda: "2026-09-09_12-36")
    first = create_run(
        root=tmp_path, dataset="dataset", method="method", command=[], metadata={}
    )
    second = create_run(
        root=tmp_path, dataset="dataset", method="method", command=[], metadata={}
    )
    assert first.name == "2026-09-09_12-36"
    assert second.name == "2026-09-09_12-36-02"


def test_throughput_run_stays_private_until_promoted(tmp_path, monkeypatch):
    monkeypatch.setenv("TXAGENT_LIVE_PUBLISH", "0")
    run_dir = create_run(
        root=tmp_path,
        dataset="dataset",
        method="method",
        command=[],
        metadata={},
        execution_mode="throughput",
    )
    write_trace(
        trace_root=tmp_path,
        experiment_id="experiment",
        task="dataset",
        harness="progressive",
        method="method",
        run_id=run_dir.name,
        sample_id="query_idx00000",
        stage="level_01",
        checkpoint_path="checkpoint.json",
        output={"llm": {"messages": [{"role": "user", "content": "prompt"}]}},
    )

    assert json.loads((run_dir / "run.json").read_text())["visibility"] == "private"
    assert json.loads((tmp_path / "catalog.json").read_text())["runs"] == []

    promote_run(run_dir)

    catalog = json.loads((tmp_path / "catalog.json").read_text())
    assert catalog["runs"][0]["run_id"] == run_dir.name
    assert json.loads((run_dir / "run.json").read_text())["visibility"] == "public"


def test_publication_pushes_only_public_projection(tmp_path, monkeypatch):
    root = tmp_path / "live"
    remote = tmp_path / "pages.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    monkeypatch.setenv("TXAGENT_LIVE_PUBLISH", "0")
    monkeypatch.setenv("TXAGENT_LIVE_PAGES_REPO", str(remote))
    run_dir = create_run(
        root=root,
        dataset="dataset",
        method="method",
        command=["secret-local-command"],
        metadata={},
        requested_id="2026-09-09_12-37",
    )
    write_trace(
        trace_root=root,
        experiment_id="experiment",
        task="dataset",
        harness="progressive",
        method="method",
        run_id=run_dir.name,
        sample_id="query_idx00000",
        stage="level_01",
        checkpoint_path="/private/checkpoint.json",
        output={"llm": {"messages": [{"role": "user", "content": "prompt"}]}},
    )
    stale = run_dir / "public/samples/query_idx99999/level_01.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}", encoding="utf-8")
    publish_snapshot(root)
    published = subprocess.run(
        ["git", f"--git-dir={remote}", "show", "live-traces:traces-data/catalog.json"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert json.loads(published)["runs"][0]["status"] == "running"
    tree = subprocess.run(
        ["git", f"--git-dir={remote}", "ls-tree", "-r", "--name-only", "live-traces"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "traces-data/dataset/method/2026-09-09_12-37/samples/query_idx00000/level_01.json" in tree
    assert "query_idx99999" not in tree
    assert ".lock" not in tree
    assert "publisher.log" not in tree


def test_prompt_clone_inherits_frozen_runtime_behavior(tmp_path, monkeypatch):
    from predict.harnesses.progressive import prompt

    source = Path(prompt.__file__).parent / "prompts/reranked_progressive_v8"
    prompt_root = tmp_path / "prompts"
    shutil.copytree(source, prompt_root / source.name)
    monkeypatch.setattr(prompt, "PROMPT_DIR", prompt_root)
    prompt.prompt_assets.cache_clear()
    run_dir = tmp_path / "live/dataset/method/2026-09-09_12-38"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"prompt_version": source.name}), encoding="utf-8"
    )
    destination = clone_prompt(run_dir, "reranked_progressive_v9")
    prompt.prompt_assets.cache_clear()

    assert destination == prompt_root / "reranked_progressive_v9"
    assert prompt.behavior_version("reranked_progressive_v9") == source.name
    assert set(prompt.prompt_assets("reranked_progressive_v9")["modes"]) == {
        "morgan",
        "assay-transfer",
        "joint",
    }


def test_prompt_relaunch_is_a_new_three_sample_run(tmp_path, monkeypatch):
    monkeypatch.setenv("TXAGENT_LIVE_PUBLISH", "0")
    run_dir = tmp_path / "live/dataset/method/2026-09-09_12-39"
    run_dir.mkdir(parents=True)
    command = [
        sys.executable,
        "-m",
        "predict.harnesses.progressive",
        "--output-root",
        "/old/output",
        "--prompt-version",
        "reranked_progressive_v8",
        "--live-run-id",
        run_dir.name,
        "--continue-after-pilot",
    ]
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "harness": "progressive",
                "resume_command": command,
                "output_root": "/old/output",
            }
        ),
        encoding="utf-8",
    )
    launched = []

    class Process:
        pid = 123

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda call, **kwargs: launched.append(call) or Process(),
    )
    launch_saved(run_dir, prompt_version="reranked_progressive_v9")

    call = launched[0]
    assert "--live-run-id" not in call
    assert "--continue-after-pilot" not in call
    assert call[call.index("--prompt-version") + 1] == "reranked_progressive_v9"
    assert call[call.index("--output-root") + 1].startswith(
        "/old/output_reranked_progressive_v9_"
    )
