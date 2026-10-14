import gzip
import json
import re
import sqlite3
from types import SimpleNamespace
import pandas as pd
import pytest
from optimization.semantic_buckets import bucket_scoring as scoring
from predict.api_client import pool


def test_manual_and_automatic_chains_share_global_reference():
    first = pd.DataFrame(
        [
            dict(
                task=task,
                level=level,
                semantic_bucket_id=f"b{i:03d}",
                weight=1 - i / 1000,
            )
            for task, level, count in (
                ("dili", "L2", 120),
                ("ames", "L3", 60),
                ("dili", "L4", 30),
                ("ames", "L5", 15),
            )
            for i in range(count)
        ]
    )
    auto = scoring._pass2_schedule(first)
    assert auto.groupby(["task", "level"]).chain.max().add(1).to_dict() == {
        ("dili", "L2"): 8,
        ("ames", "L3"): 4,
        ("dili", "L4"): 2,
        ("ames", "L5"): 1,
    }
    assert auto.candidate_bucket_ids_json.map(json.loads).map(len).min() >= 2
    manual = scoring._pass2_schedule(
        first, chain_mode="manual", chains=1, chain_counts={"ames:L3": 3}
    )
    assert (
        manual.groupby(["task", "level"]).chain.max().add(1).to_dict()["ames", "L3"]
        == 3
    )
    with pytest.raises(ValueError, match="manual chain overrides"):
        scoring._pass2_schedule(first, chain_counts={"dili:L2": 2})
    with pytest.raises(ValueError, match="requested task-levels"):
        scoring._pass2_schedule(first, chain_mode="manual", chain_counts={"skin:L9": 2})


def test_late_completions_cannot_change_frozen_anchor_overlap():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE requests (request_id TEXT,phase TEXT,status TEXT)")
    db.execute("CREATE TABLE speculative_aggregates (request_id TEXT,created_at TEXT)")
    db.execute(
        "CREATE TABLE wave_anchor_decisions (level TEXT,wave INTEGER,anchor_bucket_ids_json TEXT,source_requests_json TEXT,created_at TEXT,PRIMARY KEY(level,wave))"
    )
    rows = [
        dict(chain=i, candidate_bucket_ids_json=json.dumps([f"b{i}a", f"b{i}b"]))
        for i in range(3)
    ]
    for chain, time in ((1, "01"), (2, "02")):
        db.execute(
            "INSERT INTO requests VALUES (?,?,?)",
            (str(chain), f"pass2/dili/L2/w0000/c{chain:02d}", "complete"),
        )
        db.execute(
            "INSERT INTO speculative_aggregates VALUES (?,?)", (str(chain), time)
        )
    anchors = scoring._wave_decision(db, "dili/L2", 0, rows, overlap=1)
    assert anchors == ["b1b", "b2b"]
    db.execute("INSERT INTO requests VALUES ('0','pass2/dili/L2/w0000/c00','complete')")
    db.execute("INSERT INTO speculative_aggregates VALUES ('0','00')")
    assert scoring._wave_decision(db, "dili/L2", 0, rows, overlap=1) == anchors
    db.close()


def _inputs(tmp_path):
    world = tmp_path / "world.parquet"
    pd.DataFrame(
        [dict(level="L2", semantic_bucket_id=f"b{i:02d}") for i in range(8)]
    ).to_parquet(world)
    cards = tmp_path / "cards.json.gz"
    with gzip.open(cards, "wt") as handle:
        json.dump(
            {
                f"b{i:02d}": dict(
                    identity=dict(
                        source_components=[
                            dict(
                                source_id="source",
                                source_label="Assay",
                                canonical_dimensions={"endpoint": ["test"]},
                            )
                        ]
                    ),
                    sample_records=[dict(source_id="source", species="human")],
                )
                for i in range(8)
            },
            handle,
        )
    provider = tmp_path / "provider.json"
    provider.write_text(
        json.dumps(
            dict(
                version="openai_provider_pool.v1",
                providers=[
                    dict(
                        name="local",
                        base_url="http://dgx001:50001/v1",
                        model="deepseek-ai/DeepSeek-V4-Flash-0731",
                        api_key_env="",
                        max_inflight=4,
                    )
                ],
            )
        )
    )
    settings = dict(
        chain_mode="manual",
        chains=1,
        chain_counts={},
        max_chains=8,
        overlap=2,
        advance_after=2,
        pass1_batch_size=3,
        pass2_batch_size=3,
        pass1_fanout=1,
        pass1_average=1,
        pass2_fanout=5,
        pass2_average=3,
        pass1_provider_config=str(provider),
        pass2_provider_config=str(provider),
    )
    return (
        {
            "dili": dict(
                target="drug-induced liver injury",
                world=str(world),
                payloads=str(cards),
            )
        },
        settings,
    )


@pytest.mark.parametrize("passes", [1, 2, 3])
def test_two_pass_scoring_is_offline_at_prepare_complete_resumable_and_pinned(
    tmp_path, monkeypatch, passes
):
    inputs, settings = _inputs(tmp_path)
    settings["passes"] = passes
    calls = []

    async def completion(**kwargs):
        prompt = kwargs["messages"][-1]["content"]
        aliases = re.findall("^Candidate \\d+$", prompt, re.MULTILINE)
        calls.append(prompt)
        scores = [
            dict(
                candidate=alias,
                weight=0.9 - index * 0.1,
                rationale="Strong utility from direct evidence.",
            )
            for index, alias in enumerate(aliases)
        ]
        return SimpleNamespace(
            id=str(len(calls)),
            model=kwargs["model"],
            usage=None,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(dict(scores=scores)),
                        model_extra={},
                        reasoning_content="",
                        reasoning="",
                    )
                )
            ],
        )

    async def close():
        pass

    monkeypatch.setattr(pool, "preflight_provider_models", lambda config: [])
    monkeypatch.setattr(
        pool,
        "async_completion_pool",
        lambda *args, **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=completion)),
            close=close,
        ),
    )
    root = tmp_path / "run"
    scoring.prepare(inputs, root, settings)
    assert calls == []
    scoring.run(root, through_pass=1)
    before_pass2 = len(calls)
    scoring.run(root, through_pass=1)
    assert len(calls) == before_pass2
    scoring.run(root)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "complete_unselected_candidate"
    final = pd.read_parquet(root / "weights.parquet")
    assert len(final) == 8 and (not final.semantic_bucket_id.duplicated().any())
    assert passes == 1 or any(("locked weight" in prompt for prompt in calls))
    before = len(calls)
    assert scoring.run(root) == manifest
    assert len(calls) == before
    with sqlite3.connect(root / "requests.sqlite3") as db:
        count = db.execute(
            "SELECT count(*) FROM speculative_replica_receipts WHERE status='valid'"
        ).fetchone()[0]
    assert count == 3 + (passes - 1) * 3 * 5
    assert passes == 1 or manifest["chain_counts"] == {"dili:L2": 1}
    assert manifest["completed_passes"] == passes
    if passes == 3:
        assert set(pd.read_parquet(root / "pass3_schedule.parquet").stage) == {"pass3"}
    assert scoring.collect(root)["status"] == "complete_unselected_candidate"
    provider = tmp_path / "provider.json"
    provider.write_text(provider.read_text() + "\n")
    with pytest.raises(ValueError, match="provider route changed"):
        scoring.run(root)


def test_fixed_replica_mix_is_validated_offline_and_executed_exactly(
    tmp_path, monkeypatch
):
    inputs, settings = _inputs(tmp_path)
    profile_path = tmp_path / "mixed.json"
    payload = json.loads((tmp_path / "provider.json").read_text())
    payload["providers"].append(
        dict(payload["providers"][0], name="second", model="second-model")
    )
    payload["fixed_replica_counts"] = {"local": 2, "second": 1}
    profile_path.write_text(json.dumps(payload))
    settings.update(
        pass2_provider_config=str(profile_path), pass2_fanout=3, pass2_average=2
    )
    settings["pass2_fanout"] = 2
    with pytest.raises(ValueError, match="fixed replica counts"):
        scoring.prepare(inputs, tmp_path / "invalid", settings)
    assert not (tmp_path / "invalid").exists()
    settings["pass2_fanout"] = 3
    calls = []

    async def close():
        pass

    def client(config, **kwargs):

        async def completion(**kwargs):
            calls.append(kwargs["model"])
            aliases = re.findall(
                "^Candidate \\d+$", kwargs["messages"][-1]["content"], re.MULTILINE
            )
            content = json.dumps(
                dict(
                    scores=[
                        dict(candidate=alias, weight=0.5, rationale="Evidence utility.")
                        for alias in aliases
                    ]
                )
            )
            return SimpleNamespace(
                id=str(len(calls)),
                model=kwargs["model"],
                usage=None,
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content, model_extra={})
                    )
                ],
            )

        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=completion)),
            close=close,
        )

    monkeypatch.setattr(pool, "preflight_provider_models", lambda config: [])
    monkeypatch.setattr(pool, "async_completion_pool", client)
    root = tmp_path / "fixed-run"
    scoring.prepare(inputs, root, settings)
    assert calls == []
    scoring.run(root, through_pass=1)
    scoring.run(root)
    assert calls.count("second-model") == 3
    assert calls.count(payload["providers"][0]["model"]) == 3 + 3 * 2


def test_task_rubric_reasoning_and_token_budget_are_pinned(tmp_path, monkeypatch):
    inputs, settings = _inputs(tmp_path)
    provider_path = tmp_path / "provider.json"
    provider = json.loads(provider_path.read_text())
    provider["providers"][0]["model"] = "openai/gpt-6-luna"
    provider_path.write_text(json.dumps(provider))
    rubric = tmp_path / "rubric.jinja"
    rubric.write_text("Clinical criterion for {{ target }}: human evidence.")
    inputs["dili"]["rubric"] = str(rubric)
    settings.update(passes=1, reasoning_effort="medium", max_tokens=128000)
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        aliases = re.findall(
            "^Candidate \\d+$", kwargs["messages"][-1]["content"], re.MULTILINE
        )
        response = dict(
            scores=[
                dict(candidate=alias, weight=0.5, rationale="Human evidence.")
                for alias in aliases
            ]
        )
        return SimpleNamespace(
            id=str(len(calls)),
            model=kwargs["model"],
            usage=None,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(response), model_extra={}
                    )
                )
            ],
        )

    async def close():
        pass

    monkeypatch.setattr(pool, "preflight_provider_models", lambda config: [])
    monkeypatch.setattr(
        pool,
        "async_completion_pool",
        lambda *a, **k: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=completion)),
            close=close,
        ),
    )
    root = tmp_path / "clinical"
    manifest = scoring.prepare(inputs, root, settings)
    assert (
        "Clinical criterion" in (root / "prompt_review/dili_L2_initial.txt").read_text()
    )
    assert not calls
    scoring.run(root)
    assert all(
        (
            call["reasoning_effort"] == "medium" and call["max_tokens"] == 128000
            for call in calls
        )
    )
    rubric.write_text("Changed clinical policy.")
    with pytest.raises(ValueError, match="scoring source changed"):
        scoring.run(root)
