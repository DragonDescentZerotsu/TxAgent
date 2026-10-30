"""Paid-request boundaries: deduplication, bounded retries, concurrency and resume."""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.chembl_tool.common.record_diversity import (
    ANNOTATION_VERSION, BOUND_ANNOTATION_VERSION, cache_key,
    extraction_messages, extraction_request_body, prepare_record,
)
from tools.chembl_tool.common.openai_provider_pool import (
    ProviderPoolConfig,
    ProviderSpec,
)
from tools.chembl_tool.paper_experiments import run_record_diversity_extraction as run


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("url,field,key", [
    ("https://openrouter.ai/api/v1", "reasoning", "enabled"),
    ("http://127.0.0.1:50002/v1", "chat_template_kwargs", "thinking"),
    ("http://dgx007:50002/v1", "chat_template_kwargs", "thinking"),
    ("https://litellm.example.invalid/v1", "chat_template_kwargs", "thinking"),
])
def test_extraction_transport_controls_match_probe_and_bulk(url, field, key, thinking):
    settings = {field: {"other_setting": "preserved"}, "seed": 7}
    original = deepcopy(settings)
    result = extraction_request_body(url, settings, thinking=thinking)
    assert result[field] == {"other_setting": "preserved", key: thinking}
    assert result["seed"] == 7
    assert settings == original


def inputs(tmp_path, count=2, duplicates=True):
    records = []
    for i in range(count):
        p = prepare_record(
            {
                "canonical_record_id": str(i),
                "molecule_name": "compound A",
                "canonical_endpoint_name": "cell growth",
                "support_text": f"Compound A increased growth{0 if duplicates else i}.",
            }
        )
        records.append(
            {
                "record_id": str(i),
                "source_id": "source",
                "level": 1,
                "route": p["route"],
                "payload_json": json.dumps(p),
            }
        )
    path = tmp_path / "fixture.features.parquet"
    pq.write_table(pa.Table.from_pylist(records), path)
    contract = {
        "model_config": {"model": "fake"},
        "expected_source_rows": count,
        "expected_llm_candidate_rows": count,
    }
    return [path], contract


def annotation():
    return {
        "outcomes": [
            {
                "endpoint": "cell growth",
                "kind": "increase",
                "focal_scope": "focal",
                "evidence": [{"field": "support_text", "quote": "increased growth"}],
            }
        ],
        "assay": [],
        "system": [],
        "exposure": [],
        "comparison_role": [],
        "unresolved": [],
    }


def test_identical_inputs_share_one_request_and_contract_is_frozen(tmp_path):
    paths, contract = inputs(tmp_path)
    db = run.database(tmp_path / "db.sqlite")
    run.ingest(db, paths, contract, tmp_path / "progress.json")
    assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 2
    run.ingest(db, paths, contract, tmp_path / "progress.json")
    assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    with pytest.raises(ValueError, match="contract changed"):
        run.ingest(
            db,
            paths,
            {**contract, "model_config": {"model": "different"}},
            tmp_path / "progress.json",
        )
    db.close()


def test_persisted_hold_prevents_accidental_resume(tmp_path):
    run.check_resume_hold(tmp_path)
    hold = tmp_path / "pause.json"
    hold.write_text('{"reason":"validate four tasks before continuing"}')
    with pytest.raises(RuntimeError, match="Extraction is held"):
        run.check_resume_hold(tmp_path)
    hold.unlink()
    run.check_resume_hold(tmp_path)


def test_bound_review_notes_survive_ingestion_and_separate_cache_keys(tmp_path):
    paths, contract = inputs(tmp_path)
    contract["model_config"]["annotation_version"] = BOUND_ANNOTATION_VERSION
    rows = pq.read_table(paths[0]).to_pylist()
    for row, note in zip(rows, ["Keep each vehicle's result separate", "Preserve the comparator"]):
        prepared = json.loads(row["payload_json"])
        prepared["review_notes"] = note
        row["payload_json"] = json.dumps(prepared)
    pq.write_table(pa.Table.from_pylist(rows), paths[0])
    db = run.database(tmp_path / "db.sqlite")
    try:
        run.ingest(db, paths, contract, tmp_path / "progress.json")
        jobs = db.execute("SELECT cache_key,payload FROM jobs").fetchall()
        assert len(jobs) == 2
        for key, payload in jobs:
            prepared = run.unpack(payload)
            assert prepared["review_notes"]
            assert key == cache_key(prepared, contract["model_config"], BOUND_ANNOTATION_VERSION)
            assert prepared["review_notes"] in extraction_messages(prepared, BOUND_ANNOTATION_VERSION)[1]["content"]
    finally:
        db.close()


def test_resume_keeps_success_and_respects_attempt_limit(tmp_path):
    paths, contract = inputs(tmp_path, count=3, duplicates=False)
    db = run.database(tmp_path / "db.sqlite")
    run.ingest(db, paths, contract, tmp_path / "progress.json")
    rows = run.claim(db, 3)
    first = rows[0][0]
    run.save_attempt(db, first, 1, {"usage": {"cost": 0.01}}, annotation(), None, 0.1)
    db.execute("UPDATE jobs SET attempts=1 WHERE id=?", (rows[1][0],))
    db.execute("UPDATE jobs SET attempts=3 WHERE id=?", (rows[2][0],))
    db.commit()
    run.recover_inflight(db, 3)
    assert dict(db.execute("SELECT id,status FROM jobs")) == {
        first: "done",
        rows[1][0]: "pending",
        rows[2][0]: "failed",
    }
    assert db.execute("SELECT reported_cost FROM attempts").fetchone()[0] == 0.01
    db.close()


def test_reuse_successes_matches_native_input_and_preserves_attempts(tmp_path):
    paths, contract = inputs(tmp_path, count=2, duplicates=False)
    donor = run.database(tmp_path / "donor.sqlite")
    run.ingest(donor, paths, contract, tmp_path / "progress.json")
    job, _, _ = run.claim(donor, 1)[0]
    run.save_attempt(donor, job, 1, {"usage": {"cost": 0.01}}, annotation(), None, 0.1)
    donor.close()
    # New input inventory, same model/prompt; only exact unchanged source jobs transfer.
    changed = tmp_path / "new"
    changed.mkdir()
    new_paths, new_contract = inputs(changed, count=3, duplicates=False)
    db = run.database(tmp_path / "new.sqlite")
    run.ingest(db, new_paths, new_contract, tmp_path / "progress.json")
    receipt = run.reuse_annotations(db, tmp_path / "donor.sqlite", new_contract)
    assert receipt["imported_jobs"] == 1
    assert dict(db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status")) == {"done": 1, "pending": 2}
    assert db.execute("SELECT reported_cost FROM attempts").fetchone()[0] == 0.01
    assert run.reuse_annotations(db, tmp_path / "donor.sqlite", new_contract)["imported_jobs"] == 0
    with pytest.raises(ValueError, match="different model_config"):
        run.reuse_annotations(db, tmp_path / "donor.sqlite", {**new_contract, "model_config": {"model": "other"}})
    db.close()


def test_reuse_revalidates_quotes_and_rolls_back_on_failure(tmp_path):
    paths, contract = inputs(tmp_path, count=2, duplicates=False)
    donor = run.database(tmp_path / "donor.sqlite")
    run.ingest(donor, paths, contract, tmp_path / "progress.json")
    for job, _, _ in run.claim(donor, 2):
        value = annotation()
        if job == 2:
            value['outcomes'][0]['evidence'][0]['quote'] = 'not in original source'
        run.save_attempt(donor, job, 1, {}, value, None, 0.1)
    donor.close()
    db = run.database(tmp_path / "new.sqlite")
    run.ingest(db, paths, contract, tmp_path / "progress.json")
    with pytest.raises(ValueError):
        run.reuse_annotations(db, tmp_path / "donor.sqlite", contract)
    assert db.execute("SELECT COUNT(*) FROM annotations").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM jobs WHERE status='pending'").fetchone()[0] == 2
    db.close()


@pytest.mark.parametrize("version", [ANNOTATION_VERSION, BOUND_ANNOTATION_VERSION])
@pytest.mark.parametrize("quote_catalog", [False, True])
@pytest.mark.parametrize("thinking", [False, True])
def test_requests_are_bounded_and_validated_retries_are_cached(tmp_path, monkeypatch, version, quote_catalog, thinking):
    paths, contract = inputs(tmp_path, count=5, duplicates=False)
    contract["model_config"]["annotation_version"] = version
    db = run.database(tmp_path / "db.sqlite")
    run.ingest(db, paths, contract, tmp_path / "progress.json")
    for key, payload in db.execute("SELECT cache_key,payload FROM jobs"):
        assert key == cache_key(run.unpack(payload), contract["model_config"], version)
    stats = {"calls": 0, "active": 0, "peak": 0, "seen": set()}
    endpoint_active = {"a": 0, "b": 0}
    endpoint_peak = {"a": 0, "b": 0}

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs["async_max_connections"] == 2
            assert kwargs["temperature"] == (1 if thinking else 0)
            assert kwargs["reasoning_effort"] == ("low" if thinking else "")
            assert (
                kwargs["request_extra_body"]["chat_template_kwargs"]["thinking"]
                is thinking
            )
            self.name = kwargs["base_url"]

        async def async_chat_json(self, messages):
            expected = extraction_messages({"input": {}}, version)[0]
            if quote_catalog:
                expected["content"] += run.QUOTE_TRANSPORT_INSTRUCTION
                catalog = json.loads(messages[2]["content"])["source_quote_catalog"]
            assert messages[0] == expected
            stats["calls"] += 1
            stats["active"] += 1
            endpoint_active[self.name] += 1
            endpoint_peak[self.name] = max(
                endpoint_peak[self.name], endpoint_active[self.name]
            )
            stats["peak"] = max(stats["peak"], stats["active"])
            await asyncio.sleep(0.005)
            stats["active"] -= 1
            endpoint_active[self.name] -= 1
            value = annotation()
            if version == BOUND_ANNOTATION_VERSION:
                value["outcomes"][0].update(
                    conditions=[], comparator=[], statistical_support="not_reported",
                    attribution_basis="named_text",
                )
            key = messages[1]["content"]
            if key not in stats["seen"]:
                stats["seen"].add(key)
                value["outcomes"][0]["evidence"][0]["quote"] = "fabricated"
            else:
                assert messages[-2]['role'] == 'assistant'
                previous = json.loads(messages[-2]['content'])
                assert previous['outcomes'][0]['evidence'][0]['quote'] == 'fabricated'
                assert 'validation' in messages[-1]['content']
                if quote_catalog:
                    quote_id = next(k for k, v in catalog.items() if v["field"] == "support_text")
                    value["outcomes"][0]["evidence"][0] = {"field": "support_text", "quote": quote_id}
            return {
                "content": value,
                "usage": {"cost": 0.001, "prompt_tokens": 10, "completion_tokens": 5},
            }

        async def aclose(self):
            pass

    monkeypatch.setattr(run, "OpenAICompatibleClient", FakeClient)
    args = SimpleNamespace(
        parallelism=4,
        max_attempts=2,
        timeout=5,
        limit=0,
        output_dir=tmp_path,
        database_path=tmp_path / "db.sqlite",
    )
    providers = ProviderPoolConfig(
        tuple(
            ProviderSpec(
                name=n, base_url=n, model="fake", api_key_env="", max_inflight=2
            )
            for n in ("a", "b")
        )
    )
    config = {"model": "fake", "max_tokens": 100, "annotation_version": version,
              "reasoning_enabled": thinking, "temperature": 1 if thinking else 0,
              "reasoning_effort": "low" if thinking else ""}
    if quote_catalog:
        config.update(quote_transport="source_quote_catalog.v1", quote_repair="source_quote_repair.v1")
    keys = {"a": "EMPTY", "b": "EMPTY"}
    assert asyncio.run(run.extract(db, args, providers, config, keys)) == 0
    assert stats["calls"] == 10 and stats["peak"] == 4
    assert endpoint_peak == {"a": 2, "b": 2}
    assert dict(db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status")) == {
        "done": 5
    }
    assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 10
    before = db.execute(
        "SELECT job_id,annotation FROM annotations ORDER BY job_id"
    ).fetchall()
    assert asyncio.run(run.extract(db, args, providers, config, keys)) == 0
    assert stats["calls"] == 10
    assert (
        db.execute(
            "SELECT job_id,annotation FROM annotations ORDER BY job_id"
        ).fetchall()
        == before
    )
    responses = [run.unpack(r[0]) for r in db.execute("SELECT response FROM attempts")]
    assert {r["execution_provider"]["provider"] for r in responses} == {"a", "b"}
    backup = run.database(tmp_path / "extraction.checkpoint.sqlite")
    assert backup.execute("SELECT COUNT(*) FROM annotations").fetchone()[0] == 5
    backup.close()
    db.close()


def test_transport_switch_preserves_cache_but_rejects_semantic_changes():
    old = {
        "model": "deepseek/deepseek-v4-flash-0731",
        "base_url": "https://openrouter.ai/api/v1",
        "annotation_version": "v2",
        "temperature": 0,
        "max_tokens": 1800,
        "request_extra_body": {"reasoning": {"enabled": False}},
    }
    new = {
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "annotation_version": "v2",
        "temperature": 0,
        "max_tokens": 1800,
        "reasoning_enabled": False,
    }
    assert run.compatible_cache_config(old, new) == old
    thinking = {**new, "reasoning_enabled": True}
    assert run.compatible_cache_config(thinking, thinking) == thinking
    with pytest.raises(ValueError, match="Reasoning settings changed"):
        run.compatible_cache_config(thinking, new)
    for changed in (
        {"model": "different"},
        {"max_tokens": 2000},
        {"reasoning_enabled": True},
        {"reasoning_effort": "low"},
        {"temperature": 1},
        {"annotation_version": "v3"},
        {"quote_transport": "source_quote_catalog.v1"},
        {"quote_repair": "source_quote_repair.v1"},
    ):
        with pytest.raises(ValueError):
            run.compatible_cache_config(old, {**new, **changed})


def test_exhausted_annotation_failures_exit_nonzero_and_keep_attempts(tmp_path, monkeypatch):
    paths, contract = inputs(tmp_path, count=1)
    db = run.database(tmp_path / "db.sqlite")
    run.ingest(db, paths, contract, tmp_path / "progress.json")
    calls = []

    class InvalidClient:
        def __init__(self, **kwargs):
            pass

        async def async_chat_json(self, messages):
            calls.append(messages)
            return {"content": {"invalid": True}}

        async def aclose(self):
            pass

    monkeypatch.setattr(run, "OpenAICompatibleClient", InvalidClient)
    args = SimpleNamespace(parallelism=1, max_attempts=1, timeout=5,
                           limit=0, output_dir=tmp_path)
    providers = ProviderPoolConfig((ProviderSpec(
        name="fake", base_url="http://fake/v1", model="fake", api_key_env="", max_inflight=1,
    ),))
    try:
        for _ in range(2):
            assert asyncio.run(run.extract(
                db, args, providers, {"model": "fake", "max_tokens": 100}, {"fake": "EMPTY"},
            )) == 1
        assert len(calls) == 1  # Resume cannot silently reset exhausted attempts.
        assert db.execute("SELECT status,attempts FROM jobs").fetchone() == ("failed", 1)
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1
        assert json.loads((tmp_path / "completion.json").read_text())["status"] == "complete_with_failures"
        args.max_attempts = 2
        assert asyncio.run(run.extract(
            db, args, providers, {"model": "fake", "max_tokens": 100}, {"fake": "EMPTY"},
        )) == 1
        assert len(calls) == 2
        assert calls[-1][-2]['role'] == 'assistant'
        assert json.loads(calls[-1][-2]['content']) == {'invalid': True}
        assert db.execute("SELECT status,attempts FROM jobs").fetchone() == ('failed', 2)
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 2
    finally:
        db.close()


def test_priority_sources_change_only_schedule_and_never_repeat_jobs(tmp_path):
    db=run.database(tmp_path/'jobs.sqlite')
    paths,contract=inputs(tmp_path,count=4,duplicates=False)
    run.ingest(db,paths,contract,tmp_path/'progress.json')
    before=list(db.execute('SELECT id,cache_key,payload FROM jobs ORDER BY id'))
    ids={rid:jid for rid,jid in db.execute('SELECT record_id,job_id FROM records')}
    priority=tmp_path/'priority.json'
    priority.write_text(json.dumps([{'task':'fixture','record_id':r} for r in ['3','1','3']]))
    run.prioritize(db,priority)
    assert [r[0] for r in run.claim(db,3)]==[ids['3'],ids['1'],ids['0']]
    assert [r[0] for r in run.claim(db,3)]==[ids['2']]
    assert run.claim(db,3)==[]
    assert list(db.execute('SELECT id,cache_key,payload FROM jobs ORDER BY id'))==before
    priority.write_text(json.dumps([{'task':'fixture','record_id':'missing'}]))
    with pytest.raises(ValueError,match='Unknown priority'):
        run.prioritize(db,priority)
    db.close()
