import asyncio
import json
import sqlite3

import pandas as pd
import pytest

from semantic_buckets import official_two_pass_weights as weights


def test_complete_pass2_request_is_never_reexecuted(monkeypatch) -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("CREATE TABLE requests (request_id TEXT, status TEXT)")
    connection.execute("INSERT INTO requests VALUES ('done', 'complete')")

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("completed aggregate was reexecuted")

    monkeypatch.setattr(weights.speculative, "execute_fixed_request", fail_if_called)
    asyncio.run(weights._execute_fixed_one(connection, "done", [], "hash", []))


def test_pass2_schedule_uses_twelve_candidates_and_last_prior_score() -> None:
    rows = [
        {"level": "L2", "semantic_bucket_id": f"b{index:02d}", "weight": 1 - index / 100}
        for index in range(27)
    ]

    schedule = weights._pass2_schedule(pd.DataFrame(rows))

    assert schedule.candidate_bucket_ids_json.map(json.loads).map(len).tolist() == [12, 12, 3]
    assert schedule.wave.tolist() == [0, 1, 2]
    assert schedule.chain.tolist() == [0, 0, 0]
    assert json.loads(schedule.iloc[0].anchor_bucket_ids_json) == []
    assert json.loads(schedule.iloc[1].anchor_bucket_ids_json) == ["b11"]
    assert json.loads(schedule.iloc[2].anchor_bucket_ids_json) == ["b23"]
    assert schedule.execution_backend.unique().tolist() == ["fixed_mixed"]


def test_l4_schedule_uses_four_seed_chains_and_shared_tail_anchors() -> None:
    rows = [
        {"level": "L4", "semantic_bucket_id": f"b{index:05d}", "weight": 1 - index / 100_000}
        for index in range(100)
    ]

    schedule = weights._pass2_schedule(pd.DataFrame(rows))

    assert schedule.groupby("wave").size().tolist() == [4, 4, 1]
    assert schedule[schedule.wave.eq(0)].anchor_bucket_ids_json.map(json.loads).tolist() == [
        [], [], [], [],
    ]
    expected = ["b00011", "b00023", "b00035", "b00047"]
    assert schedule[schedule.wave.eq(1)].anchor_bucket_ids_json.map(json.loads).tolist() == [
        expected, expected, expected, expected,
    ]
    assert schedule.candidate_bucket_ids_json.map(json.loads).explode().is_unique


def test_ames_configuration_is_unreviewed_and_uses_all_later_levels() -> None:
    try:
        weights.configure_task("ames")

        assert weights.LEVELS == ("L2", "L3", "L4", "L5")
        assert weights.EXPECTED_BUCKET_COUNT == 11_391
        assert weights.FINAL_STATUS == "complete_unreviewed_candidate"
        assert weights.SEMANTIC_REVIEW_STATUS == "unreviewed_candidate"
    finally:
        weights.configure_task("skin_reaction")


def test_component_bounds_large_dimensions_and_reports_omissions() -> None:
    try:
        weights.configure_task("ames")
        rows = pd.DataFrame({
            "values_json": [json.dumps({"assay": f"assay-{index}"}) for index in range(12)]
        })

        component = weights._component("fixed_mutation", rows)
        values = component["canonical_dimensions"]["assay"]

        assert len(values) == weights.DIMENSION_VALUE_LIMIT + 1
        assert values[-1] == "[4 additional values omitted]"
    finally:
        weights.configure_task("skin_reaction")


def test_v6_card_renders_enriched_experimental_metadata() -> None:
    try:
        weights.configure_task("ames")
        payload = {
            "identity": {"source_components": [{
                "source_id": "mutagenicity_outcomes",
                "source_label": "Mutagenicity outcome evidence",
                "canonical_dimensions": {"canonical_endpoint_concept": ["micronucleus"]},
            }]},
            "sample_records": [{
                "source_id": "mutagenicity_outcomes",
                "source_name": "reviewed source",
                "test_system": "human lymphocytes",
                "metabolic_activation": "with S9",
                "result_call": "positive",
            }],
        }

        card = weights._card("Candidate 1", payload)

        assert "Evidence family: Mutagenicity outcome evidence" in card
        assert "test system: human lymphocytes" in card
        assert "metabolic activation: with S9" in card
        assert "result call: positive" in card
    finally:
        weights.configure_task("skin_reaction")


def test_selected_task_chain_widths_balance_level_depth() -> None:
    try:
        weights.configure_task("dili")
        rows = [
            {"level": level, "semantic_bucket_id": f"{level}_{index:03d}", "weight": 1 - index / 1000}
            for level, count in (("L2", 120), ("L3", 60), ("L4", 30), ("L5", 15))
            for index in range(count)
        ]
        schedule = weights._pass2_schedule(pd.DataFrame(rows))
        assert schedule.groupby("level").chain.max().add(1).to_dict() == {
            "L2": 8, "L3": 4, "L4": 2, "L5": 1,
        }
        assert schedule.groupby("level").wave.max().to_dict() == {
            "L2": 1, "L3": 1, "L4": 1, "L5": 1,
        }
        assert schedule.candidate_bucket_ids_json.map(json.loads).map(len).min() >= 2
        assert schedule.anchor_bucket_ids_json.map(json.loads).map(len).max() == 0
        assert schedule.execution_backend.unique().tolist() == ["luna_standard"]
    finally:
        weights.configure_task("skin_reaction")


def test_two_completed_chains_freeze_four_anchors_before_stragglers() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE requests (request_id TEXT, phase TEXT, status TEXT)")
    connection.execute("CREATE TABLE speculative_aggregates (request_id TEXT, created_at TEXT)")
    connection.execute("CREATE TABLE wave_anchor_decisions (level TEXT, wave INTEGER, "
                       "anchor_bucket_ids_json TEXT, source_requests_json TEXT, "
                       "created_at TEXT, PRIMARY KEY(level,wave))")
    rows = [{"chain": chain, "candidate_bucket_ids_json": json.dumps(
        [f"b{chain}_{index}" for index in range(12)])} for chain in range(4)]
    for chain, completed_at in ((2, "2026-09-23T01:00:00Z"), (0, "2026-09-23T01:00:01Z")):
        request_id = f"request-{chain}"
        connection.execute("INSERT INTO requests VALUES (?,?,?)",
                           (request_id, f"pass2/L6/w0000/c{chain:02d}", "complete"))
        connection.execute("INSERT INTO speculative_aggregates VALUES (?,?)",
                           (request_id, completed_at))
    anchors = weights._wave_decision(connection, "L6", 0, rows)
    assert anchors == ["b2_10", "b2_11", "b0_10", "b0_11"]
    connection.execute("INSERT INTO requests VALUES (?,?,?)",
                       ("late", "pass2/L6/w0000/c01", "complete"))
    connection.execute("INSERT INTO speculative_aggregates VALUES (?,?)",
                       ("late", "2026-09-23T00:00:00Z"))
    assert weights._wave_decision(connection, "L6", 0, rows) == anchors


def test_selected_luna_profile_is_three_high_reasoning_calls_without_ledger(tmp_path) -> None:
    profile = weights._write_luna_profile(tmp_path / "pool.json")
    assert profile["fixed_replica_counts"] == {"openrouter_gpt-6-luna_flex_high": 3}
    assert "spend_budget" not in profile
    provider = profile["providers"][0]
    assert provider["api_key_env"] == "OPEN_ROUTER_KEY_TWO"
    assert provider["request_extra_body"]["reasoning_effort_override"] == "high"


def test_selected_successor_profile_uses_medium_without_flex(tmp_path) -> None:
    profile = weights._write_luna_profile(tmp_path / "pool.json", standard_medium=True)
    provider = profile["providers"][0]
    assert profile["fixed_replica_counts"] == {
        "openrouter_gpt-6-luna_standard_medium": 3,
    }
    assert provider["request_extra_body"]["reasoning_effort_override"] == "medium"
    assert "service_tier" not in provider["request_extra_body"]


def test_next_wave_starts_while_two_prior_chains_are_still_running(monkeypatch) -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE requests (request_id TEXT PRIMARY KEY, phase TEXT, status TEXT)")
    connection.execute("CREATE TABLE speculative_aggregates (request_id TEXT, created_at TEXT)")
    rows = [{"stage": "pass2", "level": "L6", "wave": wave, "chain": chain,
             "batch": wave * 4 + chain,
             "candidate_bucket_ids_json": json.dumps([f"w{wave}c{chain}a", f"w{wave}c{chain}b"]),
             "anchor_bucket_ids_json": "[]"}
            for wave in range(2) for chain in range(4)]
    queued_anchors = {}

    def queue(db, row, prompt, endpoint, *, model, reasoning_effort):
        phase = f"pass2/L6/w{row['wave']:04d}/c{row['chain']:02d}"
        queued_anchors[phase] = json.loads(row["anchor_bucket_ids_json"])
        db.execute("INSERT OR IGNORE INTO requests VALUES (?,?,?)", (phase, phase, "pending"))
        db.commit()
        return phase

    monkeypatch.setattr(weights, "_queue", queue)
    monkeypatch.setattr(weights, "_render", lambda *args: "prompt")
    monkeypatch.setattr(weights, "_scores", lambda *args: pd.DataFrame())

    async def scenario():
        release_late = asyncio.Event()
        next_started = asyncio.Event()

        async def execute(db, request_id, *args):
            if request_id in {"pass2/L6/w0000/c02", "pass2/L6/w0000/c03"}:
                await release_late.wait()
            if request_id == "pass2/L6/w0001/c00":
                next_started.set()
            db.execute("UPDATE requests SET status='complete' WHERE request_id=?", (request_id,))
            db.execute("INSERT INTO speculative_aggregates VALUES (?,?)", (request_id, request_id))
            db.commit()

        monkeypatch.setattr(weights, "_retry_luna_one", execute)
        running = asyncio.create_task(weights._run_wavefront_level(
            pd.DataFrame(rows), connection, {}, [], "hash", [], 3,
            "openrouter_gpt-6-luna_standard_medium", "medium"))
        await asyncio.wait_for(next_started.wait(), timeout=1)
        assert not release_late.is_set()
        assert queued_anchors["pass2/L6/w0001/c00"] == [
            "w0c0a", "w0c0b", "w0c1a", "w0c1b",
        ]
        release_late.set()
        await asyncio.wait_for(running, timeout=1)

    asyncio.run(scenario())


def test_pass1_retries_invalid_json_without_extra_fanout(monkeypatch) -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("CREATE TABLE requests (request_id TEXT, status TEXT)")
    connection.execute("INSERT INTO requests VALUES ('one', 'pending')")
    calls = []

    async def execute(*args, **kwargs):
        calls.append(kwargs)
        return {"status": "failed" if len(calls) == 1 else "complete",
                "receipts": [{"error": "ValueError: candidate order"}]}

    def persist(db, result, **kwargs):
        db.execute("UPDATE requests SET status=? WHERE request_id='one'", (result["status"],))
        db.commit()

    monkeypatch.setattr(weights.speculative, "execute_request", execute)
    monkeypatch.setattr(weights.speculative, "persist_result", persist)
    asyncio.run(weights._execute_one(
        connection, "one", {"base_url": "http://dgx027:50001/v1"}, object(),
        "hash", asyncio.Semaphore(1), fanout=1, required=1))
    assert len(calls) == 2
    assert all(call["initial_fanout"] == call["maximum"] == 1 for call in calls)


def test_pass1_capacity_override_rejects_more_than_endpoint_inventory(monkeypatch, tmp_path) -> None:
    def load_run(_run_id):
        weights.ENDPOINTS = ({"name": "dgx027_50001", "max_inflight": 128,
                              "inventory_max_inflight": 550},)
        return tmp_path, {"status": "incomplete_pass1"}, {}

    monkeypatch.setenv("DEEPSEEK_API_KEY", "EMPTY")
    monkeypatch.setattr(weights, "_load_run", load_run)
    monkeypatch.setattr(weights, "_require_review", lambda *args: None)
    try:
        with pytest.raises(ValueError, match="inventory limit"):
            weights.run_pass1("run", "hash", max_inflight_per_endpoint=551)
    finally:
        weights.ENDPOINTS = ()
