from optimization.semantic_buckets import storage as semantic_storage
from optimization.semantic_buckets import bucket_scoring as semantic_bucket_scoring
from optimization.semantic_buckets import execution as semantic_execution
import asyncio
import json
import time
from types import SimpleNamespace
import pytest
from optimization.semantic_buckets import bucket_scoring as weights


def _response(weight: float, rationale: str) -> dict:
    return {
        "scores": [
            {"candidate": "Candidate 1", "weight": weight, "rationale": rationale},
            {"candidate": "Candidate 2", "weight": 0.25, "rationale": rationale},
        ]
    }


def _valid_receipt(replica_id: int, weight: float) -> dict:
    now = time.monotonic()
    return {
        "replica_id": replica_id,
        "status": "valid",
        "started_monotonic": now - 1,
        "finished_monotonic": now,
        "response": _response(weight, f"replica-{replica_id}"),
        "reasoning_content": f"reasoning-{replica_id}",
        "usage": None,
    }


def test_medium_reasoning_is_limited_to_approved_standard_luna() -> None:
    row = {
        "reasoning_effort": "medium",
        "prompt": "score",
        "max_tokens": 100,
        "validation_json": json.dumps({"execution_backend": "luna_standard"}),
    }
    request = semantic_execution._request_payload(row, "openai/gpt-6-luna")
    assert request["reasoning_effort"] == "medium"
    with pytest.raises(ValueError, match="approved medium Luna"):
        semantic_execution._request_payload(row, "deepseek-ai/DeepSeek-V4-Flash-0731")
    row["validation_json"] = "{}"
    with pytest.raises(ValueError, match="approved medium Luna"):
        semantic_execution._request_payload(row, "openai/gpt-6-luna")


def test_first_eight_mean_uses_half_up_rounding_and_closest_rationale() -> None:
    receipts = [_valid_receipt(index, 0.5 if index < 4 else 0.51) for index in range(8)]
    response, metadata = semantic_bucket_scoring.aggregate_responses(
        receipts, ["Candidate 1", "Candidate 2"]
    )
    assert response["scores"][0] == {
        "candidate": "Candidate 1",
        "weight": 0.51,
        "rationale": "replica-0",
    }
    component = metadata["components"]["Candidate 1"]
    assert component["mean"] == pytest.approx(0.505)
    assert component["sample_stddev"] > 0
    assert component["min"] == 0.5 and component["max"] == 0.51
    assert metadata["aggregation_method"] == "first_8_valid_mean_v1"


def test_single_replica_aggregation_has_zero_sample_stddev() -> None:
    receipt = _valid_receipt(0, 0.72)
    response, metadata = semantic_bucket_scoring.aggregate_responses(
        [receipt], ["Candidate 1", "Candidate 2"], required=1
    )
    assert response["scores"][0]["weight"] == 0.72
    assert metadata["components"]["Candidate 1"]["sample_stddev"] == 0.0


def test_collection_cancels_losers_after_required_valid_responses() -> None:

    async def scenario():

        async def call(replica_id: int) -> dict:
            started = time.monotonic()
            try:
                await asyncio.sleep(0.001 if replica_id < 2 else 10)
            except asyncio.CancelledError:
                await asyncio.sleep(10)
                return {
                    "replica_id": replica_id,
                    "status": "cancelled",
                    "started_monotonic": started,
                    "finished_monotonic": time.monotonic(),
                }
            return {
                "replica_id": replica_id,
                "status": "valid",
                "started_monotonic": started,
                "finished_monotonic": time.monotonic(),
            }

        started = time.monotonic()
        result = await semantic_execution.collect_replicas(
            call, initial_fanout=4, hedge_seconds=1, required=2, maximum=4
        )
        return (result, time.monotonic() - started)

    (receipts, chosen, elapsed), collection_seconds = asyncio.run(scenario())
    assert [row["replica_id"] for row in chosen] == [0, 1]
    assert sum((row["status"] == "cancelled" for row in receipts)) == 2
    assert elapsed is not None and elapsed < 1
    assert collection_seconds < 0.1


def test_fixed_collection_releases_quorum_and_drains_every_replica() -> None:

    async def scenario():

        async def call(replica_id: int) -> dict:
            started = time.monotonic()
            await asyncio.sleep(0.001 if replica_id < 3 else 0.05)
            return {
                "replica_id": replica_id,
                "status": "valid",
                "started_monotonic": started,
                "finished_monotonic": time.monotonic(),
            }

        started = time.monotonic()
        (
            receipts,
            chosen,
            elapsed,
            drain,
        ) = await semantic_execution.collect_fixed_replicas(call, total=6, required=3)
        released = time.monotonic() - started
        late = await drain
        return (receipts, chosen, elapsed, late, released)

    receipts, chosen, elapsed, late, released = asyncio.run(scenario())
    assert [row["replica_id"] for row in chosen] == [0, 1, 2]
    assert [row["completion_rank"] for row in chosen] == [1, 2, 3]
    assert len(receipts) == 3 and len(late) == 3
    assert [row["completion_rank"] for row in late] == [4, 5, 6]
    assert elapsed is not None and elapsed < 0.02
    assert released < 0.02


def test_fixed_request_calls_each_declared_target_twice() -> None:

    class Client:
        def __init__(self, delay: float):
            self.delay = delay
            self.calls = 0
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            self.calls += 1
            await asyncio.sleep(self.delay)
            message = SimpleNamespace(
                content=json.dumps(_response(0.5, "valid")),
                reasoning_content="reasoning",
                reasoning=None,
                model_extra={},
            )
            return SimpleNamespace(
                id=f"generation-{self.calls}",
                model=kwargs["model"],
                usage=None,
                choices=[SimpleNamespace(message=message)],
            )

    async def scenario():
        clients = [Client(0.001), Client(0.01), Client(0.02)]
        validation = {
            "candidate_aliases": ["Candidate 1", "Candidate 2"],
            "candidate_bucket_ids": ["bucket-1", "bucket-2"],
            "anchor_bucket_ids": [],
            "selected_endpoint": "fixture",
            "task": "dili",
            "level": "L2",
        }
        row = {
            "request_id": "request-1",
            "kind": "weight_assignment",
            "prompt": "prompt",
            "reasoning_effort": "high",
            "max_tokens": 100,
            "validation_json": json.dumps(validation),
        }
        targets = [
            (client, f"model-{index}")
            for index, client in enumerate(clients)
            for _ in range(2)
        ]
        result, drain = await semantic_execution.execute_fixed_request(
            targets, row, required=3
        )
        late = await drain
        return (clients, result, late)

    clients, result, late = asyncio.run(scenario())
    assert [client.calls for client in clients] == [2, 2, 2]
    assert result["status"] == "complete"
    assert result["aggregate"]["replicate_count"] == 3
    assert len(result["receipts"]) + len(late) == 6


def test_persisted_aggregate_is_exported_with_uncertainty(tmp_path) -> None:
    connection = semantic_storage._request_database(
        tmp_path / "requests.sqlite3", journal_mode="DELETE"
    )
    validation = {
        "candidate_aliases": ["Candidate 1", "Candidate 2"],
        "candidate_bucket_ids": ["bucket-1", "bucket-2"],
        "anchor_bucket_ids": [],
        "selected_endpoint": "shared_provider_pool",
        "task": "dili",
        "level": "L2",
    }
    semantic_storage._queue_request(
        connection,
        request_id="request-1",
        kind="weight_assignment",
        phase="pass1/dili/L2/0000",
        prompt="prompt",
        reasoning_effort="high",
        max_tokens=100,
        validation=validation,
    )
    receipts = [_valid_receipt(index, 0.5 if index < 4 else 0.51) for index in range(8)]
    response, aggregate = semantic_bucket_scoring.aggregate_responses(
        receipts, validation["candidate_aliases"]
    )
    result = {
        "request_id": "request-1",
        "execution_id": "execution-1",
        "receipts": receipts,
        "time_to_required_seconds": 1.0,
        "status": "complete",
        "error": None,
        "response": response,
        "aggregate": aggregate,
        "aggregation_method": semantic_execution.AGGREGATION_METHOD,
    }
    semantic_storage.ensure_tables(connection)
    semantic_storage.persist_result(
        connection,
        result,
        base_url=semantic_execution.DEFAULT_BASE_URL,
        model=semantic_execution.DEFAULT_MODEL,
        benchmark_sha256="abc",
    )
    scores = (
        weights._scores(connection, "pass1")
        .set_index("semantic_bucket_id")
        .to_dict("index")
    )
    connection.close()
    assert scores["bucket-1"]["weight"] == 0.51
    assert scores["bucket-1"]["execution_backend"] == "shared_provider_pool"
    assert len(json.loads(scores["bucket-1"]["component_weights_json"])) == 8
    assert scores["bucket-1"]["weight_stddev"] > 0
    assert json.loads(scores["bucket-1"]["component_weights_json"]) == [
        0.5,
        0.5,
        0.5,
        0.5,
        0.51,
        0.51,
        0.51,
        0.51,
    ]


def test_late_receipts_complete_the_same_execution_record(tmp_path) -> None:
    connection = semantic_storage._request_database(
        tmp_path / "requests.sqlite3", journal_mode="DELETE"
    )
    validation = {
        "candidate_aliases": ["Candidate 1", "Candidate 2"],
        "candidate_bucket_ids": ["bucket-1", "bucket-2"],
        "anchor_bucket_ids": [],
        "selected_endpoint": "fixture",
        "task": "dili",
        "level": "L2",
    }
    semantic_storage._queue_request(
        connection,
        request_id="request-1",
        kind="weight_assignment",
        phase="pass2/L2/w0000/c00",
        prompt="prompt",
        reasoning_effort="high",
        max_tokens=100,
        validation=validation,
    )
    chosen = [_valid_receipt(index, 0.5) for index in range(3)]
    response, aggregate = semantic_bucket_scoring.aggregate_responses(
        chosen, validation["candidate_aliases"], required=3
    )
    result = {
        "request_id": "request-1",
        "execution_id": "execution-1",
        "receipts": chosen,
        "time_to_required_seconds": 1.0,
        "status": "complete",
        "error": None,
        "response": response,
        "aggregate": aggregate,
        "aggregation_method": "first_3_valid_mean_v1",
        "total_replica_count": 6,
    }
    late = [_valid_receipt(index, 0.6) for index in range(3, 6)]
    semantic_storage.ensure_tables(connection)
    semantic_storage.persist_result(
        connection,
        result,
        base_url="mixed://providers",
        model="mixed",
        benchmark_sha256="abc",
    )
    assert semantic_storage.all_replicas_stored(connection, "request-1") is False
    semantic_storage.persist_late_receipts(connection, result, late)
    assert semantic_storage.all_replicas_stored(connection, "request-1") is True
    replica_count = connection.execute(
        "SELECT COUNT(*) FROM speculative_replica_receipts"
    ).fetchone()[0]
    receipt = json.loads(
        connection.execute(
            "SELECT receipt_json FROM request_attempt_receipts"
        ).fetchone()[0]
    )
    connection.close()
    assert replica_count == 6
    assert receipt["all_replicas_stored"] is True
    assert receipt["final_replica_status_counts"] == {"valid": 6}


def test_first_four_uses_only_four_components() -> None:
    receipts = [
        _valid_receipt(index, weight)
        for index, weight in enumerate((0.4, 0.5, 0.6, 0.7))
    ]
    response, metadata = semantic_bucket_scoring.aggregate_responses(
        receipts, ["Candidate 1", "Candidate 2"], required=4
    )
    assert response["scores"][0]["weight"] == 0.55
    assert metadata["aggregation_method"] == "first_4_valid_mean_v1"
    assert metadata["replicate_count"] == 4
    assert metadata["components"]["Candidate 1"]["component_weights"] == [
        0.4,
        0.5,
        0.6,
        0.7,
    ]
