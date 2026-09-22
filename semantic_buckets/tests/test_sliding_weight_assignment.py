import asyncio
import json
import sqlite3

import pandas as pd
import pytest

from semantic_buckets import bioavailability_semantic_readout_v1 as core
from semantic_buckets import sliding_weight_assignment as weights


def _payload(bucket_id: str) -> dict:
    return {
        "source": "source_a",
        "identity": {
            "source_components": [
                {
                    "source_id": "source_a",
                    "canonical_dimensions": {
                        "canonical_endpoint": ["brain exposure"],
                        "canonical_unit": ["ratio"],
                    },
                }
            ]
        },
        "sample_records": [
            {
                "source_id": "source_a",
                "canonical_endpoint": "brain exposure",
                "canonical_unit": "ratio",
            }
        ],
        "hidden_bucket_id": bucket_id,
    }


def test_schedule_uses_shared_round_anchors_and_covers_tail_once() -> None:
    buckets = [f"bucket-{index:02d}" for index in range(55)]

    rows = weights._schedule_level("bbb_martins", "L2", buckets)

    assert json.loads(rows[0]["candidate_bucket_ids_json"]) == buckets[:12]
    assert all(json.loads(row["anchor_bucket_ids_json"]) == buckets[7:12] for row in rows[1:4])
    expected = buckets[17:19] + buckets[24:26] + buckets[31:33]
    assert all(json.loads(row["anchor_bucket_ids_json"]) == expected for row in rows[4:7])
    final_anchors = buckets[38:40] + buckets[45:47] + buckets[52:54]
    assert json.loads(rows[7]["anchor_bucket_ids_json"]) == final_anchors
    scheduled = [
        bucket
        for row in rows
        for bucket in json.loads(row["candidate_bucket_ids_json"])
    ]
    assert scheduled == buckets
    assert json.loads(rows[-1]["candidate_bucket_ids_json"]) == buckets[54:]


def test_rendered_cards_hide_bucket_identity_rank_counts_and_json() -> None:
    payloads = {"secret-bucket": _payload("secret-bucket")}

    prompt = weights._render(
        "bbb_martins", "L2", ["secret-bucket"], [], payloads, {}
    )

    assert "secret-bucket" not in prompt
    assert "hidden_bucket_id" not in prompt
    assert "pair bucket count" not in prompt
    assert "Candidate 1\n  Source: source_a" in prompt
    assert "canonical endpoint: brain exposure" in prompt
    assert not prompt.lstrip().startswith("{")
    assert "0.20–0.39: Weak utility" in prompt
    assert "0.80–0.94: Very strong utility" in prompt
    assert "reserve this range for unusually strong evidence" in prompt


def test_anchored_prompt_locks_anchor_and_returns_only_candidates() -> None:
    payloads = {name: _payload(name) for name in ("anchor-secret", "candidate-secret")}
    scores = {"anchor-secret": {"weight": 0.73, "rationale": "Useful reference."}}

    prompt = weights._render(
        "bbb_martins", "L3", ["candidate-secret"], ["anchor-secret"], payloads, scores
    )

    assert "Anchor 1 — locked weight 0.73" in prompt
    assert "Useful reference." in prompt
    assert "anchor-secret" not in prompt and "candidate-secret" not in prompt
    assert "no anchor entries" in prompt


def test_weight_response_requires_ordered_complete_hundredth_scores() -> None:
    validation = {"candidate_aliases": ["Candidate 1", "Candidate 2"]}
    result = {
        "scores": [
            {"candidate": "Candidate 1", "weight": 0.73, "rationale": "Direct."},
            {"candidate": "Candidate 2", "weight": 0.4, "rationale": "Indirect."},
        ]
    }

    assert core._validate_model_response("weight_assignment", result, validation) == result
    result["scores"][1]["weight"] = 0.405
    with pytest.raises(ValueError, match="hundredth increments"):
        core._validate_weight_response(result, validation)


def test_vast_request_database_uses_rollback_journal(tmp_path) -> None:
    connection = core._request_database(tmp_path / "requests.sqlite3", journal_mode="DELETE")
    try:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]
    finally:
        connection.close()

    assert mode == "delete"
    assert synchronous == 2


def test_empty_score_checkpoint_has_a_stable_schema() -> None:
    schedule = pd.DataFrame(
        [{
            "task": "bbb_martins", "level": "L2", "round": 0, "batch": 0,
            "candidate_bucket_ids_json": '["bucket"]',
            "candidate_level_ranks_json": "[1]",
        }]
    )

    frame = weights._score_frame({}, schedule)

    assert frame.empty
    assert {"semantic_bucket_id", "level_rank", "weight", "rationale"} <= set(frame)


def test_completion_progress_tracks_task_levels_without_a_global_barrier() -> None:
    schedule = pd.DataFrame([
        {"task": "bbb", "level": "L2", "round": 0,
         "candidate_bucket_ids_json": '["bbb-0"]'},
        {"task": "bbb", "level": "L2", "round": 1,
         "candidate_bucket_ids_json": '["bbb-1"]'},
        {"task": "oral", "level": "L3", "round": 0,
         "candidate_bucket_ids_json": '["oral-0"]'},
        {"task": "oral", "level": "L3", "round": 1,
         "candidate_bucket_ids_json": '["oral-1"]'},
    ])

    global_round, task_levels = weights._completion_progress(
        {bucket: {} for bucket in ("bbb-0", "bbb-1", "oral-0")}, schedule
    )

    assert global_round == 0
    assert task_levels == {"bbb/L2": 1, "oral/L3": 0}


def test_task_level_chains_advance_without_waiting_for_each_other(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = pd.DataFrame([
        {"batch_id": f"{task}-r{round_index:04d}", "task": task, "level": level,
         "round": round_index, "batch": 0,
         "candidate_bucket_ids_json": f'["{task}-{round_index}"]'}
        for task, level in (("bbb", "L2"), ("oral", "L3"))
        for round_index in (1, 2)
    ])
    events = []

    monkeypatch.setattr(weights, "_completed_scores", lambda connection: {})
    monkeypatch.setattr(weights, "_queue_round", lambda connection, rows, *args:
                        rows.batch_id.tolist())
    monkeypatch.setattr(weights.speculative, "_pending_rows", lambda connection, ids:
                        [{"request_id": request_id} for request_id in ids])
    monkeypatch.setattr(weights.speculative, "persist_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(weights, "_checkpoint_completed", lambda *args: {})
    monkeypatch.setattr(weights, "write_json_atomic", lambda *args: None)

    async def execute(client, row, **kwargs):
        request_id = row["request_id"]
        events.append(f"start {request_id}")
        await asyncio.sleep(0.05 if request_id == "oral-r0001" else 0.001)
        events.append(f"finish {request_id}")
        return {"request_id": request_id, "status": "complete"}

    monkeypatch.setattr(weights.speculative, "execute_request", execute)
    asyncio.run(weights._execute_task_level_chains(
        object(), schedule, {}, {}, tmp_path, object(),
        {"model": "model", "base_url": "url"}, "hash",
        {"hedge_seconds": 1}, 4, 16, None,
    ))

    assert events.index("finish bbb-r0002") < events.index("finish oral-r0001")


def test_execution_uses_pro_for_seed_and_flash_for_anchored_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "MODEL", "original")
    monkeypatch.setattr(core, "BASE_URL", "http://original/v1")
    monkeypatch.setattr(core, "ENDPOINTS", ())
    monkeypatch.setattr(core, "WEIGHT_REQUEST_EXTRA_BODY", {})

    assert weights._configure_execution(0, 9) == "seed"
    assert core.MODEL == "deepseek/deepseek-v4-pro"
    assert core.BASE_URL == "https://openrouter.ai/api/v1"
    assert core.ENDPOINTS[0]["provider"] == "openrouter"
    assert core.ENDPOINTS[0]["credential_env"] == "OPEN_ROUTER_KEY"
    assert core.ENDPOINTS[0]["max_inflight"] == 9
    assert core.WEIGHT_REQUEST_EXTRA_BODY == {"thinking": {"type": "enabled"}}

    assert weights._configure_execution(1, 27) == "anchored"
    assert core.MODEL == "deepseek/deepseek-v4-flash-0731"
    assert [endpoint["name"] for endpoint in core.ENDPOINTS] == [
        "openrouter_deepseek_v4_flash_0731",
    ]
    assert sum(endpoint["max_inflight"] for endpoint in core.ENDPOINTS) == 27
    assert core.ENDPOINTS[0]["provider"] == "openrouter"
    assert core.ENDPOINTS[0]["credential_env"] == "OPEN_ROUTER_KEY"
    assert core.WEIGHT_REQUEST_EXTRA_BODY == {"thinking": {"type": "enabled"}}


def test_openrouter_rotation_changes_provider_every_three_requests() -> None:
    snapshot = {"snapshot_sha256": "abc", "routes": [
        {"model": "model-a", "canonical_model": "canonical-a", "route_tag": "fast-a",
         "supports_response_format": True, "active_input_price": 0.1,
         "active_output_price": 0.2},
        {"model": "model-b", "canonical_model": "canonical-b", "route_tag": "fast-b",
         "supports_response_format": False, "active_input_price": 0.3,
         "active_output_price": 0.4},
    ]}
    assignments = [weights.provider_pool.request_assignment(
        snapshot, index, seed_request_count=9
    ) for index in range(9, 15)]

    assert [row["selected_provider_route"] for row in assignments] == [
        "fast-a", "fast-a", "fast-a", "fast-b", "fast-b", "fast-b",
    ]
    assert assignments[0]["requested_model"] == "model-a"
    assert assignments[0]["provider_routing"]["allow_fallbacks"] is False
    assert assignments[0]["provider_routing"]["max_price"]["completion"] < 0.66
    assert assignments[3]["provider_routing"]["omit_response_format"] is True


def _binding_fixture(tmp_path, monkeypatch: pytest.MonkeyPatch):
    original = tmp_path / "live" / "records.parquet"
    replacement = tmp_path / "legacy" / "records.parquet"
    original.parent.mkdir()
    replacement.parent.mkdir()
    replacement.write_bytes(b"prepared bytes")
    original.write_bytes(b"changed bytes")
    digest = weights._sha256(replacement)
    manifest = {
        "run_id": "run-v1",
        "inputs": {"task/pair_bucket_records": {
            "path": str(original), "sha256": digest,
        }},
    }
    binding = {
        "version": weights.INPUT_BINDINGS_VERSION,
        "run_id": "run-v1",
        "bindings": {"task/pair_bucket_records": {
            "original_path": str(original), "expected_sha256": digest,
            "replacement_path": str(replacement), "replacement_sha256": digest,
        }},
    }
    monkeypatch.setattr(weights, "LEGACY_ROOT", replacement.parent)
    return manifest, binding, replacement


def test_changed_prepared_input_requires_an_explicit_binding(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _, _ = _binding_fixture(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="prepared input changed"):
        weights._resolved_prepared_inputs(tmp_path, manifest)


def test_exact_legacy_input_binding_restores_prepared_bytes(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, binding, replacement = _binding_fixture(tmp_path, monkeypatch)
    (tmp_path / weights.INPUT_BINDINGS_NAME).write_text(json.dumps(binding))

    resolved = weights._resolved_prepared_inputs(tmp_path, manifest)

    assert resolved["task/pair_bucket_records"] == replacement


def test_legacy_input_binding_fails_on_replacement_hash_drift(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, binding, replacement = _binding_fixture(tmp_path, monkeypatch)
    (tmp_path / weights.INPUT_BINDINGS_NAME).write_text(json.dumps(binding))
    replacement.write_bytes(b"later replacement")

    with pytest.raises(ValueError, match="binding hash mismatch"):
        weights._resolved_prepared_inputs(tmp_path, manifest)


def test_all_bucket_payloads_uses_resolved_task_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = {}
    monkeypatch.setattr(weights, "TASK_TARGETS", {"task": "target"})
    monkeypatch.setattr(weights, "_bucket_payloads", lambda task, paths: seen.update(
        task=task, paths=paths
    ) or {"bucket": {}})

    payloads = weights._all_bucket_payloads({
        "task/pair_bucket_records": weights.Path("/legacy/records.parquet"),
        "task/semantic_map": weights.Path("/current/map.parquet"),
    })

    assert payloads == {"bucket": {}}
    assert seen == {"task": "task", "paths": {
        "pair_bucket_records": weights.Path("/legacy/records.parquet"),
        "semantic_map": weights.Path("/current/map.parquet"),
    }}


def test_queue_round_reuses_complete_request_without_rendering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = pd.DataFrame([{
        "batch_id": "task-L2-r0001-b00", "task": "task", "level": "L2",
        "round": 1, "candidate_bucket_ids_json": '["a", "b"]',
        "anchor_bucket_ids_json": "[]",
    }])
    scores = {
        bucket: {"request_id": "request-1"} for bucket in ("a", "b")
    }
    monkeypatch.setattr(weights, "_render", lambda *args: pytest.fail(
        "completed requests must not be rendered"
    ))
    connection = sqlite3.connect(":memory:")
    try:
        request_ids = weights._queue_round(connection, rows, {}, scores, {})
    finally:
        connection.close()

    assert request_ids == ["request-1"]


def test_local_load_preflight_accepts_vllm_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __init__(self, status_code: int, text: str = "") -> None:
            self.status_code, self.text = status_code, text

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)

    metrics = "\n".join([
        'vllm:num_requests_running{engine="0"} 2.0',
        'vllm:num_requests_running{engine="1"} 3.0',
        'vllm:num_requests_waiting{engine="0"} 0.0',
        'vllm:num_requests_waiting{engine="1"} 1.0',
    ])
    monkeypatch.setattr(weights.httpx, "get", lambda url, timeout: Response(
        404 if url.endswith("/loads") else 200, metrics
    ))

    receipt = weights._flash_load_receipts()[0]

    assert receipt["running_requests"] == 5
    assert receipt["waiting_requests"] == 1
    assert receipt["data_parallel_ranks"] == 2
