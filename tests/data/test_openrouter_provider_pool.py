from datetime import datetime, timezone
import json
from types import SimpleNamespace

from data.processing import openrouter_provider_pool as pool


def test_active_completion_price_applies_utc_overrides() -> None:
    pricing = {
        "completion": "0.0000006",
        "overrides": [
            {"utc_days": ["thursday"], "utc_start": 100, "utc_end": 400,
             "completion": "0.0000012"},
            {"utc_days": ["thursday"], "utc_start": 400, "utc_end": 0,
             "completion": "0.0000006"},
        ],
    }

    assert pool.active_completion_price(
        pricing, datetime(2026, 9, 17, 2, tzinfo=timezone.utc)
    ) == 1.2
    assert pool.active_completion_price(
        pricing, datetime(2026, 9, 17, 17, tzinfo=timezone.utc)
    ) == 0.6


def test_route_without_live_tps_is_not_eligible() -> None:
    route = pool._route_record(
        "model", "canonical", "route", [{
            "provider_name": "Provider", "supported_parameters": ["reasoning_effort"],
            "quantization": "fp8", "pricing": {
                "prompt": "0.0000001", "completion": "0.0000002",
            },
            "context_length": 1_000_000, "max_completion_tokens": 100_000,
            "status": 0,
        }], datetime(2026, 9, 22, tzinfo=timezone.utc),
    )

    assert route["throughput_p50"] is None
    assert pool._eligible(route) is False


def test_default_model_is_single_and_mixed_profile_is_explicit() -> None:
    assert pool._models(False) == (pool.DEFAULT_MODEL,)
    assert pool._models(True) == (pool.DEFAULT_MODEL, pool.MIXED_MODEL)


def test_gold_requires_three_successes_and_uses_median_token_length(tmp_path) -> None:
    connection = pool._database(tmp_path / "requests.sqlite3")
    route = {"route_key": "model|provider", "model": "model", "canonical_model": "model",
             "route_tag": "provider", "provider_name": "Provider", "quantizations": ["fp8"],
             "supported_parameters": ["reasoning_effort"], "fingerprint_sha256": "hash",
             "active_output_price": 0.2, "throughput_p50": 50.0, "healthy": True,
             "supports_response_format": True, "supports_reasoning_effort": True,
             "inventory_records": 1}
    for repetition, tokens in enumerate((4_096, 15_000, 17_000), 1):
        pool._store_attempt(connection, {
            "route_key": route["route_key"], "repetition": repetition, "status": "complete",
            "started_at": "start", "finished_at": "finish", "elapsed_seconds": 1.0,
            "completion_tokens": tokens, "reasoning_tokens": tokens,
            "response_json": "{}", "receipt_json": "{}", "error": None,
        })

    assert pool._result_rows(connection, [route])[0]["gold"] is True
    connection.execute(
        "UPDATE attempts SET status='failed',error='429' WHERE repetition=1"
    )
    connection.commit()
    assert pool._result_rows(connection, [route])[0]["gold"] is False
    connection.close()


def test_weighted_ranking_selects_top_seven_with_tps_dominant() -> None:
    routes = [
        {"route_key": str(index), "model": "model", "route_tag": str(index),
         "throughput_p50": float(index + 1), "active_output_price": 0.60 - index * 0.01}
        for index in range(9)
    ]

    ranked = pool._rank_routes(routes)

    assert len(ranked) == 7
    assert ranked[0]["route_key"] == "8"
    assert ranked[0]["composite_score"] == 1.0


def test_request_assignment_rotates_every_three_requests() -> None:
    snapshot = {"snapshot_sha256": "snapshot", "routes": [
        {"model": "m1", "canonical_model": "c1", "route_tag": "one",
         "supports_response_format": True, "active_input_price": 0.1,
         "active_output_price": 0.2},
        {"model": "m2", "canonical_model": "c2", "route_tag": "two",
         "supports_response_format": False, "active_input_price": 0.1,
         "active_output_price": 0.2},
    ]}

    assignments = [pool.request_assignment(snapshot, index, seed_request_count=9)
                   for index in range(9, 18)]

    assert [row["selected_provider_route"] for row in assignments] == (
        ["one"] * 3 + ["two"] * 3 + ["one"] * 3
    )
    assert assignments[3]["requested_model"] == "m2"
    assert assignments[3]["provider_routing"]["omit_response_format"] is True
    assert assignments[0]["provider_routing"]["max_price"]["completion"] < 0.66


def test_exported_mixed_pool_has_one_shared_capacity_and_pinned_routes(
    tmp_path, monkeypatch
) -> None:
    routes = [
        {
            "model": model,
            "canonical_model": model + "-canonical",
            "route_tag": f"route-{index}",
            "provider_name": f"Provider {index}",
            "supports_response_format": index > 0,
            "active_input_price": 0.1,
            "active_output_price": 0.2,
            "max_request_cost_usd": 0.25,
        }
        for index, model in enumerate(
            (pool.DEFAULT_MODEL, pool.MIXED_MODEL, pool.DEFAULT_MODEL)
        )
    ]
    monkeypatch.setattr(
        pool,
        "load_ranked_pool",
        lambda mixed: {
            "profile": "mixed" if mixed else "0731",
            "snapshot_sha256": "snapshot",
            "qualification": {"run_id": "gold"},
            "score": {"tps_weight": 2 / 3, "output_price_weight": 1 / 3},
            "routes": routes,
        },
    )

    payload = pool.export_provider_pool(
        tmp_path / "pool.json", 512, allow_mixed_flash_models=True,
        spend_budget={
            "version": "openrouter_spend_budget.v1",
            "ledger_path": "/ledger.sqlite3",
            "epoch": "epoch-1",
            "limit_usd": 25,
            "credential_env": "OPEN_ROUTER_KEY_TWO",
        },
    )

    assert sum(row["max_inflight"] for row in payload["providers"]) == 512
    assert {row["model"] for row in payload["providers"]} == {
        pool.DEFAULT_MODEL,
        pool.MIXED_MODEL,
    }
    assert payload["openrouter_ranked_profile"]["snapshot_sha256"] == "snapshot"
    assert payload["providers"][0]["request_extra_body"]["omit_response_format"] is True
    assert payload["providers"][0]["request_extra_body"]["provider"]["max_price"] == {
        "prompt": 0.1, "completion": 0.2,
    }
    assert {row["api_key_env"] for row in payload["providers"]} == {
        "OPEN_ROUTER_KEY_TWO"
    }
    assert payload["spend_budget"]["limit_usd"] == 25


def test_qualification_uses_inline_openrouter_routing_metadata() -> None:
    route = {"model": "requested", "canonical_model": "canonical",
             "route_tag": "provider/fp8", "provider_name": "Provider",
             "supports_response_format": False}
    message = SimpleNamespace(
        content=json.dumps({"scores": [{"candidate": "Candidate 1", "weight": 0.5,
                                        "rationale": "valid"}]}),
        reasoning_content="reasoning", reasoning=None, model_extra={},
    )
    usage = SimpleNamespace(model_dump=lambda: {
        "completion_tokens": 4_500,
        "completion_tokens_details": {"reasoning_tokens": 4_096},
    })
    completion = SimpleNamespace(
        id="generation", model="canonical", usage=usage,
        model_extra={"openrouter_metadata": {"endpoints": {"available": [
            {"provider": "Provider", "selected": True}
        ]}}}, choices=[SimpleNamespace(message=message)],
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_kwargs: completion
    )))

    result = pool._call_route(client, route, "prompt", 1)

    assert result["completion_tokens"] == 4_500
    assert result["reasoning_tokens"] == 4_096
    assert result["generation_id"] == "generation"
