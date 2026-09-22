import concurrent.futures
import io
import json
import threading
import time
from types import SimpleNamespace

import pytest

from predict.api_client import pool as provider_pool
from predict.api_client.client import OpenAICompatibleClient

from tools.chembl_tool.common.openai_provider_pool import (
    OpenAIProviderPool,
    ProviderPoolConfig,
    ProviderSpec,
)


class _BlockingClient:
    def __init__(self, name, started, release):
        self.name = name
        self.started = started
        self.release = release

    def chat_json(self, messages):
        self.started.append(self.name)
        self.release.wait(timeout=2)
        return {"content": {"ok": True}, "model": self.name, "id": self.name}


def _spec(name, capacity, *, priority=0):
    return ProviderSpec(
        name=name,
        base_url=f"http://{name}/v1",
        model="same-model",
        api_key_env="",
        max_inflight=capacity,
        initial_latency_s=1,
        priority=priority,
    )


def test_pool_fills_provider_capacities_without_exceeding_them():
    started = []
    release = threading.Event()
    config = ProviderPoolConfig(providers=(_spec("a", 2), _spec("b", 1)))
    pool = OpenAIProviderPool(
        config,
        client_factory=lambda spec: _BlockingClient(spec.name, started, release),
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(pool.chat_json, []) for _ in range(3)]
        deadline = time.monotonic() + 2
        while len(started) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert sorted(started) == ["a", "a", "b"]
        snapshot = pool.snapshot()
        assert {row["name"]: row["inflight"] for row in snapshot["providers"]} == {
            "a": 2,
            "b": 1,
        }
        release.set()
        assert all(future.result()["content"]["ok"] for future in futures)


def test_pool_fails_over_and_opens_transport_failure_circuit():
    class FailingClient:
        def chat_json(self, messages):
            raise TimeoutError("slow provider")

    class WorkingClient:
        def chat_json(self, messages):
            return {"content": {"ok": True}, "model": "same-model", "id": "ok"}

    config = ProviderPoolConfig(
        providers=(_spec("a", 1), _spec("b", 1, priority=1)),
        failure_threshold=1,
        cooldown_seconds=30,
        max_failovers=1,
    )
    pool = OpenAIProviderPool(
        config,
        client_factory=lambda spec: FailingClient() if spec.name == "a" else WorkingClient(),
    )

    response = pool.chat_json([])

    assert response["execution_provider"]["provider"] == "b"
    assert [row["provider"] for row in response["execution_provider_attempts"]] == [
        "a",
        "b",
    ]
    providers = {row["name"]: row for row in pool.snapshot()["providers"]}
    assert providers["a"]["failures"] == 1
    assert providers["a"]["circuit_open"] is True
    assert providers["b"]["completed"] == 1


def test_lower_priority_provider_is_fallback_only():
    started = []
    release = threading.Event()
    config = ProviderPoolConfig(
        providers=(_spec("primary", 1), _spec("fallback", 1, priority=1))
    )
    pool = OpenAIProviderPool(
        config,
        client_factory=lambda spec: _BlockingClient(spec.name, started, release),
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(pool.chat_json, []) for _ in range(2)]
        deadline = time.monotonic() + 1
        while not started and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.05)
        assert started == ["primary"]
        release.set()
        assert all(future.result()["content"]["ok"] for future in futures)
    assert started == ["primary", "primary"]


def test_public_config_contains_only_credential_env_name():
    config = ProviderPoolConfig(providers=(_spec("a", 2),))

    public = config.public_dict()

    assert public["providers"][0]["api_key_env"] == ""
    assert "api_key" not in public["providers"][0]


def test_pool_forwards_per_call_token_ceiling():
    observed = []

    class RecordingClient:
        def chat_json(self, messages, *, max_tokens=None):
            observed.append(max_tokens)
            return {"content": {"ok": True}, "model": "same-model", "id": "ok"}

    pool = OpenAIProviderPool(
        ProviderPoolConfig(providers=(_spec("a", 1),)),
        client_factory=lambda spec: RecordingClient(),
    )

    pool.chat_json([], max_tokens=65_536)

    assert observed == [65_536]


def test_pool_records_upstream_provider_when_returned():
    class RecordingClient:
        def chat_json(self, messages, *, max_tokens=None):
            return {
                "content": {"ok": True},
                "model": "same-model",
                "id": "ok",
                "provider": "Baidu",
            }

    pool = OpenAIProviderPool(
        ProviderPoolConfig(providers=(_spec("openrouter", 1),)),
        client_factory=lambda spec: RecordingClient(),
    )

    response = pool.chat_json([])

    assert response["execution_provider"]["upstream_provider"] == "Baidu"


def test_build_pool_applies_provider_reasoning_override(monkeypatch):
    from data.processing import llm_api
    from predict.api_client import client as client_module

    created = []
    monkeypatch.setattr(
        llm_api, "openai_compatible_client", lambda **_kwargs: (object(), "KEY")
    )
    monkeypatch.setattr(
        client_module, "OpenAICompatibleClient",
        lambda **kwargs: created.append(kwargs) or object(),
    )
    config = ProviderPoolConfig(providers=(ProviderSpec(
        name="flex", base_url="https://api.openai.com/v1",
        model="gpt-6-luna", api_key_env="OPENAI_API_KEY_ONE", max_inflight=1,
        request_extra_body={
            "reasoning_effort_override": "medium", "service_tier": "flex"
        },
    ),))

    provider_pool.build_provider_pool(
        config, env_file=None, timeout_s=60, max_tokens=100,
        temperature=None, tool_service_url="http://127.0.0.1:1",
        enable_group_tools=False, max_tool_rounds=0,
        reasoning_effort="high", enable_thinking=False,
        transport_max_retries=0,
    )

    assert created[0]["reasoning_effort"] == "medium"
    assert created[0]["request_extra_body"] == {"service_tier": "flex"}


def test_shared_spend_ledger_reserves_then_reconciles_actual_cost(tmp_path):
    calls = []

    class CostedClient:
        def chat_json(self, messages, *, max_tokens=None):
            calls.append(messages)
            return {
                "content": {"ok": True}, "model": "same-model", "id": "ok",
                "usage": {"cost": 0.4},
            }

    spec = ProviderSpec(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="same-model",
        api_key_env="OPEN_ROUTER_KEY",
        max_inflight=2,
        request_extra_body={"max_request_cost_usd": 0.6},
    )
    budget = provider_pool.SpendBudgetSpec(
        ledger_path=str(tmp_path / "spend.sqlite3"),
        epoch="epoch-1",
        limit_usd=1.0,
        credential_env="OPEN_ROUTER_KEY",
    )
    pool = OpenAIProviderPool(
        ProviderPoolConfig(providers=(spec,), spend_budget=budget),
        client_factory=lambda _: CostedClient(),
    )

    pool.chat_json([{"role": "user", "content": "one"}])
    second = pool.chat_json([{"role": "user", "content": "two"}])

    assert len(calls) == 2
    assert second["execution_provider"]["spend_budget"]["committed_usd"] == 0.8
    assert pool.snapshot()["spend_budget"]["remaining_usd"] == pytest.approx(0.2)


def test_spend_ledger_waits_for_temporary_reservations(tmp_path):
    started = []
    release = threading.Event()

    class BlockingCostedClient:
        def chat_json(self, messages, *, max_tokens=None):
            started.append(messages[0]["content"])
            if len(started) == 1:
                release.wait(timeout=2)
            return {
                "content": {"ok": True}, "model": "same-model", "id": "ok",
                "usage": {"cost": 0.1},
            }

    spec = ProviderSpec(
        name="openrouter", base_url="https://openrouter.ai/api/v1",
        model="same-model", api_key_env="OPEN_ROUTER_KEY", max_inflight=2,
        request_extra_body={"max_request_cost_usd": 0.6},
    )
    budget = provider_pool.SpendBudgetSpec(
        ledger_path=str(tmp_path / "spend.sqlite3"), epoch="epoch-1",
        limit_usd=0.7, credential_env="OPEN_ROUTER_KEY",
    )
    pool = OpenAIProviderPool(
        ProviderPoolConfig(providers=(spec,), spend_budget=budget),
        client_factory=lambda _: BlockingCostedClient(),
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            pool.chat_json, [{"role": "user", "content": "one"}]
        )
        deadline = time.monotonic() + 1
        while not started and time.monotonic() < deadline:
            time.sleep(0.01)
        second = executor.submit(
            pool.chat_json, [{"role": "user", "content": "two"}]
        )
        time.sleep(0.1)
        assert started == ["one"]
        release.set()
        assert first.result()["content"]["ok"] is True
        assert second.result()["content"]["ok"] is True

    assert pool.snapshot()["spend_budget"]["committed_usd"] == 0.2


def test_spend_ledger_alerts_and_resumes_after_explicit_refresh(tmp_path):
    calls = []

    class CostedClient:
        def chat_json(self, messages, *, max_tokens=None):
            calls.append(messages)
            return {
                "content": {"ok": True}, "model": "same-model", "id": "ok",
                "usage": {"cost": 0.3},
            }

    spec = ProviderSpec(
        name="openrouter", base_url="https://openrouter.ai/api/v1",
        model="same-model", api_key_env="OPEN_ROUTER_KEY", max_inflight=1,
        request_extra_body={"max_request_cost_usd": 0.4},
    )
    budget = provider_pool.SpendBudgetSpec(
        ledger_path=str(tmp_path / "spend.sqlite3"), epoch="epoch-1",
        limit_usd=0.5, credential_env="OPEN_ROUTER_KEY",
    )
    config = ProviderPoolConfig(providers=(spec,), spend_budget=budget)
    pool = OpenAIProviderPool(config, client_factory=lambda _: CostedClient())
    pool.chat_json([{"role": "user", "content": "one"}])

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        waiting = executor.submit(
            pool.chat_json, [{"role": "user", "content": "two"}]
        )
        alert = tmp_path / "spend.sqlite3.alert.json"
        deadline = time.monotonic() + 2
        while not alert.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert alert.is_file()
        assert json.loads(alert.read_text())["status"] == "budget_refresh_required"
        assert waiting.done() is False
        refreshed = provider_pool.refresh_spend_budget(config)
        assert refreshed["authorized_usd"] == 1.0
        assert waiting.result(timeout=2)["content"]["ok"] is True

    assert alert.exists() is False
    assert len(calls) == 2


def test_spend_budget_rejects_hidden_transport_retries(tmp_path):
    spec = ProviderSpec(
        name="openrouter", base_url="https://openrouter.ai/api/v1",
        model="same-model", api_key_env="OPEN_ROUTER_KEY", max_inflight=1,
        request_extra_body={"max_request_cost_usd": 0.6},
    )
    config = ProviderPoolConfig(
        providers=(spec,),
        spend_budget=provider_pool.SpendBudgetSpec(
            ledger_path=str(tmp_path / "spend.sqlite3"), epoch="epoch-1",
            limit_usd=25, credential_env="OPEN_ROUTER_KEY",
        ),
    )

    with pytest.raises(ValueError, match="transport_max_retries=0"):
        provider_pool.build_provider_pool(
            config,
            env_file=None,
            timeout_s=60,
            max_tokens=100,
            temperature=0.0,
            tool_service_url="",
            enable_group_tools=False,
            max_tool_rounds=0,
            reasoning_effort="high",
            enable_thinking=True,
            transport_max_retries=1,
        )


def test_openrouter_control_fields_are_validated_but_not_sent():
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))],
                usage=None,
                model="canonical-model",
                id="request",
                provider="Expected Provider",
            )

    transport = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    spec = ProviderSpec(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="requested-model",
        api_key_env="OPEN_ROUTER_KEY",
        max_inflight=1,
        request_extra_body={
            "omit_response_format": True,
            "allowed_served_models": ["requested-model", "canonical-model"],
            "expected_upstream_provider": "Expected Provider",
            "provider_pool_snapshot_sha256": "snapshot",
            "provider": {"order": ["expected"]},
        },
    )
    client = OpenAICompatibleClient(
        api_key="loaded",
        base_url=spec.base_url,
        model=spec.model,
        timeout_s=60,
        max_tokens=100,
        temperature=None,
        tool_service_url="",
        enable_group_tools=False,
        max_tool_rounds=0,
        reasoning_effort="high",
        enable_thinking=False,
        transport_max_retries=0,
        request_extra_body=spec.request_extra_body,
        openai_client=transport,
    )
    pool = OpenAIProviderPool(
        ProviderPoolConfig(providers=(spec,)), client_factory=lambda _: client
    )

    response = pool.chat_json([])

    assert "response_format" not in captured
    assert captured["extra_body"] == {"provider": {"order": ["expected"]}}
    assert response["execution_provider"]["provider_pool_snapshot_sha256"] == "snapshot"


def test_endpoint_selection_skips_dead_and_wrong_model_candidates(monkeypatch):
    config = ProviderPoolConfig(
        providers=(_spec("healthy", 2), _spec("wrong", 2), _spec("dead", 2))
    )

    def fake_open(url, timeout):
        if "dead" in url:
            raise TimeoutError("offline")
        model = "different-model" if "wrong" in url else "same-model"
        return io.BytesIO(json.dumps({"data": [{"id": model}]}).encode())

    monkeypatch.setattr(provider_pool, "urlopen", fake_open)
    selection = provider_pool.select_healthy_providers(config, 5)

    assert [spec.name for spec in selection.config.providers] == ["healthy"]
    assert selection.requested_parallelism == 5
    assert selection.effective_parallelism == 2
    assert [row["status"] for row in selection.checks] == [
        "healthy",
        "model_mismatch",
        "unavailable",
    ]


def test_endpoint_selection_fails_only_when_none_are_compatible(monkeypatch):
    config = ProviderPoolConfig(providers=(_spec("dead", 1),))
    monkeypatch.setattr(
        provider_pool,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("offline")),
    )

    with pytest.raises(ValueError, match="no healthy exact-model provider"):
        provider_pool.select_healthy_providers(config, 1)
