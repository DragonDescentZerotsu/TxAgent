import concurrent.futures
import threading
import time

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


def _spec(name, capacity):
    return ProviderSpec(
        name=name,
        base_url=f"http://{name}/v1",
        model="same-model",
        api_key_env="",
        max_inflight=capacity,
        initial_latency_s=1,
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
        providers=(_spec("a", 1), _spec("b", 1)),
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


def test_public_config_contains_only_credential_env_name():
    config = ProviderPoolConfig(providers=(_spec("a", 2),))

    public = config.public_dict()

    assert public["providers"][0]["api_key_env"] == ""
    assert "api_key" not in public["providers"][0]
