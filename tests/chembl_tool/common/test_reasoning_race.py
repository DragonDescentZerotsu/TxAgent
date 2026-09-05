import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from tools.chembl_tool.common.openai_provider_pool import (
    OpenAIProviderPool, ProviderPoolConfig, ProviderPoolExhausted, ProviderSpec,
)
from tools.chembl_tool.common.openai_reasoning_client import OpenAICompatibleClient
from tools.chembl_tool.common.reasoning_race import ParallelRetryClient


def test_race_ignores_fast_invalid_response_and_awaits_loser_cancellation(tmp_path):
    class Pool:
        started = 0
        cancelled = 0

        async def async_chat_json(self, messages):
            self.started += 1
            number = self.started
            try:
                await asyncio.sleep(0.01 if number == 1 else 0.04 if number == 2 else 10)
                return {"content": {"valid": number == 2}, "id": str(number)}
            except asyncio.CancelledError:
                self.cancelled += 1
                raise

        async def aclose(self):
            pass

    pool = Pool()
    client = ParallelRetryClient(pool, parallelism=6, task_limits={"a": 6})
    try:
        result = client.chat_validated([], task="a", width=6,
                                       validate=lambda r: [] if r["content"]["valid"] else ["invalid"],
                                       receipt_path=tmp_path / "race.json")
        assert result["id"] == "2"
        assert pool.started == 6 and pool.cancelled == 4
        receipt = json.loads((tmp_path / "race.json").read_text())
        assert receipt["winner"] == 2
        assert Counter(r["status"] for r in receipt["attempts"]) == {"invalid": 1, "valid": 1, "cancelled": 4}
    finally:
        client.close()


def test_multiple_races_share_global_and_per_task_slots(tmp_path):
    class Pool:
        active = Counter()
        peaks = Counter()
        total_peak = 0

        async def async_chat_json(self, messages):
            task = messages[0]["content"]
            self.active[task] += 1
            self.peaks[task] = max(self.peaks[task], self.active[task])
            self.total_peak = max(self.total_peak, sum(self.active.values()))
            try:
                await asyncio.sleep(0.02)
                raise TimeoutError("test transport error")
            finally:
                self.active[task] -= 1

        async def aclose(self):
            pass

    pool = Pool()
    client = ParallelRetryClient(pool, parallelism=4, task_limits={"a": 3, "b": 3})

    def run(task):
        with pytest.raises(ProviderPoolExhausted):
            client.chat_validated([{"content": task}], task=task, width=6,
                                  validate=lambda _: [], receipt_path=tmp_path / f"{task}.json")

    try:
        with ThreadPoolExecutor(2) as executor:
            list(executor.map(run, ["a", "b"]))
        assert pool.total_peak <= 4
        assert all(value <= 3 for value in pool.peaks.values())
        assert sum(pool.active.values()) == 0
        for task in ("a", "b"):
            receipt = json.loads((tmp_path / f"{task}.json").read_text())
            assert len(receipt["attempts"]) == 6
            assert all(row["status"] == "error" for row in receipt["attempts"])
    finally:
        client.close()


def test_close_stops_event_loop_even_if_provider_cleanup_fails():
    class Pool:
        async def aclose(self):
            raise RuntimeError("cleanup failed")

    client = ParallelRetryClient(Pool(), parallelism=1, task_limits={"a": 1})
    with pytest.raises(RuntimeError, match="cleanup failed"):
        client.close()
    assert not client.thread.is_alive()
    assert client.loop.is_closed()


def test_real_http_race_closes_five_losing_connections():
    async def scenario():
        arrived, disconnected = set(), set()
        ready = asyncio.Event()

        async def serve(reader, writer):
            try:
                headers = await reader.readuntil(b"\r\n\r\n")
                size = next(int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n")
                            if line.lower().startswith(b"content-length:"))
                await reader.readexactly(size)
                number = len(arrived) + 1
                arrived.add(number)
                if len(arrived) == 6:
                    ready.set()
                await ready.wait()
                if number == 1:
                    body = json.dumps({"id": "winner", "object": "chat.completion", "created": 0,
                                       "model": "test", "choices": [{"index": 0, "finish_reason": "stop",
                                       "message": {"role": "assistant", "content": "{\"ok\":true}"}}]}).encode()
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                                 + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
                    await writer.drain()
                else:
                    assert await reader.read() == b""
                    disconnected.add(number)
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        spec = ProviderSpec("test", f"http://127.0.0.1:{port}/v1", "test", "", 6)

        def factory(spec):
            return OpenAICompatibleClient(api_key="test", base_url=spec.base_url, model="test",
                                          timeout_s=5, max_tokens=128, temperature=0,
                                          tool_service_url="http://unused", enable_group_tools=False,
                                          max_tool_rounds=0, reasoning_effort="", enable_thinking=False,
                                          transport_max_retries=0)

        pool = OpenAIProviderPool(ProviderPoolConfig((spec,), max_failovers=0), client_factory=factory)
        client = ParallelRetryClient.__new__(ParallelRetryClient)
        client.pool = pool
        client.slots = asyncio.Semaphore(6)
        client.task_slots = {"a": asyncio.Semaphore(6)}
        try:
            result, receipt = await asyncio.wait_for(client._race([], task="a", width=6,
                                                     validate=lambda r: [] if r["content"].get("ok") else ["bad"]), 8)
            assert result["id"] == "winner"
            for _ in range(100):
                if len(disconnected) == 5:
                    break
                await asyncio.sleep(0.01)
            assert len(disconnected) == 5
            snapshot = pool.snapshot()["providers"][0]
            assert snapshot["inflight"] == 0 and snapshot["cancelled"] == 5
            assert snapshot["failures"] == 0
        finally:
            await pool.aclose()
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())
