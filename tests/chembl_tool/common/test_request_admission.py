import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import pytest

from tools.chembl_tool.common.openai_provider_pool import (
    OpenAIProviderPool, ProviderPoolConfig, ProviderSpec,
)
from tools.chembl_tool.common.request_admission import admitted_call


def status(path):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(2)
        connection.connect(path)
        connection.sendall(b"status\n")
        return json.loads(connection.recv(4096))


@contextmanager
def broker(capacity):
    with tempfile.TemporaryDirectory(prefix="admission-") as directory:
        path = str(Path(directory) / "pool.sock")
        process = subprocess.Popen([
            sys.executable, "-m", "tools.chembl_tool.common.request_admission",
            "--socket", path, "--capacity", str(capacity),
        ], stdout=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 10
            while not Path(path).exists():
                assert process.poll() is None
                assert time.monotonic() < deadline
                time.sleep(0.01)
            yield path, process
        finally:
            process.terminate()
            process.wait(timeout=5)


async def settled(path, **expected):
    for _ in range(200):
        if all(status(path)[key] == value for key, value in expected.items()):
            return
        await asyncio.sleep(0.01)
    assert all(status(path)[key] == value for key, value in expected.items())


def test_independent_processes_share_capacity_before_http():
    async def run(path):
        active = peak = completed = 0

        async def endpoint(reader, writer):
            nonlocal active, peak, completed
            await reader.read(1)
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.04)
            writer.write(b"+")
            await writer.drain()
            active -= 1
            completed += 1
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(endpoint, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        script = '''
import asyncio, sys
from tools.chembl_tool.common.openai_provider_pool import OpenAIProviderPool, ProviderPoolConfig, ProviderSpec
class Client:
    async def async_chat_json(self, messages):
        reader, writer = await asyncio.open_connection('127.0.0.1', int(sys.argv[2]))
        try:
            writer.write(b'+'); await writer.drain()
            assert await reader.read(1) == b'+'
            return {'content': {'ok': True}, 'model': 'fake'}
        finally:
            writer.close(); await writer.wait_closed()
async def main():
    spec = ProviderSpec('fake', 'http://fake', 'fake', '', 8, timeout_s=2, admission_socket=sys.argv[1])
    pool = OpenAIProviderPool(ProviderPoolConfig((spec,)), client_factory=lambda _: Client())
    await asyncio.gather(*(pool.async_chat_json([]) for _ in range(8)))
asyncio.run(main())
'''
        children = []
        try:
            for _ in range(3):
                children.append(await asyncio.create_subprocess_exec(
                    sys.executable, "-c", script, path, str(port)))
            assert await asyncio.wait_for(asyncio.gather(*(p.wait() for p in children)), 15) == [0, 0, 0]
            assert completed == 24
            assert peak == 2
            await settled(path, active=0, waiting=0)
            assert status(path)["peak_active"] == 2
        finally:
            for child in children:
                if child.returncode is None:
                    child.kill()
                    await child.wait()
            server.close()
            await server.wait_closed()

    with broker(2) as (path, _):
        asyncio.run(run(path))


def test_waiting_does_not_consume_provider_deadline_and_cancel_returns_permits():
    async def run(path):
        hold = asyncio.Event()

        async def held_request():
            await hold.wait()

        holder = asyncio.create_task(admitted_call(path, held_request, timeout=None, on_admitted=lambda: None))
        await settled(path, active=1)

        class Client:
            called = False

            async def async_chat_json(self, messages):
                self.called = True
                return {"content": {"ok": True}, "model": "fake"}

        client = Client()
        spec = ProviderSpec("fake", "http://fake", "fake", "", 4, timeout_s=1, admission_socket=path)
        pool = OpenAIProviderPool(ProviderPoolConfig((spec,)), client_factory=lambda _: client)
        queued = asyncio.create_task(pool.async_chat_json([]))
        cancelled = asyncio.create_task(pool.async_chat_json([]))
        await settled(path, waiting=2)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        await settled(path, waiting=1)
        await asyncio.sleep(1.1)
        assert not queued.done() and not client.called
        holder.cancel()
        with pytest.raises(asyncio.CancelledError):
            await holder
        response = await asyncio.wait_for(queued, 2)
        assert response["execution_provider"]["admission_wait_seconds"] >= 1.1
        assert response["execution_provider"]["latency_seconds"] < 1
        assert pool.snapshot()["providers"][0]["inflight"] == 0
        await settled(path, active=0, waiting=0)

    with broker(1) as (path, _):
        asyncio.run(run(path))


def test_process_exit_releases_permit_and_broker_loss_cancels_request():
    async def run(path, process):
        script = '''
import sys, time
from tools.chembl_tool.common.request_admission import admitted_sync
with admitted_sync(sys.argv[1]):
    print('acquired', flush=True)
    time.sleep(60)
'''
        holder = await asyncio.create_subprocess_exec(
            sys.executable, "-c", script, path, stdout=asyncio.subprocess.PIPE)
        try:
            assert await asyncio.wait_for(holder.stdout.readline(), 5) == b"acquired\n"
            holder.kill()
            await holder.wait()
            await settled(path, active=0)
        finally:
            if holder.returncode is None:
                holder.kill()
                await holder.wait()
        cancelled = asyncio.Event()

        async def request():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        call = asyncio.create_task(admitted_call(path, request, timeout=None, on_admitted=lambda: None))
        await settled(path, active=1)
        await asyncio.sleep(0.02)
        process.kill()
        with pytest.raises(ConnectionError, match="broker lost"):
            await asyncio.wait_for(call, 2)
        assert cancelled.is_set()

    with broker(1) as (path, process):
        asyncio.run(run(path, process))
