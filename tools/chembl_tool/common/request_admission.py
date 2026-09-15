"""Host-local request admission shared by independent inference processes.

A connection owns one permit until it closes. Waiting happens before the HTTP
deadline; cancellation and process exit release permits without lease timers.
Run one broker for the lifetime of all participating launchers.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import socket


@contextmanager
def admitted_sync(path: str):
    if not path:
        yield
        return
    with socket.socket(socket.AF_UNIX) as connection:
        connection.connect(path)
        connection.sendall(b"acquire\n")
        if connection.recv(1) != b"+":
            raise ConnectionError("request admission broker closed before grant")
        yield


async def admitted_call(path, call, *, timeout, on_admitted):
    """Call after admission; cancel HTTP work if the broker connection is lost."""
    if not path:
        on_admitted()
        return await asyncio.wait_for(call(), timeout=timeout)
    reader, writer = await asyncio.open_unix_connection(path)
    request = disconnected = None
    try:
        writer.write(b"acquire\n")
        await writer.drain()
        if await reader.read(1) != b"+":
            raise ConnectionError("request admission broker closed before grant")
        on_admitted()
        request = asyncio.create_task(asyncio.wait_for(call(), timeout=timeout))
        disconnected = asyncio.create_task(reader.read(1))
        done, _ = await asyncio.wait((request, disconnected), return_when=asyncio.FIRST_COMPLETED)
        if disconnected in done:
            raise ConnectionError("request admission broker lost during request")
        return await request
    finally:
        # Await HTTP cancellation before returning the permit to the broker.
        pending = [task for task in (request, disconnected) if task is not None]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        writer.close()
        await writer.wait_closed()


class AdmissionBroker:
    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("admission capacity must be positive")
        self.capacity = capacity
        self.permits = asyncio.Semaphore(capacity)
        self.active = self.waiting = self.peak_active = self.granted = 0

    def snapshot(self):
        return {key: getattr(self, key) for key in
                ("capacity", "active", "waiting", "peak_active", "granted")}

    async def handle(self, reader, writer):
        acquire = disconnected = None
        waiting = False
        held = False
        try:
            command = await asyncio.wait_for(reader.readline(), timeout=10)
            if command == b"status\n":
                writer.write(json.dumps(self.snapshot()).encode() + b"\n")
                await writer.drain()
                return
            if command != b"acquire\n":
                return
            self.waiting += 1
            waiting = True
            acquire = asyncio.create_task(self.permits.acquire())
            disconnected = asyncio.create_task(reader.read(1))
            done, _ = await asyncio.wait((acquire, disconnected), return_when=asyncio.FIRST_COMPLETED)
            if disconnected in done:
                return
            await acquire
            self.waiting -= 1
            waiting = False
            held = True
            self.active += 1
            self.granted += 1
            self.peak_active = max(self.peak_active, self.active)
            writer.write(b"+")
            await writer.drain()
            await disconnected
        except (ConnectionError, OSError, asyncio.TimeoutError):
            pass
        finally:
            if waiting:
                self.waiting -= 1
            tasks = [task for task in (acquire, disconnected) if task is not None]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # The semaphore may have granted concurrently with disconnection.
            if acquire is not None and not acquire.cancelled() and acquire.exception() is None:
                self.permits.release()
            if held:
                self.active -= 1
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass


async def serve(path: Path, capacity: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path.unlink(missing_ok=True)
        broker = AdmissionBroker(capacity)
        server = await asyncio.start_unix_server(broker.handle, path=str(path), backlog=8192)
        path.chmod(0o600)
        print(json.dumps({"socket": str(path), **broker.snapshot()}), flush=True)
        try:
            async with server:
                await server.serve_forever()
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--capacity", type=int, required=True)
    args = parser.parse_args()
    asyncio.run(serve(args.socket, args.capacity))
