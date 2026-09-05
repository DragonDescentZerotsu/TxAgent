"""Bounded first-valid-response races over cancellable provider requests."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from pathlib import Path
import threading
import time
from typing import Any, Callable

from .json_utils import write_json_atomic
from .openai_provider_pool import OpenAIProviderPool, ProviderPoolExhausted


class ParallelRetryClient:
    """One event loop owns all HTTP calls and request slots for a synchronous runner."""

    def __init__(self, pool: OpenAIProviderPool, *, parallelism: int, task_limits: dict[str, int]):
        self.pool = pool
        self.loop = asyncio.new_event_loop()
        self.slots = asyncio.Semaphore(parallelism)
        self.task_slots = {task: asyncio.Semaphore(limit) for task, limit in task_limits.items()}
        self.thread = threading.Thread(target=self.loop.run_forever, name="reasoning-http", daemon=True)
        self.thread.start()

    def chat_validated(
        self, messages: list[dict[str, Any]], *, task: str, width: int,
        validate: Callable[[dict[str, Any]], list[str]], receipt_path: Path,
    ) -> dict[str, Any]:
        if width < 1:
            raise ValueError("race width must be positive")
        future: Future = asyncio.run_coroutine_threadsafe(
            self._race(messages, task=task, width=width, validate=validate), self.loop
        )
        response, receipt = future.result()
        if width > 1 or receipt["winner"] is None:
            write_json_atomic(receipt_path, receipt)
        if response is None:
            raise ProviderPoolExhausted("all parallel attempts failed", attempts=receipt["attempts"])
        response["parallel_retry"] = {
            "width": width, "winner": receipt["winner"],
            "receipt": str(receipt_path) if width > 1 or receipt["winner"] is None else "",
            "attempt_statuses": [row["status"] for row in receipt["attempts"]],
        }
        return response

    async def _race(self, messages, *, task, width, validate):
        started = time.time()
        attempts = [{"attempt": i + 1, "status": "queued"} for i in range(width)]
        completed_responses = []

        async def attempt(row):
            try:
                async with self.task_slots[task], self.slots:
                    row.update(status="running", started_at=time.time())
                    response = await self.pool.async_chat_json(messages)
                    errors = validate(response)
                    row.update(status="invalid" if errors else "valid", errors=errors,
                               response=response)
                    completed_responses.append(response)
                    return (row["attempt"], response) if not errors else None
            except asyncio.CancelledError:
                row["status"] = "cancelled" if "started_at" in row else "cancelled_before_start"
                raise
            except Exception as exc:
                row.update(status="error", error_type=type(exc).__name__, error=str(exc)[:500])
                if isinstance(exc, ProviderPoolExhausted):
                    row["execution_provider_attempts"] = exc.attempts
                return None
            finally:
                row["finished_at"] = time.time()

        pending = [asyncio.create_task(attempt(row)) for row in attempts]
        winner, response = None, None
        try:
            for future in asyncio.as_completed(pending):
                result = await future
                if result is not None:
                    winner, response = result
                    break
        finally:
            for future in pending:
                if not future.done():
                    future.cancel()
            # Wait for HTTP connection cleanup before releasing the race to its caller.
            await asyncio.gather(*pending, return_exceptions=True)
        if response is None and completed_responses:
            response = completed_responses[-1]  # Preserve invalid JSON for the existing repair loop.
        receipt = {
            "protocol": "parallel_first_valid.v1", "width": width, "winner": winner,
            "started_at": started, "finished_at": time.time(), "attempts": attempts,
            "selection": "first response passing the complete schema and content validator",
            "cancellation": "awaited client HTTP cancellation; server abort depends on endpoint",
        }
        return response, receipt

    def snapshot(self):
        return self.pool.snapshot()

    def close(self):
        try:
            asyncio.run_coroutine_threadsafe(self.pool.aclose(), self.loop).result()
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()
            self.loop.close()
