"""Durable OpenRouter Batch transport for existing JSON reasoning callers.

Concurrent callers contribute independent requests; the transport submits bounded
batches and polls them without changing messages or progressive dependencies.
Stopping the local client leaves remote jobs intact for journal-based recovery.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from concurrent.futures import Future
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
import uuid

import requests
from openai.types.chat import ChatCompletion

from .json_utils import canonical_json_bytes, write_json_atomic
from .openai_reasoning_client import OpenAICompatibleClient


TERMINAL = {"completed", "failed", "cancelled", "expired"}


class OpenRouterBatchClient(OpenAICompatibleClient):
    """The same chat_json interface, backed by persisted asynchronous batch jobs."""

    def __init__(self, *, batch_dir, batch_options=None, **kwargs):
        super().__init__(**kwargs)
        options = dict(batch_options or {})
        unknown = options.keys() - {"max_requests", "max_bytes", "flush_seconds", "poll_seconds"}
        if unknown:
            raise ValueError(f"Unknown batch options: {sorted(unknown)}")
        if self.enable_group_tools or self.enable_thinking or kwargs.get("transport_max_retries", 0):
            raise ValueError("Batch requires prefetched tools and no automatic POST retries")
        parsed = urlsplit(str(self.client.base_url))
        if parsed.hostname != "openrouter.ai" or parsed.scheme != "https":
            raise ValueError("openrouter_batch requires https://openrouter.ai")
        if self.model.endswith(":batch"):
            raise ValueError("Batch submissions use the base model ID, without :batch")
        self.batch_url = "https://openrouter.ai/api/beta/batches"
        self.directory = Path(batch_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_requests = int(options.get("max_requests", 128))
        self.max_bytes = int(options.get("max_bytes", 16 * 1024 * 1024))
        self.flush_seconds = float(options.get("flush_seconds", 2))
        self.poll_seconds = float(options.get("poll_seconds", 15))
        if min(self.max_requests, self.max_bytes, self.flush_seconds, self.poll_seconds) <= 0:
            raise ValueError("Batch limits and intervals must be positive")
        self._lock_file = (self.directory / ".lock").open("a")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_file.close()
            raise ValueError(f"Batch journal is already in use: {self.directory}") from None
        self._condition = threading.Condition()
        self._pending = deque()
        self._recoverable = defaultdict(deque)
        self._futures = {}
        self._jobs = {}
        self._closed = False
        self._fatal = None
        self._session = requests.Session()
        self._session.headers.update(Authorization=f"Bearer {self.client.api_key}", **{"Content-Type": "application/json"})
        try:
            for path in sorted(self.directory.glob("job_*.json")):
                job = json.loads(path.read_text())
                self._jobs[path.name] = job
                if job["status"] == "submitting":
                    raise RuntimeError(
                        f"Submission outcome unknown: {path}. Reconcile its remote job before resuming; "
                        "automatic resubmission could charge twice."
                    )
                if job["status"] == "rejected":
                    continue
                for row in job["request"]["requests"]:
                    self._recoverable[self._key(row["body"])].append((path.name, row["custom_id"]))
        except Exception:
            self._lock_file.close()
            self._session.close()
            raise
        self._worker = threading.Thread(target=self._work, name="openrouter-batches", daemon=True)
        self._worker.start()

    @staticmethod
    def _key(body):
        return hashlib.sha256(canonical_json_bytes(body)).hexdigest()

    def chat_json(self, messages):
        body = self._completion_kwargs(messages)
        body.update(body.pop("extra_body", {}))
        key = self._key(body)
        with self._condition:
            if self._closed or self._fatal:
                raise RuntimeError(self._fatal or "Batch client is closed")
            recovered = self._recoverable[key]
            was_recovered = bool(recovered)
            if recovered:
                job_name, custom_id = recovered.popleft()
                future = self._futures.setdefault(custom_id, Future())
                job = self._jobs[job_name]
                if job["status"] in TERMINAL:
                    self._deliver(job)
            else:
                custom_id = uuid.uuid4().hex
                future = self._futures[custom_id] = Future()
                row = {"custom_id": custom_id, "body": body}
                if len(canonical_json_bytes(row)) + 1024 > self.max_bytes:
                    raise ValueError("A request exceeds the configured batch byte limit")
                self._pending.append((time.monotonic(), row))
            self._condition.notify_all()
        raw, receipt = future.result()
        result = self._json_response(ChatCompletion.model_validate(raw), messages)
        result["batch"] = {**receipt, "recovered": was_recovered}
        return result

    async def async_chat_json(self, messages):
        # Cancelling a local waiter must not cancel a paid remote batch.
        return await asyncio.to_thread(self.chat_json, messages)

    def _save(self, job):
        write_json_atomic(self.directory / job["journal"], job)

    def _submit(self, rows):
        job = {"journal": f"job_{time.time_ns()}_{uuid.uuid4().hex[:8]}.json",
               "status": "submitting", "created_at": time.time(),
               "request": {"endpoint": "/v1/chat/completions", "model": self.model, "requests": rows}}
        self._jobs[job["journal"]] = job
        self._save(job)  # Write before POST; never blindly repeat an uncertain submission.
        response = self._session.post(self.batch_url, data=canonical_json_bytes(job["request"]), timeout=60)
        if 400 <= response.status_code < 500 and response.status_code not in {408, 409}:
            job.update(status="rejected", error=f"HTTP {response.status_code}: {response.text[:1000]}")
            self._save(job)
            self._deliver(job)
            return
        response.raise_for_status()
        remote = response.json()
        if not remote.get("id"):
            raise ValueError("Batch submission returned no ID; reconcile the submitting journal")
        job.update(status=remote["status"], remote=remote, next_poll_at=0)
        self._save(job)
        if job["status"] in TERMINAL:
            self._deliver(job)

    def _poll(self, job):
        previous = (job["status"], job.get("remote"), job.get("last_poll_error"))
        try:
            response = self._session.get(f"{self.batch_url}/{job['remote']['id']}", timeout=60)
            if response.status_code in {401, 403}:
                job["last_poll_error"] = f"HTTP {response.status_code}; remote job requires attention"
                self._save(job)
                raise RuntimeError(job["last_poll_error"])
            response.raise_for_status()
            remote = response.json()
            if remote["id"] != job["remote"]["id"]:
                raise ValueError("Batch poll returned a different job ID")
            job.update(status=remote["status"], remote=remote)
            job.pop("last_poll_error", None)
        except (requests.RequestException, ValueError) as exc:
            # Poll failures do not authorize resubmitting a remote job.
            # Newly created jobs can briefly return 404 before becoming readable.
            job["last_poll_error"] = f"{type(exc).__name__}: {exc}"
        job["next_poll_at"] = time.time() + self.poll_seconds
        # Keep polling deadlines in memory; unchanged queues need no disk rewrite.
        if previous != (job["status"], job.get("remote"), job.get("last_poll_error")):
            self._save(job)
        if job["status"] in TERMINAL:
            self._deliver(job)

    def _deliver(self, job):
        remote = job.get("remote", {})
        rows = remote.get("results") or []
        by_id = {row["custom_id"]: row for row in rows}
        requested = {row["custom_id"] for row in job["request"]["requests"]}
        if len(by_id) != len(rows) or by_id.keys() - requested:
            raise ValueError("Duplicate or unknown custom_id in batch results")
        with self._condition:
            for custom_id in requested:
                future = self._futures.get(custom_id)
                if future is None or future.done():
                    continue
                item = by_id.get(custom_id, {})
                response = item.get("response") or {}
                if item.get("error") or response.get("status_code") != 200 or not response.get("body"):
                    future.set_exception(RuntimeError(
                        f"Batch {remote.get('id')} request {custom_id}: "
                        f"{item.get('error') or response or remote.get('error') or job.get('error') or job['status']}"
                    ))
                else:
                    future.set_result((copy.deepcopy(response["body"]), {
                        "id": remote["id"], "custom_id": custom_id,
                        "journal": str(self.directory / job["journal"]),
                        "completion_window": remote.get("completion_window"),
                    }))

    def _work(self):
        try:
            while True:
                with self._condition:
                    if self._closed:
                        return
                    ready = self._pending and (
                        len(self._pending) >= self.max_requests
                        or time.monotonic() - self._pending[0][0] >= self.flush_seconds
                    )
                    rows, size = [], 1024
                    if ready:
                        while self._pending and len(rows) < self.max_requests:
                            row = self._pending[0][1]
                            length = len(canonical_json_bytes(row)) + 1
                            if rows and size + length > self.max_bytes:
                                break
                            self._pending.popleft()
                            rows.append(row)
                            size += length
                if rows:
                    self._submit(rows)
                for job in list(self._jobs.values()):
                    if (job["status"] not in TERMINAL | {"rejected", "submitting"}
                            and time.time() >= job.get("next_poll_at", 0)):
                        self._poll(job)
                with self._condition:
                    self._condition.wait(timeout=min(self.flush_seconds, self.poll_seconds))
        except Exception as exc:
            with self._condition:
                self._fatal = f"Batch transport stopped: {type(exc).__name__}: {exc}"
                for future in self._futures.values():
                    if not future.done():
                        future.set_exception(RuntimeError(self._fatal))

    def close(self):
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
            for future in self._futures.values():
                if not future.done():
                    future.set_exception(RuntimeError("Local batch client stopped; remote jobs remain resumable"))
        self._worker.join()
        self._session.close()
        self.client.close()
        self._lock_file.close()

    async def aclose(self):
        await asyncio.to_thread(self.close)
