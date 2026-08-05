from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from threading import Event, RLock
from typing import Any, Callable


class ToolResultCache:
    """Small thread-safe LRU backed by a process-safe local SQLite cache."""

    def __init__(self, path: Path | None, *, memory_entries: int) -> None:
        self._lock = RLock()
        self._memory_entries = memory_entries
        self._memory: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._inflight: dict[str, Event] = {}
        self._connection: sqlite3.Connection | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(
                path,
                timeout=30,
                check_same_thread=False,
                isolation_level=None,
            )
            self._connection.execute("PRAGMA busy_timeout=30000")
            try:
                self._connection.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS tool_results (cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._memory.get(key)
            if value is not None:
                self._memory.move_to_end(key)
                return deepcopy(value)
            if self._connection is None:
                return None
            row = self._connection.execute(
                "SELECT payload FROM tool_results WHERE cache_key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            value = json.loads(row[0])
            self._remember(key, value)
            return deepcopy(value)

    def put(self, key: str, value: dict[str, Any]) -> None:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        with self._lock:
            self._remember(key, value)
            if self._connection is not None:
                self._connection.execute(
                    "INSERT OR REPLACE INTO tool_results(cache_key, payload) VALUES (?, ?)",
                    (key, payload),
                )

    def get_or_compute(
        self,
        key: str,
        compute: Callable[[], dict[str, Any]],
    ) -> tuple[dict[str, Any], bool]:
        cached = self.get(key)
        if cached is not None:
            return cached, True
        with self._lock:
            event = self._inflight.get(key)
            if event is None:
                event = Event()
                self._inflight[key] = event
                owner = True
            else:
                owner = False
        if not owner:
            event.wait()
            cached = self.get(key)
            if cached is not None:
                return cached, True
            return self.get_or_compute(key, compute)
        try:
            value = compute()
            self.put(key, value)
            return deepcopy(value), False
        finally:
            with self._lock:
                self._inflight.pop(key, None)
                event.set()

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def _remember(self, key: str, value: dict[str, Any]) -> None:
        if self._memory_entries <= 0:
            return
        self._memory[key] = deepcopy(value)
        self._memory.move_to_end(key)
        while len(self._memory) > self._memory_entries:
            self._memory.popitem(last=False)
