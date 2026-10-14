"""Semantic queues, immutable bindings and replica receipts."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urlsplit

_REQUEST_DATABASE_WRITE_LOCK = threading.RLock()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_id(prefix: str, *values: str) -> str:
    digest = hashlib.sha256("\x00".join(values).encode()).hexdigest()[:20]
    return f"{prefix}_{digest}"


def _request_database(path: Path, *, journal_mode: str = "WAL") -> sqlite3.Connection:
    if journal_mode not in {"WAL", "DELETE"}:
        raise ValueError("journal_mode must be WAL or DELETE")
    connection = sqlite3.connect(path, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=60000")
    connection.execute(f"PRAGMA journal_mode={journal_mode}")
    connection.execute(
        f"PRAGMA synchronous={('FULL' if journal_mode == 'DELETE' else 'NORMAL')}"
    )
    connection.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            request_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            phase TEXT NOT NULL,
            prompt_sha256 TEXT NOT NULL,
            prompt TEXT NOT NULL,
            reasoning_effort TEXT NOT NULL,
            max_tokens INTEGER NOT NULL,
            validation_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            response_json TEXT,
            reasoning_content TEXT,
            served_model TEXT,
            provider_name TEXT,
            provider_base_url TEXT,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            error TEXT
        )
        """)
    request_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(requests)")
    }
    for column in ("provider_name", "provider_base_url"):
        if column not in request_columns:
            connection.execute(f"ALTER TABLE requests ADD COLUMN {column} TEXT")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS request_attempt_receipts (
            request_id TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            receipt_json TEXT NOT NULL,
            PRIMARY KEY (request_id, attempt)
        )
        """)
    connection.commit()
    return connection


def _seed_request_database(
    path: Path, seed: Path | None, *, policy: Mapping
) -> dict[str, Any] | None:
    if seed is None or path.exists():
        return None
    if not seed.is_file():
        raise FileNotFoundError(f"seed request cache does not exist: {seed}")
    source = sqlite3.connect(f"file:{seed}?mode=ro", uri=True, timeout=60)
    try:
        statuses = dict(
            source.execute("SELECT status,count(*) FROM requests GROUP BY status")
        )
        if not statuses.get("complete"):
            raise ValueError(f"seed request cache has no complete requests: {statuses}")
        served_models = {
            str(row[0])
            for row in source.execute(
                "SELECT DISTINCT served_model FROM requests WHERE served_model IS NOT NULL"
            )
        }
        if served_models != {policy["MODEL"]}:
            raise ValueError(
                f"seed request cache served models do not match {policy['MODEL']}: {sorted(served_models)}"
            )
        destination = sqlite3.connect(path, timeout=60)
        try:
            source.backup(destination)
            destination.execute(
                "DELETE FROM request_attempt_receipts WHERE request_id IN (SELECT request_id FROM requests WHERE status!='complete')"
            )
            destination.execute("DELETE FROM requests WHERE status!='complete'")
            destination.commit()
        finally:
            destination.close()
    finally:
        source.close()
    migrated = _request_database(path)
    try:
        copied = int(migrated.execute("SELECT count(*) FROM requests").fetchone()[0])
    finally:
        migrated.close()
    return {
        "path": str(seed),
        "sha256": _file_sha256(seed),
        "complete_requests_copied": copied,
        "source_status_counts": dict(sorted(statuses.items())),
        "noncomplete_requests_excluded": sum(
            (count for status, count in statuses.items() if status != "complete")
        ),
    }


def _queue_request(
    connection: sqlite3.Connection,
    *,
    request_id: str,
    kind: str,
    phase: str,
    prompt: str,
    reasoning_effort: str,
    max_tokens: int,
    validation: Mapping[str, Any],
    commit: bool = True,
) -> str:
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    values = (
        request_id,
        kind,
        phase,
        prompt_hash,
        prompt,
        reasoning_effort,
        max_tokens,
        _canonical_json(validation),
    )
    with _REQUEST_DATABASE_WRITE_LOCK:
        connection.execute(
            """
            INSERT OR IGNORE INTO requests
            (request_id,kind,phase,prompt_sha256,prompt,reasoning_effort,max_tokens,validation_json)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            values,
        )
        stored = connection.execute(
            """
            SELECT request_id,kind,phase,prompt_sha256,prompt,reasoning_effort,max_tokens,
                   validation_json
            FROM requests WHERE request_id=?
            """,
            (request_id,),
        ).fetchone()
        if tuple(stored) != values:
            raise ValueError(f"cached request contract changed: {request_id}")
        if commit:
            connection.commit()
    return request_id


def _response(connection: sqlite3.Connection, request_id: str) -> dict[str, Any]:
    row = connection.execute(
        "SELECT status,response_json FROM requests WHERE request_id=?", (request_id,)
    ).fetchone()
    if row is None or row["status"] != "complete":
        raise ValueError(f"request is not complete: {request_id}")
    return json.loads(row["response_json"])


def _request_id(kind: str, phase: str, prompt: str) -> str:
    return _stable_id("req", kind, phase, hashlib.sha256(prompt.encode()).hexdigest())


def review_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=120)
    connection.row_factory = sqlite3.Row
    connection.execute("""CREATE TABLE IF NOT EXISTS requests (
        request_id TEXT PRIMARY KEY, task TEXT NOT NULL, pass_number INTEGER NOT NULL,
        scope_json TEXT NOT NULL, items_json TEXT NOT NULL, prompt TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
        response_json TEXT, reasoning TEXT, receipts_json TEXT NOT NULL DEFAULT '[]', error TEXT)""")
    connection.commit()
    return connection


def insert_reviews(
    connection: sqlite3.Connection, requests: list[dict[str, Any]]
) -> None:
    connection.executemany(
        "INSERT OR IGNORE INTO requests\n        (request_id,task,pass_number,scope_json,items_json,prompt) VALUES (?,?,?,?,?,?)",
        [
            (
                row["request_id"],
                row["task"],
                row["pass_number"],
                json.dumps(row["scope"]),
                json.dumps(row["items"], ensure_ascii=False),
                row["prompt"],
            )
            for row in requests
        ],
    )
    connection.commit()


VERSION = "semantic_weight_speculative_first8.v1"
AGGREGATION_METHOD = "first_8_valid_mean_v1"
DEFAULT_BASE_URL = "http://dgx008:50001/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REQUIRED_REPLICAS = 8
MAX_FANOUT = 128
MAX_ROUND_REQUESTS = 9


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_tables(connection: sqlite3.Connection) -> None:
    connection.execute("""
        CREATE TABLE IF NOT EXISTS speculative_replica_receipts (
            request_id TEXT NOT NULL,
            execution_id TEXT NOT NULL,
            replica_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            completion_rank INTEGER,
            receipt_json TEXT NOT NULL,
            PRIMARY KEY (request_id, execution_id, replica_id)
        )
        """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS speculative_aggregates (
            request_id TEXT PRIMARY KEY,
            method TEXT NOT NULL,
            replica_count INTEGER NOT NULL,
            aggregate_json TEXT NOT NULL,
            benchmark_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)
    connection.commit()


def _usage_totals(receipts: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    prompt = completion = 0
    for receipt in receipts:
        usage = receipt.get("usage") or {}
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
    return (prompt, completion)


def persist_result(
    connection: sqlite3.Connection,
    result: Mapping[str, Any],
    *,
    base_url: str,
    model: str,
    benchmark_sha256: str,
) -> None:
    request_id = str(result["request_id"])
    row = connection.execute(
        "SELECT attempts,input_tokens,output_tokens FROM requests WHERE request_id=?",
        (request_id,),
    ).fetchone()
    attempt = int(row["attempts"]) + 1
    prompt_tokens, output_tokens = _usage_totals(result["receipts"])
    for receipt in result["receipts"]:
        connection.execute(
            "INSERT OR IGNORE INTO speculative_replica_receipts VALUES (?,?,?,?,?,?)",
            (
                request_id,
                result["execution_id"],
                receipt["replica_id"],
                receipt["status"],
                receipt.get("completion_rank"),
                _canonical_json(receipt),
            ),
        )
    _persist_aggregate(connection, result, benchmark_sha256)
    aggregate_receipt = {
        "attempt": attempt,
        "execution_id": result["execution_id"],
        "aggregation_method": result["aggregation_method"],
        "time_to_required_seconds": result["time_to_required_seconds"],
        "replica_status_counts": dict(
            Counter((receipt["status"] for receipt in result["receipts"]))
        ),
        "expected_replica_count": int(
            result.get("total_replica_count", len(result["receipts"]))
        ),
        "all_replicas_stored": len(result["receipts"])
        == int(result.get("total_replica_count", len(result["receipts"]))),
        "benchmark_sha256": benchmark_sha256,
        "response": result.get("response"),
        "aggregate": result.get("aggregate"),
    }
    connection.execute(
        "INSERT OR REPLACE INTO request_attempt_receipts VALUES (?,?,?)",
        (request_id, attempt, _canonical_json(aggregate_receipt)),
    )
    medoid = _medoid_receipt(result)
    connection.execute(
        """UPDATE requests SET status=?,attempts=?,response_json=?,reasoning_content=?,
           served_model=?,provider_name=?,provider_base_url=?,input_tokens=?,
           output_tokens=?,error=? WHERE request_id=?""",
        (
            result["status"],
            attempt,
            _canonical_json(result.get("response")) if result.get("response") else None,
            medoid.get("reasoning_content") if medoid else None,
            medoid.get("model", model) if medoid else None,
            (medoid.get("provider_name") if medoid else None)
            or f"{urlsplit(base_url).netloc.replace(':', '_')}_speculative",
            (medoid.get("provider_base_url") if medoid else None) or base_url,
            int(row["input_tokens"]) + prompt_tokens,
            int(row["output_tokens"]) + output_tokens,
            result.get("error"),
            request_id,
        ),
    )
    connection.commit()


def persist_late_receipts(
    connection: sqlite3.Connection,
    result: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> None:
    """Append replicas that finished after the aggregate quorum was released."""
    if not receipts:
        return
    request_id = str(result["request_id"])
    for receipt in receipts:
        connection.execute(
            "INSERT OR IGNORE INTO speculative_replica_receipts VALUES (?,?,?,?,?,?)",
            (
                request_id,
                result["execution_id"],
                receipt["replica_id"],
                receipt["status"],
                receipt.get("completion_rank"),
                _canonical_json(receipt),
            ),
        )
    prompt_tokens, output_tokens = _usage_totals(receipts)
    connection.execute(
        "UPDATE requests SET input_tokens=input_tokens+?,output_tokens=output_tokens+? WHERE request_id=?",
        (prompt_tokens, output_tokens, request_id),
    )
    row = connection.execute(
        "SELECT attempts FROM requests WHERE request_id=?", (request_id,)
    ).fetchone()
    attempt = int(row["attempts"])
    receipt_row = connection.execute(
        "SELECT receipt_json FROM request_attempt_receipts WHERE request_id=? AND attempt=?",
        (request_id, attempt),
    ).fetchone()
    aggregate_receipt = json.loads(receipt_row["receipt_json"])
    statuses = connection.execute(
        "SELECT status,COUNT(*) FROM speculative_replica_receipts WHERE request_id=? AND execution_id=? GROUP BY status",
        (request_id, result["execution_id"]),
    ).fetchall()
    aggregate_receipt["final_replica_status_counts"] = dict(statuses)
    aggregate_receipt["all_replicas_stored"] = sum(
        (count for _, count in statuses)
    ) == int(result["total_replica_count"])
    connection.execute(
        "UPDATE request_attempt_receipts SET receipt_json=? WHERE request_id=? AND attempt=?",
        (_canonical_json(aggregate_receipt), request_id, attempt),
    )
    connection.commit()


def all_replicas_stored(
    connection: sqlite3.Connection,
    request_id: str,
    *,
    expected_replicas: int | None = None,
) -> bool:
    """Return whether the latest execution has a receipt for every replica."""
    row = connection.execute(
        "SELECT receipt_json FROM request_attempt_receipts WHERE request_id=? ORDER BY attempt DESC LIMIT 1",
        (request_id,),
    ).fetchone()
    if row is None:
        return False
    receipt = json.loads(row["receipt_json"])
    execution_id = receipt.get("execution_id")
    expected = int(receipt.get("expected_replica_count") or 0)
    if (
        not execution_id
        or expected < 1
        or (expected_replicas is not None and expected != expected_replicas)
    ):
        return False
    stored = connection.execute(
        "SELECT COUNT(*) FROM speculative_replica_receipts WHERE request_id=? AND execution_id=?",
        (request_id, execution_id),
    ).fetchone()[0]
    return stored == expected


def _persist_aggregate(
    connection: sqlite3.Connection, result: Mapping[str, Any], benchmark_sha256: str
) -> None:
    if result["status"] != "complete":
        return
    connection.execute(
        "INSERT OR REPLACE INTO speculative_aggregates VALUES (?,?,?,?,?,?)",
        (
            result["request_id"],
            result["aggregate"]["aggregation_method"],
            result["aggregate"]["replicate_count"],
            _canonical_json(result["aggregate"]),
            benchmark_sha256,
            _now(),
        ),
    )


def _medoid_receipt(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if result["status"] != "complete":
        return None
    wanted = result["aggregate"]["medoid_replica_id"]
    return next((row for row in result["receipts"] if row["replica_id"] == wanted))


def _pending_rows(
    connection: sqlite3.Connection, request_ids: Sequence[str]
) -> list[dict[str, Any]]:
    unique = list(dict.fromkeys(request_ids))
    if not unique:
        return []
    placeholders = ",".join(("?" for _ in unique))
    rows = {
        row["request_id"]: dict(row)
        for row in connection.execute(
            f"SELECT * FROM requests WHERE request_id IN ({placeholders})", unique
        )
    }
    if set(rows) != set(unique):
        raise ValueError("requested an unknown cached request")
    return [
        rows[request_id]
        for request_id in unique
        if rows[request_id]["status"] != "complete"
    ]


def aggregate_lookup(connection: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='speculative_aggregates'"
    ).fetchone()
    if not exists:
        return {}
    return {
        row["request_id"]: json.loads(row["aggregate_json"])
        for row in connection.execute(
            "SELECT request_id,aggregate_json FROM speculative_aggregates"
        )
    }
