"""Resumable offline annotations with independently bounded provider endpoints.

Normalized source records remain in Parquet. SQLite stores unique requests,
record-to-request mappings, all attempt receipts and validated annotations.
Successful rows are never resubmitted on resume; uncertain semantics are retained.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import closing
from datetime import datetime, timezone
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import time
import zlib

import pyarrow.parquet as pq

from tools.chembl_tool.common.openai_provider_pool import (
    OpenAIProviderPool,
    load_env_file,
    load_provider_pool_config,
)
from tools.chembl_tool.common.openai_reasoning_client import OpenAICompatibleClient
from tools.chembl_tool.common.record_diversity import (
    ANNOTATION_VERSION,
    BOUND_ANNOTATION_VERSION,
    QUALITY_ANNOTATION_VERSION,
    cache_key,
    extraction_messages,
    extraction_request_body,
    validate_annotation,
    annotation_quote_catalog,
    resolve_annotation_quotes,
    repair_annotation_quotes,
    QUOTE_TRANSPORT_INSTRUCTION,
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def pack(value):
    return zlib.compress(dump(value).encode(), level=1)


def unpack(value):
    return json.loads(zlib.decompress(value))


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def database(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA cache_size=-131072")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY, cache_key TEXT NOT NULL UNIQUE, payload BLOB NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0);
        CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status,id);
        CREATE TABLE IF NOT EXISTS records (
            task TEXT NOT NULL, record_id TEXT NOT NULL, source_id TEXT, level INTEGER,
            route TEXT NOT NULL, job_id INTEGER, PRIMARY KEY(task,record_id));
        CREATE INDEX IF NOT EXISTS records_job ON records(job_id);
        CREATE TABLE IF NOT EXISTS attempts (
            job_id INTEGER NOT NULL, attempt INTEGER NOT NULL, status TEXT NOT NULL,
            response BLOB, error TEXT, elapsed_seconds REAL, reported_cost REAL,
            prompt_tokens INTEGER, completion_tokens INTEGER, finished_at TEXT,
            PRIMARY KEY(job_id,attempt));
        CREATE TABLE IF NOT EXISTS annotations (
            job_id INTEGER PRIMARY KEY, annotation BLOB NOT NULL,
            unresolved INTEGER NOT NULL, unclear_focal INTEGER NOT NULL, finished_at TEXT);
        CREATE TEMP TABLE priority_jobs (job_id INTEGER PRIMARY KEY, rank INTEGER NOT NULL);
    """)
    return db


def backup_database(source, destination):
    """Snapshot local WAL safely; atomically publish a durable shared-disk copy."""
    temporary = destination.with_suffix(".tmp.sqlite")
    with (
        closing(sqlite3.connect(source)) as reader,
        closing(sqlite3.connect(temporary)) as writer,
    ):
        # Pin a WAL snapshot so concurrent result commits cannot restart the copy.
        reader.execute("BEGIN")
        reader.execute("SELECT COUNT(*) FROM metadata").fetchone()
        reader.backup(writer, pages=2048)
    temporary.replace(destination)
    write_json(
        destination.with_suffix(".json"),
        {
            "updated_at": utc(),
            "source": str(source),
            "bytes": destination.stat().st_size,
        },
    )


def ingest(db, inputs, contract, progress_path):
    """Exact-input deduplication; resume an interrupted import idempotently."""
    encoded = dump(contract)
    old = db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
    if old and old[0] != encoded:
        raise ValueError("Input/model contract changed; use a new output directory")
    db.execute("INSERT OR IGNORE INTO metadata VALUES ('contract',?)", (encoded,))
    db.commit()
    if db.execute("SELECT 1 FROM metadata WHERE key='ingestion_complete'").fetchone():
        return
    seen = 0
    for path in inputs:
        task = path.name.removesuffix(".features.parquet")
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=2048, use_threads=False
        ):
            for row in batch.to_pylist():
                prepared = json.loads(row["payload_json"])
                assert prepared["record_id"] == row["record_id"] and row["record_id"]
                job_id = None
                if prepared["needs_llm"]:
                    key = cache_key(
                        prepared,
                        contract["model_config"],
                        contract["model_config"].get("annotation_version", ANNOTATION_VERSION),
                    )
                    existing = db.execute(
                        "SELECT id FROM jobs WHERE cache_key=?", (key,)
                    ).fetchone()
                    if existing:
                        job_id = existing[0]
                    else:
                        payload = {"input": prepared["input"]}
                        if prepared.get("review_notes"):
                            payload["review_notes"] = prepared["review_notes"]
                        job_id = db.execute(
                            "INSERT INTO jobs(cache_key,payload) VALUES (?,?)",
                            (key, pack(payload)),
                        ).lastrowid
                db.execute(
                    "INSERT OR IGNORE INTO records VALUES (?,?,?,?,?,?)",
                    (
                        task,
                        row["record_id"],
                        row["source_id"],
                        row["level"],
                        row["route"],
                        job_id,
                    ),
                )
            seen += batch.num_rows
            db.commit()
            if seen % 20480 < 2048:
                write_json(
                    progress_path,
                    {
                        "status": "preparing_requests",
                        "source_rows_scanned": seen,
                        "expected_source_rows": contract["expected_source_rows"],
                        "updated_at": utc(),
                        "model_calls": 0,
                    },
                )
        print(f"indexed {task}: {seen} source rows", flush=True)
    counts = dict(db.execute("SELECT route,COUNT(*) FROM records GROUP BY route"))
    assert sum(counts.values()) == contract["expected_source_rows"], counts
    assert counts.get("llm_candidate", 0) == contract["expected_llm_candidate_rows"], (
        counts
    )
    db.execute("INSERT INTO metadata VALUES ('ingestion_complete',?)", (utc(),))
    db.commit()


def recover_inflight(db, max_attempts):
    # Requests interrupted before a response is recorded may have reached the provider.
    # Raising the explicit attempt ceiling also permits exhausted jobs to resume;
    # their previous attempts and successful annotations are never reset.
    db.execute(
        "UPDATE jobs SET status=CASE WHEN attempts>=? THEN 'failed' ELSE 'pending' END WHERE status IN ('inflight','failed')",
        (max_attempts,),
    )
    db.commit()


def reuse_annotations(db, source, contract):
    """Import exact validated successes from a frozen cache, preserving attempts."""
    source = source.resolve()
    if source == Path(db.execute("PRAGMA database_list").fetchone()[2]).resolve():
        raise ValueError("Reuse source must differ from the destination database")
    source_hash = digest(source)
    if source.with_name(source.name + "-wal").exists():
        raise ValueError("Reuse requires a frozen SQLite backup without a WAL")
    imported = 0
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro&immutable=1", uri=True)) as donor:
        donor.execute("BEGIN")
        previous = json.loads(donor.execute(
            "SELECT value FROM metadata WHERE key='contract'"
        ).fetchone()[0])
        for field in ("model_config", "system_prompt_sha256", "effective_system_prompt_sha256"):
            if previous.get(field) != contract.get(field):
                raise ValueError(f"Reuse cache has a different {field}")
        version = contract["model_config"].get("annotation_version", ANNOTATION_VERSION)
        with db:
            for old_id, key, payload, attempts, annotation, unresolved, unclear, finished in donor.execute(
                "SELECT j.id,j.cache_key,j.payload,j.attempts,a.annotation,a.unresolved,"
                "a.unclear_focal,a.finished_at FROM jobs j JOIN annotations a ON a.job_id=j.id "
                "WHERE j.status='done'"
            ):
                target = db.execute(
                    "SELECT id,payload FROM jobs WHERE cache_key=? AND status='pending' AND attempts=0",
                    (key,),
                ).fetchone()
                if target is None:
                    continue
                job_id, target_payload = target
                prepared = unpack(target_payload)
                # Cache keys hash model-visible inputs; additionally require exact native inputs.
                if unpack(payload) != prepared:
                    continue
                assert cache_key(prepared, contract["model_config"], version) == key
                validate_annotation(unpack(annotation), prepared, version)
                for attempt in donor.execute("SELECT * FROM attempts WHERE job_id=?", (old_id,)):
                    db.execute("INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?)", (job_id, *attempt[1:]))
                db.execute("INSERT INTO annotations VALUES (?,?,?,?,?)",
                           (job_id, annotation, unresolved, unclear, finished))
                db.execute("UPDATE jobs SET status='done',attempts=? WHERE id=?", (attempts, job_id))
                imported += 1
            if digest(source) != source_hash or source.with_name(source.name + "-wal").exists():
                raise ValueError("Reuse requires a frozen SQLite backup without a WAL")
            receipt = {"source": str(source), "sha256": source_hash,
                       "imported_jobs": imported, "updated_at": utc()}
            db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)",
                       ("reuse:" + source_hash, dump(receipt)))
    return receipt


def validation_retry_messages(base_messages, response, error):
    if response is None or 'content' not in response:
        return base_messages
    return base_messages + [
        {'role':'assistant', 'content':dump(response['content'])},
        {'role':'user', 'content':'The previous attempt failed schema/source-quote validation: '
         + (error or 'invalid output')[:700]
         + '. Return complete corrected JSON; use only allowed kind values and copy quotes exactly from the specified input fields.'},
    ]


def prioritize(db, path):
    """Schedule named source records first without changing inputs or cache keys."""
    db.execute("DELETE FROM priority_jobs")
    if not path:
        return
    for rank, record in enumerate(json.loads(path.read_text())):
        row = db.execute("SELECT job_id FROM records WHERE task=? AND record_id=?",
                         (record['task'], record['record_id'])).fetchone()
        if row is None:
            raise ValueError(f"Unknown priority source record: {record}")
        if row[0] is not None:
            db.execute("INSERT OR IGNORE INTO priority_jobs VALUES (?,?)", (row[0], rank))
    db.commit()


def claim(db, limit):
    rows = db.execute(
        "SELECT j.id,j.payload,j.attempts FROM priority_jobs p JOIN jobs j ON j.id=p.job_id "
        "WHERE j.status='pending' ORDER BY p.rank LIMIT ?",
        (limit,),
    ).fetchall()
    db.executemany(
        "UPDATE jobs SET status='inflight' WHERE id=?", [(r[0],) for r in rows]
    )
    remaining = db.execute(
        "SELECT id,payload,attempts FROM jobs WHERE status='pending' ORDER BY id LIMIT ?",
        (limit-len(rows),),
    ).fetchall()
    db.executemany("UPDATE jobs SET status='inflight' WHERE id=?", [(r[0],) for r in remaining])
    db.commit()
    return rows + remaining


def save_attempt(db, job_id, attempt, response, annotation, error, elapsed):
    usage = (response or {}).get("usage", {})
    valid = annotation is not None
    db.execute(
        "INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            job_id,
            attempt,
            "schema_quote_valid" if valid else "error",
            pack(response) if response is not None else None,
            error,
            elapsed,
            usage.get("cost"),
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            utc(),
        ),
    )
    if valid:
        db.execute(
            "INSERT INTO annotations VALUES (?,?,?,?,?)",
            (
                job_id,
                pack(annotation),
                int(bool(annotation["unresolved"])),
                int(any(o["focal_scope"] == "unclear" for o in annotation["outcomes"])),
                utc(),
            ),
        )
        db.execute("UPDATE jobs SET status='done' WHERE id=?", (job_id,))
    db.commit()


def progress(db, args, state, started, initial_done, usage, active=0, pool=None):
    counts = dict(db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status"))
    elapsed = max(time.monotonic() - started, 0.001)
    rate = (counts.get("done", 0) - initial_done) / elapsed
    remaining = counts.get("pending", 0) + counts.get("inflight", 0)
    result = {
        "status": state,
        "updated_at": utc(),
        "pid": os.getpid(),
        "parallelism": args.parallelism,
        "active_jobs": active,
        "unique_requests": sum(counts.values()),
        "jobs": counts,
        "attempts": usage[0],
        "reported_cost_usd": usage[1],
        "prompt_tokens": usage[2],
        "completion_tokens": usage[3],
        "session_elapsed_seconds": round(elapsed, 2),
        "session_valid_per_second": round(rate, 3),
        "eta_hours_at_session_rate": round(remaining / rate / 3600, 2)
        if rate > 0
        else None,
        "semantic_accuracy_verified": False,
        "cost_scope": "Sum of provider-reported costs; unreported costs are not estimated",
    }
    if pool is not None:
        result["providers"] = pool.snapshot()["providers"]
    write_json(args.output_dir / "progress.json", result)
    return result


def model_identity(name):
    return name.lower().split("/")[-1]


def compatible_cache_config(previous, current):
    """Keep legacy cache keys while allowing a transport-only model-alias change."""
    fields = ("annotation_version", "temperature", "max_tokens", "quote_repair", "quote_transport")
    if model_identity(previous["model"]) != model_identity(current["model"]) or any(
        previous.get(k) != current.get(k) for k in fields
    ):
        raise ValueError("Model semantics changed; use a new output directory")
    previous_reasoning = previous.get(
        "reasoning_enabled",
        previous.get("request_extra_body", {}).get("reasoning", {}).get("enabled"),
    )
    if previous_reasoning is None or previous_reasoning != current["reasoning_enabled"]:
        raise ValueError("Reasoning settings changed; use a new output directory")
    if previous.get("reasoning_effort", "") != current.get("reasoning_effort", ""):
        raise ValueError("Reasoning effort changed; use a new output directory")
    return previous


async def extract(db, args, config, model_config, api_keys):
    annotation_version = model_config.get("annotation_version", ANNOTATION_VERSION)

    def client_factory(spec):
        return OpenAICompatibleClient(
            api_key=api_keys[spec.name],
            base_url=spec.base_url,
            model=spec.model,
            timeout_s=spec.timeout_s or args.timeout,
            max_tokens=model_config["max_tokens"],
            temperature=model_config.get("temperature", 0),
            tool_service_url="",
            enable_group_tools=False,
            max_tool_rounds=0,
            reasoning_effort=model_config.get("reasoning_effort", ""),
            enable_thinking=False,
            transport_max_retries=0,
            request_extra_body=extraction_request_body(
                spec.base_url, spec.request_extra_body,
                thinking=model_config.get("reasoning_enabled", False),
            ),
            async_max_connections=spec.max_inflight,
        )

    # Each outer attempt is exactly one HTTP attempt; retries share endpoint slots.
    client = OpenAIProviderPool(
        replace(config, max_failovers=0), client_factory=client_factory
    )
    recover_inflight(db, args.max_attempts)
    stop = asyncio.Event()
    fatal = []
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    started = time.monotonic()
    initial_done = db.execute(
        "SELECT COUNT(*) FROM jobs WHERE status='done'"
    ).fetchone()[0]
    scheduled = 0
    active = set()
    last_progress = 0
    usage_totals = list(
        db.execute(
            "SELECT COUNT(*),COALESCE(SUM(reported_cost),0),COALESCE(SUM(prompt_tokens),0),COALESCE(SUM(completion_tokens),0) FROM attempts"
        ).fetchone()
    )
    transport_streak = 0
    backup_task = None
    last_backup = time.monotonic()
    database_path = getattr(args, "database_path", None)
    backup_path = args.output_dir / "extraction.checkpoint.sqlite"

    async def one(row):
        nonlocal transport_streak
        job_id, payload, previous_attempts = row
        prepared = unpack(payload)
        base_messages = extraction_messages(prepared, annotation_version)
        catalog = annotation_quote_catalog(prepared, annotation_version) if model_config.get("quote_transport") else {}
        if catalog:
            base_messages[0]["content"] += QUOTE_TRANSPORT_INSTRUCTION
            base_messages.append({"role": "user", "content": dump({"source_quote_catalog": catalog})})
        messages = base_messages
        if previous_attempts:
            last = db.execute('SELECT response,error FROM attempts WHERE job_id=? ORDER BY attempt DESC LIMIT 1',
                              (job_id,)).fetchone()
            if last:
                messages = validation_retry_messages(base_messages, unpack(last[0]) if last[0] else None, last[1])
        for attempt in range(previous_attempts + 1, args.max_attempts + 1):
            if stop.is_set():
                db.execute("UPDATE jobs SET status='pending' WHERE id=?", (job_id,))
                db.commit()
                return
            db.execute("UPDATE jobs SET attempts=? WHERE id=?", (attempt, job_id))
            db.commit()
            response = None
            annotation = None
            error = None
            begin = time.monotonic()
            try:
                response = await client.async_chat_json(messages)
                transport_streak = 0
                if response.get("model") and model_identity(
                    response["model"]
                ) != model_identity(model_config["model"]):
                    fatal.append(
                        {
                            "error": "Provider returned an unexpected model",
                            "model": response["model"],
                        }
                    )
                    stop.set()
                    raise ValueError("Provider returned an unexpected model")
                content = response["content"]
                if catalog and isinstance(content, dict):
                    content = resolve_annotation_quotes(content, catalog)
                if model_config.get("quote_repair") and isinstance(content, dict):
                    content, response["quote_repairs"] = repair_annotation_quotes(content, prepared)
                annotation = validate_annotation(content, prepared, annotation_version)
            except Exception as exc:
                executions = getattr(exc, "attempts", [])
                error = str(exc) + (": " + dump(executions) if executions else "")
                for key in api_keys.values():
                    if key and key != "EMPTY":
                        error = error.replace(key, "<redacted>")
                error = error[:2000]
                status = getattr(exc, "status_code", None)
                if response is None:
                    transport_streak += 1
                    if executions:
                        response = {"execution_provider_attempts": executions}
                all_unavailable = all(
                    p["circuit_open"] for p in client.snapshot()["providers"]
                )
                auth_error = any(
                    f"Error code: {code}" in error for code in (401, 402, 403, 404)
                )
                if (
                    status in {401, 402, 403, 404}
                    or auth_error
                    or (transport_streak >= 64 and all_unavailable)
                ):
                    fatal.append({"http_status": status, "error": error})
                    stop.set()
            save_attempt(
                db,
                job_id,
                attempt,
                response,
                annotation,
                error,
                time.monotonic() - begin,
            )
            usage = (response or {}).get("usage", {})
            for i, value in enumerate(
                [
                    1,
                    usage.get("cost") or 0,
                    usage.get("prompt_tokens") or 0,
                    usage.get("completion_tokens") or 0,
                ]
            ):
                usage_totals[i] += value
            if annotation is not None:
                return
            if response is not None and "content" in response:
                messages = validation_retry_messages(base_messages, response, error)
            if attempt < args.max_attempts and not stop.is_set():
                await asyncio.sleep(min(2**attempt, 8))
        db.execute("UPDATE jobs SET status='failed' WHERE id=?", (job_id,))
        db.commit()

    try:
        while active or not stop.is_set():
            if backup_task is not None and backup_task.done():
                backup_task.result()
                backup_task = None
            if (
                database_path
                and backup_task is None
                and time.monotonic() - last_backup >= 600
            ):
                backup_task = asyncio.create_task(
                    asyncio.to_thread(backup_database, database_path, backup_path)
                )
                last_backup = time.monotonic()
            allowance = args.parallelism - len(active)
            if args.limit:
                allowance = min(allowance, max(0, args.limit - scheduled))
            if allowance > 0 and not stop.is_set():
                rows = claim(db, allowance)
                for row in rows:
                    active.add(asyncio.create_task(one(row)))
                scheduled += len(rows)
            if not active:
                break
            finished, active = await asyncio.wait(
                active, timeout=1, return_when=asyncio.FIRST_COMPLETED
            )
            for task in finished:
                task.result()
            if time.monotonic() - last_progress >= 10:
                result = progress(
                    db,
                    args,
                    "running",
                    started,
                    initial_done,
                    usage_totals,
                    len(active),
                    pool=client,
                )
                last_progress = time.monotonic()
                print(
                    dump(
                        {
                            k: result[k]
                            for k in (
                                "status",
                                "jobs",
                                "active_jobs",
                                "session_valid_per_second",
                                "reported_cost_usd",
                            )
                        }
                    ),
                    flush=True,
                )
        counts = dict(db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status"))
        state = (
            "paused_provider_error"
            if fatal
            else "paused"
            if stop.is_set()
            else "paused_limit"
            if counts.get("pending", 0)
            else "complete_with_failures"
            if counts.get("failed", 0)
            else "complete"
        )
        result = progress(
            db, args, state, started, initial_done, usage_totals, pool=client
        )
        result["fatal_errors"] = fatal
        write_json(args.output_dir / "progress.json", result)
        if not counts.get("pending", 0) and not counts.get("inflight", 0):
            result["per_task"] = [
                dict(zip(("task", "status", "records"), r))
                for r in db.execute(
                    "SELECT r.task,j.status,COUNT(*) FROM records r JOIN jobs j ON r.job_id=j.id GROUP BY r.task,j.status"
                )
            ]
            write_json(args.output_dir / "completion.json", result)
        print(dump(result), flush=True)
        return 2 if fatal else 1 if counts.get("failed", 0) else 0
    finally:
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        await client.aclose()
        if backup_task is not None:
            await backup_task
        if database_path:
            await asyncio.to_thread(backup_database, database_path, backup_path)


def check_resume_hold(output_dir):
    """A persisted user-requested hold survives launcher/process restarts."""
    hold = Path(output_dir) / "pause.json"
    if hold.exists():
        raise RuntimeError(
            f"Extraction is held by {hold}. Resolve its recorded conditions "
            "before removing the hold and resuming."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider-config", type=Path, required=True)
    parser.add_argument(
        "--database-path",
        type=Path,
        help="Optional node-local SQLite path; snapshot to output-dir every 10 minutes",
    )
    parser.add_argument("--parallelism", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--repair-quotes", action="store_true")
    parser.add_argument("--quote-catalog", action="store_true")
    parser.add_argument(
        "--annotation-version",
        choices=[ANNOTATION_VERSION, BOUND_ANNOTATION_VERSION, QUALITY_ANNOTATION_VERSION],
        default=ANNOTATION_VERSION,
        help="Extraction schema/prompt; changing it requires a new database",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Bounded preflight; zero processes all pending requests",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-annotations", type=Path, action="append", default=[],
                        help="Frozen SQLite backups to reuse by exact input/prompt/model identity")
    parser.add_argument("--priority-records", type=Path,
                        help="Optional ordered JSON list of task/record_id objects to annotate first; scheduling only.")
    args = parser.parse_args()
    if min(args.parallelism, args.max_attempts, args.timeout, args.max_tokens) < 1 or args.limit < 0:
        parser.error("Invalid request bounds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.prepare_only:
        check_resume_hold(args.output_dir)
    load_env_file(".env")
    config = load_provider_pool_config(args.provider_config)
    if args.parallelism > sum(s.max_inflight for s in config.providers):
        raise ValueError("Parallelism exceeds provider bound")
    spec = config.providers[0]
    if any(
        model_identity(s.model) != model_identity(spec.model) for s in config.providers
    ):
        raise ValueError("All endpoints must serve the same model")
    api_keys = {
        s.name: os.environ.get(s.api_key_env, "") if s.api_key_env else "EMPTY"
        for s in config.providers
    }
    if not all(api_keys.values()):
        raise ValueError("Empty credential alias in provider configuration")
    model_config = {
        "model": spec.model,
        "annotation_version": args.annotation_version,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "reasoning_enabled": args.enable_thinking,
        "reasoning_effort": args.reasoning_effort,
    }
    if args.repair_quotes:
        model_config["quote_repair"] = "source_quote_repair.v1"
    if args.quote_catalog:
        model_config["quote_transport"] = "source_quote_catalog.v1"
    inputs = sorted(args.inputs.glob("*.features.parquet"))
    summary = json.loads((args.inputs / "summary.json").read_text())
    assert len(inputs) == len(summary["tasks"]) and summary["features_written"]
    contract = {
        "model_config": model_config,
        "system_prompt_sha256": hashlib.sha256(
            extraction_messages({"input": {}}, args.annotation_version)[0]["content"].encode()
        ).hexdigest(),
        "inputs": {str(p.resolve()): digest(p) for p in inputs},
        "expected_source_rows": summary["source_rows"],
        "expected_llm_candidate_rows": summary["llm_candidate_rows"],
    }
    if args.quote_catalog:
        contract["effective_system_prompt_sha256"] = hashlib.sha256(
            (extraction_messages({"input": {}}, args.annotation_version)[0]["content"]
             + QUOTE_TRANSPORT_INSTRUCTION).encode()
        ).hexdigest()
    manifest = {
        "contract": contract,
        "provider_config": config.public_dict(),
        "request_parallelism": args.parallelism,
        "max_attempts": args.max_attempts,
        "request_timeout_seconds": args.timeout,
        "pid": os.getpid(),
        "started_at": utc(),
        "runner_sha256": digest(__file__),
        "model_calls_during_preparation": 0,
        "cache_semantics": "exact input/prompt/model; unknown in-flight requests may be repeated after interruption",
        "raw_sources_and_formal_retrieval_unchanged": True,
        "database_path": str(
            (args.database_path or args.output_dir / "extraction.sqlite").resolve()
        ),
        "shared_checkpoint_interval_seconds": 600 if args.database_path else None,
    }
    db_path = args.database_path or args.output_dir / "extraction.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = database(db_path)
    try:
        old = db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
        if old:
            model_config = compatible_cache_config(
                json.loads(old[0])["model_config"], model_config
            )
            contract["model_config"] = model_config
        manifest["cache_model_config_is_legacy_transport"] = "base_url" in model_config
        ingest(db, inputs, contract, args.output_dir / "progress.json")
        manifest["reused_annotations"] = [reuse_annotations(db, p, contract) for p in args.reuse_annotations]
        prioritize(db, args.priority_records)
        manifest['priority_records'] = ({'path':str(args.priority_records.resolve()),
            'sha256':digest(args.priority_records)} if args.priority_records else None)
        manifest["unique_llm_requests"] = db.execute(
            "SELECT COUNT(*) FROM jobs"
        ).fetchone()[0]
        write_json(args.output_dir / "run_manifest.json", manifest)
        with (args.output_dir / "launches.jsonl").open("a") as f:
            f.write(dump(manifest) + "\n")
        if args.prepare_only:
            write_json(
                args.output_dir / "progress.json",
                {
                    "status": "prepared",
                    "unique_requests": manifest["unique_llm_requests"],
                    "updated_at": utc(),
                    "model_calls": 0,
                },
            )
            return 0
        return asyncio.run(extract(db, args, config, model_config, api_keys))
    finally:
        db.close()
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
