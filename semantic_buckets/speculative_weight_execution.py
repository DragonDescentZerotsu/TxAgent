"""Speculative local execution for semantic-bucket weight requests."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import time
from typing import Any
from urllib.parse import urlsplit
import uuid

import httpx

from data.processing.llm_api import async_openai_compatible_client
from semantic_buckets import bioavailability_semantic_readout_v1 as core
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "semantic_weight_speculative_first8.v1"
AGGREGATION_METHOD = "first_8_valid_mean_v1"
DEFAULT_BASE_URL = "http://dgx008:50001/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REQUIRED_REPLICAS = 8
MAX_FANOUT = 128
MAX_ROUND_REQUESTS = 9


def aggregation_method(required: int) -> str:
    return f"first_{required}_valid_mean_v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS speculative_replica_receipts (
            request_id TEXT NOT NULL,
            execution_id TEXT NOT NULL,
            replica_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            completion_rank INTEGER,
            receipt_json TEXT NOT NULL,
            PRIMARY KEY (request_id, execution_id, replica_id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS speculative_aggregates (
            request_id TEXT PRIMARY KEY,
            method TEXT NOT NULL,
            replica_count INTEGER NOT NULL,
            aggregate_json TEXT NOT NULL,
            benchmark_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.commit()


def _request_payload(row: Mapping[str, Any], model: str) -> dict[str, Any]:
    effort = row["reasoning_effort"]
    approved_medium_luna = (
        effort == "medium" and model == "openai/gpt-6-luna"
        and json.loads(row["validation_json"]).get("execution_backend") == "luna_standard"
    )
    if effort != "high" and not approved_medium_luna:
        raise ValueError("speculative requests require high reasoning or approved medium Luna")
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return valid JSON."},
            {"role": "user", "content": row["prompt"]},
        ],
        "reasoning_effort": effort,
        "response_format": {"type": "json_object"},
        "max_tokens": int(row["max_tokens"]),
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }
    return request


async def _completion(client: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    completion = await client.chat.completions.create(**request)
    message = completion.choices[0].message
    extras = getattr(message, "model_extra", None) or {}
    reasoning = (
        getattr(message, "reasoning_content", None)
        or getattr(message, "reasoning", None)
        or extras.get("reasoning_content")
        or extras.get("reasoning")
    )
    return {
        "generation_id": getattr(completion, "id", None),
        "model": completion.model,
        "allowed_served_models": list(
            getattr(completion, "allowed_served_models", ()) or ()
        ),
        "provider_name": getattr(completion, "provider_name", None),
        "provider_base_url": getattr(completion, "provider_base_url", None),
        "execution_provider": getattr(completion, "execution_provider", None),
        "execution_provider_attempts": getattr(
            completion, "execution_provider_attempts", None
        ),
        "usage": completion.usage.model_dump() if completion.usage is not None else None,
        "content": message.content,
        "reasoning_content": reasoning,
    }


async def _one_replica(
    client: Any, row: Mapping[str, Any], replica_id: int, model: str
) -> dict[str, Any]:
    started_wall = _now()
    started = time.monotonic()
    raw_response = reasoning = None
    try:
        completion = await _completion(client, _request_payload(row, model))
        raw_response = completion["content"]
        reasoning = completion["reasoning_content"]
        allowed_models = set(completion.get("allowed_served_models") or [model])
        if completion["model"] not in allowed_models:
            raise ValueError(f"served model changed: {completion['model']!r}")
        validation = json.loads(row["validation_json"])
        parsed = core._validate_model_response(
            str(row["kind"]), json.loads(raw_response or ""), validation
        )
        return _replica_receipt(
            replica_id, "valid", started_wall, started, completion,
            response=parsed, reasoning_content=reasoning,
        )
    except asyncio.CancelledError:
        return _replica_receipt(
            replica_id, "cancelled", started_wall, started, {},
            reasoning_content=reasoning,
        )
    except Exception as error:
        return _replica_receipt(
            replica_id, "invalid", started_wall, started, {},
            error=f"{type(error).__name__}: {error}", raw_response=raw_response,
            reasoning_content=reasoning,
        )


def _replica_receipt(
    replica_id: int, status: str, started_at: str, started: float,
    completion: Mapping[str, Any], **extra: Any,
) -> dict[str, Any]:
    return {
        "replica_id": replica_id,
        "status": status,
        "started_at": started_at,
        "finished_at": _now(),
        "started_monotonic": started,
        "finished_monotonic": time.monotonic(),
        "generation_id": completion.get("generation_id"),
        "model": completion.get("model"),
        "provider_name": completion.get("provider_name"),
        "provider_base_url": completion.get("provider_base_url"),
        "execution_provider": completion.get("execution_provider"),
        "execution_provider_attempts": completion.get("execution_provider_attempts"),
        "usage": completion.get("usage"),
        **extra,
    }


def _consume_cancelled_task(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except BaseException:
        pass


async def collect_replicas(
    call: Callable[[int], Awaitable[dict[str, Any]]], *, initial_fanout: int,
    hedge_seconds: float, required: int = REQUIRED_REPLICAS,
    maximum: int = MAX_FANOUT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float | None]:
    if not required <= initial_fanout <= maximum:
        raise ValueError("fanout must cover the required replicas and not exceed maximum")
    active: dict[asyncio.Task[dict[str, Any]], tuple[int, float]] = {}
    receipts: list[dict[str, Any]] = []
    launched = 0

    def launch(target: int) -> None:
        nonlocal launched
        for replica_id in range(launched, target):
            task = asyncio.create_task(call(replica_id))
            active[task] = (replica_id, time.monotonic())
        launched = target

    launch(initial_fanout)
    deadline = asyncio.get_running_loop().time() + hedge_seconds
    while active:
        timeout = max(0.0, deadline - asyncio.get_running_loop().time())
        done, pending = await asyncio.wait(
            active, timeout=timeout if launched < maximum else None,
            return_when=asyncio.FIRST_COMPLETED,
        )
        receipts.extend(await asyncio.gather(*done))
        active = {task: active[task] for task in pending}
        valid = sorted(
            (row for row in receipts if row["status"] == "valid"),
            key=lambda row: (row["finished_monotonic"], row["replica_id"]),
        )
        if len(valid) >= required:
            cancelled_at = time.monotonic()
            for task, (replica_id, started) in active.items():
                task.cancel()
                task.add_done_callback(_consume_cancelled_task)
                receipts.append({
                    "replica_id": replica_id, "status": "cancelled",
                    "started_monotonic": started,
                    "finished_monotonic": cancelled_at,
                    "cancel_requested_at": _now(),
                })
            break
        if (not done or not active) and launched < maximum:
            launch(min(maximum, launched * 2))
            deadline = asyncio.get_running_loop().time() + hedge_seconds
    valid = sorted(
        (row for row in receipts if row["status"] == "valid"),
        key=lambda row: (row["finished_monotonic"], row["replica_id"]),
    )
    for rank, receipt in enumerate(valid, 1):
        receipt["completion_rank"] = rank
    chosen = valid[:required]
    elapsed = None
    if len(chosen) == required:
        elapsed = chosen[-1]["finished_monotonic"] - min(
            row["started_monotonic"] for row in receipts
        )
    return receipts, chosen, elapsed


async def collect_fixed_replicas(
    call: Callable[[int], Awaitable[dict[str, Any]]], *, total: int, required: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float | None,
           asyncio.Task[list[dict[str, Any]]] | None]:
    """Release the first valid quorum while the remaining replicas keep draining."""
    if not 1 <= required <= total:
        raise ValueError("fixed replicas require 1 <= required <= total")
    active = {asyncio.create_task(call(replica_id)) for replica_id in range(total)}
    receipts: list[dict[str, Any]] = []
    valid_count = 0
    while active:
        done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
        completed = sorted(await asyncio.gather(*done), key=lambda row: (
            row["finished_monotonic"], row["replica_id"]
        ))
        for receipt in completed:
            if receipt["status"] == "valid":
                valid_count += 1
                receipt["completion_rank"] = valid_count
            receipts.append(receipt)
        chosen = [row for row in receipts if row["status"] == "valid"][:required]
        if len(chosen) == required:
            elapsed = chosen[-1]["finished_monotonic"] - min(
                row["started_monotonic"] for row in receipts
            )

            async def drain() -> list[dict[str, Any]]:
                late: list[dict[str, Any]] = []
                next_rank = valid_count
                pending = active
                while pending:
                    finished, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )
                    batch = sorted(await asyncio.gather(*finished), key=lambda row: (
                        row["finished_monotonic"], row["replica_id"]
                    ))
                    for receipt in batch:
                        if receipt["status"] == "valid":
                            next_rank += 1
                            receipt["completion_rank"] = next_rank
                        late.append(receipt)
                return late

            return receipts, chosen, elapsed, asyncio.create_task(drain())
    return receipts, [], None, None


def aggregate_responses(
    chosen: Sequence[Mapping[str, Any]], aliases: Sequence[str], *,
    required: int = REQUIRED_REPLICAS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(chosen) != required:
        raise ValueError(f"aggregation requires exactly {required} replicas")
    vectors = []
    responses = []
    for receipt in chosen:
        response = receipt["response"]
        by_alias = {score["candidate"]: score for score in response["scores"]}
        vectors.append([float(by_alias[alias]["weight"]) for alias in aliases])
        responses.append(by_alias)
    decimal_means = [
        sum((Decimal(str(value)) for value in column), Decimal()) / Decimal(len(chosen))
        for column in zip(*vectors, strict=True)
    ]
    means = [float(value) for value in decimal_means]
    distances = [sum((value - mean) ** 2 for value, mean in zip(vector, means, strict=True))
                 for vector in vectors]
    medoid = min(range(len(chosen)), key=lambda index: (distances[index], index))
    scores, components = [], {}
    for index, alias in enumerate(aliases):
        values = [vector[index] for vector in vectors]
        rounded = float(decimal_means[index].quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        ))
        scores.append({
            "candidate": alias, "weight": rounded,
            "rationale": responses[medoid][alias]["rationale"],
        })
        components[alias] = {
            "mean": means[index], "rounded_weight": rounded,
            "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values), "component_weights": values,
        }
    metadata = {
        "aggregation_method": aggregation_method(required),
        "replicate_count": len(chosen),
        "medoid_replica_id": chosen[medoid]["replica_id"],
        "valid_replica_ids": [row["replica_id"] for row in chosen],
        "components": components,
    }
    return {"scores": scores}, metadata


async def execute_request(
    client: Any, row: Mapping[str, Any], *, model: str, initial_fanout: int,
    hedge_seconds: float, maximum: int = MAX_FANOUT,
    required: int = REQUIRED_REPLICAS,
) -> dict[str, Any]:
    execution_id = uuid.uuid4().hex
    call = lambda replica_id: _one_replica(client, row, replica_id, model)
    receipts, chosen, elapsed = await collect_replicas(
        call, initial_fanout=initial_fanout, hedge_seconds=hedge_seconds,
        maximum=maximum, required=required,
    )
    result = {
        "request_id": row["request_id"], "execution_id": execution_id,
        "receipts": receipts, "time_to_required_seconds": elapsed,
        "status": "failed", "error": None,
        "aggregation_method": aggregation_method(required),
    }
    if required == REQUIRED_REPLICAS:
        result["time_to_eight_seconds"] = elapsed
    if len(chosen) != required:
        counts = dict(Counter(receipt["status"] for receipt in receipts))
        result["error"] = f"fewer than {required} valid replicas after fanout {maximum}: {counts}"
        return result
    validation = json.loads(row["validation_json"])
    response, aggregate = aggregate_responses(
        chosen, validation["candidate_aliases"], required=required
    )
    result.update(status="complete", response=response, aggregate=aggregate)
    return result


async def execute_fixed_request(
    targets: Sequence[tuple[Any, str]], row: Mapping[str, Any], *, required: int,
) -> tuple[dict[str, Any], asyncio.Task[list[dict[str, Any]]] | None]:
    """Execute one fixed provider mix and release its first valid quorum."""
    execution_id = uuid.uuid4().hex
    call = lambda replica_id: _one_replica(
        targets[replica_id][0], row, replica_id, targets[replica_id][1]
    )
    receipts, chosen, elapsed, drain = await collect_fixed_replicas(
        call, total=len(targets), required=required
    )
    result = {
        "request_id": row["request_id"], "execution_id": execution_id,
        "receipts": receipts, "time_to_required_seconds": elapsed,
        "status": "failed", "error": None,
        "aggregation_method": aggregation_method(required),
        "total_replica_count": len(targets),
    }
    if len(chosen) != required:
        counts = dict(Counter(receipt["status"] for receipt in receipts))
        result["error"] = f"fewer than {required} valid replicas after fixed mix: {counts}"
        return result, None
    validation = json.loads(row["validation_json"])
    response, aggregate = aggregate_responses(
        chosen, validation["candidate_aliases"], required=required
    )
    result.update(status="complete", response=response, aggregate=aggregate)
    return result, drain


def _usage_totals(receipts: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    prompt = completion = 0
    for receipt in receipts:
        usage = receipt.get("usage") or {}
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
    return prompt, completion


def persist_result(
    connection: sqlite3.Connection, result: Mapping[str, Any], *, base_url: str,
    model: str, benchmark_sha256: str,
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
            (request_id, result["execution_id"], receipt["replica_id"],
             receipt["status"], receipt.get("completion_rank"),
             core._canonical_json(receipt)),
        )
    _persist_aggregate(connection, result, benchmark_sha256)
    aggregate_receipt = {
        "attempt": attempt, "execution_id": result["execution_id"],
        "aggregation_method": result["aggregation_method"],
        "time_to_required_seconds": result["time_to_required_seconds"],
        "replica_status_counts": dict(Counter(
            receipt["status"] for receipt in result["receipts"]
        )),
        "expected_replica_count": int(
            result.get("total_replica_count", len(result["receipts"]))
        ),
        "all_replicas_stored": len(result["receipts"]) == int(
            result.get("total_replica_count", len(result["receipts"]))
        ),
        "benchmark_sha256": benchmark_sha256,
        "response": result.get("response"),
        "aggregate": result.get("aggregate"),
    }
    connection.execute(
        "INSERT OR REPLACE INTO request_attempt_receipts VALUES (?,?,?)",
        (request_id, attempt, core._canonical_json(aggregate_receipt)),
    )
    medoid = _medoid_receipt(result)
    connection.execute(
        """UPDATE requests SET status=?,attempts=?,response_json=?,reasoning_content=?,
           served_model=?,provider_name=?,provider_base_url=?,input_tokens=?,
           output_tokens=?,error=? WHERE request_id=?""",
        (result["status"], attempt,
         core._canonical_json(result.get("response")) if result.get("response") else None,
         medoid.get("reasoning_content") if medoid else None,
         medoid.get("model", model) if medoid else None,
         (medoid.get("provider_name") if medoid else None)
         or f"{urlsplit(base_url).netloc.replace(':', '_')}_speculative",
         (medoid.get("provider_base_url") if medoid else None) or base_url,
         int(row["input_tokens"]) + prompt_tokens,
         int(row["output_tokens"]) + output_tokens,
         result.get("error"), request_id),
    )
    connection.commit()


def persist_late_receipts(
    connection: sqlite3.Connection, result: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> None:
    """Append replicas that finished after the aggregate quorum was released."""
    if not receipts:
        return
    request_id = str(result["request_id"])
    for receipt in receipts:
        connection.execute(
            "INSERT OR IGNORE INTO speculative_replica_receipts VALUES (?,?,?,?,?,?)",
            (request_id, result["execution_id"], receipt["replica_id"],
             receipt["status"], receipt.get("completion_rank"),
             core._canonical_json(receipt)),
        )
    prompt_tokens, output_tokens = _usage_totals(receipts)
    connection.execute(
        "UPDATE requests SET input_tokens=input_tokens+?,output_tokens=output_tokens+? "
        "WHERE request_id=?",
        (prompt_tokens, output_tokens, request_id),
    )
    row = connection.execute(
        "SELECT attempts FROM requests WHERE request_id=?", (request_id,)
    ).fetchone()
    attempt = int(row["attempts"])
    receipt_row = connection.execute(
        "SELECT receipt_json FROM request_attempt_receipts "
        "WHERE request_id=? AND attempt=?", (request_id, attempt),
    ).fetchone()
    aggregate_receipt = json.loads(receipt_row["receipt_json"])
    statuses = connection.execute(
        "SELECT status,COUNT(*) FROM speculative_replica_receipts "
        "WHERE request_id=? AND execution_id=? GROUP BY status",
        (request_id, result["execution_id"]),
    ).fetchall()
    aggregate_receipt["final_replica_status_counts"] = dict(statuses)
    aggregate_receipt["all_replicas_stored"] = sum(count for _, count in statuses) == int(
        result["total_replica_count"]
    )
    connection.execute(
        "UPDATE request_attempt_receipts SET receipt_json=? "
        "WHERE request_id=? AND attempt=?",
        (core._canonical_json(aggregate_receipt), request_id, attempt),
    )
    connection.commit()


def all_replicas_stored(
    connection: sqlite3.Connection, request_id: str, *, expected_replicas: int | None = None
) -> bool:
    """Return whether the latest execution has a receipt for every replica."""
    row = connection.execute(
        "SELECT receipt_json FROM request_attempt_receipts "
        "WHERE request_id=? ORDER BY attempt DESC LIMIT 1",
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
        "SELECT COUNT(*) FROM speculative_replica_receipts "
        "WHERE request_id=? AND execution_id=?",
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
        (result["request_id"], result["aggregate"]["aggregation_method"],
         result["aggregate"]["replicate_count"],
         core._canonical_json(result["aggregate"]), benchmark_sha256, _now()),
    )


def _medoid_receipt(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if result["status"] != "complete":
        return None
    wanted = result["aggregate"]["medoid_replica_id"]
    return next(row for row in result["receipts"] if row["replica_id"] == wanted)


async def _load_counts(base_url: str) -> tuple[int, int]:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(base_url.rstrip("/").removesuffix("/v1") + "/v1/loads")
        if response.status_code == 404:
            response = await client.get(base_url.rstrip("/").removesuffix("/v1") + "/loads")
        response.raise_for_status()
    loads = response.json().get("loads")
    if not isinstance(loads, list) or not loads:
        raise ValueError("endpoint returned invalid load telemetry")
    return (
        sum(int(row["num_running_reqs"]) for row in loads),
        sum(int(row["num_waiting_reqs"]) for row in loads),
    )


async def wait_for_drain(base_url: str, timeout_seconds: float = 120) -> float:
    started = time.monotonic()
    while True:
        running, waiting = await _load_counts(base_url)
        if running == 0 and waiting == 0:
            return time.monotonic() - started
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(
                f"endpoint did not drain in {timeout_seconds}s: running={running}, waiting={waiting}"
            )
        await asyncio.sleep(1)


def load_benchmark(path: Path) -> tuple[dict[str, Any], str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION or manifest.get("status") != "complete":
        raise ValueError("speculative benchmark is not complete")
    selection = manifest.get("selection") or {}
    if selection.get("required_replicas") != REQUIRED_REPLICAS:
        raise ValueError("speculative benchmark uses a different aggregation contract")
    return manifest, _sha256(path)


async def execute_pending(
    connection: sqlite3.Connection, request_ids: Sequence[str], benchmark_path: Path,
    *, required: int = REQUIRED_REPLICAS, fanout: int | None = None,
    maximum_requests: int = MAX_ROUND_REQUESTS, require_endpoint_drain: bool = True,
    base_url: str | None = None,
) -> dict[str, Any]:
    benchmark, benchmark_sha256 = load_benchmark(benchmark_path)
    selection = benchmark["selection"]
    endpoint = dict(benchmark["endpoint"])
    if base_url is not None:
        endpoint["base_url"] = base_url
    initial_fanout = int(fanout or selection["fanout"])
    maximum = initial_fanout if fanout is not None else MAX_FANOUT
    if not 1 <= required <= initial_fanout <= MAX_FANOUT:
        raise ValueError("speculative execution requires 1 <= required <= fanout <= 128")
    rows = _pending_rows(connection, request_ids)
    if not rows:
        return {"completed": 0, "endpoint_drained": True}
    if len(rows) > maximum_requests:
        raise ValueError(f"speculative round exceeds {maximum_requests} requests")
    ensure_tables(connection)
    if require_endpoint_drain:
        await wait_for_drain(endpoint["base_url"])
    client, credential = async_openai_compatible_client(
        base_url=endpoint["base_url"], provider="local", env_file=None,
        max_connections=len(rows) * maximum, timeout_s=300, max_retries=0,
    )
    if credential:
        raise ValueError("local speculative execution unexpectedly selected a credential")
    failures = []
    try:
        models = sorted(model.id for model in (await client.models.list()).data)
        if models != [endpoint["model"]]:
            raise ValueError(f"local endpoint model contract changed: {models}")
        tasks = [asyncio.create_task(execute_request(
            client, row, model=endpoint["model"],
            initial_fanout=initial_fanout,
            hedge_seconds=float(selection["hedge_seconds"]),
            maximum=maximum, required=required,
        )) for row in rows]
        for task in asyncio.as_completed(tasks):
            result = await task
            persist_result(
                connection, result, base_url=endpoint["base_url"],
                model=endpoint["model"], benchmark_sha256=benchmark_sha256,
            )
            if result["status"] != "complete":
                failures.append(result)
    finally:
        await client.close()
    drain_seconds = (await wait_for_drain(endpoint["base_url"])
                     if require_endpoint_drain else None)
    if failures:
        raise RuntimeError("speculative requests failed closed: " + ", ".join(
            f"{row['request_id']} ({row['error']})" for row in failures
        ))
    return {
        "completed": len(rows), "fanout": initial_fanout,
        "maximum_fanout": maximum, "required_replicas": required,
        "aggregation_method": aggregation_method(required),
        "endpoint_drained": require_endpoint_drain, "drain_seconds": drain_seconds,
        "benchmark_sha256": benchmark_sha256,
    }


def _pending_rows(
    connection: sqlite3.Connection, request_ids: Sequence[str]
) -> list[dict[str, Any]]:
    unique = list(dict.fromkeys(request_ids))
    if not unique:
        return []
    placeholders = ",".join("?" for _ in unique)
    rows = {row["request_id"]: dict(row) for row in connection.execute(
        f"SELECT * FROM requests WHERE request_id IN ({placeholders})", unique
    )}
    if set(rows) != set(unique):
        raise ValueError("requested an unknown cached request")
    return [rows[request_id] for request_id in unique
            if rows[request_id]["status"] != "complete"]


def _benchmark_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = [dict(row) for row in connection.execute(
        """SELECT * FROM requests WHERE kind='weight_assignment' AND status='complete'
           AND phase NOT LIKE '%-r0000-%' ORDER BY length(prompt),request_id"""
    )]
    if len(rows) < 10:
        raise ValueError("not enough completed anchored requests to benchmark")
    positions = [round((len(rows) - 1) * quantile) for quantile in (0.25, 0.5, 0.9)]
    return [rows[position] for position in positions]


def _openrouter_median(connection: sqlite3.Connection) -> float:
    rows = connection.execute(
        """SELECT r.request_id,a.receipt_json FROM requests r
           JOIN request_attempt_receipts a ON a.request_id=r.request_id
           WHERE r.status='complete' AND r.phase NOT LIKE '%-r0000-%'
           ORDER BY r.rowid DESC,a.attempt DESC"""
    )
    elapsed, seen = [], set()
    for row in rows:
        if row["request_id"] in seen:
            continue
        receipt = json.loads(row["receipt_json"])
        if "error" not in receipt and receipt.get("elapsed_seconds") is not None:
            elapsed.append(float(receipt["elapsed_seconds"]))
            seen.add(row["request_id"])
        if len(elapsed) == 100:
            break
    if len(elapsed) < 20:
        raise ValueError("insufficient OpenRouter timing receipts for benchmark")
    return statistics.median(elapsed)


def _benchmark_document(
    rows: Sequence[Mapping[str, Any]], fanouts: Sequence[int], baseline: float,
    base_url: str, model: str,
) -> dict[str, Any]:
    return {
        "version": VERSION, "status": "running", "created_at": _now(),
        "endpoint": {"base_url": base_url, "model": model,
                     "reasoning_effort": "high"},
        "aggregation_method": AGGREGATION_METHOD,
        "required_replicas": REQUIRED_REPLICAS, "fanouts": list(fanouts),
        "openrouter_recent_100_median_seconds": baseline,
        "prompts": [{
            "request_id": row["request_id"], "phase": row["phase"],
            "prompt_sha256": row["prompt_sha256"], "prompt_characters": len(row["prompt"]),
        } for row in rows],
        "cases": [],
    }


async def benchmark(
    connection: sqlite3.Connection, output: Path, *,
    fanouts: Sequence[int] = (8, 16, 32, 64, 128),
    base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    if sorted(set(fanouts)) != list(fanouts) or any(
        value < REQUIRED_REPLICAS or value > MAX_FANOUT for value in fanouts
    ):
        raise ValueError("fanouts must be unique ascending values from 8 through 128")
    rows = _benchmark_rows(connection)
    baseline = _openrouter_median(connection)
    document = _benchmark_document(rows, fanouts, baseline, base_url, model)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        for key in ("version", "endpoint", "fanouts", "prompts"):
            if existing.get(key) != document.get(key):
                raise ValueError(f"existing benchmark changed: {key}")
        document = existing
    write_json_atomic(output, document)
    await wait_for_drain(base_url)
    client, _ = async_openai_compatible_client(
        base_url=base_url, provider="local", env_file=None,
        max_connections=max(fanouts), timeout_s=300, max_retries=0,
    )
    try:
        models = sorted(item.id for item in (await client.models.list()).data)
        if models != [model]:
            raise ValueError(f"benchmark endpoint model contract changed: {models}")
        await _run_benchmark_cases(client, rows, document, output, fanouts, model, base_url)
    finally:
        await client.close()
    _select_fanout(document)
    write_json_atomic(output, document)
    return document


async def _run_benchmark_cases(
    client: Any, rows: Sequence[Mapping[str, Any]], document: dict[str, Any],
    output: Path, fanouts: Sequence[int], model: str, base_url: str,
) -> None:
    completed = {(case["fanout"], case["request_id"]) for case in document["cases"]}
    for fanout in fanouts:
        for row in rows:
            key = (fanout, row["request_id"])
            if key in completed:
                continue
            result = await execute_request(
                client, row, model=model, initial_fanout=fanout,
                hedge_seconds=3600, maximum=fanout,
            )
            drain_seconds = await wait_for_drain(base_url)
            aggregate = result.get("aggregate") or {}
            stddevs = [item["sample_stddev"]
                       for item in aggregate.get("components", {}).values()]
            document["cases"].append({
                "fanout": fanout, "request_id": row["request_id"],
                "phase": row["phase"], "status": result["status"],
                "time_to_eight_seconds": result["time_to_eight_seconds"],
                "drain_seconds": drain_seconds,
                "replica_status_counts": dict(Counter(
                    receipt["status"] for receipt in result["receipts"]
                )),
                "median_candidate_stddev": statistics.median(stddevs) if stddevs else None,
                "error": result.get("error"),
            })
            write_json_atomic(output, document)


def _select_fanout(document: dict[str, Any]) -> None:
    summaries = []
    for fanout in document["fanouts"]:
        cases = [case for case in document["cases"] if case["fanout"] == fanout]
        valid = len(cases) == 3 and all(case["status"] == "complete" for case in cases)
        median = statistics.median(case["time_to_eight_seconds"] for case in cases) \
            if valid else None
        summaries.append({
            "fanout": fanout, "complete": valid,
            "median_time_to_eight_seconds": median,
            "maximum_drain_seconds": max((case["drain_seconds"] for case in cases), default=None),
            "beats_openrouter_baseline": bool(valid and median < document[
                "openrouter_recent_100_median_seconds"
            ]),
        })
    valid = [row for row in summaries if row["complete"]]
    if not valid:
        document.update(status="no_eligible_fanout", fanout_summaries=summaries)
        raise ValueError("no fanout completed all benchmark prompts")
    fastest = min(row["median_time_to_eight_seconds"] for row in valid)
    eligible = [row for row in valid
                if row["median_time_to_eight_seconds"] <= fastest * 1.10
                and row["maximum_drain_seconds"] <= 30]
    if not eligible:
        document.update(status="no_eligible_fanout", fanout_summaries=summaries)
        raise ValueError("no fanout met relative-speed and drain criteria")
    selected = min(eligible, key=lambda row: row["fanout"])
    document.update(
        status="complete", completed_at=_now(), fanout_summaries=summaries,
        selection={
            "fanout": selected["fanout"], "required_replicas": REQUIRED_REPLICAS,
            "maximum_fanout": MAX_FANOUT,
            "hedge_seconds": max(15, selected["median_time_to_eight_seconds"] * 0.75),
            "rule": "smallest fanout within 10 percent of fastest median with drain at most 30 seconds",
            "beats_openrouter_baseline": selected["beats_openrouter_baseline"],
        },
    )


def aggregate_lookup(connection: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='speculative_aggregates'"
    ).fetchone()
    if not exists:
        return {}
    return {row["request_id"]: json.loads(row["aggregate_json"])
            for row in connection.execute(
                "SELECT request_id,aggregate_json FROM speculative_aggregates"
            )}
