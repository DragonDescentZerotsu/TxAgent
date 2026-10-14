"""Shared completion execution, retries and replica collection."""

from __future__ import annotations
from predict.api_client.pool import (
    ProviderPoolCompletionAdapter as _ProviderPoolCompletionAdapter,
)
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sqlite3
import threading
import time
from types import SimpleNamespace
from typing import Any
import httpx
import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
import uuid
from .storage import _canonical_json, _file_sha256, _now, _queue_request, _request_id

_REQUEST_DATABASE_WRITE_LOCK = threading.RLock()


def _endpoint_contract(*, policy: Mapping) -> tuple[int, list[str]]:
    if policy["MODEL_CONTEXT_LIMIT_OVERRIDE"] is not None:
        if policy["MODEL_CONTEXT_LIMIT_OVERRIDE"] <= 0:
            raise ValueError("MODEL_CONTEXT_LIMIT_OVERRIDE must be positive")
        return (policy["MODEL_CONTEXT_LIMIT_OVERRIDE"], [policy["MODEL"]])
    response = httpx.get(f"{policy['BASE_URL']}/models", timeout=10)
    response.raise_for_status()
    rows = response.json().get("data", [])
    models = sorted((str(row.get("id")) for row in rows))
    matching = [row for row in rows if row.get("id") == policy["MODEL"]]
    if len(matching) != 1:
        raise ValueError(
            f"endpoint does not uniquely serve {policy['MODEL']}: {models}"
        )
    maximum = int(matching[0].get("max_model_len") or 0)
    if maximum <= 0:
        raise ValueError("endpoint did not report max_model_len")
    return (maximum, models)


class _IndependentCompletionPool:
    """Route calls across independent endpoint-specific concurrency pools."""

    def __init__(
        self,
        providers: Sequence[tuple[str, str, Any, int]],
        *,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60,
    ) -> None:
        self._providers = list(providers)
        self._active = [0] * len(self._providers)
        self._completed = [0] * len(self._providers)
        self._failures = [0] * len(self._providers)
        self._consecutive_failures = [0] * len(self._providers)
        self._circuit_open_until = [0.0] * len(self._providers)
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._next = 0
        self._condition = threading.Condition()
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> Any:
        with self._condition:
            while True:
                now = time.monotonic()
                available = [
                    index
                    for index, provider in enumerate(self._providers)
                    if self._active[index] < provider[3]
                    and self._circuit_open_until[index] <= now
                ]
                if available:
                    index = min(
                        available,
                        key=lambda candidate: (
                            self._active[candidate] / self._providers[candidate][3],
                            (candidate - self._next) % len(self._providers),
                        ),
                    )
                    self._active[index] += 1
                    self._next = (index + 1) % len(self._providers)
                    break
                reopen_times = [
                    value for value in self._circuit_open_until if value > now
                ]
                timeout = min(reopen_times) - now if reopen_times else None
                self._condition.wait(timeout=timeout)
        name, base_url, client, _ = self._providers[index]
        try:
            completion = _stream_completion(client, kwargs)
        except Exception:
            with self._condition:
                self._active[index] -= 1
                self._failures[index] += 1
                self._consecutive_failures[index] += 1
                if self._consecutive_failures[index] >= self._failure_threshold:
                    self._circuit_open_until[index] = (
                        time.monotonic() + self._cooldown_seconds
                    )
                self._condition.notify_all()
            raise
        with self._condition:
            self._active[index] -= 1
            self._completed[index] += 1
            self._consecutive_failures[index] = 0
            self._circuit_open_until[index] = 0
            self._condition.notify_all()
        return SimpleNamespace(
            id=getattr(completion, "id", None),
            model=completion.model,
            usage=completion.usage,
            choices=completion.choices,
            provider_name=name,
            provider_base_url=base_url,
        )

    def snapshot(self) -> list[dict[str, Any]]:
        with self._condition:
            now = time.monotonic()
        return [
            {
                "name": provider[0],
                "base_url": provider[1],
                "max_inflight": provider[3],
                "active": self._active[index],
                "completed": self._completed[index],
                "failures": self._failures[index],
                "circuit_open": self._circuit_open_until[index] > now,
            }
            for index, provider in enumerate(self._providers)
        ]


class _CompositeCompletionPool:
    """Share work across independently bounded local and paid pools."""

    def __init__(self, pools: Sequence[tuple[str, Any, int]]) -> None:
        self._pools = list(pools)
        self._active = [0] * len(self._pools)
        self._completed = [0] * len(self._pools)
        self._failures = [0] * len(self._pools)
        self._condition = threading.Condition()
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> Any:
        with self._condition:
            while True:
                available = [
                    index
                    for index, (_, _, capacity) in enumerate(self._pools)
                    if self._active[index] < capacity
                ]
                if available:
                    index = min(
                        available,
                        key=lambda candidate: (
                            self._active[candidate] / self._pools[candidate][2],
                            self._completed[candidate],
                        ),
                    )
                    self._active[index] += 1
                    break
                self._condition.wait()
        try:
            response = self._pools[index][1].chat.completions.create(**kwargs)
        except Exception:
            with self._condition:
                self._active[index] -= 1
                self._failures[index] += 1
                self._condition.notify_all()
            raise
        with self._condition:
            self._active[index] -= 1
            self._completed[index] += 1
            self._condition.notify_all()
        return response

    def snapshot(self) -> list[dict[str, Any]]:
        with self._condition:
            return [
                {
                    "name": name,
                    "capacity": capacity,
                    "active": self._active[index],
                    "completed": self._completed[index],
                    "failures": self._failures[index],
                    "pool": pool.snapshot(),
                }
                for index, (name, pool, capacity) in enumerate(self._pools)
            ]


def _stream_completion(client: Any, kwargs: Mapping[str, Any]) -> Any:
    """Collect one streaming response while keeping the legacy completion shape."""
    request = dict(kwargs)
    request.update(stream=True, stream_options={"include_usage": True})
    content: list[str] = []
    reasoning: list[str] = []
    generation_id = model = None
    usage = None
    response = client.chat.completions.create(**request)
    if hasattr(response, "choices"):
        return response
    for chunk in response:
        generation_id = getattr(chunk, "id", None) or generation_id
        model = getattr(chunk, "model", None) or model
        usage = getattr(chunk, "usage", None) or usage
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            content.append(delta.content)
        extras = getattr(delta, "model_extra", None) or {}
        thought = (
            getattr(delta, "reasoning_content", None)
            or getattr(delta, "reasoning", None)
            or extras.get("reasoning_content")
            or extras.get("reasoning")
        )
        if thought:
            reasoning.append(thought)
    message = SimpleNamespace(
        content="".join(content),
        reasoning_content="".join(reasoning) or None,
        reasoning=None,
        model_extra={},
    )
    return SimpleNamespace(
        id=generation_id,
        model=model,
        usage=usage,
        choices=[SimpleNamespace(message=message)],
    )


def _build_completion_pool(
    per_endpoint_parallelism: int, *, policy: Mapping
) -> tuple[Any, list[dict[str, Any]], int]:
    """Build and preflight each configured endpoint without exposing credentials."""
    from predict.api_client.transport import DEFAULT_ENV_FILE, openai_compatible_client

    providers = []
    receipts = []
    for spec in policy["ENDPOINTS"]:
        provider = spec["provider"]
        max_inflight = int(spec.get("max_inflight", per_endpoint_parallelism))
        if max_inflight <= 0:
            raise ValueError(f"endpoint {spec['name']!r} max_inflight must be positive")
        env_file = None if provider == "local" else DEFAULT_ENV_FILE
        client, credential = openai_compatible_client(
            base_url=spec["base_url"],
            provider=provider,
            env_file=env_file,
            credential_env=spec.get("credential_env") or None,
            max_connections=max_inflight,
            timeout_s=policy["REQUEST_TIMEOUT_S"],
            max_retries=0,
        )
        models = sorted((model.id for model in client.models.list().data))
        if policy["MODEL"] not in models:
            raise ValueError(
                f"endpoint {spec['name']!r} does not advertise {policy['MODEL']!r}: {models}"
            )
        if provider == "local" and credential:
            raise ValueError("local DeepSeek unexpectedly selected a credential")
        providers.append((spec["name"], spec["base_url"], client, max_inflight))
        receipts.append(
            {
                "name": spec["name"],
                "base_url": spec["base_url"],
                "provider": provider,
                "credential_env": credential,
                "requested_model": policy["MODEL"],
                "advertised_models": models,
                "max_inflight": max_inflight,
            }
        )
    local_pool = _IndependentCompletionPool(providers)
    local_capacity = sum((provider[3] for provider in providers))
    if policy["PROVIDER_POOL_CONFIG"] is None:
        return (local_pool, receipts, local_capacity)
    from predict.api_client.pool import (
        build_provider_pool,
        load_provider_pool_config,
        primary_capacity,
        spend_budget_status,
    )

    paid_config = load_provider_pool_config(policy["PROVIDER_POOL_CONFIG"])
    paid_pool = build_provider_pool(
        paid_config,
        env_file=DEFAULT_ENV_FILE,
        timeout_s=3600,
        max_tokens=policy["HIGH_MAX_TOKENS"],
        temperature=None,
        tool_service_url="http://127.0.0.1:1",
        enable_group_tools=False,
        max_tool_rounds=0,
        reasoning_effort="high",
        enable_thinking=True,
        transport_max_retries=0,
        response_format={"type": "json_object"},
    )
    paid_capacity = primary_capacity(paid_config)
    paid_adapter = _ProviderPoolCompletionAdapter(paid_pool)
    receipts.append(
        {
            "name": "openrouter_provider_pool",
            "provider": "openrouter",
            "config_path": str(policy["PROVIDER_POOL_CONFIG"]),
            "config_sha256": _file_sha256(policy["PROVIDER_POOL_CONFIG"]),
            "max_inflight": paid_capacity,
            "spend_budget": spend_budget_status(paid_config),
            "pool": paid_config.public_dict(),
        }
    )
    return (
        _CompositeCompletionPool(
            (
                ("local", local_pool, local_capacity),
                ("openrouter", paid_adapter, paid_capacity),
            )
        ),
        receipts,
        local_capacity + paid_capacity,
    )


def _token_count(text: str, *, policy: Mapping) -> int:
    if policy["TOKENIZER_ENCODING_OVERRIDE"] is not None:
        import tiktoken

        return len(
            tiktoken.get_encoding(policy["TOKENIZER_ENCODING_OVERRIDE"]).encode(text)
        )
    tokenize_url = policy["BASE_URL"].rsplit("/v1", 1)[0] + "/tokenize"
    response = httpx.post(
        tokenize_url, json={"model": policy["MODEL"], "prompt": text}, timeout=60
    )
    response.raise_for_status()
    return int(response.json()["count"])


def _validate_model_response(
    kind: str, response: Any, validation: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        raise ValueError("response must be a JSON object")
    result = dict(response)
    if kind not in {"ranking", "weight_assignment"} and (
        not isinstance(result.get("rationale"), str) or not result["rationale"].strip()
    ):
        raise ValueError("response requires a non-empty rationale")
    if kind in {
        "select_column",
        "split_bucket",
        "readout_coherence",
        "retrieval_eligibility",
    }:
        from .bucket_construction import _validate_decision_response

        return _validate_decision_response(kind, result, validation)
    if kind in {
        "merge_values",
        "merge_buckets",
        "readout_merge_values",
        "readout_merge_profiles",
        "cross_source_merge",
    }:
        from .bucket_construction import _validate_merge_response

        return _validate_merge_response(kind, result, validation)
    if kind == "ranking":
        if set(result) != {"winner_bucket_id"}:
            raise ValueError("ranking response has unexpected fields")
        if result["winner_bucket_id"] not in validation["candidate_bucket_ids"]:
            raise ValueError("ranking winner was not supplied")
        return result
    if kind == "weight_assignment":
        from .bucket_scoring import _validate_weight_response

        return _validate_weight_response(result, validation)
    raise ValueError(f"unknown request kind: {kind}")


def _run_pending(
    connection: sqlite3.Connection,
    request_ids: Sequence[str],
    *,
    parallelism: int,
    client: Any,
    request_slots: threading.BoundedSemaphore,
    policy: Mapping,
) -> None:
    unique_request_ids = list(dict.fromkeys(request_ids))
    lookup = {}
    for start in range(0, len(unique_request_ids), 500):
        batch = unique_request_ids[start : start + 500]
        placeholders = ",".join(("?" for _ in batch))
        lookup.update(
            (
                (row["request_id"], row)
                for row in connection.execute(
                    f"SELECT * FROM requests WHERE request_id IN ({placeholders})",
                    batch,
                )
            )
        )
    rows = [lookup.get(request_id) for request_id in unique_request_ids]
    if any((row is None for row in rows)):
        raise ValueError("requested an unknown cached request")
    pending = [dict(row) for row in rows if row["status"] != "complete"]
    if not pending:
        return

    def call(row: Mapping[str, Any]) -> dict[str, Any]:
        attempts = int(row["attempts"])
        input_tokens = int(row["input_tokens"])
        output_tokens = int(row["output_tokens"])
        receipts = []
        last_error = None
        while attempts < policy["MAX_ATTEMPTS"]:
            attempts += 1
            started = time.time()
            raw_response = None
            attempted_model = None
            attempted_reasoning = None
            attempted_provider_name = None
            attempted_provider_base_url = None
            try:
                with request_slots:
                    validation = json.loads(row["validation_json"])
                    requested_model = validation.get("requested_model", policy["MODEL"])
                    request = {
                        "model": requested_model,
                        "messages": [
                            {"role": "system", "content": "Return valid JSON."},
                            {"role": "user", "content": row["prompt"]},
                        ],
                        "reasoning_effort": row["reasoning_effort"],
                        "response_format": {"type": "json_object"},
                    }
                    request[policy["COMPLETION_TOKEN_PARAMETER"]] = int(
                        row["max_tokens"]
                    )
                    if row["kind"] == "weight_assignment":
                        request["extra_body"] = dict(
                            policy["WEIGHT_REQUEST_EXTRA_BODY"]
                        )
                        if validation.get("provider_routing"):
                            routing = dict(validation["provider_routing"])
                            if routing.pop("omit_response_format", False):
                                request.pop("response_format")
                            request["extra_body"]["provider"] = routing
                    completion = client.chat.completions.create(**request)
                usage = completion.usage
                message = completion.choices[0].message
                raw_response = message.content
                attempted_model = str(completion.model)
                attempted_provider_name = getattr(completion, "provider_name", None)
                attempted_provider_base_url = getattr(
                    completion, "provider_base_url", None
                )
                extras = getattr(message, "model_extra", None) or {}
                reasoning = (
                    getattr(message, "reasoning_content", None)
                    or getattr(message, "reasoning", None)
                    or extras.get("reasoning_content")
                    or extras.get("reasoning")
                )
                attempted_reasoning = reasoning
                input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
                parsed = json.loads(message.content or "")
                parsed = _validate_model_response(str(row["kind"]), parsed, validation)
                allowed_models = set(
                    validation.get(
                        "allowed_served_models",
                        [requested_model, *policy["SERVED_MODEL_ALIASES"]],
                    )
                )
                allowed_models.update(getattr(completion, "allowed_served_models", ()))
                if str(completion.model) not in allowed_models:
                    raise ValueError(
                        f"served model changed: {completion.model!s} not in {sorted(allowed_models)}"
                    )
                execution = dict(getattr(completion, "execution_provider", {}) or {})
                receipt = {
                    "attempt": attempts,
                    "elapsed_seconds": time.time() - started,
                    "generation_id": getattr(completion, "id", None),
                    "requested_model": getattr(completion, "requested_model", None)
                    or requested_model,
                    "selected_provider_route": validation.get(
                        "selected_provider_route"
                    ),
                    "provider_pool_snapshot_sha256": validation.get(
                        "provider_pool_snapshot_sha256"
                    ),
                    "model": str(completion.model),
                    "provider_name": getattr(completion, "provider_name", None),
                    "provider_base_url": getattr(completion, "provider_base_url", None),
                    "usage": usage.model_dump() if usage is not None else None,
                    "response": parsed,
                    "reasoning_content": reasoning,
                    "execution_provider": execution or None,
                    "execution_provider_attempts": getattr(
                        completion, "execution_provider_attempts", None
                    ),
                }
                receipts.append(receipt)
                return {
                    "request_id": row["request_id"],
                    "status": "complete",
                    "attempts": attempts,
                    "response_json": _canonical_json(parsed),
                    "reasoning_content": reasoning,
                    "served_model": str(completion.model),
                    "provider_name": getattr(completion, "provider_name", None),
                    "provider_base_url": getattr(completion, "provider_base_url", None),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "error": None,
                    "receipts": receipts,
                }
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"
                receipts.append(
                    {
                        "attempt": attempts,
                        "elapsed_seconds": time.time() - started,
                        "error": last_error,
                        "raw_response": raw_response,
                        "model": attempted_model,
                        "provider_name": attempted_provider_name,
                        "provider_base_url": attempted_provider_base_url,
                        "reasoning_content": attempted_reasoning,
                    }
                )
                if attempts < policy["MAX_ATTEMPTS"]:
                    time.sleep(min(8, 2 ** (attempts - 1)))
        return {
            "request_id": row["request_id"],
            "status": "failed",
            "attempts": attempts,
            "response_json": None,
            "reasoning_content": None,
            "served_model": None,
            "provider_name": None,
            "provider_base_url": None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "error": last_error,
            "receipts": receipts,
        }

    results = []
    with ThreadPoolExecutor(max_workers=min(parallelism, len(pending))) as executor:
        futures = {executor.submit(call, row): row["request_id"] for row in pending}
        try:
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                with _REQUEST_DATABASE_WRITE_LOCK:
                    for receipt in result.pop("receipts"):
                        connection.execute(
                            "INSERT OR REPLACE INTO request_attempt_receipts VALUES (?,?,?)",
                            (
                                result["request_id"],
                                receipt["attempt"],
                                _canonical_json(receipt),
                            ),
                        )
                    connection.execute(
                        """
                        UPDATE requests
                        SET status=?,attempts=?,response_json=?,reasoning_content=?,served_model=?,
                            provider_name=?,provider_base_url=?,
                            input_tokens=?,output_tokens=?,error=?
                        WHERE request_id=?
                        """,
                        (
                            result["status"],
                            result["attempts"],
                            result["response_json"],
                            result["reasoning_content"],
                            result["served_model"],
                            result["provider_name"],
                            result["provider_base_url"],
                            result["input_tokens"],
                            result["output_tokens"],
                            result["error"],
                            result["request_id"],
                        ),
                    )
                    connection.commit()
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            raise
    connection.commit()
    failed = [result for result in results if result["status"] != "complete"]
    if failed:
        raise RuntimeError(
            "DeepSeek requests failed closed: "
            + ", ".join((f"{row['request_id']} ({row['error']})" for row in failed[:5]))
        )


def _queue_checked_request(
    connection: sqlite3.Connection,
    *,
    kind: str,
    phase: str,
    prompt: str,
    reasoning_effort: str,
    max_tokens: int,
    validation: Mapping[str, Any],
    maximum_context: int,
    commit: bool = True,
    policy: Mapping,
) -> str | None:
    if _token_count(prompt, policy=policy) + max_tokens > maximum_context:
        return None
    request_id = _request_id(kind, phase, prompt)
    return _queue_request(
        connection,
        request_id=request_id,
        kind=kind,
        phase=phase,
        prompt=prompt,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        validation=validation,
        commit=commit,
    )


VERSION = "semantic_weight_speculative_first8.v1"
AGGREGATION_METHOD = "first_8_valid_mean_v1"
DEFAULT_BASE_URL = "http://dgx008:50001/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REQUIRED_REPLICAS = 8
MAX_FANOUT = 128
MAX_ROUND_REQUESTS = 9


def _request_payload(row: Mapping[str, Any], model: str) -> dict[str, Any]:
    effort = row["reasoning_effort"]
    approved_medium_luna = (
        effort == "medium"
        and model == "openai/gpt-6-luna"
        and (
            json.loads(row["validation_json"]).get("execution_backend")
            == "luna_standard"
        )
    )
    if effort != "high" and (not approved_medium_luna):
        raise ValueError(
            "speculative requests require high reasoning or approved medium Luna"
        )
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
        "usage": completion.usage.model_dump()
        if completion.usage is not None
        else None,
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
        parsed = _validate_model_response(
            str(row["kind"]), json.loads(raw_response or ""), validation
        )
        return _replica_receipt(
            replica_id,
            "valid",
            started_wall,
            started,
            completion,
            response=parsed,
            reasoning_content=reasoning,
        )
    except asyncio.CancelledError:
        return _replica_receipt(
            replica_id,
            "cancelled",
            started_wall,
            started,
            {},
            reasoning_content=reasoning,
        )
    except Exception as error:
        return _replica_receipt(
            replica_id,
            "invalid",
            started_wall,
            started,
            {},
            error=f"{type(error).__name__}: {error}",
            raw_response=raw_response,
            reasoning_content=reasoning,
        )


def _replica_receipt(
    replica_id: int,
    status: str,
    started_at: str,
    started: float,
    completion: Mapping[str, Any],
    **extra: Any,
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
    call: Callable[[int], Awaitable[dict[str, Any]]],
    *,
    initial_fanout: int,
    hedge_seconds: float,
    required: int = REQUIRED_REPLICAS,
    maximum: int = MAX_FANOUT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float | None]:
    if not required <= initial_fanout <= maximum:
        raise ValueError(
            "fanout must cover the required replicas and not exceed maximum"
        )
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
            active,
            timeout=timeout if launched < maximum else None,
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
                receipts.append(
                    {
                        "replica_id": replica_id,
                        "status": "cancelled",
                        "started_monotonic": started,
                        "finished_monotonic": cancelled_at,
                        "cancel_requested_at": _now(),
                    }
                )
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
            (row["started_monotonic"] for row in receipts)
        )
    return (receipts, chosen, elapsed)


async def collect_fixed_replicas(
    call: Callable[[int], Awaitable[dict[str, Any]]], *, total: int, required: int
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    float | None,
    asyncio.Task[list[dict[str, Any]]] | None,
]:
    """Release the first valid quorum while the remaining replicas keep draining."""
    if not 1 <= required <= total:
        raise ValueError("fixed replicas require 1 <= required <= total")
    active = {asyncio.create_task(call(replica_id)) for replica_id in range(total)}
    receipts: list[dict[str, Any]] = []
    valid_count = 0
    while active:
        done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
        completed = sorted(
            await asyncio.gather(*done),
            key=lambda row: (row["finished_monotonic"], row["replica_id"]),
        )
        for receipt in completed:
            if receipt["status"] == "valid":
                valid_count += 1
                receipt["completion_rank"] = valid_count
            receipts.append(receipt)
        chosen = [row for row in receipts if row["status"] == "valid"][:required]
        if len(chosen) == required:
            elapsed = chosen[-1]["finished_monotonic"] - min(
                (row["started_monotonic"] for row in receipts)
            )

            async def drain() -> list[dict[str, Any]]:
                late: list[dict[str, Any]] = []
                next_rank = valid_count
                pending = active
                while pending:
                    finished, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )
                    batch = sorted(
                        await asyncio.gather(*finished),
                        key=lambda row: (row["finished_monotonic"], row["replica_id"]),
                    )
                    for receipt in batch:
                        if receipt["status"] == "valid":
                            next_rank += 1
                            receipt["completion_rank"] = next_rank
                        late.append(receipt)
                return late

            return (receipts, chosen, elapsed, asyncio.create_task(drain()))
    return (receipts, [], None, None)


async def execute_request(
    client: Any,
    row: Mapping[str, Any],
    *,
    model: str,
    initial_fanout: int,
    hedge_seconds: float,
    maximum: int = MAX_FANOUT,
    required: int = REQUIRED_REPLICAS,
) -> dict[str, Any]:
    from .bucket_scoring import aggregation_method, aggregate_responses

    execution_id = uuid.uuid4().hex
    call = lambda replica_id: _one_replica(client, row, replica_id, model)
    receipts, chosen, elapsed = await collect_replicas(
        call,
        initial_fanout=initial_fanout,
        hedge_seconds=hedge_seconds,
        maximum=maximum,
        required=required,
    )
    result = {
        "request_id": row["request_id"],
        "execution_id": execution_id,
        "receipts": receipts,
        "time_to_required_seconds": elapsed,
        "status": "failed",
        "error": None,
        "aggregation_method": aggregation_method(required),
    }
    if required == REQUIRED_REPLICAS:
        result["time_to_eight_seconds"] = elapsed
    if len(chosen) != required:
        counts = dict(Counter((receipt["status"] for receipt in receipts)))
        result["error"] = (
            f"fewer than {required} valid replicas after fanout {maximum}: {counts}"
        )
        return result
    validation = json.loads(row["validation_json"])
    response, aggregate = aggregate_responses(
        chosen, validation["candidate_aliases"], required=required
    )
    result.update(status="complete", response=response, aggregate=aggregate)
    return result


async def execute_fixed_request(
    targets: Sequence[tuple[Any, str]], row: Mapping[str, Any], *, required: int
) -> tuple[dict[str, Any], asyncio.Task[list[dict[str, Any]]] | None]:
    from .bucket_scoring import aggregation_method, aggregate_responses

    "Execute one fixed provider mix and release its first valid quorum."
    execution_id = uuid.uuid4().hex
    call = lambda replica_id: _one_replica(
        targets[replica_id][0], row, replica_id, targets[replica_id][1]
    )
    receipts, chosen, elapsed, drain = await collect_fixed_replicas(
        call, total=len(targets), required=required
    )
    result = {
        "request_id": row["request_id"],
        "execution_id": execution_id,
        "receipts": receipts,
        "time_to_required_seconds": elapsed,
        "status": "failed",
        "error": None,
        "aggregation_method": aggregation_method(required),
        "total_replica_count": len(targets),
    }
    if len(chosen) != required:
        counts = dict(Counter((receipt["status"] for receipt in receipts)))
        result["error"] = (
            f"fewer than {required} valid replicas after fixed mix: {counts}"
        )
        return (result, None)
    validation = json.loads(row["validation_json"])
    response, aggregate = aggregate_responses(
        chosen, validation["candidate_aliases"], required=required
    )
    result.update(status="complete", response=response, aggregate=aggregate)
    return (result, drain)


def execute_review(client, row: dict, *, model, validate) -> dict:
    receipts = json.loads(row["receipts_json"])
    attempts = int(row["attempts"])
    error = None
    for _ in range(3):
        attempts += 1
        started = time.time()
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": row["prompt"]},
                ],
                reasoning_effort="medium",
                max_completion_tokens=128000,
                response_format={"type": "json_object"},
                extra_body={
                    "provider": {
                        "order": ["openai"],
                        "allow_fallbacks": False,
                        "require_parameters": True,
                    }
                },
            )
            if completion.model != model:
                raise ValueError(f"served model changed: {completion.model}")
            parsed = validate(completion.choices[0].message.content or "", row)
            receipts.append(
                {
                    "attempt": attempts,
                    "elapsed_seconds": time.time() - started,
                    "generation_id": completion.id,
                    "served_model": completion.model,
                    "provider": getattr(completion, "provider_name", None),
                    "usage": completion.usage.model_dump() if completion.usage else {},
                }
            )
            return {
                "status": "complete",
                "attempts": attempts,
                "response_json": json.dumps(parsed),
                "reasoning": getattr(
                    completion.choices[0].message, "reasoning_content", None
                ),
                "receipts_json": json.dumps(receipts),
                "error": None,
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            receipts.append(
                {
                    "attempt": attempts,
                    "elapsed_seconds": time.time() - started,
                    "error": error[:500],
                }
            )
    return {
        "status": "failed",
        "attempts": attempts,
        "response_json": None,
        "reasoning": None,
        "receipts_json": json.dumps(receipts),
        "error": error,
    }
