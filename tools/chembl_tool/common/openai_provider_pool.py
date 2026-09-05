"""Dynamic routing across heterogeneous OpenAI-compatible providers.

The pool is intentionally transport-facing: it does not alter prompts or model
outputs.  Each provider keeps its own model alias, credential environment
variable, concurrency budget, health state, and trace provenance.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Protocol


POOL_CONFIG_VERSION = "openai_provider_pool.v1"


class JsonChatClient(Protocol):
    def chat_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ProviderSpec:
    """One provider without any credential value."""

    name: str
    base_url: str
    model: str
    api_key_env: str
    max_inflight: int
    initial_latency_s: float = 60.0
    timeout_s: int | None = None
    request_extra_body: Mapping[str, Any] | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "ProviderSpec":
        spec = cls(
            name=str(row.get("name") or "").strip(),
            base_url=str(row.get("base_url") or "").strip().rstrip("/"),
            model=str(row.get("model") or "").strip(),
            api_key_env=str(row.get("api_key_env") or "").strip(),
            max_inflight=int(row.get("max_inflight") or 0),
            initial_latency_s=float(row.get("initial_latency_s") or 60.0),
            timeout_s=(int(row["timeout_s"]) if row.get("timeout_s") is not None else None),
            request_extra_body=dict(row.get("request_extra_body") or {}),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.name:
            raise ValueError("provider name is required")
        if not self.base_url:
            raise ValueError(f"provider {self.name!r} requires base_url")
        if not self.model:
            raise ValueError(f"provider {self.name!r} requires model")
        if self.max_inflight < 1:
            raise ValueError(f"provider {self.name!r} max_inflight must be positive")
        if self.initial_latency_s <= 0:
            raise ValueError(f"provider {self.name!r} initial_latency_s must be positive")
        if self.timeout_s is not None and self.timeout_s < 1:
            raise ValueError(f"provider {self.name!r} timeout_s must be positive")

    def public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "max_inflight": self.max_inflight,
            "initial_latency_s": self.initial_latency_s,
            "timeout_s": self.timeout_s,
            "request_extra_body": dict(self.request_extra_body or {}),
        }


@dataclass(frozen=True)
class ProviderPoolConfig:
    providers: tuple[ProviderSpec, ...]
    failure_threshold: int = 3
    cooldown_seconds: float = 60.0
    max_failovers: int = 1
    latency_ewma_alpha: float = 0.2

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ProviderPoolConfig":
        version = str(payload.get("version") or "")
        if version != POOL_CONFIG_VERSION:
            raise ValueError(
                f"provider pool config version must be {POOL_CONFIG_VERSION!r}, got {version!r}"
            )
        providers = tuple(
            ProviderSpec.from_mapping(row) for row in payload.get("providers") or []
        )
        config = cls(
            providers=providers,
            failure_threshold=int(payload.get("failure_threshold") or 3),
            cooldown_seconds=float(payload.get("cooldown_seconds") or 60.0),
            max_failovers=int(payload.get("max_failovers") if payload.get("max_failovers") is not None else 1),
            latency_ewma_alpha=float(payload.get("latency_ewma_alpha") or 0.2),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.providers:
            raise ValueError("provider pool requires at least one provider")
        names = [provider.name for provider in self.providers]
        if len(names) != len(set(names)):
            raise ValueError("provider names must be unique")
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be positive")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        if self.max_failovers < 0:
            raise ValueError("max_failovers must be non-negative")
        if not 0 < self.latency_ewma_alpha <= 1:
            raise ValueError("latency_ewma_alpha must be in (0, 1]")

    def public_dict(self) -> dict[str, Any]:
        return {
            "version": POOL_CONFIG_VERSION,
            "routing": "work_conserving_least_normalized_load_with_latency_ewma",
            "providers": [provider.public_dict() for provider in self.providers],
            "total_provider_capacity": sum(provider.max_inflight for provider in self.providers),
            "failure_threshold": self.failure_threshold,
            "cooldown_seconds": self.cooldown_seconds,
            "max_failovers": self.max_failovers,
            "latency_ewma_alpha": self.latency_ewma_alpha,
        }


def load_provider_pool_config(path: str | Path) -> ProviderPoolConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("provider pool config must be a JSON object")
    return ProviderPoolConfig.from_mapping(payload)


def load_env_file(path: str | Path, *, override: bool = False) -> None:
    """Load simple KEY=VALUE entries without logging credential values."""

    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value


@dataclass
class _ProviderState:
    spec: ProviderSpec
    client: JsonChatClient
    inflight: int = 0
    completed: int = 0
    failures: int = 0
    cancelled: int = 0
    consecutive_failures: int = 0
    circuit_open_until: float = 0.0
    latency_ewma_s: float = 0.0

    def __post_init__(self) -> None:
        self.latency_ewma_s = self.spec.initial_latency_s


class ProviderPoolExhausted(RuntimeError):
    def __init__(self, message: str, *, attempts: list[dict[str, Any]]):
        super().__init__(message)
        self.attempts = attempts


class OpenAIProviderPool:
    """Thread-safe, bounded provider scheduler with circuit breaking."""

    def __init__(
        self,
        config: ProviderPoolConfig,
        *,
        client_factory: Callable[[ProviderSpec], JsonChatClient],
        clock: Callable[[], float] = time.monotonic,
    ):
        config.validate()
        self.config = config
        self._clock = clock
        self._condition = threading.Condition()
        self._states = [
            _ProviderState(spec=spec, client=client_factory(spec))
            for spec in config.providers
        ]

    def chat_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        attempts: list[dict[str, Any]] = []
        excluded: set[str] = set()
        max_attempts = min(len(self._states), self.config.max_failovers + 1)
        for _ in range(max_attempts):
            state = self._acquire(excluded)
            started = self._clock()
            try:
                response = state.client.chat_json(messages)
            except Exception as exc:  # noqa: BLE001 - provider boundary
                latency_s = max(0.0, self._clock() - started)
                circuit_failure = _is_transport_or_provider_failure(exc)
                self._release_failure(
                    state,
                    latency_s=latency_s,
                    circuit_failure=circuit_failure,
                )
                attempt = {
                    "provider": state.spec.name,
                    "base_url": state.spec.base_url,
                    "requested_model": state.spec.model,
                    "status": "error",
                    "latency_seconds": latency_s,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                    "circuit_failure": circuit_failure,
                }
                attempts.append(attempt)
                excluded.add(state.spec.name)
                continue

            latency_s = max(0.0, self._clock() - started)
            self._release_success(state, latency_s=latency_s)
            execution = {
                "provider": state.spec.name,
                "base_url": state.spec.base_url,
                "requested_model": state.spec.model,
                "served_model": str(response.get("model") or ""),
                "request_id": str(response.get("id") or ""),
                "status": "ok",
                "latency_seconds": latency_s,
            }
            attempts.append(execution)
            response["execution_provider"] = execution
            response["execution_provider_attempts"] = attempts
            return response

        raise ProviderPoolExhausted(
            "all attempted providers failed",
            attempts=attempts,
        )

    def snapshot(self) -> dict[str, Any]:
        now = self._clock()
        with self._condition:
            providers = [
                {
                    **state.spec.public_dict(),
                    "inflight": state.inflight,
                    "completed": state.completed,
                    "failures": state.failures,
                    "cancelled": state.cancelled,
                    "consecutive_failures": state.consecutive_failures,
                    "circuit_open": state.circuit_open_until > now,
                    "circuit_open_remaining_seconds": max(
                        0.0, state.circuit_open_until - now
                    ),
                    "latency_ewma_seconds": state.latency_ewma_s,
                }
                for state in self._states
            ]
        return {"version": POOL_CONFIG_VERSION, "providers": providers}

    async def async_chat_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Use the same provider slots and health state, with cancellable HTTP calls."""
        attempts, excluded = [], set()
        for _ in range(min(len(self._states), self.config.max_failovers + 1)):
            while (state := self._acquire(excluded, wait=False)) is None:
                await asyncio.sleep(0.05)
            started = self._clock()
            execution = {"provider": state.spec.name, "base_url": state.spec.base_url,
                         "requested_model": state.spec.model}
            try:
                response = await state.client.async_chat_json(messages)
            except asyncio.CancelledError:
                with self._condition:
                    state.inflight -= 1
                    state.cancelled += 1
                    self._condition.notify_all()
                raise
            except Exception as exc:
                latency = max(0.0, self._clock() - started)
                circuit_failure = _is_transport_or_provider_failure(exc)
                self._release_failure(state, latency_s=latency, circuit_failure=circuit_failure)
                attempts.append({**execution, "status": "error", "latency_seconds": latency,
                                 "error_type": type(exc).__name__, "error": str(exc)[:500],
                                 "circuit_failure": circuit_failure})
                excluded.add(state.spec.name)
                continue
            latency = max(0.0, self._clock() - started)
            self._release_success(state, latency_s=latency)
            execution.update(status="ok", latency_seconds=latency,
                             served_model=str(response.get("model") or ""),
                             request_id=str(response.get("id") or ""))
            attempts.append(execution)
            response.update(execution_provider=execution, execution_provider_attempts=attempts)
            return response
        raise ProviderPoolExhausted("all attempted providers failed", attempts=attempts)

    async def aclose(self) -> None:
        for state in self._states:
            await state.client.aclose()

    def _acquire(self, excluded: set[str], *, wait: bool = True) -> _ProviderState | None:
        with self._condition:
            while True:
                now = self._clock()
                candidates = [
                    state
                    for state in self._states
                    if state.spec.name not in excluded
                    and state.circuit_open_until <= now
                    and state.inflight < state.spec.max_inflight
                ]
                if candidates:
                    state = min(candidates, key=_routing_score)
                    state.inflight += 1
                    return state
                remaining = [
                    state for state in self._states if state.spec.name not in excluded
                ]
                if not remaining:
                    raise ProviderPoolExhausted(
                        "no untried provider remains",
                        attempts=[],
                    )
                if not wait:
                    return None
                reopen_delays = [
                    state.circuit_open_until - now
                    for state in remaining
                    if state.circuit_open_until > now
                ]
                timeout = min(reopen_delays) if reopen_delays else 0.25
                self._condition.wait(timeout=max(0.01, min(timeout, 0.25)))

    def _release_success(self, state: _ProviderState, *, latency_s: float) -> None:
        with self._condition:
            state.inflight -= 1
            state.completed += 1
            state.consecutive_failures = 0
            alpha = self.config.latency_ewma_alpha
            state.latency_ewma_s = alpha * latency_s + (1 - alpha) * state.latency_ewma_s
            self._condition.notify_all()

    def _release_failure(
        self,
        state: _ProviderState,
        *,
        latency_s: float,
        circuit_failure: bool,
    ) -> None:
        with self._condition:
            state.inflight -= 1
            state.failures += 1
            if circuit_failure:
                state.consecutive_failures += 1
                if state.consecutive_failures >= self.config.failure_threshold:
                    state.circuit_open_until = self._clock() + self.config.cooldown_seconds
                    state.consecutive_failures = 0
            alpha = self.config.latency_ewma_alpha
            state.latency_ewma_s = alpha * latency_s + (1 - alpha) * state.latency_ewma_s
            self._condition.notify_all()


def _routing_score(state: _ProviderState) -> tuple[float, int, str]:
    normalized_load = (state.inflight + 1) / state.spec.max_inflight
    return normalized_load * state.latency_ewma_s, state.completed, state.spec.name


def _is_transport_or_provider_failure(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code == 429 or status_code >= 500
    module = type(exc).__module__.lower()
    name = type(exc).__name__.lower()
    tokens = ("timeout", "connection", "transport", "ratelimit", "apierror")
    return any(token in name or token in module for token in tokens)
