"""Dynamic routing across heterogeneous OpenAI-compatible providers.

The pool is intentionally transport-facing: it does not alter prompts or model
outputs.  Each provider keeps its own model alias, credential environment
variable, concurrency budget, health state, and trace provenance.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Protocol
from urllib.request import Request, urlopen


POOL_CONFIG_VERSION = "openai_provider_pool.v1"
DEFAULT_PROVIDER_POOL_CONFIG = (
    Path(__file__).resolve().parent / "providers/current_endpoints.json"
)


class JsonChatClient(Protocol):
    def chat_json(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenized_completion: bool = False,
        reasoning_grammar: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]: ...


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
    priority: int = 0

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
            priority=int(row.get("priority") or 0),
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
        if self.priority < 0:
            raise ValueError(f"provider {self.name!r} priority must be non-negative")

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
            "priority": self.priority,
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
        capacity_by_priority: dict[int, int] = {}
        for provider in self.providers:
            capacity_by_priority[provider.priority] = (
                capacity_by_priority.get(provider.priority, 0)
                + provider.max_inflight
            )
        return {
            "version": POOL_CONFIG_VERSION,
            "routing": "priority_then_least_normalized_load_with_latency_ewma",
            "providers": [provider.public_dict() for provider in self.providers],
            "total_provider_capacity": sum(provider.max_inflight for provider in self.providers),
            "maximum_priority_tier_capacity": max(capacity_by_priority.values()),
            "failure_threshold": self.failure_threshold,
            "cooldown_seconds": self.cooldown_seconds,
            "max_failovers": self.max_failovers,
            "latency_ewma_alpha": self.latency_ewma_alpha,
        }


@dataclass(frozen=True)
class ProviderSelection:
    """Healthy exact-model providers and the effective global request budget."""

    config: ProviderPoolConfig
    requested_parallelism: int
    effective_parallelism: int
    checks: tuple[dict[str, Any], ...]

    def public_dict(self) -> dict[str, Any]:
        return {
            "requested_parallelism": self.requested_parallelism,
            "effective_parallelism": self.effective_parallelism,
            "checks": list(self.checks),
            "selected_pool": self.config.public_dict(),
        }


def load_provider_pool_config(path: str | Path) -> ProviderPoolConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("provider pool config must be a JSON object")
    return ProviderPoolConfig.from_mapping(payload)


def primary_capacity(config: ProviderPoolConfig) -> int:
    """Return capacity available before using a fallback priority tier."""
    priority = min(provider.priority for provider in config.providers)
    return sum(
        provider.max_inflight
        for provider in config.providers
        if provider.priority == priority
    )


def validate_parallelism(config: ProviderPoolConfig, parallelism: int) -> None:
    capacity = primary_capacity(config)
    if not 1 <= parallelism <= capacity:
        raise ValueError(
            f"parallelism must be between 1 and primary provider capacity {capacity}"
        )


def preflight_provider_models(
    config: ProviderPoolConfig,
    *,
    timeout_s: float = 10,
) -> list[dict[str, Any]]:
    """Verify every endpoint advertises its exact configured model."""
    receipts = []
    for provider in config.providers:
        with urlopen(provider.base_url.rstrip("/") + "/models", timeout=timeout_s) as response:
            payload = json.load(response)
        models = [str(row.get("id") or "") for row in payload.get("data") or []]
        if provider.model not in models:
            raise ValueError(
                f"provider {provider.name!r} does not advertise model {provider.model!r}"
            )
        receipts.append(
            {
                "provider": provider.name,
                "base_url": provider.base_url,
                "requested_model": provider.model,
                "models": models,
            }
        )
    return receipts


def select_healthy_providers(
    config: ProviderPoolConfig,
    requested_parallelism: int,
    *,
    timeout_s: float = 10,
) -> ProviderSelection:
    """Probe candidates concurrently and retain every healthy exact-model endpoint."""
    if requested_parallelism < 1:
        raise ValueError("requested parallelism must be positive")
    models = {provider.model for provider in config.providers}
    if len(models) != 1:
        raise ValueError("provider candidates must use one exact model")

    def probe(provider: ProviderSpec) -> dict[str, Any]:
        receipt: dict[str, Any] = {
            "provider": provider.name,
            "base_url": provider.base_url,
            "requested_model": provider.model,
            "max_inflight": provider.max_inflight,
        }
        try:
            with urlopen(
                provider.base_url.rstrip("/") + "/models", timeout=timeout_s
            ) as response:
                payload = json.load(response)
            advertised = [
                str(row.get("id") or "") for row in payload.get("data") or []
            ]
            receipt["models"] = advertised
            receipt["status"] = (
                "healthy" if provider.model in advertised else "model_mismatch"
            )
        except Exception as exc:  # noqa: BLE001 - endpoint boundary
            receipt.update(
                status="unavailable",
                error_type=type(exc).__name__,
                error=str(exc)[:500],
            )
        return receipt

    by_name: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(config.providers)) as executor:
        futures = {
            executor.submit(probe, provider): provider.name
            for provider in config.providers
        }
        for future in as_completed(futures):
            by_name[futures[future]] = future.result()
    checks = tuple(by_name[provider.name] for provider in config.providers)
    healthy = tuple(
        replace(provider, priority=0)
        for provider in config.providers
        if by_name[provider.name]["status"] == "healthy"
    )
    if not healthy:
        raise ValueError(f"no healthy exact-model provider is available: {checks}")
    active = ProviderPoolConfig(
        providers=healthy,
        failure_threshold=config.failure_threshold,
        cooldown_seconds=config.cooldown_seconds,
        max_failovers=max(config.max_failovers, len(healthy) - 1),
        latency_ewma_alpha=config.latency_ewma_alpha,
    )
    return ProviderSelection(
        config=active,
        requested_parallelism=requested_parallelism,
        effective_parallelism=min(requested_parallelism, primary_capacity(active)),
        checks=checks,
    )


def preflight_sglang_tokenized_completion(
    config: ProviderPoolConfig,
    *,
    timeout_s: float = 10,
) -> list[dict[str, Any]]:
    """Fail closed unless SGLang can grammar-constrain private reasoning."""
    receipts = []
    for provider in config.providers:
        api_root = provider.base_url.rstrip("/")
        server_root = api_root[:-3] if api_root.endswith("/v1") else api_root
        with urlopen(server_root + "/get_server_info", timeout=timeout_s) as response:
            info = json.load(response)
        expected = {
            "status": "ready",
            "served_model_name": provider.model,
            "grammar_backend": "xgrammar",
            "reasoning_parser": "deepseek-v4",
            "completion_template": None,
        }
        mismatches = {
            key: {"expected": value, "actual": info.get(key)}
            for key, value in expected.items()
            if key not in info or info.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"provider {provider.name!r} cannot run tokenized reasoning: {mismatches}"
            )
        body: dict[str, Any] = {
            "model": provider.model,
            "messages": [{"role": "user", "content": "transport preflight"}],
        }
        template_kwargs = (provider.request_extra_body or {}).get(
            "chat_template_kwargs"
        )
        if template_kwargs:
            body["chat_template_kwargs"] = template_kwargs
        request = Request(
            api_root + "/tokenize",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=timeout_s) as response:
            tokens = json.load(response).get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f"provider {provider.name!r} tokenize preflight failed")
        request = Request(
            api_root + "/detokenize",
            data=json.dumps(
                {
                    "model": provider.model,
                    "tokens": tokens[-8:],
                    "skip_special_tokens": False,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=timeout_s) as response:
            prompt_tail = str(json.load(response).get("text") or "")
        if not prompt_tail.endswith("<think>"):
            raise ValueError(
                f"provider {provider.name!r} chat template does not enter private reasoning"
            )
        receipts.append(
            {
                "provider": provider.name,
                "requested_model": provider.model,
                "status": info["status"],
                "grammar_backend": info["grammar_backend"],
                "reasoning_parser": info["reasoning_parser"],
                "completion_template": info["completion_template"],
                "tokenized_prompt_suffix": "<think>",
            }
        )
    return receipts


def build_provider_pool(
    config: ProviderPoolConfig,
    *,
    env_file: str | Path | None,
    timeout_s: int,
    max_tokens: int,
    temperature: float | None,
    tool_service_url: str,
    enable_group_tools: bool,
    max_tool_rounds: int,
    reasoning_effort: str,
    enable_thinking: bool,
    transport_max_retries: int,
    response_format: Mapping[str, Any] | None = None,
) -> "OpenAIProviderPool":
    """Build every configured provider through the shared credential loader."""
    from data.processing.llm_api import openai_compatible_client
    from predict.api_client.client import OpenAICompatibleClient

    def factory(spec: ProviderSpec) -> OpenAICompatibleClient:
        transport, _ = openai_compatible_client(
            base_url=spec.base_url,
            env_file=env_file,
            credential_env=spec.api_key_env or None,
            max_connections=spec.max_inflight,
            timeout_s=spec.timeout_s or timeout_s,
            max_retries=transport_max_retries,
        )
        return OpenAICompatibleClient(
            api_key="loaded-by-shared-client",
            base_url=spec.base_url,
            model=spec.model,
            timeout_s=spec.timeout_s or timeout_s,
            max_tokens=max_tokens,
            temperature=temperature,
            tool_service_url=tool_service_url,
            enable_group_tools=enable_group_tools,
            max_tool_rounds=max_tool_rounds,
            reasoning_effort=reasoning_effort,
            enable_thinking=enable_thinking,
            transport_max_retries=transport_max_retries,
            request_extra_body=spec.request_extra_body,
            response_format=response_format,
            openai_client=transport,
        )

    return OpenAIProviderPool(config, client_factory=factory)


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

    enable_group_tools = False

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

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenized_completion: bool = False,
        reasoning_grammar: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        attempts: list[dict[str, Any]] = []
        excluded: set[str] = set()
        max_attempts = min(len(self._states), self.config.max_failovers + 1)
        for _ in range(max_attempts):
            state = self._acquire(excluded)
            started = self._clock()
            try:
                token_override = {"max_tokens": max_tokens} if max_tokens else {}
                if tokenized_completion:
                    response = state.client.chat_json(
                        messages,
                        tokenized_completion=True,
                        reasoning_grammar=reasoning_grammar,
                        **token_override,
                    )
                else:
                    response = state.client.chat_json(messages, **token_override)
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
                "upstream_provider": response.get("provider"),
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

    def _acquire(self, excluded: set[str]) -> _ProviderState:
        with self._condition:
            while True:
                now = self._clock()
                available = [
                    state
                    for state in self._states
                    if state.spec.name not in excluded
                    and state.circuit_open_until <= now
                ]
                priority = min(
                    (state.spec.priority for state in available), default=None
                )
                candidates = [
                    state
                    for state in available
                    if state.spec.priority == priority
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
