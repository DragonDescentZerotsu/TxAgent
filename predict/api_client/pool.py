"""Dynamic routing across heterogeneous OpenAI-compatible providers.

The pool is intentionally transport-facing: it does not alter prompts or model
outputs.  Each provider keeps its own model alias, credential environment
variable, concurrency budget, health state, and trace provenance.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable, Mapping, Protocol
from urllib.request import Request, urlopen
import uuid


POOL_CONFIG_VERSION = "openai_provider_pool.v1"
SPEND_BUDGET_VERSION = "openrouter_spend_budget.v1"
_NANODOLLARS = 1_000_000_000
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
        effort = (self.request_extra_body or {}).get("reasoning_effort_override")
        if effort is not None and effort not in {"low", "medium", "high"}:
            raise ValueError(f"provider {self.name!r} has invalid reasoning override")

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
class SpendBudgetSpec:
    """One shared, explicit spending epoch for an OpenRouter credential."""

    ledger_path: str
    epoch: str
    limit_usd: float
    credential_env: str
    stop_on_exhaustion: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SpendBudgetSpec":
        version = str(payload.get("version") or "")
        if version != SPEND_BUDGET_VERSION:
            raise ValueError(
                f"spend budget version must be {SPEND_BUDGET_VERSION!r}, got {version!r}"
            )
        spec = cls(
            ledger_path=str(payload.get("ledger_path") or "").strip(),
            epoch=str(payload.get("epoch") or "").strip(),
            limit_usd=float(payload.get("limit_usd") or 0),
            credential_env=str(payload.get("credential_env") or "").strip(),
            stop_on_exhaustion=bool(payload.get("stop_on_exhaustion", False)),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.ledger_path:
            raise ValueError("spend budget requires ledger_path")
        if not self.epoch:
            raise ValueError("spend budget requires epoch")
        if self.limit_usd <= 0:
            raise ValueError("spend budget limit_usd must be positive")
        if not self.credential_env:
            raise ValueError("spend budget requires credential_env")

    def public_dict(self) -> dict[str, Any]:
        return {
            "version": SPEND_BUDGET_VERSION,
            "ledger_path": self.ledger_path,
            "epoch": self.epoch,
            "limit_usd": self.limit_usd,
            "credential_env": self.credential_env,
            "stop_on_exhaustion": self.stop_on_exhaustion,
        }


@dataclass(frozen=True)
class ProviderPoolConfig:
    providers: tuple[ProviderSpec, ...]
    failure_threshold: int = 3
    cooldown_seconds: float = 60.0
    max_failovers: int = 1
    latency_ewma_alpha: float = 0.2
    spend_budget: SpendBudgetSpec | None = None

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
            spend_budget=(
                SpendBudgetSpec.from_mapping(payload["spend_budget"])
                if payload.get("spend_budget") else None
            ),
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
        if self.spend_budget is not None:
            self.spend_budget.validate()
            for provider in self.providers:
                if provider.api_key_env != self.spend_budget.credential_env:
                    raise ValueError(
                        f"provider {provider.name!r} credential does not match spend budget"
                    )
                maximum = (provider.request_extra_body or {}).get(
                    "max_request_cost_usd"
                )
                if not isinstance(maximum, (int, float)) or maximum <= 0:
                    raise ValueError(
                        f"provider {provider.name!r} requires max_request_cost_usd"
                    )

    def public_dict(self) -> dict[str, Any]:
        capacity_by_priority: dict[int, int] = {}
        for provider in self.providers:
            capacity_by_priority[provider.priority] = (
                capacity_by_priority.get(provider.priority, 0)
                + provider.max_inflight
            )
        result = {
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
        if self.spend_budget is not None:
            result["spend_budget"] = self.spend_budget.public_dict()
        return result


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
        spend_budget=config.spend_budget,
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

    if config.spend_budget is not None and transport_max_retries:
        raise ValueError(
            "spend-budgeted provider pools require transport_max_retries=0"
        )

    def factory(spec: ProviderSpec) -> OpenAICompatibleClient:
        request_extra_body = dict(spec.request_extra_body or {})
        provider_reasoning_effort = request_extra_body.pop(
            "reasoning_effort_override", reasoning_effort
        )
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
            reasoning_effort=provider_reasoning_effort,
            enable_thinking=enable_thinking,
            transport_max_retries=transport_max_retries,
            request_extra_body=request_extra_body,
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

    def __str__(self) -> str:
        attempts = ", ".join(
            f"{item['provider']}:{item['error_type']}:{item.get('status_code') or 'unknown'}"
            for item in self.attempts
        )
        return f"{super().__str__()} [{attempts}]" if attempts else super().__str__()


def _nanodollars(value: float) -> int:
    return math.ceil(float(value) * _NANODOLLARS)


class _SpendLedger:
    """SQLite-backed reservations shared by concurrent launchers."""

    def __init__(self, spec: SpendBudgetSpec):
        self.spec = spec
        self.path = Path(spec.ledger_path)
        self.alert_path = self.path.with_suffix(self.path.suffix + ".alert.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS epochs (
                  epoch TEXT PRIMARY KEY, limit_nanodollars INTEGER NOT NULL,
                  credential_env TEXT NOT NULL, created_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reservations (
                  reservation_id TEXT PRIMARY KEY, epoch TEXT NOT NULL,
                  provider TEXT NOT NULL, status TEXT NOT NULL,
                  reserved_nanodollars INTEGER NOT NULL,
                  actual_nanodollars INTEGER, created_at REAL NOT NULL,
                  settled_at REAL
                )
                """
            )
            initial_limit = _nanodollars(spec.limit_usd)
            current = connection.execute(
                "SELECT limit_nanodollars,credential_env FROM epochs WHERE epoch=?",
                (spec.epoch,),
            ).fetchone()
            if current is None:
                connection.execute(
                    "INSERT INTO epochs VALUES (?,?,?,?)",
                    (spec.epoch, initial_limit, spec.credential_env, time.time()),
                )
            elif current[1] != spec.credential_env or int(current[0]) < initial_limit:
                raise ValueError("spend budget epoch metadata is incompatible")
            connection.commit()

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=60)
        connection.execute("PRAGMA busy_timeout=60000")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @staticmethod
    def _totals(connection: sqlite3.Connection, epoch: str) -> tuple[int, int]:
        row = connection.execute(
            """
            SELECT
              COALESCE(SUM(CASE WHEN status='settled' THEN actual_nanodollars
                                WHEN status='uncertain' THEN reserved_nanodollars
                                ELSE 0 END),0),
              COALESCE(SUM(CASE WHEN status='reserved' THEN reserved_nanodollars
                                ELSE 0 END),0)
            FROM reservations WHERE epoch=?
            """,
            (epoch,),
        ).fetchone()
        return int(row[0]), int(row[1])

    def reserve(self, provider: str, maximum_usd: float) -> str:
        reserved = _nanodollars(maximum_usd)
        reservation_id = uuid.uuid4().hex
        while True:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                limit = int(connection.execute(
                    "SELECT limit_nanodollars FROM epochs WHERE epoch=?",
                    (self.spec.epoch,),
                ).fetchone()[0])
                fixed, inflight = self._totals(connection, self.spec.epoch)
                if fixed + reserved > limit:
                    self._write_alert(
                        authorized=limit, committed=fixed + inflight,
                        required=reserved,
                    )
                    connection.rollback()
                    if self.spec.stop_on_exhaustion:
                        raise RuntimeError("OpenRouter spend budget exhausted")
                elif fixed + inflight + reserved <= limit:
                    connection.execute(
                        "INSERT INTO reservations VALUES (?,?,?,?,?,?,?,?)",
                        (
                            reservation_id, self.spec.epoch, provider, "reserved",
                            reserved, None, time.time(), None,
                        ),
                    )
                    connection.commit()
                    break
                else:
                    connection.rollback()
            time.sleep(0.05)
        return reservation_id

    def _write_alert(self, *, authorized: int, committed: int, required: int) -> None:
        payload = {
            "version": SPEND_BUDGET_VERSION,
            "status": "budget_refresh_required",
            "epoch": self.spec.epoch,
            "credential_env": self.spec.credential_env,
            "authorized_usd": authorized / _NANODOLLARS,
            "committed_usd": committed / _NANODOLLARS,
            "next_reservation_usd": required / _NANODOLLARS,
            "created_at_unix": time.time(),
        }
        temporary = self.alert_path.with_name(
            f".{self.alert_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, self.alert_path)

    def refresh(self) -> dict[str, Any]:
        increment = _nanodollars(self.spec.limit_usd)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE epochs SET limit_nanodollars=limit_nanodollars+?
                WHERE epoch=? AND credential_env=?
                """,
                (increment, self.spec.epoch, self.spec.credential_env),
            )
            if connection.total_changes != 1:
                connection.rollback()
                raise ValueError("spend budget epoch is unavailable")
            connection.commit()
        self.alert_path.unlink(missing_ok=True)
        return self.status()

    def settle(self, reservation_id: str, actual_usd: float | None) -> dict[str, Any]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT reserved_nanodollars FROM reservations
                WHERE reservation_id=? AND epoch=? AND status='reserved'
                """,
                (reservation_id, self.spec.epoch),
            ).fetchone()
            if row is None:
                raise ValueError("unknown or already settled spend reservation")
            reserved = int(row[0])
            actual = _nanodollars(actual_usd) if actual_usd is not None else None
            if actual is not None and actual > reserved:
                connection.rollback()
                raise RuntimeError("OpenRouter cost exceeded its reserved maximum")
            status = "settled" if actual is not None else "uncertain"
            connection.execute(
                """
                UPDATE reservations SET status=?,actual_nanodollars=?,settled_at=?
                WHERE reservation_id=?
                """,
                (status, actual, time.time(), reservation_id),
            )
            connection.commit()
        return self.status(reservation_id=reservation_id)

    def uncertain(self, reservation_id: str) -> dict[str, Any]:
        return self.settle(reservation_id, None)

    def status(self, *, reservation_id: str | None = None) -> dict[str, Any]:
        with self._connection() as connection:
            authorized = int(connection.execute(
                "SELECT limit_nanodollars FROM epochs WHERE epoch=?",
                (self.spec.epoch,),
            ).fetchone()[0])
            fixed, inflight = self._totals(connection, self.spec.epoch)
            counts = dict(connection.execute(
                """
                SELECT status,COUNT(*) FROM reservations
                WHERE epoch=? GROUP BY status
                """,
                (self.spec.epoch,),
            ).fetchall())
        committed = fixed + inflight
        result = {
            "version": SPEND_BUDGET_VERSION,
            "epoch": self.spec.epoch,
            "budget_increment_usd": self.spec.limit_usd,
            "authorized_usd": authorized / _NANODOLLARS,
            "committed_usd": committed / _NANODOLLARS,
            "remaining_usd": max(
                0.0, (authorized - committed) / _NANODOLLARS
            ),
            "reservation_counts": counts,
            "alert_path": str(self.alert_path),
            "alert_active": self.alert_path.is_file(),
        }
        if reservation_id is not None:
            result["reservation_id"] = reservation_id
        return result


def spend_budget_status(config: ProviderPoolConfig) -> dict[str, Any] | None:
    """Return the current shared budget without constructing API clients."""
    return _SpendLedger(config.spend_budget).status() if config.spend_budget else None


def refresh_spend_budget(config: ProviderPoolConfig) -> dict[str, Any]:
    """Add one explicitly authorized tranche and release paused requests."""
    if config.spend_budget is None:
        raise ValueError("provider pool has no spend budget")
    return _SpendLedger(config.spend_budget).refresh()


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
        self._spend_ledger = (
            _SpendLedger(config.spend_budget) if config.spend_budget else None
        )
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
            reservation_id = None
            if self._spend_ledger is not None:
                try:
                    reservation_id = self._spend_ledger.reserve(
                        state.spec.name,
                        float(
                            (state.spec.request_extra_body or {})[
                                "max_request_cost_usd"
                            ]
                        ),
                    )
                except Exception:
                    self._release_unstarted(state)
                    raise
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
                controls = dict(state.spec.request_extra_body or {})
                allowed_models = set(controls.get("allowed_served_models") or [])
                served_model = str(response.get("model") or "")
                if allowed_models and served_model not in allowed_models:
                    raise ValueError(
                        f"provider {state.spec.name!r} served unexpected model "
                        f"{served_model!r}"
                    )
                expected_provider = controls.get("expected_upstream_provider")
                if expected_provider and response.get("provider") != expected_provider:
                    raise ValueError(
                        f"provider {state.spec.name!r} returned upstream "
                        f"{response.get('provider')!r}"
                    )
            except Exception as exc:  # noqa: BLE001 - provider boundary
                latency_s = max(0.0, self._clock() - started)
                budget_error = None
                if reservation_id is not None:
                    try:
                        self._spend_ledger.uncertain(reservation_id)
                    except Exception as ledger_exc:  # fail closed after releasing capacity
                        budget_error = ledger_exc
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
                    "status_code": getattr(exc, "status_code", None),
                    "error": str(exc)[:500],
                    "circuit_failure": circuit_failure,
                }
                attempts.append(attempt)
                excluded.add(state.spec.name)
                if budget_error is not None:
                    raise budget_error
                continue

            latency_s = max(0.0, self._clock() - started)
            budget_receipt = None
            if reservation_id is not None:
                usage = response.get("usage") or {}
                raw_cost = usage.get("cost")
                actual_cost = (
                    float(raw_cost)
                    if isinstance(raw_cost, (int, float)) and raw_cost >= 0
                    else None
                )
                try:
                    budget_receipt = self._spend_ledger.settle(
                        reservation_id, actual_cost
                    )
                except Exception:
                    self._release_success(state, latency_s=latency_s)
                    raise
            self._release_success(state, latency_s=latency_s)
            execution = {
                "provider": state.spec.name,
                "upstream_provider": response.get("provider"),
                "provider_pool_snapshot_sha256": (
                    (state.spec.request_extra_body or {}).get(
                        "provider_pool_snapshot_sha256"
                    )
                ),
                "base_url": state.spec.base_url,
                "requested_model": state.spec.model,
                "served_model": str(response.get("model") or ""),
                "request_id": str(response.get("id") or ""),
                "status": "ok",
                "latency_seconds": latency_s,
            }
            if budget_receipt is not None:
                execution["spend_budget"] = budget_receipt
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
        result = {"version": POOL_CONFIG_VERSION, "providers": providers}
        if self._spend_ledger is not None:
            result["spend_budget"] = self._spend_ledger.status()
        return result

    def _release_unstarted(self, state: _ProviderState) -> None:
        with self._condition:
            state.inflight -= 1
            self._condition.notify_all()

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
