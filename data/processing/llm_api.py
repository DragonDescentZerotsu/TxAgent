"""One provider-aware API credential and client loader for data processing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import httpx
from openai import DefaultHttpxClient, OpenAI


Provider = Literal["openai", "openrouter", "local"]
DEFAULT_ENV_FILE = (
    Path(__file__).resolve().parents[3]
    / "therapeutic-tuning/distillation/.env"
)
PROVIDER_CREDENTIALS = {
    "openai": ("OPENAI_API_KEY",),
    "openrouter": (
        "OPEN_ROUTER_KEY",
        "OPENROUTER_KEY",
        "OPENROUTER_API_KEY",
    ),
    "local": (),
}


def read_env_file(path: str | Path = DEFAULT_ENV_FILE) -> dict[str, str]:
    """Read simple dotenv assignments without mutating the process environment."""
    env_path = Path(path)
    if not env_path.is_file():
        raise FileNotFoundError(f"API environment file not found: {env_path}")
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        if name:
            values[name] = value
    return values


def provider_from_base_url(base_url: str) -> Provider:
    lowered = base_url.lower()
    if "openrouter.ai" in lowered:
        return "openrouter"
    if "api.openai.com" in lowered:
        return "openai"
    if any(host in lowered for host in ("localhost", "127.0.0.1", "dgx")):
        return "local"
    raise ValueError(
        f"cannot infer API provider from {base_url!r}; pass provider explicitly"
    )


def resolve_api_key(
    provider: Provider,
    *,
    env_file: str | Path | None = DEFAULT_ENV_FILE,
    credential_env: str | None = None,
) -> tuple[str, str]:
    """Return the selected secret and its exact variable name, or fail closed."""
    if provider == "local":
        return "EMPTY", ""
    names = (credential_env,) if credential_env else PROVIDER_CREDENTIALS[provider]
    file_values = read_env_file(env_file) if env_file is not None else {}
    for name in names:
        value = file_values.get(name) or os.environ.get(name)
        if not value:
            continue
        value = value.strip()
        if len(value) < 20:
            raise ValueError(f"{name} appears to be a placeholder")
        if provider == "openrouter" and not value.startswith("sk-or-"):
            raise ValueError(
                f"{name} is not an OpenRouter credential; expected an sk-or- key"
            )
        return value, name
    searched = ", ".join(names)
    location = str(env_file) if env_file is not None else "the process environment"
    raise RuntimeError(f"missing {provider} credential ({searched}) in {location}")


def openai_compatible_client(
    *,
    base_url: str,
    provider: Provider | None = None,
    env_file: str | Path | None = DEFAULT_ENV_FILE,
    credential_env: str | None = None,
    max_connections: int | None = None,
    timeout_s: float | None = None,
) -> tuple[OpenAI, str]:
    """Build one OpenAI-compatible client and report its credential variable."""
    selected_provider = provider or provider_from_base_url(base_url)
    api_key, selected_name = resolve_api_key(
        selected_provider,
        env_file=env_file,
        credential_env=credential_env,
    )
    http_kwargs: dict[str, object] = {}
    limits = None
    if max_connections:
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        )
        http_kwargs["limits"] = limits
    if selected_provider != "local":
        # Several compute nodes advertise IPv6 without a working IPv6 route.
        # Binding remote API traffic to IPv4 avoids minutes of SYN retries.
        http_kwargs["transport"] = httpx.HTTPTransport(
            local_address="0.0.0.0", limits=limits or httpx.Limits()
        )
    if timeout_s is not None:
        http_kwargs["timeout"] = timeout_s
    kwargs: dict[str, object] = {
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
    }
    if http_kwargs:
        kwargs["http_client"] = DefaultHttpxClient(**http_kwargs)
    return OpenAI(**kwargs), selected_name


__all__ = [
    "DEFAULT_ENV_FILE",
    "PROVIDER_CREDENTIALS",
    "Provider",
    "openai_compatible_client",
    "provider_from_base_url",
    "read_env_file",
    "resolve_api_key",
]
