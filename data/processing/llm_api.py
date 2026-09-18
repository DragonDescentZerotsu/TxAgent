"""One provider-aware API credential and client loader for data processing."""

from __future__ import annotations

import json
import ipaddress
import os
import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient, DefaultHttpxClient, OpenAI

Provider = Literal["openai", "openrouter", "deepseek", "parcc", "local"]
DEFAULT_ENV_FILE = (
    Path(__file__).resolve().parents[3] / "therapeutic-tuning/distillation/.env"
)
PROVIDER_CREDENTIALS = {
    "openai": ("OPENAI_API_KEY",),
    "openrouter": (
        "OPEN_ROUTER_KEY",
        "OPENROUTER_KEY",
        "OPENROUTER_API_KEY",
    ),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "parcc": ("LITE_LLM_KEY",),
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
    host = (urlsplit(base_url).hostname or "").casefold().rstrip(".")
    if host == "openrouter.ai" or host.endswith(".openrouter.ai"):
        return "openrouter"
    if host == "litellm.parcc.upenn.edu":
        return "parcc"
    if host == "api.openai.com":
        return "openai"
    if host == "api.deepseek.com":
        return "deepseek"
    try:
        private_ip = ipaddress.ip_address(host).is_private
    except ValueError:
        private_ip = False
    if private_ip or host == "localhost" or re.fullmatch(r"dgx\d+", host):
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


def _validate_local_model_response(response: httpx.Response) -> None:
    """Reject a successful local chat response from any unrequested model."""
    if not response.is_success or not response.request.url.path.endswith(
        "/chat/completions"
    ):
        return
    request = json.loads(response.request.content)
    if isinstance(request, dict) and request.get("stream") is True:
        return
    response.read()
    payload = response.json()
    requested = request.get("model") if isinstance(request, dict) else None
    returned = payload.get("model") if isinstance(payload, dict) else None
    if not requested or returned != requested:
        raise ValueError(
            f"local endpoint returned model {returned!r} for request {requested!r}"
        )


async def _validate_local_model_response_async(response: httpx.Response) -> None:
    """Async equivalent of the local non-streaming model guard."""
    if not response.is_success or not response.request.url.path.endswith(
        "/chat/completions"
    ):
        return
    request = json.loads(response.request.content)
    if isinstance(request, dict) and request.get("stream") is True:
        return
    await response.aread()
    payload = response.json()
    requested = request.get("model") if isinstance(request, dict) else None
    returned = payload.get("model") if isinstance(payload, dict) else None
    if not requested or returned != requested:
        raise ValueError(
            f"local endpoint returned model {returned!r} for request {requested!r}"
        )


def openai_compatible_client(
    *,
    base_url: str,
    provider: Provider | None = None,
    env_file: str | Path | None = DEFAULT_ENV_FILE,
    credential_env: str | None = None,
    max_connections: int | None = None,
    timeout_s: float | None = None,
    max_retries: int | None = None,
) -> tuple[OpenAI, str]:
    """Build one OpenAI-compatible client and report its credential variable."""
    if provider is not None:
        try:
            inferred_provider = provider_from_base_url(base_url)
        except ValueError:
            inferred_provider = None
        if provider == "local" and inferred_provider is None:
            raise ValueError(
                f"credential-free local provider requires a loopback, private IP, "
                f"or dgxNNN host: "
                f"{base_url!r}"
            )
        if inferred_provider is not None and inferred_provider != provider:
            raise ValueError(
                f"provider {provider!r} does not match endpoint provider "
                f"{inferred_provider!r} for {base_url!r}"
            )
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
    else:
        http_kwargs["event_hooks"] = {"response": [_validate_local_model_response]}
    if timeout_s is not None:
        http_kwargs["timeout"] = timeout_s
    kwargs: dict[str, object] = {
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
    }
    if max_retries is not None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        kwargs["max_retries"] = max_retries
    if http_kwargs:
        kwargs["http_client"] = DefaultHttpxClient(**http_kwargs)
    return OpenAI(**kwargs), selected_name


def async_openai_compatible_client(
    *,
    base_url: str,
    provider: Provider | None = None,
    env_file: str | Path | None = DEFAULT_ENV_FILE,
    credential_env: str | None = None,
    max_connections: int | None = None,
    timeout_s: float | None = None,
    max_retries: int | None = None,
) -> tuple[AsyncOpenAI, str]:
    """Build an async OpenAI-compatible client with the shared credential policy."""
    selected_provider = provider or provider_from_base_url(base_url)
    if provider is not None and provider_from_base_url(base_url) != provider:
        raise ValueError(f"provider {provider!r} does not match endpoint {base_url!r}")
    api_key, selected_name = resolve_api_key(
        selected_provider, env_file=env_file, credential_env=credential_env
    )
    limits = httpx.Limits(
        max_connections=max_connections or 100,
        max_keepalive_connections=max_connections or 20,
    )
    http_kwargs: dict[str, object] = {"limits": limits}
    if selected_provider != "local":
        http_kwargs["transport"] = httpx.AsyncHTTPTransport(
            local_address="0.0.0.0", limits=limits
        )
    else:
        http_kwargs["event_hooks"] = {
            "response": [_validate_local_model_response_async]
        }
    if timeout_s is not None:
        http_kwargs["timeout"] = timeout_s
    kwargs: dict[str, object] = {
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
        "http_client": DefaultAsyncHttpxClient(**http_kwargs),
    }
    if max_retries is not None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        kwargs["max_retries"] = max_retries
    return AsyncOpenAI(**kwargs), selected_name


__all__ = [
    "DEFAULT_ENV_FILE",
    "PROVIDER_CREDENTIALS",
    "Provider",
    "async_openai_compatible_client",
    "openai_compatible_client",
    "provider_from_base_url",
    "read_env_file",
    "resolve_api_key",
]
