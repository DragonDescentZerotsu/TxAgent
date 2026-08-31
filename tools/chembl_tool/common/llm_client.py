"""Shared OpenAI-compatible client and distillation credential loading."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import httpx
from openai import DefaultHttpxClient, OpenAI


REPO_ROOT = Path(__file__).resolve().parents[3]
DISTILLATION_KEYS_ENV = "DISTILLATION_KEYS_PATH"


def distillation_keys_path() -> Path:
    configured = os.environ.get(DISTILLATION_KEYS_ENV)
    if configured:
        return Path(configured)
    return REPO_ROOT.parent / "therapeutic-tuning/distillation/keys.py"


def load_distillation_secret(name: str, *, keys_path: Path | None = None) -> str:
    value = os.environ.get(name)
    if value:
        return value
    path = keys_path or distillation_keys_path()
    if not path.is_file():
        raise RuntimeError(f"{name} is unset and distillation keys.py is absent: {path}")
    spec = importlib.util.spec_from_file_location("_txagent_distillation_keys", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load distillation keys.py: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = getattr(module, name, None)
    if not value:
        raise RuntimeError(f"{name} is absent from distillation keys.py")
    return str(value)


def openai_client(
    *,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    keys_path: Path | None = None,
    unauthenticated: bool = False,
    max_connections: int | None = None,
    timeout_s: float | None = None,
) -> OpenAI:
    kwargs: dict[str, object] = {
        "api_key": "EMPTY"
        if unauthenticated
        else load_distillation_secret(api_key_env, keys_path=keys_path),
    }
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    if max_connections or timeout_s is not None:
        http_kwargs: dict[str, object] = {}
        if max_connections:
            http_kwargs["limits"] = httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            )
        if timeout_s is not None:
            http_kwargs["timeout"] = timeout_s
        kwargs["http_client"] = DefaultHttpxClient(**http_kwargs)
    return OpenAI(**kwargs)
