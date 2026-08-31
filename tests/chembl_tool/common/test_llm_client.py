from __future__ import annotations

from pathlib import Path

import pytest

from tools.chembl_tool.common.llm_client import (
    distillation_keys_path,
    load_distillation_secret,
    openai_client,
)


def test_secret_prefers_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    assert (
        load_distillation_secret(
            "OPENAI_API_KEY", keys_path=tmp_path / "missing.py"
        )
        == "environment-key"
    )


def test_secret_falls_back_to_distillation_keys(monkeypatch, tmp_path: Path) -> None:
    keys_path = tmp_path / "keys.py"
    keys_path.write_text('OPENAI_API_KEY = "distillation-key"\n', encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert (
        load_distillation_secret("OPENAI_API_KEY", keys_path=keys_path)
        == "distillation-key"
    )


def test_unauthenticated_client_never_loads_distillation_keys(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = openai_client(
        base_url="http://dgx019:8000/v1/",
        keys_path=tmp_path / "missing.py",
        unauthenticated=True,
        max_connections=1024,
        timeout_s=321,
    )
    assert str(client.base_url) == "http://dgx019:8000/v1/"
    assert client.api_key == "EMPTY"
    assert client.timeout.read == 321
    client.close()


def test_default_keys_path_uses_sibling_checkout(monkeypatch) -> None:
    monkeypatch.delenv("DISTILLATION_KEYS_PATH", raising=False)
    assert distillation_keys_path().as_posix().endswith(
        "/therapeutic-tuning/distillation/keys.py"
    )


def test_missing_secret_fails_closed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MISSING_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="distillation keys.py is absent"):
        load_distillation_secret(
            "MISSING_API_KEY", keys_path=tmp_path / "missing.py"
        )
