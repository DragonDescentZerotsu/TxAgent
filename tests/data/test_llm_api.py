from pathlib import Path

import pytest

from data.processing.llm_api import (
    provider_from_base_url,
    read_env_file,
    resolve_api_key,
)


def test_dotenv_provider_aliases_are_loaded_without_mutating_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=sk-openai-example-123456789\n"
        "OPEN_ROUTER_KEY='sk-or-v1-example-123456789'\n"
    )
    monkeypatch.delenv("OPEN_ROUTER_KEY", raising=False)

    assert read_env_file(env_file)["OPEN_ROUTER_KEY"].startswith("sk-or-")
    _, name = resolve_api_key("openrouter", env_file=env_file)
    assert name == "OPEN_ROUTER_KEY"


def test_openrouter_refuses_an_openai_key(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-openai-example-123456789\n")

    with pytest.raises(RuntimeError, match="missing openrouter credential"):
        resolve_api_key("openrouter", env_file=env_file)
    with pytest.raises(ValueError, match="not an OpenRouter credential"):
        resolve_api_key(
            "openrouter", env_file=env_file, credential_env="OPENAI_API_KEY"
        )


def test_provider_is_inferred_from_known_endpoint() -> None:
    assert provider_from_base_url("https://openrouter.ai/api/v1") == "openrouter"
    assert provider_from_base_url("https://api.openai.com/v1") == "openai"
    assert provider_from_base_url("http://127.0.0.1:8000/v1") == "local"
