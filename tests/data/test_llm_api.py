from pathlib import Path

import httpx
import pytest

from predict.api_client import transport as llm_api
from predict.api_client.transport import (
    async_openai_compatible_client,
    openai_compatible_client,
    provider_from_base_url,
    read_env_file,
    resolve_api_key,
)


@pytest.fixture(autouse=True)
def isolated_provider_credentials(monkeypatch):
    for names in llm_api.PROVIDER_CREDENTIALS.values():
        for name in names:
            monkeypatch.delenv(name, raising=False)


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


def test_provider_is_inferred_from_known_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_BASE_URL", "https://gateway.example.org/v1")
    assert provider_from_base_url("https://openrouter.ai/api/v1") == "openrouter"
    assert provider_from_base_url("https://gateway.example.org/v1") == "gateway"
    assert provider_from_base_url("https://api.openai.com/v1") == "openai"
    assert provider_from_base_url("https://api.deepseek.com") == "deepseek"
    assert provider_from_base_url("http://127.0.0.1:8000/v1") == "local"
    assert provider_from_base_url("http://10.218.16.108:50001/v1") == "local"
    assert provider_from_base_url("http://dgx027:50001/v1") == "local"
    assert provider_from_base_url("https://y5in3e8saiu8uu-8000.proxy.runpod.net/v1") == "local"
    for misleading in (
        "https://notdgx.example/v1",
        "https://example.org/v1/dgx027",
        "https://api.openai.com.example/v1",
        "https://example.org/openrouter.ai/v1",
        "https://runpod.net.example/v1",
    ):
        with pytest.raises(ValueError, match="cannot infer API provider"):
            provider_from_base_url(misleading)


def test_gateway_uses_its_own_credential_alias(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LITE_LLM_KEY=sk-gateway-example-123456789\n")

    _, name = resolve_api_key("gateway", env_file=env_file)

    assert name == "LITE_LLM_KEY"


def test_deepseek_uses_its_own_credential(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=sk-deepseek-example-123456789\n")

    _, name = resolve_api_key("deepseek", env_file=env_file)

    assert name == "DEEPSEEK_API_KEY"


def test_explicit_provider_must_match_a_recognized_endpoint() -> None:
    with pytest.raises(ValueError, match="does not match endpoint provider"):
        openai_compatible_client(
            base_url="https://openrouter.ai/api/v1",
            provider="gateway",
            credential_env="LITE_LLM_KEY",
            env_file=None,
        )
    with pytest.raises(ValueError, match="does not match endpoint provider"):
        openai_compatible_client(
            base_url="http://dgx027:50001/v1",
            provider="openrouter",
            env_file=None,
        )
    with pytest.raises(ValueError, match="credential-free local provider"):
        openai_compatible_client(
            base_url="https://notdgx.example/v1",
            provider="local",
            env_file=None,
        )


def test_local_client_uses_no_credential_and_rejects_wrong_returned_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    http_options = {}

    def capture_http_client(**kwargs):
        http_options.update(kwargs)
        return object()

    monkeypatch.setattr(llm_api, "DefaultHttpxClient", capture_http_client)
    monkeypatch.setattr(llm_api, "OpenAI", lambda **kwargs: kwargs)
    client, credential = openai_compatible_client(
        base_url="http://dgx027:50001/v1",
        provider="local",
        env_file=tmp_path / "missing.env",
        max_connections=512,
        max_retries=0,
    )
    assert credential == "" and client["api_key"] == "EMPTY"
    assert client["max_retries"] == 0
    assert http_options["limits"].max_connections == 512
    hook = http_options["event_hooks"]["response"][0]
    request = httpx.Request(
        "POST",
        "http://dgx027:50001/v1/chat/completions",
        json={"model": "deepseek-ai/DeepSeek-V4-Flash-0731"},
    )
    wrong = httpx.Response(200, request=request, json={"model": "wrong/model"})
    with pytest.raises(ValueError, match="local endpoint returned model"):
        hook(wrong)


def test_runpod_proxy_uses_ipv4_and_local_model_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    options = {}
    monkeypatch.setattr(llm_api, "DefaultHttpxClient", lambda **kwargs: options.update(kwargs) or object())
    monkeypatch.setattr(llm_api, "OpenAI", lambda **kwargs: kwargs)
    client, credential = openai_compatible_client(
        base_url="https://y5in3e8saiu8uu-8000.proxy.runpod.net/v1",
        env_file=None,
    )
    assert client["api_key"] == "EMPTY" and credential == ""
    assert "transport" in options
    assert options["event_hooks"]["response"] == [llm_api._validate_local_model_response]


def test_async_local_client_uses_shared_credential_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    http_options = {}

    def capture_http_client(**kwargs):
        http_options.update(kwargs)
        return object()

    monkeypatch.setattr(llm_api, "DefaultAsyncHttpxClient", capture_http_client)
    monkeypatch.setattr(llm_api, "AsyncOpenAI", lambda **kwargs: kwargs)

    client, credential = async_openai_compatible_client(
        base_url="http://dgx008:50001/v1", provider="local", env_file=None,
        max_connections=1152, max_retries=0,
    )

    assert credential == "" and client["api_key"] == "EMPTY"
    assert client["max_retries"] == 0
    assert http_options["limits"].max_connections == 1152
    assert "response" in http_options["event_hooks"]


def test_environment_credentials_override_file_fallback(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPEN_ROUTER_KEY=sk-or-file-example-123456789\n")
    monkeypatch.setenv("OPEN_ROUTER_KEY", "sk-or-process-example-123456789")
    assert resolve_api_key("openrouter", env_file=env_file) == ("sk-or-process-example-123456789", "OPEN_ROUTER_KEY")
    monkeypatch.delenv("OPEN_ROUTER_KEY")
    monkeypatch.setenv("OPENROUTER_KEY", "sk-or-alias-example-123456789")
    assert resolve_api_key("openrouter", env_file=env_file) == ("sk-or-alias-example-123456789", "OPENROUTER_KEY")
