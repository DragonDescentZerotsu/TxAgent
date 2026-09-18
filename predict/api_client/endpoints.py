"""Provider profiles for the shared OpenAI-compatible transport."""

from predict.api_client.pool import ProviderSpec


def litellm_endpoint(
    *, name: str, base_url: str, model: str, api_key_env: str, max_inflight: int
) -> ProviderSpec:
    return ProviderSpec(
        name=name,
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        max_inflight=max_inflight,
    )


def openrouter_endpoint(
    *, name: str, model: str, api_key_env: str, max_inflight: int
) -> ProviderSpec:
    return ProviderSpec(
        name=name,
        base_url="https://openrouter.ai/api/v1",
        model=model,
        api_key_env=api_key_env,
        max_inflight=max_inflight,
        request_extra_body={"reasoning": {"enabled": True}},
    )


def vllm_endpoint(
    *, name: str, base_url: str, model: str, max_inflight: int, api_key_env: str = ""
) -> ProviderSpec:
    return ProviderSpec(
        name=name,
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        max_inflight=max_inflight,
    )


__all__ = ["litellm_endpoint", "openrouter_endpoint", "vllm_endpoint"]
