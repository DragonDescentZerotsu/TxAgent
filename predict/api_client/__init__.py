"""Provider-neutral model inference."""

from .client import OpenAICompatibleClient
from .pool import (
    DEFAULT_PROVIDER_POOL_CONFIG,
    OpenAIProviderPool,
    ProviderPoolConfig,
    ProviderSelection,
    ProviderSpec,
    build_provider_pool,
    load_provider_pool_config,
    preflight_provider_models,
    primary_capacity,
    select_healthy_providers,
    validate_parallelism,
)

__all__ = [
    "OpenAICompatibleClient",
    "DEFAULT_PROVIDER_POOL_CONFIG",
    "OpenAIProviderPool",
    "ProviderPoolConfig",
    "ProviderSelection",
    "ProviderSpec",
    "build_provider_pool",
    "load_provider_pool_config",
    "preflight_provider_models",
    "primary_capacity",
    "select_healthy_providers",
    "validate_parallelism",
]
