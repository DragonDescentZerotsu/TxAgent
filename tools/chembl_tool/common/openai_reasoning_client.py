"""Compatibility imports for the canonical prediction engine."""

from predict.api_client import client as _impl
from predict.api_client.client import OpenAICompatibleClient
from predict.tools.client import ToolServiceClient


def __getattr__(name: str):
    return getattr(_impl, name)


__all__ = ["OpenAICompatibleClient", "ToolServiceClient"]
