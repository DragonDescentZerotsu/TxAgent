"""Compatibility imports for the canonical prediction engine."""

from predict.llm_engine import client as _impl
from predict.llm_engine.client import OpenAICompatibleClient
from predict.tools.client import ToolServiceClient


def __getattr__(name: str):
    return getattr(_impl, name)


__all__ = ["OpenAICompatibleClient", "ToolServiceClient"]
