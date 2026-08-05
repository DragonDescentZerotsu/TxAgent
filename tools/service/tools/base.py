from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tools.service.config import ServiceSettings


class BaseTool(ABC):
    name: str
    version: str = "v1"
    description: str = ""
    input_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] = {}

    def __init__(self) -> None:
        self.initialized = False
        self.initialization_error: str | None = None

    def initialize(self, settings: ServiceSettings) -> None:
        self.initialized = True
        self.initialization_error = None

    def close(self) -> None:
        pass

    @abstractmethod
    def invoke(self, payload: dict[str, Any], *, return_debug: bool = False) -> dict[str, Any]:
        raise NotImplementedError

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "initialized": self.initialized,
            "initialization_error": self.initialization_error,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }
