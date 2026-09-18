from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import time
from typing import Iterable

from tools.service.cache import ToolResultCache
from tools.service.config import ServiceSettings
from tools.service.errors import ToolError
from tools.service.schemas import (
    ToolErrorPayload,
    ToolInfo,
    ToolRequest,
    ToolResponse,
    ToolResponseMetadata,
    utc_now,
)
from tools.service.tools.base import BaseTool


TOOL_CACHE_NAMESPACE = "tool-service-2026-09-07-molgpka-serialized-v2"


class ToolRegistry:
    def __init__(self, tools: Iterable[BaseTool] = ()):
        self._tools: dict[str, BaseTool] = {}
        self._cache: ToolResultCache | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._cache_namespace = TOOL_CACHE_NAMESPACE
        for tool in tools:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool registered: {tool.name}")
        self._tools[tool.name] = tool

    def initialize_all(self, settings: ServiceSettings) -> None:
        self._cache_namespace = (
            f"{TOOL_CACHE_NAMESPACE}:molgpka={settings.enable_molgpka}:"
            f"logd_ph={settings.logd_ph}"
        )
        self._cache = ToolResultCache(
            settings.cache_path,
            memory_entries=settings.cache_memory_entries,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=settings.batch_workers,
            thread_name_prefix="tool-worker",
        )
        for tool in self._tools.values():
            try:
                tool.initialize(settings)
            except Exception as exc:
                tool.initialized = False
                tool.initialization_error = f"{type(exc).__name__}: {exc}"

    def list_tools(self) -> list[ToolInfo]:
        return [ToolInfo(**tool.info()) for tool in self._tools.values()]

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError("UNKNOWN_TOOL", f"Unknown tool: {name}", recoverable=True) from exc

    def invoke(self, request: ToolRequest) -> ToolResponse:
        started_at = utc_now()
        start = time.perf_counter()
        warnings: list[str] = []
        output = None
        errors: list[ToolErrorPayload] = []
        status = "ok"
        cache_hit = False

        try:
            tool = self.get(request.tool_name)
            if request.version != tool.version:
                raise ToolError(
                    "UNSUPPORTED_TOOL_VERSION",
                    f"{tool.name} supports {tool.version}, got {request.version}",
                    recoverable=True,
                )
            cache_key = self._cache_key(request)
            if self._cache is not None and tool.initialization_error is None:
                cached, cache_hit = self._cache.get_or_compute(
                    cache_key,
                    lambda: self._invoke_uncached(request, tool),
                )
            else:
                cached = self._invoke_uncached(request, tool)
            output = deepcopy(cached["output"])
            warnings.extend(str(item) for item in cached.get("warnings") or [])
            if tool.initialization_error:
                warnings.append(f"Tool initialization warning: {tool.initialization_error}")
        except ToolError as exc:
            status = "error"
            errors.append(ToolErrorPayload(code=exc.code, message=exc.message, recoverable=exc.recoverable))
        except Exception as exc:
            status = "error"
            errors.append(
                ToolErrorPayload(
                    code="TOOL_RUNTIME_ERROR",
                    message=f"{type(exc).__name__}: {exc}",
                    recoverable=False,
                )
            )

        finished_at = utc_now()
        return ToolResponse(
            request_id=request.request_id,
            tool_name=request.tool_name,
            version=request.version,
            status=status,
            output=output,
            warnings=warnings,
            errors=errors,
            metadata=ToolResponseMetadata(
                started_at=started_at,
                finished_at=finished_at,
                latency_ms=int((time.perf_counter() - start) * 1000),
                model_or_index_version=request.version,
                cache_hit=cache_hit,
            ),
        )

    def invoke_many(self, requests: list[ToolRequest]) -> list[ToolResponse]:
        if self._executor is None:
            return [self.invoke(request) for request in requests]
        return list(self._executor.map(self.invoke, requests))

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
        if self._cache is not None:
            self._cache.close()
            self._cache = None
        for tool in self._tools.values():
            tool.close()

    @staticmethod
    def _invoke_uncached(request: ToolRequest, tool: BaseTool) -> dict[str, object]:
        output = tool.invoke(request.input, return_debug=request.options.return_debug)
        tool_warnings = output.pop("_warnings", None)
        return {
            "output": output,
            "warnings": tool_warnings if isinstance(tool_warnings, list) else [],
        }

    def _cache_key(self, request: ToolRequest) -> str:
        payload = {
            "namespace": self._cache_namespace,
            "tool_name": request.tool_name,
            "version": request.version,
            "input": request.input,
            "return_debug": request.options.return_debug,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
