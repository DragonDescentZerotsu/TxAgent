from __future__ import annotations

from threading import Lock

from tools.service.config import ServiceSettings
from tools.service.registry import ToolRegistry
from tools.service.schemas import ToolRequest
from tools.service.tools.base import BaseTool


class EchoTool(BaseTool):
    name = "echo"
    version = "v1"
    description = "test echo tool"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self._lock = Lock()

    def invoke(self, payload: dict[str, object], *, return_debug: bool = False) -> dict[str, object]:
        with self._lock:
            self.calls += 1
        return {"payload": payload}


def test_registry_wraps_successful_tool_response():
    registry = ToolRegistry([EchoTool()])
    registry.initialize_all(ServiceSettings(enable_molgpka=False, prewarm_molgpka=False))

    response = registry.invoke(ToolRequest(tool_name="echo", input={"x": 1}))

    assert response.status == "ok"
    assert response.output == {"payload": {"x": 1}}
    assert response.errors == []
    registry.close()


def test_registry_reports_unknown_tool():
    registry = ToolRegistry([])

    response = registry.invoke(ToolRequest(tool_name="missing", input={}))

    assert response.status == "error"
    assert response.errors[0].code == "UNKNOWN_TOOL"


def test_registry_batch_single_flight_computes_identical_requests_once():
    tool = EchoTool()
    registry = ToolRegistry([tool])
    registry.initialize_all(
        ServiceSettings(
            enable_molgpka=False,
            prewarm_molgpka=False,
            batch_workers=8,
            cache_path=None,
        )
    )
    responses = registry.invoke_many(
        [ToolRequest(tool_name="echo", input={"x": 1}) for _ in range(32)]
    )

    assert tool.calls == 1
    assert all(response.status == "ok" for response in responses)
    assert sum(response.metadata.cache_hit for response in responses) == 31
    registry.close()


def test_registry_persistent_cache_survives_registry_restart(tmp_path):
    cache_path = tmp_path / "tool-cache.sqlite3"
    settings = ServiceSettings(
        enable_molgpka=False,
        prewarm_molgpka=False,
        cache_path=cache_path,
    )
    first_tool = EchoTool()
    first = ToolRegistry([first_tool])
    first.initialize_all(settings)
    first.invoke(ToolRequest(tool_name="echo", input={"x": 1}))
    first.close()

    second_tool = EchoTool()
    second = ToolRegistry([second_tool])
    second.initialize_all(settings)
    response = second.invoke(ToolRequest(tool_name="echo", input={"x": 1}))

    assert response.metadata.cache_hit is True
    assert second_tool.calls == 0
    second.close()
