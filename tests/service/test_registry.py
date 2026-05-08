from __future__ import annotations

from tools.service.config import ServiceSettings
from tools.service.registry import ToolRegistry
from tools.service.schemas import ToolRequest
from tools.service.tools.base import BaseTool


class EchoTool(BaseTool):
    name = "echo"
    version = "v1"
    description = "test echo tool"

    def invoke(self, payload: dict[str, object], *, return_debug: bool = False) -> dict[str, object]:
        return {"payload": payload}


def test_registry_wraps_successful_tool_response():
    registry = ToolRegistry([EchoTool()])
    registry.initialize_all(ServiceSettings(enable_molgpka=False, prewarm_molgpka=False))

    response = registry.invoke(ToolRequest(tool_name="echo", input={"x": 1}))

    assert response.status == "ok"
    assert response.output == {"payload": {"x": 1}}
    assert response.errors == []


def test_registry_reports_unknown_tool():
    registry = ToolRegistry([])

    response = registry.invoke(ToolRequest(tool_name="missing", input={}))

    assert response.status == "error"
    assert response.errors[0].code == "UNKNOWN_TOOL"

