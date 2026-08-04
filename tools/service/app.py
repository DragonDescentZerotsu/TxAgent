from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from tools.service.config import get_settings
from tools.service.registry import ToolRegistry
from tools.service.runtime import configure_compute_runtime
from tools.service.schemas import ToolBatchRequest, ToolRequest
from tools.service.tools.mmp_structure_compare import MmpStructureCompareTool
from tools.service.tools.properties_compare import PropertiesCompareTool
from tools.service.tools.rdkit_properties import MoleculePropertiesTool


def build_registry() -> ToolRegistry:
    properties_tool = MoleculePropertiesTool()
    return ToolRegistry(
        [
            properties_tool,
            PropertiesCompareTool(properties_tool),
            MmpStructureCompareTool(),
        ]
    )


registry = build_registry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_compute_runtime(settings)
    registry.initialize_all(settings)
    try:
        yield
    finally:
        registry.close()


app = FastAPI(title="TxAgent Resident Tool Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    tools = registry.list_tools()
    return {
        "status": "ok",
        "tools_total": len(tools),
        "tools_initialized": sum(1 for tool in tools if tool.initialized),
        "tools_with_initialization_errors": [
            tool.name for tool in tools if tool.initialization_error is not None
        ],
    }


@app.get("/tools")
def list_tools():
    return {"tools": [tool.model_dump() for tool in registry.list_tools()]}


@app.post("/tools/{tool_name}/invoke")
def invoke_tool(tool_name: str, request: ToolRequest):
    if request.tool_name != tool_name:
        request = request.model_copy(update={"tool_name": tool_name})
    return registry.invoke(request).model_dump(mode="json")


@app.post("/tools/invoke")
def invoke_tool_from_body(request: ToolRequest):
    return registry.invoke(request).model_dump(mode="json")


@app.post("/tools/batch")
def invoke_tools_batch(request: ToolBatchRequest):
    return {
        "responses": [
            response.model_dump(mode="json")
            for response in registry.invoke_many(request.requests)
        ]
    }


@app.post("/tools/{tool_name}")
def invoke_tool_shortcut(tool_name: str, payload: dict[str, Any]):
    if "input" in payload or "tool_name" in payload:
        request = ToolRequest(**{**payload, "tool_name": tool_name})
    else:
        request = ToolRequest(tool_name=tool_name, input=payload)
    response = registry.invoke(request)
    if response.status == "error":
        raise HTTPException(status_code=400, detail=response.model_dump(mode="json"))
    return response.model_dump(mode="json")
