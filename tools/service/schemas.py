from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ToolRequestOptions(BaseModel):
    timeout_s: float | None = 120
    return_debug: bool = False


class ToolRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    tool_name: str
    version: str = "v1"
    input: dict[str, Any] = Field(default_factory=dict)
    options: ToolRequestOptions = Field(default_factory=ToolRequestOptions)


class ToolErrorPayload(BaseModel):
    code: str
    message: str
    recoverable: bool = True


class ToolResponseMetadata(BaseModel):
    started_at: datetime
    finished_at: datetime
    latency_ms: int
    model_or_index_version: str | None = None


class ToolResponse(BaseModel):
    request_id: str
    tool_name: str
    version: str
    status: Literal["ok", "error"]
    output: dict[str, Any] | None
    warnings: list[str] = Field(default_factory=list)
    errors: list[ToolErrorPayload] = Field(default_factory=list)
    metadata: ToolResponseMetadata


class ToolInfo(BaseModel):
    name: str
    version: str
    description: str
    initialized: bool
    initialization_error: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)

