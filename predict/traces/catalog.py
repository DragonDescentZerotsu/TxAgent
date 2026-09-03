"""Discover centralized prediction traces for the viewer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from predict.traces.io import DEFAULT_TRACE_ROOT, TRACE_SCHEMA_VERSION


def discover_traces(root: str | Path = DEFAULT_TRACE_ROOT) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(Path(root).glob("*/*/*/*/*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("schema_version") == TRACE_SCHEMA_VERSION:
            rows.append({**row, "path": str(path)})
    return rows

