"""Directory creation for evidence-library builders."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
