"""Load a frozen retrieval artifact for controlled paired experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_retrieval_replay(source_run_dir: str, query_smiles: str) -> dict[str, Any] | None:
    """Return the source run's retrieval payload after checking query alignment."""

    if not source_run_dir:
        return None
    source_path = Path(source_run_dir) / "retrieval.json"
    if not source_path.exists():
        raise FileNotFoundError(f"Retrieval replay artifact does not exist: {source_path}")
    retrieval = json.loads(source_path.read_text(encoding="utf-8"))
    if retrieval.get("status") != "ok":
        raise ValueError(f"Retrieval replay artifact is not successful: {source_path}")
    source_smiles = str((retrieval.get("query") or {}).get("input_smiles") or "")
    if source_smiles and source_smiles != query_smiles:
        raise ValueError(
            "Retrieval replay query mismatch: "
            f"expected {query_smiles!r}, found {source_smiles!r} in {source_path}"
        )
    return retrieval
