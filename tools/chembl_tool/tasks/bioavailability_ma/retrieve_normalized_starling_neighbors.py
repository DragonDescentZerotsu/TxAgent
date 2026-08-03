"""Retrieve Bioavailability analogs from the compact normalized-v6 index."""

from __future__ import annotations

from pathlib import Path

from tools.chembl_tool.common.task_workflows.retrieve_neighbors import main as run_retrieval
from tools.chembl_tool.tasks.bioavailability_ma.build_normalized_starling_evidence_library import (
    DEFAULT_OUT_DIR,
)


DEFAULT_INDEX = str(Path(DEFAULT_OUT_DIR) / "05_neighbor_index")


def main(argv: list[str] | None = None) -> int:
    return run_retrieval(DEFAULT_INDEX, __doc__ or "", argv)


if __name__ == "__main__":
    raise SystemExit(main())
