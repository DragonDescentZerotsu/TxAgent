"""Retrieve top analog neighbors for each Skin_Reaction Tier.endpoint_group."""

from __future__ import annotations

from pathlib import Path

from tools.chembl_tool.tasks.skin_reaction.build_evidence_library import DEFAULT_OUT_DIR
from tools.chembl_tool.common.task_workflows.retrieve_neighbors import (
    load_index,
    main as run_retrieval,
    retrieve_neighbors,
    similarity_bucket,
)


DEFAULT_INDEX = str(Path(DEFAULT_OUT_DIR) / "skin_reaction_neighbor_index.pkl")


def main(argv: list[str] | None = None) -> int:
    return run_retrieval(DEFAULT_INDEX, __doc__ or "", argv)


if __name__ == "__main__":
    raise SystemExit(main())
