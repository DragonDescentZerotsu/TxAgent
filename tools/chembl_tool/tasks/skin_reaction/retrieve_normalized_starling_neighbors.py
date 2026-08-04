"""Retrieve Skin_Reaction analogs from the compact normalized-v6 index."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.chembl_tool.common.task_workflows.retrieve_neighbors import main as run_retrieval
from tools.chembl_tool.tasks.skin_reaction.build_normalized_starling_evidence_library import (
    DEFAULT_OUT_DIR,
)


DEFAULT_INDEX_ROOT = Path(DEFAULT_OUT_DIR) / "08_neighbor_index"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--benchmark-split", choices=("random", "scaffold"), required=True
    )
    split_args, remaining = parser.parse_known_args(argv)
    default_index = str(DEFAULT_INDEX_ROOT / split_args.benchmark_split)
    return run_retrieval(default_index, __doc__ or "", remaining)


if __name__ == "__main__":
    raise SystemExit(main())
