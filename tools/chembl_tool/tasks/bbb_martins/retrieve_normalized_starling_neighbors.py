"""Retrieve BBB analogs from the current heldout-filtered BBB v3 index."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.chembl_tool.common.task_workflows.retrieve_neighbors import main as run_retrieval
DEFAULT_INDEX = Path(
    "outputs/paper/molecular_evidence_agent_starling_scaffold_"
    "experimental_meaningful_cns_access_v3/evidence/"
    "bbb_starling_v7/08_neighbor_index"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--benchmark-split", choices=("scaffold",), required=True
    )
    _, remaining = parser.parse_known_args(argv)
    return run_retrieval(str(DEFAULT_INDEX), __doc__ or "", remaining)


if __name__ == "__main__":
    raise SystemExit(main())
