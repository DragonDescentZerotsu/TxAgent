"""Shared retrieval-strategy CLI arguments.

Both the per-molecule pipeline (``run_reasoning_pipeline``) and the shared batch
runner (``reasoning_batch``) expose the same retrieval-strategy selection. Keeping
the definitions here guarantees the two entrypoints stay in lockstep -- the exact
drift that made the ``main``/``joseph`` retrieval rewrites collide.
"""

from __future__ import annotations

import argparse

from tools.chembl_tool.common.experiment_retrieval import (
    MORGAN_FINGERPRINT_STRATEGY,
    RETRIEVAL_STRATEGIES,
)
from tools.chembl_tool.common.neighbor_selection import (
    NEIGHBOR_SELECTORS,
    SIMILARITY_SELECTOR,
)


def add_retrieval_strategy_args(parser: argparse.ArgumentParser) -> None:
    """Add the ``--retrieval-strategy`` group shared by the reasoning CLIs.

    The strategy is the source of truth for the retrieval mechanic and the
    compatible ``--group-prompt-format``; each entrypoint validates that pairing.
    """
    group = parser.add_argument_group("Retrieval strategy")
    group.add_argument(
        "--retrieval-strategy",
        choices=list(RETRIEVAL_STRATEGIES),
        default=MORGAN_FINGERPRINT_STRATEGY,
        help=(
            "Retrieval mechanic and source of truth for the group-prompt format. "
            "morgan_fingerprint uses --morgan-neighbor-selector and pairs with "
            "--group-prompt-format legacy or morganfingerprint; assay_transfer_tool "
            "uses the assay-transfer reranker and requires --group-prompt-format "
            "assay_transfer_tool."
        ),
    )
    group.add_argument(
        "--morgan-neighbor-selector",
        choices=NEIGHBOR_SELECTORS,
        default=SIMILARITY_SELECTOR,
        help="morgan_fingerprint strategy only: neighbor selection policy.",
    )
    group.add_argument(
        "--assay-transfer-initial-morgan-filter",
        type=int,
        default=100,
        help=(
            "assay_transfer_tool strategy only: size of the initial top-N tanimoto pool "
            "fed to candidate validation and reranking before the final top-k."
        ),
    )
