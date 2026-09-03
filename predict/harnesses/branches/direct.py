"""Run the direct-evidence experiment.

For each query molecule, direct inference retrieves only task-declared direct
outcome evidence. Each returned direct family remains a separate group branch.
The shared stage runtime also runs the query-only single-molecule branch, then
combines that branch with all direct-family outputs in one final prediction.

``retrieve_direct_evidence`` implements the direct-only evidence decision below.
``main`` delegates only shared CLI parsing, scheduling, checkpoints, and metrics
to ``batch.mode_main``.
"""

from __future__ import annotations

from typing import Any, Mapping

from predict.harnesses.branches.batch import mode_main
from predict.harnesses.branches.assay_transfer import (
    ASSAY_TRANSFER_DIVERSITY_NONE,
    ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
    ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
)
from predict.harnesses.branches.retrieval import (
    BranchRetrievalConfig,
    retrieve_branches,
)
from predict.harnesses.branches.reranker import RetrievalReranker
from predict.retrieval.policies import NeighborIdentityPolicy, SIMILARITY_SELECTOR


def retrieve_direct_evidence(
    query_smiles: str,
    index: Mapping[str, Any],
    *,
    config: BranchRetrievalConfig,
    top_k_per_group: int,
    min_similarity: float,
    neighbor_identity_policy: str = NeighborIdentityPolicy.OPERATIONAL.value,
    neighbor_selector: str = SIMILARITY_SELECTOR,
    reranker: RetrievalReranker | None = None,
    assay_transfer_initial_morgan_filter: int = 100,
    assay_transfer_min_score: float | None = None,
    assay_transfer_diversity_mode: str = ASSAY_TRANSFER_DIVERSITY_NONE,
    assay_transfer_diversity_score_slack: float = 0.0,
    assay_transfer_selection_unit: str = ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    assay_transfer_records_per_molecule: int = ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
) -> dict[str, Any]:
    """Retrieve only task-declared direct outcome groups for one query."""
    return retrieve_branches(
        query_smiles,
        index,
        branch_definitions=config.direct_groups,
        source_name=config.source_name,
        mode="direct",
        top_k_per_group=top_k_per_group,
        min_similarity=min_similarity,
        neighbor_identity_policy=neighbor_identity_policy,
        neighbor_selector=neighbor_selector,
        reranker=reranker,
        assay_transfer_initial_morgan_filter=assay_transfer_initial_morgan_filter,
        assay_transfer_min_score=assay_transfer_min_score,
        assay_transfer_diversity_mode=assay_transfer_diversity_mode,
        assay_transfer_diversity_score_slack=assay_transfer_diversity_score_slack,
        assay_transfer_selection_unit=assay_transfer_selection_unit,
        assay_transfer_records_per_molecule=assay_transfer_records_per_molecule,
    )


def main(argv: list[str] | None = None) -> int:
    return mode_main("direct", argv)


if __name__ == "__main__":
    raise SystemExit(main())
