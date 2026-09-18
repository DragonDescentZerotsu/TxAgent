"""Run the full mechanism-family experiment.

For each query molecule, full inference retrieves every task-declared
mechanism family and keeps each non-empty family as an independent group
branch. The shared runtime also runs the query-only single-molecule branch,
then combines it with every mechanism-family output in one final prediction.

``retrieve_full_evidence`` preserves the mechanism-family groups returned by the
retrieval shared with ``flat.py``. CLI parsing, scheduling, checkpoints, and
metrics are delegated to ``batch.mode_main``.
"""

from __future__ import annotations

from typing import Any, Mapping

from predict.harnesses.branches.assay_transfer import (
    ASSAY_TRANSFER_DIVERSITY_NONE,
    ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
    ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
)
from predict.harnesses.branches.retrieval import (
    BranchRetrievalConfig,
    RetrievalReranker,
    retrieve_mechanism_evidence,
)
from predict.retrieval.policies import NeighborIdentityPolicy, SIMILARITY_SELECTOR


def retrieve_full_evidence(
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
    """Retrieve mechanism families and preserve one branch per family."""
    return retrieve_mechanism_evidence(
        query_smiles,
        index,
        config=config,
        mode="full_mechanism",
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


if __name__ == "__main__":
    raise SystemExit(
        "Use: python -m predict.harnesses.branches --organization full ..."
    )
