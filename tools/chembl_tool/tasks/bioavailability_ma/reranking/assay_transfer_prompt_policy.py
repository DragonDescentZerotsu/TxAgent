"""Opt-in LLM visibility policy for assay-transfer-ranked neighbors."""

from __future__ import annotations

import math
from typing import Any


SCORED_NEIGHBORS_POLICY_NAME = "assay_transfer_scored_neighbors.v1"
SCORED_NEIGHBORS_SCORE_FIELD = "assay_transfer_score"
SCORED_NEIGHBORS_SCORE_DECIMALS = 2


def validate_scored_neighbors_configuration(
    *,
    enabled: bool,
    experiment_mode: str,
    retrieval_source: str,
    retrieval_reranker: str,
) -> None:
    """Validate score visibility without changing the requested neighbor count."""
    if not enabled:
        return
    if experiment_mode != "full_mechanism":
        raise ValueError("--enable-assay-transfer-scores requires --experiment-mode full_mechanism")
    if retrieval_source not in {"starling", "starling_in_distribution"}:
        raise ValueError(
            "--enable-assay-transfer-scores requires --retrieval-source starling or starling_in_distribution"
        )
    if retrieval_reranker != "assay_transfer":
        raise ValueError("--enable-assay-transfer-scores requires --retrieval-reranker assay_transfer")


def prepare_assay_transfer_selected_neighbors(
    retrieval: dict[str, Any], *, expose_scores: bool
) -> None:
    """Apply scoreability filtering independently from LLM score visibility."""
    experiment = retrieval.get("experiment") or {}
    if experiment.get("mode") != "full_mechanism":
        raise ValueError("Scored-neighbor prompt policy requires full_mechanism retrieval")
    reranker = experiment.get("retrieval_reranker") or {}
    if reranker.get("name") != "assay_transfer":
        raise ValueError("Scored-neighbor prompt policy requires assay_transfer retrieval metadata")
    coverage = retrieval.get("coverage") or {}
    effective_top_k = int(coverage.get("top_k_per_group") or 0)
    if effective_top_k <= 0:
        raise ValueError("Scored-neighbor prompt policy requires a positive top_k_per_group")

    n_unscoreable_dropped = int(coverage.get("n_unscoreable_selected_dropped") or 0)
    for group in retrieval.get("groups") or []:
        selection = group.get("transfer_neighbor_selection")
        if selection is not None:
            selection["selection_metadata_is_llm_hidden"] = not expose_scores
            selection["selection_score_is_llm_visible"] = expose_scores
            selection["remaining_audit_metadata_is_llm_hidden"] = True
        scoreable_neighbors = []
        for neighbor in group.get("neighbors") or []:
            if "transfer_selection_score" not in neighbor:
                raise ValueError(
                    "Missing cached transfer_selection_score for selected neighbor "
                    f"{neighbor.get('molecule_chembl_id')!r} in group {group.get('group_id')!r}"
                )
            score = float(neighbor["transfer_selection_score"])
            if score == -1.0 and int(neighbor.get("transfer_scored_record_count") or 0) == 0:
                n_unscoreable_dropped += 1
                continue
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(
                    "Invalid transfer_selection_score for selected neighbor "
                    f"{neighbor.get('molecule_chembl_id')!r}: {score!r}"
                )
            scoreable_neighbors.append(neighbor)
        for rank, neighbor in enumerate(scoreable_neighbors, start=1):
            neighbor["rank"] = rank
        group["neighbors"] = scoreable_neighbors
        if selection is not None:
            selection["n_selected"] = len(scoreable_neighbors)

    coverage["n_groups_with_neighbors"] = sum(
        bool(group.get("neighbors")) for group in retrieval.get("groups") or []
    )
    coverage["n_neighbors_total"] = sum(
        len(group.get("neighbors") or []) for group in retrieval.get("groups") or []
    )
    coverage["n_unscoreable_selected_dropped"] = n_unscoreable_dropped

    if expose_scores:
        experiment["llm_neighbor_score_policy"] = {
            "name": SCORED_NEIGHBORS_POLICY_NAME,
            "public_field": SCORED_NEIGHBORS_SCORE_FIELD,
            "source_field": "transfer_selection_score",
            "round_decimals": SCORED_NEIGHBORS_SCORE_DECIMALS,
            "effective_top_k_per_group": effective_top_k,
            "final_synthesis_visibility": "group_analysis_only",
        }
    else:
        experiment.pop("llm_neighbor_score_policy", None)
    retrieval["experiment"] = experiment


def enable_scored_neighbors_prompt_policy(retrieval: dict[str, Any]) -> None:
    """Validate selected scores and mark only their rounded value as LLM-visible."""
    prepare_assay_transfer_selected_neighbors(retrieval, expose_scores=True)


def scored_neighbors_prompt_enabled(retrieval: dict[str, Any]) -> bool:
    policy = (retrieval.get("experiment") or {}).get("llm_neighbor_score_policy") or {}
    return policy.get("name") == SCORED_NEIGHBORS_POLICY_NAME


def public_assay_transfer_score(neighbor: dict[str, Any]) -> float:
    return round(float(neighbor["transfer_selection_score"]), SCORED_NEIGHBORS_SCORE_DECIMALS)
