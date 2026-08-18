"""Opt-in LLM visibility policy for assay-transfer-ranked records."""

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
    if not enabled:
        return
    if experiment_mode not in {"full_flat", "full_mechanism"}:
        raise ValueError(
            "--enable-assay-transfer-scores requires --experiment-mode "
            "full_flat or full_mechanism"
        )
    if retrieval_source not in {"starling", "starling_in_distribution"}:
        raise ValueError(
            "--enable-assay-transfer-scores requires --retrieval-source "
            "starling or starling_in_distribution"
        )
    if retrieval_reranker != "assay_transfer":
        raise ValueError("--enable-assay-transfer-scores requires assay_transfer retrieval")


def prepare_assay_transfer_selected_neighbors(
    retrieval: dict[str, Any], *, expose_scores: bool
) -> None:
    experiment = retrieval.get("experiment") or {}
    mode = str(experiment.get("mode") or "")
    if mode not in {"full_flat", "full_mechanism"}:
        raise ValueError(
            "Scored-neighbor prompt policy requires full_flat or full_mechanism retrieval"
        )
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
        scoreable = []
        for neighbor in group.get("neighbors") or []:
            family_selections = neighbor.get("transfer_family_selections") or []
            if mode == "full_flat" and not family_selections:
                raise ValueError("Flat assay-transfer neighbor is missing family selections")
            for family in family_selections:
                _validate_public_score_bundle(
                    family,
                    context=f"{group.get('group_id')!r}/{family.get('group_id')!r}",
                )
            if "transfer_selection_score" not in neighbor:
                raise ValueError(
                    "Missing cached transfer_selection_score for selected record "
                    f"{neighbor.get('transfer_winning_record_id')!r} in group "
                    f"{group.get('group_id')!r}"
                )
            score = float(neighbor["transfer_selection_score"])
            if score == -1.0 and int(neighbor.get("transfer_scored_record_count") or 0) == 0:
                n_unscoreable_dropped += 1
                continue
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"Invalid transfer_selection_score: {score!r}")
            _validate_public_score_bundle(
                neighbor,
                context=f"{group.get('group_id')!r}",
            )
            scoreable.append(neighbor)
        for rank, neighbor in enumerate(scoreable, start=1):
            neighbor["rank"] = rank
        group["neighbors"] = scoreable
        if selection is not None:
            selection["n_selected"] = len(scoreable)

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
    prepare_assay_transfer_selected_neighbors(retrieval, expose_scores=True)


def scored_neighbors_prompt_enabled(retrieval: dict[str, Any]) -> bool:
    policy = (retrieval.get("experiment") or {}).get("llm_neighbor_score_policy") or {}
    return policy.get("name") == SCORED_NEIGHBORS_POLICY_NAME


def public_assay_transfer_score(neighbor: dict[str, Any]) -> float:
    return round(float(neighbor["transfer_selection_score"]), SCORED_NEIGHBORS_SCORE_DECIMALS)


def public_assay_transfer_records(neighbor: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the ordered, score-visible record cards without audit identifiers."""
    bundled = neighbor.get("transfer_selected_records") or []
    if not bundled:
        bundled = [
            {
                "record_rank": 1,
                "transfer_selection_score": neighbor["transfer_selection_score"],
                "transfer_winning_record": neighbor.get("transfer_winning_record") or {},
            }
        ]
    return [
        {
            "record_rank": int(record.get("record_rank") or rank),
            "assay_transfer_score": round(
                float(record["transfer_selection_score"]),
                SCORED_NEIGHBORS_SCORE_DECIMALS,
            ),
            "record": record.get("transfer_winning_record") or {},
        }
        for rank, record in enumerate(bundled, start=1)
    ]


def public_assay_transfer_families(
    neighbor: dict[str, Any], group: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return family-attributed score bundles without internal record IDs."""
    families = neighbor.get("transfer_family_selections") or []
    if not families:
        families = [
            {
                "group_id": group.get("group_id"),
                "tier": group.get("tier"),
                "endpoint_group": group.get("endpoint_group"),
                "family_rank": neighbor.get("rank"),
                "transfer_selection_score": neighbor.get("transfer_selection_score"),
                "transfer_molecule_mean_score": neighbor.get(
                    "transfer_molecule_mean_score"
                ),
                "transfer_selected_records": neighbor.get("transfer_selected_records") or [],
                "transfer_winning_record": neighbor.get("transfer_winning_record") or {},
            }
        ]
    output = []
    for family in families:
        mean_score = family.get("transfer_molecule_mean_score")
        output.append(
            {
                "group_id": str(family.get("group_id") or ""),
                "tier": str(family.get("tier") or ""),
                "endpoint_group": str(family.get("endpoint_group") or ""),
                "family_rank": int(family.get("family_rank") or 0),
                "score_kind": "molecule_mean" if mean_score is not None else "selected_record",
                "assay_transfer_score": round(
                    float(
                        mean_score
                        if mean_score is not None
                        else family["transfer_selection_score"]
                    ),
                    SCORED_NEIGHBORS_SCORE_DECIMALS,
                ),
                "records": public_assay_transfer_records(family),
            }
        )
    return output


def _validate_public_score_bundle(bundle: dict[str, Any], *, context: str) -> None:
    score = float(bundle["transfer_selection_score"])
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"Invalid transfer_selection_score in {context}: {score!r}")
    for selected_record in bundle.get("transfer_selected_records") or []:
        selected_score = float(selected_record["transfer_selection_score"])
        if not math.isfinite(selected_score) or not 0.0 <= selected_score <= 1.0:
            raise ValueError(
                f"Invalid bundled transfer_selection_score in {context}: {selected_score!r}"
            )
        if not selected_record.get("transfer_winning_record"):
            raise ValueError("Bundled assay-transfer record is missing its source payload")
