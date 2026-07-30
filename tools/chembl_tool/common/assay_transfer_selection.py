"""Deterministic diversity selection for scored assay-transfer records.

The assay-transfer model score is not calibrated against a structural or assay
coverage metric.  Rather than combining incomparable values in a weighted sum,
the selector uses a score-slack frontier: at every output slot it may choose
only records within a declared score loss of the best remaining record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


ASSAY_TRANSFER_DIVERSITY_NONE = "none"
ASSAY_TRANSFER_DIVERSITY_STRUCTURAL = "structural"
ASSAY_TRANSFER_DIVERSITY_ASSAY = "assay"
ASSAY_TRANSFER_DIVERSITY_MODES = (
    ASSAY_TRANSFER_DIVERSITY_NONE,
    ASSAY_TRANSFER_DIVERSITY_STRUCTURAL,
    ASSAY_TRANSFER_DIVERSITY_ASSAY,
)
ASSAY_TRANSFER_SELECTION_VERSION = "assay_transfer_score_slack_diversity.v1"


def validate_assay_transfer_diversity(
    *, mode: str, score_slack: float
) -> None:
    if mode not in ASSAY_TRANSFER_DIVERSITY_MODES:
        raise ValueError(
            "Unknown assay-transfer diversity mode "
            f"{mode!r}; expected one of {', '.join(ASSAY_TRANSFER_DIVERSITY_MODES)}"
        )
    if not 0.0 <= score_slack <= 1.0:
        raise ValueError("assay-transfer diversity score slack must be between 0 and 1 inclusive")
    if mode == ASSAY_TRANSFER_DIVERSITY_NONE and score_slack != 0.0:
        raise ValueError("assay-transfer diversity score slack requires a non-none diversity mode")


def assay_transfer_selection_policy(*, mode: str, score_slack: float) -> dict[str, Any]:
    """Return replayable, audit-only provenance for record selection."""
    validate_assay_transfer_diversity(mode=mode, score_slack=score_slack)
    return {
        "version": ASSAY_TRANSFER_SELECTION_VERSION,
        "mode": mode,
        "score_slack": score_slack,
        "selection_unit": "scored_record",
        "score_frontier": "best_remaining_score_minus_score_slack.v1",
        "structural_metric": "greedy_marginal_query_morgan_bit_coverage"
        if mode == ASSAY_TRANSFER_DIVERSITY_STRUCTURAL
        else "not_selected",
        "assay_metric": "canonical_endpoint_key_novelty"
        if mode == ASSAY_TRANSFER_DIVERSITY_ASSAY
        else "not_selected",
        "rank1_policy": (
            "max_marginal_query_morgan_bit_coverage"
            if mode == ASSAY_TRANSFER_DIVERSITY_STRUCTURAL
            else "highest_transfer_score_endpoint_novelty_tie"
            if mode == ASSAY_TRANSFER_DIVERSITY_ASSAY
            else "transfer_score"
        ),
    }


def select_assay_transfer_records(
    records: Sequence[dict[str, Any]],
    *,
    top_k: int,
    mode: str,
    score_slack: float,
    query_fingerprint: Any,
    fingerprints_by_molecule: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select records under a score frontier without mutating their payloads.

    ``records`` must already be score-ranked and pass all retrieval eligibility
    and minimum-score policies. A zero slack intentionally returns the existing
    score order verbatim, including its deterministic tie behavior.
    """
    validate_assay_transfer_diversity(mode=mode, score_slack=score_slack)
    if top_k <= 0 or not records:
        return [], _selection_audit([], [], mode=mode, score_slack=score_slack, query_bits=frozenset())
    if mode == ASSAY_TRANSFER_DIVERSITY_NONE or score_slack == 0.0:
        selected = list(records[:top_k])
        return selected, _selection_audit(
            selected,
            [
                {
                    "rank": rank,
                    "selection_reason": "score_ranking",
                    "best_remaining_score": float(record["transfer_selection_score"]),
                    "selected_score": float(record["transfer_selection_score"]),
                    "score_loss": 0.0,
                }
                for rank, record in enumerate(selected, start=1)
            ],
            mode=mode,
            score_slack=score_slack,
            query_bits=frozenset(int(bit) for bit in query_fingerprint.GetOnBits()),
            fingerprints_by_molecule=fingerprints_by_molecule,
        )

    query_bits = frozenset(int(bit) for bit in query_fingerprint.GetOnBits())
    shared_bits = {
        id(record): _shared_query_bits(record, query_bits, fingerprints_by_molecule)
        for record in records
    }
    remaining = list(records)
    selected: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    covered_bits: set[int] = set()
    seen_endpoints: set[str] = set()
    while remaining and len(selected) < top_k:
        best_score = max(float(record["transfer_selection_score"]) for record in remaining)
        frontier = [
            record
            for record in remaining
            if float(record["transfer_selection_score"]) >= best_score - score_slack - 1e-12
        ]
        if mode == ASSAY_TRANSFER_DIVERSITY_STRUCTURAL:
            choice = min(
                frontier,
                key=lambda record: (
                    -len(shared_bits[id(record)] - covered_bits),
                    *_score_tie_break(record),
                ),
            )
            marginal_bits = len(shared_bits[id(choice)] - covered_bits)
            reason = "max_marginal_query_morgan_bit_coverage"
        else:
            novel = [record for record in frontier if _endpoint_key(record) not in seen_endpoints and _endpoint_key(record)]
            choice = min(novel or frontier, key=_score_tie_break)
            marginal_bits = len(shared_bits[id(choice)] - covered_bits)
            reason = "new_canonical_endpoint_key" if novel else "score_fallback_after_endpoint_coverage"
        score = float(choice["transfer_selection_score"])
        endpoint_key = _endpoint_key(choice)
        trace.append(
            {
                "rank": len(selected) + 1,
                "selection_reason": reason,
                "best_remaining_score": best_score,
                "selected_score": score,
                "score_loss": best_score - score,
                "frontier_size": len(frontier),
                "marginal_query_morgan_bits": marginal_bits,
                "selected_canonical_endpoint_key": endpoint_key,
                "endpoint_was_new": bool(endpoint_key and endpoint_key not in seen_endpoints),
            }
        )
        selected.append(choice)
        remaining.remove(choice)
        covered_bits.update(shared_bits[id(choice)])
        if endpoint_key:
            seen_endpoints.add(endpoint_key)
    return selected, _selection_audit(
        selected,
        trace,
        mode=mode,
        score_slack=score_slack,
        query_bits=query_bits,
        fingerprints_by_molecule=fingerprints_by_molecule,
    )


def _selection_audit(
    selected: Sequence[dict[str, Any]],
    trace: list[dict[str, Any]],
    *,
    mode: str,
    score_slack: float,
    query_bits: frozenset[int],
    fingerprints_by_molecule: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fingerprints = fingerprints_by_molecule or {}
    covered = set()
    for record in selected:
        covered.update(_shared_query_bits(record, query_bits, fingerprints))
    endpoints = {key for key in (_endpoint_key(record) for record in selected) if key}
    return {
        **assay_transfer_selection_policy(mode=mode, score_slack=score_slack),
        "n_selected": len(selected),
        "n_distinct_selected_molecules": len(
            {str(record.get("molecule_chembl_id") or "") for record in selected}
        ),
        "n_distinct_selected_canonical_endpoint_keys": len(endpoints),
        "query_morgan_bit_coverage": len(covered) / len(query_bits) if query_bits else None,
        "selection_trace": trace,
    }


def _shared_query_bits(
    record: Mapping[str, Any], query_bits: frozenset[int], fingerprints_by_molecule: Mapping[str, Any]
) -> frozenset[int]:
    fingerprint = fingerprints_by_molecule.get(str(record.get("molecule_chembl_id") or ""))
    if fingerprint is None:
        return frozenset()
    return frozenset(int(bit) for bit in fingerprint.GetOnBits() if int(bit) in query_bits)


def _endpoint_key(record: Mapping[str, Any]) -> str:
    winning_record = record.get("transfer_winning_record") or {}
    return str(winning_record.get("canonical_endpoint_key") or "")


def _score_tie_break(record: Mapping[str, Any]) -> tuple[float, int, str, str]:
    return (
        -float(record["transfer_selection_score"]),
        int(record.get("structural_rank") or 0),
        str(record.get("transfer_winning_record_id") or ""),
        str(record.get("molecule_chembl_id") or ""),
    )
