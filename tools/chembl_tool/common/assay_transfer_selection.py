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
ASSAY_TRANSFER_SELECTION_VERSION = "assay_transfer_score_slack_diversity.v2"
ASSAY_TRANSFER_SELECTION_SCORED_RECORD = "scored_record"
ASSAY_TRANSFER_SELECTION_UNIQUE_MOLECULE = "unique_molecule"
ASSAY_TRANSFER_SELECTION_UNITS = (
    ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    ASSAY_TRANSFER_SELECTION_UNIQUE_MOLECULE,
)
ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT = 1
ASSAY_TRANSFER_RECORDS_PER_MOLECULE_MAX = 10


def validate_assay_transfer_selection_unit(selection_unit: str) -> None:
    if selection_unit not in ASSAY_TRANSFER_SELECTION_UNITS:
        raise ValueError(
            f"Unknown assay-transfer selection unit {selection_unit!r}; expected one of "
            + ", ".join(ASSAY_TRANSFER_SELECTION_UNITS)
        )


def validate_assay_transfer_records_per_molecule(
    records_per_molecule: int,
    *,
    selection_unit: str,
) -> None:
    validate_assay_transfer_selection_unit(selection_unit)
    if not (
        ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT
        <= records_per_molecule
        <= ASSAY_TRANSFER_RECORDS_PER_MOLECULE_MAX
    ):
        raise ValueError(
            "assay-transfer records per molecule must be between "
            f"{ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT} and "
            f"{ASSAY_TRANSFER_RECORDS_PER_MOLECULE_MAX} inclusive"
        )
    if (
        records_per_molecule > ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT
        and selection_unit != ASSAY_TRANSFER_SELECTION_UNIQUE_MOLECULE
    ):
        raise ValueError(
            "assay-transfer records per molecule greater than 1 requires "
            "selection unit unique_molecule"
        )


def collapse_assay_transfer_records_by_molecule(
    records: Sequence[dict[str, Any]],
    *,
    records_per_molecule: int = ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep one molecule and up to N score-ranked, endpoint-distinct records."""
    validate_assay_transfer_records_per_molecule(
        records_per_molecule,
        selection_unit=ASSAY_TRANSFER_SELECTION_UNIQUE_MOLECULE,
    )
    records_by_molecule: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        molecule_id = str(record.get("molecule_chembl_id") or "")
        records_by_molecule.setdefault(molecule_id, []).append(record)

    output = []
    n_duplicate_endpoints_skipped = 0
    n_underfilled_molecules = 0
    for molecule_rank, (molecule_id, molecule_records) in enumerate(
        records_by_molecule.items(), start=1
    ):
        distinct_records: list[dict[str, Any]] = []
        seen_endpoints: set[str] = set()
        duplicate_endpoints_skipped = 0
        for record in molecule_records:
            endpoint_key = _endpoint_key(record)
            if records_per_molecule > 1 and not endpoint_key:
                raise ValueError(
                    "Cannot select endpoint-distinct assay records because canonical "
                    f"endpoint identity is missing for {record.get('transfer_winning_record_id')!r}"
                )
            if endpoint_key in seen_endpoints:
                duplicate_endpoints_skipped += 1
                continue
            seen_endpoints.add(endpoint_key)
            distinct_records.append(record)

        displayed_records = distinct_records[:records_per_molecule]
        winner = displayed_records[0]
        selected_record_payloads = [
            {
                "record_rank": rank,
                "transfer_selection_score": float(record["transfer_selection_score"]),
                "transfer_winning_record_id": record.get("transfer_winning_record_id"),
                "canonical_endpoint_key": _endpoint_key(record),
                "transfer_winning_record": record.get("transfer_winning_record") or {},
            }
            for rank, record in enumerate(displayed_records, start=1)
        ]
        underfilled = len(displayed_records) < records_per_molecule
        n_duplicate_endpoints_skipped += duplicate_endpoints_skipped
        n_underfilled_molecules += int(underfilled)
        collapsed = {
            **winner,
            "transfer_scored_record_count": len(molecule_records),
            "transfer_molecule_selection_rank": molecule_rank,
            "transfer_records_per_molecule_limit": records_per_molecule,
            "transfer_selected_record_count": len(displayed_records),
            "transfer_distinct_endpoint_records_available": len(distinct_records),
            "transfer_duplicate_endpoint_records_skipped": duplicate_endpoints_skipped,
            "transfer_records_underfilled": underfilled,
        }
        if records_per_molecule > 1:
            collapsed["transfer_selected_records"] = selected_record_payloads
        output.append(collapsed)
    return output, {
        "selection_unit": ASSAY_TRANSFER_SELECTION_UNIQUE_MOLECULE,
        "molecule_identity_field": "molecule_chembl_id",
        "winning_record_policy": "highest_transfer_score",
        "within_molecule_record_policy": "highest_score_per_canonical_endpoint",
        "records_per_molecule": records_per_molecule,
        "duplicate_endpoint_backfill": False,
        "n_valid_records_before_collapse": len(records),
        "n_unique_molecules_after_collapse": len(output),
        "n_selected_records_after_collapse": sum(
            int(row["transfer_selected_record_count"]) for row in output
        ),
        "n_duplicate_endpoint_records_skipped": n_duplicate_endpoints_skipped,
        "n_underfilled_molecules": n_underfilled_molecules,
    }


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


def assay_transfer_selection_policy(
    *,
    mode: str,
    score_slack: float,
    selection_unit: str = ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    records_per_molecule: int = ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
) -> dict[str, Any]:
    """Return replayable, audit-only provenance for record selection."""
    validate_assay_transfer_diversity(mode=mode, score_slack=score_slack)
    validate_assay_transfer_records_per_molecule(
        records_per_molecule,
        selection_unit=selection_unit,
    )
    return {
        "version": ASSAY_TRANSFER_SELECTION_VERSION,
        "mode": mode,
        "score_slack": score_slack,
        "selection_unit": selection_unit,
        "records_per_molecule": records_per_molecule,
        "within_molecule_record_policy": (
            "highest_score_per_canonical_endpoint_no_duplicate_backfill"
            if selection_unit == ASSAY_TRANSFER_SELECTION_UNIQUE_MOLECULE
            else "not_applicable"
        ),
        "molecule_identity_field": (
            "molecule_chembl_id"
            if selection_unit == ASSAY_TRANSFER_SELECTION_UNIQUE_MOLECULE
            else "not_applicable"
        ),
        "winning_record_policy": (
            "highest_transfer_score"
            if selection_unit == ASSAY_TRANSFER_SELECTION_UNIQUE_MOLECULE
            else "not_applicable"
        ),
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
    selection_unit: str = ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    records_per_molecule: int = ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select records under a score frontier without mutating their payloads.

    ``records`` must already be score-ranked and pass all retrieval eligibility
    and minimum-score policies. A zero slack intentionally returns the existing
    score order verbatim, including its deterministic tie behavior.
    """
    validate_assay_transfer_diversity(mode=mode, score_slack=score_slack)
    validate_assay_transfer_records_per_molecule(
        records_per_molecule,
        selection_unit=selection_unit,
    )
    if top_k <= 0 or not records:
        return [], _selection_audit(
            [],
            [],
            mode=mode,
            score_slack=score_slack,
            query_bits=frozenset(),
            selection_unit=selection_unit,
            records_per_molecule=records_per_molecule,
        )
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
            selection_unit=selection_unit,
            records_per_molecule=records_per_molecule,
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
        selection_unit=selection_unit,
        records_per_molecule=records_per_molecule,
    )


def _selection_audit(
    selected: Sequence[dict[str, Any]],
    trace: list[dict[str, Any]],
    *,
    mode: str,
    score_slack: float,
    query_bits: frozenset[int],
    fingerprints_by_molecule: Mapping[str, Any] | None = None,
    selection_unit: str = ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    records_per_molecule: int = ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
) -> dict[str, Any]:
    fingerprints = fingerprints_by_molecule or {}
    covered = set()
    for record in selected:
        covered.update(_shared_query_bits(record, query_bits, fingerprints))
    endpoints = {key for key in (_endpoint_key(record) for record in selected) if key}
    return {
        **assay_transfer_selection_policy(
            mode=mode,
            score_slack=score_slack,
            selection_unit=selection_unit,
            records_per_molecule=records_per_molecule,
        ),
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
