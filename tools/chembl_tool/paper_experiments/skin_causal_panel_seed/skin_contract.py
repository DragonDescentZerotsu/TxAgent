"""Compile outcome-calibrated, multi-event Skin sensitization evidence cards.

This module contains only deterministic source compilation.  It does not inspect
benchmark labels or decide predictions for query molecules.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from copy import deepcopy
import hashlib
import json
from typing import Any

from tools.chembl_tool.common.evidence_contract import attach_minimal_evidence
from tools.chembl_tool.common.experiment_retrieval import (
    SourceExperimentConfig,
    evidence_family,
)
from tools.chembl_tool.tasks.skin_reaction.experiment_config import STARLING


CONTRACT_VERSION = "skin_sensitization_causal_panel.v1"
SOURCE_GROUP_ID = "Mechanism.sensitization_causal_panel"
ENDPOINT_GROUP = "sensitisation_causal_panel"
VALID_DIRECTIONS = frozenset({"positive", "negative"})
MIE_EVENT = "MIE_protein_binding"
DOWNSTREAM_EVENTS = frozenset(
    {
        "KE2_keratinocyte_activation",
        "KE3_dendritic_cell_activation",
        "KE4_T_cell_activation",
    }
)
AGREEMENT_THRESHOLD = 0.70


def causal_panel_retrieval_config() -> SourceExperimentConfig:
    """Return the seed-only view without registering it in the production task."""
    return SourceExperimentConfig(
        source_name="starling_causal_panel_seed_v1",
        direct_groups=STARLING.direct_groups,
        mechanism_groups=(
            STARLING.mechanism_groups[0],
            evidence_family(
                ENDPOINT_GROUP,
                family_label="Tier 2",
                source_group_ids=(SOURCE_GROUP_ID,),
                legacy_output_group_id="Mechanism.causal_panel",
            ),
        ),
    )


def compile_skin_causal_panel_rows(
    direct_records: Iterable[Mapping[str, Any]],
    aop_records: Iterable[Mapping[str, Any]],
    canonical_evidence_rows: Iterable[Mapping[str, Any]],
    *,
    max_examples_per_event: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return one causal card per strict, outcome-calibrated reference parent."""
    direct_by_parent = _group_by_parent(direct_records)
    aop_by_parent = _group_by_parent(aop_records)
    identity_rows = _canonical_identity_rows(canonical_evidence_rows)

    cards: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    candidate_parents = sorted(set(direct_by_parent) & set(aop_by_parent))
    for parent_key in candidate_parents:
        direct_rows = direct_by_parent[parent_key]
        aop_rows = aop_by_parent[parent_key]
        direct_direction = consensus_direction(
            row.get("outcome_label") for row in direct_rows
        )
        if direct_direction is None:
            rejection_counts["direct_without_70pct_direction"] += 1
            continue

        event_rows: dict[str, list[dict[str, Any]]] = {}
        event_directions: dict[str, str] = {}
        for event, rows in _group_by_event(aop_rows).items():
            direction = consensus_direction(row.get("result_label") for row in rows)
            if direction is not None:
                event_rows[event] = rows
                event_directions[event] = direction
        if MIE_EVENT not in event_directions:
            rejection_counts["missing_consensus_mie"] += 1
            continue
        downstream = sorted(DOWNSTREAM_EVENTS & set(event_directions))
        if not downstream:
            rejection_counts["missing_consensus_downstream_event"] += 1
            continue
        if len(set(event_directions.values())) != 1:
            rejection_counts["aop_event_direction_conflict"] += 1
            continue
        panel_direction = next(iter(event_directions.values()))
        if panel_direction != direct_direction:
            rejection_counts["aop_outcome_direction_conflict"] += 1
            continue

        identity = identity_rows.get(parent_key)
        if identity is None:
            rejection_counts["missing_canonical_evidence_identity"] += 1
            continue
        cards.append(
            _build_card(
                identity=identity,
                parent_key=parent_key,
                direct_rows=direct_rows,
                direct_direction=direct_direction,
                event_rows=event_rows,
                event_directions=event_directions,
                max_examples_per_event=max_examples_per_event,
            )
        )

    direction_counts = Counter(
        str(row["causal_panel"]["observed_direction"]) for row in cards
    )
    event_count_distribution = Counter(
        len(row["causal_panel"]["events"]) for row in cards
    )
    stats = {
        "contract_version": CONTRACT_VERSION,
        "agreement_threshold": AGREEMENT_THRESHOLD,
        "n_direct_parents": len(direct_by_parent),
        "n_aop_parents": len(aop_by_parent),
        "n_candidate_direct_aop_parents": len(candidate_parents),
        "n_compiled_cards": len(cards),
        "direction_counts": dict(sorted(direction_counts.items())),
        "event_count_distribution": {
            str(key): value for key, value in sorted(event_count_distribution.items())
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }
    return cards, stats


def consensus_direction(values: Iterable[Any]) -> str | None:
    """Return a strict 70%-agreement direction; ties and unknowns are rejected."""
    normalized = [
        text
        for value in values
        if (text := str(value or "").strip().lower()) in VALID_DIRECTIONS
    ]
    if not normalized:
        return None
    counts = Counter(normalized)
    direction, count = counts.most_common(1)[0]
    if list(counts.values()).count(count) > 1:
        return None
    if count / len(normalized) < AGREEMENT_THRESHOLD:
        return None
    return direction


def panel_direction_from_evidence_row(row: Mapping[str, Any]) -> str:
    panel = row.get("causal_panel") or {}
    return str(panel.get("observed_direction") or "").strip().lower()


def _build_card(
    *,
    identity: Mapping[str, Any],
    parent_key: str,
    direct_rows: list[dict[str, Any]],
    direct_direction: str,
    event_rows: Mapping[str, list[dict[str, Any]]],
    event_directions: Mapping[str, str],
    max_examples_per_event: int,
) -> dict[str, Any]:
    events = sorted(event_directions)
    event_counts = {
        event: Counter(
            str(row.get("result_label") or "").strip().lower()
            for row in event_rows[event]
            if str(row.get("result_label") or "").strip().lower() in VALID_DIRECTIONS
        )
        for event in events
    }
    direct_counts = Counter(
        str(row.get("outcome_label") or "").strip().lower()
        for row in direct_rows
        if str(row.get("outcome_label") or "").strip().lower() in VALID_DIRECTIONS
    )
    examples = []
    for event in events:
        ranked = sorted(
            event_rows[event],
            key=lambda row: (
                -float(row.get("confidence") or 0.0),
                str(row.get("pmid") or ""),
                str(row.get("source_record_id") or ""),
            ),
        )
        for source_row in ranked[:max_examples_per_event]:
            examples.append(
                {
                    "source_id": "skin_aop_sensitization_canonical_v3",
                    "source_record_id": str(source_row.get("source_record_id") or ""),
                    "endpoint_type": event,
                    "reported_value": str(source_row.get("result_label") or ""),
                    "reported_units": "",
                    "context": {
                        key: str(source_row.get(key) or "")
                        for key in (
                            "assay_type",
                            "endpoint_or_target",
                            "experimental_conditions",
                            "qualifying_conditions",
                        )
                        if str(source_row.get(key) or "").strip()
                    },
                    "support_text": str(source_row.get("support_text") or ""),
                }
            )
    confidence_values = [
        float(row.get("confidence"))
        for row in [*direct_rows, *(row for rows in event_rows.values() for row in rows)]
        if row.get("confidence") not in (None, "")
    ]
    min_confidence = min(confidence_values) if confidence_values else ""
    event_summary = "; ".join(
        f"{event}: {event_directions[event]} "
        f"({dict(sorted(event_counts[event].items()))})"
        for event in events
    )
    source_record_ids = sorted(
        {
            str(row.get("source_record_id") or "")
            for row in [*direct_rows, *(row for rows in event_rows.values() for row in rows)]
            if str(row.get("source_record_id") or "")
        }
    )
    digest = hashlib.sha256("\n".join(source_record_ids).encode("utf-8")).hexdigest()
    row = {
        "molecule_chembl_id": str(identity["molecule_chembl_id"]),
        "canonical_smiles": str(identity["canonical_smiles"]),
        "assay_chembl_id": "STARLING_SKIN_CAUSAL_PANEL_SEED_V1",
        "assay_tier": "Tier 2",
        "endpoint_group": ENDPOINT_GROUP,
        "group_id": SOURCE_GROUP_ID,
        "standard_type": "outcome-calibrated multi-event sensitization causal panel",
        "standard_relation": "",
        "standard_value": direct_direction,
        "standard_units": "qualitative observed direction",
        "pchembl_value": "",
        "activity_comment": (
            f"Direct sensitization outcome and {len(events)} AOP events show the same "
            f"observed {direct_direction} direction."
        ),
        "data_validity_comment": "",
        "assay_description": (
            f"direct outcome: {direct_direction} ({dict(sorted(direct_counts.items()))})\n"
            f"AOP events: {event_summary}\n"
            "Panel inclusion requires a protein-binding MIE, at least one downstream "
            "key event, event-level agreement, and agreement with the observed direct outcome."
        ),
        "target_pref_name": "skin sensitization adverse-outcome pathway",
        "target_genes": "",
        "organism": "",
        "confidence_score": min_confidence,
        "relationship_type": "",
        "evidence_source": "Starling/Skin_Reaction/causal_panel_seed_v1",
        "evidence_role": "mechanistic_factor",
        "evidence_scope": {
            "direct_outcome_direction": direct_direction,
            "aop_events": events,
            "panel_contract": CONTRACT_VERSION,
        },
        "transferability": "not_assessed",
        "uncertainty": [
            "This is a reference-molecule panel; transferability to the query must be assessed independently."
        ],
        "source_molecule_names": [],
        "source_record_count": len(source_record_ids),
        "source_numeric_record_count": 0,
        "source_qualitative_record_count": len(source_record_ids),
        "source_endpoint_counts": {
            "direct_outcome": dict(sorted(direct_counts.items())),
            **{event: dict(sorted(event_counts[event].items())) for event in events},
        },
        "source_support_texts": [example["support_text"] for example in examples],
        "source_record_examples": examples,
        "parent_inchi_key": parent_key,
        "causal_panel": {
            "contract_version": CONTRACT_VERSION,
            "observed_direction": direct_direction,
            "events": events,
            "source_record_ids_sha256": digest,
        },
    }
    return attach_minimal_evidence(row)


def _group_by_parent(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for source_row in rows:
        row = dict(source_row)
        key = str(row.get("parent_inchi_key") or "").strip()
        if key:
            grouped.setdefault(key, []).append(row)
    return grouped


def _group_by_event(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        event = str(row.get("aop_event") or "").strip()
        if event:
            grouped.setdefault(event, []).append(dict(row))
    return grouped


def _canonical_identity_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    identities: dict[str, dict[str, str]] = {}
    for source_row in rows:
        row = dict(source_row)
        minimal = row.get("minimal_evidence") or {}
        molecule = minimal.get("molecule") or {}
        parent_key = str(row.get("parent_inchi_key") or row.get("standard_inchi_key") or "").strip()
        if not parent_key:
            continue
        molecule_id = str(row.get("molecule_chembl_id") or molecule.get("id") or "").strip()
        canonical_smiles = str(row.get("canonical_smiles") or molecule.get("canonical_smiles") or "").strip()
        if molecule_id and canonical_smiles:
            candidate = {
                "molecule_chembl_id": molecule_id,
                "canonical_smiles": canonical_smiles,
            }
            previous = identities.get(parent_key)
            if previous is None or (
                candidate["molecule_chembl_id"], candidate["canonical_smiles"]
            ) < (previous["molecule_chembl_id"], previous["canonical_smiles"]):
                # A small number of source SMILES variants collapse to the same
                # normalized parent.  Keep one stable representative instead of
                # duplicating a correlated causal card across variants.
                identities[parent_key] = candidate
    return identities


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clone_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Small public helper used by the index materializer to avoid shared mutation."""
    return deepcopy(dict(row))
