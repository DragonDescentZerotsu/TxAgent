"""Spacing- and spelling-only endpoint normalization for Starling records.

This layer may normalize whitespace and explicitly reviewed typographical errors.
It preserves case and must not merge scientifically related endpoint concepts.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any, Iterable

from data.processing.evidence_library.shared.v2.endpoint_concepts import (
    load_endpoint_concept_maps,
)
from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    clean_text,
    endpoint_inventory_hash,
)
from data.processing.evidence_library.shared.v2.normalization.contracts import FamilyAssignment
from data.processing.evidence_library.shared.v2.normalization.measurements import (
    EndpointOrthography,
    canonicalize_endpoint,
)
from data.processing.evidence_library.versions.v9.tasks.bioavailability_ma.starling_record_canonicalization import (
    DIRECT_EVIDENCE_SCOPE,
    bioavailability_evidence_scope,
)


SPACING_AND_SPELLING_VERSION = "bioavailability_spacing_and_spelling.v1"
ENDPOINT_CONCEPT_VERSION = "bioavailability_endpoint_concepts.v1"

EXPECTED_ENDPOINT_INVENTORIES = {
    "oral_exposure": {
        "count": 149,
        "sha256": "a55791a100173ee82a96679ac75a3d3381452d7df8d981b23cbd3ff7e4453c1b",
    },
    "fa": {
        "count": 33,
        "sha256": "43d72d3407ede660852220ff1e986c9815773e0a1333957c5b54c248036edbd4",
    },
    "fg": {
        "count": 17,
        "sha256": "24c9afdc81db8b4c261163a1b9cb04b5a12b4c510377ac08608c69c5588087b2",
    },
    "fh": {
        "count": 19,
        "sha256": "40917d39b6a569927834157be69c4c91f5806784c9cc699b5f310efa9e31793d",
    },
    "hf_bioavailability": {
        "count": 1,
        "sha256": "95f78038b5cde8c7484c93f22f15a0c21d6ae2ba7451fdd2754df3be0d184c1a",
    },
}
ENDPOINT_CONCEPT_PATHS = tuple(
    Path(__file__).resolve().parent
    / "data_processing/canonicalization_v7/endpoint_concepts"
    / f"{source_id}.json"
    for source_id in sorted(EXPECTED_ENDPOINT_INVENTORIES)
)


_REVIEWED_CORRECTIONS = {
    "t max": "tmax",
    "solubidity": "solubility",
    "intest inal_absorption": "intestinal_absorption",
    "intestina l_absorption": "intestinal_absorption",
    "intest inal_effective_permeability": "intestinal_effective_permeability",
    "intestina\u00adl_effective_permeability": "intestinal_effective_permeability",
    "intestinaleffective_permeability": "intestinal_effective_permeability",
    "hepatic_first_pass_m etabolism": "hepatic_first_pass_metabolism",
    "uptake_or absorptive_transport": "uptake_or_absorptive_transport",
    "efflux_or_secretary_transport": "efflux_or_secretory_transport",
    "caco2_mdck_pampa_perability": "caco2_mdck_pampa_permeability",
    "caco2_mdck_pampa_permeibility": "caco2_mdck_pampa_permeability",
    "caco2_mdck_pamp\u0430_permeability": "caco2_mdck_pampa_permeability",
    "caco2_mdck_p permeability": "caco2_mdck_permeability",
    "caco2_mdck_pemiability": "caco2_mdck_permeability",
    "caco2_mdck_permiability": "caco2_mdck_permeability",
    "caco2_mdck_perva_permeability": "caco2_mdck_permeability",
    "caco2_mdck_poraineability": "caco2_mdck_permeability",
    "caco2_mdck_ppera_permeability": "caco2_mdck_permeability",
}

SpacingAndSpellingDecision = EndpointOrthography


def _apply_source_case(source: str, corrected: str) -> str:
    """Preserve simple source casing while replacing a reviewed misspelling."""
    letters = "".join(character for character in source if character.isalpha())
    if letters.isupper():
        return corrected.upper()
    if letters and letters[0].isupper() and letters[1:].islower():
        return corrected[:1].upper() + corrected[1:]
    return corrected


def spacing_and_spelling_decision(
    source_id: str,
    endpoint_name: str,
) -> SpacingAndSpellingDecision:
    """Return a conservative orthographic correction for one endpoint."""
    del source_id
    cleaned = clean_text(endpoint_name) or "missing_endpoint"
    key = cleaned.casefold()
    reviewed = _REVIEWED_CORRECTIONS.get(key)
    corrected = _apply_source_case(cleaned, reviewed) if reviewed else cleaned
    changed = corrected != cleaned
    return SpacingAndSpellingDecision(
        endpoint_name=cleaned,
        spacing_and_spelling_endpoint=corrected,
        status="reviewed_correction" if changed else "unchanged",
        reason="explicit_reviewed_spacing_or_spelling_error" if changed else "no_semantic_change",
        spacing_and_spelling_version=SPACING_AND_SPELLING_VERSION,
    )


def spacing_and_spelling_endpoint(source_id: str, endpoint_name: str) -> str:
    return spacing_and_spelling_decision(
        source_id, endpoint_name
    ).spacing_and_spelling_endpoint


@cache
def _endpoint_concept_maps() -> dict[str, dict[tuple[str, str], str]]:
    return load_endpoint_concept_maps(
        ENDPOINT_CONCEPT_PATHS,
        version=ENDPOINT_CONCEPT_VERSION,
        expected_inventories=EXPECTED_ENDPOINT_INVENTORIES,
        canonical_resolver=lambda source_id, endpoint: canonicalize_endpoint(
            spacing_and_spelling_endpoint(source_id, endpoint)
        ),
    )


def endpoint_concept(
    source_id: str,
    endpoint_name: str,
    canonical_endpoint_name: str | None = None,
) -> str:
    endpoint = clean_text(endpoint_name) or ""
    canonical = canonical_endpoint_name or canonicalize_endpoint(
        spacing_and_spelling_endpoint(source_id, endpoint)
    )
    try:
        return _endpoint_concept_maps()[source_id][(endpoint, canonical)]
    except KeyError as error:
        raise ValueError(
            f"unreviewed endpoint concept: {source_id}/{endpoint}/{canonical}"
        ) from error


def family_assignment(
    source_id: str,
    endpoint_name: str,
    record: Mapping[str, Any] | None = None,
) -> FamilyAssignment | None:
    """Map source records to the five stable retrieval families without changing endpoints."""
    endpoint = (clean_text(endpoint_name) or "").casefold()
    if str((record or {}).get("group_id") or "") == (
        "Observed.direct_oral_bioavailability"
    ):
        return FamilyAssignment(
            "Observed.direct_oral_bioavailability",
            "Observed",
            "direct_oral_bioavailability",
            "direct_outcome",
            "oral bioavailability",
        )
    if source_id == "hf_bioavailability":
        record = record or {}
        scope = str(
            record.get("canonical_bioavailability_evidence_scope") or ""
        ) or bioavailability_evidence_scope(
            record.get("bioavailability_report_type")
        )
        if scope == DIRECT_EVIDENCE_SCOPE:
            return FamilyAssignment(
                "Observed.direct_oral_bioavailability",
                "Observed",
                "direct_oral_bioavailability",
                "direct_outcome",
                "oral bioavailability",
            )
        return FamilyAssignment(
            "Observed.nondirect_oral_bioavailability",
            "Observed",
            "nondirect_oral_bioavailability",
            "surrogate_proxy",
            "relative or apparent oral bioavailability",
        )
    if source_id == "oral_exposure":
        paper_scope = str((record or {}).get("canonical_paper_direct_scope") or "")
        if paper_scope == "direct" or (
            not paper_scope
            and endpoint in {"bioavailability", "absolute_bioavailability"}
        ):
            return FamilyAssignment(
                "Observed.direct_oral_bioavailability",
                "Observed",
                "direct_oral_bioavailability",
                "direct_outcome",
                "oral bioavailability",
            )
        return FamilyAssignment(
            "Observed.oral_auc_cmax_exposure",
            "Observed",
            "oral_auc_cmax_exposure",
            "surrogate_proxy",
            "oral systemic exposure",
        )
    if source_id == "fa":
        return FamilyAssignment(
            "Fa.absorption_solubility_permeability",
            "Fa",
            "absorption_solubility_permeability",
            "mechanistic_factor",
            "Fa absorption, solubility, permeability, dissolution, and GI stability",
        )
    if source_id == "fg":
        return FamilyAssignment(
            "Fg.gut_wall_efflux_intestinal_metabolism",
            "Fg",
            "gut_wall_efflux_intestinal_metabolism",
            "mechanistic_factor",
            "Fg gut-wall escape, efflux, and intestinal metabolism",
        )
    if source_id == "fh":
        return FamilyAssignment(
            "Fh.hepatic_clearance_metabolic_stability",
            "Fh",
            "hepatic_clearance_metabolic_stability",
            "mechanistic_factor",
            "Fh hepatic clearance and metabolic stability",
        )
    return None


def validate_endpoint_inventory(source_id: str, endpoint_names: Iterable[str]) -> dict:
    """Validate the frozen source inventory and record only orthographic decisions."""
    values = sorted(set(endpoint_names))
    actual = {"count": len(values), "sha256": endpoint_inventory_hash(values)}
    expected = EXPECTED_ENDPOINT_INVENTORIES.get(source_id)
    if expected is None:
        raise ValueError(f"no frozen endpoint inventory for source {source_id!r}")
    if actual != expected:
        raise ValueError(
            f"endpoint inventory drift for {source_id}: expected {expected}, found {actual}"
        )
    decisions = [spacing_and_spelling_decision(source_id, value) for value in values]
    return {
        "source_id": source_id,
        **actual,
        "n_reviewed_corrections": sum(
            decision.status == "reviewed_correction" for decision in decisions
        ),
        "coverage": 1.0,
        "endpoints": [decision.to_dict() for decision in decisions],
    }


__all__ = [
    "ENDPOINT_CONCEPT_PATHS",
    "ENDPOINT_CONCEPT_VERSION",
    "EXPECTED_ENDPOINT_INVENTORIES",
    "SPACING_AND_SPELLING_VERSION",
    "SpacingAndSpellingDecision",
    "endpoint_concept",
    "family_assignment",
    "spacing_and_spelling_decision",
    "spacing_and_spelling_endpoint",
    "validate_endpoint_inventory",
]
