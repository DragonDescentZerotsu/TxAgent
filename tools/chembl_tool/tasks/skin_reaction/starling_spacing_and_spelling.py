"""Spacing- and spelling-only endpoint normalization for Skin_Reaction records.

This layer may normalize whitespace and explicitly reviewed typographical errors.
It preserves case and must not merge scientifically related endpoint concepts.
Every entry in ``_REVIEWED_CORRECTIONS`` was read off the frozen source inventory
and is a typo, truncation, or separator variant of a value that already exists in
the same column; distinct dermatological concepts (``systemic_contact_dermatitis``,
``contact_urticaria``, ``anaphylaxis``, ``irritant``, ``photocontact``) are left
untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable

from tools.chembl_tool.common.starling.normalization.cleaning import (
    clean_text,
    endpoint_inventory_hash,
)
from tools.chembl_tool.common.starling.normalization.contracts import FamilyAssignment
from tools.chembl_tool.common.starling.normalization.measurements import EndpointOrthography
from tools.chembl_tool.tasks.skin_reaction.canonical_starling_source import (
    AOP_PARTITION,
    DIRECT_PARTITION,
    REJECT_PARTITION,
    partition_for_record,
)


SPACING_AND_SPELLING_VERSION = "skin_reaction_spacing_and_spelling.v1"

EXPECTED_ENDPOINT_INVENTORIES = {
    "direct_skin_reaction": {
        "count": 30,
        "sha256": "aba62d623303471553c59ff0df509c40570120b91a56bb3937a87c34f95704ee",
    },
    "sensitization_aop": {
        "count": 7349,
        "sha256": "1db864a92ac2fd927751541be3c35b5978a404dd4c5c11873ae566ac4a61ccc6",
    },
    "phototoxicity_irritation_local_damage": {
        "count": 115,
        "sha256": "612830311fc38d17947c74682619f61bfb5187549e6fa58f5c9533f41bbff650",
    },
    "skin_exposure": {
        "count": 34,
        "sha256": "107769ca1582a04c261a6596b9434837b438648a566a547922ccecabac05675d",
    },
}


# direct_skin_reaction / reaction_type: the extractor's controlled vocabulary with
# character-level corruption of the trailing "allergy" token, of "dermatitis", and
# of "urticarial".
# phototoxicity_irritation_local_damage / evidence_endpoint and
# skin_exposure / evidence_type: separator variants and truncations of the same
# controlled vocabulary.
# sensitization_aop / endpoint_or_target is free prose (7,349 distinct values) with
# no reviewable orthographic cluster, so it contributes no corrections.
_REVIEWED_CORRECTIONS = {
    # "allergic_contact_dermatitis_contact_allergy" — corrupted final token.
    "allergic_contact_dermatitis_contact_ally": "allergic_contact_dermatitis_contact_allergy",
    "allergic_contact_dermatitis_contact_alergy": "allergic_contact_dermatitis_contact_allergy",
    "allergic_contact_dermatitis_contact_addergy": "allergic_contact_dermatitis_contact_allergy",
    "allergic_contact_dermatitis_contact_allxiety": "allergic_contact_dermatitis_contact_allergy",
    "allergic_contact_dermatitis_contact_anergy": "allergic_contact_dermatitis_contact_allergy",
    "allergic_contact_dermatitis_contact_alleged": "allergic_contact_dermatitis_contact_allergy",
    "allergic_contact_dermatitis_contact_allity": "allergic_contact_dermatitis_contact_allergy",
    "allergic_contact_dermatitis_contact_alloy": "allergic_contact_dermatitis_contact_allergy",
    "allergic_contact_dermatitis_contact_allyergy": "allergic_contact_dermatitis_contact_allergy",
    "allergic_contact_dermatitis_contact_all allergy": "allergic_contact_dermatitis_contact_allergy",
    "allergic_contact_dermatitis/contact_allergy": "allergic_contact_dermatitis_contact_allergy",
    # "allergic" / "dermatitis" misspellings of the same controlled value.
    "alergic_contact_dermatitis_contact_allergy": "allergic_contact_dermatitis_contact_allergy",
    "allergenic_contact_dermatitis_contact_allergy": "allergic_contact_dermatitis_contact_allergy",
    "allergic_contact_dermatatitis_contact_allergy": "allergic_contact_dermatitis_contact_allergy",
    "allergic_contact_dermatatis_contact_allergy": "allergic_contact_dermatitis_contact_allergy",
    # "urticarial_wheal" — noun form and doubled vowel.
    "urticaria_wheal": "urticarial_wheal",
    "urticariial_wheal": "urticarial_wheal",
    # Source uses a non-breaking hyphen U+2011, which NFKC folds to U+2010.
    "cross‐reaction": "cross-reaction",
    # phototoxicity_irritation_local_damage separator variants and typos.
    "local tissue injury": "local_tissue_injury",
    "local_tissue injury": "local_tissue_injury",
    "inflammatory response": "inflammatory_response",
    "local inflammatory response": "local_inflammatory_response",
    "skin irritation": "skin_irritation",
    "keratinocyte damage": "keratinocyte_damage",
    "light-dependent cytotoxicity": "light_dependent_cytotoxicity",
    "light-dependent_cytotoxicity": "light_dependent_cytotoxicity",
    "light_dependent_cyttotoxicity": "light_dependent_cytotoxicity",
    "photooxicity_or_photosensitivity": "phototoxicity_or_photosensitivity",
    "photosensitivity_or_photosensitivity": "phototoxicity_or_photosensitivity",
    # skin_exposure separator variants and truncations.
    "dermally applied dose": "dermally_applied_dose",
    "skins_retention": "skin_retention",
    "skin_exposure_urement": "skin_exposure_measurement",
    "flx": "flux",
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


def family_assignment(
    source_id: str,
    endpoint_name: str,
    record: Mapping[str, Any] | None = None,
) -> FamilyAssignment | None:
    """Map source records to the four stable retrieval families without changing endpoints."""
    del endpoint_name
    record = record or {}
    if str(record.get("group_id") or "") == "Direct.skin_reaction":
        return FamilyAssignment(
            "Direct.skin_reaction",
            "Tier 1",
            "direct_skin_reaction",
            "direct_outcome",
            "skin reaction",
        )
    partition = str(record.get("canonical_sensitization_partition") or "")
    if (
        not partition
        and source_id in {"direct_skin_reaction", "sensitization_aop"}
        and record.get("source_row_number") is not None
    ):
        decision = partition_for_record(record)
        partition = decision.partition if decision else ""
    if partition == REJECT_PARTITION:
        return None
    if partition == DIRECT_PARTITION or (
        not partition and source_id == "direct_skin_reaction"
    ):
        return FamilyAssignment(
            "Direct.skin_reaction",
            "Tier 1",
            "direct_skin_reaction",
            "direct_outcome",
            "skin reaction",
        )
    if partition == AOP_PARTITION or (
        not partition and source_id == "sensitization_aop"
    ):
        return FamilyAssignment(
            "Mechanism.sensitization_aop",
            "Tier 2",
            "sensitisation_aop",
            "mechanistic_factor",
            "skin sensitization adverse outcome pathway",
        )
    if source_id == "phototoxicity_irritation_local_damage":
        return FamilyAssignment(
            "Mechanism.phototoxicity_irritation_local_damage",
            "Tier 3",
            "phototoxicity_irritation_local_damage",
            "mechanistic_factor",
            "phototoxicity, irritation, and local tissue damage",
        )
    if source_id == "skin_exposure":
        return FamilyAssignment(
            "Mechanism.skin_exposure",
            "Tier 4",
            "skin_exposure",
            "exposure_context",
            "skin permeation and dermal exposure",
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
    "EXPECTED_ENDPOINT_INVENTORIES",
    "SPACING_AND_SPELLING_VERSION",
    "SpacingAndSpellingDecision",
    "family_assignment",
    "spacing_and_spelling_decision",
    "spacing_and_spelling_endpoint",
    "validate_endpoint_inventory",
]
