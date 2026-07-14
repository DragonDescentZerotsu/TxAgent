"""Generic molecule-relation policies shared by all retrieval sources and tasks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from tools.chembl_tool.common.molecule_identity import MoleculeIdentity, identity_from_record


class MoleculeRelation(str, Enum):
    EXACT_RECORD = "exact_record"
    SAME_PARENT = "same_parent"
    SAME_CONNECTIVITY_VARIANT = "same_connectivity_variant"
    STRUCTURAL_ANALOG = "structural_analog"
    UNRESOLVED = "unresolved"


class NeighborIdentityPolicy(str, Enum):
    OPERATIONAL = "operational"
    PARENT_DISJOINT = "parent_disjoint"


NEIGHBOR_IDENTITY_POLICIES = tuple(policy.value for policy in NeighborIdentityPolicy)


@dataclass(frozen=True)
class CandidateDecision:
    relation: MoleculeRelation
    excluded: bool
    reason: str


def classify_molecule_relation(query: MoleculeIdentity, candidate: MoleculeIdentity) -> MoleculeRelation:
    """Classify identity overlap without treating covalent analogs as the same parent."""
    if query.status != "ok" or candidate.status != "ok":
        return MoleculeRelation.UNRESOLVED
    if _same_nonempty(query.standard_inchi_key, candidate.standard_inchi_key) or _same_nonempty(
        query.canonical_smiles, candidate.canonical_smiles
    ):
        return MoleculeRelation.EXACT_RECORD

    if _same_nonempty(_connectivity_key(query.standard_inchi_key), _connectivity_key(candidate.standard_inchi_key)):
        return MoleculeRelation.SAME_CONNECTIVITY_VARIANT

    query_components = set(query.component_parent_inchi_keys)
    candidate_components = set(candidate.component_parent_inchi_keys)
    if (
        _same_nonempty(query.parent_inchi_key, candidate.parent_inchi_key)
        or bool(query.parent_inchi_key and query.parent_inchi_key in candidate_components)
        or bool(candidate.parent_inchi_key and candidate.parent_inchi_key in query_components)
    ):
        return MoleculeRelation.SAME_PARENT
    return MoleculeRelation.STRUCTURAL_ANALOG


def decide_candidate(
    query: MoleculeIdentity,
    candidate_record: Mapping[str, Any],
    policy: str | NeighborIdentityPolicy,
) -> CandidateDecision:
    """Apply one reusable exclusion policy to a candidate molecule record."""
    normalized_policy = NeighborIdentityPolicy(policy)
    relation = classify_molecule_relation(query, identity_from_record(candidate_record))
    excluded_relations = {MoleculeRelation.EXACT_RECORD, MoleculeRelation.SAME_CONNECTIVITY_VARIANT}
    if normalized_policy is NeighborIdentityPolicy.PARENT_DISJOINT:
        excluded_relations.add(MoleculeRelation.SAME_PARENT)
    excluded = relation in excluded_relations
    return CandidateDecision(
        relation=relation,
        excluded=excluded,
        reason=f"{normalized_policy.value}:{relation.value}" if excluded else "retained",
    )


def policy_metadata(policy: str | NeighborIdentityPolicy) -> dict[str, Any]:
    normalized = NeighborIdentityPolicy(policy)
    return {
        "neighbor_identity_policy": normalized.value,
        "excluded_relations": (
            [
                MoleculeRelation.EXACT_RECORD.value,
                MoleculeRelation.SAME_CONNECTIVITY_VARIANT.value,
                MoleculeRelation.SAME_PARENT.value,
            ]
            if normalized is NeighborIdentityPolicy.PARENT_DISJOINT
            else [MoleculeRelation.EXACT_RECORD.value, MoleculeRelation.SAME_CONNECTIVITY_VARIANT.value]
        ),
        "threshold_preserving_backfill": True,
    }


def _same_nonempty(left: str, right: str) -> bool:
    return bool(left and right and left == right)


def _connectivity_key(inchi_key: str) -> str:
    return inchi_key.split("-", 1)[0] if inchi_key else ""
