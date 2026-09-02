"""Molecule normalization, exclusion, and candidate-selection policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from functools import lru_cache
from typing import Any

from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import AllChem, inchi
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize


IDENTITY_NORMALIZER_VERSION = "rdkit_fragment_parent.v1"
FP_RADIUS = 2
FP_BITS = 2048
SIMILARITY_SELECTOR = "similarity"
QUERY_FEATURE_COVERAGE_SELECTOR = "query_feature_coverage"
NEIGHBOR_SELECTORS = (SIMILARITY_SELECTOR, QUERY_FEATURE_COVERAGE_SELECTOR)
SELECTOR_VERSIONS = {
    SIMILARITY_SELECTOR: "similarity.v1",
    QUERY_FEATURE_COVERAGE_SELECTOR: "query_feature_coverage.v1",
}

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


class MoleculeRelation(str, Enum):
    EXACT_RECORD = "exact_record"
    SAME_PARENT = "same_parent"
    SAME_CONNECTIVITY_VARIANT = "same_connectivity_variant"
    SAME_SCAFFOLD = "same_scaffold"
    STRUCTURAL_ANALOG = "structural_analog"
    UNRESOLVED = "unresolved"


class NeighborIdentityPolicy(str, Enum):
    OPERATIONAL = "operational"
    PARENT_DISJOINT = "parent_disjoint"
    SCAFFOLD_DISJOINT = "scaffold_disjoint"


NEIGHBOR_IDENTITY_POLICIES = tuple(policy.value for policy in NeighborIdentityPolicy)


@dataclass(frozen=True)
class MoleculeIdentity:
    """Normalized whole-record and active-parent identifiers for one SMILES."""

    input_smiles: str
    canonical_smiles: str = ""
    standard_inchi_key: str = ""
    parent_smiles: str = ""
    parent_inchi_key: str = ""
    parent_connectivity_key: str = ""
    component_parent_inchi_keys: tuple[str, ...] = ()
    status: str = "ok"
    normalizer_version: str = IDENTITY_NORMALIZER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateDecision:
    relation: MoleculeRelation
    excluded: bool
    reason: str


@dataclass(frozen=True)
class NeighborCandidate:
    """Minimal selector-facing candidate contract."""

    molecule_index: int
    molecule_id: str
    similarity: float


@lru_cache(maxsize=200_000)
def normalize_molecule_identity(smiles: str) -> MoleculeIdentity:
    """Return deterministic whole-record and parent identities for a SMILES."""
    with rdBase.BlockLogs():
        return _normalize_molecule_identity(smiles)


def _normalize_molecule_identity(smiles: str) -> MoleculeIdentity:
    input_smiles = str(smiles or "").strip()
    mol = Chem.MolFromSmiles(input_smiles) if input_smiles else None
    if mol is None:
        return MoleculeIdentity(input_smiles=input_smiles, status="invalid_smiles")

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    standard_inchi_key = _inchi_key(mol)
    parent = _standardize_parent(mol)
    parent_smiles = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True) if parent is not None else ""
    parent_inchi_key = _inchi_key(parent)
    component_keys = tuple(sorted(set(_component_parent_keys(mol))))
    if parent_inchi_key and parent_inchi_key not in component_keys:
        component_keys = tuple(sorted((*component_keys, parent_inchi_key)))
    return MoleculeIdentity(
        input_smiles=input_smiles,
        canonical_smiles=canonical_smiles,
        standard_inchi_key=standard_inchi_key,
        parent_smiles=parent_smiles,
        parent_inchi_key=parent_inchi_key,
        parent_connectivity_key=_connectivity_key(parent_inchi_key),
        component_parent_inchi_keys=component_keys,
    )


def identity_from_record(record: Mapping[str, Any]) -> MoleculeIdentity:
    """Use stored identity metadata when available, otherwise normalize its SMILES."""
    stored = record.get("molecule_identity")
    if isinstance(stored, Mapping) and stored.get("normalizer_version") == IDENTITY_NORMALIZER_VERSION:
        return MoleculeIdentity(
            input_smiles=str(stored.get("input_smiles") or record.get("canonical_smiles") or ""),
            canonical_smiles=str(stored.get("canonical_smiles") or ""),
            standard_inchi_key=str(stored.get("standard_inchi_key") or ""),
            parent_smiles=str(stored.get("parent_smiles") or ""),
            parent_inchi_key=str(stored.get("parent_inchi_key") or ""),
            parent_connectivity_key=str(stored.get("parent_connectivity_key") or ""),
            component_parent_inchi_keys=tuple(stored.get("component_parent_inchi_keys") or ()),
            status=str(stored.get("status") or "ok"),
            normalizer_version=IDENTITY_NORMALIZER_VERSION,
        )
    return normalize_molecule_identity(str(record.get("canonical_smiles") or record.get("smiles") or ""))


@lru_cache(maxsize=200_000)
def bemis_murcko_scaffold(smiles: str) -> str:
    """Return a canonical Bemis-Murcko scaffold; acyclic molecules map to empty."""
    molecule = Chem.MolFromSmiles(str(smiles or ""))
    if molecule is None:
        raise ValueError(f"cannot calculate scaffold for invalid SMILES: {smiles}")
    with rdBase.BlockLogs():
        try:
            return _murcko_scaffold_smiles(molecule)
        except RuntimeError:
            # Some valid source molecules carry inconsistent double-bond stereo.
            # Stereo is outside this non-chiral scaffold contract, so remove it
            # and retry on the same molecular graph instead of losing the molecule.
            without_stereo = Chem.Mol(molecule)
            Chem.RemoveStereochemistry(without_stereo)
            return _murcko_scaffold_smiles(without_stereo)


def _murcko_scaffold_smiles(molecule: Chem.Mol) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule,
        includeChirality=False,
    )


def _standardize_parent(mol: Chem.Mol) -> Chem.Mol | None:
    try:
        clean = rdMolStandardize.Cleanup(mol)
        fragment_parent = rdMolStandardize.FragmentParent(clean)
        uncharged = rdMolStandardize.Uncharger().uncharge(fragment_parent)
        Chem.SanitizeMol(uncharged)
        return uncharged
    except Exception:  # noqa: BLE001 - invalid standardization should remain auditable.
        return None


def _component_parent_keys(mol: Chem.Mol) -> list[str]:
    keys: list[str] = []
    try:
        components = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except Exception:  # noqa: BLE001 - malformed mixtures should remain unresolved, not abort retrieval.
        return keys
    for component in components:
        parent = _standardize_parent(component)
        key = _inchi_key(parent)
        if key:
            keys.append(key)
    return keys


def _inchi_key(mol: Chem.Mol | None) -> str:
    if mol is None:
        return ""
    try:
        return str(inchi.MolToInchiKey(mol) or "")
    except Exception:  # noqa: BLE001 - InChI support can fail for unusual records.
        return ""


def _connectivity_key(inchi_key: str) -> str:
    return inchi_key.split("-", 1)[0] if inchi_key else ""


def standardize_smiles_and_fp(
    smiles: str,
) -> tuple[str, str, DataStructs.ExplicitBitVect | None]:
    """Return canonical SMILES, InChIKey, and the retrieval fingerprint."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return "", "", None
    canonical_smiles, inchi_key = _standardize_molecule(molecule)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        FP_RADIUS,
        nBits=FP_BITS,
        useFeatures=False,
        useChirality=False,
        useBondTypes=True,
    )
    return canonical_smiles, inchi_key, fingerprint


def standardize_smiles(smiles: str) -> tuple[str, str]:
    molecule = Chem.MolFromSmiles(smiles)
    return _standardize_molecule(molecule) if molecule is not None else ("", "")


def _standardize_molecule(molecule: Chem.Mol) -> tuple[str, str]:
    canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
    try:
        inchi_key = inchi.MolToInchiKey(molecule)
    except Exception:
        inchi_key = ""
    return canonical_smiles, inchi_key


def classify_molecule_relation(
    query: MoleculeIdentity, candidate: MoleculeIdentity
) -> MoleculeRelation:
    """Classify identity overlap without treating covalent analogs as one parent."""
    if query.status != "ok" or candidate.status != "ok":
        return MoleculeRelation.UNRESOLVED
    if _same_nonempty(
        query.standard_inchi_key, candidate.standard_inchi_key
    ) or _same_nonempty(query.canonical_smiles, candidate.canonical_smiles):
        return MoleculeRelation.EXACT_RECORD
    if _same_nonempty(
        _connectivity_key(query.standard_inchi_key),
        _connectivity_key(candidate.standard_inchi_key),
    ):
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
    """Apply one exclusion policy to a candidate molecule record."""
    normalized_policy = NeighborIdentityPolicy(policy)
    candidate = identity_from_record(candidate_record)
    relation = classify_molecule_relation(query, candidate)
    if (
        normalized_policy is NeighborIdentityPolicy.SCAFFOLD_DISJOINT
        and relation is MoleculeRelation.STRUCTURAL_ANALOG
        and _same_nonempty_scaffold(query, candidate)
    ):
        relation = MoleculeRelation.SAME_SCAFFOLD
    excluded = relation.value in _excluded_relation_values(normalized_policy)
    return CandidateDecision(
        relation=relation,
        excluded=excluded,
        reason=f"{normalized_policy.value}:{relation.value}" if excluded else "retained",
    )


def policy_metadata(policy: str | NeighborIdentityPolicy) -> dict[str, Any]:
    normalized = NeighborIdentityPolicy(policy)
    return {
        "neighbor_identity_policy": normalized.value,
        "excluded_relations": _excluded_relation_values(normalized),
        "threshold_preserving_backfill": True,
    }


def _excluded_relation_values(policy: NeighborIdentityPolicy) -> list[str]:
    relations = [MoleculeRelation.EXACT_RECORD, MoleculeRelation.SAME_CONNECTIVITY_VARIANT]
    if policy in {
        NeighborIdentityPolicy.PARENT_DISJOINT,
        NeighborIdentityPolicy.SCAFFOLD_DISJOINT,
    }:
        relations.append(MoleculeRelation.SAME_PARENT)
    if policy is NeighborIdentityPolicy.SCAFFOLD_DISJOINT:
        relations.append(MoleculeRelation.SAME_SCAFFOLD)
    return [relation.value for relation in relations]


def _same_nonempty_scaffold(
    query: MoleculeIdentity, candidate: MoleculeIdentity
) -> bool:
    return bool(
        query.parent_smiles
        and candidate.parent_smiles
        and bemis_murcko_scaffold(query.parent_smiles)
        == bemis_murcko_scaffold(candidate.parent_smiles)
    )


def _same_nonempty(left: str, right: str) -> bool:
    return bool(left and right and left == right)


def select_neighbor_candidates(
    candidates: Sequence[NeighborCandidate],
    *,
    query_fingerprint: Any,
    candidate_fingerprints: Sequence[Any],
    top_k: int,
    selector: str = SIMILARITY_SELECTOR,
) -> list[NeighborCandidate]:
    """Order already eligible candidates by similarity or query-bit coverage."""
    _validate_selector(selector)
    if top_k <= 0 or not candidates:
        return []
    if selector == SIMILARITY_SELECTOR:
        return sorted(candidates, key=_similarity_sort_key)[:top_k]
    return _select_query_feature_coverage(
        candidates,
        query_fingerprint=query_fingerprint,
        candidate_fingerprints=candidate_fingerprints,
        top_k=top_k,
    )


def selector_metadata(selector: str) -> dict[str, Any]:
    _validate_selector(selector)
    metadata: dict[str, Any] = {
        "name": selector,
        "version": SELECTOR_VERSIONS[selector],
    }
    if selector == QUERY_FEATURE_COVERAGE_SELECTOR:
        metadata.update(
            {
                "objective": "greedy_marginal_query_morgan_bit_coverage",
                "similarity_role": "eligibility_threshold_and_tie_break",
                "rank1_forced": False,
            }
        )
    return metadata


def _select_query_feature_coverage(
    candidates: Sequence[NeighborCandidate],
    *,
    query_fingerprint: Any,
    candidate_fingerprints: Sequence[Any],
    top_k: int,
) -> list[NeighborCandidate]:
    query_bits = frozenset(int(bit) for bit in query_fingerprint.GetOnBits())
    if not query_bits:
        return sorted(candidates, key=_similarity_sort_key)[:top_k]
    shared_bits = {
        candidate.molecule_index: frozenset(
            int(bit)
            for bit in candidate_fingerprints[candidate.molecule_index].GetOnBits()
            if int(bit) in query_bits
        )
        for candidate in candidates
    }
    remaining, selected, covered = list(candidates), [], set()
    while remaining and len(selected) < top_k:
        choice = min(
            remaining,
            key=lambda candidate: (
                -len(shared_bits[candidate.molecule_index] - covered),
                -candidate.similarity,
                candidate.molecule_id,
                candidate.molecule_index,
            ),
        )
        selected.append(choice)
        covered.update(shared_bits[choice.molecule_index])
        remaining.remove(choice)
    return selected


def _similarity_sort_key(candidate: NeighborCandidate) -> tuple[float, str, int]:
    return (-candidate.similarity, candidate.molecule_id, candidate.molecule_index)


def _validate_selector(selector: str) -> None:
    if selector not in NEIGHBOR_SELECTORS:
        raise ValueError(
            f"Unknown neighbor selector {selector!r}; expected one of "
            f"{', '.join(NEIGHBOR_SELECTORS)}"
        )
