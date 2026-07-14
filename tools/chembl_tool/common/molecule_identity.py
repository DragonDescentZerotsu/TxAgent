"""Source-independent molecule identity normalization for retrieval policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Mapping

from rdkit import Chem, rdBase
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize


IDENTITY_NORMALIZER_VERSION = "rdkit_fragment_parent.v1"

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
