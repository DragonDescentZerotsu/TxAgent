from __future__ import annotations
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Any
from rdkit import Chem, rdBase
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize

IDENTITY_NORMALIZER_VERSION = "rdkit_fragment_parent.v2"

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
        status="ok" if parent is not None else "unresolved_parent",
    )

def _standardize_parent(mol: Chem.Mol, *, allow_charged_fallback: bool = True) -> Chem.Mol | None:
    try:
        clean = rdMolStandardize.Cleanup(mol)
        fragment_parent = rdMolStandardize.FragmentParent(clean)
        Chem.SanitizeMol(fragment_parent)
        try:
            uncharged = rdMolStandardize.Uncharger().uncharge(Chem.Mol(fragment_parent))
            Chem.SanitizeMol(uncharged)
            return uncharged
        except Exception:
            # Permanent hypervalent ions can be valid only in their charged form.
            return fragment_parent if allow_charged_fallback else None
    except Exception:  # noqa: BLE001 - invalid standardization should remain auditable.
        return None

def _component_parent_keys(mol: Chem.Mol) -> list[str]:
    keys: list[str] = []
    try:
        components = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except Exception:  # noqa: BLE001 - malformed mixtures should remain unresolved, not abort retrieval.
        return keys
    for component in components:
        # Do not introduce shared counterion matches for already-resolved salts.
        parent = _standardize_parent(component, allow_charged_fallback=False)
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

