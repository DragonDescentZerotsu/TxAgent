"""Opt-in tautomer identity for DILI/Carcinogens; existing tasks are unchanged.

Gold retains stable stereochemistry. A separate stereo-free tautomer identity is
used only for split/retrieval leakage grouping, never to merge gold labels.
"""
from functools import lru_cache
import hashlib

from rdkit import Chem, rdBase
from rdkit.Chem import inchi, rdMolHash, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity

VERSION = 'new_task_tautomer_identity.v2'
TASKS = ('dili', 'carcinogens')


def _enumerator():
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(10000)
    enumerator.SetMaxTransforms(10000)
    enumerator.SetRemoveSp3Stereo(False)
    # Only bonds participating in tautomerism lose their obsolete stereo tags;
    # stable E/Z double bonds remain distinct in the gold identity.
    enumerator.SetRemoveBondStereo(True)
    enumerator.SetReassignStereo(True)
    return enumerator


def _canonicalize(molecule):
    enumerator = _enumerator()
    result = enumerator.Enumerate(molecule)
    complete = str(result.status) == 'Completed'
    # Never select a purported canonical tautomer from a truncated enumeration.
    chosen = enumerator.PickCanonical(result) if complete else Chem.Mol(molecule)
    return chosen, {'status': str(result.status), 'tautomers': len(result), 'canonicalization_applied': complete}


def _tautomer_hash(molecule):
    no_stereo = Chem.Mol(molecule)
    Chem.RemoveStereochemistry(no_stereo)
    value = rdMolHash.MolHash(no_stereo, rdMolHash.HashFunction.HetAtomTautomer)
    return 'HTV1_' + hashlib.sha256(value.encode()).hexdigest(), value


def _scaffold_core(parent):
    # Murcko retains exocyclic double-bond atoms (e.g. =O), whose presence can
    # depend on the tautomer. Prune all terminal atoms from that scaffold to get
    # a conservative ring/linker core; atom elements and connectivity remain.
    core = MurckoScaffold.GetScaffoldForMol(parent)
    Chem.RemoveStereochemistry(core)
    while True:
        terminal = [a.GetIdx() for a in core.GetAtoms() if a.GetDegree() == 1]
        if not terminal:
            break
        editable = Chem.RWMol(core)
        for index in sorted(terminal, reverse=True):
            editable.RemoveAtom(index)
        core = editable.GetMol()
    return core


@lru_cache(maxsize=200_000)
def leakage_identity(task: str, smiles: str) -> dict:
    """Fast conservative split/retrieval grouping; no tautomer enumeration."""
    if task not in TASKS:
        raise ValueError(f'Tautomer identity is restricted to {TASKS}: {task}')
    old = normalize_molecule_identity(smiles)
    if not old.parent_smiles:
        raise ValueError(f'Invalid parent identity: {smiles}')
    parent = Chem.MolFromSmiles(old.parent_smiles)
    graph_parent = Chem.Mol(parent)
    Chem.RemoveStereochemistry(graph_parent)
    graph_hash = rdMolHash.MolHash(graph_parent, rdMolHash.HashFunction.ElementGraph)
    formula = rdMolDescriptors.CalcMolFormula(parent, separateIsotopes=True)
    charge = Chem.GetFormalCharge(parent)
    key = 'EGFC_' + hashlib.sha256(f'{graph_hash}|{formula}|{charge}'.encode()).hexdigest()
    core = _scaffold_core(parent)
    scaffold_hash = rdMolHash.MolHash(core, rdMolHash.HashFunction.ElementGraph)
    scaffold_key = 'MURCKO_TOPOLOGY_' + hashlib.sha256(scaffold_hash.encode()).hexdigest()
    return {'leakage_group': key, 'leakage_tautomer_hash': graph_hash, 'parent_formula': formula, 'parent_net_charge': charge,
            'scaffold_leakage_group': scaffold_key if core.GetNumAtoms() else '',
            'scaffold_tautomer_hash': scaffold_hash,
            'bemis_murcko_scaffold': MurckoScaffold.MurckoScaffoldSmiles(mol=parent, includeChirality=False),
            'scaffold_group_smiles': scaffold_hash,
            'source_parent_smiles': old.parent_smiles,
            'identity_contract': VERSION, 'leakage_hash_policy': 'ElementGraph_plus_isotope_separated_formula_and_net_charge.v1',
            'scaffold_group_policy': 'terminal_pruned_Murcko_ring_linker_ElementGraph.v1'}


def _stereo_counts(molecule):
    return {
        'specified_tetrahedral_centers': sum(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in molecule.GetAtoms()),
        'specified_double_bonds': sum(b.GetStereo() not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY) for b in molecule.GetBonds()),
    }


@lru_cache(maxsize=200_000)
def normalize_new_task_identity(task: str, smiles: str) -> dict:
    if task not in TASKS:
        raise ValueError(f'Tautomer identity is restricted to {TASKS}: {task}')
    old = normalize_molecule_identity(smiles)
    if not old.parent_smiles or not old.parent_inchi_key:
        raise ValueError(f'Invalid parent identity: {smiles}')
    return _normalize_parent(old.parent_smiles, old.parent_inchi_key)


@lru_cache(maxsize=200_000)
def _normalize_parent(parent_smiles, parent_key):
    with rdBase.BlockLogs():
        parent = Chem.MolFromSmiles(parent_smiles)
        gold, gold_enumeration = _canonicalize(parent)
        before, after = _stereo_counts(parent), _stereo_counts(gold)
        grouping = leakage_identity('dili', parent_smiles)
        return {
            'identity_contract': VERSION,
            'leakage_hash_policy': 'ElementGraph_plus_isotope_separated_formula_and_net_charge.v1',
            'rdkit_version': rdBase.rdkitVersion,
            'enumeration': {'gold': gold_enumeration},
            'max_tautomers': 10000, 'max_transforms': 10000,
            'source_parent_smiles': parent_smiles,
            'source_parent_inchi_key': parent_key,
            'drug': Chem.MolToSmiles(gold, canonical=True, isomericSmiles=True),
            'molecule_identity_key': inchi.MolToInchiKey(gold),
            **grouping,
            'stereo_before': before, 'stereo_after': after,
            'tautomer_stereo_changed': before != after,
            'gold_identity_policy': 'complete_stereo_preserving_tautomer' if gold_enumeration['status'] == 'Completed' else 'retain_verified_old_parent_noncomplete_enumeration',
        }
