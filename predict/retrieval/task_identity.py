"""Runtime refinement of frozen DILI/Carcinogens leakage groups.

Only explicitly declared legacy indices use these rules. Benchmark split and
Gold construction are not modified; incomplete enumeration remains excluded.
"""
from functools import lru_cache
import hashlib

from rdkit import Chem
from rdkit.Chem import rdMolHash, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from predict.retrieval.policies import normalize_molecule_identity, CandidateDecision, MoleculeRelation

VERSION = 'new_task_tautomer_identity.v2'
TASKS = ('dili', 'carcinogens')
PARENT_REFINEMENT_POLICY = 'complete_stereo_free_tautomer_within_legacy_group.v1'


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


def validate_contract(task, contract):
    if task not in TASKS or contract != VERSION:
        raise ValueError(f'Unsupported opt-in retrieval identity: {task}/{contract}')


@lru_cache(maxsize=16384)
def _canonical_parent(smiles):
    """Return a completed stereo-free tautomer representative, or unresolved.

    Only queried for matching legacy parent groups, never during index builds.
    Incomplete enumeration must not release a potentially equivalent candidate.
    """
    if not smiles:
        return None
    try:
        parent = normalize_molecule_identity(smiles).parent_smiles
        if not parent:
            return None
        mol = Chem.MolFromSmiles(parent)
        Chem.RemoveStereochemistry(mol)
        canonical, info = _canonicalize(mol)
        if info['status'] != 'Completed':
            return None
        return Chem.MolToSmiles(canonical, canonical=True, isomericSmiles=True)
    except (ValueError, RuntimeError):
        return None


def decide_candidate(query, candidate, policy, contract):
    """Refine frozen parent groups; the caller owns the shared Murcko check."""
    if candidate.get('retrieval_identity_contract') != contract:
        raise ValueError('Missing or mixed candidate retrieval identity contract')
    if policy not in {'parent_disjoint', 'scaffold_disjoint'}:
        raise ValueError('Tautomer-aware indices require parent_disjoint or scaffold_disjoint')
    if 'retrieval_scaffold_leakage_group' not in candidate:
        raise ValueError('Candidate lacks a scaffold leakage identity')
    if not candidate.get('retrieval_leakage_group'):
        raise ValueError('Candidate lacks a reliable leakage identity')
    same_parent = query['leakage_group'] == candidate['retrieval_leakage_group']
    if same_parent:
        query_parent = _canonical_parent(query.get('source_parent_smiles'))
        candidate_parent = _canonical_parent(
            candidate.get('retrieval_identity_source_smiles')
            or (candidate.get('molecule_identity') or {}).get('parent_smiles')
            or candidate.get('canonical_smiles'))
        if query_parent is None or candidate_parent is None:
            return CandidateDecision(MoleculeRelation.SAME_PARENT, True, policy + ':unresolved_legacy_parent_group')
        if query_parent == candidate_parent:
            return CandidateDecision(MoleculeRelation.SAME_PARENT, True, policy + ':same_completed_tautomer_parent')
    return CandidateDecision(MoleculeRelation.STRUCTURAL_ANALOG, False, 'retained')
