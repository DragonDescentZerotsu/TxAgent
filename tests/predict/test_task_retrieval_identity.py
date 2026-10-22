"""Approved local identity rules through shared and branch retrieval entrypoints."""
import pytest
from rdkit import Chem

from data.processing.evidence_library.index_builder import build_neighbor_index
from predict.retrieval import task_identity as identity
from predict.retrieval.policies import decide_candidate, normalize_molecule_identity, policy_metadata
from predict.retrieval.retrieve import retrieve_neighbors
from predict.harnesses.branches.retrieval import BranchDefinition, retrieve_branches

SOURCE = {"retrieval_identity_task": "dili", "retrieval_identity_contract": identity.VERSION}


def candidate(smiles):
    group = identity.leakage_identity('dili', smiles)
    return dict(canonical_smiles=smiles, retrieval_identity_source_smiles=smiles,
                retrieval_identity_contract=identity.VERSION,
                retrieval_leakage_group=group['leakage_group'],
                retrieval_scaffold_leakage_group=group['scaffold_leakage_group'])


@pytest.mark.parametrize('task', identity.TASKS)
@pytest.mark.parametrize('left,right,excluded', [
    ('C=CCC', 'CC=CC', False),
    ('C=C=CC', 'C=CC=C', False),
    ('CC(=O)C', 'C=C(C)O', True),
    ('O=c1cccc[nH]1', 'Oc1ccccn1', True),
    ('CN(C)/N=N/c1[nH]cnc1C(N)=O', 'CN(C)NN=C1N=CN=C1C(N)=O', True),
    ('C[C@H](O)C(=O)O', 'C[C@@H](O)C(=O)[O-].[Na+]', True),
])
def test_refined_parent_preserves_approved_pairs(task, left, right, excluded):
    source = {**SOURCE, 'retrieval_identity_task': task}
    for query, other in [(left, right), (right, left)]:
        assert identity.leakage_identity(task, query)['leakage_group'] == candidate(other)['retrieval_leakage_group']
        assert decide_candidate(normalize_molecule_identity(query), candidate(other),
                                'parent_disjoint', source=source).excluded is excluded


@pytest.mark.parametrize('target', ['C=CCC', 'CC=CC'])
def test_incomplete_enumeration_stays_excluded(target, monkeypatch):
    original = identity._canonicalize
    def incomplete(mol):
        if Chem.MolToSmiles(mol) == target:
            return mol, {'status': 'MaxTransformsReached'}
        return original(mol)
    identity._canonical_parent.cache_clear()
    monkeypatch.setattr(identity, '_canonicalize', incomplete)
    try:
        decision = decide_candidate(normalize_molecule_identity('C=CCC'), candidate('CC=CC'),
                                    'parent_disjoint', source=SOURCE)
        assert decision.excluded and decision.reason.endswith(':unresolved_legacy_parent_group')
    finally:
        identity._canonical_parent.cache_clear()


def test_original_structure_precedes_display_and_missing_structure_is_conservative():
    query = identity.leakage_identity('dili', 'C=CCC')
    row = candidate('CC=CC')
    row['canonical_smiles'] = 'C=CCC'
    assert not identity.decide_candidate(query, row, 'parent_disjoint', identity.VERSION).excluded
    row.pop('retrieval_identity_source_smiles')
    row.pop('canonical_smiles')
    assert identity.decide_candidate(query, row, 'parent_disjoint', identity.VERSION).excluded


def test_scaffold_rule_stays_topological_and_skips_enumeration(monkeypatch):
    monkeypatch.setattr(identity, '_canonical_parent', lambda _: pytest.fail('Already scaffold-excluded'))
    decision = decide_candidate(normalize_molecule_identity('c1ccccc1'), candidate('C1CCCCC1'),
                                'scaffold_disjoint', source=SOURCE)
    assert decision.excluded and decision.reason.endswith(':same_tautomer_scaffold_group')
    assert not decide_candidate(normalize_molecule_identity('c1ccccc1'), candidate('C1CCCCC1'),
                                'parent_disjoint', source=SOURCE).excluded


@pytest.mark.parametrize('broken', ['contract', 'task', 'candidate', 'scaffold', 'group', 'source', 'policy'])
def test_incompatible_contracts_fail_before_selection(broken):
    source, row, policy = dict(SOURCE), candidate('CC=CC'), 'parent_disjoint'
    if broken == 'contract': source['retrieval_identity_contract'] = 'unknown'
    if broken == 'task': source['retrieval_identity_task'] = 'ames'
    if broken == 'candidate': row.pop('retrieval_identity_contract')
    if broken == 'scaffold': row.pop('retrieval_scaffold_leakage_group')
    if broken == 'group': row['retrieval_leakage_group'] = ''
    if broken == 'source': source = {}
    if broken == 'policy': policy = 'operational'
    with pytest.raises(ValueError):
        decide_candidate(normalize_molecule_identity('C=CCC'), row, policy, source=source)
    if broken in {'contract', 'task', 'policy'}:
        with pytest.raises(ValueError):
            policy_metadata(policy, source=source)


@pytest.mark.parametrize('path', ['indexed', 'branches', 'reranked'])
@pytest.mark.parametrize('query,other,policy,expected', [
    ('C=CCC', 'CC=CC', 'parent_disjoint', 'isomer'),
    ('C=CCC', 'CC=CC', 'scaffold_disjoint', 'isomer'),
    ('c1ccccc1', 'C1CCCCC1', 'scaffold_disjoint', 'eligible'),
    ('CC(=O)C', 'C=C(C)O', 'parent_disjoint', 'eligible'),
])
def test_runtime_filters_before_top_k(path, query, other, policy, expected):
    rows = [dict(molecule_chembl_id=mid, canonical_smiles=smiles, group_id='Assay.test',
                 assay_tier='Assay', endpoint_group='test', standard_type='endpoint',
                 standard_value=1, standard_units='unit')
            for mid, smiles in [('same', query), ('isomer', other), ('eligible', 'c1ccncc1')]]
    index = build_neighbor_index(rows, index_version='test', workers=1)
    index['source'] = dict(SOURCE)
    for molecule in index['molecules']:
        molecule.update(candidate(molecule['canonical_smiles']))
    kwargs = dict(top_k_per_group=1, min_similarity=0, neighbor_identity_policy=policy)
    if path == 'indexed':
        result = retrieve_neighbors(query, index, **kwargs)
    else:
        class Reranker:
            name = "fixture"
            def provenance(self):
                return {'candidate_contract': 'tanimoto_identity_exclusion_then_eligible_pool.v1'}
            def rerank_records(self, **kwargs):
                return kwargs['candidates']
            def rerank(self, **kwargs):
                return kwargs['candidates']
        result = retrieve_branches(query, index, mode='full_flat', source_name='fixture',
            branch_definitions=(BranchDefinition('Assay.test', 'Assay', 'test', source_groups=('Assay.test',)),),
            reranker=Reranker() if path == 'reranked' else None, **kwargs)
    assert [n['molecule_chembl_id'] for n in result['groups'][0]['neighbors']] == [expected]
    assert result['retrieval_policy']['retrieval_parent_refinement_policy'] == identity.PARENT_REFINEMENT_POLICY


@pytest.mark.parametrize('path', ['indexed', 'branches'])
def test_monatomic_fingerprint_collision_cannot_fill_neighbor_budget(path):
    from predict.retrieval.features import STRUCTURAL_ELIGIBILITY_VERSION
    rows = [dict(molecule_chembl_id=mid, canonical_smiles=smiles, group_id='Assay.test',
                 assay_tier='Assay', endpoint_group='test', standard_type='endpoint',
                 standard_value=1, standard_units='unit')
            for mid, smiles in [('collision', '[U]'), ('eligible', 'C[Pb](C)(C)C')]]
    index = build_neighbor_index(rows, index_version='test', workers=1)
    kwargs = dict(top_k_per_group=1, min_similarity=0, neighbor_identity_policy='parent_disjoint')
    if path == 'indexed':
        result = retrieve_neighbors('[Pb]', index, **kwargs)
    else:
        result = retrieve_branches('[Pb]', index, mode='full_flat', source_name='fixture',
            branch_definitions=(BranchDefinition('Assay.test', 'Assay', 'test', source_groups=('Assay.test',)),),
            **kwargs)
    assert [n['molecule_chembl_id'] for n in result['groups'][0]['neighbors']] == ['eligible']
    assert result['retrieval_policy']['structural_eligibility']['version'] == STRUCTURAL_ELIGIBILITY_VERSION
