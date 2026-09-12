from copy import deepcopy

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolHash

from tools.chembl_tool.common.starling import new_task_identity as identity
from tools.chembl_tool.common.starling.build_new_task_tautomer_repair import regroup_votes

DACARBAZINE = ('CN(C)/N=N/c1[nH]cnc1C(N)=O', 'CN(C)NN=C1N=CN=C1C(N)=O')


def test_dacarbazine_tautomers_share_gold_leakage_and_scaffold_groups():
    rows = [identity.normalize_new_task_identity('dili', s) for s in DACARBAZINE]
    for field in ('drug', 'molecule_identity_key', 'leakage_group', 'scaffold_leakage_group'):
        assert rows[0][field] == rows[1][field]
    assert rows[0]['tautomer_stereo_changed']
    assert rows[0]['enumeration']['gold']['status'] == 'Completed'
    # v2 is not assumed superior: it misses this actual benchmark leak.
    hashes = []
    for smiles in DACARBAZINE:
        molecule = Chem.MolFromSmiles(smiles)
        Chem.RemoveStereochemistry(molecule)
        hashes.append(rdMolHash.MolHash(molecule, rdMolHash.HashFunction.HetAtomTautomerv2))
    assert hashes[0] != hashes[1]


@pytest.mark.parametrize('smiles', [('C[C@H](O)C(=O)O', 'C[C@@H](O)C(=O)O'), ('C/C=C/C', 'C/C=C\\C')])
def test_stable_stereo_distinct_gold_but_same_leakage_group(smiles):
    a, b = [identity.normalize_new_task_identity('carcinogens', s) for s in smiles]
    assert a['molecule_identity_key'] != b['molecule_identity_key']
    assert a['leakage_group'] == b['leakage_group']
    assert not a['tautomer_stereo_changed'] and not b['tautomer_stereo_changed']


def test_scaffold_topology_is_conservative_without_collapsing_gold():
    benzene, cyclohexane = [identity.normalize_new_task_identity('dili', s) for s in ('c1ccccc1', 'C1CCCCC1')]
    assert benzene['scaffold_leakage_group'] == cyclohexane['scaffold_leakage_group']
    assert benzene['leakage_group'] != cyclohexane['leakage_group']
    assert benzene['molecule_identity_key'] != cyclohexane['molecule_identity_key']
    a, b = [identity.leakage_identity('dili', s) for s in ('O=c1cccc[nH]1', 'Oc1ccccn1')]
    assert a['scaffold_leakage_group'] == b['scaffold_leakage_group']
    assert a['leakage_group'] == b['leakage_group']


def test_fast_leakage_has_no_enumerator_dependency(monkeypatch):
    identity.leakage_identity.cache_clear()
    monkeypatch.setattr(identity, '_enumerator', lambda: pytest.fail('leakage grouping must not enumerate'))
    assert identity.leakage_identity('dili', DACARBAZINE[0])['leakage_group'] == identity.leakage_identity('dili', DACARBAZINE[1])['leakage_group']


def test_noncomplete_enumeration_retains_original_graph(monkeypatch):
    class Truncated:
        status = 'MaxTransformsReached'
        def __len__(self): return 100
    class Enumerator:
        def Enumerate(self, molecule): return Truncated()
        def PickCanonical(self, result): pytest.fail('must not select a truncated canonical identity')
    monkeypatch.setattr(identity, '_enumerator', Enumerator)
    molecule = Chem.MolFromSmiles('C/C=C/C')
    retained, status = identity._canonicalize(molecule)
    assert Chem.MolToSmiles(retained) == Chem.MolToSmiles(molecule)
    assert status['canonicalization_applied'] is False


def _votes():
    rows = []
    for index, smiles in enumerate(DACARBAZINE):
        rows.append({'source_record_id': f'uid{index}', 'source_row_uid': f'uid{index}', 'drug': smiles,
                     'molecule_identity_key': f'old{index}', 'bemis_murcko_scaffold': f'oldscaffold{index}',
                     'study_id': 'pmid:123', 'condition_group': 'population=human', 'pmid': '123', 'Y': 1,
                     'reviewer': 'codex_record_semantic_review', 'raw_record': {'SMILES': smiles},
                     'study_source_row_uids': [f'uid{index}', f'duplicate{index}']})
    return rows


def test_tautomer_equivalence_cannot_double_count_one_study_or_resolve_conflict():
    rows = _votes()
    original = deepcopy(rows)
    mapping = {s: identity.normalize_new_task_identity('dili', s) for s in DACARBAZINE}
    votes, conflicts, merged = regroup_votes(rows, mapping)
    assert rows == original
    assert len(votes) == 1 and not conflicts and len(merged) == 1
    assert votes[0]['identity_merged_vote_uids'] == ['uid0', 'uid1']
    assert votes[0]['raw_record'] == rows[0]['raw_record']
    rows[1]['Y'] = 0
    votes, conflicts, merged = regroup_votes(rows, mapping)
    assert not votes and len(conflicts) == 1 and not merged
    assert len(conflicts[0]['old_source_votes']) == 2


def test_adapter_cannot_change_existing_five_tasks():
    for task in ('ames', 'bbb_martins', 'bioavailability_ma', 'skin_reaction', 'clintox'):
        with pytest.raises(ValueError, match='restricted'):
            identity.normalize_new_task_identity(task, 'CCO')



def test_element_formula_group_catches_keto_enol_without_hydrogenation_merge():
    keto, enol = [identity.leakage_identity('dili', s) for s in ('CC(=O)C', 'C=C(O)C')]
    assert keto['leakage_group'] == enol['leakage_group']
    v1_hashes = [rdMolHash.MolHash(Chem.MolFromSmiles(s), rdMolHash.HashFunction.HetAtomTautomer) for s in ('CC(=O)C', 'C=C(O)C')]
    assert v1_hashes[0] != v1_hashes[1]
    assert keto['leakage_group'] != identity.leakage_identity('dili', 'CC(C)O')['leakage_group']
    assert identity.leakage_identity('dili', 'c1ccccc1')['leakage_group'] != identity.leakage_identity('dili', 'C1CCCCC1')['leakage_group']
    assert identity.leakage_identity('dili', '[13CH3]C(=O)C')['leakage_group'] != keto['leakage_group']


def test_failed_gold_identity_writes_pending_audit_before_raising(tmp_path, monkeypatch):
    import json
    from tools.chembl_tool.common.starling import build_new_task_tautomer_repair as repair
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "data/starling_data/dili/gold_v2/source_votes.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"drug": "CC", "source_record_id": "uid-1"}) + "\n")
    class Pool:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def map(self, *args, **kwargs): return iter([("CC", None, "enumeration failed")])
    monkeypatch.setattr(repair, "ProcessPoolExecutor", Pool)
    import pytest
    with pytest.raises(ValueError, match="non-complete enumeration"):
        repair.build_source("dili", output_root=tmp_path / "staged", workers=1)
    pending = tmp_path / "staged/DILI/gold/identity_pending.jsonl"
    assert json.loads(pending.read_text())["reason"] == "enumeration failed"
