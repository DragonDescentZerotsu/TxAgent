import json
from pathlib import Path

import pandas as pd
import pytest

from tools.chembl_tool.common.starling import new_task_retrieval_identity as retrieval
from tools.chembl_tool.common.starling.new_task_identity import VERSION, leakage_identity
from tools.chembl_tool.common.assay_retrieval import _filter_heldout_direct_records


def candidate(smiles):
    value = leakage_identity('dili', smiles)
    return {'retrieval_identity_contract': VERSION,
            'retrieval_leakage_group': value['leakage_group'],
            'retrieval_scaffold_leakage_group': value['scaffold_leakage_group']}


def test_tautomer_forms_removed_from_direct_records_across_sources(tmp_path):
    keto, enol = 'CC(=O)C', 'C=C(O)C'
    assert leakage_identity('dili', keto)['leakage_group'] == leakage_identity('dili', enol)['leakage_group']
    heldout = tmp_path / 'heldout.jsonl'
    heldout.write_text(json.dumps({'drug': keto, **leakage_identity('dili', keto)}) + '\n')
    rows = pd.DataFrame([
        {'canonical_smiles': enol, 'source_id': source, 'heldout_filter_scope': scope, 'uid': uid}
        for source, scope, uid in [('base', 'direct_outcome', 'voter'),
                                   ('other', 'direct_outcome', 'nonvoter'),
                                   ('other', '', 'mechanism')]])
    kept, stats = _filter_heldout_direct_records(rows, heldout_molecules_path=heldout,
        heldout_smiles_field='drug', filter_source_id='', filter_scope_field='heldout_filter_scope',
        filter_scope_value='direct_outcome', task='dili', identity_contract=VERSION,
        identity_cache={enol: leakage_identity('dili', enol)})
    assert kept['uid'].tolist() == ['mechanism']
    assert stats['n_direct_heldout_records_excluded'] == 2
    assert stats['n_heldout_leakage_groups'] == 1


def test_conservative_topology_scaffold_replaces_raw_bm():
    benzene = leakage_identity('dili', 'c1ccccc1')
    cyclohexane = leakage_identity('dili', 'C1CCCCC1')
    assert benzene['bemis_murcko_scaffold'] != cyclohexane['bemis_murcko_scaffold']
    assert benzene['scaffold_leakage_group'] == cyclohexane['scaffold_leakage_group']
    assert retrieval.decide_candidate(benzene, candidate('C1CCCCC1'), 'scaffold_disjoint', VERSION).excluded
    assert not retrieval.decide_candidate(benzene, candidate('C1CCCCC1'), 'parent_disjoint', VERSION).excluded
    assert retrieval.decide_candidate(leakage_identity('dili', 'CC(=O)C'), candidate('C=C(O)C'), 'parent_disjoint', VERSION).excluded


def test_identity_opt_in_rejects_legacy_tasks_and_incomplete_candidates():
    with pytest.raises(ValueError, match='Unsupported'):
        retrieval.validate_contract('ames', VERSION)
    with pytest.raises(ValueError, match='scaffold'):
        retrieval.decide_candidate(leakage_identity('dili', 'CCO'),
            {'retrieval_identity_contract': VERSION, 'retrieval_leakage_group': 'x'}, 'scaffold_disjoint', VERSION)


def test_cache_uses_light_helper_and_persistent_hits(tmp_path, monkeypatch):
    from contextlib import contextmanager
    calls = []
    class Serial:
        def map(self, fn, values, chunksize):
            return map(fn, values)
    @contextmanager
    def pool(_):
        yield Serial()
    original = leakage_identity
    def light(task, smiles):
        calls.append(smiles)
        return original(task, smiles)
    monkeypatch.setattr(retrieval, 'worker_pool', pool)
    monkeypatch.setattr(retrieval.adapter, 'leakage_identity', light)
    monkeypatch.setattr(retrieval.adapter, 'normalize_new_task_identity', lambda *_: pytest.fail('Gold enumeration called'))
    path = tmp_path / 'cache.jsonl'
    cache = retrieval.prepare_cache('dili', ['CCO', 'CCO', 'CCN'], path, workers=2)
    assert sorted(calls) == ['CCN', 'CCO']
    assert retrieval.prepare_cache('dili', ['CCO', 'CCN'], path) == cache
    assert len(calls) == 2
    retrieval.query_identity.cache_clear()
    retrieval.query_identity('dili', 'CCO', VERSION)
    retrieval.query_identity('dili', 'CCO', VERSION)
    assert len(calls) == 3


def test_index_annotations_preserve_original_identity():
    cache = {'CCO': leakage_identity('dili', 'CCO')}
    index = {'source': {}, 'molecules': [{'canonical_smiles': 'CCO', 'molecule_identity': {'parent_smiles': 'CCO'}}]}
    retrieval.annotate_index(index, 'dili', cache, VERSION)
    assert index['molecules'][0]['molecule_identity'] == {'parent_smiles': 'CCO'}
    assert index['source']['retrieval_identity_contract'] == VERSION
    assert index['molecules'][0]['retrieval_leakage_group'] == cache['CCO']['leakage_group']


def test_runtime_excludes_conservative_scaffold_before_top_k():
    from tools.chembl_tool.common.task_workflows.evidence_library import build_neighbor_index
    from tools.chembl_tool.common.task_workflows.retrieve_neighbors import retrieve_neighbors
    rows = []
    for mid, smiles in [('same', 'c1ccccc1'), ('topology', 'C1CCCCC1'), ('eligible', 'c1ccncc1')]:
        rows.append({'molecule_chembl_id': mid, 'canonical_smiles': smiles,
                     'assay_chembl_id': 'a', 'assay_tier': 'Assay', 'endpoint_group': 'test',
                     'group_id': 'Assay.test', 'standard_type': 'endpoint', 'standard_value': '1',
                     'standard_units': 'unit', 'evidence_source': 'Starling', 'assay_description': 'endpoint'})
    index = build_neighbor_index(rows, index_version='test', workers=1)
    index['source'] = {}
    retrieval.annotate_index(index, 'dili', {r['canonical_smiles']: leakage_identity('dili', r['canonical_smiles']) for r in rows}, VERSION)
    result = retrieve_neighbors('c1ccccc1', index, top_k_per_group=1, min_similarity=0,
                                neighbor_identity_policy='scaffold_disjoint')
    assert [r['molecule_chembl_id'] for r in result['groups'][0]['neighbors']] == ['eligible']


def test_formal_runner_rejects_old_index_even_if_heldout_hash_matches(tmp_path):
    from tools.chembl_tool.common.json_utils import sha256_file
    from tools.chembl_tool.paper_experiments.run_conditioned_assay_progressive_curve import _heldout_filter_validation
    heldout = tmp_path / 'heldout.jsonl'
    heldout.write_text('{}\n')
    metadata = {'heldout_molecules_jsonl_sha256': sha256_file(heldout)}
    with pytest.raises(ValueError, match='tautomer-aware'):
        _heldout_filter_validation(task='DILI', split_scheme='random', index_manifest=metadata, heldout_path=heldout)
    metadata.update(retrieval_identity_contract=VERSION, retrieval_identity_task='dili')
    assert _heldout_filter_validation(task='DILI', split_scheme='random', index_manifest=metadata, heldout_path=heldout)['status'] == 'ok'
    metadata['retrieval_identity_task'] = 'carcinogens'
    with pytest.raises(ValueError, match='task mismatch'):
        _heldout_filter_validation(task='DILI', split_scheme='random', index_manifest=metadata, heldout_path=heldout)


def test_thin_overlay_preserves_raw_and_only_exact_hold(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tools.chembl_tool.common.starling.stage_new_task_retrieval import stage_overlay
    from tools.chembl_tool.common.starling.source_gold_review import payload_hash
    monkeypatch.chdir(tmp_path)
    source = Path('data/starling_data/dili/progressive_v1')
    stage = Path('stage')
    source.mkdir(parents=True)
    (stage/'DILI/gold').mkdir(parents=True)
    records = []
    for uid in ['hold', 'keep']:
        records.append({'source_row_uid': uid, 'raw_record_json': json.dumps({'same_name': 'mixture', 'same_smiles': 'CCO', 'uid': uid}),
            'canonical_smiles': 'CCO', 'group_id': 'Group.dili_actual_voter', 'retrieval_eligible': True,
            'is_gold_voter': True, 'progressive_level': 1, 'family_key': 'dili_actual_voter',
            'level_assignment_reason': 'actual_voter', 'heldout_filter_scope': 'direct_outcome'})
    pq.write_table(pa.Table.from_pylist(records), source/'records.parquet')
    pq.write_table(pa.Table.from_pylist([{'source_row_uid': r['source_row_uid'], 'level': 1,
        'family_key': 'dili_actual_voter', 'reason': 'actual_voter', 'is_gold_voter': True} for r in records]), source/'record_audit.parquet')
    (stage/'DILI/gold/actual_voter_membership.jsonl').write_text(json.dumps({'source_record_id': 'keep'})+'\n')
    (stage/'specimen_identity_withdrawal_candidates.jsonl').write_text(json.dumps({'task': 'DILI', 'source_row_uid': 'hold',
        'source_payload_sha256': payload_hash(json.loads(records[0]['raw_record_json']))})+'\n')
    output = stage_overlay('dili', stage)
    changed = pq.read_table(output/'records.parquet').to_pylist()
    assert [r['raw_record_json'] for r in changed] == [r['raw_record_json'] for r in records]
    assert [r['progressive_level'] for r in changed] == [0, 1]
    assert changed[1] == records[1]


def test_annotation_uses_original_evidence_identity_after_native_smiles_reserialization():
    # Native index serialization can change stereo annotation of large molecules;
    # cache membership must follow the original source molecule id, not its text.
    index = {'source': {}, 'molecules': [{'molecule_chembl_id':'source1', 'canonical_smiles':'native-reserialized',
                                         'molecule_identity':{'parent_smiles':'native-reserialized'}}]}
    cache = {'CCO': leakage_identity('dili','CCO')}
    retrieval.annotate_index(index,'dili',cache,VERSION,{'source1':'CCO'})
    assert index['molecules'][0]['retrieval_identity_source_smiles'] == 'CCO'
    assert index['molecules'][0]['retrieval_leakage_group'] == cache['CCO']['leakage_group']


def test_builder_captures_input_before_native_builder_mutates_evidence(tmp_path, monkeypatch):
    from tools.chembl_tool.common import assay_retrieval as assay
    cache = {'CCO': leakage_identity('dili','CCO')}
    monkeypatch.setattr(retrieval,'load_cache',lambda *_: cache)
    rows = [{'molecule_chembl_id':'m1','canonical_smiles':'CCO'}]
    monkeypatch.setattr(assay,'build_assay_evidence_rows',lambda **_: (rows,[],{}))
    def native(values, **kwargs):
        values[0]['canonical_smiles'] = 'native-reserialized'
        return {'molecules':[{'molecule_chembl_id':'m1','canonical_smiles':'native-reserialized',
                              'molecule_identity':{'parent_smiles':'native-reserialized'}}]}
    monkeypatch.setattr(assay,'build_neighbor_index',native)
    path=tmp_path/'input';path.write_text('fixture')
    index, _, _ = assay.build_assay_index(task='dili',records_path=path,membership_path=None,
        ranked_assays_path=path,workers=1,max_record_examples=3,max_support_text_chars=0,max_assays=0,
        identity_contract=VERSION,identity_cache_path=path,heldout_molecules_path=path)
    assert index['molecules'][0]['retrieval_identity_source_smiles'] == 'CCO'
