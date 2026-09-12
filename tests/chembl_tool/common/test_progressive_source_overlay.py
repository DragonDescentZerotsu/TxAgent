"""Current source membership, deduplication and heldout filtering gates."""
import json
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tools.chembl_tool.common.starling.source_gold_review import payload_hash


def row(uid, direct=False):
    return {'source_row_uid': uid, 'source_id': 'example_v5',
            'molecule_identity_key': 'parent', 'group_id': 'Source.example_v5',
            'retrieval_eligible': True, 'identity_review_reason': '',
            'raw_record_json': json.dumps({'uid':uid, 'direct':direct}),
            'source_smiles':'CCO', 'molecule_name':'ethanol',
            'support_text': 'clinical outcome' if direct else 'mechanism'}


@pytest.mark.parametrize('flag,function', [
    ('--refresh-votes', 'refresh_membership'), ('--record-review', 'apply_record_review'),
])
def test_maintenance_cli_dispatches_explicit_release(tmp_path, monkeypatch, flag, function):
    import sys
    from tools.chembl_tool.common.starling import stage_new_task_retrieval as maintenance

    calls = []
    monkeypatch.setattr(maintenance, function, lambda *args, **kw: calls.append((args, kw)))
    source, output, ledger = [tmp_path / name for name in ('source', 'output', 'ledger')]
    monkeypatch.setattr(sys, 'argv', ['maintenance', '--task', 'dili', '--source', str(source),
                                    '--output', str(output), flag, str(ledger)])
    maintenance.main()
    assert len(calls) == 1
    assert calls[0][0] == ('dili', source, output, ledger)


@pytest.mark.parametrize('exact_copy', [False, True])
@pytest.mark.parametrize('identity_hold', [False, True])
@pytest.mark.parametrize('identity_clearance', [False, True])
def test_new_vote_release_promotes_representative_only_and_preserves_raw(tmp_path, exact_copy, identity_hold, identity_clearance):
    from tools.chembl_tool.common.starling.stage_new_task_retrieval import refresh_membership
    source=tmp_path/'source';source.mkdir();output=tmp_path/'release'
    rows=[]
    for uid,level in [('new',2),('duplicate',2),('withdrawn',1),('mechanism',5)]:
        r=row(uid,level<3)
        r.update(progressive_level=level,family_key='old',level_assignment_reason='old',
            canonical_smiles='CCO',identity_verification='source_structure_only',
            is_gold_voter=level==1,heldout_filter_scope='direct_outcome' if level<3 else '',direct_signal=level<3)
        rows.append(r)
    if identity_clearance:
        for r in rows[:2]:r.update(identity_review_reason='old_lookup_error',retrieval_eligible=False,progressive_level=0)
    if exact_copy:
        rows[1]['raw_record_json']=rows[0]['raw_record_json']
    pq.write_table(pa.Table.from_pylist(rows),source/'records.parquet')
    audit=[{'source_row_uid':r['source_row_uid'],'level':r['progressive_level'],
            'family_key':'old','reason':'old','is_gold_voter':r['is_gold_voter'],'direct_signal':r['direct_signal']} for r in rows]
    pq.write_table(pa.Table.from_pylist(audit),source/'record_audit.parquet')
    if identity_clearance:
        from tools.chembl_tool.common.starling.new_task_retrieval_identity import prepare_cache,load_cache
        prepare_cache('dili',['CC'],source/'retrieval_identity_cache.jsonl',workers=1)
    else:
        for name in ('retrieval_identity_cache.jsonl','retrieval_identity_cache.manifest.json'):(source/name).write_text('{}')
    vote={'source_record_id':'new','source_payload_sha256':payload_hash(json.loads(rows[0]['raw_record_json'])),
        'study_id':'pmid:1','molecule_identity_key':'parent','condition_group':'no_reported_external_condition',
        'Y':0,'study_source_row_uids':['new','duplicate']}
    votes=tmp_path/'votes.jsonl';votes.write_text(json.dumps(vote)+'\n')
    hold_path=None
    if identity_hold:
        hold_path=tmp_path/'holds.jsonl'
        hold_path.write_text(json.dumps({'source_row_uid':'withdrawn', 'source_payload_sha256':payload_hash(json.loads(rows[2]['raw_record_json'])), 'reason':'verified_wrong_subject'})+'\n')
    clearance_path=None
    if identity_clearance:
        clearance_path=tmp_path/'clearances.jsonl'
        clearance_path.write_text(''.join(json.dumps({'source_row_uid':r['source_row_uid'],
            'source_payload_sha256':payload_hash(json.loads(r['raw_record_json'])),
            'previous_reason':'old_lookup_error','canonical_smiles':'CCO','reason':'verified exact aliases'})+'\n' for r in rows[:2]))
    result=refresh_membership('dili',source,output,votes,deduplicate_exact=True,identity_holds_path=hold_path,identity_clearances_path=clearance_path)
    out=pq.read_table(output/'records.parquet').to_pylist()
    expected=[rows[0],*rows[2:]] if exact_copy else rows
    withdrawn_level=0 if identity_hold else 2
    assert [r['progressive_level'] for r in out]==([1,withdrawn_level,5] if exact_copy else [1,2,withdrawn_level,5])
    assert result['new_hash_bound_identity_holds']==int(identity_hold)
    assert result['hash_bound_identity_clearances']==2*int(identity_clearance)
    if identity_clearance:
        assert out[0]['identity_review_reason']=='' and out[0]['retrieval_eligible']
        assert set(load_cache('dili',output/'retrieval_identity_cache.jsonl'))=={'CC','CCO'}
        assert set(load_cache('dili',source/'retrieval_identity_cache.jsonl'))=={'CC'}
    if identity_hold:
        held=next(r for r in out if r['source_row_uid']=='withdrawn')
        assert held['identity_review_reason']=='verified_wrong_subject' and held['retrieval_eligible'] is False
    assert [r['raw_record_json'] for r in out]==[r['raw_record_json'] for r in expected]
    assert out[-1]==rows[-1]
    assert result['source_rows_deleted']==int(exact_copy)
    assert pq.read_table(output/'record_audit.parquet').num_rows==len(expected)
    assert result['new_voter_labels']=={0:1}
    assert result['nonvoter_duplicate_support_records']==1
    if identity_clearance:
        bad=[json.loads(line) for line in clearance_path.read_text().splitlines()]
        bad[0]['source_payload_sha256']='stale'
        clearance_path.write_text(''.join(json.dumps(r)+'\n' for r in bad))
        with pytest.raises(ValueError,match='Stale identity clearance'):
            refresh_membership('dili',source,tmp_path/'bad_release',votes,identity_clearances_path=clearance_path)


def test_heldout_metadata_preserves_labels_and_rejects_conflicting_identity(tmp_path):
    from tools.chembl_tool.common.starling.stage_new_task_retrieval import stage_heldout_inputs
    row={'drug':'CCO','Y':0,'molecule_identity_key':'LFQSCWFLJHTTHZ-UHFFFAOYSA-N',
        'benchmark_row_id':'q1','condition_group':'no_reported_external_condition'}
    for scheme in ('scaffold','random'):
        p=tmp_path/'benchmark'/scheme;p.mkdir(parents=True)
        (p/'heldout_molecule_condition_labels.jsonl').write_text(json.dumps(row)+'\n')
    receipt=stage_heldout_inputs('dili',tmp_path/'benchmark',tmp_path/'heldout')
    out=json.loads((tmp_path/'heldout/scaffold.jsonl').read_text())
    assert all(out[k]==v for k,v in row.items())
    assert out['leakage_group'].startswith('EGFC_')
    assert receipt['scaffold']['ordered_query_labels_unchanged']
    row['leakage_group']='wrong'
    (tmp_path/'benchmark/scaffold/heldout_molecule_condition_labels.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='conflicts'):
        stage_heldout_inputs('dili',tmp_path/'benchmark',tmp_path/'bad')


def test_exact_duplicates_keep_voter_but_preserve_distinct_evidence(tmp_path):
    from tools.chembl_tool.common.starling.stage_new_task_retrieval import exact_duplicate_plan
    raw={'pmid':'123','support_text':'No injury at the reported dose.', 'confidence':0.9,
         'qualifying_conditions':'10 mg','SMILES':'CCO','molecule_name':'ethanol'}
    rows=[]
    for uid, extra in [('copy',{}),('voter',{}),('passage',{'support_text':'Injury at a higher dose.'}),
                       ('study',{'pmid':'456'}),('condition',{'qualifying_conditions':'20 mg'})]:
        rows.append({'source_row_uid':uid,'canonical_smiles':'CCO','identity_review_reason':'',
            'raw_record_json':json.dumps({**raw,**extra,'source_row_uid':uid,'extraction_id':uid,'paragraph_idx':len(rows)})})
    path=tmp_path/'records.parquet';pq.write_table(pa.Table.from_pylist(rows),path)
    result=exact_duplicate_plan(pq.ParquetFile(path),{'voter'})
    assert set(result)=={'copy'}
    assert result['copy']['retained_source_row_uid']=='voter'
    with pytest.raises(ValueError,match='both gold voters'):
        exact_duplicate_plan(pq.ParquetFile(path),{'copy','voter'})


def test_l1_only_filter_is_explicit_and_legacy_scope_is_identified():
    from tools.chembl_tool.common.starling.validate_new_task_retrieval_identity import heldout_excluded_levels
    assert heldout_excluded_levels('dili',{'filter_scope_field':'group_id','filter_scope_value':'Group.dili_actual_voter'})=={1}
    assert heldout_excluded_levels('dili',{'filter_scope_field':'heldout_filter_scope','filter_scope_value':'direct_outcome'})=={1,2}
    with pytest.raises(ValueError,match='Unsupported'):
        heldout_excluded_levels('dili',{'filter_scope_field':'group_id','filter_scope_value':'Group.dili_direct_outcome'})


def test_l1_prefilter_keeps_heldout_l2_and_training_l1(tmp_path):
    import pandas as pd
    from tools.chembl_tool.common.assay_retrieval import _filter_heldout_direct_records
    from tools.chembl_tool.common.starling.new_task_identity import leakage_identity, VERSION
    cache={s:leakage_identity('dili',s) for s in ('CCO','CCN')}
    heldout=tmp_path/'heldout.jsonl'
    heldout.write_text(json.dumps({'drug':'CCO','leakage_group':cache['CCO']['leakage_group']})+'\n')
    rows=pd.DataFrame([
        {'canonical_smiles':'CCO','group_id':'Group.dili_actual_voter','uid':'heldout_l1'},
        {'canonical_smiles':'CCO','group_id':'Group.dili_direct_outcome','uid':'heldout_l2'},
        {'canonical_smiles':'CCN','group_id':'Group.dili_actual_voter','uid':'train_l1'}])
    retained,receipt=_filter_heldout_direct_records(rows,heldout_molecules_path=heldout,
        heldout_smiles_field='drug',filter_source_id='',filter_scope_field='group_id',
        filter_scope_value='Group.dili_actual_voter',task='dili',identity_contract=VERSION,identity_cache=cache)
    assert set(retained['uid'])=={'heldout_l2','train_l1'}
    assert receipt['n_direct_heldout_records_excluded']==1
