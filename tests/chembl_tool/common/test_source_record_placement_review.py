import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.chembl_tool.common.json_utils import sha256_file
from tools.chembl_tool.common.starling.source_gold_review import payload_hash
from tools.chembl_tool.common.starling.stage_new_task_retrieval import apply_record_review
from tools.chembl_tool.common.starling.stage_new_task_retrieval import _repair_source_row
from tools.chembl_tool.common.starling.stage_new_task_retrieval import apply_content_repairs


def fixture(tmp_path):
    source=tmp_path/'source';source.mkdir()
    rows=[]
    for uid,level in [('voter',1),('move',4),('noise',7)]:
        raw={'support_text':uid+' observation'}
        rows.append(dict(source_row_uid=uid,raw_record_json=json.dumps(raw),support_text=raw['support_text'],
            group_id='old',retrieval_eligible=True,is_gold_voter=level==1,progressive_level=level,
            family_key='old',level_assignment_reason='old',heldout_filter_scope='',placement_reviewed=False))
    pq.write_table(pa.Table.from_pylist(rows),source/'records.parquet')
    pq.write_table(pa.Table.from_pylist([dict(source_row_uid=r['source_row_uid'],level=r['progressive_level'],
        family_key='old',reason='old',reviewed=False) for r in rows]),source/'record_audit.parquet')
    review={'task':'dili','source_records_sha256':sha256_file(source/'records.parquet'),'decisions':[]}
    for uid,old,new in [('move',4,3),('noise',7,0)]:
        review['decisions'].append(dict(source_row_uid=uid,source_payload_sha256=payload_hash({'support_text':uid+' observation'}),
            previous_level=old,level=new,action='move' if new else 'exclude',reason='individually reviewed',quote=uid+' observation'))
    ledger=tmp_path/'ledger.json';ledger.write_text(json.dumps(review))
    return source,ledger,review


def test_preserves_raw_rows_voters_and_input_snapshot(tmp_path):
    source,ledger,review=fixture(tmp_path)
    before=pq.read_table(source/'records.parquet')
    receipt=apply_record_review('dili',source,tmp_path/'out',ledger)
    after=pq.read_table(tmp_path/'out/records.parquet')
    assert receipt['records_changed']==2 and receipt['source_rows_deleted']==0
    assert after['progressive_level'].to_pylist()==[1,3,0]
    assert after['retrieval_eligible'].to_pylist()==[True,True,False]
    for column in ('raw_record_json','support_text','source_row_uid','is_gold_voter'):
        assert before[column].equals(after[column])
    assert pq.read_table(source/'records.parquet').equals(before)


def test_review_cannot_overwrite_source_or_published_hardlink(tmp_path):
    import os

    source, ledger, _ = fixture(tmp_path)
    output = tmp_path / 'published'
    output.mkdir()
    os.link(source / 'records.parquet', output / 'records.parquet')
    before = sha256_file(source / 'records.parquet')
    for destination in (source, output):
        with pytest.raises(ValueError, match='fresh versioned directory'):
            apply_record_review('dili', source, destination, ledger)
    assert sha256_file(source / 'records.parquet') == before
    assert sha256_file(output / 'records.parquet') == before


def test_direct_containment_exemption_requires_current_source_payload(tmp_path):
    source, ledger, review = fixture(tmp_path)
    records = pq.read_table(source / 'records.parquet').to_pylist()
    raw = {'support_text': 'Drug-induced liver injury was observed.'}
    records[1].update(raw_record_json=json.dumps(raw), support_text=raw['support_text'])
    pq.write_table(pa.Table.from_pylist(records), source / 'records.parquet')
    review['source_records_sha256'] = sha256_file(source / 'records.parquet')
    decision = review['decisions'][0]
    decision.update(source_payload_sha256=payload_hash(raw), quote=raw['support_text'])
    ledger.write_text(json.dumps(review))
    with pytest.raises(ValueError, match='Direct containment'):
        apply_record_review('dili', source, tmp_path / 'blocked', ledger)
    decision['direct_guard_exemption'] = True
    ledger.write_text(json.dumps(review))
    result = apply_record_review('dili', source, tmp_path / 'reviewed', ledger)
    assert result['records_changed'] == 2
    decision['source_payload_sha256'] = 'stale'
    ledger.write_text(json.dumps(review))
    with pytest.raises(ValueError, match='stale'):
        apply_record_review('dili', source, tmp_path / 'stale', ledger)


@pytest.mark.parametrize('problem',['source_hash','payload_hash','duplicate','voter','quote','missing'])
def test_rejects_unbound_or_voter_edits(tmp_path,problem):
    source,ledger,review=fixture(tmp_path)
    r=review['decisions'][0]
    if problem=='source_hash':review['source_records_sha256']='bad'
    elif problem=='payload_hash':r['source_payload_sha256']='bad'
    elif problem=='duplicate':review['decisions'].append(dict(r))
    elif problem=='voter':r.update(source_row_uid='voter',previous_level=1)
    elif problem=='quote':r['quote']='not present'
    else:r['source_row_uid']='absent'
    ledger.write_text(json.dumps(review))
    with pytest.raises(ValueError):apply_record_review('dili',source,tmp_path/'out',ledger)


def content_fixture(voter=False):
    raw={'source_row_uid':'u','pmid':'123','SMILES':'CCO','molecule_name':'wrong',
         'support_text':'Original source observation','human_evidence_basis':'clinical'}
    row={'source_row_uid':'u','source_id':'dili_base','raw_record_json':json.dumps(raw),
         'source_smiles':'CCO','canonical_smiles':'CCO','is_gold_voter':voter,
         'progressive_level':1 if voter else 2,'identity_review_reason':''}
    decision={'source_row_uid':'u','source_payload_sha256':payload_hash(raw),
              'previous_level':row['progressive_level'],'level':2,'reason':'verified correction',
              'sources':['primary study'],'corrected_record':{**raw,'SMILES':'CC(=O)O',
              'molecule_name':'acetic acid','support_text':'Corrected bounded observation'}}
    return row,decision


def test_identity_repair_updates_visible_fields_but_preserves_acquisition():
    row,d=content_fixture();fixed=_repair_source_row('dili',row,d)
    assert fixed['canonical_smiles']=='CC(=O)O'
    assert fixed['molecule_name']=='acetic acid'
    assert fixed['support_text']=='Corrected bounded observation'
    assert fixed['raw_record_json']==row['raw_record_json']
    assert fixed['source_smiles']=='CCO' and not fixed['is_gold_voter']
    assert json.loads(fixed['reviewed_record_json'])==d['corrected_record']


def test_published_identity_repair_keeps_record_audit_aligned(tmp_path, monkeypatch):
    monkeypatch.setattr(
        'tools.chembl_tool.common.starling.new_task_retrieval_identity.prepare_cache',
        lambda *args: None,
    )
    original, decision = content_fixture()
    fixed = _repair_source_row('dili', original, decision)
    row = {**fixed, **original, 'molecule_identity_key': 'old_identity'}
    source = tmp_path / 'source'
    source.mkdir()
    pq.write_table(pa.Table.from_pylist([row]), source / 'records.parquet')
    pq.write_table(pa.Table.from_pylist([{
        'source_row_uid': 'u', 'molecule_identity_key': 'old_identity', 'level': 2,
    }]), source / 'record_audit.parquet')
    ledger = tmp_path / 'review.json'
    ledger.write_text(json.dumps({
        'task': 'dili', 'source_records_sha256': sha256_file(source / 'records.parquet'),
        'decisions': [decision],
    }))
    output = tmp_path / 'repaired'
    apply_content_repairs('dili', source, output, ledger)
    repaired = pq.read_table(output / 'records.parquet').to_pylist()[0]
    audit = pq.read_table(output / 'record_audit.parquet').to_pylist()[0]
    assert audit['molecule_identity_key'] == repaired['molecule_identity_key']
    assert audit['molecule_identity_key'] != 'old_identity'
    assert repaired['raw_record_json'] == original['raw_record_json']


def test_material_correction_cannot_inherit_single_molecule_or_vote():
    row,d=content_fixture(voter=True);d['corrected_record']['SMILES']=None
    with pytest.raises(ValueError,match='voter'):_repair_source_row('dili',row,d)
    d['level']=0;fixed=_repair_source_row('dili',row,d)
    assert not fixed['canonical_smiles'] and not fixed['retrieval_eligible']
    assert fixed['is_gold_voter'] and fixed['raw_record_json']==row['raw_record_json']
    row,d=content_fixture();d['corrected_record']['SMILES']=None
    with pytest.raises(ValueError,match='single-molecule'):_repair_source_row('dili',row,d)


@pytest.mark.parametrize('problem',['payload','uid','pmid','level','unbound','direct'])
def test_content_repairs_reject_stale_or_unsafe_promotions(problem):
    row,d=content_fixture()
    if problem=='payload':d['source_payload_sha256']='bad'
    elif problem=='uid':d['corrected_record']['source_row_uid']='other'
    elif problem=='pmid':d['corrected_record']['pmid']='other'
    elif problem=='level':d['level']=1
    elif problem=='unbound':d['sources']=[]
    else:d['level']=5;d['corrected_record']['support_text']='Clinical liver injury was observed.'
    with pytest.raises(ValueError):_repair_source_row('dili',row,d)
