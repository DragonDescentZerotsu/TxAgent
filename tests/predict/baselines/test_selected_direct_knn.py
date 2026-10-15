"""Selected-card voting uses the same algorithms for Gold and canonical TDC."""
import csv
import json
import sqlite3
from pathlib import Path

import pytest

from predict.baselines import selected_direct_knn as knn
from predict.utils.json import sha256_file


def _grid(tmp_path, monkeypatch, benchmark, *, subset='valid', missing_score=False, ambiguous=False):
    gold, tdc = tmp_path / 'gold.jsonl', tmp_path / 'tdc.jsonl'
    for path, name in ((gold, 'gold-q'), (tdc, 'tdc-q')):
        path.write_text(json.dumps(dict(benchmark_row_id=name, drug='CCO', Y=1)) + '\n')
    monkeypatch.setattr(knn, 'split_path', lambda task, selected: gold if selected == subset else pytest.fail('wrong Gold subset'))
    monkeypatch.setattr(knn, 'tdc_split_path', lambda task, selected: tdc if selected == subset else pytest.fail('wrong TDC subset'))
    target = gold if benchmark == 'gold_v1' else tdc
    qid = 'gold-q' if benchmark == 'gold_v1' else 'tdc-q'
    database = tmp_path / 'rankings.sqlite3'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE contexts(context_id,parent_id,gold_label)')
        db.execute('CREATE TABLE rankings(benchmark_row_id,morgan_context_id,assay_context_id,assay_rank,assay_transfer_score,morgan_similarity)')
        for i in range(10):
            label = -1 if ambiguous and i == 9 else int(i >= 5)
            db.execute('INSERT INTO contexts VALUES (?,?,?)', (f'c{i}', f'p{i}', label))
            db.execute('INSERT INTO rankings VALUES (?,?,?,?,?,?)',
                       (qid, f'c{i}', f'c{i}', None if missing_score and i == 9 else 10-i,
                        None if missing_score and i == 9 else .2 + i*.02, .95 - i*.05))
    index, l1 = tmp_path / 'RELEASE_INDEX.json', tmp_path / 'VERSION.json'
    index.write_text('{}');l1.write_text('{}')
    selected = tmp_path / 'selected_contexts.tsv'
    with selected.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=('benchmark_row_id','selection_rank','context_id','parent_id','gold_label'), delimiter='\t')
        writer.writeheader()
        for i in range(10):
            writer.writerow(dict(benchmark_row_id=qid,selection_rank=i+1,context_id=f'c{i}',parent_id=f'p{i}',gold_label=-1 if ambiguous and i == 9 else int(i>=5)))
    leaf = dict(schema_version='direct_context_selection.v2',status='complete',benchmark=benchmark,
                subset=subset,task_id='ames',profile='fixture',budget=10,benchmark_row_ids=[qid],
                objective=dict(gated_assay_lambda=1),
                inputs=dict(queries=str(target),queries_sha256=sha256_file(target),direct_cache=dict(
                    label_release='v1' if benchmark=='gold_v1' else {'benchmark':'tdc'},
                    release_index=str(index),release_index_sha256=sha256_file(index),l1_manifest=str(l1),l1_manifest_sha256=sha256_file(l1),
                    database=str(database),database_sha256=sha256_file(database))),
                records=dict(path=selected.name,sha256=sha256_file(selected),row_count=10))
    leaf_path = tmp_path / 'manifest.json';leaf_path.write_text(json.dumps(leaf))
    grid = tmp_path / 'grid.json';grid.write_text(json.dumps(dict(
        schema_version='direct_context_selection_grid.v2',status='complete',kind='direct',benchmark=benchmark,
        subset=subset,tasks=['ames'],profile_count=1,profiles=[dict(name='fixture',task_manifests=dict(
            ames=dict(path=str(leaf_path),sha256=sha256_file(leaf_path))))])))
    return grid, target


@pytest.mark.parametrize('benchmark', ['gold_v1','tdc'])
@pytest.mark.parametrize('order,prediction', [('assay-rank',1),('policy-relevance',0)])
def test_both_benchmarks_reach_voting_and_honor_neighbor_order(tmp_path, monkeypatch, benchmark, order, prediction):
    grid,target = _grid(tmp_path,monkeypatch,benchmark)
    output=tmp_path/'output'
    result=knn.run(grid,output,benchmark=benchmark,subset='valid',task='ames',neighbor_order=order,ks=(3,),ambiguous_card_policy=None)
    assert json.loads(result.read_text())['target_path']==str(target.resolve())
    with (output/'predictions.tsv').open() as handle:
        rows=list(csv.DictReader(handle,delimiter='\t'))
    assert len(rows)==1 and int(rows[0]['prediction'])==prediction


def test_gold_validation_view_uses_its_own_labels(tmp_path,monkeypatch):
    grid,target=_grid(tmp_path,monkeypatch,'gold_v1',subset='valid_small')
    result=knn.run(grid,tmp_path/'output',benchmark='gold_v1',subset='valid',task='ames',neighbor_order='policy-relevance',ks=(3,),ambiguous_card_policy=None)
    assert json.loads(result.read_text())['subset']=='valid_small'


@pytest.mark.parametrize('benchmark', ['gold_v1','tdc'])
def test_missing_scores_only_block_assay_order(tmp_path,monkeypatch,benchmark):
    grid,_=_grid(tmp_path,monkeypatch,benchmark,missing_score=True)
    with pytest.raises(ValueError,match='rank or score is missing'):
        knn.run(grid,tmp_path/'bad',benchmark=benchmark,subset='valid',task='ames',neighbor_order='assay-rank',ks=(3,),ambiguous_card_policy=None)
    assert not (tmp_path/'bad').exists()
    knn.run(grid,tmp_path/'good',benchmark=benchmark,subset='valid',task='ames',neighbor_order='policy-relevance',ks=(3,),ambiguous_card_policy=None)


def test_incompatible_tdc_label_identity_is_rejected(tmp_path,monkeypatch):
    grid,_=_grid(tmp_path,monkeypatch,'tdc')
    path=tmp_path/'manifest.json';leaf=json.loads(path.read_text())
    leaf['inputs']['direct_cache']['label_release']['version']='tdc'
    path.write_text(json.dumps(leaf));doc=json.loads(grid.read_text());doc['profiles'][0]['task_manifests']['ames']['sha256']=sha256_file(path);grid.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='identity or score policy'):
        knn.run(grid,tmp_path/'bad',benchmark='tdc',subset='valid',task='ames',neighbor_order='assay-rank',ks=(3,),ambiguous_card_policy=None)


def test_ambiguous_labels_still_require_explicit_abstention(tmp_path,monkeypatch):
    grid,_=_grid(tmp_path,monkeypatch,'gold_v1',ambiguous=True)
    args=dict(benchmark='gold_v1',subset='valid',task='ames',neighbor_order='assay-rank',ks=(3,))
    with pytest.raises(ValueError,match='Ambiguous cards'):
        knn.run(grid,tmp_path/'bad',**args,ambiguous_card_policy=None)
    knn.run(grid,tmp_path/'good',**args,ambiguous_card_policy='abstain')


@pytest.mark.parametrize('benchmark', ['gold_v1','tdc'])
def test_cli_accepts_both_benchmarks(tmp_path,monkeypatch,benchmark):
    import sys
    grid,_=_grid(tmp_path,monkeypatch,benchmark)
    monkeypatch.setattr(sys,'argv',['knn','--benchmark',benchmark,'--selection-grid',str(grid),
                                  '--subset','valid','--task','ames','--neighbor-order','assay-rank',
                                  '--k','3','--output-dir',str(tmp_path/'cli')])
    knn.main()
    assert json.loads((tmp_path/'cli/manifest.json').read_text())['benchmark']==benchmark


def test_different_condition_context_cannot_supply_selected_card_score(tmp_path,monkeypatch):
    grid,_=_grid(tmp_path,monkeypatch,'gold_v1')
    with sqlite3.connect(tmp_path/'rankings.sqlite3') as db:
        db.execute("INSERT INTO contexts VALUES ('alternate','p9',0)")
        db.execute("UPDATE rankings SET assay_context_id='alternate' WHERE morgan_context_id='c9'")
    knn.run(grid,tmp_path/'output',benchmark='gold_v1',subset='valid',task='ames',neighbor_order='policy-relevance',ks=(3,),ambiguous_card_policy=None)
    with (tmp_path/'output/neighbors.tsv').open() as handle:
        row=next(r for r in csv.DictReader(handle,delimiter='\t') if r['context_id']=='c9')
    assert float(row['relevance'])==pytest.approx(.5)
    with pytest.raises(ValueError,match='rank or score is missing'):
        knn.run(grid,tmp_path/'bad',benchmark='gold_v1',subset='valid',task='ames',neighbor_order='assay-rank',ks=(3,),ambiguous_card_policy=None)


def test_explicitly_missing_overlay_score_does_not_use_base_score(tmp_path, monkeypatch):
    grid, _ = _grid(tmp_path, monkeypatch, 'tdc')
    qid = 'tdc-q'
    leaf_path = tmp_path / 'manifest.json'
    leaf = json.loads(leaf_path.read_text())
    selected = tmp_path / 'selected_contexts.tsv'
    rows = list(csv.DictReader(selected.open(), delimiter='\t'))
    overlay_db = tmp_path / 'scores.sqlite3'
    with sqlite3.connect(overlay_db) as db:
        db.execute('CREATE TABLE scores(query_id,level,item_id,parent_id,score,score_key)')
        for row in rows:
            i = int(row['context_id'][1:])
            row['assay_transfer_score'] = str(.2+i*.02) if i != 9 else ''
            row['score_key'] = 'key'+str(i) if i != 9 else ''
            if i != 9:
                db.execute('INSERT INTO scores VALUES (?,?,?,?,?,?)', (qid, 'L1', row['context_id'], row['parent_id'], float(row['assay_transfer_score']), row['score_key']))
    with selected.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    overlay = tmp_path / 'overlay.json'
    overlay.write_text(json.dumps(dict(status='complete', task='ames', database=overlay_db.name, database_sha256=sha256_file(overlay_db))))
    coverage = tmp_path / 'coverage.tsv'
    coverage.write_text('benchmark_row_id\tscored_candidates\tcandidate_count\tdirect_scored\tindirect_scored\n'+qid+'\t9\t100\t9\t0\n')
    policy = dict(version='available_scores.v1', overlay_manifest=str(overlay), overlay_manifest_sha256=sha256_file(overlay), database_sha256=sha256_file(overlay_db), coverage=dict(path=coverage.name, sha256=sha256_file(coverage), row_count=1))
    for field in ('cache_bundle','later_index'):
        source = tmp_path / (field+'.json')
        source.write_text('{}')
        policy[field] = str(source)
        policy[field+'_sha256'] = sha256_file(source)
    leaf['score_policy'] = policy
    leaf['records']['sha256'] = sha256_file(selected)
    leaf_path.write_text(json.dumps(leaf))
    doc = json.loads(grid.read_text())
    doc['profiles'][0]['task_manifests']['ames']['sha256'] = sha256_file(leaf_path)
    grid.write_text(json.dumps(doc))
    knn.run(grid,tmp_path/'out',benchmark='tdc',subset='valid',task='ames',neighbor_order='policy-relevance',ks=(3,),ambiguous_card_policy=None)
    result = next(row for row in csv.DictReader((tmp_path/'out/neighbors.tsv').open(), delimiter='\t') if row['context_id'] == 'c9')
    assert result['exact_score'] == ''
    assert float(result['relevance']) == pytest.approx(float(result['similarity']))
