import argparse
import json

import numpy as np
import joblib
import pytest

from predict.baselines import retrieval_features as features
from predict.baselines.retrieval_forest import fit, finalize
from predict.utils.json import sha256_file


def test_bucket_argmax_is_local_stable_and_handles_empty():
    chosen, scores = features.bucket_neighbors(
        np.array([-np.inf, .7, .7, .9]), np.array([0, 1, 2, 0, 3]), np.array([0, 1, 3]))
    assert chosen.tolist() == [-1, 1, 4]
    assert scores.tolist() == [-np.inf, .7, .9]


def test_condition_selects_members_then_sorts_by_similarity():
    reference = [dict(drug=str(i), Y=i % 2, condition_group=c)
                 for i, c in enumerate(['a','b','a','no_reported_external_condition'])]
    similarity = np.array([.5,.99,.4,.8])
    assert features.direct_neighbors(similarity, reference, {'condition_group':'a'},
                                     'same_condition_then_null') == [3,0,2]
    assert features.direct_neighbors(similarity, reference, {}, 'all_train_unique_molecules') == [1,3,0]


def test_query_label_and_same_parent_variants_cannot_enter_features(monkeypatch):
    reference = [dict(drug=str(i), Y=i % 2) for i in range(5)]
    context = dict(identities={'q':('P','K')}, fingerprints={'q':None}, all_fps=[],
                   identity_molecules={'P':[0],'K':[0,1]}, target_molecules=np.arange(5),
                   starts=np.array([0,2]), targets=np.arange(5,dtype=float),
                   reference=reference, reference_molecules=np.arange(5),
                   rows=[{'drug':'q','Y':0}], policies=['all_train_unique_molecules','same_condition_then_null'])
    monkeypatch.setattr(features, '_CONTEXT', context)
    monkeypatch.setattr(features.DataStructs, 'BulkTanimotoSimilarity', lambda *_:[1,1,.8,.7,.6])
    first = features._query(('q',[0]))
    assert first[2].tolist() == [-1,2]
    assert np.isnan(first[1][:2]).all()
    assert first[4].tolist() == [[[2,3,4],[2,3,4]]]
    context['rows'][0]['Y']=1
    reference[0]['Y']=1-reference[0]['Y'];reference[1]['Y']=1-reference[1]['Y']
    second = features._query(('q',[0]))
    for a,b in zip(first[1:],second[1:]):np.testing.assert_equal(a,b)


@pytest.mark.parametrize('feature_set', ['direct', 'direct_indirect'])
def test_rf_selection_and_prediction_reuse(tmp_path, feature_set):
    prepared=tmp_path/'prepared';prepared.mkdir();data_dir=tmp_path/'data';data_dir.mkdir()
    rng=np.random.default_rng(4);x=rng.normal(size=(36,8)).astype('float32');x[::3,6:]=np.nan
    direct=np.repeat(x[:,:6,None],2,axis=2).transpose(0,2,1)
    np.save(prepared/'direct.npy',direct);np.save(prepared/'auxiliary.npy',x[:,6:])
    offsets={'train':[0,20],'valid':[20,28],'test':[28,36]}
    for s,(a,b) in offsets.items():
        (data_dir/f'{s}.jsonl').write_text('\n'.join(json.dumps(dict(drug=str(i),Y=i%2)) for i in range(a,b))+'\n')
    m=dict(data_dir=str(data_dir),offsets=offsets,features=8,policies=['all','condition'],
           inputs={str(p):sha256_file(p) for p in data_dir.glob('*.jsonl')},
           files={p.name:sha256_file(p) for p in prepared.glob('*.npy')})
    (prepared/'manifest.json').write_text(json.dumps(m));out=tmp_path/'model'
    if feature_set == 'direct':
        # Direct-only loading must not depend on any indirect values or file.
        (prepared/'auxiliary.npy').unlink()
        loaded, split_features, _ = features.load_features(prepared, 0, feature_set)
        assert loaded['features'] == 6
        np.testing.assert_array_equal(split_features['train'], x[:20, :6])
    for leaf in (1,3):
        args=argparse.Namespace(prepared=prepared,policy=0,output=out/'candidates'/str(leaf),
                                max_features='sqrt',min_leaf=leaf,workers=1,trees=8,seeds=[1,2],
                                feature_set=feature_set)
        fit(args)
        assert joblib.load(args.output/'seed_1.joblib').n_features_in_ == (6 if feature_set == 'direct' else 8)
    args.output=out;args.expected_candidates=2
    finalize(args)
    selection=json.loads((out/'selection.json').read_text())
    assert selection['validation_bce']==min(selection['candidate_validation_bce'].values())
    assert not selection['test_used']
    path=out/'test_predictions.jsonl';before=(sha256_file(path),path.stat().st_mtime_ns)
    finalize(args)
    assert before==(sha256_file(path),path.stat().st_mtime_ns)
    if feature_set == 'direct':
        np.save(prepared/'auxiliary.npy',x[:,6:])
        args.feature_set = 'direct_indirect'
        with pytest.raises(ValueError, match='Candidate input mismatch'):
            finalize(args)
