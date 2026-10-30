from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import time
import zlib

import pytest

from tools.chembl_tool.common import budgeted_preparation as preparation


def manifest():
    return {'code': {'retriever.py': 'implementation'}, 'inputs': {'ames': {
        'index': {'path': '/index.pkl', 'sha256': 'index'},
        'input': {'path': '/valid.jsonl', 'sha256': 'valid'},
        'catalog': {'sha256': 'families'}, 'source': {'sha256': 'source'}}}}


def options(**overrides):
    return dict(task='ames', manifest=manifest(), split_scheme='scaffold',
                candidate_molecules=128, min_similarity=0.0, **overrides)


def test_candidate_cache_singleflight_and_reload_preserve_full_pool(tmp_path, monkeypatch):
    calls = []
    pool = {'direct': [{'id': 'a', 'similarity': .123456, 'card': {'text': 'μg + S9'}}],
            'indirect': [{'id': 'b', 'card': {'nested': ['x', 1, None]}}]}
    def retrieve(*a, **kw):
        calls.append(kw)
        time.sleep(.02)
        return deepcopy(pool)
    monkeypatch.setattr(preparation, 'retrieve_candidates', retrieve)
    cache = preparation.PreparationCache(tmp_path)
    with ThreadPoolExecutor(8) as workers:
        results = list(workers.map(lambda _: cache.candidates('CCO', {}, **options()), range(8)))
    assert len(calls) == 1 and results == [pool] * 8
    assert cache.snapshot() == {'hits': 7, 'misses': 1, 'index_loads': 0}
    results[0]['direct'][0]['card']['text'] = 'changed caller copy'
    other = preparation.PreparationCache(tmp_path)
    altered_cohort = manifest()
    altered_cohort['inputs']['ames']['input']['sha256'] = 'test'
    args = options(); args['manifest'] = altered_cohort
    assert other.candidates('CCO', {}, **args) == pool  # Retrieval never reads gold/conditions.
    assert len(calls) == 1 and other.hits == 1


@pytest.mark.parametrize('field', ['code', 'index', 'source', 'catalog', 'threshold', 'count', 'split', 'smiles', 'indirect_policy'])
def test_retrieval_contract_changes_invalidate_cache(tmp_path, monkeypatch, field):
    calls = []
    monkeypatch.setattr(preparation, 'retrieve_candidates',
                        lambda *a, **kw: calls.append(kw) or {'direct': [], 'indirect': []})
    cache = preparation.PreparationCache(tmp_path)
    args = options(); cache.candidates('CCO', {}, **args)
    smiles = 'CCO'
    if field == 'code': args['manifest']['code']['retriever.py'] = 'changed'
    elif field in ('index', 'source', 'catalog'): args['manifest']['inputs']['ames'][field]['sha256'] = 'changed'
    elif field == 'threshold': args['min_similarity'] = .3
    elif field == 'count': args['candidate_molecules'] = 64
    elif field == 'split': args['split_scheme'] = 'random'
    elif field == 'indirect_policy': args['indirect_identity_policy'] = 'scaffold_disjoint'
    else: smiles = 'CCCO'
    cache.candidates(smiles, {}, **args)
    assert len(calls) == 2


def test_resident_view_is_loaded_once_and_invalidated_on_source_change(tmp_path, monkeypatch):
    loads = []
    monkeypatch.setattr(preparation, 'local_input', lambda p: p)
    monkeypatch.setattr(preparation, 'load_retrieval_index', lambda p: loads.append(p) or {})
    monkeypatch.setattr(preparation, 'build_evidence_view', lambda i: {'resident': True})
    cache = preparation.PreparationCache(tmp_path)
    with ThreadPoolExecutor(4) as workers:
        views = list(workers.map(lambda _: cache.view('ames', manifest()), range(4)))
    assert len(loads) == 1 and all(v is views[0] for v in views)
    changed = manifest(); changed['inputs']['ames']['index']['sha256'] = 'changed'
    cache.view('ames', changed)
    assert len(loads) == 2
    def fail_load(path):
        raise OSError('index temporarily unavailable')
    monkeypatch.setattr(preparation, 'load_retrieval_index', fail_load)
    with pytest.raises(OSError):
        cache.view('ames', manifest())
    monkeypatch.setattr(preparation, 'load_retrieval_index', lambda p: {})
    # Retry the previously loaded key after a failed switch, not an empty view.
    assert cache.view('ames', changed) == {'resident': True}


def test_corrupt_binding_cannot_be_used(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, 'retrieve_candidates', lambda *a, **kw: {'direct': [], 'indirect': []})
    cache = preparation.PreparationCache(tmp_path)
    cache.candidates('CCO', {}, **options())
    path = next(tmp_path.glob('*/*.zlib'))
    saved = json.loads(zlib.decompress(path.read_bytes()))
    saved['binding']['min_similarity'] = .3
    path.write_bytes(zlib.compress(json.dumps(saved).encode()))
    with pytest.raises(ValueError, match='binding mismatch'):
        cache.candidates('CCO', {}, **options())


def test_lazy_view_only_loads_on_cache_miss(tmp_path, monkeypatch):
    cache = preparation.PreparationCache(tmp_path)
    views=[]; view={'index':'loaded'}
    monkeypatch.setattr(cache, 'view', lambda *a: views.append(a) or view)
    def retrieve(smiles, actual, **kw):
        assert actual is view
        return {'direct': [], 'indirect': []}
    monkeypatch.setattr(preparation, 'retrieve_candidates', retrieve)
    cache.candidates('CCO', None, **options())
    cache.candidates('CCO', None, **options())
    assert len(views)==1 and cache.hits==1 and cache.misses==1
