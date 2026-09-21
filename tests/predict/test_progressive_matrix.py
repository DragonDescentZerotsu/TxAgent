"""Exercise matrix sharing, dependencies, retry limits and resumed responses."""

import csv
from dataclasses import replace
import threading
import time
import json
import queue
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace

import pytest

from predict.harnesses.progressive import matrix
from predict.harnesses.progressive.matrix import AttemptClient, request_key, schedule
from predict.harnesses.progressive import runner
from predict.api_client.client import OpenAICompatibleClient
from predict.api_client.pool import ProviderSelection
from predict.utils.json import sha256_file
from predict.harnesses.progressive.level_selection import select_policy_records


MODEL = matrix.load_provider_pool_config(matrix.DEFAULT_PROVIDER_CONFIG).providers[0].model


@pytest.fixture(autouse=True)
def _use_configured_test_endpoints(monkeypatch):
    monkeypatch.setattr(
        matrix,
        'select_healthy_providers',
        lambda config, requested: ProviderSelection(
            config=config,
            requested_parallelism=requested,
            effective_parallelism=min(requested, matrix.primary_capacity(config)),
            checks=(),
        ),
    )


class FakeEndpoint:
    status = 200
    def __enter__(self): return self
    def __exit__(self,*_): pass
    def read(self): return json.dumps({'data':[{'id':MODEL}]}).encode()


def test_full_matrix_requires_explicit_parallelism(tmp_path):
    with pytest.raises(SystemExit, match='2'):
        matrix.main(['--output-root', str(tmp_path / 'matrix')])
    with pytest.raises(SystemExit):
        matrix.main([
            '--output-root', str(tmp_path / 'test-matrix'),
            '--evaluation-subset', 'test', '--parallelism', '1',
        ])


class FakePool:
    def __init__(self, failures=0):
        self.failures = failures
        self.calls = []
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def chat_json(self, messages):
        with self.lock:
            self.calls.append(messages)
            self.active += 1
            self.peak = max(self.peak, self.active)
            fail = len(self.calls) <= self.failures
        time.sleep(.005)
        with self.lock:
            self.active -= 1
        if fail:
            raise ConnectionError('simulated disconnect')
        return dict(content={'value': messages[0]['content']},
                    structured_output_validation={'valid': True})

    def snapshot(self):
        return {'calls': len(self.calls)}


def chain(client, name, received, second=None):
    messages = [{'role':'user','content':name}]
    response = yield dict(messages=messages,task='test',level=1,
                         execute=lambda:client.chat_json(messages))
    received.append(response)
    if second:
        later = [{'role':'user','content':second+response['content']['value']}]
        response = yield dict(messages=later,task='test',level=2,
                             execute=lambda:client.chat_json(later))
        received.append(response)
    return {'status':'ok'}


def test_shared_roots_unlock_branches_and_resume(tmp_path):
    pool = FakePool()
    client = AttemptClient(pool)
    received = []
    results = schedule([chain(client,'root',received,'A'),chain(client,'root',received,'B'),
                        chain(client,'other',received,'C')],client,{},tmp_path,parallelism=2)
    assert all(r['status']=='ok' for r in results)
    assert len(pool.calls)==5
    assert pool.peak<=2
    assert {m[0]['content'] for m in pool.calls[:2]}=={'root','other'}
    assert sum(r['inference_reused'] for r in received)==1
    schedule([chain(client,'root',[],'A')],client,{},tmp_path,parallelism=2)
    assert len(pool.calls)==5


def test_streaming_starts_before_producer_finishes_and_reuses_late_duplicates(tmp_path):
    pool = FakePool()
    client = AttemptClient(pool)
    incoming = queue.Queue(maxsize=1)
    received = []
    started = threading.Event()
    original = pool.chat_json
    def chat(messages):
        started.set()
        return original(messages)
    pool.chat_json = chat
    def produce():
        incoming.put(chain(client,'root',received))
        assert started.wait(5), 'Inference waited for all preparation'
        incoming.put(chain(client,'root',received))
        incoming.put(None)
    producer = threading.Thread(target=produce)
    producer.start()
    results = schedule(incoming,client,{},tmp_path,parallelism=2)
    producer.join()
    assert len(results)==2 and all(r['status']=='ok' for r in results)
    assert len(pool.calls)==1 and sum(r['inference_reused'] for r in received)==1


def test_key_includes_prior_and_generation_contract():
    job=dict(task='t',level=2,messages=[{'role':'user','content':'prior positive'}])
    changed=deepcopy(job)
    changed['messages'][0]['content']='prior negative'
    assert request_key(job,{}) != request_key(changed,{})
    assert request_key(job,{'temperature':0}) != request_key(job,{'temperature':1})
    changed = deepcopy(job)
    changed['reasoning_grammar'] = 'root ::= "structured"'
    assert request_key(job,{}) != request_key(changed,{})
    changed = dict(job, request_namespace='replicate:2')
    assert request_key(job,{}) != request_key(changed,{})
    assert request_key(job,{}) == request_key(dict(job, request_namespace=''),{})
    assert request_key(job,{}) != request_key(dict(job, stage_kind='record_filter'),{})


def test_larger_output_budget_reuses_completed_smaller_budget_response(tmp_path):
    job = dict(task='test',level=1,messages=[{'role':'user','content':'root'}])
    old, new = {'max_tokens':262144}, {'max_tokens':1000000}
    key = request_key(job,old)
    path = tmp_path/'responses'/key[:2]/f'{key}.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(key=key,response={
        'content':{'value':'root'},'structured_output_validation':{'valid':True},
        'usage':{'completion_tokens':100},
    })))
    pool=FakePool()
    received=[]
    result=schedule([chain(AttemptClient(pool),'root',received)],AttemptClient(pool),new,tmp_path,
                    parallelism=2,reuse_settings=(old,))
    assert result == [{'status':'ok'}] and not pool.calls
    assert received[0]['inference_reused'] is True


def test_schedule_reuses_exact_request_from_external_batch(tmp_path):
    job = dict(task='test', level=1, messages=[{'role':'user','content':'root'}])
    old = {'model':'m', 'temperature':0, 'request_extra_body':{},
           'response_format':{'type':'json_object'}, 'max_tokens':65536,
           'code':'old'}
    new = dict(old, code='new')
    source = tmp_path/'old'/'inference'/'responses'
    key = request_key(job, old)
    path = source/key[:2]/f'{key}.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(key=key, response={
        'content':{'value':'root'}, 'structured_output_validation':{'valid':True},
        'usage':{'completion_tokens':100},
    })))
    pool, received = FakePool(), []
    result = schedule(
        [chain(AttemptClient(pool),'root',received)], AttemptClient(pool), new,
        tmp_path/'new'/'inference', parallelism=2,
        reuse_sources=((old, source),),
    )
    assert result == [{'status':'ok'}] and not pool.calls
    assert received[0]['inference_reuse']['source'] == str(path)


def test_context_overflow_retries_with_remaining_window():
    calls=[]
    def create(**kwargs):
        calls.append(kwargs)
        if len(calls)==1:
            raise RuntimeError('maximum context length of 1048576 tokens; 60115 tokens from the input messages')
        return object()
    client=OpenAICompatibleClient.__new__(OpenAICompatibleClient)
    client.client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.model=MODEL
    client.max_tokens=1_000_000
    client.temperature=0.0
    client.reasoning_effort=''
    client.enable_thinking=False
    client.request_extra_body={}
    client.response_format={'type':'json_object'}
    assert client._create_completion([{'role':'user','content':'x'}]) is not None
    assert [row['max_tokens'] for row in calls] == [1_000_000,988_461]


def test_rolling_resume_uses_prepared_evidence_without_a_pilot(tmp_path, monkeypatch):
    source = tmp_path/'source'
    source.mkdir()
    prepared = tmp_path/'prepared'
    condition = prepared/'conditions/bbb_martins/morgan_all_records10'
    qdir = runner._query_dir(condition,'bbb_martins',0)
    qdir.mkdir(parents=True)
    input_path = tmp_path/'valid.jsonl'
    input_path.write_text('{}\n')
    settings = dict(model=MODEL,max_tokens=100,
        prompt=matrix.prompt_asset_manifest('reranked_progressive_v8'),
        runner_code_sha256=sha256_file(Path(runner.__file__)),
        validation_code_sha256=sha256_file(Path(runner.__file__).resolve().parents[2]/'llm_io/response.py'))
    (source/'matrix.json').write_text(json.dumps(dict(settings=settings,modes=['morgan'],
        pools=['all'],caps=[10,50],limit=1,final_levels={})))
    reuse = tmp_path/'reuse.json'
    reuse.write_text(json.dumps({'settings': dict(settings, max_tokens=200)}))
    (condition/'experiment_manifest.json').write_text(json.dumps(dict(
        evaluation_indices_by_task={'bbb_martins':[0]},reranking='morgan',record_pool='all',
        inputs={'bbb_martins':dict(input_jsonl=str(input_path),input_sha256=sha256_file(input_path))})))
    (qdir/'prepared_manifest.json').write_text(json.dumps(dict(status='ok',
        task='bbb_martins',query_index=0,
        tool_prefetch_complete=True,prompt_version='reranked_progressive_v8',context_limit=10,
        record_limit_per_context_level=10,indirect_record_limit_per_level=10)))
    pool = FakePool()
    client = AttemptClient(pool)
    monkeypatch.setattr(matrix,'provider_client',lambda *a:client)
    monkeypatch.setattr(matrix,'preflight_provider_models',lambda *a,**kw:[])
    monkeypatch.setattr(runner,'_validate_inputs',lambda args:{'bbb_martins':[{}]})
    monkeypatch.setattr(runner,'query_steps',lambda args,q,c:chain(c,'saved',[]))
    monkeypatch.setattr(runner,'run',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Repeated preparation')))
    summarized = []
    monkeypatch.setattr(matrix,'summarize',lambda conditions,root:summarized.extend(conditions))
    output = tmp_path/'rolling'
    assert matrix.main(['--output-root',str(output),'--rolling-from',str(source),
        '--tasks','bbb_martins','--reranking-modes','morgan','--record-pools','all',
        '--records-per-level','10','--limit','1','--parallelism','2',
        '--execution-mode','throughput','--trace-root',str(tmp_path/'traces'),
        '--prepared-from',str(prepared),
        '--reuse-settings-from',str(reuse)]) == 0
    assert len(pool.calls)==1 and len(summarized)==1
    receipt = json.loads((output/'rolling.json').read_text())
    assert receipt['pilot']=='skipped_by_throughput_mode'
    assert receipt['prepared_from'] == str(prepared.resolve())
    assert len(receipt['compatible_reuse_settings']) == 2
    assert receipt['reuse_settings_receipts'][0]['path'] == str(reuse.resolve())


def test_rolling_new_task_prepares_bounded_batches_while_calls_run(tmp_path, monkeypatch):
    source = tmp_path/'source'
    source.mkdir()
    settings = dict(model=MODEL,max_tokens=100,
        prompt=matrix.prompt_asset_manifest('reranked_progressive_v8'),
        runner_code_sha256=sha256_file(Path(runner.__file__)),
        validation_code_sha256=sha256_file(Path(runner.__file__).resolve().parents[2]/'llm_io/response.py'))
    (source/'matrix.json').write_text(json.dumps(dict(settings=settings,modes=['morgan'],
        pools=['all'],caps=[10],limit=0,final_levels={'bbb_martins':5})))
    pool = FakePool()
    client = AttemptClient(pool)
    started = threading.Event()
    chat = pool.chat_json
    def call(messages):
        started.set()
        return chat(messages)
    pool.chat_json = call
    monkeypatch.setattr(matrix,'provider_client',lambda *a:client)
    monkeypatch.setattr(matrix,'preflight_provider_models',lambda *a,**kw:[])
    monkeypatch.setattr(runner,'_validate_inputs',lambda args:{'bioavailability_ma':[{}]*17})
    monkeypatch.setattr(runner,'query_steps',lambda args,q,c:chain(c,str(q.index),[]))
    batches = []
    cache_configs = []
    def prepare(args, prepared_callback):
        if batches:
            assert started.wait(5), 'Inference blocked until all batches prepared'
        batches.append(args.indices)
        cache_configs.append(args.assay_transfer_cache)
        queries = []
        for i in args.indices:
            path = runner._query_dir(Path(args.output_root),'bioavailability_ma',i)
            path.mkdir(parents=True)
            queries.append(runner.PreparedQuery('bioavailability_ma',i,path))
        prepared_callback(queries,{}, {})
    monkeypatch.setattr(runner,'run',prepare)
    monkeypatch.setattr(matrix,'summarize',lambda *a:None)
    output=tmp_path/'rolling'
    assert matrix.main(['--output-root',str(output),'--rolling-from',str(source),
        '--tasks','bioavailability_ma','--reranking-modes','morgan','--record-pools','all',
        '--records-per-level','10','--parallelism','2',
        '--execution-mode','throughput','--trace-root',str(tmp_path/'traces')]) == 0
    assert batches == [list(range(16)),[16]] and len(pool.calls)==17
    assert cache_configs == [matrix.DEFAULT_CACHE_BUNDLE.resolve()] * 2
    links=list((output/'conditions').glob('bioavailability_ma/*/bioavailability_ma/queries/*'))
    assert len(links)==17 and all(p.is_symlink() and p.exists() for p in links)


def test_endpoint_allocations_respect_endpoint_caps():
    config = matrix.load_provider_pool_config(matrix.DEFAULT_PROVIDER_CONFIG)
    capacity = matrix.primary_capacity(config)
    allocations = matrix.endpoint_allocations(capacity, config)
    assert sum(slots for _, slots in allocations) == capacity
    assert [(provider.name, slots) for provider, slots in allocations] == [
        (provider.name, provider.max_inflight) for provider in config.providers
    ]
    with __import__('pytest').raises(ValueError, match=str(capacity)):
        matrix.endpoint_allocations(capacity + 1, config)


def test_deepseek_pool_allocates_all_endpoint_slots():
    path = Path(
        "predict/api_client/providers/"
        "progressive_deepseek_v4_flash_four_endpoint_high.json"
    )
    config = matrix.load_provider_pool_config(path)
    allocations = matrix.endpoint_allocations(1536, config)
    assert [(provider.name, slots) for provider, slots in allocations] == [
        ("dgx007_50002", 256),
        ("dgx011_50001", 256),
        ("dgx014_50001", 256),
        ("dgx014_50002", 256),
        ("dgx015_50001", 256),
        ("dgx018_50001", 256),
    ]


def test_matrix_tops_up_endpoint_load_after_preparation(tmp_path, monkeypatch):
    from predict.harnesses.progressive import diagnostics

    monkeypatch.setenv('TXAGENT_LIVE_PUBLISH', '0')
    pool = FakePool()
    client = AttemptClient(pool)
    monkeypatch.setattr(matrix, 'provider_client', lambda *args: client)
    def select_available(config, requested):
        active = replace(config, providers=config.providers[:-1])
        return ProviderSelection(
            config=active,
            requested_parallelism=requested,
            effective_parallelism=matrix.primary_capacity(active),
            checks=tuple(
                {'provider': provider.name,
                 'status': 'healthy' if provider in active.providers else 'unavailable'}
                for provider in config.providers
            ),
        )

    monkeypatch.setattr(matrix, 'select_healthy_providers', select_available)
    observed = [510, 511, 512]
    monkeypatch.setattr(matrix, 'sample_provider_loads', lambda config, **kwargs: [
        {'provider': provider.name, 'total': total}
        for provider, total in zip(config.providers, observed)
    ])
    monkeypatch.setattr(matrix, 'summarize', lambda *args: None)
    monkeypatch.setattr(diagnostics, 'build_run_diagnostics', lambda *args: None)
    monkeypatch.setattr(
        runner, 'query_steps',
        lambda args, query, active_client: chain(active_client, 'prepared', []),
    )

    def prepare(args, prepared_callback, candidate_loader, **kwargs):
        assert kwargs.get('prepared_query_callback') is None
        query_dir = Path(args.output_root) / 'query'
        query_dir.mkdir(parents=True)
        query = runner.PreparedQuery('bbb_martins', 0, query_dir)
        prepared_callback([query], {'bbb_martins': []}, {'bbb_martins': []})

    monkeypatch.setattr(runner, 'run', prepare)
    output = tmp_path / 'matrix'
    assert matrix.main([
        '--output-root', str(output), '--tasks', 'bbb_martins',
        '--harness-version', 'reranked-progressive-l1-context-v1',
        '--reranking-modes', 'assay-transfer-contrastive',
        '--record-pools', 'assay-transfer-trained', '--records-per-level', '50',
        '--l1-molecules', '10', '--l1-min-contrasts', '0',
        '--query-prior-modes', 'none', '--prompt-version',
        'reranked_progressive_l1_context_v3', '--max-level', '1',
        '--assay-transfer-cache',
        'predict/retrieval/assay_reranking/l1_context_morgan25_v2.yaml',
        '--provider-pool-config',
        'predict/api_client/providers/full_flat_progressive_v2_four_endpoint_512_high.json',
        '--target-total-load-per-endpoint', '512',
        '--execution-mode', 'throughput',
        '--trace-root', str(tmp_path / 'traces'),
    ]) == 0
    receipt = json.loads((output / 'matrix.json').read_text())
    assert receipt['parallelism'] == 3
    assert [row['launcher_slots'] for row in receipt['top_up_allocations']] == [
        2, 1, 0,
    ]
    assert (
        receipt['endpoint_preflight']['models']['checks'][-1]['status']
        == 'unavailable'
    )
    assert len(receipt['provider_pool']['providers']) == 2
    assert len(pool.calls) == 1


def test_matrix_expands_l1_molecule_counts(tmp_path, monkeypatch):
    monkeypatch.setenv('TXAGENT_LIVE_PUBLISH', '0')
    monkeypatch.setattr(matrix, 'preflight_provider_models', lambda *_: [])
    prepared = []

    def prepare(
        args, prepared_callback, candidate_loader,
        prepared_query_callback=None,
    ):
        prepared.append((Path(args.output_root).name, args.context_limit, args.live_method))
        prepared_callback([], {}, {})

    monkeypatch.setattr(runner, 'run', prepare)
    output = tmp_path/'matrix'
    assert matrix.main([
        '--output-root',str(output),'--tasks','bbb_martins',
        '--reranking-modes','morgan','assay-transfer',
        '--record-pools','all','--records-per-level','50',
        '--l1-molecules','3','5','7','--query-prior-modes','none',
        '--prompt-version','reranked_progressive_l1_simple_v12','--max-level','1',
        '--parallelism','2','--trace-root',str(tmp_path/'traces'),'--prepare-only',
    ]) == 0
    assert {row[1] for row in prepared} == {3, 5, 7}
    assert {row[0] for row in prepared} == {
        f'{mode}_all_records50_k{k}_no_query_prior'
        for mode in ('morgan', 'assay-transfer') for k in (3, 5, 7)
    }
    assert all(f'_k{k}_' in method for _, k, method in prepared)
    assert json.loads((output/'matrix.json').read_text())['l1_molecules'] == [3, 5, 7]


def test_l1_context_matrix_accepts_two_contrastive_modes(tmp_path, monkeypatch):
    monkeypatch.setenv('TXAGENT_LIVE_PUBLISH', '0')
    monkeypatch.setattr(matrix, 'preflight_provider_models', lambda *_: [])
    prepared = []

    def prepare(args, prepared_callback, candidate_loader):
        prepared.append((
            args.reranking, args.context_limit, args.l1_min_contrast,
            args.molecule_description_mode,
        ))
        prepared_callback([], {}, {})

    monkeypatch.setattr(runner, 'run', prepare)
    output = tmp_path / 'matrix'
    assert matrix.main([
        '--output-root', str(output), '--tasks', 'bioavailability_ma',
        '--harness-version', 'reranked-progressive-l1-context-v1',
        '--reranking-modes', 'assay-transfer-contrastive', 'morgan-contrastive',
        '--record-pools', 'assay-transfer-trained', '--records-per-level', '50',
        '--l1-molecules', '15', '--l1-min-contrasts', '3',
        '--prompt-version', 'reranked_progressive_l1_context_v4',
        '--molecule-description-modes', 'motif',
        '--max-level', '1', '--parallelism', '2',
        '--assay-transfer-cache',
        'predict/retrieval/assay_reranking/l1_context_morgan15_v2.yaml',
        '--trace-root', str(tmp_path / 'traces'), '--prepare-only',
    ]) == 0
    assert prepared == [
        ('assay-transfer-contrastive', 15, 3, 'motif'),
        ('morgan-contrastive', 15, 3, 'motif'),
    ]


def test_v10_4_context_matrix_combines_contrast_and_coarse_descriptions(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv('TXAGENT_LIVE_PUBLISH', '0')
    prepared = []
    cache_manifests = {}
    for task in ('bbb_martins', 'bioavailability_ma'):
        manifest = tmp_path / 'cache' / task / 'VERSION.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({
            'schema_version': 'l1_context_retrieval.v2',
            'status': 'complete',
            'task_id': task,
            'subset': 'valid',
            'morgan_primary_parent_width': 25,
        }))
        cache_manifests[task] = manifest
    bundle = tmp_path / 'cache.yaml'
    bundle.write_text(
        'version: 9\ncaches:\n'
        f'  bbb_martins:\n    valid: {cache_manifests["bbb_martins"]}\n'
        '  bioavailability_ma:\n'
        f'    valid: {cache_manifests["bioavailability_ma"]}\n'
    )

    def prepare(args, prepared_callback, candidate_loader):
        prepared.append((
            tuple(args.tasks), args.reranking, args.context_limit,
            args.l1_min_contrast, args.molecule_description_mode,
            args.molecule_description_cache_version, args.query_prior,
        ))
        prepared_callback([], {}, {})

    monkeypatch.setattr(runner, 'run', prepare)
    assert matrix.main([
        '--output-root', str(tmp_path / 'matrix'),
        '--tasks', 'bbb_martins', 'bioavailability_ma',
        '--harness-version', 'reranked-progressive-l1-context-v1',
        '--reranking-modes', 'assay-transfer-contrastive',
        '--record-pools', 'all', '--records-per-level', '50',
        '--l1-molecules', '10', '--l1-min-contrasts', '1', '2', '3',
        '--query-prior-modes', 'cached',
        '--prompt-version', 'reranked_progressive_l1_context_v4',
        '--molecule-description-modes', 'coarse',
        '--molecule-description-cache-version', 'v2',
        '--replicates', '1', '--max-level', '1', '--parallelism', '2',
        '--assay-transfer-cache', str(bundle),
        '--trace-root', str(tmp_path / 'traces'), '--prepare-only',
    ]) == 0
    assert prepared == [
        (
            ('bbb_martins', 'bioavailability_ma'),
            'assay-transfer-contrastive', 10, minimum,
            'coarse', 'v2', 'cached',
        )
        for minimum in (1, 2, 3)
    ]


def test_matrix_pools_replicates_with_endpoint_cap_and_task_exclusion(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv('TXAGENT_LIVE_PUBLISH', '0')
    prepared = []

    def prepare(args, prepared_callback, candidate_loader):
        prepared.append((
            args.query_prior,
            args.molecule_description_mode,
            tuple(args.tasks),
            args.request_namespace,
            args.molecule_description_cache_version,
        ))
        prepared_callback([], {}, {})

    monkeypatch.setattr(runner, 'run', prepare)
    output = tmp_path / 'matrix'
    capacity = sum(
        min(provider.max_inflight, 128)
        for provider in matrix.load_provider_pool_config(
            matrix.DEFAULT_PROVIDER_CONFIG
        ).providers
    )
    assert matrix.main([
        '--output-root', str(output),
        '--tasks', 'bbb_martins', 'bioavailability_ma',
        '--harness-version', 'reranked-progressive-l1-context-v1',
        '--reranking-modes', 'morgan',
        '--record-pools', 'assay-transfer-trained',
        '--records-per-level', '50',
        '--l1-molecules', '10', '--l1-min-contrasts', '0',
        '--query-prior-modes', 'cached', 'none',
        '--prompt-version', 'reranked_progressive_l1_context_v4',
        '--molecule-description-modes', 'none', 'coarse',
        '--molecule-description-cache-version', 'v2',
        '--replicates', '2',
        '--exclude-task-condition', 'bioavailability_ma', 'none', 'coarse',
        '--max-inflight-per-endpoint', '128', '--parallelism', str(capacity),
        '--max-level', '1', '--prepare-only',
        '--assay-transfer-cache',
        'predict/retrieval/assay_reranking/l1_context_morgan10_v2.yaml',
        '--trace-root', str(tmp_path / 'traces'),
    ]) == 0
    assert len(prepared) == 8
    assert {row[3] for row in prepared} == {'replicate:1', 'replicate:2'}
    assert {row[4] for row in prepared} == {'v2'}
    excluded = [
        row for row in prepared
        if row[0] == 'none' and row[1] == 'coarse'
    ]
    assert len(excluded) == 2
    assert all(row[2] == ('bbb_martins',) for row in excluded)
    assert all(
        row[2] == ('bbb_martins', 'bioavailability_ma')
        for row in prepared if row not in excluded
    )
    receipt = json.loads((output / 'matrix.json').read_text())
    assert receipt['replicates'] == 2
    assert receipt['excluded_task_conditions'] == [{
        'task': 'bioavailability_ma',
        'query_prior': 'none',
        'molecule_description_mode': 'coarse',
    }]
    assert receipt['max_inflight_per_endpoint'] == 128
    assert receipt['molecule_description_artifact']['cache_version'] == 'v2'
    assert receipt['molecule_description_artifact']['sha256'] == (
        runner.MOLECULE_DESCRIPTION_ARTIFACTS['v2']['sha256']
    )
    assert {
        provider['max_inflight']
        for provider in receipt['provider_pool']['providers']
    } == {128}


def test_matrix_expands_retrieval_widths_in_one_batch(tmp_path, monkeypatch):
    monkeypatch.setenv('TXAGENT_LIVE_PUBLISH', '0')
    monkeypatch.setattr(matrix, 'preflight_provider_models', lambda *_: [])
    prepared = []

    def prepare(args, prepared_callback, candidate_loader):
        prepared.append((
            Path(args.assay_transfer_cache).name,
            args.morgan_primary_parent_width,
            Path(args.output_root).name,
        ))
        prepared_callback([], {}, {})

    monkeypatch.setattr(runner, 'run', prepare)
    output = tmp_path / 'matrix'
    cache_root = 'predict/retrieval/assay_reranking'
    assert matrix.main([
        '--output-root', str(output), '--tasks', 'bbb_martins',
        '--harness-version', 'reranked-progressive-l1-context-v1',
        '--reranking-modes', 'assay-transfer-contrastive',
        '--record-pools', 'assay-transfer-trained', '--records-per-level', '50',
        '--l1-molecules', '10', '--l1-min-contrasts', '0',
        '--query-prior-modes', 'none',
        '--prompt-version', 'reranked_progressive_l1_context_v3',
        '--max-level', '1', '--parallelism', '2',
        '--assay-transfer-caches',
        f'{cache_root}/l1_context_morgan15_v2.yaml',
        f'{cache_root}/l1_context_morgan25_v2.yaml',
        '--trace-root', str(tmp_path / 'traces'), '--prepare-only',
    ]) == 0
    assert prepared == [
        (
            'l1_context_morgan15_v2.yaml', 15,
            'assay-transfer-contrastive_assay-transfer-trained_records50_'
            'k10_m0_no_query_prior_w15',
        ),
        (
            'l1_context_morgan25_v2.yaml', 25,
            'assay-transfer-contrastive_assay-transfer-trained_records50_'
            'k10_m0_no_query_prior_w25',
        ),
    ]
    receipt = json.loads((output / 'matrix.json').read_text())
    assert receipt['assay_transfer_cache'] is None
    assert [row['morgan_primary_parent_width']
            for row in receipt['assay_transfer_caches']] == [15, 25]


def test_matrix_explicit_variants_avoid_cartesian_conditions(tmp_path, monkeypatch):
    monkeypatch.setenv('TXAGENT_LIVE_PUBLISH', '0')
    monkeypatch.setattr(matrix, 'preflight_provider_models', lambda *_: [])
    prepared = []

    def prepare(args, prepared_callback, candidate_loader):
        prepared.append((
            args.reranking,
            args.morgan_primary_parent_width,
            args.assay_transfer_prompt_version,
            Path(args.output_root).name,
        ))
        prepared_callback([], {}, {})

    monkeypatch.setattr(runner, 'run', prepare)
    results = tmp_path / 'results'
    cache_root = 'predict/retrieval/assay_reranking'
    assert matrix.main([
        '--study', 'contrastive', '--results-root', str(results),
        '--batch-id', 'variants-1', '--tasks', 'bbb_martins',
        '--harness-version', 'reranked-progressive-l1-context-v1',
        '--variant', 'morgan-contrastive',
        f'{cache_root}/l1_context_morgan10_v2.yaml',
        'reranked_progressive_l1_context_v4',
        '--variant', 'assay-transfer-contrastive',
        f'{cache_root}/l1_context_morgan10_v2.yaml',
        'reranked_progressive_l1_context_v3',
        '--variant', 'assay-transfer-contrastive',
        f'{cache_root}/l1_context_morgan25_v2.yaml',
        'reranked_progressive_l1_context_v3',
        '--record-pools', 'assay-transfer-trained', '--records-per-level', '50',
        '--l1-molecules', '10', '--l1-min-contrasts', '3',
        '--query-prior-modes', 'cached', 'none', '--max-level', '1',
        '--parallelism', '2', '--trace-root', str(tmp_path / 'traces'),
        '--prepare-only',
    ]) == 0
    assert [row[:3] for row in prepared] == [
        ('morgan-contrastive', 10, 'reranked_progressive_l1_context_v4'),
        ('assay-transfer-contrastive', 10, 'reranked_progressive_l1_context_v3'),
        ('morgan-contrastive', 10,
         'reranked_progressive_l1_context_v4_no_query_prior'),
        ('assay-transfer-contrastive', 10,
         'reranked_progressive_l1_context_v3_no_query_prior'),
        ('assay-transfer-contrastive', 25, 'reranked_progressive_l1_context_v3'),
        ('assay-transfer-contrastive', 25,
         'reranked_progressive_l1_context_v3_no_query_prior'),
    ]
    assert prepared[0][3].startswith('k10_m3_')
    assert prepared[1][3].startswith('k10_m3_w10_')
    assert prepared[4][3].startswith('k10_m3_w25_')
    receipt = json.loads((results / '_batches/variants-1/matrix.json').read_text())
    assert len(receipt['variants']) == 3
    assert {row['prompt_versions']['cached'] for row in receipt['variants']} == {
        'reranked_progressive_l1_context_v3',
        'reranked_progressive_l1_context_v4',
    }


def test_throughput_matrix_starts_requests_before_next_width_prepares(tmp_path, monkeypatch):
    from predict.harnesses.progressive import diagnostics

    monkeypatch.setenv('TXAGENT_LIVE_PUBLISH', '0')
    monkeypatch.setattr(matrix, 'preflight_provider_models', lambda *_: [])
    pool = FakePool()
    client = AttemptClient(pool)
    started = threading.Event()
    original_chat = pool.chat_json

    def chat(messages):
        started.set()
        return original_chat(messages)

    pool.chat_json = chat
    monkeypatch.setattr(matrix, 'provider_client', lambda *args: client)
    monkeypatch.setattr(matrix, 'summarize', lambda *args: None)
    monkeypatch.setattr(diagnostics, 'build_run_diagnostics', lambda *args: None)
    monkeypatch.setattr(
        runner, 'query_steps',
        lambda args, query, active_client: chain(
            active_client, str(args.morgan_primary_parent_width), []
        ),
    )
    widths = []

    def prepare(
        args, prepared_callback, candidate_loader,
        prepared_query_callback=None,
    ):
        if widths:
            assert started.wait(5), 'Inference waited for all widths to prepare'
        widths.append(args.morgan_primary_parent_width)
        query_dir = Path(args.output_root) / f'query-{len(widths)}'
        query_dir.mkdir(parents=True)
        query = runner.PreparedQuery('bbb_martins', len(widths), query_dir)
        if prepared_query_callback is not None:
            prepared_query_callback(query)
        prepared_callback(
            [query],
            {'bbb_martins': []},
            {'bbb_martins': []},
        )

    monkeypatch.setattr(runner, 'run', prepare)
    cache_root = 'predict/retrieval/assay_reranking'
    assert matrix.main([
        '--output-root', str(tmp_path / 'matrix'), '--tasks', 'bbb_martins',
        '--harness-version', 'reranked-progressive-l1-context-v1',
        '--reranking-modes', 'assay-transfer-contrastive',
        '--record-pools', 'assay-transfer-trained', '--records-per-level', '50',
        '--l1-molecules', '10', '--l1-min-contrasts', '0',
        '--query-prior-modes', 'none',
        '--prompt-version', 'reranked_progressive_l1_context_v3',
        '--max-level', '1', '--parallelism', '2',
        '--assay-transfer-caches',
        f'{cache_root}/l1_context_morgan15_v2.yaml',
        f'{cache_root}/l1_context_morgan25_v2.yaml',
        '--execution-mode', 'throughput',
        '--trace-root', str(tmp_path / 'traces'),
    ]) == 0
    assert widths == [15, 25]
    assert len(pool.calls) == 2


@pytest.mark.parametrize(('level_args', 'levels'), [
    (['--indirect-levels', '2', '3', '4'], (2, 3, 4)),
    (['--indirect-level', '3'], (3,)),
])
def test_indirect_levels_share_one_streaming_queue(
    tmp_path, monkeypatch, level_args, levels,
):
    from predict.harnesses.progressive import diagnostics

    monkeypatch.setenv('TXAGENT_LIVE_PUBLISH', '0')
    pool = FakePool()
    client = AttemptClient(pool)
    started = threading.Event()
    original_chat = pool.chat_json

    def chat(messages):
        started.set()
        return original_chat(messages)

    pool.chat_json = chat
    monkeypatch.setattr(matrix, 'provider_client', lambda *args: client)
    monkeypatch.setattr(matrix, 'summarize', lambda *args: None)
    monkeypatch.setattr(diagnostics, 'build_run_diagnostics', lambda *args: None)
    monkeypatch.setattr(runner, 'query_steps', lambda args, query, active_client: chain(
        active_client, f'{args.indirect_level}-{args.reranking}', []
    ))
    prepared = []

    def prepare(args, prepared_callback, candidate_loader, prepared_query_callback=None):
        if prepared:
            assert started.wait(5), 'Inference waited for the next indirect level'
        prepared.append((args.indirect_level, args.reranking))
        query_dir = Path(args.output_root) / 'query'
        query_dir.mkdir(parents=True)
        query = runner.PreparedQuery('bbb_martins', len(prepared), query_dir)
        if prepared_query_callback is not None:
            prepared_query_callback(query)
        prepared_callback([query], {'bbb_martins': []}, {'bbb_martins': []})

    monkeypatch.setattr(runner, 'run', prepare)
    output = tmp_path / 'matrix'
    assert matrix.main([
        '--output-root', str(output), '--tasks', 'bbb_martins',
        '--harness-version', runner.INDIRECT_ONLY_HARNESS,
        *level_args,
        '--reranking-modes', 'morgan-parent-control', 'morgan-parent-semantic',
        '--record-pools', 'all', '--records-per-level', '25',
        '--l1-molecules', '10', '--l1-min-contrasts', '0',
        '--query-prior-modes', 'none', '--prompt-version',
        'reranked_progressive_l1_context_v4', '--max-tokens', '65536',
        '--provider-pool-config',
        'predict/api_client/providers/progressive_deepseek_v4_flash_four_endpoint_high.json',
        '--parallelism', '2', '--execution-mode', 'throughput',
        '--trace-root', str(tmp_path / 'traces'),
    ]) == 0
    assert prepared == [
        (level, mode) for level in levels
        for mode in ('morgan-parent-control', 'morgan-parent-semantic')
    ]
    receipt = json.loads((output / 'matrix.json').read_text())
    assert receipt['indirect_levels'] == list(levels)
    assert receipt['indirect_level'] == (levels[0] if len(levels) == 1 else None)
    assert len(pool.calls) == 2 * len(levels)


def test_context_l2_matrix_expands_molecule_description_axis(tmp_path, monkeypatch):
    monkeypatch.setenv('TXAGENT_LIVE_PUBLISH', '0')
    monkeypatch.setattr(matrix, 'preflight_provider_models', lambda *_: [])
    prepared = []

    def prepare(args, prepared_callback, candidate_loader):
        prepared.append((
            args.reranking, args.molecule_description_mode,
            Path(args.output_root).name, args.live_method,
        ))
        prepared_callback([], {}, {})

    monkeypatch.setattr(runner, 'run', prepare)
    output = tmp_path / 'matrix'
    assert matrix.main([
        '--output-root', str(output), '--tasks', 'bbb_martins',
        '--harness-version', 'reranked-progressive-l1-context-l2-v1',
        '--reranking-modes', 'morgan', 'semantic-lap',
        '--record-pools', 'all', '--records-per-level', '12',
        '--l1-molecules', '10', '--max-level', '2',
        '--prompt-version',
        'reranked_progressive_l1_context_l2_semantic_molecule_metadata_v1',
        '--molecule-description-modes', 'none', 'motif',
        '--parallelism', '2', '--trace-root', str(tmp_path / 'traces'),
        '--prepare-only',
    ]) == 0
    assert {(mode, description) for mode, description, _, _ in prepared} == {
        ('morgan', 'none'), ('morgan', 'motif'),
        ('semantic-lap', 'none'), ('semantic-lap', 'motif'),
    }
    assert all(f'_description_{description}' in name
               for _, description, name, _ in prepared)
    assert all(
        method.endswith('_description_none')
        if description == 'none' else method.endswith('_description_motif')
        for _, description, _, method in prepared
    )
    receipt = json.loads((output / 'matrix.json').read_text())
    assert receipt['molecule_description_modes'] == ['none', 'motif']


def test_l1_context_matrix_accepts_zero_and_maximum_contrast(tmp_path, monkeypatch):
    monkeypatch.setenv('TXAGENT_LIVE_PUBLISH', '0')
    monkeypatch.setattr(matrix, 'preflight_provider_models', lambda *_: [])
    prepared = []

    def prepare(args, prepared_callback, candidate_loader):
        prepared.append((Path(args.output_root).name, args.l1_min_contrast))
        prepared_callback([], {}, {})

    monkeypatch.setattr(runner, 'run', prepare)
    output = tmp_path / 'matrix'
    assert matrix.main([
        '--output-root', str(output), '--tasks', 'bbb_martins',
        '--harness-version', 'reranked-progressive-l1-context-v1',
        '--reranking-modes', 'assay-transfer-contrastive',
        '--record-pools', 'assay-transfer-trained', '--records-per-level', '50',
        '--l1-molecules', '10', '--l1-min-contrasts', '0', '4', '5',
        '--prompt-version', 'reranked_progressive_l1_context_v2',
        '--max-level', '1', '--parallelism', '2',
        '--assay-transfer-cache',
        'predict/retrieval/assay_reranking/l1_context_morgan25_v1.yaml',
        '--trace-root', str(tmp_path / 'traces'), '--prepare-only',
    ]) == 0
    assert prepared == [
        ('assay-transfer-contrastive_assay-transfer-trained_records50_k10_m0_query_prior', 0),
        ('assay-transfer-contrastive_assay-transfer-trained_records50_k10_m4_query_prior', 4),
        ('assay-transfer-contrastive_assay-transfer-trained_records50_k10_m5_query_prior', 5),
    ]


def test_l1_context_matrix_names_order_only_pair(tmp_path, monkeypatch):
    monkeypatch.setenv('TXAGENT_LIVE_PUBLISH', '0')
    monkeypatch.setattr(matrix, 'preflight_provider_models', lambda *_: [])
    prepared = []

    def prepare(args, prepared_callback, candidate_loader):
        prepared.append((Path(args.output_root).name, args.reranking))
        prepared_callback([], {}, {})

    monkeypatch.setattr(runner, 'run', prepare)
    output = tmp_path / 'matrix'
    assert matrix.main([
        '--output-root', str(output), '--tasks', 'bbb_martins',
        '--harness-version', 'reranked-progressive-l1-context-v1',
        '--reranking-modes', 'morgan', 'assay-transfer-within-morgan',
        '--record-pools', 'assay-transfer-trained', '--records-per-level', '50',
        '--l1-molecules', '10', '--prompt-version',
        'reranked_progressive_l1_context_order_only_v1', '--max-level', '1',
        '--parallelism', '2', '--assay-transfer-cache',
        'predict/retrieval/assay_reranking/l1_context_morgan100_v1.yaml',
        '--trace-root', str(tmp_path / 'traces'), '--prepare-only',
    ]) == 0
    assert prepared == [
        ('morgan_assay-transfer-trained_records50_k10_query_prior', 'morgan'),
        (
            'assay-transfer-within-morgan_assay-transfer-trained_records50_'
            'k10_query_prior',
            'assay-transfer-within-morgan',
        ),
    ]


def test_organized_matrix_splits_methods_into_study_leaves(tmp_path, monkeypatch):
    monkeypatch.setenv('TXAGENT_LIVE_PUBLISH', '0')
    monkeypatch.setattr(matrix, 'preflight_provider_models', lambda *_: [])
    prepared = []

    def prepare(args, prepared_callback, candidate_loader):
        prepared.append(Path(args.output_root))
        prepared_callback([], {}, {})

    monkeypatch.setattr(runner, 'run', prepare)
    results = tmp_path / 'results'
    assert matrix.main([
        '--study', 'contrastive', '--results-root', str(results),
        '--batch-id', 'batch-1', '--tasks', 'bbb_martins',
        '--harness-version', 'reranked-progressive-l1-context-v1',
        '--reranking-modes', 'morgan-contrastive', 'assay-transfer-contrastive',
        '--record-pools', 'assay-transfer-trained', '--records-per-level', '50',
        '--l1-molecules', '10', '--l1-min-contrasts', '0', '3',
        '--prompt-version', 'reranked_progressive_l1_context_v2_references_v1',
        '--max-level', '1',
        '--parallelism', '2', '--assay-transfer-cache',
        'predict/retrieval/assay_reranking/l1_context_morgan25_v1.yaml',
        '--trace-root', str(tmp_path / 'traces'), '--prepare-only',
        '--query-prior-modes', 'cached', 'none',
    ]) == 0
    assert {(path.parents[1].name, path.parent.name, tuple(path.name.split('_', 3)[:3]))
            for path in prepared} == {
        (method, prior, ('k10', f'm{minimum}', matrix.time.strftime('%Y%m%d')))
        for method in ('morgan', 'assay_transfer')
        for prior in ('with_query_prior', 'no_query_prior')
        for minimum in (0, 3)
    }
    receipts = json.loads((results / '_batches/batch-1/runs.json').read_text())
    assert len(receipts['runs']) == 8
    assert all((path / 'run.json').is_file() for path in prepared)
    catalog = json.loads((results / 'catalog.json').read_text())
    assert catalog['schema_version'] == 'progressive_results_catalog.v3'
    assert {row['query_prior'] for row in catalog['runs']} == {'cached', 'none'}
    ledger_path = results / catalog['results_ledger']['path']
    assert sha256_file(ledger_path) == catalog['results_ledger']['sha256']
    with ledger_path.open(newline='') as handle:
        ledger = list(csv.DictReader(handle, delimiter='\t'))
    assert len(ledger) == 8
    assert {row['status'] for row in ledger} == {'prepared'}
    assert {row['task'] for row in ledger} == {'bbb_martins'}
    assert {row['macro_f1'] for row in ledger} == {''}

    completed = prepared[0]
    run_document = json.loads((completed / 'run.json').read_text())
    (completed / 'run.json').write_text(json.dumps({
        **run_document, 'status': 'complete', 'metric_status': 'complete',
    }))
    metrics = completed / 'bbb_martins/levels/level_1/metrics.json'
    metrics.parent.mkdir(parents=True)
    metrics.write_text(json.dumps({
        'level': 1, 'n_total': 397, 'n_successful': 397,
        'n_failed_runs': 0, 'macro_f1': 0.75, 'accuracy': 0.8,
    }))
    matrix._refresh_results_catalog(results)
    with (results / 'results_ledger.tsv').open(newline='') as handle:
        ledger = list(csv.DictReader(handle, delimiter='\t'))
    row = next(row for row in ledger if row['path'] == completed.relative_to(results).as_posix())
    assert row['status'] == 'complete'
    assert row['level'] == '1'
    assert row['macro_f1'] == '0.75'
    assert row['metrics_sha256'] == sha256_file(metrics)

    flat = results / 'contrastive/flat/with_query_prior/k10_m0_flat'
    flat.mkdir(parents=True)
    (flat / 'run.json').write_text(json.dumps({
        **run_document, 'method': 'flat', 'run_id': flat.name,
        'status': 'complete', 'metric_status': 'complete',
    }))
    flat_metrics = flat / 'bbb_martins/bbb_martins__flat/metrics.json'
    flat_metrics.parent.mkdir(parents=True)
    flat_metrics.write_text(json.dumps({
        'n_total': 397, 'n_successful': 397, 'n_failed_runs': 0,
        'macro_f1': 0.8, 'accuracy': 0.82,
    }))
    matrix._refresh_results_catalog(results)
    with (results / 'results_ledger.tsv').open(newline='') as handle:
        ledger = list(csv.DictReader(handle, delimiter='\t'))
    flat_row = next(row for row in ledger if row['method'] == 'flat')
    assert flat_row['macro_f1'] == '0.8'
    assert flat_row['metrics_sha256'] == sha256_file(flat_metrics)


def test_staged_publication_is_complete_only_and_hash_validated(tmp_path):
    staging = tmp_path / 'local'
    results = tmp_path / 'results'
    batch = staging / '_batches/batch-1'
    source = staging / 'study/morgan/no_query_prior/k10_m0_test'
    destination = results / source.relative_to(staging)
    output = source / 'bbb_martins/queries/query_idx00000/levels/level_1/output.json'
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps({'status': 'ok'}))
    (source / 'run.json').write_text(json.dumps({
        'status': 'complete', 'metric_status': 'complete',
    }))
    (source / 'experiment_manifest.json').write_text(json.dumps({
        'evaluation_indices_by_task': {'bbb_martins': [0]},
    }))
    (source / 'diagnostics_manifest.json').write_text(json.dumps({'status': 'complete'}))
    for name in (
        'visible_evidence.tsv', 'neighborhood_label_mix.per_query.tsv',
        'neighborhood_label_mix.summary.tsv',
        'reasoning_reference_coverage.per_query.tsv',
        'reasoning_reference_coverage.summary.tsv', 'report.md',
    ):
        (source / name).write_text(f'{name}\n')
    batch.mkdir(parents=True)
    (batch / 'runs.json').write_text(json.dumps({'runs': [{
        'study': 'study', 'method': 'morgan', 'query_prior': 'none',
        'run_id': 'k10_m0_test',
        'path': str(destination), 'staging_path': str(source),
    }]}))
    (batch / 'matrix.json').write_text('{}\n')
    (batch / 'status.json').write_text(json.dumps({'status': 'incomplete'}))
    (batch / 'launcher.lock').touch()
    staged_traces = staging / 'live'
    canonical_traces = tmp_path / 'traces'
    live_run = (
        staged_traces / 'study/morgan/no_query_prior/k10_m0_test'
        / 'bbb_martins/condition'
    )
    live_run.mkdir(parents=True)
    (live_run / 'run.json').write_text(json.dumps({'status': 'complete'}))
    options = SimpleNamespace(
        results_root=results, staging_root=staging, batch_id='batch-1',
        trace_root=staged_traces, canonical_trace_root=canonical_traces,
    )
    with pytest.raises(ValueError, match='not complete'):
        matrix._publish_staged_study(options, batch)
    assert not destination.exists()

    (batch / 'status.json').write_text(json.dumps({'status': 'complete'}))
    matrix._publish_staged_study(options, batch)
    assert source.is_dir() and destination.is_dir()
    canonical_batch = results / '_batches/batch-1'
    assert not (canonical_batch / 'launcher.lock').exists()
    publication = json.loads((canonical_batch / 'publication.json').read_text())
    assert publication['status'] == 'complete'
    assert publication['runs'][0]['l1_outputs'] == 1
    assert publication['live_runs'][0]['task_runs'] == 1
    assert (
        canonical_traces / 'study/morgan/no_query_prior/k10_m0_test'
        / 'bbb_martins/condition/run.json'
    ).is_file()
    assert publication['runs'][0]['run_json_sha256'] == sha256_file(
        destination / 'run.json'
    )


def test_staged_run_validates_declared_indirect_level(tmp_path):
    source = tmp_path / 'run'
    output = source / 'bbb_martins/queries/query_idx00000/levels/level_3/output.json'
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps({'status': 'ok'}))
    (source / 'run.json').write_text(json.dumps({
        'status': 'complete', 'metric_status': 'complete', 'indirect_level': 3,
    }))
    (source / 'experiment_manifest.json').write_text(json.dumps({
        'evaluation_indices_by_task': {'bbb_martins': [0]},
    }))
    (source / 'diagnostics_manifest.json').write_text(json.dumps({'status': 'complete'}))
    for name in (
        'visible_evidence.tsv', 'neighborhood_label_mix.per_query.tsv',
        'neighborhood_label_mix.summary.tsv',
        'reasoning_reference_coverage.per_query.tsv',
        'reasoning_reference_coverage.summary.tsv', 'report.md',
    ):
        (source / name).write_text(f'{name}\n')
    receipt = matrix._validate_staged_run(source)
    assert receipt['l3_outputs'] == 1
    assert 'l1_outputs' not in receipt


def test_v15_matrix_records_tokenized_transport_preflight(tmp_path, monkeypatch):
    monkeypatch.setenv('TXAGENT_LIVE_PUBLISH', '0')
    monkeypatch.setattr(matrix, 'preflight_provider_models', lambda *_: ['model-ok'])
    monkeypatch.setattr(
        matrix,
        'preflight_sglang_tokenized_completion',
        lambda *_: ['transport-ok'],
    )
    monkeypatch.setattr(
        runner,
        'run',
        lambda args, prepared_callback, candidate_loader: prepared_callback([], {}, {}),
    )
    output = tmp_path / 'matrix'
    assert matrix.main([
        '--output-root', str(output), '--tasks', 'bbb_martins',
        '--reranking-modes', 'morgan', '--record-pools', 'all',
        '--records-per-level', '50', '--l1-molecules', '3',
        '--query-prior-modes', 'cached',
        '--prompt-version', 'reranked_progressive_l1_simple_v15',
        '--max-level', '1', '--parallelism', '2',
        '--trace-root', str(tmp_path / 'traces'), '--prepare-only',
    ]) == 0
    receipt = json.loads((output / 'matrix.json').read_text())
    assert receipt['settings']['reasoning_transport'] == 'sglang_tokenized_completion.v1'
    assert receipt['settings']['response_format'] is None
    assert receipt['endpoint_preflight'] == {
        'models': None, 'tokenized_reasoning': None
    }


def test_matrix_names_composed_visibility_and_prior_variants(tmp_path, monkeypatch):
    monkeypatch.setenv('TXAGENT_LIVE_PUBLISH', '0')
    monkeypatch.setattr(matrix, 'preflight_provider_models', lambda *_: [])
    prepared = []

    def prepare(args, prepared_callback, candidate_loader):
        prepared.append((args.query_prior, args.prompt_version, args.live_method))
        prepared_callback([], {}, {})

    monkeypatch.setattr(runner, 'run', prepare)
    output = tmp_path/'matrix'
    assert matrix.main([
        '--output-root', str(output), '--tasks', 'bbb_martins',
        '--reranking-modes', 'morgan', '--record-pools', 'all',
        '--records-per-level', '50', '--l1-molecules', '5',
        '--query-prior-modes', 'cached', 'none',
        '--prompt-version',
        'reranked_progressive_l1_simple_v12_no_smiles_no_query_smiles',
        '--max-level', '1', '--parallelism', '2',
        '--trace-root', str(tmp_path/'traces'), '--prepare-only',
    ]) == 0
    assert prepared == [
        (
            'cached',
            'reranked_progressive_l1_simple_v12_no_smiles_no_query_smiles',
            'progressive_morgan_all_records50_k5_'
            'no_smiles_no_query_smiles_query_prior',
        ),
        (
            'none',
            'reranked_progressive_l1_simple_v12_no_smiles_no_query_smiles_no_query_prior',
            'progressive_morgan_all_records50_k5_'
            'no_smiles_no_query_smiles_no_query_prior',
        ),
    ]


def test_matrix_threads_v4_joint_contract_and_score_visibility(tmp_path, monkeypatch):
    monkeypatch.setenv('TXAGENT_LIVE_PUBLISH', '0')
    monkeypatch.setattr(matrix, 'preflight_provider_models', lambda *_: [])
    prepared = []

    def prepare(args, prepared_callback, candidate_loader):
        prepared.append((args.harness_version, args.joint_panel_sizes, args.live_method))
        prepared_callback([], {}, {})

    monkeypatch.setattr(runner, 'run', prepare)
    output = tmp_path / 'matrix'
    with pytest.raises(ValueError, match='does not support joint'):
        matrix.main([
            '--output-root', str(output), '--tasks', 'bbb_martins',
            '--harness-version', 'reranked-progressive-v4',
            '--reranking-modes', 'joint',
            '--record-pools', 'all', '--records-per-level', '50',
            '--l1-molecules', '10', '--query-prior-modes', 'cached', 'none',
            '--prompt-version', 'reranked_progressive_l1_simple_v13',
            '--max-level', '1', '--parallelism', '2',
            '--trace-root', str(tmp_path / 'traces'), '--prepare-only',
            '--execution-mode', 'throughput', '--publish-review-traces',
        ])
    assert prepared == []


def test_records_only_visibility_is_named_in_live_method(tmp_path, monkeypatch):
    monkeypatch.setenv('TXAGENT_LIVE_PUBLISH', '0')
    monkeypatch.setattr(matrix, 'preflight_provider_models', lambda *_: [])
    methods = []

    def prepare(args, prepared_callback, candidate_loader):
        methods.append(args.live_method)
        prepared_callback([], {}, {})

    monkeypatch.setattr(runner, 'run', prepare)
    assert matrix.main([
        '--output-root', str(tmp_path / 'matrix'), '--tasks', 'bbb_martins',
        '--reranking-modes', 'morgan', '--record-pools', 'all',
        '--records-per-level', '50', '--l1-molecules', '5',
        '--query-prior-modes', 'cached',
        '--prompt-version',
        'reranked_progressive_l1_simple_v13_no_scores_no_smiles_no_query_smiles',
        '--max-level', '1', '--parallelism', '2',
        '--trace-root', str(tmp_path / 'traces'), '--prepare-only',
    ]) == 0
    assert methods == [
        'progressive_morgan_all_records50_k5_'
        'no_scores_no_smiles_no_query_smiles_query_prior'
    ]


def test_retry_budget_is_bounded_per_resume_and_failed_results_not_reused(tmp_path):
    pool=FakePool(failures=100)
    client=AttemptClient(pool,max_attempts=3)
    result=schedule([chain(client,'root',[]),chain(client,'root',[])],client,{},tmp_path,
                    parallelism=2,retry_delay=0)
    assert len(pool.calls)==3
    assert len(result)==2 and all(r['status']=='error' for r in result)
    assert not [p for p in (tmp_path/'responses').rglob('*.json') if not p.name.endswith('.attempts.json')]
    schedule([chain(client,'root',[])],client,{},tmp_path,parallelism=2,retry_delay=0)
    assert len(pool.calls)==6
    ledger=next((tmp_path/'responses').rglob('*.attempts.json'))
    assert json.loads(ledger.read_text())['attempts']==6


def test_transient_failure_retries_once_for_shared_consumers(tmp_path):
    pool=FakePool(failures=1)
    client=AttemptClient(pool)
    result=schedule([chain(client,'root',[]),chain(client,'root',[])],client,{},tmp_path,
                    parallelism=2,retry_delay=0)
    assert len(pool.calls)==2
    assert all(r['status']=='ok' for r in result)


def test_real_stage_iterator_preserves_citations_and_prior(tmp_path, monkeypatch):
    args=runner.parse_args(['--tasks','bbb_martins','--reranking','morgan','--max-level','2',
                           '--query-prior','none','--output-root',str(tmp_path/'condition')])
    args.trace_root=str(tmp_path/'traces')
    policy=args.retrieval_policies['bbb_martins']
    by_level={'L1':{'a':['r1']},'L2':{'a':['r2']}}
    molecules,later=select_policy_records(by_level,{k:list(v) for k,v in by_level.items()},
        {'a':.5},{},policy['stages'],task='bbb_martins',query_id='q',molecule_limit=1,later_limit=1)
    molecules[0]['canonical_smiles']='CCO'
    for row in molecules[0]['l1_records']+later['L2']['records']:
        row['payload']={'canonical_smiles':'CCO','source_fields':{'support_text':'Experimental observation'}}
    monkeypatch.setattr(runner,'_prefetch_analog_tools',lambda **kwargs: ({},[]))
    prepared=runner._prepare_context_record_query(task='bbb_martins',query_index=0,
        record={'drug':'CCN'},contexts=molecules,levels=runner._run_levels(args,'bbb_martins'),
        output_root=tmp_path/'condition',single_root=tmp_path,query_prior_mode='none',
        record_limit=10,l2_record_limit=10,indirect_record_limit=1,
        indirect_records=later,prompt_version=args.assay_transfer_prompt_version,
        retrieval_policy=policy)
    contract=runner._task_contract('bbb_martins',args.assay_transfer_prompt_version)

    class EvidencePool(FakePool):
        def chat_json(self,messages):
            self.calls.append(messages)
            payload=json.loads(messages[1]['content'])
            alias=payload['active_evidence'][0]['evidence_cards'][0]['card_id']
            return {'content':{contract.prediction_field:contract.positive_prediction,
                'confidence':'moderate','revision_action':'initial' if len(self.calls)==1 else 'keep',
                'supportive_card_ids':[alias],'contradictory_card_ids':[],
                'prediction_basis_card_ids':[alias],
                'claims':[{'claim':'Observed evidence','card_ids':[alias]}],
                'new_evidence_assessment':[],'evidence_gaps':[],'decision_summary':'Supported by evidence'}}
    pool=EvidencePool()
    client=AttemptClient(pool)
    result=schedule([runner.query_steps(args,prepared,client)],client,{},tmp_path/'inference',
                    parallelism=2,retry_delay=0)
    assert result[0]['status']=='ok'
    assert len(pool.calls)==2
    assert 'prior_state' in json.loads(pool.calls[1][1]['content'])
    output=json.loads((prepared.query_dir/'levels/level_2/output.json').read_text())
    assert output['state']['claims'][0]['card_ids'][0] in output['card_alias_map'].values()
