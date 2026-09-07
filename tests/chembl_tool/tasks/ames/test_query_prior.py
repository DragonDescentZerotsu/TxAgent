import json
from types import SimpleNamespace

import pytest

from tools.chembl_tool.tasks.ames.query_prior import build_query_prior_messages
from tools.chembl_tool.paper_experiments import run_conditioned_assay_progressive_curve as runner
from tools.chembl_tool.common import identity_blind


def test_ames_paths_and_prior_condition_visibility():
    for scheme in ('scaffold', 'random'):
        spec = runner._progressive_task_specs(scheme)['ames']
        assert str(spec.index).endswith(f'indices/ames/{scheme}/assay_neighbor_index.pkl')
    query = {'canonical_smiles': 'CC', 'external_condition': 'strain panel: TA98'}
    from tools.chembl_tool.common.reasoning_payload import llm_query_payload, llm_evidence_query_payload
    single = json.loads(build_query_prior_messages('single', llm_query_payload(query))[1]['content'])
    final = json.loads(build_query_prior_messages('final', llm_evidence_query_payload(query), {'confidence': 'low'})[1]['content'])
    assert 'external_condition' not in single['query']
    assert final['query']['external_condition'] == query['external_condition']
    assert final['required_json_schema']['ames_prediction'] == 'positive | negative'
    assert 'Y' not in final


@pytest.mark.parametrize('race_width', [1, 6])
def test_fresh_priors_resume_and_reject_changed_inputs(tmp_path, monkeypatch, race_width):
    source = tmp_path / 'valid.jsonl'
    source.write_text(json.dumps({'drug':'CC', 'Y':1, 'molecule_identity_key':'parent', 'condition_group':'metabolic_activation=absent+strain_panel=TA98'})+'\n')
    args = SimpleNamespace(output_root=str(tmp_path / 'run'), prepare_only=False, query_prior_source_root='', tasks=['ames'], limit=0, indices=None, model=runner.MODEL, base_url=runner.BASE_URL, max_tokens=20480, parallelism=2, endpoint_concurrency_budget=2, tool_service_url='http://unused', timeout_s=1, parallelism_per_task=0, max_stage_requeues=0, retry_delay_s=0)
    calls=[]
    args.retry_race_width = race_width
    class Client:
        def chat_json(self, messages):
            payload=json.loads(messages[1]['content']);calls.append(payload)
            content={k: [] if isinstance(v,list) else 'text' for k,v in payload['required_json_schema'].items()}
            content.update(confidence='low', ames_prior='mixed_or_unclear', ames_prediction='negative')
            return {'content':content, 'messages':messages}
    races=[]
    class Pool:
        async def async_chat_json(self, messages):
            if not calls:
                calls.append({'query':{}, 'invalid_first_attempt':True})
                return {'content':{}}
            return Client().chat_json(messages)

        async def aclose(self):
            pass

    def make_client(*unused):
        if race_width == 1:
            return Client()
        client = runner.ParallelRetryClient(Pool(), parallelism=2, task_limits={'ames':2})
        races.append(client)
        return client
    monkeypatch.setattr(runner, '_make_client', make_client)
    def prefetch(retrieval, *a, **kw):
        retrieval['query']['prefetched_molecule_properties']={'status':'ok','tool_name':'molecule_properties','text':'MW: 30.07'}
        return retrieval
    monkeypatch.setattr(identity_blind,'prepare_reasoning_retrieval',prefetch)
    specs={'ames':SimpleNamespace(input_jsonl=source)}
    assert runner._prepare_fresh_query_priors(args,specs,None)==0
    count = len(calls)
    assert count >= 2
    if race_width > 1:
        receipts = list((tmp_path / 'run').glob('single_cache/ames/none/runs/*/retry_races/*/*.json'))
        assert any(json.loads(p.read_text())['width'] == 6 for p in receipts)
        assert all(not client.thread.is_alive() for client in races)
    manifest_path = tmp_path / 'run/single_cache/ames/none/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['runner_sha256'] = 'previous operational runner'
    manifest_path.write_text(json.dumps(manifest))
    args.parallelism, args.endpoint_concurrency_budget = 4, 8
    assert runner._prepare_fresh_query_priors(args,specs,None)==0
    assert len(calls)==count
    resumed = json.loads(manifest_path.read_text())
    assert (resumed['parallelism'], resumed['endpoint_concurrency_budget']) == (4, 8)
    assert resumed['execution_history'][-1]['runner_sha256'] == 'previous operational runner'
    assert resumed['execution_history'][-1]['parallelism'] == 2
    assert resumed['execution_history'][-1]['endpoint_concurrency_budget'] == 2
    # An interrupted run keeps single; only the missing None final is called again.
    final = next((tmp_path / 'run').glob('single_cache/ames/none/runs/*/final_reasoning_output.json'))
    single = final.with_name('single_molecule_reasoning_output.json')
    single_bytes = single.read_bytes()
    final.unlink()
    args.parallelism, args.endpoint_concurrency_budget = 8, 16
    assert runner._prepare_fresh_query_priors(args,specs,None)==0
    assert len(calls) == count + 1
    assert single.read_bytes() == single_bytes
    assert json.loads(final.read_text())['status'] == 'ok'
    args.max_tokens += 1
    with pytest.raises(ValueError,match='changed fresh prior'):
        runner._prepare_fresh_query_priors(args,specs,None)
    args.max_tokens -= 1
    assert all('Y' not in p and 'Y' not in p['query'] for p in calls)
    source.write_text(source.read_text().replace('"Y": 1','"Y": 0'))
    with pytest.raises(ValueError,match='changed fresh prior'):
        runner._prepare_fresh_query_priors(args,specs,None)
