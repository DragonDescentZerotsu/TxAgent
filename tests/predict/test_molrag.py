"""The single MolRAG evidence/low-effort contract preserves frozen requests and results."""

from argparse import Namespace
import threading

import pytest
from rdkit import Chem, DataStructs

from predict.baselines import molrag as m
from predict.utils.json import read_json, write_json_atomic, write_jsonl_atomic, sha256_file


def row(uid, smiles, label=0, group='species=human'):
    return dict(benchmark_row_id=uid, drug=smiles, Y=label, condition_group=group,
                full_record='SECRET_EXPERIMENT_DESCRIPTION', citation='SECRET_CITATION')


def test_explicit_terminal_answers():
    assert m.parse_answer('Some examples say NO.\nQuestion Answer:"YES".') == 1
    assert m.parse_answer('Question Answer: <NO>') == 0
    assert m.parse_answer('Question Answer: " <YES> " .') == 1
    assert m.parse_answer('{"type":"text","text":"Explanation.\\nQuestion Answer: YES"}') == 1
    assert m.parse_answer('{"summary":"Explanation.","Question Answer":"NO"}') == 0
    assert m.parse_answer('{"type":"text":"Explanation.\\nQuestion Answer: YES"}') == 1
    for bad in ('YES', 'Question Answer: YES or NO', 'Question Answer: YES\nadditional text',
                '{}', '{"text":"YES"}', '{"Question Answer":"YES","confidence":0.8}',
                'Question Answer: YES\nQuestion Answer: NO',
                '{"earlier":"Question Answer: NO","Question Answer":"YES"}',
                '{"text":"Question Answer: YES or NO"}'):
        with pytest.raises(ValueError):
            m.parse_answer(bad)
def test_wrong_model_stops_without_accepting_prediction(tmp_path):
    class Client:
        def chat_json(self, messages):
            return dict(raw_content='Question Answer: YES', model='wrong-model', usage=dict(cost=.02))

    request = dict(uid='q', request_hash='r', messages=[], examples=[{}] * 4)
    stop = threading.Event()
    entry = (tmp_path, dict(model=m.MODEL, max_attempts=2, generation=m.generation_settings(m.MODEL), retry_token_exhaustion=False), {}, request, tmp_path / 'query')
    assert m.execute_query(entry, Client(), stop) == 'fatal_error'
    assert stop.is_set() and not (entry[-1] / 'output.json').exists()
    assert read_json(entry[-1] / 'attempt_0001.json')['llm']['usage']['cost'] == .02


def test_four_shot_is_required_before_model_calls(tmp_path):
    class Client:
        calls = 0
        def chat_json(self, messages):
            self.calls += 1
            return dict(raw_content='Question Answer: NO', model=m.MODEL)
    client = Client()
    request = dict(uid='q', request_hash='r', messages=[], examples=[{}])
    entry = (tmp_path, dict(model=m.MODEL, max_attempts=2), {}, request, tmp_path / 'query')
    assert m.execute_query(entry, client, threading.Event()) == 'insufficient_neighbors'
    assert client.calls == 0


def test_retry_limits_survive_resume_and_token_exhaustion(tmp_path, monkeypatch):
    monkeypatch.setattr(m.time, 'sleep', lambda _: None)
    class Client:
        calls = 0
        tokens = 10
        def chat_json(self, messages):
            self.calls += 1
            return dict(raw_content='unfinished', model=m.MODEL, usage=dict(cost=.01, completion_tokens=self.tokens))
    client = Client()
    request = dict(uid='q', request_hash='r', messages=[], examples=[{}] * 4)
    manifest = dict(model=m.MODEL, max_attempts=2,
                    retry_token_exhaustion=False, generation=dict(max_tokens=8192))
    entry = (tmp_path, manifest, {}, request, tmp_path / 'query')
    assert m.execute_query(entry, client, threading.Event()) == 'error'
    assert m.execute_query(entry, client, threading.Event()) == 'error'
    assert client.calls == 2
    client.tokens = 8190
    entry = (*entry[:-1], tmp_path / 'truncated')
    assert m.execute_query(entry, client, threading.Event()) == 'error'
    assert m.execute_query(entry, client, threading.Event()) == 'error'
    assert client.calls == 3
    class BudgetClient:
        def chat_json(self, messages):
            raise RuntimeError('OpenRouter spend budget exhausted')
    stop = threading.Event()
    assert m.execute_query((*entry[:-1], tmp_path / 'budget'), BudgetClient(), stop) == 'budget_stopped'
    assert stop.is_set()


def test_failure_penalty_scores_both_labels_without_changing_predictions(tmp_path):
    root = tmp_path / 'run'
    paths, originals = [], {}
    for task in ('bbb', 'oral', 'skin', 'ames', 'dili', 'carcinogens'):
        batch = root / task
        (batch / 'inputs').mkdir(parents=True)
        gold = [row(str(i), 'CCO', label) for i, label in enumerate((0, 1, 0, 1))]
        gold_path = batch / 'inputs/test.jsonl'
        write_jsonl_atomic(gold_path, gold)
        m.write_batch(batch / 'manifest.json', dict(model=m.MODEL, benchmark='tdc', task=task, subset='test', prompt=m.PROMPT,
                           protocol='molrag_direct_evidence.v1', generation=m.generation_settings(m.MODEL)),
                      records=list(enumerate(gold)))
        write_json_atomic(batch / 'receipt.json', {'source': {'splits': {'test': {'sha256': sha256_file(gold_path)}}}})
        predictions = [dict(uid=q['benchmark_row_id'], label=q['Y'],
                            prediction=q['Y'] if i < 2 else None, status='ok' if i < 2 else 'error')
                       for i, q in enumerate(gold)]
        write_jsonl_atomic(batch / 'predictions.jsonl', predictions)
        originals[batch / 'predictions.jsonl'] = (batch / 'predictions.jsonl').read_bytes()
        paths.append(task)
    write_json_atomic(root / 'suite.json', dict(batches=paths))
    for task in paths:
        batch = root / task
        requests = [dict(uid=str(i), request_hash=str(i),
                         messages=[dict(role='user', content='Predict the molecule.')]) for i in range(4)]
        write_jsonl_atomic(batch / 'requests.jsonl', requests)
        receipt = read_json(batch / 'receipt.json')
        receipt.update(manifest_sha256=sha256_file(batch / 'manifest.json'),
                       requests_sha256=sha256_file(batch / 'requests.jsonl'))
        write_json_atomic(batch / 'receipt.json', receipt)
        for i, text in [(2, '{"Question Answer":"NO"}'), (3, 'Question Answer: YES or NO')]:
            attempt = batch / 'runs' / f'query_{i:05d}' / 'attempt_0001.json'
            write_json_atomic(attempt, dict(**requests[i], status='error',
                llm=dict(model=m.MODEL, raw_content=text, messages=requests[i]['messages'])))
            originals[attempt] = attempt.read_bytes()
    reparsed = m.score_saved_runs([root], tmp_path / 'reparsed')
    assert reparsed['means'][0]['accuracy'] == .75
    assert reparsed['means'][0]['n_failures_counted_wrong'] == 6
    assert all(c['n_reparsed_answers'] == 1 for c in reparsed['cells'])
    recovered = [r for r in m.read_jsonl(tmp_path / 'reparsed/scored_predictions.jsonl') if 'reparsed_prediction' in r]
    assert len(recovered) == 6
    assert all(r['prediction'] is None and r['status'] == 'error' and r['scoring_prediction'] == 0
               and not r['failure_counted_wrong'] and r['reparsed_attempt_sha256'] for r in recovered)
    assert all(path.read_bytes() == before for path, before in originals.items())
    requests_path = batch / 'requests.jsonl'
    original_requests = requests_path.read_bytes()
    requests_path.write_bytes(original_requests + b'\n')
    with pytest.raises(ValueError, match='Frozen reparse'):
        m.score_saved_runs([root], tmp_path / 'changed_request')
    requests_path.write_bytes(original_requests)
    bad_attempt = batch / 'runs/query_00002/attempt_0001.json'
    changed = read_json(bad_attempt)
    changed['llm']['messages'] = [dict(role='user', content='A different query.')]
    write_json_atomic(bad_attempt, changed)
    with pytest.raises(ValueError, match='response messages'):
        m.score_saved_runs([root], tmp_path / 'message_mismatch')
    changed['request_hash'] = 'another-query'
    write_json_atomic(bad_attempt, changed)
    with pytest.raises(ValueError, match='different request'):
        m.score_saved_runs([root], tmp_path / 'mismatch')
    bad_attempt.write_bytes(originals[bad_attempt])
    predictions[-1]['status'] = 'pending'
    write_jsonl_atomic(batch / 'predictions.jsonl', predictions)
    with pytest.raises(ValueError, match='Only explicit failed'):
        m.score_saved_runs([root], tmp_path / 'pending_rejected')


@pytest.mark.parametrize('model', [m.MODEL, 'openai/gpt-6-luna', 'google/gemini-3.8-flash'])
def test_evidence_preparation_execution_and_resume(tmp_path, monkeypatch, model):
    from predict.baselines.molrag_evidence import build_library
    from predict.utils.json import read_jsonl
    task = 'bbb_martins'
    heldout = tmp_path / 'heldout.jsonl'
    train = tmp_path / 'train.jsonl'
    write_jsonl_atomic(heldout, [row('test', 'c1ccccc1', 1)])
    train_rows = [dict(row('label', 'CCN', 1), tdc_dataset='BBB_Martins'),
                  dict(row('overlap', 'CCO', 0), tdc_dataset='BBB_Martins')]
    write_jsonl_atomic(train, train_rows)
    card = dict(endpoint_type='brain uptake', reported_value='0.5', reported_units='ratio',
                assay_context='brain perfusion', species_context='rat', qualifying_conditions='dose=1 mg/kg',
                support_text='Observed brain uptake in rats.', evidence_family='direct_brain_exposure', evidence_family_level=1)
    evidence = tmp_path / 'evidence.jsonl'
    write_jsonl_atomic(evidence, [dict(canonical_smiles=smiles, assay_chembl_id='assay',
        source_record_examples=cards) for smiles, cards in [
        ('OCC', [card, {**card, 'reported_value': '99'}]),
        ('CCCC', [card]), ('CC(=O)O', [card]),
        ('c1ccccc1', [card]), ('CCC', [{**card, 'evidence_family_level': 2}])]])
    source = tmp_path / 'naive'
    source.mkdir()
    inputs = dict(index=dict(path='index.pkl', sha256='frozen-index'),
                  input=dict(path=str(heldout), sha256=sha256_file(heldout)),
                  train=dict(path=str(train), sha256=sha256_file(train)),
                  heldout=dict(path=str(heldout), sha256=sha256_file(heldout)))
    write_json_atomic(source / 'experiment_manifest.json', dict(split_scheme='scaffold', setting='direct', inputs={task: inputs}))
    write_json_atomic(tmp_path / 'manifest.json', dict(index_sha256='frozen-index',
        n_direct_heldout_records_after_filter=0, heldout_molecules_jsonl=str(heldout),
        evidence=str(evidence), evidence_sha256=sha256_file(evidence)))
    sources = tmp_path / 'sources.json'
    write_json_atomic(sources, dict(release='test', datasets=[dict(benchmark='tdc', task=task, batch='naive')]))
    output = tmp_path / 'library'
    build_library(sources, tmp_path, output)
    references = read_jsonl(output / 'tdc' / task / 'references.jsonl')
    assert {r['smiles'] for r in references} == {'CCO', 'CCN', 'CCCC', 'CC(=O)O'}
    experimental = next(r for r in references if r['smiles'] == 'CCO')
    assert experimental['card'] == card and experimental['provenance']['line'] == 1
    assert experimental['provenance']['source_row_uid'] is None  # Never invent a physical UID.
    fps = [m.GENERATOR.GetFingerprint(Chem.MolFromSmiles(r['smiles'])) for r in references]
    query = row('q', 'c1ccccc1', 1, 'species=human')
    selected = m.select_examples(query, references, fps)
    assert len(selected) == 4  # Raw source conditions do not become benchmark condition groups.
    query_fp = m.GENERATOR.GetFingerprint(Chem.MolFromSmiles(query['drug']))
    ref_fp = m.GENERATOR.GetFingerprint(Chem.MolFromSmiles(selected[0]['smiles']))
    assert selected[0]['dice'] == DataStructs.DiceSimilarity(query_fp, ref_fp)
    task_fields = dict(endpoint_name='BBB', positive_prediction='permeable', negative_prediction='not permeable')
    messages = m.render(query, selected, task_fields)
    assert messages == m.render({**query, 'Y': 0}, selected, task_fields)
    text = messages[-1]['content']
    assert 'Observed brain uptake in rats.' in text and 'Label: 1' in text
    assert 'species=human' in text and 'Species context: rat' in text
    assert 'source_row_uid' not in text and '99' not in text and text.count('SMILES:') == 4
    # A same-scaffold query loses that reference even when its Dice similarity is highest.
    ring = m.normalize_molecule_identity('c1ccccc1O')
    additional = dict(experimental, reference_id='ring', smiles=ring.canonical_smiles,
                      canonical_smiles=ring.canonical_smiles, molecule_identity=ring.to_dict())
    ring_fp = m.GENERATOR.GetFingerprint(Chem.MolFromSmiles(additional['smiles']))
    selected = m.select_examples(query, [additional, *references], [ring_fp, *fps])
    assert all(r['reference_id'] != 'ring' for r in selected)
    with pytest.raises(ValueError, match='four eligible'):
        m.select_examples(query, [additional], [ring_fp])
    cohorts = tmp_path / 'cohorts.json'
    write_json_atomic(cohorts, dict(release='test', datasets=[dict(benchmark='tdc', task=task,
        splits={name: dict(path=str(path), sha256=sha256_file(path))
                for name, path in [('train', train), ('valid', heldout), ('test', heldout)]})]))
    provider = tmp_path / 'provider.json'
    config = dict(version='openai_provider_pool.v1', max_failovers=0, providers=[dict(
        name='test', base_url='https://openrouter.ai/api/v1', model=model, api_key_env='TEST_KEY',
        max_inflight=1, request_extra_body=dict(reasoning_effort_override='low'))])
    write_json_atomic(provider, config)
    args = Namespace(output_root=tmp_path / 'runs', run_name='evidence', sources=cohorts,
        source_root=tmp_path, provider_config=provider, benchmark=None, task=None,
        reference_library=output / 'manifest.json')
    run = m.prepare(args)
    entries = m.validate(run)
    assert entries[0][1]['protocol'] == 'molrag_direct_evidence.v1'
    assert entries[0][1]['generation'] == m.generation_settings(model)
    assert len(entries[0][3]['examples']) == 4

    class Client:
        calls = 0
        def chat_json(self, messages):
            self.calls += 1
            return dict(raw_content='{"Question Answer":"YES"}', model=model,
                        messages=messages, usage=dict(cost=.01))
    client = Client()
    assert m.execute_query(entries[0], client, threading.Event()) == 'ok'
    assert m.execute_query(entries[0], client, threading.Event()) == 'reused'
    assert client.calls == 1
    (entries[0][-1] / 'output.json').unlink()
    (entries[0][-1] / 'stage_resume.json').unlink()
    assert m.execute_query(entries[0], client, threading.Event()) == 'recovered'
    assert client.calls == 1
    checked = m.validate(run)
    result = m.summarize(run, checked, require_complete=True)
    assert result['n_successful'] == 1 and result['usage']['reported_cost_usd'] == .01
    assert result['cells'][0]['accuracy'] == 1
    local_provider = tmp_path/'local-provider.json'
    local_config = {**config, 'providers': [{**config['providers'][0], 'base_url': 'http://localhost:50001/v1'}]}
    if model == m.MODEL:
        local_config['providers'][0]['model'] = 'deepseek-ai/DeepSeek-V4-Flash-0731'
    write_json_atomic(local_provider, local_config)
    with monkeypatch.context() as patch:
        copyfile = m.shutil.copyfile
        def interrupted_copy(source, destination, **kwargs):
            if str(destination).endswith('requests.jsonl'):
                raise OSError('interrupted copy')
            return copyfile(source, destination, **kwargs)
        patch.setattr(m.shutil, 'copyfile', interrupted_copy)
        with pytest.raises(OSError, match='interrupted copy'):
            m.repeat(run, tmp_path/'interrupted', local_provider)
        assert not (tmp_path/'interrupted').exists()
        assert not list(tmp_path.glob('.molrag-repeat-*'))
    repeated = tmp_path/'repeated'
    m.repeat(run, repeated, local_provider)
    repeated_entries = m.validate(repeated)
    assert repeated_entries[0][3] == entries[0][3]
    assert repeated_entries[0][1]['run_uid'] != entries[0][1]['run_uid']
    assert not list(repeated.rglob('output.json')) and not list(repeated.rglob('attempt_*.json'))
    assert m.execute_query(repeated_entries[0], client, threading.Event()) == 'ok'
    assert client.calls == 2
    # Capacity changes are operational; saved provider routing and requests stay frozen.
    provider_before = (repeated / 'provider.json').read_bytes()
    captured = []
    def pool(config, **kwargs):
        captured.append(config)
        return client
    monkeypatch.setattr(client, 'close', lambda: None, raising=False)
    monkeypatch.setattr(m, 'build_provider_pool', pool)
    monkeypatch.setattr('sys.argv', ['molrag', 'run', '--run-root', str(repeated),
                                  '--parallelism', '2', '--provider-max-inflight', '2'])
    m.main()
    assert client.calls == 2
    assert captured[0].providers[0].max_inflight == 2
    assert captured[0].providers[0].model == local_config['providers'][0]['model']
    assert (repeated / 'provider.json').read_bytes() == provider_before
    assert read_json(repeated / 'execution_sessions.json')[-1]['provider_max_inflight'] == 2
    with pytest.raises(ValueError, match='new output'):
        m.repeat(run, repeated, local_provider)
    with monkeypatch.context() as patch:
        def reject_write(*args, **kwargs):
            pytest.fail('Offline verification must not rewrite run artifacts')
        patch.setattr(m, 'write_json_atomic', reject_write)
        patch.setattr(m, 'write_jsonl_atomic', reject_write)
        assert m.summarize(run, checked, require_complete=True, write=False) == result
    saved_output = entries[0][-1] / 'output.json'
    before = saved_output.read_bytes()
    saved = read_json(saved_output)
    saved['llm']['model'] = 'wrong-model'
    write_json_atomic(saved_output, saved)
    with pytest.raises(ValueError, match='incompatible'):
        m.validate(run)
    saved_output.write_bytes(before)
    config['providers'][0]['request_extra_body']['reasoning_effort_override'] = 'high'
    write_json_atomic(provider, config)
    args.run_name = 'wrong-effort'
    with pytest.raises(ValueError, match='low-effort'):
        m.prepare(args)
    assert not (args.output_root / 'molrag' / m.PROMPT / 'wrong-effort').exists()
    frozen = entries[0][0] / 'inputs/references.jsonl'
    frozen.write_text(frozen.read_text() + '\n')
    with pytest.raises(ValueError, match='reference pool changed'):
        m.validate(run)
    evidence.write_text(evidence.read_text() + '\n')
    with pytest.raises(ValueError, match='sidecar changed'):
        build_library(sources, tmp_path, tmp_path / 'changed-library')
