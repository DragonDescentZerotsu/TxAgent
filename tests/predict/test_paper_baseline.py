"""Frozen paper contracts must survive rendering and response replay unchanged."""

from copy import deepcopy
import json

import pytest

from predict.prompts import paper_baseline as paper
from predict.utils.response import response_validation_errors


def example(task='bbb_martins', benchmark=''):
    spec = paper.profiles()['conditioned'][task]
    if benchmark:
        spec = paper.profiles()['tdc_overrides'].get(task, spec)
    payload = {
        'task_definition': {'task': task, 'benchmark_profile': benchmark},
        'protocol': {'citation_rule': spec['citation_rule'], 'record_rendering': paper.RECORD_RENDERING},
        'query': {'canonical_smiles': 'CCO', 'external_condition': 'reported condition'},
        'query_prior': {'reasoning_summary': 'A frozen property prior.'},
        'neighbors': [], 'required_json_schema': deepcopy(spec['required_json_schema']),
    }
    manifest = {'task_contracts': {task: spec['contract_sha256']}, 'benchmark_profile': benchmark}
    messages = paper.render(payload)
    prepared = {'version': 'direct_indirect_budgeted.v3', 'prompt_version': 'joseph_v5',
                'prompt_payload': payload, 'messages': messages, 'request_hash': paper.content_hash(messages),
                'manifest_hash': paper.content_hash(manifest), 'card_alias_map': {},
                'direct_ids': [], 'indirect_ids': [], 'n_direct': 0, 'n_indirect': 0}
    content = {spec['task']['prediction_field']: spec['task']['positive_prediction'],
               'confidence': 'moderate', 'supportive_card_ids': [], 'contradictory_card_ids': [],
               'prediction_basis_card_ids': [], 'claims': [], 'evidence_gaps': [],
               'decision_summary': 'No experimental evidence was supplied.'}
    output = {'status': 'ok', 'request_hash': prepared['request_hash'],
              'llm': {'content': content, 'structured_output_validation': {'valid': True}}}
    return prepared, manifest, output


@pytest.mark.parametrize('task', paper.profiles()['conditioned'])
@pytest.mark.parametrize('benchmark', ['', paper.TDC_PROFILE])
def test_native_labels_and_conditioned_tdc_profiles(task, benchmark):
    prepared, manifest, output = example(task, benchmark)
    original = deepcopy(prepared)
    assert paper.verify_prepared(prepared, manifest, output) == 1
    spec = paper.profile(prepared['prompt_payload'])['task']
    output['llm']['content'][spec['prediction_field']] = spec['negative_prediction']
    assert paper.verify_prepared(prepared, manifest, output) == 0
    assert prepared == original
    assert 'final_prediction' not in prepared['prompt_payload']['required_json_schema']
    if task in ('ames', 'carcinogens') and benchmark:
        assert paper.profile(prepared['prompt_payload']) != paper.profiles()['conditioned'][task]


@pytest.mark.parametrize('change', ['message', 'payload', 'schema', 'manifest', 'response_hash', 'receipt', 'aliases', 'raw_response'])
def test_changed_requests_and_failed_responses_are_not_reusable(change):
    p, m, o = example()
    if change == 'message': p['messages'][0]['content'] += 'changed'
    if change == 'payload': p['prompt_payload']['query_prior']['reasoning_summary'] = 'changed'
    if change == 'schema': p['prompt_payload']['required_json_schema']['new_field'] = 'string'
    if change == 'manifest': m['benchmark_profile'] = 'unknown'
    if change == 'response_hash': o['request_hash'] = 'different request'
    if change == 'receipt': o['llm']['structured_output_validation']['valid'] = False
    if change == 'aliases': p['card_alias_map']['C001'] = 'foreign record'
    if change == 'raw_response': o['llm']['raw_content'] = '{"different": true}'
    with pytest.raises(ValueError): paper.verify_prepared(p, m, o)


@pytest.mark.parametrize('field,value', [
    ('supportive_card_ids', ['C002']),
    ('contradictory_card_ids', ['C001']),
    ('prediction_basis_card_ids', ['C002']),
    ('claims', [{'claim': 'foreign', 'card_ids': ['C002']}]),
    ('claims', [{'card_ids': []}] * 9),
    ('evidence_gaps', ['missing'] * 7),
    ('decision_summary', ''),
])
def test_citation_and_output_limits(field, value):
    p, _, o = example()
    content = o['llm']['content']
    content['supportive_card_ids'] = ['C001']
    content[field] = value
    assert response_validation_errors(o['llm'], **paper.validation_kwargs(p['prompt_payload'], {'C001': 'record'}))


def test_exact_duplicate_removal_preserves_source_and_different_values():
    evidence = {'endpoint': {'name': 'test', 'measurement': {'value': 0, 'unit': 'mg'}},
                'text': {'evidence': 'observed'}, 'group': {'endpoint_group': 'family'},
                'examples': [{'reported_value': 0, 'reported_units': 'different', 'support_text': 'observed'}]}
    original = deepcopy(evidence)
    card = dict(endpoint='test', reported_value=0, reported_unit='mg', support_text='observed',
                evidence_family='family', assay_context='', species='', qualifying_conditions='')
    result = paper._deduplicated_details(evidence, card)
    assert evidence == original
    assert result['endpoint']['measurement']['value'] == 0
    assert result['examples'][0]['reported_units'] == 'different'
    assert 'support_text' not in result['examples'][0]
    assert 'name' not in result['endpoint']


def test_offline_cli_verifies_saved_batch_without_flat_runtime(tmp_path, capsys):
    from predict.harnesses.branches.__main__ import main
    p, m, o = example()
    m['inputs'] = {'bbb_martins': {}}
    p['manifest_hash'] = paper.content_hash(m)
    query = tmp_path / 'runs/bbb_martins/query_00000'
    summary = tmp_path / 'summary/bbb_martins'
    query.mkdir(parents=True); summary.mkdir(parents=True)
    for path, value in [(tmp_path/'experiment_manifest.json', m), (query/'prepared.json', p),
                        (query/'output.json', o), (query/'evaluation.json', {'label': 1}),
                        (summary/'metrics.json', {'n_total': 1, 'n_successful': 1, 'accuracy': 1, 'macro_f1': 0.5})]:
        path.write_text(json.dumps(value))
    (summary/'predictions.jsonl').write_text(json.dumps({'index': 0, 'label': 1, 'pred_label': 1})+'\n')
    before = {str(f): f.read_bytes() for f in tmp_path.rglob('*') if f.is_file()}
    assert main(['--verify-paper-baseline', str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)['tasks'][0]['queries'] == 1
    assert before == {str(f): f.read_bytes() for f in tmp_path.rglob('*') if f.is_file()}
    with pytest.raises(ValueError, match='newly prepared'):
        main(['--run-paper-baseline', str(tmp_path), '--provider-config', 'unused.json'])
    with pytest.raises(ValueError, match='newly prepared'):
        main(['--bundle-paper-baseline', str(tmp_path), '--run-name', 'must-not-exist'])
    assert before == {str(f): f.read_bytes() for f in tmp_path.rglob('*') if f.is_file()}
    (summary/'predictions.jsonl').write_text('')
    with pytest.raises(ValueError, match='empty'): main(['--verify-paper-baseline', str(tmp_path)])


def test_neighbor_fill_exhausts_nearest_parent_and_is_order_invariant():
    from optimization.selection.record_budget import select_simple_records
    records = [dict(id=f'{m}{i}', molecule_id=m, group=f'L{i%2+2}', similarity=sim)
               for m, sim in [('near', .8), ('far', .4)] for i in range(60)]
    chosen, audit = select_simple_records(records, budget=50)
    reversed_chosen, _ = select_simple_records(records[::-1], budget=50)
    assert chosen == reversed_chosen and {r['molecule_id'] for r in chosen} == {'near'}
    assert len(chosen) == 50 and audit['shortfall'] == 0
    capped, audit = select_simple_records(records, budget=50, policy='molecule_cap', per_molecule=5)
    assert len(capped) == 10 and audit['shortfall'] == 40
    with pytest.raises(ValueError, match='duplicate'):
        select_simple_records(records + records[:1], budget=50)


def test_source_cards_keep_order_aliases_values_and_prior():
    payload = example()[0]['prompt_payload']
    def record(rid, kind, similarity):
        return dict(id=rid, kind=kind, similarity=similarity, native_molecule_id=rid,
                    canonical_smiles='CCC', group='observed', card=dict(endpoint_type='endpoint',
                    reported_value=0, reported_units='mg', support_text='Observed in source.',
                    assay_context='measured', species_context='', qualifying_conditions='', evidence_family_level=2))
    records = [record('i', 'indirect', .9), record('d', 'direct', .1)]
    original = deepcopy(payload)
    projected, aliases = paper.project_records(payload, records)
    assert payload == original and projected['query_prior'] == payload['query_prior']
    assert aliases == {'C001': 'd', 'C002': 'i'}
    assert projected['neighbors'][0]['records'][0]['evidence']['endpoint']['measurement']['value'] == 0
    assert 'Record C001' in paper.render(projected)[1]['content']


def test_prepared_execution_retry_resume_and_mismatched_request(tmp_path):
    from predict.harnesses.branches.paper_baseline import execute_query
    p, m, saved = example()
    p['context_ready'] = True
    (tmp_path/'prepared.json').write_text(json.dumps(p))
    calls = []
    def call(messages):
        calls.append(deepcopy(messages))
        return {'content': {}} if len(calls) < 4 else deepcopy(saved['llm'])
    result = execute_query(tmp_path, m, call)
    assert len(calls) == 4 and result['status'] == 'ok'
    assert calls[0] == p['messages']
    assert calls[-1][-1]['content'].startswith('Retry the bbb_martins budgeted decision response (attempt 4 of 4).')
    before = (tmp_path/'output.json').read_bytes()
    assert execute_query(tmp_path, m, lambda _: pytest.fail('resume called model')) == result
    assert (tmp_path/'output.json').read_bytes() == before
    result['request_hash'] = 'foreign'
    (tmp_path/'output.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='different request'):
        execute_query(tmp_path, m, lambda _: pytest.fail('invalid resume called model'))


def test_new_candidate_path_uses_selected_parent_and_keeps_scaffold_policy():
    from predict.retrieval.paper_baseline import build_evidence_view, retrieve_candidates
    from predict.retrieval.policies import standardize_smiles_and_fp
    smiles = ['C[C@@H](O)C(=O)O', 'C[C@H](O)C(=O)[O-].[Na+]', 'CCCO', 'CCCO.Cl', 'CCCCO.Cl', 'c1ccccc1', 'Cc1ccccc1']
    molecules, fps, evidence = [], [], {}
    for i, smi in enumerate(smiles):
        canonical, inchi, fp = standardize_smiles_and_fp(smi)
        mid = str(i)
        molecules.append(dict(molecule_chembl_id=mid, canonical_smiles=canonical, standard_inchi_key=inchi))
        fps.append(fp)
        evidence[mid] = {'source': [dict(canonical_smiles=canonical, source_record_examples=[
            dict(evidence_family='observed', evidence_family_level=level, endpoint_type='endpoint') for level in (1, 2)])]}
    index = dict(molecules=molecules, fingerprints=fps, evidence_by_molecule_group=evidence,
                 group_to_molecule_indices={}, source={})
    view = build_evidence_view(index)
    for query, absent, present in [(smiles[0], '1', '2'), (smiles[3], '3', '4')]:
        pool = retrieve_candidates(query, view, split_scheme='scaffold', min_similarity=0)
        for kind in ('direct', 'indirect'):
            ids = {r['native_molecule_id'] for r in pool[kind]}
            assert absent not in ids and present in ids
    pool = retrieve_candidates(smiles[5], view, split_scheme='scaffold', min_similarity=0)
    assert '6' not in {r['native_molecule_id'] for r in pool['direct']}
    assert '6' in {r['native_molecule_id'] for r in pool['indirect']}


@pytest.mark.parametrize('portable', [False, True])
def test_batch_execution_freezes_generation_cohort_prior_and_resume(tmp_path, monkeypatch, portable):
    from predict.harnesses.branches import paper_baseline as runtime
    from predict.api_client import pool
    from predict.utils.json import write_json_atomic, sha256_file
    task = 'bbb_martins'
    source, target = tmp_path/'source', tmp_path/'target'
    directory = target/'runs'/task/'query_00000'
    frozen, original, response = example()
    cohort = tmp_path/'test.jsonl'
    cohort.write_text(json.dumps({'drug': 'CCO', 'Y': 1})+'\n')
    original.update(model='google/gemini-3.8-flash', generation=dict(temperature=None, thinking='provider_default', reasoning_effort='medium'),
                    max_tokens=16384, request_extra_body={}, inputs={task: {'input': dict(path=str(cohort), sha256=sha256_file(cohort))}})
    frozen.update(context_ready=True, manifest_hash=paper.content_hash(original),
                  query_hash=paper.content_hash(dict(benchmark_profile='', canonical_smiles='CCO', condition_group='')))
    write_json_atomic(source/'experiment_manifest.json', original)
    write_json_atomic(source/'runs'/task/'query_00000/prepared.json', frozen)
    manifest = {**original, 'migration': dict(version=runtime.REVISION, source_batch=str(source), source_root=str(tmp_path),
        source_manifest_sha256=sha256_file(source/'experiment_manifest.json'), query_indices={task: [0]})}
    prepared = {**frozen, 'manifest_hash': paper.content_hash(manifest)}
    write_json_atomic(target/'experiment_manifest.json', manifest)
    write_json_atomic(directory/'prepared.json', prepared)
    write_json_atomic(directory/'evaluation.json', {'label': 1})
    config = pool.ProviderPoolConfig.from_mapping({'version': pool.POOL_CONFIG_VERSION, 'providers': [
        dict(name='test', base_url='https://invalid.example/v1', model=original['model'], api_key_env='NOT_USED', max_inflight=1)]})
    monkeypatch.setattr(pool, 'load_provider_pool_config', lambda _: config)
    from types import SimpleNamespace
    builds, calls, closed = [], [], []
    def build(config, **kwargs):
        builds.append(kwargs)
        return SimpleNamespace(chat_json=lambda messages: calls.append(messages) or deepcopy(response['llm']),
                               close=lambda: closed.append(True))
    monkeypatch.setattr(pool, 'build_provider_pool', build)
    assert runtime.execute_batch(target, None, None, 1) == 0
    assert builds[0]['max_tokens'] == 16384 and builds[0]['reasoning_effort'] == 'medium'
    assert builds[0]['token_limit_parameter'] == 'max_tokens' and builds[0]['enable_thinking'] is False
    assert calls == [prepared['messages']] and closed == [True]
    assert paper.verify_batch(target)['tasks'][0]['queries'] == 1
    if portable:
        (target/'.env').write_text('DUMMY=not-an-export-input')
        exported = tmp_path/'portable'
        runtime.bundle(target, exported)
        assert not list(exported.rglob('output.json'))
        assert not list(exported.rglob('.env'))
        assert not (exported/'execution_provider.json').exists()
        with pytest.raises(ValueError, match='new, separate'):
            runtime.bundle(target, exported)
        # Remove both original locators, then move the whole bundle before execution.
        source.rename(tmp_path/'unavailable-source')
        cohort.rename(tmp_path/'unavailable-cohort.jsonl')
        target = tmp_path/'relocated'
        exported.rename(target)
        directory = target/'runs'/task/'query_00000'
        prepared = json.loads((directory/'prepared.json').read_text())
        assert prepared['messages'] == frozen['messages']
        assert runtime.execute_batch(target, None, None, 1) == 0
        assert len(calls) == 2
        assert paper.verify_batch(target)['tasks'][0]['queries'] == 1
    expected_calls = len(calls)
    expected_builds = len(builds) + 1
    before = (directory/'output.json').read_bytes()
    assert runtime.execute_batch(target, None, None, 1) == 0
    assert len(calls) == expected_calls and (directory/'output.json').read_bytes() == before
    changed = deepcopy(prepared)
    changed['prompt_payload']['query_prior']['reasoning_summary'] = 'silently replaced'
    changed['messages'] = paper.render(changed['prompt_payload'])
    changed['request_hash'] = paper.content_hash(changed['messages'])
    write_json_atomic(directory/'prepared.json', changed)
    with pytest.raises(ValueError, match='prior or evaluation'):
        runtime.execute_batch(target, None, None, 1)
    assert len(builds) == expected_builds
    write_json_atomic(directory/'prepared.json', prepared)
    write_json_atomic(directory/'evaluation.json', {'label': 0})
    with pytest.raises(ValueError, match='prior or evaluation'):
        runtime.execute_batch(target, None, None, 1)
    assert len(builds) == expected_builds


def test_tdc_train_merge_filters_heldout_before_card_ranking():
    from predict.retrieval.paper_baseline import merge_train_labels
    rows = [dict(drug=smi, Y=y, benchmark_row_id=str(i), tdc_dataset='BBB_Martins')
            for i, (smi, y) in enumerate([('CCO', 1), ('CCC', 0), ('CCC', 0)])]
    index = dict(molecules=[], fingerprints=[], evidence_by_molecule_group={}, source={})
    merged = merge_train_labels(index, 'bbb_martins', rows, rows[:1])
    assert merged['tdc_train_audit'] == {'label_cards': 1, 'heldout_parent_excluded_row_ids': ['0']}
    cards = [r['source_record_examples'][0] for groups in merged['evidence_by_molecule_group'].values() for r in groups['TDC.train_label']]
    assert len(cards) == 1 and cards[0]['source_row_ids'] == ['1', '2'] and cards[0]['reported_value'] == '0'
