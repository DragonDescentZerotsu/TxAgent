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
    (summary/'predictions.jsonl').write_text('')
    with pytest.raises(ValueError, match='empty'): main(['--verify-paper-baseline', str(tmp_path)])
