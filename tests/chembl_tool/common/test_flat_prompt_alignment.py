"""upstream prose mapping preserves scientific scope, provenance and local output."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import yaml

from tools.chembl_tool.common import flat_prompt_alignment as alignment
from tools.chembl_tool.common.budgeted_reasoning import build_messages, prepared_payload, validation_kwargs
from tools.chembl_tool.common.json_utils import sha256_file
from tools.chembl_tool.paper_experiments import budgeted_evidence as runner


def record():
    return dict(id='record-1', native_molecule_id='native', molecule_id='parent', group='mechanism',
        kind='indirect', canonical_smiles='CCO', molecule_names=['ethanol'], similarity=.4,
        card=dict(endpoint_type='measured endpoint', reported_value='negative', reported_units='',
            assay_context='experimental system', species_context='mouse', qualifying_conditions='condition',
            support_text='Measured observation.', evidence_family_level=2))


@pytest.mark.parametrize('task', alignment.TASKS)
def test_six_tasks_render_without_tools_and_keep_local_output(task):
    contract = runner.runtime._task_contract(task)
    prior = {'reasoning_summary': 'A physicochemical hypothesis.', 'confidence': 'low',
             'exact_chembl_evidence_assessment': 'LEGACY_CHEMBL',
             'tool_results': ['SECRET_TOOL'], contract.prediction_field: 'SECRET_LABEL'}
    messages, aliases, payload = build_messages(contract, {'canonical_smiles': 'CCCO'}, [record()],
        {'content': 'SECRET_TOOL'}, {'native': [{'content': 'SECRET_TOOL'}]},
        visibility_mode='deployment_visible_prefetched', prompt_version='full_flat_v5',
        query_prior=prior, return_payload=True)
    text = json.dumps(messages)
    assert 'SECRET_TOOL' not in text and 'SECRET_LABEL' not in text and 'LEGACY_CHEMBL' not in text
    assert 'exact_chembl_evidence_assessment' not in text
    assert 'A physicochemical hypothesis.' in text and 'C001' in text and 'Measured observation.' in text
    assert 'Pitfalls to avoid' in text and 'Do not skip records' in text
    assert 'molecule_ids' not in text and 'revision_action' not in text
    assert set(payload['required_json_schema']) == set(validation_kwargs(contract, aliases)['required_fields'])
    prepared = dict(prompt_version='full_flat_v5', messages=messages, prompt_payload=payload)
    assert prepared_payload(prepared) == payload
    tampered = deepcopy(prepared)
    tampered['prompt_payload']['query']['canonical_smiles'] = 'N'
    with pytest.raises(ValueError, match='actual rendered'):
        prepared_payload(tampered)


def test_none_and_blind_render_without_fabricated_citations():
    contract = runner.runtime._task_contract('ames')
    messages, aliases = build_messages(contract, {'canonical_smiles': 'CCCCO'}, [], {}, {},
        visibility_mode='identity_blind', prompt_version='full_flat_v5',
        query_prior={'reasoning_summary': 'CCCCO has uncertain reactivity.'})
    assert not aliases and 'CCCCO' not in json.dumps(messages)
    assert 'at least one identifier' not in json.dumps(messages).lower()


def test_portable_tasks_keep_scope_and_map_both_label_directions(tmp_path):
    contracts = [runner.runtime._task_contract(task) for task in alignment.TASKS]
    alignment.export_bundle(tmp_path, contracts)
    tasks = yaml.safe_load((tmp_path/'tasks.yaml').read_text())
    receipt = json.loads((tmp_path/'provenance.json').read_text())
    for contract in contracts:
        exported = tasks[contract.task]
        assert exported['endpoint_name'] == contract.endpoint_name
        assert exported['required_fields'] == ['claims', 'final_prediction']
        assert receipt['label_mapping'][contract.task]['pass'] == contract.positive_prediction
        assert receipt['label_mapping'][contract.task]['fail'] == contract.negative_prediction
        assert not any('evidence_gaps' in s for s in exported['task_instructions'])
    assert 'both_reported' in str(tasks['ames'])
    assert 'rabbit' in str(tasks['carcinogens'])
    assert 'Phototoxicity' in str(tasks['skin_reaction'])
    assert 'human' in str(tasks['dili'])


def test_prior_is_exact_condition_bound_and_snapshot_pinned(tmp_path):
    root = tmp_path/'source'
    path = root/'ames/queries/query_idx00000/levels/level_1/prepared.json'
    path.parent.mkdir(parents=True)
    row = {'drug': 'CCCO', 'condition_group': 'condition-a'}
    data = tmp_path/'input.jsonl'
    data.write_text(json.dumps(row)+'\n')
    (root/'experiment_manifest.json').write_text(json.dumps({'inputs': {'ames': {
        'input_jsonl': str(data), 'input_sha256': sha256_file(data)}}}))
    path.write_text(json.dumps({'query_smiles': 'CCCO', 'condition_group': 'condition-a',
        'tool_prefetch_complete': True, 'query_prior': {'reasoning_summary': 'prior'},
        'reused_none_final': {'gold_label': 'SECRET'}}))
    mapping = tmp_path/'sources.json'
    mapping.write_text(json.dumps({'ames': str(root)}))
    source = alignment.prior_sources(mapping, ['ames'])['ames']
    prior, _ = alignment.load_prior('ames', row, source)
    assert prior == {'reasoning_summary': 'prior'}
    with pytest.raises(ValueError, match='query/condition'):
        alignment.load_prior('ames', {**row, 'condition_group': 'condition-b'}, source)
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError, match='snapshot changed'):
        alignment.load_prior('ames', row, source)


def test_aligned_preparation_does_not_call_tool_service(tmp_path, monkeypatch):
    monkeypatch.setattr(alignment, 'load_prior', lambda *args: ({'reasoning_summary': 'prior'}, {}))
    def forbidden(**kwargs):
        raise AssertionError('raw tool service must not be called')
    monkeypatch.setattr(runner.runtime, '_prefetch_analog_tools', forbidden)
    args = SimpleNamespace(budgeted_prompt='full_flat_v5', skip_tool_prefetch=False,
                           visibility_mode='deployment_visible_prefetched')
    query = runner.runtime.PreparedQuery('ames', 0, tmp_path)
    runner._write_prepared_query(args, 'ames', {'drug': 'CCCO', 'Y': 0}, query,
        {'prior_sources': {'ames': {}}}, {'indirect': []}, [], [], {}, [])
    prepared = json.loads((tmp_path/'prepared.json').read_text())
    assert prepared['context_ready'] and not prepared['tool_prefetch_complete']
    assert 'tools' not in prepared['prompt_payload']['query']


def test_dedup_preserves_raw_disagreements_unknown_fields_and_input():
    contract = runner.runtime._task_contract('ames')
    source = record()
    source['card'].update(support_text=('Long original observation. ' * 100).rstrip(),
                         custom_reference='study-12345678', dose='5 mg/kg', duration='28 days',
                         unusual={'negative': False, 'zero': 0, 'unknown': None, 'blank': ''})
    messages, _, payload = build_messages(contract, {'canonical_smiles': 'CCCO'}, [source], {}, {},
        visibility_mode='deployment_visible_prefetched', prompt_version='full_flat_v5',
        query_prior={'reasoning_summary': 'prior'}, return_payload=True)
    snapshot = deepcopy(payload)
    legacy = deepcopy(payload)
    del legacy['protocol']['record_rendering']
    old = alignment.render(contract, legacy)
    matched_old, _ = build_messages(contract, {'canonical_smiles': 'CCCO'}, [source], {}, {},
        visibility_mode='deployment_visible_prefetched', prompt_version='full_flat_v5',
        query_prior={'reasoning_summary': 'prior'}, record_rendering='full_fields')
    assert matched_old == old
    text, old_text = messages[1]['content'], old[1]['content']
    assert text.count(source['card']['support_text']) == 1
    assert old_text.count(source['card']['support_text']) == 3
    details = json.loads(next(line.removeprefix('Additional details: ') for line in text.splitlines()
                              if line.startswith('Additional details: ')))
    raw = details['examples'][0]
    normalized = snapshot['neighbors'][0]['records'][0]['evidence']['examples'][0]
    for field in ('custom_reference', 'dose', 'duration', 'unusual'):
        assert raw[field] == normalized[field]
    assert payload == snapshot
    assert messages == alignment.render(contract, payload)
    # Old prepared inputs still render byte-for-byte using the legacy path.
    prepared = dict(prompt_version='full_flat_v5', prompt_payload=legacy, messages=old)
    assert prepared_payload(prepared) == legacy
    changed = deepcopy(payload)
    evidence = changed['neighbors'][0]['records'][0]['evidence']
    evidence['examples'][0]['support_text'] = 'A different observation, not a duplicate.'
    evidence['annotations']['scope']['qualifying_conditions'] = ['condition', 'additional restriction']
    changed_text = alignment.render(contract, changed)[1]['content']
    assert 'A different observation, not a duplicate.' in changed_text
    assert 'additional restriction' in changed_text


def test_dedup_does_not_drop_hidden_values_or_multiple_source_examples():
    evidence = {'endpoint': {'name': '', 'measurement': {'value': None, 'unit': 'mg'}},
                'text': {'evidence': ''}, 'group': {'endpoint_group': ''},
                'annotations': {'scope': {'qualifying_conditions': [True]}},
                'examples': [{'support_text': 'one'}, {'support_text': 'one'}]}
    card = dict(endpoint='', reported_value=None, reported_unit='mg', evidence_family='',
                support_text='', assay_context='', species='', qualifying_conditions=1)
    assert alignment._deduplicated_details(evidence, card) == evidence


def test_unknown_rendering_version_fails_closed():
    with pytest.raises(ValueError, match='unsupported record rendering'):
        alignment.render(None, {'protocol': {'record_rendering': 'unknown'}})
