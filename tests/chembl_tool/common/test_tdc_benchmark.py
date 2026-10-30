"""TDC source integration must preserve leakage, budget and prompt boundaries."""
from copy import deepcopy
import json
import pytest

from tools.chembl_tool.common.budgeted_retrieval import build_evidence_view, retrieve_candidates
from tools.chembl_tool.common.budgeted_evidence import SelectionConfig, select_records
from tools.chembl_tool.common.budgeted_reasoning import build_messages, prepared_payload
from tools.chembl_tool.common.task_workflows.evidence_library import standardize_smiles_and_fp
from tools.chembl_tool.common.tdc_benchmark import PROFILE, task_contract
from tools.chembl_tool.common.tdc_evidence import merge_train_labels, parent_key
from tools.chembl_tool.tasks.ames.experiment_config import get_progressive_task_contract


def row(smiles, label, name):
    return dict(drug=smiles, Y=label, benchmark_row_id=name, tdc_dataset='AMES')


def index():
    molecules, fps, evidence = [], [], {}
    for name, smiles in [('shared', 'CCCO'), ('heldout', 'c1ccccc1')]:
        canonical, key, fp = standardize_smiles_and_fp(smiles)
        molecules.append(dict(molecule_chembl_id=name, canonical_smiles=canonical, standard_inchi_key=key))
        fps.append(fp)
        cards = [dict(endpoint_type='Ames', reported_value='negative', reported_units='',
            assay_context='', species_context='', qualifying_conditions='', support_text='Reported negative Ames result.',
            evidence_family=f'family_{level}', evidence_family_level=level) for level in (1, 2)]
        evidence[name] = {'assay': [dict(canonical_smiles=canonical, assay_chembl_id='assay', source_record_examples=cards)]}
    return dict(molecules=molecules, fingerprints=fps, evidence_by_molecule_group=evidence, source={})


def test_merge_labels_before_retrieval_and_preserve_sources():
    contract = task_contract(get_progressive_task_contract(), PROFILE)
    train = [row('CCCO', 1, 'a'), row('CCCO', 1, 'duplicate'), row('CCCCN', 0, 'new'), row('c1ccccc1', 1, 'excluded')]
    merged = merge_train_labels(index(), 'ames', train, [row('c1ccccc1', 0, 'test')])
    assert merged['tdc_train_audit']['label_cards'] == 2
    assert merged['tdc_train_audit']['heldout_parent_excluded_row_ids'] == ['excluded']
    pool = retrieve_candidates('Cc1ccncc1', build_evidence_view(merged), split_scheme='scaffold', min_similarity=0)
    assert len(pool['direct']) == 3  # one Starling + two labels
    assert parent_key('ames', 'c1ccccc1') not in {r['molecule_id'] for r in pool['direct']}
    assert parent_key('ames', 'c1ccccc1') in {r['molecule_id'] for r in pool['indirect']}
    shared = [r for r in pool['direct'] if r['molecule_id'] == parent_key('ames', 'CCCO')]
    assert {r['evidence_source'] for r in shared} == {'Starling', 'TDC'}
    chosen, _ = select_records(pool['direct'], SelectionConfig(budget=2, policy='molecule_cap', per_molecule=1))
    assert len(chosen) == len({r['molecule_id'] for r in chosen}) == 2
    query = dict(canonical_smiles='Cc1ccncc1', benchmark_profile=PROFILE)
    messages, aliases, payload = build_messages(contract, query, pool['direct'], {}, {},
        visibility_mode='deployment_visible_prefetched', prompt_version='full_flat_v5',
        query_prior={'reasoning_summary': 'A structural prior.'}, return_payload=True)
    assert len(payload['neighbors']) == 2
    assert 'TDC' in messages[1]['content'] and 'Starling' in messages[1]['content']
    assert 'dataset_label' in messages[1]['content']
    assert prepared_payload(dict(prompt_version='full_flat_v5', prompt_payload=payload, messages=messages)) == payload
    assert len(aliases) == 3


def test_tdc_contract_is_opt_in_and_retains_molecular_role_rules():
    original = get_progressive_task_contract()
    before = deepcopy(original)
    assert task_contract(original) == original
    adapted = task_contract(original, PROFILE)
    assert original == before and adapted.prediction_field == original.prediction_field
    assert 'molecule-level' in adapted.endpoint_name
    assert any('modifier/protectant' in s for s in adapted.task_instructions)
    assert not any('only under the query' in s for s in adapted.task_instructions)


def test_preparation_reads_manifest_paths_and_retains_conflicting_labels(tmp_path):
    from tools.chembl_tool.common.budgeted_preparation import evidence_view
    train = tmp_path/'train.jsonl'
    heldout = tmp_path/'heldout.jsonl'
    train.write_text(''.join(json.dumps(row('CCCO', y, str(y)))+'\n' for y in (0, 1)))
    heldout.write_text(json.dumps({'drug': 'c1ccccc1'})+'\n')
    view = evidence_view(index(), 'ames', {'benchmark_profile': PROFILE, 'inputs': {'ames': {
        'train': {'path': str(train)}, 'heldout': {'path': str(heldout)}}}})
    pool = retrieve_candidates('Cc1ccncc1', view, split_scheme='scaffold', min_similarity=0)
    labels = [r for r in pool['direct'] if r['evidence_source'] == 'TDC']
    assert {r['card']['reported_value'] for r in labels} == {'0', '1'}
    assert len({r['molecule_id'] for r in labels}) == 1
    assert len({r['card']['source_record_id'] for r in labels}) == 2
    assert all(r['record_smiles'] == 'CCCO' for r in labels)


@pytest.mark.parametrize('task', ['dili', 'carcinogens'])
def test_new_task_identity_is_preserved_for_tdc_only_molecules(task):
    from tools.chembl_tool.common.starling.new_task_identity import VERSION, leakage_identity
    from tools.chembl_tool.common.starling.new_task_retrieval_identity import annotate_index
    source = index()
    cache = {m['canonical_smiles']: leakage_identity(task, m['canonical_smiles']) for m in source['molecules']}
    annotate_index(source, task, cache, VERSION)
    merged = merge_train_labels(source, task, [row('CCCCN', 1, 'new')], [row('c1ccccc1', 0, 'heldout')])
    assert all(m['retrieval_identity_contract'] == VERSION for m in merged['molecules'])
    pool = retrieve_candidates('CC(C)O', build_evidence_view(merged), split_scheme='scaffold', min_similarity=0)
    assert any(r['evidence_source'] == 'TDC' for r in pool['direct'])
    assert parent_key(task, 'c1ccccc1') not in {r['molecule_id'] for r in pool['direct']}


def test_single_analysis_cache_binds_task_query_condition_and_input(tmp_path):
    from tools.chembl_tool.common import flat_prompt_alignment as alignment
    from tools.chembl_tool.common.budgeted_evidence import content_hash
    from tools.chembl_tool.common.json_utils import sha256_file
    query = {'drug': 'CCO', 'condition_group': 'no_reported_external_condition', 'Y': 1}
    source = tmp_path/'test.jsonl'
    source.write_text(json.dumps(query)+'\n')
    cache = tmp_path/'single.json'
    cache.write_text(json.dumps(dict(version='single_analysis_cache.v1', task='ames',
        benchmark_profile=PROFILE, task_contract_hash='pinned-contract',
        input=dict(path=str(source), sha256=sha256_file(source)), rows=[dict(
            query_smiles=query['drug'], condition_group=query['condition_group'],
            query_hash=content_hash([query['drug'], query['condition_group']]),
            analysis={'reasoning_summary': 'Structure-derived assessment.'})])))
    mapping = tmp_path/'sources.json'
    mapping.write_text(json.dumps({'ames':str(cache)}))
    bound = alignment.prior_sources(mapping, ['ames'])['ames']
    assert alignment.load_prior('ames', query, bound)[0]['reasoning_summary']
    with pytest.raises(KeyError):
        alignment.load_prior('ames', {**query, 'condition_group':'TA98'}, bound)
    with pytest.raises(ValueError, match='task mismatch'):
        alignment.load_prior('dili', query, bound)
    alignment._single_cache.cache_clear()
    source.write_text(json.dumps({**query,'drug':'CCC'})+'\n')
    with pytest.raises(ValueError, match='input changed'):
        alignment.load_prior('ames', query, bound)


def test_matched_knn_uses_exact_train_rows_and_never_query_labels(tmp_path):
    from tools.chembl_tool.common.tdc_evidence import matched_knn_records, _bound_rows
    from tools.chembl_tool.common.json_utils import sha256_file
    train = [dict(row(smiles, i % 2, f'train:{i}'), molecule_identity_key=smiles)
             for i, smiles in enumerate(('C[C@H](O)CC', 'CCCO', 'CCCCN'))]
    query = dict(drug=train[0]['drug'], benchmark_row_id='test:0', Y=0)
    saved = dict(query_index=0, drug=query['drug'], benchmark_row_id='test:0',
                 status='ok', Y=0, prediction=0, neighbors=[
                     dict(r, train_index=i, reference_split='train', similarity=1.)
                     for i, r in enumerate(train)])
    inputs = {}
    for name, rows in [('train', train), ('direct_knn', [saved])]:
        path = tmp_path/f'{name}.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        inputs[name] = dict(path=str(path), sha256=sha256_file(path))
    records = matched_knn_records('ames', query, 0, inputs)
    assert [r['canonical_smiles'] for r in records] == [r['drug'] for r in train]
    assert [r['card']['reported_value'] for r in records] == ['0', '1', '0']
    # Same-parent train neighbor is deliberately retained to match the KNN.
    assert records[0]['canonical_smiles'] == query['drug']
    saved.update(Y=1, prediction=1)
    path = tmp_path/'direct_knn.jsonl'
    path.write_text(json.dumps(saved)+'\n')
    inputs['direct_knn']['sha256'] = sha256_file(path)
    assert matched_knn_records('ames', {**query, 'Y': 1}, 0, inputs) == records
    contract = task_contract(get_progressive_task_contract(), PROFILE)
    _, _, payload = build_messages(contract, {'canonical_smiles': query['drug']},
        records, {}, {}, visibility_mode='deployment_visible_prefetched', return_payload=True)
    assert [n['canonical_smiles'] for n in payload['neighbors']] == [r['drug'] for r in train]
    saved['neighbors'][0]['Y'] = 1
    path.write_text(json.dumps(saved)+'\n')
    inputs['direct_knn']['sha256'] = sha256_file(path)
    with pytest.raises(ValueError, match='differs from TDC train'):
        matched_knn_records('ames', query, 0, inputs)
    _bound_rows.cache_clear()
    path.write_text('{}\n')
    with pytest.raises(ValueError, match='input changed'):
        matched_knn_records('ames', query, 0, inputs)
