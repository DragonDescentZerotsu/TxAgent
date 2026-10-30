"""Identity filtering, source binding, prompt isolation and resumable inference."""
from argparse import Namespace
from copy import deepcopy
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.chembl_tool.common import budgeted_retrieval as retrieval
from tools.chembl_tool.common.budgeted_evidence import content_hash
from tools.chembl_tool.common.budgeted_reasoning import build_messages, validation_kwargs
from tools.chembl_tool.common.task_workflows.evidence_library import standardize_smiles_and_fp
from tools.chembl_tool.common.reasoning_validation import response_validation_errors
from tools.chembl_tool.paper_experiments import budgeted_evidence as runner


def card(level=2, **changes):
    return dict(endpoint_type='bacterial reverse mutation', reported_value='negative', reported_units='',
                assay_context='bacterial assay', species_context='bacteria', qualifying_conditions='TA98; +S9',
                support_text='TestCompound was negative in the supplied experiment.', evidence_family=f'family_{level}',
                evidence_family_level=level, **changes)


def index():
    molecules, fingerprints, evidence = [], [], {}
    for name, structure in [('query', 'Cc1ccccc1'), ('same_parent', 'Cc1ccccc1.[Na+]'),
                             ('same_scaffold', 'Oc1ccccc1'), ('different', 'CCCO')]:
        smiles, key, fp = standardize_smiles_and_fp(structure)
        molecules.append(dict(molecule_chembl_id=name, canonical_smiles=smiles, standard_inchi_key=key))
        fingerprints.append(fp)
        evidence[name] = {'Assay.mixed': [dict(assay_chembl_id='mixed', source_molecule_names=['TestCompound'],
                                             source_record_examples=[card(1), card(2), card(7)])]}
    return dict(molecules=molecules, fingerprints=fingerprints, evidence_by_molecule_group=evidence, source={})


def test_every_indirect_level_uses_parent_disjoint_and_mixed_assays_are_split():
    original = index()
    view = retrieval.build_evidence_view(original)
    candidates = retrieval.retrieve_candidates('Cc1ccccc1', view, split_scheme='scaffold', min_similarity=0)
    assert {r['native_molecule_id'] for r in candidates['direct']} == {'different'}
    assert {r['native_molecule_id'] for r in candidates['indirect']} == {'same_scaffold', 'different'}
    assert {r['card']['evidence_family_level'] for r in candidates['indirect']} == {2, 7}
    assert len(original['evidence_by_molecule_group']['query']['Assay.mixed'][0]['source_record_examples']) == 3
    random = retrieval.retrieve_candidates('Cc1ccccc1', view, split_scheme='random', min_similarity=0)
    assert {r['native_molecule_id'] for r in random['direct']} == {'same_scaffold', 'different'}


def test_untagged_cards_fail():
    raw = index()
    raw['evidence_by_molecule_group']['query']['Assay.mixed'][0]['source_record_examples'][0]['evidence_family_level'] = 0
    with pytest.raises(ValueError, match='family tags'):
        retrieval.build_evidence_view(raw)


def test_indirect_scaffold_exclusion_backfills_before_neighbor_limit():
    view = retrieval.build_evidence_view(index())
    parent = retrieval.retrieve_candidates('Cc1ccccc1', view, split_scheme='scaffold',
        min_similarity=0, candidate_molecules=1)
    strict = retrieval.retrieve_candidates('Cc1ccccc1', view, split_scheme='scaffold',
        min_similarity=0, candidate_molecules=1, indirect_identity_policy='scaffold_disjoint')
    assert strict['direct'] == parent['direct']
    assert {r['native_molecule_id'] for r in parent['indirect']} == {'same_scaffold'}
    assert {r['native_molecule_id'] for r in strict['indirect']} == {'different'}


def test_direct_candidates_have_no_hidden_ten_molecule_limit(monkeypatch):
    calls = []
    def retrieve(*args, **kwargs):
        calls.append(kwargs['top_k_per_group'])
        return {'status':'ok', 'groups':[{'neighbors':[]}]}
    monkeypatch.setattr(retrieval, 'retrieve_neighbors', retrieve)
    retrieval.retrieve_candidates('CC', {'budgeted_parent_groups':{}}, split_scheme='scaffold', candidate_molecules=128)
    assert calls == [128,128]
    records = [{**candidate(), 'id':f'{m}:{r}', 'molecule_id':f'parent{m}'} for m in range(25) for r in range(8)]
    chosen = runner._direct_selection(records, cap=5, budget=100, policy='molecule_cap')
    assert len(chosen)==100 and len({r['molecule_id'] for r in chosen})==20


def test_unresolved_parent_is_excluded_before_ranking_and_backfilled():
    raw = index()
    wildcard = deepcopy(raw['molecules'][0])
    wildcard.update(molecule_chembl_id='wildcard', canonical_smiles='[1*]CCO')
    wildcard.pop('molecule_identity', None)
    raw['molecules'].append(wildcard)
    raw['fingerprints'].append(standardize_smiles_and_fp('CCO')[2])
    raw['evidence_by_molecule_group']['wildcard'] = deepcopy(raw['evidence_by_molecule_group']['different'])
    view = retrieval.build_evidence_view(raw)
    assert view['budgeted_unresolved_parent_ids'] == ['wildcard']
    assert 'wildcard' not in view['evidence_by_molecule_group']
    candidates = retrieval.retrieve_candidates('Cc1ccccc1', view, split_scheme='scaffold',
                                               min_similarity=0, candidate_molecules=2)
    assert {r['native_molecule_id'] for r in candidates['indirect']} == {'same_scaffold', 'different'}
    assert 'wildcard' in raw['evidence_by_molecule_group']  # Frozen index is unchanged.


def test_exact_source_binding_not_fuzzy_or_reintroducing_unselected_records(tmp_path, monkeypatch):
    source, catalog = tmp_path/'records.parquet', tmp_path/'catalog.json'
    row = dict(canonical_record_id='r1', molecule_name='TestCompound', canonical_smiles='CCCO', molecule_id='different',
        canonical_endpoint_name='bacterial reverse mutation', canonical_measurement_text='negative', canonical_unit_text='',
        canonical_assay_context='bacterial assay', canonical_species_context='bacteria', qualifying_conditions='TA98; +S9',
        support_text='TestCompound was negative in the supplied experiment.', group_id='Native.g', confidence=.8, retrieval_eligible=True)
    pq.write_table(pa.Table.from_pylist([row, {**row, 'canonical_record_id':'r2', 'canonical_measurement_text':'positive'}]), source)
    catalog.write_text(json.dumps({'levels':[{'level':2,'endpoint_group':'family_2','source_groups':['Native.g']}]}))
    monkeypatch.setattr(retrieval, 'source_paths', lambda task: (source,catalog))
    c = dict(id=content_hash(['different',card()]), native_molecule_id='different', card=card())
    bound = retrieval.bind_source_records('bbb_martins',[c])
    assert bound[c['id']]['source_record_ids'] == ['r1']
    wrong = {**c, 'id':'wrong'}
    with pytest.raises(ValueError, match='do not bind'):
        retrieval.bind_source_records('bbb_martins',[wrong])


def candidate():
    return dict(id='record_internal_id', native_molecule_id='M12345', molecule_id='parent', group='family_2',
        kind='indirect', card=card(), canonical_smiles='CCCO', molecule_names=['TestCompound'],
        similarity=.45786, quality=.99, features={'internal_features':1}, quality_audit={'gold_label':1})


def response(contract):
    return {contract.prediction_field: contract.negative_prediction, 'confidence':'moderate',
            'supportive_card_ids':['C001'], 'contradictory_card_ids':[], 'prediction_basis_card_ids':['C001'],
            'claims':[{'claim':'Observed negative readout in the neighbor, with uncertain transfer.', 'card_ids':['C001']}],
            'evidence_gaps':[], 'decision_summary':'The supplied scoped negative observation supports this prediction.'}


def test_prompt_has_raw_evidence_only_and_visibility_contracts():
    c = candidate()
    contract = runner.runtime._task_contract('ames')
    query = dict(canonical_smiles='CCCCO', condition_group='no_reported_external_condition', Y=1, gold_label=1)
    tools = {'tool_name':'molecule_properties','content':'[molecule_properties]\nMW: 88.00\nquery_smiles: CCCCO', 'raw_features':{'secret':1}}
    for mode in ('identity_blind','deployment_visible_prefetched'):
        messages, aliases = build_messages(contract, query, [c], tools, {}, visibility_mode=mode)
        text = messages[1]['content']; payload = json.loads(text)
        assert aliases == {'C001':'record_internal_id'}
        for forbidden in ('quality_audit','gold_label','raw_features','internal_features','query_prior','level_context','revision_action','new_evidence_assessment'):
            assert forbidden not in text
        assert 'negative' in text and 'TA98' in text
        assert payload['neighbors'][0]['similarity'] == .46
        assert payload['neighbors'][0]['records'][0]['evidence']['contract_version'] == 'minimal_evidence.v1'
        if mode == 'identity_blind':
            assert 'CCCCO' not in text and 'CCCO' not in text and 'TestCompound' not in text and 'M12345' not in text
        else:
            assert payload['query']['canonical_smiles'] == 'CCCCO'
    assert not response_validation_errors({'content':response(contract)}, **validation_kwargs(contract, {'C001'}))
    invalid = response(contract); invalid['claims'][0]['card_ids'] = ['absent']
    assert response_validation_errors({'content':invalid}, **validation_kwargs(contract, {'C001'}))


def test_one_prediction_and_exact_checkpoint_resume(tmp_path):
    contract = runner.runtime._task_contract('ames')
    messages, aliases = build_messages(contract, {'canonical_smiles':'CCCCO'}, [candidate()], {}, {}, visibility_mode='identity_blind')
    prepared = dict(messages=messages, request_hash=content_hash(messages), manifest_hash='manifest',
                    tool_prefetch_complete=True, card_alias_map=aliases)
    (tmp_path/'prepared.json').write_text(json.dumps(prepared))
    q = runner.runtime.PreparedQuery('ames', 0, tmp_path)
    args = Namespace(prepare_only=False, manifest_hash='manifest', retry_race_width=1)
    class Client:
        calls = 0
        def chat_json(self, messages):
            self.calls += 1
            return {'content':response(contract)}
    client = Client()
    assert runner._run_query(args,q,client)['status'] == 'ok' and client.calls == 1
    assert runner._run_query(args,q,client)['status'] == 'ok' and client.calls == 1
    prepared['messages'][1]['content'] += 'changed'
    (tmp_path/'prepared.json').write_text(json.dumps(prepared))
    with pytest.raises(ValueError, match='request hash'):
        runner._run_query(args,q,client)


def test_manifest_rejects_changes_and_offline_cannot_infer(tmp_path):
    path = tmp_path/'manifest.json'
    runner._pin(path, {'budget':64})
    with pytest.raises(ValueError, match='fresh output root'):
        runner._pin(path, {'budget':128})
    prepared = dict(messages=[], request_hash=content_hash([]), manifest_hash='m', tool_prefetch_complete=False)
    (tmp_path/'prepared.json').write_text(json.dumps(prepared))
    with pytest.raises(ValueError, match='tool prefetch'):
        runner._run_query(Namespace(prepare_only=False, manifest_hash='m'), runner.runtime.PreparedQuery('ames',0,tmp_path),None)


@pytest.mark.parametrize('retry_round,expected_widths', [(0, [1, 6]), (1, [6, 6])])
def test_prepared_retry_races_validate_and_resume(tmp_path, retry_round, expected_widths):
    from tools.chembl_tool.common.reasoning_race import ParallelRetryClient
    contract = runner.runtime._task_contract('ames')
    messages, aliases = build_messages(contract, {'canonical_smiles':'CCCCO'}, [candidate()], {}, {}, visibility_mode='identity_blind')
    (tmp_path/'prepared.json').write_text(json.dumps(dict(messages=messages,
        request_hash=content_hash(messages), manifest_hash='m', tool_prefetch_complete=True,
        card_alias_map=aliases)))
    class Client(ParallelRetryClient):
        def __init__(self):
            self.widths = []
        def chat_validated(self, messages, *, task, width, validate, receipt_path):
            self.widths.append(width)
            result = {'content': {} if len(self.widths) == 1 else response(contract)}
            assert bool(validate(result)) == (len(self.widths) == 1)
            assert task == 'ames' and receipt_path.parent.name == 'retry_races'
            return result
    client = Client()
    args = Namespace(prepare_only=False, manifest_hash='m', retry_race_width=6, retry_round=retry_round)
    query = runner.runtime.PreparedQuery('ames', 0, tmp_path)
    assert runner._run_query(args, query, client)['status'] == 'ok'
    assert client.widths == expected_widths
    assert runner._run_query(args, query, client)['status'] == 'ok'
    assert client.widths == expected_widths


def test_all_seven_settings_share_direct_and_candidates_end_to_end(tmp_path, monkeypatch):
    """Full CLI -> native binding -> tools -> reader -> summaries, with a fake reader."""
    import pickle
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_family_curve as cli
    task = 'bbb_martins'
    input_path = tmp_path/'valid.jsonl'
    input_path.write_text(json.dumps({'drug':'Cc1ccccc1', 'Y':0, 'condition_group':'no_reported_external_condition'})+'\n')
    source, catalog = tmp_path/'records.parquet', tmp_path/'catalog.json'
    raw_index = index()
    source_rows = []
    for m in raw_index['molecules']:
        for level in (1,2,7):
            c = card(level)
            source_rows.append(dict(canonical_record_id=f"{m['molecule_chembl_id']}_{level}",
                molecule_name='TestCompound', canonical_smiles=m['canonical_smiles'], molecule_id=m['molecule_chembl_id'],
                canonical_endpoint_name=c['endpoint_type'], canonical_measurement_text=c['reported_value'],
                canonical_unit_text=c['reported_units'], canonical_assay_context=c['assay_context'],
                canonical_species_context=c['species_context'], qualifying_conditions=c['qualifying_conditions'],
                support_text=c['support_text'], group_id=f'Native.g{level}', confidence=.8, retrieval_eligible=True))
    pq.write_table(pa.Table.from_pylist(source_rows), source)
    catalog.write_text(json.dumps({'levels':[{'level':level, 'endpoint_group':f'family_{level}',
        'source_groups':[f'Native.g{level}']} for level in (1,2,7)]}))
    index_root = tmp_path/'indices'; index_path = index_root/task/'scaffold/assay_neighbor_index.pkl'
    index_path.parent.mkdir(parents=True)
    index_path.write_bytes(pickle.dumps(raw_index))
    index_path.with_name('manifest.json').write_text(json.dumps({'reference_pool':runner.runtime.REFERENCE_POOL,
        'n_direct_heldout_records_after_filter':0, 'max_record_examples':0, 'records_sha256':runner.sha256_file(source)}))
    spec = runner.runtime.ProgressiveTaskSpec(input_path,index_path,catalog)
    monkeypatch.setattr(runner.runtime, '_progressive_task_specs', lambda *a: {task:spec})
    monkeypatch.setattr(runner.runtime, '_heldout_filter_validation', lambda **kw: {'status':'ok'})
    monkeypatch.setattr(runner, 'source_paths', lambda t: (source,catalog))
    monkeypatch.setattr(retrieval, 'source_paths', lambda t: (source,catalog))
    monkeypatch.setattr(runner, '_code_hashes', lambda: {})
    monkeypatch.setattr(runner.runtime, '_prefetch_analog_tools', lambda **kw: ({},[],{'tool_name':'molecule_properties','content':'MW: 92.00'}))
    calls = []
    class Client:
        def chat_json(self,messages):
            calls.append(messages)
            contract = runner.runtime._task_contract(task)
            payload = json.loads(messages[1]['content'])
            ids = [r['card_id'] for n in payload['neighbors'] for r in n['records']][:1]
            value = response(contract)
            value.update(supportive_card_ids=ids, prediction_basis_card_ids=ids, claims=[])
            return {'content':value}
        def snapshot(self):
            return {'providers':[]}
        def close(self):
            pass
    monkeypatch.setattr(runner.runtime, '_make_client', lambda *a: Client())
    def annotations(path, task, bindings):
        return {key:{'outcomes':[{
            'endpoint':bound['prepared']['endpoint'], 'kind':'positive', 'focal_scope':'focal',
            'evidence_basis':'measured', 'result_scope':'focal_effect', 'limitations':[],
            'conditions':[], 'comparator':[], 'statistical_support':'not_reported',
        }], 'unresolved':[]} for key,bound in bindings.items()}
    # Formal mode cannot silently run with native-only features.
    with pytest.raises(ValueError, match='lack input-bound v4'):
        cli.main(['--budgeted-prompt','budgeted_v1','--evidence-setting','direct_indirect','--tasks',task,
                  '--retrieval-index-root',str(index_root),'--min-similarity','0',
                  '--output-root',str(tmp_path/'missing_cache')])
    assert not calls
    export = tmp_path/'feature_export'
    assert cli.main(['--budgeted-prompt','budgeted_v1','--evidence-setting','direct_indirect','--feature-inputs-only','--tasks',task,
        '--retrieval-index-root',str(index_root),'--min-similarity','0','--output-root',str(export)]) == 0
    table = pq.read_table(export/'feature_inputs'/f'{task}.features.parquet').to_pylist()
    assert table and all(json.loads(row['payload_json'])['needs_llm'] for row in table)
    assert not calls
    monkeypatch.setattr(runner, 'cached_annotations', annotations)
    rows = []
    for setting, policy in [('none','joint'), ('direct','joint'), *[('direct_indirect',s) for s in runner.POLICIES]]:
        root = tmp_path/f'{setting}_{policy}'
        assert cli.main(['--budgeted-prompt','budgeted_v1','--evidence-setting',setting,'--indirect-selector',policy,'--tasks',task,
            '--retrieval-index-root',str(index_root),'--min-similarity','0','--output-root',str(root)]) == 0
        prepared = json.loads((root/'runs'/task/'query_00000/prepared.json').read_text())
        metrics = json.loads((root/'summary'/task/'metrics.json').read_text())
        assert metrics['n_successful'] == 1 and metrics['accuracy'] == 1
        rows.append(prepared)
    def forbid(*a, **kw):
        raise AssertionError('simple controls must not read assessment caches or native source features')
    monkeypatch.setattr(runner, 'cached_annotations', forbid)
    monkeypatch.setattr(runner, 'bind_source_records', forbid)
    paired = {}
    for direct_policy in ('neighbor_fill', 'molecule_cap'):
        for setting in ('direct', 'direct_indirect'):
            root = tmp_path/f'raw_{setting}_{direct_policy}'
            assert cli.main(['--budgeted-prompt','budgeted_v1','--evidence-setting',setting,'--direct-selector',direct_policy,
                '--indirect-selector',direct_policy,'--direct-budget','50','--indirect-budget','50',
                '--records-per-molecule','5','--tasks',task,'--retrieval-index-root',str(index_root),
                '--min-similarity','0','--output-root',str(root)]) == 0
            value = json.loads((root/'runs'/task/'query_00000/prepared.json').read_text())
            paired[setting,direct_policy] = value['direct_ids']
            manifest = json.loads((root/'experiment_manifest.json').read_text())
            assert manifest['assessment_required'] is False and manifest['direct_selector'] == direct_policy
        assert paired['direct',direct_policy] == paired['direct_indirect',direct_policy]
    assert len(calls) == 11
    assert rows[0]['n_direct'] == rows[0]['n_indirect'] == 0
    assert rows[1]['n_direct'] > 0 and rows[1]['n_indirect'] == 0
    assert all(r['direct_ids'] == rows[1]['direct_ids'] for r in rows[2:])
    assert len({r['candidate_pool_hash'] for r in rows[1:]}) == 1

    # Exercise the real suite CLI, preparation handoff, inference and metrics
    # against exactly the same scientific fixtures. Only process transport and
    # the model are fake; a resume must not call the model or rewrite outputs.
    from threading import Thread
    from concurrent.futures import ThreadPoolExecutor
    from tools.chembl_tool.paper_experiments import budgeted_suite
    # Keep fake tools and fixture-source patches within this process.
    monkeypatch.setattr(budgeted_suite, 'ProcessPoolExecutor',
                        lambda *, mp_context, **kw: ThreadPoolExecutor(**kw))
    class Preparation:
        def __init__(self, command):
            self.code = None
            def run():
                try:
                    self.code = cli.main(command[3:])
                except Exception:
                    self.code = 1
                    raise
            self.thread = Thread(target=run)
            self.thread.start()
        def poll(self):
            return self.code
        def wait(self):
            self.thread.join(timeout=10)
            assert not self.thread.is_alive()
            return self.code
        def terminate(self):
            self.wait()
    def start(cells):
        for cell in cells:
            cell.root.mkdir(parents=True, exist_ok=True)
            for name in ('preparation_ready.json', 'preparation_done.json'):
                (cell.root/name).unlink(missing_ok=True)
        process = Preparation(cells[0].command)
        for cell in cells:
            cell.process = process
    monkeypatch.setattr(budgeted_suite, '_start_preparation', start)
    monkeypatch.setattr(budgeted_suite, '_stop_preparation', lambda p: p.wait())
    root = tmp_path/'suite'
    options = ['--budgeted-prompt','budgeted_v1','--baseline-suite','--tasks',task,'--retrieval-index-root',str(index_root),
        '--min-similarity','0','--output-root',str(root),'--parallelism','2',
        '--visibility-mode','identity_blind',
        '--preparation-processes','2','--direct-budget','50','--indirect-budget','50']
    assert cli.main(options) == 0
    assert len(calls) == 17
    receipt = json.loads((root/'runs/both_group'/task/'preparation_done.json').read_text())
    assert receipt['error'] == '' and receipt['query_groups'] == 1
    assert receipt['preparation_processes'] == 2
    assert len(list((root/'candidate_cache').glob('*/*.json.zlib'))) == 1
    saved = {p:p.read_bytes() for p in root.glob('runs/*/*/runs/*/query_*/output.json')}
    assert len(saved) == 6
    for name, evidence, direct, indirect in budgeted_suite.SETTINGS:
        metrics = json.loads((root/'runs'/name/task/'summary'/task/'metrics.json').read_text())
        assert metrics['n_successful'] == 1 and metrics['accuracy'] == 1
        if name in ('direct_fill','both_fill','direct_cap','both_cap'):
            before = tmp_path/f'raw_{evidence}_{direct}'/'runs'/task/'query_00000/prepared.json'
            after = root/'runs'/name/task/'runs'/task/'query_00000/prepared.json'
            assert json.loads(before.read_text())['messages'] == json.loads(after.read_text())['messages']
    assert cli.main(options) == 0
    assert len(calls) == 17
    assert all(p.read_bytes() == value for p,value in saved.items())


def test_direct_budget_and_parent_cap_are_independent():
    records = [{'id': f'{m:02d}_{i:02d}', 'molecule_id': str(m), 'group': 'direct', 'similarity': 1-m/100}
               for m in range(12) for i in range(8)]
    selected = runner._direct_selection(records, cap=5, budget=50)
    assert len(selected) == 50
    assert len({r['molecule_id'] for r in selected}) == 10
    assert all(sum(r['molecule_id'] == str(m) for r in selected) <= 5 for m in range(12))
    assert runner._direct_selection(list(reversed(records)), cap=5, budget=7) == selected[:7]
    assert runner._direct_selection(records, cap=5, budget=0) == []
    assert len(runner._direct_selection(records[:3], cap=5, budget=50)) == 3
    filled = runner._direct_selection(records, cap=5, budget=8, policy='neighbor_fill')
    assert {r['molecule_id'] for r in filled} == {'0'}


def test_transfer_mapping_rejects_invalid_and_duplicate_scores(tmp_path):
    path = tmp_path/'transfer.json'
    row = {'task':'ames','query_hash':'q','record_id':'r','score':.75}
    def write(rows):
        path.write_text(json.dumps({'version':'query_record_transfer.v1','model':'frozen tool','scores':rows}))
    write([row]); assert runner._transfer_map(path)['ames','q','r'] == .75
    write([row,row])
    with pytest.raises(ValueError, match='duplicate'):
        runner._transfer_map(path)
    write([{**row,'score':float('nan')}])
    with pytest.raises(ValueError, match='finite'):
        runner._transfer_map(path)


def test_cached_annotations_are_input_bound_and_version_validated(tmp_path):
    import sqlite3
    import zlib
    from tools.chembl_tool.common.record_diversity import prepare_record, BOUND_ANNOTATION_VERSION
    p = prepare_record(dict(canonical_record_id='r', canonical_endpoint_name='mutation',
                            support_text='The compound was negative.', tested_entity_role='test_compound'))
    a = {'outcomes':[{'endpoint':'mutation','kind':'negative','focal_scope':'focal',
         'evidence':[{'field':'support_text','quote':'The compound was negative.'}],
         'conditions':[], 'comparator':[], 'statistical_support':'not_reported','attribution_basis':'record_fields'}],
         'assay':[], 'system':[], 'exposure':[], 'comparison_role':[], 'unresolved':[]}
    db_path = tmp_path/'annotations.sqlite'
    pack = lambda v: zlib.compress(json.dumps(v).encode())
    with sqlite3.connect(db_path) as db:
        db.executescript('CREATE TABLE metadata(key TEXT,value TEXT); CREATE TABLE records(task TEXT,record_id TEXT,job_id INTEGER); '
            'CREATE TABLE jobs(id INTEGER,payload BLOB); CREATE TABLE annotations(job_id INTEGER,annotation BLOB);')
        db.execute('INSERT INTO metadata VALUES (?,?)',('contract',json.dumps({'model_config':{'annotation_version':BOUND_ANNOTATION_VERSION}})))
        db.execute('INSERT INTO records VALUES (?,?,?)',('ames',p['record_id'],1))
        db.execute('INSERT INTO jobs VALUES (?,?)',(1,pack({'input':p['input']})))
        db.execute('INSERT INTO annotations VALUES (?,?)',(1,pack(a)))
    assert retrieval.cached_annotations(db_path,'ames',{'card':{'prepared':p}})['card']['outcomes'][0]['kind'] == 'negative'
    changed = deepcopy(p); changed['input']['support_text'] = 'A different study.'
    assert retrieval.cached_annotations(db_path,'ames',{'card':{'prepared':changed}}) == {}


def test_assessed_cache_rejects_stale_prompt_and_requires_all_native_alternatives(tmp_path):
    import hashlib
    import sqlite3
    import zlib
    from tools.chembl_tool.common.record_diversity import prepare_record, QUALITY_ANNOTATION_VERSION, extraction_messages
    p=prepare_record(dict(canonical_record_id='a',canonical_endpoint_name='mutation',support_text='Negative.'))
    other=prepare_record(dict(canonical_record_id='b',canonical_endpoint_name='mutation',support_text='Positive.'))
    def annotation(text,kind):
        quote={'field':'support_text','quote':text}
        return {'outcomes':[{'endpoint':'mutation','kind':kind,'focal_scope':'focal','evidence':[quote],
            'conditions':[],'comparator':[],'statistical_support':'not_reported','attribution_basis':'record_fields',
            'evidence_basis':'measured','basis_evidence':[quote],'result_scope':'focal_effect','limitations':[]}],
            'assay':[],'system':[],'exposure':[],'comparison_role':[],'unresolved':[]}
    path=tmp_path/'assessed.sqlite'
    pack=lambda v:zlib.compress(json.dumps(v).encode())
    contract={'model_config':{'annotation_version':QUALITY_ANNOTATION_VERSION},'system_prompt_sha256':'old'}
    bound={'prepared':p,'alternatives':[p,other],'ambiguous_native_metadata':True}
    with sqlite3.connect(path) as db:
        db.executescript('CREATE TABLE metadata(key TEXT,value TEXT); CREATE TABLE records(task TEXT,record_id TEXT,job_id INTEGER); '
            'CREATE TABLE jobs(id INTEGER,payload BLOB); CREATE TABLE annotations(job_id INTEGER,annotation BLOB);')
        db.execute('INSERT INTO metadata VALUES (?,?)',('contract',json.dumps(contract)))
        db.execute('INSERT INTO records VALUES (?,?,?)',('ames','a',1))
        db.execute('INSERT INTO jobs VALUES (?,?)',(1,pack(p)))
        db.execute('INSERT INTO annotations VALUES (?,?)',(1,pack(annotation('Negative.','negative'))))
    with pytest.raises(ValueError,match='different extraction prompt'):
        retrieval.cached_annotations(path,'ames',{'card':bound})
    contract['system_prompt_sha256']=hashlib.sha256(extraction_messages({'input':{}},QUALITY_ANNOTATION_VERSION)[0]['content'].encode()).hexdigest()
    with sqlite3.connect(path) as db:
        db.execute('UPDATE metadata SET value=?',(json.dumps(contract),))
    assert retrieval.cached_annotations(path,'ames',{'card':bound})=={}
    with sqlite3.connect(path) as db:
        db.execute('INSERT INTO records VALUES (?,?,?)',('ames','b',2))
        db.execute('INSERT INTO jobs VALUES (?,?)',(2,pack(other)))
        db.execute('INSERT INTO annotations VALUES (?,?)',(2,pack(annotation('Positive.','positive'))))
    result=retrieval.cached_annotations(path,'ames',{'card':bound})['card']
    assert len(result['alternatives'])==2
    assert {a['outcomes'][0]['kind'] for a in result['alternatives'].values()}=={'positive','negative'}


@pytest.mark.parametrize('corruption', ['model', 'input', 'request', 'label'])
def test_prepared_replay_rejects_changed_contract_before_client(tmp_path, monkeypatch, corruption):
    from tools.chembl_tool.common.json_utils import write_json_atomic, sha256_file
    from tools.chembl_tool.common.openai_provider_pool import ProviderPoolConfig, ProviderSpec
    source = tmp_path / 'test.jsonl'
    source.write_text('{"drug":"CCO","Y":1}\n')
    provider = ProviderPoolConfig(providers=(ProviderSpec(name='test', base_url='http://example.invalid',
        model='model', api_key_env='', max_inflight=1, request_extra_body={}),), max_failovers=0)
    manifest = {'version':runner.VERSION, 'model':'model', 'max_tokens':100,
        'generation':runner.runtime._generation_settings(provider), 'request_extra_body':{},
        'inputs':{'bbb_martins':{'input':{'path':str(source),'sha256':sha256_file(source)}}},
        'setting':'direct_indirect','selection':{'policy':'neighbor_fill','budget':1}}
    messages = [{'role':'user','content':'evidence'}]
    query = tmp_path / 'runs/bbb_martins/query_00000'
    prepared = {'manifest_hash':content_hash(manifest),'messages':messages,'request_hash':content_hash(messages),
        'query_hash':content_hash({'benchmark_profile':'','canonical_smiles':'CCO','condition_group':''})}
    write_json_atomic(tmp_path / 'experiment_manifest.json',manifest)
    if corruption == 'input': source.write_text('{"drug":"CCC","Y":1}\n')
    if corruption == 'request': prepared['messages'][0]['content']='changed'
    write_json_atomic(query / 'prepared.json',prepared)
    write_json_atomic(query / 'evaluation.json',{'label':0 if corruption=='label' else 1})
    args = Namespace(output_root=str(tmp_path), tasks=['bbb_martins'], model='other' if corruption=='model' else 'model',
        max_tokens=100, matched_progressive_root='', baseline_suite=False, skip_tool_prefetch=False,
        retry_race_width=1,parallelism=1,endpoint_concurrency_budget=1,benchmark_manifest='',indices=None,limit=0)
    monkeypatch.setattr(runner.runtime,'_resolve_provider_pool_config',lambda args:provider)
    monkeypatch.setattr(runner.runtime,'_make_client',lambda *args:pytest.fail('must reject before network client'))
    with pytest.raises(ValueError,match='mismatch|changed'):
        runner._run_prepared(args)
