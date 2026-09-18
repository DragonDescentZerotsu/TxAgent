"""Exercise cache-assignment selection, source joins, score scope, and fail-closed inputs."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from predict.harnesses.progressive import runner
from predict.harnesses.progressive import _records as profile
from predict.harnesses.progressive.prompt import prompt_assets
from predict.retrieval.assay_reranking import cache_matched as matched, v24_1_levels, v9
from predict.retrieval.policies import normalize_molecule_identity as identity
from predict.utils.json import sha256_file


@pytest.fixture
def cache_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(matched, 'DEFAULT_GOLD_CONTEXT_MAPPING', None)
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    library = tmp_path/'library'
    source = library/'03_pair_buckets/records.parquet'
    source.parent.mkdir(parents=True)
    mapping = tmp_path/'levels.parquet'
    mapper = tmp_path/'mapper.json'
    source_contract = library/'02_canonicalized/source_contract.json'
    fields = ['measurement_text', 'qualifying_conditions']
    save(source_contract, dict(contract_version='source_column_contract.v1', sources={'toy': {
        'normalized_artifact_columns': {k: {'source_or_simply_cleaned': True} for k in fields}}}))
    parents = ['C' * i + 'O' for i in range(1, 76)]
    rows, memberships = [], []
    for level in range(1, 6):
        for i, smiles in enumerate(parents):
            for condition in (['rat', 'mouse'] if level == 1 else ['rat']):
                rid = f'{level}_{i}_{condition}'
                rows.append(dict(canonical_record_id=rid, source_row_uid=rid, canonical_smiles=smiles,
                    source_id='toy', measurement_kind='continuous', measurement_text=str(i),
                    qualifying_conditions=condition))
                memberships.append(dict(source_row_uid=rid, level=level,
                    family_key=prompt_assets('reranked_progressive_v8')['levels']['bbb_martins'][f'L{level}']['evidence_family']))
    pd.DataFrame(rows).to_parquet(source)
    pd.DataFrame(memberships).to_parquet(mapping)
    save(mapper, {'tasks': {'bbb_martins': {'path': mapping.name, 'sha256': sha256_file(mapping)}}})
    stage = save(source.with_name('manifest.json'), {'task_id': 'bbb_martins', 'outputs': {'records.parquet': sha256_file(source)}})
    query = dict(benchmark_row_id='q', drug='c1ccccc1', Y=1, condition_group='none')
    train = [dict(benchmark_row_id=f'gold_{i}', drug=smiles, Y=1,
                  condition_group='none', molecule_identity_key=identity(smiles).parent_inchi_key)
             for i, smiles in enumerate(parents)]
    for name in ['train.jsonl', 'train_molecule_condition_labels.jsonl']:
        (tmp_path/name).write_text(''.join(json.dumps(row)+'\n' for row in train))
    for name in ['valid.jsonl', 'valid_molecule_condition_labels.jsonl']:
        (tmp_path/name).write_text(json.dumps(query)+'\n')
    monkeypatch.setattr(matched, 'split_path', lambda task, split: tmp_path/f'{split}.jsonl')
    gold_rows = [dict(retrieval_molecule_identity_key=identity(s).parent_inchi_key,
        retrieval_smiles=s, retrieval_record_id=f'gold_{i}', retrieval_condition_group='none',
        query_record_id='q', model_rank=74-i,
        morgan_tanimoto_similarity=0.0, prob_transfer=i/75) for i,s in enumerate(parents)]
    rankings = tmp_path/'v9/rankings.parquet'
    rankings.parent.mkdir(parents=True)
    pd.DataFrame(gold_rows).to_parquet(rankings)
    v9path = save(tmp_path/'v9/VERSION.json', {
        'schema_version': v9.RANKING_SCHEMA_VERSION, 'status': 'complete',
        'candidate_policy': 'morgan_top75_distinct_gold_train_parents_then_all_context_rows',
        'morgan_pool_size': 75, 'rankings_sha256': sha256_file(rankings),
        'model': v9.model_profile('bbb_martins'), 'prompt_assets': v9.verify_vendored_assets(),
        'training_record_ids': [row['benchmark_row_id'] for row in train],
        'inputs': {
            key: value
            for split in ['train', 'valid']
            for key, value in (
                (split, str(tmp_path/f'{split}_molecule_condition_labels.jsonl')),
                (f'{split}_sha256', sha256_file(tmp_path/f'{split}_molecule_condition_labels.jsonl')),
            )
        }})
    monkeypatch.setattr(profile, 'load_candidates', lambda **kw: ({'q':gold_rows}, {
        'rankings_sha256': sha256_file(v9path.parent/'rankings.parquet')}))
    cachepath = tmp_path/'later/VERSION.json'
    cachepath.parent.mkdir()
    db = cachepath.with_name('scores.sqlite3')
    with sqlite3.connect(db) as c:
        c.executescript('''CREATE TABLE queries(query_id,query_smiles);
            CREATE TABLE groups_dim(group_key,group_id);
            CREATE TABLE molecules(molecule_key,molecule_chembl_id);
            CREATE TABLE records(record_key,external_record_id);
            CREATE TABLE scores(score_key,transfer_probability);
            CREATE TABLE assignments(query_id,group_key,molecule_key,record_key,score_key);''')
        c.execute('INSERT INTO queries VALUES (1,?)', (query['drug'],))
        c.executemany('INSERT INTO molecules VALUES (?,?)', [(i,identity(s).parent_inchi_key) for i,s in enumerate(parents)])
        for level in range(2,5):
            c.execute('INSERT INTO groups_dim VALUES (?,?)', (level,f'L{level}'))
            for i in range(75):
                key = level*100+i
                c.execute('INSERT INTO records VALUES (?,?)', (key,f'{level}_{i}_rat'))
                c.execute('INSERT INTO scores VALUES (?,?)', (key,i/75))
                c.execute('INSERT INTO assignments VALUES (1,?,?,?,?)', (level,i,key,key))
    files = dict(stage3_records=source, stage3_manifest=stage, source_contract=source_contract,
        level_manifest=mapper, level_mapping=mapping, queries=tmp_path/'valid_molecule_condition_labels.jsonl')
    cache = dict(status='complete', task_id='bbb_martins', subset='valid', profile=v24_1_levels.PROFILE,
        schema_version='txagent_assay_transfer_compact_cache.v1', models=v24_1_levels.MODELS,
        levels_in_cache=['L2','L3','L4'], neighbor_identity_policy='scaffold_disjoint', morgan_pool_size=75,
        implementation_sha256=sha256_file(Path(v24_1_levels.__file__)),
        cache_sha256=sha256_file(db), cache_quick_check='ok',
        inputs={key: dict(path=str(path),sha256=sha256_file(path)) for key,path in files.items()})
    save(cachepath, cache)
    policy = dict(stages={f'L{i}':'morgan' for i in range(1,6)},
        cache_manifests={'L1':str(v9path), **{f'L{i}':str(cachepath) for i in range(2,5)}}, inputs={})
    return dict(queries={'q':query['drug']}, task='bbb_martins', subset='valid', library=library,
                mapper=mapper, policy=policy, molecule_limit=2, l1_limit=2, later_limit=2)


@pytest.mark.parametrize('invalid_uid', [False, True])
def test_approved_empty_context_retirement_precedes_selection(cache_inputs, invalid_uid):
    library = cache_inputs['library']
    base = cache_inputs['mapper'].parent
    parent = identity('N#N').parent_inchi_key
    context = dict(benchmark_row_id='retired', drug='N#N', Y=0,
                   condition_group='none', molecule_identity_key=parent,
                   source_record_ids=['claim'])
    existing = [json.loads(line) for line in (base/'train.jsonl').read_text().splitlines()]
    for name in ('train.jsonl', 'train_molecule_condition_labels.jsonl'):
        (base/name).write_text(''.join(json.dumps(row)+'\n' for row in [*existing, context]))
    v9path = Path(cache_inputs['policy']['cache_manifests']['L1'])
    version = json.loads(v9path.read_text())
    version['inputs']['train_sha256'] = sha256_file(base/'train_molecule_condition_labels.jsonl')
    version['training_record_ids'] = [row['benchmark_row_id'] for row in [*existing, context]]
    ranks = pd.read_parquet(v9path.with_name('rankings.parquet'))
    ranks = pd.concat([pd.DataFrame([dict(retrieval_record_id='retired',
        retrieval_molecule_identity_key=parent, retrieval_smiles='N#N',
        retrieval_condition_group='none', query_record_id='q', model_rank=-1,
        morgan_tanimoto_similarity=0., prob_transfer=1.)]), ranks.iloc[1:]], ignore_index=True)
    ranks.to_parquet(v9path.with_name('rankings.parquet'))
    version['rankings_sha256'] = sha256_file(v9path.with_name('rankings.parquet'))
    v9path.write_text(json.dumps(version))
    path = library/'reviews/l1_training_context_retirement/manifest.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(version='l1_training_context_retirement.v1', task_id='bbb_martins',
        stage3_records_sha256=sha256_file(library/'03_pair_buckets/records.parquet'),
        gold_train_sha256=version['inputs']['train_sha256'],
        mapper_sha256=sha256_file(base/'levels.parquet'), contexts=[dict(context_id='retired',
            parent=parent, source_record_ids=['claim'],
            source_row_uids=['1_0_rat' if invalid_uid else 'unmapped'])])))
    if invalid_uid:
        with pytest.raises(ValueError, match='Retired training context'):
            matched.load_candidates(**cache_inputs)
    else:
        molecules, _, _ = matched.load_candidates(**cache_inputs)
        assert len(molecules['q']) == 2
        assert all(m['reference_molecule_id'] != parent for m in molecules['q'])


def test_matched_pools_and_record_score_scope(cache_inputs):
    morgan = matched.load_candidates(**cache_inputs)
    assay_inputs = deepcopy(cache_inputs)
    assay_inputs['policy']['stages'] = {f'L{i}':'morgan' if i == 5 else 'assay_transfer' for i in range(1,6)}
    assay = matched.load_candidates(**assay_inputs)
    assert {k:v['candidate_sha256'] for k,v in morgan[2]['query_audits']['q'].items()} == {
        k:v['candidate_sha256'] for k,v in assay[2]['query_audits']['q'].items()}
    assert [r['record_id'] for r in assay[1]['q']['L2']['records']] == ['2_74_rat','2_73_rat']
    assert morgan[1]['q']['L5']['records'] == assay[1]['q']['L5']['records']
    molecules = assay[0]['q']
    assert all('transfer_likelihood' in m for m in molecules)
    assert all('transfer_likelihood' not in r for m in molecules for r in m['l1_records'])
    assert all({r['payload']['source_fields']['qualifying_conditions'] for r in m['l1_records']} == {'rat','mouse'} for m in molecules)
    snapshots = profile.stage_ranked_snapshots(molecules, task='bbb_martins', records_by_level=assay[1]['q'], prompt_version='reranked_progressive_v8')
    assert all('transfer_likelihood' not in c and 'morgan_similarity' not in c
               for m in snapshots[1].values() for c in m['cards'].values())
    assert matched.load_candidates(**assay_inputs) == assay
    with pytest.raises(ValueError, match='one fixed pool'):
        matched.load_candidates(**assay_inputs,cache_pool='all')


@pytest.mark.parametrize('failure', ['source_hash','mapper_hash','task','level','missing_score','parent'])
def test_cache_failures_are_not_morgan_fallback(cache_inputs, failure):
    path = Path(cache_inputs['policy']['cache_manifests']['L2'])
    version = json.loads(path.read_text())
    if failure in {'source_hash','mapper_hash'}:
        key = 'stage3_records' if failure == 'source_hash' else 'level_manifest'
        version['inputs'][key]['sha256'] = 'wrong'
    elif failure == 'task':
        version['task_id'] = 'bioavailability_ma'
    elif failure == 'level':
        version['levels_in_cache'] = ['L3','L4']
        version['models'] = {k: v for k,v in version['models'].items() if k != 'L2'}
    else:
        db = path.with_name('scores.sqlite3')
        with sqlite3.connect(db) as c:
            c.execute('DELETE FROM scores WHERE score_key=200' if failure == 'missing_score'
                      else "UPDATE molecules SET molecule_chembl_id='wrong' WHERE molecule_key=0")
        version['cache_sha256'] = sha256_file(db)
    path.write_text(json.dumps(version))
    with pytest.raises(ValueError):
        matched.load_candidates(**cache_inputs)


def test_cli_inputs_and_offline_provider_do_not_resolve_credentials(tmp_path, monkeypatch):
    from data.processing import llm_api

    args = runner.parse_args(['--tasks','bbb_martins','--evidence-library',f'bbb_martins={tmp_path}',
        '--l1-molecules','20','--l1-records-per-molecule','4','--records-per-level','25',
        '--query-prior','none','--prepare-only','--skip-tool-prefetch'])
    assert args.evidence_libraries == {'bbb_martins':tmp_path}
    assert (args.context_limit,args.record_limit_per_context_level,args.indirect_record_limit_per_level) == (20,4,25)
    def forbidden(*a, **kw):
        raise AssertionError('Offline preparation must not resolve provider credentials')
    monkeypatch.setattr(llm_api,'openai_compatible_client',forbidden)
    assert len(runner._resolve_provider_pool_config(args).providers) >= 1
    args.provider_pool_config = ''
    args.base_url = 'https://openrouter.ai/api/v1'
    assert len(runner._resolve_provider_pool_config(args).providers) == 1
    for flags in [['--context-limit','10'], ['--query-prior','fresh'], ['--v7-root','x'],
                  ['--evidence-root','x'], ['--benchmark-data-root','x'], ['--single-source-root','x']]:
        with pytest.raises(SystemExit):
            runner.parse_args(flags)


def test_summary_uses_the_same_level_catalog_as_preparation(tmp_path, monkeypatch):
    observed = []
    monkeypatch.setattr(runner, '_write_prediction_summary', lambda **kw: observed.append(kw['metric_fields']['family']))
    args = runner.parse_args(['--tasks','bioavailability_ma'])
    runner._summarize_task(task='bioavailability_ma', records=[{'Y':1}], indices=[0],
        output_root=tmp_path, max_level=0, query_prior_mode='none', profile='context_records',
        context_record_l3_l5=True, prompt_version=args.assay_transfer_prompt_version)
    assert observed == [row['endpoint_group'] for row in runner._run_levels(args,'bioavailability_ma')]


def test_missing_l1_records_stop_instead_of_refilling(cache_inputs):
    mapper = Path(cache_inputs['mapper'])
    manifest = json.loads(mapper.read_text())
    mapping = mapper.parent/manifest['tasks']['bbb_martins']['path']
    rows = pd.read_parquet(mapping)
    rows.loc[rows.level == 1, 'family_key'] = 'central_functional_access_proxy'
    rows.loc[rows.level == 1, 'level'] = 2
    rows.to_parquet(mapping)
    manifest['tasks']['bbb_martins']['sha256'] = sha256_file(mapping)
    mapper.write_text(json.dumps(manifest))
    cache_inputs['policy']['stages'] = {'L1':'morgan'}
    cache_inputs['policy']['cache_manifests'] = {'L1':cache_inputs['policy']['cache_manifests']['L1']}
    with pytest.raises(ValueError,match='no mapped V10 L1 records'):
        matched.load_candidates(**cache_inputs)


@pytest.fixture
def mapped_context_inputs(cache_inputs):
    args = deepcopy(cache_inputs)
    args['policy']['stages'] = {'L1':'morgan'}
    args['policy']['cache_manifests'] = {'L1':args['policy']['cache_manifests']['L1']}
    root = args['library'].parent
    source = args['library']/'03_pair_buckets/records.parquet'
    records = pd.read_parquet(source)
    gold, _ = profile.load_candidates()
    ranks = deepcopy(gold['q'])
    train, membership = [], []
    destination = ranks[1]['retrieval_molecule_identity_key']
    for i, rank in enumerate(ranks):
        cid = rank['retrieval_record_id']
        train.append(dict(benchmark_row_id=cid, drug=rank['retrieval_smiles'], Y=1,
            molecule_identity_key=rank['retrieval_molecule_identity_key'], condition_group='none',
            source_record_ids=[f'1_{i}_rat',f'1_{i}_mouse'], label_counts={'1':2}))
        rank.update(query_record_id='q', retrieval_condition_group='none', prompt_hash='not-used-by-Morgan')
        for condition in ('rat','mouse'):
            rid = f'1_{i}_{condition}'
            membership.append(dict(source_context_id=cid, source_record_id=rid,
                destination_context_id='gold_1' if i == 0 else cid, canonical_record_id=rid,
                source_row_uid=rid, source_parent=rank['retrieval_molecule_identity_key'],
                destination_parent=destination if i == 0 else rank['retrieval_molecule_identity_key'],
                level='1', status='redistributed' if i == 0 else 'unchanged'))
    records.loc[records.canonical_record_id.isin(['1_0_rat','1_0_mouse']), 'canonical_smiles'] = ranks[1]['retrieval_smiles']
    records.to_parquet(source)
    stage = source.with_name('manifest.json')
    stage.write_text(json.dumps(dict(task_id='bbb_martins',outputs={'records.parquet':sha256_file(source)})))
    for name in ['train.jsonl','train_molecule_condition_labels.jsonl']:
        (root/name).write_text(''.join(json.dumps(r)+'\n' for r in train))
    v9path = Path(args['policy']['cache_manifests']['L1'])
    version = json.loads(v9path.read_text())
    version['inputs']['train_sha256'] = sha256_file(root/'train_molecule_condition_labels.jsonl')
    version['training_record_ids'] = [row['benchmark_row_id'] for row in train]
    pd.DataFrame(ranks).to_parquet(v9path.with_name('rankings.parquet'))
    version['rankings_sha256'] = sha256_file(v9path.with_name('rankings.parquet'))
    v9path.write_text(json.dumps(version))
    table = root/'gold_context_record_mapping.tsv'
    pd.DataFrame(membership).to_csv(table, sep='\t', index=False)
    mapper = json.loads(args['mapper'].read_text())
    receipt = dict(version='bbb_gold_context_record_mapping.v2',task_id='bbb_martins',
        stage3_records_sha256=sha256_file(source),gold_train_sha256=version['inputs']['train_sha256'],
        mapper_sha256=mapper['tasks']['bbb_martins']['sha256'],mapping_sha256=sha256_file(table),
        retired_source_contexts=['gold_0'],label_audit=[dict(context_id='gold_1',retired=False,label_counts_after={'1':4})])
    args['gold_context_mapping'] = root/'mapping_manifest.json'
    args['gold_context_mapping'].write_text(json.dumps(receipt))
    return args


def test_mapping_retires_context_and_joins_records(mapped_context_inputs):
    args = mapped_context_inputs
    args['molecule_limit'] = 74
    args['l1_limit'] = 10
    molecules, _, audit = matched.load_candidates(**args)
    merged = next(m for m in molecules['q'] if m['gold_context_ids']['morgan']=='gold_1')
    assert {r['record_id'] for r in merged['l1_records']} == {'1_0_rat','1_0_mouse','1_1_rat','1_1_mouse'}
    assert all(m['gold_context_ids']['morgan']!='gold_0' for m in molecules['q'])
    assert audit['gold_context_mapping']['destination_contexts']==74
    assert str(args['gold_context_mapping']) in audit['inputs']


@pytest.mark.parametrize('failure', ['hash','uid','parent','missing','stale_score'])
def test_context_mapping_fails_closed(mapped_context_inputs, failure):
    args = mapped_context_inputs
    p = args['gold_context_mapping']
    receipt = json.loads(p.read_text())
    table = p.with_name('gold_context_record_mapping.tsv')
    rows = pd.read_csv(table,sep='\t',dtype=str)
    if failure=='hash': receipt['stage3_records_sha256']='wrong'
    elif failure=='stale_score': args['policy']['stages']['L1']='assay_transfer'
    else:
        if failure=='missing': rows=rows.iloc[1:]
        else: rows.loc[0, 'source_row_uid' if failure=='uid' else 'destination_parent']='wrong'
        rows.to_csv(table,sep='\t',index=False)
        receipt['mapping_sha256']=sha256_file(table)
    p.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='needs rescoring' if failure=='stale_score' else 'mapping|hash'):
        matched.load_candidates(**args)


def test_only_exact_rendered_transfer_inputs_can_reuse_scores(mapped_context_inputs):
    from predict.retrieval.assay_reranking.v9 import V9PromptRenderer
    import hashlib
    args = mapped_context_inputs
    args['policy']['stages']['L1']='assay_transfer'
    ranks_path = Path(args['policy']['cache_manifests']['L1']).with_name('rankings.parquet')
    ranks = pd.read_parquet(ranks_path)
    renderer = V9PromptRenderer('bbb_martins')
    ranks['prompt_hash'] = [hashlib.sha256(renderer.render(
        dict(smiles=s,value=1,condition_group='none'),
        dict(smiles=args['queries']['q'],condition_group='none')).encode()).hexdigest()
        for s in ranks.retrieval_smiles]
    ranks.to_parquet(ranks_path)
    version_path = Path(args['policy']['cache_manifests']['L1'])
    version = json.loads(version_path.read_text())
    version['rankings_sha256'] = sha256_file(ranks_path)
    version_path.write_text(json.dumps(version))
    _, _, audit = matched.load_candidates(**args)
    assert audit['gold_context_mapping']['score_policy']=='exact_rendered_prompt_hash'
    path = args['gold_context_mapping']
    receipt = json.loads(path.read_text())
    receipt['label_audit'][0]['label_counts_after']={'1':3,'0':1}
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='needs rescoring'):
        matched.load_candidates(**args)
    args['allow_frozen_l1_vote_scores']=True
    _, _, accepted = matched.load_candidates(**args)
    mapping = accepted['gold_context_mapping']
    assert mapping['score_policy']=='frozen_pre_redistribution_vote_scores'
    assert mapping['changed_vote_contexts']=={'gold_1':dict(frozen_label_counts={'1':2},mapped_label_counts={'1':3,'0':1})}
    assert accepted['contract']['allow_frozen_l1_vote_scores'] is True
    ranks.loc[ranks.retrieval_record_id=='gold_1','prompt_hash']='corrupted'
    ranks.to_parquet(ranks_path)
    version = json.loads(version_path.read_text())
    version['rankings_sha256'] = sha256_file(ranks_path)
    version_path.write_text(json.dumps(version))
    with pytest.raises(ValueError,match='not a mapped vote-percentage change'):
        matched.load_candidates(**args)


@pytest.mark.parametrize('flag', ['--record_pool', '-record_pool'])
def test_active_record_pool_cli_selects_all_view(flag):
    args=runner.parse_args([flag, 'all'])
    assert args.record_pool == args.cache_pool == 'all'


def test_record_pool_cli_default_and_invalid_choices():
    assert runner.parse_args([]).record_pool == 'all'
    for argv in [
        ['--record_pool', 'assay-transfer-trained'],
        ['--record_pool', 'all_transfer_eligible'],
        ['--record_pool','tool-accepted'],
        ['--cache-pool','all'],
    ]:
        with pytest.raises(SystemExit):
            runner.parse_args(argv)


def test_active_cache_supports_per_level_limits_above_100():
    args = runner.parse_args([
        '--records-per-level', '7',
        '--level-record-limit', 'L2=3',
        '--level-record-limit', 'L5=8',
    ])
    assert runner._indirect_record_limits(args, 'bioavailability_ma') == {
        'L2': 3, 'L3': 7, 'L4': 7, 'L5': 8, 'L6': 7,
    }
    args = runner.parse_args([
        '--tasks', 'bbb_martins',
        '--records-per-level', '101', '--level-record-limit', 'L2=121',
    ])
    assert runner._indirect_record_limits(args, 'bbb_martins') == {
        'L2': 121, 'L3': 101, 'L4': 101, 'L5': 101,
    }
    for argv in [['--records-per-level', '0'], ['--level-record-limit', 'L2=0']]:
        with pytest.raises(SystemExit):
            runner.parse_args(argv)


def test_frozen_vote_scores_require_explicit_cli_opt_in():
    assert runner.parse_args(['--tasks','bbb_martins']).allow_frozen_l1_vote_scores is False
    assert runner.parse_args(['--tasks','bbb_martins','--allow-frozen-l1-vote-scores']).allow_frozen_l1_vote_scores is True


@pytest.mark.parametrize('pool_size,public,stored', [
    (3,'assay-transfer-trained','tool-accepted'),
    (75,'all_transfer_eligible','tool-compatible'), (200,'all','all')])
def test_three_pool_schema_uses_declared_parent_count(cache_inputs, pool_size, public, stored):
    from predict.retrieval.assay_reranking import three_pools, runtime
    args = deepcopy(cache_inputs)
    args['cache_pool'] = {
        'assay-transfer-trained': 'tool-accepted',
        'all_transfer_eligible': 'tool-compatible',
        'all': 'all',
    }[public]
    args['policy']['stages']={'L1':'morgan','L2':'assay_transfer'}
    source = args['library']/'03_pair_buckets/records.parquet'
    records = pd.read_parquet(source)
    mapper = json.loads(args['mapper'].read_text())
    mapping_path = args['mapper'].parent/mapper['tasks']['bbb_martins']['path']
    memberships = pd.read_parquet(mapping_path)
    extra, extra_levels = [], []
    for i in range(75,pool_size):
        rid=f'2_{i}_rat'
        extra.append(dict(records.loc[records.canonical_record_id=='2_0_rat'].iloc[0],
                          canonical_record_id=rid,source_row_uid=rid,canonical_smiles='C'*(i+1)+'O'))
        extra_levels.append(dict(memberships.loc[memberships.source_row_uid=='2_0_rat'].iloc[0],source_row_uid=rid))
    if extra:
        records=pd.concat([records,pd.DataFrame(extra)],ignore_index=True)
        memberships=pd.concat([memberships,pd.DataFrame(extra_levels)],ignore_index=True)
        records.to_parquet(source); memberships.to_parquet(mapping_path)
        mapper['tasks']['bbb_martins']['sha256']=sha256_file(mapping_path)
        args['mapper'].write_text(json.dumps(mapper))
        source.with_name('manifest.json').write_text(json.dumps(dict(task_id='bbb_martins',outputs={'records.parquet':sha256_file(source)})))
    path=args['library'].parent/'multi/VERSION.json'
    path.parent.mkdir()
    db=path.with_name('scores.sqlite3')
    with sqlite3.connect(db) as c:
        c.executescript('''CREATE TABLE benchmark_queries(benchmark_row_id,drug,query_id);
            CREATE TABLE records(record_key,external_record_id,parent_id);
            CREATE TABLE scores(score_key,transfer_probability);
            CREATE TABLE assignments(query_id,pool,level,record_key,score_key);''')
        c.execute('INSERT INTO benchmark_queries VALUES (?,?,1)',('q',args['queries']['q']))
        for i in range(pool_size):
            c.execute('INSERT INTO records VALUES (?,?,?)',(i,f'2_{i}_rat',identity('C'*(i+1)+'O').parent_inchi_key))
            c.execute('INSERT INTO scores VALUES (?,?)',(i,i/pool_size))
            c.execute('INSERT INTO assignments VALUES (1,?,?,?,?)',(stored,'L2',i,i))
    files=dict(records=source,source_contract=args['library']/'02_canonicalized/source_contract.json',
        mapping=mapping_path,mapping_manifest=args['mapper'],queries=args['library'].parent/'valid_molecule_condition_labels.jsonl',
        builder=Path(three_pools.__file__))
    version=dict(schema_version=three_pools.SCHEMA,status='complete',task_id='bbb_martins',subset='valid',
        models={'L2':v24_1_levels.MODELS['L2']},pools=[stored],morgan_pool_size=pool_size,
        morgan_fingerprint=dict(radius=2,bits=2048,similarity='Tanimoto'),
        neighbor_identity_policy='all_parent_forms_scaffold_disjoint',scoring_contract_version=runtime.SCORING_CONTRACT_VERSION,
        inputs={k:dict(path=str(p),sha256=sha256_file(p)) for k,p in files.items()},cache_sha256=sha256_file(db),cache_integrity='ok')
    path.write_text(json.dumps(version))
    args['policy']['cache_manifests']={'L1':args['policy']['cache_manifests']['L1'],'L2':str(path)}
    _, later, audit=matched.load_candidates(**args)
    assert later['q']['L2']['candidate_molecules']==pool_size
    assert later['q']['L2']['records'][0]['record_id']==f'2_{pool_size-1}_rat'
    assert audit['cache_pool']==stored
    args['cache_pool']='missing'
    with pytest.raises(ValueError,match='Unknown pool'):
        matched.load_candidates(**args)
    args['cache_pool']=stored
    version['status']='prepared'
    path.write_text(json.dumps(version))
    with pytest.raises(ValueError,match='not finalized'):
        matched.load_candidates(**args)


def test_library_compatibility_requires_all_cached_level_fields(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    old,new=tmp_path/'old.parquet',tmp_path/'new.parquet'
    rows=[dict(canonical_record_id='a',source_row_uid='a',canonical_smiles='CO',value=1),
          dict(canonical_record_id='b',source_row_uid='b',canonical_smiles='CCO',value=2)]
    membership=pd.DataFrame([dict(source_row_uid='a',level=1),dict(source_row_uid='b',level=2)])
    pq.write_table(pa.Table.from_pylist(rows),old)
    rows[0]['canonical_smiles']='CCC'
    pq.write_table(pa.Table.from_pylist(rows),new)
    assert matched.verify_cached_level_records(old,new,membership,['L2'])['records']==1
    rows[1]['value']=3
    pq.write_table(pa.Table.from_pylist(rows),new)
    with pytest.raises(ValueError,match='Cached-level library records changed'):
        matched.verify_cached_level_records(old,new,membership,['L2'])


def test_l1_pool_metadata_can_select_a_different_existing_width(mapped_context_inputs):
    args = mapped_context_inputs
    path=Path(args['policy']['cache_manifests']['L1'])
    version=json.loads(path.read_text())
    _,_,audit=matched.load_candidates(**args)
    assert audit['l1_pool_size']==75
    version['morgan_pool_size']=200
    path.write_text(json.dumps(version))
    with pytest.raises(ValueError,match='declared parent count'):
        matched.load_candidates(**args)


def test_parcc_uses_shared_credential_loader(monkeypatch):
    from data.processing import llm_api

    observed=[]
    monkeypatch.setattr(
        llm_api,
        'openai_compatible_client',
        lambda **kwargs: (observed.append(kwargs) or (object(), 'LITE_LLM_KEY')),
    )
    args=runner.parse_args(['--tasks','bbb_martins','--provider-pool-config',''])
    config=runner._resolve_provider_pool_config(args)
    assert config.providers[0].base_url=='https://litellm.parcc.upenn.edu/v1'
    runner._make_client(args, config)
    assert observed[0]['credential_env']=='LITE_LLM_KEY'
