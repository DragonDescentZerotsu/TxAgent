"""Exercise per-stage ranking, independent L2 selection, card routing and resume."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from predict.harnesses.progressive import runner
from predict.harnesses.progressive import _records as profile
from predict.harnesses.progressive.prompt import (
    prompt_asset_manifest, prompt_assets, render_progressive_messages,
)
from predict.harnesses.progressive.inference import _derive_claim_provenance
from predict.harnesses.progressive.state import progressive_state_errors
from predict.harnesses.progressive.level_selection import (
    _active_heldout_parents,
    load_mapped_candidates,
    load_retrieval_policy,
    select_policy_records,
)
from predict.retrieval.policies import normalize_molecule_identity
from predict.utils.json import sha256_file
from predict.harnesses.progressive._visibility import validate_visible_messages


VERSION = 'reranked_progressive_v8'
SIMPLE_L1_VERSION = 'reranked_progressive_l1_simple_v1'
MOLECULE_METADATA_VERSION = (
    'reranked_progressive_l1_context_l2_semantic_molecule_metadata_v1'
)
MOLECULE_BUCKET_VERSION = 'reranked_progressive_l1_context_l2_morgan_bucket_v1'


def test_progressive_heldout_parents_follow_active_gold_release():
    parents, inputs = _active_heldout_parents('bbb_martins')
    active = Path('data/gold_labels/BBB_Martins/CURRENT').read_text().strip()
    assert active == 'v1'
    paths = [
        Path(
            f'data/gold_labels/BBB_Martins/{active}/scaffold/'
            f'{split}_molecule_condition_labels.jsonl'
        ).resolve()
        for split in ('valid', 'test')
    ]
    expected = set()
    for path in paths:
        for line in path.read_text().splitlines():
            identity = normalize_molecule_identity(json.loads(line)['drug'])
            expected.add(identity.parent_inchi_key or identity.parent_smiles)
    assert parents == expected
    assert inputs == {str(path): sha256_file(path) for path in paths}

    v1 = set()
    for path in paths:
        old = Path(str(path).replace(f'/{active}/', '/v1/'))
        for line in old.read_text().splitlines():
            identity = normalize_molecule_identity(json.loads(line)['drug'])
            v1.add(identity.parent_inchi_key or identity.parent_smiles)
    assert parents == v1


def test_molecule_bucket_prompt_is_self_contained_and_l2_capable():
    assets = prompt_assets(MOLECULE_BUCKET_VERSION)
    manifest = prompt_asset_manifest(MOLECULE_BUCKET_VERSION)
    assert assets['settings']['max_level'] == 2
    assert assets['settings']['query_prior'] is True
    assert assets['settings']['tools'] is False
    assert not any(key.startswith('@') for key in manifest['files_sha256'])


def test_molecule_description_parquet_is_hash_and_schema_pinned(tmp_path):
    path = tmp_path / 'descriptions.parquet'
    table = pa.table({
        'canonical_smiles': pa.array(['C'], type=pa.string()),
        'description_raw': pa.array(['raw sentinel'], type=pa.string()),
        'description_motif': pa.array(['motif sentinel'], type=pa.string()),
        'description_coarse': pa.array(['coarse sentinel'], type=pa.string()),
        'error': pa.array([None], type=pa.string()),
    })
    pq.write_table(table, path)
    cache = runner._load_molecule_descriptions(
        'motif', path=path, expected_sha256=runner.sha256_file(path)
    )
    assert cache['column'] == 'description_motif'
    assert cache['entries']['C']['description'] == 'motif sentinel'
    with pytest.raises(ValueError, match='hash mismatch'):
        runner._load_molecule_descriptions('raw', path=path, expected_sha256='0' * 64)


def test_query_prior_overlay_adds_a_hash_pinned_stable_identity(tmp_path):
    task = 'bbb_martins'
    base_root = tmp_path / 'base'
    base_batch = base_root / task / f'{task}__none'
    base_run = base_batch / 'runs' / f'{task}__none_idx00000'
    base_run.mkdir(parents=True)
    base_input = tmp_path / 'v1.jsonl'
    base_input.write_text(json.dumps({
        'molecule_identity_key': 'OLD',
        'condition_group': 'old_condition',
    }) + '\n')
    base_manifest = base_batch / 'manifest.json'
    base_manifest.write_text(json.dumps({'n_items': 1}))

    override_run = tmp_path / 'new_run'
    override_run.mkdir()
    single_path = override_run / 'single_molecule_reasoning_output.json'
    final_path = override_run / 'final_reasoning_output.json'
    single_path.write_text(json.dumps({
        'status': 'ok',
        'llm': {
            'content': {'prior': 'new'},
            'tool_results': [{'status': 'ok'}],
            'structured_output_validation': {'valid': True},
        },
    }))
    final_path.write_text(json.dumps({'status': 'ok'}))

    overlay_root = tmp_path / 'overlay'
    overlay_batch = overlay_root / task / f'{task}__none'
    overlay_batch.mkdir(parents=True)
    overlay_manifest = overlay_batch / 'manifest.json'
    overlay_manifest.write_text(json.dumps({
        'schema_version': 'progressive_query_prior_overlay.v1',
        'base_root': str(base_root),
        'base_manifest_sha256': sha256_file(base_manifest),
        'base_input_jsonl': str(base_input),
        'base_input_sha256': sha256_file(base_input),
        'overrides': [{
            'molecule_identity_key': 'NEW',
            'condition_group': 'new_condition',
            'source_index': 7,
            'run_dir': str(override_run),
            'files_sha256': {
                single_path.name: sha256_file(single_path),
                final_path.name: sha256_file(final_path),
            },
        }],
    }))

    runner._single_source_index.cache_clear()
    sources = runner._single_source_index(task, str(overlay_root))
    assert set(sources) == {('OLD', 'old_condition'), ('NEW', 'new_condition')}
    prior, tool_summary, final, source_index = runner._load_reused_query_prior(
        task,
        {'molecule_identity_key': 'NEW', 'condition_group': 'new_condition'},
        overlay_root,
    )
    assert (prior, tool_summary, final, source_index) == (
        {'prior': 'new'}, {'status': 'ok'}, {'status': 'ok'}, 7,
    )

    final_path.write_text(json.dumps({'status': 'changed'}))
    runner._single_source_index.cache_clear()
    with pytest.raises(ValueError, match='override hash mismatch'):
        runner._single_source_index(task, str(overlay_root))


def test_molecule_description_preflight_reports_selected_failures(tmp_path):
    cache = {
        'mode': 'raw', 'cache_version': 'v2',
        'column': 'description_raw', 'path': '/fixed.parquet',
        'sha256': 'a' * 64, 'row_count': 2,
        'entries': {
            'QUERY': {'description': 'query description', 'error': None},
            'L1': {'description': None, 'error': 'generation failed'},
        },
    }
    records = {'bbb_martins': [{'benchmark_row_id': 'q1', 'drug': 'QUERY'}]}
    contexts = {'bbb_martins': {'q1': [{'canonical_smiles': 'L1'}]}}
    later = {'bbb_martins': {'q1': {'L2': {'records': [{
        'payload': {'canonical_smiles': 'MISSING'}
    }]}}}}
    receipt = tmp_path / 'preflight.json'
    with pytest.raises(ValueError, match='2 selected molecules'):
        runner._validate_molecule_description_coverage(
            cache, records_by_task=records, indices_by_task={'bbb_martins': [0]},
            contexts_by_task=contexts, later_by_task=later, receipt_path=receipt,
        )
    saved = json.loads(receipt.read_text())
    assert saved['status'] == 'failed'
    assert saved['cache_version'] == 'v2'
    assert {row['reason'] for row in saved['invalid']} == {'error', 'missing'}
    assert {ref['role'] for row in saved['invalid'] for ref in row['references']} == {
        'L1', 'L2'
    }
    values = runner._validate_molecule_description_coverage(
        cache, records_by_task=records, indices_by_task={'bbb_martins': [0]},
        contexts_by_task=contexts, later_by_task=later, receipt_path=receipt,
        missing_policy='omit',
    )
    assert values == {'QUERY': 'query description'}
    saved = json.loads(receipt.read_text())
    assert saved['status'] == 'ok_with_omissions'
    assert saved['missing_policy'] == 'omit'


def test_molecule_description_cli_requires_explicit_successor_prompt():
    base = [
        '--harness-version', 'reranked-progressive-l1-context-l2-v1',
        '--reranking', 'semantic-lap', '--molecule-description-mode', 'raw',
    ]
    with pytest.raises(SystemExit):
        runner.parse_args(base)
    args = runner.parse_args([
        *base, '--prompt-version', MOLECULE_METADATA_VERSION,
        '--skip-tool-prefetch',
    ])
    assert args.molecule_description_mode == 'raw'
    assert '_molecule_description_raw_' in args.output_root
    weighted = runner.parse_args([
        '--harness-version', 'reranked-progressive-l1-context-l2-weighted-v1',
        '--reranking', 'semantic-weighted', '--molecule-description-mode', 'coarse',
        '--prompt-version',
        'reranked_progressive_l1_context_l2_weighted_molecule_metadata_v1',
        '--skip-tool-prefetch',
    ])
    assert weighted.molecule_description_mode == 'coarse'
    l1 = runner.parse_args([
        '--harness-version', 'reranked-progressive-l1-context-v1',
        '--reranking', 'morgan', '--molecule-description-mode', 'motif',
        '--molecule-description-cache-version', 'v2',
        '--prompt-version', 'reranked_progressive_l1_context_v4',
        '--skip-tool-prefetch',
    ])
    assert l1.molecule_description_missing_policy == 'omit'
    assert l1.molecule_description_cache_version == 'v2'
    assert '_molecule_description_motif_cache_v2_' in l1.output_root
    no_prior = runner.parse_args([
        *base, '--prompt-version', f'{MOLECULE_METADATA_VERSION}_no_query_prior',
        '--query-prior', 'none', '--skip-tool-prefetch',
    ])
    assert no_prior.query_prior == 'none'
    assert runner.parse_args([]).molecule_description_mode == 'none'
    assert runner.parse_args([]).molecule_description_cache_version == 'v1'


def test_progressive_can_select_frozen_v1_gold_labels():
    args = runner.parse_args([
        '--tasks', 'bbb_martins', '--gold-label-version', 'v1',
    ])
    assert args.gold_label_version == 'v1'
    assert runner.PROGRESSIVE_TASKS['bbb_martins'].input_jsonl == Path(
        'data/gold_labels/BBB_Martins/v1/scaffold/valid.jsonl'
    )


def test_molecule_descriptions_render_for_query_l1_and_grouped_l2_parent():
    policy = {'L1': 'morgan', 'L2': 'semantic_lap'}
    molecules, later = selection({'L1': 'morgan', 'L2': 'morgan'})
    for row in later['L2']['records']:
        row['semantic_bucket_id'] = 'bucket'
        row['ranking_method'] = 'semantic_lap'
        row['payload']['canonical_smiles'] = 'CCF'
    descriptions = {
        'CCO': 'L1 DESCRIPTION SENTINEL',
        'CCF': 'L2 PARENT DESCRIPTION SENTINEL',
    }
    snapshots = profile.stage_ranked_snapshots(
        molecules, task='bbb_martins', records_by_level=later,
        prompt_version=MOLECULE_METADATA_VERSION,
        molecule_descriptions=descriptions,
    )
    contract = runner._task_contract('bbb_martins', MOLECULE_METADATA_VERSION)
    messages = profile.build_tianang_aligned_messages(
        contract=contract,
        levels=profile.tianang_aligned_levels(
            'bbb_martins', 2, prompt_version=MOLECULE_METADATA_VERSION
        ),
        current_level=2, query_smiles='N',
        query_molecule_description='QUERY DESCRIPTION SENTINEL',
        condition_sentence='', query_prior=None, query_tool_summary=None,
        active=snapshots[2],
        prior_state={contract.prediction_field: contract.positive_prediction, 'claims': []},
        prompt_version=MOLECULE_METADATA_VERSION,
        record_limit=10, l2_record_limit=10, indirect_record_limit=12,
        retrieval_policy=policy,
    )
    user = messages[1]['content']
    assert user.count('QUERY DESCRIPTION SENTINEL') == 1
    assert user.count('L1 DESCRIPTION SENTINEL') == 1
    assert user.count('L2 PARENT DESCRIPTION SENTINEL') == 1
    assert messages[0]['content'].count('not experimental measurements or evidence cards') == 1


def test_claim_provenance_is_derived_from_claims_only():
    content = {
        "supportive_card_ids": ["C99"],
        "contradictory_card_ids": ["C98"],
        "prediction_basis_card_ids": ["C97"],
        "claims": [
            {"claim": "a", "card_ids": ["C01"], "evidence_role": "supportive"},
            {"claim": "b", "card_ids": ["C02"], "evidence_role": "contradictory"},
            {"claim": "c", "card_ids": ["C03"], "evidence_role": "supportive"},
        ],
    }
    normalized, errors = _derive_claim_provenance(content)
    assert errors == []
    assert normalized["supportive_card_ids"] == ["C01", "C03"]
    assert normalized["contradictory_card_ids"] == ["C02"]
    assert normalized["prediction_basis_card_ids"] == ["C01", "C03", "C02"]

    malformed, errors = _derive_claim_provenance(
        {"claims": [{"claim": "a", "card_ids": ["C01", "C02"]}] * 3}
    )
    assert malformed["supportive_card_ids"] == []
    assert "every claim must cite exactly one card" in errors
    assert "every claim needs evidence_role supportive or contradictory" in errors

    wrong_count, errors = _derive_claim_provenance({"claims": [{}] * 9})
    assert errors == ["claims must contain exactly 3 items"]
    assert all(isinstance(wrong_count[field], list) for field in (
        "supportive_card_ids", "contradictory_card_ids", "prediction_basis_card_ids"
    ))

    variable, errors = _derive_claim_provenance(
        {
            "bbb_prediction": "pass",
            "confidence": "high",
            "revision_action": "initial",
            "claims": [
                {"claim": str(index), "card_ids": [f"C{index:02d}"], "evidence_role": "supportive"}
                for index in range(1, 10)
            ],
            "new_evidence_assessment": [],
            "evidence_gaps": [],
            "decision_summary": "nine supported claims",
        },
        required_claim_count=None,
    )
    assert errors == []
    assert len(variable["supportive_card_ids"]) == 9
    visible = {f"C{index:02d}" for index in range(1, 10)}
    assert progressive_state_errors(
        variable,
        contract=runner._task_contract('bbb_martins', 'reranked_progressive_l1_simple_v11'),
        visible_card_ids=visible,
        new_card_ids=visible,
        prior_state=None,
        max_claims=None,
    ) == []


def policy_file(tmp_path, stages):
    path = tmp_path/'policy.yaml'
    path.write_text(yaml.safe_dump(dict(version=1, stages=stages, score_caches={})))
    return path


def selection(policy, later_limit=2):
    by_level = {'L1': {'a': ['a1'], 'b': ['b1']},
                'L2': {'a': ['a2'], 'b': ['b2'], 'c': ['c2']},
                'L3': {'a': ['a3']}}
    by_level = {level: by_level[level] for level in policy}
    similarities = {'a': .9, 'b': .8, 'c': .7}
    scores = {'a1': .1, 'b1': .9, 'a2': .4, 'b2': .5, 'c2': .95, 'a3': .2}
    molecules, later = select_policy_records(by_level, {k: list(v) for k,v in by_level.items()},
        similarities, scores, policy, task='bbb_martins', query_id='q',
        molecule_limit=1, later_limit=later_limit)
    smiles = {'a': 'CCO', 'b': 'CCN', 'c': 'CCF'}
    for molecule in molecules:
        molecule['canonical_smiles'] = smiles[molecule['reference_molecule_id']]
    rows = [r for m in molecules for r in m['l1_records']]
    rows += [r for level in later.values() for r in level['records']]
    for row in rows:
        row['payload'] = dict(canonical_smiles=smiles[row['reference_molecule_id']],
            source_id='source', source_fields={'support_text': row['record_id'],
                                              'qualifying_conditions': 'mouse' if row['record_id'].endswith('2') else 'rat'})
    return molecules, later


def test_policy_yaml_and_cli(tmp_path):
    path = policy_file(tmp_path, {'L1':'joint', 'L2':'assay_transfer', 'L3':'morgan'})
    with pytest.raises(SystemExit):
        runner.parse_args(['--reranking','joint','--tasks','bbb_martins',
                           '--max-level','3','--query-prior','none','--prepare-only',
                           '--skip-tool-prefetch'])
    args = runner.parse_args(['--reranking','morgan','--tasks','bbb_martins',
                             '--max-level','3','--query-prior','none','--prepare-only', '--skip-tool-prefetch'])
    assert args.assay_transfer_prompt_version == 'reranked_progressive_v8'
    assert args.context_ranking == 'morgan'
    assert runner._record_level_names(args, 'bbb_martins') == ('L2','L3')
    assert list(runner._indirect_record_limits(args,'bbb_martins').values()) == [50,50]
    assert len(runner._run_levels(args,'bbb_martins')) == 3
    for stages in ({'L1':'morgan','L2':'joint'}, {'L1':'unknown'}, {'L2':'morgan'}):
        path = policy_file(tmp_path, stages)
        with pytest.raises(ValueError):
            load_retrieval_policy(path,'bbb_martins',2)
    with pytest.raises(SystemExit):
        runner.parse_args(['--assay-transfer-prompt-version',VERSION])


def test_independent_l2_rankings_and_shortfall():
    molecules, later = selection({'L1':'morgan','L2':'assay_transfer'})
    assert [m['reference_molecule_id'] for m in molecules] == ['a']
    assert [r['reference_molecule_id'] for r in later['L2']['records']] == ['c','b']
    assert all(not m['l2_records'] for m in molecules)
    _, later = selection({'L1':'assay_transfer','L2':'morgan'}, later_limit=50)
    assert [r['reference_molecule_id'] for r in later['L2']['records']] == ['a','b','c']
    assert later['L2']['shortfall'] == 47


def test_joint_union_deduplication_and_missing_scores():
    pool = {str(i):[f'r{i}'] for i in range(7)}
    kwargs = dict(by_level={'L1':pool},pools={'L1':list(pool)},
        similarities={str(i):1-i/10 for i in range(7)}, scores={f'r{i}':i/10 for i in range(7)},
        policy={'L1':'joint'}, task='bbb_martins',query_id='q')
    molecules,_ = select_policy_records(**kwargs)
    assert len(molecules) == 7  # Panels overlap in three molecules; no refill.
    assert sum(isinstance(m.get('morgan_top5_rank'),int) for m in molecules) == 5
    assert sum(isinstance(m.get('assay_transfer_top5_rank'),int) for m in molecules) == 5
    assert len({r['record_id'] for m in molecules for r in m['l1_records']}) == 7
    with pytest.raises(ValueError,match='Missing mapping-compatible score'):
        select_policy_records(**{**kwargs,'scores':{}})
    assert select_policy_records(**kwargs) == select_policy_records(**kwargs)


def test_scores_fail_closed_before_source_loading(tmp_path):
    policy = load_retrieval_policy(policy_file(tmp_path, {'L1': 'joint'}), 'bbb_martins', 1)
    with pytest.raises(ValueError, match='mapping-compatible score_caches'):
        load_mapped_candidates([], sampler='plain', retrieval_policy=policy,
                               v7_root=tmp_path/'nonexistent')
    for value in (float('nan'), float('inf'), -0.1, 1.1):
        with pytest.raises(ValueError, match='Invalid transfer score'):
            select_policy_records({'L1': {'a': ['r']}}, {'L1': ['a']}, {'a': .5},
                {'r': value}, {'L1': 'assay_transfer'}, task='bbb_martins', query_id='q')


@pytest.mark.parametrize('task', ['bbb_martins','bioavailability_ma'])
def test_new_records_inside_old_cards_and_policy_visibility(task):
    policy = {'L1':'morgan','L2':'morgan','L3':'morgan'}
    molecules, later = selection(policy)
    before = deepcopy((molecules,later))
    snapshots = profile.stage_ranked_snapshots(molecules,task=task,records_by_level=later,prompt_version=VERSION)
    assert (molecules,later) == before
    old_id = next(iter(snapshots[1]))
    old_record = next(iter(snapshots[1][old_id]['cards']))
    assert snapshots[3][old_id]['cards'][old_record] == snapshots[1][old_id]['cards'][old_record]
    assert snapshots[2][old_id]['first_seen_level'] == 1
    assert len(snapshots[2][old_id]['cards']) == 2
    assert len(snapshots[2]) == 2
    assert 'selected_condition' not in snapshots[2][old_id]
    assert {c['qualifying_conditions'] for c in snapshots[2][old_id]['cards'].values()} == {'rat','mouse'}
    contract = runner._task_contract(task,VERSION)
    messages = profile.build_tianang_aligned_messages(contract=contract,
        levels=profile.tianang_aligned_levels(task,3,prompt_version=VERSION),current_level=2,
        query_smiles='C',condition_sentence='',query_prior=None,query_tool_summary=None,
        active=snapshots[2],prior_state={contract.prediction_field:contract.positive_prediction,'claims':[]},
        prompt_version=VERSION,record_limit=10,l2_record_limit=10,indirect_record_limit=50,
        retrieval_policy=policy)
    validate_visible_messages(messages)
    payload = json.loads(messages[1]['content'])
    assert 'retrieval_by_level' not in payload['protocol']
    cards = [c for m in payload['active_evidence'] for c in m['evidence_cards']]
    assert set(payload['level_context']['new_card_ids']) == {c['card_id'] for c in cards if c['new_this_level']}
    assert sum(c['new_this_level'] for c in payload['active_evidence'][0]['evidence_cards']) == 1
    assert 'transfer_likelihood' not in snapshots[3][old_id]  # No inherited/fabricated stage score.
    with pytest.raises(ValueError,match='Duplicate physical record'):
        bad = deepcopy(later);bad['L2']['records'].append(molecules[0]['l1_records'][0])
        profile.stage_ranked_snapshots(molecules,task=task,records_by_level=bad,prompt_version=VERSION)


def test_preparation_and_resume_policy_identity(tmp_path):
    path = policy_file(tmp_path,{'L1':'morgan','L2':'morgan'})
    policy = load_retrieval_policy(path,'bbb_martins',2)
    molecules,later = selection(policy['stages'])
    kwargs = dict(task='bbb_martins',query_index=0,record={'drug':'C','benchmark_row_id':'q'},
        contexts=molecules,levels=profile.tianang_aligned_levels('bbb_martins',2,prompt_version=VERSION),
        output_root=tmp_path/'outputs',single_root=tmp_path,query_prior_mode='none',
        record_limit=10,l2_record_limit=10,indirect_record_limit=50,indirect_records=later,
        prompt_version=VERSION,prefetch_tools=False,retrieval_policy=policy)
    query = runner._prepare_context_record_query(**kwargs)
    l2 = json.loads((query.query_dir/'levels/level_2/prepared.json').read_text())
    assert l2['retrieval_policy'] == policy
    assert len(l2['new_card_ids']) == 2
    assert l2['selection_audit']['l2_record_limit_per_context'] is None
    previous = {'inputs': {'retrieval_policy':policy}}
    changed = deepcopy(previous);changed['inputs']['retrieval_policy']['stages']['L2']='assay_transfer'
    with pytest.raises(ValueError,match='changed inputs'):
        runner._merge_resume_manifest(previous,changed)


@pytest.mark.parametrize('task,last', [('bbb_martins',5), ('bioavailability_ma',6)])
@pytest.mark.parametrize('mode,l1', [('morgan','morgan'), ('assay-transfer','assay_transfer')])
def test_named_modes_select_and_render_the_correct_stages(task, last, mode, l1):
    version = VERSION
    args = runner.parse_args(['--tasks',task,'--reranking',mode])
    policy = args.retrieval_policies[task]['stages']
    assert args.assay_transfer_prompt_version == 'reranked_progressive_v8'
    assert list(policy) == [f'L{i}' for i in range(1,last+1)]
    assert policy['L1'] == l1
    assert policy['L5'] == 'morgan'
    assert {policy[k] for k in ('L2','L3','L4')} == {'morgan' if mode == 'morgan' else 'assay_transfer'}
    if last == 6:
        assert policy['L6'] == ('morgan' if mode == 'morgan' else 'assay_transfer')
    by_level = {k: {str(i):[f'{k}_{i}'] for i in range(12)} for k in policy}
    # L5 has no assay scores: every mode must select it successfully by Morgan.
    scores = {f'{k}_{i}':i/12 for k in policy if k != 'L5' for i in range(12)}
    molecules, later = select_policy_records(by_level, {k:list(v) for k,v in by_level.items()},
        {str(i):1-i/12 for i in range(12)}, scores, policy, task=task,query_id='q', later_limit=2)
    selected = {m['reference_molecule_id'] for m in molecules}
    assert selected == ({str(i) for i in range(10)} if mode == 'morgan' else
                        {str(i) for i in range(2,12)} if mode == 'assay-transfer' else
                        {str(i) for i in [0,1,2,3,4,7,8,9,10,11]})
    assert [r['record_id'] for r in later['L5']['records']] == ['L5_0','L5_1']
    for level in ('L2','L3','L4'):
        expected = [f'{level}_0',f'{level}_1'] if mode == 'morgan' else [f'{level}_11',f'{level}_10']
        assert [r['record_id'] for r in later[level]['records']] == expected
    # Render every stage of each mode, not just the CLI's parsed policy.
    for molecule in molecules:
        molecule['canonical_smiles'] = 'C'*(int(molecule['reference_molecule_id'])+2)
    rows = [r for m in molecules for r in m['l1_records']] + [r for b in later.values() for r in b['records']]
    for row in rows:
        row['payload'] = {'canonical_smiles':'C'*(int(row['reference_molecule_id'])+2), 'source_fields':{}}
    snapshots = profile.stage_ranked_snapshots(molecules,task=task,records_by_level=later,prompt_version=version)
    before = deepcopy(snapshots)
    contract = runner._task_contract(task,version)
    for level,active in snapshots.items():
        messages = profile.build_tianang_aligned_messages(contract=contract,
            levels=runner._run_levels(args,task), current_level=level,query_smiles='N',condition_sentence='',
            query_prior=None,query_tool_summary=None,active=active,
            prior_state=None if level == 1 else {contract.prediction_field:contract.positive_prediction,'claims':[]},
            prompt_version=version,record_limit=4,l2_record_limit=4,indirect_record_limit=2,retrieval_policy=policy)
        payload = json.loads(messages[1]['content'])
        assert not {'retrieval_by_level','prompt_mode','version'} & payload['protocol'].keys()
        plan = payload['level_context']['full_level_plan']
        source_levels = profile.tianang_aligned_levels(task, prompt_version=version)
        assert [row['level'] for row in plan] == list(range(1,last+1))
        assert [row['description'] for row in plan] == [row['description'] for row in source_levels]
        assert payload['level_context']['current_family'] == plan[level-1]
        assert payload['level_context']['current_level']==level
        assert set(payload['level_context']['new_card_ids'])=={
            c['card_id'] for m in payload['active_evidence'] for c in m['evidence_cards'] if c['new_this_level']}
        for molecule in payload['active_evidence']:
            cards = molecule['evidence_cards']
            morgan_levels = sorted({c['first_seen_level'] for c in cards
                                    if policy[f"L{c['first_seen_level']}"] in {'morgan','joint'}})
            assert ('morgan_similarity' in molecule) == bool(morgan_levels)
            if morgan_levels:
                assert molecule['morgan_score_levels'] == morgan_levels
            expected_transfer = molecule['first_seen_level'] == 1 and (
                mode == 'assay-transfer' or mode == 'joint' and isinstance(molecule.get('assay_transfer_top5_rank'), int))
            assert ('transfer_likelihood' in molecule) == expected_transfer
            for card in cards:
                assert 'evidence_family' not in card
                assert 'morgan_similarity' not in card
                assert ('transfer_likelihood' in card) == (card['first_seen_level'] > 1 and policy[f"L{card['first_seen_level']}"] == 'assay_transfer')
            assert 'analog_id' not in molecule
            assert all('retrieved_by' not in c for c in cards)
        assert snapshots == before


@pytest.mark.parametrize('task', ['bbb_martins','bioavailability_ma'])
def test_semantic_record_details_omit_nulls_metadata_and_duplicates(task):
    version=VERSION
    policy={'L1':'morgan'}
    molecules,later=selection(policy)
    row=molecules[0]['l1_records'][0]
    row['payload'].update(source_id='internal_source',measurement_kind='continuous',
        source_projection='library_source_contract.v2',source_fields={
            'assay_model':'rat assay', 'assay_system':'second meaningful system',
            'species':None, 'measurement_text':None, 'dose':0, 'perturbation':False,
            'formulation_or_solid_form':{'form':'solution','vehicle':None},
            'transporter_identifier':'ABCB1', 'metric_uncertainty':None, 'incubation_duration':'2 h',
            'extra_details':{'internal':'do not expose'},'source_record_id':'private-id',
            'source_id':'internal_source','measurement_kind':'continuous',
            'support_text':'A complete experimental observation.'})
    original=deepcopy((molecules,later))
    snapshots=profile.stage_ranked_snapshots(molecules,task=task,records_by_level=later,prompt_version=version)
    contract=runner._task_contract(task,version)
    def render(active):
        messages=profile.build_tianang_aligned_messages(contract=contract,
            levels=profile.tianang_aligned_levels(task,1,prompt_version=version),
            current_level=1,query_smiles='N',condition_sentence='',query_prior=None,
            query_tool_summary=None,active=active,prior_state=None,prompt_version=version,
            record_limit=10,l2_record_limit=10,indirect_record_limit=50,retrieval_policy=policy)
        return json.loads(messages[1]['content'])['active_evidence'][0]['evidence_cards']
    card=render(snapshots[1])[0]
    assert not {'source_id','measurement_kind','species','reported_value'} & card.keys()
    assert card['experimental_details']=={
        'assay_system':'second meaningful system','dose':0,'perturbation':False,
        'formulation_or_solid_form':{'form':'solution'},'transporter_identifier':'ABCB1',
        'incubation_duration':'2 h'}
    assert (molecules,later)==original
    second=deepcopy(row)
    second['record_id']='different_schema_same_level'
    second['payload']['source_fields']={'intestinal_site':'jejunum','condition_medium':None}
    molecules[0]['l1_records'].append(second)
    mixed=profile.stage_ranked_snapshots(molecules,task=task,records_by_level=later,prompt_version=version)
    rendered=render(mixed[1])
    assert len(rendered)==2
    assert rendered[0]['experimental_details']==card['experimental_details']
    assert rendered[1]['experimental_details']=={'intestinal_site':'jejunum'}
    molecules[0]['l1_records'].pop()
    row['payload']['source_fields']={'dose':None,'species':None,'assay_system':None,'unit_text':None}
    empty=profile.stage_ranked_snapshots(molecules,task=task,records_by_level=later,prompt_version=version)
    assert 'experimental_details' not in render(empty[1])[0]


def test_stage_score_view_filters_receipts_without_mutation():
    policy = {'L1':'assay_transfer','L2':'assay_transfer','L5':'morgan'}
    active = {'a': dict(first_seen_level=1, morgan_similarity=.6, transfer_likelihood=.8,
        transfer_score_level=1, morgan_top5_rank=1, query_analog_tool_summaries=[
            {'content':'Morgan fingerprint Tanimoto similarity: 0.60\nMolecular weight difference: 2.00'}],
        cards={'r1':dict(first_seen_level=1,retrieved_by='assay_transfer',transfer_likelihood=.9),
               'r2':dict(first_seen_level=2,retrieved_by='assay_transfer',transfer_likelihood=.7)})}
    original=deepcopy(active)
    view=profile.stage_score_view(active,policy)['a']
    assert active==original
    assert 'morgan_similarity' not in view and 'morgan_top5_rank' not in view
    assert view['transfer_likelihood']==.8
    assert 'transfer_likelihood' not in view['cards']['r1']
    assert view['cards']['r2']['transfer_likelihood']==.7
    assert view['query_analog_tool_summaries'][0]['content']=='Molecular weight difference: 2.00'
    active['a']['cards']['r5']=dict(first_seen_level=5,retrieved_by='morgan',transfer_likelihood=.2)
    view=profile.stage_score_view(active,policy)['a']
    assert view['morgan_similarity']==.6 and view['morgan_score_levels']==[5]
    assert 'transfer_likelihood' not in view['cards']['r5']
    active['a'].pop('transfer_likelihood')
    assert 'transfer_likelihood' not in profile.stage_score_view(active,policy)['a']
    active['a']['cards']['r5']['retrieved_by']='assay_transfer'
    with pytest.raises(ValueError,match='disagrees'):
        profile.stage_score_view(active,policy)


def test_mode_prompt_requires_matching_policy_and_hashes_mode_assets():
    from predict.harnesses.progressive.prompt import prompt_asset_manifest, render_progressive_messages
    version=VERSION
    manifest=prompt_asset_manifest(version)
    for mode in ('morgan','assay-transfer','joint'):
        assert f'modes/{mode}/system.jinja' in manifest['files_sha256']
        assert f'modes/{mode}/user.yaml' in manifest['files_sha256']
    with pytest.raises(ValueError,match='prompt_mode'):
        render_progressive_messages(system_role='Test',payload={'task_definition':{'task':'bbb_martins'}},prompt_version=version)
    with pytest.raises(ValueError,match='disagrees'):
        profile.build_tianang_aligned_messages(contract=runner._task_contract('bbb_martins',version),
            levels=profile.tianang_aligned_levels('bbb_martins',2,prompt_version=version),
            current_level=1,query_smiles='N',condition_sentence='',query_prior=None,
            query_tool_summary=None,active={},prior_state=None,prompt_version=version,
            record_limit=10,l2_record_limit=10,indirect_record_limit=50,
            retrieval_policy={'L1':'morgan','L2':'assay_transfer'})


def test_simple_l1_prompt_is_readable_auditable_and_tool_free():
    policy = {'L1': 'assay_transfer'}
    molecules, later = selection(policy)
    snapshots = profile.stage_ranked_snapshots(
        molecules,
        task='bioavailability_ma',
        records_by_level=later,
        prompt_version=SIMPLE_L1_VERSION,
    )
    messages = profile.build_tianang_aligned_messages(
        contract=runner._task_contract('bioavailability_ma', SIMPLE_L1_VERSION),
        levels=profile.tianang_aligned_levels(
            'bioavailability_ma', 1, prompt_version=SIMPLE_L1_VERSION
        ),
        current_level=1,
        query_smiles='N',
        condition_sentence='',
        query_prior={'hidden': True},
        query_tool_summary={'content': 'hidden'},
        active=snapshots[1],
        prior_state=None,
        prompt_version=SIMPLE_L1_VERSION,
        record_limit=10,
        l2_record_limit=10,
        indirect_record_limit=50,
        retrieval_policy=policy,
    )
    visible = '\n'.join(message['content'] for message in messages)
    assert [message['role'] for message in messages] == ['system', 'user']
    assert not messages[1]['content'].lstrip().startswith('{')
    assert 'C01' in visible
    for hidden in ('query_prior', 'query_analog_tool_summaries', 'analog_id', 'record_'):
        assert hidden not in visible

    manifest = profile.prompt_assets(SIMPLE_L1_VERSION)
    assert manifest['settings'] == {
        'max_level': 1,
        'query_prior': False,
        'tools': False,
        'user_template': 'user.jinja',
    }
    args = runner.parse_args([
        '--tasks', 'bioavailability_ma',
        '--reranking', 'assay-transfer',
        '--max-level', '1',
        '--query-prior', 'none',
        '--skip-tool-prefetch',
        '--prompt-version', SIMPLE_L1_VERSION,
    ])
    assert args.output_root.endswith('reranked_progressive_l1_simple_v1_assay_transfer')
    for incomplete in (
        ['--max-level', '1', '--query-prior', 'none'],
        ['--max-level', '1', '--skip-tool-prefetch'],
        ['--query-prior', 'none', '--skip-tool-prefetch'],
    ):
        with pytest.raises(SystemExit):
            runner.parse_args(['--prompt-version', SIMPLE_L1_VERSION, *incomplete])


def test_active_default_and_retired_cli_boundary(tmp_path):
    from predict.harnesses.progressive.prompt import prompt_asset_manifest, prompt_directory
    args = runner.parse_args([])
    assert (args.harness_version,args.reranking) == ('reranked-progressive-v2','assay-transfer')
    assert args.retrieval_policies['bioavailability_ma']['stages']['L1'] == 'assay_transfer'
    assert prompt_directory(VERSION).parent.name == 'prompts'
    with pytest.raises(ValueError, match='inactive prompt version'):
        prompt_asset_manifest('reranked_progressive_v7')
    for options in (['--legacy'], ['--profile','standard'], ['--context-ranking','morgan'],
                    ['--legacy','--reranking','morgan'], ['--reranking','joint','--context-limit','9'],
                    ['--tasks','skin_reaction'], ['--evaluation-subset','test'],
                    ['--tasks','bbb_martins','--max-level','6']):
        with pytest.raises(SystemExit):
            runner.parse_args(options)
    prepared_test = runner.parse_args(['--evaluation-subset','test','--prepare-only'])
    assert prepared_test.evaluation_subset == 'test'
    approved_test = runner.parse_args([
        '--evaluation-subset', 'test', '--allow-test-inference', '--parallelism', '1',
    ])
    assert approved_test.evaluation_subset == 'test'
    custom = policy_file(tmp_path, {'L1':'morgan'})
    with pytest.raises(SystemExit):
        runner.parse_args(['--score-cache-config',str(custom)])
    with pytest.raises(ValueError,match='changed inputs'):
        runner._merge_resume_manifest({'inputs':{'policy':args.retrieval_policies}},
            {'inputs':{'policy':runner.parse_args(['--reranking','morgan']).retrieval_policies}})


def test_query_prior_prompt_requires_cached_prior():
    version = 'reranked_progressive_l1_simple_v9'
    args = runner.parse_args([
        '--tasks', 'bbb_martins',
        '--max-level', '1',
        '--query-prior', 'cached',
        '--skip-tool-prefetch',
        '--prompt-version', version,
    ])
    assert args.query_prior == 'cached'
    with pytest.raises(SystemExit):
        runner.parse_args([
            '--tasks', 'bbb_martins',
            '--max-level', '1',
            '--query-prior', 'none',
            '--skip-tool-prefetch',
            '--prompt-version', version,
        ])


@pytest.mark.parametrize(
    'version,shows_query_smiles,shows_evidence_smiles',
    [
        ('reranked_progressive_l1_simple_v10', True, True),
        ('reranked_progressive_l1_simple_v10_no_smiles', True, False),
        ('reranked_progressive_l1_simple_v10_no_query_smiles', False, True),
        ('reranked_progressive_l1_simple_v10_no_smiles_no_query_smiles', False, False),
    ],
)
def test_simple_prompt_smiles_modifiers_compose_without_variant_bundles(
    version, shows_query_smiles, shows_evidence_smiles
):
    policy = {'L1': 'assay_transfer'}
    molecules, later = selection(policy)
    snapshots = profile.stage_ranked_snapshots(
        molecules, task='bbb_martins', records_by_level=later, prompt_version=version
    )
    prior = {
        'passive_bbb_plausibility': 'moderate',
        'efflux_or_transporter_prior': 'uncertain',
        'confidence': 'low',
        'reasoning_summary': 'Compact prior.',
        'property_drivers': ['driver'],
        'caveats': ['caveat'],
    }
    messages = profile.build_tianang_aligned_messages(
        contract=runner._task_contract('bbb_martins', version),
        levels=profile.tianang_aligned_levels('bbb_martins', 1, prompt_version=version),
        current_level=1,
        query_smiles='N',
        condition_sentence='',
        query_prior=prior,
        query_tool_summary=None,
        active=snapshots[1],
        prior_state=None,
        prompt_version=version,
        record_limit=10,
        l2_record_limit=10,
        indirect_record_limit=50,
        retrieval_policy=policy,
    )
    user = messages[1]['content']
    query_smiles = 'N'
    evidence_smiles = molecules[0]['canonical_smiles']
    assert (f'SMILES: {query_smiles}' in user) is shows_query_smiles
    assert (f'SMILES: {evidence_smiles}' in user) is shows_evidence_smiles
    assert 'Query-property prior' in user
    assert profile.prompt_assets(version)['settings']['query_prior'] is True
    args = runner.parse_args([
        '--tasks', 'bbb_martins',
        '--max-level', '1',
        '--query-prior', 'cached',
        '--skip-tool-prefetch',
        '--prompt-version', version,
    ])
    assert args.prompt_version == version


def test_simple_prompt_modifier_manifests_pin_base_and_applied_rules():
    from predict.harnesses.progressive.prompt import prompt_asset_manifest

    versions = [
        'reranked_progressive_l1_simple_v10',
        'reranked_progressive_l1_simple_v10_no_smiles',
        'reranked_progressive_l1_simple_v10_no_query_smiles',
        'reranked_progressive_l1_simple_v10_no_smiles_no_query_smiles',
    ]
    manifests = [prompt_asset_manifest(version) for version in versions]
    assert len({manifest['directory'] for manifest in manifests}) == 1
    assert len({manifest['sha256'] for manifest in manifests}) == len(versions)
    assert '@applied_modifiers' not in manifests[0]['files_sha256']
    assert all('@applied_modifiers' in manifest['files_sha256'] for manifest in manifests[1:])


@pytest.mark.parametrize('prior_mode', ['cached', 'none'])
def test_v13_records_only_modifier_hides_retrieval_metadata_but_keeps_records(
    prior_mode,
):
    base = 'reranked_progressive_l1_simple_v13_no_scores_no_smiles_no_query_smiles'
    version = base if prior_mode == 'cached' else f'{base}_no_query_prior'
    policy = {'L1': 'assay_transfer'}
    molecules, later = selection(policy)
    snapshots = profile.stage_ranked_snapshots(
        molecules, task='bbb_martins', records_by_level=later, prompt_version=version
    )
    prior = {
        'passive_bbb_plausibility': 'moderate',
        'efflux_or_transporter_prior': 'uncertain',
        'confidence': 'low',
        'reasoning_summary': 'Compact prior.',
        'property_drivers': ['driver'],
        'caveats': ['caveat'],
    } if prior_mode == 'cached' else None
    messages = profile.build_tianang_aligned_messages(
        contract=runner._task_contract('bbb_martins', version),
        levels=profile.tianang_aligned_levels('bbb_martins', 1, prompt_version=version),
        current_level=1,
        query_smiles='N',
        condition_sentence='',
        query_prior=prior,
        query_tool_summary=None,
        active=snapshots[1],
        prior_state=None,
        prompt_version=version,
        record_limit=10,
        l2_record_limit=10,
        indirect_record_limit=50,
        retrieval_policy=policy,
    )
    user = messages[1]['content']
    assert 'SMILES:' not in user
    assert 'Transfer likelihood:' not in user
    assert 'Morgan similarity:' not in user
    assert 'panel rank:' not in user
    assert 'Record C01' in user
    assert 'Reported evidence:' in user
    assert ('Query-property prior' in user) is (prior_mode == 'cached')
    assert profile.prompt_assets(version)['settings']['retrieval_scores'] is False


def test_v4_fixes_assay_three_plus_morgan_seven_on_v3_cache():
    args = runner.parse_args([
        '--harness-version', 'reranked-progressive-v4',
        '--tasks', 'bbb_martins',
        '--reranking', 'joint',
        '--max-level', '1',
        '--query-prior', 'cached',
        '--skip-tool-prefetch',
        '--prompt-version', 'reranked_progressive_l1_simple_v13',
        '--assay-transfer-cache',
        'predict/retrieval/assay_reranking/cache_matched_retrieval_v3.yaml',
    ])
    assert args.joint_panel_sizes == (3, 7)
    assert args.retrieval_policies['bbb_martins']['selection_contract'] == (
        'cache_matched_retrieval.v3'
    )
    with pytest.raises(SystemExit):
        runner.parse_args([
            '--harness-version', 'reranked-progressive-v4',
            '--reranking', 'morgan',
        ])


def test_simple_prompt_no_query_prior_keeps_smiles_and_changes_runtime_requirement():
    version = 'reranked_progressive_l1_simple_v11_no_query_prior'
    assets = profile.prompt_assets(version)
    assert assets['settings']['query_prior'] is False
    args = runner.parse_args([
        '--tasks', 'bbb_martins',
        '--max-level', '1',
        '--query-prior', 'none',
        '--skip-tool-prefetch',
        '--prompt-version', version,
    ])
    assert args.prompt_version == version


def test_simple_prompt_all_smiles_hidden_composes_with_no_query_prior():
    version = (
        'reranked_progressive_l1_simple_v12_no_smiles_no_query_smiles_no_query_prior'
    )
    policy = {'L1': 'assay_transfer'}
    molecules, later = selection(policy)
    snapshots = profile.stage_ranked_snapshots(
        molecules, task='bbb_martins', records_by_level=later, prompt_version=version
    )
    messages = profile.build_tianang_aligned_messages(
        contract=runner._task_contract('bbb_martins', version),
        levels=profile.tianang_aligned_levels('bbb_martins', 1, prompt_version=version),
        current_level=1,
        query_smiles='N',
        condition_sentence='',
        query_prior=None,
        query_tool_summary=None,
        active=snapshots[1],
        prior_state=None,
        prompt_version=version,
        record_limit=10,
        l2_record_limit=10,
        indirect_record_limit=50,
        retrieval_policy=policy,
    )
    user = messages[1]['content']
    assert 'SMILES:' not in user
    assert 'Query-property prior' not in user
    assert profile.prompt_assets(version)['settings']['query_prior'] is False
    assert profile.prompt_assets(version)['applied_modifiers'] == (
        'no_smiles', 'no_query_smiles', 'no_query_prior'
    )
    args = runner.parse_args([
        '--tasks', 'bbb_martins',
        '--max-level', '1',
        '--query-prior', 'none',
        '--skip-tool-prefetch',
        '--prompt-version', version,
    ])
    assert args.prompt_version == version


@pytest.mark.parametrize("parent,successor", [
    (
        "reranked_progressive_l1_simple_v13_no_scores_no_smiles_no_query_smiles_no_query_prior",
        "reranked_progressive_l1_simple_v13_references_v1_no_scores_no_smiles_no_query_smiles_no_query_prior",
    ),
    (
        "reranked_progressive_l1_simple_v14_no_query_prior",
        "reranked_progressive_l1_simple_v14_references_v1_no_query_prior",
    ),
])
def test_reference_successors_preserve_model_visible_prompt(parent, successor):
    policy = {"L1": "assay_transfer"}
    molecules, later = selection(policy)
    snapshots = profile.stage_ranked_snapshots(
        molecules, task="bbb_martins", records_by_level=later, prompt_version=parent
    )

    def messages(version):
        return profile.build_tianang_aligned_messages(
            contract=runner._task_contract("bbb_martins", version),
            levels=profile.tianang_aligned_levels(
                "bbb_martins", 1, prompt_version=version
            ),
            current_level=1,
            query_smiles="N",
            condition_sentence="",
            query_prior=None,
            query_tool_summary=None,
            active=snapshots[1],
            prior_state=None,
            prompt_version=version,
            record_limit=10,
            l2_record_limit=10,
            indirect_record_limit=50,
            retrieval_policy=policy,
        )

    assert messages(successor) == messages(parent)


def test_front_shelf_reference_successors_pin_observational_contract():
    versions = {
        'reranked_progressive_l1_simple_v12_references_v1': 'l1',
        'reranked_progressive_l1_simple_v13_references_v1': 'l1',
        'reranked_progressive_l1_simple_v14_references_v1': 'l1',
        'reranked_progressive_l1_context_order_only_v1_references_v1': 'l1',
        'reranked_progressive_l1_context_v2_references_v1': 'l1',
        'reranked_progressive_l1_context_l2_semantic_v1_references_v1': 'semantic_l2',
        'reranked_progressive_l1_context_l2_weighted_v1_references_v1': 'weighted_l2',
    }
    for version, layout in versions.items():
        contract = profile.prompt_assets(version)['settings'][
            'reasoning_reference_contract'
        ]
        assert contract == {
            'schema_version': 'progressive_reasoning_references.v1',
            'layout': layout,
            'required_mentions': False,
            'matching': 'case_insensitive_exact_visible_id',
        }
        manifest = prompt_asset_manifest(version)
        assert 'provenance.json' in manifest['files_sha256']
        assert 'references.py' in manifest['assembly_files_sha256']
    assert profile.prompt_assets(
        'reranked_progressive_l1_simple_v14_references_v1'
    )['settings'].get('reasoning_transport') is None
