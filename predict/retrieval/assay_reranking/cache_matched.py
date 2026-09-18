"""Consume frozen candidate assignments for cache-matched evidence selection.

Direct-gold caches select training parents at L1; verified BBB context membership supplies
their cards and retires empty contexts before parent selection. Transfer-score
reuse requires unchanged rendered model inputs after redistribution unless the
caller explicitly accepts frozen pre-redistribution vote-percentage scores.
Later cache assignments define the shared Morgan/transfer pool, except full-library
Morgan L5. This module validates lineage and selects records; harnesses own their
presentation, tools, inference, and checkpoints. No builder runs.
"""
from collections import defaultdict
import csv
from contextlib import ExitStack
import hashlib
import json
import math
import re
from pathlib import Path
import sqlite3

import yaml

from predict.utils.json import sha256_file

TASKS = {"bbb_martins": ("BBB_Martins", 5), "bioavailability_ma": ("Bioavailability_Ma", 6)}
DEFAULT_CACHE_BUNDLE = Path(__file__).with_name('ranked_level_retrieval_v3.yaml')
SEMANTIC_BUCKET_CACHE_BUNDLE = Path(__file__).with_name('semantic_bucket_reranking_v1.yaml')
L1_CONTEXT_CACHE_BUNDLE = Path(__file__).with_name('l1_context_morgan25_v1.yaml')
CONTEXT_L2_CACHE_BUNDLE = Path(__file__).with_name('l1_context_semantic_l2_v1.yaml')
CONTEXT_L2_WEIGHTED_CACHE_BUNDLE = Path(__file__).with_name(
    'l1_context_semantic_weighted_l2_v1.yaml'
)
CONTEXT_L2_MORGAN_SEMANTIC_CACHE_BUNDLE = Path(__file__).with_name(
    'l1_context_morgan_semantic_l2_v1.yaml'
)
CONTEXT_L2_MORGAN_SEMANTIC_V2_CACHE_BUNDLE = Path(__file__).with_name(
    'l1_context_morgan_semantic_l2_v2.yaml'
)
CONTEXT_L2_MORGAN_SEMANTIC_V3_CACHE_BUNDLE = Path(__file__).with_name(
    'l1_context_morgan_semantic_l2_v3.yaml'
)
CONTEXT_L2_MORGAN_SEMANTIC_V4_CACHE_BUNDLE = Path(__file__).with_name(
    'l1_context_morgan_semantic_l2_v4.yaml'
)
INDIRECT_MORGAN_SEMANTIC_CACHE_BUNDLE = Path(__file__).with_name(
    'indirect_morgan_semantic_l2_l4_v1.yaml'
)
INDIRECT_MORGAN_SEMANTIC_V2_CACHE_BUNDLE = Path(__file__).with_name(
    'indirect_morgan_semantic_l2_l4_v2.yaml'
)
INDIRECT_MORGAN_SEMANTIC_V3_CACHE_BUNDLE = Path(__file__).with_name(
    'indirect_morgan_semantic_l2_l4_v3.yaml'
)
DEFAULT_GOLD_CONTEXT_MAPPING = Path(__file__).resolve().parents[3] / 'outputs/paper/assay_transfer_harness/joseph/bbb_remaining49_rebuild_20260908/mapping_manifest.json'


def split_path(task, subset):
    """Load the legacy benchmark resolver only when legacy replay needs it."""
    from data.processing.gold_labels.conditioned_benchmark import split_path as resolve

    return resolve(task, subset)


def verify_cached_level_records(cached_path, current_path, membership, levels):
    """Permit whole-library drift only when every cached-level row is identical."""
    import pyarrow.parquet as pq

    uids = membership.loc[membership.level.isin([int(k[1:]) for k in levels]), 'source_row_uid'].tolist()
    tables = [pq.read_table(path, filters=[('source_row_uid', 'in', uids)]).sort_by('canonical_record_id')
              for path in (cached_path, current_path)]
    if not tables[0].equals(tables[1]):
        raise ValueError('Cached-level library records changed; rebuild the candidate cache')
    return dict(levels=sorted(levels), records=tables[0].num_rows, all_fields_identical=True)


def load_cache_policy(path, task, subset, reranking, max_level=0):
    path = Path(path).resolve()
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict) or set(document) != {'version', 'caches'}:
        raise ValueError('Cache bundle requires version and caches only')
    if task not in TASKS or reranking not in {
        'morgan', 'morgan-contrastive', 'joint', 'assay-transfer',
        'assay-transfer-within-morgan', 'assay-transfer-contrastive',
        'semantic-lap', 'semantic-weighted',
        'morgan-parent-control', 'morgan-parent-semantic',
        'morgan-parent-llm-semantic'
    }:
        raise ValueError('Unsupported task or reranking mode')
    if ((reranking in {'morgan-contrastive', 'assay-transfer-within-morgan'}
         and document['version'] not in {6, 9})
            or (reranking == 'assay-transfer-contrastive'
                and document['version'] not in {6, 9, 19})
            or (document['version'] in {6, 9} and reranking not in {
                'morgan', 'morgan-contrastive', 'assay-transfer',
                'assay-transfer-within-morgan',
                'assay-transfer-contrastive'
            })):
        raise ValueError('L1 context cache and contrastive reranking must be selected together')
    last = max_level or TASKS[task][1]
    if not 1 <= last <= TASKS[task][1]:
        raise ValueError(f'{task} ends at L{TASKS[task][1]}')
    stages = {f'L{i}': 'morgan' if reranking == 'morgan' or i == 5 else 'assay_transfer'
              for i in range(1, last + 1)}
    if reranking == 'assay-transfer-contrastive':
        stages['L1'] = 'assay_transfer_contrastive'
    if reranking == 'morgan-contrastive':
        stages['L1'] = 'morgan_contrastive'
    if reranking == 'assay-transfer-within-morgan':
        stages['L1'] = 'assay_transfer_within_morgan'
    if reranking == 'joint':
        stages['L1'] = 'joint'
    if document['version'] in {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
        if document['version'] in {18, 19}:
            if reranking == 'joint':
                raise ValueError('ranked level retrieval does not support joint mode')
            index_path = (path.parent / document['caches'][task]).resolve()
            index = json.loads(index_path.read_text())
            expected = ('ranked_level_task_release_index.v2' if document['version'] == 18
                        else 'ranked_uid_task_release_index.v1')
            if (index.get('schema_version') != expected
                    or index.get('status') != 'complete'
                    or index.get('task_id') != task
                    or index.get('pool') != 'all'):
                raise ValueError(f'{task}: incomplete independent level release index')
            split = index['splits'][subset]
            cache_manifests = {}
            for level, method in stages.items():
                try:
                    entry = split['levels'][level]
                    relative = (entry[method] if document['version'] == 18 else entry)['manifest']
                except KeyError as exc:
                    raise ValueError(f'{task}/{subset}: missing {level}/{method} cache') from exc
                cache_manifests[level] = str((index_path.parent / relative).resolve())
            return dict(
                selection_contract=(
                    'ranked_level_retrieval.v2' if document['version'] == 18
                    else 'ranked_uid_retrieval.v1'
                ), reranking=reranking,
                stages=stages, cache_index=str(index_path), cache_manifests=cache_manifests,
                inputs={str(path): sha256_file(path), str(index_path): sha256_file(index_path)},
            )
        manifest = document['caches'][task][subset]
        if not isinstance(manifest, str) or not manifest.strip():
            raise ValueError(f'{task}/{subset}: missing cache manifest')
        contract = ({
            5: 'semantic_bucket_reranking.v1',
            6: 'l1_context_retrieval.v1',
            7: 'l1_context_semantic_l2.v1',
            8: 'l1_context_semantic_weighted_l2.v1',
            9: 'l1_context_retrieval.v2',
            10: 'l1_context_morgan_semantic_l2.v1',
            11: 'l1_context_morgan_semantic_l2.v2',
            12: 'l1_context_morgan_semantic_l2.v3',
            13: 'l1_context_morgan_semantic_l2.v4',
            14: 'indirect_morgan_semantic_l2_l4.v1',
            15: 'indirect_morgan_semantic_l2_l4.v2',
            16: 'indirect_morgan_semantic_l2_l4.v3',
            17: 'ranked_evidence_retrieval.v1',
            18: 'ranked_level_retrieval.v2',
            19: 'ranked_uid_retrieval.v1',
        }.get(document['version']) or f"cache_matched_retrieval.v{document['version'] - 1}")
        if document['version'] == 7:
            if reranking not in {'morgan', 'semantic-lap'} or last != 2:
                raise ValueError('Context-L2 cache requires Morgan or semantic-lap through L2')
            stages = {'L1': 'morgan', 'L2': reranking.replace('-', '_')}
        if document['version'] == 8:
            if reranking not in {'morgan', 'semantic-weighted'} or last != 2:
                raise ValueError('Weighted context-L2 cache requires Morgan or semantic-weighted through L2')
            stages = {'L1': 'morgan', 'L2': reranking.replace('-', '_')}
        if document['version'] in {10, 11, 12, 13}:
            if reranking not in {'morgan-parent-control', 'morgan-parent-semantic'} or last != 2:
                raise ValueError('Molecule-first context-L2 requires a matched parent mode through L2')
            stages = {'L1': 'morgan', 'L2': reranking.replace('-', '_')}
        if document['version'] in {14, 15, 16}:
            allowed = {'morgan-parent-control', 'morgan-parent-semantic'}
            if document['version'] == 16:
                allowed.add('morgan-parent-llm-semantic')
            if reranking not in allowed or last not in {2, 3, 4}:
                raise ValueError('Indirect-only cache requires one Morgan-parent arm at L2, L3, or L4')
            stages = {f'L{last}': reranking.replace('-', '_')}
        return dict(selection_contract=contract, reranking=reranking,
                    stages=stages, cache_manifest=str((path.parent / manifest).resolve()),
                    inputs={str(path): sha256_file(path)})
    if document['version'] != 2:
        raise ValueError(
            'Cache bundle version must be 2 (legacy), 3 (v2), 4 (v3), '
            '5 (semantic), 6 (L1 context v1), 7 (context L2), '
            '8 (weighted context L2), 9 (L1 context v2), or '
            '10 (molecule-first context L2), or 11 (global-cap molecule-first L2)'
        )
    paths = document['caches'][task][subset]
    if set(paths) - {f'L{i}' for i in range(1, TASKS[task][1]+1) if i != 5}:
        raise ValueError('Cache bundle cannot override L5 or add unsupported levels')
    required = set(stages) - {'L5'}
    if required - paths.keys() or any(not isinstance(paths[k], str) or not paths[k].strip() for k in required):
        raise ValueError(f'{task}/{subset}: missing cache manifests for {sorted(required)}')
    return dict(selection_contract='cache_matched_retrieval.v1', reranking=reranking, stages=stages,
                cache_manifests={k: str((path.parent / paths[k]).resolve()) for k in stages if k != 'L5'},
                inputs={str(path): sha256_file(path)})


def _load_candidates_v1(queries, *, task, subset, library, mapper, policy,
                        molecule_limit=10, l1_limit=10, later_limit=50, tie_seed=0,
                        gold_context_mapping=None, allow_frozen_l1_vote_scores=False,
                        cache_pool='tool-accepted', _all_l1_candidates=False,
                        _accept_pinned_legacy_builder=False,
                        _parent_scoped_l5_similarity=False,
                        _filter_l1_record_overlap=False,
                        _skip_redundant_quick_check=False):
    """Return L1 molecule selections, later record selections, and an input/pool audit."""
    import pandas as pd
    import pyarrow.parquet as pq
    from rdkit import DataStructs

    from predict.retrieval.policies import (
        decide_candidate,
        normalize_molecule_identity,
        seeded_rank_tie_key,
        standardize_smiles_and_fp,
    )
    from predict.utils.json import read_jsonl

    if (not queries or min(l1_limit, later_limit) < 1
            or (not _all_l1_candidates and molecule_limit < 1)):
        raise ValueError('Queries and positive shape limits are required')
    if policy['stages']['L1'] == 'joint' and molecule_limit != 10:
        raise ValueError('Joint requires ten slots, five per ranker')
    library, mapper = Path(library).resolve(), Path(mapper).resolve()
    source = library / '03_pair_buckets/records.parquet'
    source_manifest = source.with_name('manifest.json')
    source_contract = library / '02_canonicalized/source_contract.json'
    inputs = dict(policy['inputs'])

    def check(path, expected=None):
        path = Path(path).resolve()
        actual = inputs.get(str(path))
        if actual is None:
            actual = sha256_file(path)
            inputs[str(path)] = actual
        if expected is not None and actual != expected:
            raise ValueError(f'Input hash mismatch: {path}')
        return actual

    stage = json.loads(source_manifest.read_text())
    if stage.get('task_id') != task:
        raise ValueError('Evidence library task mismatch')
    check(source, stage['outputs']['records.parquet'])
    check(source_manifest)
    check(source_contract)
    check(mapper)
    mapping_manifest = json.loads(mapper.read_text())
    item = mapping_manifest['tasks'][task]
    mapping_path = mapper.parent / item['path']
    check(mapping_path, item['sha256'])
    membership = pd.read_parquet(mapping_path, columns=['source_row_uid', 'level', 'family_key'])
    if membership.source_row_uid.isna().any() or membership.source_row_uid.duplicated().any():
        raise ValueError('Duplicate or missing mapping UIDs')
    if membership.level.isna().any() or membership.family_key.isna().any():
        raise ValueError('Mapper levels and families must be complete')
    inventory = pd.read_parquet(source, columns=['canonical_record_id', 'source_row_uid', 'canonical_smiles'])
    if inventory.canonical_record_id.duplicated().any() or inventory.source_row_uid.isna().any():
        raise ValueError('Duplicate records or missing source UIDs in library')
    mapped = inventory.merge(membership, on='source_row_uid', how='left', validate='many_to_one')
    unmapped = mapped.loc[mapped.level.isna(), 'canonical_record_id'].tolist()
    mapped = mapped.loc[mapped.level.notna()].copy()
    mapped['level'] = mapped.level.map(lambda value: f'L{int(value)}')
    record_info = mapped.set_index('canonical_record_id').to_dict('index')
    query_path = split_path(task, subset).with_name(f'{subset}_molecule_condition_labels.jsonl')
    train_path = split_path(task, 'train').with_name('train_molecule_condition_labels.jsonl')
    for compact, detailed in [(split_path(task, subset), query_path), (split_path(task, 'train'), train_path)]:
        check(compact)
        compact_rows, detailed_rows = read_jsonl(compact), read_jsonl(detailed)
        keys = ('benchmark_row_id', 'drug', 'Y', 'condition_group')
        if [tuple(row.get(k) for k in keys) for row in compact_rows] != [tuple(row.get(k) for k in keys) for row in detailed_rows]:
            raise ValueError('Detailed cache ledger disagrees with the active compact benchmark')
    check(query_path)
    check(train_path)
    frozen = read_jsonl(query_path)
    frozen_by_id = {row['benchmark_row_id']: row for row in frozen}
    for qid, smiles in queries.items():
        if qid not in frozen_by_id or frozen_by_id[qid]['drug'] != smiles:
            raise ValueError(f'Query differs from the official {subset} ledger: {qid}')

    # Validate the direct-gold ID, context, model, asset, and ranking integrity so
    # this reusable selector does not depend on a harness-owned context renderer.
    v9_path = Path(policy['cache_manifests']['L1'])
    v9 = json.loads(v9_path.read_text())
    pool = re.fullmatch(r'morgan_top([1-9][0-9]*)_distinct_gold_train_parents_then_all_context_rows', v9.get('candidate_policy', ''))
    if pool is None:
        raise ValueError('L1 requires a frozen gold-train parent-pool cache')
    v9_pool_size = int(pool[1])
    if v9.get('morgan_pool_size', v9_pool_size) != v9_pool_size or molecule_limit > v9_pool_size:
        raise ValueError('L1 shape or declared parent count disagrees with its cache pool')
    check(v9_path)
    check(train_path, v9['inputs']['train_sha256'])
    query_input_key = subset if f'{subset}_sha256' in v9['inputs'] else 'valid'
    cached_query_path = Path(v9['inputs'][query_input_key]).resolve()
    if cached_query_path != query_path.resolve():
        raise ValueError(f'L1 direct-gold cache points to a different {subset} query ledger')
    check(query_path, v9['inputs'][f'{query_input_key}_sha256'])
    from predict.retrieval.assay_reranking.v9 import (
        RANKING_SCHEMA_VERSION, model_profile, reference_provenance, verify_vendored_assets,
    )
    lineage = str(v9.get('model_lineage') or 'v9')
    rankings_path = v9_path.with_name('rankings.parquet')
    if (v9.get('schema_version') != RANKING_SCHEMA_VERSION or v9.get('status') != 'complete'
            or v9.get('rankings_sha256') != check(rankings_path)
            or v9.get('model') != model_profile(task, lineage)
            or v9.get('prompt_assets') != verify_vendored_assets()
            or v9.get('reference_provenance', {}) != reference_provenance(task, lineage)):
        raise ValueError('L1 direct-gold ranking cache is incomplete or incompatible')
    train = {r['benchmark_row_id']: r for r in read_jsonl(train_path)}
    if set(v9.get('training_record_ids') or []) != set(train):
        raise ValueError('L1 direct-gold cache training IDs differ from active gold')
    ranking_rows = pq.read_table(rankings_path).to_pylist()
    if {str(row['query_record_id']) for row in ranking_rows} != set(frozen_by_id):
        raise ValueError(f'L1 direct-gold cache queries differ from active {subset} gold')
    gold_ranks = defaultdict(list)
    for row in ranking_rows:
        known = train.get(row['retrieval_record_id'])
        if (known is None or row['retrieval_molecule_identity_key'] != known['molecule_identity_key']
                or row['retrieval_condition_group'] != known['condition_group']):
            raise ValueError('Frozen direct-gold context identity mismatch')
        gold_ranks[row['query_record_id']].append(row)
    if any(len({r['retrieval_molecule_identity_key'] for r in gold_ranks[qid]}) != v9_pool_size for qid in queries):
        raise ValueError('Incomplete declared direct-gold parent pool')
    v9_audit = {
        'gold_train_labels': str(train_path), 'gold_train_labels_sha256': check(train_path),
        'ranking_version': str(v9_path), 'ranking_version_sha256': check(v9_path),
        'rankings': str(rankings_path), 'rankings_sha256': check(rankings_path),
        'candidate_policy': v9.get('candidate_policy'),
        'model_lineage': lineage,
    }

    source_fields = json.loads(source_contract.read_text())
    if source_fields.get('contract_version') not in {'source_column_contract.v1', 'source_column_contract.v2'}:
        raise ValueError('Unsupported source-field contract')
    fields = {key: [name for name, value in spec['normalized_artifact_columns'].items()
                    if value.get('source_or_simply_cleaned') is True]
              for key, spec in source_fields['sources'].items()}
    if task == 'bbb_martins':
        from data.processing.evidence_library.versions.v10.tasks.bbb_martins import semantic_display
        check(Path(semantic_display.__file__))
    identities, fps = {}, {}

    def identity(smiles):
        if smiles not in identities:
            result = normalize_molecule_identity(smiles)
            if result.status != 'ok' or not result.parent_smiles:
                raise ValueError(f'Unresolved record parent: {smiles}')
            identities[smiles] = result
        return identities[smiles]

    parent_smiles_by_id = {}
    if _parent_scoped_l5_similarity:
        parent_forms = defaultdict(set)
        for row in mapped.itertuples(index=False):
            parent = identity(row.canonical_smiles)
            parent_id = parent.parent_inchi_key or parent.parent_smiles
            parent_forms[parent_id].add(row.canonical_smiles)
        for parent_id, raw_forms in parent_forms.items():
            parent = identity(sorted(raw_forms)[0])
            if parent_id != (parent.parent_inchi_key or parent.parent_smiles):
                raise ValueError(f'Global parent identity mismatch: {parent_id}')
            parent_smiles_by_id[parent_id] = parent.parent_smiles

    by_level = {'L1': defaultdict(list), 'L5': defaultdict(list)}
    forms = {level: defaultdict(set) for level in by_level}
    # Only uncached levels need a full-library parent inventory.
    local = mapped[mapped.level.isin(set(by_level) & set(policy['stages']))]
    for row in local.itertuples(index=False):
        parent = identity(row.canonical_smiles)
        mid = parent.parent_inchi_key or parent.parent_smiles
        by_level[row.level][mid].append(row.canonical_record_id)
        forms[row.level][mid].add(row.canonical_smiles)

    context_mapping_audit = None
    retirement_path = library / 'reviews/l1_training_context_retirement/manifest.json'
    if retirement_path.exists():
        retirement = json.loads(retirement_path.read_text())
        if retirement.get('version') != 'l1_training_context_retirement.v1' or retirement.get('task_id') != task:
            raise ValueError('Training context retirement version/task mismatch')
        check(retirement_path)
        check(source, retirement['stage3_records_sha256'])
        check(train_path, retirement['gold_train_sha256'])
        check(mapping_path, retirement['mapper_sha256'])
        train = {r['benchmark_row_id']: r for r in read_jsonl(train_path)}
        retired = set()
        for row in retirement['contexts']:
            cid = row['context_id']
            known = train.get(cid)
            if (cid in retired or known is None
                    or row['parent'] != known['molecule_identity_key']
                    or set(row['source_record_ids']) != set(known['source_record_ids'])
                    or len(row['source_row_uids']) != len(row['source_record_ids'])
                    or row['parent'] in by_level['L1']
                    or set(row['source_row_uids']) & set(inventory.source_row_uid)
                    or set(row['source_row_uids']) & set(membership.source_row_uid)):
                raise ValueError('Retired training context still has evidence or incompatible source membership')
            retired.add(cid)
        gold_ranks = {qid: [r for r in rows if r['retrieval_record_id'] not in retired]
                      for qid, rows in gold_ranks.items()}
        context_mapping_audit = dict(manifest=str(retirement_path), retired_contexts=sorted(retired),
                                    score_policy='unchanged_frozen_scores_for_retained_contexts')
    mapping_receipt = (DEFAULT_GOLD_CONTEXT_MAPPING if gold_context_mapping is None
                       and task == 'bbb_martins' else gold_context_mapping)
    if mapping_receipt is False:
        mapping_receipt = None
    if mapping_receipt is not None:
        receipt_path = Path(mapping_receipt).resolve()
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('version') != 'bbb_gold_context_record_mapping.v2' or receipt.get('task_id') != task:
            raise ValueError('Gold context mapping version/task mismatch')
        check(receipt_path)
        check(source, receipt['stage3_records_sha256'])
        check(train_path, receipt['gold_train_sha256'])
        check(mapping_path, receipt['mapper_sha256'])
        table_path = receipt_path.with_name('gold_context_record_mapping.tsv')
        check(table_path, receipt['mapping_sha256'])
        with table_path.open() as handle:
            context_rows = list(csv.DictReader(handle, delimiter='\t'))
        train = {r['benchmark_row_id']: r for r in read_jsonl(train_path)}
        expected = {(cid, sid) for cid, c in train.items() for sid in c['source_record_ids']}
        actual = [(r['source_context_id'], r['source_record_id']) for r in context_rows]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError('Gold context mapping does not cover frozen source references exactly')
        context_ids = set()
        by_level['L1'], forms['L1'] = defaultdict(list), defaultdict(set)
        for row in context_rows:
            info = record_info.get(row['canonical_record_id'])
            dest = train.get(row['destination_context_id'])
            origin = train[row['source_context_id']]
            if (row['status'] not in {'unchanged', 'redistributed'} or row['level'] != '1'
                    or info is None or info['level'] != 'L1' or info['source_row_uid'] != row['source_row_uid']
                    or dest is None or dest['condition_group'] != origin['condition_group']
                    or row['source_parent'] != origin['molecule_identity_key']):
                raise ValueError('Invalid UID/level/condition in gold context mapping')
            parent = identity(info['canonical_smiles'])
            mid = parent.parent_inchi_key or parent.parent_smiles
            if mid != row['destination_parent'] or mid != dest['molecule_identity_key']:
                raise ValueError('Gold context mapping parent mismatch')
            context_ids.add(row['destination_context_id'])
            by_level['L1'][mid].append(row['canonical_record_id'])
            forms['L1'][mid].add(info['canonical_smiles'])
        for mid, ids in by_level['L1'].items():
            by_level['L1'][mid] = sorted(set(ids))
        retired = set(train) - context_ids
        if retired != set(receipt['retired_source_contexts']):
            raise ValueError('Gold context retirement mismatch')
        # V9 has validated the frozen cache. Re-select from all its contexts so
        # retirement occurs before parent deduplication, never by redirecting scores.
        rows = pq.read_table(v9_path.with_name('rankings.parquet')).to_pylist()
        gold_ranks = defaultdict(list)
        changed_values = {}
        from predict.retrieval.assay_reranking.v9 import V9PromptRenderer, _record_value
        renderer = V9PromptRenderer(task)
        after = {r['context_id']: r for r in receipt['label_audit'] if not r['retired']}
        for row in rows:
            cid, qid = row['retrieval_record_id'], row['query_record_id']
            if cid not in train or row['retrieval_molecule_identity_key'] != train[cid]['molecule_identity_key'] or row['retrieval_condition_group'] != train[cid]['condition_group']:
                raise ValueError('Frozen V9 context identity mismatch')
            if cid in retired or qid not in queries:
                continue
            if policy['stages']['L1'] != 'morgan':
                known = train[cid]
                counts = after[cid]['label_counts_after'] if cid in after else known['label_counts']
                prompt = renderer.render(dict(smiles=known['drug'], value=_record_value({'label_counts': counts}),
                    condition_group=known['condition_group'], condition_atoms=known.get('condition_atoms', [])),
                    dict(smiles=frozen_by_id[qid]['drug'], condition_group=frozen_by_id[qid]['condition_group'],
                         condition_atoms=frozen_by_id[qid].get('condition_atoms', [])))
                if hashlib.sha256(prompt.encode()).hexdigest() != row['prompt_hash']:
                    original = renderer.render(dict(smiles=known['drug'], value=_record_value(known),
                        condition_group=known['condition_group'], condition_atoms=known.get('condition_atoms', [])),
                        dict(smiles=frozen_by_id[qid]['drug'], condition_group=frozen_by_id[qid]['condition_group'],
                             condition_atoms=frozen_by_id[qid].get('condition_atoms', [])))
                    if hashlib.sha256(original.encode()).hexdigest() != row['prompt_hash']:
                        raise ValueError('L1 cache needs rescoring: frozen prompt mismatch is not a mapped vote-percentage change')
                    changed_values[cid] = dict(frozen_label_counts=known['label_counts'], mapped_label_counts=counts)
            gold_ranks[qid].append(row)
        if changed_values and not allow_frozen_l1_vote_scores:
            raise ValueError(f'L1 assay-transfer cache needs rescoring for {len(changed_values)} mapped contexts with changed model inputs; no stale scores reused')
        context_mapping_audit = dict(manifest=str(receipt_path), retired_contexts=sorted(retired),
            source_references=len(context_rows), destination_contexts=len(context_ids),
            frozen_vote_score_reuse_allowed=allow_frozen_l1_vote_scores,
            changed_vote_contexts=changed_values,
            score_policy=('frozen_pre_redistribution_vote_scores' if changed_values else 'exact_rendered_prompt_hash')
                if policy['stages']['L1'] != 'morgan' else 'scores_not_used')

    output_molecules, output_later, query_audits = {}, {}, {}
    l1_record_overlap_exclusions = defaultdict(set)
    versions, connections, library_compatibility = {}, {}, {}
    with ExitStack() as stack:
        for level, manifest_name in policy['cache_manifests'].items():
            if level == 'L1':
                continue
            path = Path(manifest_name)
            if manifest_name not in versions:
                version = json.loads(path.read_text())
                from predict.retrieval.assay_reranking import v24_1_levels, v25_oral_levels
                owner = v24_1_levels if task == 'bbb_martins' else v25_oral_levels
                multi_pool = version.get('schema_version') == 'assay_transfer_three_pools.v1'
                if version.get('schema_version') not in {'assay_transfer_three_pools.v1', 'txagent_assay_transfer_compact_cache.v1'}:
                    raise ValueError(f'Unsupported cache schema: {path}')
                if version.get('status') != 'complete':
                    raise ValueError(f'Cache is not finalized: {path}; complete missing scores first')
                cache_levels = list(version['models']) if multi_pool else version['levels_in_cache']
                if not cache_levels or not set(cache_levels) <= owner.MODELS.keys():
                    raise ValueError(f'Unsupported cached levels: {path}')
                expected_models = {k: owner.MODELS[k] for k in cache_levels}
                expected = dict(task_id=task, subset=subset, models=expected_models,
                    neighbor_identity_policy='all_parent_forms_scaffold_disjoint' if multi_pool else 'scaffold_disjoint')
                if any(version.get(key) != value for key, value in expected.items()):
                    raise ValueError(f'Cache task/split/model/schema/pool mismatch: {path}')
                if type(version.get('morgan_pool_size')) is not int or version['morgan_pool_size'] < 1:
                    raise ValueError(f'Invalid declared parent-pool size: {path}')
                if multi_pool:
                    from predict.retrieval.assay_reranking import three_pools
                    from predict.retrieval.assay_reranking.runtime import SCORING_CONTRACT_VERSION
                    if cache_pool not in version.get('pools', []) or version.get('scoring_contract_version') != SCORING_CONTRACT_VERSION:
                        raise ValueError(f'Unknown pool or scoring contract: {path}')
                    if version.get('morgan_fingerprint') != dict(radius=2, bits=2048, similarity='Tanimoto'):
                        raise ValueError(f'Unsupported cached fingerprint: {path}')
                    if not _accept_pinned_legacy_builder:
                        check(Path(three_pools.__file__), version['inputs']['builder']['sha256'])
                else:
                    if cache_pool != 'tool-accepted':
                        raise ValueError('Compact cache has one fixed pool; select a multi-pool manifest for another view')
                    accepted = {sha256_file(Path(owner.__file__))}
                    audited_builder = getattr(owner, 'AUDITED_BUILDERS', {}).get(version.get('profile'))
                    if audited_builder:
                        accepted.add(audited_builder)
                    if version.get('implementation_sha256') not in accepted:
                        raise ValueError(f'Cache builder provenance changed: {path}')
                check(path)
                for name, value in version['inputs'].items():
                    if name == 'builder' and _accept_pinned_legacy_builder:
                        continue
                    check(value['path'], value['sha256'])
                targets = (dict(source_contract=source_contract, mapping_manifest=mapper, mapping=mapping_path, queries=query_path)
                           if multi_pool else dict(source_contract=source_contract, level_manifest=mapper,
                                                   level_mapping=mapping_path, queries=query_path))
                for name, target in targets.items():
                    check(target, version['inputs'][name]['sha256'])
                cached_source = version['inputs']['records' if multi_pool else 'stage3_records']
                if check(source) != cached_source['sha256']:
                    library_compatibility[manifest_name] = verify_cached_level_records(
                        cached_source['path'], source, membership, cache_levels)
                elif not multi_pool:
                    check(source_manifest, version['inputs']['stage3_manifest']['sha256'])
                db = path.with_name('scores.sqlite3')
                check(db, version['cache_sha256'])
                connection = sqlite3.connect(f'file:{db.resolve()}?mode=ro', uri=True)
                stack.callback(connection.close)
                declared_integrity = version.get('cache_integrity' if multi_pool else 'cache_quick_check')
                if (declared_integrity != 'ok' or (not _skip_redundant_quick_check
                        and connection.execute('PRAGMA quick_check').fetchone() != ('ok',))):
                    raise ValueError(f'Cache integrity failure: {db}')
                versions[manifest_name], connections[manifest_name] = version, connection
            if level not in versions[manifest_name]['models']:
                raise ValueError(f'Cache does not contain {level}: {path}')

        for qid, smiles in queries.items():
            query = identity(smiles)
            _, _, qfp = standardize_smiles_and_fp(query.parent_smiles)
            similarities, allowed = {}, {}

            def admissible(raw_smiles):
                parent = identity(raw_smiles)
                if raw_smiles not in allowed:
                    allowed[raw_smiles] = not decide_candidate(query, {
                        'canonical_smiles': parent.parent_smiles, 'molecule_identity': parent.to_dict()
                    }, 'scaffold_disjoint').excluded
                return allowed[raw_smiles]

            def similarity(raw_smiles):
                parent = identity(raw_smiles)
                if not admissible(raw_smiles):
                    raise ValueError(f'Candidate scaffold/parent overlap: {qid}/{raw_smiles}')
                if parent.parent_smiles not in fps:
                    _, _, fps[parent.parent_smiles] = standardize_smiles_and_fp(parent.parent_smiles)
                if parent.parent_smiles not in similarities:
                    similarities[parent.parent_smiles] = DataStructs.TanimotoSimilarity(qfp, fps[parent.parent_smiles])
                return similarities[parent.parent_smiles]

            def tie(level, mid, rid):
                return seeded_rank_tie_key(tie_seed, task, qid, level, mid, rid)

            ranked = gold_ranks[qid]
            panels = ['morgan', 'assay_transfer'] if policy['stages']['L1'] == 'joint' else [policy['stages']['L1']]
            selected, audit_levels = {}, {}
            for method in panels:
                ordered = sorted(ranked, key=lambda row: (
                    int(row['model_rank']) if method == 'assay_transfer' else -float(row['morgan_tanimoto_similarity']),
                    tie('L1', row['retrieval_molecule_identity_key'], row['retrieval_record_id'])))
                unique = {}
                for row in ordered:
                    unique.setdefault(row['retrieval_molecule_identity_key'], row)
                ordered = list(unique.values())
                take = len(ordered) if _all_l1_candidates else (5 if len(panels) == 2 else molecule_limit)
                if not _all_l1_candidates and len(ordered) < take:
                    raise ValueError(f'Insufficient surviving L1 parents in frozen pool: {qid}')
                for rank, row in enumerate(ordered[:take], 1):
                    mid = row['retrieval_molecule_identity_key']
                    parent = identity(row['retrieval_smiles'])
                    if mid != (parent.parent_inchi_key or parent.parent_smiles):
                        raise ValueError('V9 parent identity mismatch')
                    rids = by_level['L1'].get(mid)
                    if not rids:
                        raise ValueError(f'Selected gold parent has no mapped V10 L1 records: {qid}/{mid}')
                    if _filter_l1_record_overlap:
                        retained = [rid for rid in rids
                                    if admissible(record_info[rid]['canonical_smiles'])]
                        l1_record_overlap_exclusions[qid].update(set(rids) - set(retained))
                        rids = retained
                        if not rids or not admissible(row['retrieval_smiles']):
                            continue
                    else:
                        for form in forms['L1'][mid]:
                            similarity(form)
                    sim = similarity(row['retrieval_smiles'])
                    if not math.isclose(sim, row['morgan_tanimoto_similarity'], abs_tol=1e-10):
                        raise ValueError('V9 Morgan fingerprint mismatch')
                    if mid not in selected:
                        selected[mid] = dict(reference_molecule_id=mid, canonical_smiles=parent.parent_smiles,
                            morgan_similarity=sim, selection_rank=len(selected), available_l1=len(rids), available_l2=0,
                            ranking_method=policy['stages']['L1'], l2_records=[], gold_context_ids={},
                            l1_records=[dict(record_id=rid, reference_molecule_id=mid, morgan_similarity=sim,
                                ranking_method=policy['stages']['L1'])
                                for rid in sorted(rids, key=lambda rid: tie('L1', mid, rid))[:l1_limit]])
                    selected[mid]['gold_context_ids'][method] = row['retrieval_record_id']
                    if method == 'assay_transfer':
                        score = float(row['prob_transfer'])
                        if not math.isfinite(score) or not 0 <= score <= 1:
                            raise ValueError('Invalid V9 transfer probability')
                        selected[mid]['transfer_likelihood'] = score
                    if len(panels) == 2:
                        selected[mid].setdefault('morgan_top5_rank', 'not_selected_in_top5')
                        selected[mid].setdefault('assay_transfer_top5_rank', 'not_selected_in_top5')
                        selected[mid][f'{method}_top5_rank'] = rank
            if len(panels) == 2:
                assay_rows = {}
                for row in sorted(ranked, key=lambda item: int(item['model_rank'])):
                    assay_rows.setdefault(row['retrieval_molecule_identity_key'], row)
                for mid, molecule in selected.items():
                    score = float(assay_rows[mid]['prob_transfer'])
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError('Invalid V9 transfer probability')
                    molecule['transfer_likelihood'] = score
            seen = {r['record_id'] for m in selected.values() for r in m['l1_records']}
            audit_levels['L1'] = dict(candidate_molecules=len({r['retrieval_molecule_identity_key'] for r in ranked}),
                candidate_contexts=len(ranked), selected_molecules=len(selected),
                selected_records=len(seen), candidate_sha256=hashlib.sha256(json.dumps(sorted(
                    (r['retrieval_molecule_identity_key'], r['retrieval_record_id']) for r in ranked)).encode()).hexdigest())
            later = {}
            for level, method in policy['stages'].items():
                if level == 'L1':
                    continue
                candidates = []
                if level == 'L5':
                    for mid, rids in by_level['L5'].items():
                        if not all(admissible(form) for form in forms['L5'][mid]):
                            continue
                        parent_smiles = (parent_smiles_by_id[mid]
                                         if _parent_scoped_l5_similarity else None)
                        similarity_smiles = parent_smiles or record_info[rids[0]]['canonical_smiles']
                        sim = similarity(similarity_smiles)
                        candidates.extend(dict(record_id=rid, reference_molecule_id=mid,
                            morgan_similarity=sim, ranking_method='morgan',
                            **({'reference_parent_smiles': parent_smiles} if parent_smiles else {}))
                            for rid in rids)
                else:
                    connection = connections[policy['cache_manifests'][level]]
                    version = versions[policy['cache_manifests'][level]]
                    if version['schema_version'] == 'assay_transfer_three_pools.v1':
                        registered = connection.execute('SELECT query_id,drug FROM benchmark_queries WHERE benchmark_row_id=?', (qid,)).fetchone()
                        if registered is None or registered[1] != smiles:
                            raise ValueError(f'Query differs from frozen pool ledger: {qid}')
                        rows = connection.execute('''SELECT r.external_record_id,r.parent_id,s.transfer_probability
                            FROM assignments a JOIN records r USING(record_key) LEFT JOIN scores s USING(score_key)
                            WHERE a.query_id=? AND a.pool=? AND a.level=?''', (registered[0],cache_pool,level)).fetchall()
                    else:
                        rows = connection.execute('''SELECT r.external_record_id, m.molecule_chembl_id,
                        s.transfer_probability FROM assignments a JOIN queries q USING(query_id)
                        JOIN groups_dim g USING(group_key) JOIN records r USING(record_key)
                        JOIN molecules m USING(molecule_key) LEFT JOIN scores s USING(score_key)
                        WHERE q.query_smiles=? AND g.group_id=?''', (query.parent_smiles, level)).fetchall()
                    for rid, mid, score in rows:
                        info = record_info.get(rid)
                        if info is None or info['level'] != level:
                            raise ValueError(f'Cached record missing from mapped {level}: {rid}')
                        parent = identity(info['canonical_smiles'])
                        if mid != (parent.parent_inchi_key or parent.parent_smiles):
                            raise ValueError(f'Cached record parent mismatch: {rid}')
                        sim = similarity(info['canonical_smiles'])
                        if score is None or not math.isfinite(score) or not 0 <= score <= 1:
                            raise ValueError(f'Missing or invalid cached score: {rid}')
                        candidates.append(dict(record_id=rid, reference_molecule_id=mid,
                            morgan_similarity=sim, ranking_method=method,
                            **({'transfer_likelihood': score} if method == 'assay_transfer' else {})))
                    if len({r['reference_molecule_id'] for r in candidates}) != version['morgan_pool_size']:
                        raise ValueError(f'Incomplete cached {version["morgan_pool_size"]}-parent pool: {qid}/{level}')
                ids = [r['record_id'] for r in candidates]
                if len(ids) != len(set(ids)):
                    raise ValueError(f'Duplicate cache assignments: {qid}/{level}')
                field = 'transfer_likelihood' if method == 'assay_transfer' else 'morgan_similarity'
                ordered = sorted((r for r in candidates if r['record_id'] not in seen),
                    key=lambda r: (-r[field], tie(level, r['reference_molecule_id'], r['record_id'])))
                chosen = ordered[:later_limit]
                seen.update(r['record_id'] for r in chosen)
                audit_levels[level] = dict(candidate_records=len(candidates),
                    candidate_molecules=len({r['reference_molecule_id'] for r in candidates}),
                    selected_records=len(chosen), selected_molecules=len({r['reference_molecule_id'] for r in chosen}),
                    candidate_sha256=hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest(),
                    shortfall=later_limit-len(chosen))
                later[level] = dict(records=chosen, available_record_count=len(ordered),
                    allow_shortfall=True, ranking_method=method, **audit_levels[level])
            output_molecules[qid], output_later[qid] = list(selected.values()), later
            query_audits[qid] = audit_levels

    selected_rows = [r for molecules in output_molecules.values() for m in molecules for r in m['l1_records']]
    selected_rows += [r for levels in output_later.values() for entry in levels.values() for r in entry['records']]
    wanted = {row['record_id'] for row in selected_rows}
    payloads = {}
    for batch in pq.ParquetFile(source).iter_batches(batch_size=10000):
        ids = batch.column(batch.schema.get_field_index('canonical_record_id')).to_pylist()
        for row in batch.filter([rid in wanted for rid in ids]).to_pylist():
            rid = row['canonical_record_id']
            values = {name: row.get(name) for name in fields[row['source_id']]}
            if task == 'bbb_martins':
                values = semantic_display.semantic_prompt_payload(row, values)
            payloads[rid] = dict(record_id=rid, source_row_uid=row['source_row_uid'], task_id=task,
                source_id=row['source_id'], progressive_level=record_info[rid]['level'],
                family_key=record_info[rid]['family_key'],
                canonical_smiles=identity(row['canonical_smiles']).parent_smiles,
                source_canonical_smiles=row['canonical_smiles'], source_fields=values,
                source_contract={
                    'contract_version': source_fields['contract_version'],
                    'source_or_simply_cleaned': {name: True for name in values},
                },
                source_projection='library_source_contract.v2', measurement_kind=row['measurement_kind'])
    if set(payloads) != wanted:
        raise ValueError('Selected records missing library payloads')
    for row in selected_rows:
        row['payload'] = payloads[row['record_id']]
    audit = dict(
        selection_policy='cache_matched_retrieval.v1', inputs=inputs,
        contract=dict(policy=policy, inputs=inputs, molecule_limit=molecule_limit, l1_limit=l1_limit,
                      later_limit=later_limit, tie_seed=tie_seed,
                      allow_frozen_l1_vote_scores=allow_frozen_l1_vote_scores, cache_pool=cache_pool),
        v9=v9_audit, l1_pool_size=v9_pool_size, gold_context_mapping=context_mapping_audit, cache_versions=versions, query_audits=query_audits,
        library_compatibility=library_compatibility, cache_pool=cache_pool,
        mapped_level_counts=mapped.level.value_counts().to_dict(), unmapped_record_ids=unmapped,
        l1_record_overlap_exclusions={qid: sorted(record_ids) for qid, record_ids
                                      in sorted(l1_record_overlap_exclusions.items()) if record_ids},
        neighbor_identity_policy='scaffold_disjoint', similarity_floor=None,
        scaffold_overlap=0, parent_overlap=0)
    if _parent_scoped_l5_similarity:
        audit['_parent_smiles_by_id'] = parent_smiles_by_id
    return output_molecules, output_later, audit


def load_candidates(queries, *, task, subset, library=None, mapper=None, policy,
                    molecule_limit=10, l1_limit=10, later_limit=50, tie_seed=0,
                    gold_context_mapping=None, allow_frozen_l1_vote_scores=False,
                    cache_pool='tool-accepted', joint_panel_sizes=None,
                    min_contrast=3, morgan_primary_parent_width=100):
    """Load a complete v2 cache directly, or explicitly replay a legacy bundle."""
    if policy.get('selection_contract') in {
        'ranked_level_retrieval.v2', 'ranked_uid_retrieval.v1'
    }:
        if policy['selection_contract'] == 'ranked_uid_retrieval.v1':
            from .ranked_uid_retrieval import load_candidates as load_level_candidates
        else:
            from .ranked_level_retrieval import load_candidates as load_level_candidates
        return load_level_candidates(
            queries, task=task, subset=subset, policy=policy,
            molecule_limit=molecule_limit, l1_limit=l1_limit,
            later_limit=later_limit, tie_seed=tie_seed, cache_pool=cache_pool,
            min_contrast=min_contrast,
            morgan_primary_parent_width=morgan_primary_parent_width,
        )
    if policy.get('selection_contract') == 'cache_matched_retrieval.v2':
        from .cache_matched_v2 import load_candidates as load_v2_candidates
        return load_v2_candidates(
            queries, task=task, subset=subset, policy=policy,
            molecule_limit=molecule_limit, l1_limit=l1_limit,
            later_limit=later_limit, tie_seed=tie_seed, cache_pool=cache_pool,
            joint_panel_sizes=joint_panel_sizes,
        )
    if policy.get('selection_contract') == 'cache_matched_retrieval.v3':
        from .cache_matched_v3 import load_candidates as load_v3_candidates
        return load_v3_candidates(
            queries, task=task, subset=subset, policy=policy,
            molecule_limit=molecule_limit, l1_limit=l1_limit,
            later_limit=later_limit, tie_seed=tie_seed, cache_pool=cache_pool,
            joint_panel_sizes=joint_panel_sizes,
        )
    if policy.get('selection_contract') == 'ranked_evidence_retrieval.v1':
        from .ranked_retrieval import load_candidates as load_ranked_candidates
        return load_ranked_candidates(
            queries, task=task, subset=subset, policy=policy,
            molecule_limit=molecule_limit, l1_limit=l1_limit,
            later_limit=later_limit, tie_seed=tie_seed, cache_pool=cache_pool,
            joint_panel_sizes=joint_panel_sizes,
        )
    if policy.get('selection_contract') == 'semantic_bucket_reranking.v1':
        from .semantic_bucket_cache import load_candidates as load_semantic_candidates
        return load_semantic_candidates(
            queries, task=task, subset=subset, policy=policy,
            molecule_limit=molecule_limit, l1_limit=l1_limit,
            later_limit=later_limit, tie_seed=tie_seed, cache_pool=cache_pool,
        )
    if policy.get('selection_contract') in {
        'l1_context_retrieval.v1', 'l1_context_retrieval.v2',
        'l1_context_semantic_l2.v1',
        'l1_context_semantic_weighted_l2.v1',
        'l1_context_morgan_semantic_l2.v1',
        'l1_context_morgan_semantic_l2.v2',
        'l1_context_morgan_semantic_l2.v3',
        'l1_context_morgan_semantic_l2.v4'
    }:
        from .l1_context_cache import load_candidates as load_l1_context_candidates
        return load_l1_context_candidates(
            queries, task=task, subset=subset, policy=policy,
            molecule_limit=molecule_limit, l1_limit=l1_limit,
            later_limit=later_limit, tie_seed=tie_seed,
            min_contrast=min_contrast, cache_pool=cache_pool,
        )
    if policy.get('selection_contract') in {
        'indirect_morgan_semantic_l2_l4.v1',
        'indirect_morgan_semantic_l2_l4.v2',
        'indirect_morgan_semantic_l2_l4.v3',
    }:
        from .indirect_cache import load_candidates as load_indirect_candidates
        return load_indirect_candidates(
            queries, task=task, subset=subset, policy=policy,
            later_limit=later_limit, tie_seed=tie_seed, cache_pool=cache_pool,
        )
    if library is None or mapper is None:
        raise ValueError('Legacy cache replay requires the evidence library and level mapper')
    if joint_panel_sizes is not None:
        raise ValueError('Custom joint panels require cache-matched retrieval v2 or v3')
    return _load_candidates_v1(
        queries, task=task, subset=subset, library=library, mapper=mapper,
        policy=policy, molecule_limit=molecule_limit, l1_limit=l1_limit,
        later_limit=later_limit, tie_seed=tie_seed,
        gold_context_mapping=gold_context_mapping,
        allow_frozen_l1_vote_scores=allow_frozen_l1_vote_scores,
        cache_pool=cache_pool,
    )
