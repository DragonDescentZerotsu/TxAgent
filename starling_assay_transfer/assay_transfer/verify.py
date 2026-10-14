"""Verify lineage, constraints, targets and deterministic released rows."""

import math
from collections import Counter, defaultdict
from itertools import zip_longest

import pyarrow.parquet as pq

from .storage import digest, read_json, read_parquet, verify_manifest
from .filtering import ratio_diagnostics,gold_calibration
from .dataset import rendered_rows, release_schema
from .split import check_disjointness


def lineage(records, gate, calibration, policy, heldout=()):
    if policy['record_levels'] is not None and any(row['level'] not in policy['record_levels'] for row in records + gate):
        raise ValueError('Constructed record level is excluded by the contract')
    index = {row['record_id']: row for row in records + gate}
    for bucket, entry in calibration.items():
        fit = [index[key] for key in entry['calibration_fit_record_ids']]
        if any(row['dataset_split'] in ('test','test_gate_only','gate_only') for row in fit):
            raise ValueError('Test/gate-only values fit targets')
        rows = [index[key] for key in entry['ratio_gate_record_ids']]
        actual = ratio_diagnostics(rows, entry)['within_to_parent_mean_variance_ratio']
        if actual is None or not math.isfinite(actual) or actual > policy['variance_ratio_maximum']:
            raise ValueError(f'Invalid variance gate: {bucket}')
        if not math.isclose(actual,entry['within_to_parent_mean_variance_ratio'],rel_tol=1e-12,abs_tol=1e-15):
            raise ValueError('Recorded variance diagnostics differ from source records')
    training = [row for row in records if row['dataset_split']=='train']
    queries = [row for row in records if row.get('query_validation') or row.get('query_test')]
    audit = check_disjointness(training, list(heldout) + queries, policy)
    return {**audit, 'buckets_checked':len(calibration), 'evaluation_queries_checked':len(queries), 'test_target_fit':False}


def constraints(pairs, records, policy, family):
    ids, unique, directed = set(), set(), set()
    usage, binary, query_counts, parents = Counter(), defaultdict(Counter), Counter(), Counter()
    for item in pairs:
        query, reference = records[item['b_id']], records[item['a_id']]
        if item['pair_id'] in ids or not math.isfinite(item['probability']) or not 0 <= item['probability'] <= 1:
            raise ValueError('Duplicate pair or invalid probability')
        ids.add(item['pair_id'])
        if query['record_id']==reference['record_id']:
            raise ValueError('Pair repeats its query record')
        if item['split'] != 'train':
            if family == 'context':
                expected_split = 'valid' if item['split']=='validation_ranking' else 'test'
                if item['component'] != 'direct' or reference['dataset_split'] != 'train' or query['dataset_split'] != expected_split:
                    raise ValueError('Context evaluation requires heldout Gold queries and Gold training references')
            continue
        if query['parent_id'] == reference['parent_id'] or query['dataset_split'] != 'train' or reference['dataset_split'] != 'train':
            raise ValueError('Invalid training parent or split')
        key = item['component'], query['record_id'], reference['parent_id']
        if key in unique:
            raise ValueError('Repeated reference parent per query')
        unique.add(key)
        if item['component'] == 'direct':
            edge = reference['parent_id'], query['parent_id']
            if edge[::-1] in directed:
                raise ValueError('Reverse direct parent edge')
            directed.add(edge)
            query_counts[query['record_id']] += 1
            continue
        if query['pair_bucket_key'] != reference['pair_bucket_key']:
            raise ValueError('Indirect training pair crosses assay/level bucket')
        cap = policy['record_degree_caps_by_level'].get(query['level'], policy['record_degree_cap']) if family == 'record' else policy['context_indirect_degree_cap']
        for role, row in (('query',query),('reference',reference)):
            usage[role,row['record_id']] += 1
            if usage[role,row['record_id']] > cap:
                raise ValueError('Record degree exceeds effective contract')
            group = 'continuous' if policy['task'] == 'Bioavailability_Ma' and row['measurement_kind'] != 'binary' else row['measurement_kind']
            parents[role,row['level'],group,row['parent_id']] += 1
            if policy['evaluation_levels'] and parents[role,row['level'],group,row['parent_id']] > 4096:
                raise ValueError('Parent family budget exceeded')
        if query['measurement_kind'] == 'binary':
            binary[query['record_id']][int(query['category_id'] == reference['category_id'])] += 1
    if family == 'record' and policy['task'] in ('BBB_Martins','Bioavailability_Ma'):
        if any(max(counts.values()) > 3 * min(counts[0],counts[1]) for counts in binary.values()):
            raise ValueError('Binary training query violates the 75/25 constraint')
    expected = policy['context_direct_degree_cap'] // 2
    if family == 'context' and any(count > expected for count in query_counts.values()):
        raise ValueError('Direct query degree differs from contract')
    return {'pairs':len(ids),'training_query_maximum':max(usage.values(),default=0),
            'direct_training_queries':len(query_counts),'constraints_passed':True}


def verify_task(root, policy, family, workers=1):
    split_root = root / 'pool'
    pair_root = root / 'targets'
    release = root / 'dataset' / family
    manifests = {name:verify_manifest(path) for name,path in (('pool',split_root),('sampling',root/'sampling'),('augmentation',root/'augmentation'),('targets',pair_root),('dataset',release))}
    record_root = split_root if family=='record' else split_root / 'indirect'
    split_audit = {}
    if record_root.exists():
        records = read_parquet(record_root / 'records.parquet')
        gate = read_parquet(record_root / 'gate_records.parquet')
        calibration = read_json(record_root / 'calibration.json')['accepted_buckets']
        gold = read_json(record_root / 'gold.json')
        for flag, split in (('query_validation','valid'), ('query_test','test')):
            parents = {row['parent_id'] for row in gold[split]}
            if any(row.get(flag) and row['parent_id'] not in parents for row in records):
                raise ValueError('Record evaluation query is outside its pinned Gold split')
        split_audit = lineage(records, gate, calibration, policy, read_json(record_root / 'heldout_identities.json'))
    index = {row['record_id']:row for row in records} if family=='record' else read_json(root / 'sampling/records.json')
    if family=='context':
        groups = read_json(split_root / 'gold.json')
        split_audit['gold'] = check_disjointness(groups['train'], groups['valid'] + groups['test'], policy)
        calibration = read_json(pair_root / 'calibration.json')
        expected_calibration = gold_calibration(groups, split_root / 'gold.json')
        if any(calibration.get(key) != value for key,value in expected_calibration.items()):
            raise ValueError('Gold calibration differs from retained train plus validation')
        training = read_parquet(pair_root / 'train.parquet')
        components = Counter(item['component'] for item in training)
        budget = int(components['direct'] * policy['augmentation_ratio'])
        coverage = read_json(root / 'augmentation/coverage.json')
        if coverage['direct_rows'] != components['direct'] or coverage['indirect_budget']['requested'] != budget or coverage['indirect_budget']['selected'] != components['indirect']:
            raise ValueError('Augmentation coverage differs from emitted training')
        if manifests['pool']['scope']=='full' and components['indirect'] != budget:
            raise ValueError('Full augmentation budget shortfall')
    checks, counts = {}, {}
    for split in ('train','validation_ranking','test_ranking'):
        pairs = read_parquet(pair_root / f'{split}.parquet')
        checks[split] = constraints(pairs,index,policy,family)
        path = release / split / 'data.parquet'
        if pq.read_schema(path) != release_schema():
            raise ValueError('Released schema differs from the standalone schema')
        actual = (row for batch in pq.ParquetFile(path).iter_batches(2000) for row in batch.to_pylist())
        expected = rendered_rows(pairs,index,policy,family,workers)
        for got, wanted in zip_longest(actual,expected):
            if got != wanted:
                raise ValueError(f'Released row differs from its source pair: {split}')
        counts[split] = len(pairs)
    receipt = {'schema_version':'assay_transfer.verification.v2', 'status':'verified',
               'scope':manifests['pool']['scope'], 'family':family, 'task':policy['task'],
               'effective_contract_sha256':policy['contract_sha256'], 'split_audit':split_audit,
               'checks':checks,'coverage':counts,'manifest_sha256':{key:digest(value) for key,value in manifests.items()}}
    return receipt
