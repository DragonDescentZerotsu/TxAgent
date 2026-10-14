"""Generate sigmoid transfer targets and per-level empirical binary targets after selection."""
import hashlib
import math
import statistics
from collections import defaultdict
from .storage import read_json,read_parquet,write_json,write_parquet

def stable_sha256(value):
    return hashlib.sha256(value.encode()).hexdigest()


def effective_value(record, calibration):
    if '_effective_geometry_value' in record:
        return record['_effective_geometry_value']
    value = float(record['geometry_value'])
    return math.log10(value) if calibration.get('logit_geometry_transform') == 'log10' else value


def prepare_geometry(records, calibration):
    return {key: {**row, '_effective_geometry_value': effective_value(row, calibration[row['pair_bucket_key']])}
            for key, row in records.items()}


def probability(reference, query, calibration, target):
    sd = calibration.get('logit_calibration_sample_standard_deviation') or calibration['calibration_sample_standard_deviation']
    distance = abs(effective_value(reference, calibration) - effective_value(query, calibration)) / sd
    logit = (target['center_sd'] - distance) / target['temperature_sd']
    return 1 / (1 + math.exp(-max(-700, min(700, logit))))


def binary_targets(pairs, records, calibration, target, reject_missing=False):
    values = defaultdict(lambda: defaultdict(list))
    binary_levels = set()
    for pair in pairs:
        query, reference = records[pair['b_id']], records[pair['a_id']]
        if query['measurement_kind'] == 'binary':
            binary_levels.add(query['level'])
            continue
        score = probability(reference, query, calibration[query['pair_bucket_key']], target)
        values[query['level']]['near' if score >= 0.5 else 'far'].append(score)
    output = {}
    for level in sorted(binary_levels):
        near, far = values[level]['near'], values[level]['far']
        if (not near or not far) and reject_missing:
            output[level] = {'status':'rejected','reason':'no_numeric_near_far_population_within_level'}
            continue
        if not near or not far:
            raise ValueError(f'No numeric near/far population for binary level {level}')
        output[level] = {'match_probability': statistics.fmean(near), 'mismatch_probability': statistics.fmean(far),
                         'near_numeric_pair_count': len(near), 'far_numeric_pair_count': len(far)}
    return output


def generate(stage,task_root,policy,family,smoke):
    from .sampling import PAIR_SCHEMA
    pool,sample,selection=task_root/'pool',task_root/'sampling',task_root/'augmentation'
    records=({r['record_id']:r for r in read_parquet(pool/'records.parquet')}
             if family=='record' else read_json(sample/'records.json'))
    calibration=read_json(sample/'calibration.json')
    buckets=calibration['accepted_buckets'] if family=='record' else calibration.get('indirect_calibration',{}).get('accepted_buckets',{})
    population=[p for p in read_parquet(sample/'pool_train.parquet') if p['component']=='indirect']
    fitted=binary_targets(population,records,buckets,policy['common']['numeric_target'],reject_missing=smoke)
    for level in read_json(sample/'sampling.json')['rejected_binary_levels']:
        fitted[level]={'status':'rejected','reason':'no_numeric_near_far_population_within_level'}
    counts={}
    for split in ('train','validation_ranking','test_ranking'):
        pairs=read_parquet(selection/f'{split}.parquet')
        for item in pairs:
            reference,query=records[item['a_id']],records[item['b_id']]
            if query['measurement_kind']=='binary':
                entry=fitted[query['level']]
                item['probability']=entry['match_probability' if reference['category_id']==query['category_id'] else 'mismatch_probability']
            else:
                entry=calibration if item['component']=='direct' else buckets[query['pair_bucket_key']]
                item['probability']=probability(reference,query,entry,policy['common']['numeric_target'])
        counts[split]=write_parquet(stage/f'{split}.parquet',pairs,PAIR_SCHEMA)
    calibration['empirical_binary_targets']=fitted
    write_json(stage/'calibration.json',calibration)
    write_json(stage/'effective_contract.json',policy)
    return {'scope':'smoke' if smoke else 'full','family':family,'coverage':counts,'target_fit_scope':'full_sampled_pool_before_augmentation'}
