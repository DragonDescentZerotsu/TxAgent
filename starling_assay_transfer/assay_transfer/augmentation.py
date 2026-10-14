"""Apply indirect augmentation and task-specific mixtures to sampled pairs."""
from collections import Counter,defaultdict,deque
from itertools import zip_longest
from .storage import read_json,read_parquet,write_json,write_parquet
from .targets import stable_sha256
from .sampling import PAIR_SCHEMA,coverage
from .dataset import prompt,context_prompt

def oral_selection(candidates, limit):
    grouped = defaultdict(lambda: defaultdict(list))
    for item in candidates:
        grouped[item['_bucket']][item['b_id']].append(item)
    queues = {bucket: deque(values for _, values in sorted(queries.items(), key=lambda item: (
        stable_sha256(f'v10.3:{item[0]}'), item[0]))) for bucket, queries in grouped.items()}
    active = deque(sorted(queues, key=lambda bucket: (stable_sha256(f'v10.3:bucket:{bucket}'), bucket)))
    selected, references, labels = [], set(), defaultdict(Counter)
    while active and len(selected) < limit:
        bucket = active.popleft()
        row = min(queues[bucket].popleft(), key=lambda item: (labels[bucket][item['near']],
                  item['a_id'] in references, -item['similarity'], item['pair_id']))
        selected.append(row)
        references.add(row['a_id'])
        labels[bucket][row['near']] += 1
        if queues[bucket]:
            active.append(bucket)
    return selected


def mixture(candidates, records, policy, limit, smoke):
    task = policy['task']
    eligible = [item for item in candidates if records[item['b_id']]['measurement_kind'] != 'binary']
    if task == 'BBB_Martins':
        eligible = [item for item in eligible if records[item['b_id']]['source_id'] in
                    ('efflux_transport','influx_transport','passive_permeability')]
    elif task == 'Bioavailability_Ma':
        eligible = [item for item in eligible if records[item['b_id']]['source_id'] != 'hf_bioavailability']
        for item in eligible:
            item['_bucket'] = records[item['b_id']]['pair_bucket_key']
        selected = oral_selection(eligible, limit)
    if task == 'Skin_Reaction':
        weights = {'L1':22767,'L2':22767,'L3':2100}
        quotas = {level: limit * weight // sum(weights.values()) for level, weight in weights.items()}
        for level in list(weights)[:limit-sum(quotas.values())]:
            quotas[level] += 1
        selected = select(candidates, records, quotas, smoke)
        expected = limit
    else:
        expected = limit
        if task != 'Bioavailability_Ma':
            selected = sorted(eligible if task == 'BBB_Martins' else candidates,
                              key=lambda item: (stable_sha256(f'v10.3:bbb:{item["pair_id"]}'), item['pair_id']))[:limit]
    if not smoke and len(selected) != expected:
        raise ValueError(f'Indirect mixture budget shortfall: {len(selected)} < {expected}')
    for item in selected:
        item.pop('_bucket', None)
    return selected, {'requested': expected, 'selected': len(selected), 'shortfall': max(0, expected-len(selected))}


def clean_prompts(pairs, records):
    groups = defaultdict(list)
    for item in pairs:
        reference, query = records[item['a_id']],records[item['b_id']]
        text = context_prompt(reference,query) if item['component']=='direct' else prompt(reference,query)
        groups[text].append(item)
    keep, conflicts, duplicates = set(),0,0
    for items in groups.values():
        if len({item['near'] for item in items}) > 1:
            conflicts += len(items)
        else:
            keep.add(min(items,key=lambda item:item['pair_id'])['pair_id'])
            duplicates += len(items)-1
    output = [item for item in pairs if item['pair_id'] in keep]
    return output, {'policy':'drop_conflicting_full_prompts_then_keep_smallest_pair_id_for_same_completion',
                    'input_rows':len(pairs),'output_rows':len(output),'conflicting_rows_dropped':conflicts,
                    'duplicate_rows_dropped':duplicates}

def select(candidates, records, limits, smoke):
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for pair in candidates:
        row = records[pair['b_id']]
        if row['level'] in limits:
            grouped[row['level']][row['pair_bucket_key']][int(pair['near'])].append(pair)
    chosen = []
    for level, limit in limits.items():
        buckets = grouped[level]
        queues = []
        for bucket in sorted(buckets, key=stable_sha256):
            labels = buckets[bucket]
            near = sorted(labels[1], key=lambda p: p['pair_id'])
            far = sorted(labels[0], key=lambda p: p['pair_id'])
            queues.append([item for batch in zip_longest(far,near) for item in batch if item])
        rounds = [item for batch in zip_longest(*queues) for item in batch if item]
        if not smoke and len(rounds) < limit:
            raise ValueError(f'Insufficient Skin {level} indirect quota: {len(rounds)} < {limit}')
        chosen.extend(rounds[:limit])
    return chosen


def record_mixture(pairs, records, smoke):
    native=[p for p in pairs if records[p['b_id']]['level'] in ('L2','L3')]
    quotas={level:len(native)//3+int(index<len(native)%3) for index,level in enumerate(('L1','L4','EXPOSURE'))}
    selected=[]
    for level,limit in quotas.items():
        candidates=[p for p in pairs if records[p['b_id']]['level']==level]
        buckets=defaultdict(lambda:defaultdict(list))
        for p in candidates:
            buckets[records[p['b_id']]['pair_bucket_key']][int(p['near'])].append(p)
        ranked=[]
        for bucket,labels in buckets.items():
            near=sorted(labels[1],key=lambda p:p['pair_id']);far=sorted(labels[0],key=lambda p:p['pair_id'])
            ordered=[p for batch in zip_longest(far,near) for p in batch if p]
            for index,p in enumerate(ordered):
                ranked.append((index,-records[p['b_id']]['bucket_train_record_count'],stable_sha256(bucket),p['pair_id'],p))
        chosen=[entry[-1] for entry in sorted(ranked)[:limit]]
        if not smoke and len(chosen)!=limit:
            raise ValueError(f'Skin donor quota shortfall: {level}: {len(chosen)} < {limit}')
        selected.extend(chosen)
    return native+selected, {'native':len(native),'donors':len(selected),'quotas':quotas,
                             'selected':dict(Counter(records[p['b_id']]['level'] for p in selected))}


def augment_record(stage,pool,sample,policy,smoke):
    records={r['record_id']:r for r in read_parquet(pool/'records.parquet')}
    training=read_parquet(sample/'train.parquet')
    report=read_json(sample/'sampling.json')
    if policy['task']=='Skin_Reaction':
        training,report['donor_mixture']=record_mixture(training,records,smoke)
        report['coverage']=coverage(training,[r for r in records.values() if r['dataset_split']=='train'])
    counts={}
    for split in ('train','validation_ranking','test_ranking'):
        pairs=training if split=='train' else read_parquet(sample/f'{split}.parquet')
        counts[split]=write_parquet(stage/f'{split}.parquet',pairs,PAIR_SCHEMA)
    write_json(stage/'sampling.json',report)
    return {'scope':'smoke' if smoke else 'full','family':'record','coverage':report['coverage'],'pairs':counts}


def augment_context(stage,sample,policy,smoke):
    records=read_json(sample/'records.json')
    candidates=read_parquet(sample/'train.parquet')
    training,cleanup=clean_prompts(candidates,records)
    direct=[p for p in training if p['component']=='direct']
    candidates=[p for p in training if p['component']=='indirect']
    if not direct:raise ValueError('No direct examples remain after prompt cleanup')
    indirect,budget=mixture(candidates,records,policy,int(len(direct)*policy['augmentation_ratio']),smoke)
    counts={}
    for split in ('train','validation_ranking','test_ranking'):
        pairs=direct+indirect if split=='train' else read_parquet(sample/f'{split}.parquet')
        counts[split]=write_parquet(stage/f'{split}.parquet',pairs,PAIR_SCHEMA)
    source=read_json(sample/'sampling.json')
    report={'scope':'smoke' if smoke else 'full','family':'context','pairs':counts,'indirect_budget':budget,
            'direct_rows':len(direct),'augmentation_ratio_requested':policy['augmentation_ratio'],
            'augmentation_ratio_emitted':len(indirect)/len(direct),'prompt_cleanup':cleanup,
            'sampling':source['indirect_sampling'],'gold_rows':source['gold_rows']}
    write_json(stage/'coverage.json',report)
    return report
