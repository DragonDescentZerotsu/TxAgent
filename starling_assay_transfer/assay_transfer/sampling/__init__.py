"""Select record and Gold pairs from a filtered pool, before augmentation and targets."""
from collections import Counter,defaultdict
import pyarrow as pa
from rdkit import DataStructs
from assay_transfer.storage import digest,read_json,read_parquet,write_json,write_parquet
from assay_transfer.targets import effective_value,prepare_geometry,stable_sha256
from . import bbb as bbb_sampler,bbb_repair as bbb_sampler_repair,oral as oral_sampler,oral_repair as oral_sampler_repair,safety
from .safety import _fingerprints
from .bbb_trim import trim_binary
from .oral_trim import trim_binary as oral_trim

PAIR_SCHEMA=pa.schema([pa.field(key,pa.string()) for key in
    ('pair_id','a_id','b_id','split','component','calibration_scope','transform','calibration_sha256','source_manifest_sha256')]
    +[pa.field(key,pa.float64()) for key in ('probability','similarity','sd','effective_distance','standardized_distance')]
    +[pa.field('retrieval_parent_rank',pa.int64()),pa.field('near',pa.bool_())])


def edge(reference, query, split, component, similarity, calibration, policy, rank=None):
    distance=abs(effective_value(reference,calibration)-effective_value(query,calibration))
    binary=query['measurement_kind']=='binary'
    sd=calibration.get('logit_calibration_sample_standard_deviation') or calibration.get('calibration_sample_standard_deviation')
    near=reference['category_id']==query['category_id'] if binary else distance/sd <= policy['common']['numeric_target']['center_sd']
    return {'pair_id':digest([policy['contract_sha256'],component,split,reference['record_id'],query['record_id']]),
            'a_id':reference['record_id'],'b_id':query['record_id'],'split':split,'component':component,
            'near':near,'probability':None,'similarity':similarity,'retrieval_parent_rank':rank,
            'calibration_scope':calibration['calibration_scope'],'calibration_sha256':calibration.get('calibration_sha256',''),
            'source_manifest_sha256':calibration.get('source_manifest_sha256',''),
            'transform':calibration.get('logit_geometry_transform','identity'),'sd':sd,
            'effective_distance':distance,'standardized_distance':None if binary else distance/sd}

def sample_level(train, calibration, policy, component='record', limit=None, smoke=False):
    rows = sorted(train, key=lambda row: row['record_id'])
    rows = list(prepare_geometry({row['record_id']: row for row in rows}, calibration).values())
    fps = safety._fingerprints(rows)
    caps = {row['pair_bucket_key']: {'resolved_degree_cap': policy['record_degree_caps_by_level'].get(
        row['level'], policy['record_degree_cap']) if component == 'record' else policy['context_indirect_degree_cap']}
        for row in rows}
    if not rows:
        raise ValueError('No training records')
    if not policy['evaluation_levels']:
        cap = policy['record_degree_cap'] if component == 'record' else policy['context_indirect_degree_cap']
        edges, phases, _ = safety._selected_edges(rows, calibration, fps, limit, smoke, cap)
        report = {'sampling_phases': phases, 'method': policy['record_sampling_policy']}
    else:
        oral = policy['task'] == 'Bioavailability_Ma'
        sampler, repairer = (oral_sampler, oral_sampler_repair) if oral else (bbb_sampler, bbb_sampler_repair)
        if oral:
            edges, phases = sampler.sample(rows, calibration, caps)
        else:
            edges, phases = sampler.sample(rows, calibration, caps, fps=fps if policy['task'] == 'Skin_Reaction' else None)
        if len(edges) == 0:
            raise ValueError('No sampled training pairs')
        edges, report = repairer.repair(rows, edges, caps, 4096, solver_time_limit=120, separate_family_caps=True)
        edges = oral_trim(edges, rows) if oral else trim_binary(edges, rows)
        report.update(sampling_phases=phases, method=policy['record_sampling_policy'])
    raw = [{'a_id': rows[int(ref)]['record_id'], 'b_id': rows[int(query)]['record_id']} for query, ref, _ in edges]
    index = {row['record_id']: row for row in rows}
    output=[edge(index[p['a_id']],index[p['b_id']],'train','indirect',
                 DataStructs.TanimotoSimilarity(fps[index[p['a_id']]['parent_id']],fps[index[p['b_id']]['parent_id']]),
                 calibration[index[p['b_id']]['pair_bucket_key']],policy) for p in raw]
    numeric=defaultdict(set)
    for p in output:
        row=index[p['b_id']]
        if row['measurement_kind']!='binary':numeric[row['level']].add(p['near'])
    rejected={index[p['b_id']]['level'] for p in output if index[p['b_id']]['measurement_kind']=='binary'
              and numeric[index[p['b_id']]['level']]!={True,False}}
    if rejected and not smoke:raise ValueError(f'No numeric near/far population for binary levels: {sorted(rejected)}')
    dropped=sum(index[p['b_id']]['measurement_kind']=='binary' and index[p['b_id']]['level'] in rejected for p in output)
    output=[p for p in output if not (index[p['b_id']]['measurement_kind']=='binary' and index[p['b_id']]['level'] in rejected)]
    report.update(pairs=len(output),rejected_binary_levels=sorted(rejected),binary_target_pairs_rejected=dropped,
                  coverage=coverage(output,rows),caps=caps)
    return output,report


def coverage(pairs, rows):
    query, reference = Counter(p['b_id'] for p in pairs), Counter(p['a_id'] for p in pairs)
    return {'records': len(rows), 'query_records_used': sum(query[row['record_id']] > 0 for row in rows),
            'reference_records_used': sum(reference[row['record_id']] > 0 for row in rows),
            'query_maximum': max(query.values(), default=0), 'reference_maximum': max(reference.values(), default=0),
            'unused_record_ids': [r['record_id'] for r in rows if not query[r['record_id']] and not reference[r['record_id']]]}


def sampled(train, calibration, policy, component='record', limit=None, smoke=False):
    if not policy['evaluation_levels']:
        return sample_level(train,calibration,policy,component,limit,smoke)
    levels = defaultdict(list)
    for row in train:
        levels[row['level']].append(row)
    pairs, reports, rejected = [], {}, set()
    for level, rows in sorted(levels.items()):
        edges, report = sample_level(rows,calibration,policy,component,limit,smoke)
        pairs.extend(edges)
        reports[level] = report
        rejected.update(report['rejected_binary_levels'])
    return pairs, {'levels':reports,'rejected_binary_levels':sorted(rejected),'coverage':coverage(pairs,train)}

def direct_graph(train, calibration, policy):
    sd, center = calibration['calibration_sample_standard_deviation'], policy['common']['numeric_target']['center_sd']
    fps, directed, output = _fingerprints(train), set(), []
    def scarcity(query):
        classes = defaultdict(set)
        for row in train:
            if query['parent_id'] != row['parent_id']:
                classes[abs(row['geometry_value'] - query['geometry_value']) <= center * sd].add(row['parent_id'])
        return min(len(classes[True]), len(classes[False])), query['record_id']
    quota = policy['context_direct_degree_cap'] // 4
    for query in sorted(train, key=scarcity):
        choices = {True: {}, False: {}}
        for row in train:
            parent = row['parent_id']
            if parent == query['parent_id'] or (query['parent_id'], parent) in directed:
                continue
            near = abs(row['geometry_value'] - query['geometry_value']) <= center * sd
            item = (-DataStructs.TanimotoSimilarity(fps[query['parent_id']], fps[parent]), parent, row['record_id'], row)
            if parent not in choices[near] or item[:3] < choices[near][parent][:3]:
                choices[near][parent] = item
        buckets = {near: sorted(values.values()) for near, values in choices.items()}
        selected, seen = [], set()
        for near in sorted((True,False), key=lambda near: (len(buckets[near]), not near)):
            take = [item for item in buckets[near] if item[1] not in seen][:quota]
            selected.extend(take)
            seen.update(item[1] for item in take)
        for item in sorted(item for values in buckets.values() for item in values):
            if len(selected) == 2 * quota:
                break
            if item[1] not in seen:
                selected.append(item)
                seen.add(item[1])
        if len(selected) != 2 * quota:
            raise ValueError(f'Insufficient direct graph references: {query["record_id"]}')
        for similarity, parent, _, reference in selected:
            output.append(edge(reference, query, 'train', 'direct', -similarity, calibration, policy))
            directed.add((parent, query['parent_id']))
    return output


def sample_record(stage,pool,policy,smoke):
    from assay_transfer.evaluation import record_ranking
    records=read_parquet(pool/'records.parquet')
    calibration=read_json(pool/'calibration.json')['accepted_buckets']
    training=[r for r in records if r['dataset_split']=='train']
    gold=read_json(pool/'gold.json')
    limit=len(gold['train'])*(policy['context_direct_degree_cap']//2)
    pairs,report=sampled(training,calibration,policy,limit=limit,smoke=smoke)
    write_parquet(stage/'pool_train.parquet',pairs,PAIR_SCHEMA)
    ranks=record_ranking(records,calibration,report['rejected_binary_levels'],policy)
    counts={}
    for split in ('train','validation_ranking','test_ranking'):
        counts[split]=write_parquet(stage/f'{split}.parquet',[p for p in pairs+ranks if p['split']==split],PAIR_SCHEMA)
    write_json(stage/'sampling.json',report)
    write_json(stage/'calibration.json',read_json(pool/'calibration.json'))
    return {'scope':'smoke' if smoke else 'full','family':'record','pairs':counts,'coverage':report['coverage']}


def sample_context(stage,pool,policy,smoke):
    from assay_transfer.evaluation import direct_ranking
    from assay_transfer.filtering import gold_calibration
    groups=read_json(pool/'gold.json')
    calibration=gold_calibration(groups,pool/'gold.json')
    if smoke:
        groups={split:sorted(rows,key=lambda r:stable_sha256(r['record_id']))[:512 if split=='train' else 12]
                for split,rows in groups.items()}
    for split,rows in groups.items():
        for row in rows:row.update(measurement_kind='continuous',level='GOLD',dataset_split=split,task_id=policy['task'])
    direct=direct_graph(groups['train'],calibration,policy)
    ranks=direct_ranking(groups,calibration,policy)
    records={r['record_id']:r for rows in groups.values() for r in rows}
    candidates,report=[],{'rejected_binary_levels':[]}
    if policy['augmentation_ratio']>0:
        indirect={r['record_id']:r for r in read_parquet(pool/'indirect/records.parquet')}
        if records.keys() & indirect.keys():raise ValueError('Gold and indirect identities collide')
        records.update(indirect)
        training=[indirect[key] for key in read_json(pool/'indirect_training_ids.json')]
        calibration['indirect_calibration']=read_json(pool/'indirect/calibration.json')
        if training:
            candidates,report=sampled(training,calibration['indirect_calibration']['accepted_buckets'],policy,
                                      'context_indirect',limit=None,smoke=smoke)
        elif not smoke:raise ValueError('No eligible indirect augmentation records')
    all_train=direct+candidates
    write_parquet(stage/'pool_train.parquet',all_train,PAIR_SCHEMA)
    counts={}
    for split in ('train','validation_ranking','test_ranking'):
        counts[split]=write_parquet(stage/f'{split}.parquet',[p for p in all_train+ranks if p['split']==split],PAIR_SCHEMA)
    write_json(stage/'records.json',records)
    write_json(stage/'calibration.json',calibration)
    write_json(stage/'sampling.json',{'indirect_sampling':report,'rejected_binary_levels':report['rejected_binary_levels'],
               'gold_rows':{split:len(rows) for split,rows in groups.items()}})
    return {'scope':'smoke' if smoke else 'full','family':'context','pairs':counts}
