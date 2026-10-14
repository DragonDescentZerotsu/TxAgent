"""Filter source records and fit the geometry/variance policies on the retained pool."""
import json
import math
import statistics
from collections import Counter,defaultdict
import numpy as np
from .storage import digest,read_json,read_parquet,write_json,write_parquet,file_sha256
from .inputs import source_fields,raw_rows,identity,skin_routes,gold_groups
from .split import assign,filter_gold,filter_training,check_disjointness
from .targets import effective_value
from .dataset import fields
from .evaluation import support_reason,support_diagnostics,select_record_evaluation,choose_skin_ood

def project(files, policy, smoke=False):
    fields = source_fields(read_json(files['source_contract']['path']))
    output, rejected = [], Counter()
    routes = skin_routes(files) if 'skin_routes' in files else {}
    for raw in raw_rows(files, smoke):
        level = f'L{raw["mapped_level"]}' if raw['mapped_level'] is not None else None
        if routes or policy['slug'] == 'skin_reaction':
            key = raw['source_id'], raw['source_row_number'], raw['source_record_id'] or ''
            old = routes.get(key)
            level = level if level in ('L1','L2','L3') else (
                'L4' if level is None and old == 'L4' else
                'EXPOSURE' if level is None and old is None and raw['source_id'] == 'skin_exposure' else None)
        if level is None:
            rejected['missing_or_excluded_level'] += 1
            continue
        if policy['record_levels'] is not None and level not in policy['record_levels']:
            rejected['level_not_in_contract'] += 1
            continue
        try:
            row = project_row(raw, fields, policy, level)
        except (ValueError, TypeError) as error:
            rejected[str(error)] += 1
            continue
        output.append(row)
    if len({row['record_id'] for row in output}) != len(output):
        raise ValueError('Duplicate source-row projection identity')
    return output, {'scope': 'smoke' if smoke else 'full', 'projected_records': len(output),
                    'rejections': dict(rejected), 'levels': dict(Counter(row['level'] for row in output))}


def project_row(raw, fields, policy, level):
    kind = raw['measurement_kind']
    if kind not in ('continuous', 'binary', 'ordinal'):
        raise ValueError('unsupported_measurement_kind')
    field = 'canonical_category_rank' if kind in ('binary', 'ordinal') else 'finite_scalar_value'
    value = float(raw[field])
    if not math.isfinite(value):
        raise ValueError('nonfinite_geometry')
    parent, smiles, scaffold = identity(raw['canonical_smiles'])
    payload = {field: raw.get(field) for field in fields[raw['source_id']]}
    bucket = json.dumps([*json.loads(raw['pair_bucket_key']), level], separators=(',', ':'))
    return {'record_id': f'{policy["slug"]}:{raw["canonical_record_id"]}', 'task_id': policy['task'],
            'source_row_uid': raw['source_row_uid'], 'source_id': raw['source_id'], 'level': level,
            'parent_id': parent, 'parent_smiles': smiles, 'scaffold': scaffold, 'pair_bucket_key': bucket,
            'upstream_pair_bucket_key': raw['pair_bucket_key'], 'geometry_value': value, 'measurement_kind': kind,
            'category_id': str(raw.get('canonical_category_id') or ''), 'endpoint': raw['canonical_endpoint_name'],
            'canonical_measurement_text': str(raw.get('canonical_measurement_text') or ''),
            'canonical_unit_text': str(raw.get('canonical_unit_text') or ''),
            'canonical_measurement_scale_id': str(raw.get('canonical_measurement_scale_id') or ''),
            'prompt_payload_json': json.dumps(payload, ensure_ascii=False, default=str),
            'pair_fields_json': raw['canonical_pair_fields_json'], 'dataset_split': '',
            'query_validation': False, 'query_test': False, 'evaluation_regime': ''}

def ratio_diagnostics(rows, calibration):
    groups = defaultdict(list)
    for row in rows:
        groups[row['parent_id']].append(effective_value(row, calibration))
    means = [statistics.fmean(values) for values in groups.values()]
    degrees = len(rows) - len(groups)
    squared = sum(sum((value - statistics.fmean(values)) ** 2 for value in values) for values in groups.values())
    within = squared / degrees if degrees > 0 else None
    between = statistics.variance(means) if len(means) > 1 else None
    ratio = within / between if within is not None and between and between > 0 else None
    return {'within_to_parent_mean_variance_ratio': ratio, 'pooled_within_parent_variance': within,
            'unweighted_parent_mean_variance': between, 'ratio_gate_record_count': len(rows),
            'ratio_gate_parent_count': len(groups), 'ratio_gate_within_df': degrees}


def domain(rows, strict_ordinal=False):
    kinds = {row['measurement_kind'] for row in rows}
    if len(kinds) != 1:
        return None, 'mixed_measurement_kinds'
    kind = next(iter(kinds))
    entry = {'measurement_kind': kind, 'train_record_count': len(rows),
             'train_parent_count': len({row['parent_id'] for row in rows})}
    if kind == 'binary':
        counts = Counter(row['category_id'] for row in rows)
        if len(counts) != 2 or min(counts.values()) < 3:
            return None, 'incomplete_or_sparse_binary_domain'
        return {**entry, 'category_counts': dict(sorted(counts.items())),
                'calibration_sample_standard_deviation': None}, None
    if len(rows) < 2:
        return None, 'fewer_than_2_records'
    sd = statistics.stdev(float(row['geometry_value']) for row in rows)
    if not math.isfinite(sd) or sd <= 0:
        return None, 'invalid_sample_standard_deviation'
    if kind == 'ordinal' and strict_ordinal:
        counts = Counter(row['category_id'] for row in rows)
        if len(counts) < 3 or min(counts.values()) < 3:
            return None, 'incomplete_or_sparse_ordinal_domain'
    return {**entry, 'calibration_sample_standard_deviation': sd,
            'calibration_standard_deviation_ddof': 1}, None


def sd_inflation(values, multiplier=3):
    q1, q3 = np.percentile(values, [25, 75])
    width = q3 - q1
    retained = [value for value in values if q1 - multiplier * width <= value <= q3 + multiplier * width]
    full = statistics.stdev(values) if len(values) > 1 else 0
    trimmed = statistics.stdev(retained) if len(retained) > 1 else 0
    return full / trimmed if trimmed > 0 else (float('inf') if full > 0 else 1)

def conflicts(rows):
    surfaces, categories = {}, defaultdict(set)
    for row in rows:
        if row['measurement_kind'] not in ('binary', 'ordinal'):
            continue
        key = (row['dataset_split'] == 'test_gate_only', row['pair_bucket_key'], row['parent_id'],
               json.dumps(fields(row), ensure_ascii=False))
        surfaces[row['record_id']] = key
        categories[key].add(row['category_id'])
    bad = {key for key, values in categories.items() if len(values) > 1}
    return ([row for row in rows if surfaces.get(row['record_id']) not in bad],
            [row for row in rows if surfaces.get(row['record_id']) in bad])


def geometry(fit, entry, policy, reviewed):
    if entry['measurement_kind'] == 'binary':
        return entry, None
    values = [row['geometry_value'] for row in fit]
    inflation = sd_inflation(values)
    task, bucket = policy['task'], fit[0]['pair_bucket_key']
    upstream = fit[0]['upstream_pair_bucket_key']
    alert = len(values) >= 20 and inflation >= 2
    transform, reason = 'identity', None
    if task == 'BBB_Martins':
        transform = 'log10' if upstream in reviewed['interventions'] else 'identity'
        if alert and upstream not in reviewed['interventions'] and upstream not in reviewed['reviewed_no_intervention']:
            reason = 'unreviewed_geometry_alert'
    elif task == 'Bioavailability_Ma':
        decision = reviewed['decisions'].get(upstream)
        transform = decision['transform'] if decision else 'identity'
        if alert and not decision:
            reason = 'unreviewed_geometry_alert'
    elif task == 'Skin_Reaction':
        decision = reviewed['decisions'].get(bucket)
        if alert and (not decision or decision['transform'] != 'identity'):
            reason = 'unreviewed_geometry_alert'
    elif alert:
        decision = reviewed['bucket_overrides'].get(bucket)
        transform = decision['transform'] if decision else (
            'identity' if fit[0]['canonical_unit_text'] in reviewed['identity_units'] else
            reviewed['default_positive_transform'] if min(values) > 0 else 'unreviewed')
    if transform == 'reject':
        reason = 'reviewed_geometry_reject'
    if transform == 'unreviewed':
        reason = 'nonpositive_unreviewed_geometry_alert'
    if transform == 'log10' and min(values) <= 0:
        reason = 'nonpositive_log10_value'
    if reason:
        return None, reason
    if transform not in ('identity', 'log10'):
        raise ValueError(f'Unknown geometry transform: {transform}')
    effective = [math.log10(value) for value in values] if transform == 'log10' else values
    sd, post = statistics.stdev(effective), sd_inflation(effective)
    if not math.isfinite(sd) or sd <= 0 or transform == 'log10' and post >= 2:
        return None, 'invalid_effective_geometry_or_tail_gate'
    entry.update(logit_geometry_transform=transform, logit_calibration_sample_standard_deviation=sd,
                 logit_geometry_sd_inflation=post if math.isfinite(post) else None,
                 raw_geometry_alert=bool(alert))
    return entry, None


def fit_bucket(evidence, policy, reviewed, ood=False):
    non_test = [row for row in evidence if row['dataset_split'] not in ('test_gate_only', 'gate_only')]
    train = [row for row in non_test if row['dataset_split'] == 'train']
    reason = support_reason(non_test, train, ood)
    bounds = policy['ood_bucket_records']
    if ood and not bounds[0] <= len(non_test) <= bounds[1]:
        reason = 'ood_size_outside_contract'
    safety = not policy['evaluation_levels']
    numeric = evidence[0]['measurement_kind'] != 'binary'
    fit = non_test if ood or policy['task'] == 'Skin_Reaction' or (numeric and not safety) else train
    if reason:
        return None, reason
    entry, reason = domain(fit, strict_ordinal=safety)
    if entry is None:
        return None, reason
    entry, reason = geometry(fit, entry, policy, reviewed)
    if entry is None:
        return None, reason
    gate = non_test if policy['variance_gate_fit'] == 'non_test_records' else evidence
    if entry.get('logit_geometry_transform') == 'log10' and any(row['geometry_value'] <= 0 for row in gate):
        return None, 'nonpositive_gate_geometry'
    diagnostics = ratio_diagnostics(gate, entry)
    ratio = diagnostics['within_to_parent_mean_variance_ratio']
    if ratio is None or not math.isfinite(ratio) or ratio > policy['variance_ratio_maximum']:
        return None, 'undefined_variance_ratio' if ratio is None else 'variance_ratio_above_contract'
    entry.update(diagnostics, **support_diagnostics(non_test, train),
                 numeric_center_sd=policy['common']['numeric_target']['center_sd'],
                 ratio_gate_maximum=policy['variance_ratio_maximum'], ratio_gate_fit=policy['variance_gate_fit'],
                 calibration_scope='whole_non_test_bucket' if ood else policy['calibration_policy'],
                 calibration_fit_record_ids=sorted(row['record_id'] for row in fit),
                 ratio_gate_record_ids=sorted(row['record_id'] for row in gate),
                 level=evidence[0]['level'])
    return entry, None


def calibrate(rows, policy, reviewed, original_sizes, frozen_ood=None):
    groups = defaultdict(list)
    for row in rows:
        groups[row['pair_bucket_key']].append(row)
    candidates, rejected, accepted = {}, {}, {}
    for bucket, members in sorted(groups.items()):
        modes = (False, True) if policy['task'] == 'Skin_Reaction' and members[0]['level'] in policy['evaluation_levels'] else (False,)
        for ood in modes:
            entry, reason = fit_bucket(members, policy, reviewed, ood)
            if entry is not None:
                candidates[bucket, ood] = entry
            else:
                rejected[f'{bucket}/{ood}'] = {'reason': reason}
    selection = {}
    selected = set()
    if frozen_ood is not None:
        selected = set(frozen_ood)
    elif policy['task'] == 'Skin_Reaction':
        selected, selection = choose_skin_ood(rows, candidates, policy['evaluation_levels'], policy['ood_bucket_records'])
    elif policy['ood_bucket_records']:
        lower, upper = policy['ood_bucket_records']
        selected = {bucket for bucket in groups if lower <= original_sizes[bucket] <= upper}
    selection['selected_buckets'] = sorted(selected)
    for bucket in selected:
        if (bucket, True) not in candidates and bucket in groups:
            entry, reason = fit_bucket(groups[bucket], policy, reviewed, True)
            if entry is not None:
                candidates[bucket, True] = entry
            else:
                rejected[f'{bucket}/True'] = {'reason': reason}
    retained, gate = [], []
    for bucket, members in sorted(groups.items()):
        ood = bucket in selected
        entry = candidates.get((bucket, ood))
        if entry is None:
            continue
        accepted[bucket] = entry
        for row in members:
            if row['dataset_split'] in ('test_gate_only', 'gate_only'):
                gate.append(row)
                continue
            row.update(evaluation_regime='ood' if ood else 'id', **{key: entry[key] for key in ('bucket_total_record_count', 'bucket_train_record_count',
                'bucket_total_parent_count', 'bucket_train_parent_count')})
            if ood:
                row['dataset_split'] = 'ood'
            retained.append(row)
    return retained, gate, accepted, rejected, selection


def prepare_record_pool(stage, files, policy, smoke, record_evaluation=True):
    stage.mkdir(parents=True, exist_ok=True)
    gold, gold_excluded = filter_gold(gold_groups(files), policy)
    rows, projection = project(files, policy, smoke)
    assignment = assign(rows, gold, policy)
    original = Counter(r['pair_bucket_key'] for r in rows if r['dataset_split'] not in ('test_gate_only','gate_only'))
    heldout = gold['valid'] + gold['test'] + [row for row in rows if row['dataset_split'] != 'train']
    rows, dropped = conflicts(rows)
    reviewed = read_json(files['geometry_policy']['path'])
    _, _, _, _, initial_selection = calibrate([dict(row) for row in rows], policy, reviewed, original)
    frozen = initial_selection['selected_buckets']
    for row in rows:
        if row['pair_bucket_key'] in frozen and row['dataset_split'] == 'train':
            row['dataset_split'] = 'ood'
    heldout += [row for row in rows if row['dataset_split'] == 'ood']
    rows, exclusions = filter_training(rows, heldout, policy)
    rows, gate, accepted, rejected, selection = calibrate(rows, policy, reviewed, original, frozen)
    for entry in accepted.values():
        entry['source_manifest_sha256'] = files['stage3_manifest']['sha256']
        entry['calibration_sha256'] = digest(entry)
    test, panels = select_record_evaluation(rows, gate, accepted, policy) if record_evaluation else ([], {})
    if record_evaluation and not smoke and policy['task']=='Bioavailability_Ma' and any(p['queries']!=500 or p['selected_buckets']!=p['eligible_buckets'] for p in panels.values()):
        raise ValueError('Full Oral panels require 500 queries and coverage of every eligible bucket')
    if not any(row['dataset_split'] == 'train' for row in rows):
        raise ValueError(f'No eligible training records: {policy["task"]}; see rejection policies')
    write_parquet(stage / 'records.parquet', rows + test)
    write_parquet(stage / 'gate_records.parquet', gate)
    write_parquet(stage / 'conflicts.parquet', dropped)
    write_json(stage / 'gold.json', gold)
    write_json(stage / 'gold_exclusions.json', gold_excluded)
    write_parquet(stage / 'disjointness_exclusions.parquet', exclusions)
    write_json(stage / 'heldout_identities.json', [{key:row.get(key) for key in ('record_id','parent_id','scaffold','level')} for row in heldout])
    write_json(stage / 'calibration.json', {'accepted_buckets': accepted, 'rejected_buckets': rejected})
    audit = {'projection': projection, 'initial_assignment': assignment, 'panels': panels,
             'disjointness': check_disjointness([r for r in rows if r['dataset_split']=='train'], heldout, policy),
             'training_exclusions': len(exclusions), 'gold_training_exclusions': len(gold_excluded),
             'ood_selection': selection, 'conflict_count': len(dropped), 'test_target_fit': False,
             'accepted_buckets': len(accepted), 'rejected_regimes': len(rejected),
             'records': dict(Counter(f'{r["level"]}/{r["dataset_split"]}' for r in rows + test)),
             'scope': 'smoke' if smoke else 'full', 'holdout_policy': policy['holdout_policy']}
    write_json(stage / 'split_audit.json', audit)
    write_json(stage / 'effective_contract.json', policy)
    return audit

def gold_calibration(groups, source):
    fit = groups['train'] + groups['valid']
    sd = statistics.stdev(row['geometry_value'] for row in fit)
    if sd <= 0:
        raise ValueError('Retained Gold train/validation has zero target standard deviation')
    calibration = {'calibration_sample_standard_deviation':sd,
                   'calibration_scope':'retained_gold_train_plus_validation',
                   'source_manifest_sha256':file_sha256(source),
                   'fit_record_ids':sorted(row['record_id'] for row in fit)}
    return {**calibration, 'calibration_sha256':digest(calibration)}


def prepare_context_pool(stage,files,policy,smoke):
    groups,excluded=filter_gold(gold_groups(files),policy)
    stage.mkdir(parents=True,exist_ok=True)
    write_json(stage/'gold.json',groups)
    write_json(stage/'gold_exclusions.json',excluded)
    audit=check_disjointness(groups['train'],groups['valid']+groups['test'],policy)
    if policy['augmentation_ratio']>0:
        audit['indirect']=prepare_record_pool(stage/'indirect',files,policy,smoke,record_evaluation=False)
        training=[r for r in read_parquet(stage/'indirect/records.parquet') if r['dataset_split']=='train']
        task=policy['task']
        if task=='BBB_Martins':
            training=[r for r in training if r['measurement_kind']!='binary' and r['source_id'] in
                      ('efflux_transport','influx_transport','passive_permeability')]
        elif task=='Bioavailability_Ma':
            training=[r for r in training if r['measurement_kind']!='binary' and r['source_id']!='hf_bioavailability']
        elif task=='Skin_Reaction':
            training=[r for r in training if r['level'] in ('L1','L2','L3')]
        write_json(stage/'indirect_training_ids.json',sorted(r['record_id'] for r in training))
        audit['eligible_indirect_training_records']=len(training)
    write_json(stage/'split_audit.json',{**audit,'gold_training_exclusions':len(excluded)})
    write_json(stage/'effective_contract.json',policy)
    return {'scope':'smoke' if smoke else 'full','family':'context','disjointness':audit}
