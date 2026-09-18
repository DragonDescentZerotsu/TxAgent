"""Percentage-only routing overlay and provenance-checked prior-assignment reuse."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.evidence_library.versions.v10.measurement_routing import RouteDecision

RULE = 'previously_accepted_percentage_review.v1'
PERCENT = re.compile(r'[%％]|\bpercent(?:age)?\b|\bper\s+cent\b', re.I)


def review_percentage(record, decision):
    if decision.bucket == 'accept' and any(PERCENT.search(str(record.get(k) or '')) for k in ('measurement_text', 'unit_text')):
        return RouteDecision('extract')
    return decision


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def prepare(task, output, mappings=(), caches=()):
    from data.processing.evidence_library.versions.v10.build_measurement_resolution_mapping import TaskConfig
    from data.processing.evidence_library.versions.v10.build_reference_semantics_mapping import SubmissionCache

    output = Path(output)
    if output.exists():
        raise ValueError(f'Preparation output already exists: {output}; reuse its manifest, do not overwrite')
    config = TaskConfig(task)
    source = Path('data/evidence_libraries') / task / 'v9/01_cleaned/records.parquet'
    source_hash = digest(source)
    table = pq.read_table(source)
    records = table.to_pylist()
    by_id = {r['cleaned_record_id']: r for r in records}
    if len(by_id) != len(records) or len({r['source_row_uid'] for r in records}) != len(records):
        raise ValueError('Source identities are not unique')
    delta = []
    for row in records:
        prior = row['measurement_resolution_route']
        new = review_percentage(row, RouteDecision(
            prior, row['measurement_resolution_rule_id'],
            row['measurement_resolution_exact_measurement'], row['measurement_resolution_exact_unit'],
            bool(row['measurement_resolution_exact_unit_is_canonical'])))
        if new.bucket == prior:
            continue
        row = dict(row)
        row.update(previous_measurement_resolution_route=prior,
                   previous_measurement_resolution_rule_id=row['measurement_resolution_rule_id'],
                   measurement_resolution_route='extract', measurement_resolution_rule_id=None,
                   measurement_resolution_exact_measurement=None, measurement_resolution_exact_unit=None,
                   measurement_resolution_exact_unit_is_canonical=False,
                   percentage_routing_version=RULE)
        delta.append(row)
    delta_ids = {r['cleaned_record_id'] for r in delta}
    reused = {}
    receipts = []
    # Caller supplies mappings in priority order. Overlap is retained in original
    # artifacts, but never generates a second current assignment or another call.
    for mapping in mappings:
        mapping = Path(mapping)
        manifest = json.loads(mapping.with_suffix('.manifest.json').read_text())
        if digest(mapping) != manifest['mapping_sha256']:
            raise ValueError(f'Mapping hash mismatch: {mapping}')
        prior_source = Path(manifest['cleaned_records_path'])
        if digest(prior_source) != manifest['cleaned_records_sha256']:
            raise ValueError(f'Prior source hash mismatch: {prior_source}')
        fields = ['source_id','endpoint_name','measurement_text','unit_text','support_text']
        old = {r['cleaned_record_id']: r for r in pq.read_table(prior_source, columns=['cleaned_record_id', *fields]).to_pylist()}
        count = 0
        for assignment in pq.read_table(mapping).to_pylist():
            key = assignment['cleaned_record_id']
            if key not in by_id or key in delta_ids or key in reused:
                continue
            row = by_id[key]
            if row['measurement_resolution_route'] != 'extract':
                continue
            if key not in old or any(row.get(f) != old[key].get(f) for f in fields):
                raise ValueError(f'Reuse input mismatch: {key}')
            if assignment.get('source_row_uid') not in (None, row['source_row_uid']):
                raise ValueError(f'Reuse UID mismatch: {key}')
            if not str(assignment.get('assignment_method','')).startswith('model_'):
                continue
            assignment.update(source_row_uid=row['source_row_uid'], reuse_mapping_path=str(mapping),
                              reuse_mapping_sha256=manifest['mapping_sha256'], assignment_origin='v9_reused')
            reused[key] = assignment
            count += 1
        receipts.append({'mapping':str(mapping), 'sha256':manifest['mapping_sha256'], 'reused_rows':count})
    for cache_path in caches:
        cache_path = Path(cache_path)
        contract = json.loads((cache_path.parent/'input_contract.json').read_text())
        # The cache pins the current source byte-for-byte; no semantic fuzzy join.
        if contract.get('records_sha256') != source_hash or contract.get('task') != task:
            raise ValueError(f'Cache does not pin current source: {cache_path}')
        cache = SubmissionCache(cache_path)
        count = 0
        for key, assignment in cache.assignments.items():
            if key in reused or key in delta_ids or key not in by_id:
                continue
            if by_id[key]['measurement_resolution_route'] != 'extract' or not str(assignment.get('assignment_method','')).startswith('model_'):
                continue
            assignment = dict(assignment)
            assignment.update(cache.provenance.get(key, {}))
            assignment.update(source_row_uid=by_id[key]['source_row_uid'], reuse_mapping_path=str(cache_path),
                              reuse_mapping_sha256=digest(cache_path), assignment_origin='v9_reused')
            reused[key] = assignment
            count += 1
        receipts.append({'cache':str(cache_path), 'sha256':digest(cache_path), 'reused_rows':count})
    excluded = set()
    for fixture in Path('tests/chembl_tool/common/measurement_resolution_quality/gold').glob(task+'.*.jsonl'):
        for line in fixture.read_text().splitlines()[1:]:
            case = json.loads(line)
            excluded.add(case['audit_case_id'].split(':',1)[-1])
    grouped = defaultdict(list)
    for row in delta:
        if row['cleaned_record_id'] not in excluded:
            grouped[row['source_id']].append(row)
    rng = random.Random(20260905)
    for rows in grouped.values():
        rows.sort(key=lambda r:r['source_row_uid'])
        rng.shuffle(rows)
    pilot = []
    while len(pilot) < min(100,sum(len(rows) for rows in grouped.values())+len(pilot)):
        for key in sorted(grouped):
            if grouped[key] and len(pilot)<100:
                pilot.append(grouped[key].pop())
    output.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(delta),output/'delta_records.parquet')
    if reused:
        # Union fields before Arrow inference so provenance is not lost across models.
        keys = set().union(*(r.keys() for r in reused.values()))
        pq.write_table(pa.Table.from_pylist([{k:r.get(k) for k in keys} for r in reused.values()]),output/'reused_assignments.parquet')
    inputs=[]
    for ordinal,row in enumerate(pilot,1):
        inputs.append({'audit_case_id':task+':'+row['cleaned_record_id'], 'source_id':row['source_id'],
                       'source_row_uid':row['source_row_uid'], 'canonical_endpoint_name':row['canonical_endpoint_name'],
                       'route_bucket':'extract', 'review_ordinal':ordinal,
                       'input':{f:row.get(f) for f in config.prompt_row_fields(row['source_id'])}})
    # This is an active annotation input until consolidated into reviewed gold.
    (output/'pilot_inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in inputs))
    manifest={'task_id':task, 'source_path':str(source), 'source_sha256':source_hash,
              'routing_policy':RULE, 'source_rows':len(records), 'new_percentage_rows':len(delta),
              'delta_sha256':digest(output/'delta_records.parquet'),
              'delta_by_source':dict(Counter(r['source_id'] for r in delta)),
              'prior_route_counts':dict(Counter(r['measurement_resolution_route'] for r in records)),
              'reused_rows':len(reused),
              'prior_extraction_pending':sum(r['measurement_resolution_route']=='extract' and r['cleaned_record_id'] not in reused for r in records),
              'reuse_receipts':receipts,'pilot_rows':len(pilot), 'pilot_seed':20260905,
              'existing_gold_delta_rows_excluded':len(delta_ids & excluded),
              'full_delta_launched':False,'published':False}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


def merge_pilot(output, mapping):
    """Combine verified old assignments with completed delta calls, without inference."""
    output, mapping = Path(output), Path(mapping)
    manifest = json.loads((output/'manifest.json').read_text())
    pilot_manifest = json.loads(mapping.with_suffix('.manifest.json').read_text())
    if digest(mapping) != pilot_manifest['mapping_sha256'] or pilot_manifest['cleaned_records_sha256'] != manifest['delta_sha256']:
        raise ValueError('Pilot mapping or source hash mismatch')
    delta = {r['cleaned_record_id']:r['source_row_uid'] for r in pq.read_table(output/'delta_records.parquet',columns=['cleaned_record_id','source_row_uid']).to_pylist()}
    old = pq.read_table(output/'reused_assignments.parquet').to_pylist()
    new = pq.read_table(mapping).to_pylist()
    old_ids = {r['cleaned_record_id'] for r in old}
    for row in new:
        key = row['cleaned_record_id']
        if key in old_ids or key not in delta or row.get('source_row_uid') != delta[key]:
            raise ValueError(f'Pilot identity overlap or mismatch: {key}')
        if not str(row.get('assignment_method','')).startswith('model_'):
            raise ValueError(f'Pilot has an unresolved technical failure: {key}')
        row['assignment_origin'] = 'v10_percentage_pilot'
    combined = old + new
    if len({r['cleaned_record_id'] for r in combined}) != len(combined):
        raise ValueError('Duplicate combined assignment IDs')
    keys = sorted(set().union(*(r.keys() for r in combined)))
    target = output/'partial_measurement_resolution.parquet'
    if target.exists():
        raise ValueError(f'Partial mapping already exists: {target}')
    pq.write_table(pa.Table.from_pylist([{k:r.get(k) for k in keys} for r in combined]),target)
    receipt = {'task_id':manifest['task_id'], 'partial':True, 'published':False,
               'reused_rows':len(old), 'new_rows':len(new),
               'prior_extraction_pending':manifest['prior_extraction_pending'],
               'percentage_extraction_pending':len(delta)-len(new),
               'mapping_sha256':digest(target), 'reuse_sha256':digest(output/'reused_assignments.parquet'),
               'pilot_mapping_sha256':digest(mapping), 'source_sha256':manifest['source_sha256'],
               'reuse_policy':'Preserve valid previous assignments, including prior prompt versions; no retrospective re-extraction.'}
    target.with_suffix('.manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,choices=['bbb_martins','bioavailability_ma'])
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--prior-mapping',action='append',default=[],type=Path)
    parser.add_argument('--prior-cache',action='append',default=[],type=Path)
    args=parser.parse_args()
    prepare(args.task,args.output,args.prior_mapping,args.prior_cache)
