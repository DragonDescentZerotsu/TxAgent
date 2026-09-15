"""Stage source-vote membership, identity holds and exact-record deduplication."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tools.chembl_tool.common.json_utils import atomic_output_path, read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic
from tools.chembl_tool.common.starling.source_gold_review import payload_hash

TASKS = {'dili': 'DILI', 'carcinogens': 'Carcinogens'}
DERIVED = {'group_id', 'retrieval_eligible', 'is_gold_voter', 'progressive_level',
           'family_key', 'level_assignment_reason', 'heldout_filter_scope'}


def exact_duplicate_key(row):
    """Ignore extraction coordinates only; retain every scientific field and PMID."""
    raw = json.loads(row['raw_record_json'])
    for key in ('source_row_uid', 'extraction_id', 'paragraph_idx'):
        raw.pop(key, None)
    return hashlib.sha256(json.dumps(
        [raw, row.get('canonical_smiles'), row.get('identity_review_reason')],
        sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def exact_duplicate_plan(parquet, voters):
    """Prefer the actual voter; never silently withdraw a second gold voter."""
    seen, groups = {}, {}
    columns = [c for c in ('raw_record_json', 'source_row_uid', 'canonical_smiles',
                          'identity_review_reason') if c in parquet.schema_arrow.names]
    for batch in parquet.iter_batches(batch_size=32768, columns=columns):
        for row in batch.to_pylist():
            key, uid = exact_duplicate_key(row), row['source_row_uid']
            if key not in seen:
                seen[key] = uid
                continue
            previous = seen[key]
            if previous in voters and uid in voters:
                raise ValueError('Identical records are both gold voters: '+previous+', '+uid)
            groups.setdefault(key, [previous]).append(uid)
            if uid in voters:
                seen[key] = uid
    return {uid: {'source_row_uid': uid, 'retained_source_row_uid': seen[key],
                  'exact_content_sha256': key}
            for key, uids in groups.items() for uid in uids if uid != seen[key]}


def stage_heldout_inputs(task, benchmark, output):
    """Add explicit leakage metadata to an identical heldout query cohort."""
    from tools.chembl_tool.common.starling.new_task_identity import leakage_identity, VERSION
    benchmark,output=Path(benchmark),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    receipt={}
    for scheme in ('scaffold','random'):
        source=benchmark/scheme/'heldout_molecule_condition_labels.jsonl'
        rows=read_jsonl(source);staged=[]
        for row in rows:
            identity=leakage_identity(task,row['drug'])
            if row.get('leakage_group') and row['leakage_group']!=identity['leakage_group']:
                raise ValueError('Existing heldout identity conflicts with adapter')
            staged.append({**{k:row[k] for k in ('drug','Y','molecule_identity_key','benchmark_row_id','condition_group')},
                'leakage_group':identity['leakage_group'],'scaffold_leakage_group':identity['scaffold_leakage_group'],
                'identity_contract':VERSION})
        assert [(r['drug'],r['Y'],r['benchmark_row_id']) for r in rows]==[(r['drug'],r['Y'],r['benchmark_row_id']) for r in staged]
        path=output/(scheme+'.jsonl');write_jsonl_atomic(path,staged)
        receipt[scheme]={'source':str(source),'source_sha256':sha256_file(source),'path':str(path),
            'sha256':sha256_file(path),'rows':len(staged),'ordered_query_labels_unchanged':True}
    write_json_atomic(output/'receipt.json',receipt)
    return receipt


def refresh_membership(task, source, output, votes_path, *, deduplicate_exact=False, identity_holds_path=None, identity_clearances_path=None):
    """Refresh exact source-vote membership without replaying mechanism rules.

    Unlike the historical withdrawal-only repair, this explicitly accepts a new
    source release. Duplicate support UIDs remain nonvoters. Every changed voter
    is bound to its original payload hash; identity holds cannot be bypassed.
    """
    from tools.chembl_tool.common.build_runtime import local_input, local_workdir, publish_file
    source,output,votes_path=Path(source),Path(output),Path(votes_path)
    if source.resolve()==output.resolve():raise ValueError('Use a separate release directory')
    output.mkdir(parents=True,exist_ok=True)
    votes=read_jsonl(votes_path)
    voters={r['source_record_id']:r for r in votes}
    if len(voters)!=len(votes):raise ValueError('Duplicate representative UID')
    holds = {r['source_row_uid']: r for r in read_jsonl(Path(identity_holds_path))} if identity_holds_path else {}
    clearances = {r['source_row_uid']: r for r in read_jsonl(Path(identity_clearances_path))} if identity_clearances_path else {}
    if set(holds)&set(clearances):raise ValueError('Identity hold and clearance overlap')
    if set(holds) & set(voters):raise ValueError('Identity-held record cannot be a voter')
    matched_holds = set()
    matched_clearances = set()
    policy=importlib.import_module(f'tools.chembl_tool.tasks.{task}.starling_levels')
    source_path=source/'records.parquet'
    inputs={str(p):sha256_file(p) for p in (source_path,votes_path,Path(__file__))}
    counts=Counter();transitions=Counter();receipts=[];matched=set();old_voters=set();newly_eligible_smiles=set()
    changed_derived=DERIVED|{'direct_signal'}
    if holds or clearances:
        for path in (identity_holds_path,identity_clearances_path):
            if path:inputs[str(path)] = sha256_file(Path(path))
        changed_derived |= {'identity_review_reason'}
    with local_workdir() as staging:
        local=local_input(source_path)
        parquet=pq.ParquetFile(local)
        if clearances and 'identity_verification' in parquet.schema_arrow.names:
            changed_derived.add('identity_verification')
        duplicates=exact_duplicate_plan(parquet, voters) if deduplicate_exact else {}
        with pq.ParquetWriter(staging/'records.parquet',parquet.schema_arrow,compression='zstd',compression_level=3) as writer:
            for batch in parquet.iter_batches(batch_size=32768):
                table=pa.Table.from_batches([batch]); original=table
                uids=table['source_row_uid'].to_pylist()
                levels=table['progressive_level'].to_pylist()
                changed=[i for i,uid in enumerate(uids) if uid in voters or uid in holds or uid in clearances or levels[i]==1]
                if changed:
                    data={k:table[k].to_pylist() for k in changed_derived}
                    for i in changed:
                        uid=uids[i];old=levels[i];is_voter=uid in voters
                        if old==1:old_voters.add(uid)
                        if uid in clearances:
                            clearance=clearances[uid]
                            if table['identity_review_reason'][i].as_py()!=clearance['previous_reason']:
                                raise ValueError('Identity clearance previous state changed: '+uid)
                            if table['canonical_smiles'][i].as_py()!=clearance['canonical_smiles'] or payload_hash(json.loads(table['raw_record_json'][i].as_py()))!=clearance['source_payload_sha256']:
                                raise ValueError('Stale identity clearance payload or structure: '+uid)
                            data['identity_review_reason'][i]='';matched_clearances.add(uid)
                            if 'identity_verification' in data:data['identity_verification'][i]='hash_bound_reviewed_exact_parent_synonyms'
                        if is_voter:
                            matched.add(uid)
                            if table['identity_review_reason'][i].as_py() and uid not in clearances:raise ValueError('Voter has unresolved source identity: '+uid)
                            raw_hash=payload_hash(json.loads(table['raw_record_json'][i].as_py()))
                            if raw_hash!=voters[uid]['source_payload_sha256']:raise ValueError('Stale voter raw payload: '+uid)
                        level=1 if is_voter else 2
                        reason='actual_source_vote_membership:'+votes_path.parent.name if is_voter else 'nonvoter_direct_after_source_vote_refresh'
                        values={'group_id':'Group.'+policy.FAMILIES[level],'retrieval_eligible':True,
                            'is_gold_voter':is_voter,'progressive_level':level,'family_key':policy.FAMILIES[level],
                            'level_assignment_reason':reason,'heldout_filter_scope':'direct_outcome','direct_signal':True}
                        if uid in holds:
                            hold = holds[uid]
                            if payload_hash(json.loads(table['raw_record_json'][i].as_py())) != hold['source_payload_sha256']:
                                raise ValueError('Stale source identity hold: '+uid)
                            matched_holds.add(uid); level=0; reason=hold['reason']
                            values.update(group_id='',retrieval_eligible=False,is_gold_voter=False,progressive_level=0,
                                family_key='',level_assignment_reason=reason,heldout_filter_scope='',identity_review_reason=reason)
                        if values['retrieval_eligible'] and not original['retrieval_eligible'][i].as_py():
                            newly_eligible_smiles.add(table['canonical_smiles'][i].as_py())
                        for k,value in values.items():data[k][i]=value
                        if old!=level:receipts.append({'source_row_uid':uid,'previous_level':old,'level':level,'reason':reason})
                        transitions[f'{old}->{level}']+=1
                    for k,values in data.items():
                        pos=table.schema.get_field_index(k)
                        table=table.set_column(pos,table.schema.field(k),pa.array(values,type=table.schema.field(k).type))
                    assert all(table[k].equals(original[k]) for k in table.column_names if k not in changed_derived)
                if duplicates:
                    table=table.filter(pa.array([uid not in duplicates for uid in uids]))
                counts.update(table['progressive_level'].to_pylist());writer.write_table(table)
        assert matched==set(voters) and counts[1]==len(voters)
        assert matched_holds == set(holds)
        assert matched_clearances == set(clearances)
        assert sum(counts.values())+len(duplicates)==parquet.metadata.num_rows
        publish_file(staging/'records.parquet',output/'records.parquet')
    write_jsonl_atomic(output/'membership_changes.jsonl',receipts)
    write_jsonl_atomic(output/'exact_duplicate_removals.jsonl',duplicates.values())
    write_jsonl_atomic(output/'actual_voter_membership.jsonl',(
        {'source_record_id':uid,'study_representative_uid':uid,'study_id':r['study_id'],
         'molecule_identity_key':r['molecule_identity_key'],'condition_group':r['condition_group'],'Y':r['Y'],
         'source_payload_sha256':r['source_payload_sha256']} for uid,r in sorted(voters.items())))
    # The linkage table does not confer voter status on supporting duplicates.
    links=[{'source_row_uid':uid,'representative_uid':r['source_record_id'],
            'is_gold_voter':uid==r['source_record_id'], 'removed_exact_duplicate':uid in duplicates,
            'retained_source_row_uid':duplicates.get(uid,{}).get('retained_source_row_uid',uid)}
           for r in votes for uid in r['study_source_row_uids']]
    pq.write_table(pa.Table.from_pylist(links),output/'vote_support_links.parquet',compression='zstd')
    audit=pq.read_table(source/'record_audit.parquet').to_pandas()
    changes={r['source_row_uid']:r for r in receipts}
    for level,ids in ((1,set(voters)),(2,set(changes)-set(voters))):
        mask=audit['source_row_uid'].isin(ids)
        audit.loc[mask,'level']=level;audit.loc[mask,'family_key']=policy.FAMILIES[level]
        audit.loc[mask,'reason']='actual_source_vote_membership:'+votes_path.parent.name if level==1 else 'nonvoter_direct_after_source_vote_refresh'
        audit.loc[mask,'is_gold_voter']=level==1;audit.loc[mask,'direct_signal']=True
    for uid, hold in holds.items():
        mask = audit['source_row_uid'].eq(uid)
        audit.loc[mask,'level']=0; audit.loc[mask,'family_key']=''
        audit.loc[mask,'reason']=hold['reason']; audit.loc[mask,'is_gold_voter']=False
    audit=audit.loc[~audit['source_row_uid'].isin(duplicates)]
    pq.write_table(pa.Table.from_pandas(audit,preserve_index=False),output/'record_audit.parquet',compression='zstd')
    import shutil
    for name in ('retrieval_identity_cache.jsonl','retrieval_identity_cache.manifest.json'):
        shutil.copy2(source/name,output/name)
    if newly_eligible_smiles:
        from tools.chembl_tool.common.starling.new_task_retrieval_identity import prepare_cache
        prepare_cache(task,newly_eligible_smiles,output/'retrieval_identity_cache.jsonl')
    for path in (identity_holds_path,identity_clearances_path):
        if path:shutil.copy2(path,output/Path(path).name)
    assert all(sha256_file(Path(p))==h for p,h in inputs.items())
    manifest={'task':task,'status':'passed','contract':'source_vote_membership_refresh.v2',
        'source_rows':sum(counts.values()),'source_rows_deleted':len(duplicates),'raw_and_identity_fields_unchanged':not bool(holds or clearances),
        'raw_and_structural_identity_fields_unchanged':True,'new_hash_bound_identity_holds':len(holds),
        'hash_bound_identity_clearances':len(clearances),
        'newly_eligible_molecular_forms_cache_checked':len(newly_eligible_smiles),
        'exact_deduplication_enabled':deduplicate_exact,
        'exact_deduplication_policy':'All raw fields including PMID, support, conditions and confidence plus canonical identity and identity hold; ignore only source_row_uid/extraction_id/paragraph_idx. Prefer actual voter.',
        'level_counts':dict(sorted(counts.items())),'transitions':dict(transitions),
        'old_voters':len(old_voters),'new_voters':len(voters),'new_voter_labels':dict(Counter(r['Y'] for r in votes)),
        'nonvoter_duplicate_support_records':sum(not r['is_gold_voter'] for r in links),
        'records_with_level_change':len(receipts),'inputs':inputs,
        'retained_nonvoter_study_support_records':sum(not r['is_gold_voter'] and not r['removed_exact_duplicate'] for r in links),
        'files':{p.name:sha256_file(p) for p in output.iterdir() if p.is_file() and p.suffix!='.lock' and p.name!='membership_refresh_receipt.json'}}
    write_json_atomic(output/'membership_refresh_receipt.json',manifest)
    print(json.dumps(manifest,sort_keys=True),flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--refresh-votes',type=Path,help='Explicit new source-vote release; permits exact new voter membership')
    parser.add_argument('--identity-holds',type=Path,help='Reviewed source UID/payload-hash identity holds to exclude from gold/retrieval')
    parser.add_argument('--identity-clearances',type=Path,help='Reviewed UID/payload/structure-bound clearances of old identity holds')
    parser.add_argument('--deduplicate-exact',action='store_true',help='Remove identical record copies; preserve differing passages and study provenance')
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    action.add_argument('--record-review',type=Path,help='Hash-bound source-only placement/exclusion ledger; never changes voters or raw records')
    parser.add_argument('--heldout-benchmark-root',type=Path,help='Task root of the new benchmark; stage explicit identity metadata')
    args = parser.parse_args()
    if args.record_review:
        apply_record_review(args.task,args.source,args.output,args.record_review)
    elif args.refresh_votes:
        refresh_membership(args.task,args.source,args.output,args.refresh_votes,deduplicate_exact=args.deduplicate_exact,identity_holds_path=args.identity_holds,identity_clearances_path=args.identity_clearances)
        if args.heldout_benchmark_root:
            stage_heldout_inputs(args.task,args.heldout_benchmark_root,args.output.parent/'heldout')


def apply_record_review(task, source, output, ledger):
    """Stage exact reviewed nonvoter changes; leave the input release immutable."""
    import os
    policy=importlib.import_module(f'tools.chembl_tool.tasks.{task}.starling_levels')
    review=json.loads(ledger.read_text())
    before=sha256_file(source/'records.parquet')
    if review.get('task')!=task or review.get('source_records_sha256')!=before:
        raise ValueError('Record review task/source hash mismatch')
    rows=review['decisions']; decisions={r['source_row_uid']:r for r in rows}
    if len(rows)!=len(decisions):raise ValueError('Duplicate record review UID')
    for r in rows:
        if r['action'] not in {'move','exclude'} or not r.get('reason') or not r.get('quote'):
            raise ValueError('Expected grounded move/exclude decisions only')
        if r['previous_level'] not in range(2,8) or r['level'] not in ({0} if r['action']=='exclude' else range(2,8)):
            raise ValueError('Record review cannot change voter membership')
        if r['level']==r['previous_level']:raise ValueError('No-op record review')
    if source.resolve()==output.resolve() or output.exists():
        raise ValueError('Review output must be a fresh versioned directory')
    output.mkdir(parents=True)
    matched=set();counts=Counter();transitions=Counter()
    changed=DERIVED-{'is_gold_voter'} | {'placement_reviewed'}
    parquet=pq.ParquetFile(source/'records.parquet')
    with atomic_output_path(output/'records.parquet') as tmp:
        with pq.ParquetWriter(tmp,parquet.schema_arrow,compression='zstd') as writer:
            for table in parquet.iter_batches(batch_size=8192):
                table=pa.Table.from_batches([table]);original=table
                uids=table['source_row_uid'].to_pylist()
                positions=[i for i,u in enumerate(uids) if u in decisions]
                if positions:
                    data={k:table[k].to_pylist() for k in changed if k in table.column_names}
                    for i in positions:
                        uid=uids[i];r=decisions[uid];row=table.slice(i,1).to_pylist()[0]
                        raw=json.loads(row['raw_record_json'])
                        if uid in matched or payload_hash(raw)!=r['source_payload_sha256']:
                            raise ValueError('Duplicate/stale reviewed payload: '+uid)
                        if row['is_gold_voter'] or row['progressive_level']!=r['previous_level'] or not row['retrieval_eligible']:
                            raise ValueError('Reviewed source membership mismatch: '+uid)
                        if not any(r['quote'] in str(raw.get(k) or '') for k in ('support_text','extra_details','qualifying_conditions')):
                            raise ValueError('Review quote is not in raw source: '+uid)
                        level=r['level']
                        if level>2 and policy.direct_guard(row) and not r.get('direct_guard_exemption'):
                            raise ValueError('Direct containment needs an explicit record-specific review: '+uid)
                        values={'group_id':'Group.'+policy.FAMILIES[level] if level else 'Excluded.'+task,
                            'retrieval_eligible':bool(level),'progressive_level':level,'family_key':policy.FAMILIES.get(level,''),
                            'level_assignment_reason':r['reason'],'heldout_filter_scope':'direct_outcome' if level==2 else '',
                            'placement_reviewed':True}
                        for k in data:data[k][i]=values[k]
                        matched.add(uid);transitions[f'{r["previous_level"]}->{level}']+=1
                    for k,v in data.items():
                        pos=table.schema.get_field_index(k);field=table.schema.field(k)
                        table=table.set_column(pos,field,pa.array(v,type=field.type))
                    assert all(table[k].equals(original[k]) for k in table.column_names if k not in changed)
                counts.update(table['progressive_level'].to_pylist());writer.write_table(table)
    if matched!=set(decisions):raise ValueError('Reviewed records missing from source')
    # Keep full audit and identity/vote provenance; only reviewed rows change placement.
    audit=pq.read_table(source/'record_audit.parquet').to_pandas()
    for uid,r in decisions.items():
        mask=audit['source_row_uid'].eq(uid)
        if int(mask.sum())!=1:raise ValueError('Review UID absent/duplicated in audit: '+uid)
        for k,v in {'level':r['level'],'family_key':policy.FAMILIES.get(r['level'],''),'reason':r['reason'],'reviewed':True}.items():
            if k in audit.columns:audit.loc[mask,k]=v
    pq.write_table(pa.Table.from_pandas(audit,preserve_index=False),output/'record_audit.parquet',compression='zstd')
    for p in source.iterdir():
        if p.is_file() and p.name not in {'records.parquet','record_audit.parquet'}:os.link(p,output/p.name)
    write_json_atomic(output/'record_review.json',review)
    receipt={'task':task,'status':'passed','source_records_sha256':before,'records_sha256':sha256_file(output/'records.parquet'),
        'ledger_sha256':sha256_file(ledger),'source_rows':sum(counts.values()),'source_rows_deleted':0,
        'raw_fields_and_voter_membership_unchanged':True,'records_changed':len(matched),
        'transitions':dict(transitions),'level_counts':dict(sorted(counts.items()))}
    assert sha256_file(source/'records.parquet')==before
    write_json_atomic(output/'record_review_receipt.json',receipt)
    return receipt


def _repair_source_row(task, row, decision):
    """Apply reviewed content to retrieval fields; retain immutable acquisition/votes."""
    from tools.chembl_tool.common.starling.build_source_records import identity, _joined, text
    original = json.loads(row['raw_record_json'])
    if (row['source_row_uid'] != decision['source_row_uid']
            or payload_hash(original) != decision['source_payload_sha256']
            or row['progressive_level'] != decision['previous_level']):
        raise ValueError('Stale content repair: '+row['source_row_uid'])
    corrected = decision.get('corrected_record')
    result = dict(row)
    if corrected is not None:
        if (corrected.get('source_row_uid') != row['source_row_uid']
                or str(corrected.get('pmid')) != str(original.get('pmid'))):
            raise ValueError('Content repair cannot replace source UID or PMID')
        if not decision.get('sources'):
            raise ValueError('Content repair requires source evidence')
        policy = importlib.import_module(f'tools.chembl_tool.tasks.{task}.starling_source')
        fields = policy.FIELDS[row['source_id'].rsplit('_', 1)[1]]
        used = {key for names in fields.values() for key in names}
        excluded = used | {'SMILES', 'source_row_uid', 'pmid', 'paragraph_idx', 'extraction_id',
                           'support_text', 'confidence', 'molecule_name', 'agent_name', 'entity_name'}
        result.update({key: _joined(corrected, names) for key, names in fields.items()})
        result.update(qualifying_conditions=_joined(corrected, [k for k in corrected if k not in excluded]),
                      support_text=text(corrected.get('support_text')),
                      molecule_name=next((text(corrected.get(k)) for k in
                          ('molecule_name', 'agent_name', 'entity_name') if text(corrected.get(k))), ''),
                      reviewed_record_json=json.dumps(corrected, ensure_ascii=False, separators=(',', ':')))
        smiles = text(corrected.get('SMILES'))
        if smiles:
            _, molecule = identity(smiles)
            if molecule['identity_review_reason']:
                raise ValueError('Repaired single-molecule identity remains invalid')
            result.update(canonical_smiles=molecule['canonical_smiles'],
                          molecule_identity_key=molecule['parent_inchi_key'],
                          molecule_id=molecule['molecule_id'], parent_smiles=molecule['parent_smiles'],
                          bemis_murcko_scaffold=molecule['bemis_murcko_scaffold'],
                          identity_review_reason='', identity_verification='primary_source_reviewed_repair')
        else:
            result.update({k: '' for k in ('canonical_smiles', 'molecule_identity_key', 'molecule_id',
                                          'parent_smiles', 'bemis_murcko_scaffold')})
            result.update(identity_review_reason='reviewed_subject_not_single_small_molecule',
                          identity_verification='primary_source_reviewed_material_or_combination')
    level = decision['level']
    levels = importlib.import_module(f'tools.chembl_tool.tasks.{task}.starling_levels')
    if level not in (0, 2, 3, 4, 5, 6, 7) or not decision.get('reason'):
        raise ValueError('Content review cannot grant L1 membership')
    # A frozen voter may lose retrieval eligibility, but its corrected identity
    # cannot inherit the historical vote. The source vote itself stays frozen.
    if row['is_gold_voter'] and level:
        raise ValueError('Corrected voter requires separate gold review before retrieval promotion')
    if level and (not result['canonical_smiles'] or result['identity_review_reason']):
        raise ValueError('Eligible repaired record requires a verified single-molecule identity')
    if level > 2 and levels.direct_guard(result) and not decision.get('direct_guard_exemption'):
        raise ValueError('Repaired direct-containing record must remain in L2')
    result.update(progressive_level=level, retrieval_eligible=bool(level),
                  group_id='Group.'+levels.FAMILIES[level] if level else 'Excluded.'+task,
                  family_key=levels.FAMILIES.get(level, ''), level_assignment_reason=decision['reason'],
                  heldout_filter_scope='direct_outcome' if level == 2 else '', placement_reviewed=True,
                  direct_signal=levels.direct_guard(result))
    for key in ('raw_record_json', 'source_smiles', 'source_row_uid', 'is_gold_voter'):
        assert result[key] == row[key]
    return result


def apply_content_repairs(task, source, output, ledger):
    """Publish a hash-bound staged source without rerunning votes or global rules."""
    import shutil
    from tools.chembl_tool.common.build_runtime import local_input
    from tools.chembl_tool.common.starling.new_task_retrieval_identity import prepare_cache
    source, output, ledger = Path(source), Path(output), Path(ledger)
    review = json.loads(ledger.read_text())
    before = sha256_file(source/'records.parquet')
    if review['task'] != task or review['source_records_sha256'] != before:
        raise ValueError('Repair ledger task/source mismatch')
    decisions = {d['source_row_uid']: d for d in review['decisions']}
    if len(decisions) != len(review['decisions']) or output.exists():
        raise ValueError('Duplicate repair UID or output already exists')
    output.mkdir(parents=True)
    parquet = pq.ParquetFile(local_input(source/'records.parquet'))
    schema = parquet.schema_arrow
    if 'reviewed_record_json' not in schema.names:
        schema = schema.append(pa.field('reviewed_record_json', pa.string()))
    matched, counts, changes, new_smiles = set(), Counter(), [], set()
    with atomic_output_path(output/'records.parquet') as tmp, pq.ParquetWriter(tmp, schema, compression='zstd') as writer:
        for batch in parquet.iter_batches(batch_size=16384):
            table = pa.Table.from_batches([batch])
            if 'reviewed_record_json' not in table.column_names:
                table = table.append_column('reviewed_record_json', pa.nulls(len(table), pa.string()))
            positions = [i for i, uid in enumerate(table['source_row_uid'].to_pylist()) if uid in decisions]
            if positions:
                rows = table.to_pylist()
                for i in positions:
                    old = rows[i];uid = old['source_row_uid']
                    if uid in matched: raise ValueError('Duplicate source UID')
                    rows[i] = _repair_source_row(task, old, decisions[uid]);matched.add(uid)
                    if rows[i]['retrieval_eligible']: new_smiles.add(rows[i]['canonical_smiles'])
                    changes.append({'source_row_uid':uid,'previous_level':old['progressive_level'],
                                    'level':rows[i]['progressive_level'],'reason':rows[i]['level_assignment_reason'],
                                    'molecule_identity_key':rows[i].get('molecule_identity_key', ''),
                                    'direct_signal':rows[i]['direct_signal']})
                table = pa.Table.from_pylist(rows, schema=schema)
            counts.update(table['progressive_level'].to_pylist());writer.write_table(table)
    if matched != set(decisions): raise ValueError('Missing repair UID')
    audit = pq.read_table(source/'record_audit.parquet').to_pandas()
    for c in changes:
        mask = audit['source_row_uid'].eq(c['source_row_uid'])
        if int(mask.sum()) != 1: raise ValueError('Repair audit UID mismatch')
        policy = importlib.import_module(f'tools.chembl_tool.tasks.{task}.starling_levels')
        for key, value in {'level':c['level'],'family_key':policy.FAMILIES.get(c['level'],''),
                           'molecule_identity_key':c['molecule_identity_key'],
                           'reason':c['reason'],'reviewed':True,'direct_signal':c['direct_signal']}.items():
            if key in audit.columns: audit.loc[mask,key] = value
    pq.write_table(pa.Table.from_pandas(audit,preserve_index=False),output/'record_audit.parquet',compression='zstd')
    for p in source.iterdir():
        if p.is_file() and p.name not in {'records.parquet','record_audit.parquet'} and p.suffix != '.lock':
            shutil.copy2(p,output/p.name)
    prepare_cache(task,new_smiles,output/'retrieval_identity_cache.jsonl')
    write_json_atomic(output/'content_repairs.json',review)
    receipt = {'status':'passed','task':task,'source_records_sha256':before,
               'records_sha256':sha256_file(output/'records.parquet'),'records_changed':len(changes),
               'source_rows':sum(counts.values()),'source_rows_deleted':0,'level_counts':dict(sorted(counts.items())),
               'changes':changes,'raw_acquisition_and_frozen_vote_membership_unchanged':True,
               'gold_labels_modified':False,'ledger_sha256':sha256_file(ledger)}
    assert sha256_file(source/'records.parquet') == before
    write_json_atomic(output/'content_repair_receipt.json',receipt)
    return receipt


if __name__ == '__main__':
    main()
