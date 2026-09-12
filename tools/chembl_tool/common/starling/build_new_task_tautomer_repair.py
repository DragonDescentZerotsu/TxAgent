"""Stage a DILI/Carcinogens identity repair without changing any active release."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tools.chembl_tool.common.json_utils import read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic
from tools.chembl_tool.common.starling.new_task_identity import normalize_new_task_identity, leakage_identity, VERSION
from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
import json
import shutil
from rdkit import rdBase
from tools.chembl_tool.common.starling.conditioned_benchmark import TASK_DIRECTORIES
from tools.chembl_tool.common.starling.fresh_conditioned_benchmark import build_fresh_conditioned_benchmark

DEFAULT_ROOT = Path('data/.build/new_task_tautomer_repair')


def _identity(args):
    task, smiles = args
    try:
        return smiles, normalize_new_task_identity(task, smiles), None
    except ValueError as exc:
        return smiles, None, str(exc)


def regroup_votes(original, identities):
    groups = defaultdict(list)
    for old in original:
        identity = identities[old['drug']]
        row = dict(old)
        row['identity_before_tautomer_repair'] = {k: old[k] for k in ('drug', 'molecule_identity_key', 'bemis_murcko_scaffold')}
        row.update({k: identity[k] for k in ('drug', 'molecule_identity_key', 'bemis_murcko_scaffold', 'leakage_group', 'scaffold_leakage_group', 'scaffold_group_smiles')})
        row['identity_contract'] = VERSION
        row['original_bemis_murcko_scaffold'] = identity['bemis_murcko_scaffold']
        groups[row['study_id'], row['molecule_identity_key'], row['condition_group']].append(row)
    accepted, conflicts, changes = [], [], []
    for key, rows in sorted(groups.items()):
        uids = sorted(r['source_record_id'] for r in rows)
        if len({r['Y'] for r in rows}) != 1:
            conflicts.append({'study_id': key[0], 'molecule_identity_key': key[1], 'condition_group': key[2],
                              'reason': 'within_study_label_conflict_after_tautomer_identity_merge', 'old_source_votes': rows})
            continue
        representative = dict(min(rows, key=lambda r: (r['reviewer'] != 'codex_record_semantic_review', r['source_record_id'])))
        representative['study_source_row_uids'] = sorted({uid for r in rows for uid in r.get('study_source_row_uids', [r['source_record_id']])})
        representative['study_source_pmids'] = sorted({pmid for r in rows for pmid in r.get('study_source_pmids', [r['pmid']])})
        representative['identity_merged_vote_uids'] = uids
        accepted.append(representative)
        if len(rows) > 1:
            changes.append({'retained_source_record_id': representative['source_record_id'], 'old_vote_uids': uids,
                            'study_id': key[0], 'molecule_identity_key': key[1], 'condition_group': key[2],
                            'Y': representative['Y'], 'action': 'one_study_one_parent_one_condition_one_vote'})
    scaffold_representatives = {}
    for row in accepted:
        group = row['scaffold_leakage_group']
        scaffold_representatives[group] = row['scaffold_group_smiles']
    for row in accepted:
        row['identity_scaffold_before_group_representative'] = row['bemis_murcko_scaffold']
        row['bemis_murcko_scaffold'] = scaffold_representatives[row['scaffold_leakage_group']]
    representations = {}
    for row in accepted:
        key = row['molecule_identity_key']
        pair = row['drug'], row['bemis_murcko_scaffold']
        representations[key] = min(pair, representations.get(key, pair))
    for row in accepted:
        row['drug'], row['bemis_murcko_scaffold'] = representations[row['molecule_identity_key']]
    return accepted, conflicts, changes


def build_source(task, output_root=DEFAULT_ROOT, workers=16, reuse_enumeration_cache=False):
    source = Path('data/starling_data') / task / 'gold_v2/source_votes.jsonl'
    previous_sha = sha256_file(source)
    original = read_jsonl(source)
    withdrawals_path = output_root / 'specimen_identity_withdrawal_candidates.jsonl'
    withdrawals = [r for r in read_jsonl(withdrawals_path) if r['task'].lower() == task] if withdrawals_path.exists() else []
    by_uid = {r['source_record_id']: r for r in original}
    for review in withdrawals:
        row = by_uid[review['source_row_uid']]
        if row['source_payload_sha256'] != review['source_payload_sha256'] or row['raw_record'] != review['raw_record']:
            raise ValueError('Stale specimen identity withdrawal')
    eligible = [r for r in original if r['source_record_id'] not in {r['source_row_uid'] for r in withdrawals}]
    identities, pending = {}, []
    cache_receipt = None
    cache_path = output_root / TASK_DIRECTORIES[task] / 'gold' / 'identity_lineage.jsonl'
    if reuse_enumeration_cache and cache_path.exists():
        frozen_cache = cache_path.with_name('enumeration_cache_before_grouping.jsonl')
        if not frozen_cache.exists():
            shutil.copy2(cache_path, frozen_cache)
        cache_receipt = {'path': str(frozen_cache), 'sha256': sha256_file(frozen_cache),
                         'reuse': 'gold enumeration only; current graph/scaffold groups recomputed without enumeration'}
        for item in read_jsonl(frozen_cache):
            if item['rdkit_version'] != rdBase.rdkitVersion or item['max_tautomers'] != 10000 or item['max_transforms'] != 10000:
                raise ValueError('Incompatible gold enumeration cache')
            if normalize_molecule_identity(item['input_drug']).parent_inchi_key != item['source_parent_inchi_key']:
                raise ValueError('Changed source parent in enumeration cache')
            item['enumeration'] = {'gold': item['enumeration']['gold']}
            item.update(leakage_identity(task, item['input_drug']))
            identities[item['input_drug']] = item
    needed = sorted({r['drug'] for r in eligible} - set(identities))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for smiles, identity, error in pool.map(_identity, ((task, s) for s in needed), chunksize=8):
            if error:
                pending.append({'source_smiles': smiles, 'reason': error})
            else:
                identities[smiles] = identity
    root = output_root / TASK_DIRECTORIES[task] / 'gold'
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(root / 'identity_pending.jsonl', pending)
    write_jsonl_atomic(root / 'identity_lineage.jsonl', [{'input_drug': s, **i} for s, i in sorted(identities.items())])
    if pending:
        raise ValueError(f'{task}: {len(pending)} identities have non-complete enumeration; inspect {root}/identity_pending.jsonl')
    identities = {s: identities[s] for s in sorted({r['drug'] for r in eligible})}
    limits = [{'input_drug': smiles, **identity,
               'source_names': sorted({r['molecule_name'] for r in original if r['drug'] == smiles}),
               'source_vote_uids': [r['source_record_id'] for r in original if r['drug'] == smiles]}
              for smiles, identity in sorted(identities.items())
              if any(x['status'] != 'Completed' for x in identity['enumeration'].values())]
    write_jsonl_atomic(root / 'enumeration_limit_reviews.jsonl', limits)
    accepted, conflicts, changes = regroup_votes(eligible, identities)
    write_jsonl_atomic(root / 'specimen_identity_withdrawals.jsonl', withdrawals)
    old_by_uid = {r['source_record_id']: r for r in original}
    assert len(old_by_uid) == len(original)
    for row in accepted:
        assert row['raw_record'] == old_by_uid[row['source_record_id']]['raw_record']
        assert row['source_payload_sha256'] == old_by_uid[row['source_record_id']]['source_payload_sha256']
    write_jsonl_atomic(root / 'source_votes.jsonl', accepted)
    write_jsonl_atomic(root / 'actual_voter_membership.jsonl',
        ({'source_record_id': uid, 'study_representative_uid': row['source_record_id'],
          'study_id': row['study_id'], 'molecule_identity_key': row['molecule_identity_key'],
          'condition_group': row['condition_group'], 'Y': row['Y']}
         for row in accepted for uid in row['identity_merged_vote_uids']))
    write_jsonl_atomic(root / 'study_conflicts.jsonl', conflicts)
    write_jsonl_atomic(root / 'merged_study_votes.jsonl', changes)
    old_keys = defaultdict(set)
    for i in identities.values():
        old_keys[i['molecule_identity_key']].add(i['source_parent_inchi_key'])
    write_json_atomic(root / 'manifest.json', {'task': task, 'identity_contract': VERSION,
        'input_source_votes': str(source), 'input_sha256': previous_sha,
        'source_votes': len(accepted), 'old_source_votes': len(original),
        'labels': dict(Counter(r['Y'] for r in accepted)), 'merged_studies': len(changes),
        'conflicting_studies': len(conflicts), 'merged_parent_groups': sum(len(v)>1 for v in old_keys.values()),
        'stereo_changed_identity_forms': sum(i['tautomer_stereo_changed'] for i in identities.values()),
        'noncomplete_identity_forms_retained': len(limits),
        'new_source_vote_eligibility_granted': 0, 'raw_records_changed': 0,
        'specimen_identity_withdrawn_votes': len(withdrawals), 'enumeration_cache': cache_receipt,
        'adapter_sha256': sha256_file(Path(__file__).with_name('new_task_identity.py'))})
    result = build_fresh_conditioned_benchmark(task=task, record_votes=accepted,
        output_root=output_root / 'source_only' / TASK_DIRECTORIES[task] / 'scaffold',
        source_artifacts=(source, root / 'source_votes.jsonl', root / 'identity_lineage.jsonl', Path(__file__), Path(__file__).with_name('new_task_identity.py')),
        required_labels=(0, 1))
    assert sha256_file(source) == previous_sha
    return result


def validate_staged_repair(output_root=DEFAULT_ROOT):
    from tools.chembl_tool.common.starling.conditioned_benchmark import NO_REPORTED_CONDITION
    report = {'identity_contract': VERSION, 'tasks': {}, 'status': 'passed'}
    for task in ('dili', 'carcinogens'):
        directory = TASK_DIRECTORIES[task]
        gold = output_root / directory / 'gold'
        old_path = Path('data/starling_data') / task / 'gold_v2/source_votes.jsonl'
        old = {r['source_record_id']: r for r in read_jsonl(old_path)}
        votes = read_jsonl(gold / 'source_votes.jsonl')
        membership = read_jsonl(gold / 'actual_voter_membership.jsonl')
        expected_members = {uid for r in votes for uid in r['identity_merged_vote_uids']}
        assert {r['source_record_id'] for r in membership} == expected_members
        assert len(membership) == len(expected_members)
        assert expected_members <= set(old)
        for row in votes:
            assert row['raw_record'] == old[row['source_record_id']]['raw_record']
            assert row['source_payload_sha256'] == old[row['source_record_id']]['source_payload_sha256']
        assert len(votes) == len({(r['study_id'], r['molecule_identity_key'], r['condition_group']) for r in votes})
        source_root = output_root / 'source_only' / directory / 'scaffold'
        for row in read_jsonl(source_root / 'accepted_parent_conditions_before_group_gate.jsonl'):
            ys = [v['Y'] for v in row['source_votes']]
            positive = sum(ys)
            threshold = .70 if row['condition_group'] == NO_REPORTED_CONDITION else .60
            assert positive * 2 != len(ys)
            assert max(positive, len(ys)-positive) / len(ys) >= threshold
            assert row['Y'] == int(positive * 2 > len(ys))
        augmented = output_root / 'augmented' / directory
        manifest = json.loads((augmented / 'manifest.json').read_text())
        checks = {}
        for scheme in ('scaffold', 'random'):
            split_rows = {split: read_jsonl(augmented / scheme / f'{split}_molecule_condition_labels.jsonl') for split in ('train', 'valid', 'test')}
            union = [r for rows in split_rows.values() for r in rows]
            assert len(union) == len({(r['molecule_identity_key'],r['condition_group']) for r in union})
            leakage = {split: {leakage_identity(task, r['drug'])['leakage_group'] for r in rows} for split, rows in split_rows.items()}
            scaffold = {split: {leakage_identity(task, r['drug'])['scaffold_leakage_group'] for r in rows} for split, rows in split_rows.items()}
            pairs = [('train','valid'),('train','test'),('valid','test')]
            overlaps = {a+'__'+b:len(leakage[a]&leakage[b]) for a,b in pairs}
            scaffold_overlaps = {a+'__'+b:len(scaffold[a]&scaffold[b]) for a,b in pairs}
            assert not any(overlaps.values())
            if scheme == 'scaffold':
                assert not any(scaffold_overlaps.values())
            singles = {split: sum(r['label_source']=='starling' and r['source_record_count']==1 for r in split_rows[split]) for split in ('valid','test')}
            assert sum(singles.values()) == manifest['optimizers'][scheme]['minimum_singletons_in_heldout']
            assert abs(singles['valid']-singles['test']) == manifest['optimizers'][scheme]['minimum_singleton_imbalance']
            for row in union:
                if row['label_source'] == 'tdc':
                    assert row['condition_group'] == NO_REPORTED_CONDITION
                    assert row['source_record_count'] == 0 and row['record_support_eligible'] is False
                    assert all(r['Y'] == row['Y'] for r in row['raw_source_rows'])
            checks[scheme] = {'rows': {split:len(rows) for split,rows in split_rows.items()},
                              'tautomer_leakage_group_overlap': overlaps, 'scaffold_topology_overlap': scaffold_overlaps,
                              'starling_singletons': singles, 'quality_objective_matches_optimizer': True}
        report['tasks'][task] = {'source_votes': len(votes), 'actual_voter_uids':len(membership),
            'old_actual_voter_uids_removed': sorted(set(old)-expected_members),
            'raw_source_fields_unchanged': True, 'agreement_70_60_recomputed': True, 'splits': checks,
            'source_gold_sha256': sha256_file(old_path),
            'files': {str(p):sha256_file(p) for root in (gold, source_root, augmented) for p in root.rglob('*') if p.is_file()}}
    write_json_atomic(output_root / 'validation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', nargs='+', choices=['dili', 'carcinogens'], default=['dili', 'carcinogens'])
    parser.add_argument('--output-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--phase', choices=['source', 'validate'], default='source')
    parser.add_argument('--reuse-enumeration-cache', action='store_true', help='Reuse pinned matching gold enumeration results; recompute current leakage/scaffold groups')
    args = parser.parse_args()
    if args.output_root.resolve() == Path('data/conditioned_benchmark').resolve() or Path('data/starling_data').resolve() in args.output_root.resolve().parents:
        parser.error('Identity repair must remain in an isolated staging root')
    if args.phase == 'validate':
        print(json.dumps(validate_staged_repair(args.output_root), indent=2))
        return
    for task in args.tasks:
        result = build_source(task, args.output_root, args.workers, args.reuse_enumeration_cache)
        print(task, result['n_accepted_parent_conditions'], flush=True)


if __name__ == '__main__':
    main()
