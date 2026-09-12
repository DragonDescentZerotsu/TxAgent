"""Validate an explicit reviewed source and its index without repeating extraction."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import importlib
import json
from pathlib import Path
import pickle

import pyarrow.parquet as pq

from tools.chembl_tool.common.assay_retrieval import build_family_molecule_prefix_view, retrieve_family_molecule_prefixes
from tools.chembl_tool.common.build_runtime import local_input
from tools.chembl_tool.common.json_utils import read_jsonl, sha256_file, write_json_atomic
from tools.chembl_tool.common.progressive_assay_reasoning import append_evidence, extract_cumulative_evidence, select_initial_evidence, select_progressive_delta
from tools.chembl_tool.common.starling.new_task_identity import VERSION, leakage_identity
from tools.chembl_tool.common.starling.new_task_retrieval_identity import load_cache, decide_candidate

TASKS = {'dili': 'DILI', 'carcinogens': 'Carcinogens'}
LEVELS = list(range(1, 8))


def heldout_excluded_levels(task, manifest):
    """Validate the declared filter; support frozen legacy indices explicitly."""
    family = importlib.import_module(f'tools.chembl_tool.tasks.{task}.starling_levels').FAMILIES[1]
    scope = (manifest['filter_scope_field'], manifest['filter_scope_value'])
    if scope == ('group_id', 'Group.'+family):
        return {1}
    if scope == ('heldout_filter_scope', 'direct_outcome'):
        return {1, 2}
    raise ValueError('Unsupported heldout filter: '+str(scope))


def validate(task, scheme, benchmark, *, source_root, index_dir, heldout_file=None):
    source_root, directory = Path(source_root), Path(index_dir)
    manifest = json.loads((directory/'manifest.json').read_text())
    cache = load_cache(task, source_root/'retrieval_identity_cache.jsonl')
    overlay = source_root/'records.parquet'
    heldout_path = Path(heldout_file) if heldout_file else benchmark/TASKS[task]/scheme/'heldout_molecule_condition_labels.jsonl'
    heldout_rows = read_jsonl(heldout_path)
    heldout = {leakage_identity(task, r['drug'])['leakage_group'] for r in heldout_rows}
    assert heldout == {r['leakage_group'] for r in heldout_rows}
    excluded_levels = heldout_excluded_levels(task, manifest)
    required = {'retrieval_identity_contract': VERSION, 'retrieval_identity_task': task,
                'records_sha256': sha256_file(overlay), 'heldout_molecules_jsonl_sha256': sha256_file(heldout_path),
                'n_direct_heldout_records_after_filter': 0, 'filter_source_id': ''}
    for key, value in required.items():
        assert manifest[key] == value, (key, manifest[key], value)
    frame = pq.read_table(local_input(overlay), columns=['canonical_smiles','retrieval_eligible','progressive_level','source_row_uid','identity_review_reason']).to_pandas()
    holds = set(frame.loc[frame['identity_review_reason'].fillna('').ne(''),'source_row_uid'])
    assert not frame.loc[frame['source_row_uid'].isin(holds), 'retrieval_eligible'].any()
    assert len(frame.loc[frame['source_row_uid'].isin(holds)]) == len(holds)
    selected = frame[frame['retrieval_eligible'].eq(True)]
    source_groups = selected['canonical_smiles'].map({s: r['leakage_group'] for s,r in cache.items()})
    assert not source_groups.isna().any()
    excluded_mask = selected['progressive_level'].isin(excluded_levels) & source_groups.isin(heldout)
    excluded = int(excluded_mask.sum())
    assert manifest['n_direct_heldout_records_excluded'] == excluded
    retained_direct_by_level={level:int((selected['progressive_level'].eq(level) & ~excluded_mask).sum()) for level in (1,2)}
    heldout_l2_source = int((selected['progressive_level'].eq(2) & source_groups.isin(heldout) & ~excluded_mask).sum())
    del frame, selected, source_groups
    artifacts = {}
    for filename, key in [('assay_neighbor_index.pkl','index_sha256'),('assay_molecule_evidence.jsonl','evidence_sha256')]:
        artifacts[filename] = sha256_file(directory/filename)
        assert artifacts[filename] == manifest[key]
    with local_input(directory/'assay_neighbor_index.pkl').open('rb') as handle:
        index = pickle.load(handle)
    assert index['source']['retrieval_identity_contract'] == VERSION
    molecules = {m['molecule_chembl_id']: m for m in index['molecules']}
    by_parent = {v['source_parent_smiles']: v for v in cache.values()}
    for molecule in molecules.values():
        fresh = cache.get(molecule['retrieval_identity_source_smiles']) or by_parent.get(molecule['molecule_identity']['parent_smiles'])
        assert fresh is not None
        assert molecule['retrieval_leakage_group'] == fresh['leakage_group']
        assert molecule['retrieval_scaffold_leakage_group'] == fresh['scaffold_leakage_group']
    cards, groups = Counter(), 0
    heldout_cards = Counter()
    heldout_l2_molecules = set()
    for mid, by_group in index['evidence_by_molecule_group'].items():
        is_heldout = molecules[mid]['retrieval_leakage_group'] in heldout
        for rows in by_group.values():
            for row in rows:
                groups += 1
                for card in row['source_record_examples']:
                    level = int(card['evidence_family_level'])
                    assert not (is_heldout and level in excluded_levels), (mid, level, 'heldout excluded-level card')
                    if is_heldout:
                        heldout_cards[level] += 1
                        if level == 2: heldout_l2_molecules.add(mid)
                    cards[level] += 1
    assert set(cards) == set(LEVELS)
    if heldout_l2_source:
        assert heldout_cards[2] > 0, 'L2 was silently dropped despite the L1-only filter'
    print(f'{task}/{scheme}: {len(molecules)} identities and {groups} groups passed; running seven-level smoke', flush=True)
    view = build_family_molecule_prefix_view(index, levels=LEVELS)
    valid = read_jsonl(benchmark/TASKS[task]/scheme/'valid_molecule_condition_labels.jsonl')
    queries = [next(r for r in valid if int(r['Y']) == label) for label in (0,1)]
    receipts = []
    policy = 'scaffold_disjoint' if scheme == 'scaffold' else 'parent_disjoint'
    for query in queries:
        qid = leakage_identity(task, query['drug'])
        l2_decisions = Counter()
        for mid in heldout_l2_molecules:
            excluded_candidate = decide_candidate(qid, molecules[mid], policy, VERSION).excluded
            same_parent = molecules[mid]['retrieval_leakage_group'] == qid['leakage_group']
            if same_parent: assert excluded_candidate
            l2_decisions['excluded' if excluded_candidate else 'allowed_other_heldout_analog'] += 1
        results = retrieve_family_molecule_prefixes(query['drug'], view, levels=LEVELS,
            min_similarity=.3, neighbor_identity_policy=policy)
        active, previous, previous_ids = {}, {}, set()
        levels = []
        for level in LEVELS:
            result = results[level]
            assert result['status'] == 'ok'
            neighbors = [n for group in result['groups'] for n in group['neighbors']]
            for neighbor in neighbors:
                assert not decide_candidate(qid, molecules[neighbor['molecule_chembl_id']], policy, VERSION).excluded
                for row in neighbor['evidence_rows']:
                    assert all(int(c['evidence_family_level']) <= level for c in row['source_record_examples'])
            cumulative = extract_cumulative_evidence(result)
            ids = {(aid,cid) for aid,analog in cumulative.items() for cid in analog['cards']}
            assert previous_ids <= ids
            old_ids = {(aid,cid) for aid,analog in active.items() for cid in analog['cards']}
            if level == 1:
                active, _ = select_initial_evidence(cumulative, card_limit=4)
            else:
                new, augmented, _ = select_progressive_delta(previous, cumulative, active, level=level, card_limit=2)
                active = append_evidence(active, new, augmented)
            active_ids = {(aid,cid) for aid,analog in active.items() for cid in analog['cards']}
            assert old_ids <= active_ids
            levels.append({'level':level,'neighbors':len(neighbors),'cumulative_cards':len(ids),'active_cards':len(active_ids)})
            previous, previous_ids = cumulative, ids
        receipts.append({'molecule_identity_key':query['molecule_identity_key'],'Y':query['Y'],
            'heldout_l2_query_identity_decisions':dict(l2_decisions),'levels':levels})
    assert any(r['levels'][-1]['neighbors'] for r in receipts)
    report = {'status':'passed','task':task,'scheme':scheme,'identity_contract':VERSION,
              'manifest_sha256':sha256_file(directory/'manifest.json'), 'artifacts':artifacts,
              'molecule_identities_checked_against_pinned_unique_cache':len(molecules), 'exact_specimen_hold_source_records_excluded':len(holds),'all_indexed_groups_checked':groups,
              'cards_by_level':dict(cards),'heldout_leakage_groups':len(heldout),
              'excluded_direct_source_records':excluded,'retained_direct_records_by_level':retained_direct_by_level,
              'heldout_prefilter_levels':sorted(excluded_levels),
              'heldout_l1_cards':heldout_cards[1], 'heldout_l2_cards':heldout_cards[2],
              'heldout_l2_source_records_retained':heldout_l2_source,
              'query_smoke':receipts}
    write_json_atomic(directory/'identity_validation.json', report)
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--scheme', choices=['scaffold','random'], required=True)
    parser.add_argument('--benchmark-root', type=Path, default=Path('data/conditioned_benchmark'))
    parser.add_argument('--source-root',type=Path,required=True,help='Explicit new source release with records and identity cache')
    parser.add_argument('--index-dir',type=Path,required=True,help='Explicit index of that source release')
    parser.add_argument('--heldout-file',type=Path,help='Identity-enriched input with identical heldout query rows')
    args = parser.parse_args()
    # Loaded JSON/index records are acyclic; avoid repeated heap traversal.
    gc.disable()
    validate(args.task,args.scheme,args.benchmark_root,source_root=args.source_root,index_dir=args.index_dir,heldout_file=args.heldout_file)


if __name__ == '__main__':
    main()
