"""Independently validate new-task source conservation and progressive retrieval.

This tool never calls a language model or modifies scientific inputs. Successful
receipts are written only after every requested check passes; failures emit a
separate diagnostic ledger. Source semantics are checked independently of the
production classifier, over every complete model-visible field.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
from pathlib import Path
import pickle
import re

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tools.chembl_tool.common.assay_retrieval import (
    build_family_molecule_prefix_view,
    retrieve_family_molecule_prefixes,
)
from tools.chembl_tool.common.build_runtime import local_input, worker_pool
from tools.chembl_tool.common.json_utils import (
    read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic,
)
from tools.chembl_tool.common.molecule_identity import (
    bemis_murcko_scaffold, normalize_molecule_identity,
)
from tools.chembl_tool.common.progressive_assay_reasoning import (
    append_evidence, extract_cumulative_evidence, select_initial_evidence,
    select_progressive_delta,
)
from tools.chembl_tool.common.retrieval_policy import decide_candidate
from tools.chembl_tool.common.starling.assay_catalog import assay_id, assay_unit
from tools.chembl_tool.common.starling.source_gold_review import payload_hash
from tools.chembl_tool.paper_experiments.build_assay_family_catalog import TASKS

ROOT = Path('data/starling_data')
BENCHMARK = Path('data/.build/tdc_augmented_conditioned_benchmark')
DIRECTORIES = {'dili': 'DILI', 'carcinogens': 'Carcinogens'}
SCHEMES = ('scaffold', 'random')
LEVELS = list(range(1, 8))
FIELDS = {
    'endpoint_type': 'canonical_endpoint_name',
    'reported_value': 'canonical_measurement_text',
    'reported_units': 'canonical_unit_text',
    'assay_context': 'canonical_assay_context',
    'species_context': 'canonical_species_context',
    'qualifying_conditions': 'qualifying_conditions',
    'support_text': 'support_text',
}
# Independent audit vocabulary. These expressions intentionally do not import
# or call either task's production direct_guard/classify functions.
PATTERNS = {
    'dili': {
        'explicit_hepatotoxicity': r'\b(?:dili|hepatotoxi\w*|hepatotox\w*)\b',
        'organ_injury_outcome': r'\b(?:liver|hepatic|hepatocellular)[ -]+(?:injur\w*|damage|failure|necrosis|toxic\w*|fibrosis|cirrhosis)\b',
        'drug_hepatitis': r'\b(?:drug[ -]induced|toxic|clinical|acute|fulminant|cholestatic)[ -]+hepatitis\b',
        'clinical_liver_chemistry': r'\b(?:serum|plasma|patients?|clinical|in[ -]vivo)\b.{0,150}\b(?:ALT|AST|aminotransferase\w*|transaminase\w*|bilirubin)\b.{0,55}\b(?:elevat\w*|increas\w*|normal|unchanged|reduc\w*)\b',
    },
    'carcinogens': {
        'hazard_assertion': r'\b(?:carcinogen\w*|tumou?rigen\w*|neoplas\w*)\b',
        'tumor_outcome': r'\btumou?rs?[ -]+(?:incidence|induction|formation|burden|development|latency|promotion|outcome|bearing|volume|weight|growth)\b',
        'tumor_action': r'\b(?:induc\w*|caus\w*|develop\w*|produc\w*|prevent\w*|suppress\w*|inhibit\w*|without|no)\b.{0,70}\btumou?rs?\b',
        'cancer_subtype_outcome': r'\b(?:induc\w*|caus\w*|develop\w*|incidence|risk)\b.{0,50}\b(?:cancer|carcinoma|adenoma|sarcoma|lymphoma|leuk[ae]mia|mesothelioma)\b',
    },
}
PATTERNS = {task: {key: re.compile(value, re.I | re.S) for key, value in patterns.items()}
            for task, patterns in PATTERNS.items()}
_state = None


def clean(value):
    if value is None or isinstance(value, float) and math.isnan(value):
        return ''
    return str(value).strip()


def mentions(task, values):
    # Do not invent an outcome by joining e.g. units="no numeric value" to
    # assay="tumor cells". Inspect all complete fields, preserving boundaries.
    texts = [value.replace('_', ' ') for value in values]
    return tuple(name for name, pattern in PATTERNS[task].items()
                 if any(pattern.search(text) for text in texts))


def surface_signature(mid, aid, family, values):
    content = json.dumps([mid, aid, family, *values], ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(content.encode()).digest()


def columns_equal(left, right):
    if left.equals(right):
        return True
    # Arrow's equals treats NaN != NaN. Preserve NaN positions separately so
    # unchanged numeric source columns pass without equating NaN and null.
    if left.type == right.type and pa.types.is_floating(left.type):
        left_nan, right_nan = pc.is_nan(left), pc.is_nan(right)
        return left_nan.equals(right_nan) and pc.if_else(
            left_nan, pa.scalar(None, type=left.type), left
        ).equals(pc.if_else(right_nan, pa.scalar(None, type=right.type), right))
    return False


def _identity_hold_applies(row, raw, hold):
    if row['source_smiles'] != hold['source_smiles']:
        return False
    scope = hold.get('scope', 'record')
    if scope == 'record':
        return row['source_row_uid'] == hold['source_row_uid']
    if scope == 'source_smiles':
        return True
    if scope in {'source_smiles_name', 'source_smiles_and_molecule_name', 'source_smiles_and_molecule_name_and_text'}:
        if row['molecule_name'] != hold['molecule_name']:
            return False
    elif scope != 'claimed_name_in_text':
        raise ValueError('Unknown reviewed identity scope: ' + scope)
    if hold.get('required_text_pattern'):
        return bool(re.search(hold['required_text_pattern'], raw, re.I))
    return bool(hold.get('molecule_name')) and row['molecule_name'] == hold['molecule_name']


def _scan_group(position):
    task, paths, families, voters, reviews, collect, holds = _state
    hold_anchors = {hold['source_row_uid']: hold for group in holds.values() for hold in group}
    source = pq.ParquetFile(paths[0]).read_row_group(position)
    overlay = pq.ParquetFile(paths[1]).read_row_group(position)
    audit = pq.ParquetFile(paths[2]).read_row_group(position)
    assert len(source) == len(overlay) == len(audit), ('row conservation', position)
    for name in source.column_names:
        if name not in {'group_id', 'retrieval_eligible', 'is_gold_voter'}:
            assert columns_equal(source[name], overlay[name]), ('changed original column', position, name)
    assert source['source_row_uid'].equals(audit['source_row_uid'])
    assert source['group_id'].equals(audit['previous_group_id'])
    for source_key, audit_key in [('progressive_level', 'level'), ('family_key', 'family_key'),
                                 ('level_assignment_reason', 'reason'), ('is_gold_voter', 'is_gold_voter'),
                                 ('placement_reviewed', 'reviewed'), ('direct_signal', 'direct_signal')]:
        assert overlay[source_key].equals(audit[audit_key]), ('audit mismatch', position, source_key)
    selected = ['source_row_uid', 'canonical_record_id', 'source_id', 'molecule_id',
                'molecule_identity_key', 'progressive_level', 'family_key', 'group_id',
                'retrieval_eligible', 'heldout_filter_scope', 'is_gold_voter',
                'placement_reviewed', 'level_assignment_reason', 'source_smiles',
                'molecule_name', *FIELDS.values()]
    selected = list(dict.fromkeys(selected))
    rows = overlay.select(selected).to_pylist()
    raw = overlay['raw_record_json'].to_pylist()
    original_eligible = source['retrieval_eligible'].to_pylist()
    uids, canonical_ids, facts, anomalies, reviewed, voter_found = [], [], [], [], [], []
    counts, pattern_counts = Counter(), Counter()
    for ordinal, row in enumerate(rows):
        uid, level = row['source_row_uid'], int(row['progressive_level'])
        uids.append(uid); canonical_ids.append(row['canonical_record_id'])
        assert uid and row['canonical_record_id'] == uid, ('unstable UID', uid)
        assert level in range(8), ('invalid level', uid, level)
        assert row['retrieval_eligible'] == (level > 0), ('eligibility', uid)
        assert original_eligible[ordinal] or level == 0, ('identity hold promoted', uid)
        assert row['is_gold_voter'] == (uid in voters) == (level == 1), ('voter mismatch', uid)
        assert row['family_key'] == families.get(level, ''), ('family mismatch', uid)
        assert row['group_id'] == ('Group.' + families[level] if level else 'Excluded.' + task)
        assert row['heldout_filter_scope'] == ('direct_outcome' if level in (1, 2) else '')
        review = reviews.get(uid)
        assert row['placement_reviewed'] == bool(review), ('review membership', uid)
        matched_holds = [hold for hold in holds.get(row['source_smiles'], [])
                         if _identity_hold_applies(row, raw[ordinal], hold)]
        if uid in hold_anchors:
            hold = hold_anchors[uid]
            assert row['source_smiles'] == hold['source_smiles'], ('changed identity hold structure', uid)
            assert payload_hash(json.loads(raw[ordinal])) == hold['source_payload_sha256'], ('stale identity hold', uid)
        if matched_holds:
            assert level == 0, ('reviewed identity hold remains retrievable', uid)
        if row['level_assignment_reason'].startswith('reviewed_identity_hold:'):
            assert matched_holds, ('untraceable identity hold', uid)
        if review:
            assert payload_hash(json.loads(raw[ordinal])) == review['source_payload_sha256'], ('stale review', uid)
            if original_eligible[ordinal] and not matched_holds:
                forced_direct = (level == 2 and int(review['level']) > 2
                                 and not review.get('direct_guard_exemption')
                                 and row['level_assignment_reason'] == 'broad_direct_containment')
                assert level == int(review['level']) or forced_direct, ('review placement not applied', uid)
            reviewed.append(uid)
        if level == 1:
            assert payload_hash(json.loads(raw[ordinal])) == voters[uid], ('voter payload mismatch', uid)
            voter_found.append(uid)
        counts[level] += 1
        if not level:
            continue
        values = [clean(row[name]) for name in FIELDS.values()]
        hits = mentions(task, values)
        pattern_counts.update((name, level) for name in hits)
        exempt = bool(review and review.get('direct_guard_exemption'))
        if level > 2 and hits and not exempt:
            anomalies.append({'source_row_uid': uid, 'source_id': row['source_id'],
                              'level': level, 'patterns': hits,
                              'source_payload_sha256': payload_hash(json.loads(raw[ordinal])),
                              **{key: row[key] for key in FIELDS.values()}})
        if collect:
            aid = assay_id(task, assay_unit(row['canonical_assay_context'], row['canonical_endpoint_name'])[0])
            sig = surface_signature(row['molecule_id'], aid, row['family_key'], values)
            facts.append((sig, row['molecule_identity_key'], level, uid, aid, row['molecule_id']))
    return uids, canonical_ids, facts, anomalies, reviewed, voter_found, counts, pattern_counts


def _track(path, tracked):
    path = Path(path)
    digest = sha256_file(path)
    tracked[str(path)] = digest
    return digest


def validate_source(task, workers, collect, tracked):
    global _state
    root = ROOT/task
    source, overlay, audit = (root/'canonical_v1/records.parquet',
                             root/'progressive_v1/records.parquet',
                             root/'progressive_v1/record_audit.parquet')
    manifest_path = overlay.with_name('manifest.json')
    manifest = json.loads(manifest_path.read_text())
    _track(manifest_path, tracked)
    for path, digest in manifest['inputs'].items():
        assert _track(path, tracked) == digest, ('source build input changed', path)
    for name, digest in manifest['files'].items():
        assert _track(overlay.parent/name, tracked) == digest, ('source build output changed', name)
    config = importlib.import_module(TASKS[task]['config_module'])
    families = {level: spec.family_key for level, spec in enumerate(config.STARLING.mechanism_groups, 1)}
    _track(config.__file__, tracked)
    voters_path = root/'gold_v2/source_votes.jsonl'
    _track(voters_path, tracked)
    voter_rows = read_jsonl(voters_path)
    voters = {row['source_record_id']: row['source_payload_sha256'] for row in voter_rows}
    assert len(voters) == len(voter_rows), 'duplicate vote UID'
    review_path = root/'level_review_v1/placement_decisions.jsonl'
    _track(review_path, tracked)
    review_rows = read_jsonl(review_path)
    reviews = {row['source_row_uid']: row for row in review_rows}
    assert len(reviews) == len(review_rows), 'duplicate placement review UID'
    holds_path = review_path.with_name('identity_holds.jsonl')
    holds = {}
    if holds_path.exists():
        _track(holds_path, tracked)
        for hold in read_jsonl(holds_path):
            holds.setdefault(hold['source_smiles'], []).append(hold)
    paths = [local_input(path) for path in (source, overlay, audit)]
    files = [pq.ParquetFile(path) for path in paths]
    assert len({file.metadata.num_rows for file in files}) == 1
    assert len({file.num_row_groups for file in files}) == 1
    _state = task, paths, families, voters, reviews, collect, holds
    uid_set, canonical_set, source_facts, found_reviews, found_voters = set(), set(), {}, set(), set()
    retained, counts, patterns, anomalies = [], Counter(), Counter(), []
    try:
        with worker_pool(min(workers, files[0].num_row_groups)) as pool:
            for uids, canonical, facts, errors, reviewed, actual, n, p in pool.map(_scan_group, range(files[0].num_row_groups)):
                assert len(uids) == len(set(uids)) and not uid_set.intersection(uids), 'duplicate source UID'
                assert len(canonical) == len(set(canonical)) and not canonical_set.intersection(canonical), 'duplicate canonical UID'
                uid_set.update(uids); canonical_set.update(canonical)
                counts.update(n); patterns.update(p); anomalies.extend(errors)
                found_reviews.update(reviewed); found_voters.update(actual)
                for sig, parent, level, uid, aid, mid in facts:
                    previous = source_facts.setdefault(sig, (parent, level, uid))
                    assert previous[:2] == (parent, level), ('ambiguous surface identity', uid)
                    retained.append((aid, mid, parent, level))
    finally:
        _state = None
    assert found_voters == set(voters) and found_reviews == set(reviews)
    assert {hold['source_row_uid'] for group in holds.values() for hold in group} <= uid_set
    assert sum(counts.values()) == files[0].metadata.num_rows == len(uid_set)
    for level, count in counts.items():
        assert manifest['counts'][f'L{level}'] == count
    diagnostic = overlay.parent/'independent_direct_diagnostics.jsonl'
    write_jsonl_atomic(diagnostic, anomalies)
    if anomalies:
        print(json.dumps({'task': task, 'independent_direct_anomalies': len(anomalies),
                          'ledger': str(diagnostic), 'counts': dict(Counter(r['level'] for r in anomalies))}), flush=True)
        raise ValueError(f'{task}: {len(anomalies)} independently detected direct-containing records remain in L3+; see {diagnostic}')
    result = {'n_source_records': len(uid_set), 'n_unique_uids': len(uid_set),
              'all_original_fields_and_raw_payloads_unchanged': True,
              'levels': dict(sorted(counts.items())), 'actual_voters': len(voters),
              'payload_pinned_reviews': len(found_reviews), 'independent_direct_anomalies': 0,
              'independent_guard_hits': {f'{name}:L{level}': count for (name, level), count in sorted(patterns.items())}}
    print(json.dumps({'task': task, 'source_validation': result}), flush=True)
    return result, source_facts, retained


def _identity(smiles, cache):
    if smiles not in cache:
        cache[smiles] = normalize_molecule_identity(smiles)
    return cache[smiles]


def validate_indices(task, benchmark_root, source_facts, retained, tracked):
    config = importlib.import_module(TASKS[task]['config_module'])
    families = {level: spec.family_key for level, spec in enumerate(config.STARLING.mechanism_groups, 1)}
    spec = TASKS[task]
    catalog = Path(spec['output_root'])/spec.get('output_name', task)/'family_assays.jsonl'
    catalog_manifest = json.loads(catalog.with_name('manifest.json').read_text())
    _track(catalog.with_name('manifest.json'), tracked)
    assert catalog_manifest['catalog_sha256'] == _track(catalog, tracked)
    overlay = ROOT/task/'progressive_v1/records.parquet'
    assert catalog_manifest['records_sha256'] == tracked[str(overlay)]
    assert catalog_manifest['family_assignment_unit'] == 'source_record'
    output = {}
    for scheme in SCHEMES:
        split_root = benchmark_root/DIRECTORIES[task]/scheme
        splits, parents = {}, {}
        for split in ('train', 'valid', 'test'):
            path = split_root/f'{split}_molecule_condition_labels.jsonl'
            _track(path, tracked)
            splits[split] = read_jsonl(path)
            parents[split] = {row['molecule_identity_key'] for row in splits[split]}
        assert not (parents['train'] & parents['valid'] or parents['train'] & parents['test'] or parents['valid'] & parents['test'])
        heldout_path = split_root/'heldout_molecule_condition_labels.jsonl'
        heldout_rows = read_jsonl(heldout_path)
        _track(heldout_path, tracked)
        row_key = lambda r: (r['molecule_identity_key'], r['condition_group'], int(r['Y']))
        assert Counter(map(row_key, heldout_rows)) == Counter(map(row_key, splits['valid'] + splits['test']))
        heldout = parents['valid'] | parents['test']
        expected = Counter((aid, mid) for aid, mid, parent, level in retained
                           if not (level <= 2 and parent in heldout))
        excluded = sum(level <= 2 and parent in heldout for _, _, parent, level in retained)
        index_path = overlay.parent/'indices'/scheme/'assay_neighbor_index.pkl'
        metadata_path = index_path.with_name('manifest.json')
        _track(metadata_path, tracked)
        metadata = json.loads(metadata_path.read_text())
        policy = 'scaffold_disjoint' if scheme == 'scaffold' else 'parent_disjoint'
        wanted = {'records_sha256': tracked[str(overlay)], 'ranked_assays_sha256': tracked[str(catalog)],
                  'heldout_molecules_jsonl_sha256': tracked[str(heldout_path)],
                  'filter_source_id': '', 'filter_scope_field': 'heldout_filter_scope',
                  'filter_scope_value': 'direct_outcome', 'n_direct_heldout_records_after_filter': 0,
                  'n_direct_heldout_records_excluded': excluded,
                  'neighbor_identity_policy_default': policy,
                  'evidence_prompt_profile': 'assay_compact.mechanism_tagged_v4'}
        for key, value in wanted.items():
            assert metadata[key] == value, ('index lineage', task, scheme, key, metadata[key], value)
        assert metadata['index_sha256'] == _track(index_path, tracked)
        assert metadata['evidence_sha256'] == _track(index_path.with_name('assay_molecule_evidence.jsonl'), tracked)
        with local_input(index_path).open('rb') as handle:
            index = pickle.load(handle)
        indexed, cards_by_level, traces = Counter(), Counter(), {}
        for mid, groups in index['evidence_by_molecule_group'].items():
            for evidence_rows in groups.values():
                for row in evidence_rows:
                    aid = row['assay_chembl_id']
                    indexed[aid, mid] += row['source_record_count']
                    for card in row['source_record_examples']:
                        level = int(card['evidence_family_level'])
                        assert card['evidence_family'] == families[level]
                        sig = surface_signature(mid, aid, card['evidence_family'], [clean(card.get(key)) for key in FIELDS])
                        assert sig in source_facts, ('untraceable card surface', task, scheme, mid, aid)
                        parent, source_level, uid = source_facts[sig]
                        assert source_level == level and not (level <= 2 and parent in heldout), ('heldout leak', uid)
                        cards_by_level[level] += 1
                        traces.setdefault(level, {'source_row_uid': uid, 'surface_sha256': sig.hex()})
        assert indexed == expected, ('assay-molecule source counts differ', task, scheme)
        assert set(cards_by_level) == set(LEVELS), ('missing indexed family', task, scheme)
        view = build_family_molecule_prefix_view(index, levels=LEVELS)
        pool_counts = Counter()
        for groups in view['evidence_by_molecule_group'].values():
            for level in LEVELS:
                gid = view['family_molecule_prefix_view']['group_ids'][str(level)]
                for row in groups.get(gid, []):
                    for card in row['source_record_examples']:
                        assert 1 <= int(card['evidence_family_level']) <= level
                        pool_counts[level] += 1
        assert dict(pool_counts) == {level: sum(count for own_level, count in cards_by_level.items() if own_level <= level) for level in LEVELS}
        queries = []
        for label in (0, 1):
            for source in ('starling', 'tdc'):
                available = [r for r in splits['valid'] if int(r['Y']) == label and r.get('label_source') == source]
                if available:
                    queries.append(available[0])
        assert {int(row['Y']) for row in queries} == {0, 1}
        receipts, identity_cache = [], {}
        for query in queries:
            qid = _identity(query['drug'], identity_cache)
            assert qid.parent_inchi_key == query['molecule_identity_key']
            qscaffold = bemis_murcko_scaffold(query['drug'])
            results = retrieve_family_molecule_prefixes(query['drug'], view, levels=LEVELS,
                min_similarity=0.3, neighbor_identity_policy=policy)
            active, previous, previous_ids, level_receipts = {}, {}, set(), []
            for level in LEVELS:
                retrieval = results[level]
                assert retrieval['status'] == 'ok', ('retrieval failed', task, scheme, level)
                neighbors = [neighbor for group in retrieval['groups'] for neighbor in group['neighbors']]
                for neighbor in neighbors:
                    assert not decide_candidate(qid, neighbor, policy).excluded
                    nid = _identity(neighbor['canonical_smiles'], identity_cache)
                    assert nid.parent_inchi_key != qid.parent_inchi_key
                    if scheme == 'scaffold' and qscaffold:
                        assert qscaffold != bemis_murcko_scaffold(neighbor['canonical_smiles'])
                    for row in neighbor['evidence_rows']:
                        assert all(int(card['evidence_family_level']) <= level for card in row['source_record_examples'])
                cumulative = extract_cumulative_evidence(retrieval)
                ids = {(aid, cid) for aid, analog in cumulative.items() for cid in analog['cards']}
                assert previous_ids <= ids, ('cumulative card loss', task, scheme, level)
                old_ids = {(aid, cid) for aid, analog in active.items() for cid in analog['cards']}
                if level == 1:
                    active, _ = select_initial_evidence(cumulative, card_limit=4)
                else:
                    new, augmented, _ = select_progressive_delta(previous, cumulative, active, level=level, card_limit=2)
                    active = append_evidence(active, new, augmented)
                active_ids = {(aid, cid) for aid, analog in active.items() for cid in analog['cards']}
                assert old_ids <= active_ids, ('append-only card loss', task, scheme, level)
                level_receipts.append({'level': level, 'neighbors': len(neighbors), 'cumulative_cards': len(ids),
                                       'active_molecules': len(active), 'active_cards': len(active_ids)})
                previous, previous_ids = cumulative, ids
            receipts.append({'molecule_identity_key': query['molecule_identity_key'], 'Y': query['Y'],
                             'label_source': query.get('label_source'), 'levels': level_receipts})
        assert any(row['levels'][-1]['neighbors'] for row in receipts), ('vacuous smoke', task, scheme)
        output[scheme] = {'heldout_parents': len(heldout), 'excluded_direct_records': excluded,
                          'indexed_assay_molecule_groups': len(indexed), 'all_cards_traceable': True,
                          'cards_by_level': dict(cards_by_level), 'card_trace_examples': traces,
                          'cumulative_pool_cards': dict(pool_counts), 'query_smoke': receipts}
        print(json.dumps({'task': task, 'scheme': scheme, 'cards': dict(cards_by_level), 'smoke_queries': len(receipts)}), flush=True)
        del index, view
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', nargs='+', choices=list(DIRECTORIES), default=list(DIRECTORIES))
    parser.add_argument('--phase', choices=['source', 'index', 'all'], default='all')
    parser.add_argument('--benchmark-root', type=Path, default=BENCHMARK)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 64:
        parser.error('workers must be between 1 and 64')
    for task in args.tasks:
        tracked = {}
        _track(__file__, tracked)
        source, facts, retained = validate_source(task, args.workers, args.phase != 'source', tracked)
        result = {'task': task, 'phase': args.phase, 'source': source, 'model_evaluations_run': 0}
        if args.phase != 'source':
            result['indices'] = validate_indices(task, args.benchmark_root, facts, retained, tracked)
        for path, digest in tracked.items():
            assert sha256_file(Path(path)) == digest, ('input changed during validation', path)
        result['verified_inputs_sha256'] = tracked
        result['status'] = 'passed'
        write_json_atomic(ROOT/task/'progressive_v1'/f'validation_{args.phase}.json', result)


if __name__ == '__main__':
    main()
