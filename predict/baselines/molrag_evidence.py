"""Freeze direct reference pools from the historical naive baseline's source indexes."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from predict.baselines.molrag import digest
from predict.retrieval.paper_baseline import merge_train_labels, parent_key
from predict.retrieval.policies import normalize_molecule_identity
from predict.utils.json import read_json, read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic


def build_library(sources, source_root, output):
    specification = read_json(sources)
    output.mkdir(parents=True, exist_ok=True)
    datasets = []
    for spec in specification['datasets']:
        benchmark, task = spec['benchmark'], spec['task']
        destination = output / benchmark / task
        destination.mkdir(parents=True, exist_ok=True)
        if (destination / 'receipt.json').exists():
            receipt = read_json(destination / 'receipt.json')
            if receipt['source'] != spec or sha256_file(destination / 'references.jsonl') != receipt['sha256']:
                raise ValueError('Existing reference pool changed')
            datasets.append(receipt)
            continue
        original = read_json(source_root / spec['batch'] / 'experiment_manifest.json')
        inputs = original['inputs'][task]
        if original['split_scheme'] != 'scaffold' or original['setting'] != 'direct':
            raise ValueError('Expected the frozen scaffold direct baseline')
        index_manifest = (source_root / inputs['index']['path']).parent / 'manifest.json'
        index = read_json(index_manifest)
        if index['index_sha256'] != inputs['index']['sha256'] or index['n_direct_heldout_records_after_filter'] != 0:
            raise ValueError('Source index or held-out filtering differs from the naive baseline')
        heldout = read_jsonl(Path(index['heldout_molecules_jsonl']))
        forbidden = {parent_key(task, row['drug']) for row in heldout}
        references, identities, counts = {}, {}, Counter()

        def add(smiles, card, provenance):
            identity = identities.setdefault(smiles, None)
            if identity is None:
                identity = identities[smiles] = normalize_molecule_identity(smiles)
            if identity.status != 'ok' or not identity.parent_connectivity_key:
                counts['unresolved_structure'] += 1
                return
            # Apply the historical held-out identity contract, then full-molecule
            # scaffold exclusion at query selection time.
            if parent_key(task, smiles) in forbidden:
                counts['heldout_excluded'] += 1
                return
            canonical = identity.canonical_smiles
            counts['eligible_label_cards' if card.get('record_type') == 'dataset_label' else 'eligible_evidence_cards'] += 1
            if canonical in references:
                counts['additional_records_same_molecule'] += 1
                return
            references[canonical] = dict(reference_id=digest([benchmark, task, canonical, card]),
                smiles=canonical, canonical_smiles=canonical, molecule_identity=identity.to_dict(),
                card=card, provenance=provenance)

        # The JSONL is the hash-bound evidence sidecar of the exact frozen index.
        # Stream it instead of loading multi-GB pickle objects or rebuilding evidence.
        hasher = hashlib.sha256()
        with Path(index['evidence']).open('rb') as handle:
            for line_number, line in enumerate(handle, 1):
                hasher.update(line)
                if b'"evidence_family_level": 1' not in line and b'"evidence_family_level":1' not in line:
                    continue
                row = json.loads(line)
                for position, card in enumerate(row['source_record_examples']):
                    if int(card['evidence_family_level']) != 1:
                        continue
                    add(row['canonical_smiles'], card, dict(kind='experimental_evidence',
                        evidence_file=index['evidence'], line=line_number, card_index=position,
                        assay_id=row['assay_chembl_id'], source=row.get('evidence_source', ''),
                        source_row_uid=card.get('source_row_uid')))
        if hasher.hexdigest() != index['evidence_sha256']:
            raise ValueError('Frozen index evidence sidecar changed')
        if benchmark == 'tdc':
            for name in ('train', 'heldout'):
                if sha256_file(source_root / inputs[name]['path']) != inputs[name]['sha256']:
                    raise ValueError('Frozen TDC split changed')
            train = read_jsonl(source_root / inputs['train']['path'])
            heldout = read_jsonl(source_root / inputs['heldout']['path'])
            labels = merge_train_labels(dict(molecules=[], fingerprints=[], evidence_by_molecule_group={}), task, train, heldout)
            for groups in labels['evidence_by_molecule_group'].values():
                for rows in groups.values():
                    for row in rows:
                        for card in row['source_record_examples']:
                            add(row['canonical_smiles'], card, dict(kind='tdc_train_label',
                                source_row_ids=card['source_row_ids'], source_record_id=card['source_record_id']))
        frozen = sorted(references.values(), key=lambda row: row['canonical_smiles'])
        if not frozen:
            raise ValueError('Empty direct reference pool')
        write_jsonl_atomic(destination / 'references.jsonl', frozen)
        receipt = dict(benchmark=benchmark, task=task, source=spec,
            source_manifest_sha256=sha256_file(source_root / spec['batch'] / 'experiment_manifest.json'),
            index_manifest_sha256=sha256_file(index_manifest), index_sha256=inputs['index']['sha256'],
            evidence_sha256=hasher.hexdigest(), counts=dict(counts), n_molecules=len(frozen),
            selected_record_types=dict(Counter(r['provenance']['kind'] for r in frozen)),
            path=str((destination / 'references.jsonl').relative_to(output)),
            sha256=sha256_file(destination / 'references.jsonl'))
        write_json_atomic(destination / 'receipt.json', receipt)
        datasets.append(receipt)
        print(json.dumps({k: receipt[k] for k in ('benchmark', 'task', 'n_molecules', 'selected_record_types')}), flush=True)
    write_json_atomic(output / 'manifest.json', dict(release=specification['release'], datasets=datasets,
        record_policy='first_frozen_evidence_record_then_tdc_train_label_per_full_molecule',
        provenance_note='Legacy cards without physical row UIDs retain exact evidence file/line/card coordinates; no UIDs are fabricated.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build_library(args.sources, args.source_root, args.output)
