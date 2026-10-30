"""Merge TDC train-label cards into a heldout-filtered Starling index in memory."""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from .json_utils import read_jsonl, sha256_file

from .budgeted_evidence import content_hash
from .molecule_identity import identity_from_record, normalize_molecule_identity
from .task_workflows.evidence_library import standardize_smiles_and_fp

LABELS = {
    'bbb_martins': ('BBB penetration', 'BBB-permeable', 'not BBB-permeable'),
    'bioavailability_ma': ('oral bioavailability', 'F >= 20%', 'F < 20%'),
    'skin_reaction': ('skin sensitization', 'sensitizer', 'non-sensitizer'),
    'ames': ('Ames mutagenicity', 'mutagenic', 'non-mutagenic'),
    'dili': ('drug-induced liver injury', 'DILI-positive', 'DILI-negative'),
    'carcinogens': ('rodent carcinogenicity', 'carcinogenic', 'non-carcinogenic'),
}


def parent_key(task, smiles):
    if task in ('dili', 'carcinogens'):
        from .starling.new_task_identity import leakage_identity
        return leakage_identity(task, smiles)['leakage_group']
    return normalize_molecule_identity(smiles).parent_connectivity_key


def merge_train_labels(index, task, train, heldout):
    """Merge before candidate truncation; no source quotas or selector changes."""
    forbidden = {parent_key(task, r['drug']) for r in heldout}
    entries, evidence = {}, defaultdict(lambda: defaultdict(list))
    for molecule, fp in zip(index['molecules'], index['fingerprints'], strict=True):
        parent = molecule.get('retrieval_leakage_group') or identity_from_record(molecule).parent_connectivity_key
        if not parent:
            continue
        native = molecule['molecule_chembl_id']
        entries.setdefault(parent, ({**molecule, 'molecule_chembl_id': parent}, fp))
        for group, rows in index['evidence_by_molecule_group'].get(native, {}).items():
            for row in rows:
                cards = row['source_record_examples']
                if parent in forbidden:
                    cards = [c for c in cards if int(c['evidence_family_level']) != 1]
                if cards:
                    evidence[parent][group].append({**row, 'source_record_examples': cards})
    labels = {}
    excluded = []
    for row in train:
        parent = parent_key(task, row['drug'])
        if not parent:
            raise ValueError('TDC train record lacks a parent identity')
        if parent in forbidden:
            excluded.append(row['benchmark_row_id'])
            continue
        identity = normalize_molecule_identity(row['drug'])
        key = (identity.canonical_smiles, int(row['Y']))
        if key in labels:
            labels[key]['source_row_ids'].append(row['benchmark_row_id'])
            continue
        labels[key] = {'parent': parent, 'row': row, 'source_row_ids': [row['benchmark_row_id']]}
    for (smiles, label), item in labels.items():
        parent, row = item['parent'], item['row']
        if parent not in entries:
            canonical, inchi, fp = standardize_smiles_and_fp(smiles)
            if fp is None:
                raise ValueError(f'invalid TDC train structure: {row["benchmark_row_id"]}')
            molecule = dict(molecule_chembl_id=parent, canonical_smiles=canonical,
                standard_inchi_key=inchi, molecule_identity=normalize_molecule_identity(smiles).to_dict())
            source = index.get('source', {})
            if source.get('retrieval_identity_contract'):
                from .starling.new_task_retrieval_identity import annotate_index, query_identity
                identity_contract = source['retrieval_identity_contract']
                annotate_index({'molecules': [molecule], 'source': {}}, task,
                    {canonical: query_identity(task, canonical, identity_contract)}, identity_contract)
            entries[parent] = (molecule, fp)
        card = train_label_card(task, row, smiles, item['source_row_ids'])
        evidence[parent]['TDC.train_label'].append(dict(canonical_smiles=smiles,
            molecule_chembl_id=parent, assay_chembl_id='TDC.train_label', group_id='TDC.train_label',
            source_record_examples=[card], evidence_source='TDC'))
    molecules, fingerprints, groups = [], [], defaultdict(list)
    for parent in sorted(evidence):
        molecule, fp = entries[parent]
        molecule = {**molecule, 'molecule_chembl_id': parent, 'groups': sorted(evidence[parent])}
        for group in evidence[parent]:
            groups[group].append(len(molecules))
        molecules.append(molecule)
        fingerprints.append(fp)
    return {**index, 'molecules': molecules, 'fingerprints': fingerprints,
        'evidence_by_molecule_group': {p: dict(rows) for p, rows in evidence.items()},
        'group_to_molecule_indices': dict(groups), 'tdc_merged': True,
        'tdc_train_audit': {'label_cards': len(labels), 'heldout_parent_excluded_row_ids': excluded}}


def train_label_card(task, row, smiles, source_row_ids):
    """One shared label-card format for merged retrieval and matched KNN."""
    label = int(row['Y'])
    endpoint, positive, negative = LABELS[task]
    meaning = positive if label else negative
    return dict(endpoint_type=endpoint, reported_value=str(label), reported_units='',
        assay_context='Dataset label; assay-level conditions not provided', species_context='', qualifying_conditions='',
        evidence_family='TDC.train_label', evidence_family_level=1,
        support_text=f'TDC {row["tdc_dataset"]} training-set label: {label} ({meaning}). This is a dataset classification; no assay-level conditions are supplied. It is not necessarily independent of other source records.',
        record_type='dataset_label', evidence_source='TDC', dataset=row['tdc_dataset'], split='train',
        source_record_id='tdc:' + content_hash([task, smiles, label]), source_row_ids=source_row_ids)


@lru_cache(maxsize=12)
def _bound_rows(path, digest):
    path = Path(path)
    if sha256_file(path) != digest:
        raise ValueError(f'matched KNN input changed: {path}')
    return read_jsonl(path)


def matched_knn_records(task, query, query_index, inputs):
    """Replay three saved neighbors; labels come only from hash-bound train rows.

    No heldout-parent or scaffold filter: this control follows the existing KNN
    exactly. Prediction/gold fields in its result file never enter the context.
    """
    train = _bound_rows(**dict(path=inputs['train']['path'], digest=inputs['train']['sha256']))
    results = _bound_rows(**dict(path=inputs['direct_knn']['path'], digest=inputs['direct_knn']['sha256']))
    saved = results[query_index]
    if (saved['query_index'] != query_index or saved['drug'] != query['drug']
            or saved['benchmark_row_id'] != query['benchmark_row_id']
            or saved.get('status') != 'ok' or len(saved['neighbors']) != 3):
        raise ValueError('matched KNN query identity/status/count mismatch')
    records, seen = [], set()
    for rank, neighbor in enumerate(saved['neighbors']):
        if neighbor['reference_split'] != 'train':
            raise ValueError('matched KNN must use train neighbors only')
        index = neighbor['train_index']
        if not isinstance(index, int) or not 0 <= index < len(train) or index in seen:
            raise ValueError('invalid or duplicate matched KNN train index')
        seen.add(index)
        row = train[index]
        if any(neighbor[k] != row[k] for k in ('drug', 'Y', 'benchmark_row_id', 'molecule_identity_key')):
            raise ValueError('matched KNN neighbor differs from TDC train row')
        smiles = row['drug']  # Preserve the exact KNN structure, including stereochemistry.
        card = train_label_card(task, row, smiles, [row['benchmark_row_id']])
        native = 'tdc-train:' + row['molecule_identity_key']
        records.append(dict(id=content_hash([task, row['benchmark_row_id'], card]),
            molecule_id=parent_key(task, smiles), native_molecule_id=native,
            canonical_smiles=smiles, record_smiles=smiles, evidence_source='TDC',
            molecule_names=[], assay_ids=['TDC.train_label'], group='TDC.train_label',
            kind='direct', card=card, similarity=neighbor['similarity'], knn_rank=rank,
            train_index=index))
    return records
