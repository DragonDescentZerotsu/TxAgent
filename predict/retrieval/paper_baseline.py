"""Frozen paper card pools; shared retrieval applies the approved identity rules."""
from collections import defaultdict

from predict.prompts.paper_baseline import content_hash
from predict.retrieval.policies import identity_from_record, normalize_molecule_identity, standardize_smiles_and_fp
from predict.retrieval.retrieve import retrieve_neighbors


def build_evidence_view(index):
    """Partition each physical assay's cards BEFORE applying identity policies."""
    positions = {str(m['molecule_chembl_id']): i for i, m in enumerate(index['molecules'])}
    parents, unresolved = {}, []
    for molecule in index['molecules']:
        key = str(molecule['molecule_chembl_id'])
        parent = molecule.get('retrieval_leakage_group') or identity_from_record(molecule).parent_connectivity_key
        if parent:
            parents[key] = parent
        else:
            unresolved.append(key)
    evidence, groups = {}, {'Evidence.direct': [], 'Evidence.indirect': []}
    for molecule, assays in index['evidence_by_molecule_group'].items():
        # Unresolved structures (e.g. wildcard fragments) cannot establish a
        # parent-disjoint relation. Remove them before ranking, allowing backfill.
        if str(molecule) not in parents:
            continue
        partition = defaultdict(list)
        for rows in assays.values():
            for row in rows:
                examples = defaultdict(list)
                for card in row.get('source_record_examples', []):
                    level = int(card.get('evidence_family_level') or 0)
                    if level < 1 or not card.get('evidence_family'):
                        raise ValueError('budgeted retrieval requires record-level family tags')
                    examples['Evidence.direct' if level == 1 else 'Evidence.indirect'].append(card)
                for group, cards in examples.items():
                    partition[group].append({**row, 'source_record_examples': cards})
        if partition:
            evidence[molecule] = dict(partition)
            for group in partition:
                groups[group].append(positions[str(molecule)])
    return {**index, 'group_to_molecule_indices': groups, 'evidence_by_molecule_group': evidence,
            'budgeted_parent_groups': parents, 'budgeted_unresolved_parent_ids': sorted(unresolved)}


def retrieve_candidates(smiles, view, *, split_scheme, candidate_molecules=128, min_similarity=.3,
                        indirect_identity_policy='parent_disjoint'):
    if split_scheme not in ('scaffold', 'random') or candidate_molecules < 1 or not 0 <= min_similarity <= 1:
        raise ValueError('invalid paper retrieval limits')
    if indirect_identity_policy not in ('parent_disjoint', 'scaffold_disjoint'):
        raise ValueError('unsupported indirect identity policy')
    result = {}
    for kind, count, policy in [('direct', candidate_molecules, 'scaffold_disjoint' if split_scheme == 'scaffold' else 'parent_disjoint'),
                                ('indirect', candidate_molecules, indirect_identity_policy)]:
        retrieval = retrieve_neighbors(smiles, view, top_k_per_group=count, min_similarity=min_similarity,
                                       groups=['Evidence.' + kind], neighbor_identity_policy=policy)
        if retrieval['status'] != 'ok':
            raise ValueError(f'invalid query structure: {smiles}')
        records = {}
        for neighbor in retrieval['groups'][0]['neighbors']:
            molecule = str(neighbor['molecule_chembl_id'])
            parent = view['budgeted_parent_groups'].get(molecule)
            if not parent:
                raise ValueError('retrieved molecule has no normalized parent')
            for row in neighbor['evidence_rows']:
                for example in row['source_record_examples']:
                    # The same rendered experiment across physical assays is one card.
                    key = content_hash([molecule, example, row['canonical_smiles']] if view.get('tdc_merged') else [molecule, example])
                    records.setdefault(key, {'id': key, 'molecule_id': parent, 'native_molecule_id': molecule,
                        'group': str(example['evidence_family']), 'kind': kind, 'card': dict(example),
                        'similarity': neighbor['similarity'], 'canonical_smiles': neighbor['canonical_smiles'],
                        **({'record_smiles': row['canonical_smiles'], 'evidence_source': row.get('evidence_source', 'Starling')} if view.get('tdc_merged') else {}),
                        'molecule_names': row.get('source_molecule_names') or row.get('molecule_names', []), 'assay_ids': []})
                    assay = str(row.get('assay_chembl_id') or row.get('group_id') or '')
                    if assay not in records[key]['assay_ids']:
                        records[key]['assay_ids'].append(assay)
        result[kind] = list(records.values())
    return result



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
        from predict.retrieval.task_identity import leakage_identity
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
                from predict.retrieval.task_identity import leakage_identity, validate_contract
                identity_contract = source['retrieval_identity_contract']
                validate_contract(task, identity_contract)
                value = leakage_identity(task, canonical)
                molecule.update(retrieval_identity_source_smiles=canonical,
                    retrieval_identity_contract=identity_contract,
                    retrieval_leakage_group=value['leakage_group'],
                    retrieval_scaffold_leakage_group=value['scaffold_leakage_group'],
                    retrieval_scaffold_smiles=value['bemis_murcko_scaffold'])
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
