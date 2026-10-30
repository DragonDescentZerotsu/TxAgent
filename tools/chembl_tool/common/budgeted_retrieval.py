"""Direct/indirect views and exact source binding for budgeted selection.

The frozen index remains the eligibility authority. Native source records only
supply features for cards already present in the retrieved candidate pool.
"""
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import zlib

from .assay_retrieval import NATIVE_CARD_FIELDS, _NATIVE_MISSING, _clean, _representative_records
from .budgeted_evidence import content_hash
from .molecule_identity import identity_from_record
from .record_diversity import prepare_record
from .task_workflows.retrieve_neighbors import retrieve_neighbors

REGISTRY = Path("tools/chembl_tool/paper_experiments/current_starling_retrieval.json")


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


def source_paths(task):
    registry = json.loads(REGISTRY.read_text())
    spec, root = registry['tasks'][task], Path(registry['artifact_root'])
    source = Path(spec['source_records']) if spec.get('overlay_is_canonical') else root / spec['overlay'] / 'records.parquet'
    return source, root / spec['catalog'] / 'manifest.json'


def _raw_ames_rows(rows):
    """Read only required source row groups; source_row_number is zero based."""
    import pyarrow.parquet as pq
    positions, result = defaultdict(set), {}
    for row in rows:
        positions[row['source_id']].add(int(row['source_row_number']))
    for source, wanted in positions.items():
        parquet = pq.ParquetFile(Path('data/starling_data/ames/raw_v1') / (source + '.parquet'))
        offset = 0
        for group in range(parquet.num_row_groups):
            size = parquet.metadata.row_group(group).num_rows
            selected = sorted(p for p in wanted if offset <= p < offset + size)
            if selected:
                values = parquet.read_row_group(group).take([p-offset for p in selected]).to_pylist()
                result.update({(source, p): value for p, value in zip(selected, values)})
            offset += size
        if any((source, p) not in result for p in wanted):
            raise ValueError('Ames native source row missing')
    return result


def bind_source_records(task, candidates):
    """Match exact indexed card surfaces, never approximate names or prose.

    Multiple source rows with the same surface stay in provenance. Their
    native inputs are retained so the scorer can handle ambiguity conservatively.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    candidates = {r['id']: r for r in candidates}
    if not candidates:
        return {}
    source, catalog = source_paths(task)
    families = [{'source_group_id': group, 'level': int(level['level']),
                 'endpoint_group': level.get('endpoint_group') or level.get('family_id')}
                for level in json.loads(catalog.read_text())['levels'] for group in level['source_groups']]
    molecules = pa.array(sorted({r['native_molecule_id'] for r in candidates.values()}))
    matched = defaultdict(list)
    native_fields = NATIVE_CARD_FIELDS.get(task, {})
    for batch in pq.ParquetFile(source).iter_batches(batch_size=8192, use_threads=False):
        table = pa.Table.from_batches([batch])
        mask = pc.and_(pc.is_in(table['molecule_id'], value_set=molecules),
                       pc.fill_null(table['retrieval_eligible'], False))
        for row in table.filter(mask).to_pylist():
            for destination, keys in native_fields.items():
                row['_native_' + destination] = ' | '.join(f'{key}={_clean(row[key])}' for key in keys
                    if key in row and _clean(row[key]) and not _NATIVE_MISSING.fullmatch(_clean(row[key])))
            example = _representative_records([row], limit=0, assay_metadata={'source_families': families})[0]
            key = content_hash([str(row['molecule_id']), example])
            if key in candidates:
                matched[key].append(row)
    missing = set(candidates) - set(matched)
    if missing:
        raise ValueError(f'{task}: {len(missing)} indexed cards do not bind to the current raw source; rebuild an uncapped index. Examples: {sorted(missing)[:3]}')
    raw = _raw_ames_rows([row for rows in matched.values() for row in rows]) if task == 'ames' else {}
    result = {}
    for key, rows in matched.items():
        prepared = [{**prepare_record(row, raw.get((row.get('source_id'), row.get('source_row_number')))),
                     'task':task, 'source_id':row.get('source_id',''),
                     'level':candidates[key]['card']['evidence_family_level']} for row in rows]
        prepared.sort(key=lambda p: p['record_id'])
        chosen = prepared[0]
        result[key] = {'prepared': chosen, 'source_record_ids': sorted(p['record_id'] for p in prepared),
                       'alternatives': prepared,
                       'source_input_hashes': sorted({content_hash(p['input']) for p in prepared}),
                       'ambiguous_native_metadata': len({content_hash(p['input']) for p in prepared}) > 1}
    return result


def cached_annotations(path, task, bindings):
    """Only adopt an annotation whose native input is exactly current."""
    if not path:
        return {}
    from .record_diversity import (validate_annotation, ANNOTATION_VERSION,
                                   QUALITY_ANNOTATION_VERSION, extraction_messages)
    import hashlib
    output = {}
    with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True) as db:
        contract = db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
        if not contract:
            raise ValueError('annotation cache is missing its extraction contract')
        contract = json.loads(contract[0])
        version = contract['model_config'].get('annotation_version', ANNOTATION_VERSION)
        if version == QUALITY_ANNOTATION_VERSION:
            expected = hashlib.sha256(extraction_messages({'input':{}}, version)[0]['content'].encode()).hexdigest()
            if contract.get('system_prompt_sha256') != expected:
                raise ValueError('assessed annotation cache uses a different extraction prompt; regenerate with the current contract')
        for key, bound in bindings.items():
            matched = {}
            alternatives = bound.get('alternatives', [bound['prepared']])
            for prepared in alternatives:
                rows = db.execute('SELECT j.payload,a.annotation FROM records r JOIN jobs j ON j.id=r.job_id '
                                  'JOIN annotations a ON a.job_id=j.id WHERE r.task=? AND r.record_id=?',
                                  (task, prepared['record_id'])).fetchall()
                for payload, annotation in rows:
                    payload, annotation = json.loads(zlib.decompress(payload)), json.loads(zlib.decompress(annotation))
                    if payload['input'] == prepared['input']:
                        matched[content_hash(prepared['input'])] = validate_annotation(annotation, prepared, annotation_version=version)
            expected = {content_hash(p['input']) for p in alternatives}
            if expected <= matched.keys():
                output[key] = ({'alternatives':matched} if bound.get('ambiguous_native_metadata') else
                               matched[content_hash(bound['prepared']['input'])])
    return output


def write_feature_inputs(root, task, bindings):
    """Export exact retrieved native inputs to the shared offline extractor.

    Export every bound alternative; a feature cache cannot resolve ambiguous
    source identities by choosing an arbitrary matching row.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    records = {}
    for bound in bindings.values():
        for prepared in bound.get('alternatives', [bound['prepared']]):
            rid = prepared['record_id']
            if rid in records and records[rid]['input'] != prepared['input']:
                raise ValueError(f'conflicting native inputs for source record {rid}')
            records[rid] = {**prepared, 'needs_llm': True, 'route':'llm_candidate'}
    rows = [{'record_id':rid, 'source_id':p.get('source_id',''), 'level':p.get('level',0),
             'route':'llm_candidate', 'payload_json':json.dumps(p, ensure_ascii=False, separators=(',',':'))}
            for rid,p in sorted(records.items())]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([('record_id',pa.string()),('source_id',pa.string()),('level',pa.int64()),
                        ('route',pa.string()),('payload_json',pa.string())])
    target = root / f'{task}.features.parquet'
    temporary = target.with_suffix('.tmp.parquet')
    pq.write_table(pa.Table.from_pylist(rows,schema=schema),temporary,compression='zstd')
    temporary.replace(target)
    return {'task':task, 'rows':len(rows), 'llm_candidates':len(rows)}
