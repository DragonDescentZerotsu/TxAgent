"""Pin current V10 files and project eligible source records without a checkout dependency."""

import gzip
import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

from .storage import PACKAGE, digest, file_sha256, read_json, write_json
from .canonical import TASKS, task_contract
from .chemistry import normalize_molecule_identity

FILES = {'release_manifest': 'manifest.json', 'stage3_manifest': '03_pair_buckets/manifest.json',
         'records': '03_pair_buckets/records.parquet', 'source_contract': '02_canonicalized/source_contract.json',
         'upstream_calibration': '03_pair_buckets/pair_bucket_distance_calibration.json.gz',
         'level_manifest': 'level_mapping/manifest.json', 'levels': 'level_mapping/records.parquet',
         'record_eligibility': 'level_mapping/assay_transfer_record_eligibility.parquet'}


@lru_cache(maxsize=200_000)
def identity(smiles):
    value = normalize_molecule_identity(smiles)
    if value.status != 'ok' or not value.parent_smiles:
        raise ValueError(f'Invalid normalized parent: {smiles}')
    parent = value.parent_inchi_key or value.parent_smiles
    with rdBase.BlockLogs():
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(value.parent_smiles))
    return parent, value.parent_smiles, scaffold or f'parent:{parent}'


def pin(source_root, output, contract, tasks=TASKS, gold_only=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    source_root = Path(source_root).resolve()
    document = {'schema_version': 'assay_transfer.inputs.v1', 'tasks': {}}
    for name in tasks:
        policy = task_contract(name, contract)
        release, paths = None, {}
        if not gold_only:
            release, paths = evidence_paths(source_root, policy)
        gold = source_root / 'gold_labels' / name / 'v1/scaffold'
        for split in ('train', 'valid', 'test'):
            preferred = gold / f'{split}_molecule_condition_labels.jsonl'
            paths[f'gold_{split}'] = preferred if preferred.exists() else gold/f'{split}.jsonl'
        if (gold/'gold_label_record_index.parquet').exists():
            paths['gold_index'] = gold/'gold_label_record_index.parquet'
        if (gold / 'summary.json').exists():
            paths['gold_summary'] = gold / 'summary.json'
        if not gold_only and policy['slug'] == 'skin_reaction':
            paths['skin_routes'] = source_root / 'artifacts/evidence_library_assets/skin_reaction/construction_assets/progressive_level_mapping_v1/record_levels.parquet'
        if not gold_only:
            paths['geometry_policy'] = PACKAGE / 'assets' / f'{policy["slug"].replace("bioavailability_ma", "oral").replace("bbb_martins", "bbb").replace("skin_reaction", "skin")}_geometry.json'
        files = {role: {'path': str(path.resolve()), 'sha256': file_sha256(path), 'bytes': path.stat().st_size}
                 for role, path in paths.items()}
        document['tasks'][name] = {'evidence_release': str(release) if release else None, 'files': files}
        print(f'Pinned {name}: {release.name if release else "Gold-v1 only"}', flush=True)
    document['inputs_sha256'] = digest(document)
    write_json(output, document)
    return document


def source_fields(document):
    output = {}
    for source, spec in document['sources'].items():
        columns = spec.get('normalized_artifact_columns')
        output[source] = ([key for key, value in columns.items() if value.get('source_or_simply_cleaned')]
                          if columns else spec.get('source_visible_fields', spec.get('cleaned_source_fields', [])))
    return output


def raw_rows(files, smoke=False):
    paths = {key: value['path'] for key, value in files.items()}
    with gzip.open(paths['upstream_calibration'], 'rt') as handle:
        valid = [key for key, value in json.load(handle)['buckets'].items() if value.get('calibration_valid')]
    with duckdb.connect() as connection:
        connection.execute('CREATE TEMP TABLE valid(bucket VARCHAR)')
        connection.executemany('INSERT INTO valid VALUES (?)', [(key,) for key in valid])
        query = '''SELECT r.*, m.level AS mapped_level FROM read_parquet(?) r
                   LEFT JOIN read_parquet(?) m USING(source_row_uid)
                   LEFT JOIN read_parquet(?) e USING(source_row_uid)
                   JOIN valid v ON r.pair_bucket_key=v.bucket
                   WHERE r.assay_transfer_eligible AND (e.assay_transfer_eligible OR m.level IS NULL)'''
        parameters = [paths['records'], paths['levels'], paths['record_eligibility']]
        if smoke:
            buckets = connection.execute(f'''SELECT pair_bucket_key, mapped_level FROM (
                    SELECT pair_bucket_key, mapped_level, measurement_kind, count(*) n,
                    row_number() OVER(PARTITION BY mapped_level, measurement_kind, count(*) <= 11 ORDER BY count(*) DESC, pair_bucket_key) ranking
                    FROM ({query}) GROUP BY pair_bucket_key, mapped_level, measurement_kind
                    HAVING count(*) BETWEEN 9 AND 1000) WHERE ranking <= 2 ORDER BY pair_bucket_key''', parameters).fetchall()
            connection.execute('CREATE TEMP TABLE selected(bucket VARCHAR, level BIGINT)')
            connection.executemany('INSERT INTO selected VALUES (?, ?)', [(row[0],row[1]) for row in buckets])
            query = f'SELECT base.* FROM ({query}) base JOIN selected s ON base.pair_bucket_key=s.bucket AND base.mapped_level IS NOT DISTINCT FROM s.level'
        reader = connection.execute(query+' ORDER BY source_row_uid', parameters).fetch_record_batch(2000)
        for batch in reader:
            yield from batch.to_pylist()


def gold_groups(files):
    output = {}
    voters = {}
    if 'gold_index' in files:
        rows = pq.read_table(files['gold_index']['path']).to_pylist()
        voters = {row['benchmark_row_id']:row for row in rows}
        if len(voters)!=len(rows):
            raise ValueError('Duplicate Gold voter identities')
    for split in ('train', 'valid', 'test'):
        groups = []
        with Path(files[f'gold_{split}']['path']).open() as handle:
            for line in handle:
                raw = json.loads(line)
                parent, smiles, scaffold = identity(raw.get('molecule_identity', {}).get('parent_smiles') or raw['drug'])
                count, value = gold_value(raw,voters)
                condition = raw.get('condition_group','no_reported_external_condition')
                atoms = raw.get('condition_atoms',[] if condition=='no_reported_external_condition' else condition.split('+'))
                groups.append({'record_id': raw.get('benchmark_row_id') or f'{split}:{parent}',
                               'parent_id': parent, 'parent_smiles': smiles, 'scaffold': scaffold,
                               'geometry_value': value, 'Y':int(raw['Y']), 'source_record_count':count, 'condition_atoms': atoms,
                               'condition_group': raw.get('condition_group', ''), 'raw_json': json.dumps(raw)})
        if len({row['record_id'] for row in groups}) != len(groups):
            raise ValueError(f'Duplicate gold rows: {split}')
        output[split] = groups
    if {row['parent_id'] for row in output['valid']} & {row['parent_id'] for row in output['test']}:
        raise ValueError('Gold validation/test parents overlap')
    return output


def skin_routes(files):
    routes = {}
    for row in pq.read_table(files['skin_routes']['path']).to_pylist():
        key = row['source_id'], row['source_row_number'], row['source_record_id'] or ''
        if key in routes and routes[key] != row['progressive_level']:
            raise ValueError(f'Conflicting Skin route: {key}')
        routes[key] = row['progressive_level']
    return routes


def gold_value(raw, voters):
    if 'label_counts' in raw:
        count = int(raw['source_record_count'])
        counts = raw['label_counts']
        if count<=0 or sum(counts.values())!=count or set(counts)-{'0','1'} or any(n<0 for n in counts.values()):
            raise ValueError('Invalid Gold-v1 vote counts')
        return count,int(counts.get('1',0))/count
    voter = voters[raw['benchmark_row_id']]
    count, mean = int(voter['vote_unit_count']),float(voter['voter_mean'])
    positive = round(mean*count)
    if count<=0 or not math.isfinite(mean) or not 0<=mean<=1 or not math.isclose(positive,mean*count,abs_tol=1e-10):
        raise ValueError('Invalid pinned Gold voter mean/count')
    return count,positive/count


def evidence_paths(source_root, policy):
    current = source_root / 'evidence_libraries' / policy['slug'] / 'CURRENT'
    release = current.resolve() if current.is_symlink() else current.parent / current.read_text().strip()
    if not release.name.startswith('v10'):
        raise ValueError(f'Current evidence is not V10: {release}')
    complete = read_json(release / 'manifest.json')
    if complete.get('completed_stages') != ['source', 'clean', 'normalize'] or '03_pair_buckets' not in complete.get('completed_artifact_stages', []):
        raise ValueError(f'Incomplete V10 evidence: {release}')
    return release.resolve(), {role: release / relative for role, relative in FILES.items()}
