"""Morgan-neighbor features from main labels and frozen indirect target pools."""

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
import torch

from predict.baselines.conditioned_knn import select_conditioned_ranked_indices
from predict.baselines.minimol.indirect_data import parent_keys
from predict.baselines.structure_knn.run import FP_BITS, FP_RADIUS
from predict.utils.json import sha256_file, write_json_atomic


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def molecule(smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FP_RADIUS, fpSize=FP_BITS, includeChirality=False, useBondTypes=True)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid structure: {smiles}")
    return parent_keys(smiles), generator.GetFingerprint(mol)


def bucket_neighbors(similarities, target_molecules, starts):
    """Segmented argmax; ties choose the first frozen parent-sorted target."""
    scores = similarities[target_molecules]
    maxima = np.maximum.reduceat(scores, starts)
    sizes = np.diff(np.r_[starts, len(scores)])
    candidates = np.where(scores == np.repeat(maxima, sizes),
                          np.arange(len(scores)), len(scores))
    selected = np.minimum.reduceat(candidates, starts).astype(np.int32)
    selected[~np.isfinite(maxima)] = -1
    return selected, maxima


def direct_neighbors(similarities, reference, query, policy):
    ranked = np.argsort(-similarities, kind="stable")
    ranked = ranked[np.isfinite(similarities[ranked])].tolist()
    selected = [i for i, _ in select_conditioned_ranked_indices(
        ranked, reference, query, k=3, policy=policy)]
    # Condition priority chooses membership; feature positions always rank by similarity.
    return sorted(selected, key=lambda i: (-similarities[i], i))


_CONTEXT = None


def _query(job):
    smiles, row_ids = job
    c = _CONTEXT
    parent, connectivity = c['identities'][smiles]
    scores = np.asarray(DataStructs.BulkTanimotoSimilarity(
        c['fingerprints'][smiles], c['all_fps']), dtype=np.float64)
    for key in (parent, connectivity):
        if key:
            scores[c['identity_molecules'][key]] = -np.inf
    neighbors, similarities = bucket_neighbors(scores, c['target_molecules'], c['starts'])
    aux = np.full((len(neighbors), 2), np.nan, dtype=np.float32)
    valid = neighbors >= 0
    aux[valid, 0] = c['targets'][neighbors[valid]]
    aux[valid, 1] = similarities[valid]
    direct = np.full((len(row_ids), 2, 6), np.nan, dtype=np.float32)
    indices = np.full((len(row_ids), 2, 3), -1, dtype=np.int32)
    for j, row_id in enumerate(row_ids):
        for policy_id, policy in enumerate(c['policies']):
            selected = direct_neighbors(scores[c['reference_molecules']], c['reference'],
                                        c['rows'][row_id], policy)
            for rank, i in enumerate(selected):
                direct[j, policy_id, 2 * rank:2 * rank + 2] = (
                    c['reference'][i]['Y'], scores[c['reference_molecules'][i]])
                indices[j, policy_id, rank] = i
    return row_ids, aux.reshape(-1), neighbors, direct, indices


def prepare(data_dir, source_prepared, output, condition_policy, workers, *, auxiliary=None):
    """Reuse an audited MiniMol target cache; never use its learned embeddings."""
    global _CONTEXT
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Feature output must be empty")
    source = json.loads((source_prepared / 'manifest.json').read_text())
    if sha256_file(source_prepared / 'data.pt') != source['data_sha256']:
        raise ValueError("Indirect target cache changed")
    for path, digest in source['inputs'].items():
        relocated = (Path(auxiliary or path) if path == source['auxiliary'] else
                     data_dir / Path(path).relative_to(source['data_dir']))
        if sha256_file(relocated) != digest:
            raise ValueError(f"Prepared source input changed: {path}")
    data = torch.load(source_prepared / 'data.pt', map_location='cpu', weights_only=False)
    split_bytes = {s: (data_dir / f'{s}.jsonl').read_bytes() for s in ('train', 'valid', 'test')}
    splits = {s: [json.loads(line) for line in content.splitlines()] for s, content in split_bytes.items()}
    for split, rows in splits.items():
        packed = data['splits'][split]
        if ([data['smiles'][int(i)] for i in packed['indices']] != [r['drug'] for r in rows]
                or packed['labels'].tolist() != [r['Y'] for r in rows]):
            raise ValueError(f"Main split differs from source cache: {split}")
    rows = sum(splits.values(), [])
    smiles = sorted(set(r['drug'] for r in rows) |
                    {data['smiles'][int(i)] for i in data['auxiliary']['indices']})
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
        structures = list(pool.map(molecule, smiles, chunksize=32))
    identities = {s: value[0] for s, value in zip(smiles, structures)}
    fingerprints = {s: value[1] for s, value in zip(smiles, structures)}
    lookup = {s: i for i, s in enumerate(smiles)}
    aux = data['auxiliary']
    target_molecules = np.array([lookup[data['smiles'][int(i)]] for i in aux['indices']])
    heldout = {k for s in ('valid', 'test') for row in splits[s]
               for k in identities[row['drug']] if k}
    for i in np.unique(target_molecules):
        if any(k in heldout for k in identities[smiles[i]] if k):
            raise ValueError("Auxiliary pool contains heldout parent/connectivity")
    bids = aux['buckets'].numpy()
    if not np.all(bids[1:] >= bids[:-1]) or set(bids) != set(range(len(data['buckets']))):
        raise ValueError("Auxiliary targets must be grouped by bucket")
    starts = np.r_[0, np.flatnonzero(np.diff(bids)) + 1]
    reference = [{**row, 'molecule_identity_key': identities[row['drug']][0]}
                 for row in splits['train']]
    policies = ['all_train_unique_molecules', condition_policy]
    identity_molecules = {}
    for i, s in enumerate(smiles):
        for key in identities[s]:
            if key:
                identity_molecules.setdefault(key, []).append(i)
    _CONTEXT = dict(identities=identities, fingerprints=fingerprints,
                    all_fps=[fingerprints[s] for s in smiles],
                    identity_molecules=identity_molecules,
                    reference=reference, reference_molecules=np.array([lookup[r['drug']] for r in reference]),
                    target_molecules=target_molecules, targets=aux['targets'].numpy(),
                    starts=starts, rows=rows, policies=policies)
    jobs = {}
    for i, row in enumerate(rows):
        jobs.setdefault(row['drug'], []).append(i)
    shapes = {'auxiliary': ((len(rows), 2 * len(starts)), np.float32),
              'auxiliary_neighbors': ((len(rows), len(starts)), np.int32),
              'direct': ((len(rows), 2, 6), np.float32),
              'direct_neighbors': ((len(rows), 2, 3), np.int32)}
    arrays = {name: np.lib.format.open_memmap(output / f'{name}.npy', mode='w+',
                                             dtype=dtype, shape=shape)
              for name, (shape, dtype) in shapes.items()}
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
        for count, (ids, values, neighbors, direct, direct_ids) in enumerate(
                pool.map(_query, jobs.items(), chunksize=8), 1):
            arrays['auxiliary'][ids] = values
            arrays['auxiliary_neighbors'][ids] = neighbors
            arrays['direct'][ids] = direct
            arrays['direct_neighbors'][ids] = direct_ids
            if count % 500 == 0:
                print(f"Prepared {count}/{len(jobs)} query structures", flush=True)
    for a in arrays.values():
        a.flush()
    policies_identical = np.array_equal(arrays['direct'][:, 0], arrays['direct'][:, 1], equal_nan=True)
    offset, offsets = 0, {}
    for split, rs in splits.items():
        offsets[split] = [offset, offset + len(rs)]
        offset += len(rs)
    (output / 'splits').mkdir()
    for split, content in split_bytes.items():
        (output / 'splits' / f'{split}.jsonl').write_bytes(content)
    write_json_atomic(output / 'manifest.json', dict(
        version='retrieval_forest_features.v2', data_dir='splits',
        source_prepared=str(source_prepared), source_manifest_sha256=sha256_file(source_prepared / 'manifest.json'),
        inputs={f'splits/{s}.jsonl': sha256_file(output / 'splits' / f'{s}.jsonl') for s in splits},
        files={f'{name}.npy': sha256_file(output / f'{name}.npy') for name in arrays},
        offsets=offsets, buckets=len(starts), features=6 + 2 * len(starts), policies=policies,
        policies_identical=policies_identical, auxiliary_heldout_overlap=0,
        exclusion='All queries: exclude own normalized parent/connectivity from direct and auxiliary donors; auxiliary pool also excludes valid+test union',
        direct_order='Select with existing condition policy, then descending Morgan similarity; stable source-row ties',
        auxiliary_order='Each bucket independently: maximum Morgan similarity, stable frozen parent order for ties',
        feature_schema='[label,similarity] x 3 + [standardized target,similarity] x bucket; unavailable pairs are NaN',
        fingerprint=dict(radius=FP_RADIUS, bits=FP_BITS, chirality=False),
        missing_auxiliary_pairs=int(np.isnan(arrays['auxiliary'][:, ::2]).sum()),
        incomplete_direct_queries=[int(np.isnan(arrays['direct'][:, p, ::2]).any(axis=1).sum()) for p in range(2)],
    ))
    _CONTEXT = None


def load_features(prepared, policy, feature_set='direct_indirect', *, data_dir=None):
    if feature_set not in ('direct', 'direct_indirect'):
        raise ValueError(f"Unknown feature set: {feature_set}")
    manifest = json.loads((prepared / 'manifest.json').read_text())
    input_root = prepared if manifest.get('version') == 'retrieval_forest_features.v2' else Path('.')
    original_data_dir = Path(manifest['data_dir'])
    data_dir = Path(data_dir) if data_dir is not None else input_root / original_data_dir
    for file, digest in manifest['files'].items():
        if feature_set == 'direct' and file not in ('direct.npy', 'direct_neighbors.npy'):
            continue
        if sha256_file(prepared / file) != digest:
            raise ValueError(f"Feature cache changed: {file}")
    for path, digest in manifest['inputs'].items():
        relocated = data_dir / Path(path).relative_to(original_data_dir)
        if sha256_file(relocated) != digest:
            raise ValueError(f"Main data changed: {path}")
    direct = np.load(prepared / 'direct.npy', mmap_mode='r')[:, policy]
    if feature_set == 'direct':
        features = direct
    else:
        auxiliary = np.load(prepared / 'auxiliary.npy', mmap_mode='r')
        features = np.concatenate((direct, auxiliary), axis=1)
    manifest = dict(manifest, feature_set=feature_set, features=features.shape[1])
    rows = {s: read_rows(data_dir / f'{s}.jsonl') for s in manifest['offsets']}
    return manifest, {s: features[a:b] for s, (a, b) in manifest['offsets'].items()}, rows
