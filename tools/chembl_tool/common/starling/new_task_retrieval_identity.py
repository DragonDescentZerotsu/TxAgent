"""Explicit new-task retrieval identity cache; legacy task identity is unchanged."""
from __future__ import annotations

import argparse
import fcntl
from functools import lru_cache
from pathlib import Path

from rdkit import rdBase

from tools.chembl_tool.common.build_runtime import local_input, worker_pool
from tools.chembl_tool.common.json_utils import read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic
from tools.chembl_tool.common.retrieval_policy import CandidateDecision, MoleculeRelation
from tools.chembl_tool.common.starling import new_task_identity as adapter


def validate_contract(task, contract):
    if task not in adapter.TASKS or contract != adapter.VERSION:
        raise ValueError(f'Unsupported opt-in retrieval identity: {task}/{contract}')


def _lineage(task):
    return {'task': task, 'identity_contract': adapter.VERSION,
            'adapter_sha256': sha256_file(Path(adapter.__file__)),
            'rdkit_version': rdBase.rdkitVersion}


def _compute(item):
    task, smiles = item
    try:
        value = adapter.leakage_identity(task, smiles)
        if not value.get('leakage_group') or 'scaffold_leakage_group' not in value:
            raise ValueError('Adapter did not return a complete leakage identity')
        return {'input_smiles': smiles, 'identity': value}, None
    except ValueError as exc:
        return None, {'input_smiles': smiles, 'reason': str(exc)}


def prepare_cache(task, smiles, path, workers=16):
    """Normalize each unique molecular form once; serialize competing builders."""
    validate_contract(task, adapter.VERSION)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = _lineage(task)
    values = sorted({str(value) for value in smiles if value})
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = load_cache(task, path) if path.exists() else {}
        missing = [value for value in values if value not in cached]
        pending = []
        if missing:
            with worker_pool(min(workers, len(missing))) as pool:
                for row, error in pool.map(_compute, ((task, value) for value in missing), chunksize=8):
                    if error:
                        pending.append(error)
                    else:
                        cached[row['input_smiles']] = row['identity']
            write_jsonl_atomic(path, [{'input_smiles': key, 'identity': value} for key, value in sorted(cached.items())])
            write_json_atomic(path.with_suffix('.manifest.json'), {
                **expected, 'cache_sha256': sha256_file(path), 'unique_smiles': len(cached),
                'identity_unit': 'unique_smiles', 'source_records_modified': 0,
            })
        write_jsonl_atomic(path.with_suffix('.pending.jsonl'), pending)
        if pending:
            raise ValueError(f'{task}: {len(pending)} unresolved retrieval identities; see {path.with_suffix(".pending.jsonl")}')
        assert expected == _lineage(task), 'Identity adapter changed during cache computation'
    return cached


def load_cache(task, path):
    import json
    path = Path(path)
    metadata = json.loads(path.with_suffix('.manifest.json').read_text())
    for key, value in _lineage(task).items():
        if metadata.get(key) != value:
            raise ValueError(f'Stale retrieval identity cache {path}: {key}')
    if sha256_file(path) != metadata['cache_sha256']:
        raise ValueError(f'Retrieval identity cache hash mismatch: {path}')
    rows = read_jsonl(local_input(path))
    result = {row['input_smiles']: row['identity'] for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate retrieval identity cache input')
    return result


def annotate_index(index, task, cache, contract, source_smiles_by_id=None):
    validate_contract(task, contract)
    by_parent = {value['source_parent_smiles']: value for value in cache.values()}
    for molecule in index['molecules']:
        smiles = (source_smiles_by_id or {}).get(molecule['molecule_chembl_id'], molecule['canonical_smiles']) if source_smiles_by_id is not None else molecule['canonical_smiles']
        value = cache.get(smiles) or by_parent.get((molecule.get('molecule_identity') or {}).get('parent_smiles'))
        if value is None:
            raise ValueError(f'Index molecular form absent from prepared identity cache: {smiles}')
        molecule.update({
            'retrieval_identity_source_smiles': smiles,
            'retrieval_identity_contract': contract,
            'retrieval_leakage_group': value['leakage_group'],
            'retrieval_scaffold_leakage_group': value['scaffold_leakage_group'],
            'retrieval_scaffold_smiles': value['bemis_murcko_scaffold'],
        })
    sample = next(iter(cache.values()))
    index['source'].update({'retrieval_identity_contract': contract, 'retrieval_identity_task': task,
        'retrieval_identity_adapter_sha256': _lineage(task)['adapter_sha256'],
        'retrieval_parent_group_policy': sample['leakage_hash_policy'],
        'retrieval_scaffold_group_policy': sample['scaffold_group_policy']})


@lru_cache(maxsize=4096)
def query_identity(task, smiles, contract):
    validate_contract(task, contract)
    return adapter.leakage_identity(task, smiles)


def decide_candidate(query, candidate, policy, contract):
    """Compare precomputed keys before ranking; no per-candidate RDKit work."""
    if candidate.get('retrieval_identity_contract') != contract:
        raise ValueError('Missing or mixed candidate retrieval identity contract')
    if policy not in {'parent_disjoint', 'scaffold_disjoint'}:
        raise ValueError('Tautomer-aware indices require parent_disjoint or scaffold_disjoint')
    if 'retrieval_scaffold_leakage_group' not in candidate:
        raise ValueError('Candidate lacks a scaffold leakage identity')
    if not candidate.get('retrieval_leakage_group'):
        raise ValueError('Candidate lacks a reliable leakage identity')
    if query['leakage_group'] == candidate['retrieval_leakage_group']:
        return CandidateDecision(MoleculeRelation.SAME_PARENT, True, policy + ':same_tautomer_leakage_group')
    if policy == 'scaffold_disjoint' and query['scaffold_leakage_group'] and query['scaffold_leakage_group'] == candidate.get('retrieval_scaffold_leakage_group'):
        return CandidateDecision(MoleculeRelation.SAME_SCAFFOLD, True, policy + ':same_tautomer_scaffold_group')
    return CandidateDecision(MoleculeRelation.STRUCTURAL_ANALOG, False, 'retained')


def main(argv=None):
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=adapter.TASKS, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 64:
        parser.error('workers must be between 1 and 64')
    frame = pq.read_table(local_input(args.records), columns=['canonical_smiles', 'retrieval_eligible']).to_pandas()
    smiles = frame.loc[frame['retrieval_eligible'].eq(True), 'canonical_smiles'].dropna().unique()
    result = prepare_cache(args.task, smiles, args.cache, args.workers)
    print(f'{args.task}: cached {len(result):,} unique identities', flush=True)


if __name__ == '__main__':
    main()
