"""Replay the seven primary-source corrections against the pinned Stage-03 source.

Run from the repository root with --input and --output. This review receipt is
not a runner: the ordinary snapshot publisher and retrieval builder consume its
output. Acquisition IDs survive; corrected canonical IDs are in rebindings.json.
"""
import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem

from tools.chembl_tool.common.json_utils import atomic_output_path, sha256_file, write_json_atomic
from tools.chembl_tool.common.starling.evidence_library import starling_molecule_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    ledger = json.loads((root / 'rebindings.json').read_text())
    if sha256_file(args.input) != ledger['input_sha256']:
        raise ValueError('Source differs from the reviewed snapshot')
    decisions = {r['source_record_key']: r for r in ledger['decisions']}
    for review in decisions.values():
        if sha256_file(Path(review['source_document'])) != review['document_sha256']:
            raise ValueError('Primary document changed')
        update = review['updates']
        mol = Chem.MolFromSmiles(update['canonical_smiles'])
        if mol is None or Chem.MolToInchiKey(mol) != review['target_inchi_key']:
            raise ValueError('Reviewed molecule identity mismatch')
        if starling_molecule_id(update['canonical_smiles']) != update['molecule_id']:
            raise ValueError('Stale molecule identifier')
    parquet = pq.ParquetFile(args.input)
    seen = set()
    changed = []
    with atomic_output_path(args.output) as target:
        with pq.ParquetWriter(target, parquet.schema_arrow, compression='zstd') as writer:
            for batch in parquet.iter_batches(batch_size=20000):
                patches = {}
                for i, row in enumerate(batch.to_pylist()):
                    key = f"{row['source_id']}:{int(row['source_row_number']) - 1}"
                    if key not in decisions:
                        continue
                    review = decisions[key]
                    if key in seen or row != review['expected']:
                        raise ValueError(f'Duplicate or changed reviewed row: {key}')
                    seen.add(key)
                    for field, value in review['updates'].items():
                        if row[field] != value:
                            patches.setdefault(field, {})[i] = value
                    changed.append({'source_record_key': key, 'fields': [k for k, v in review['updates'].items() if row[k] != v]})
                for field, values in patches.items():
                    index = batch.schema.get_field_index(field)
                    column = batch.column(index).to_pylist()
                    for i, value in values.items():
                        column[i] = value
                    batch = batch.set_column(index, batch.schema.field(index), pa.array(column, type=batch.schema.field(index).type))
                writer.write_batch(batch)
        if seen != set(decisions):
            raise ValueError('Missing reviewed source rows')
    write_json_atomic(args.output.with_name('rebinding_receipt.json'), {
        'status': 'complete_verified', 'input_sha256': ledger['input_sha256'],
        'output_sha256': sha256_file(args.output), 'n_rows': parquet.metadata.num_rows,
        'ledger_sha256': sha256_file(root / 'rebindings.json'), 'changed': changed,
    })


if __name__ == '__main__':
    main()
