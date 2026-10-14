"""Immutable, dependency-checked build stages and streaming artifact IO."""

import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
import platform
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PACKAGE = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def environment():
    return {'python':platform.python_version(), **{name: importlib.metadata.version(name) for name in ('numpy', 'pyarrow', 'duckdb', 'Jinja2', 'rdkit', 'scipy')}}


def implementation():
    return digest({str(path.relative_to(PACKAGE)): file_sha256(path)
                   for path in sorted(PACKAGE.rglob('*')) if path.suffix in ('.py', '.json', '.jinja') and path.name != 'canonical.json'})


def verify_manifest(root):
    root = Path(root)
    manifest = read_json(root / 'manifest.json')
    for relative, expected in manifest['artifacts'].items():
        path = root / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f'Artifact hash mismatch: {path}')
    return manifest


def verify_inputs(manifest):
    for task in manifest['tasks'].values():
        for role, item in task['files'].items():
            path = Path(item['path'])
            if not path.is_file() or file_sha256(path) != item['sha256']:
                raise ValueError(f'Pinned input changed: {role}: {path}')


def stage(root, dependencies, produce, resume=False):
    root = Path(root)
    signature = digest(dependencies)
    if root.exists():
        manifest = verify_manifest(root)
        if resume and manifest['build_signature'] == signature:
            return manifest
        raise FileExistsError(f'Immutable stage exists: {root}; use a new output root or matching --resume')
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{root.name}.', dir=root.parent))
    try:
        details = produce(temporary)
        files = {str(path.relative_to(temporary)): file_sha256(path) for path in sorted(temporary.rglob('*')) if path.is_file()}
        manifest = {**details, 'artifacts': files, 'build_dependencies': dependencies,
                    'build_signature': signature, 'environment': environment(), 'status': 'verified'}
        write_json(temporary / 'manifest.json', manifest)
        verify_manifest(temporary)
        if root.exists():
            raise FileExistsError(root)
        os.rename(temporary, root)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def write_parquet(path, rows, schema=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = iter(rows)
    writer, total = None, 0
    try:
        while True:
            batch = [row for _, row in zip(range(2000), rows)]
            if not batch:
                break
            table = pa.Table.from_pylist(batch, schema=schema)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(path, schema, compression='zstd')
            writer.write_table(table.cast(schema))
            total += len(batch)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        pq.write_table(pa.Table.from_pylist([], schema=schema or pa.schema([pa.field('_empty', pa.bool_())])), path)
    return total


def read_parquet(path):
    return pq.read_table(path).to_pylist()
