"""Build and read the shared, immutable Morgan fingerprint cache."""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import socket
import sqlite3
from typing import Iterable, Iterator, Mapping, Sequence

import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from predict.retrieval.policies import (
    IDENTITY_NORMALIZER_VERSION,
    bemis_murcko_scaffold,
    normalize_molecule_identity,
)


CONTRACTS = {
    "rdkit_fragment_parent.v1": "morgan_fingerprint_cache.v1",
    "rdkit_fragment_parent.v2": "morgan_fingerprint_cache.v2",
}
CONTRACT = CONTRACTS[IDENTITY_NORMALIZER_VERSION]
FP_BYTES = 2048 // 8
_DATA_ROOT = Path(__file__).parents[1]
DEFAULT_SOURCES = tuple(
    (task, _DATA_ROOT / path)
    for task, path in (
        ("bbb_martins", "evidence_libraries/bbb_martins/v10/03_pair_buckets/records.parquet"),
        ("bioavailability_ma", "evidence_libraries/bioavailability_ma/v10/03_pair_buckets/records.parquet"),
        ("skin_reaction", "evidence_libraries/skin_reaction/v9/02_canonicalized/records.parquet"),
    )
)
DEFAULT_ROOT = Path(__file__).parents[1] / "caches/morgan_fingerprints/v2"
VARIANT_COLUMNS = {"vanilla": "vanilla", "ring": "vanilla", "feature": "feature"}
FINGERPRINT_CONTRACT = {
    "radius": 2,
    "bits": 2048,
    "count_simulation": False,
    "use_chirality": False,
    "use_bond_types": True,
    "stored_variants": ["vanilla", "feature"],
    "aliases": {"ring": "vanilla"},
    "encoding": "rdkit_binary_text",
}
IDENTITY_CONTRACT = {
    "canonicalization": "rdkit_canonical_isomeric_smiles",
    "digest": "sha256_utf8",
    "database_key": ["identity_sha256", "canonical_smiles"],
    "structure_roles": ["canonical", "normalized_parent"],
    "parent_normalizer": IDENTITY_NORMALIZER_VERSION,
    "parent_id": "parent_inchi_key_or_parent_smiles",
    "scaffold": "rdkit_murcko_nonchiral_or_parent_id_fallback",
}
_GENERATORS = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _generators():
    global _GENERATORS
    if _GENERATORS is None:
        vanilla = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        feature = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
        )
        _GENERATORS = vanilla, feature
    return _GENERATORS


def _fingerprint_row(smiles: str):
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(str(smiles or ""))
        if molecule is None:
            return None
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        molecule = Chem.MolFromSmiles(canonical)
        if molecule is None:
            return None
        vanilla, feature = _generators()
        return (
            hashlib.sha256(canonical.encode()).hexdigest(),
            canonical,
            DataStructs.BitVectToBinaryText(vanilla.GetFingerprint(molecule)),
            DataStructs.BitVectToBinaryText(feature.GetFingerprint(molecule)),
        )


def _source_rows(task: str, path: Path) -> Iterator[tuple[str, str, str]]:
    source = pq.ParquetFile(path)
    columns = ("source_id", "canonical_smiles")
    if any(column not in source.schema_arrow.names for column in columns):
        raise ValueError(f"missing source_id or canonical_smiles: {path}")
    for batch in source.iter_batches(batch_size=20_000, columns=list(columns)):
        for source_id, smiles in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            yield task, str(source_id or "<missing>"), smiles


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA page_size=65536")
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        "CREATE TABLE fingerprints ("
        "identity_sha256 TEXT NOT NULL, canonical_smiles TEXT NOT NULL, "
        "vanilla BLOB NOT NULL, feature BLOB NOT NULL, "
        "PRIMARY KEY (identity_sha256, canonical_smiles)) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE source_membership ("
        "task_id TEXT NOT NULL, source_id TEXT NOT NULL, structure_role TEXT NOT NULL, "
        "identity_sha256 TEXT NOT NULL, "
        "canonical_smiles TEXT NOT NULL, record_count INTEGER NOT NULL, "
        "PRIMARY KEY (task_id, source_id, structure_role, identity_sha256, canonical_smiles), "
        "FOREIGN KEY (identity_sha256, canonical_smiles) "
        "REFERENCES fingerprints(identity_sha256, canonical_smiles)) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE molecule_identity ("
        "identity_sha256 TEXT NOT NULL, canonical_smiles TEXT NOT NULL, "
        "parent_id TEXT NOT NULL, parent_smiles TEXT NOT NULL, scaffold TEXT NOT NULL, "
        "PRIMARY KEY (identity_sha256, canonical_smiles), "
        "FOREIGN KEY (identity_sha256, canonical_smiles) "
        "REFERENCES fingerprints(identity_sha256, canonical_smiles)) WITHOUT ROWID"
    )
    return connection


def _fingerprint_source_row(item: tuple[str, str, str, int]):
    task, source, smiles, records = item
    canonical = _fingerprint_row(smiles)
    identity = normalize_molecule_identity(smiles)
    parent = _fingerprint_row(identity.parent_smiles) if identity.parent_smiles else None
    parent_id = str(identity.parent_inchi_key or identity.parent_smiles)
    scaffold = bemis_murcko_scaffold(identity.parent_smiles) if identity.parent_smiles else ""
    identity_row = (canonical[0], canonical[1], parent_id, identity.parent_smiles,
                    scaffold or (f"parent:{parent_id}" if parent_id else "")) if canonical else None
    return task, source, records, canonical, parent, identity_row


def _computed_rows(rows: Iterable[tuple[str, str, str, int]], workers: int) -> Iterator[tuple]:
    if workers == 1:
        yield from map(_fingerprint_source_row, rows)
        return
    with multiprocessing.Pool(workers) as pool:
        yield from pool.imap(_fingerprint_source_row, rows, chunksize=256)


def build_cache(sources: Sequence[tuple[str, Path]], output: Path, workers: int) -> dict:
    """Build one immutable cache version from complete task-level canonical records."""
    if output.exists():
        raise FileExistsError(f"cache version already exists: {output}")
    resolved = [(task, path.resolve()) for task, path in sources]
    source_manifest = [
        {"task_id": task, "path": str(path), "sha256": _sha256(path),
         "rows": pq.ParquetFile(path).metadata.num_rows,
         "columns": ["source_id", "canonical_smiles"]}
        for task, path in resolved
    ]
    stage = output.parent / f".{output.name}.building-{os.getpid()}"
    stage.mkdir(parents=True)
    database = stage / "fingerprints.sqlite3"
    connection = _connect(database)
    stats, valid_rows, invalid_rows = {}, 0, 0
    fingerprints, memberships, identities = [], [], []
    source_counts = Counter(
        row for task, path in resolved for row in _source_rows(task, path)
    )
    inputs = ((*key, records) for key, records in source_counts.items())
    for task, source, records, canonical, parent, identity_row in _computed_rows(inputs, workers):
        group = stats.setdefault((task, source), {"record_rows": 0, "valid_rows": 0,
                                                   "invalid_rows": 0,
                                                   "parent_unavailable_rows": 0})
        group["record_rows"] += records
        if canonical is None:
            invalid_rows += records
            group["invalid_rows"] += records
            continue
        valid_rows += records
        group["valid_rows"] += records
        identities.append(identity_row)
        for role, row in (("canonical", canonical), ("normalized_parent", parent)):
            if row is None:
                if role == "normalized_parent":
                    group["parent_unavailable_rows"] += records
                continue
            fingerprints.append(row)
            memberships.append((task, source, role, row[0], row[1], records))
        if len(fingerprints) >= 10_000:
            connection.executemany("INSERT OR IGNORE INTO fingerprints VALUES (?, ?, ?, ?)", fingerprints)
            connection.executemany(
                "INSERT INTO source_membership VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT DO UPDATE SET record_count=record_count + excluded.record_count",
                memberships,
            )
            connection.executemany(
                "INSERT OR IGNORE INTO molecule_identity VALUES (?, ?, ?, ?, ?)", identities)
            connection.commit()
            fingerprints.clear()
            memberships.clear()
            identities.clear()
    if fingerprints:
        connection.executemany("INSERT OR IGNORE INTO fingerprints VALUES (?, ?, ?, ?)", fingerprints)
        connection.executemany(
            "INSERT INTO source_membership VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT DO UPDATE SET record_count=record_count + excluded.record_count",
            memberships,
        )
        connection.executemany(
            "INSERT OR IGNORE INTO molecule_identity VALUES (?, ?, ?, ?, ?)", identities)
        connection.commit()
    unique_rows = connection.execute("SELECT count(*) FROM fingerprints").fetchone()[0]
    identity_rows = connection.execute("SELECT count(*) FROM molecule_identity").fetchone()[0]
    role_unique = {
        role: connection.execute(
            "SELECT count(*) FROM (SELECT identity_sha256, canonical_smiles "
            "FROM source_membership WHERE structure_role = ? GROUP BY identity_sha256, canonical_smiles)",
            (role,),
        ).fetchone()[0]
        for role in ("canonical", "normalized_parent")
    }
    membership_stats = [
        {"task_id": task, "source_id": source, "structure_role": role,
         "unique_smiles": unique, "record_rows": records}
        for task, source, role, unique, records in connection.execute(
            "SELECT task_id, source_id, structure_role, count(*), sum(record_count) "
            "FROM source_membership GROUP BY task_id, source_id, structure_role "
            "ORDER BY task_id, source_id, structure_role")
    ]
    connection.close()
    if invalid_rows:
        raise ValueError(f"canonical sources contain {invalid_rows} invalid SMILES")
    if any(_sha256(path) != item["sha256"] for (_, path), item in zip(resolved, source_manifest)):
        raise ValueError("Morgan cache source changed during construction")
    source_rows = sum(item["record_rows"] for item in stats.values())
    manifest = {
        "schema_version": CONTRACT,
        "sources": source_manifest,
        "source_statistics": [dict(task_id=task, source_id=source, **values)
                              for (task, source), values in sorted(stats.items())],
        "source_membership_statistics": membership_stats,
        "counts": {"source_rows": source_rows, "valid_rows": valid_rows,
                   "invalid_rows": invalid_rows, "unique_cached_smiles": unique_rows,
                   "unique_source_canonical_smiles": role_unique["canonical"],
                   "unique_normalized_parent_smiles": role_unique["normalized_parent"],
                   "molecule_identity_rows": identity_rows,
                   "source_canonical_duplicates": valid_rows - role_unique["canonical"]},
        "identity": IDENTITY_CONTRACT,
        "fingerprints": FINGERPRINT_CONTRACT,
        "software": {"rdkit": rdBase.rdkitVersion},
        "build": {"hostname": socket.gethostname(), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                  "workers": workers, "builder_sha256": _sha256(Path(__file__))},
        "database": {"path": database.name, "sha256": _sha256(database)},
    }
    stage.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verify_cache(stage, verify_hash=True)
    stage.rename(output)
    return manifest


def _canonical_id(smiles: str) -> tuple[str, str]:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(str(smiles or ""))
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    return hashlib.sha256(canonical.encode()).hexdigest(), canonical


def _load_manifest(cache_root: Path, parent_normalizer: str | None = None) -> dict:
    manifest = json.loads(Path(cache_root).joinpath("manifest.json").read_text())
    expected_normalizer = parent_normalizer or IDENTITY_NORMALIZER_VERSION
    identity_contract = {
        **IDENTITY_CONTRACT,
        "parent_normalizer": expected_normalizer,
    }
    if (manifest.get("schema_version") != CONTRACTS.get(expected_normalizer)
            or manifest.get("identity") != identity_contract
            or manifest.get("fingerprints") != FINGERPRINT_CONTRACT):
        raise ValueError("Morgan cache contract or parameters do not match this reader")
    return manifest


@lru_cache(maxsize=4)
def load_molecule_identity_index(
    cache_root: Path = DEFAULT_ROOT,
    parent_normalizer: str | None = None,
) -> dict[str, tuple[str, str, str]]:
    """Load source-canonical SMILES to normalized parent and scaffold mappings."""
    manifest = _load_manifest(cache_root, parent_normalizer)
    database = Path(cache_root) / manifest["database"]["path"]
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT canonical_smiles, parent_id, parent_smiles, scaffold FROM molecule_identity"
        )
        return {canonical: (parent, parent_smiles, scaffold)
                for canonical, parent, parent_smiles, scaffold in rows}


def load_morgan_fingerprints(
    smiles: Iterable[str], variant: str = "vanilla", cache_root: Path = DEFAULT_ROOT,
    parent_normalizer: str | None = None,
) -> dict[str, DataStructs.ExplicitBitVect]:
    """Load fingerprints keyed by the caller's SMILES; fail closed on cache misses."""
    if variant not in VARIANT_COLUMNS:
        raise ValueError(f"unknown Morgan variant: {variant}")
    manifest = _load_manifest(cache_root, parent_normalizer)
    identities = {value: _canonical_id(value) for value in dict.fromkeys(map(str, smiles))}
    by_hash = {}
    for digest, canonical in identities.values():
        by_hash.setdefault(digest, []).append(canonical)
    found = {}
    database = Path(cache_root) / "fingerprints.sqlite3"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        hashes = list(by_hash)
        for start in range(0, len(hashes), 900):
            chunk = hashes[start:start + 900]
            marks = ",".join("?" for _ in chunk)
            column = VARIANT_COLUMNS[variant]
            query = f"SELECT identity_sha256, canonical_smiles, {column} FROM fingerprints WHERE identity_sha256 IN ({marks})"
            for digest, canonical, packed in connection.execute(query, chunk):
                if canonical in by_hash[digest]:
                    found[(digest, canonical)] = DataStructs.CreateFromBinaryText(packed)
    missing = [value for value, identity in identities.items() if identity not in found]
    if missing:
        raise KeyError(f"Morgan cache misses {len(missing)} molecule(s); first: {missing[0]!r}")
    return {value: found[identity] for value, identity in identities.items()}


def verify_cache(
    root: Path = DEFAULT_ROOT, verify_hash: bool = False,
    parent_normalizer: str | None = None,
    source_overrides: Mapping[str, Path] | None = None,
) -> dict:
    """Validate schema, row payloads, deterministic samples, and optionally the file hash."""
    root = Path(root)
    manifest = _load_manifest(root, parent_normalizer)
    database = root / manifest["database"]["path"]
    if verify_hash and _sha256(database) != manifest["database"]["sha256"]:
        raise ValueError("Morgan cache database hash differs from its manifest")
    for source in manifest["sources"]:
        path = Path((source_overrides or {}).get(source["path"], source["path"]))
        if pq.ParquetFile(path).metadata.num_rows != source["rows"] or _sha256(path) != source["sha256"]:
            raise ValueError(f"Morgan cache source differs from its manifest: {path}")
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Morgan cache SQLite integrity check failed")
        count = connection.execute("SELECT count(*) FROM fingerprints").fetchone()[0]
        bad = connection.execute(
            "SELECT count(*) FROM fingerprints WHERE length(vanilla) != ? OR length(feature) != ?",
            (FP_BYTES, FP_BYTES),
        ).fetchone()[0]
        samples = connection.execute(
            "SELECT identity_sha256, canonical_smiles, vanilla, feature "
            "FROM fingerprints ORDER BY identity_sha256 LIMIT 8"
        ).fetchall()
        missing = connection.execute(
            "SELECT count(*) FROM source_membership AS source LEFT JOIN fingerprints AS fp "
            "USING (identity_sha256, canonical_smiles) WHERE fp.identity_sha256 IS NULL"
        ).fetchone()[0]
        missing_identities = connection.execute(
            "SELECT count(*) FROM source_membership AS source LEFT JOIN molecule_identity AS identity "
            "USING (identity_sha256, canonical_smiles) "
            "WHERE source.structure_role = 'canonical' AND identity.identity_sha256 IS NULL"
        ).fetchone()[0]
        identity_count = connection.execute("SELECT count(*) FROM molecule_identity").fetchone()[0]
        membership_stats = [
            {"task_id": task, "source_id": source, "structure_role": role,
             "unique_smiles": unique, "record_rows": records}
            for task, source, role, unique, records in connection.execute(
                "SELECT task_id, source_id, structure_role, count(*), sum(record_count) "
                "FROM source_membership GROUP BY task_id, source_id, structure_role "
                "ORDER BY task_id, source_id, structure_role")
        ]
    if (count != manifest["counts"]["unique_cached_smiles"] or bad or missing
            or missing_identities
            or identity_count != manifest["counts"]["molecule_identity_rows"]):
        raise ValueError("Morgan cache counts or fingerprint widths are invalid")
    if membership_stats != manifest["source_membership_statistics"]:
        raise ValueError("Morgan cache task/source coverage differs from its manifest")
    for digest, canonical, vanilla, feature in samples:
        expected = _fingerprint_row(canonical)
        if expected != (digest, canonical, vanilla, feature):
            raise ValueError(f"Morgan cache fingerprint differs from RDKit: {canonical}")
    return {"status": "verified", "rows": count, "database_sha256": manifest["database"]["sha256"]}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument(
        "--source", action="append", metavar="TASK=PARQUET",
        help="repeatable complete canonical-record source; defaults to the three active V9 tasks",
    )
    build.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    build.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    verify.add_argument("--verify-hash", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "build":
        sources = []
        for value in args.source or []:
            task, separator, path = value.partition("=")
            if not separator or not task or not path:
                parser.error(f"invalid --source {value!r}; expected TASK=PARQUET")
            sources.append((task, Path(path)))
        result = build_cache(sources or DEFAULT_SOURCES, args.output, args.workers)
    else:
        result = verify_cache(args.root, args.verify_hash)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
