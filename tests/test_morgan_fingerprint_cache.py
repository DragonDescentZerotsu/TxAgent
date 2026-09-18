from pathlib import Path
import shutil

import pytest

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from data.processing.morgan_fingerprint_cache import (
    build_cache,
    load_molecule_identity_index,
    load_morgan_fingerprints,
    verify_cache,
)


def test_build_load_and_verify_morgan_cache(tmp_path: Path):
    source = tmp_path / "smiles.parquet"
    pq.write_table(pa.table({
        "source_id": ["direct", "direct", "direct", "indirect", "indirect"],
        "canonical_smiles": ["CCO", "CCO", "OCC", "c1ccccc1", "N#N"],
    }), source)
    root = tmp_path / "cache"

    manifest = build_cache([("test_task", source)], root, workers=1)
    assert manifest["counts"] == {
        "source_rows": 5,
        "valid_rows": 5,
        "invalid_rows": 0,
        "unique_cached_smiles": 3,
        "unique_source_canonical_smiles": 3,
        "unique_normalized_parent_smiles": 3,
        "molecule_identity_rows": 3,
        "source_canonical_duplicates": 2,
    }
    assert manifest["source_statistics"] == [
        {"task_id": "test_task", "source_id": "direct", "record_rows": 3,
         "valid_rows": 3, "invalid_rows": 0, "parent_unavailable_rows": 0},
        {"task_id": "test_task", "source_id": "indirect", "record_rows": 2,
         "valid_rows": 2, "invalid_rows": 0, "parent_unavailable_rows": 0},
    ]
    assert len(manifest["source_membership_statistics"]) == 4
    direct = [row for row in manifest["source_membership_statistics"]
              if row["source_id"] == "direct"]
    assert {row["record_rows"] for row in direct} == {3}
    assert verify_cache(root, verify_hash=True)["rows"] == 3
    parent_id, parent_smiles, scaffold = load_molecule_identity_index(root)["CCO"]
    assert parent_id and parent_smiles == "CCO" and scaffold == f"parent:{parent_id}"

    cached = load_morgan_fingerprints(["CCO", "OCC"], cache_root=root)
    expected = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(
        Chem.MolFromSmiles("CCO")
    )
    assert DataStructs.TanimotoSimilarity(cached["CCO"], expected) == 1.0
    assert DataStructs.TanimotoSimilarity(cached["CCO"], cached["OCC"]) == 1.0

    archived = tmp_path / "archived.parquet"
    shutil.copy2(source, archived)
    pq.write_table(pa.table({"source_id": ["changed"], "canonical_smiles": ["C"]}), source)
    with pytest.raises(ValueError, match="source differs"):
        verify_cache(root, verify_hash=True)
    assert verify_cache(root, verify_hash=True, source_overrides={str(source): archived})["rows"] == 3
