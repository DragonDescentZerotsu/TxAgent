import json

import pandas as pd
import pytest

from semantic_buckets import materialize_overmerge_candidate as candidate
from semantic_buckets.bioavailability_semantic_readout_v1 import _stable_id
from semantic_buckets.publication import sha256_file


def test_candidate_maps_frozen_scoped_records_without_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(candidate, "FROZEN", tmp_path / "frozen")
    frozen = candidate.FROZEN / "dili"
    inputs = frozen / "input"
    semantic_run = frozen / "semantic_run"
    inputs.mkdir(parents=True)
    semantic_run.mkdir()
    atom = _stable_id("atom", "L2", "source_a", "pair_a")
    records = pd.DataFrame([{"canonical_record_id": "record_a", "source_row_uid": "uid_a",
                             "level": "L2", "source_id": "source_a", "pair_bucket_key": "pair_a"}])
    record_path = inputs / "record_relevance_map.parquet"
    records.to_parquet(record_path, index=False)
    (inputs / "manifest.json").write_text(json.dumps({
        "task": "dili", "status": "complete", "evidence_release": "release_a",
        "output": {"sha256": sha256_file(record_path)}}))
    atoms_path = semantic_run / "input_atoms.parquet"
    pd.DataFrame([{"atom_id": atom, "level": "L2", "source_id": "source_a"}]).to_parquet(
        atoms_path, index=False)
    (semantic_run / "semantic_bucket_map_manifest.json").write_text(json.dumps({
        "task": "dili", "input_atoms_sha256": sha256_file(atoms_path)}))
    output = tmp_path / "candidate"
    (output / "dili").mkdir(parents=True)
    pd.DataFrame([{"atom_id": atom, "level": "L2", "source_id": "source_a",
                   "source_semantic_bucket_id": "child_a"}]).to_parquet(
        output / "dili/source_semantic_bucket_map.parquet", index=False)
    receipt = candidate.materialize("dili", output, version="pass3.candidate.v1")
    mapped = pd.read_parquet(output / "dili/record_semantic_bucket_map.parquet")
    assert receipt["status"] == "candidate_unselected"
    assert receipt["version"] == "pass3.candidate.v1"
    assert receipt["record_count"] == 1
    assert mapped.iloc[0].semantic_bucket_id == "child_a"
    with pytest.raises(FileExistsError):
        candidate.materialize("dili", output)
