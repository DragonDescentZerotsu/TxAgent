"""Attach frozen retrieval-scoped records to an unselected semantic successor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from semantic_buckets import source_local_semantic_v5 as v5
from semantic_buckets.publication import sha256_file
from tools.chembl_tool.common.json_utils import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / ("semantic_buckets/provenance/source_local_semantic_v5/"
                 "ames_dili_carcinogens_main_universe_v3_frozen_queued_boundary_20260922")
TASKS = ("dili", "carcinogens")


def materialize(task: str, candidate_root: Path, *,
                version: str = "source_local_overmerge_reconcile.v1.candidate.v1") -> dict:
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task}")
    source = candidate_root / task / "source_semantic_bucket_map.parquet"
    destination = candidate_root / task
    semantic_path = destination / "semantic_bucket_map.parquet"
    record_path = destination / "record_semantic_bucket_map.parquet"
    manifest_path = destination / "semantic_bucket_map_manifest.json"
    if any(path.exists() for path in (semantic_path, record_path, manifest_path)):
        raise FileExistsError(f"candidate outputs already exist for {task}")
    frozen = FROZEN / task
    input_path = frozen / "input/record_relevance_map.parquet"
    input_manifest = frozen / "input/manifest.json"
    receipt = json.loads(input_manifest.read_text(encoding="utf-8"))
    if receipt["task"] != task or receipt["status"] != "complete":
        raise ValueError("frozen input manifest identity mismatch")
    if sha256_file(input_path) != receipt["output"]["sha256"]:
        raise ValueError("frozen scoped records changed")
    atoms_path = frozen / "semantic_run/input_atoms.parquet"
    frozen_semantic = json.loads((frozen / "semantic_run/semantic_bucket_map_manifest.json").read_text())
    if (frozen_semantic["task"] != task or
            sha256_file(atoms_path) != frozen_semantic["input_atoms_sha256"]):
        raise ValueError("frozen semantic atom input changed")
    semantic = pd.read_parquet(source)
    atoms = pd.read_parquet(atoms_path, columns=["atom_id", "level", "source_id"])
    if len(semantic) != len(atoms) or semantic.atom_id.duplicated().any():
        raise ValueError("candidate does not assign each frozen atom once")
    identity = semantic[["atom_id", "level", "source_id"]].merge(
        atoms, on="atom_id", how="outer", validate="one_to_one", indicator=True,
        suffixes=("_candidate", "_frozen"))
    if (not identity._merge.eq("both").all() or
            not identity.level_candidate.eq(identity.level_frozen).all() or
            not identity.source_id_candidate.eq(identity.source_id_frozen).all()):
        raise ValueError("candidate changed the frozen atom universe, source, or level")
    semantic = semantic.assign(semantic_bucket_id=semantic.source_semantic_bucket_id)
    mapped = v5._record_map(SimpleNamespace(input=input_path), semantic)
    semantic.to_parquet(semantic_path, index=False)
    mapped.sort_values(["level", "source_row_uid"]).to_parquet(record_path, index=False)
    result = {"version": version,
              "status": "candidate_unselected", "task": task,
              "evidence_release": receipt["evidence_release"],
              "source_semantic_bucket_map_sha256": sha256_file(source),
              "semantic_bucket_map_sha256": sha256_file(semantic_path),
              "record_semantic_bucket_map_sha256": sha256_file(record_path),
              "input_manifest_sha256": sha256_file(input_manifest),
              "input_records_sha256": sha256_file(input_path),
              "input_atoms_sha256": sha256_file(atoms_path),
              "atom_count": len(semantic), "record_count": len(mapped),
              "semantic_bucket_count": semantic.semantic_bucket_id.nunique()}
    write_json_atomic(manifest_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.task, args.candidate_root.resolve()), indent=2))


if __name__ == "__main__":
    main()
