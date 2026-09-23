"""Publish approved DILI/Carcinogens semantic maps without inventing weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from semantic_buckets.artifacts import REPOSITORY_ROOT, RELEASE_ROOT, resolve_semantic_bucket_artifacts
from semantic_buckets.publication import sha256_file
from tools.chembl_tool.common.json_utils import write_json_atomic


TASKS = ("dili", "carcinogens")
RELEASE = "v10_main_universe_v3"
GENERATION = "source_local_small_bucket_luna_v1_20260923"
CANDIDATE = REPOSITORY_ROOT / (
    "semantic_buckets/provenance/source_local_small_bucket_luna_v1/"
    "20260923_le10_tfidf100_luna_flex_v1"
)
FROZEN = REPOSITORY_ROOT / (
    "semantic_buckets/provenance/source_local_semantic_v5/"
    "ames_dili_carcinogens_main_universe_v3_frozen_queued_boundary_20260922"
)
MAPS = ("source_semantic_bucket_map.parquet", "semantic_bucket_map.parquet",
        "record_semantic_bucket_map.parquet")


def verify_candidate(task: str, approved_sha256: str) -> dict:
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task}")
    global_manifest = CANDIDATE / "manifest.json"
    if sha256_file(global_manifest) != approved_sha256:
        raise ValueError("approved candidate manifest hash does not match")
    if json.loads(global_manifest.read_text())["state"] != "candidate_unselected":
        raise ValueError("candidate is not complete")
    current = REPOSITORY_ROOT / "data/evidence_libraries" / task / "CURRENT"
    if current.read_text().strip() != RELEASE:
        raise ValueError(f"{task} active evidence release changed")
    candidate = CANDIDATE / task
    manifest = json.loads((candidate / "semantic_bucket_map_manifest.json").read_text())
    if (manifest["status"] != "candidate_unselected" or manifest["task"] != task
            or manifest["evidence_release"] != RELEASE):
        raise ValueError("candidate manifest identity mismatch")
    for name in MAPS:
        if sha256_file(candidate / name) != manifest[name.removesuffix(".parquet") + "_sha256"]:
            raise ValueError(f"candidate map changed: {name}")
    frozen = FROZEN / task
    source_manifest = frozen / "input/manifest.json"
    atoms_path = frozen / "semantic_run/input_atoms.parquet"
    records_path = frozen / "input/record_relevance_map.parquet"
    if (sha256_file(source_manifest) != manifest["input_manifest_sha256"]
            or sha256_file(atoms_path) != manifest["input_atoms_sha256"]
            or sha256_file(records_path) != manifest["input_records_sha256"]):
        raise ValueError("frozen semantic input changed")
    source = json.loads(source_manifest.read_text())
    for item in ("stage3_records", "level_mapping"):
        receipt = source["inputs"][item]
        if sha256_file(Path(receipt["path"])) != receipt["sha256"]:
            raise ValueError(f"active evidence {item} changed")
    atoms = pd.read_parquet(atoms_path, columns=["atom_id", "level", "source_id"])
    mapping = pd.read_parquet(candidate / "semantic_bucket_map.parquet")
    if len(mapping) != len(atoms) or mapping.atom_id.duplicated().any():
        raise ValueError("semantic map does not cover each atom once")
    joined = mapping.merge(atoms, on="atom_id", how="outer", indicator=True,
                           suffixes=("_map", "_frozen"), validate="one_to_one")
    if (not joined._merge.eq("both").all() or
            not joined.level_map.eq(joined.level_frozen).all() or
            not joined.source_id_map.eq(joined.source_id_frozen).all()):
        raise ValueError("semantic map changed atom identity")
    scope = mapping.groupby("semantic_bucket_id")[["level", "source_id"]].nunique()
    if scope.gt(1).any().any():
        raise ValueError("semantic bucket crosses a source or level")
    records = pd.read_parquet(candidate / "record_semantic_bucket_map.parquet")
    if (len(records) != manifest["record_count"] or
            records.duplicated(["canonical_record_id", "source_row_uid", "level"]).any() or
            not records.semantic_bucket_id.isin(mapping.semantic_bucket_id).all()):
        raise ValueError("record semantic assignments are incomplete or repeated")
    return manifest


def publish(task: str, approved_sha256: str) -> dict:
    candidate_manifest = verify_candidate(task, approved_sha256)
    destination = RELEASE_ROOT / task / RELEASE
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{RELEASE}.", dir=destination.parent))
    try:
        generation = stage / "generations" / GENERATION
        generation.mkdir(parents=True)
        for name in MAPS:
            shutil.copy2(CANDIDATE / task / name, generation / name)
        successor = dict(candidate_manifest, status="complete_approved_unweighted",
                         candidate_manifest_sha256=sha256_file(
                             CANDIDATE / task / "semantic_bucket_map_manifest.json"),
                         approval_candidate_sha256=approved_sha256)
        write_json_atomic(generation / "semantic_bucket_map_manifest.json", successor)
        selected = {
            "semantic_map": f"generations/{GENERATION}/semantic_bucket_map.parquet",
            "semantic_map_manifest": f"generations/{GENERATION}/semantic_bucket_map_manifest.json",
            "record_semantic_bucket_map": f"generations/{GENERATION}/record_semantic_bucket_map.parquet",
        }
        files = {key: {"path": path, "sha256": sha256_file(stage / path)}
                 for key, path in selected.items()}
        source_path = f"generations/{GENERATION}/source_semantic_bucket_map.parquet"
        files["source_semantic_bucket_map"] = {
            "path": source_path, "sha256": sha256_file(stage / source_path)}
        document = {
            "schema_version": "semantic_buckets.release.v1",
            "status": "complete_approved_unweighted", "task": task,
            "evidence_library_version": RELEASE, "semantic_generation": GENERATION,
            "selected": selected, "files": files,
            "counts": {"semantic_atoms": successor["atom_count"],
                       "semantic_buckets": successor["semantic_bucket_count"],
                       "record_assignments": successor["record_count"]},
            "upstream": {"candidate_manifest_sha256": approved_sha256,
                         "candidate_task_manifest_sha256": successor["candidate_manifest_sha256"],
                         "stage3_records_sha256": json.loads((FROZEN / task / "input/manifest.json")
                                                              .read_text())["inputs"]["stage3_records"]["sha256"]},
        }
        write_json_atomic(stage / "manifest.json", document)
        for item in files.values():
            if sha256_file(stage / item["path"]) != item["sha256"]:
                raise ValueError("staged release hash mismatch")
        stage.replace(destination)
        resolve_semantic_bucket_artifacts(task)
        return document
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--approved-candidate-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.task, args.approved_candidate_sha256), indent=2))


if __name__ == "__main__":
    main()
