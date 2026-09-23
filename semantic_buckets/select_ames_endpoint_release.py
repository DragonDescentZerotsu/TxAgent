"""Select the user-approved AMES endpoint-family map and completed weights."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from predict.utils.json import sha256_file, write_json_atomic
from semantic_buckets.artifacts import (
    REPOSITORY_ROOT, resolve_semantic_bucket_artifacts, semantic_bucket_root,
)
from semantic_buckets.publish_ames_endpoint_candidate_weights import DESTINATION as WEIGHTED
from semantic_buckets.score_ames_endpoint_children import CANDIDATE


RELEASE = "v10_main_universe_v3"
GENERATION = "ames_endpoint_family_v1_20260923"
ROOT = semantic_bucket_root("ames", RELEASE)
INPUT = REPOSITORY_ROOT / (
    "semantic_buckets/provenance/source_local_semantic_v5/"
    "ames_dili_carcinogens_main_universe_v3_frozen_queued_boundary_20260922/ames/input"
)


def _validate() -> tuple[dict, dict, dict]:
    if (REPOSITORY_ROOT / "data/evidence_libraries/ames/CURRENT").read_text().strip() != RELEASE:
        raise ValueError("AMES evidence CURRENT no longer names the pinned release")
    source = json.loads((INPUT / "manifest.json").read_text(encoding="utf-8"))
    semantic = json.loads((CANDIDATE / "manifest.json").read_text(encoding="utf-8"))
    weighted = json.loads((WEIGHTED / "manifest.json").read_text(encoding="utf-8"))
    if (source["status"] != "complete" or source["evidence_release"] != RELEASE
            or semantic["status"] != "candidate_unreviewed_unweighted"
            or weighted["status"] != "complete_unreviewed_candidate"
            or weighted["activation_allowed"] is not False
            or weighted["evidence_library_version"] != RELEASE):
        raise ValueError("AMES publication inputs have incompatible identities")
    for entry in source["inputs"].values():
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Frozen AMES evidence input changed")
    if sha256_file(INPUT / source["output"]["path"]) != source["output"]["sha256"]:
        raise ValueError("Frozen AMES retrieval world changed")
    for name, digest in semantic["files"].items():
        if sha256_file(CANDIDATE / name) != digest:
            raise ValueError(f"AMES candidate map changed: {name}")
    for name, digest in weighted["files"].items():
        if sha256_file(WEIGHTED / name) != digest:
            raise ValueError(f"AMES weighted candidate changed: {name}")
    for field in ("semantic_map", "record_map"):
        if sha256_file(Path(weighted[field])) != weighted[field + "_sha256"]:
            raise ValueError(f"AMES selected {field} changed")
    candidate_records = pd.read_parquet(CANDIDATE / "record_semantic_bucket_map.parquet")
    atoms = pd.read_parquet(CANDIDATE / "semantic_bucket_map.parquet")
    weights = pd.read_parquet(WEIGHTED / "semantic_bucket_weights.parquet")
    ranked_records = pd.read_parquet(WEIGHTED / "record_relevance_rankings.parquet")
    if (len(atoms) != semantic["counts"]["atoms"] or atoms.atom_id.duplicated().any()
            or len(weights) != weighted["bucket_count"]
            or weights.duplicated(["level", "semantic_bucket_id"]).any()
            or atoms.groupby("semantic_bucket_id")[["level", "source_id"]].nunique()
            .gt(1).any().any()):
        raise ValueError("AMES semantic atoms or weights fail exact coverage")
    world = pd.read_parquet(INPUT / source["output"]["path"])
    key = ["source_row_uid", "level", "canonical_record_id"]
    if (len(candidate_records) != len(world)
            or not candidate_records[key].sort_values(key).reset_index(drop=True).equals(
                world[key].sort_values(key).reset_index(drop=True))):
        raise ValueError("AMES candidate does not exactly cover the frozen retrieval world")
    rank = weights[["level", "semantic_bucket_id", "weight", "final_rank"]].rename(
        columns={"final_rank": "weight_rank"})
    expected = candidate_records.merge(rank, on=["level", "semantic_bucket_id"],
                                       how="left", validate="many_to_one")
    fields = [*key, "semantic_bucket_id", "weight", "weight_rank"]
    if (expected.weight.isna().any() or len(ranked_records) != len(expected)
            or not expected[fields].sort_values(key).reset_index(drop=True).equals(
                ranked_records[fields].sort_values(key).reset_index(drop=True))):
        raise ValueError("AMES record rankings differ from selected map and weights")
    return source, semantic, weighted


def _stage(stage: Path, source: dict, semantic: dict, weighted: dict) -> dict:
    generation = stage / "generations" / GENERATION
    weighting = stage / "weighting" / GENERATION
    generation.mkdir(parents=True)
    weighting.mkdir(parents=True)
    for name in ("semantic_bucket_map.parquet", "record_semantic_bucket_map.parquet"):
        shutil.copy2(CANDIDATE / name, generation / name)
    for name in weighted["files"]:
        shutil.copy2(WEIGHTED / name, weighting / name)
    review = {
        "status": "user_approved_selection", "task": "ames",
        "approval": "User approved publishing AMES after conservative grouping and completed weighting",
        "candidate_manifest_sha256": sha256_file(CANDIDATE / "manifest.json"),
        "weighted_candidate_manifest_sha256": sha256_file(WEIGHTED / "manifest.json"),
        "frozen_input_manifest_sha256": sha256_file(INPUT / "manifest.json"),
        "record_coverage": weighted["record_count"],
        "bucket_coverage": weighted["bucket_count"],
        "new_child_scores": weighted["new_child_count"],
    }
    write_json_atomic(generation / "review.json", review)
    write_json_atomic(generation / "semantic_bucket_map_manifest.json", {
        "schema_version": "ames_endpoint_family_reviewed.v1", "status": "complete_reviewed",
        "task": "ames", "evidence_release": RELEASE,
        "semantic_bucket_count": weighted["bucket_count"],
        "atom_count": semantic["counts"]["atoms"],
        "record_count": weighted["record_count"],
        "review_sha256": sha256_file(generation / "review.json"),
        "source_input_manifest_sha256": sha256_file(INPUT / "manifest.json"),
        "semantic_bucket_map_sha256": sha256_file(generation / "semantic_bucket_map.parquet"),
        "record_semantic_bucket_map_sha256": sha256_file(
            generation / "record_semantic_bucket_map.parquet"),
    })
    selected = {
        "semantic_map": f"generations/{GENERATION}/semantic_bucket_map.parquet",
        "semantic_map_manifest": f"generations/{GENERATION}/semantic_bucket_map_manifest.json",
        "record_semantic_bucket_map": f"generations/{GENERATION}/record_semantic_bucket_map.parquet",
        "semantic_bucket_weights": f"weighting/{GENERATION}/semantic_bucket_weights.parquet",
        "semantic_bucket_rankings": f"weighting/{GENERATION}/semantic_bucket_rankings.parquet",
        "record_relevance_rankings": f"weighting/{GENERATION}/record_relevance_rankings.parquet",
    }
    files = {key: {"path": name, "sha256": sha256_file(stage / name)}
             for key, name in selected.items()}
    files["review"] = {"path": f"generations/{GENERATION}/review.json",
                       "sha256": sha256_file(generation / "review.json")}
    manifest = {
        "schema_version": "semantic_buckets.release.v1", "status": "complete_reviewed",
        "task": "ames", "evidence_library_version": RELEASE,
        "semantic_generation": GENERATION, "weight_generation": GENERATION,
        "selected": selected, "files": files,
        "counts": {"semantic_atoms": semantic["counts"]["atoms"],
                   "semantic_buckets": weighted["bucket_count"],
                   "record_assignments": weighted["record_count"]},
        "upstream": {"stage3_records_sha256": source["inputs"]["stage3_records"]["sha256"],
                     "frozen_input_manifest_sha256": sha256_file(INPUT / "manifest.json"),
                     "candidate_manifest_sha256": sha256_file(CANDIDATE / "manifest.json"),
                     "weighted_candidate_manifest_sha256": sha256_file(WEIGHTED / "manifest.json")},
    }
    write_json_atomic(stage / "manifest.json", manifest)
    return manifest


def publish() -> dict:
    if ROOT.exists():
        raise FileExistsError(ROOT)
    source, semantic, weighted = _validate()
    ROOT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{RELEASE}.", dir=ROOT.parent) as temporary:
        stage = Path(temporary)
        manifest = _stage(stage, source, semantic, weighted)
        for entry in manifest["files"].values():
            if sha256_file(stage / entry["path"]) != entry["sha256"]:
                raise ValueError("Staged AMES release hash changed")
        stage.rename(ROOT)
    resolve_semantic_bucket_artifacts("ames", "CURRENT")
    return manifest


if __name__ == "__main__":
    print(json.dumps(publish(), indent=2, sort_keys=True))
