"""Select completed two-pass weights for reviewed semantic releases.

Reads frozen scoring runs and the selected record map, checks exact bucket and
record coverage, then publishes immutable ranking files and atomically selects
them in the task's existing semantic release. It never changes evidence CURRENT.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile

import pandas as pd

from semantic_buckets.artifacts import REPOSITORY_ROOT, resolve_semantic_bucket_artifacts
from semantic_buckets.publication import sha256_file
from tools.chembl_tool.common.json_utils import write_json_atomic


RUNS = {
    "dili": "dili_morgan100_official_two_pass_v6_cards_luna_standard_medium_balanced8_v2_20260923",
    "carcinogens": "carcinogens_morgan100_official_two_pass_v6_cards_luna_standard_medium_balanced8_v2_20260923",
}
GENERATION = "official_two_pass_v6_selected_20260923_v1"
RUN_ROOT = REPOSITORY_ROOT / "semantic_buckets/provenance/semantic_weight_two_pass_v4"


def _source(task: str) -> tuple[Path, dict, pd.DataFrame]:
    path = RUN_ROOT / RUNS[task]
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("task") != task or manifest.get("status") != "complete_unselected_candidate"
            or manifest.get("completed_bucket_count") != manifest.get("bucket_count")):
        raise ValueError(f"Incomplete or incompatible scoring run: {path}")
    for name, digest in manifest["artifacts"].items():
        if sha256_file(path / name) != digest:
            raise ValueError(f"Scoring artifact changed: {path / name}")
    weights = pd.read_parquet(path / "weights.parquet")
    if (len(weights) != manifest["bucket_count"] or weights.duplicated(
            ["level", "semantic_bucket_id"]).any() or not weights.weight.map(
            lambda value: math.isfinite(value) and 0 <= value <= 1).all()):
        raise ValueError(f"Invalid completed weights: {path}")
    return path, manifest, weights


def _rankings(task: str, weights: pd.DataFrame, record_map: Path) -> pd.DataFrame:
    mapping = pd.read_parquet(record_map)
    keys = ["level", "semantic_bucket_id"]
    if mapping.duplicated(["source_row_uid", "level"]).any():
        raise ValueError(f"Duplicate record-level mapping: {task}")
    selected = mapping.merge(weights[keys + ["weight", "final_rank"]], on=keys,
                             how="left", validate="many_to_one")
    if selected.weight.isna().any() or len(selected) != len(mapping):
        raise ValueError(f"Weights omit selected records: {task}")
    return selected.rename(columns={"final_rank": "weight_rank"}).sort_values(
        ["level", "source_row_uid"])


def _validate_buckets(task: str, weights: pd.DataFrame, semantic_map: Path) -> None:
    atoms = pd.read_parquet(semantic_map, columns=[
        "level", "source_id", "semantic_bucket_id"])
    if (atoms.groupby("semantic_bucket_id")[["level", "source_id"]].nunique()
            .gt(1).any().any()):
        raise ValueError(f"Cross-source or cross-level semantic bucket: {task}")
    keys = ["level", "semantic_bucket_id"]
    coverage = atoms[keys].drop_duplicates().merge(
        weights[keys], on=keys, how="outer", indicator=True, validate="one_to_one")
    if not coverage["_merge"].eq("both").all():
        raise ValueError(f"Weights do not exactly cover selected buckets: {task}")


def publish(task: str) -> dict:
    if task not in RUNS:
        raise ValueError(f"Unsupported approved task: {task}")
    artifacts = resolve_semantic_bucket_artifacts(task, "CURRENT")
    release = artifacts.root
    old_path = artifacts.manifest
    old_hash = sha256_file(old_path)
    old = json.loads(old_path.read_text(encoding="utf-8"))
    if old.get("status") != "complete_approved_unweighted" or artifacts.record_semantic_bucket_map is None:
        raise ValueError(f"Expected a reviewed bucket-only release: {release}")
    run_path, run, weights = _source(task)
    if run["inputs"]["semantic_map"]["sha256"] != sha256_file(artifacts.semantic_map):
        raise ValueError(f"Scoring run targets another semantic map: {task}")
    _validate_buckets(task, weights, artifacts.semantic_map)
    records = _rankings(task, weights, artifacts.record_semantic_bucket_map)
    destination = release / "weighting" / GENERATION
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{GENERATION}.", dir=destination.parent) as temporary:
        stage = Path(temporary)
        weights.to_parquet(stage / "semantic_bucket_weights.parquet", index=False)
        weights.to_csv(stage / "semantic_bucket_weights.tsv", sep="\t", index=False)
        weights.sort_values(["level", "final_rank", "semantic_bucket_id"]).to_parquet(
            stage / "semantic_bucket_rankings.parquet", index=False)
        records.to_parquet(stage / "record_relevance_rankings.parquet", index=False)
        names = ("semantic_bucket_weights.parquet", "semantic_bucket_weights.tsv",
                 "semantic_bucket_rankings.parquet", "record_relevance_rankings.parquet")
        bundle = {
            "schema_version": "semantic_weight_selection.v1", "status": "complete_reviewed",
            "task": task, "evidence_library_version": artifacts.release,
            "generation": GENERATION, "bucket_count": len(weights), "record_count": len(records),
            "source_run_manifest": str(run_path / "manifest.json"),
            "source_run_manifest_sha256": sha256_file(run_path / "manifest.json"),
            "selected_semantic_map_sha256": sha256_file(artifacts.semantic_map),
            "previous_release_manifest_sha256": old_hash,
            "files": {name: sha256_file(stage / name) for name in names},
        }
        write_json_atomic(stage / "manifest.json", bundle)
        stage.rename(destination)
    prefix = destination.relative_to(release).as_posix()
    selected = dict(old["selected"])
    selected.update({key: f"{prefix}/{key}.parquet" for key in (
        "semantic_bucket_weights", "semantic_bucket_rankings", "record_relevance_rankings")})
    updated = dict(old, status="complete_reviewed", selected=selected,
                   weight_generation=GENERATION,
                   files={key: {"path": name, "sha256": sha256_file(release / name)}
                          for key, name in selected.items()},
                   previous_release_manifest_sha256=old_hash)
    if sha256_file(old_path) != old_hash:
        raise ValueError("Release manifest changed during publication")
    write_json_atomic(destination / "selection_receipt.json", {
        "previous_release_manifest": old, "previous_release_manifest_sha256": old_hash,
        "new_selected": selected,
    })
    write_json_atomic(old_path, updated)
    if sha256_file(artifacts.semantic_map) != updated["files"]["semantic_map"]["sha256"]:
        raise ValueError("Selected semantic map changed during publication")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(RUNS), required=True)
    args = parser.parse_args()
    result = publish(args.task)
    print(json.dumps({"task": args.task, "status": result["status"],
                      "weight_generation": result["weight_generation"]}, sort_keys=True))


if __name__ == "__main__":
    main()
