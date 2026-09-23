"""Publish complete AMES endpoint-family rankings without selecting the unreviewed map."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sqlite3
import tempfile

import pandas as pd

from predict.utils.json import sha256_file, write_json_atomic
from semantic_buckets.artifacts import REPOSITORY_ROOT
from semantic_buckets.score_ames_endpoint_children import CANDIDATE, PREDECESSOR, ROOT


DESTINATION = CANDIDATE.parent / "ames_endpoint_family_weighted_candidate_v2_20260923"


def _weights(original: pd.DataFrame, children: pd.DataFrame,
             semantic_map: pd.DataFrame) -> pd.DataFrame:
    ids = set(semantic_map.semantic_bucket_id)
    if (set(children.semantic_bucket_id) & set(original.semantic_bucket_id)
            or not set(children.semantic_bucket_id) <= ids):
        raise ValueError("New child scores conflict with the original universe")
    retained = original[original.semantic_bucket_id.isin(ids)].copy()
    retained["weight_origin"] = "original_two_pass"
    children["weight_origin"] = "endpoint_child_two_pass"
    columns = [column for column in children if column != "weight_origin"]
    weights = pd.concat([retained[columns + ["weight_origin"]],
                         children[columns + ["weight_origin"]]], ignore_index=True)
    weights = weights.sort_values(["level", "weight", "semantic_bucket_id"],
                                  ascending=[True, False, True]).reset_index(drop=True)
    weights["final_rank"] = weights.groupby("level").cumcount() + 1
    if (len(weights) != len(ids) or weights.semantic_bucket_id.duplicated().any()
            or set(weights.semantic_bucket_id) != ids
            or not weights.weight.map(lambda value: math.isfinite(value) and 0 <= value <= 1).all()):
        raise ValueError("Combined weights do not exactly cover the candidate map")
    return weights


def _validated_inputs() -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    candidate = json.loads((CANDIDATE / "manifest.json").read_text(encoding="utf-8"))
    run = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    predecessor = json.loads((PREDECESSOR / "manifest.json").read_text(encoding="utf-8"))
    if (candidate["status"] != "candidate_unreviewed_unweighted"
            or run["status"] != "complete_unreviewed_candidate"
            or predecessor["status"] != "complete_unreviewed_candidate"
            or run["candidate_manifest_sha256"] != sha256_file(CANDIDATE / "manifest.json")
            or run["parent_run_manifest_sha256"] != sha256_file(PREDECESSOR / "manifest.json")
            or candidate["inputs"][str((PREDECESSOR / "weights.parquet").relative_to(
                REPOSITORY_ROOT))] != sha256_file(PREDECESSOR / "weights.parquet")
            or sha256_file(ROOT / "child_weights.parquet") != run["child_weights_sha256"]):
        raise ValueError("AMES source generation or scoring run is not complete")
    for name, digest in candidate["files"].items():
        if sha256_file(CANDIDATE / name) != digest:
            raise ValueError(f"Candidate map changed: {name}")
    connection = sqlite3.connect(f"file:{ROOT / 'requests.sqlite3'}?mode=ro", uri=True)
    try:
        completed, replicas = connection.execute(
            "SELECT (SELECT count(*) FROM requests WHERE phase LIKE 'pass2/%' AND status='complete'), "
            "(SELECT count(*) FROM speculative_replica_receipts WHERE status='valid' AND "
            "request_id IN (SELECT request_id FROM requests WHERE phase LIKE 'pass2/%'))"
        ).fetchone()
    finally:
        connection.close()
    if completed != run["pass2_request_count"] or replicas != 3 * completed:
        raise ValueError("Pass-2 request or replica receipts are incomplete")
    semantic = pd.read_parquet(CANDIDATE / "semantic_bucket_map.parquet")
    records = pd.read_parquet(CANDIDATE / "record_semantic_bucket_map.parquet")
    if (len(semantic) != candidate["counts"]["atoms"]
            or len(records) != candidate["counts"]["records"]
            or semantic.atom_id.duplicated().any()
            or records.duplicated(["source_row_uid", "level"]).any()
            or semantic.groupby("semantic_bucket_id")[["level", "source_id"]].nunique().gt(1).any().any()):
        raise ValueError("Candidate map coverage or identity is invalid")
    return candidate, run, semantic, records


def publish() -> dict:
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    candidate, run, semantic, records = _validated_inputs()
    weights = _weights(pd.read_parquet(PREDECESSOR / "weights.parquet"),
                       pd.read_parquet(ROOT / "child_weights.parquet"), semantic)
    ranking = weights[["task", "level", "semantic_bucket_id", "final_rank", "weight",
                       "rationale", "weight_origin"]]
    record_ranking = records.merge(ranking[["level", "semantic_bucket_id", "weight",
                                           "final_rank"]].rename(columns={
                                               "final_rank": "weight_rank"}),
                                   on=["level", "semantic_bucket_id"],
                                   how="left", validate="many_to_one")
    if len(record_ranking) != len(records) or record_ranking.weight.isna().any():
        raise ValueError("Combined weights omit candidate records")
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ames_weighted_candidate.",
                                     dir=DESTINATION.parent) as temporary:
        stage = Path(temporary)
        weights.to_parquet(stage / "semantic_bucket_weights.parquet", index=False)
        ranking.to_parquet(stage / "semantic_bucket_rankings.parquet", index=False)
        ranking.to_csv(stage / "semantic_bucket_rankings.tsv", sep="\t", index=False)
        record_ranking.to_parquet(stage / "record_relevance_rankings.parquet", index=False)
        summary = ranking.groupby("level").weight.agg(["count", "min", "mean", "max"])
        summary.to_csv(stage / "summary.tsv", sep="\t")
        names = ("semantic_bucket_weights.parquet", "semantic_bucket_rankings.parquet",
                 "semantic_bucket_rankings.tsv", "record_relevance_rankings.parquet", "summary.tsv")
        manifest = {
            "schema_version": "ames_endpoint_family_weighted_candidate.v2",
            "status": "complete_unreviewed_candidate", "task": "ames",
            "evidence_library_version": candidate["evidence_library_version"],
            "activation_allowed": False, "bucket_count": len(weights),
            "record_count": len(record_ranking), "new_child_count": run["bucket_count"],
            "source_manifests": {str(path): sha256_file(path) for path in (
                CANDIDATE / "manifest.json", ROOT / "manifest.json", PREDECESSOR / "manifest.json")},
            "semantic_map": str(CANDIDATE / "semantic_bucket_map.parquet"),
            "semantic_map_sha256": candidate["files"]["semantic_bucket_map.parquet"],
            "record_map": str(CANDIDATE / "record_semantic_bucket_map.parquet"),
            "record_map_sha256": candidate["files"]["record_semantic_bucket_map.parquet"],
            "files": {name: sha256_file(stage / name) for name in names},
        }
        write_json_atomic(stage / "manifest.json", manifest)
        stage.rename(DESTINATION)
    return manifest


if __name__ == "__main__":
    print(json.dumps(publish(), indent=2, sort_keys=True))
