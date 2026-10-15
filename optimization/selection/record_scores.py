"""Cache pinned offline record scores for retrieval candidates."""

from __future__ import annotations
import argparse
import json
import math
import sqlite3
from functools import partial
from pathlib import Path
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from optimization.dili_score import DILI_SIMPLE_VERSION, score_dili_record
from predict.utils.json import read_json, sha256_file, write_json_atomic

SCHEMA = "dili_record_scores.v1"
SCORER_COMMIT = "dbbede84b725003032b38848411cd230b0b774e8"
SCORER_FILES = ("__init__.py", "_conditions.py", "_schema.py", "scoring.py")
LEVELS = ("L2", "L3", "L4", "L5", "L6", "L7")
BIOAVAILABILITY_SCHEMA = "bioavailability_record_scores.v1"
TASK_SCHEMAS = {"dili": SCHEMA, "bioavailability_ma": BIOAVAILABILITY_SCHEMA,
                **{task: f"{task}_record_scores.v1" for task in ("bbb_martins", "ames", "carcinogens")}}


def _scorer(task):
    root = Path(__file__).resolve().parents[1]
    if task == "dili":
        return DILI_SIMPLE_VERSION, score_dili_record, root / "dili_score", SCORER_FILES, SCHEMA
    if task == "bioavailability_ma":
        from optimization.bioavailability_score import BIOAVAILABILITY_SIMPLE_VERSION, score_bioavailability_record
        files = ("bioavailability_score/__init__.py", "bioavailability_score/scoring.py",
                 *("dili_score/" + name for name in SCORER_FILES))
        return BIOAVAILABILITY_SIMPLE_VERSION, score_bioavailability_record, root, files, BIOAVAILABILITY_SCHEMA
    if task in ("bbb_martins", "ames", "carcinogens"):
        from optimization.record_scoring import VERSIONS, score_record
        files = ("record_scoring.py", "bioavailability_score/__init__.py", "bioavailability_score/scoring.py",
                 *("dili_score/" + name for name in SCORER_FILES))
        return VERSIONS[task], partial(score_record, task=task), root, files, TASK_SCHEMAS[task]
    raise ValueError(f"Unsupported record-scoring task: {task}")


def candidate_uids(release_index: Path) -> set[str]:
    index = json.loads(release_index.read_text(encoding="utf-8"))
    task = index.get("task_id")
    _scorer(task)
    uids: set[str] = set()
    for subset in ("valid", "test"):
        available = index["splits"][subset]["levels"]
        if task == "dili":
            levels = LEVELS
        else:
            if any(not level.startswith("L") or not level[1:].isdigit() for level in available):
                raise ValueError("Retrieval index has an invalid level name")
            levels = tuple(sorted((level for level in available if int(level[1:]) >= 2),
                                  key=lambda level: int(level[1:])))
        if not levels:
            raise ValueError(f"{task} release has no indirect candidates in {subset}")
        for level in levels:
            entry = available[level]
            manifest_path = (release_index.parent / entry["manifest"]).resolve()
            if sha256_file(manifest_path) != entry["manifest_sha256"]:
                raise ValueError(f"Cache manifest hash mismatch: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            database = manifest_path.with_name(manifest["database"])
            with sqlite3.connect(
                f"file:{database}?mode=ro&immutable=1", uri=True
            ) as connection:
                uids.update(
                    (
                        str(row[0])
                        for row in connection.execute(
                            "SELECT DISTINCT item_id FROM rankings"
                        )
                    )
                )
    if not uids:
        raise ValueError(f"{task} cache has no candidates")
    return uids


def build_cache(
    release_index: Path, canonical_records: Path, output_root: Path
) -> Path:
    release_index, canonical_records = (
        release_index.resolve(),
        canonical_records.resolve(),
    )
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Record score cache output is already populated: {output_root}"
        )
    index = json.loads(release_index.read_text())
    task = index.get("task_id")
    version, scorer, package_root, scorer_files, schema = _scorer(task)
    code_hashes = {name: sha256_file(package_root / name) for name in scorer_files}
    records_hash = sha256_file(canonical_records)
    if task != "dili":
        if not index.get("evidence"):
            raise ValueError(f"{task} release index needs pinned evidence")
        evidence = index["evidence"]
        path = (release_index.parent / evidence["manifest"]).resolve()
        if sha256_file(path) != evidence["manifest_sha256"]:
            raise ValueError(f"{task} evidence manifest changed")
        projection = json.loads(path.read_text())
        if projection.get("task_id") != task or projection.get("status") != "complete":
            raise ValueError(f"{task} evidence projection is incompatible")
        if projection.get("inputs", {}).get("records") != records_hash:
            raise ValueError(f"{task} canonical records differ from the retrieval release")
    uids = candidate_uids(release_index)
    connection = duckdb.connect()
    connection.register("candidate_uids", pa.table({"source_row_uid": sorted(uids)}))
    batches = connection.execute(
        "SELECT c.* FROM read_parquet(?) c SEMI JOIN candidate_uids u USING (source_row_uid)",
        [str(canonical_records)],
    ).to_arrow_reader(2048)
    scores: dict[str, dict] = {}
    for batch in batches:
        for row in batch.to_pylist():
            uid = str(row["source_row_uid"])
            if uid in scores:
                raise ValueError(f"Canonical records repeat candidate UID: {uid}")
            result = scorer(row)
            score = float(result["score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid record score: {uid}")
            scores[uid] = {
                "source_row_uid": uid,
                "tool_score": score,
                "relevance": float(result["components"]["relevance"]),
                "completeness": float(result["components"]["completeness"]),
                "result_content_status": result["result_content_status"],
            }
            if task != "dili":
                scores[uid].update(
                    source_id=str(row.get("source_id") or ""),
                    endpoint_concept=str(row.get("canonical_endpoint_concept") or row.get("endpoint_name") or ""),
                    rule_ids=result["reasons"], ambiguity_flags=result["ambiguity_flags"],
                    missing_fields=result["missing_fields"],
                )
    connection.close()
    missing = uids - scores.keys()
    if missing:
        raise ValueError(
            f"Canonical records omit {len(missing)} candidates: {sorted(missing)[:5]}"
        )
    if code_hashes != {name: sha256_file(package_root / name) for name in scorer_files}:
        raise ValueError("Scoring code changed during the cache build")
    output_root.mkdir(parents=True, exist_ok=True)
    scores_path = output_root / "scores.parquet"
    pq.write_table(
        pa.Table.from_pylist([scores[uid] for uid in sorted(scores)]), scores_path
    )
    manifest_path = output_root / "manifest.json"
    write_json_atomic(
        manifest_path,
        {
            "schema_version": schema,
            "task_id": task,
            "status": "complete",
            "scorer_commit": SCORER_COMMIT if task == "dili" else "",
            "scorer_version": version,
            "scorer_files_sha256": code_hashes,
            "release_index": str(release_index),
            "release_index_sha256": sha256_file(release_index),
            "canonical_records": str(canonical_records),
            "canonical_records_sha256": records_hash,
            "candidate_uid_count": len(uids),
            "scores": {
                "path": scores_path.name,
                "sha256": sha256_file(scores_path),
                "row_count": len(scores),
            },
        },
    )
    return manifest_path


def load_scores(manifest_path: Path) -> tuple[dict[str, float], dict]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest.get("schema_version")
    if schema not in TASK_SCHEMAS.values() or manifest.get("status") != "complete":
        raise ValueError(f"Incomplete Record score cache: {manifest_path}")
    task = next(task for task, expected in TASK_SCHEMAS.items() if schema == expected)
    version, _, package_root, scorer_files, _ = _scorer(task)
    if manifest.get("task_id", task) != task:
        raise ValueError("Record score cache task differs from its schema")
    if ((task == "dili" and manifest.get("scorer_commit") != SCORER_COMMIT)
            or manifest.get("scorer_version") != version):
        raise ValueError("Record score cache uses different scientific rules")
    if any(
        (
            sha256_file(package_root / name) != manifest["scorer_files_sha256"].get(name)
            for name in scorer_files
        )
    ):
        raise ValueError("Record scoring code differs from the score cache")
    for key in ("release_index", "canonical_records"):
        if sha256_file(Path(manifest[key])) != manifest[f"{key}_sha256"]:
            raise ValueError(f"Record score source changed: {key}")
    if read_json(Path(manifest["release_index"])).get("task_id") != task:
        raise ValueError("Score cache release belongs to a different task")
    scores_path = manifest_path.parent / manifest["scores"]["path"]
    if sha256_file(scores_path) != manifest["scores"]["sha256"]:
        raise ValueError("Record score payload changed")
    rows = pq.read_table(
        scores_path, columns=["source_row_uid", "tool_score"]
    ).to_pylist()
    scores = {str(row["source_row_uid"]): float(row["tool_score"]) for row in rows}
    if (len(scores) != len(rows) or len(rows) != manifest["scores"]["row_count"]
            or len(rows) != manifest.get("candidate_uid_count", len(rows))):
        raise ValueError("Record score payload repeats or omits UIDs")
    if any(
        (not math.isfinite(score) or not 0 <= score <= 1 for score in scores.values())
    ):
        raise ValueError("Record score payload contains an invalid score")
    return (
        scores,
        {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "release_index": manifest["release_index"],
            "release_index_sha256": manifest["release_index_sha256"],
            "scorer_commit": manifest.get("scorer_commit", ""),
            "scorer_version": version,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-index", type=Path, required=True)
    parser.add_argument("--canonical-records", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(build_cache(args.release_index, args.canonical_records, args.output_root))


if __name__ == "__main__":
    main()
