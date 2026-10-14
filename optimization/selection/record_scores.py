"""Cache pinned, offline DILI scores for the Gold V27 candidate universe."""

from __future__ import annotations
import argparse
import json
import math
import sqlite3
from pathlib import Path
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from optimization.dili_score import DILI_SIMPLE_VERSION, score_dili_record
from predict.utils.json import sha256_file, write_json_atomic

SCHEMA = "dili_record_scores.v1"
SCORER_COMMIT = "dbbede84b725003032b38848411cd230b0b774e8"
SCORER_FILES = ("__init__.py", "_conditions.py", "_schema.py", "scoring.py")
LEVELS = ("L2", "L3", "L4", "L5", "L6", "L7")


def candidate_uids(release_index: Path) -> set[str]:
    index = json.loads(release_index.read_text(encoding="utf-8"))
    if index.get("task_id") != "dili":
        raise ValueError("DILI score cache requires a DILI release index")
    uids: set[str] = set()
    for subset in ("valid", "test"):
        for level in LEVELS:
            entry = index["splits"][subset]["levels"][level]
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
        raise ValueError("Gold V27 DILI cache has no candidates")
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
            f"DILI score cache output is already populated: {output_root}"
        )
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
                raise ValueError(f"Canonical DILI records repeat candidate UID: {uid}")
            result = score_dili_record(row)
            score = float(result["score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid DILI score: {uid}")
            scores[uid] = {
                "source_row_uid": uid,
                "tool_score": score,
                "relevance": float(result["components"]["relevance"]),
                "completeness": float(result["components"]["completeness"]),
                "result_content_status": result["result_content_status"],
            }
    missing = uids - scores.keys()
    if missing:
        raise ValueError(
            f"Canonical DILI records omit {len(missing)} candidates: {sorted(missing)[:5]}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    scores_path = output_root / "scores.parquet"
    pq.write_table(
        pa.Table.from_pylist([scores[uid] for uid in sorted(scores)]), scores_path
    )
    manifest_path = output_root / "manifest.json"
    package_root = Path(__file__).resolve().parents[1] / "dili_score"
    write_json_atomic(
        manifest_path,
        {
            "schema_version": SCHEMA,
            "status": "complete",
            "scorer_commit": SCORER_COMMIT,
            "scorer_version": DILI_SIMPLE_VERSION,
            "scorer_files_sha256": {
                name: sha256_file(package_root / name) for name in SCORER_FILES
            },
            "release_index": str(release_index),
            "release_index_sha256": sha256_file(release_index),
            "canonical_records": str(canonical_records),
            "canonical_records_sha256": sha256_file(canonical_records),
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
    if manifest.get("schema_version") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError(f"Incomplete DILI score cache: {manifest_path}")
    if (
        manifest.get("scorer_commit") != SCORER_COMMIT
        or manifest.get("scorer_version") != DILI_SIMPLE_VERSION
    ):
        raise ValueError("DILI score cache uses different scientific rules")
    package_root = Path(__file__).resolve().parents[1] / "dili_score"
    if any(
        (
            sha256_file(package_root / name) != manifest["scorer_files_sha256"][name]
            for name in SCORER_FILES
        )
    ):
        raise ValueError("DILI scoring code differs from the score cache")
    for key in ("release_index", "canonical_records"):
        if sha256_file(Path(manifest[key])) != manifest[f"{key}_sha256"]:
            raise ValueError(f"DILI score source changed: {key}")
    scores_path = manifest_path.parent / manifest["scores"]["path"]
    if sha256_file(scores_path) != manifest["scores"]["sha256"]:
        raise ValueError("DILI score payload changed")
    rows = pq.read_table(
        scores_path, columns=["source_row_uid", "tool_score"]
    ).to_pylist()
    scores = {str(row["source_row_uid"]): float(row["tool_score"]) for row in rows}
    if len(scores) != len(rows) or len(rows) != manifest["scores"]["row_count"]:
        raise ValueError("DILI score payload repeats or omits UIDs")
    if any(
        (not math.isfinite(score) or not 0 <= score <= 1 for score in scores.values())
    ):
        raise ValueError("DILI score payload contains an invalid score")
    return (
        scores,
        {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "release_index": manifest["release_index"],
            "release_index_sha256": manifest["release_index_sha256"],
            "scorer_commit": SCORER_COMMIT,
            "scorer_version": DILI_SIMPLE_VERSION,
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
