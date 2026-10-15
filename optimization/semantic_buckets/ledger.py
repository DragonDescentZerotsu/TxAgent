"""Build an immutable UID/pair/semantic census from reviewed evidence inputs.

The census may contain pending assignments. It is not a selectable semantic
release until every levelled UID is assigned and retrieval weights validate.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
import duckdb
import pandas as pd
import yaml
from predict.utils.json import sha256_file, write_json_atomic
from optimization.semantic_buckets.inputs import (
    resolve_semantic_bucket_artifacts,
    resolve_release,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "semantic_uid_pair_ledger.v1"


def _inputs(task: str, release: str) -> dict[str, Path]:
    release = resolve_release(task, release)
    evidence = ROOT / "data/evidence_libraries" / task / release
    selected = resolve_semantic_bucket_artifacts(task, release)
    return {
        "stage3": evidence / "03_pair_buckets/records.parquet",
        "levels": evidence / "level_mapping/records.parquet",
        "record_map": selected.record_semantic_bucket_map
        or selected.record_readout_bucket_map,
        "semantic_manifest": selected.manifest,
    }


def _check_sources(db: duckdb.DuckDBPyConnection, paths: dict[str, Path]) -> None:
    stage, levels, mapping = (
        str(paths[key]) for key in ("stage3", "levels", "record_map")
    )
    for name, path in (("stage", stage), ("levels", levels), ("mapping", mapping)):
        rows, unique = db.execute(
            "SELECT count(*),count(DISTINCT source_row_uid) FROM read_parquet(?)",
            [path],
        ).fetchone()
        if rows != unique:
            raise ValueError(f"{name} repeats a physical UID")
    missing_pair = db.execute(
        "SELECT count(*) FROM read_parquet(?) WHERE pair_bucket_key IS NULL", [stage]
    ).fetchone()[0]
    if missing_pair:
        raise ValueError(f"Stage 3 has {missing_pair} records without a pair key")
    foreign = db.execute(
        """SELECT count(*) FROM read_parquet(?) m
        LEFT JOIN read_parquet(?) s USING (source_row_uid,canonical_record_id)
        LEFT JOIN read_parquet(?) l USING (source_row_uid,canonical_record_id)
        WHERE s.source_row_uid IS NULL OR l.source_row_uid IS NULL
          OR m.source_id <> s.source_id
          OR m.level <> 'L' || CAST(l.level AS VARCHAR)""",
        [mapping, stage, levels],
    ).fetchone()[0]
    if foreign:
        raise ValueError(
            f"Selected semantic map has {foreign} foreign or mislevelled UIDs"
        )
    foreign_levels = db.execute(
        "SELECT count(*) FROM read_parquet(?) l\n        LEFT JOIN read_parquet(?) s USING (source_row_uid,canonical_record_id)\n        WHERE s.source_row_uid IS NULL",
        [levels, stage],
    ).fetchone()[0]
    if foreign_levels:
        raise ValueError(f"Level mapping has {foreign_levels} foreign UIDs")
    conflicting_pairs = db.execute(
        """SELECT count(*) FROM (
        SELECT m.level,s.source_id,s.pair_bucket_key
        FROM read_parquet(?) m JOIN read_parquet(?) s
        USING (source_row_uid,canonical_record_id)
        GROUP BY m.level,s.source_id,s.pair_bucket_key
        HAVING count(DISTINCT m.semantic_bucket_id)>1)""",
        [mapping, stage],
    ).fetchone()[0]
    if conflicting_pairs:
        raise ValueError(
            f"{conflicting_pairs} level-specific pairs have conflicting families"
        )


def build(
    task: str,
    release: str,
    output: Path,
    *,
    record_map: Path | None = None,
    semantic_manifest: Path | None = None,
) -> dict:
    """Write a hash-pinned census without changing any selected release."""
    paths = _inputs(task, release)
    if record_map is not None:
        if semantic_manifest is None:
            raise ValueError("Candidate record map requires its manifest")
        paths["record_map"] = record_map.resolve()
        paths["semantic_manifest"] = semantic_manifest.resolve()
    if output.exists():
        raise FileExistsError(output)
    if any((path is None or not path.is_file() for path in paths.values())):
        raise FileNotFoundError("Central-ledger input is missing")
    db = duckdb.connect()
    _check_sources(db, paths)
    output.mkdir(parents=True)
    ledger = output / "central_ledger.parquet"
    query = """WITH stage AS (
        SELECT canonical_record_id,source_row_uid,source_id,pair_bucket_key
        FROM read_parquet(?)
    ), levels AS (
        SELECT canonical_record_id,source_row_uid,'L' || CAST(NULLIF(level,0) AS VARCHAR) AS level
        FROM read_parquet(?)
    ), mapping AS (
        SELECT canonical_record_id,source_row_uid,level,semantic_bucket_id
        FROM read_parquet(?)
    ), pair_assignments AS (
        SELECT DISTINCT m.level,s.source_id,s.pair_bucket_key,m.semantic_bucket_id
        FROM mapping m JOIN stage s USING (canonical_record_id,source_row_uid)
    ), members AS (
        SELECT source_id,pair_bucket_key,
            list_sort(list_distinct(list(struct_pack(
                level := level, semantic_bucket_id := semantic_bucket_id))))
                AS semantic_memberships
        FROM pair_assignments GROUP BY source_id,pair_bucket_key
    )
    SELECT s.canonical_record_id,s.source_row_uid,s.source_id,s.pair_bucket_key,
        l.level,p.semantic_bucket_id AS own_level_semantic_bucket_id,
        members.semantic_memberships,
        CASE WHEN l.level IS NULL THEN 'unleveled'
             WHEN p.semantic_bucket_id IS NULL THEN 'unassigned'
             ELSE 'assigned' END AS assignment_status
    FROM stage s LEFT JOIN levels l USING (canonical_record_id,source_row_uid)
    LEFT JOIN pair_assignments p ON l.level=p.level AND s.source_id=p.source_id
        AND s.pair_bucket_key=p.pair_bucket_key
    LEFT JOIN members ON members.source_id=s.source_id
        AND members.pair_bucket_key=s.pair_bucket_key
    ORDER BY s.source_row_uid"""
    db.execute(
        f"COPY ({query}) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
        [str(ledger)] + [str(paths[key]) for key in ("stage3", "levels", "record_map")],
    )
    counts = dict(
        db.execute(
            "SELECT assignment_status,count(*) FROM read_parquet(?)\n        GROUP BY assignment_status ORDER BY assignment_status",
            [str(ledger)],
        ).fetchall()
    )
    total = sum(counts.values())
    expected = db.execute(
        "SELECT count(*) FROM read_parquet(?)", [str(paths["stage3"])]
    ).fetchone()[0]
    if total != expected:
        raise ValueError(f"Central ledger lost Stage-3 UIDs: {total} != {expected}")
    manifest = {
        "schema_version": SCHEMA,
        "status": "candidate_unreviewed",
        "task": task,
        "evidence_release": resolve_release(task, release),
        "inputs": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
        "ledger": {"path": ledger.name, "sha256": sha256_file(ledger), "rows": total},
        "assignment_status_counts": counts,
    }
    write_json_atomic(output / "manifest.json", manifest)
    db.close()
    return manifest


def missing_weight_summary(ledger: Path, weights: Path) -> dict[str, int]:
    """Count mapped families without weights; never invent a numeric fallback."""
    db = duckdb.connect()
    result = db.execute(
        """WITH members AS (
        SELECT DISTINCT m.member.level AS level,
            m.member.semantic_bucket_id AS semantic_bucket_id
        FROM read_parquet(?) l, unnest(l.semantic_memberships) m(member)
    ), weights AS (
        SELECT level,semantic_bucket_id FROM read_parquet(?)
    ) SELECT count(*),count(*) FILTER (WHERE w.semantic_bucket_id IS NULL)
      FROM members m LEFT JOIN weights w USING (level,semantic_bucket_id)""",
        [str(ledger), str(weights)],
    ).fetchone()
    db.close()
    return {"mapped_families": result[0], "unweighted_families": result[1]}


def collect_retrieval_world(task: str, bundles: tuple[Path, ...], output: Path) -> dict:
    """Freeze the valid/test L2+ physical-UID union of explicit cache bundles."""
    if output.exists():
        raise FileExistsError(output)
    if not bundles:
        raise ValueError("At least one selected cache bundle is required")
    uids: set[tuple[str, str]] = set()
    inputs: dict[str, str] = {}
    for bundle in bundles:
        config = yaml.safe_load(bundle.read_text(encoding="utf-8"))
        if config.get("schema_version") == "retrieval_cache_set.v1":
            indexes = {ROOT / tasks[task]["retrieval"]["later"]
                       for tasks in config["benchmarks"].values() if task in tasks}
        else:
            indexes = {(bundle.parent / config["caches"][task]["later"]).resolve()}
        for index in sorted(indexes):
            document = json.loads(index.read_text(encoding="utf-8"))
            if document.get("status") != "complete" or document.get("task_id") != task:
                raise ValueError(f"Incomplete or foreign retrieval index: {index}")
            inputs[str(bundle.resolve())] = sha256_file(bundle)
            inputs[str(index)] = sha256_file(index)
            for split in ("valid", "test"):
                for level, entry in document["splits"][split]["levels"].items():
                    if level == "L1":
                        continue
                    version = index.parent / entry["manifest"]
                    if sha256_file(version) != entry["manifest_sha256"]:
                        raise ValueError(f"Retrieval manifest changed: {version}")
                    metadata = json.loads(version.read_text(encoding="utf-8"))
                    database = version.with_name(metadata["database"])
                    with sqlite3.connect(
                        f"file:{database.resolve()}?mode=ro&immutable=1", uri=True
                    ) as connection:
                        uids.update(
                            (
                                (str(uid), level)
                                for (uid,) in connection.execute(
                                    "SELECT DISTINCT item_id FROM rankings"
                                )
                            )
                        )
                    inputs[str(version)] = entry["manifest_sha256"]
    if not uids:
        raise ValueError("Selected retrieval bundles contain no indirect UIDs")
    output.mkdir(parents=True)
    selected = output / "retrieval_uids.parquet"
    pd.DataFrame(sorted(uids), columns=["source_row_uid", "level"]).to_parquet(
        selected, index=False
    )
    manifest = {
        "schema_version": "semantic_retrieval_world.v1",
        "status": "candidate_unreviewed",
        "task": task,
        "inputs": inputs,
        "uids": {
            "path": selected.name,
            "sha256": sha256_file(selected),
            "rows": len(uids),
        },
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def audit_retrieval_coverage(
    ledger: Path, weights: Path, retrieval_uids: Path
) -> dict[str, int]:
    """Measure the two publication gates without writing or selecting artifacts."""
    db = duckdb.connect()
    db.execute(
        """CREATE TEMP TABLE targets AS SELECT r.source_row_uid,r.level,
        l.assignment_status,l.semantic_memberships
        FROM (SELECT DISTINCT source_row_uid,level FROM read_parquet(?)) r
        LEFT JOIN read_parquet(?) l USING (source_row_uid,level)""",
        [str(retrieval_uids), str(ledger)],
    )
    total, missing_uid, unassigned = db.execute(
        "SELECT count(*),\n        count(*) FILTER (WHERE assignment_status IS NULL),\n        count(*) FILTER (WHERE assignment_status='unassigned')\n        FROM targets"
    ).fetchone()
    db.execute(
        """CREATE TEMP TABLE missing_components AS
        SELECT t.source_row_uid,m.member.level AS family_level,
            m.member.semantic_bucket_id AS semantic_bucket_id
        FROM targets t,unnest(t.semantic_memberships) m(member)
        LEFT JOIN read_parquet(?) w ON w.level=m.member.level
            AND w.semantic_bucket_id=m.member.semantic_bucket_id
        WHERE w.semantic_bucket_id IS NULL""",
        [str(weights)],
    )
    affected, families = db.execute(
        "SELECT count(DISTINCT source_row_uid),\n        count(DISTINCT (family_level,semantic_bucket_id)) FROM missing_components"
    ).fetchone()
    db.close()
    return {
        "retrieval_uids": total,
        "missing_uid": missing_uid,
        "unassigned_uid": unassigned,
        "uids_missing_component_weight": affected,
        "unweighted_touched_families": families,
    }


def prepare_pending_atoms(ledger: Path, output: Path) -> dict:
    """Freeze only unresolved level-specific pair atoms for scientific review."""
    if output.exists():
        raise FileExistsError(output)
    db = duckdb.connect()
    output.mkdir(parents=True)
    atoms = output / "pending_atoms.parquet"
    db.execute(
        """COPY (SELECT level,source_id,pair_bucket_key,
        MIN(source_row_uid) AS representative_uid,count(*) AS record_count,
        first(semantic_memberships) AS reviewed_other_level_memberships
        FROM read_parquet(?) WHERE assignment_status='unassigned'
        GROUP BY level,source_id,pair_bucket_key
        ORDER BY level,source_id,pair_bucket_key)
        TO ? (FORMAT PARQUET, COMPRESSION ZSTD)""",
        [str(atoms), str(ledger)],
    )
    count = db.execute("SELECT count(*) FROM read_parquet(?)", [str(atoms)]).fetchone()[
        0
    ]
    db.close()
    manifest = {
        "schema_version": "semantic_pending_pair_atoms.v1",
        "status": "candidate_unreviewed",
        "ledger": {"path": str(ledger), "sha256": sha256_file(ledger)},
        "atoms": {"path": atoms.name, "sha256": sha256_file(atoms), "rows": count},
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def project_retrieval_weights(
    ledger: Path,
    weights: Path,
    retrieval_uids: Path,
    output: Path,
    *,
    provisional_mean: bool = False,
    weight_source: str = "reviewed_family_mean",
) -> dict:
    """Materialize reviewed means, optionally using level means for pending UIDs."""
    if output.exists():
        raise FileExistsError(output)
    db = duckdb.connect()
    duplicate_weights = db.execute(
        """SELECT count(*)-count(DISTINCT (level,semantic_bucket_id)),
        count(*) FILTER (WHERE weight IS NULL OR NOT isfinite(weight)
            OR weight < 0 OR weight > 1) FROM read_parquet(?)""",
        [str(weights)],
    ).fetchone()
    if any(duplicate_weights):
        raise ValueError(f"Invalid family weights: {duplicate_weights}")
    db.execute(
        "CREATE TEMP TABLE targets AS SELECT l.* FROM read_parquet(?) l\n        JOIN (SELECT DISTINCT source_row_uid,level FROM read_parquet(?)) r\n        USING (source_row_uid,level)",
        [str(ledger), str(retrieval_uids)],
    )
    required = db.execute(
        "SELECT count(*) FROM (\n        SELECT DISTINCT source_row_uid,level FROM read_parquet(?))",
        [str(retrieval_uids)],
    ).fetchone()[0]
    matched, pending = db.execute(
        "SELECT count(*),count(*) FILTER (\n        WHERE assignment_status = 'unassigned')\n        FROM targets"
    ).fetchone()
    invalid = db.execute("""SELECT count(*) FROM targets WHERE assignment_status IS NULL
        OR assignment_status='unleveled' OR (assignment_status='assigned'
        AND semantic_memberships IS NULL)""").fetchone()[0]
    if matched != required or invalid or (pending and (not provisional_mean)):
        raise ValueError(
            f"Retrieval semantic coverage incomplete: {matched}/{required}; pending={pending}; invalid={invalid}"
        )
    db.execute(
        """CREATE TEMP TABLE components AS SELECT t.source_row_uid,t.level,
        t.own_level_semantic_bucket_id AS semantic_bucket_id,
        m.member.level AS component_level,
        m.member.semantic_bucket_id AS component_bucket_id,w.weight
        FROM targets t, unnest(t.semantic_memberships) m(member)
        LEFT JOIN read_parquet(?) w ON w.level=m.member.level
            AND w.semantic_bucket_id=m.member.semantic_bucket_id
        WHERE t.assignment_status='assigned'""",
        [str(weights)],
    )
    missing = db.execute(
        "SELECT count(*) FROM components WHERE weight IS NULL"
    ).fetchone()[0]
    if missing:
        sample = db.execute(
            "SELECT DISTINCT component_level,component_bucket_id\n            FROM components WHERE weight IS NULL ORDER BY 1,2 LIMIT 5"
        ).fetchall()
        raise ValueError(
            f"Retrieval requires {missing} unweighted memberships; sample={sample}"
        )
    output.mkdir(parents=True)
    target = output / "record_relevance_rankings.parquet"
    db.execute(
        """CREATE TEMP TABLE projected AS SELECT source_row_uid,level,
        semantic_bucket_id,AVG(weight) AS weight,count(*) AS component_count,
        ? AS weight_source FROM components
        GROUP BY source_row_uid,level,semantic_bucket_id""",
        [weight_source],
    )
    if pending:
        db.execute(
            """INSERT INTO projected SELECT t.source_row_uid,t.level,
            NULL AS semantic_bucket_id,w.mean_weight AS weight,
            0 AS component_count,'provisional_task_level_mean' AS weight_source
            FROM targets t JOIN (SELECT level,AVG(weight) AS mean_weight
                FROM read_parquet(?) GROUP BY level) w USING (level)
            WHERE t.assignment_status='unassigned'""",
            [str(weights)],
        )
    final_rows = db.execute("SELECT count(*) FROM projected").fetchone()[0]
    if final_rows != required:
        raise ValueError(f"Projected {final_rows}/{required} retrieval UIDs")
    db.execute(
        "COPY (SELECT * FROM projected ORDER BY level,source_row_uid)\n        TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
        [str(target)],
    )
    manifest = {
        "schema_version": "semantic_retrieval_mean_weights.v2",
        "status": "candidate_unreviewed",
        "aggregation": "mean_distinct_pair_families_else_task_level_bucket_mean",
        "provisional_uids": pending,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in (
                ("ledger", ledger),
                ("weights", weights),
                ("retrieval_uids", retrieval_uids),
            )
        },
        "projection": {
            "path": target.name,
            "sha256": sha256_file(target),
            "rows": matched,
        },
    }
    write_json_atomic(output / "manifest.json", manifest)
    db.close()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task",
        choices=(
            "bbb_martins",
            "bioavailability_ma",
            "ames",
            "dili",
            "carcinogens",
            "skin_reaction",
        ),
    )
    parser.add_argument("--release", default="CURRENT")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.task, args.release, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
