"""Export independent top-100 rankings from verified L1 and combined releases."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any

import pyarrow.parquet as pq

from predict.utils.json import sha256_file

from .ranked_level_retrieval import CAPACITY, SCHEMA_VERSION
from .runtime import cache_profile_root


PROFILE = "ranked_level_retrieval_v2"
TASK_LEVELS = {
    "bbb_martins": ("L1", "L2", "L3", "L4", "L5"),
    "bioavailability_ma": ("L1", "L2", "L3", "L4", "L5", "L6"),
}
METHODS = ("morgan", "assay_transfer")
DEFAULT_OUTPUT = cache_profile_root(PROFILE)
DEFAULT_SOURCE = cache_profile_root("ranked_evidence_retrieval_parent_v1")
DEFAULT_L1 = cache_profile_root("v10_3_best_scaffold_morgan100_v1")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _schema(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        PRAGMA journal_mode=DELETE;
        PRAGMA user_version=2;
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE queries(
            benchmark_row_id TEXT PRIMARY KEY,
            drug TEXT NOT NULL,
            query_parent_id TEXT NOT NULL,
            query_parent_smiles TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE rankings(
            benchmark_row_id TEXT NOT NULL,
            rank INTEGER NOT NULL,
            item_id TEXT NOT NULL,
            parent_id TEXT NOT NULL,
            parent_smiles TEXT NOT NULL,
            morgan_similarity REAL NOT NULL,
            transfer_likelihood REAL,
            member_count INTEGER NOT NULL,
            PRIMARY KEY(benchmark_row_id,rank),
            UNIQUE(benchmark_row_id,item_id)
        ) WITHOUT ROWID;
        CREATE INDEX rankings_item ON rankings(item_id);
    """)


def _source_queries(source_db: Path) -> tuple[list[tuple[str, str, str, str]], dict[str, int]]:
    with sqlite3.connect(f"file:{source_db.resolve()}?mode=ro", uri=True) as connection:
        rows = [
            (str(row_id), str(drug), str(parent_id), str(parent_smiles))
            for row_id, drug, parent_id, parent_smiles in connection.execute(
                "SELECT benchmark_row_id,drug,query_parent_id,query_parent_smiles "
                "FROM benchmark_queries JOIN queries USING(query_id) ORDER BY benchmark_row_id"
            )
        ]
        ids = {
            str(row_id): int(query_id)
            for row_id, query_id in connection.execute(
                "SELECT benchmark_row_id,query_id FROM benchmark_queries"
            )
        }
    return rows, ids


def _l1_rows(
    rankings_path: Path, method: str, query_ids: set[str], evidence_db: Path
) -> tuple[dict[str, list[tuple[Any, ...]]], dict[str, dict[str, int]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in pq.read_table(rankings_path).to_pylist():
        query_id = str(row["query_record_id"])
        if query_id in query_ids:
            grouped[query_id][str(row["retrieval_molecule_identity_key"])].append(row)
    context_counts: dict[str, int] = {}
    with sqlite3.connect(f"file:{evidence_db.resolve()}?mode=ro", uri=True) as evidence:
        context_counts = {
            str(context): int(count)
            for context, count in evidence.execute(
                "SELECT context_id,COUNT(*) FROM context_records GROUP BY context_id"
            )
        }
    output: dict[str, list[tuple[Any, ...]]] = {}
    counts: dict[str, dict[str, int]] = {}
    for query_id in sorted(query_ids):
        parents = grouped.get(query_id) or {}
        if len(parents) != CAPACITY:
            raise ValueError(f"L1 cache lacks {CAPACITY} parents: {query_id}")
        chosen = []
        for parent_id, contexts in parents.items():
            if method == "morgan":
                context = min(
                    contexts,
                    key=lambda row: (
                        int(row["retrieval_parent_context_index"]),
                        str(row["retrieval_record_id"]),
                    ),
                )
                order = int(context["retrieval_parent_rank"])
            else:
                context = min(
                    contexts,
                    key=lambda row: (int(row["model_rank"]), str(row["retrieval_record_id"])),
                )
                order = int(context["model_rank"])
            context_id = str(context["retrieval_record_id"])
            member_count = context_counts.get(context_id)
            if not member_count:
                raise ValueError(f"L1 context has no evidence members: {context_id}")
            chosen.append((
                order, context_id, parent_id, str(context["retrieval_smiles"]),
                float(context["morgan_tanimoto_similarity"]),
                float(context["prob_transfer"]) if method == "assay_transfer" else None,
                member_count,
            ))
        chosen.sort(key=lambda row: (row[0], row[2]))
        output[query_id] = [
            (query_id, rank, *row[1:]) for rank, row in enumerate(chosen[:CAPACITY], 1)
        ]
        counts[query_id] = {
            "candidate_records": sum(row[-1] for row in chosen),
            "candidate_parents": len(chosen),
            "stored": len(output[query_id]),
        }
    return output, counts


def _later_rows(
    source_db: Path, query_key_by_id: dict[str, int], level: str, method: str
) -> tuple[dict[str, list[tuple[Any, ...]]], dict[str, dict[str, int]]]:
    rank_column = "morgan_rank" if method == "morgan" else "assay_rank"
    output: dict[str, list[tuple[Any, ...]]] = {}
    counts: dict[str, dict[str, int]] = {}
    with sqlite3.connect(f"file:{source_db.resolve()}?mode=ro", uri=True) as source:
        source.row_factory = sqlite3.Row
        for benchmark_row_id, query_id in query_key_by_id.items():
            count = source.execute(
                "SELECT candidate_record_count,candidate_parent_count FROM selection_counts "
                "WHERE query_id=? AND pool='all' AND level=?", (query_id, level),
            ).fetchone()
            if count is None:
                raise ValueError(f"Source cache lacks {benchmark_row_id}/{level}/all")
            rows = source.execute(
                f"""SELECT source_row_uid,parent_id,parent_smiles,morgan_similarity,
                           assay_transfer_score,{rank_column} AS native_rank
                    FROM rankings WHERE query_id=? AND pool='all' AND level=?
                    AND {rank_column} IS NOT NULL ORDER BY {rank_column} LIMIT ?""",
                (query_id, level, CAPACITY),
            ).fetchall()
            output[benchmark_row_id] = [(
                benchmark_row_id, int(row["native_rank"]), str(row["source_row_uid"]),
                str(row["parent_id"]), str(row["parent_smiles"]),
                float(row["morgan_similarity"]),
                float(row["assay_transfer_score"]) if method == "assay_transfer" else None,
                1,
            ) for row in rows]
            counts[benchmark_row_id] = {
                "candidate_records": int(count[0]),
                "candidate_parents": int(count[1]),
                "stored": len(rows),
            }
    return output, counts


def build_level(
    *, task: str, subset: str, level: str, method: str,
    source_root: Path, l1_root: Path, output_root: Path,
) -> dict[str, Any]:
    if task not in TASK_LEVELS or subset not in {"valid", "test"}:
        raise ValueError("Unsupported task or split")
    if level not in TASK_LEVELS[task] or method not in METHODS:
        raise ValueError("Unsupported level or method")
    if level == "L5" and method != "morgan":
        raise ValueError("L5 is Morgan-only")
    target = output_root.resolve() / task / "scaffold" / subset / level / method
    if target.exists():
        raise FileExistsError(f"Refusing to replace independent cache: {target}")
    source_manifest_path = source_root.resolve() / task / "scaffold" / subset / "VERSION.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_db = source_manifest_path.with_name(str(source_manifest["rankings_database"]))
    queries, query_keys = _source_queries(source_db)
    evidence_manifest_path = source_root.resolve() / task / "evidence" / "VERSION.json"
    evidence_manifest = json.loads(evidence_manifest_path.read_text(encoding="utf-8"))
    evidence_db = evidence_manifest_path.with_name(str(evidence_manifest["database"]))

    if level == "L1":
        l1_manifest_path = l1_root.resolve() / task / "scaffold" / subset / "VERSION.json"
        l1_manifest = json.loads(l1_manifest_path.read_text(encoding="utf-8"))
        if ((l1_manifest.get("inputs") or {}).get("neighbor_identity_policy") != "scaffold_disjoint"
                or l1_manifest.get("morgan_pool_size") != CAPACITY
                or l1_manifest.get("status") != "complete"):
            raise ValueError("L1 source is not the complete scaffold-disjoint Morgan-100 cache")
        l1_rankings = l1_manifest_path.with_name(str(l1_manifest["rankings"]))
        if l1_manifest.get("rankings_sha256") != sha256_file(l1_rankings):
            raise ValueError("L1 ranking hash mismatch")
        rows, query_counts = _l1_rows(l1_rankings, method, set(query_keys), evidence_db)
        source_input = {
            "manifest": str(l1_manifest_path.relative_to(l1_root.resolve())),
            "manifest_sha256": sha256_file(l1_manifest_path),
            "rankings_sha256": l1_manifest["rankings_sha256"],
        }
        identity_policy = "scaffold_disjoint"
    else:
        rows, query_counts = _later_rows(source_db, query_keys, level, method)
        source_input = {
            "manifest": str(source_manifest_path.relative_to(source_root.resolve())),
            "manifest_sha256": sha256_file(source_manifest_path),
            "content_id": source_manifest["content_id"],
        }
        identity_policy = "parent_disjoint"

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{method}.", dir=target.parent) as temporary:
        temporary_root = Path(temporary)
        database = temporary_root / "rankings.sqlite3"
        with sqlite3.connect(database) as connection:
            _schema(connection)
            connection.executemany("INSERT INTO queries VALUES (?,?,?,?)", queries)
            connection.executemany(
                "INSERT INTO rankings VALUES (?,?,?,?,?,?,?,?)",
                [row for query_rows in rows.values() for row in query_rows],
            )
            identity = {
                "schema_version": SCHEMA_VERSION,
                "task_id": task,
                "subset": subset,
                "level": level,
                "ranking_method": method,
                "pool": "fixed" if level == "L1" else "all",
                "gold_release": "v1",
                "neighbor_identity_policy": identity_policy,
                "candidate_parent_capacity": CAPACITY,
                "capacity": CAPACITY,
                "query_count": len(queries),
                "stored_rows": sum(len(value) for value in rows.values()),
                "evidence_content_id": evidence_manifest["content_id"],
                "source": source_input,
            }
            content_id = _digest(identity)
            connection.executemany("INSERT INTO metadata VALUES (?,?)", [
                ("schema_version", SCHEMA_VERSION), ("content_id", content_id),
                ("task_id", task), ("subset", subset), ("level", level),
                ("ranking_method", method),
            ])
            connection.commit()
            connection.execute("VACUUM")
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise ValueError("Independent level database failed integrity check")
        manifest = {
            **identity,
            "content_id": content_id,
            "status": "complete",
            "database": database.name,
            "database_sha256": sha256_file(database),
            "query_counts": query_counts,
        }
        (temporary_root / "VERSION.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_root, target)
    return manifest


def copy_evidence(task: str, source_root: Path, output_root: Path) -> dict[str, Any]:
    source = source_root.resolve() / task / "evidence"
    target = output_root.resolve() / task / "evidence"
    if target.exists():
        raise FileExistsError(f"Refusing to replace evidence store: {target}")
    manifest = json.loads((source / "VERSION.json").read_text(encoding="utf-8"))
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".evidence.", dir=target.parent) as temporary:
        temporary_root = Path(temporary)
        shutil.copy2(source / manifest["database"], temporary_root / manifest["database"])
        shutil.copy2(source / "VERSION.json", temporary_root / "VERSION.json")
        if sha256_file(temporary_root / manifest["database"]) != manifest["database_sha256"]:
            raise ValueError("Copied evidence hash mismatch")
        os.replace(temporary_root, target)
    return manifest


def write_release_index(task: str, output_root: Path) -> dict[str, Any]:
    task_root = output_root.resolve() / task
    evidence_manifest = task_root / "evidence/VERSION.json"
    evidence = json.loads(evidence_manifest.read_text(encoding="utf-8"))
    splits: dict[str, Any] = {}
    complete = True
    for subset in ("valid", "test"):
        levels: dict[str, Any] = {}
        for level in TASK_LEVELS[task]:
            methods = ("morgan",) if level == "L5" else METHODS
            entries = {}
            for method in methods:
                path = task_root / "scaffold" / subset / level / method / "VERSION.json"
                if not path.is_file():
                    complete = False
                    continue
                manifest = json.loads(path.read_text(encoding="utf-8"))
                entries[method] = {
                    "manifest": str(path.relative_to(task_root)),
                    "manifest_sha256": sha256_file(path),
                    "content_id": manifest["content_id"],
                }
            levels[level] = entries
        splits[subset] = {"levels": levels}
    index = {
        "schema_version": "ranked_level_task_release_index.v2",
        "profile": PROFILE,
        "task_id": task,
        "status": "complete" if complete else "partial",
        "gold_release": "v1",
        "pool": "all",
        "capacity": CAPACITY,
        "neighbor_identity_policy_by_level": {
            level: "scaffold_disjoint" if level == "L1" else "parent_disjoint"
            for level in TASK_LEVELS[task]
        },
        "evidence": {
            "manifest": str(evidence_manifest.relative_to(task_root)),
            "manifest_sha256": sha256_file(evidence_manifest),
            "content_id": evidence["content_id"],
        },
        "splits": splits,
    }
    temporary = task_root / ".RELEASE_INDEX.json.tmp"
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, task_root / "RELEASE_INDEX.json")
    return index


def validate(root: Path, *, deep: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "tasks": {}}
    for task, task_levels in TASK_LEVELS.items():
        task_root = root.resolve() / task
        index_path = task_root / "RELEASE_INDEX.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("status") != "complete" or index.get("pool") != "all":
            raise ValueError(f"Incomplete independent release: {task}")
        checked = 0
        for subset, split in index["splits"].items():
            for level, methods in split["levels"].items():
                for method, entry in methods.items():
                    manifest_path = task_root / entry["manifest"]
                    if sha256_file(manifest_path) != entry["manifest_sha256"]:
                        raise ValueError(f"Manifest hash mismatch: {task}/{subset}/{level}/{method}")
                    manifest = _read_manifest(manifest_path)
                    database = manifest_path.with_name(manifest["database"])
                    if sha256_file(database) != manifest["database_sha256"]:
                        raise ValueError(f"Database hash mismatch: {task}/{subset}/{level}/{method}")
                    if deep:
                        with sqlite3.connect(database) as connection:
                            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                                raise ValueError(f"Integrity failure: {database}")
                            overflow = connection.execute(
                                "SELECT COUNT(*) FROM rankings WHERE rank<1 OR rank>?", (CAPACITY,)
                            ).fetchone()[0]
                            duplicates = connection.execute(
                                "SELECT COUNT(*) FROM (SELECT benchmark_row_id,parent_id,COUNT(*) n "
                                "FROM rankings WHERE ?='L1' GROUP BY benchmark_row_id,parent_id HAVING n>1)",
                                (level,),
                            ).fetchone()[0]
                            if overflow or duplicates:
                                raise ValueError(f"Ranking validation failure: {database}")
                    checked += 1
        report["tasks"][task] = {"artifacts": checked}
    report["status"] = "complete"
    report["validation_mode"] = "deep" if deep else "hashes"
    return report


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "complete":
        raise ValueError(f"Invalid independent manifest: {path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("copy-evidence", "build-level", "build-release", "index", "validate"))
    parser.add_argument("--task", choices=tuple(TASK_LEVELS))
    parser.add_argument("--subset", choices=("valid", "test"))
    parser.add_argument("--level")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--l1-root", type=Path, default=DEFAULT_L1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hashes-only", action="store_true")
    args = parser.parse_args()
    if args.command == "validate":
        result = validate(args.output_root, deep=not args.hashes_only)
    elif args.command == "copy-evidence":
        if not args.task:
            parser.error("--task is required")
        result = copy_evidence(args.task, args.source_root, args.output_root)
    elif args.command == "index":
        if not args.task:
            parser.error("--task is required")
        result = write_release_index(args.task, args.output_root)
    elif args.command == "build-level":
        if not all((args.task, args.subset, args.level, args.method)):
            parser.error("--task, --subset, --level, and --method are required")
        result = build_level(
            task=args.task, subset=args.subset, level=args.level, method=args.method,
            source_root=args.source_root, l1_root=args.l1_root, output_root=args.output_root,
        )
    else:
        if not args.task:
            parser.error("--task is required")
        if not (args.output_root / args.task / "evidence").exists():
            copy_evidence(args.task, args.source_root, args.output_root)
        for subset in ("valid", "test"):
            for level in TASK_LEVELS[args.task]:
                for method in (("morgan",) if level == "L5" else METHODS):
                    build_level(
                        task=args.task, subset=subset, level=level, method=method,
                        source_root=args.source_root, l1_root=args.l1_root,
                        output_root=args.output_root,
                    )
        result = write_release_index(args.task, args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
