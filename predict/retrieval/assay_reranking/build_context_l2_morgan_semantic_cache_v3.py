"""Publish matched L2 panels, expanding Morgan parents until both contain 25 records."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any

from predict.utils.json import sha256_file

from . import build_context_l2_morgan_semantic_cache_v2 as base
from . import three_pools
from .build_context_l2_morgan_semantic_cache import _eligibility
from .runtime import cache_profile_root
from .semantic_bucket_selection import select_global_molecule_records


SCHEMA_VERSION = "l1_context_morgan_semantic_l2.v3"
PROFILE = "l1_context_morgan_semantic_l2_v3"
OUTPUT_ROOT = cache_profile_root(PROFILE)
MAX_PARENT_LIMIT = 100


def _matched_parent_prefix(
    rows: list[dict[str, Any]],
) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["parent_id"])].append(row)
    ordered = sorted(
        grouped, key=lambda parent: (-grouped[parent][0]["morgan_similarity"], parent)
    )[:MAX_PARENT_LIMIT]
    candidates: list[dict[str, Any]] = []
    for rank, parent in enumerate(ordered, 1):
        for row in grouped[parent]:
            row["parent_rank"] = rank
        candidates.extend(grouped[parent])
        sizes = [len(select_global_molecule_records(
            candidates, method=method, limit=base.TOTAL_RECORD_LIMIT,
            per_parent_limit=base.PER_PARENT_LIMIT,
        )) for method in ("control", "semantic")]
        if sizes == [base.TOTAL_RECORD_LIMIT, base.TOTAL_RECORD_LIMIT]:
            return ordered[:rank], grouped
    raise ValueError("Morgan-100 parents cannot fill both matched 25-record panels")


def _populate(
    output: sqlite3.Connection, source: sqlite3.Connection,
    binding: dict[str, dict[str, Any]], task: str,
) -> Counter:
    source_queries = {str(row[0]): int(row[1]) for row in source.execute(
        "SELECT benchmark_row_id,query_id FROM benchmark_queries"
    )}
    record_keys, summary = {}, Counter()
    queries = output.execute(
        """SELECT b.benchmark_row_id,b.query_id,q.query_parent_smiles
           FROM benchmark_queries AS b JOIN queries AS q USING(query_id)
           ORDER BY b.query_id"""
    ).fetchall()
    for query in queries:
        query_id, benchmark = int(query["query_id"]), str(query["benchmark_row_id"])
        rows, counts = base._query_rows(
            source, source_queries[benchmark], binding,
            base._l1_visible_record_ids(output, query_id), task, benchmark,
            str(query["query_parent_smiles"]),
        )
        parents, grouped = _matched_parent_prefix(rows)
        for mode in base.MODES:
            base._store_mode(output, query_id, mode, parents, grouped, record_keys, summary)
        summary.update(counts)
        summary["selected_parent_slots"] += len(parents)
        summary["maximum_parent_prefix"] = max(summary["maximum_parent_prefix"], len(parents))
    summary["l2_unique_records"], summary["queries"] = len(record_keys), len(queries)
    return summary


def _manifest(
    task: str, subset: str, l1: dict, source_path: Path, source: dict,
    eligibility: dict, database: Path, summary: Counter, content_id: str,
) -> dict[str, Any]:
    manifest = base._manifest(
        task, subset, l1, source_path, source, eligibility, database, summary, content_id,
    )
    builder = Path(__file__).resolve()
    manifest.update(
        schema_version=SCHEMA_VERSION,
        capacities={**manifest["capacities"], "l2_molecules": MAX_PARENT_LIMIT},
        selection={
            **manifest["selection"],
            "parent_limit": MAX_PARENT_LIMIT,
            "parent_selection": "minimal_morgan_ranked_prefix_filling_both_25_record_arms",
            "parent_backfill_after_filter": True,
            "full_panel_required": True,
        },
    )
    manifest["inputs"]["l2_builder"] = {
        "path": str(builder), "sha256": sha256_file(builder),
    }
    manifest["inputs"]["shared_v2_builder"] = {
        "path": str(Path(base.__file__).resolve()), "sha256": sha256_file(Path(base.__file__)),
    }
    return manifest


def build(task: str, subset: str = "valid", output_root: Path | None = None) -> dict:
    if task not in three_pools.MODULES or subset != "valid":
        raise ValueError("Morgan L2 V3 supports BBB/oral validation only")
    destination = (output_root or OUTPUT_ROOT / task / "scaffold" / subset).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing cache: {destination}")
    binding, eligibility = _eligibility(task)
    l1_root, l1, source_root, source = base._source_contract(task, subset)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    database = temporary / "retrieval.sqlite3"
    try:
        shutil.copy2(l1_root / l1["database"], database)
        database.chmod(0o600)
        with sqlite3.connect(database) as output, sqlite3.connect(
            f"file:{(source_root / 'scores.sqlite3').resolve()}?mode=ro&immutable=1", uri=True,
        ) as source_db:
            output.row_factory = source_db.row_factory = sqlite3.Row
            base._create_tables(output)
            summary = _populate(output, source_db, binding, task)
            identity = base._digest({"schema_version": SCHEMA_VERSION, "task": task,
                "l1_content_id": l1["content_id"], "source": source["cache_sha256"],
                "eligibility": eligibility["record_map_sha256"],
                "builder": sha256_file(Path(__file__)), "summary": dict(sorted(summary.items()))})
            output.execute("UPDATE metadata SET value=? WHERE key='schema_version'", (SCHEMA_VERSION,))
            output.execute("UPDATE metadata SET value=? WHERE key='content_id'", (identity,))
            output.commit()
            output.execute("VACUUM")
            if output.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Morgan L2 V3 cache failed SQLite quick_check")
        source_manifest = source_root / "VERSION.json"
        manifest = _manifest(task, subset, l1, source_manifest, source, eligibility,
                             database, summary, identity)
        (temporary / "VERSION.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.chmod(database, 0o444)
        os.chmod(temporary / "VERSION.json", 0o444)
        temporary.rename(destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(three_pools.MODULES))
    parser.add_argument("--subset", default="valid", choices=("valid",))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.task, args.subset, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
