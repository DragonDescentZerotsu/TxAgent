"""Publish matched L2 panels with a five-record parent/bucket semantic cap."""
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
from . import semantic_bucket_selection as selector
from . import three_pools
from .build_context_l2_morgan_semantic_cache import _eligibility
from .runtime import cache_profile_root


SCHEMA_VERSION = "l1_context_morgan_semantic_l2.v4"
PROFILE = "l1_context_morgan_semantic_l2_v4"
OUTPUT_ROOT = cache_profile_root(PROFILE)
MAX_PARENT_LIMIT = 100
SEMANTIC_PARENT_BUCKET_LIMIT = 5


def _select(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return selector.select_global_molecule_records(
        rows, method=method, limit=base.TOTAL_RECORD_LIMIT,
        per_parent_limit=base.PER_PARENT_LIMIT,
        per_parent_bucket_limit=(
            SEMANTIC_PARENT_BUCKET_LIMIT if method == "semantic" else None
        ),
    )


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
        if all(len(_select(candidates, method)) == base.TOTAL_RECORD_LIMIT
               for method in ("control", "semantic")):
            return ordered[:rank], grouped
    raise ValueError("Morgan-100 parents cannot fill both matched 25-record panels")


def _store_mode(
    connection: sqlite3.Connection, query_id: int, mode: str, parents: list[str],
    grouped: dict[str, list[dict]], record_keys: dict[str, int], summary: Counter,
) -> None:
    candidates = [row for parent in parents for row in grouped[parent]]
    selected = _select(candidates, "semantic" if mode.endswith("semantic") else "control")
    selected_counts = Counter(str(row["parent_id"]) for row in selected)
    within_counts: Counter[str] = Counter()
    for selection_rank, row in enumerate(selected, 1):
        parent = str(row["parent_id"])
        within_counts[parent] += 1
        record_id = str(row["record_id"])
        if record_id not in record_keys:
            record_keys[record_id] = len(record_keys) + 1
            connection.execute("INSERT INTO l2_records VALUES (?,?,?,?,?)", (
                record_keys[record_id], record_id, parent, row["parent_smiles"],
                json.dumps(row["payload"], sort_keys=True, separators=(",", ":")),
            ))
        connection.execute("INSERT INTO l2_assignments VALUES (?,?,?,?,?,?,?,?,?,?)", (
            query_id, mode, selection_rank, row["parent_rank"], within_counts[parent],
            record_keys[record_id], row["semantic_bucket_id"], row["semantic_rank"],
            row["retrieval_eligible"], row["morgan_similarity"],
        ))
    for parent_rank, parent in enumerate(parents, 1):
        rows = grouped[parent]
        connection.execute("INSERT INTO l2_parent_assignments VALUES (?,?,?,?,?,?,?,?)", (
            query_id, mode, parent_rank, parent, rows[0]["morgan_similarity"], len(rows),
            sum(row["retrieval_eligible"] is True for row in rows), selected_counts[parent],
        ))
    mapped = sum(row["semantic_rank"] is not None for row in candidates)
    eligible = sum(row["retrieval_eligible"] is True for row in candidates)
    values = (query_id, mode, len(candidates), mapped, eligible, mapped - eligible,
              len(candidates) - mapped, len(parents), len(selected_counts), len(selected),
              base.TOTAL_RECORD_LIMIT - len(selected))
    connection.execute("INSERT INTO l2_selection_counts VALUES (?,?,?,?,?,?,?,?,?,?,?)", values)
    summary[f"{mode}_records"] += len(selected)
    summary[f"{mode}_shortfall"] += base.TOTAL_RECORD_LIMIT - len(selected)


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
            _store_mode(output, query_id, mode, parents, grouped, record_keys, summary)
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
    selector_path = Path(selector.__file__).resolve()
    manifest.update(
        schema_version=SCHEMA_VERSION,
        capacities={**manifest["capacities"], "l2_molecules": MAX_PARENT_LIMIT,
                    "l2_records_per_parent_semantic_bucket": SEMANTIC_PARENT_BUCKET_LIMIT},
        selection={
            **manifest["selection"], "parent_limit": MAX_PARENT_LIMIT,
            "parent_selection": "minimal_morgan_ranked_prefix_filling_both_25_record_arms",
            "parent_backfill_after_filter": True, "full_panel_required": True,
            "semantic_records_per_parent_bucket": SEMANTIC_PARENT_BUCKET_LIMIT,
        },
    )
    manifest["inputs"]["l2_builder"] = {
        "path": str(builder), "sha256": sha256_file(builder),
    }
    manifest["inputs"]["selection_helper"] = {
        "path": str(selector_path), "sha256": sha256_file(selector_path),
    }
    manifest["inputs"]["shared_v2_builder"] = {
        "path": str(Path(base.__file__).resolve()), "sha256": sha256_file(Path(base.__file__)),
    }
    return manifest


def build(task: str, subset: str = "valid", output_root: Path | None = None) -> dict:
    if task not in three_pools.MODULES or subset != "valid":
        raise ValueError("Morgan L2 V4 supports BBB/oral validation only")
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
            selector_hash = sha256_file(Path(selector.__file__))
            identity = base._digest({"schema_version": SCHEMA_VERSION, "task": task,
                "l1_content_id": l1["content_id"], "source": source["cache_sha256"],
                "eligibility": eligibility["record_map_sha256"],
                "builder": sha256_file(Path(__file__)), "selector": selector_hash,
                "summary": dict(sorted(summary.items()))})
            output.execute("UPDATE metadata SET value=? WHERE key='schema_version'", (SCHEMA_VERSION,))
            output.execute("UPDATE metadata SET value=? WHERE key='content_id'", (identity,))
            output.commit()
            output.execute("VACUUM")
            if output.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Morgan L2 V4 cache failed SQLite quick_check")
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
