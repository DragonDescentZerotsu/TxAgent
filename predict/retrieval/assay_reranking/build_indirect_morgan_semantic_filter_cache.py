"""Publish V1-matched controls and 100 query-filter candidates for L2-L4."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

from predict.utils.json import sha256_file

from . import build_indirect_morgan_semantic_cache as v1
from .runtime import cache_profile_root
from .semantic_bucket_selection import select_global_molecule_records


SCHEMA_VERSION = "indirect_morgan_semantic_l2_l4.v2"
PROFILE = "indirect_morgan_semantic_l2_l4_v2"
OUTPUT_ROOT = cache_profile_root(PROFILE)
SEMANTIC_LIMIT = 100
CONTROL_LIMIT = 25
PER_PARENT_LIMIT = 15
PER_PARENT_BUCKET_LIMIT = 8


def _semantic_selection(rows: list[dict]) -> list[dict]:
    return select_global_molecule_records(
        rows, method="semantic", limit=SEMANTIC_LIMIT,
        per_parent_limit=PER_PARENT_LIMIT,
        per_parent_bucket_limit=PER_PARENT_BUCKET_LIMIT,
    )


def _semantic_prefix(rows: list[dict]) -> tuple[list[str], dict[str, list[dict]]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["parent_id"]].append(row)
    ordered = sorted(grouped, key=lambda key: (-grouped[key][0]["morgan_similarity"], key))
    candidates: list[dict] = []
    for rank, parent in enumerate(ordered[:v1.PARENT_LIMIT], 1):
        for row in grouped[parent]:
            row["parent_rank"] = rank
        candidates.extend(grouped[parent])
        if len(_semantic_selection(candidates)) == SEMANTIC_LIMIT:
            return ordered[:rank], grouped
    raise ValueError("Morgan-100 parents cannot fill 100 semantic candidates")


def _insert_record(connection, keys, row, level):
    record_id = row["record_id"]
    if record_id in keys:
        return keys[record_id]
    key = keys[record_id] = len(keys) + 1
    connection.execute("INSERT INTO records VALUES (?,?,?,?,?,?)", (
        key, record_id, row["parent_id"], row["parent_smiles"], level,
        json.dumps(row["payload"], sort_keys=True, separators=(",", ":")),
    ))
    return key


def _store_counts(connection, query_id, level, mode, parents, grouped, selected, limit):
    candidates = [row for parent in parents for row in grouped[parent]]
    selected_counts = Counter(row["parent_id"] for row in selected)
    for rank, parent in enumerate(parents, 1):
        rows = grouped[parent]
        connection.execute("INSERT INTO parent_assignments VALUES (?,?,?,?,?,?,?,?,?)", (
            query_id, level, mode, rank, parent, rows[0]["morgan_similarity"], len(rows),
            sum(row["retrieval_eligible"] is True for row in rows), selected_counts[parent],
        ))
    mapped = sum(row["semantic_rank"] is not None for row in candidates)
    eligible = sum(row["retrieval_eligible"] is True for row in candidates)
    values = (query_id, level, mode, len(candidates), mapped, eligible, mapped - eligible,
              len(candidates) - mapped, len(parents), len(selected_counts), len(selected),
              limit - len(selected))
    connection.execute("INSERT INTO selection_counts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", values)


def _store_mode(connection, query_id, level, mode, parents, grouped, keys, summary):
    candidates = [row for parent in parents for row in grouped[parent]]
    semantic = mode.endswith("semantic")
    selected = _semantic_selection(candidates) if semantic else v1._select(candidates, "control")
    within = Counter()
    for rank, row in enumerate(selected, 1):
        within[row["parent_id"]] += 1
        key = _insert_record(connection, keys, row, level)
        connection.execute("INSERT INTO assignments VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
            query_id, level, mode, rank, row["parent_rank"], within[row["parent_id"]], key,
            row["semantic_bucket_id"], row["semantic_rank"], row["retrieval_eligible"],
            row["morgan_similarity"],
        ))
    limit = SEMANTIC_LIMIT if semantic else CONTROL_LIMIT
    _store_counts(connection, query_id, level, mode, parents, grouped, selected, limit)
    summary[f"{level}/{mode}/records"] += len(selected)
    summary[f"{level}/{mode}/parent_slots"] += len(parents)


def _populate(output, source, binding, task):
    source_ids = dict(source.execute("SELECT benchmark_row_id,query_id FROM benchmark_queries"))
    benchmarks = list(source.execute(
        """SELECT b.benchmark_row_id,b.drug,q.query_smiles FROM benchmark_queries b
           JOIN queries q USING(query_id) ORDER BY b.benchmark_row_id"""
    ))
    output.executemany("INSERT INTO queries VALUES (?,?)", ((i, row[2]) for i, row in enumerate(benchmarks)))
    output.executemany("INSERT INTO benchmark_queries VALUES (?,?,?)", (
        (row[0], row[1], i) for i, row in enumerate(benchmarks)
    ))
    keys, summary = {}, Counter()
    for query_id, (benchmark, _, query_smiles) in enumerate(benchmarks):
        for level in v1.LEVELS:
            rows = v1._query_rows(source, source_ids[benchmark], level, binding, task,
                                  str(benchmark), str(query_smiles))
            control_parents, control_grouped = v1._parent_prefix(rows)
            semantic_parents, semantic_grouped = _semantic_prefix(rows)
            _store_mode(output, query_id, level, v1.MODES[0], control_parents,
                        control_grouped, keys, summary)
            _store_mode(output, query_id, level, v1.MODES[1], semantic_parents,
                        semantic_grouped, keys, summary)
    summary["queries"], summary["unique_records"] = len(benchmarks), len(keys)
    return summary


def _verify_v1_controls(task, subset, database):
    old_manifest = v1.OUTPUT_ROOT / task / "scaffold" / subset / "VERSION.json"
    old_database = old_manifest.with_name(json.loads(old_manifest.read_text())["database"])
    with sqlite3.connect(database) as new, sqlite3.connect(old_database) as old:
        for benchmark, query_id in new.execute("SELECT benchmark_row_id,query_id FROM benchmark_queries"):
            old_id = old.execute("SELECT query_id FROM benchmark_queries WHERE benchmark_row_id=?",
                                 (benchmark,)).fetchone()[0]
            query = """SELECT r.external_record_id FROM assignments a JOIN records r USING(record_key)
                       WHERE a.query_id=? AND a.level=? AND a.mode=? ORDER BY a.selection_rank"""
            for level in v1.LEVELS:
                current = [row[0] for row in new.execute(query, (query_id, level, v1.MODES[0]))]
                frozen = [row[0] for row in old.execute(query, (old_id, level, v1.MODES[0]))]
                if current != frozen:
                    raise ValueError(f"V1 control selection differs: {benchmark}/{level}")
    return {"manifest": str(old_manifest.resolve()), "manifest_sha256": sha256_file(old_manifest),
            "database_sha256": sha256_file(old_database), "control_order_identical": True}


def _manifest(task, subset, database, source_path, source, eligibility, equivalence, summary, content_id):
    return {"schema_version": SCHEMA_VERSION, "status": "complete", "task_id": task,
        "subset": subset, "database": database.name, "content_id": content_id,
        "database_sha256": sha256_file(database), "levels_in_cache": list(v1.LEVELS),
        "tie_seed": 0, "contains_direct_records": False,
        "capacities": {"parent_candidates": 100, "control_records_per_level": CONTROL_LIMIT,
            "semantic_candidates_per_level": SEMANTIC_LIMIT, "semantic_records_per_parent": PER_PARENT_LIMIT,
            "semantic_records_per_parent_semantic_bucket": PER_PARENT_BUCKET_LIMIT,
            "final_records_min": 10, "final_records_max": 25},
        "selection": {"candidate_pool": "all", "full_panel_required": True,
            "semantic_metadata_visible": False, "control": "exact_v1_control_order",
            "semantic": "minimal_morgan_parent_prefix_then_semantic_order",
            "semantic_filter": "retrieval_eligible_true_and_ranked"},
        "assignment_counts": dict(sorted(summary.items())), "eligibility": eligibility,
        "v1_control_equivalence": equivalence, "neighbor_identity_policy": source["neighbor_identity_policy"],
        "inputs": {"source_database": {"path": str(source_path.resolve()), "sha256": source["cache_sha256"]},
            "builder": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
            "v1_builder": {"path": str(Path(v1.__file__).resolve()), "sha256": sha256_file(Path(v1.__file__))}}}


def build(task: str, subset: str = "valid", output_root: Path | None = None) -> dict:
    if task not in v1.three_pools.MODULES or subset != "valid":
        raise ValueError("indirect cache supports BBB/oral validation only")
    destination = (output_root or OUTPUT_ROOT / task / "scaffold" / subset).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing cache: {destination}")
    binding, eligibility = v1._eligibility(task)
    source_path, source = v1._source(task, subset)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    database = temporary / "retrieval.sqlite3"
    try:
        with sqlite3.connect(database) as output, sqlite3.connect(
                f"file:{source_path.resolve()}?mode=ro&immutable=1", uri=True) as source_db:
            v1._create_tables(output)
            summary = _populate(output, source_db, binding, task)
            content_id = v1._digest({"schema": SCHEMA_VERSION, "task": task,
                "source": source["cache_sha256"], "eligibility": eligibility["record_map_sha256"],
                "builder": sha256_file(Path(__file__)), "summary": dict(sorted(summary.items()))})
            output.executemany("INSERT INTO metadata VALUES (?,?)", (
                ("schema_version", SCHEMA_VERSION), ("content_id", content_id)))
            output.commit()
            output.execute("VACUUM")
        equivalence = _verify_v1_controls(task, subset, database)
        manifest = _manifest(task, subset, database, source_path, source, eligibility,
                             equivalence, summary, content_id)
        (temporary / "VERSION.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.chmod(database, 0o444); os.chmod(temporary / "VERSION.json", 0o444)
        temporary.rename(destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(v1.three_pools.MODULES))
    parser.add_argument("--subset", default="valid", choices=("valid",))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.task, args.subset, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
