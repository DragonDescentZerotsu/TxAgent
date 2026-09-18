"""Build Gold-v2 hybrid-disjoint Morgan control and top-quartile semantic caches."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any

from rdkit import DataStructs
import pyarrow.parquet as pq

from predict.retrieval.policies import (
    bemis_murcko_scaffold,
    normalize_molecule_identity,
    seeded_rank_tie_key,
    standardize_smiles_and_fp,
)
from predict.utils.json import sha256_file

from . import build_indirect_morgan_semantic_cache as v1
from . import three_pools
from .runtime import cache_profile_root


SCHEMA_VERSION = "indirect_morgan_semantic_l2_l4.v3"
PROFILE = "indirect_morgan_semantic_l2_l4_v3"
OUTPUT_ROOT = cache_profile_root(PROFILE)
LEVELS = ("L2", "L3", "L4")
CONTROL_MODE = "morgan_parent_control"
SEMANTIC_MODE = "morgan_parent_semantic"
LLM_SEMANTIC_MODE = "morgan_parent_llm_semantic"
MODES = (CONTROL_MODE, SEMANTIC_MODE, LLM_SEMANTIC_MODE)
CONTROL_LIMIT = 25
SEMANTIC_LIMIT = 25
LLM_SEMANTIC_LIMIT = 100
PER_PARENT_LIMIT = 10
WEIGHTS = Path(__file__).resolve().parents[3] / "semantic_buckets/policies/semantic_weighted_top10_v2.json"
GOLD = {
    "bbb_martins": Path("data/gold_labels/BBB_Martins/v2/scaffold"),
    "bioavailability_ma": Path("data/gold_labels/Bioavailability_Ma/v2/scaffold"),
}
CORE_PAYLOAD_FIELDS = {
    "source_row_uid", "canonical_record_id", "canonical_smiles", "source_id",
    "pair_bucket_key", "finite_scalar_value", "canonical_measurement_text",
    "canonical_unit_text", "measurement_kind", "canonical_measurement_scale_id",
    "canonical_category_id", "canonical_transporter_identifier",
}


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _eligibility(path: Path, task: str) -> tuple[dict[str, dict[str, Any]], dict]:
    frame = pq.read_table(path).to_pandas()
    required = {"canonical_record_id", "source_row_uid", "level", "semantic_bucket_id",
                "level_rank", "retrieval_eligible"}
    if not required <= set(frame) or frame.source_row_uid.duplicated().any():
        raise ValueError("current eligibility mapping has an invalid schema or repeated UID")
    if set(frame.level) != set(LEVELS):
        raise ValueError("current eligibility mapping must cover exactly L2-L4")
    binding = {str(row.source_row_uid): row._asdict() for row in frame.itertuples(index=False)}
    return binding, {"path": str(path.resolve()), "sha256": sha256_file(path), "rows": len(frame),
                     "task_id": task}


def _top_quartile(binding: dict[str, dict[str, Any]]) -> tuple[set[str], dict]:
    counts: dict[str, Counter] = defaultdict(Counter)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for row in binding.values():
        if not bool(row["retrieval_eligible"]):
            continue
        level, bucket = str(row["level"]), str(row["semantic_bucket_id"])
        counts[level][bucket] += 1
        ranks[level][bucket] = int(row["level_rank"])
    selected, audit = set(), {}
    for level in LEVELS:
        total, cumulative, buckets = sum(counts[level].values()), 0, []
        target = math.ceil(total * 0.25)
        for bucket in sorted(counts[level], key=lambda key: (ranks[level][key], key)):
            selected.add(bucket)
            buckets.append(bucket)
            cumulative += counts[level][bucket]
            if cumulative >= target:
                break
        audit[level] = {"eligible_records": total, "target_records": target,
                        "selected_records": cumulative, "selected_buckets": len(buckets),
                        "last_bucket_rank": max((ranks[level][key] for key in buckets), default=None)}
    return selected, audit


def _record_payloads(task: str, binding: dict[str, dict[str, Any]]) -> list[dict]:
    module = three_pools.MODULES[task]
    fields, union = module._source_fields()
    available = set(pq.read_schema(module.STAGE3).names)
    columns = sorted((CORE_PAYLOAD_FIELDS | union) & available)
    rows = pq.read_table(module.STAGE3, columns=columns).to_pylist()
    output = []
    for raw in rows:
        info = binding.get(str(raw["source_row_uid"]))
        if info is None:
            continue
        identity = normalize_molecule_identity(str(raw["canonical_smiles"] or ""))
        if identity.status != "ok" or not identity.parent_smiles:
            continue
        output.append(_record_payload(task, raw, info, fields, identity))
    if len(output) != len(binding):
        raise ValueError(f"{len(binding) - len(output)} current semantic records lack usable structures")
    return output


def _record_payload(task: str, raw: dict, info: dict, fields: dict, identity: Any) -> dict:
    source = str(raw["source_id"])
    payload = {**raw, "record_id": str(raw["canonical_record_id"]), "task_id": task,
               "progressive_level": str(info["level"]),
               "canonical_smiles": identity.parent_smiles,
               "parent_id": identity.parent_inchi_key or identity.parent_smiles,
               "identity_smiles_forms": sorted({str(raw["canonical_smiles"]), identity.parent_smiles}),
               "source_fields": {field: raw.get(field) for field in fields[source]}}
    return {"record_id": payload["record_id"], "parent_id": payload["parent_id"],
            "parent_smiles": identity.parent_smiles, "level": str(info["level"]),
            "semantic_bucket_id": str(info["semantic_bucket_id"]),
            "semantic_rank": int(info["level_rank"]),
            "retrieval_eligible": bool(info["retrieval_eligible"]), "payload": payload}


def _parent_index(records: list[dict]) -> tuple[dict, dict, dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    smiles, scaffolds, fingerprints = {}, {}, {}
    for row in records:
        parent = row["parent_id"]
        grouped[row["level"], parent].append(row)
        smiles[parent] = row["parent_smiles"]
    for parent, value in smiles.items():
        scaffolds[parent] = bemis_murcko_scaffold(value)
        fingerprints[parent] = standardize_smiles_and_fp(value)[2]
        if fingerprints[parent] is None:
            raise ValueError(f"parent lacks Morgan fingerprint: {parent}")
    return grouped, smiles, {"scaffolds": scaffolds, "fingerprints": fingerprints}


def _eligible_parents(query: dict, level: str, parents: list[str], metadata: dict) -> list[str]:
    identity = normalize_molecule_identity(str(query["drug"]))
    if identity.status != "ok" or not identity.parent_smiles:
        raise ValueError(f"Gold-v2 query identity is invalid: {query['benchmark_row_id']}")
    query_parent = identity.parent_inchi_key or identity.parent_smiles
    query_scaffold = bemis_murcko_scaffold(identity.parent_smiles)
    output = []
    for parent in parents:
        if parent == query_parent:
            continue
        if level == "L2" and query_scaffold and metadata["scaffolds"][parent] == query_scaffold:
            continue
        output.append(parent)
    return output


def _rank_parents(query: dict, level: str, grouped: dict, metadata: dict) -> list[tuple[str, float]]:
    parents = sorted(parent for found, parent in grouped if found == level)
    parents = _eligible_parents(query, level, parents, metadata)
    query_smiles = query["molecule_identity"]["parent_smiles"]
    query_fp = standardize_smiles_and_fp(query_smiles)[2]
    similarities = DataStructs.BulkTanimotoSimilarity(
        query_fp, [metadata["fingerprints"][parent] for parent in parents]
    )
    return sorted(zip(parents, similarities, strict=True), key=lambda row: (-row[1], row[0]))


def _sample(query: dict, task: str, level: str, ranked: list[tuple[str, float]],
            grouped: dict, semantic_buckets: set[str], semantic: bool,
            limit: int) -> list[dict]:
    selected = []
    for parent_rank, (parent, similarity) in enumerate(ranked, 1):
        rows = grouped[level, parent]
        if semantic:
            rows = [row for row in rows if row["retrieval_eligible"]
                    and row["semantic_bucket_id"] in semantic_buckets]
        rows = sorted(rows, key=lambda row: seeded_rank_tie_key(
            0, task, query["benchmark_row_id"], level, parent, row["record_id"]))
        for within, source in enumerate(rows[:PER_PARENT_LIMIT], 1):
            selected.append(dict(source, parent_rank=parent_rank,
                                 within_parent_rank=within, morgan_similarity=float(similarity)))
            if len(selected) == limit:
                return selected
    raise ValueError(f"{query['benchmark_row_id']}/{level} cannot fill {limit} records")


def _create_tables(connection: sqlite3.Connection) -> None:
    v1._create_tables(connection)


def _insert_record(connection: sqlite3.Connection, keys: dict, row: dict) -> int:
    record_id = row["record_id"]
    if record_id in keys:
        return keys[record_id]
    key = keys[record_id] = len(keys) + 1
    connection.execute("INSERT INTO records VALUES (?,?,?,?,?,?)", (
        key, record_id, row["parent_id"], row["parent_smiles"], row["level"],
        json.dumps(row["payload"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    ))
    return key


def _store_selection(connection: sqlite3.Connection, query_id: int, level: str, mode: str,
                     selected: list[dict], ranked: list[tuple[str, float]], grouped: dict,
                     semantic_buckets: set[str], keys: dict, summary: Counter) -> None:
    for rank, row in enumerate(selected, 1):
        key = _insert_record(connection, keys, row)
        connection.execute("INSERT INTO assignments VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
            query_id, level, mode, rank, row["parent_rank"], row["within_parent_rank"], key,
            row["semantic_bucket_id"], row["semantic_rank"], row["retrieval_eligible"],
            row["morgan_similarity"],
        ))
    _store_audits(connection, query_id, level, mode, selected, ranked, grouped,
                  semantic_buckets)
    summary[f"{level}/{mode}/records"] += len(selected)


def _store_audits(connection: sqlite3.Connection, query_id: int, level: str,
                  mode: str, selected: list[dict], ranked: list[tuple[str, float]],
                  grouped: dict, semantic_buckets: set[str]) -> None:
    by_parent: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        by_parent[row["parent_id"]].append(row)
    for parent, rows in sorted(by_parent.items(), key=lambda item: item[1][0]["parent_rank"]):
        first = rows[0]
        available = grouped[level, parent]
        semantic = [row for row in available if row["retrieval_eligible"]
                    and row["semantic_bucket_id"] in semantic_buckets]
        connection.execute("INSERT INTO parent_assignments VALUES (?,?,?,?,?,?,?,?,?)", (
            query_id, level, mode, first["parent_rank"], parent, first["morgan_similarity"],
            len(available), len(semantic), len(rows),
        ))
    candidates = [row for parent, _ in ranked for row in grouped[level, parent]]
    eligible = [row for row in candidates if row["retrieval_eligible"]
                and row["semantic_bucket_id"] in semantic_buckets]
    limit = LLM_SEMANTIC_LIMIT if mode == LLM_SEMANTIC_MODE else CONTROL_LIMIT
    values = (query_id, level, mode, len(candidates), len(candidates), len(eligible),
              len(candidates) - len(eligible), 0, len(ranked), len(by_parent),
              len(selected), limit - len(selected))
    connection.execute("INSERT INTO selection_counts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", values)


def _populate(connection: sqlite3.Connection, task: str, queries: list[dict], records: list[dict],
              semantic_buckets: set[str]) -> Counter:
    grouped, _smiles, metadata = _parent_index(records)
    connection.executemany("INSERT INTO queries VALUES (?,?)", (
        (index, row["molecule_identity"]["parent_smiles"]) for index, row in enumerate(queries)
    ))
    connection.executemany("INSERT INTO benchmark_queries VALUES (?,?,?)", (
        (row["benchmark_row_id"], row["drug"], index) for index, row in enumerate(queries)
    ))
    keys, summary = {}, Counter()
    for query_id, query in enumerate(queries):
        for level in LEVELS:
            ranked = _rank_parents(query, level, grouped, metadata)
            control = _sample(query, task, level, ranked, grouped, semantic_buckets,
                              False, CONTROL_LIMIT)
            semantic = _sample(query, task, level, ranked, grouped, semantic_buckets,
                               True, SEMANTIC_LIMIT)
            llm_semantic = _sample(query, task, level, ranked, grouped, semantic_buckets,
                                   True, LLM_SEMANTIC_LIMIT)
            _store_selection(connection, query_id, level, CONTROL_MODE, control, ranked,
                             grouped, semantic_buckets, keys, summary)
            _store_selection(connection, query_id, level, SEMANTIC_MODE, semantic, ranked,
                             grouped, semantic_buckets, keys, summary)
            _store_selection(connection, query_id, level, LLM_SEMANTIC_MODE, llm_semantic, ranked,
                             grouped, semantic_buckets, keys, summary)
    summary.update(queries=len(queries), unique_records=len(keys))
    return summary


def build(task: str, eligibility: Path, subset: str = "valid",
          output_root: Path | None = None) -> dict[str, Any]:
    if task not in GOLD or subset != "valid":
        raise ValueError("V3 cache supports Gold-v2 BBB/oral validation only")
    destination = (output_root or OUTPUT_ROOT / task / "hybrid" / subset).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing cache: {destination}")
    binding, eligibility_receipt = _eligibility(eligibility, task)
    semantic_buckets, quartile = _top_quartile(binding)
    records = _record_payloads(task, binding)
    query_path = GOLD[task] / f"{subset}_molecule_condition_labels.jsonl"
    queries = _jsonl(query_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    return _build_into(temporary, destination, task, subset, records, queries,
                       semantic_buckets, quartile, eligibility_receipt, query_path)


def _build_into(temporary: Path, destination: Path, task: str, subset: str,
                records: list[dict], queries: list[dict], semantic_buckets: set[str],
                quartile: dict, eligibility: dict, query_path: Path) -> dict[str, Any]:
    database = temporary / "retrieval.sqlite3"
    try:
        with sqlite3.connect(database) as connection:
            _create_tables(connection)
            summary = _populate(connection, task, queries, records, semantic_buckets)
            content_id = _content_id(task, subset, summary, quartile, eligibility, query_path)
            connection.executemany("INSERT INTO metadata VALUES (?,?)", (
                ("schema_version", SCHEMA_VERSION), ("content_id", content_id)))
            connection.commit()
            connection.execute("VACUUM")
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("V3 cache failed SQLite quick_check")
        manifest = _manifest(task, subset, database, content_id, summary, quartile,
                             eligibility, query_path)
        (temporary / "VERSION.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.chmod(database, 0o444)
        os.chmod(temporary / "VERSION.json", 0o444)
        temporary.rename(destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _content_id(task: str, subset: str, summary: Counter, quartile: dict,
                eligibility: dict, query_path: Path) -> str:
    return _digest({"schema": SCHEMA_VERSION, "task": task, "subset": subset,
                    "eligibility": eligibility["sha256"], "quartile": quartile,
                    "summary": dict(summary), "query_ledger": sha256_file(query_path),
                    "weights": sha256_file(WEIGHTS),
                    "builder": sha256_file(Path(__file__))})


def _manifest(task: str, subset: str, database: Path, content_id: str, summary: Counter,
              quartile: dict, eligibility: dict, query_path: Path) -> dict[str, Any]:
    policy = {"L2": "scaffold_disjoint", "L3": "parent_disjoint", "L4": "parent_disjoint"}
    return {"schema_version": SCHEMA_VERSION, "status": "complete", "task_id": task,
            "subset": subset, "gold_release": "v2", "database": database.name,
            "database_sha256": sha256_file(database), "content_id": content_id,
            "levels_in_cache": list(LEVELS), "tie_seed": 0, "contains_direct_records": False,
            "neighbor_identity_policy": "hybrid_l2_scaffold_l3_l4_parent_disjoint",
            "neighbor_identity_policy_by_level": policy,
            "capacities": {"control_records_per_level": CONTROL_LIMIT,
                           "semantic_records_per_level": SEMANTIC_LIMIT,
                           "llm_semantic_candidates_per_level": LLM_SEMANTIC_LIMIT,
                           "records_per_parent": PER_PARENT_LIMIT},
            "selection": {"candidate_pool": "all", "full_panel_required": True,
                          "semantic_metadata_visible": False,
                          "semantic_universe": "whole_bucket_roundup_to_25_percent_of_eligible_records",
                          "parent_order": "morgan_tanimoto_then_parent_id",
                          "within_parent_order": "deterministic_seeded_random"},
            "semantic_universe_by_level": quartile,
            "assignment_counts": dict(sorted(summary.items())), "eligibility": eligibility,
            "inputs": {"query_ledger": {"path": str(query_path.resolve()),
                                         "sha256": sha256_file(query_path)},
                       "weights": {"path": str(WEIGHTS.resolve()), "sha256": sha256_file(WEIGHTS)},
                       "builder": {"path": str(Path(__file__).resolve()),
                                   "sha256": sha256_file(Path(__file__))}}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(GOLD), required=True)
    parser.add_argument("--eligibility", type=Path, required=True)
    parser.add_argument("--subset", default="valid", choices=("valid",))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.task, args.eligibility, args.subset, args.output_root),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
