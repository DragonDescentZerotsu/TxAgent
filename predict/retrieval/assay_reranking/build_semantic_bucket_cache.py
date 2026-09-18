"""Publish the top-100-parent semantic/pair-bucket retrieval cache."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import tempfile
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from rdkit import DataStructs

from data.processing.gold_labels.conditioned_benchmark import split_path
from predict.retrieval.policies import normalize_molecule_identity, seeded_rank_tie_key
from predict.utils.json import read_jsonl, sha256_file

from . import three_pools
from .build_cache_matched_v2 import Writer, _schema, _similarity
from .build_cache_matched_v3 import _l5_catalog, _load_l5_payloads
from .cache_matched_v2 import FIXED_POOL
from .semantic_bucket_selection import select_records


SCHEMA_VERSION = "semantic_bucket_reranking.v1"
from predict.retrieval.assay_reranking.runtime import cache_profile_root

ROOT = cache_profile_root("semantic_bucket_reranking_v1")
SOURCE_ROOT = three_pools.ROOT
L1_ROOT = cache_profile_root("cache_matched_retrieval_v3")
CAPACITY = 50


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _semantic_index(
    path: Path, task: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    manifest_path = path / "manifest.json"
    mapping_path = path / "record_relevance_map.parquet"
    ranking_path = path / "relevance_bucket_rankings.parquet"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("task") != task:
        raise ValueError(f"Completed current-task semantic ranking required: {path}")
    mapping = pd.read_parquet(mapping_path)
    rankings = pd.read_parquet(ranking_path)
    required_map = {"source_row_uid", "level", "node_key"}
    required_rank = {"level", "node_key", "level_percentile"}
    if not required_map <= set(mapping) or not required_rank <= set(rankings):
        raise ValueError("Semantic artifacts lack UID, level, node, pair-bucket, or percentile fields")
    if rankings.duplicated(["level", "node_key"]).any():
        raise ValueError("Semantic rankings contain duplicate identities")
    reviewed = mapping.merge(
        rankings[["level", "node_key", "level_percentile"]],
        on=["level", "node_key"], how="left", validate="many_to_one",
    )
    if reviewed.level_percentile.isna().any():
        raise ValueError("Semantic ranking does not cover every mapped record")
    duplicate_rows = reviewed[reviewed.source_row_uid.duplicated(keep=False)]
    variants = duplicate_rows.groupby("source_row_uid").agg(
        levels=("level", "nunique"), nodes=("node_key", "nunique")
    )
    ambiguous_uids = set(variants.index[(variants.levels > 1) | (variants.nodes > 1)])
    reviewed = reviewed[~reviewed.source_row_uid.isin(ambiguous_uids)].drop_duplicates(
        "source_row_uid"
    )
    module = three_pools.MODULES[task]
    current = pq.read_table(
        module.STAGE3,
        columns=["source_row_uid", "canonical_record_id", "pair_bucket_key"],
    ).to_pandas()
    levels = pq.read_table(
        module.LEVEL_MAPPING, columns=["source_row_uid", "level"]
    ).to_pandas()
    current = current.merge(levels, on="source_row_uid", how="inner", validate="one_to_one")
    current["level"] = current.level.map(lambda value: f"L{int(value)}")
    joined = current.merge(
        reviewed[["source_row_uid", "level", "level_percentile"]],
        on=["source_row_uid", "level"], how="left", validate="one_to_one",
    )
    joined["level_percentile"] = joined.level_percentile.fillna(-1.0)
    index = {
        str(row.source_row_uid): {
            "level": str(row.level),
            "pair_bucket_key": str(row.pair_bucket_key),
            "semantic_percentile": float(row.level_percentile),
        }
        for row in joined.itertuples(index=False)
    }
    inputs = {
        name: {"path": str(file.resolve()), "sha256": sha256_file(file)}
        for name, file in {
            "semantic_manifest": manifest_path,
            "semantic_record_map": mapping_path,
            "semantic_rankings": ranking_path,
        }.items()
    }
    inputs["current_stage3"] = {
        "path": str(module.STAGE3.resolve()), "sha256": sha256_file(module.STAGE3),
    }
    inputs["current_level_mapping"] = {
        "path": str(module.LEVEL_MAPPING.resolve()), "sha256": sha256_file(module.LEVEL_MAPPING),
    }
    audit = {
        "binding": "stable_source_row_uid_and_level",
        "pair_bucket_source": "current_V10",
        "reviewed_unique_uids": int(reviewed.source_row_uid.nunique()),
        "current_mapped_uids": len(index),
        "current_uids_without_rank": sum(
            row["semantic_percentile"] < 0 for row in index.values()
        ),
        "ambiguous_historical_uids_excluded": len(ambiguous_uids),
    }
    return index, inputs, audit


def _candidate(
    *, task: str, benchmark_row_id: str, level: str, parent_id: str,
    parent_smiles: str, payload: dict[str, Any], score: float | None,
    similarity: float, semantics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    uid = str(payload.get("source_row_uid") or "")
    semantic = semantics.get(uid)
    if semantic is None or semantic["level"] != level:
        raise ValueError(f"Current semantic ranking lacks {task}/{level}/{uid}")
    if str(payload.get("pair_bucket_key") or "") != semantic["pair_bucket_key"]:
        raise ValueError(f"Current pair-bucket identity differs from semantic review: {uid}")
    record_id = str(payload["record_id"])
    return {
        "record_id": record_id,
        "parent_id": parent_id,
        "parent_smiles": parent_smiles,
        "payload": payload,
        "pair_bucket_key": semantic["pair_bucket_key"],
        "semantic_percentile": semantic["semantic_percentile"],
        "morgan_similarity": similarity,
        "assay_transfer_score": score,
        "tie_key": seeded_rank_tie_key(
            0, task, benchmark_row_id, level, parent_id, record_id,
        ),
    }


def _source_candidates(
    connection: sqlite3.Connection, query_id: int, *, task: str,
    benchmark_row_id: str, level: str, query_parent_smiles: str,
    semantics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """SELECT r.parent_id,r.payload,s.transfer_probability
           FROM assignments AS a
           JOIN records AS r USING(record_key)
           JOIN scores AS s USING(score_key)
           WHERE a.query_id=? AND a.pool='all' AND a.level=?""",
        (query_id, level),
    )
    output = []
    for raw in rows:
        payload = json.loads(raw["payload"])
        parent_smiles = str(payload["canonical_smiles"])
        output.append(_candidate(
            task=task, benchmark_row_id=benchmark_row_id, level=level,
            parent_id=str(raw["parent_id"]), parent_smiles=parent_smiles,
            payload=payload, score=float(raw["transfer_probability"]),
            similarity=_similarity(query_parent_smiles, parent_smiles), semantics=semantics,
        ))
    if len({row["parent_id"] for row in output}) != three_pools.POOL_SIZE:
        raise ValueError(f"Source cache is not a top-{three_pools.POOL_SIZE} parent pool")
    return output


def _l5_candidates(
    task: str, subset: str, semantics: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    catalog = _l5_catalog(task)
    queries = {
        str(row["benchmark_row_id"]): row
        for row in read_jsonl(split_path(task, subset).with_name(
            f"{subset}_molecule_condition_labels.jsonl"
        ))
    }
    selected: dict[str, list[dict[str, Any]]] = {}
    selected_ids: set[str] = set()
    staged: dict[str, list[tuple[dict[str, Any], float]]] = {}
    for qid, query in queries.items():
        identity = normalize_molecule_identity(query["drug"])
        query_parent_smiles = str(identity.parent_smiles or "")
        query_parent_id = str(identity.parent_inchi_key or query_parent_smiles)
        query_scaffold = str(query.get("bemis_murcko_scaffold") or "")
        query_fp = catalog["fingerprints"].get(query_parent_smiles)
        if query_fp is None:
            from predict.retrieval.policies import standardize_smiles_and_fp
            query_fp = standardize_smiles_and_fp(query_parent_smiles)[2]
        groups = {
            parent: records for parent, records in catalog["grouped"]["all"].items()
            if parent != query_parent_id and (
                not query_scaffold or query_scaffold not in catalog["parent_scaffolds"][parent]
            )
        }
        similarities = {
            parent: float(DataStructs.TanimotoSimilarity(
                query_fp, catalog["fingerprints"][catalog["parent_smiles"][parent]]
            ))
            for parent in groups
        }
        parents = sorted(groups, key=lambda parent: (-similarities[parent], parent))[:three_pools.POOL_SIZE]
        rows = [(record, similarities[parent]) for parent in parents for record in groups[parent]]
        staged[qid] = rows
        selected_ids.update(str(record["record_id"]) for record, _ in rows)
    payloads = _load_l5_payloads(task, selected_ids, catalog)
    for qid, rows in staged.items():
        candidates = []
        for base, similarity in rows:
            payload = payloads[str(base["record_id"])]
            payload["pair_bucket_key"] = semantics[str(base["source_row_uid"])]["pair_bucket_key"]
            candidates.append(_candidate(
                task=task, benchmark_row_id=qid, level="L5",
                parent_id=str(base["parent_id"]), parent_smiles=str(base["canonical_smiles"]),
                payload=payload, score=None, similarity=similarity, semantics=semantics,
            ))
        selected[qid] = candidates
    inputs = dict(catalog["inputs"])
    inputs["l5_records"] = {
        "path": str(catalog["module"].STAGE3.resolve()),
        "sha256": sha256_file(catalog["module"].STAGE3),
    }
    return selected, inputs


def _copy_l1(source: sqlite3.Connection, output: sqlite3.Connection, writer: Writer) -> None:
    source.row_factory = sqlite3.Row
    for row in source.execute(
        """SELECT a.*,r.payload FROM assignments AS a JOIN records AS r USING(record_key)
           WHERE a.pool=? AND a.level='L1' ORDER BY a.query_id,a.morgan_rank""",
        (FIXED_POOL,),
    ):
        writer.assignment(
            query_id=int(row["query_id"]), pool=FIXED_POOL, level="L1",
            parent_id=str(row["parent_id"]), parent_smiles=str(row["parent_smiles"]),
            payload=json.loads(row["payload"]), within_parent_rank=int(row["within_parent_rank"]),
            similarity=float(row["morgan_similarity"]), morgan_rank=int(row["morgan_rank"]),
            assay_score=float(row["assay_transfer_score"]), assay_rank=int(row["assay_rank"]),
        )
    output.executemany(
        "INSERT INTO selection_counts VALUES (?,?,?,?,?)",
        source.execute(
            """SELECT query_id,pool,level,candidate_record_count,candidate_parent_count
               FROM selection_counts WHERE pool='fixed' AND level='L1'"""
        ).fetchall(),
    )


def build(*, task: str, subset: str, semantic_artifact: Path, output_root: Path | None = None) -> dict[str, Any]:
    if task not in three_pools.MODULES or subset not in {"valid", "test"}:
        raise ValueError("Unsupported task or split")
    output_root = (output_root or ROOT / task / "scaffold" / subset).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to replace existing cache: {output_root}")
    semantics, semantic_inputs, semantic_audit = _semantic_index(
        semantic_artifact.resolve(), task
    )
    source_root = SOURCE_ROOT / task / "scaffold" / subset
    source_manifest_path = source_root / "VERSION.json"
    source_database = source_root / "scores.sqlite3"
    l1_root = L1_ROOT / task / "scaffold" / subset
    l1_manifest_path = l1_root / "VERSION.json"
    l1_database = l1_root / "retrieval.sqlite3"
    source_manifest = json.loads(source_manifest_path.read_text())
    l1_manifest = json.loads(l1_manifest_path.read_text())
    if source_manifest.get("status") != "complete" or l1_manifest.get("status") != "complete":
        raise ValueError("Complete source-score and L1 caches are required")

    levels = [f"L{i}" for i in range(2, (6 if task == "bbb_martins" else 7))]
    l5_by_query, l5_inputs = _l5_candidates(task, subset, semantics)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    database = temporary / "retrieval.sqlite3"
    output = sqlite3.connect(database)
    _schema(output)
    output.execute("PRAGMA user_version=4")
    writer = Writer(output)
    assignment_counts: dict[str, int] = defaultdict(int)
    eligibility_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with sqlite3.connect(f"file:{l1_database.resolve()}?mode=ro&immutable=1", uri=True) as l1, \
            sqlite3.connect(f"file:{source_database.resolve()}?mode=ro&immutable=1", uri=True) as source:
        for query_id, parent_id, parent_smiles in l1.execute(
            "SELECT query_id,query_parent_id,query_parent_smiles FROM queries ORDER BY query_id"
        ):
            output.execute("INSERT INTO queries VALUES (?,?,?)", (query_id, parent_id, parent_smiles))
        for row in l1.execute("SELECT benchmark_row_id,drug,query_id FROM benchmark_queries ORDER BY query_id"):
            output.execute("INSERT INTO benchmark_queries VALUES (?,?,?)", row)
        _copy_l1(l1, output, writer)
        assignment_counts["fixed/L1"] = output.execute(
            "SELECT count(*) FROM assignments WHERE pool='fixed' AND level='L1'"
        ).fetchone()[0]
        source_queries = {
            str(qid): int(query_id)
            for qid, query_id in source.execute("SELECT benchmark_row_id,query_id FROM benchmark_queries")
        }
        for qid, query_id in output.execute("SELECT benchmark_row_id,query_id FROM benchmark_queries ORDER BY query_id"):
            parent_smiles = output.execute(
                "SELECT query_parent_smiles FROM queries WHERE query_id=?", (query_id,)
            ).fetchone()[0]
            for level in levels:
                candidates = l5_by_query[str(qid)] if level == "L5" else _source_candidates(
                    source, source_queries[str(qid)], task=task, benchmark_row_id=str(qid),
                    level=level, query_parent_smiles=str(parent_smiles), semantics=semantics,
                )
                eligible = [row for row in candidates if row["semantic_percentile"] >= 20]
                eligibility_counts[level]["candidate_records"] += len(candidates)
                eligibility_counts[level]["eligible_records"] += len(eligible)
                rankings = {
                    method: select_records(eligible, method=method, limit=len(eligible))
                    for method in (("morgan",) if level == "L5" else ("morgan", "assay_transfer"))
                }
                ranks = {
                    method: {row["record_id"]: rank for rank, row in enumerate(rows, 1)}
                    for method, rows in rankings.items()
                }
                chosen = {row["record_id"]: row for rows in rankings.values() for row in rows[:CAPACITY]}
                output.execute(
                    "INSERT INTO selection_counts VALUES (?,?,?,?,?)",
                    (query_id, "all", level, len(eligible), len({row["parent_id"] for row in eligible})),
                )
                within: dict[str, int] = defaultdict(int)
                for row in sorted(chosen.values(), key=lambda value: (ranks["morgan"][value["record_id"]], value["record_id"])):
                    within[row["parent_id"]] += 1
                    assay_rank = ranks.get("assay_transfer", {}).get(row["record_id"])
                    writer.assignment(
                        query_id=query_id, pool="all", level=level,
                        parent_id=row["parent_id"], parent_smiles=row["parent_smiles"], payload=row["payload"],
                        within_parent_rank=within[row["parent_id"]], similarity=row["morgan_similarity"],
                        morgan_rank=ranks["morgan"][row["record_id"]],
                        assay_score=(row["assay_transfer_score"] if assay_rank is not None else None),
                        assay_rank=assay_rank,
                    )
                    assignment_counts[f"all/{level}"] += 1
    identity = {
        "schema_version": SCHEMA_VERSION, "task_id": task, "subset": subset,
        "source_cache_sha256": sha256_file(source_database),
        "l1_cache_sha256": sha256_file(l1_database), "semantic_inputs": semantic_inputs,
        "assignment_counts": dict(sorted(assignment_counts.items())),
    }
    content_id = _digest(identity)
    output.executemany("INSERT INTO metadata VALUES (?,?)", [
        ("schema_version", SCHEMA_VERSION), ("content_id", content_id),
    ])
    output.commit()
    output.execute("VACUUM")
    output.close()
    manifest = {
        "schema_version": SCHEMA_VERSION, "status": "complete", "task_id": task,
        "subset": subset, "database": "retrieval.sqlite3", "content_id": content_id,
        "tie_seed": 0, "joint_materialized": False, "pools": ["all"],
        "fixed_pool_levels": ["L1"],
        "capacities": {"l1_molecules": 10, "l1_records_per_molecule": 10,
                       "later_records_per_level": CAPACITY, "morgan_parent_pool": three_pools.POOL_SIZE},
        "selection": {
            "parent_gate": "top_100_distinct_scaffold_disjoint_morgan_no_floor",
            "semantic_filter": "within_level_percentile_gte_20",
            "allocation": "one_record_per_pair_bucket_per_pass",
            "modes": ["morgan", "assay-transfer"], "L5": "morgan",
        },
        "semantic_ranking_binding": semantic_audit,
        "assignment_counts": dict(sorted(assignment_counts.items())),
        "eligibility_counts": {level: dict(counts) for level, counts in sorted(eligibility_counts.items())},
        "record_count": len(writer.record_keys),
        "inputs": {
            "source_manifest": {"path": str(source_manifest_path.resolve()), "sha256": sha256_file(source_manifest_path)},
            "source_database": {"path": str(source_database.resolve()), "sha256": identity["source_cache_sha256"]},
            "l1_manifest": {"path": str(l1_manifest_path.resolve()), "sha256": sha256_file(l1_manifest_path)},
            "l1_database": {"path": str(l1_database.resolve()), "sha256": identity["l1_cache_sha256"]},
            **semantic_inputs, **l5_inputs,
        },
        "database_sha256": sha256_file(database),
        "validations": {"source_scores_reused": True, "scaffold_disjoint_source_pool": True,
                        "similarity_floor": None, "runtime_sqlite_only": True},
    }
    (temporary / "VERSION.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.rename(output_root)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(three_pools.MODULES))
    parser.add_argument("--subset", default="valid", choices=("valid", "test"))
    parser.add_argument("--semantic-artifact", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(task=args.task, subset=args.subset,
                           semantic_artifact=args.semantic_artifact,
                           output_root=args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
