"""Publish a Skin Gold+TDC L1 cache using only existing TDC assay scores.

Gold scores are the mean of scored TDC candidates in each query's joint
Morgan-top-100 universe. No model is loaded or called by this builder.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from rdkit import DataStructs

from data.processing.gold_labels.conditioned_benchmark import tdc_task_root
from predict.retrieval.policies import standardize_smiles_and_fp
from predict.utils.json import read_jsonl, sha256_file

from .build_ranked_retrieval import _digest
from .build_ranked_uid_retrieval import _schema, _write_complete
from .build_skin_gold_v1_l1_adapter import GOLD, _cards, _projection
from .build_tdc_mixed_ranked_l1 import _projection_entry
from .ranked_uid_retrieval import CAPACITY, SCHEMA_VERSION
from .runtime import cache_profile_root


TASK = "skin_reaction"
PROFILE = "flat_v5/tdc_v1/skin_reaction/l1/assay_transfer/joint_gold_tdc/cached_mean_v2"
TEST_PROFILE = "flat_v5/tdc_v1/skin_reaction/l1/assay_transfer/joint_gold_tdc/cached_mean_test_v1"
TDC_PROFILE = "ranked_level_retrieval_tdc_v1_l1_assay_task_best_v1"


def _load_tdc_cache(subset: str) -> tuple[Path, dict[str, dict[str, float]], dict[str, list[str]]]:
    index_path = cache_profile_root(TDC_PROFILE) / TASK / "RELEASE_INDEX.json"
    index = json.loads(index_path.read_text())
    entry = index["splits"][subset]["levels"]["L1"]
    manifest_path = index_path.parent / entry["manifest"]
    if sha256_file(manifest_path) != entry["manifest_sha256"]:
        raise ValueError("TDC L1 manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    members: dict[str, list[str]] = defaultdict(list)
    with sqlite3.connect(manifest_path.parent / manifest["database"]) as connection:
        for query, parent, score in connection.execute(
            "SELECT benchmark_row_id,parent_id,assay_transfer_score FROM rankings"
        ):
            if score is None or not 0 <= float(score) <= 1:
                raise ValueError(f"Missing TDC assay score: {query}/{parent}")
            scores[str(query)][str(parent)] = float(score)
        for context, uid in connection.execute(
            "SELECT context_id,source_row_uid FROM context_records ORDER BY context_id,within_context_rank"
        ):
            members[str(context)].append(str(uid))
    return index_path, scores, members


def build(output_root: Path, *, subset: str = "valid", profile: str = PROFILE) -> Path:
    """Build one immutable split release under a canonical cache root."""
    if subset not in {"valid", "test"}:
        raise ValueError(f"Unsupported TDC Skin split: {subset}")
    destination = output_root / TASK
    if destination.exists():
        raise FileExistsError(destination)
    gold = _cards()
    tdc_path = tdc_task_root(TASK) / "train_molecule_condition_labels.jsonl"
    tdc = {str(row["benchmark_row_id"]): row for row in read_jsonl(tdc_path)}
    if set(gold) & set(tdc):
        raise ValueError("Gold and TDC context IDs overlap")
    contexts = {**gold, **tdc}
    by_parent: dict[str, list[str]] = defaultdict(list)
    for context_id, row in contexts.items():
        by_parent[str(row["molecule_identity_key"])].append(context_id)
    parents = sorted(by_parent)
    fingerprints = [
        standardize_smiles_and_fp(str(contexts[by_parent[parent][0]]["molecule_identity"]["parent_smiles"]))[2]
        for parent in parents
    ]
    if any(fp is None for fp in fingerprints):
        raise ValueError("Invalid training parent fingerprint")
    tdc_index_path, cached_scores, tdc_members = _load_tdc_cache(subset)
    query_path = tdc_task_root(TASK) / f"{subset}_molecule_condition_labels.jsonl"
    queries = read_jsonl(query_path)
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".skin_joint_l1_", dir=output_root) as temporary:
        root = Path(temporary)
        level = root / "scaffold" / subset / "L1"
        level.mkdir(parents=True)
        database = level / "rankings.sqlite3"
        connection = sqlite3.connect(database)
        _schema(connection, "L1")
        used_contexts: set[str] = set()
        imputed, cached, means = 0, 0, {}
        query_counts = {}
        for query in queries:
            query_id = str(query["benchmark_row_id"])
            query_parent = str(query["molecule_identity_key"])
            smiles = str(query["molecule_identity"]["parent_smiles"])
            fp = standardize_smiles_and_fp(smiles)[2]
            similarities = DataStructs.BulkTanimotoSimilarity(fp, fingerprints)
            candidates = sorted(
                (
                    (float(similarity), parent)
                    for similarity, parent in zip(similarities, parents)
                    if parent != query_parent and (
                        not query.get("bemis_murcko_scaffold")
                        or str(contexts[by_parent[parent][0]].get("bemis_murcko_scaffold") or "")
                        != str(query["bemis_murcko_scaffold"])
                    )
                ), key=lambda value: (-value[0], value[1]),
            )[:CAPACITY]
            if len(candidates) != CAPACITY:
                raise ValueError(f"Only {len(candidates)} Skin parents for {query_id}")
            # A TDC parent can have a Gold card too; identify TDC by membership.
            tdc_values = [
                cached_scores[query_id][parent] for _, parent in candidates
                if any(cid in tdc for cid in by_parent[parent]) and parent in cached_scores[query_id]
            ]
            if not tdc_values:
                raise ValueError(f"No cached TDC scores in joint top 100: {query_id}")
            mean = sum(tdc_values) / len(tdc_values)
            means[query_id] = {"mean": mean, "tdc_count": len(tdc_values)}
            connection.execute("INSERT INTO queries VALUES (?,?,?,?)", (query_id, str(query["drug"]), query_parent, smiles))
            rows = []
            for rank, (similarity, parent) in enumerate(candidates, 1):
                ids = sorted(by_parent[parent], key=lambda cid: (0 if cid in gold else 1, cid))
                used_contexts.update(ids)
                scored = []
                for cid in ids:
                    if cid in gold:
                        score, origin = mean, "imputed_joint_pool_mean"
                        imputed += 1
                    else:
                        if parent not in cached_scores[query_id]:
                            raise ValueError(f"TDC score absent for joint parent: {query_id}/{parent}")
                        score, origin = cached_scores[query_id][parent], "cached_tdc"
                        cached += 1
                    scored.append((score, cid, origin))
                score, chosen, origin = min(scored, key=lambda item: (-item[0], item[1]))
                chosen_row = contexts[chosen]
                member_count = len(chosen_row["source_row_uids"]) if chosen in gold else len(tdc_members[chosen])
                rows.append((
                    query_id, parent, parent,
                    str(chosen_row["molecule_identity"]["parent_smiles"]), similarity,
                    rank, 1, rank, score, None, ids[0], chosen,
                    len(contexts[ids[0]]["source_row_uids"]) if ids[0] in gold else len(tdc_members[ids[0]]), member_count,
                    f"{origin}:{query_id}:{chosen}",
                ))
            assay_order = sorted(rows, key=lambda row: (-row[8], row[1]))
            assay_ranks = {row[1]: rank for rank, row in enumerate(assay_order, 1)}
            connection.executemany("INSERT INTO rankings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
                (*row[:9], assay_ranks[row[1]], *row[10:]) for row in rows
            ])
            query_counts[query_id] = {
                "candidate_parents": CAPACITY,
                "morgan_candidate_records": sum(row[12] for row in rows),
                "assay_candidate_records": sum(row[13] for row in rows),
            }
        for cid in sorted(used_contexts):
            row = contexts[cid]
            parent = str(row["molecule_identity_key"])
            connection.execute("INSERT INTO contexts VALUES (?,?,?,?)", (
                cid, parent, str(row.get("condition_group") or ""), int(row["Y"]),
            ))
            members = (
                row["source_row_uids"]
                if cid in gold else tdc_members[cid]
            )
            if not members:
                raise ValueError(f"Context has no evidence members: {cid}")
            connection.executemany("INSERT INTO context_records VALUES (?,?,?)", [
                (cid, uid, rank) for rank, uid in enumerate(members, 1)
            ])
        gold_projection = _projection(root, gold, used_contexts & set(gold))
        identity = {
            "schema_version": SCHEMA_VERSION, "status": "complete", "task_id": TASK,
            "subset": subset, "level": "L1", "pool": "fixed", "capacity": CAPACITY,
            "parent_capacity": CAPACITY, "query_count": len(queries),
            "stored_rows": len(queries) * CAPACITY, "query_counts": query_counts,
            "label_release": {"benchmark": "tdc"},
            "l1_sources": ["gold_v1", "tdc_v1"],
            "neighbor_identity_policy": "scaffold_disjoint",
            "shared_candidate_universe": True, "assay_transfer_status": "complete",
            "score_policy": "tdc_cached_gold_joint_pool_mean.v1",
            "score_counts": {"cached_tdc": cached, "imputed_joint_pool_mean": imputed},
            "query_imputation_means": means,
            "inputs": {
                "gold_train_sha256": sha256_file(GOLD / "train_molecule_condition_labels.jsonl"),
                "tdc_train_sha256": sha256_file(tdc_path),
                "tdc_query_sha256": sha256_file(query_path),
                "tdc_cache_index": str(tdc_index_path.resolve()),
                "tdc_cache_index_sha256": sha256_file(tdc_index_path),
            },
        }
        manifest = _write_complete(connection, database, task=TASK, subset=subset,
                                   level="L1", identity=identity, target=level)
        connection.close()
        tdc_evidence = _projection_entry(tdc_index_path)
        gold_evidence = {
            "manifest": str((destination / "evidence" / "VERSION.json").resolve()),
            "manifest_sha256": sha256_file(gold_projection),
            "record_count": json.loads(gold_projection.read_text())["record_count"],
            "content_id": json.loads(gold_projection.read_text())["content_id"],
        }
        index = {
            "schema_version": "ranked_uid_task_release_index.v1",
            "selection_contract": SCHEMA_VERSION, "profile": profile,
            "task_id": TASK, "status": "complete", "pool": "all",
            "label_release": {"benchmark": "tdc"},
            "l1_sources": ["gold_v1", "tdc_v1"], "parent_capacity": CAPACITY,
            "levels_independent": True, "ranking_modes": ["morgan", "assay-transfer"],
            "assay_transfer_status": "complete",
            "neighbor_identity_policy_by_level": {"L1": "scaffold_disjoint"},
            "evidence_sources": [gold_evidence, tdc_evidence],
            "splits": {subset: {"levels": {"L1": {
                "manifest": str((level / "VERSION.json").relative_to(root)),
                "manifest_sha256": sha256_file(level / "VERSION.json"),
                "content_id": manifest["content_id"],
            }}}},
        }
        (root / "RELEASE_INDEX.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
        (root / "VALIDATION.json").write_text(json.dumps({
            "status": "complete", "query_count": len(queries),
            "score_counts": identity["score_counts"], "score_policy": identity["score_policy"],
            "content_id": _digest(identity),
        }, indent=2, sort_keys=True) + "\n")
        os.replace(root, destination)
    return destination / "RELEASE_INDEX.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("valid", "test"), default="valid")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    profile = PROFILE if args.subset == "valid" else TEST_PROFILE
    print(build(args.output_root or cache_profile_root(profile), subset=args.subset, profile=profile))


if __name__ == "__main__":
    main()
