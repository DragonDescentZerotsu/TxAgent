"""Publish global top-10 expert-weighted L2 selections beside frozen controls."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any

import pandas as pd

from predict.retrieval.policies import (
    decide_candidate,
    normalize_molecule_identity,
    seeded_rank_tie_key,
)
from predict.utils.json import sha256_file

from . import three_pools
from .build_cache_matched_v2 import _similarity
from .build_context_l2_semantic_cache import _l1_visible_record_ids
from .runtime import cache_profile_root
from .semantic_bucket_selection import select_semantic_weighted


SCHEMA_VERSION = "l1_context_semantic_weighted_l2.v1"
PROFILE = "l1_context_semantic_weighted_l2_v1"
OUTPUT_ROOT = cache_profile_root(PROFILE)
BASE_ROOT = cache_profile_root("l1_context_semantic_l2_v1")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = REPOSITORY_ROOT / "semantic_buckets/policies/semantic_weighted_top10_v2.json"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _policy(task: str) -> tuple[dict[str, tuple[str, int, float]], dict[str, Any]]:
    document = json.loads(POLICY_PATH.read_text())
    if document.get("version") != "semantic_weighted_top10.v2":
        raise ValueError("unsupported semantic weight policy")
    selection = document.get("selection") or {}
    if selection != {
        "record_limit": 12,
        "per_bucket_limit": 8,
        "utility": "expert_weight_times_morgan_similarity",
        "rank_limit": 10,
    }:
        raise ValueError("semantic weight selection contract changed")
    spec = document["tasks"][task]
    ranking_path = REPOSITORY_ROOT / spec["ranking_path"]
    record_path = REPOSITORY_ROOT / spec["record_ranking_path"]
    for path, expected in (
        (ranking_path, spec["ranking_sha256"]),
        (record_path, spec["record_ranking_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"semantic ranking input changed: {path}")

    rankings = pd.read_parquet(
        ranking_path, columns=["level", "semantic_bucket_id", "level_rank"]
    )
    record_rankings = pd.read_parquet(
        record_path,
        columns=[
            "canonical_record_id", "source_row_uid", "level",
            "semantic_bucket_id", "level_rank",
        ],
    )
    resolved: dict[str, list[dict[str, Any]]] = {}
    for level, level_spec in spec["levels"].items():
        bucket_ids = level_spec["bucket_ids"]
        weights = [float(value) for value in level_spec["weights"]]
        rationales = level_spec["rationales"]
        if not (len(bucket_ids) == len(weights) == len(rationales) == 10):
            raise ValueError(f"top-10 policy length changed: {task}/{level}")
        if max(weights) != 1 or any(
            value < 0 or value > 1 or value * 10 != round(value * 10)
            for value in weights
        ):
            raise ValueError(f"weights must be normalized tenths: {task}/{level}")
        observed = (
            rankings[rankings.level.eq(level)]
            .nsmallest(10, "level_rank")
            .sort_values("level_rank")
        )
        if observed.level_rank.tolist() != list(range(1, 11)) or (
            observed.semantic_bucket_id.tolist() != bucket_ids
        ):
            raise ValueError(f"pinned bucket IDs disagree with rankings: {task}/{level}")
        level_records = record_rankings[
            record_rankings.level.eq(level)
            & record_rankings.level_rank.le(10)
        ]
        exemplars = (
            level_records.sort_values("canonical_record_id")
            .groupby("semantic_bucket_id")["canonical_record_id"]
            .apply(lambda values: [str(value) for value in values.head(3)])
            .to_dict()
        )
        resolved[level] = [
            {
                "rank": rank,
                "semantic_bucket_id": bucket,
                "expert_weight": weight,
                "rationale": rationale,
                "representative_record_ids": exemplars.get(bucket, []),
            }
            for rank, (bucket, weight, rationale) in enumerate(
                zip(bucket_ids, weights, rationales), 1
            )
        ]

    l2 = record_rankings[record_rankings.level.eq("L2")]
    if l2.source_row_uid.duplicated().any():
        raise ValueError(f"duplicate semantic identity: {task}/L2")
    weights = {
        item["semantic_bucket_id"]: (item["rank"], item["expert_weight"])
        for item in resolved["L2"]
    }
    binding = {
        str(row.source_row_uid): (
            str(row.semantic_bucket_id),
            int(row.level_rank),
            weights.get(str(row.semantic_bucket_id), (0, 0.0))[1],
        )
        for row in l2.itertuples(index=False)
        if str(row.semantic_bucket_id) in weights
    }
    return binding, {
        "version": document["version"],
        "selection": selection,
        "levels": resolved,
        "inputs": {
            str(POLICY_PATH.resolve()): sha256_file(POLICY_PATH),
            str(ranking_path.resolve()): spec["ranking_sha256"],
            str(record_path.resolve()): spec["record_ranking_sha256"],
        },
    }


def build(task: str, subset: str = "valid", output_root: Path | None = None) -> dict[str, Any]:
    if task not in three_pools.MODULES or subset != "valid":
        raise ValueError("the weighted context-L2 experiment supports BBB/oral validation only")
    destination = (output_root or OUTPUT_ROOT / task / "scaffold" / subset).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing cache: {destination}")
    binding, policy = _policy(task)
    base_root = BASE_ROOT / task / "scaffold" / subset
    base_manifest_path = base_root / "VERSION.json"
    base_manifest = json.loads(base_manifest_path.read_text())
    base_database = base_root / str(base_manifest.get("database", "retrieval.sqlite3"))
    if (
        base_manifest.get("schema_version") != "l1_context_semantic_l2.v1"
        or base_manifest.get("status") != "complete"
        or sha256_file(base_database) != base_manifest.get("database_sha256")
    ):
        raise ValueError("complete frozen context-L2 control cache required")

    records, source_inputs, issues = three_pools.load_source(task)
    if issues:
        raise ValueError(f"global V10 source has unresolved records: {issues}")
    grouped: dict[str, dict[str, Any]] = {}
    inventory_counts: Counter[tuple[str, str]] = Counter()
    for record in records.values():
        if record["progressive_level"] != "L2":
            continue
        assignment = binding.get(str(record.get("source_row_uid") or ""))
        if assignment is None:
            continue
        bucket, rank, weight = assignment
        parent = grouped.setdefault(
            record["parent_id"],
            {
                "identity_smiles_forms": set(),
                "records": [],
            },
        )
        parent["identity_smiles_forms"].update(record["identity_smiles_forms"])
        parent["records"].append((record, bucket, rank, weight))
        inventory_counts[bucket, "records"] += 1
    for parent_id, parent in grouped.items():
        identity = normalize_molecule_identity(sorted(parent["identity_smiles_forms"])[0])
        resolved_id = identity.parent_inchi_key or identity.parent_smiles
        if identity.status != "ok" or not identity.parent_smiles or resolved_id != parent_id:
            raise ValueError(f"cannot resolve global parent identity: {parent_id}")
        parent["parent_smiles"] = identity.parent_smiles
        for bucket in {item[1] for item in parent["records"]}:
            inventory_counts[bucket, "parents"] += 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    database = temporary / "retrieval.sqlite3"
    shutil.copy2(base_database, database)
    database.chmod(0o664)
    output = sqlite3.connect(database)
    output.row_factory = sqlite3.Row
    output.executescript(
        """
        ALTER TABLE l2_assignments ADD COLUMN expert_weight REAL;
        ALTER TABLE l2_assignments ADD COLUMN semantic_utility REAL;
        ALTER TABLE l2_selection_counts ADD COLUMN weighted_candidate_record_count INTEGER;
        ALTER TABLE l2_selection_counts ADD COLUMN weighted_candidate_parent_count INTEGER;
        ALTER TABLE l2_selection_counts ADD COLUMN weighted_candidate_bucket_count INTEGER;
        ALTER TABLE l2_selection_counts ADD COLUMN weighted_capped_capacity INTEGER;
        CREATE TABLE l2_weighted_bucket_counts(
          query_id INTEGER NOT NULL,semantic_bucket_id TEXT NOT NULL,
          semantic_rank INTEGER NOT NULL,expert_weight REAL NOT NULL,
          eligible_record_count INTEGER NOT NULL,selected_record_count INTEGER NOT NULL,
          PRIMARY KEY(query_id,semantic_bucket_id)
        ) WITHOUT ROWID;
        """
    )
    record_keys = {
        str(row[0]): int(row[1])
        for row in output.execute("SELECT external_record_id,record_key FROM l2_records")
    }
    next_record_key = max(record_keys.values(), default=0) + 1
    summary: Counter[str] = Counter()
    failures = []
    query_rows = output.execute(
        """SELECT b.benchmark_row_id,b.drug,b.query_id,q.query_parent_id,
                  q.query_parent_smiles
           FROM benchmark_queries AS b JOIN queries AS q USING(query_id)
           ORDER BY b.query_id"""
    ).fetchall()
    for query in query_rows:
        identity = normalize_molecule_identity(str(query["drug"]))
        expected_parent = identity.parent_inchi_key or identity.parent_smiles
        if (
            identity.status != "ok"
            or expected_parent != str(query["query_parent_id"])
            or identity.parent_smiles != str(query["query_parent_smiles"])
        ):
            raise ValueError(f"query identity changed: {query['benchmark_row_id']}")
        seen = _l1_visible_record_ids(output, int(query["query_id"]))
        candidates_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        candidate_parents, candidate_records = set(), 0
        for parent_id, parent in grouped.items():
            if any(
                decide_candidate(identity, {"canonical_smiles": form}, "scaffold_disjoint").excluded
                for form in parent["identity_smiles_forms"]
            ):
                continue
            similarity = _similarity(
                str(query["query_parent_smiles"]), str(parent["parent_smiles"])
            )
            parent_contributed = False
            for record, bucket, rank, weight in parent["records"]:
                if weight == 0 or record["record_id"] in seen:
                    continue
                candidates_by_bucket[bucket].append(
                    {
                        "record_id": record["record_id"],
                        "parent_id": parent_id,
                        "parent_smiles": parent["parent_smiles"],
                        "payload": record,
                        "semantic_bucket_id": bucket,
                        "semantic_rank": rank,
                        "expert_weight": weight,
                        "morgan_similarity": similarity,
                        "tie_key": seeded_rank_tie_key(
                            0, task, str(query["benchmark_row_id"]), "L2",
                            parent_id, record["record_id"],
                        ),
                    }
                )
                candidate_records += 1
                parent_contributed = True
            if parent_contributed:
                candidate_parents.add(parent_id)
        capped = []
        for bucket_rows in candidates_by_bucket.values():
            bucket_rows.sort(key=lambda row: (-row["morgan_similarity"], row["tie_key"]))
            capped.extend(bucket_rows[: policy["selection"]["per_bucket_limit"]])
        selected = select_semantic_weighted(
            capped,
            limit=policy["selection"]["record_limit"],
            per_bucket_limit=policy["selection"]["per_bucket_limit"],
        )
        if len(selected) != policy["selection"]["record_limit"]:
            failures.append({
                "benchmark_row_id": str(query["benchmark_row_id"]),
                "eligible_records": candidate_records,
                "represented_buckets": len(candidates_by_bucket),
                "capped_capacity": len(capped),
            })
            continue
        selected_counts = Counter(row["semantic_bucket_id"] for row in selected)
        for bucket, rows in sorted(candidates_by_bucket.items()):
            first = rows[0]
            output.execute(
                "INSERT INTO l2_weighted_bucket_counts VALUES (?,?,?,?,?,?)",
                (
                    int(query["query_id"]), bucket, int(first["semantic_rank"]),
                    float(first["expert_weight"]), len(rows), selected_counts[bucket],
                ),
            )
        for selection_rank, row in enumerate(selected, 1):
            record_id = row["record_id"]
            if record_id not in record_keys:
                record_keys[record_id] = next_record_key
                next_record_key += 1
                output.execute(
                    "INSERT INTO l2_records VALUES (?,?,?,?,?)",
                    (
                        record_keys[record_id], record_id, row["parent_id"],
                        row["parent_smiles"],
                        json.dumps(row["payload"], sort_keys=True, separators=(",", ":")),
                    ),
                )
            output.execute(
                "INSERT INTO l2_assignments VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    int(query["query_id"]), "semantic_weighted", selection_rank,
                    record_keys[record_id], row["semantic_bucket_id"],
                    row["semantic_rank"], None, row["morgan_similarity"],
                    row["expert_weight"], row["semantic_utility"],
                ),
            )
        output.execute(
            """UPDATE l2_selection_counts
               SET weighted_candidate_record_count=?,weighted_candidate_parent_count=?,
                   weighted_candidate_bucket_count=?,weighted_capped_capacity=?
               WHERE query_id=?""",
            (
                candidate_records, len(candidate_parents), len(candidates_by_bucket),
                len(capped), int(query["query_id"]),
            ),
        )
        summary["semantic_weighted_assignments"] += len(selected)
        summary["semantic_weighted_selected_parents"] += len(
            {row["parent_id"] for row in selected}
        )
        output.commit()
    if failures:
        output.close()
        shutil.rmtree(temporary)
        raise ValueError(f"weighted top-10 capacity failure: {failures[:10]}")

    summary["l2_unique_records"] = int(
        output.execute("SELECT COUNT(*) FROM l2_records").fetchone()[0]
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task,
        "subset": subset,
        "base_content_id": base_manifest["content_id"],
        "base_database_sha256": base_manifest["database_sha256"],
        "policy_sha256": policy["inputs"][str(POLICY_PATH.resolve())],
        "global_source_inputs": source_inputs,
        "assignment_counts": dict(sorted(summary.items())),
    }
    content_id = _digest(identity)
    output.execute("UPDATE metadata SET value=? WHERE key='schema_version'", (SCHEMA_VERSION,))
    output.execute("UPDATE metadata SET value=? WHERE key='content_id'", (content_id,))
    output.commit()
    output.execute("VACUUM")
    output.close()

    inventory = {
        bucket: {
            "records": inventory_counts[bucket, "records"],
            "parents": inventory_counts[bucket, "parents"],
        }
        for bucket in sorted({key[0] for key in inventory_counts})
    }
    manifest = {
        **base_manifest,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "database": database.name,
        "content_id": content_id,
        "capacities": {
            **base_manifest["capacities"],
            "l2_records": policy["selection"]["record_limit"],
            "semantic_weighted_per_bucket": policy["selection"]["per_bucket_limit"],
            "semantic_rank_limit": policy["selection"]["rank_limit"],
        },
        "assignment_counts": {
            **base_manifest["assignment_counts"], **dict(sorted(summary.items()))
        },
        "l2_record_count": summary["l2_unique_records"],
        "record_count": int(base_manifest["l1_record_count"]) + summary["l2_unique_records"],
        "selection": {
            **base_manifest["selection"],
            "semantic_weighted": "expert_weight_times_morgan_similarity_then_seeded_tie",
            "semantic_weighted_candidate_pool": "global_scaffold_disjoint_nonzero_top10_v10_l2_records",
            "semantic_metadata_visible": False,
        },
        "semantic_weight_policy": policy,
        "semantic_weighted_inventory": inventory,
        "inputs": {
            **base_manifest.get("inputs", {}),
            "base_manifest": {
                "path": str(base_manifest_path.resolve()),
                "sha256": sha256_file(base_manifest_path),
            },
            "base_database": {
                "path": str(base_database.resolve()),
                "sha256": base_manifest["database_sha256"],
            },
            "weighted_builder": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
            "weighted_selector": {
                "path": str(Path(__file__).with_name("semantic_bucket_selection.py").resolve()),
                "sha256": sha256_file(Path(__file__).with_name("semantic_bucket_selection.py")),
            },
            **policy["inputs"],
            **source_inputs,
        },
        "database_sha256": sha256_file(database),
    }
    (temporary / "VERSION.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    temporary.rename(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(three_pools.MODULES))
    parser.add_argument("--subset", default="valid", choices=("valid",))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.task, args.subset, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
