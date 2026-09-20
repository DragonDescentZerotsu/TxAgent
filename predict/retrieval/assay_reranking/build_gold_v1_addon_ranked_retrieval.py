"""Build Morgan-only Gold-v1 L1 caches for Ames, DILI, and Carcinogens."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import DataStructs

from predict.retrieval.policies import normalize_molecule_identity, standardize_smiles_and_fp
from predict.utils.json import read_jsonl, sha256_file

from .build_ranked_retrieval import _clean, _digest
from .build_ranked_uid_retrieval import _schema, _write_complete
from .ranked_uid_retrieval import CAPACITY, SCHEMA_VERSION
from .runtime import cache_profile_root


PROFILE = "ranked_level_retrieval_gold_v1_addon_v1"
TASKS = {
    "ames": ("Ames", "ames", "v10"),
    "dili": ("DILI", "dili", "v10"),
    "carcinogens": ("Carcinogens", "carcinogens", "v10_main_universe_v1"),
}
SUBSETS = ("valid", "test")
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = cache_profile_root(PROFILE)


def _gold_root(task: str) -> Path:
    return REPO_ROOT / "data/gold_labels" / TASKS[task][0] / "v1/scaffold"


def _card(row: dict[str, Any]) -> tuple[str, str, str]:
    identity = normalize_molecule_identity(str(row["drug"]))
    parent_id = str(row["molecule_identity_key"])
    if identity.status != "ok" or identity.parent_inchi_key != parent_id:
        raise ValueError(f"Frozen Gold parent cannot be reproduced: {row['benchmark_row_id']}")
    return parent_id, identity.parent_smiles, str(row.get("bemis_murcko_scaffold") or "")


def _load_train(task: str) -> tuple[Path, list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    path = _gold_root(task) / "train.jsonl"
    rows = read_jsonl(path)
    cards = {str(row["benchmark_row_id"]): row for row in rows}
    if len(cards) != len(rows):
        raise ValueError(f"Duplicate Gold train card: {task}")
    parents: dict[str, str] = {}
    order = []
    for row in rows:
        parent_id, _, _ = _card(row)
        if parent_id not in parents:
            parents[parent_id] = str(row["benchmark_row_id"])
            order.append(parent_id)
    return path, rows, cards, order


def _write_projection(task: str, root: Path, cards: dict[str, dict[str, Any]]) -> Path:
    gold = _gold_root(task)
    membership_path = gold / "voter_membership.parquet"
    contract_path = gold / "voter_contract_manifest.json"
    membership_rows = pq.read_table(membership_path).to_pylist()
    train_rows = [row for row in membership_rows if row["split"] == "train"]
    by_uid = {str(row["source_row_uid"]): row for row in train_rows}
    if len(by_uid) != len(train_rows):
        raise ValueError(f"Gold train physical UID is reused: {task}")
    expected_cards = set(cards)
    if {str(row["benchmark_row_id"]) for row in train_rows} != expected_cards:
        raise ValueError(f"Gold train membership does not cover every card: {task}")

    stage1_path = (
        REPO_ROOT / "data/evidence_libraries" / TASKS[task][1]
        / TASKS[task][2] / "01_cleaned/records.parquet"
    )
    projected = []
    seen: set[str] = set()
    for batch in pq.ParquetFile(stage1_path).iter_batches(batch_size=50_000):
        for raw in batch.to_pylist():
            uid = str(raw["source_row_uid"])
            member = by_uid.get(uid)
            if member is None:
                continue
            if uid in seen:
                raise ValueError(f"Stage-1 UID is duplicated: {task}/{uid}")
            seen.add(uid)
            card = cards[str(member["benchmark_row_id"])]
            parent_id, parent_smiles, _ = _card(card)
            source_fields = {
                key: value for key, value in raw.items()
                if value is not None and str(value).strip() and key not in {
                    "source_row_uid", "cleaned_record_id", "canonical_smiles",
                }
            }
            payload = {
                "task_id": task, "progressive_level": "L1",
                "label_source": "gold_v1", "benchmark_row_id": member["benchmark_row_id"],
                "canonical_smiles": parent_smiles,
                "source_canonical_smiles": str(raw["canonical_smiles"]),
                "condition_group": str(card["condition_group"]), "Y": int(card["Y"]),
                "vote_id": str(member["vote_id"]), "vote_label": int(member["vote_label"]),
                "source_record_id": str(member["physical_source_record_id"]),
                "source_fields": source_fields,
                "source_contract": {
                    "contract_version": "gold_v1_stage1_projection.v1",
                    "source_id": str(raw["source_id"]),
                    "source_or_simply_cleaned": {key: True for key in source_fields},
                },
                "display_measurement_text": str(raw.get("measurement_text") or ""),
                "display_unit_text": str(raw.get("unit_text") or ""),
                "display_origin": "v10_stage1_source_pair",
            }
            projected.append({
                "source_row_uid": uid,
                "external_record_id": str(member["physical_source_record_id"]),
                "parent_id": parent_id, "parent_smiles": parent_smiles, "level": "L1",
                "payload": json.dumps(_clean(payload), sort_keys=True, separators=(",", ":")),
            })
    for uid in sorted(set(by_uid) - seen):
        member = by_uid[uid]
        card = cards[str(member["benchmark_row_id"])]
        parent_id, parent_smiles, _ = _card(card)
        payload = {
            "task_id": task, "progressive_level": "L1", "label_source": "gold_v1",
            "benchmark_row_id": member["benchmark_row_id"],
            "canonical_smiles": parent_smiles,
            "source_canonical_smiles": str(member["canonical_smiles"]),
            "condition_group": str(card["condition_group"]), "Y": int(card["Y"]),
            "vote_id": str(member["vote_id"]), "vote_label": int(member["vote_label"]),
            "source_record_id": str(member["physical_source_record_id"]),
            "source_fields": {
                "source_record_id": str(member["physical_source_record_id"]),
                "gold_vote_label": int(member["vote_label"]),
            },
            "source_contract": {
                "contract_version": "gold_v1_membership_fallback.v1",
                "source_id": str(member["source_id"]),
                "source_or_simply_cleaned": {
                    "source_record_id": True, "gold_vote_label": True,
                },
            },
            "display_measurement_text": f"Frozen Gold-v1 label={int(member['vote_label'])}",
            "display_unit_text": "", "display_origin": "gold_v1_membership_fallback",
        }
        projected.append({
            "source_row_uid": uid,
            "external_record_id": str(member["physical_source_record_id"]),
            "parent_id": parent_id, "parent_smiles": parent_smiles, "level": "L1",
            "payload": json.dumps(payload, sort_keys=True, separators=(",", ":")),
        })
        seen.add(uid)

    projection = root / "evidence"
    projection.mkdir(parents=True)
    records_path = projection / "records.parquet"
    schema = pa.schema([
        ("source_row_uid", pa.string()), ("external_record_id", pa.string()),
        ("parent_id", pa.string()), ("parent_smiles", pa.string()),
        ("level", pa.string()), ("payload", pa.string()),
    ])
    pq.write_table(
        pa.Table.from_pylist(sorted(projected, key=lambda row: row["source_row_uid"]), schema=schema),
        records_path, compression="zstd", row_group_size=10_000,
    )
    identity = {
        "schema_version": "ranked_evidence_projection.v1", "status": "complete",
        "task_id": task, "record_count": len(projected),
        "source_identity": "gold_v1_source_row_uid",
        "inputs": {
            "voter_membership_sha256": sha256_file(membership_path),
            "voter_contract_manifest_sha256": sha256_file(contract_path),
            "stage1_sha256": sha256_file(stage1_path),
        },
    }
    identity["content_id"] = _digest(identity)
    manifest = {
        **identity, "records": records_path.name,
        "records_sha256": sha256_file(records_path),
        "source_paths": {
            "voter_membership": str(membership_path.relative_to(REPO_ROOT)),
            "voter_contract_manifest": str(contract_path.relative_to(REPO_ROOT)),
            "stage1": str(stage1_path.relative_to(REPO_ROOT)),
        },
    }
    manifest_path = projection / "VERSION.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _build_split(
    task: str, subset: str, train_path: Path, train_rows: list[dict[str, Any]],
    cards: dict[str, dict[str, Any]], parent_order: list[str], evidence_manifest: Path,
    root: Path,
) -> dict[str, Any]:
    parent_cards: dict[str, dict[str, Any]] = {}
    fingerprints = []
    for parent_id in parent_order:
        row = next(row for row in train_rows if str(row["molecule_identity_key"]) == parent_id)
        _, parent_smiles, _ = _card(row)
        fingerprint = standardize_smiles_and_fp(parent_smiles)[2]
        if fingerprint is None:
            raise ValueError(f"Gold train parent is invalid: {task}/{parent_id}")
        parent_cards[parent_id] = row
        fingerprints.append(fingerprint)

    query_path = _gold_root(task) / f"{subset}.jsonl"
    queries = read_jsonl(query_path)
    membership_rows = pq.read_table(
        _gold_root(task) / "voter_membership.parquet",
        columns=["benchmark_row_id", "source_row_uid", "physical_member_index", "split"],
    ).to_pylist()
    members: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in membership_rows:
        if row["split"] == "train":
            members[str(row["benchmark_row_id"])].append(
                (int(row["physical_member_index"]), str(row["source_row_uid"]))
            )
    ordered_members = {
        context_id: [uid for _, uid in sorted(values)] for context_id, values in members.items()
    }

    target = root / "scaffold" / subset / "L1"
    target.mkdir(parents=True)
    database = target / "rankings.sqlite3"
    connection = sqlite3.connect(database)
    _schema(connection, "L1")
    query_ledger = []
    rankings = []
    selected_contexts: set[str] = set()
    query_counts = {}
    for query in queries:
        query_id = str(query["benchmark_row_id"])
        query_parent, query_smiles, query_scaffold = _card(query)
        query_fp = standardize_smiles_and_fp(query_smiles)[2]
        query_ledger.append((query_id, str(query["drug"]), query_parent, query_smiles))
        similarities = DataStructs.BulkTanimotoSimilarity(query_fp, fingerprints)
        candidates = sorted(
            (
                (float(similarity), parent_id, parent_cards[parent_id])
                for similarity, parent_id in zip(similarities, parent_order)
                if parent_id != query_parent
                and (not query_scaffold or _card(parent_cards[parent_id])[2] != query_scaffold)
            ),
            key=lambda item: (-item[0], item[1]),
        )[:CAPACITY]
        if len(candidates) != CAPACITY:
            raise ValueError(f"Gold query lacks {CAPACITY} scaffold-disjoint parents: {task}/{query_id}")
        member_count = 0
        for rank, (similarity, parent_id, card) in enumerate(candidates, 1):
            context_id = str(card["benchmark_row_id"])
            context_members = ordered_members[context_id]
            selected_contexts.add(context_id)
            member_count += len(context_members)
            rankings.append((
                query_id, parent_id, parent_id, _card(card)[1], similarity,
                rank, 1, rank, None, None, context_id, None,
                len(context_members), None, None,
            ))
        query_counts[query_id] = {
            "candidate_parents": CAPACITY, "morgan_candidate_records": member_count,
            "assay_candidate_records": 0,
        }
    connection.executemany("INSERT INTO queries VALUES (?,?,?,?)", query_ledger)
    connection.executemany("INSERT INTO rankings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rankings)
    connection.executemany(
        "INSERT INTO contexts VALUES (?,?,?,?)",
        [
            (context_id, str(cards[context_id]["molecule_identity_key"]),
             str(cards[context_id]["condition_group"]), int(cards[context_id]["Y"]))
            for context_id in sorted(selected_contexts)
        ],
    )
    connection.executemany(
        "INSERT INTO context_records VALUES (?,?,?)",
        [
            (context_id, uid, rank)
            for context_id in sorted(selected_contexts)
            for rank, uid in enumerate(ordered_members[context_id], 1)
        ],
    )
    identity = {
        "task_id": task, "subset": subset, "level": "L1", "pool": "fixed",
        "capacity": CAPACITY, "parent_capacity": CAPACITY,
        "gold_release": "v1", "neighbor_identity_policy": "scaffold_disjoint",
        "shared_candidate_universe": True,
        "morgan_fingerprint": {"radius": 2, "bits": 2048, "similarity": "Tanimoto"},
        "query_count": len(queries), "stored_rows": len(rankings),
        "query_counts": query_counts, "model": None,
        "assay_transfer_status": "not_computed",
        "inputs": {
            "query_sha256": sha256_file(query_path), "train_sha256": sha256_file(train_path),
            "voter_membership_sha256": sha256_file(_gold_root(task) / "voter_membership.parquet"),
            "evidence_manifest_sha256": sha256_file(evidence_manifest),
        },
    }
    manifest = _write_complete(
        connection, database, task=task, subset=subset, level="L1",
        identity=identity, target=target,
    )
    connection.close()
    return manifest


def _write_index(task: str, root: Path, evidence_manifest: Path) -> dict[str, Any]:
    evidence = json.loads(evidence_manifest.read_text(encoding="utf-8"))
    splits = {}
    for subset in SUBSETS:
        path = root / "scaffold" / subset / "L1/VERSION.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        splits[subset] = {"levels": {"L1": {
            "manifest": str(path.relative_to(root)), "manifest_sha256": sha256_file(path),
            "content_id": manifest["content_id"],
        }}}
    index = {
        "schema_version": "ranked_uid_task_release_index.v1",
        "selection_contract": SCHEMA_VERSION, "profile": PROFILE, "task_id": task,
        "status": "complete", "gold_release": "v1", "pool": "all",
        "parent_capacity": CAPACITY, "levels_independent": True,
        "ranking_modes": ["morgan"], "assay_transfer_status": "not_computed",
        "morgan_fingerprint": {"radius": 2, "bits": 2048, "similarity": "Tanimoto"},
        "neighbor_identity_policy_by_level": {"L1": "scaffold_disjoint"},
        "evidence": {
            "manifest": os.path.relpath(evidence_manifest, root),
            "manifest_sha256": sha256_file(evidence_manifest),
            "content_id": evidence["content_id"], "record_count": evidence["record_count"],
        },
        "splits": splits,
    }
    (root / "RELEASE_INDEX.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return index


def _validate(task: str, root: Path, cards: dict[str, dict[str, Any]]) -> dict[str, Any]:
    index_path = root / "RELEASE_INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    evidence_manifest = root / str(index["evidence"]["manifest"])
    evidence = json.loads(evidence_manifest.read_text(encoding="utf-8"))
    records_path = evidence_manifest.with_name(str(evidence["records"]))
    if (
        index.get("schema_version") != "ranked_uid_task_release_index.v1"
        or index.get("profile") != PROFILE or index.get("task_id") != task
        or index["ranking_modes"] != ["morgan"]
        or index["assay_transfer_status"] != "not_computed"
        or sha256_file(evidence_manifest) != index["evidence"]["manifest_sha256"]
        or sha256_file(records_path) != evidence["records_sha256"]
    ):
        raise ValueError(f"Invalid Gold-v1 add-on projection: {task}")
    projected_uids = set(pq.read_table(records_path, columns=["source_row_uid"])["source_row_uid"].to_pylist())
    membership = pq.read_table(
        _gold_root(task) / "voter_membership.parquet",
        columns=["benchmark_row_id", "source_row_uid", "physical_member_index", "split"],
    ).to_pylist()
    expected_members: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in membership:
        if row["split"] == "train":
            expected_members[str(row["benchmark_row_id"])].append(
                (int(row["physical_member_index"]), str(row["source_row_uid"]))
            )
    expected = {
        context_id: [uid for _, uid in sorted(values)]
        for context_id, values in expected_members.items()
    }
    if projected_uids != {uid for values in expected.values() for uid in values}:
        raise ValueError(f"Gold-v1 add-on projection does not equal train membership: {task}")
    train_scaffolds = {
        str(row["molecule_identity_key"]): str(row.get("bemis_murcko_scaffold") or "")
        for row in cards.values()
    }
    report = {}
    for subset in SUBSETS:
        target = root / "scaffold" / subset / "L1"
        manifest_path = target / "VERSION.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = index["splits"][subset]["levels"]["L1"]
        database = target / str(manifest["database"])
        query_rows = {
            str(row["benchmark_row_id"]): row
            for row in read_jsonl(_gold_root(task) / f"{subset}.jsonl")
        }
        with sqlite3.connect(database) as connection:
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise ValueError(f"Gold-v1 add-on database failed integrity: {task}/{subset}")
            for query_id, query_parent in connection.execute(
                "SELECT benchmark_row_id,query_parent_id FROM queries"
            ):
                rows = connection.execute(
                    "SELECT parent_id,morgan_rank,assay_transfer_score,assay_rank,assay_context_id "
                    "FROM rankings WHERE benchmark_row_id=? ORDER BY morgan_rank", (query_id,),
                ).fetchall()
                if (
                    len(rows) != CAPACITY or len({row[0] for row in rows}) != CAPACITY
                    or query_parent in {row[0] for row in rows}
                    or [row[1] for row in rows] != list(range(1, CAPACITY + 1))
                    or any(value is not None for row in rows for value in row[2:])
                ):
                    raise ValueError(f"Invalid Gold-v1 add-on ranks: {task}/{subset}/{query_id}")
                query_scaffold = str(query_rows[str(query_id)].get("bemis_murcko_scaffold") or "")
                if query_scaffold and any(train_scaffolds[row[0]] == query_scaffold for row in rows):
                    raise ValueError(f"Gold-v1 add-on scaffold leakage: {task}/{subset}/{query_id}")
            for context_id, in connection.execute("SELECT context_id FROM contexts"):
                actual = [
                    row[0] for row in connection.execute(
                        "SELECT source_row_uid FROM context_records WHERE context_id=? "
                        "ORDER BY within_context_rank", (context_id,),
                    )
                ]
                if actual != expected[context_id]:
                    raise ValueError(f"Gold-v1 add-on context membership changed: {task}/{context_id}")
        if (
            sha256_file(manifest_path) != entry["manifest_sha256"]
            or sha256_file(database) != manifest["database_sha256"]
            or manifest["content_id"] != entry["content_id"]
        ):
            raise ValueError(f"Gold-v1 add-on hashes changed: {task}/{subset}")
        report[subset] = {"queries": manifest["query_count"], "stored_rows": manifest["stored_rows"]}
    receipt = {
        "schema_version": "ranked_uid_release_validation.v1", "status": "complete",
        "task_id": task, "ranking_modes": ["morgan"],
        "assay_transfer_status": "not_computed", "splits": report,
    }
    receipt["content_id"] = _digest(receipt)
    (root / "VALIDATION.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def build_task(task: str, output_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    destination = output_root / task
    if destination.exists():
        raise FileExistsError(f"Refusing to replace Gold-v1 add-on cache: {destination}")
    train_path, train_rows, cards, parent_order = _load_train(task)
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{task}.", dir=output_root) as temporary:
        root = Path(temporary)
        evidence_manifest = _write_projection(task, root, cards)
        for subset in SUBSETS:
            _build_split(
                task, subset, train_path, train_rows, cards, parent_order,
                evidence_manifest, root,
            )
        index = _write_index(task, root, evidence_manifest)
        validation = _validate(task, root, cards)
        os.replace(root, destination)
    return {
        "task_id": task, "status": index["status"],
        "ranking_modes": index["ranking_modes"],
        "validation_content_id": validation["content_id"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=("all", *TASKS), default=["all"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    tasks = tuple(TASKS) if args.tasks == ["all"] else tuple(dict.fromkeys(args.tasks))
    print(json.dumps({task: build_task(task, args.output_root) for task in tasks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
