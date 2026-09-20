"""Build generic Morgan-100 L1 caches for the six TDC-v1 tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import DataStructs

from data.processing.gold_labels.conditioned_benchmark import tdc_task_root
from predict.retrieval.policies import standardize_smiles_and_fp
from predict.utils.json import read_jsonl, sha256_file

from .build_ranked_retrieval import _digest
from .build_ranked_uid_retrieval import _schema, _write_complete
from .ranked_uid_retrieval import CAPACITY, SCHEMA_VERSION
from .runtime import cache_profile_root


PROFILE = "ranked_level_retrieval_tdc_v1"
TASKS = (
    "bbb_martins",
    "bioavailability_ma",
    "skin_reaction",
    "ames",
    "dili",
    "carcinogens",
)
SUBSETS = ("valid", "test")
DEFAULT_OUTPUT = cache_profile_root(PROFILE)


def _parent(row: dict[str, Any]) -> tuple[str, str]:
    identity = row["molecule_identity"]
    return str(row["molecule_identity_key"]), str(identity["parent_smiles"])


def _write_projection(task: str, train_path: Path, train: list[dict[str, Any]], root: Path) -> Path:
    projection = root / "evidence"
    projection.mkdir(parents=True)
    records = []
    seen: set[str] = set()
    for row in train:
        parent_id, parent_smiles = _parent(row)
        for source_record_id in row["source_record_ids"]:
            uid = str(source_record_id)
            if uid in seen:
                raise ValueError(f"duplicate TDC source record ID for {task}: {uid}")
            seen.add(uid)
            payload = {
                "task_id": task,
                "progressive_level": "L1",
                "label_source": "tdc_v1",
                "benchmark_row_id": str(row["benchmark_row_id"]),
                "canonical_smiles": parent_smiles,
                "source_canonical_smiles": str(row["drug"]),
                "condition_group": str(row.get("condition_group") or ""),
                "Y": int(row["Y"]),
                "label_counts": dict(row.get("label_counts") or {str(row["Y"]): 1}),
                "source_record_id": uid,
                "source_fields": {
                    "source_record_id": uid,
                    "label": int(row["Y"]),
                },
                "display_measurement_text": f"Frozen TDC label={int(row['Y'])}",
                "display_unit_text": "",
            }
            records.append({
                "source_row_uid": uid,
                "external_record_id": uid,
                "parent_id": parent_id,
                "parent_smiles": parent_smiles,
                "level": "L1",
                "payload": json.dumps(payload, sort_keys=True, separators=(",", ":")),
            })
    records.sort(key=lambda row: row["source_row_uid"])
    records_path = projection / "records.parquet"
    pq.write_table(pa.Table.from_pylist(records), records_path, compression="zstd")
    identity = {
        "schema_version": "ranked_evidence_projection.v1",
        "status": "complete",
        "task_id": task,
        "record_count": len(records),
        "source_identity": "tdc_source_record_id",
        "input_sha256": sha256_file(train_path),
    }
    identity["content_id"] = _digest(identity)
    manifest = {
        **identity,
        "records": records_path.name,
        "records_sha256": sha256_file(records_path),
        "source_path": str(train_path),
    }
    manifest_path = projection / "VERSION.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _build_split(
    task: str,
    subset: str,
    train_path: Path,
    train: list[dict[str, Any]],
    evidence_manifest: Path,
    root: Path,
) -> dict[str, Any]:
    query_path = tdc_task_root(task) / f"{subset}_molecule_condition_labels.jsonl"
    query_rows = read_jsonl(query_path)
    parents: dict[str, dict[str, Any]] = {}
    fingerprints = []
    for row in train:
        parent_id, parent_smiles = _parent(row)
        if parent_id in parents:
            raise ValueError(f"TDC train contains duplicate parent: {task}/{parent_id}")
        fingerprint = standardize_smiles_and_fp(parent_smiles)[2]
        if fingerprint is None:
            raise ValueError(f"TDC train parent is invalid: {task}/{parent_id}")
        parents[parent_id] = row
        fingerprints.append(fingerprint)
    parent_ids = list(parents)

    target = root / "scaffold" / subset / "L1"
    target.mkdir(parents=True)
    database = target / "rankings.sqlite3"
    connection = sqlite3.connect(database)
    _schema(connection, "L1")
    query_ledger = []
    rankings = []
    selected_contexts: set[str] = set()
    query_counts = {}
    for query in query_rows:
        query_id = str(query["benchmark_row_id"])
        query_parent, query_smiles = _parent(query)
        query_fp = standardize_smiles_and_fp(query_smiles)[2]
        if query_fp is None:
            raise ValueError(f"TDC query parent is invalid: {task}/{query_id}")
        query_ledger.append((query_id, str(query["drug"]), query_parent, query_smiles))
        similarities = DataStructs.BulkTanimotoSimilarity(query_fp, fingerprints)
        candidates = sorted(
            (
                (float(similarity), parent_id, parents[parent_id])
                for similarity, parent_id in zip(similarities, parent_ids)
                if parent_id != query_parent
                and (
                    not query.get("bemis_murcko_scaffold")
                    or str(parents[parent_id].get("bemis_murcko_scaffold") or "")
                    != str(query["bemis_murcko_scaffold"])
                )
            ),
            key=lambda item: (-item[0], item[1]),
        )[:CAPACITY]
        if len(candidates) != CAPACITY:
            raise ValueError(f"TDC query lacks {CAPACITY} disjoint parents: {task}/{query_id}")
        member_count = 0
        for rank, (similarity, parent_id, row) in enumerate(candidates, 1):
            context_id = str(row["benchmark_row_id"])
            members = [str(value) for value in row["source_record_ids"]]
            selected_contexts.add(context_id)
            member_count += len(members)
            rankings.append((
                query_id, parent_id, parent_id, _parent(row)[1], similarity,
                rank, 1, rank, None, None, context_id, None,
                len(members), None, None,
            ))
        query_counts[query_id] = {
            "candidate_parents": CAPACITY,
            "morgan_candidate_records": member_count,
            "assay_candidate_records": 0,
        }
    connection.executemany("INSERT INTO queries VALUES (?,?,?,?)", query_ledger)
    connection.executemany(
        "INSERT INTO rankings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rankings
    )
    by_context = {str(row["benchmark_row_id"]): row for row in train}
    connection.executemany(
        "INSERT INTO contexts VALUES (?,?,?,?)",
        [
            (
                context_id,
                str(by_context[context_id]["molecule_identity_key"]),
                str(by_context[context_id].get("condition_group") or ""),
                int(by_context[context_id]["Y"]),
            )
            for context_id in sorted(selected_contexts)
        ],
    )
    connection.executemany(
        "INSERT INTO context_records VALUES (?,?,?)",
        [
            (context_id, uid, rank)
            for context_id in sorted(selected_contexts)
            for rank, uid in enumerate(
                map(str, by_context[context_id]["source_record_ids"]), 1
            )
        ],
    )
    identity = {
        "task_id": task,
        "subset": subset,
        "level": "L1",
        "pool": "fixed",
        "capacity": CAPACITY,
        "parent_capacity": CAPACITY,
        "label_release": {"benchmark": "tdc", "version": "v1"},
        "neighbor_identity_policy": "scaffold_disjoint",
        "shared_candidate_universe": True,
        "morgan_fingerprint": {"radius": 2, "bits": 2048, "similarity": "Tanimoto"},
        "query_count": len(query_rows),
        "stored_rows": len(rankings),
        "query_counts": query_counts,
        "model": None,
        "assay_transfer_status": "not_computed",
        "inputs": {
            "query_sha256": sha256_file(query_path),
            "train_sha256": sha256_file(train_path),
            "evidence_manifest_sha256": sha256_file(evidence_manifest),
        },
    }
    manifest = _write_complete(
        connection,
        database,
        task=task,
        subset=subset,
        level="L1",
        identity=identity,
        target=target,
    )
    connection.close()
    return manifest


def _write_index(task: str, root: Path, evidence_manifest: Path) -> dict[str, Any]:
    evidence = json.loads(evidence_manifest.read_text(encoding="utf-8"))
    splits = {}
    for subset in SUBSETS:
        path = root / "scaffold" / subset / "L1" / "VERSION.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        splits[subset] = {"levels": {"L1": {
            "manifest": str(path.relative_to(root)),
            "manifest_sha256": sha256_file(path),
            "content_id": manifest["content_id"],
        }}}
    index = {
        "schema_version": "ranked_uid_task_release_index.v1",
        "selection_contract": SCHEMA_VERSION,
        "profile": PROFILE,
        "task_id": task,
        "status": "complete",
        "label_release": {"benchmark": "tdc", "version": "v1"},
        "pool": "all",
        "parent_capacity": CAPACITY,
        "levels_independent": True,
        "ranking_modes": ["morgan"],
        "assay_transfer_status": "not_computed",
        "morgan_fingerprint": {"radius": 2, "bits": 2048, "similarity": "Tanimoto"},
        "neighbor_identity_policy_by_level": {
            "L1": "scaffold_disjoint"
        },
        "evidence": {
            "manifest": os.path.relpath(evidence_manifest, root),
            "manifest_sha256": sha256_file(evidence_manifest),
            "content_id": evidence["content_id"],
            "record_count": evidence["record_count"],
        },
        "splits": splits,
    }
    (root / "RELEASE_INDEX.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return index


def _validate(task: str, root: Path, train: list[dict[str, Any]]) -> dict[str, Any]:
    index_path = root / "RELEASE_INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if (
        index.get("schema_version") != "ranked_uid_task_release_index.v1"
        or index.get("status") != "complete"
        or index.get("task_id") != task
        or index.get("ranking_modes") != ["morgan"]
        or index.get("assay_transfer_status") != "not_computed"
    ):
        raise ValueError(f"invalid TDC release index: {task}")
    evidence_manifest = root / str(index["evidence"]["manifest"])
    evidence = json.loads(evidence_manifest.read_text(encoding="utf-8"))
    evidence_records = evidence_manifest.with_name(str(evidence["records"]))
    if (
        sha256_file(evidence_manifest) != index["evidence"]["manifest_sha256"]
        or sha256_file(evidence_records) != evidence["records_sha256"]
        or evidence["content_id"] != index["evidence"]["content_id"]
    ):
        raise ValueError(f"invalid TDC evidence projection hashes: {task}")
    expected_members = {
        str(row["benchmark_row_id"]): list(map(str, row["source_record_ids"]))
        for row in train
    }
    train_scaffolds = {
        str(row["molecule_identity_key"]): str(row.get("bemis_murcko_scaffold") or "")
        for row in train
    }
    report = {}
    for subset in SUBSETS:
        target = root / "scaffold" / subset / "L1"
        manifest_path = target / "VERSION.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = index["splits"][subset]["levels"]["L1"]
        if (
            sha256_file(manifest_path) != entry["manifest_sha256"]
            or manifest["content_id"] != entry["content_id"]
        ):
            raise ValueError(f"invalid TDC level manifest hashes: {task}/{subset}")
        database = target / manifest["database"]
        query_rows = {
            str(row["benchmark_row_id"]): row
            for row in read_jsonl(
                tdc_task_root(task) / f"{subset}_molecule_condition_labels.jsonl"
            )
        }
        with sqlite3.connect(database) as connection:
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise ValueError(f"TDC ranking database failed integrity: {task}/{subset}")
            queries = connection.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
            for query_id, query_parent in connection.execute(
                "SELECT benchmark_row_id,query_parent_id FROM queries"
            ):
                rows = connection.execute(
                    "SELECT parent_id,morgan_rank,assay_transfer_score,assay_rank,"
                    "assay_context_id FROM rankings WHERE benchmark_row_id=? "
                    "ORDER BY morgan_rank", (query_id,),
                ).fetchall()
                if (
                    len(rows) != CAPACITY
                    or len({row[0] for row in rows}) != CAPACITY
                    or query_parent in {row[0] for row in rows}
                    or [row[1] for row in rows] != list(range(1, CAPACITY + 1))
                    or any(value is not None for row in rows for value in row[2:])
                ):
                    raise ValueError(f"invalid TDC ranking rows: {task}/{subset}/{query_id}")
                query_scaffold = str(
                    query_rows[str(query_id)].get("bemis_murcko_scaffold") or ""
                )
                if query_scaffold and any(
                    train_scaffolds[row[0]] == query_scaffold for row in rows
                ):
                    raise ValueError(
                        f"TDC scaffold leakage: {task}/{subset}/{query_id}"
                    )
            for context_id, in connection.execute("SELECT context_id FROM contexts"):
                members = [
                    row[0] for row in connection.execute(
                        "SELECT source_row_uid FROM context_records WHERE context_id=? "
                        "ORDER BY within_context_rank", (context_id,),
                    )
                ]
                if members != expected_members[context_id]:
                    raise ValueError(f"TDC context membership changed: {task}/{context_id}")
        if sha256_file(database) != manifest["database_sha256"]:
            raise ValueError(f"TDC ranking database hash changed: {task}/{subset}")
        report[subset] = {
            "queries": queries,
            "stored_rows": manifest["stored_rows"],
            "content_id": manifest["content_id"],
        }
    receipt = {
        "schema_version": "ranked_uid_release_validation.v1",
        "status": "complete",
        "task_id": task,
        "ranking_modes": ["morgan"],
        "assay_transfer_status": "not_computed",
        "splits": report,
    }
    receipt["content_id"] = _digest(receipt)
    (root / "VALIDATION.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def build_task(task: str, output_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    destination = output_root / task
    if destination.exists():
        raise FileExistsError(f"Refusing to replace TDC ranked cache: {destination}")
    train_path = tdc_task_root(task) / "train_molecule_condition_labels.jsonl"
    train = read_jsonl(train_path)
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{task}.", dir=output_root) as temporary:
        root = Path(temporary)
        evidence_manifest = _write_projection(task, train_path, train, root)
        for subset in SUBSETS:
            _build_split(task, subset, train_path, train, evidence_manifest, root)
        index = _write_index(task, root, evidence_manifest)
        validation = _validate(task, root, train)
        os.replace(root, destination)
    return {
        "task_id": task,
        "status": index["status"],
        "ranking_modes": index["ranking_modes"],
        "validation_content_id": validation["content_id"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=("all", *TASKS), default=["all"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    tasks = TASKS if args.tasks == ["all"] else tuple(dict.fromkeys(args.tasks))
    print(json.dumps({task: build_task(task, args.output_root) for task in tasks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
