"""Publish a compact receipt for the Gold-v1-keyed BBB and Oral V10 rebuild."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket

import pyarrow.parquet as pq

from semantic_buckets.artifacts import semantic_bucket_root
from tools.chembl_tool.common.json_utils import write_json_atomic


ROOT = Path(__file__).resolve().parents[5]
OUTPUT = ROOT / "data/artifacts/evidence_library_assets/gold_v1_protected_v10/publication_receipt.json"
TASKS = {
    "bbb_martins": "BBB_Martins",
    "bioavailability_ma": "Bioavailability_Ma",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": _sha(path)}


def publish(output: Path = OUTPUT) -> dict:
    tasks = {}
    for task, public in TASKS.items():
        library = ROOT / f"data/evidence_libraries/{task}/v10"
        stage3 = library / "03_pair_buckets/records.parquet"
        levels = library / "level_mapping/records.parquet"
        voter = ROOT / f"data/gold_labels/{public}/v1/scaffold/voter_membership.parquet"
        semantic = semantic_bucket_root(task, "v10") / "manifest.json"
        level_rows = pq.read_table(levels, columns=["source_row_uid", "level"]).to_pylist()
        voter_uids = set(pq.read_table(voter, columns=["source_row_uid"])["source_row_uid"].to_pylist())
        l1_uids = {row["source_row_uid"] for row in level_rows if int(row["level"]) == 1}
        if voter_uids != l1_uids:
            raise ValueError(f"{task} L1 does not exactly equal frozen voter membership")
        stage3_rows = pq.read_metadata(stage3).num_rows
        stage3_uids = set(
            pq.read_table(stage3, columns=["source_row_uid"])["source_row_uid"].to_pylist()
        )
        if stage3_uids != {row["source_row_uid"] for row in level_rows}:
            raise ValueError(f"{task} Stage-3 and level-map UID universes differ")
        tasks[task] = {
            "stage3_rows": stage3_rows,
            "l1_physical_voters": len(l1_uids),
            "stage3_records": _file(stage3),
            "stage3_manifest": _file(library / "03_pair_buckets/manifest.json"),
            "uid_level_map": _file(levels),
            "uid_level_map_manifest": _file(levels.with_name("manifest.json")),
            "voter_contract": _file(ROOT / f"data/gold_labels/{public}/v1/scaffold/voter_contract_manifest.json"),
            "semantic_release": _file(semantic),
        }
    gold = {}
    for public in ("BBB_Martins", "Bioavailability_Ma", "Skin_Reaction"):
        root = ROOT / f"data/gold_labels/{public}/v1/scaffold"
        gold[public] = {
            "current": (ROOT / f"data/gold_labels/{public}/CURRENT").read_text().strip(),
            "splits": {
                split: _file(root / f"{split}_molecule_condition_labels.jsonl")
                for split in ("train", "valid", "test")
            },
        }
        if gold[public]["current"] != "v1":
            raise ValueError(f"{public} CURRENT is not v1")
    receipt = {
        "version": "gold_v1_protected_v10.publication.v1",
        "status": "complete",
        "hostname": socket.gethostname(),
        "repository_root": str(ROOT),
        "policy": {
            "gold_release": "v1",
            "protected_tasks": sorted(TASKS),
            "pair_bucket_contract": "gold context excluded from intrinsic pair identity",
            "skin_scope": "Gold-v1 splits frozen; Skin evidence rebuild deferred",
        },
        "tasks": tasks,
        "gold_releases": gold,
        "archived_predecessor": "data/legacy/artifacts/evidence_libraries/v10_before_gold_v1_protected_rebuild_20260916",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(publish(), indent=2))
