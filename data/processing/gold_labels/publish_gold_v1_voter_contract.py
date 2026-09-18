"""Publish immutable Gold-v1 vote units and their physical L1 members."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v7.tasks.bbb_martins.experimental_meaningful_cns_access_benchmark_v4 import (
    label_record as label_bbb_record,
)
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "gold_v1_voter_contract.v1"
PROTECTION_VERSION = "gold_v1_voter_protection.v1"
TASKS = {
    "BBB_Martins": "bbb_martins",
    "Bioavailability_Ma": "bioavailability_ma",
}
EXPECTED = {
    "bbb_martins": (7_634, 7_634),
    "bioavailability_ma": (18_853, 19_479),
}


def publish(
    *,
    gold_root: Path,
    lineage_root: Path,
    frozen_stage1_root: Path,
    bbb_raw_records: Path,
) -> dict[str, Any]:
    outputs = {}
    for public_task, task in TASKS.items():
        scaffold = gold_root / public_task / "v1/scaffold"
        cards = _cards(scaffold)
        lineage_path = lineage_root / f"{task}.voter_lineage.jsonl"
        lineage = _jsonl(lineage_path)
        labels = (
            _bbb_labels(lineage, bbb_raw_records)
            if task == "bbb_martins"
            else _oral_labels(gold_root)
        )
        frozen = _frozen_rows(frozen_stage1_root / task / "v10")
        membership = _membership(task, lineage, labels, frozen)
        vote_units, card_index = _vote_units_and_card_index(cards, membership)
        expected_votes, expected_members = EXPECTED[task]
        if (len(vote_units), len(membership)) != (expected_votes, expected_members):
            raise ValueError(
                f"{task} contract count mismatch: "
                f"{len(vote_units)} votes, {len(membership)} physical members"
            )
        _validate_cards(cards, vote_units)

        membership_path = scaffold / "voter_membership.parquet"
        vote_units_path = scaffold / "vote_units.parquet"
        card_index_path = scaffold / "gold_label_record_index.parquet"
        _write_parquet(membership_path, membership, PROTECTION_VERSION)
        _write_parquet(vote_units_path, vote_units, VERSION)
        _write_parquet(card_index_path, card_index, VERSION)
        manifest = {
            "version": VERSION,
            "status": "complete",
            "task": task,
            "policy": {
                "gold_release": "v1",
                "l1_visibility": "all physical members of published-card vote units",
                "aggregation": "each vote_id contributes exactly once",
                "physical_member_weight": "1 / physical_member_count",
                "structure_identity": "frozen from the Gold-v1-compatible V10 Stage 1",
            },
            "counts": {
                "gold_label_cards": len(cards),
                "vote_units": len(vote_units),
                "physical_members": len(membership),
            },
            "inputs": {
                "voter_lineage": _file(lineage_path),
                "frozen_stage1_root": str(frozen_stage1_root / task / "v10"),
                "gold_split_files": {
                    split: _file(scaffold / f"{split}_molecule_condition_labels.jsonl")
                    for split in ("train", "valid", "test")
                },
            },
            "outputs": {
                "voter_membership": _file(membership_path),
                "vote_units": _file(vote_units_path),
                "gold_label_record_index": _file(card_index_path),
            },
        }
        manifest_path = scaffold / "voter_contract_manifest.json"
        write_json_atomic(manifest_path, manifest)
        outputs[task] = {**manifest["counts"], "manifest": _file(manifest_path)}
    return {"version": VERSION, "tasks": outputs}


def _membership(
    task: str,
    lineage: list[dict[str, Any]],
    labels: dict[str, int],
    frozen: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in lineage:
        grouped[str(row["v1_vote_id"])].append(row)
    output = []
    for vote_id, members in sorted(grouped.items()):
        label = labels[vote_id]
        ordered = sorted(members, key=lambda row: str(row["source_row_uid"]))
        for index, row in enumerate(ordered):
            uid = str(row["source_row_uid"])
            source = frozen.get(uid)
            if source is None:
                raise ValueError(f"Frozen Stage 1 lacks protected voter {uid}")
            output.append(
                {
                    "contract_version": PROTECTION_VERSION,
                    "task": task,
                    "source_row_uid": uid,
                    "physical_source_record_id": str(row["physical_source_record_id"]),
                    "source_id": str(source["source_id"]),
                    "source_sha256": str(source["source_sha256"]),
                    "canonical_smiles": str(source["canonical_smiles"]),
                    "vote_id": vote_id,
                    "vote_label": label,
                    "physical_member_index": index,
                    "physical_member_count": len(ordered),
                    "physical_vote_weight": 1.0 / len(ordered),
                    "benchmark_row_id": str(row["v1_benchmark_row_id"]),
                    "molecule_identity_key": str(row["v1_molecule_identity_key"]),
                    "condition_group": str(row["v1_condition_group"]),
                    "split": str(row["v1_split"]),
                }
            )
    return sorted(output, key=lambda row: row["source_row_uid"])


def _vote_units_and_card_index(
    cards: list[dict[str, Any]], membership: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_vote: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in membership:
        by_vote[row["vote_id"]].append(row)
    vote_units = []
    by_card: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vote_id, members in sorted(by_vote.items()):
        first = members[0]
        if any(
            row[field] != first[field]
            for row in members
            for field in (
                "vote_label", "benchmark_row_id", "molecule_identity_key",
                "condition_group", "split",
            )
        ):
            raise ValueError(f"Vote unit has inconsistent members: {vote_id}")
        unit = {
            "contract_version": VERSION,
            "task": first["task"],
            "vote_id": vote_id,
            "vote_label": first["vote_label"],
            "benchmark_row_id": first["benchmark_row_id"],
            "molecule_identity_key": first["molecule_identity_key"],
            "condition_group": first["condition_group"],
            "split": first["split"],
            "physical_member_count": len(members),
            "physical_source_row_uids": sorted(row["source_row_uid"] for row in members),
            "physical_source_record_ids": sorted(
                row["physical_source_record_id"] for row in members
            ),
        }
        vote_units.append(unit)
        by_card[first["benchmark_row_id"]].append(unit)
    card_index = []
    for card in sorted(cards, key=lambda row: str(row["benchmark_row_id"])):
        units = sorted(by_card[str(card["benchmark_row_id"])], key=lambda row: row["vote_id"])
        card_index.append(
            {
                "contract_version": VERSION,
                "task": units[0]["task"],
                "benchmark_row_id": str(card["benchmark_row_id"]),
                "molecule_identity_key": str(card["molecule_identity_key"]),
                "condition_group": str(card["condition_group"]),
                "split": str(card["split"]),
                "Y": int(card["Y"]),
                "voter_mean": sum(unit["vote_label"] for unit in units) / len(units),
                "vote_unit_count": len(units),
                "physical_member_count": sum(unit["physical_member_count"] for unit in units),
                "vote_units": [
                    {
                        "vote_id": unit["vote_id"],
                        "vote_label": unit["vote_label"],
                        "physical_source_row_uids": unit["physical_source_row_uids"],
                        "physical_source_record_ids": unit["physical_source_record_ids"],
                    }
                    for unit in units
                ],
            }
        )
    return vote_units, card_index


def _validate_cards(cards: list[dict[str, Any]], vote_units: list[dict[str, Any]]) -> None:
    by_card: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in vote_units:
        by_card[row["benchmark_row_id"]].append(row)
    for card in cards:
        units = by_card[str(card["benchmark_row_id"])]
        found = Counter(unit["vote_label"] for unit in units)
        expected = Counter({int(key): int(value) for key, value in card["label_counts"].items()})
        if found != expected or len(units) != int(card["source_record_count"]):
            raise ValueError(f"Frozen vote replay differs for {card['benchmark_row_id']}")


def _bbb_labels(
    lineage: list[dict[str, Any]], raw_path: Path
) -> dict[str, int]:
    raw = {
        str(row["source_row_uid"]): row
        for row in pq.read_table(raw_path).to_pylist()
    }
    labels = {}
    for row in lineage:
        uid = str(row["source_row_uid"])
        label, _ = label_bbb_record(
            raw[uid],
            source_index=int(row["physical_source_record_id"]),
            allow_conditioned_context=(
                row["v1_condition_group"] != "no_reported_external_condition"
            ),
        )
        if label not in (0, 1):
            raise ValueError(f"Cannot replay BBB Gold-v1 vote {row['v1_vote_id']}")
        labels[str(row["v1_vote_id"])] = label
    return labels


def _oral_labels(gold_root: Path) -> dict[str, int]:
    paths = (
        gold_root / "legacy/record_supported_v2_processed/Bioavailability_Ma/scaffold/voting_records.jsonl",
        gold_root / "legacy/processed_starling_context_conditioned_continuous_v1/Bioavailability_Ma/scaffold/voting_records.jsonl",
    )
    labels = {}
    for path in paths:
        for row in _jsonl(path):
            labels[str(row["source_record_id"])] = int(row["record_vote"])
    for row in _jsonl(gold_root / "Bioavailability_Ma/v1/scaffold/source_condition_review.jsonl"):
        if row["review_status"] == "accepted":
            labels[str(row["source_record_id"])] = int(row["Y"])
    return labels


def _frozen_rows(root: Path) -> dict[str, dict[str, Any]]:
    inventory = json.loads((root / "00_source/source_inventory.json").read_text())
    source_hashes = inventory["source_hashes"]
    output = {}
    for row in pq.read_table(
        root / "01_cleaned/records.parquet",
        columns=["source_row_uid", "source_id", "canonical_smiles"],
    ).to_pylist():
        row["source_sha256"] = source_hashes[str(row["source_id"])]
        output[str(row["source_row_uid"])] = row
    return output


def _cards(root: Path) -> list[dict[str, Any]]:
    rows = []
    for split in ("train", "valid", "test"):
        rows.extend(_jsonl(root / f"{split}_molecule_condition_labels.jsonl"))
    return rows


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_parquet(path: Path, rows: list[dict[str, Any]], version: str) -> None:
    table = pa.Table.from_pylist(rows).replace_schema_metadata(
        {b"schema_version": version.encode()}
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)


def _file(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": file_sha256(path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-root", type=Path, default=Path("data/gold_labels"))
    parser.add_argument(
        "--lineage-root", type=Path,
        default=Path("data/artifacts/gold_labels/conditioned_benchmark/v2_voter_lineage"),
    )
    parser.add_argument(
        "--frozen-stage1-root", type=Path,
        default=Path("data/legacy/artifacts/evidence_libraries/v10_before_level_mapping_compatibility_20260908"),
    )
    parser.add_argument(
        "--bbb-raw-records", type=Path,
        default=Path("data/raw/starling/bbb_martins/Direct_BBB/records.parquet"),
    )
    args = parser.parse_args(argv)
    print(json.dumps(publish(**vars(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
