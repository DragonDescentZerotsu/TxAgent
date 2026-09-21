"""Publish Gold-v1 context-card membership for Ames, DILI, and Carcinogens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from predict.utils.json import read_jsonl, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[3]
REPLAY_COMMIT = "8972834103833c18a8d269a92f898bad07f7e5ab"
AMES_MEMBERSHIP_COMMIT = "26d44f121141f3738764c0b00dc09e185445341b"
TASKS = {
    "ames": {
        "gold": "Ames",
        "stage1": "ames",
        "stage1_release": "v10",
        "replay": None,
        "contract": "gold_v1_voter_contract.v1",
        "corrections": None,
    },
    "dili": {
        "gold": "DILI",
        "stage1": "dili",
        "stage1_release": "v10",
        "replay": "data/starling_data/dili/gold_v4/source_only_benchmark/DILI/scaffold",
        "contract": "gold_v1_voter_contract.v2",
        "corrections": (
            "data/gold_labels/DILI/v1/reviews/voter_membership_v2/"
            "reviewed_exclusions.json"
        ),
    },
    "carcinogens": {
        "gold": "Carcinogens",
        "stage1": "carcinogens",
        "stage1_release": "v10_main_universe_v1",
        "replay": "data/starling_data/carcinogens/gold_v4/source_only_benchmark/Carcinogens/scaffold",
        "contract": "gold_v1_voter_contract.v1",
        "corrections": None,
    },
}
SPLITS = ("train", "valid", "test")


MEMBERSHIP_SCHEMA = pa.schema(
    [
        ("contract_version", pa.string()), ("task", pa.string()),
        ("source_row_uid", pa.string()), ("physical_source_record_id", pa.string()),
        ("source_id", pa.string()), ("source_sha256", pa.string()),
        ("canonical_smiles", pa.string()), ("vote_id", pa.string()),
        ("vote_label", pa.int64()), ("physical_member_index", pa.int64()),
        ("physical_member_count", pa.int64()), ("physical_vote_weight", pa.float64()),
        ("benchmark_row_id", pa.string()), ("molecule_identity_key", pa.string()),
        ("condition_group", pa.string()), ("split", pa.string()),
    ],
    metadata={b"schema_version": b"gold_v1_voter_protection.v1"},
)
VOTE_UNITS_SCHEMA = pa.schema(
    [
        ("contract_version", pa.string()), ("task", pa.string()),
        ("vote_id", pa.string()), ("vote_label", pa.int64()),
        ("benchmark_row_id", pa.string()), ("molecule_identity_key", pa.string()),
        ("condition_group", pa.string()), ("split", pa.string()),
        ("physical_member_count", pa.int64()),
        ("physical_source_row_uids", pa.list_(pa.string())),
        ("physical_source_record_ids", pa.list_(pa.string())),
    ],
)
INDEX_SCHEMA = pa.schema(
    [
        ("contract_version", pa.string()), ("task", pa.string()),
        ("benchmark_row_id", pa.string()), ("molecule_identity_key", pa.string()),
        ("condition_group", pa.string()), ("split", pa.string()),
        ("Y", pa.int64()), ("voter_mean", pa.float64()),
        ("vote_unit_count", pa.int64()), ("physical_member_count", pa.int64()),
        ("vote_units", pa.list_(pa.struct([
            ("vote_id", pa.string()), ("vote_label", pa.int64()),
            ("physical_source_row_uids", pa.list_(pa.string())),
            ("physical_source_record_ids", pa.list_(pa.string())),
        ]))),
    ],
)


def _card_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["benchmark_row_id"]), str(row["drug"]), int(row["Y"]),
        str(row["molecule_identity_key"]), str(row["condition_group"]),
    )


def _detail_paths(task: str, replay_root: Path | None) -> dict[str, Path]:
    spec = TASKS[task]
    local = REPO_ROOT / "data/gold_labels" / spec["gold"] / "v1/scaffold"
    paths = {split: local / f"{split}_molecule_condition_labels.jsonl" for split in SPLITS}
    if spec["replay"]:
        if replay_root is None:
            raise ValueError(f"--replay-root is required for {task}")
        paths["train"] = replay_root / str(spec["replay"]) / "train_molecule_condition_labels.jsonl"
    return paths


def _load_cards(task: str, replay_root: Path | None) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any]]:
    spec = TASKS[task]
    gold = REPO_ROOT / "data/gold_labels" / spec["gold"] / "v1/scaffold"
    detail_paths = _detail_paths(task, replay_root)
    cards: list[tuple[str, dict[str, Any]]] = []
    inputs: dict[str, Any] = {"gold_split_files": {}, "detail_files": {}}
    for split in SPLITS:
        compact_path = gold / f"{split}.jsonl"
        detail_path = detail_paths[split]
        compact = read_jsonl(compact_path)
        details = read_jsonl(detail_path)
        if list(map(_card_identity, compact)) != list(map(_card_identity, details)):
            raise ValueError(f"Gold card replay mismatch: {task}/{split}")
        inputs["gold_split_files"][split] = {
            "path": str(compact_path.relative_to(REPO_ROOT)), "sha256": sha256_file(compact_path)
        }
        inputs["detail_files"][split] = {
            "path": (
                f"git:{REPLAY_COMMIT}:{spec['replay']}/train_molecule_condition_labels.jsonl"
                if split == "train" and spec["replay"]
                else str(detail_path.relative_to(REPO_ROOT))
            ),
            "sha256": sha256_file(detail_path),
        }
        cards.extend((split, row) for row in details)
    return cards, inputs


def _apply_reviewed_corrections(
    task: str,
    cards: list[tuple[str, dict[str, Any]]],
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any] | None]:
    """Apply hash-pinned, reviewed voter exclusions without changing Gold labels."""
    relative = TASKS[task]["corrections"]
    if relative is None:
        return cards, None
    path = REPO_ROOT / str(relative)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "reviewed" or payload.get("task") != task:
        raise ValueError(f"Invalid reviewed voter corrections: {relative}")
    for item in payload.get("inputs", []):
        input_path = REPO_ROOT / str(item["path"])
        if sha256_file(input_path) != str(item["sha256"]):
            raise ValueError(f"Stale voter-correction input: {item['path']}")

    decisions = payload.get("decisions", [])
    by_uid = {str(item["source_row_uid"]): item for item in decisions}
    if len(by_uid) != len(decisions) or any(
        item.get("action") != "exclude" for item in decisions
    ):
        raise ValueError("Voter corrections must be unique reviewed exclusions")

    applied: list[dict[str, Any]] = []
    corrected: list[tuple[str, dict[str, Any]]] = []
    for split, original in cards:
        card = dict(original)
        source_votes = {
            str(vote["source_record_id"]): vote for vote in card["source_votes"]
        }
        ids = list(map(str, card["source_record_ids"]))
        excluded = [source_id for source_id in ids if source_id in by_uid]
        for source_id in excluded:
            decision = by_uid[source_id]
            vote = source_votes[source_id]
            expected = {
                "split": split,
                "benchmark_row_id": str(card["benchmark_row_id"]),
                "molecule_identity_key": str(card["molecule_identity_key"]),
                "condition_group": str(card["condition_group"]),
                "vote_label": int(vote["Y"]),
                "source_payload_sha256": str(vote["source_payload_sha256"]),
            }
            for field, actual in expected.items():
                if decision.get(f"expected_{field}") != actual:
                    raise ValueError(
                        f"Stale voter correction {source_id}: expected_{field}"
                    )

        if excluded:
            old_labels = [int(source_votes[source_id]["Y"]) for source_id in ids]
            ids = [source_id for source_id in ids if source_id not in by_uid]
            if not ids:
                raise ValueError(
                    f"Voter correction emptied Gold card: {card['benchmark_row_id']}"
                )
            card["source_record_ids"] = ids
            card["source_votes"] = [source_votes[source_id] for source_id in ids]
            new_labels = [int(source_votes[source_id]["Y"]) for source_id in ids]
            old_mean = sum(old_labels) / len(old_labels)
            new_mean = sum(new_labels) / len(new_labels)
            if int(old_mean >= 0.5) != int(card["Y"]) or int(
                new_mean >= 0.5
            ) != int(card["Y"]):
                raise ValueError(
                    f"Voter correction changes Gold label: {card['benchmark_row_id']}"
                )
            for source_id in excluded:
                decision = by_uid[source_id]
                observed_impact = {
                    "expected_gold_label_before": int(old_mean >= 0.5),
                    "expected_gold_label_after": int(new_mean >= 0.5),
                    "expected_voter_mean_before": old_mean,
                    "expected_voter_mean_after": new_mean,
                }
                if any(
                    decision.get(field) != value
                    for field, value in observed_impact.items()
                ):
                    raise ValueError(f"Stale voter-impact review: {source_id}")
            applied.extend(
                {
                    "source_row_uid": source_id,
                    "benchmark_row_id": str(card["benchmark_row_id"]),
                    "old_voter_mean": old_mean,
                    "new_voter_mean": new_mean,
                    "old_vote_count": len(old_labels),
                    "new_vote_count": len(new_labels),
                    "gold_label": int(card["Y"]),
                }
                for source_id in excluded
            )
        corrected.append((split, card))

    if {item["source_row_uid"] for item in applied} != set(by_uid):
        missing = sorted(set(by_uid) - {item["source_row_uid"] for item in applied})
        raise ValueError(f"Reviewed voter corrections were not found: {missing}")
    return corrected, {
        "path": str(relative),
        "sha256": sha256_file(path),
        "version": str(payload["version"]),
        "applied": applied,
    }


def _ames_uid_map(membership_root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    historical = ds.dataset(membership_root, format="parquet").to_table(
        columns=["source_record_id", "canonical_record_id"],
        filter=ds.field("level") == 1,
    ).to_pylist()
    canonical_to_uid = {
        str(row["canonical_record_id"]): str(row["source_row_uid"])
        for row in ds.dataset(
            REPO_ROOT / "data/gold_labels/Ames/level_mappings/v1/level_mapping",
            format="parquet",
        ).to_table(columns=["source_row_uid", "canonical_record_id"], filter=ds.field("level") == 1).to_pylist()
    }
    mapping = {
        str(row["source_record_id"]): canonical_to_uid[str(row["canonical_record_id"])]
        for row in historical
    }
    if len(mapping) != 3333 or len(set(mapping.values())) != len(mapping):
        raise ValueError("Pinned Ames L1 membership is not one-to-one")
    parts = sorted(membership_root.glob("*.parquet"))
    return mapping, {
        "upstream_commit": AMES_MEMBERSHIP_COMMIT,
        "source_membership_parts": [
            {"name": path.name, "sha256": sha256_file(path)} for path in parts
        ],
        "active_level_mapping_manifest_sha256": sha256_file(
            REPO_ROOT / "data/gold_labels/Ames/level_mappings/v1/manifest.json"
        ),
    }


def _stage1_rows(
    task: str, source_ids: set[str], ames_membership_root: Path | None,
    votes: dict[str, dict[str, Any]], replay_root: Path | None,
) -> tuple[dict[str, dict[str, str]], Path, dict[str, Any] | None, list[dict[str, Any]]]:
    spec = TASKS[task]
    path = (
        REPO_ROOT / "data/evidence_libraries" / spec["stage1"]
        / spec["stage1_release"] / "01_cleaned/records.parquet"
    )
    columns = [
        "source_id", "source_name", "source_row_number", "source_record_id",
        "source_row_uid", "canonical_smiles",
    ]
    ames_uids = None
    ames_provenance = None
    found: dict[str, dict[str, str]] = {}
    if task == "ames":
        if ames_membership_root is None:
            raise ValueError("--ames-membership-root is required for Ames")
        ames_uids, ames_provenance = _ames_uid_map(ames_membership_root)
        if not source_ids <= set(ames_uids):
            raise ValueError("Pinned Ames membership does not cover active Gold voters")
        uid_to_source_id = {
            uid: source_id
            for source_id, uid in ames_uids.items()
            if source_id in source_ids
        }
    for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000, columns=columns):
        for row in batch.to_pylist():
            if task == "ames":
                source_record_id = uid_to_source_id.get(str(row["source_row_uid"]))
                if source_record_id is None:
                    continue
            else:
                source_record_id = str(row["source_row_uid"])
            if source_record_id not in source_ids:
                continue
            if source_record_id in found:
                raise ValueError(f"Stage-1 identity is not unique: {task}/{source_record_id}")
            found[source_record_id] = {
                "source_row_uid": str(row["source_row_uid"]),
                "source_id": str(row["source_id"]),
                "canonical_smiles": str(row["canonical_smiles"]),
            }
    missing = source_ids - set(found)
    if missing:
        if replay_root is None:
            raise ValueError(f"Stage-1 is missing {len(missing)} Gold voters for {task}")
        by_path: dict[str, set[str]] = {}
        for source_record_id in missing:
            raw_path = str(votes[source_record_id].get("raw_source_path") or "")
            if not raw_path:
                raise ValueError(f"Missing Stage-1 voter lacks pinned raw source: {source_record_id}")
            by_path.setdefault(raw_path, set()).add(source_record_id)
        raw_inputs = []
        for relative, expected in by_path.items():
            raw_path = replay_root / relative
            rows = pq.read_table(raw_path, filters=[("source_row_uid", "in", sorted(expected))]).to_pylist()
            if {str(row["source_row_uid"]) for row in rows} != expected:
                raise ValueError(f"Pinned raw source does not contain every omitted voter: {relative}")
            for row in rows:
                source_record_id = str(row["source_row_uid"])
                raw_smiles = str(row.get("SMILES") or row.get("smiles") or "")
                vote_smiles = str(votes[source_record_id]["drug"])
                if raw_smiles != vote_smiles:
                    raise ValueError(f"Pinned raw structure changed: {source_record_id}")
                found[source_record_id] = {
                    "source_row_uid": source_record_id,
                    "source_id": Path(relative).stem,
                    "canonical_smiles": raw_smiles,
                }
            raw_inputs.append({
                "path": f"git:{REPLAY_COMMIT}:{relative}", "sha256": sha256_file(raw_path),
                "source_row_uids": sorted(expected),
            })
    else:
        raw_inputs = []
    return found, path, ames_provenance, raw_inputs


def publish_task(
    task: str, output_root: Path, replay_root: Path | None,
    ames_membership_root: Path | None,
) -> dict[str, Any]:
    spec = TASKS[task]
    cards, inputs = _load_cards(task, replay_root)
    cards, correction_input = _apply_reviewed_corrections(task, cards)
    contract = str(spec["contract"])
    source_ids = {
        str(source_id)
        for _, card in cards
        for source_id in card["source_record_ids"]
    }
    votes = {
        str(vote["source_record_id"]): vote
        for _, card in cards for vote in card["source_votes"]
    }
    stage1, stage1_path, ames_provenance, raw_fallbacks = _stage1_rows(
        task, source_ids, ames_membership_root, votes, replay_root
    )
    membership = []
    vote_units = []
    index = []
    seen_vote_ids: set[str] = set()
    for split, card in cards:
        source_votes = {str(vote["source_record_id"]): vote for vote in card["source_votes"]}
        ids = list(map(str, card["source_record_ids"]))
        if set(ids) != set(source_votes) or len(ids) != len(set(ids)):
            raise ValueError(f"Gold source-vote mismatch: {task}/{card['benchmark_row_id']}")
        units = []
        labels = []
        for source_record_id in ids:
            vote = source_votes[source_record_id]
            vote_id = source_record_id
            if vote_id in seen_vote_ids:
                raise ValueError(f"Gold vote reused across cards: {task}/{vote_id}")
            seen_vote_ids.add(vote_id)
            vote_label = int(vote["Y"])
            labels.append(vote_label)
            physical = stage1[source_record_id]
            unit = {
                "vote_id": vote_id, "vote_label": vote_label,
                "physical_source_row_uids": [physical["source_row_uid"]],
                "physical_source_record_ids": [source_record_id],
            }
            units.append(unit)
            vote_units.append({
                "contract_version": contract, "task": task, **unit,
                "benchmark_row_id": str(card["benchmark_row_id"]),
                "molecule_identity_key": str(card["molecule_identity_key"]),
                "condition_group": str(card["condition_group"]), "split": split,
                "physical_member_count": 1,
            })
            membership.append({
                "contract_version": contract, "task": task,
                "source_row_uid": physical["source_row_uid"],
                "physical_source_record_id": source_record_id,
                "source_id": physical["source_id"],
                "source_sha256": str(vote["source_payload_sha256"]),
                "canonical_smiles": physical["canonical_smiles"],
                "vote_id": vote_id, "vote_label": vote_label,
                "physical_member_index": 1, "physical_member_count": 1,
                "physical_vote_weight": 1.0,
                "benchmark_row_id": str(card["benchmark_row_id"]),
                "molecule_identity_key": str(card["molecule_identity_key"]),
                "condition_group": str(card["condition_group"]), "split": split,
            })
        voter_mean = sum(labels) / len(labels)
        if int(voter_mean >= 0.5) != int(card["Y"]):
            raise ValueError(f"Gold label does not match voters: {task}/{card['benchmark_row_id']}")
        index.append({
            "contract_version": contract, "task": task,
            "benchmark_row_id": str(card["benchmark_row_id"]),
            "molecule_identity_key": str(card["molecule_identity_key"]),
            "condition_group": str(card["condition_group"]), "split": split,
            "Y": int(card["Y"]), "voter_mean": voter_mean,
            "vote_unit_count": len(units), "physical_member_count": len(units),
            "vote_units": units,
        })
    target = output_root / spec["gold"] / "v1/scaffold"
    target.mkdir(parents=True, exist_ok=True)
    outputs = {}
    contract_metadata = {b"schema_version": contract.encode()}
    for name, rows, schema in (
        ("voter_membership", membership, MEMBERSHIP_SCHEMA),
        ("vote_units", vote_units, VOTE_UNITS_SCHEMA.with_metadata(contract_metadata)),
        ("gold_label_record_index", index, INDEX_SCHEMA.with_metadata(contract_metadata)),
    ):
        path = target / f"{name}.parquet"
        if path.exists():
            raise FileExistsError(f"Refusing to replace Gold contract: {path}")
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")
        outputs[name] = {
            "path": f"data/gold_labels/{spec['gold']}/v1/scaffold/{path.name}",
            "sha256": sha256_file(path),
        }
    inputs["stage1"] = {"path": str(stage1_path.relative_to(REPO_ROOT)), "sha256": sha256_file(stage1_path)}
    inputs["replay_commit"] = REPLAY_COMMIT if spec["replay"] else None
    inputs["stage1_omission_raw_fallbacks"] = raw_fallbacks
    if correction_input is not None:
        inputs["reviewed_voter_corrections"] = correction_input
    if ames_provenance is not None:
        inputs["ames_uid_membership"] = ames_provenance
    manifest = {
        "version": contract, "status": "complete", "task": task,
        "policy": {
            "gold_release": "v1",
            "l1_visibility": "all physical members of published-card vote units",
            "aggregation": "each vote_id contributes exactly once",
            "physical_member_weight": "1 / physical_member_count",
            "structure_identity": (
                "pinned V10 Stage 1 source_row_uid, with exact pinned raw-source fallback "
                "for recorded Stage-1 omissions"
            ),
        },
        "counts": {
            "gold_label_cards": len(index), "vote_units": len(vote_units),
            "physical_members": len(membership),
        },
        "inputs": inputs, "outputs": outputs,
    }
    manifest_path = target / "voter_contract_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to replace Gold contract: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"task": task, "status": "complete", **manifest["counts"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=("all", *TASKS), default=["all"])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path)
    parser.add_argument("--ames-membership-root", type=Path)
    args = parser.parse_args()
    tasks = tuple(TASKS) if args.tasks == ["all"] else tuple(dict.fromkeys(args.tasks))
    print(json.dumps({
        task: publish_task(
            task, args.output_root, args.replay_root, args.ames_membership_root
        ) for task in tasks
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
