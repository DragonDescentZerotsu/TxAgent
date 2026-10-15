"""Publish supported Gold-v1 memberships or audit all six frozen voter contracts."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from collections import Counter, defaultdict
import json
import importlib.util
import math
from pathlib import Path
import sys
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
                f"generated:{spec['replay']}/train_molecule_condition_labels.jsonl"
                if split == "train" and spec["replay"]
                else str(detail_path.relative_to(REPO_ROOT))
            ),
            "sha256": sha256_file(detail_path),
        }
        if split == "train" and spec["replay"]:
            inputs["detail_files"][split]["reconstructed_from"] = {
                "commit": REPLAY_COMMIT,
                "root": f"data/starling_data/{task}/gold_v4",
                "files": ["record_dispositions.parquet", "identity_ledger.parquet",
                          "tautomer_identities.jsonl", "gold_labels.jsonl"],
                "builder": f"tools/chembl_tool/tasks/{task}/build_reviewed_starling.py",
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


def verify_membership(gold: Path, library: Path) -> dict[str, Any]:
    """Audit card votes and physical members against the published library."""
    membership = pq.read_table(gold / "voter_membership.parquet").to_pylist()
    units = pq.read_table(gold / "vote_units.parquet").to_pylist()
    index = pq.read_table(gold / "gold_label_record_index.parquet").to_pylist()
    cards = {(str(row["benchmark_row_id"]), split): row
             for split in SPLITS for row in read_jsonl(gold / f"{split}.jsonl")}
    indexed = {(str(row["benchmark_row_id"]), str(row["split"])): row for row in index}
    if len(indexed) != len(index) or set(indexed) != set(cards):
        raise ValueError("Membership index does not cover the frozen Gold cards exactly")
    edges = defaultdict(list)
    seen = set()
    for row in membership:
        key = (str(row["benchmark_row_id"]), str(row["split"]), str(row["vote_id"]))
        edge = (*key, str(row["source_row_uid"]))
        if edge in seen:
            raise ValueError(f"Repeated physical vote member: {edge}")
        seen.add(edge)
        edges[key].append(row)
    by_card = defaultdict(list)
    for unit in units:
        key = (str(unit["benchmark_row_id"]), str(unit["split"]), str(unit["vote_id"]))
        members = edges.pop(key, [])
        if (not members or {str(row["source_row_uid"]) for row in members}
                != set(unit["physical_source_row_uids"])
                or len(members) != int(unit["physical_member_count"])):
            raise ValueError(f"Vote unit and physical membership differ: {key}")
        if any(int(row["vote_label"]) != int(unit["vote_label"])
               or row["molecule_identity_key"] != unit["molecule_identity_key"]
               or row["condition_group"] != unit["condition_group"]
               or int(row["physical_member_count"]) != len(members)
               or not math.isfinite(float(row["physical_vote_weight"]))
               or abs(float(row["physical_vote_weight"]) - 1 / len(members)) > 1e-12
               for row in members):
            raise ValueError(f"Invalid physical vote label or weight: {key}")
        by_card[key[:2]].append(unit)
    if edges or set(by_card) != set(cards):
        raise ValueError("Vote units and membership cards differ")
    for key, card in cards.items():
        votes = by_card[key]
        labels = [int(unit["vote_label"]) for unit in votes]
        mean = sum(labels) / len(labels)
        physical_count = sum(int(unit["physical_member_count"]) for unit in votes)
        row = indexed[key]
        indexed_votes = {(str(vote["vote_id"]), int(vote["vote_label"]),
                          tuple(vote["physical_source_row_uids"]))
                         for vote in row.get("vote_units", [])}
        actual_votes = {(str(vote["vote_id"]), int(vote["vote_label"]),
                         tuple(vote["physical_source_row_uids"])) for vote in votes}
        if (any(label not in (0, 1) for label in labels)
                or int(mean >= 0.5) != int(card["Y"]) or int(row["Y"]) != int(card["Y"])
                or row["vote_unit_count"] != len(votes)
                or row["physical_member_count"] != physical_count
                or ("molecule_identity_key" in row and row["molecule_identity_key"] != card["molecule_identity_key"])
                or ("condition_group" in row and row["condition_group"] != card["condition_group"])
                or ("vote_units" in row and indexed_votes != actual_votes)
                or not math.isfinite(row["voter_mean"])
                or abs(row["voter_mean"] - mean) > 1e-12):
            raise ValueError(f"Gold label or membership index differs: {key}")
        for unit in votes:
            if (unit["molecule_identity_key"] != card["molecule_identity_key"]
                    or unit["condition_group"] != card["condition_group"]):
                raise ValueError(f"Vote belongs to a different parent or condition: {key}")
    physical = Counter(pq.read_table(library / "03_pair_buckets/records.parquet",
                                    columns=["source_row_uid"])["source_row_uid"].to_pylist())
    selected = {str(row["source_row_uid"]) for row in membership}
    invalid = sorted(uid for uid in selected if physical[uid] != 1)
    if invalid:
        raise ValueError(f"Members absent or nonunique in canonical evidence: {invalid[:25]}")
    return {"status": "verified", "gold_label_cards": len(cards), "vote_units": len(units),
            "physical_members": len(membership), "distinct_physical_uids": len(selected),
            "evidence_release": library.name}


def replay_skin_votes(replay_root: Path):
    """Bind original measured votes to the permanent acquisition-row UIDs."""
    paths = {
        "classifier": ("tools/chembl_tool/tasks/skin_reaction/canonical_starling_source.py",
                       "8f1d82719f4831ba3eddfadaf3a1643012bc2e2434b3c48b281c55df9e1100cb"),
        "identity": ("tools/chembl_tool/common/molecule_identity.py",
                     "d9f5e9e005ee2e17ca33c7aadf298ef9d59c0831b36b1fa1f0227e0776617050"),
        "raw": ("data/starling_data/skin_reaction/direct_skin_reaction/extractions.parquet",
                "e7c4819c98af47eae4dc0563f40dc39f8b09eb8e03c9626daee603147125f9c1"),
    }
    for relative, expected in paths.values():
        if sha256_file(replay_root / relative) != expected:
            raise ValueError(f"Skin replay input differs from pinned Git source: {relative}")
    modules = {}
    for kind in ("classifier", "identity"):
        name = f"_gold_voter_replay_skin_{kind}"
        spec = importlib.util.spec_from_file_location(name, replay_root / paths[kind][0])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            del sys.modules[name]
        modules[kind] = module
    raw = pq.read_table(replay_root / paths["raw"][0])
    current = pq.read_table(REPO_ROOT / "data/raw/starling/skin_reaction/direct_skin_reaction/extractions.parquet")
    if not current.select(raw.column_names).equals(raw, check_metadata=False):
        raise ValueError("Skin acquisition rows differ from the pinned source; cannot bind UIDs by row")
    uids = current["source_row_uid"].to_pylist()
    groups = defaultdict(list)
    for position, row in enumerate(raw.to_pandas().to_dict("records")):
        decision = modules["classifier"].classify_direct_source_record(row)
        outcome = str(row.get("outcome_label") or "").strip().lower()
        if decision.partition != "direct" or outcome not in {"positive", "negative"}:
            continue
        identity = modules["identity"].normalize_molecule_identity(str(row.get("SMILES") or row.get("smiles") or "").strip())
        if identity.status == "ok" and identity.parent_smiles:
            groups[identity.parent_inchi_key or identity.parent_smiles].append(
                (uids[position], int(outcome == "positive")))
    inputs = {kind: {"git_path": relative, "sha256": expected}
              for kind, (relative, expected) in paths.items()}
    return groups, current, inputs


def audit_skin_replay(replay_root: Path, gold: Path, library: Path) -> dict[str, Any]:
    """Report conflicts between the frozen cards and the pinned source replay."""
    groups, _, inputs = replay_skin_votes(replay_root)
    differences, members = [], set()
    frozen_count = replay_count = cards_count = 0
    for split in SPLITS:
        for card in read_jsonl(gold / f"{split}_molecule_condition_labels.jsonl"):
            if card["condition_group"] != "no_reported_external_condition":
                continue
            votes = groups[card["molecule_identity_key"]]
            expected = +Counter({str(key): int(value) for key, value in card["label_counts"].items()})
            actual = Counter(str(label) for _, label in votes)
            if expected != actual:
                differences.append({"benchmark_row_id": card["benchmark_row_id"], "split": split,
                                    "molecule_identity_key": card["molecule_identity_key"],
                                    "frozen_label_counts": dict(expected), "replayed_label_counts": dict(actual)})
            cards_count += 1
            frozen_count += sum(expected.values())
            replay_count += len(votes)
            members.update(uid for uid, _ in votes)
    canonical = set(pq.read_table(library / "03_pair_buckets/records.parquet",
                                  columns=["source_row_uid"])["source_row_uid"].to_pylist())
    missing = sorted(members - canonical)
    return {"status": "blocked" if differences or missing else "verified",
            "upstream_commit": AMES_MEMBERSHIP_COMMIT,
            "inputs": inputs,
            "scope": "condition-free Skin cards only; external reviewed votes must be audited separately",
            "condition_free_cards": cards_count, "frozen_votes": frozen_count,
            "replayed_votes": replay_count, "card_differences": differences,
            "canonical_missing_uids": missing}


def publish_skin_replay(replay_root: Path, output_root: Path) -> dict[str, Any]:
    """Stage the approved count correction and uncapped physical membership."""
    gold = REPO_ROOT / "data/gold_labels/Skin_Reaction/v1/scaffold"
    target = output_root / "Skin_Reaction/v1/scaffold"
    if target.exists() or target.resolve() == gold.resolve():
        raise FileExistsError(f"Skin replay requires a new staging directory: {target}")
    groups, direct, inputs = replay_skin_votes(replay_root)
    original_path = replay_root / "data/processed_starling/Skin_Reaction/molecule_labels.jsonl"
    if sha256_file(original_path) != "ff8a04abaaff104ae2d61943bcd5ab8d19abba2920b150639a6e18bef96d8263":
        raise ValueError("Original Skin molecule table differs from the pinned migration")
    originals = {str(row["molecule_identity_key"]): row for row in read_jsonl(original_path)}
    physical = {str(row["source_row_uid"]): (position, row, "direct_skin_reaction")
                for position, row in enumerate(direct.to_pylist())}
    aop_path = REPO_ROOT / "data/raw/starling/skin_reaction/sensitization_aop/extractions.parquet"
    aop = pq.read_table(aop_path).to_pylist()
    external = defaultdict(list)
    for review in read_jsonl(gold / "source_condition_review.jsonl"):
        if review["review_status"] != "accepted":
            continue
        source, position = review["source_record_id"].rsplit(":", 1)
        position = int(position)
        row = direct.slice(position, 1).to_pylist()[0] if source == "direct_skin_reaction" else aop[position]
        if str(row.get("support_text") or "").strip() != str(review["support_text"]).strip():
            raise ValueError(f"Reviewed Skin source payload changed: {review['source_record_id']}")
        uid = str(row["source_row_uid"])
        physical[uid] = (position, row, source)
        external[(review["molecule_identity_key"], review["condition_group"])].append((uid, int(review["Y"])))
    membership, units, index, corrected, differences = [], [], [], {}, []
    vote_fields = ("label_counts", "source_record_count", "source_ids", "label_methods", "source_pmids",
                   "raw_value_examples", "context_examples", "majority_label", "majority_record_count",
                   "minority_record_count", "agreement_fraction", "agreement_threshold", "vote_unit")
    for split in SPLITS:
        for original in read_jsonl(gold / f"{split}_molecule_condition_labels.jsonl"):
            card = dict(original)
            parent, condition = card["molecule_identity_key"], card["condition_group"]
            null = condition == "no_reported_external_condition"
            votes = groups[parent] if null else external[(parent, condition)]
            counts = Counter(str(label) for _, label in votes)
            truth = originals[parent] if null else card
            if not votes or counts != +Counter(truth["label_counts"]):
                raise ValueError(f"Skin votes do not reproduce source truth: {card['benchmark_row_id']}")
            if len(votes) != len({uid for uid, _ in votes}):
                raise ValueError(f"Repeated Skin voter within card: {card['benchmark_row_id']}")
            mean = sum(label for _, label in votes) / len(votes)
            if int(mean >= 0.5) != int(card["Y"]):
                raise ValueError(f"Skin replay changes the frozen label: {card['benchmark_row_id']}")
            if counts != +Counter(card["label_counts"]):
                differences.append({"benchmark_row_id": card["benchmark_row_id"], "split": split,
                                    "old_label_counts": card["label_counts"], "new_label_counts": dict(counts)})
            if null:
                card.update({field: truth[field] for field in vote_fields if field in truth})
            card["source_record_ids"] = [uid for uid, _ in votes]
            card_units = []
            for uid, label in votes:
                position, raw, source = physical[uid]
                unit = dict(vote_id=uid, vote_label=label, physical_source_row_uids=[uid],
                            physical_source_record_ids=[str(raw["extraction_id"])])
                common = dict(contract_version="gold_v1_voter_contract.v1", task="skin_reaction",
                              benchmark_row_id=card["benchmark_row_id"], molecule_identity_key=parent,
                              condition_group=condition, split=split)
                units.append({**common, **unit, "physical_member_count": 1})
                card_units.append(unit)
                membership.append({**common, "source_row_uid": uid,
                    "physical_source_record_id": str(raw["extraction_id"]), "source_id": source,
                    "source_sha256": hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest(),
                    "canonical_smiles": str(raw["SMILES"]), "vote_id": uid, "vote_label": label,
                    "physical_member_index": 1, "physical_member_count": 1, "physical_vote_weight": 1.0})
            index.append({**common, "Y": int(card["Y"]), "voter_mean": mean,
                          "vote_unit_count": len(votes), "physical_member_count": len(votes), "vote_units": card_units})
            corrected[card["benchmark_row_id"]] = card
    counts = dict(gold_label_cards=len(index), vote_units=len(units), physical_members=len(membership),
                  distinct_physical_uids=len({row["source_row_uid"] for row in membership}))
    if counts != dict(gold_label_cards=2421, vote_units=35004, physical_members=35004, distinct_physical_uids=34950):
        raise ValueError(f"Unexpected Skin reconstruction counts: {counts}")
    shutil.copytree(gold, target)
    outputs = {}
    for name, rows, schema in (("voter_membership", membership, MEMBERSHIP_SCHEMA),
                               ("vote_units", units, VOTE_UNITS_SCHEMA),
                               ("gold_label_record_index", index, INDEX_SCHEMA)):
        path = target / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")
        outputs[name] = dict(path=f"data/gold_labels/Skin_Reaction/v1/scaffold/{path.name}", sha256=sha256_file(path))
    changed_files = {}
    for path in target.glob("*.jsonl"):
        rows = read_jsonl(path)
        if "molecule_condition_labels" not in path.name:
            continue
        replacements = [corrected.get(row.get("benchmark_row_id"), row) for row in rows]
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in replacements))
        changed_files[path.name] = {"before_sha256": sha256_file(gold / path.name), "after_sha256": sha256_file(path)}
    small = json.loads((target / "valid_small_manifest.json").read_text())
    small["source"]["valid_molecule_condition_labels_sha256"] = sha256_file(target / "valid_molecule_condition_labels.jsonl")
    small["valid_small"]["molecule_condition_labels_sha256"] = sha256_file(target / "valid_small_molecule_condition_labels.jsonl")
    (target / "valid_small_manifest.json").write_text(json.dumps(small, indent=2) + "\n")
    summary = json.loads((target / "summary.json").read_text())
    summary["voter_counts"] = counts
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = dict(version="gold_v1_voter_contract.v1", status="staged", task="skin_reaction", counts=counts,
                    policy={"gold_release": "v1", "vote_unit": "one accepted physical source row per context",
                            "aggregation": "one vote per (benchmark_row_id, vote_id)", "physical_member_weight": "1"},
                    inputs={"upstream_commit": AMES_MEMBERSHIP_COMMIT, **inputs,
                            "molecule_labels_sha256": sha256_file(original_path),
                            "source_condition_review_sha256": sha256_file(gold / "source_condition_review.jsonl")},
                    outputs=outputs, corrections=differences, corrected_files=changed_files)
    (target / "voter_contract_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=("all", "bbb_martins", "bioavailability_ma", "skin_reaction", *TASKS), default=["all"])
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data/gold_labels")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--skin-replay-root", type=Path,
                        help="Export of the pinned original Git inputs for the Skin audit")
    parser.add_argument("--replay-root", type=Path)
    parser.add_argument("--ames-membership-root", type=Path)
    args = parser.parse_args()
    if "all" in args.tasks and len(args.tasks) != 1:
        parser.error("--tasks all cannot be combined with task names")
    if args.skin_replay_root and not args.verify_only:
        if args.tasks != ["skin_reaction"]:
            parser.error("Skin publication requires --tasks skin_reaction")
        print(json.dumps(publish_skin_replay(args.skin_replay_root, args.output_root), indent=2))
        return 0
    if args.verify_only:
        gold_tasks = {"bbb_martins": "BBB_Martins", "bioavailability_ma": "Bioavailability_Ma",
                      "skin_reaction": "Skin_Reaction", **{task: spec["gold"] for task, spec in TASKS.items()}}
        report = {}
        for task, name in gold_tasks.items():
            if args.tasks != ["all"] and task not in args.tasks:
                continue
            library = REPO_ROOT / "data/evidence_libraries" / task
            try:
                report[task] = verify_membership(args.output_root / name / "v1/scaffold",
                                                  library / (library / "CURRENT").read_text().strip())
            except (ValueError, FileNotFoundError) as exc:
                report[task] = {"status": "blocked", "reason": str(exc)}
            if task == "skin_reaction" and args.skin_replay_root:
                report[task]["reconstruction"] = audit_skin_replay(
                    args.skin_replay_root, args.output_root / name / "v1/scaffold",
                    library / (library / "CURRENT").read_text().strip())
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2))
        return int(any(row["status"] != "verified" for row in report.values()))
    tasks = tuple(TASKS) if args.tasks == ["all"] else tuple(dict.fromkeys(args.tasks))
    if any(task not in TASKS for task in tasks):
        parser.error("Publication supports Ames, DILI and Carcinogens; use --verify-only to audit all six tasks")
    print(json.dumps({
        task: publish_task(
            task, args.output_root, args.replay_root, args.ames_membership_root
        ) for task in tasks
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
