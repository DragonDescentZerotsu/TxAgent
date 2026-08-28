"""Compare corrected BBB gold/conditioned v3 lineages with their v2 sources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic


V2_GOLD = Path(
    "data/processed_starling_experimental_meaningful_cns_access_v2/BBB_Martins"
)
V3_GOLD = Path(
    "data/processed_starling_experimental_meaningful_cns_access_v3/BBB_Martins"
)
V1_CONDITIONED = Path(
    "data/processed_starling_context_conditioned_selected_v1/BBB_Martins/scaffold"
)
V2_CONDITIONED = Path(
    "data/processed_starling_context_conditioned_selected_v2/BBB_Martins/scaffold"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _gold_by_identity(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["molecule_identity_key"]): row
        for row in _read_jsonl(root / "molecule_labels.jsonl")
    }


def _all_vote_rows_by_identity(root: Path) -> dict[str, dict[str, Any]]:
    """Return each parent's accepted-vote ledger, including rejected parents."""

    output = _gold_by_identity(root)
    for name in ("rejected_parent_molecules.jsonl", "conflicting_molecules.jsonl"):
        for row in _read_jsonl(root / name):
            output.setdefault(str(row["molecule_identity_key"]), row)
    return output


def _source_record_ids(rows: Mapping[str, Mapping[str, Any]]) -> set[str]:
    return {
        str(record_id)
        for row in rows.values()
        for record_id in row.get("source_record_ids") or ()
    }


def _split_by_identity(root: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        for row in _read_jsonl(root / "scaffold" / f"{split}_molecule_labels.jsonl"):
            output[str(row["molecule_identity_key"])] = split
    return output


def _conditioned_by_key(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for split in ("train", "valid", "test"):
        for row in _read_jsonl(root / f"{split}_molecule_condition_labels.jsonl"):
            key = (str(row["molecule_identity_key"]), str(row["condition_group"]))
            output[key] = {**row, "split": split}
    return output


def audit(output_path: Path) -> dict[str, Any]:
    old = _gold_by_identity(V2_GOLD)
    new = _gold_by_identity(V3_GOLD)
    old_vote_rows = _all_vote_rows_by_identity(V2_GOLD)
    new_vote_rows = _all_vote_rows_by_identity(V3_GOLD)
    old_split = _split_by_identity(V2_GOLD)
    new_split = _split_by_identity(V3_GOLD)
    shared = set(old) & set(new)
    removed = sorted(set(old) - set(new))
    added = sorted(set(new) - set(old))
    label_changes = sorted(key for key in shared if int(old[key]["Y"]) != int(new[key]["Y"]))
    split_changes = sorted(key for key in shared if old_split[key] != new_split[key])
    removed_vote_ids = sorted(
        _source_record_ids(old_vote_rows) - _source_record_ids(new_vote_rows)
    )
    removed_vote_id_set = set(removed_vote_ids)
    vote_affected_keys = sorted(
        key
        for key, row in old.items()
        if removed_vote_id_set & set(map(str, row.get("source_record_ids") or ()))
    )
    vote_affected_rows = [
        {
            "molecule_identity_key": key,
            "drug": old[key].get("drug"),
            "removed_source_record_ids": sorted(
                removed_vote_id_set
                & set(map(str, old[key].get("source_record_ids") or ()))
            ),
            "old_label_counts": old[key].get("label_counts"),
            "new_label_counts": (new_vote_rows.get(key) or {}).get("label_counts"),
            "old_label": old[key].get("Y"),
            "new_label": (new.get(key) or {}).get("Y"),
            "old_split": old_split.get(key),
            "new_split": new_split.get(key),
            "new_parent_status": "retained" if key in new else "removed_after_revote",
        }
        for key in vote_affected_keys
    ]

    old_conditioned = _conditioned_by_key(V1_CONDITIONED)
    new_conditioned = _conditioned_by_key(V2_CONDITIONED)
    shared_conditioned = set(old_conditioned) & set(new_conditioned)
    conditioned_label_changes = sorted(
        key
        for key in shared_conditioned
        if int(old_conditioned[key]["Y"]) != int(new_conditioned[key]["Y"])
    )
    conditioned_split_changes = sorted(
        key
        for key in shared_conditioned
        if old_conditioned[key]["split"] != new_conditioned[key]["split"]
    )
    changed_rows = [
        {
            "molecule_identity_key": key,
            "drug": old[key].get("drug"),
            "change": "removed_after_revote",
            "old_label": old[key].get("Y"),
            "old_split": old_split.get(key),
            "source_record_ids": old[key].get("source_record_ids"),
        }
        for key in removed
    ] + [
        {
            "molecule_identity_key": key,
            "drug": old[key].get("drug"),
            "change": "label_changed",
            "old_label": old[key].get("Y"),
            "new_label": new[key].get("Y"),
            "old_split": old_split.get(key),
            "new_split": new_split.get(key),
        }
        for key in label_changes
    ]
    changed_path = output_path.with_name("changed_gold_parents.jsonl")
    vote_affected_path = output_path.with_name("affected_gold_vote_parents.jsonl")
    write_jsonl_atomic(changed_path, changed_rows)
    write_jsonl_atomic(vote_affected_path, vote_affected_rows)
    summary = {
        "old_gold_lineage": "experimental_meaningful_cns_access_v2",
        "new_gold_lineage": "experimental_meaningful_cns_access_v3",
        "old_binary_parents": len(old),
        "new_binary_parents": len(new),
        "shared_binary_parents": len(shared),
        "removed_binary_parents": len(removed),
        "added_binary_parents": len(added),
        "shared_parent_label_changes": len(label_changes),
        "shared_parent_split_changes": len(split_changes),
        "removed_accepted_source_votes": len(removed_vote_ids),
        "removed_source_record_ids": removed_vote_ids,
        "vote_affected_parents": len(vote_affected_rows),
        "vote_affected_parent_acceptance_losses": sum(
            row["new_parent_status"] == "removed_after_revote"
            for row in vote_affected_rows
        ),
        "split_overlap": {
            split: sum(
                old_split.get(key) == split and new_split.get(key) == split
                for key in shared
            )
            for split in ("train", "valid", "test")
        },
        "conditioned": {
            "old_rows": len(old_conditioned),
            "new_rows": len(new_conditioned),
            "shared_rows": len(shared_conditioned),
            "removed_rows": len(set(old_conditioned) - set(new_conditioned)),
            "added_rows": len(set(new_conditioned) - set(old_conditioned)),
            "shared_row_label_changes": len(conditioned_label_changes),
            "shared_row_split_changes": len(conditioned_split_changes),
        },
        "changed_gold_parents": str(changed_path),
        "affected_gold_vote_parents": str(vote_affected_path),
    }
    write_json_atomic(output_path, summary)
    return summary


def main() -> None:
    output = V3_GOLD / "migration_from_v2.json"
    print(json.dumps(audit(output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
