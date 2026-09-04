import json

import pytest

from tools.chembl_tool.common.json_utils import read_jsonl, sha256_file
from tools.chembl_tool.common.starling.conditioned_benchmark import (
    BENCHMARK_ROOT,
    CONTRACT,
    NO_REPORTED_CONDITION,
    SPLIT_SCHEMES,
    TASK_DIRECTORIES,
    split_path,
    task_root,
)
from tools.chembl_tool.common.starling.build_conditioned_random_split import (
    RANDOM_SPLIT_CONTRACT,
    allocate_parent_groups,
    preserve_parent_assignment,
)


EXPECTED_COUNTS = {
    "bbb_martins": (3053, 397, 393),
    "bioavailability_ma": (1956, 262, 269),
    "clintox": (1144, 142, 142),
    "skin_reaction": (1941, 239, 241),
}

EXPECTED_RANDOM_COUNTS = {
    "bbb_martins": (3075, 384, 384),
    "bioavailability_ma": (1989, 249, 249),
    "clintox": (1142, 143, 143),
    "skin_reaction": (1937, 242, 242),
}


def test_manifest_has_one_canonical_root_per_task() -> None:
    manifest = json.loads((BENCHMARK_ROOT / "manifest.json").read_text())
    assert manifest["contract"] == CONTRACT
    assert manifest["construction_contract"]["publisher_revoting_allowed"] is False
    assert set(manifest["tasks"]) == set(TASK_DIRECTORIES)
    for task, counts in EXPECTED_COUNTS.items():
        task_manifest = manifest["tasks"][task]
        assert tuple(
            task_manifest["split_counts"][split]
            for split in ("train", "valid", "test")
        ) == counts
        assert task_manifest["roots"] == {
            scheme: str(task_root(task, scheme)) for scheme in SPLIT_SCHEMES
        }
        assert tuple(
            task_manifest["split_counts_by_scheme"]["random"][split]
            for split in ("train", "valid", "test")
        ) == EXPECTED_RANDOM_COUNTS[task]


def test_split_rows_have_uniform_condition_schema_and_no_overlap() -> None:
    required = {
        "drug",
        "Y",
        "condition_group",
        "condition_scope",
        "molecule_identity_key",
        "bemis_murcko_scaffold",
        "benchmark_row_id",
    }
    for task in TASK_DIRECTORIES:
        identities: dict[str, set[str]] = {}
        scaffolds: dict[str, set[str]] = {}
        for split in ("train", "valid", "test"):
            rows = read_jsonl(split_path(task, split))
            assert all(required <= set(row) for row in rows)
            identities[split] = {str(row["molecule_identity_key"]) for row in rows}
            scaffolds[split] = {
                str(row["bemis_murcko_scaffold"])
                for row in rows
                if str(row["bemis_murcko_scaffold"])
            }
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
            assert identities[left].isdisjoint(identities[right])
            assert scaffolds[left].isdisjoint(scaffolds[right])


def test_migration_receipt_matches_published_split_hashes() -> None:
    receipt = json.loads((BENCHMARK_ROOT / "migration_receipt.json").read_text())
    assert receipt["contract"] == CONTRACT
    for task, task_receipt in receipt["tasks"].items():
        for split, expected in task_receipt["splits"].items():
            path = split_path(task, split)
            assert sha256_file(path) == expected["canonical_sha256"]
            assert len(read_jsonl(path)) == expected["n"]
            assert expected["same_ordered_drug_labels"] is True


def test_clintox_uses_only_the_shared_null_condition() -> None:
    groups = {
        row["condition_group"]
        for split in ("train", "valid", "test")
        for row in read_jsonl(split_path("clintox", split))
    }
    assert groups == {NO_REPORTED_CONDITION}


def _row_content(row: dict) -> dict:
    return {
        key: value
        for key, value in row.items()
        if key not in {"split", "split_policy", "split_assignments"}
    }


def test_random_split_preserves_rows_and_groups_parents() -> None:
    assert SPLIT_SCHEMES == ("scaffold", "random")
    receipt = json.loads((BENCHMARK_ROOT / "random_split_receipt.json").read_text())
    assert receipt["split_contract"] == RANDOM_SPLIT_CONTRACT
    for task in TASK_DIRECTORIES:
        scaffold_rows = [
            row
            for split in ("train", "valid", "test")
            for row in read_jsonl(
                task_root(task, "scaffold")
                / f"{split}_molecule_condition_labels.jsonl"
            )
        ]
        random_by_split = {
            split: read_jsonl(
                task_root(task, "random")
                / f"{split}_molecule_condition_labels.jsonl"
            )
            for split in ("train", "valid", "test")
        }
        random_rows = [
            row for split_rows in random_by_split.values() for row in split_rows
        ]
        assert sorted(map(_row_content, scaffold_rows), key=str) == sorted(
            map(_row_content, random_rows), key=str
        )
        parents = {
            split: {str(row["molecule_identity_key"]) for row in rows}
            for split, rows in random_by_split.items()
        }
        assert parents["train"].isdisjoint(parents["valid"])
        assert parents["train"].isdisjoint(parents["test"])
        assert parents["valid"].isdisjoint(parents["test"])


def test_random_split_covers_every_condition_in_every_partition() -> None:
    for task in TASK_DIRECTORIES:
        condition_sets = [
            {
                str(row["condition_group"])
                for row in read_jsonl(split_path(task, split, "random"))
            }
            for split in ("train", "valid", "test")
        ]
        assert condition_sets[0] == condition_sets[1] == condition_sets[2]


def test_random_allocator_prioritizes_multi_vote_rows_for_heldout() -> None:
    rows = [
        {
            "Y": index % 2,
            "condition_group": "no_reported_external_condition",
            "molecule_identity_key": f"parent-{index:02d}",
            "source_record_count": 2 if index < 8 else 1,
        }
        for index in range(20)
    ]
    assignment, audit = allocate_parent_groups(rows, seed=17)
    repeated_assignment, repeated_audit = allocate_parent_groups(rows, seed=17)
    heldout = [
        row
        for row in rows
        if assignment[row["molecule_identity_key"]] in {"valid", "test"}
    ]
    assert len(heldout) == 4
    assert all(row["source_record_count"] > 1 for row in heldout)
    assert audit["minimum_singletons_in_heldout"] == 0
    assert audit["selected_valid_singletons"] == 0
    assert audit["selected_test_singletons"] == 0
    assert audit["record_support_field"] == "source_record_count"
    assert audit["multi_vote_definition"] == "source_record_count >= 2"
    assert repeated_assignment == assignment
    assert repeated_audit == audit


def test_random_assignment_can_be_preserved_after_train_only_deletion() -> None:
    rows = [
        {
            "benchmark_row_id": f"row-{index:02d}",
            "Y": index % 2,
            "condition_group": "no_reported_external_condition",
            "molecule_identity_key": f"parent-{index:02d}",
            "source_record_count": 2,
        }
        for index in range(22)
    ]
    reference_assignment, _ = allocate_parent_groups(rows, seed=17)
    reference_rows = [
        {**row, "split": reference_assignment[row["molecule_identity_key"]]}
        for row in rows
    ]
    train_row = next(row for row in rows if reference_assignment[row["molecule_identity_key"]] == "train")
    rebuilt_rows = [row for row in rows if row is not train_row]

    preserved, audit = preserve_parent_assignment(rebuilt_rows, reference_rows)
    assert preserved == {
        row["molecule_identity_key"]: reference_assignment[row["molecule_identity_key"]]
        for row in rebuilt_rows
    }
    assert audit["n_removed_rows"] == 1
    assert audit["n_added_rows"] == 0


def test_preserved_random_assignment_rejects_new_or_changed_rows() -> None:
    reference_rows = [
        {
            "benchmark_row_id": "row-01",
            "Y": 1,
            "condition_group": "no_reported_external_condition",
            "molecule_identity_key": "parent-01",
            "source_record_count": 2,
            "split": "train",
        }
    ]
    added = [
        reference_rows[0],
        {**reference_rows[0], "benchmark_row_id": "row-02"},
    ]
    changed = [{**reference_rows[0], "Y": 0}]

    with pytest.raises(ValueError, match="new benchmark_row_id"):
        preserve_parent_assignment(added, reference_rows)
    with pytest.raises(ValueError, match="changed content"):
        preserve_parent_assignment(changed, reference_rows)
