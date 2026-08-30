import json

from tools.chembl_tool.common.json_utils import read_jsonl, sha256_file
from tools.chembl_tool.common.starling.conditioned_benchmark import (
    BENCHMARK_ROOT,
    CONTRACT,
    NO_REPORTED_CONDITION,
    TASK_DIRECTORIES,
    split_path,
)


EXPECTED_COUNTS = {
    "bbb_martins": (3053, 397, 393),
    "bioavailability_ma": (1958, 262, 269),
    "clintox": (1144, 142, 142),
    "skin_reaction": (1997, 246, 248),
}


def test_manifest_has_one_canonical_root_per_task() -> None:
    manifest = json.loads((BENCHMARK_ROOT / "manifest.json").read_text())
    assert manifest["contract"] == CONTRACT
    assert set(manifest["tasks"]) == set(TASK_DIRECTORIES)
    for task, counts in EXPECTED_COUNTS.items():
        assert tuple(
            manifest["tasks"][task]["split_counts"][split]
            for split in ("train", "valid", "test")
        ) == counts


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
