import json
from pathlib import Path

import pytest

from tools.chembl_tool.common.json_utils import read_jsonl, sha256_file
from data.processing.gold_labels.conditioned_benchmark import (
    CONTRACT,
    TASK_DIRECTORIES,
    split_path,
)
from data.processing.gold_labels.build_valid_small import (
    build_all,
    select_scaffold_groups,
)


EXPECTED_COUNTS = {
    "bbb_martins": (3053, 397, 393),
    "bioavailability_ma": (1958, 262, 269),
    "skin_reaction": (1941, 239, 241),
    "ames": (1926, 274, 274),
    "dili": (3220, 402, 402),
    "carcinogens": (3754, 469, 469),
}

METADATA_ROOT = Path("data/artifacts/gold_labels/conditioned_benchmark")


def test_manifest_has_one_canonical_root_per_task() -> None:
    manifest = json.loads((METADATA_ROOT / "manifest.json").read_text())
    assert manifest["contract"] == CONTRACT
    assert set(TASK_DIRECTORIES) <= set(manifest["tasks"])
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
    receipt = json.loads((METADATA_ROOT / "migration_receipt.json").read_text())
    assert receipt["contract"] == CONTRACT
    for task in TASK_DIRECTORIES:
        task_receipt = receipt["tasks"][task]
        for split, expected in task_receipt["splits"].items():
            path = split_path(task, split)
            assert sha256_file(path) == expected["canonical_sha256"]
            assert len(read_jsonl(path)) == expected["n"]
            assert expected["same_ordered_drug_labels"] is True


def test_clintox_is_archived_not_an_active_gold_default() -> None:
    assert Path("data/legacy/clintox/gold_labels/conditioned_benchmark").is_dir()
    with pytest.raises(ValueError, match="Unknown conditioned benchmark task"):
        split_path("clintox", "test")


def test_valid_small_is_a_scaffold_disjoint_validation_subset() -> None:
    for task in TASK_DIRECTORIES:
        valid = read_jsonl(split_path(task, "valid"))
        selected_path = split_path(task, "valid_small")
        selected = read_jsonl(selected_path)
        selected_ids = [row["benchmark_row_id"] for row in selected]
        selected_id_set = set(selected_ids)
        valid_ids = [row["benchmark_row_id"] for row in valid]
        assert len(selected) == 100
        assert selected_ids == [row_id for row_id in valid_ids if row_id in selected_id_set]

        selected_scaffolds = {row["bemis_murcko_scaffold"] for row in selected}
        remainder_scaffolds = {
            row["bemis_murcko_scaffold"]
            for row in valid
            if row["benchmark_row_id"] not in selected_id_set
        }
        assert selected_scaffolds.isdisjoint(remainder_scaffolds)

        detailed = read_jsonl(
            selected_path.with_name("valid_small_molecule_condition_labels.jsonl")
        )
        assert [row["benchmark_row_id"] for row in detailed] == selected_ids
        manifest = json.loads(
            selected_path.with_name("valid_small_manifest.json").read_text()
        )
        assert manifest["valid_small"]["sha256"] == sha256_file(selected_path)
        assert manifest["valid_small"]["scaffold_overlap_with_valid_remainder"] == 0


def test_valid_small_selection_prioritizes_size_then_label_ratio() -> None:
    rows = [
        {"bemis_murcko_scaffold": "a", "molecule_identity_key": "a1", "Y": 1},
        {"bemis_murcko_scaffold": "a", "molecule_identity_key": "a1", "Y": 1},
        {"bemis_murcko_scaffold": "b", "molecule_identity_key": "b1", "Y": 0},
        {"bemis_murcko_scaffold": "b", "molecule_identity_key": "b1", "Y": 0},
        {"bemis_murcko_scaffold": "c", "molecule_identity_key": "c1", "Y": 1},
        {"bemis_murcko_scaffold": "d", "molecule_identity_key": "d1", "Y": 0},
    ]
    selected = select_scaffold_groups(rows, target_size=3, seed=20260723)
    selected_rows = [row for row in rows if row["bemis_murcko_scaffold"] in selected]
    assert len(selected_rows) == 3
    assert sum(row["Y"] for row in selected_rows) in {1, 2}


def test_valid_small_can_maximize_parents_before_label_ratio() -> None:
    rows = [
        {"bemis_murcko_scaffold": "a", "molecule_identity_key": "a1", "Y": 1},
        {"bemis_murcko_scaffold": "a", "molecule_identity_key": "a1", "Y": 1},
        {"bemis_murcko_scaffold": "b", "molecule_identity_key": "b1", "Y": 0},
        {"bemis_murcko_scaffold": "b", "molecule_identity_key": "b2", "Y": 0},
    ]
    selected = select_scaffold_groups(
        rows, target_size=2, seed=20260723, maximize_parents=True
    )
    assert selected == {"b"}


def test_valid_small_rebuild_is_byte_identical(tmp_path: Path) -> None:
    build_all(output_root=tmp_path)
    for task, directory in TASK_DIRECTORIES.items():
        version = (Path("data/gold_labels") / directory / "CURRENT").read_text().strip()
        expected = Path("data/gold_labels") / directory / version / "scaffold"
        rebuilt = tmp_path / directory / version / "scaffold"
        for name in (
            "valid_small.jsonl",
            "valid_small_molecule_condition_labels.jsonl",
            "valid_small_manifest.json",
        ):
            assert (rebuilt / name).read_bytes() == (expected / name).read_bytes()
