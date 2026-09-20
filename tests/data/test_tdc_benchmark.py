from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.processing.gold_labels.conditioned_benchmark import tdc_split_path, tdc_task_root
from data.processing.gold_labels.build_tdc_benchmark import (
    _collapse_parents,
    _repair_provided_splits,
)
from predict.utils.json import read_jsonl, sha256_file


EXPECTED = {
    "bbb_martins": ("BBB_Martins", 1956, 1950, (1223, 197, 530)),
    "bioavailability_ma": ("Bioavailability_Ma", 640, 640, (448, 64, 128)),
    "skin_reaction": ("Skin_Reaction", 404, 404, (282, 40, 82)),
    "ames": ("Ames", 7278, 7246, (5077, 720, 1449)),
    "dili": ("DILI", 475, 474, (331, 47, 96)),
    "carcinogens": ("Carcinogens", 280, 276, (193, 27, 56)),
}


@pytest.mark.parametrize("task", EXPECTED)
def test_tdc_release_is_hash_pinned_and_split_disjoint(task: str) -> None:
    directory, source_rows, accepted_parents, counts = EXPECTED[task]
    root = tdc_task_root(task)
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())

    assert manifest["contract"] == "tdc_conditioned_benchmark.v1"
    assert manifest["status"] == "complete"
    assert manifest["version"] == "v1"
    assert manifest["release_id"] == "tdc_v1"
    assert manifest["source_rows"] == source_rows
    assert manifest["accepted_parents"] == accepted_parents
    assert manifest["current_pointer_changed"] is False
    assert manifest["summary_sha256"] == sha256_file(root / "summary.json")
    assert summary["outputs"] == {
        name: sha256_file(root / name) for name in summary["outputs"]
    }

    rows = {
        split: read_jsonl(root / f"{split}_molecule_condition_labels.jsonl")
        for split in ("train", "valid", "test")
    }
    assert tuple(len(rows[split]) for split in ("train", "valid", "test")) == counts
    for split_rows in rows.values():
        assert {row["Y"] for row in split_rows} == {0, 1}
        assert all(row["vote_unit"] == "external_dataset_label" for row in split_rows)
        assert all(row["record_support_eligible"] is False for row in split_rows)
        assert all(row["label_source"] == "tdc" for row in split_rows)
        assert all(row["source_pmids"] == [] for row in split_rows)

    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        assert not (
            {row["molecule_identity_key"] for row in rows[left]}
            & {row["molecule_identity_key"] for row in rows[right]}
        )
        assert not (
            {row["bemis_murcko_scaffold"] for row in rows[left] if row["bemis_murcko_scaffold"]}
            & {row["bemis_murcko_scaffold"] for row in rows[right] if row["bemis_murcko_scaffold"]}
        )

    current = (Path("data/gold_labels") / directory / "CURRENT").read_text().strip()
    assert current != "tdc_v1"
    assert tdc_split_path(task, "valid") == root / "valid.jsonl"
    assert not (Path("data/gold_labels") / directory / "tdc_v1").exists()


def test_ames_source_manifest_names_the_supplied_workspace_source() -> None:
    path = Path("data/raw/tdc/Ames/v1/source_manifest.json")
    manifest = json.loads(path.read_text())
    supplied = Path(
        "/vast/projects/myatskar/design-documents/joseph/"
        "therapeutic-tuning/data/raw/original/AMES"
    )
    assert Path(manifest["source_location"]) == supplied
    for row in manifest["files"]:
        source = Path(row["source_path"])
        assert source.parent == supplied
        assert sha256_file(Path(row["path"])) == row["sha256"]


def test_tdc_parent_conflicts_are_dropped_and_scaffolds_move_to_protected_split() -> None:
    rows = [
        {
            "source_order": 0,
            "source_record_id": "a",
            "source_split": "train",
            "molecule_identity_key": "parent-a",
            "bemis_murcko_scaffold": "shared",
            "Y": 0,
        },
        {
            "source_order": 1,
            "source_record_id": "b",
            "source_split": "valid",
            "molecule_identity_key": "parent-b",
            "bemis_murcko_scaffold": "shared",
            "Y": 1,
        },
        {
            "source_order": 2,
            "source_record_id": "c0",
            "source_split": "train",
            "molecule_identity_key": "conflict",
            "bemis_murcko_scaffold": "conflict-a",
            "Y": 0,
        },
        {
            "source_order": 3,
            "source_record_id": "c1",
            "source_split": "test",
            "molecule_identity_key": "conflict",
            "bemis_murcko_scaffold": "conflict-b",
            "Y": 1,
        },
    ]
    accepted, conflicts = _collapse_parents(rows)
    repairs = _repair_provided_splits(accepted)

    assert {row["molecule_identity_key"] for row in accepted} == {"parent-a", "parent-b"}
    assert conflicts[0]["molecule_identity_key"] == "conflict"
    assert {row["split"] for row in accepted} == {"valid"}
    assert repairs == [{
        "molecule_identity_key": "parent-a",
        "bemis_murcko_scaffold": "shared",
        "from_split": "train",
        "to_split": "valid",
        "reason": "cross_split_scaffold_precedence",
    }]
