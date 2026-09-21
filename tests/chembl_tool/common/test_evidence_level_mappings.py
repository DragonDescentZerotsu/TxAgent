import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from data.processing.evidence_library.level_mappings import (
    evidence_level_mapping_release,
)


ROOT = Path("data")
EXPECTED = {
    "bbb_martins": {
        "public_task": "BBB_Martins",
        "sha256": "51c3a59524e8b69e9783164814d1529d08335b8abcfdcc3eb3113dd9e9451d0b",
        "rows": 496_135,
        "l1_rows": 7_634,
    },
    "bioavailability_ma": {
        "public_task": "Bioavailability_Ma",
        "sha256": "53bb5b26862d24e0928497dabfd9df8b0b3b5c108e990dc71d2c23d14dc06b7c",
        "rows": 431_457,
        "l1_rows": 19_479,
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("task", EXPECTED)
def test_current_mapping_exactly_covers_stage3_and_gold_v1_voters(task):
    expected = EXPECTED[task]
    manifest_path, mapping_path, receipt = evidence_level_mapping_release(
        task, "v10_main_universe_v3"
    )
    manifest = json.loads(manifest_path.read_text())

    assert mapping_path == (
        ROOT
        / f"evidence_libraries/{task}/v10_main_universe_v3/level_mapping/records.parquet"
    ).resolve()
    assert digest(mapping_path) == receipt["sha256"] == expected["sha256"]
    assert pq.read_metadata(mapping_path).num_rows == receipt["rows"] == expected["rows"]
    assert receipt["rows_by_level"]["1"] == expected["l1_rows"]
    assert manifest["validation"] == {
        "stage3_uid_coverage": "exact",
        "l1_equals_gold_physical_voters": True,
    }

    mapping = pq.read_table(mapping_path, columns=["source_row_uid", "level"])
    stage3_path = ROOT.parent / manifest["inputs"]["stage3_records"]["path"]
    stage3 = pq.read_table(stage3_path, columns=["source_row_uid"])
    mapping_uids = mapping.column("source_row_uid").to_pylist()
    assert len(mapping_uids) == len(set(mapping_uids))
    assert set(mapping_uids) == set(stage3.column("source_row_uid").to_pylist())

    voter_path = (
        ROOT
        / f"gold_labels/{expected['public_task']}/v1/scaffold/voter_membership.parquet"
    )
    voters = set(pq.read_table(voter_path, columns=["source_row_uid"])[0].to_pylist())
    levels = mapping.column("level").to_pylist()
    assert {uid for uid, level in zip(mapping_uids, levels) if level == 1} == voters


def test_runtime_index_points_to_release_owned_mappings_and_manifests():
    index_path = ROOT / "evidence_libraries/level_mappings.v1.json"
    index = json.loads(index_path.read_text())
    assert index["status"] == "complete"
    assert set(index["tasks"]) >= set(EXPECTED)

    for task, indexed in index["tasks"].items():
        manifest_path, mapping_path, receipt = evidence_level_mapping_release(
            task, indexed["release"]
        )
        assert mapping_path == (index_path.parent / indexed["path"]).resolve()
        assert manifest_path == (index_path.parent / indexed["manifest"]).resolve()
        assert digest(mapping_path) == indexed["sha256"] == receipt["sha256"]
        assert digest(manifest_path) == indexed["manifest_sha256"]
