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
        "sha256": "6a0730e4704f02cb1dd54159b9318fcde4c3d777fffe0343ffc3ee59f308b766",
        "rows": 496_148,
        "l1_rows": 7_634,
    },
    "bioavailability_ma": {
        "public_task": "Bioavailability_Ma",
        "sha256": "86cd88de820420febc7c17b7f4f791ddbc4e9acc8f24a50a9511726d2f664f3b",
        "rows": 431_440,
        "l1_rows": 19_479,
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("task", EXPECTED)
def test_v10_mapping_exactly_covers_stage3_and_gold_v1_voters(task):
    expected = EXPECTED[task]
    manifest_path, mapping_path, receipt = evidence_level_mapping_release(task, "v10")
    manifest = json.loads(manifest_path.read_text())

    assert mapping_path == (
        ROOT / f"evidence_libraries/{task}/v10/level_mapping/records.parquet"
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
    assert set(index["tasks"]) == set(EXPECTED)

    for task, indexed in index["tasks"].items():
        manifest_path, mapping_path, receipt = evidence_level_mapping_release(task, "v10")
        assert mapping_path == (index_path.parent / indexed["path"]).resolve()
        assert manifest_path == (index_path.parent / indexed["manifest"]).resolve()
        assert digest(mapping_path) == indexed["sha256"] == receipt["sha256"]
        assert digest(manifest_path) == indexed["manifest_sha256"]
