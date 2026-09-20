import csv
import json
from pathlib import Path

import pytest

from predict.harnesses.branches.flat import (
    PRESELECTED_UID_SCHEMA,
    _load_preselected_uids,
    _preselected_query_indices,
)
from predict.utils.json import sha256_file


def _manifest(tmp_path: Path) -> Path:
    cache_index = tmp_path / "RELEASE_INDEX.json"
    cache_index.write_text("{}\n", encoding="utf-8")
    records = tmp_path / "selected_uids.tsv"
    with records.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("benchmark_row_id", "level", "selection_rank", "source_row_uid"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows([
            {"benchmark_row_id": "q1", "level": "L2", "selection_rank": 1, "source_row_uid": "u2"},
            {"benchmark_row_id": "q1", "level": "L2", "selection_rank": 2, "source_row_uid": "u1"},
        ])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": PRESELECTED_UID_SCHEMA,
        "status": "complete",
        "task_id": "bbb_martins",
        "subset": "valid",
        "benchmark_row_ids": ["q1"],
        "k_per_level": {"L2": 2},
        "release_index_sha256": sha256_file(cache_index),
        "objective": {"version": "test"},
        "records": {
            "path": records.name,
            "sha256": sha256_file(records),
            "row_count": 2,
        },
    }), encoding="utf-8")
    return manifest


def test_manifest_preserves_optimizer_uid_order(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    selected, receipt = _load_preselected_uids(
        manifest,
        task="bbb_martins",
        subset="valid",
        queries={"q1": "CCO"},
        levels={"L2"},
        limits={"L2": 2},
        cache_index=tmp_path / "RELEASE_INDEX.json",
    )

    assert selected == {"q1": {"L2": ["u2", "u1"]}}
    assert receipt["objective"] == {"version": "test"}


def test_manifest_rejects_tampered_uid_table(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    (tmp_path / "selected_uids.tsv").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        _load_preselected_uids(
            manifest,
            task="bbb_martins",
            subset="valid",
            queries={"q1": "CCO"},
            levels={"L2"},
            limits={"L2": 2},
            cache_index=tmp_path / "RELEASE_INDEX.json",
        )


def test_manifest_query_ids_select_the_matching_canonical_rows(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    indices = _preselected_query_indices(
        manifest,
        [
            {"benchmark_row_id": "q0", "drug": "CC"},
            {"benchmark_row_id": "q1", "drug": "CCO"},
        ],
        task="bbb_martins",
        subset="valid",
    )

    assert indices == [1]
