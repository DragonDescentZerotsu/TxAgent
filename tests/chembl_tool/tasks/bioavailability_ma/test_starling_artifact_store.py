from __future__ import annotations

import pytest

from tools.chembl_tool.tasks.bioavailability_ma.starling_artifact_store import (
    package_stages,
    restore_stages,
    verify_local,
    verify_tracked,
)


def test_stage_packages_are_deterministic_split_and_selectively_restorable(tmp_path):
    local = tmp_path / "local"
    stage = local / "01_cleaned"
    stage.mkdir(parents=True)
    (stage / "records.parquet").write_bytes(b"records" * 100)
    (stage / "manifest.json").write_text("{}\n", encoding="utf-8")

    tracked_a = tmp_path / "tracked-a"
    tracked_b = tmp_path / "tracked-b"
    manifest_a = package_stages(local, tracked_a, stages=["01_cleaned"], part_size=100)
    manifest_b = package_stages(local, tracked_b, stages=["01_cleaned"], part_size=100)
    assert manifest_a["stages"]["01_cleaned"]["archive_sha256"] == (
        manifest_b["stages"]["01_cleaned"]["archive_sha256"]
    )
    assert all(
        part["size"] <= 100
        for part in manifest_a["stages"]["01_cleaned"]["parts"]
    )
    verify_tracked(tracked_a, stages=["01_cleaned"])

    restored = tmp_path / "restored"
    restore_stages(tracked_a, restored, stages=["01_cleaned"])
    verify_local(restored, tracked_a, stages=["01_cleaned"])
    assert (restored / "01_cleaned" / "records.parquet").read_bytes() == b"records" * 100
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        restore_stages(tracked_a, restored, stages=["01_cleaned"])


def test_tracked_verification_detects_corrupt_part(tmp_path):
    local = tmp_path / "local"
    stage = local / "01_cleaned"
    stage.mkdir(parents=True)
    (stage / "records.parquet").write_bytes(b"records")
    tracked = tmp_path / "tracked"
    manifest = package_stages(local, tracked, stages=["01_cleaned"])
    part = tracked / manifest["stages"]["01_cleaned"]["parts"][0]["path"]
    part.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="metadata mismatch"):
        verify_tracked(tracked, stages=["01_cleaned"])
