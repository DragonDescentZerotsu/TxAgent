from pathlib import Path

from tools.chembl_tool.common.starling.stage_artifact_store import (
    StageArtifactStoreProfile,
    package_stages,
    restore_stages,
    verify_local,
    verify_tracked,
)
from tools.chembl_tool.tasks.bbb_martins.starling_artifact_store import PROFILE


def test_historical_v6_store_keeps_its_frozen_stage_inventory():
    assert "02_normalized" in PROFILE.stages
    assert "02_canonicalized" not in PROFILE.stages


def test_stage_store_packages_verifies_and_restores(tmp_path):
    local = tmp_path / "local"
    tracked = tmp_path / "tracked"
    stages = ("01_cleaned", "02_normalized")
    for number, stage in enumerate(stages):
        target = local / stage
        target.mkdir(parents=True)
        (target / "value.txt").write_text(f"value-{number}\n", encoding="utf-8")
    (local / "manifest.json").write_text('{"complete": true}\n', encoding="utf-8")
    profile = StageArtifactStoreProfile(
        store_version="test.v1",
        task_id="bbb_martins",
        stages=stages,
        local_root=local,
        tracked_root=tracked,
    )

    package_stages(profile, part_size=64)
    verify_tracked(profile)
    verify_local(profile)

    restored = tmp_path / "restored"
    restore_stages(profile, local_root=restored)
    assert (restored / "01_cleaned/value.txt").read_text(encoding="utf-8") == "value-0\n"
    assert (restored / "manifest.json").read_text(encoding="utf-8") == '{"complete": true}\n'
    verify_local(profile, local_root=restored)
