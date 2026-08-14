from tools.chembl_tool.tasks.clintox.starling_raw_artifact_store import (
    ARTIFACT_STAGES,
    PROFILE,
)


def test_clintox_raw_artifact_profile_covers_all_declared_stages():
    assert PROFILE.task_id == "clintox"
    assert PROFILE.stages == ARTIFACT_STAGES
    assert ARTIFACT_STAGES[0] == "01_cleaned"
    assert ARTIFACT_STAGES[-1] == "06_record_supported_v2_scaffold_view"
