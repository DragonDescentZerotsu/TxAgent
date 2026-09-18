from data.processing.evidence_library.versions.v10.stage1_artifact_store import (
    STAGE1_STAGES,
    _profile,
)


def test_all_imported_task_stores_use_the_stage1_layout() -> None:
    skin = _profile("skin_reaction")
    assert skin.stages == STAGE1_STAGES
    assert skin.store_version == "skin_reaction.normalized_v10_stage1_store.v1"

    for task in ("ames", "dili", "carcinogens"):
        assert _profile(task).stages == STAGE1_STAGES
